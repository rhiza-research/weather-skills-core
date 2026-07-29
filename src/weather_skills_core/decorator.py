"""The ``@weather_skill`` decorator: a declarative CLI for weather skills.

A skill declares its surface (input/output envelope types, standard parameter
toggles, extra arguments) and keeps only its domain logic; the decorator owns
argparse construction, input reading, envelope validation, date resolution,
provenance, the cache-hit short-circuit, and output writing.

The wrapped function receives the input dataset(s) positionally, then ONE
dict holding every argument -- the extra arguments and the standard toggles
alike, keyed by dest, with the dates already resolved -- and returns its
output:

- a Dataset for a zarr-writing skill (the decorator stamps provenance,
  writes it, and removes a partial store when the write fails);
- a generator of per-period Datasets in streaming mode (the decorator writes
  the first with ``mode="w"`` and appends the rest, re-stamping provenance on
  every append and rolling back a partial store on failure);
- a Figure-like object (anything with ``savefig``) for a PNG-writing skill
  (the decorator saves it with provenance embedded in the PNG metadata);
- anything (ignored) for a no-artifact skill.

A zarr-writing skill may return ``(dataset, EntryOverride)`` instead of a
bare dataset, rewriting the recorded provenance args; a streaming skill
yields the override from its generator.

Calling the decorated function runs the CLI on ``sys.argv``; pass ``argv`` to
run it on an explicit argument list. Usage/validation failures exit 2 and
occur before any network work; data-availability and hard failures exit 1
(see :mod:`weather_skills_core.errors`).
"""

import argparse
import datetime
import functools
import inspect
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from weather_skills_core import dates as _dates
from weather_skills_core import envelope as _envelope
from weather_skills_core import provenance as _provenance
from weather_skills_core import types as _types
from weather_skills_core.errors import DataError, SkillError, UsageError

#: Core owns the date grammar, so core states it: the decorator appends this to
#: the help of every date flag it registers, whether the help is the default
#: below or a skill's own sentence. --end cross-references --start instead, so
#: one --help never prints the grammar twice. Exported for the rare skill whose
#: date flag is an ``extra_args`` entry rather than a standard toggle.
DATE_GRAMMAR = (
    "Either YYYY-MM-DD, 'now'/'today', 'latest', or an offset "
    "'now-<int>{d|w}' / 'latest-<int>{d|w}' (w = 7 days)."
)
_START_HELP = "Range start, inclusive."
_END_HELP = "Range end, inclusive. Same date grammar as --start."
_DATE_HELP = "Date."
_BBOX_REQUIRED_HELP = (
    "N/W/S/E decimal degrees (use the resolve-region skill to get a country's bbox)"
)
_BBOX_OPTIONAL_HELP = "Spatial subset N/W/S/E decimal degrees. Omit for the full grid."


@dataclass
class RunContext:
    """Run-scoped context shared by the decorator's hooks and the wrapped function.

    Created once per invocation and passed by keyword (``context=``) to every
    declaration hook whose signature names a ``context`` parameter --
    ``latest_resolver``, ``validate_args``, ``normalize_args``,
    ``completeness_probe``, ``write_encoding``, and ``post_write`` -- and to
    the wrapped function itself when its signature names one. Callables
    without the parameter keep their plain call shapes.

    ``args`` is the parsed argparse namespace. ``start_time`` holds the
    resolved absolute window start once the date grammar has run, and is
    ``None`` before that or when the toggle is off. ``state`` is a mutable
    scratch dict reserved for the skill: hooks and the function share it
    within one run (memoize an opened remote store, stash a value the
    write-encoding hook needs) and it starts empty on every run. Use it
    instead of module-level globals for run-scoped side channels.
    """

    args: argparse.Namespace
    start_time: datetime.date | None = None
    state: dict = field(default_factory=dict)


def _wants_context(fn) -> bool:
    """True when a callable's signature names a ``context`` parameter.

    The opt-in is the literal parameter name: a ``**kwargs`` catch-all does
    not opt in (a function receiving its CLI parameters as ``**params`` must
    not silently gain a ``context`` key).
    """
    try:
        return "context" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _call_hook(hook, *hook_args, wants_context, context):
    """Invoke a declaration hook, passing the run context only when it opts in."""
    if wants_context:
        return hook(*hook_args, context=context)
    return hook(*hook_args)


def _wrote_line(output, detail):
    """The ``Wrote:`` stderr line, with the write mode's detail parenthesized."""
    suffix = f" ({detail})" if detail else ""
    return f"Wrote: {output}{suffix}"


@dataclass
class EntryOverride:
    """Post-run provenance-entry rewrite.

    ``args`` is merged over the entry's recorded args. A standard-mode skill
    returns ``(dataset, EntryOverride(...))`` instead of a bare dataset; a
    streaming skill yields it from the generator at any point -- every
    subsequent stamp uses the rewritten entry, and an override yielded after
    the final dataset is applied by re-stamping the written store, so the
    persisted chain always reflects every override. This supports
    effective-end rewrites: a fetcher that discovers mid-run that trailing
    days are unavailable records the effective window rather than the
    requested one.
    """

    args: dict


def _split_extras(result, *, allow_override=True):
    """Unpack a wrapped function's return into ``(primary, override)``.

    A bare return is the primary result (the dataset or figure) alone; the
    only tuple form is ``(primary, EntryOverride)``, and PNG mode
    (``allow_override=False``) has no tuple form at all because its entries
    are embedded before the function runs. Anything else raises
    :class:`TypeError`.
    """
    if not isinstance(result, tuple):
        return result, None
    primary, *extras = result
    if not (allow_override and len(extras) == 1 and isinstance(extras[0], EntryOverride)):
        raise TypeError(
            f"unexpected extra return value(s) {extras!r}: a tuple return holds "
            "the output followed by exactly one EntryOverride (zarr mode only)"
        )
    return primary, extras[0]


def _order_lists(raw, preserve_order=()):
    """Sort every list-valued entry arg, so flag order cannot cause a cache miss.

    Ordered, never deduped: a value given twice is recorded twice. Runs last,
    after any ``normalize_args`` hook, so the guarantee holds for a list the
    hook itself produced.

    ``preserve_order`` names the dests whose order is data -- the skill's
    output changes with it -- which core cannot infer and sorting would
    destroy, collapsing two different requests onto one cache key.
    """
    ordered = {}
    for key, value in raw.items():
        # Tuples count: a hook may return one, and the JSON round-trip below
        # would turn it into a list that had never been ordered.
        if isinstance(value, list | tuple) and key not in preserve_order:
            try:
                value = sorted(value)
            except TypeError:
                # Mixed element types have no total order; keep as given.
                value = list(value)
        ordered[key] = value
    return ordered


def _remove_existing(path):
    """Remove whatever occupies an output path before a rewrite.

    A prior zarr store is a directory (``rmtree``); a regular file or a
    symlink at the path (e.g. an unrelated artifact written under the same
    name) is unlinked instead of crashing ``rmtree``. A dangling symlink --
    which ``exists()`` misses because it follows the link -- is unlinked
    too, and a vacant path is a no-op, so callers need no existence check.
    """
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def rewrite_bbox_argv(argv):
    """Rewrite ``--bbox VAL`` to ``--bbox=VAL`` in an argv list.

    argparse rejects a space-separated ``--bbox`` value that starts with ``-``
    (a bbox whose North latitude is negative); the equals form parses either
    way.
    """
    out, i = [], 0
    while i < len(argv):
        if argv[i] == "--bbox" and i + 1 < len(argv):
            out.append(f"--bbox={argv[i + 1]}")
            i += 2
        else:
            out.append(argv[i])
            i += 1
    return out


def _split_arg_spec(spec):
    """Split one ``extra_args`` tuple into ``(add_argument args, kwargs)``.

    The tuple IS an ``add_argument`` call: every leading element is a flag
    (or, without a leading dash, a positional's name) and an optional
    trailing dict holds argparse's own keywords.
    """
    if isinstance(spec, str) or not isinstance(spec, tuple | list):
        raise ValueError(  # noqa: TRY004 -- every declaration error is a ValueError
            f"each extra_args entry is a tuple of add_argument arguments, not {spec!r}"
        )
    names = list(spec)
    kwargs = names.pop() if names and isinstance(names[-1], dict) else {}
    if not names:
        raise ValueError(f"extra_args entry {spec!r} names no flag or positional")
    return names, dict(kwargs)


def _arg_dest(spec):
    """The dest argparse will give one ``extra_args`` entry.

    Mirrors argparse's own rule: an explicit ``dest`` wins, a positional is
    its own name verbatim, and a flag uses the first long spelling with the
    leading dashes stripped and inner dashes underscored. The dash-to-
    underscore rewrite is argparse's rule for optionals only, so a positional
    named ``target-grid`` keeps that dest and reaches the body under that key.
    """
    names, kwargs = _split_arg_spec(spec)
    if "dest" in kwargs:
        return kwargs["dest"]
    if not names[0].startswith("-"):
        return names[0]
    longs = [n for n in names if n.startswith("--")]
    return (longs[0] if longs else names[0]).lstrip("-").replace("-", "_")


def _normalize_date_toggle(name, value, *, extra_keys=()):
    """Normalize a ``start_time``/``end_time``/``date`` toggle to None or a config dict.

    ``True`` enables the flag with the decorator-owned help text and
    ``required=True``. A dict enables it with overrides: ``help``,
    ``required`` (default True), and ``choices`` -- plus, for ``date`` only,
    ``context`` (the resolved-date log line's parenthetical label).
    """
    if value is None or value is False:
        return None
    if value is True:
        return {}
    if isinstance(value, dict):
        allowed = {"help", "required", "choices", *extra_keys}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"{name} toggle has unknown keys {sorted(unknown)}")
        return dict(value)
    raise ValueError(f"{name} must be a bool or a dict, not {value!r}")


def _normalize_mode_toggle(name, value, modes, *, extra_keys=()):
    """Normalize a ``bbox``/``variable`` toggle to None or a config dict.

    A bare mode string becomes ``{"mode": <string>}``; a dict must carry a
    ``mode`` key from ``modes`` and may add ``help``/``choices`` (plus any
    ``extra_keys``, e.g. ``required`` for ``variable``).
    """
    if value is None:
        return None
    if isinstance(value, str):
        value = {"mode": value}
    if not isinstance(value, dict):
        raise ValueError(  # noqa: TRY004 -- ValueError is the observable contract asserted by callers/tests
            f"{name} must be one of {modes} or a dict with a 'mode' key, not {value!r}"
        )
    allowed = {"mode", "help", "choices", *extra_keys}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{name} toggle has unknown keys {sorted(unknown)}")
    if value.get("mode") not in modes:
        raise ValueError(f"{name} mode must be one of {modes}, not {value.get('mode')!r}")
    return dict(value)


def _normalize_workers_toggle(value):
    """Normalize the ``workers`` toggle to None or a config dict.

    A bare int becomes ``{"default": <int>}``; a dict may carry ``default``,
    ``help``, ``required``, and ``choices``.
    """
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return {"default": value}
    if isinstance(value, dict):
        unknown = set(value) - {"default", "help", "required", "choices"}
        if unknown:
            raise ValueError(f"workers toggle has unknown keys {sorted(unknown)}")
        return dict(value)
    raise ValueError(f"workers must be an int default or a dict, not {value!r}")


def _date_toggle_kwargs(cfg, default_help, *, grammar=True):
    """argparse keywords for a standard date toggle from its normalized config.

    The flag's own sentence is the default or the toggle's ``help`` override; a
    skill states only what is source-specific and the date grammar is appended
    here. ``grammar=False`` is ``--end``, whose default sentence points at
    ``--start`` rather than repeating it.
    """
    help_text = cfg.get("help", default_help)
    if grammar:
        help_text = f"{help_text} {DATE_GRAMMAR}"
    kwargs = {"required": cfg.get("required", True), "help": help_text}
    if "choices" in cfg:
        kwargs["choices"] = cfg["choices"]
    return kwargs


def _normalize_mutex_groups(mutex_groups, specs_by_dest):
    """Validate a ``mutex_groups`` declaration against the declared arguments.

    ``specs_by_dest`` maps each ``extra_args`` dest to its spec. Returns
    ``(group_required, dest_to_group)``: the per-group ``required`` flag and
    the dest-to-group-name membership map. Raises :class:`ValueError` for a
    group naming an undeclared dest, a dest in two groups, a group with fewer
    than two members, a positional member, or a member carrying its own
    ``required`` (requiredness belongs to the group).
    """
    group_required = {}
    dest_to_group = {}
    for group_name, group_spec in (mutex_groups or {}).items():
        if isinstance(group_spec, dict):
            unknown = set(group_spec) - {"args", "required"}
            if unknown:
                raise ValueError(f"mutex group {group_name!r} has unknown keys {sorted(unknown)}")
            if "args" not in group_spec:
                raise ValueError(f"mutex group {group_name!r} must list member dests under 'args'")
            dests = list(group_spec["args"])
            required = bool(group_spec.get("required", False))
        else:
            dests = list(group_spec)
            required = False
        if len(dests) < 2:
            raise ValueError(f"mutex group {group_name!r} needs at least two member dests")
        for dest in dests:
            if dest not in specs_by_dest:
                raise ValueError(
                    f"mutex group {group_name!r} names {dest!r}, which is not an extra_args dest"
                )
            if dest in dest_to_group:
                raise ValueError(
                    f"extra arg {dest!r} is in both mutex groups "
                    f"{dest_to_group[dest]!r} and {group_name!r}"
                )
            names, kwargs = _split_arg_spec(specs_by_dest[dest])
            if not names[0].startswith("-"):
                raise ValueError(
                    f"mutex group {group_name!r} member {dest!r} is positional; "
                    "mutually exclusive arguments must be flags"
                )
            if kwargs.get("required"):
                raise ValueError(
                    f"mutex group {group_name!r} member {dest!r} sets required=True; "
                    "declare requiredness on the group, not the member"
                )
            dest_to_group[dest] = group_name
        group_required[group_name] = required
    return group_required, dest_to_group


def weather_skill(
    name,
    version,
    *,
    input_type=None,
    output_type=None,
    input_names=None,
    input_help=None,
    variadic_input=False,
    start_time=False,
    end_time=False,
    date=False,
    bbox=None,
    variable=None,
    workers=None,
    title=False,
    dims=False,
    time_dim=False,
    extra_args=None,
    mutex_groups=None,
    latest_resolver=None,
    streaming=False,
    cache=True,
    hash_input=True,
    completeness_probe=None,
    validate_args=None,
    normalize_args=None,
    exclude_args=(),
    preserve_order=(),
    reference_args=(),
    history_labels=None,
    write_encoding=None,
    post_write=None,
    append_dim="time",
    savefig_kwargs=None,
    software=_provenance.DEFAULT_SOFTWARE,
):
    """Declare a weather skill: its CLI, its envelope contract, its cache, and its write.

    The wrapped function is called as ``fn(*datasets, arguments)`` -- the
    opened inputs positionally, then one :class:`argparse.Namespace` holding
    every ``extra_args`` value and every enabled toggle under its dest, read
    as ``arguments.start_time`` (dates already resolved to
    :class:`datetime.date`, ``bbox`` to an (N, W, S, E) tuple). It is built
    from the resolved values, not from the parsed namespace, which keeps the
    raw tokens the provenance entry records. A variadic skill gets
    ``fn(datasets, arguments)``. The function and every hook below opt into
    the :class:`RunContext` by naming a ``context`` parameter, which the
    decorator then passes as a separate keyword.

    ``skills/weather-skill-authoring/SKILL.md`` is the authoring guide: worked
    examples per skill class, the date grammar, cache semantics, and the
    envelope contract in full. Every parameter after ``version`` is
    keyword-only.

    Args:
        name: canonical skill name; recorded in every provenance entry.
        version: the script's ``_SKILL_VERSION``; shown in the ``--help`` epilog and recorded.
        input_type: envelope type(s) of the zarr input(s), from
            :mod:`weather_skills_core.types`. A type or a tuple of them declares ONE input
            (the tuple its allowed set); a LIST declares one entry per input; ``None``
            declares no zarr input. Each input is validated as the type it is detected to
            be, so a wider declaration accepts more shapes without checking less.
        output_type: ``None`` for a no-artifact skill (CLI only: no provenance, cache, or
            write), a zarr envelope type, a tuple of them (a union the returned shape must
            be a member of, checked before the write), or ``types.PNG``. A list is rejected:
            a skill has one output. A single type is unchecked; a shape-preserving transform
            declares ``types.ALL`` and asserts the preservation itself with
            :func:`weather_skills_core.envelope.validate_type`.
        input_names: one dedicated input flag per declared input, replacing ``--input``.
        input_help: help for the input flag(s): one string, or one per ``input_names`` entry.
        variadic_input: accept two or more ``--input`` repeats of the one declared type.
        start_time: enable ``--start``; ``True``, or a dict overriding ``help``/``required``/
            ``choices``.
        end_time: enable ``--end``; same forms as ``start_time``.
        date: enable ``--date``; same forms, plus ``context``, the resolved-date log label.
        bbox: enable ``--bbox``: ``types.REQUIRED`` or ``types.OPTIONAL``, or a dict.
        variable: enable ``--variable``: ``types.SINGLE`` or ``types.REPEAT``, or a dict.
        workers: enable ``--workers`` with this int default; excluded from the cache key.
        title: enable ``--title``.
        dims: enable ``--dims LAT,LON``, naming the spatial axes for input validation and the
            output union check; the body passes it to ``validate_type`` (WSK103).
        time_dim: enable ``--time-dim``; a string sets its default.
        extra_args: a sequence of ``add_argument`` calls, one tuple each -- leading strings
            are the flags (or a positional's name), an optional trailing dict is argparse's
            own keywords, verbatim. Dests are the ones argparse derives, must be unique, and
            may not reuse a name the decorator resolves itself.
        mutex_groups: group name to its member dests, or to ``{"args": (...), "required":
            True}``; each becomes an argparse mutually exclusive group.
        latest_resolver: ``callable(args) -> date`` resolving the ``latest`` token, invoked
            lazily and at most once per run.
        streaming: the function is a generator of per-period datasets, written then appended.
        cache: ``False`` skips the cache check -- the function runs and the output is
            rewritten every invocation, provenance still stamped.
        hash_input: ``False`` defers the input's content hash until after the cheap check.
        completeness_probe: ``callable(Path) -> bool`` confirming a candidate cache hit reads
            back; receives the output store's path.
        validate_args: ``callable(args)`` validating arguments before the cache check.
        normalize_args: ``callable(dict) -> dict`` canonicalizing the recorded entry args.
        exclude_args: dests dropped from the recorded entry args.
        preserve_order: dests whose list ORDER is data, exempt from the entry-arg sort.
        reference_args: dests holding secondary store paths, content-hashed into the entry's
            ``reference_inputs``.
        history_labels: per-input suffix for a PNG's embedded history keys (defaults to
            ``input_names``).
        write_encoding: ``callable(ds)`` setting write encodings, after the encoding clear.
        post_write: ``callable(path)`` run after the artifact is written and before the
            ``Wrote:`` line; raising a ``SkillError`` fails the run.
        append_dim: the dim a streaming write appends along.
        savefig_kwargs: extends the PNG ``savefig`` call (default ``{"dpi": 150}``).
        software: the ``Software`` key stamped into a PNG's metadata.

    Raises:
        ValueError: the declaration is malformed -- an unknown envelope type, a list
            ``output_type``, colliding or unknown dests, a toggle whose value is not one of
            its modes, or a combination the decorator cannot build a parser from.
    """
    input_types = _normalize_input_types(input_type)
    for declared in input_types:
        # Every alternative an input allows must be a known envelope type,
        # checked at import rather than at first run.
        unknown = [t for t in declared if t not in _types.ALL]
        if unknown:
            raise ValueError(
                f"unknown envelope type(s) {unknown} in input_type {input_type!r}; "
                f"valid types: {list(_types.ALL)}"
            )
    if variadic_input and len(input_types) != 1:
        raise ValueError("variadic_input requires exactly one declared input type")
    if input_names is not None and len(input_names) != len(input_types):
        raise ValueError("input_names must declare one flag per declared input type")
    if input_help is not None:
        if not input_types:
            raise ValueError("input_help requires a declared input_type")
        if input_names is not None:
            if isinstance(input_help, str) or len(list(input_help)) != len(input_names):
                raise ValueError(
                    "with input_names, input_help must give one help string per input flag"
                )
            input_help = list(input_help)
        elif not isinstance(input_help, str):
            raise ValueError(
                "without input_names, input_help is a single help string for the --input flag"
            )
    output_union = None
    if isinstance(output_type, list):
        # A list is the one-entry-per-input spelling, and a skill has one
        # output; a union is a tuple, the same as on input_type.
        raise ValueError(  # noqa: TRY004 -- every declaration error is a ValueError
            f"output_type takes one type or a tuple of them, not a list: {output_type!r}"
        )
    if isinstance(output_type, tuple | set | frozenset):
        members = list(output_type)
        if not members:
            raise ValueError("a union output_type needs at least one envelope type")
        bad = [t for t in members if t not in _types.ALL]
        if bad:
            raise ValueError(
                f"a union output_type may hold only zarr envelope types "
                f"{list(_types.ALL)}; got {bad}"
            )
        output_union = tuple(dict.fromkeys(members))
        zarr_output = True
    elif output_type not in (None, _types.PNG, *_types.ALL):
        raise ValueError(f"unknown output_type {output_type!r}")
    else:
        zarr_output = output_type in _types.ALL
    if streaming and not zarr_output:
        raise ValueError("streaming requires a zarr output_type")
    if cache is False and not zarr_output:
        raise ValueError(
            "cache=False requires a zarr output_type; PNG and no-artifact skills have no cache"
        )
    if output_type is None and input_types:
        raise ValueError("no-artifact skills do not declare input_type")
    if post_write is not None and output_type is None:
        raise ValueError("post_write requires an artifact-writing output_type")
    start_cfg = _normalize_date_toggle("start_time", start_time)
    end_cfg = _normalize_date_toggle("end_time", end_time)
    date_cfg = _normalize_date_toggle("date", date, extra_keys=("context",))
    bbox_cfg = _normalize_mode_toggle("bbox", bbox, (_types.OPTIONAL, _types.REQUIRED))
    variable_cfg = _normalize_mode_toggle(
        "variable", variable, (_types.SINGLE, _types.REPEAT), extra_keys=("required",)
    )
    workers_cfg = _normalize_workers_toggle(workers)
    if (start_cfg is None) != (end_cfg is None):
        raise ValueError("start_time and end_time must be enabled together")
    if start_cfg is not None and start_cfg.get("required", True) != end_cfg.get("required", True):
        raise ValueError("start_time and end_time must agree on required")
    # Dests the decorator itself resolves and passes to the function; an
    # extra_args entry under one of these names would be silently clobbered.
    reserved_dests = set()
    if start_cfg is not None:
        reserved_dests.update(("start_time", "end_time"))
    if date_cfg is not None:
        reserved_dests.add("date")
    if bbox_cfg is not None:
        reserved_dests.add("bbox")
    # Every argument reaches the function under one key, so the dests argparse
    # derives must not collide with each other or with a standard parameter's.
    extra_specs = list(extra_args or ())
    extra_dests = [_arg_dest(spec) for spec in extra_specs]
    duplicates = sorted({d for d in extra_dests if extra_dests.count(d) > 1})
    if duplicates:
        raise ValueError(f"extra_args declares dest(s) {duplicates} more than once")
    specs_by_dest = dict(zip(extra_dests, extra_specs, strict=True))
    collisions = sorted(reserved_dests & set(extra_dests))
    if collisions:
        raise ValueError(
            f"extra_args dest(s) {collisions} collide with standard parameter names "
            "the decorator resolves and passes itself; rename the extra argument(s)"
        )
    orderable = set(extra_dests) | {"variable"}
    unknown_preserved = sorted(set(preserve_order) - orderable)
    if unknown_preserved:
        raise ValueError(
            f"preserve_order names {unknown_preserved}, which are not list-valued "
            f"dests of this skill; expected some of {sorted(orderable)}"
        )
    png_labels = history_labels if history_labels is not None else input_names
    if output_type == _types.PNG and not input_types:
        raise ValueError("a png output_type requires at least one declared zarr input")
    if output_type == _types.PNG and len(input_types) > 1:
        if png_labels is None or len(png_labels) != len(input_types):
            raise ValueError("a multi-input PNG skill must declare one history label per input")
        if len(set(png_labels)) != len(png_labels):
            raise ValueError(
                "history labels must be unique; duplicates would collide on the "
                "embedded PNG metadata keys"
            )

    input_dests = list(input_names) if input_names else (["input"] if input_types else [])
    input_dests = [d.replace("-", "_") for d in input_dests]

    group_required, dest_to_group = _normalize_mutex_groups(mutex_groups, specs_by_dest)

    # Per-hook run-context opt-in, resolved once at declaration time.
    resolver_wants_ctx = latest_resolver is not None and _wants_context(latest_resolver)
    validate_wants_ctx = validate_args is not None and _wants_context(validate_args)
    normalize_wants_ctx = normalize_args is not None and _wants_context(normalize_args)
    probe_wants_ctx = completeness_probe is not None and _wants_context(completeness_probe)
    encoding_wants_ctx = write_encoding is not None and _wants_context(write_encoding)
    post_wants_ctx = post_write is not None and _wants_context(post_write)

    def decorate(fn):
        fn_wants_ctx = _wants_context(fn)
        if fn_wants_ctx and "context" in extra_dests:
            raise ValueError(
                "extra_args may not use the dest 'context' when the wrapped function "
                "declares a context parameter (the run context would clobber it)"
            )
        parser = _build_parser(fn)

        @functools.wraps(fn)
        def wrapper(argv=None):
            args_list = list(sys.argv[1:]) if argv is None else list(argv)
            if bbox_cfg is not None:
                args_list = rewrite_bbox_argv(args_list)
            args = parser.parse_args(args_list)
            try:
                _execute(fn, args, fn_wants_ctx)
            except SkillError as exc:
                message = f"Error: {exc}" if exc.prefix else str(exc)
                print(message, file=sys.stderr)
                sys.exit(exc.exit_code)

        wrapper.parser = parser
        return wrapper

    def _build_parser(fn):
        parser = argparse.ArgumentParser(
            description=fn.__doc__,
            epilog=f"skill version: {version}",
        )
        if input_names:
            helps = input_help if input_help is not None else [None] * len(input_names)
            for flag_name, help_text in zip(input_names, helps, strict=True):
                kwargs = {"required": True}
                if help_text is not None:
                    kwargs["help"] = help_text
                parser.add_argument(f"--{flag_name}", **kwargs)
        elif variadic_input:
            parser.add_argument(
                "--input",
                "-i",
                action="append",
                required=True,
                help=input_help
                if input_help is not None
                else "Input Zarr (repeat the flag for each input; need at least 2)",
            )
        elif len(input_types) == 1:
            kwargs = {"required": True}
            if input_help is not None:
                kwargs["help"] = input_help
            parser.add_argument("--input", "-i", **kwargs)
        elif len(input_types) > 1:
            parser.add_argument(
                "--input",
                "-i",
                action="append",
                required=True,
                help=input_help
                if input_help is not None
                else f"Input Zarr; pass exactly {len(input_types)} times, in order",
            )
        if output_type is not None:
            parser.add_argument("--output", "-o", required=True)
        if start_cfg is not None:
            parser.add_argument("--start", **_date_toggle_kwargs(start_cfg, _START_HELP))
        if end_cfg is not None:
            parser.add_argument("--end", **_date_toggle_kwargs(end_cfg, _END_HELP, grammar=False))
        if date_cfg is not None:
            parser.add_argument("--date", **_date_toggle_kwargs(date_cfg, _DATE_HELP))
        if bbox_cfg is not None:
            required = bbox_cfg["mode"] == _types.REQUIRED
            default_help = _BBOX_REQUIRED_HELP if required else _BBOX_OPTIONAL_HELP
            kwargs = {"help": bbox_cfg.get("help", default_help)}
            if required:
                kwargs["required"] = True
            if "choices" in bbox_cfg:
                kwargs["choices"] = bbox_cfg["choices"]
            parser.add_argument("--bbox", **kwargs)
        if variable_cfg is not None:
            kwargs = {}
            if variable_cfg["mode"] == _types.REPEAT:
                kwargs.update(action="append", default=None)
            if "help" in variable_cfg:
                kwargs["help"] = variable_cfg["help"]
            if variable_cfg.get("required"):
                kwargs["required"] = True
            if "choices" in variable_cfg:
                kwargs["choices"] = variable_cfg["choices"]
            parser.add_argument("--variable", "-v", **kwargs)
        if workers_cfg is not None:
            kwargs = {"type": int}
            if "default" in workers_cfg:
                kwargs["default"] = workers_cfg["default"]
                default_help = f"Max concurrent fetch threads (default {workers_cfg['default']})."
            else:
                default_help = "Max concurrent fetch threads."
            kwargs["help"] = workers_cfg.get("help", default_help)
            if workers_cfg.get("required"):
                kwargs["required"] = True
            if "choices" in workers_cfg:
                kwargs["choices"] = workers_cfg["choices"]
            parser.add_argument("--workers", **kwargs)
        if title:
            parser.add_argument("--title", help="Optional figure title.")
        if dims:
            parser.add_argument("--dims", help="Override LAT,LON dim names")
        if time_dim:
            kwargs = {"help": "Name of the time-like dim when not auto-detectable."}
            if isinstance(time_dim, str):
                kwargs["default"] = time_dim
            parser.add_argument("--time-dim", **kwargs)
        groups = {
            group_name: parser.add_mutually_exclusive_group(required=required)
            for group_name, required in group_required.items()
        }
        for dest, spec in specs_by_dest.items():
            target = groups.get(dest_to_group.get(dest), parser)
            names, kwargs = _split_arg_spec(spec)
            target.add_argument(*names, **kwargs)
        return parser

    def _execute(fn, args, fn_wants_ctx):
        context = RunContext(args=args)
        if workers_cfg is not None and args.workers is not None and args.workers < 1:
            raise UsageError("--workers must be >= 1.")
        if validate_args is not None:
            _call_hook(validate_args, args, wants_context=validate_wants_ctx, context=context)

        paths = _input_paths(args)
        for p in paths:
            if not p.exists():
                raise UsageError(f"{p} not found.")

        out = Path(args.output) if output_type is not None else None
        if zarr_output:
            _overlap_guard(paths, out, args)

        # Resolve bbox and dates before any provenance or network work: a
        # malformed value must exit 2 without side effects, and the recorded
        # args carry resolved absolute dates, never relative tokens.
        params = {}
        resolved_dates = {}
        # "Given" means present on the command line (not None): an explicit
        # empty-string value is a malformed token rejected by the parsers
        # below with exit 2, never a silently omitted flag.
        # bbox parses first: a malformed bbox exits before date resolution
        # can trigger any `latest` network discovery.
        if bbox_cfg is not None:
            params["bbox"] = _envelope.parse_bbox(args.bbox) if args.bbox is not None else None

        latest_fn = None
        if latest_resolver is not None:
            latest_cache = {}

            def latest_fn():
                # Memoized for the whole run, so a declaration with both a
                # window and a date discovers `latest` at most once.
                if "value" not in latest_cache:
                    latest_cache["value"] = _call_hook(
                        latest_resolver, args, wants_context=resolver_wants_ctx, context=context
                    )
                return latest_cache["value"]

        if start_cfg is not None:
            if args.start is not None and args.end is not None:
                start_d, end_d, log_line = _dates.resolve_window(args.start, args.end, latest_fn)
                if log_line is not None:
                    print(log_line, file=sys.stderr)
                params["start_time"], params["end_time"] = start_d, end_d
                context.start_time = start_d
                resolved_dates["start"] = start_d.isoformat()
                resolved_dates["end"] = end_d.isoformat()
            elif args.start is not None or args.end is not None:
                # Reachable only when the toggles declare required=False.
                raise UsageError("--start and --end must be given together.")
            else:
                params["start_time"] = params["end_time"] = None
        if date_cfg is not None:
            if args.date is not None:
                date_d, log_line = _dates.resolve_date(
                    args.date, latest_fn, context=date_cfg.get("context", "single date")
                )
                if log_line is not None:
                    print(log_line, file=sys.stderr)
                params["date"] = date_d
                resolved_dates["date"] = date_d.isoformat()
            else:
                params["date"] = None
        if variable_cfg is not None:
            params["variable"] = args.variable
        if workers_cfg is not None:
            params["workers"] = args.workers
        if title:
            params["title"] = args.title
        if dims:
            params["dims"] = args.dims
        if time_dim:
            params["time_dim"] = args.time_dim
        for dest in extra_dests:
            params[dest] = getattr(args, dest)
        # The run context stays a separate opt-in keyword, never a dict entry.
        ctx_kwargs = {"context": context} if fn_wants_ctx else {}

        if output_type is None:
            # No inputs to pass: a no-artifact skill may not declare any.
            _call(fn, [], params, ctx_kwargs)
            return

        entry_args = _entry_args(args, resolved_dates, context)

        if output_type == _types.PNG:
            _run_png(fn, args, paths, out, entry_args, params, ctx_kwargs, context)
            return
        _run_zarr(fn, args, paths, out, entry_args, params, ctx_kwargs, context)

    def _post_write(out, context):
        """Run the post-write hook, after the artifact write and before the Wrote line."""
        if post_write is not None:
            _call_hook(post_write, out, wants_context=post_wants_ctx, context=context)

    def _check_output_union(ds, args):
        """Validate a returned dataset's detected shape against a union output_type."""
        # The --dims override names the run's spatial axes on the way out too.
        detected = _envelope.detect_type(ds, args.dims if dims else None)
        if detected not in output_union:
            raise DataError(
                f"{name} returned a {detected} envelope, but its declared "
                f"output_type allows {' or '.join(output_union)}."
            )

    def _input_paths(args):
        if input_names:
            return [Path(getattr(args, d)) for d in input_dests]
        if not input_types:
            return []
        if variadic_input:
            values = args.input
            if len(values) < 2:
                raise UsageError("need at least 2 inputs.")
            return [Path(v) for v in values]
        if len(input_types) == 1:
            return [Path(args.input)]
        values = args.input
        if len(values) != len(input_types):
            raise UsageError(
                f"--input must be passed exactly {len(input_types)} times; got {len(values)}."
            )
        return [Path(v) for v in values]

    def _overlap_guard(paths, out, args):
        # rmtree of the output must never precede lazy reads of an input; the
        # same-store and nested-store cases would corrupt the input before its
        # lazily-backed values are read.
        out_r = out.resolve()
        for p in paths:
            p_r = p.resolve()
            if p_r == out_r or out_r.is_relative_to(p_r) or p_r.is_relative_to(out_r):
                raise UsageError(
                    f"--output ({args.output}) overlaps with input ({p}) as the same "
                    f"store or one nested inside the other; {name} writes to a "
                    "distinct output path."
                )

    def _entry_args(args, resolved_dates, context):
        path_dests = set(input_dests) | {"output"}
        raw = {k: v for k, v in vars(args).items() if k not in path_dests}
        raw.update(resolved_dates)
        raw.pop("workers", None)
        for dest in exclude_args:
            raw.pop(dest, None)
        if normalize_args is not None:
            raw = _call_hook(
                normalize_args, raw, wants_context=normalize_wants_ctx, context=context
            )
        raw = _order_lists(raw, preserve_order)
        # Canonicalize through a JSON round-trip so the compared entry equals
        # the stamped entry's decoded form: a tuple from a normalize hook
        # serializes as a list, and without the round-trip the stamped store
        # would never match its own cache key again.
        return json.loads(json.dumps(raw))

    def _open_inputs(paths, args):
        import xarray as xr

        if variadic_input:
            declared_per_path = [input_types[0]] * len(paths)
        else:
            declared_per_path = input_types
        # User-supplied --dims/--time-dim overrides apply to every validated
        # input: validation checks the overridden names instead of detecting.
        dims_override = args.dims if dims else None
        time_dim_override = args.time_dim if time_dim else None
        datasets = []
        for p, declared in zip(paths, declared_per_path, strict=True):
            try:
                ds = xr.open_zarr(p, consolidated=False)
            except Exception as exc:  # noqa: BLE001 -- broad by design: xr.open_zarr can fail many ways; all map to one UsageError
                raise UsageError(
                    f"{p} is not a readable Zarr store ({type(exc).__name__}: {exc})."
                ) from None
            _envelope.validate_input(
                ds,
                list(declared),
                str(p),
                dims=dims_override,
                time_dim=time_dim_override,
            )
            # The path rides on the dataset rather than a parallel argument,
            # so it cannot desync from the inputs; stamp_zarr strips it before
            # any write.
            ds.attrs[_provenance.INPUT_PATH_ATTR] = str(p)
            datasets.append(ds)
        return datasets

    def _call(fn, datasets, params, ctx_kwargs):
        # Datasets stay positional; every argument arrives as one namespace
        # after them. It is built HERE, from params, and never from the parsed
        # namespace: that one keeps the raw --bbox string and the raw date
        # tokens, and _entry_args records it verbatim.
        arguments = argparse.Namespace(**params)
        if variadic_input:
            return fn(datasets, arguments, **ctx_kwargs)
        return fn(*datasets, arguments, **ctx_kwargs)

    def _reference_inputs(args):
        refs = []
        for dest in reference_args:
            value = getattr(args, dest, None)
            if value:
                ref_p = Path(value)
                if not ref_p.exists():
                    raise UsageError(f"--{dest.replace('_', '-')} {ref_p} not found.")
                refs.append(ref_p)
        return _provenance.reference_ref(refs) if refs else None

    def _run_png(fn, args, paths, out, entry_args, params, ctx_kwargs, context):
        if out.is_dir():
            raise UsageError(
                f"--output {args.output} exists and is a directory; the png "
                "output must be a file path."
            )
        # Plot skills carry no cache: they always render. Each input branch
        # gets its own entry (same args, that input's basename + hash) on top
        # of that input's chain.
        upstreams = [_provenance.load_history(p) for p in paths]
        for p, upstream in zip(paths, upstreams, strict=True):
            if not upstream:
                print(
                    f"Warning: no upstream weather_skills_history on {p.name}; "
                    f"embedding {name} step alone.",
                    file=sys.stderr,
                )
        chains = []
        labels = png_labels if len(paths) > 1 else [None]
        for label, p, upstream in zip(labels, paths, upstreams, strict=True):
            entry = _provenance.build_entry(
                name, version, entry_args, _provenance.input_ref(p, include_hash=True)
            )
            chains.append((label, upstream + [entry]))

        datasets = _open_inputs(paths, args)
        fig, _ = _split_extras(_call(fn, datasets, params, ctx_kwargs), allow_override=False)

        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out,
            metadata=_provenance.png_metadata(chains, software=software),
            **{"dpi": 150, **(savefig_kwargs or {})},
        )
        # matplotlib is deliberately not a dependency of this package; close
        # the figure only when it is importable (a real Figure was returned).
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pass
        else:
            plt.close(fig)
        _post_write(out, context)
        print(_wrote_line(args.output, ""), file=sys.stderr)

    def _run_zarr(fn, args, paths, out, entry_args, params, ctx_kwargs, context):
        # The provenance entry is computed BEFORE the function runs: the entry
        # is the cache key, and a hit returns without calling the function or
        # touching the store.
        probe = None
        if completeness_probe is not None:

            def probe(candidate):
                return _call_hook(
                    completeness_probe, candidate, wants_context=probe_wants_ctx, context=context
                )

        reference_inputs = _reference_inputs(args)
        if not paths:
            upstream = []
            entry = _provenance.build_entry(name, version, entry_args, None, reference_inputs)
            if cache and _provenance.cache_hit(out, entry, fetcher=True, completeness_probe=probe):
                print(
                    f"Cache hit: {args.output} already matches requested params; skipping {name}.",
                    file=sys.stderr,
                )
                return
        elif len(paths) == 1:
            upstream = _provenance.load_history(paths[0])
            entry = _provenance.build_entry(
                name,
                version,
                entry_args,
                _provenance.input_ref(paths[0], include_hash=hash_input),
                reference_inputs,
            )
            if cache and _provenance.cache_hit(
                out, entry, upstream, compare_hash=hash_input, completeness_probe=probe
            ):
                print(
                    f"Cache hit: {args.output} already matches requested params; skipping {name}.",
                    file=sys.stderr,
                )
                return
            if not hash_input:
                # Cache miss: now pay for the content hash so the stamped
                # entry is complete even though the check skipped it.
                entry["input"] = _provenance.input_ref(paths[0], include_hash=True)
            if not upstream:
                print(
                    "Warning: no upstream weather_skills_history on input; "
                    "treating input as opaque.",
                    file=sys.stderr,
                )
        else:
            histories = [_provenance.load_history(p) for p in paths]
            upstream = histories[0]
            entry = _provenance.build_entry(
                name,
                version,
                entry_args,
                _provenance.multi_input_ref(paths, histories),
                reference_inputs,
            )
            if cache and _provenance.cache_hit(out, entry, upstream, completeness_probe=probe):
                print(
                    f"Cache hit: {args.output} already matches requested params; skipping {name}.",
                    file=sys.stderr,
                )
                return
            for p, hist in zip(paths, histories, strict=True):
                if not hist:
                    print(
                        f"Warning: no upstream weather_skills_history on input "
                        f"{p.name}; treating input as opaque.",
                        file=sys.stderr,
                    )

        datasets = _open_inputs(paths, args)
        result = _call(fn, datasets, params, ctx_kwargs)

        if streaming:
            _write_streaming(result, out, upstream, entry, args, context)
            return

        result, override = _split_extras(result)
        if output_union is not None:
            _check_output_union(result, args)
        if override is not None:
            entry = {**entry, "args": {**entry["args"], **override.args}}
        # Carry the first input's attrs (source metadata, upstream history)
        # under the function's own attrs, then stamp the new chain over both.
        if datasets:
            result.attrs = {**datasets[0].attrs, **result.attrs}
        _provenance.stamp_zarr(result, upstream + [entry])
        if write_encoding is not None:
            _call_hook(write_encoding, result, wants_context=encoding_wants_ctx, context=context)
        _remove_existing(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            result.to_zarr(out, mode="w", consolidated=True)
        except BaseException:
            # zarr stamps the root attrs before the chunk data, so a store
            # truncated by a failed write carries the full history attr and
            # would exactly match a later cache check; remove it before the
            # error propagates. Only the new partial store can be present
            # here -- any prior store was removed before the write started.
            if out.exists():
                shutil.rmtree(out)
                print(
                    f"Removed partial store {args.output} after a failed write "
                    "so it is not mistaken for a complete cache on a later run.",
                    file=sys.stderr,
                )
            raise
        _post_write(out, context)
        print(_wrote_line(args.output, f"{dict(result.sizes)}"), file=sys.stderr)

    def _write_streaming(gen, out, upstream, entry, args, context):
        # First write is mode="w"; later periods append along append_dim.
        # Provenance is re-stamped on every append because a to_zarr append
        # rewrites the root group attrs from the appended dataset. The
        # store_created flag flips BEFORE the first write (after any
        # pre-existing store is removed): a failure during the first write may
        # already have created a partial directory that must be cleaned up,
        # while a complete store from a previous run is gone before the flag
        # flips and so can never be deleted by the rollback.
        store_created = False
        total = 0
        written_entry = None
        try:
            for item in gen:
                if isinstance(item, EntryOverride):
                    entry = {**entry, "args": {**entry["args"], **item.args}}
                    continue
                piece = item
                if output_union is not None:
                    _check_output_union(piece, args)
                _provenance.stamp_zarr(piece, upstream + [entry])
                if write_encoding is not None:
                    _call_hook(
                        write_encoding, piece, wants_context=encoding_wants_ctx, context=context
                    )
                if not store_created:
                    _remove_existing(out)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    store_created = True
                    piece.to_zarr(out, mode="w", consolidated=True)
                else:
                    piece.to_zarr(out, mode="a", append_dim=append_dim, consolidated=True)
                written_entry = entry
                total += piece.sizes.get(append_dim, 0)
        except BaseException:
            if store_created and out.exists():
                shutil.rmtree(out)
                print(
                    f"Removed partial store {args.output} after a mid-stream failure "
                    "so it is not mistaken for a complete cache on a later run.",
                    file=sys.stderr,
                )
            raise
        if not store_created:
            raise DataError(f"{name} produced no data for the requested window; nothing written.")
        if entry != written_entry:
            # An EntryOverride yielded after the final dataset arrived after
            # the last stamp; correct the persisted chain in place.
            _provenance.restamp_zarr(out, upstream + [entry])
        _post_write(out, context)
        print(_wrote_line(args.output, f"{append_dim}={total}"), file=sys.stderr)

    return decorate


def _normalize_input_types(input_type):
    """Normalize the ``input_type`` declaration to one allowed-type tuple per input.

    A single type, or a tuple/set of them, declares one input -- the tuple
    being that input's allowed set. A list declares one entry per input, each
    entry itself a type or a tuple/set of them.
    """
    if input_type is None:
        return []
    if isinstance(input_type, str):
        return [(input_type,)]
    if isinstance(input_type, tuple | set | frozenset):
        return [tuple(input_type)]
    return [(entry,) if isinstance(entry, str) else tuple(entry) for entry in input_type]


@dataclass(frozen=True)
class StandardParameter:
    """One standard CLI parameter owned by the ``@weather_skill`` decorator.

    Describes the parameter's argparse surface as the decorator constructs it:

    - ``name`` -- the declaration keyword on :func:`weather_skill` (``"input"``
      and ``"output"`` for the I/O flags, which are driven by ``input_type``/
      ``input_names``/``variadic_input`` and ``output_type`` rather than a
      keyword of their own).
    - ``kind`` -- ``"io"`` for the input/output flags, ``"toggle"`` for the
      standard parameter toggles.
    - ``dest`` -- the argparse dest the flag parses into (``--start`` parses
      into ``start``; the decorator passes the resolved value to the wrapped
      function under ``start_time``).
    - ``flags`` -- every flag spelling the decorator registers. ``input_names``
      replaces the ``input`` entry's flags with one dedicated flag per input;
      the flags here are the default ``--input``/``-i`` surface.
    - ``arity`` -- ``"single"`` for one value per invocation, or
      ``"single_or_append"`` when a declaration mode selects between one value
      and a repeatable flag (``variable=types.REPEAT``; a multi-input or variadic
      ``--input``).
    - ``type_name`` -- the argparse ``type`` callable's name when the flag
      converts its value (``--workers`` parses through ``int``), else ``None``
      (the raw CLI string).
    - ``accepts_choices`` / ``accepts_help`` / ``accepts_required`` -- whether
      the toggle's dict form accepts that key. I/O flags and the plain toggles
      (``title``, ``dims``, ``time_dim``) have no dict form, so all three are
      ``False`` for them; ``input_help`` covers input-flag help, and ``bbox``
      requiredness is its ``mode`` key, not a dict ``required``.
    """

    name: str
    kind: str
    dest: str
    flags: tuple[str, ...]
    arity: str
    type_name: str | None
    accepts_choices: bool
    accepts_help: bool
    accepts_required: bool


def standard_parameters() -> tuple[StandardParameter, ...]:
    """The decorator's standard CLI parameter surface, for introspection.

    Enumerates every flag the decorator itself registers -- the input/output
    flags plus the standard parameter toggles -- with each flag's dest,
    spelling(s), arity, value type, and dict-form capability. Read-only: the
    parser construction in :func:`weather_skill` is the behavior; this
    function describes it (a conformance linter enumerates the standard
    surface from here instead of hardcoding a list).
    """
    return (
        StandardParameter(
            name="input",
            kind="io",
            dest="input",
            flags=("--input", "-i"),
            arity="single_or_append",
            type_name=None,
            accepts_choices=False,
            accepts_help=False,
            accepts_required=False,
        ),
        StandardParameter(
            name="output",
            kind="io",
            dest="output",
            flags=("--output", "-o"),
            arity="single",
            type_name=None,
            accepts_choices=False,
            accepts_help=False,
            accepts_required=False,
        ),
        StandardParameter(
            name="start_time",
            kind="toggle",
            dest="start",
            flags=("--start",),
            arity="single",
            type_name=None,
            accepts_choices=True,
            accepts_help=True,
            accepts_required=True,
        ),
        StandardParameter(
            name="end_time",
            kind="toggle",
            dest="end",
            flags=("--end",),
            arity="single",
            type_name=None,
            accepts_choices=True,
            accepts_help=True,
            accepts_required=True,
        ),
        StandardParameter(
            name="date",
            kind="toggle",
            dest="date",
            flags=("--date",),
            arity="single",
            type_name=None,
            accepts_choices=True,
            accepts_help=True,
            accepts_required=True,
        ),
        StandardParameter(
            name="bbox",
            kind="toggle",
            dest="bbox",
            flags=("--bbox",),
            arity="single",
            type_name=None,
            accepts_choices=True,
            accepts_help=True,
            accepts_required=False,
        ),
        StandardParameter(
            name="variable",
            kind="toggle",
            dest="variable",
            flags=("--variable", "-v"),
            arity="single_or_append",
            type_name=None,
            accepts_choices=True,
            accepts_help=True,
            accepts_required=True,
        ),
        StandardParameter(
            name="workers",
            kind="toggle",
            dest="workers",
            flags=("--workers",),
            arity="single",
            type_name="int",
            accepts_choices=True,
            accepts_help=True,
            accepts_required=True,
        ),
        StandardParameter(
            name="title",
            kind="toggle",
            dest="title",
            flags=("--title",),
            arity="single",
            type_name=None,
            accepts_choices=False,
            accepts_help=False,
            accepts_required=False,
        ),
        StandardParameter(
            name="dims",
            kind="toggle",
            dest="dims",
            flags=("--dims",),
            arity="single",
            type_name=None,
            accepts_choices=False,
            accepts_help=False,
            accepts_required=False,
        ),
        StandardParameter(
            name="time_dim",
            kind="toggle",
            dest="time_dim",
            flags=("--time-dim",),
            arity="single",
            type_name=None,
            accepts_choices=False,
            accepts_help=False,
            accepts_required=False,
        ),
    )
