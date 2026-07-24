"""The ``@weather_skill`` decorator: a declarative CLI for weather skills.

A skill declares its surface (input/output envelope types, standard parameter
toggles, extra arguments) and keeps only its domain logic; the decorator owns
argparse construction, input reading, envelope validation, date resolution,
provenance, the cache-hit short-circuit, and output writing.

The wrapped function receives the input dataset(s) positionally followed by
the resolved parameters as keyword arguments, and returns its output:

- a Dataset for a zarr-writing skill (the decorator stamps provenance,
  writes it, and removes a partial store when the write fails);
- a generator of per-period Datasets in streaming mode (the decorator writes
  the first with ``mode="w"`` and appends the rest, re-stamping provenance on
  every append and rolling back a partial store on failure);
- a Figure-like object (anything with ``savefig``) for a PNG-writing skill
  (the decorator saves it with provenance embedded in the PNG metadata);
- anything (ignored) for a no-artifact skill.

An artifact-writing skill may return a tuple -- the output first, followed by
marker objects: an :class:`EntryOverride` rewriting the recorded provenance
args (zarr modes) and/or a :class:`WroteSummary` customizing the ``Wrote:``
stderr line. A streaming skill yields either marker from its generator.

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
from weather_skills_core.errors import DataError, SkillError, UsageError

_START_HELP = (
    "Range start, inclusive. Either YYYY-MM-DD, 'now'/'today', 'latest', "
    "or an offset 'now-<int>{d|w}' / 'latest-<int>{d|w}' (w = 7 days)."
)
_END_HELP = "Range end, inclusive. Same date grammar as --start."
_DATE_HELP = (
    "Date. Either YYYY-MM-DD, 'now'/'today', 'latest', "
    "or an offset 'now-<int>{d|w}' / 'latest-<int>{d|w}' (w = 7 days)."
)
_BBOX_REQUIRED_HELP = (
    "N/W/S/E decimal degrees (use the resolve-region skill to get a country's bbox)"
)
_BBOX_OPTIONAL_HELP = "Spatial subset N/W/S/E decimal degrees. Omit for the full grid."

_ZARR_OUTPUT_TYPES = (_envelope.GRIDDED, _envelope.FORECAST, _envelope.STATION)
PNG = "png"
SAME = "same"


@dataclass
class RunContext:
    """Run-scoped context shared by the decorator's hooks and the wrapped function.

    Created once per invocation and passed by keyword (``context=``) to every
    declaration hook whose signature names a ``context`` parameter --
    ``latest_resolver``, ``validate_args``, ``normalize_args``,
    ``completeness_probe``, and ``write_encoding`` -- and to the wrapped
    function itself when its signature names one. Callables without the
    parameter keep their plain call shapes.

    Fields fill in as the run proceeds: ``args`` (the parsed argparse
    namespace) and ``output_path`` exist from the start; ``input_paths``
    holds the CLI-given input paths once collected (before the inputs are
    opened); ``start_time``/``end_time``/``date`` hold the resolved absolute
    dates once the date grammar has run, and are ``None`` before that or when
    the toggle is off.

    ``state`` is a mutable scratch dict reserved for the skill: hooks and the
    function share it within one run (memoize an opened remote store, stash a
    value the write-encoding hook needs) and it starts empty on every run.
    Use it instead of module-level globals for run-scoped side channels.
    """

    args: argparse.Namespace
    output_path: Path | None = None
    input_paths: list = field(default_factory=list)
    start_time: datetime.date | None = None
    end_time: datetime.date | None = None
    date: datetime.date | None = None
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


@dataclass
class WroteSummary:
    """Customize the detail of the decorator's ``Wrote:`` stderr line.

    ``detail`` is extra text for the line's parenthetical: appended after the
    default detail (``"; "``-separated) unless ``replace=True``, which makes
    it the whole parenthetical. The defaults are the output sizes for a
    standard zarr skill, ``<append_dim>=<total>`` for a streaming skill, and
    nothing for a PNG skill.

    A standard-mode skill returns it alongside the dataset -- ``(dataset,
    WroteSummary(...))``, in any combination with an :class:`EntryOverride`;
    a PNG skill returns ``(figure, WroteSummary(...))``; a streaming skill
    yields it from the generator (the last one yielded wins).
    """

    detail: str
    replace: bool = False


def _wrote_line(output, default_detail, summary):
    """Compose the ``Wrote:`` stderr line from the default detail and a summary."""
    if summary is None:
        detail = default_detail
    elif summary.replace or not default_detail:
        detail = summary.detail
    else:
        detail = f"{default_detail}; {summary.detail}"
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
    """Unpack a wrapped function's return into ``(primary, override, summary)``.

    A non-tuple return is the primary result alone. A tuple return's first
    element is the primary result (the dataset or figure); each remaining
    element must be a marker -- at most one :class:`EntryOverride` (rejected
    with ``allow_override=False``, i.e. in PNG mode, where the entries are
    embedded before the function runs) and at most one :class:`WroteSummary`.
    Anything else raises :class:`TypeError`.
    """
    if not isinstance(result, tuple):
        return result, None, None
    primary, *extras = result
    override = summary = None
    for extra in extras:
        if allow_override and isinstance(extra, EntryOverride) and override is None:
            override = extra
        elif isinstance(extra, WroteSummary) and summary is None:
            summary = extra
        else:
            raise TypeError(
                f"unexpected extra return value {extra!r}: a tuple return holds the "
                "output first, then at most one EntryOverride (zarr mode only) and "
                "at most one WroteSummary"
            )
    return primary, override, summary


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


def _add_extra_argument(parser, dest, spec):
    """Add one ``extra_args`` entry to the parser.

    ``spec`` is a bare type (``int``; ``bool`` becomes a store-true flag), a
    tuple/list of literal string choices, a constraint set combining a type
    with a value domain (``{int, range(0, 2)}`` derives ``choices``; the set
    must name the element type alongside any choices), or a dict of argparse
    keywords for full control with the extra keys ``positional``, ``flag``,
    ``aliases``, and ``repeat``.
    """
    flag_name = "--" + dest.replace("_", "-")
    if isinstance(spec, dict):
        spec = dict(spec)
        positional = spec.pop("positional", False)
        flag = spec.pop("flag", flag_name)
        aliases = list(spec.pop("aliases", ()))
        if spec.pop("repeat", False):
            spec["action"] = "append"
        if positional:
            parser.add_argument(dest, **spec)
        else:
            parser.add_argument(flag, *aliases, dest=dest, **spec)
        return
    kwargs = {}
    if spec is bool:
        kwargs["action"] = "store_true"
    elif isinstance(spec, type):
        kwargs["type"] = spec
    elif isinstance(spec, tuple | list):
        # A top-level tuple/list spec lists the flag's choices literally,
        # matched against the raw CLI string.
        kwargs["choices"] = list(spec)
    elif isinstance(spec, set | frozenset):
        for element in spec:
            if element is bool:
                kwargs["action"] = "store_true"
            elif isinstance(element, type):
                kwargs["type"] = element
            elif isinstance(element, range | tuple | list):
                kwargs["choices"] = list(element)
            else:
                raise ValueError(f"unsupported constraint {element!r} for extra arg {dest!r}")
        if "choices" in kwargs and "type" not in kwargs:
            raise ValueError(
                f"extra arg {dest!r} constrains choices without a type; the raw CLI "
                "string would never match a typed choice. Add the element type to the "
                "constraint set, or declare literal string choices as a top-level tuple."
            )
    else:
        raise ValueError(f"unsupported extra_args spec {spec!r} for {dest!r}")
    parser.add_argument(flag_name, dest=dest, **kwargs)


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


def _date_toggle_kwargs(cfg, default_help):
    """argparse keywords for a standard date toggle from its normalized config."""
    kwargs = {"required": cfg.get("required", True), "help": cfg.get("help", default_help)}
    if "choices" in cfg:
        kwargs["choices"] = cfg["choices"]
    return kwargs


def _normalize_mutex_groups(mutex_groups, extra_args):
    """Validate a ``mutex_groups`` declaration against ``extra_args``.

    Returns ``(group_required, dest_to_group)``: the per-group ``required``
    flag and the dest-to-group-name membership map. Raises :class:`ValueError`
    for a group naming an undeclared dest, a dest in two groups, a group with
    fewer than two members, a positional member, or a member carrying its own
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
            if dest not in (extra_args or {}):
                raise ValueError(
                    f"mutex group {group_name!r} names {dest!r}, which is not an extra_args dest"
                )
            if dest in dest_to_group:
                raise ValueError(
                    f"extra arg {dest!r} is in both mutex groups "
                    f"{dest_to_group[dest]!r} and {group_name!r}"
                )
            spec = extra_args[dest]
            if isinstance(spec, dict):
                if spec.get("positional"):
                    raise ValueError(
                        f"mutex group {group_name!r} member {dest!r} is positional; "
                        "mutually exclusive arguments must be flags"
                    )
                if spec.get("required"):
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
    input_paths=False,
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
    source=None,
    streaming=False,
    cache=True,
    hash_input=True,
    completeness_probe=None,
    validate_args=None,
    normalize_args=None,
    exclude_args=(),
    reference_args=(),
    history_labels=None,
    write_encoding=None,
    post_write=None,
    append_dim="time",
    savefig_kwargs=None,
    cache_hit_label=None,
    software=_provenance.DEFAULT_SOFTWARE,
):
    """Declare a weather skill.

    Declaration surface:

    - ``name`` / ``version`` -- canonical skill name and its version (the
      script's ``_SKILL_VERSION``); the version appears in the argparse epilog
      and every provenance entry.
    - ``input_type`` -- envelope type(s) of the zarr input(s): ``None`` (no
      zarr inputs), one type string, or a comma string / list declaring one
      type per input (each from ``gridded``/``forecast``/``station``/``any``).
      Inputs arrive via ``--input``/``-i`` (repeated when there are several)
      unless ``input_names`` names a dedicated flag per input (e.g.
      ``["forecast", "mclimate"]``), or ``variadic_input=True`` accepts two or
      more ``--input`` repeats of a single declared type (the function then
      receives one list of datasets).
    - ``input_help`` -- help text for the input flag(s). With ``input_names``,
      a sequence giving one help string per named flag (a ``None`` entry
      leaves that flag without help). Otherwise a single string shown on the
      ``--input``/``-i`` flag, replacing the decorator's default help in the
      variadic and fixed-multi cases. Requires a declared ``input_type``.
    - ``input_paths`` -- with ``True``, the function also receives an
      ``input_paths`` keyword argument: the CLI-given input path(s) as a list
      of :class:`~pathlib.Path`, in input order (declaration order for
      ``input_names``, repeat order otherwise). For diagnostics and messages;
      the datasets still arrive positionally, and the paths never enter the
      recorded provenance args. Requires a declared ``input_type``.
    - ``output_type`` -- ``None`` for a no-artifact skill (argparse + version
      epilog only: no provenance, no cache, no write), a zarr envelope type,
      a tuple/set of zarr envelope types (a union), ``"same"``, or ``"png"``
      for a Figure-writing skill. ``"same"`` declares a shape-preserving
      transform: the output is whatever envelope type the (first) input
      carries, useful for skills whose ``input_type`` admits several shapes
      (``"gridded|forecast"``, ``"any"``, ...). It requires at least one
      declared zarr input and is written through the zarr path exactly like
      an explicit zarr type; it asserts nothing new about the output's shape
      beyond "the input's shape, preserved". A union (e.g. ``("gridded",
      "forecast")``, for a fetcher whose source decides the shape) VALIDATES
      rather than selects: the returned dataset's detected shape must be a
      member, checked before the write (a mismatch exits 1); the declaration
      never coerces the output toward any member. A single-type declaration
      stays unchecked.
    - standard parameter toggles: ``start_time``/``end_time``/``date`` (the
      relative-or-absolute date grammar; resolved dates are passed to the
      function and recorded in provenance), ``bbox`` (``"required"`` or
      ``"optional"``; parsed to an (N, W, S, E) tuple), ``variable``
      (``"single"`` or ``"repeat"``), ``workers`` (an int default; excluded
      from the cache key), ``title``, ``dims`` (LAT,LON override), and
      ``time_dim`` (pass a string to set a default). A user-supplied
      ``--dims``/``--time-dim`` value is honored during input validation:
      typed inputs are validated against the overridden dim names instead of
      relying on CF/heuristic detection (see
      :func:`weather_skills_core.envelope.validate_input`).
    - toggle dict form: ``start_time``/``end_time``/``date``, ``bbox``,
      ``variable``, and ``workers`` also accept a dict overriding the flag's
      argparse surface -- ``help`` replaces the decorator-owned help text,
      ``required`` overrides requiredness (``--start``/``--end``/``--date``
      default to required; with ``"required": False`` an omitted value is
      passed to the function as ``None`` and no resolved date is recorded),
      and ``choices`` constrains the accepted values. The string/int forms
      become the dict's ``mode``/``default`` key -- ``bbox={"mode":
      "optional", ...}``, ``variable={"mode": "repeat", ...}``,
      ``workers={"default": 4, ...}`` -- and ``date`` additionally accepts
      ``context``: the parenthetical label on the resolved-date stderr line
      (default ``"single date"``, e.g. ``"single forecast init date"``).
    - ``extra_args`` -- mapping of dest name to a bare type, a tuple/list of
      literal string choices, a constraint set combining a type with a value
      domain (``{int, range(0, 2)}``), or an argparse-keyword dict. A dest
      may not reuse a name the decorator resolves and passes itself
      (``start_time``/``end_time``/``date``/``bbox``/``input_paths``, and
      ``context`` when the function opts into the run context): the resolved
      value would clobber the extra argument's.
    - ``mutex_groups`` -- mapping of group name to either a sequence of
      ``extra_args`` dests (an optional group) or a dict
      ``{"args": (dests...), "required": True}``. Each group becomes a real
      argparse mutually exclusive group: at most one member may be given, and
      ``required=True`` demands exactly one. Members must be non-positional
      ``extra_args`` entries that do not set their own ``required``; the group
      name labels the declaration only (argparse mutex groups are untitled).
    - ``latest_resolver`` -- ``callable(args) -> date`` resolving the
      ``latest`` token; invoked lazily, at most once per run.
    - ``source`` -- ``weather_skills_source`` value stamped on fetcher output.
    - ``streaming`` -- the function is a generator yielding per-period
      datasets, written as ``mode="w"`` then appends along ``append_dim``.
    - cache behavior: ``cache=False`` disables the cache check entirely -- the
      function runs and the output is rewritten on every invocation, with the
      provenance entry still built and stamped (for skills whose recompute is
      cheaper than a meaningful cache key); ``hash_input`` compares the
      input's content hash in the cache key (``False`` defers the expensive
      hash until after a cheap check); ``completeness_probe`` (``callable(Path) -> bool``) verifies a
      candidate cache hit actually reads back -- it receives the output store's
      path and applies to fetcher and chained (transform) checks alike;
      ``reference_args`` names
      arg dests holding secondary reference-store paths, content-hashed into
      the entry's ``reference_inputs``.
    - hooks: ``validate_args(args)`` for pre-cache argument validation (raise
      ``UsageError``); ``normalize_args(dict) -> dict`` canonicalizes the
      recorded entry args (sort/dedupe) so flag order cannot cause spurious
      misses -- the returned dict is passed through a JSON round-trip before
      the cache compare and the stamp, so JSON-equivalent containers (a
      tuple vs. the list it serializes to) compare equal;
      ``exclude_args`` drops further dests from the entry args;
      ``write_encoding(ds)`` sets controlled write encodings after the
      encoding clear; ``post_write(path)`` runs after the artifact is written
      (zarr, streaming, or PNG -- requires an artifact output_type),
      receiving the output path, and may fail the run by raising a
      ``SkillError`` (which maps to the usual exit codes) -- use it for
      read-back verification of the written store. It runs before the
      ``Wrote:`` line, and a cache hit skips it (nothing was written).
    - run context: every hook above (plus ``latest_resolver`` and
      ``completeness_probe``) and the wrapped function itself opt into the
      run context by naming a ``context`` parameter; the decorator then also
      passes ``context=`` -- a :class:`RunContext` carrying the parsed args
      namespace, the resolved dates, the input/output paths, and a run-scoped
      ``state`` scratch dict shared across the hooks and the function.
      Callables without the parameter keep their plain call shapes.
    - PNG: ``history_labels`` gives the per-input suffix for the embedded
      history keys (defaults to ``input_names``); ``savefig_kwargs`` extends
      the ``savefig`` call (default ``{"dpi": 150}``).
    - stderr messages: ``cache_hit_label`` replaces the skill name as the
      word after "skipping" in the cache-hit line (default: ``name``); the
      ``Wrote:`` line's detail is customized by returning or yielding a
      :class:`WroteSummary` (see its docstring). All other decorator-emitted
      stderr lines are fixed.
    """
    input_types = _normalize_input_types(input_type)
    for declared in input_types:
        # An input may declare alternatives with "|" (e.g. "gridded|forecast");
        # every alternative must be a known envelope type, checked at import.
        unknown = [t for t in (a.strip() for a in declared.split("|")) if t not in _envelope.TYPES]
        if unknown:
            raise ValueError(
                f"unknown envelope type(s) {unknown} in input_type {declared!r}; "
                f"valid types: {list(_envelope.TYPES)}"
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
    if input_paths and not input_types:
        raise ValueError("input_paths=True requires a declared input_type")
    output_union = None
    if isinstance(output_type, tuple | set | frozenset | list):
        members = list(output_type)
        if not members:
            raise ValueError("a union output_type needs at least one envelope type")
        bad = [t for t in members if t not in _ZARR_OUTPUT_TYPES]
        if bad:
            raise ValueError(
                f"a union output_type may hold only zarr envelope types "
                f"{list(_ZARR_OUTPUT_TYPES)}; got {bad}"
            )
        output_union = tuple(dict.fromkeys(members))
        zarr_output = True
    elif output_type not in (None, PNG, SAME, *_ZARR_OUTPUT_TYPES):
        raise ValueError(f"unknown output_type {output_type!r}")
    else:
        zarr_output = output_type in (SAME, *_ZARR_OUTPUT_TYPES)
    if output_type == SAME and not input_types:
        raise ValueError('output_type="same" requires at least one declared zarr input')
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
    bbox_cfg = _normalize_mode_toggle("bbox", bbox, ("optional", "required"))
    variable_cfg = _normalize_mode_toggle(
        "variable", variable, ("single", "repeat"), extra_keys=("required",)
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
    if input_paths:
        reserved_dests.add("input_paths")
    collisions = sorted(reserved_dests & set(extra_args or {}))
    if collisions:
        raise ValueError(
            f"extra_args dest(s) {collisions} collide with standard parameter names "
            "the decorator resolves and passes itself; rename the extra argument(s)"
        )
    png_labels = history_labels if history_labels is not None else input_names
    if output_type == PNG and not input_types:
        raise ValueError('output_type="png" requires at least one declared zarr input')
    if output_type == PNG and len(input_types) > 1:
        if png_labels is None or len(png_labels) != len(input_types):
            raise ValueError("a multi-input PNG skill must declare one history label per input")
        if len(set(png_labels)) != len(png_labels):
            raise ValueError(
                "history labels must be unique; duplicates would collide on the "
                "embedded PNG metadata keys"
            )

    input_dests = list(input_names) if input_names else (["input"] if input_types else [])
    input_dests = [d.replace("-", "_") for d in input_dests]

    group_required, dest_to_group = _normalize_mutex_groups(mutex_groups, extra_args)
    hit_label = cache_hit_label if cache_hit_label is not None else name

    # Per-hook run-context opt-in, resolved once at declaration time.
    resolver_wants_ctx = latest_resolver is not None and _wants_context(latest_resolver)
    validate_wants_ctx = validate_args is not None and _wants_context(validate_args)
    normalize_wants_ctx = normalize_args is not None and _wants_context(normalize_args)
    probe_wants_ctx = completeness_probe is not None and _wants_context(completeness_probe)
    encoding_wants_ctx = write_encoding is not None and _wants_context(write_encoding)
    post_wants_ctx = post_write is not None and _wants_context(post_write)

    def decorate(fn):
        fn_wants_ctx = _wants_context(fn)
        if fn_wants_ctx and "context" in (extra_args or {}):
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
            parser.add_argument("--end", **_date_toggle_kwargs(end_cfg, _END_HELP))
        if date_cfg is not None:
            parser.add_argument("--date", **_date_toggle_kwargs(date_cfg, _DATE_HELP))
        if bbox_cfg is not None:
            required = bbox_cfg["mode"] == "required"
            default_help = _BBOX_REQUIRED_HELP if required else _BBOX_OPTIONAL_HELP
            kwargs = {"help": bbox_cfg.get("help", default_help)}
            if required:
                kwargs["required"] = True
            if "choices" in bbox_cfg:
                kwargs["choices"] = bbox_cfg["choices"]
            parser.add_argument("--bbox", **kwargs)
        if variable_cfg is not None:
            kwargs = {}
            if variable_cfg["mode"] == "repeat":
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
        for dest, spec in (extra_args or {}).items():
            target = groups.get(dest_to_group.get(dest), parser)
            _add_extra_argument(target, dest, spec)
        return parser

    def _execute(fn, args, fn_wants_ctx):
        context = RunContext(
            args=args,
            output_path=Path(args.output) if output_type is not None else None,
        )
        if workers_cfg is not None and args.workers is not None and args.workers < 1:
            raise UsageError("--workers must be >= 1.")
        if validate_args is not None:
            _call_hook(validate_args, args, wants_context=validate_wants_ctx, context=context)

        paths = _input_paths(args)
        context.input_paths = list(paths)
        for p in paths:
            if not p.exists():
                raise UsageError(f"{p} not found.")

        out = context.output_path
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
                context.start_time, context.end_time = start_d, end_d
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
                context.date = date_d
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
        for dest in extra_args or {}:
            params[dest] = getattr(args, dest)
        if input_paths:
            params["input_paths"] = list(paths)
        if fn_wants_ctx:
            params["context"] = context

        if output_type is None:
            fn(**params)
            return

        entry_args = _entry_args(args, resolved_dates, context)

        if output_type == PNG:
            _run_png(fn, args, paths, out, entry_args, params, context)
            return
        _run_zarr(fn, args, paths, out, entry_args, params, context)

    def _post_write(out, context):
        """Run the post-write hook, after the artifact write and before the Wrote line."""
        if post_write is not None:
            _call_hook(post_write, out, wants_context=post_wants_ctx, context=context)

    def _check_output_union(ds):
        """Validate a returned dataset's detected shape against a union output_type."""
        detected = _envelope.detect_type(ds)
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
            # An input may declare alternatives with "|" (e.g. "gridded|forecast").
            _envelope.validate_input(
                ds,
                [t.strip() for t in declared.split("|")],
                str(p),
                dims=dims_override,
                time_dim=time_dim_override,
            )
            datasets.append(ds)
        return datasets

    def _call(fn, datasets, params):
        if variadic_input:
            return fn(datasets, **params)
        return fn(*datasets, **params)

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

    def _run_png(fn, args, paths, out, entry_args, params, context):
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
        fig, _, summary = _split_extras(_call(fn, datasets, params), allow_override=False)

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
        print(_wrote_line(args.output, "", summary), file=sys.stderr)

    def _run_zarr(fn, args, paths, out, entry_args, params, context):
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
                    f"Cache hit: {args.output} already matches requested params; skipping {hit_label}.",
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
                    f"Cache hit: {args.output} already matches requested params; skipping {hit_label}.",
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
                    f"Cache hit: {args.output} already matches requested params; skipping {hit_label}.",
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
        result = _call(fn, datasets, params)

        if streaming:
            _write_streaming(result, out, upstream, entry, args, context)
            return

        result, override, summary = _split_extras(result)
        if output_union is not None:
            _check_output_union(result)
        if override is not None:
            entry = {**entry, "args": {**entry["args"], **override.args}}
        # Carry the first input's attrs (source metadata, upstream history)
        # under the function's own attrs, then stamp the new chain over both.
        if datasets:
            result.attrs = {**datasets[0].attrs, **result.attrs}
        _provenance.stamp_zarr(result, upstream + [entry], source=source)
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
        print(_wrote_line(args.output, f"{dict(result.sizes)}", summary), file=sys.stderr)

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
        summary = None
        written_entry = None
        try:
            for item in gen:
                if isinstance(item, EntryOverride):
                    entry = {**entry, "args": {**entry["args"], **item.args}}
                    continue
                if isinstance(item, WroteSummary):
                    summary = item
                    continue
                piece = item
                if output_union is not None:
                    _check_output_union(piece)
                _provenance.stamp_zarr(piece, upstream + [entry], source=source)
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
        print(_wrote_line(args.output, f"{append_dim}={total}", summary), file=sys.stderr)

    return decorate


def _normalize_input_types(input_type):
    """Normalize the ``input_type`` declaration to a list of per-input types."""
    if input_type is None:
        return []
    if isinstance(input_type, str):
        return [t.strip() for t in input_type.split(",")]
    return list(input_type)


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
      and a repeatable flag (``variable="repeat"``; a multi-input or variadic
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
