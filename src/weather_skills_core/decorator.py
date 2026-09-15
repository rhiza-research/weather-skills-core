"""``@weather_skill`` decorator: CLI, open Dataset inputs, stamp/write outputs."""

import argparse
import functools
import inspect
import json
import shutil
import sys
from pathlib import Path

import xarray as xr

from weather_skills_core import provenance as provenance_mod
from weather_skills_core import standard_args
from weather_skills_core import standard_dataset as std
from weather_skills_core.dataset_type import Dataset
from weather_skills_core.errors import SkillError, UsageError
from weather_skills_core.standard_utils import (
    fill_missing_data_var_attrs,
    normalize_latlon_coords,
    normalize_step_coord,
)
from weather_skills_core.units import (
    dequantify_dataset,
    normalize_unit_strings,
    quantify_dataset,
    stamp_precip_amounts,
)

# Accumulated by ``@weather_skill.argument`` (bottom-up); weather_skill reads it.
ARGS_ATTR = "__weather_skill_arguments__"


def argv_has_option(argv: list[str], option_strings) -> bool:
    """True when one of ``option_strings`` is present (bare, space, or ``--flag=``)."""
    for flag in option_strings:
        for arg in argv:
            if arg == flag or (flag.startswith("--") and arg.startswith(f"{flag}=")):
                return True
    return False


class Argument:
    """One stacked CLI flag (same kwargs as ``argparse.add_argument``).

    ``probe=True`` is reserved: when that flag is on argv, the decorator
    drops every ``required=True`` flag (including ``-o``) and skips writing.
    """

    def __init__(self, *option_strings, **kwargs):
        if not option_strings:
            raise ValueError("Argument requires at least one option string or positional name")
        kwargs = dict(kwargs)
        self.probe = bool(kwargs.pop("probe", False))
        self.option_strings = option_strings
        self.kwargs = kwargs
        self.dataset_type: Dataset | None = None
        type_kw = self.kwargs.get("type")
        if isinstance(type_kw, Dataset):
            self.dataset_type = type_kw

    @property
    def dest(self) -> str:
        if "dest" in self.kwargs:
            return self.kwargs["dest"]
        flag = next(
            (f for f in self.option_strings if f.startswith("--")),
            self.option_strings[0],
        )
        return flag.lstrip("-").replace("-", "_")


def build_history(name, version, args, params, input_paths, upstream, strip_dests):
    """Append this skill's provenance entry to the first input's history."""
    entry_args = {k: v for k, v in vars(args).items() if k not in strip_dests}
    for dest in ("date", "start_time", "end_time"):
        if dest in params and params[dest] is not None:
            entry_args[dest] = params[dest].isoformat()
    entry_args = json.loads(json.dumps(entry_args, default=str))

    if not input_paths:
        input_field, base_history = None, []
    elif len(input_paths) == 1:
        input_field = provenance_mod.input_ref(input_paths[0])
        base_history = upstream[0]
    else:
        input_field = [
            {
                "basename": path.name,
                "hash": provenance_mod.hash_zarr(path),
                "history": hist,
            }
            for path, hist in zip(input_paths, upstream, strict=True)
        ]
        base_history = upstream[0]

    return base_history + [provenance_mod.build_entry(name, version, entry_args, input_field)]


def prepare_dataset_output(ds, *, first_ds=None):
    """Normalize coords/units metadata every skill writes.

    - GRIB-style ``kg m**-2`` → pint/CF strings
    - precip amount units → amount CF ``standard_name``
    - ``step`` timedelta → ``timedelta64[ns]``
    - lat/lon coords → 5 decimal places, ``float32``
    - fill attrs stripped by geometry ops from the first input (same var names)
    """
    if first_ds is not None:
        ref = first_ds
        if any(getattr(first_ds[v].pint, "units", None) is not None for v in first_ds.data_vars):
            ref = dequantify_dataset(first_ds)
        ds = fill_missing_data_var_attrs(ref, ds)
    ds = normalize_unit_strings(ds)
    ds = stamp_precip_amounts(ds)
    ds = normalize_step_coord(ds)
    return normalize_latlon_coords(ds)


def write_output(value, out_path, history, first_ds):
    """Write one skill result: stamp a returned Path, or ``to_zarr`` a Dataset."""
    if isinstance(value, (str, Path)):
        written = Path(value)
        if written.resolve() != out_path.resolve():
            raise SkillError(f"returned path {written} does not match --output {out_path}")
        if written.is_dir():
            provenance_mod.restamp_zarr(written, history)
        else:
            provenance_mod.stamp_figure(written, history)
        print(f"Wrote: {out_path}", file=sys.stderr)
        return

    if not hasattr(value, "dims"):
        raise SkillError(f"skill returned {type(value).__name__}; expected xr.Dataset or Path")

    if hasattr(value, "pint"):
        value = dequantify_dataset(value)
    value = prepare_dataset_output(value, first_ds=first_ds)
    if first_ds is not None:
        value.attrs = {**first_ds.attrs, **value.attrs}
    provenance_mod.stamp_zarr(value, history)
    if out_path.exists():
        shutil.rmtree(out_path)
    try:
        value.to_zarr(out_path, mode="w", consolidated=True)
    except Exception:
        if out_path.exists():
            shutil.rmtree(out_path)
        raise
    print(f"Wrote: {out_path}", file=sys.stderr)


def _open_zarr(path, io_spec=None):
    """Open, validate, and quantify one Zarr store."""
    if not path.exists():
        raise UsageError(f"input not found: {path}")
    ds = xr.open_zarr(path, consolidated=True)
    if io_spec is not None:
        std.validate_input(ds, io_spec, str(path))
    ds = normalize_unit_strings(ds)
    ds = normalize_step_coord(ds)
    return quantify_dataset(ds)


def _iter_zarr_path_holders(value):
    """Yield objects that expose ``zarr_paths()`` (e.g. plot ``LayerSpec``)."""
    if value is None:
        return
    items = value if isinstance(value, (list, tuple)) else (value,)
    for item in items:
        fn = getattr(item, "zarr_paths", None)
        if callable(fn):
            yield item


def open_dataset_params(params, arguments):
    """Replace Dataset-typed Path values with opened/validated/quantified datasets.

    Values (or list items) that expose ``zarr_paths() -> list[Path]`` are also
    opened and hashed. Identical paths are opened once and reused. Each holder
    gets ``.ds`` (one path) or ``.datasets`` (several).
    """
    input_paths: list[Path] = []
    upstream: list = []
    first_ds = None
    cache: dict[Path, xr.Dataset] = {}

    def remember(path, ds):
        nonlocal first_ds
        resolved = path.resolve()
        cache[resolved] = ds
        if resolved not in {p.resolve() for p in input_paths}:
            input_paths.append(path)
            upstream.append(provenance_mod.load_history(path))
        if first_ds is None:
            first_ds = ds
        return ds

    for arg in arguments:
        if arg.dataset_type is None:
            continue
        dest, raw = arg.dest, params.get(arg.dest)
        if raw is None:
            continue
        paths = [Path(p) for p in raw] if isinstance(raw, (list, tuple)) else [Path(raw)]
        if not paths:
            continue
        opened = []
        for path in paths:
            resolved = path.resolve()
            ds = cache.get(resolved)
            if ds is None:
                ds = _open_zarr(path, arg.dataset_type.io_spec)
            opened.append(remember(path, ds))
        params[dest] = opened if isinstance(raw, (list, tuple)) else opened[0]

    seen_holders: list[object] = []
    for raw in list(params.values()):
        for holder in _iter_zarr_path_holders(raw):
            if holder in seen_holders:
                continue
            seen_holders.append(holder)
            opened = []
            for path in holder.zarr_paths() or []:
                path = Path(path)
                resolved = path.resolve()
                ds = cache.get(resolved)
                if ds is None:
                    ds = _open_zarr(path)
                opened.append(remember(path, ds))
            if not opened:
                continue
            if len(opened) == 1:
                holder.ds = opened[0]
            else:
                holder.datasets = opened

    return params, input_paths, upstream, first_ds


def argument(*option_strings, **kwargs):
    """Declare a CLI flag under ``@weather_skill`` (argparse-style)."""
    arg = Argument(*option_strings, **kwargs)

    def decorate(fn):
        existing = list(getattr(fn, ARGS_ATTR, []))
        existing.insert(0, arg)
        setattr(fn, ARGS_ATTR, existing)
        return fn

    return decorate


def weather_skill(
    *,
    name,
    version,
    output: bool = True,
):
    """Turn a function into a weather skill CLI with validated I/O and provenance.

    Stack ``@weather_skill.argument`` for flags. Use ``type=Dataset(...)`` for
    Zarr inputs (opened and dim-checked before the skill runs). Dataset
    ``--input`` is passed to the skill as ``ds`` (a list when ``action="append"``).
    Converted values that expose ``zarr_paths() -> list[Path]`` are opened and
    hashed the same way (identical paths are reused); each holder gets ``.ds``
    or ``.datasets``.

    When ``output=True`` (default), the decorator owns ``-o/--output``
    (repeatable). It injects ``output`` as a ``Path`` (one path) or
    ``list[Path]`` (several). Returning an ``xr.Dataset`` writes Zarr there;
    returning a ``Path`` stamps that file; returning a sequence writes one
    artifact per ``--output``. The number of returned values must match the
    number of ``--output`` paths. Returning ``None`` skips decorator write
    (skill already wrote). Set ``output=False`` for inspect-only skills.
    ``probe=True`` on an argument drops required flags and skips writing.

    Skills open precip rates and amounts alike. The operation that cannot take
    amounts is ``rate_to_total`` (used by ``convert-to-totals``): multiplying
    an amount by ``aggregation_period`` would double-count.

    On every Dataset write the decorator also normalizes GRIB unit strings,
    stamps precip-amount CF names when units are amounts, casts ``step`` to
    ``timedelta64[ns]``, rounds lat/lon to 5 decimal places as ``float32``,
    and fills data-var attrs stripped by the skill from the first input
    (same variable names). Value conversion (``to_standard_units``) stays
    skill-owned.
    """

    def decorator(fn):
        arguments = list(getattr(fn, ARGS_ATTR, []))
        for arg in arguments:
            if not isinstance(arg, Argument):
                raise TypeError(
                    f"skill {name!r}: stacked arguments must be Argument instances; "
                    f"got {type(arg).__name__}"
                )
            flags = set(arg.option_strings)
            if output and (flags & {"-o", "--output"} or arg.dest == "output"):
                raise ValueError(
                    f"skill {name!r}: do not declare -o/--output; the decorator owns it "
                    "(pass output=False to opt out)"
                )

        if not any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in inspect.signature(fn).parameters.values()
        ):
            raise TypeError(
                f"skill {name!r} must accept **kwargs so the decorator can pass "
                "extra runtime information"
            )

        has_bbox = any(arg.dest == "bbox" for arg in arguments)
        parser = argparse.ArgumentParser(
            description=fn.__doc__,
            epilog=f"skill version: {version}",
        )
        if output:
            parser.add_argument(
                "-o",
                "--output",
                dest="output",
                action="append",
                required=True,
                metavar="PATH",
                help="Output path (repeat once per returned artifact).",
            )
        for arg in arguments:
            kwargs = dict(arg.kwargs)
            if arg.dataset_type is not None:
                label = arg.dataset_type.help_label()
                kwargs.setdefault("metavar", "PATH")
                if "help" not in arg.kwargs:
                    kwargs["help"] = f"Input Zarr ({label})."
                elif label not in str(kwargs.get("help", "")):
                    kwargs["help"] = f"{kwargs['help']} [{label}]"
            if arg.dest in standard_args.STANDARD_HELP:
                kwargs = standard_args.add_standard_help(
                    kwargs, standard_args.STANDARD_HELP[arg.dest]
                )
            parser.add_argument(*arg.option_strings, **kwargs)

        strip_dests = {arg.dest for arg in arguments if arg.dataset_type is not None} | {"output"}

        @functools.wraps(fn)
        def wrapper(argv=None):
            argv = sys.argv[1:] if argv is None else list(argv)
            if has_bbox:
                argv = standard_args.rewrite_bbox_argv(argv)

            try:
                probing = any(
                    arg.probe and argv_has_option(argv, arg.option_strings) for arg in arguments
                )
                saved_required = [(action, action.required) for action in parser._actions]
                try:
                    if probing:
                        for action in parser._actions:
                            action.required = False
                    args = parser.parse_args(argv)
                finally:
                    for action, required in saved_required:
                        action.required = required
                params = standard_args.convert_standard_args(args, arguments)
                params, input_paths, upstream, first_ds = open_dataset_params(params, arguments)
                if (
                    any(a.dataset_type is not None and a.dest == "input" for a in arguments)
                    and "input" in params
                ):
                    params["ds"] = params.pop("input")

                output_paths: list[Path] = []
                if output:
                    output_paths = [Path(p) for p in (args.output or [])]
                    if not output_paths:
                        params["output"] = None
                    else:
                        params["output"] = (
                            output_paths[0] if len(output_paths) == 1 else output_paths
                        )

                result = fn(**params)
                if not output or probing or result is None:
                    return result

                results = list(result) if isinstance(result, (list, tuple)) else [result]
                if len(results) != len(output_paths):
                    raise SkillError(
                        f"skill returned {len(results)} value(s), "
                        f"but {len(output_paths)} --output path(s) were given"
                    )

                history = build_history(
                    name, version, args, params, input_paths, upstream, strip_dests
                )
                for value, out_path in zip(results, output_paths, strict=True):
                    write_output(value, out_path, history, first_ds)
                return result
            except SkillError as exc:
                msg = str(exc)
                print(msg if not exc.prefix else f"Error: {msg}", file=sys.stderr)
                sys.exit(exc.exit_code)

        wrapper.parser = parser
        return wrapper

    return decorator


weather_skill.argument = argument  # type: ignore[attr-defined]
