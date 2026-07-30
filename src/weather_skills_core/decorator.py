"""@weather_skill: argparse, I/O, cache, provenance, write."""

from __future__ import annotations

import argparse
import functools
import inspect
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from weather_skills_core import dates as _dates
from weather_skills_core import dataset as _dataset
from weather_skills_core import provenance as _provenance
from weather_skills_core.errors import DataError, SkillError, UsageError
from weather_skills_core.types import STANDARD_ARGS, Types, normalize_io_list, standard_args
from weather_skills_core.util import require_env

_ARG_SPECS_ATTR = "_weather_skill_argument_specs"


@dataclass
class EntryOverride:
    args: dict


def argument(*args, **kwargs):
    """Stack under @weather_skill; same API as parser.add_argument."""

    def deco(fn):
        if not hasattr(fn, _ARG_SPECS_ATTR):
            setattr(fn, _ARG_SPECS_ATTR, [])
        getattr(fn, _ARG_SPECS_ATTR).append((args, kwargs))
        return fn

    return deco


def rewrite_bbox_argv(argv):
    out, i = [], 0
    while i < len(argv):
        if argv[i] == "--bbox" and i + 1 < len(argv):
            out.append(f"--bbox={argv[i + 1]}")
            i += 2
        else:
            out.append(argv[i])
            i += 1
    return out


def _sanitize(d: dict) -> dict:
    return json.loads(json.dumps(d, default=str))


def _date_entry(v):
    return v.isoformat() if isinstance(v, date) else v


def _infer_dest(a_args, a_kwargs):
    if "dest" in a_kwargs:
        return a_kwargs["dest"]
    opt = next((x for x in a_args if str(x).startswith("--")), a_args[0] if a_args else None)
    if opt is None:
        raise ValueError(f"could not infer dest for {a_args}")
    return str(opt).lstrip("-").replace("-", "_")


def _split_result(result, n_outputs):
    override = None
    if isinstance(result, tuple):
        parts = list(result)
        if parts and isinstance(parts[-1], EntryOverride):
            override = parts.pop()
        if len(parts) != n_outputs:
            raise TypeError(f"expected {n_outputs} output(s); got {len(parts)}")
        return parts, override
    if n_outputs == 1:
        return [result], None
    if isinstance(result, list) and len(result) == n_outputs:
        return result, None
    raise TypeError(f"expected {n_outputs} outputs; got {type(result).__name__}")


def _rm(path: Path):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _check_catalog(required, optional):
    required, optional = tuple(required or ()), tuple(optional or ())
    for n in (*required, *optional):
        if n not in STANDARD_ARGS:
            raise ValueError(f"unknown standard arg {n!r}; valid: {sorted(STANDARD_ARGS)}")
    if set(required) & set(optional):
        raise ValueError("arg listed in both required_args and optional_args")
    listed = set(required) | set(optional)
    ranged = "start_time" in listed or "end_time" in listed
    if ranged and listed & {"start_time", "end_time"} != {"start_time", "end_time"}:
        raise ValueError("start_time and end_time must be listed together")
    if ranged and ("start_time" in required) != ("end_time" in required):
        raise ValueError("start_time and end_time must share required/optional")
    if "time" in listed and ranged:
        raise ValueError("time is mutually exclusive with start_time/end_time")
    return required, optional


def weather_skill(
    *,
    name: str,
    version: str,
    inputs=None,
    outputs=None,
    required_args=(),
    optional_args=(),
    exclude_args=(),
    required_env=(),
    check_cache: bool = True,
):
    input_specs, input_variadic_min = normalize_io_list(inputs, name="inputs", allow_variadic=True)
    output_specs, _ = normalize_io_list(outputs, name="outputs")
    required, optional = _check_catalog(required_args, optional_args)
    catalog = required + optional
    exclude_args = tuple(exclude_args or ())
    required_env = tuple(required_env or ())
    has_artifact = bool(output_specs)
    if not has_artifact and input_specs:
        raise ValueError("no-artifact skills do not declare inputs")

    def decorator(fn):
        arg_specs = list(getattr(fn, _ARG_SPECS_ATTR, []) or [])
        custom = []
        reserved = {"check_cache", "input", "output", *catalog}
        for a, kw in arg_specs:
            dest = _infer_dest(a, kw)
            if dest in reserved or dest in STANDARD_ARGS:
                raise ValueError(f"custom dest {dest!r} shadows a reserved name")
            custom.append(dest)

        params = inspect.signature(fn).parameters
        if "check_cache" in params:
            raise ValueError("check_cache must not appear in the skill signature")
        missing = [n for n in (*catalog, *custom) if n not in params]
        if missing:
            raise ValueError(f"skill function missing arg(s) {missing}")

        @functools.wraps(fn)
        def wrapper(argv=None):
            argv = list(sys.argv[1:] if argv is None else argv)
            if "bbox" in catalog:
                argv = rewrite_bbox_argv(argv)

            p = argparse.ArgumentParser(prog=name, description=fn.__doc__, epilog=f"{name} {version}")
            ni, no = len(input_specs), len(output_specs)
            if input_variadic_min is not None:
                p.add_argument(
                    "--input", "-i", action="append", default=[],
                    required=input_variadic_min > 0,
                )
            elif ni > 1:
                p.add_argument("--input", "-i", action="append", required=True)
            elif ni == 1:
                p.add_argument("--input", "-i", required=True)
            if no == 1:
                p.add_argument("--output", "-o", required=True)
            elif no > 1:
                p.add_argument("--output", "-o", nargs=no, required=True, metavar="PATH")
            if "time" in catalog:
                p.add_argument("--time", required="time" in required, help="YYYY-MM-DD or latest")
            if "start_time" in catalog:
                p.add_argument("--start", dest="start_time", required="start_time" in required,
                               help="YYYY-MM-DD or latest")
                p.add_argument("--end", dest="end_time", required="end_time" in required,
                               help="YYYY-MM-DD or latest")
            if "bbox" in catalog:
                p.add_argument("--bbox", required="bbox" in required, help="N/W/S/E")
            if "variable" in catalog:
                p.add_argument("--variable", "-v", action="append", required="variable" in required)
            if has_artifact:
                p.add_argument("--check-cache", action=argparse.BooleanOptionalAction,
                               default=check_cache)
            for a, kw in arg_specs:
                p.add_argument(*a, **kw)

            try:
                ns = p.parse_args(argv)
            except SystemExit as e:
                raise SystemExit(e.code if isinstance(e.code, int) else 2) from None

            try:
                return _run(fn, ns, custom)
            except UsageError as e:
                print(f"error: {e}", file=sys.stderr)
                raise SystemExit(2) from None
            except (DataError, SkillError) as e:
                print(f"error: {e}", file=sys.stderr)
                raise SystemExit(1) from None

        def _run(fn, ns, custom):
            import xarray as xr

            ni, no = len(input_specs), len(output_specs)
            if required_env:
                require_env(*required_env)

            kw, entry = {}, {}
            if "time" in catalog:
                raw = getattr(ns, "time", None)
                kw["time"] = None if raw is None else _dates.parse_date_value(raw, flag="--time")
                if raw is not None:
                    entry["time"] = _date_entry(kw["time"])
            if "start_time" in catalog:
                rs, re_ = getattr(ns, "start_time", None), getattr(ns, "end_time", None)
                if rs is None and re_ is None:
                    kw["start_time"] = kw["end_time"] = None
                elif rs is None or re_ is None:
                    raise UsageError("--start and --end must be given together.")
                else:
                    kw["start_time"] = _dates.parse_date_value(rs, flag="--start")
                    kw["end_time"] = _dates.parse_date_value(re_, flag="--end")
                    entry["start_time"] = _date_entry(kw["start_time"])
                    entry["end_time"] = _date_entry(kw["end_time"])
                    if isinstance(kw["start_time"], date) and isinstance(kw["end_time"], date):
                        if kw["start_time"] > kw["end_time"]:
                            raise UsageError("resolved --start is after --end")
            if "bbox" in catalog:
                raw = getattr(ns, "bbox", None)
                kw["bbox"] = _dataset.parse_bbox(raw) if raw is not None else None
                if raw is not None:
                    entry["bbox"] = raw
            if "variable" in catalog:
                kw["variable"] = getattr(ns, "variable", None)
                if kw["variable"] is not None:
                    entry["variable"] = list(kw["variable"])
            for dest in custom:
                kw[dest] = getattr(ns, dest)
                if dest not in exclude_args:
                    entry[dest] = kw[dest]

            in_paths = []
            if ni == 0:
                pass
            elif input_variadic_min is not None:
                vals = ns.input or []
                if len(vals) < input_variadic_min:
                    raise UsageError(
                        f"--input must be passed at least {input_variadic_min} times; got {len(vals)}"
                    )
                in_paths = [Path(x) for x in vals]
                entry["input"] = [str(x) for x in in_paths]
            elif ni == 1:
                in_paths = [Path(ns.input)]
                entry["input"] = str(in_paths[0])
            else:
                vals = ns.input or []
                if len(vals) != ni:
                    raise UsageError(f"--input must be passed {ni} times; got {len(vals)}")
                in_paths = [Path(x) for x in vals]
                entry["input"] = [str(x) for x in in_paths]
            for dest in exclude_args:
                entry.pop(dest, None)
            entry = _sanitize(entry)

            out_paths = []
            if no == 1:
                out_paths = [Path(ns.output)]
            elif no > 1:
                out_paths = [Path(x) for x in ns.output]

            datasets, chains = [], []
            if input_variadic_min is not None:
                path_specs = [input_specs[0]] * len(in_paths)
            else:
                path_specs = input_specs
            for path, spec in zip(in_paths, path_specs, strict=True):
                allowed = spec if isinstance(spec, tuple) else (spec,)
                ds = xr.open_zarr(path, consolidated=False)
                _dataset.validate_type(ds, allowed, str(path))
                datasets.append(ds)
                chains.append(_provenance.load_history(path) or [])

            multi = input_variadic_min is not None or ni > 1
            if ni == 0:
                entry_input, fetcher, upstream = None, True, None
            elif multi:
                fetcher, upstream = False, []
                entry_input = [
                    {"basename": p.name, "history": ch}
                    for p, ch in zip(in_paths, chains, strict=True)
                ]
            else:
                fetcher, upstream = False, chains[0]
                entry_input = {"basename": in_paths[0].name}

            prov = _provenance.build_entry(name, version, entry, entry_input)
            if has_artifact and getattr(ns, "check_cache", True):
                if _provenance.cache_hit(out_paths[0], prov, upstream, fetcher=fetcher):
                    print(f"cache hit: skipping {name}; using {out_paths[0]}", file=sys.stderr)
                    return 0

            # Variadic skills receive the dataset list as the first positional.
            result = fn(datasets, **kw) if input_variadic_min is not None else fn(*datasets, **kw)
            if not has_artifact:
                return 0

            items, override = _split_result(result, no)
            if override:
                prov["args"] = _sanitize({**prov["args"], **override.args})

            for item, out, spec in zip(items, out_paths, output_specs, strict=True):
                _write(item, out, spec, prov, upstream, fetcher)
            print(f"Wrote: {', '.join(str(p) for p in out_paths)}", file=sys.stderr)
            return 0

        def _write(item, out, spec, prov, upstream, fetcher):
            import xarray as xr

            members = spec if isinstance(spec, tuple) else (spec,)
            history = [prov] if fetcher or upstream == [] else (upstream or []) + [prov]

            if isinstance(item, (str, Path)):
                path = Path(item)
                if not path.exists():
                    raise DataError(f"returned path does not exist: {path}")
                if path.resolve() != out.resolve() and path != out:
                    raise DataError(f"returned path {path} != output {out}")
                return

            if members == (Types.PNG,) or hasattr(item, "savefig"):
                _rm(out)
                item.savefig(out, dpi=150, metadata={_provenance.HISTORY_ATTR: json.dumps(history)})
                return

            if not isinstance(item, xr.Dataset):
                raise TypeError(f"unsupported return type {type(item).__name__}")
            actual = _dataset.detect_type(item)
            if Types.ANY not in members and actual not in members:
                raise DataError(f"output is {actual}; declared {' or '.join(members)}")
            _provenance.stamp_zarr(item, history)
            _rm(out)
            try:
                item.to_zarr(out, mode="w", consolidated=True)
            except Exception:
                if out.exists():
                    _rm(out)
                raise

        return wrapper

    return decorator


weather_skill.argument = argument  # type: ignore[attr-defined]


__all__ = [
    "EntryOverride",
    "argument",
    "rewrite_bbox_argv",
    "standard_args",
    "weather_skill",
]
