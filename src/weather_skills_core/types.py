"""Types and standard-arg catalog."""

from __future__ import annotations

import re


class Types:
    GRIDDED = "gridded"
    FORECAST = "forecast"
    STATION = "station"
    ANY = "any"
    PNG = "png"


ZARR_TYPES = (Types.GRIDDED, Types.FORECAST, Types.STATION, Types.ANY)
ALL_TYPES = (*ZARR_TYPES, Types.PNG)
STANDARD_ARGS = frozenset({"time", "start_time", "end_time", "bbox", "variable"})

# argparse-style: "any+" (≥1), "any*" (≥0), "any+2" (≥2).
_VARIADIC_RE = re.compile(r"^(.+)([+*])(\d*)$")


def coerce_type(value) -> str:
    text = value if isinstance(value, str) else str(value)
    if text not in ALL_TYPES:
        raise ValueError(f"unknown type {value!r}; valid: {list(ALL_TYPES)}")
    return text


def _coerce_entry(entry, *, name: str):
    if isinstance(entry, (list, tuple, set, frozenset)):
        members = tuple(coerce_type(t) for t in entry)
        if not members:
            raise ValueError(f"{name} union must be non-empty")
        return members
    return coerce_type(entry)


def _parse_variadic(entry) -> tuple[str, int] | None:
    """Parse 'any+' / 'any*' / 'any+2' → (type, min_count). Else None."""
    if not isinstance(entry, str):
        return None
    m = _VARIADIC_RE.fullmatch(entry)
    if not m:
        return None
    base, kind, n = m.group(1), m.group(2), m.group(3)
    if kind == "*":
        if n:
            raise ValueError(f"variadic '*' does not take a count (got {entry!r})")
        return coerce_type(base), 0
    return coerce_type(base), (int(n) if n else 1)


def normalize_io_list(values, *, name: str, allow_variadic: bool = False) -> tuple[list, int | None]:
    """Normalize an inputs=/outputs= list.

    Returns ``(specs, variadic_min)``. Fixed length → ``variadic_min is None``.
    A single ``"any+"`` / ``"any*"`` / ``"any+2"`` entry (argparse ``nargs``
    style) sets the minimum count; the skill receives one list of datasets.
    """
    if values is None:
        return [], None
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a list of Types; got {values!r}")
    values = list(values)
    if len(values) == 1:
        parsed = _parse_variadic(values[0])
        if parsed is not None:
            if not allow_variadic:
                raise ValueError(f"{name} does not support type+/type* variadic form")
            typ, vmin = parsed
            if vmin < 0:
                raise ValueError(f"{name} variadic minimum must be >= 0")
            return [typ], vmin
    if any(isinstance(v, str) and _VARIADIC_RE.fullmatch(v) for v in values):
        raise ValueError(f"{name} variadic form must be a single 'type+', 'type*', or 'type+N' entry")
    return [_coerce_entry(entry, name=name) for entry in values], None


def standard_args() -> tuple[str, ...]:
    return tuple(sorted(STANDARD_ARGS))
