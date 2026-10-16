"""Shared CLI flags (``--bbox``, dates, …) for the decorator and linting catalog."""

from __future__ import annotations

from dataclasses import dataclass

from weather_skills_core.errors import UsageError
from weather_skills_core.standard_utils import parse_bbox, parse_date

STANDARD_HELP = {
    "bbox": "N/W/S/E decimal degrees.",
    "date": "Absolute date YYYY-MM-DD.",
    "start_time": "Range start, inclusive. Absolute YYYY-MM-DD.",
    "end_time": "Range end, inclusive. Absolute YYYY-MM-DD.",
}


@dataclass(frozen=True)
class StandardParameter:
    name: str
    flags: tuple


def standard_parameters():
    """Shared flags (``--bbox``, dates, …) used by the decorator (and the linting package).

    Input/output paths are ordinary ``@weather_skill.argument`` declarations
    (``type=Dataset(...)`` / ``type=Path``); they are not catalogued here.
    """
    return (
        StandardParameter("start_time", ("--start-time",)),
        StandardParameter("end_time", ("--end-time",)),
        StandardParameter("date", ("--date",)),
        StandardParameter("bbox", ("--bbox",)),
        StandardParameter("variable", ("--variable", "-v")),
    )


def rewrite_bbox_argv(argv):
    """Turn ``--bbox N/W/S/E`` into ``--bbox=…`` so a negative north is not a flag."""
    out = []
    i = 0
    while i < len(argv):
        if argv[i] == "--bbox" and i + 1 < len(argv):
            out.append(f"--bbox={argv[i + 1]}")
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return out


def add_standard_help(kwargs: dict, canonical: str) -> dict:
    """Attach standard help text to an ``add_argument`` kwargs dict."""
    out = dict(kwargs)
    existing = out.get("help")
    if existing:
        text = str(existing).rstrip()
        if canonical not in text:
            out["help"] = f"{text} {canonical}"
    else:
        out["help"] = canonical
    return out


def convert_standard_args(args, arguments) -> dict:
    """Build skill kwargs from parsed CLI.

    ``--bbox`` becomes an ``(N, W, S, E)`` float tuple. Dates become ``datetime.date``.
    Named places are not a decorator flag — compose with the resolve-region skill
    and pass the printed bbox here. Relative dates are not a decorator flag —
    compose with the resolve-time skill and pass the printed `--date` /
    `--start-time` / `--end-time`.

    Raises UsageError if ``start_time`` is after ``end_time``.
    """
    params = {}
    for arg in arguments:
        dest = arg.dest
        raw = getattr(args, dest)
        if dest == "bbox":
            params[dest] = parse_bbox(raw) if raw is not None else None
        elif dest in ("date", "start_time", "end_time"):
            params[dest] = parse_date(raw) if raw is not None else None
        else:
            params[dest] = raw

    start = params.get("start_time")
    end = params.get("end_time")
    if start is not None and end is not None and start > end:
        raise UsageError(f"--start-time {start.isoformat()} is after --end-time {end.isoformat()}.")
    return params
