"""Provenance: weather_skills_history chains on zarr/PNG artifacts."""

import json
import sys
from pathlib import Path

HISTORY_ATTR = "weather_skills_history"
# CF global attribute; fetchers set this on the Dataset (not via the decorator).
SOURCE_ATTR = "source"

_ENTRY_KNOWN_KEYS = {"skill", "version", "args", "input"}
_INPUT_ITEM_KNOWN_KEYS = {"basename", "history"}


def read_chain(raw, *, strict: bool = False, label: str = "artifact") -> list | None:
    """Parse history JSON. Strict raises ValueError; else warn and return None."""
    try:
        chain = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        if strict:
            raise ValueError("value is not valid JSON") from None
        chain = None
    if not isinstance(chain, list):
        if strict:
            raise ValueError("value is not a JSON array")  # noqa: TRY004
        print(
            f"ignoring malformed weather_skills_history on {label}; "
            "run `provenance --check` for details",
            file=sys.stderr,
        )
        return None
    return chain


def _validate_input(value, loc: str, violations: list, notes: list) -> None:
    if value is None:
        return

    def _check_item(item, item_loc: str) -> None:
        if not isinstance(item, dict):
            violations.append(f"{item_loc}: input entry is not an object")
            return
        if "basename" not in item:
            violations.append(f"{item_loc}: missing required key 'basename'")
        elif not isinstance(item["basename"], str):
            violations.append(f"{item_loc}.basename: must be a string")
        if "history" in item:
            _validate_chain(item["history"], f"{item_loc}.history", violations, notes)
        for key in item:
            if key not in _INPUT_ITEM_KNOWN_KEYS:
                notes.append(f"{item_loc}: unknown key {key!r}")

    if isinstance(value, list):
        for j, item in enumerate(value):
            _check_item(item, f"{loc}[{j}]")
        return
    if isinstance(value, dict):
        _check_item(value, loc)
        return
    violations.append(f"{loc}: must be null, an object, or an array of objects")


def _validate_chain(chain, loc: str, violations: list, notes: list) -> None:
    if not isinstance(chain, list):
        violations.append(f"{loc}: value is not a JSON array")
        return
    for i, entry in enumerate(chain):
        eloc = f"{loc}[{i}]"
        if not isinstance(entry, dict):
            violations.append(f"{eloc}: entry is not an object")
            continue
        if "skill" not in entry:
            violations.append(f"{eloc}: missing required key 'skill'")
        elif not isinstance(entry["skill"], str):
            violations.append(f"{eloc}.skill: must be a string")
        elif not entry["skill"]:
            violations.append(f"{eloc}.skill: must be a non-empty string")
        if "version" not in entry:
            violations.append(f"{eloc}: missing required key 'version'")
        elif not isinstance(entry["version"], str):
            violations.append(f"{eloc}.version: must be a string")
        if "args" not in entry:
            violations.append(f"{eloc}: missing required key 'args'")
        elif not isinstance(entry["args"], dict):
            violations.append(f"{eloc}.args: must be an object")
        if "input" not in entry:
            violations.append(f"{eloc}: missing required key 'input'")
        else:
            _validate_input(entry["input"], f"{eloc}.input", violations, notes)
        for key in entry:
            if key not in _ENTRY_KNOWN_KEYS:
                notes.append(f"{eloc}: unknown key {key!r}")


def validate_chain(chain, loc: str) -> tuple[list, list]:
    """Return (violations, notes) for a parsed chain."""
    violations: list = []
    notes: list = []
    _validate_chain(chain, loc, violations, notes)
    return violations, notes


def load_history(zarr_path: Path) -> list:
    """Read history from a zarr; empty list on miss or malformation."""
    zarr_path = Path(zarr_path)
    try:
        import xarray as xr

        with xr.open_zarr(zarr_path, consolidated=False) as ds:
            raw = ds.attrs.get(HISTORY_ATTR)
    except (OSError, KeyError, ValueError):
        return []
    if not raw:
        return []
    parsed = read_chain(raw, label=str(zarr_path))
    return [] if parsed is None else parsed


def build_entry(skill: str, version: str, args: dict, input) -> dict:
    """Build {skill, version, args, input}."""
    return {"skill": skill, "version": version, "args": args, "input": input}


def _chained_input_match(last_input, entry_input) -> bool:
    """Match on basename (+ nested history for multi-input). No content hashing."""
    if isinstance(entry_input, list):
        if not isinstance(last_input, list) or len(last_input) != len(entry_input):
            return False
        return all(
            isinstance(li, dict)
            and li.get("basename") == ei["basename"]
            and li.get("history") == ei["history"]
            for li, ei in zip(last_input, entry_input, strict=True)
        )
    if last_input is not None and not isinstance(last_input, dict):
        return False
    last_input = last_input or {}
    entry_input = entry_input or {}
    return last_input.get("basename") == entry_input.get("basename")


def cache_hit(out: Path, entry: dict, upstream: list | None = None, *, fetcher: bool = False) -> bool:
    """True if out's history matches this entry (fetcher or upstream+[entry])."""
    out = Path(out)
    if not out.exists():
        return False

    history = load_history(out)
    if fetcher:
        if not history or not isinstance(history[0], dict):
            return False
        existing = history[0]
        return (
            existing.get("skill") == entry["skill"]
            and existing.get("version") == entry["version"]
            and existing.get("args") == entry["args"]
            and existing.get("input") == entry["input"]
        )

    upstream = upstream or []
    if len(history) != len(upstream) + 1 or history[:-1] != upstream:
        return False
    last = history[-1]
    if not isinstance(last, dict):
        return False
    return (
        last.get("skill") == entry["skill"]
        and last.get("version") == entry["version"]
        and last.get("args") == entry["args"]
        and _chained_input_match(last.get("input"), entry.get("input"))
    )


def stamp_zarr(ds, history: list) -> None:
    """Set weather_skills_history on ds. Fetchers set CF ``source`` themselves."""
    ds.attrs[HISTORY_ATTR] = json.dumps(history, sort_keys=True)
