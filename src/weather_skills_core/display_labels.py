"""Human-readable dataset labels for plots and legends."""

from __future__ import annotations

import json
import re
from pathlib import Path

from weather_skills_core.errors import UsageError

_DATE_TAIL = re.compile(r"[_-]\d{4}-\d{2}-\d{2}(?:[_-]\d{2}-\d{2}-\d{2})?$")
_VAR_TAIL = re.compile(r"[_-](?:precip|tp)$", re.I)
_ACRONYMS = frozenset({"ecmwf", "s2s", "ghcn", "cmip6", "gefs", "oisst", "imerg", "smap"})


def _prettify_token(text: str) -> str:
    parts = [p for p in re.split(r"[_-]+", text.strip()) if p]
    if not parts:
        return text
    out = []
    for part in parts:
        low = part.lower()
        if low in _ACRONYMS:
            out.append(part.upper())
        elif low == "chirps":
            out.append("CHIRPS")
        else:
            out.append(part.capitalize())
    return " ".join(out)


def _normalize_stem(stem: str) -> str:
    base = _DATE_TAIL.sub("", stem)
    return _VAR_TAIL.sub("", base)


def _scheme_payload(token: str) -> str:
    """Return the product id from a ``scheme:tail`` source token."""
    head, _, tail = token.partition(":")
    payload = (tail or head).strip()
    if "/" in payload or "\\" in payload:
        first = re.split(r"[/\\]", payload, maxsplit=1)[0].strip()
        return first or payload
    if payload.endswith(".zarr"):
        return Path(payload).stem
    return payload


def label_from_history(ds) -> str | None:
    raw = ds.attrs.get("weather_skills_history")
    if not raw:
        return None
    try:
        history = json.loads(raw) if isinstance(raw, str) else list(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not history or not isinstance(history[0], dict):
        return None
    skill = history[0].get("skill")
    if not isinstance(skill, str) or not skill.strip():
        return None
    skill = skill.strip()
    if skill.endswith("-fetch"):
        return _prettify_token(skill[: -len("-fetch")])
    return _prettify_token(skill)


def label_from_source_token(token: str) -> str:
    text = token.strip()
    if not text:
        return ""
    if ":" in text:
        return _prettify_token(_normalize_stem(_scheme_payload(text)))
    if "/" in text or "\\" in text or text.endswith(".zarr"):
        text = Path(text).stem
    return _prettify_token(_normalize_stem(text))


def dataset_display_label(ds, fallback) -> str:
    """Short product name from ``weather_skills_source``, provenance, or ``fallback``."""
    src = ds.attrs.get("weather_skills_source")
    if isinstance(src, str) and src.strip():
        label = label_from_source_token(src)
        if label:
            return label
    enc = ds.encoding.get("source")
    if isinstance(enc, str) and enc.strip():
        label = label_from_source_token(enc)
        if label:
            return label
    label = label_from_history(ds)
    if label:
        return label
    return fallback() if callable(fallback) else str(fallback)


def resolve_input_labels(
    labels: list[str] | None,
    n: int,
    *,
    input_flag: str = "--input",
) -> list[str | None]:
    """Map optional ``--label`` values to *n* inputs; ``None`` slots use auto labels."""
    if not labels:
        return [None] * n
    if len(labels) != n:
        raise UsageError(f"expected {n} --label values (one per {input_flag}), got {len(labels)}")
    return list(labels)


def combine_display_labels(labels: list[str]) -> str:
    """Join unique labels for a shared row title (e.g. one forecast row)."""
    unique = list(dict.fromkeys(label for label in labels if label))
    if not unique:
        return ""
    if len(unique) == 1:
        return unique[0]
    return " / ".join(unique)
