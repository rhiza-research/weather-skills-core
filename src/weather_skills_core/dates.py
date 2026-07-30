"""CLI date values: YYYY-MM-DD or the string 'latest'; stride date lists."""

from __future__ import annotations

import calendar
import re
from datetime import date, datetime, timedelta

from weather_skills_core.errors import DataError, UsageError

_ABS = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_WEEKDAYS = {name.lower(): i for i, name in enumerate(calendar.day_name)}


def np_to_date(value) -> date:
    import numpy as np

    if np.isnat(value):
        raise DataError("time coordinate value is NaT")
    return date.fromisoformat(np.datetime_as_string(value, unit="D"))


def parse_date_value(value: str, *, flag: str = "date") -> date | str:
    if value == "latest":
        return "latest"
    if _ABS.match(value):
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    raise UsageError(f"invalid {flag} value {value!r}: expected YYYY-MM-DD or 'latest'")


def _as_datetime(value) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, date) and not isinstance(value, datetime):
        return datetime(value.year, value.month, value.day)
    import numpy as np

    # numpy datetime64 → midnight UTC datetime (date portion only).
    if isinstance(value, np.datetime64):
        return datetime.fromisoformat(np.datetime_as_string(value, unit="D"))
    text = str(value)
    if "T" in text:
        text = text.split("T", 1)[0]
    return datetime.fromisoformat(text[:10])


def stride_dates(start, end, stride: str = "day"):
    """Inclusive date list from start to end (sheerwater get_dates-style).

    ``stride`` is ``day``/``week``/``month``/``year``, or weekday names
    (``Monday``, ``Monday/Thursday``).
    """
    import numpy as np

    start_dt, end_dt = _as_datetime(start), _as_datetime(end)
    if end_dt < start_dt:
        raise UsageError(f"stride start {start_dt.date()} is after end {end_dt.date()}")

    parts = [p.strip().lower() for p in stride.split("/") if p.strip()]
    if parts and all(p in _WEEKDAYS for p in parts):
        want = {_WEEKDAYS[p] for p in parts}
        out = []
        cur = start_dt
        while cur <= end_dt:
            if cur.weekday() in want:
                out.append(cur)
            cur += timedelta(days=1)
        return np.array(out, dtype="datetime64[ns]")

    key = stride.strip().lower()
    if key == "day":
        delta, months, years = timedelta(days=1), 0, 0
    elif key == "week":
        delta, months, years = timedelta(days=7), 0, 0
    elif key == "month":
        delta, months, years = None, 1, 0
    elif key == "year":
        delta, months, years = None, 0, 1
    else:
        raise UsageError(
            f"invalid stride {stride!r}; use day/week/month/year or weekday "
            "names (e.g. Monday, Monday/Thursday)"
        )

    out = []
    cur = start_dt
    while cur <= end_dt:
        out.append(cur)
        if delta is not None:
            cur = cur + delta
        else:
            y = cur.year + years + (cur.month - 1 + months) // 12
            m = (cur.month - 1 + months) % 12 + 1
            d = min(cur.day, calendar.monthrange(y, m)[1])
            cur = datetime(y, m, d)
    return np.array(out, dtype="datetime64[ns]")
