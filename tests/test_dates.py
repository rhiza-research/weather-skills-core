"""Absolute YYYY-MM-DD + latest passthrough; stride dates."""

import datetime

import numpy as np
import pytest

from weather_skills_core.dates import parse_date_value, stride_dates
from weather_skills_core.errors import UsageError


def test_absolute():
    assert parse_date_value("2026-01-15") == datetime.date(2026, 1, 15)


def test_latest_passthrough():
    assert parse_date_value("latest") == "latest"


def test_rejects_offsets():
    with pytest.raises(UsageError, match="YYYY-MM-DD"):
        parse_date_value("latest-2w")


def test_stride_dates_weekdays():
    # 2026-01-05 is Monday; through 2026-01-15 includes Mon 5/12 and Thu 8/15.
    out = stride_dates("2026-01-05", "2026-01-15", stride="Monday/Thursday")
    assert list(out) == [
        np.datetime64("2026-01-05"),
        np.datetime64("2026-01-08"),
        np.datetime64("2026-01-12"),
        np.datetime64("2026-01-15"),
    ]


def test_stride_dates_week():
    out = stride_dates("2026-01-01", "2026-01-22", stride="week")
    assert list(out) == [
        np.datetime64("2026-01-01"),
        np.datetime64("2026-01-08"),
        np.datetime64("2026-01-15"),
        np.datetime64("2026-01-22"),
    ]
