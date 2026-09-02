"""Tests for weather_skills_core.display_labels."""

import json

from weather_skills_core import display_labels as dl
from weather_skills_core.errors import UsageError
from conftest import make_gridded


def test_dataset_display_label_from_history():
    ds = make_gridded()
    ds.attrs["weather_skills_history"] = json.dumps([{"skill": "chirps-fetch"}])
    assert dl.dataset_display_label(ds, "fallback") == "CHIRPS"


def test_dataset_display_label_from_source_token():
    ds = make_gridded()
    ds.attrs["weather_skills_source"] = "ecmwf-s2s"
    assert dl.dataset_display_label(ds, "fallback") == "ECMWF S2S"


def test_dataset_display_label_fallback():
    assert dl.dataset_display_label(make_gridded(), "input 1") == "input 1"


def test_resolve_input_labels_empty():
    assert dl.resolve_input_labels(None, 2) == [None, None]
    assert dl.resolve_input_labels([], 2) == [None, None]


def test_resolve_input_labels_count_mismatch():
    try:
        dl.resolve_input_labels(["A"], 2)
    except UsageError as exc:
        assert "expected 2 --label" in str(exc)
    else:
        raise AssertionError("expected UsageError")


def test_combine_display_labels():
    assert dl.combine_display_labels(["ECMWF S2S", "ECMWF S2S"]) == "ECMWF S2S"
    assert dl.combine_display_labels(["GEFS", "ECMWF S2S"]) == "GEFS / ECMWF S2S"
