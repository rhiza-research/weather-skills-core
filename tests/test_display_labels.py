"""Tests for weather_skills_core.display_labels."""

import json

from conftest import make_gridded

from weather_skills_core import display_labels as dl
from weather_skills_core.errors import UsageError


def test_dataset_display_label_from_history():
    ds = make_gridded()
    ds.attrs["weather_skills_history"] = json.dumps([{"skill": "chirps-fetch"}])
    assert dl.dataset_display_label(ds, "fallback") == "CHIRPS"


def test_dataset_display_label_from_source_token():
    ds = make_gridded()
    ds.attrs["weather_skills_source"] = "ecmwf-s2s"
    assert dl.dataset_display_label(ds, "fallback") == "ECMWF S2S"


def test_dataset_display_label_source_wins_over_history():
    ds = make_gridded()
    ds.attrs["weather_skills_source"] = "dynamical:nasa-imerg-analysis-late"
    ds.attrs["weather_skills_history"] = json.dumps([{"skill": "dynamical-fetch"}])
    assert dl.dataset_display_label(ds, "fallback") == "Nasa IMERG Analysis Late"


def test_label_from_source_token_colon_catalog():
    assert (
        dl.label_from_source_token("dynamical:nasa-imerg-analysis-late")
        == "Nasa IMERG Analysis Late"
    )


def test_label_from_source_token_colon_path():
    assert dl.label_from_source_token("kenya-forecasting-data:gefs/gefs_kenya.zarr") == "GEFS"
    assert dl.label_from_source_token("cmip6:ACCESS-CM2/ssp585/r1i1p1f1/Amon/pr/gn") == "Access Cm2"


def test_label_from_source_token_path_stem():
    assert dl.label_from_source_token("/tmp/chirps_2024-01-01.zarr") == "CHIRPS"


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
