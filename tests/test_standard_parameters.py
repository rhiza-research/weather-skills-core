"""Catalog and Types surface."""

import pytest

from weather_skills_core import Types, standard_args
from weather_skills_core.types import normalize_io_list


def test_standard_args_catalog():
    assert set(standard_args()) == {"time", "start_time", "end_time", "bbox", "variable"}


def test_types_constants():
    assert Types.GRIDDED == "gridded"
    assert Types.PNG == "png"


def test_inputs_variadic_plus_star():
    specs, vmin = normalize_io_list([Types.ANY + "+"], name="inputs", allow_variadic=True)
    assert specs == ["any"]
    assert vmin == 1
    specs, vmin = normalize_io_list([Types.ANY + "*"], name="inputs", allow_variadic=True)
    assert specs == ["any"]
    assert vmin == 0
    specs, vmin = normalize_io_list([Types.ANY + "+2"], name="inputs", allow_variadic=True)
    assert specs == ["any"]
    assert vmin == 2
    with pytest.raises(ValueError, match="does not support"):
        normalize_io_list([Types.ANY + "*"], name="outputs")
    with pytest.raises(ValueError, match="does not take a count"):
        normalize_io_list([Types.ANY + "*2"], name="inputs", allow_variadic=True)
    with pytest.raises(ValueError, match="single"):
        normalize_io_list([Types.ANY, Types.ANY + "+"], name="inputs", allow_variadic=True)
