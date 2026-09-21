"""Plot PNG hash and plotted-data null check."""

import numpy as np
import pytest
from conftest import make_gridded
from PIL import Image

from weather_skills_core.plot.qa import data_status, hash_png_pixels, report_figure


def test_hash_png_pixels_is_stable_and_changes_with_pixels(tmp_path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(a)
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(b)
    Image.new("RGB", (8, 8), color=(11, 20, 30)).save(tmp_path / "c.png")
    assert hash_png_pixels(a) == hash_png_pixels(b)
    assert hash_png_pixels(a) != hash_png_pixels(tmp_path / "c.png")


def test_data_status_not_null():
    status, detail = data_status({"a": make_gridded(n_time=1)})
    assert status == "not null"
    assert "precip 12/12 finite" in detail


def test_data_status_null():
    status, detail = data_status({"a": make_gridded(n_time=1, fill=np.nan)})
    assert status == "NULL"
    assert "precip 0/12 finite" in detail


def test_data_status_no_datasets():
    assert data_status(None) == ("not checked", "no numeric data variables")


def test_report_figure_prints_hash_and_not_null(tmp_path, capsys):
    png = tmp_path / "fig.png"
    Image.new("RGB", (4, 4), color=(1, 2, 3)).save(png)
    digest = report_figure(png, {"a": make_gridded(n_time=1)})
    out = capsys.readouterr().out
    assert f"plot hash: {digest}" in out
    assert "data: not null (precip 12/12 finite)" in out
    assert "all NaN" not in out


def test_report_figure_prints_null(tmp_path, capsys):
    png = tmp_path / "fig.png"
    Image.new("RGB", (4, 4), color=(255, 255, 255)).save(png)
    report_figure(png, make_gridded(n_time=1, fill=np.nan))
    out = capsys.readouterr().out
    assert "data: NULL (precip 0/12 finite)" in out
    assert "inspect-zarr" in out


def test_export_prints_hash_and_not_null(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    from weather_skills_core.plot import compile, export
    from weather_skills_core.plot.spec import spec_from_flags

    ds = make_gridded(n_time=1)
    compiled = compile(spec_from_flags(variable="precip", kind="heatmap"), {"a": ds})
    out = tmp_path / "map.png"
    export(compiled, out, datasets={"a": ds})
    text = capsys.readouterr().out
    assert out.is_file()
    assert text.startswith("plot hash: ")
    assert "data: not null (precip 12/12 finite)" in text


def test_export_prints_null_when_field_is_all_nan(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    from weather_skills_core.plot import compile, export
    from weather_skills_core.plot.spec import spec_from_flags

    ds = make_gridded(n_time=1, fill=np.nan)
    compiled = compile(spec_from_flags(variable="precip", kind="heatmap"), {"a": ds})
    export(compiled, tmp_path / "empty.png", datasets={"a": ds})
    text = capsys.readouterr().out
    assert "data: NULL (precip 0/12 finite)" in text
