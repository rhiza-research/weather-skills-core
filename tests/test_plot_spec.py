"""Tests for the Plotly plot spec / compiler."""

from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import make_forecast, make_gridded

from weather_skills_core.plot_spec import (
    PlotSpec,
    dump_spec,
    load_spec,
    overlay_spec,
    panel_shape,
    parse_index,
    spec_from_flags,
)
from weather_skills_core.plot_style import resolve_colorscale


def test_panel_shape_default_caps_columns_at_four():
    assert panel_shape(1) == (1, 1)
    assert panel_shape(3) == (1, 3)
    assert panel_shape(4) == (1, 4)
    assert panel_shape(5) == (2, 4)
    assert panel_shape(8) == (2, 4)


def test_panel_shape_rows_and_columns_allow_blank_cells():
    from weather_skills_core import UsageError

    assert panel_shape(5, rows=2, columns=3) == (2, 3)
    with pytest.raises(UsageError, match="must hold at least"):
        panel_shape(7, rows=2, columns=3)


def test_parse_index_list_and_scalar():
    assert parse_index("step=0,1,2") == {"step": [0, 1, 2]}
    assert parse_index("number=0") == {"number": 0}


def test_spec_roundtrip_json(tmp_path):
    spec = spec_from_flags(
        input_path="/tmp/in.zarr",
        variable="precip",
        style="heatmap",
        title="Precip",
        rows=2,
        columns=3,
    )
    path = tmp_path / "out.plot.json"
    dump_spec(spec, path)
    loaded = load_spec(path)
    assert loaded.data["title"] == "Precip"
    assert loaded.zarr_paths()[0].name == "in.zarr"
    merged = overlay_spec(loaded.data, {"title": "Edited", "plotly": {"layout": {"width": 800}}})
    assert merged["title"] == "Edited"
    assert merged["plotly"]["layout"]["width"] == 800


def test_plot_spec_holder_zarr_paths():
    spec = PlotSpec({"inputs": [{"id": "a", "path": "/tmp/a.zarr"}]})
    assert spec.zarr_paths()[0].as_posix().endswith("a.zarr")


def test_precip_default_colorscale_is_chirps():
    da = make_gridded()["precip"]
    da.attrs["units"] = "mm"
    da.attrs["standard_name"] = "lwe_thickness_of_precipitation_amount"
    scale = resolve_colorscale(da, None)
    assert scale["name"] == "chirps_total"
    assert scale["bounds"][0] == 2


def test_compile_heatmap_facets_time():
    plotly = pytest.importorskip("plotly")
    from weather_skills_core.plot_compile import compile_figure

    ds = make_gridded(n_time=5)
    spec = spec_from_flags(variable="precip", style="heatmap")
    fig, resolved = compile_figure(spec, {"a": ds})
    assert resolved["layout"]["facet"]["columns"] == 4
    assert resolved["layout"]["facet"]["rows"] == 2
    assert resolved["layout"]["facet"]["n_panels"] == 5
    heatmaps = [t for t in fig.data if t.type == "heatmap"]
    assert len(heatmaps) == 5
    assert plotly.__name__


def test_compile_timeseries_forecast_valid_time():
    pytest.importorskip("plotly")
    from weather_skills_core.plot_compile import compile_figure

    ds = make_forecast()
    spec = spec_from_flags(variable="tp", style="timeseries")
    fig, resolved = compile_figure(spec, {"a": ds})
    assert resolved["traces"][0]["type"] == "timeseries"
    assert fig.data[0].type == "scatter"
    assert fig.layout.title.text.endswith("(timeseries)")


def test_compile_applies_plotly_patch():
    pytest.importorskip("plotly")
    from weather_skills_core.plot_compile import compile_figure

    ds = make_gridded(n_time=1)
    spec = spec_from_flags(variable="precip", style="heatmap")
    spec["plotly"] = {"layout": {"title": {"text": "Patched"}}}
    fig, _resolved = compile_figure(spec, {"a": ds})
    assert fig.layout.title.text == "Patched"


def test_export_png_and_sidecar(tmp_path):
    pytest.importorskip("plotly")
    pytest.importorskip("kaleido")
    from weather_skills_core.plot_compile import compile_figure
    from weather_skills_core.plot_export import write_plot_outputs

    ds = make_gridded(n_time=1)
    spec = spec_from_flags(variable="precip", style="heatmap", title="Map")
    fig, resolved = compile_figure(spec, {"a": ds})
    out = tmp_path / "map.png"
    write_plot_outputs(fig, resolved, out, datasets={"a": ds})
    assert out.is_file() and out.stat().st_size > 0
    sidecar = tmp_path / "map.plot.json"
    data = json.loads(sidecar.read_text())
    assert data["title"] == "Map"
    assert data["style"]["colormap"]


def test_compile_contour_uses_contour_traces():
    pytest.importorskip("plotly")
    from weather_skills_core.plot_compile import compile_figure

    ds = make_gridded(n_time=1)
    spec = spec_from_flags(variable="precip", style="contour")
    fig, resolved = compile_figure(spec, {"a": ds})
    assert resolved["traces"][0]["type"] == "contour"
    assert any(t.type == "contour" for t in fig.data)


def test_export_plotly_json_layout_only(tmp_path):
    pytest.importorskip("plotly")
    from weather_skills_core.plot_compile import compile_figure
    from weather_skills_core.plot_export import export_plotly_json

    ds = make_gridded(n_time=1)
    spec = spec_from_flags(variable="precip", style="heatmap")
    fig, _resolved = compile_figure(spec, {"a": ds})
    path = tmp_path / "fig.plotly.json"
    export_plotly_json(fig, path, include_data=False)
    payload = json.loads(path.read_text())
    assert "layout" in payload
    assert all("z" not in (t or {}) for t in payload.get("data") or [])


def test_compile_heatmap_grid_blank_and_heatmap():
    pytest.importorskip("plotly")
    from weather_skills_core.plot_recipes import blank_cell, compile_heatmap_grid, heatmap_cell
    from weather_skills_core.plot_style import resolve_colorscale

    ds = make_gridded(n_time=1)
    da = ds["precip"].isel(time=0)
    scale = resolve_colorscale(da, None)
    scale["label"] = "precip"
    fig = compile_heatmap_grid(
        [[heatmap_cell(da, "latitude", "longitude"), blank_cell("n/a")]],
        extent=[9.5, 13.5, 0.5, 3.5],
        col_titles=["t0", "missing"],
        coloraxes={"coloraxis": scale},
        overlays=False,
    )
    assert any(t.type == "heatmap" for t in fig.data)


def test_compile_line_and_mediogram():
    pytest.importorskip("plotly")
    from weather_skills_core.plot_recipes import compile_line_figure, compile_mediogram

    fig = compile_line_figure(
        [([1, 2, 3], [0.0, 1.0, 2.0], "a"), ([1, 2, 3], [2.0, 1.0, 0.0], "b")],
        title="lines",
        xlabel="t",
        ylabels=["mm", "mm"],
    )
    assert len(fig.data) == 2
    fc = np.arange(12.0).reshape(3, 4)
    mc = np.arange(12.0, 24.0).reshape(3, 4)
    medio = compile_mediogram(fc, mc, ["+0d", "+1d", "+2d", "+3d"], title="Medio")
    boxes = [t for t in medio.data if t.type == "box"]
    assert len(boxes) == 8
    assert medio.layout.title.text == "Medio"
