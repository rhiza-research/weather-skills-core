"""Tests for the matplotlib plot spec / compiler."""

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
    merged = overlay_spec(
        loaded.data, {"title": "Edited", "patch": {"layout": {"title": "Edited"}}}
    )
    assert merged["title"] == "Edited"
    assert merged["patch"]["layout"]["title"] == "Edited"


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
    assert scale["colors"][0] == "#ffffff"
    assert scale["colors"][1] == "#c8ffbe"
    assert scale["colors"][-1] == "#ffe6e6"
    aliased = resolve_colorscale(da, "ppt_total")
    assert aliased["colors"] == scale["colors"]
    assert aliased["bounds"] == scale["bounds"]


def test_precip_anomaly_and_poa_and_spi_colorscales():
    da = make_gridded(fill=-20.0)["precip"]
    da.attrs["units"] = "mm"
    da.attrs["standard_name"] = "lwe_thickness_of_precipitation_amount"
    anom = resolve_colorscale(da, None)
    assert anom["name"] == "chirps_anom"
    assert anom["colors"][0] == "#c00000"
    assert anom["colors"][3] == "#ffe878"
    assert resolve_colorscale(da, "ppt_anomaly")["colors"] == anom["colors"]

    poa = make_gridded(fill=80.0)["precip"]
    poa.attrs.update(
        units="percent",
        standard_name="lwe_thickness_of_precipitation_amount",
        long_name="percent of normal rainfall",
    )
    poa_scale = resolve_colorscale(poa, None)
    assert poa_scale["name"] == "ppt_poa"
    assert poa_scale["bounds"][0] == 30
    assert poa_scale["colors"][0] == "#e1beb4"

    spi = make_gridded(fill=-1.0, name="spi")["spi"]
    spi.attrs["long_name"] = "Standardized Precipitation Index"
    spi_scale = resolve_colorscale(spi, None)
    assert spi_scale["name"] == "spi"
    assert spi_scale["bounds"][0] == -2.5
    assert spi_scale["colors"][0] == "#730000"


def test_rank_colorscale_bounds_depend_on_n_seasons():
    from weather_skills_core.plot_style import rank_colorscale

    scale = rank_colorscale(40)
    assert scale["name"] == "ppt_rank"
    assert scale["bounds"][0] == -0.5
    assert scale["bounds"][-2] == 39.5
    assert len(scale["colors"]) == len(scale["bounds"]) - 1


def _quadmeshes(fig):
    from matplotlib.collections import QuadMesh

    return [
        c
        for ax in fig.axes
        for c in ax.collections
        if isinstance(c, QuadMesh) and ax.get_label() != "<colorbar>"
    ]


def test_compile_heatmap_facets_time():
    pytest.importorskip("matplotlib")
    from weather_skills_core.plot_compile import compile_figure

    ds = make_gridded(n_time=5)
    spec = spec_from_flags(variable="precip", style="heatmap")
    fig, resolved = compile_figure(spec, {"a": ds})
    assert resolved["layout"]["facet"]["columns"] == 4
    assert resolved["layout"]["facet"]["rows"] == 2
    assert resolved["layout"]["facet"]["n_panels"] == 5
    assert len(_quadmeshes(fig)) == 5


def test_compile_timeseries_forecast_valid_time():
    pytest.importorskip("matplotlib")
    from weather_skills_core.plot_compile import compile_figure

    ds = make_forecast()
    spec = spec_from_flags(variable="tp", style="timeseries")
    fig, resolved = compile_figure(spec, {"a": ds})
    assert resolved["traces"][0]["type"] == "timeseries"
    assert fig.axes[0].lines
    assert fig._suptitle.get_text().endswith("(timeseries)")


def test_compile_applies_patch():
    pytest.importorskip("matplotlib")
    from weather_skills_core.plot_compile import compile_figure

    ds = make_gridded(n_time=1)
    spec = spec_from_flags(variable="precip", style="heatmap")
    spec["patch"] = {"layout": {"title": {"text": "Patched"}}}
    fig, _resolved = compile_figure(spec, {"a": ds})
    assert fig._suptitle.get_text() == "Patched"


def _colorbar_box(fig):
    fig.canvas.draw()
    ax = next(a for a in fig.axes if a.get_label() == "<colorbar>")
    return ax.get_position()


def test_compile_applies_colorbar_size_patch():
    pytest.importorskip("matplotlib")
    from weather_skills_core.plot_compile import compile_figure

    ds = make_gridded(n_time=2)
    spec = spec_from_flags(variable="precip", style="heatmap", columns=2)
    fig_default, _ = compile_figure(spec, {"a": ds})
    default = _colorbar_box(fig_default)
    spec["patch"] = {"layout": {"coloraxis": {"colorbar": {"len": 0.45, "thickness": 12}}}}
    fig_patched, resolved = compile_figure(spec, {"a": ds})
    patched = _colorbar_box(fig_patched)
    assert resolved["layout"]["colorbar"]["len"] == 0.45
    assert patched.width < default.width
    assert patched.height < default.height * 0.6


def test_export_png_and_sidecar(tmp_path):
    pytest.importorskip("matplotlib")
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


def test_compile_contour_uses_contour_collections():
    pytest.importorskip("matplotlib")
    from matplotlib.contour import QuadContourSet

    from weather_skills_core.plot_compile import compile_figure

    ds = make_gridded(n_time=1)
    spec = spec_from_flags(variable="precip", style="contour")
    fig, resolved = compile_figure(spec, {"a": ds})
    assert resolved["traces"][0]["type"] == "contour"
    assert any(isinstance(c, QuadContourSet) for ax in fig.axes for c in ax.collections)


def test_compile_heatmap_grid_blank_and_heatmap():
    pytest.importorskip("matplotlib")
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
        scales={"field": scale},
        overlays=False,
    )
    assert _quadmeshes(fig)


def test_compile_line_and_mediogram():
    pytest.importorskip("matplotlib")
    from weather_skills_core.plot_recipes import compile_line_figure, compile_mediogram

    fig = compile_line_figure(
        [([1, 2, 3], [0.0, 1.0, 2.0], "a"), ([1, 2, 3], [2.0, 1.0, 0.0], "b")],
        title="lines",
        xlabel="t",
        ylabels=["mm", "mm"],
    )
    assert len(fig.axes[0].lines) == 2
    fc = np.arange(12.0).reshape(3, 4)
    mc = np.arange(12.0, 24.0).reshape(3, 4)
    medio = compile_mediogram(fc, mc, ["+0d", "+1d", "+2d", "+3d"], title="Medio")
    assert len(medio.axes[0].lines) >= 1
    assert medio._suptitle.get_text() == "Medio"


def test_export_png_timeseries_and_mediogram(tmp_path):
    pytest.importorskip("matplotlib")
    from weather_skills_core.plot_export import export_png
    from weather_skills_core.plot_recipes import compile_line_figure, compile_mediogram

    lines = compile_line_figure(
        [([1, 2, 3], [0.0, 1.0, 2.0], "a"), ([1, 2, 3], [2.0, 1.0, 0.0], "b")],
        title="lines",
    )
    line_png = tmp_path / "lines.png"
    export_png(lines, line_png)
    assert line_png.is_file() and line_png.stat().st_size > 0

    fc = np.arange(12.0).reshape(3, 4)
    mc = np.arange(12.0, 24.0).reshape(3, 4)
    medio = compile_mediogram(fc, mc, ["+0d", "+1d", "+2d", "+3d"], title="Medio")
    medio_png = tmp_path / "medio.png"
    export_png(medio, medio_png)
    assert medio_png.is_file() and medio_png.stat().st_size > 0


def test_export_png_heatmap_grid_with_blank(tmp_path):
    pytest.importorskip("matplotlib")
    from weather_skills_core.plot_export import export_png
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
        scales={"field": scale},
        overlays=False,
    )
    out = tmp_path / "grid.png"
    export_png(fig, out)
    assert out.is_file() and out.stat().st_size > 0


def test_format_step_dates_and_leads():
    from weather_skills_core.plot_compile import format_step

    assert format_step(np.datetime64("2026-01-01T00:00:00")) == "1 Jan '26"
    assert format_step(np.timedelta64(0, "D")) == "+0d"
    assert format_step(np.timedelta64(3, "D")) == "+3d"


def test_panel_title_weekly_range_and_daily_date():
    import xarray as xr

    from weather_skills_core.plot_compile import panel_title, timeseries_axis

    weekly = np.arange("2026-08-04", "2026-09-01", dtype="datetime64[D]")[::7]
    da = xr.DataArray(
        np.zeros((len(weekly), 2, 2)),
        dims=("time", "latitude", "longitude"),
        coords={
            "time": weekly,
            "latitude": [0.0, 1.0],
            "longitude": [36.0, 37.0],
        },
        name="precip",
    )
    da.attrs["aggregation_period"] = "7 day"
    assert panel_title(da, "time", weekly[0], weekly) == "4–10 Aug '26"

    daily = np.arange("2026-08-04", "2026-08-08", dtype="datetime64[D]")
    daily_da = xr.DataArray(
        np.zeros((len(daily), 2, 2)),
        dims=("time", "latitude", "longitude"),
        coords={
            "time": daily,
            "latitude": [0.0, 1.0],
            "longitude": [36.0, 37.0],
        },
        name="precip",
    )
    assert panel_title(daily_da, "time", daily[0], daily) == "4 Aug '26"

    fc = make_forecast()["tp"]
    xvals, xlabel = timeseries_axis(fc, "step")
    assert xlabel == "Valid time"
    assert np.datetime_as_string(xvals[0], unit="D") == "2026-01-01"


def test_parse_band_and_along_dim():
    from weather_skills_core.plot_style import along_dim, parse_band, resolve_colorscale

    assert parse_band("10,90") == (10.0, 90.0)
    assert parse_band(True) == (10.0, 90.0)
    da = make_forecast()["tp"]
    assert along_dim(da, "member") == "number"
    temp = make_gridded(name="t2m", units="K")["t2m"]
    temp.attrs["standard_name"] = "air_temperature"
    scale = resolve_colorscale(temp, None)
    assert scale["name"] == "rocket"


def test_compile_timeseries_along_and_band():
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    from matplotlib.collections import PolyCollection

    from weather_skills_core.plot_compile import compile_figure

    ds = make_forecast(n_number=5)
    spec = spec_from_flags(
        variable="tp",
        style="timeseries",
        along="number",
        reduce=["latitude", "longitude"],
        band="10,90",
    )
    fig, resolved = compile_figure(spec, {"a": ds})
    assert resolved["layout"]["shared_colorscale"] is True
    fills = [c for ax in fig.axes for c in ax.collections if isinstance(c, PolyCollection)]
    assert fills
    assert fig.axes[0].lines


def test_compile_timeseries_align_dayofyear():
    pytest.importorskip("matplotlib")
    from weather_skills_core.plot_compile import compile_figure

    ds = make_gridded(n_time=4)
    spec = spec_from_flags(
        variable="precip",
        style="timeseries",
        align="dayofyear",
        reduce=["latitude", "longitude"],
    )
    fig, resolved = compile_figure(spec, {"a": ds})
    assert resolved.get("align") == "dayofyear" or spec.get("align") == "dayofyear"
    xdata = fig.axes[0].lines[0].get_xdata()
    assert float(xdata[0]) >= 1


def test_colorblind_template_sets_style():
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    from weather_skills_core.plot_compile import compile_figure
    from weather_skills_core.plot_style import normalize_template

    assert normalize_template("colorblind") == "colorblind"
    ds = make_gridded(n_time=1)
    spec = spec_from_flags(variable="precip", style="heatmap", template="colorblind")
    fig, resolved = compile_figure(spec, {"a": ds})
    assert resolved["style"]["template"] == "colorblind"
    assert _quadmeshes(fig)


def test_shared_colorscale_one_norm_across_panels():
    pytest.importorskip("matplotlib")
    from weather_skills_core.plot_compile import compile_figure

    ds = make_gridded(n_time=3)
    spec = spec_from_flags(variable="precip", style="heatmap")
    fig, resolved = compile_figure(spec, {"a": ds})
    assert resolved["layout"]["shared_colorscale"] is True
    meshes = _quadmeshes(fig)
    norms = {id(m.norm) for m in meshes}
    assert len(norms) == 1
