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
        trace_type="heatmap",
        title="Precip",
        rows=2,
        columns=3,
    )
    path = tmp_path / "out.plot.json"
    dump_spec(spec, path)
    loaded = load_spec(path)
    assert loaded.data["title"] == "Precip"
    assert loaded.data["layout"]["facet"] == {"max_columns": 4, "rows": 2, "columns": 3}
    assert loaded.zarr_paths()[0].name == "in.zarr"
    merged = overlay_spec(loaded.data, {"title": "Edited"})
    assert merged["title"] == "Edited"


def test_normalize_spec_reports_where_a_relocated_key_moved():
    from weather_skills_core import UsageError
    from weather_skills_core.plot_spec import normalize_spec

    for bad, expected in [
        ({"patch": {"title": "x"}}, "pass --patch"),
        ({"layered": True}, "traces\\[0\\].type"),
        ({"rc": {"axes.grid": False}}, "style.rc"),
        ({"mesh": {"alpha": 0.4}}, r"traces\[\].mesh"),
        ({"band": [10, 90]}, r"traces\[\].band"),
        ({"layout": {"title": "x"}}, "moved to title"),
        ({"layout": {"axes": {}}}, "moved to axes"),
        ({"style": {"dpi": 200}}, "layout.dpi"),
    ]:
        with pytest.raises(UsageError, match=expected):
            normalize_spec(bad)
    with pytest.raises(UsageError, match="not a known key"):
        normalize_spec({"bogus": 1})


def test_flag_table_writes_and_reads_one_canonical_path():
    from weather_skills_core.plot_spec import overlay_flags, resolve_flags, spec_get

    spec = overlay_flags({}, title="T", colormap="magma", rows=2, band=[10, 90], bbox=(5, 1, -5, 9))
    assert spec == {
        "title": "T",
        "style": {"colormap": "magma"},
        "layout": {"facet": {"rows": 2}},
        "traces": [{"band": [10, 90]}],
        "geo": {"bbox": [5, 1, -5, 9]},
    }
    assert spec_get(spec, "band") == [10, 90]
    assert spec_get(spec, "vmin") is None
    # A set CLI value wins; an unset one falls back to the spec.
    assert resolve_flags(spec, title="CLI", colormap=None) == {
        "title": "CLI",
        "colormap": "magma",
    }


def test_plot_spec_holder_zarr_paths():
    spec = PlotSpec({"inputs": [{"id": "a", "path": "/tmp/a.zarr"}]})
    assert spec.zarr_paths()[0].as_posix().endswith("a.zarr")


def test_plot_spec_zarr_paths_layers_and_xy_skip_geojson():
    spec = PlotSpec(
        {
            "inputs": [{"id": "a", "path": "/tmp/a.zarr"}],
            "layers": [
                {"kind": "heatmap", "path": "/tmp/a.zarr"},
                {"kind": "scatter", "path": "/tmp/stations.zarr"},
                {"kind": "outline", "path": "/tmp/kenya.geojson"},
            ],
            "traces": [{"type": "xy", "x": "/tmp/x.zarr", "y": "/tmp/y.zarr"}],
        }
    )
    names = [p.name for p in spec.zarr_paths()]
    assert names == ["a.zarr", "stations.zarr", "x.zarr", "y.zarr"]


def test_parse_plot_spec_and_dump_dest(tmp_path):
    from weather_skills_core.plot_spec import dump_spec_dest, parse_plot_spec

    path = tmp_path / "fig.plot.json"
    path.write_text('{"version": 1, "inputs": [{"id": "a", "path": "/tmp/a.zarr"}]}\n')
    loaded = parse_plot_spec(str(path))
    assert loaded.zarr_paths()[0].name == "a.zarr"
    assert dump_spec_dest(None) is None
    assert dump_spec_dest("none") is False
    assert dump_spec_dest("-") == "-"


def test_named_datasets_from_spec_and_cli_fallback():
    from weather_skills_core import UsageError
    from weather_skills_core.plot_spec import (
        datasets_from_cli_or_spec,
        named_datasets_from_spec,
        spec_role_datasets,
    )

    spec = PlotSpec(
        {
            "inputs": [
                {"id": "obs", "path": "/tmp/obs.zarr"},
                {"id": "forecast1", "path": "/tmp/fc.zarr"},
                {"id": "verify1", "path": "/tmp/v.zarr"},
            ]
        }
    )
    spec.datasets = ["OBS", "FC", "V"]
    named = named_datasets_from_spec(spec)
    assert named == {"obs": "OBS", "forecast1": "FC", "verify1": "V"}
    assert spec_role_datasets(named, "forecast") == ["FC"]
    assert datasets_from_cli_or_spec(["A", "B"], spec, exactly=2) == ["A", "B"]
    assert datasets_from_cli_or_spec(None, spec, min_count=2) == ["OBS", "FC", "V"]
    with pytest.raises(UsageError, match="--spec"):
        datasets_from_cli_or_spec(None, None, min_count=1)


def test_precip_default_colorscale_is_chirps():
    da = make_gridded()["precip"]
    da.attrs["units"] = "mm"
    da.attrs["standard_name"] = "lwe_thickness_of_precipitation_amount"
    scale = resolve_colorscale(da, None)
    assert scale["name"] == "chirps_total"
    assert scale["bounds"][0] == 0
    assert scale["colors"][0] == "#ffffff"
    assert scale["colors"][1] == "#ffffff"
    assert scale["colors"][2] == "#c8ffbe"
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
    spec = spec_from_flags(variable="precip", trace_type="heatmap")
    fig, resolved = compile_figure(spec, {"a": ds})
    assert resolved["layout"]["facet"]["columns"] == 4
    assert resolved["layout"]["facet"]["rows"] == 2
    assert resolved["layout"]["facet"]["n_panels"] == 5
    assert len(_quadmeshes(fig)) == 5


def test_compile_timeseries_forecast_valid_time():
    pytest.importorskip("matplotlib")
    from weather_skills_core.plot_compile import compile_figure

    ds = make_forecast()
    spec = spec_from_flags(
        variable="tp",
        trace_type="timeseries",
        reduce=["number", "latitude", "longitude"],
    )
    fig, resolved = compile_figure(spec, {"a": ds})
    assert resolved["traces"][0]["type"] == "timeseries"
    assert fig.axes[0].lines
    assert fig._suptitle.get_text().endswith("(timeseries)")


def test_compile_applies_title():
    pytest.importorskip("matplotlib")
    from weather_skills_core.plot_compile import compile_figure

    ds = make_gridded(n_time=1)
    spec = spec_from_flags(variable="precip", trace_type="heatmap")
    spec["title"] = "Patched"
    fig, _resolved = compile_figure(spec, {"a": ds})
    assert fig._suptitle.get_text() == "Patched"


def _colorbar_box(fig):
    fig.canvas.draw()
    ax = next(a for a in fig.axes if a.get_label() == "<colorbar>")
    return ax.get_position()


def test_compile_applies_colorbar_size():
    pytest.importorskip("matplotlib")
    from weather_skills_core.plot_compile import compile_figure

    ds = make_gridded(n_time=2)
    spec = spec_from_flags(variable="precip", trace_type="heatmap", columns=2)
    fig_default, _ = compile_figure(spec, {"a": ds})
    default = _colorbar_box(fig_default)
    spec["layout"]["colorbar"] = {"len": 0.45, "thickness": 12}
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
    spec = spec_from_flags(variable="precip", trace_type="heatmap", title="Map")
    fig, resolved = compile_figure(spec, {"a": ds})
    out = tmp_path / "map.png"
    write_plot_outputs(fig, resolved, out, datasets={"a": ds})
    assert out.is_file() and out.stat().st_size > 0
    sidecar = tmp_path / "map.plot.json"
    data = json.loads(sidecar.read_text())
    assert data["title"] == "Map"
    assert data["style"]["colormap"]
    # Only resolved values are dumped; unset knobs stay out of the sidecar.
    assert data["axes"] == {}
    assert "None" not in sidecar.read_text()


def test_compile_contour_uses_contour_collections():
    pytest.importorskip("matplotlib")
    from matplotlib.contour import QuadContourSet

    from weather_skills_core.plot_compile import compile_figure

    ds = make_gridded(n_time=1)
    spec = spec_from_flags(variable="precip", trace_type="contour")
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
        trace_type="timeseries",
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
        trace_type="timeseries",
        align="dayofyear",
        reduce=["latitude", "longitude"],
    )
    fig, resolved = compile_figure(spec, {"a": ds})
    assert resolved["traces"][0]["align"] == "dayofyear"
    xdata = fig.axes[0].lines[0].get_xdata()
    assert float(xdata[0]) >= 1


def test_colorblind_template_sets_style():
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    from weather_skills_core.plot_compile import compile_figure
    from weather_skills_core.plot_style import normalize_template

    assert normalize_template("colorblind") == "colorblind"
    ds = make_gridded(n_time=1)
    spec = spec_from_flags(variable="precip", trace_type="heatmap", template="colorblind")
    fig, resolved = compile_figure(spec, {"a": ds})
    assert resolved["style"]["template"] == "colorblind"
    assert _quadmeshes(fig)


def test_shared_colorscale_one_norm_across_panels():
    pytest.importorskip("matplotlib")
    from weather_skills_core.plot_compile import compile_figure

    ds = make_gridded(n_time=3)
    spec = spec_from_flags(variable="precip", trace_type="heatmap")
    fig, resolved = compile_figure(spec, {"a": ds})
    assert resolved["layout"]["shared_colorscale"] is True
    meshes = _quadmeshes(fig)
    norms = {id(m.norm) for m in meshes}
    assert len(norms) == 1


def test_pick_rejects_unknown_and_non_json():
    from weather_skills_core import UsageError
    from weather_skills_core.plot_mpl import LINE_KEYS, pick

    assert pick({"linewidth": 2.5}, LINE_KEYS, loc="line")["linewidth"] == 2.5
    with pytest.raises(UsageError, match="unknown key"):
        pick({"linewidth": 2, "bogus": 1}, LINE_KEYS, loc="line")
    with pytest.raises(UsageError, match="must be JSON"):
        pick({"linewidth": object()}, LINE_KEYS, loc="line")


def test_apply_axes_xlabel_string_and_object():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    from weather_skills_core import UsageError
    from weather_skills_core.plot_mpl import apply_axes

    fig, ax = plt.subplots()
    apply_axes(ax, {"xlabel": "Lon", "ylabel": "Lat"})
    assert ax.get_xlabel() == "Lon"
    assert ax.get_ylabel() == "Lat"

    apply_axes(ax, {"ylabel": {"text": "Frequency (%)", "pad": 12, "rotation": 0}})
    assert ax.get_ylabel() == "Frequency (%)"
    assert ax.yaxis.label.get_rotation() == 0

    apply_axes(ax, {"ylabel": {"coords": [1.15, 0.5]}})
    assert ax.get_ylabel() == "Frequency (%)"
    pos = ax.yaxis.label.get_position()
    assert pos == pytest.approx((1.15, 0.5))

    fig_p = plt.figure()
    ax_p = fig_p.add_subplot(111, projection="polar")
    ax_p.set_ylabel("Frequency (%)")
    apply_axes(ax_p, {"ylabel": {"coords": [1.15, 0.5], "rotation": 0}})
    assert ax_p.get_ylabel() == "Frequency (%)"
    assert ax_p.yaxis.label.get_position() == pytest.approx((1.15, 0.5))
    assert ax_p.yaxis.label.get_rotation() == 0
    plt.close(fig_p)

    with pytest.raises(UsageError, match="unknown key"):
        apply_axes(ax, {"xlabel": {"text": "x", "bogus": 1}})
    with pytest.raises(UsageError, match="must be a string or object"):
        apply_axes(ax, {"xlabel": ["Lon"]})
    plt.close(fig)


def test_apply_rc_sets_and_rejects_backend():
    pytest.importorskip("matplotlib")
    import matplotlib as mpl

    from weather_skills_core import UsageError
    from weather_skills_core.plot_mpl import apply_rc

    apply_rc({"axes.grid": False, "lines.linewidth": 3.0})
    assert mpl.rcParams["lines.linewidth"] == 3.0
    with pytest.raises(UsageError, match="backend"):
        apply_rc({"backend": "TkAgg"})
    with pytest.raises(UsageError, match="unknown matplotlib rcParam"):
        apply_rc({"not.a.real.param": 1})


def test_compile_mpl_axes_annotate_mesh_and_log():
    pytest.importorskip("matplotlib")
    from matplotlib.patches import FancyArrowPatch

    from weather_skills_core.plot_compile import compile_figure

    ds = make_gridded(n_time=1)
    spec = spec_from_flags(variable="precip", trace_type="heatmap")
    spec["traces"][0]["mesh"] = {"alpha": 0.4}
    spec["axes"] = {
        "spines": {"top": False, "right": False},
        "grid": {"visible": True, "alpha": 0.3},
    }
    spec["annotations"] = [
        {
            "text": "peak",
            "xy": [10.0, 2.0],
            "xytext": [11.0, 3.0],
            "arrowprops": {"arrowstyle": "->", "color": "black"},
        }
    ]
    spec["shapes"] = [{"type": "hline", "y": 2.0, "linestyle": "--", "color": "red"}]
    fig, _resolved = compile_figure(spec, {"a": ds})
    ax = next(a for a in fig.axes if a.get_label() != "<colorbar>")
    assert ax.spines["top"].get_visible() is False
    assert ax.collections[0].get_alpha() == 0.4
    assert any(isinstance(p, FancyArrowPatch) for p in ax.patches) or ax.texts
    texts = [t.get_text() for t in ax.texts]
    assert "peak" in texts

    ts = spec_from_flags(
        variable="precip",
        trace_type="timeseries",
        reduce=["latitude", "longitude"],
    )
    ts["axes"] = {"yscale": "log", "legend": {"loc": "lower right"}}
    ts["traces"][0]["line"] = {"linewidth": 4, "linestyle": "--"}
    fig_ts, _ = compile_figure(ts, {"a": ds})
    assert fig_ts.axes[0].get_yscale() == "log"
    assert fig_ts.axes[0].lines[0].get_linewidth() == 4


def test_compile_contour_levels_from_spec():
    pytest.importorskip("matplotlib")
    from matplotlib.contour import QuadContourSet

    from weather_skills_core.plot_compile import compile_figure

    ds = make_gridded(n_time=1)
    spec = spec_from_flags(variable="precip", trace_type="contour")
    spec["traces"][0]["contour"] = {"levels": 5, "linewidths": 1.2, "lines": True}
    fig, _ = compile_figure(spec, {"a": ds})
    filled = next(c for ax in fig.axes for c in ax.collections if isinstance(c, QuadContourSet))
    assert len(filled.levels) >= 2


def test_compile_line_twin_and_mediogram_colors():
    pytest.importorskip("matplotlib")
    from weather_skills_core.plot_recipes import compile_line_figure, compile_mediogram

    fig = compile_line_figure(
        [([1, 2, 3], [0.0, 1.0, 2.0], "a"), ([1, 2, 3], [20.0, 10.0, 0.0], "b")],
        styles=[{}, {"twin": "y", "color": "#d62728", "line": {"linewidth": 3}}],
    )
    assert len(fig.axes) == 2
    assert fig.axes[1].lines[0].get_linewidth() == 3

    fc = np.arange(12.0).reshape(3, 4)
    mc = np.arange(12.0, 24.0).reshape(3, 4)
    medio = compile_mediogram(
        fc,
        mc,
        ["+0d", "+1d", "+2d", "+3d"],
        spec={
            "traces": [
                {
                    "type": "mediogram",
                    "mediogram": {"forecast": {"facecolor": "#ff00ff"}, "width": 0.2},
                }
            ]
        },
    )
    box = medio.axes[0].patches[0]
    facecolor = box.get_facecolor()
    assert facecolor[0] > 0.9 and facecolor[2] > 0.9
    assert facecolor[1] < 0.2


def test_compile_dumps_axes_ticks_and_applies_xticks():
    pytest.importorskip("matplotlib")
    from weather_skills_core.plot_compile import compile_figure
    from weather_skills_core.plot_mpl import AXES_TEMPLATE, resolve_axes_block

    # An untouched figure dumps no axes knobs; the editable set is AXES_TEMPLATE.
    assert resolve_axes_block({}) == {}
    assert {"xticks", "yticks", "tick_params", "xlocator"} <= set(AXES_TEMPLATE)
    assert resolve_axes_block({"axes": {"yscale": "log", "xlim": None}}) == {"yscale": "log"}

    ds = make_gridded(n_time=1)
    spec = spec_from_flags(
        variable="precip",
        trace_type="timeseries",
        reduce=["latitude", "longitude"],
    )
    spec["axes"] = {"yticks": [0.0, 0.5, 1.0], "xticks": {"values": [1, 2], "labels": ["a", "b"]}}
    fig, resolved = compile_figure(spec, {"a": ds})
    assert resolved["axes"]["yticks"] == [0.0, 0.5, 1.0]
    yticks = [
        float(t)
        for t in fig.axes[0].get_yticks()
        if fig.axes[0].get_ylim()[0] <= t <= fig.axes[0].get_ylim()[1]
    ]
    assert yticks == [0.0, 0.5, 1.0]
    assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == ["a", "b"]


def test_heatmap_grid_and_sidecar_use_shared_axes_spec(tmp_path):
    pytest.importorskip("matplotlib")
    from weather_skills_core.plot_export import write_plot_outputs
    from weather_skills_core.plot_recipes import compile_heatmap_grid, heatmap_cell
    from weather_skills_core.plot_style import resolve_colorscale

    ds = make_gridded(n_time=1)
    da = ds["precip"].isel(time=0)
    scale = resolve_colorscale(da, None)
    spec = {"axes": {"spines": {"top": False}}}
    fig = compile_heatmap_grid(
        [[heatmap_cell(da, "latitude", "longitude")]],
        extent=[9.5, 13.5, 0.5, 3.5],
        scales={"field": scale},
        overlays=False,
        spec=spec,
    )
    ax = next(a for a in fig.axes if a.get_label() != "<colorbar>")
    assert ax.spines["top"].get_visible() is False
    out = tmp_path / "grid.png"
    write_plot_outputs(fig, {"title": "grid", "layout": {}}, out, spec=spec)
    dumped = json.loads(out.with_name("grid.plot.json").read_text())
    assert dumped["axes"] == {"spines": {"top": False}}
