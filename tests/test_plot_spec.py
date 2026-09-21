"""Tests for the matplotlib plot spec / compiler."""

from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import make_forecast, make_gridded

from weather_skills_core.plot.spec import (
    PlotSpec,
    dump_spec,
    load_spec,
    overlay_spec,
    panel_shape,
    parse_index,
    spec_from_flags,
)
from weather_skills_core.plot.theme import resolve_colorscale


def _fig(obj):
    return obj.fig if hasattr(obj, "fig") else obj


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
        kind="heatmap",
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
    from weather_skills_core.plot.spec import normalize_spec

    for bad, expected in [
        ({"patch": {"title": "x"}}, "pass --patch"),
        ({"layered": True}, "traces\\[0\\].kind"),
        ({"rc": {"axes.grid": False}}, "theme.rc"),
        ({"mesh": {"alpha": 0.4}}, r"traces\[\].mesh"),
        ({"band": [10, 90]}, r"traces\[\].band"),
        ({"along_color": "cycle"}, r"traces\[\].along_color"),
        ({"layout": {"title": "x"}}, "moved to title"),
        ({"layout": {"wspace": 0.4}}, "moved to layout.facet.wspace"),
        ({"layout": {"facet": {"horizontal_spacing": 0.25}}}, "moved to layout.facet.wspace"),
        ({"layout": {"axes": {}}}, "moved to axes"),
        ({"style": {"dpi": 200}}, "theme"),
        ({"traces": [{"type": "heatmap"}]}, r"traces\[0\].type moved to traces\[\].kind"),
        (
            {"traces": [{"kind": "heatmap", "style": "line"}]},
            r"traces\[0\].style moved to traces\[\].mark",
        ),
        ({"version": 1}, "version 1 is not supported"),
    ]:
        with pytest.raises(UsageError, match=expected):
            normalize_spec(bad)
    with pytest.raises(UsageError, match="not a known key"):
        normalize_spec({"bogus": 1})


def test_flag_table_writes_and_reads_one_canonical_path():
    from weather_skills_core.plot.spec import overlay_flags, resolve_flags, spec_get

    spec = overlay_flags({}, title="T", colormap="magma", rows=2, band=[10, 90], bbox=(5, 1, -5, 9))
    assert spec == {
        "title": "T",
        "theme": {"colormap": "magma"},
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
    from weather_skills_core.plot.spec import FLAG_TO_SPEC

    assert FLAG_TO_SPEC["kind"] == ("traces", 0, "kind")
    assert FLAG_TO_SPEC["mark"] == ("traces", 0, "mark")
    assert FLAG_TO_SPEC["lead"] == ("traces", 0, "leads")
    assert FLAG_TO_SPEC["align_day_of_year"] == ("traces", 0, "align")
    assert FLAG_TO_SPEC["subplots"] == ("layout", "subplots")
    assert FLAG_TO_SPEC["bar_mode"] == ("layout", "bar_mode")
    assert FLAG_TO_SPEC["wspace"] == ("layout", "facet", "wspace")
    assert FLAG_TO_SPEC["hspace"] == ("layout", "facet", "hspace")
    spaced = overlay_flags({}, panel_spacing=(0.4, 0.2))
    assert spaced["layout"]["facet"] == {"wspace": 0.4, "hspace": 0.2}
    assert spec_get(spaced, "panel_spacing") == (0.4, 0.2)


def test_theme_file_rejects_unknown_keys(tmp_path):
    from weather_skills_core import UsageError
    from weather_skills_core.plot.theme import load_user_theme

    path = tmp_path / "theme.json"
    path.write_text('{"bogus": 1, "fontsize": 18}\n', encoding="utf-8")
    with pytest.raises(UsageError, match="unknown keys"):
        load_user_theme(path)


def test_colormap_bounds_fold_into_theme_colormap():
    from weather_skills_core import UsageError
    from weather_skills_core.plot.spec import overlay_flags, spec_from_flags, spec_get
    from weather_skills_core.plot.theme import parse_colormap_spec, resolve_colorscale

    spec = overlay_flags({}, colormap="viridis", colormap_bounds=[0, 10, 50], colormap_under="grey")
    assert spec["theme"]["colormap"] == {
        "name": "viridis",
        "bounds": [0.0, 10.0, 50.0],
        "under": "grey",
    }
    folded = spec_from_flags(
        colormap="white,green,blue",
        colormap_bounds=[0, 10, 50, 100],
        cbar_ticks=[0, 50, 100],
        cbar_labels=["dry", "mid", "wet"],
    )
    assert folded["theme"]["colormap"]["colors"] == ["white", "green", "blue"]
    assert folded["theme"]["colormap"]["bounds"] == [0.0, 10.0, 50.0, 100.0]
    assert spec_get(folded, "cbar_ticks") == [0.0, 50.0, 100.0]
    assert spec_get(folded, "cbar_labels") == ["dry", "mid", "wet"]

    parsed = parse_colormap_spec(
        {"colors": ["#fff", "#080", "#040"], "bounds": [0, 10, 50, 100], "over": "magenta"}
    )
    assert parsed["over"] == "magenta"
    with pytest.raises(UsageError, match="3 class colors"):
        parse_colormap_spec({"colors": ["a", "b"], "bounds": [0, 1, 2, 3]})
    with pytest.raises(UsageError, match="strictly increasing"):
        parse_colormap_spec({"colors": ["a"], "bounds": [10, 0]})

    da = make_gridded()["precip"]
    scale = resolve_colorscale(
        da,
        "drought",
        registry={"drought": {"colors": ["#ffffff", "#cc0000"], "bounds": [0, 1, 2]}},
    )
    assert scale["name"] == "drought"
    assert scale["bounds"] == [0, 1, 2]
    named_bounds = resolve_colorscale(da, {"name": "viridis", "bounds": [0, 10, 50]})
    assert named_bounds["cmap"] == "viridis"
    assert named_bounds["bounds"] == [0, 10, 50]


def test_normalize_spec_colorbar_labels_need_ticks():
    from weather_skills_core import UsageError
    from weather_skills_core.plot.spec import normalize_spec

    with pytest.raises(UsageError, match="requires layout.colorbar.ticks"):
        normalize_spec({"layout": {"colorbar": {"labels": ["a"]}}})
    with pytest.raises(UsageError, match="not a known key"):
        normalize_spec({"theme": {"colormap": {"colors": ["#fff", "#000"], "bogus": 1}}})
    ok = normalize_spec(
        {
            "theme": {
                "colormap": {
                    "colors": ["#fff", "#080", "#040"],
                    "bounds": [0, 10, 50, 100],
                    "under": "grey",
                }
            },
            "layout": {"colorbar": {"ticks": [0, 50, 100], "labels": ["dry", "mid", "wet"]}},
        }
    )
    assert ok["layout"]["colorbar"]["labels"] == ["dry", "mid", "wet"]


def test_compile_custom_discrete_colormap_and_cbar_labels():
    pytest.importorskip("matplotlib")
    from matplotlib.colors import BoundaryNorm

    from weather_skills_core.plot import compile

    ds = make_gridded()
    spec = spec_from_flags(
        variable="precip",
        kind="heatmap",
        colormap={
            "colors": ["#ffffff", "#88cc88", "#006600"],
            "bounds": [0, 10, 50, 100],
            "under": "0.5",
            "over": "magenta",
        },
        cbar_ticks=[0, 50, 100],
        cbar_labels=["dry", "ok", "wet"],
    )
    compiled = compile(spec, {"a": ds})
    fig, resolved = compiled.fig, compiled.spec
    meshes = _quadmeshes(fig)
    assert meshes
    assert isinstance(meshes[0].norm, BoundaryNorm)
    assert list(meshes[0].norm.boundaries) == [0.0, 10.0, 50.0, 100.0]
    cbar_ax = next(ax for ax in fig.axes if ax.get_label() == "<colorbar>")
    fig.canvas.draw()
    labels = [t.get_text() for t in cbar_ax.get_yticklabels() if t.get_text()]
    if not labels:
        labels = [t.get_text() for t in cbar_ax.get_xticklabels() if t.get_text()]
    assert labels == ["dry", "ok", "wet"]
    assert resolved["theme"]["colormap"]["bounds"] == [0.0, 10.0, 50.0, 100.0]
    assert resolved["layout"]["colorbar"]["labels"] == ["dry", "ok", "wet"]


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
            "traces": [{"kind": "xy", "x": "/tmp/x.zarr", "y": "/tmp/y.zarr"}],
        }
    )
    names = [p.name for p in spec.zarr_paths()]
    assert names == ["a.zarr", "stations.zarr", "x.zarr", "y.zarr"]


def test_parse_plot_spec_and_dump_dest(tmp_path):
    from weather_skills_core.plot.spec import dump_spec_dest, parse_plot_spec

    path = tmp_path / "fig.plot.json"
    path.write_text('{"version": 2, "inputs": [{"id": "a", "path": "/tmp/a.zarr"}]}\n')
    loaded = parse_plot_spec(str(path))
    assert loaded.zarr_paths()[0].name == "a.zarr"
    assert dump_spec_dest(None) is None
    assert dump_spec_dest("none") is False
    assert dump_spec_dest("-") == "-"
    assert dump_spec_dest("") == "-"


def test_maybe_emit_spec_writes_json_and_signals_skip(tmp_path):
    from weather_skills_core.plot.spec import maybe_emit_spec

    dest = tmp_path / "fig.plot.json"
    spec = {"version": 2, "title": "T", "traces": [{"kind": "heatmap", "input": "a"}]}
    assert maybe_emit_spec(spec, dest) is True
    dumped = json.loads(dest.read_text())
    assert dumped["title"] == "T"
    assert dumped["traces"][0]["kind"] == "heatmap"
    assert maybe_emit_spec(spec, None) is False
    assert maybe_emit_spec(spec, "none") is False


def test_parse_plot_patch_inline_and_file(tmp_path):
    from weather_skills_core.plot.spec import parse_plot_patch

    assert parse_plot_patch('{"title": "Patched"}') == {"title": "Patched"}
    path = tmp_path / "edit.json"
    path.write_text('{"axes": {"xticks": ["2026-08-17"]}}\n')
    assert parse_plot_patch(str(path)) == {"axes": {"xticks": ["2026-08-17"]}}
    long_raw = json.dumps({"title": "x" * 220, "axes": {"xticks": ["2026-08-17"]}})
    assert len(long_raw) > 255
    assert parse_plot_patch(long_raw)["title"].startswith("x")


def test_load_spec_accepts_long_inline_json():
    colors = [
        "#f7f5d8",
        "#e8e6a8",
        "#c8dd8c",
        "#8ecb72",
        "#4db56a",
        "#2a9d6e",
        "#1d8a86",
        "#1f6fa9",
        "#2a55a0",
        "#333f88",
        "#3b2c6e",
    ]
    raw = json.dumps(
        {
            "title": "CHIRPS average annual rainfall, Ethiopia (climatology)",
            "theme": {
                "colormap": {
                    "colors": colors,
                    "bounds": [0, 100, 200, 300, 400, 600, 800, 1000, 1400, 1800, 2400, 3300],
                }
            },
            "inputs": [{"variable": "precip_avg"}],
        }
    )
    assert len(raw) > 255
    loaded = load_spec(raw)
    assert loaded.data["title"].startswith("CHIRPS")
    assert loaded.data["theme"]["colormap"]["colors"][-1] == "#3b2c6e"


def test_named_datasets_from_spec_and_cli_fallback():
    from weather_skills_core import UsageError
    from weather_skills_core.plot.spec import (
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


def test_precip_default_colorscale_is_nested_week_window():
    da = make_gridded()["precip"]
    da.attrs["units"] = "mm"
    da.attrs["standard_name"] = "lwe_thickness_of_precipitation_amount"
    scale = resolve_colorscale(da, None)
    assert scale["name"] == "ppt_week"
    assert scale["bounds"][0] == 0
    assert scale["bounds"][-1] == 200
    assert scale["colors"][0] == "#ffffff"
    assert scale["colors"][1] == "#ffffff"
    assert scale["colors"][2] == "#f6e8c3"
    assert scale["colors"][3] == "#dfc27d"
    assert scale["colors"][4] == "#1eb41e"
    assert scale["colors"][5] == "#50a5f5"
    chc = resolve_colorscale(da, "ppt_total")
    assert chc["name"] == "ppt_total"
    assert chc["colors"][0] == "#ffffff"
    assert chc["colors"][1] == "#ffffff"
    assert chc["colors"][2] == "#c8ffbe"
    assert chc["colors"][-1] == "#ffe6e6"
    assert chc["bounds"][-1] == 2500


def test_precip_nested_windows_keep_absolute_mm_colors():
    from weather_skills_core.plot.theme import (
        PRECIP_COLORS,
        PRECIP_MASTER_BOUNDS,
        PRECIP_MASTER_COLORS,
        PRECIP_OVER,
        PRECIP_UNDER,
        precip_nested_palette,
        precip_window_name,
        widest_precip_window,
    )

    assert len(PRECIP_MASTER_COLORS) == len(PRECIP_MASTER_BOUNDS) - 1
    assert PRECIP_MASTER_COLORS == [
        PRECIP_COLORS[1],
        "#f6e8c3",
        "#dfc27d",
        PRECIP_COLORS[4],
        *PRECIP_COLORS[6:8],
        *PRECIP_COLORS[9:16],
        "#7a0000",
    ]

    assert precip_window_name(None) == "ppt_week"
    assert precip_window_name(1) == "ppt_daily"
    assert precip_window_name(1.9) == "ppt_daily"
    assert precip_window_name(2) == "ppt_week"
    assert precip_window_name(9.9) == "ppt_week"
    assert precip_window_name(10) == "ppt_month"
    assert precip_window_name(39.9) == "ppt_month"
    assert precip_window_name(40) == "ppt_season"
    assert widest_precip_window(1, 7, 90) == "ppt_season"
    assert widest_precip_window(1, None) == "ppt_week"

    daily = make_gridded()["precip"]
    daily.attrs.update(
        units="mm",
        standard_name="lwe_thickness_of_precipitation_amount",
        aggregation_period="1 day",
    )
    weekly = daily.copy()
    weekly.attrs["aggregation_period"] = "7 day"
    monthly = daily.copy()
    monthly.attrs["aggregation_period"] = "10 day"
    seasonal = daily.copy()
    seasonal.attrs["aggregation_period"] = "90 day"

    sd, sw, sm, ss = (
        resolve_colorscale(daily, None),
        resolve_colorscale(weekly, None),
        resolve_colorscale(monthly, None),
        resolve_colorscale(seasonal, None),
    )
    assert sd["name"] == "ppt_daily" and sd["bounds"][-1] == 50
    assert sw["name"] == "ppt_week" and sw["bounds"][-1] == 200
    assert sm["name"] == "ppt_month" and sm["bounds"][-1] == 400
    assert ss["name"] == "ppt_season" and ss["bounds"][-1] == 1000
    assert sd["colors"][5] == sw["colors"][5] == sm["colors"][5] == "#50a5f5"
    season = precip_nested_palette("ppt_season")
    assert season["colors"][0] == PRECIP_UNDER
    assert season["colors"][1:-1] == PRECIP_MASTER_COLORS
    assert season["colors"][-1] == PRECIP_OVER
    aliased = resolve_colorscale(daily, "ppt_daily")
    assert aliased["colors"] == sd["colors"]
    assert aliased["bounds"] == sd["bounds"]


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
    from weather_skills_core.plot.theme import rank_colorscale

    scale = rank_colorscale(40)
    assert scale["name"] == "ppt_rank"
    assert scale["bounds"][0] == -0.5
    assert scale["bounds"][-2] == 39.5
    assert len(scale["colors"]) == len(scale["bounds"]) - 1


def _quadmeshes(fig):
    from matplotlib.collections import QuadMesh

    fig = _fig(fig)
    return [
        c
        for ax in fig.axes
        for c in ax.collections
        if isinstance(c, QuadMesh) and ax.get_label() != "<colorbar>"
    ]


def test_compile_heatmap_facets_time():
    pytest.importorskip("matplotlib")
    from weather_skills_core.plot import compile

    ds = make_gridded(n_time=5)
    spec = spec_from_flags(variable="precip", kind="heatmap")
    compiled = compile(spec, {"a": ds})
    fig, resolved = compiled.fig, compiled.spec
    assert resolved["layout"]["facet"]["columns"] == 4
    assert resolved["layout"]["facet"]["rows"] == 2
    assert resolved["layout"]["facet"]["n_panels"] == 5
    assert len(_quadmeshes(fig)) == 5


def _visible_map_axes(fig):
    fig = _fig(fig)
    return [ax for ax in fig.axes if ax.get_visible() and ax.get_label() != "<colorbar>"]


def test_compile_heatmap_facet_spacing_separates_panels():
    pytest.importorskip("matplotlib")
    from weather_skills_core.plot import compile

    ds = make_gridded(n_time=2)
    spec = spec_from_flags(variable="precip", kind="heatmap", columns=2)
    fig_tight = compile(spec, {"a": ds}).fig
    fig_tight.canvas.draw()
    left, right = _visible_map_axes(fig_tight)[:2]
    gap_tight = right.get_position().x0 - left.get_position().x1

    spec["layout"]["facet"]["wspace"] = 0.45
    compiled = compile(spec, {"a": ds})
    fig_spaced, resolved = compiled.fig, compiled.spec
    fig_spaced.canvas.draw()
    left_s, right_s = _visible_map_axes(fig_spaced)[:2]
    gap_spaced = right_s.get_position().x0 - left_s.get_position().x1
    assert resolved["layout"]["facet"]["wspace"] == 0.45
    assert gap_spaced > gap_tight + 0.03
    assert fig_spaced.get_layout_engine() is None


def test_normalize_spec_rejects_negative_facet_spacing():
    from weather_skills_core import UsageError
    from weather_skills_core.plot.spec import normalize_spec

    with pytest.raises(UsageError, match="layout.facet.wspace must be >= 0"):
        normalize_spec({"layout": {"facet": {"wspace": -0.1}}})


def test_compile_timeseries_forecast_valid_time():
    pytest.importorskip("matplotlib")
    from weather_skills_core.plot import compile

    ds = make_forecast()
    spec = spec_from_flags(
        variable="tp",
        kind="timeseries",
        reduce=["number", "latitude", "longitude"],
    )
    compiled = compile(spec, {"a": ds})
    fig, resolved = compiled.fig, compiled.spec
    assert resolved["traces"][0]["kind"] == "timeseries"
    assert fig.axes[0].lines
    assert fig._suptitle.get_text().endswith("(timeseries)")


def test_compile_applies_title():
    pytest.importorskip("matplotlib")
    from weather_skills_core.plot import compile

    ds = make_gridded(n_time=1)
    spec = spec_from_flags(variable="precip", kind="heatmap")
    spec["title"] = "Patched"
    fig = compile(spec, {"a": ds}).fig
    assert fig._suptitle.get_text() == "Patched"


def _colorbar_box(fig):
    fig = _fig(fig)
    fig.canvas.draw()
    ax = next(a for a in fig.axes if a.get_label() == "<colorbar>")
    return ax.get_position()


def test_compile_applies_colorbar_size():
    pytest.importorskip("matplotlib")
    from weather_skills_core.plot import compile

    ds = make_gridded(n_time=2)
    spec = spec_from_flags(variable="precip", kind="heatmap", columns=2)
    fig_default = compile(spec, {"a": ds}).fig
    default = _colorbar_box(fig_default)
    spec["layout"]["colorbar"] = {"len": 0.45, "thickness": 12}
    compiled_patched = compile(spec, {"a": ds})
    fig_patched, resolved = compiled_patched.fig, compiled_patched.spec
    patched = _colorbar_box(fig_patched)
    assert resolved["layout"]["colorbar"]["len"] == 0.45
    assert patched.width < default.width
    assert patched.height < default.height * 0.6


def test_export_skips_spec_dump_by_default(tmp_path):
    pytest.importorskip("matplotlib")
    from weather_skills_core.plot import compile, export

    ds = make_gridded(n_time=1)
    spec = spec_from_flags(variable="precip", kind="heatmap", title="Map")
    compiled = compile(spec, {"a": ds})
    out = tmp_path / "map.png"
    export(compiled, out, datasets={"a": ds})
    assert out.is_file() and out.stat().st_size > 0
    assert not (tmp_path / "map.plot.json").exists()


def test_export_png_and_optional_spec_dump(tmp_path):
    pytest.importorskip("matplotlib")
    from weather_skills_core.plot import compile, export

    ds = make_gridded(n_time=1)
    spec = spec_from_flags(variable="precip", kind="heatmap", title="Map")
    compiled = compile(spec, {"a": ds})
    out = tmp_path / "map.png"
    dumped = tmp_path / "map.plot.json"
    export(compiled, out, datasets={"a": ds}, dump_spec_path=dumped)
    assert out.is_file() and out.stat().st_size > 0
    data = json.loads(dumped.read_text())
    assert data["title"] == "Map"
    assert data["theme"]["colormap"]
    # Only resolved values are dumped; unset knobs stay out of the JSON.
    assert data["axes"] == {}
    assert "None" not in dumped.read_text()


def test_export_dump_spec_stdout(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    from weather_skills_core.plot import compile, export

    ds = make_gridded(n_time=1)
    spec = spec_from_flags(variable="precip", kind="heatmap", title="Map")
    compiled = compile(spec, {"a": ds})
    out = tmp_path / "map.png"
    export(compiled, out, datasets={"a": ds}, dump_spec_path="-")
    data = json.loads(capsys.readouterr().out)
    assert data["title"] == "Map"
    assert not (tmp_path / "map.plot.json").exists()


def test_compile_contour_uses_contour_collections():
    pytest.importorskip("matplotlib")
    from matplotlib.contour import QuadContourSet

    from weather_skills_core.plot import compile

    ds = make_gridded(n_time=1)
    spec = spec_from_flags(variable="precip", kind="contour")
    compiled = compile(spec, {"a": ds})
    fig, resolved = compiled.fig, compiled.spec
    assert resolved["traces"][0]["kind"] == "contour"
    assert any(isinstance(c, QuadContourSet) for ax in fig.axes for c in ax.collections)


def test_compile_grid_bottom_colorbars_side_by_side():
    pytest.importorskip("matplotlib")
    from weather_skills_core.plot.maps import blank_cell, compile_grid, heatmap_cell
    from weather_skills_core.plot.theme import resolve_colorscale

    ds = make_gridded(n_time=1)
    da = ds["precip"].isel(time=0)
    field = resolve_colorscale(da, None)
    field["label"] = "precip"
    verify = dict(field)
    verify["label"] = "bias"
    compiled = compile_grid(
        [
            [heatmap_cell(da, "latitude", "longitude"), heatmap_cell(da, "latitude", "longitude")],
            [blank_cell(""), heatmap_cell(da, "latitude", "longitude", scale="verify")],
        ],
        extent=[9.5, 13.5, 0.5, 3.5],
        scales={"field": field, "verify": verify},
        spec={"layout": {"colorbar": {"location": "bottom"}}},
        overlays=False,
    )
    fig = _fig(compiled)
    cbars = [ax for ax in fig.axes if ax.get_label() == "<colorbar>"]
    assert len(cbars) == 2
    left, right = sorted(cbars, key=lambda ax: ax.get_position().x0)
    assert left.get_position().x1 <= right.get_position().x0 + 0.02
    assert left.get_position().y1 < 0.35
    assert right.get_position().y1 < 0.35


def test_compile_grid_facet_spacing_separates_columns():
    pytest.importorskip("matplotlib")
    from weather_skills_core.plot.maps import compile_grid, heatmap_cell
    from weather_skills_core.plot.theme import resolve_colorscale

    ds = make_gridded(n_time=1)
    da = ds["precip"].isel(time=0)
    field = resolve_colorscale(da, None)
    cells = [[heatmap_cell(da, "latitude", "longitude"), heatmap_cell(da, "latitude", "longitude")]]
    extent = [9.5, 13.5, 0.5, 3.5]
    fig_tight = _fig(compile_grid(cells, extent=extent, scales={"field": field}, overlays=False))
    fig_tight.canvas.draw()
    left, right = _visible_map_axes(fig_tight)[:2]
    gap_tight = right.get_position().x0 - left.get_position().x1
    fig_spaced = _fig(
        compile_grid(
            cells,
            extent=extent,
            scales={"field": field},
            overlays=False,
            spec={"layout": {"facet": {"wspace": 0.45}}},
        )
    )
    fig_spaced.canvas.draw()
    left_s, right_s = _visible_map_axes(fig_spaced)[:2]
    gap_spaced = right_s.get_position().x0 - left_s.get_position().x1
    assert gap_spaced > gap_tight + 0.03
    assert fig_spaced.get_layout_engine() is None


def test_compile_heatmap_grid_blank_and_heatmap():
    pytest.importorskip("matplotlib")
    from weather_skills_core.plot.maps import blank_cell, compile_grid, heatmap_cell
    from weather_skills_core.plot.theme import resolve_colorscale

    ds = make_gridded(n_time=1)
    da = ds["precip"].isel(time=0)
    scale = resolve_colorscale(da, None)
    scale["label"] = "precip"
    fig = compile_grid(
        [[heatmap_cell(da, "latitude", "longitude"), blank_cell("n/a")]],
        extent=[9.5, 13.5, 0.5, 3.5],
        col_titles=["t0", "missing"],
        scales={"field": scale},
        overlays=False,
    )
    assert _quadmeshes(fig)


def test_compile_bars_grouped_stacked_overlay():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from weather_skills_core import UsageError
    from weather_skills_core.plot.charts import compile_lines

    x = [0, 1, 2]
    a = [1.0, 2.0, 3.0]
    b = [0.5, 0.5, 0.5]
    series = [(x, a, "a"), (x, b, "b")]

    def rects(compiled):
        return [p for ax in _fig(compiled).axes for p in ax.patches if isinstance(p, Rectangle)]

    grouped = compile_lines(series, kinds=["bar", "bar"], spec={"layout": {"bar_mode": "grouped"}})
    g = rects(grouped)
    assert len(g) == 6
    assert g[0].get_x() < g[3].get_x()
    plt.close(_fig(grouped))

    stacked = compile_lines(series, kinds=["bar", "bar"], spec={"layout": {"bar_mode": "stacked"}})
    s = rects(stacked)
    assert s[0].get_y() == pytest.approx(0.0)
    assert s[3].get_y() == pytest.approx(1.0)
    assert s[0].get_x() == pytest.approx(s[3].get_x())
    plt.close(_fig(stacked))

    overlay = compile_lines(series, kinds=["bar", "bar"], spec={"layout": {"bar_mode": "overlay"}})
    o = rects(overlay)
    assert o[0].get_y() == pytest.approx(0.0)
    assert o[3].get_y() == pytest.approx(0.0)
    assert o[0].get_x() == pytest.approx(o[3].get_x())
    plt.close(_fig(overlay))

    aliased = compile_lines(
        series,
        kinds=["bar", "bar"],
        spec={"traces": [{"bar": {"mode": "stacked"}}, {}]},
    )
    alias_rects = rects(aliased)
    assert alias_rects[3].get_y() == pytest.approx(1.0)
    plt.close(_fig(aliased))

    with pytest.raises(UsageError, match="bar_mode"):
        compile_lines(series, kinds=["bar", "bar"], spec={"layout": {"bar_mode": "dodged"}})


def test_compile_line_and_mediogram():
    pytest.importorskip("matplotlib")
    from weather_skills_core.plot.charts import compile_lines, compile_mediogram

    fig = compile_lines(
        [([1, 2, 3], [0.0, 1.0, 2.0], "a"), ([1, 2, 3], [2.0, 1.0, 0.0], "b")],
        title="lines",
        xlabel="t",
        ylabels=["mm", "mm"],
    )
    assert len(_fig(fig).axes[0].lines) == 2
    fc = np.arange(12.0).reshape(3, 4)
    mc = np.arange(12.0, 24.0).reshape(3, 4)
    medio = compile_mediogram(fc, mc, ["+0d", "+1d", "+2d", "+3d"], title="Medio")
    assert len(_fig(medio).axes[0].lines) >= 1
    assert _fig(medio)._suptitle.get_text() == "Medio"


def test_export_png_timeseries_and_mediogram(tmp_path):
    pytest.importorskip("matplotlib")
    from weather_skills_core.plot.charts import compile_lines, compile_mediogram
    from weather_skills_core.plot.figure import export_png

    lines = compile_lines(
        [([1, 2, 3], [0.0, 1.0, 2.0], "a"), ([1, 2, 3], [2.0, 1.0, 0.0], "b")],
        title="lines",
    )
    line_png = tmp_path / "lines.png"
    export_png(_fig(lines), line_png)
    assert line_png.is_file() and line_png.stat().st_size > 0

    fc = np.arange(12.0).reshape(3, 4)
    mc = np.arange(12.0, 24.0).reshape(3, 4)
    medio = compile_mediogram(fc, mc, ["+0d", "+1d", "+2d", "+3d"], title="Medio")
    medio_png = tmp_path / "medio.png"
    export_png(_fig(medio), medio_png)
    assert medio_png.is_file() and medio_png.stat().st_size > 0


def test_export_png_heatmap_grid_with_blank(tmp_path):
    pytest.importorskip("matplotlib")
    from weather_skills_core.plot.figure import export_png
    from weather_skills_core.plot.maps import blank_cell, compile_grid, heatmap_cell
    from weather_skills_core.plot.theme import resolve_colorscale

    ds = make_gridded(n_time=1)
    da = ds["precip"].isel(time=0)
    scale = resolve_colorscale(da, None)
    scale["label"] = "precip"
    fig = compile_grid(
        [[heatmap_cell(da, "latitude", "longitude"), blank_cell("n/a")]],
        extent=[9.5, 13.5, 0.5, 3.5],
        col_titles=["t0", "missing"],
        scales={"field": scale},
        overlays=False,
    )
    out = tmp_path / "grid.png"
    export_png(_fig(fig), out)
    assert out.is_file() and out.stat().st_size > 0


def test_format_step_dates_and_leads():
    from weather_skills_core.plot.maps import format_step

    assert format_step(np.datetime64("2026-01-01T00:00:00")) == "1 Jan '26"
    assert format_step(np.timedelta64(0, "D")) == "+0d"
    assert format_step(np.timedelta64(3, "D")) == "+3d"


def test_panel_title_weekly_range_and_daily_date():
    import xarray as xr

    from weather_skills_core.plot.maps import panel_title, timeseries_axis

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
    from weather_skills_core import UsageError
    from weather_skills_core.plot.theme import (
        along_dim,
        along_member_label,
        parse_along_color,
        parse_band,
        resolve_colorscale,
    )

    assert parse_band("10,90") == (10.0, 90.0)
    assert parse_band(True) == (10.0, 90.0)
    assert parse_along_color(None) == "same"
    assert parse_along_color("cycle") == "cycle"
    assert parse_along_color("distinct") == "cycle"
    with pytest.raises(UsageError, match="same"):
        parse_along_color("rainbow")
    assert along_member_label(3) == "3"
    assert along_member_label(2015.0) == "2015"
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

    from weather_skills_core.plot import compile

    ds = make_forecast(n_number=5)
    spec = spec_from_flags(
        variable="tp",
        kind="timeseries",
        along="number",
        reduce=["latitude", "longitude"],
        band="10,90",
    )
    compiled = compile(spec, {"a": ds})
    fig, resolved = compiled.fig, compiled.spec
    assert resolved["layout"]["shared_colorscale"] is True
    assert resolved["traces"][0]["along_color"] == "same"
    fills = [c for ax in fig.axes for c in ax.collections if isinstance(c, PolyCollection)]
    assert fills
    assert fig.axes[0].lines


def test_compile_timeseries_along_color_cycle_and_same():
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    from matplotlib.colors import to_hex

    from weather_skills_core import UsageError
    from weather_skills_core.plot import compile

    ds = make_forecast(n_number=3)
    shared = spec_from_flags(
        variable="tp",
        kind="timeseries",
        along="number",
        reduce=["latitude", "longitude"],
    )
    compiled_same = compile(shared, {"a": ds})
    fig_same, resolved_same = compiled_same.fig, compiled_same.spec
    same_colors = [to_hex(ln.get_color()) for ln in fig_same.axes[0].lines]
    assert len(same_colors) == 3
    assert len(set(same_colors)) == 1
    assert resolved_same["traces"][0]["along_color"] == "same"
    assert fig_same.axes[0].lines[0].get_label() != "_nolegend_"
    assert all(ln.get_label() == "_nolegend_" for ln in fig_same.axes[0].lines[1:])

    cycled = spec_from_flags(
        variable="tp",
        kind="timeseries",
        along="number",
        along_color="cycle",
        reduce=["latitude", "longitude"],
    )
    compiled_cycle = compile(cycled, {"a": ds})
    fig_cycle, resolved_cycle = compiled_cycle.fig, compiled_cycle.spec
    cycle_colors = [to_hex(ln.get_color()) for ln in fig_cycle.axes[0].lines]
    assert len(cycle_colors) == 3
    assert len(set(cycle_colors)) == 3
    assert resolved_cycle["traces"][0]["along_color"] == "cycle"
    labels = [ln.get_label() for ln in fig_cycle.axes[0].lines]
    assert labels == ["0", "1", "2"]

    with pytest.raises(UsageError, match="cannot be combined"):
        compile(
            spec_from_flags(
                variable="tp",
                kind="timeseries",
                along="number",
                along_color="cycle",
                band="10,90",
                reduce=["latitude", "longitude"],
            ),
            {"a": ds},
        )


def test_compile_timeseries_align_dayofyear():
    pytest.importorskip("matplotlib")
    from weather_skills_core.plot import compile

    ds = make_gridded(n_time=4)
    spec = spec_from_flags(
        variable="precip",
        kind="timeseries",
        align="dayofyear",
        reduce=["latitude", "longitude"],
    )
    compiled = compile(spec, {"a": ds})
    fig, resolved = compiled.fig, compiled.spec
    assert resolved["traces"][0]["align"] == "dayofyear"
    xdata = fig.axes[0].lines[0].get_xdata()
    assert float(xdata[0]) >= 1


def test_colorblind_template_sets_style():
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    from weather_skills_core.plot import compile
    from weather_skills_core.plot.theme import normalize_template

    assert normalize_template("colorblind") == "colorblind"
    ds = make_gridded(n_time=1)
    spec = spec_from_flags(variable="precip", kind="heatmap", template="colorblind")
    compiled = compile(spec, {"a": ds})
    fig, resolved = compiled.fig, compiled.spec
    assert resolved["theme"]["template"] == "colorblind"
    assert _quadmeshes(fig)


def test_shared_colorscale_one_norm_across_panels():
    pytest.importorskip("matplotlib")
    from weather_skills_core.plot import compile

    ds = make_gridded(n_time=3)
    spec = spec_from_flags(variable="precip", kind="heatmap")
    compiled = compile(spec, {"a": ds})
    fig, resolved = compiled.fig, compiled.spec
    assert resolved["layout"]["shared_colorscale"] is True
    meshes = _quadmeshes(fig)
    norms = {id(m.norm) for m in meshes}
    assert len(norms) == 1


def test_pick_rejects_unknown_and_non_json():
    from weather_skills_core import UsageError
    from weather_skills_core.plot.figure import LINE_KEYS, pick

    assert pick({"linewidth": 2.5}, LINE_KEYS, loc="line")["linewidth"] == 2.5
    with pytest.raises(UsageError, match="unknown key"):
        pick({"linewidth": 2, "bogus": 1}, LINE_KEYS, loc="line")
    with pytest.raises(UsageError, match="must be JSON"):
        pick({"linewidth": object()}, LINE_KEYS, loc="line")


def test_apply_axes_xlabel_string_and_object():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    from weather_skills_core import UsageError
    from weather_skills_core.plot.figure import apply_axes

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
    from weather_skills_core.plot.figure import apply_rc

    apply_rc({"axes.grid": False, "lines.linewidth": 3.0})
    assert mpl.rcParams["lines.linewidth"] == 3.0
    with pytest.raises(UsageError, match="backend"):
        apply_rc({"backend": "TkAgg"})
    with pytest.raises(UsageError, match="unknown matplotlib rcParam"):
        apply_rc({"not.a.real.param": 1})


def test_compile_mpl_axes_annotate_mesh_and_log():
    pytest.importorskip("matplotlib")
    from matplotlib.patches import FancyArrowPatch

    from weather_skills_core.plot import compile

    ds = make_gridded(n_time=1)
    spec = spec_from_flags(variable="precip", kind="heatmap")
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
    fig = compile(spec, {"a": ds}).fig
    ax = next(a for a in fig.axes if a.get_label() != "<colorbar>")
    assert ax.spines["top"].get_visible() is False
    assert ax.collections[0].get_alpha() == 0.4
    assert any(isinstance(p, FancyArrowPatch) for p in ax.patches) or ax.texts
    texts = [t.get_text() for t in ax.texts]
    assert "peak" in texts

    ts = spec_from_flags(
        variable="precip",
        kind="timeseries",
        reduce=["latitude", "longitude"],
    )
    ts["axes"] = {"yscale": "log", "legend": {"loc": "lower right"}}
    ts["traces"][0]["line"] = {"linewidth": 4, "linestyle": "--"}
    fig_ts = compile(ts, {"a": ds}).fig
    assert fig_ts.axes[0].get_yscale() == "log"
    assert fig_ts.axes[0].lines[0].get_linewidth() == 4


def test_compile_contour_levels_from_spec():
    pytest.importorskip("matplotlib")
    from matplotlib.contour import QuadContourSet

    from weather_skills_core.plot import compile

    ds = make_gridded(n_time=1)
    spec = spec_from_flags(variable="precip", kind="contour")
    spec["traces"][0]["contour"] = {"levels": 5, "linewidths": 1.2, "lines": True}
    fig = compile(spec, {"a": ds}).fig
    filled = next(c for ax in fig.axes for c in ax.collections if isinstance(c, QuadContourSet))
    assert len(filled.levels) >= 2


def test_compile_line_twin_and_mediogram_colors():
    pytest.importorskip("matplotlib")
    from weather_skills_core.plot.charts import compile_lines, compile_mediogram

    fig = compile_lines(
        [([1, 2, 3], [0.0, 1.0, 2.0], "a"), ([1, 2, 3], [20.0, 10.0, 0.0], "b")],
        styles=[{}, {"twin": "y", "color": "#d62728", "line": {"linewidth": 3}}],
    )
    assert len(_fig(fig).axes) == 2
    assert _fig(fig).axes[1].lines[0].get_linewidth() == 3

    fc = np.arange(12.0).reshape(3, 4)
    mc = np.arange(12.0, 24.0).reshape(3, 4)
    medio = compile_mediogram(
        fc,
        mc,
        ["+0d", "+1d", "+2d", "+3d"],
        spec={
            "traces": [
                {
                    "kind": "mediogram",
                    "mediogram": {"forecast": {"facecolor": "#ff00ff"}, "width": 0.2},
                }
            ]
        },
    )
    box = _fig(medio).axes[0].patches[0]
    facecolor = box.get_facecolor()
    assert facecolor[0] > 0.9 and facecolor[2] > 0.9
    assert facecolor[1] < 0.2


def test_compile_dumps_axes_ticks_and_applies_xticks():
    pytest.importorskip("matplotlib")
    from weather_skills_core.plot import compile
    from weather_skills_core.plot.figure import AXES_TEMPLATE, resolve_axes_block

    # An untouched figure dumps no axes knobs; the editable set is AXES_TEMPLATE.
    assert resolve_axes_block({}) == {}
    assert {"xticks", "yticks", "tick_params", "xlocator"} <= set(AXES_TEMPLATE)
    assert resolve_axes_block({"axes": {"yscale": "log", "xlim": None}}) == {"yscale": "log"}

    ds = make_gridded(n_time=1)
    spec = spec_from_flags(
        variable="precip",
        kind="timeseries",
        reduce=["latitude", "longitude"],
    )
    spec["axes"] = {"yticks": [0.0, 0.5, 1.0], "xticks": {"values": [1, 2], "labels": ["a", "b"]}}
    compiled = compile(spec, {"a": ds})
    fig, resolved = compiled.fig, compiled.spec
    assert resolved["axes"]["yticks"] == [0.0, 0.5, 1.0]
    yticks = [
        float(t)
        for t in fig.axes[0].get_yticks()
        if fig.axes[0].get_ylim()[0] <= t <= fig.axes[0].get_ylim()[1]
    ]
    assert yticks == [0.0, 0.5, 1.0]
    assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == ["a", "b"]


def test_heatmap_grid_and_dumped_spec_use_shared_axes_spec(tmp_path):
    pytest.importorskip("matplotlib")
    from weather_skills_core.plot import export
    from weather_skills_core.plot.maps import compile_grid, heatmap_cell
    from weather_skills_core.plot.theme import resolve_colorscale

    ds = make_gridded(n_time=1)
    da = ds["precip"].isel(time=0)
    scale = resolve_colorscale(da, None)
    spec = {"axes": {"spines": {"top": False}}}
    fig = compile_grid(
        [[heatmap_cell(da, "latitude", "longitude")]],
        extent=[9.5, 13.5, 0.5, 3.5],
        scales={"field": scale},
        overlays=False,
        spec=spec,
    )
    ax = next(a for a in _fig(fig).axes if a.get_label() != "<colorbar>")
    assert ax.spines["top"].get_visible() is False
    out = tmp_path / "grid.png"
    dumped_path = out.with_name("grid.plot.json")
    export(fig, out, spec=spec, dump_spec_path=dumped_path)
    dumped = json.loads(dumped_path.read_text())
    assert dumped["axes"] == {"spines": {"top": False}}
