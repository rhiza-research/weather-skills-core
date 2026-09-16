"""Compile a weather-skills plot spec + Datasets into a Plotly figure."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

import numpy as np

from weather_skills_core.cf import auto_variable, cf_dim
from weather_skills_core.errors import UsageError
from weather_skills_core.figure import format_plot_date, format_plot_date_range
from weather_skills_core.plot_spec import apply_index, overlay_spec, panel_shape, parse_index
from weather_skills_core.plot_style import (
    DEFAULT_DPI,
    DEFAULT_FONTSIZE,
    DEFAULT_MAX_COLUMNS,
    register_template,
    resolve_colorscale,
)
from weather_skills_core.standard_utils import (
    ensure_normalized_longitude,
    lat_slice,
    parse_bbox,
    polygon_from_geojson,
)
from weather_skills_core.units import (
    DATA_INTERVAL_ATTR,
    parse_aggregation_period,
    precip_for_display,
    to_standard_units,
    variable_label_for_display,
)


def _plain(da):
    if getattr(da.pint, "units", None) is not None:
        return da.pint.dequantify()
    return da


def _step_dim(da):
    for cand in ("step", "time", "valid_time"):
        if cand in da.dims:
            return cand
    cf = cf_dim(da, "time")
    return cf if cf and cf in da.dims else None


def _pad_cell_extent(lat_vals, lon_vals):
    lat_vals = np.asarray(lat_vals, dtype=float)
    lon_vals = np.asarray(lon_vals, dtype=float)
    dlat = float(np.nanmean(np.abs(np.diff(lat_vals)))) if lat_vals.size > 1 else 0.5
    dlon = float(np.nanmean(np.abs(np.diff(lon_vals)))) if lon_vals.size > 1 else 0.5
    lon_min = float(np.nanmin(lon_vals)) - dlon / 2
    lon_max = float(np.nanmax(lon_vals)) + dlon / 2
    lat_min = float(np.nanmin(lat_vals)) - dlat / 2
    lat_max = float(np.nanmax(lat_vals)) + dlat / 2
    if lon_max - lon_min >= 360.0 - 1e-6:
        lon_min, lon_max = -180.0, 180.0
    lat_min = max(lat_min, -90.0)
    lat_max = min(lat_max, 90.0)
    return [lon_min, lon_max, lat_min, lat_max]


def _figsize_from_extent(lon_min, lon_max, lat_min, lat_max, base_height=5.0):
    lat_range = abs(lat_max - lat_min)
    lon_range = abs(lon_max - lon_min)
    if lat_range == 0 or lon_range == 0:
        return base_height, base_height
    height = base_height
    width = height * lon_range / lat_range
    return max(width, 2.0), height


def _subset_spatial(da, lat_dim, lon_dim, bbox_nwse, region_polygon, extent_vals):
    if bbox_nwse is None and region_polygon is None:
        return da, extent_vals
    import xarray as xr

    da = ensure_normalized_longitude(da, lon_dim)
    if bbox_nwse is not None:
        r_n, r_w, r_s, r_e = bbox_nwse
        da = da.sel({lat_dim: lat_slice(da[lat_dim].values, r_n, r_s)})
        if r_w > r_e:
            da = da.where((da[lon_dim] >= r_w) | (da[lon_dim] <= r_e), drop=True)
        else:
            da = da.sel({lon_dim: slice(r_w, r_e)})
    if region_polygon is not None:
        import shapely

        lon_grid, lat_grid = np.meshgrid(da[lon_dim].values, da[lat_dim].values)
        mask = shapely.contains_xy(region_polygon, lon_grid, lat_grid)
        da = da.where(xr.DataArray(mask, dims=(lat_dim, lon_dim)))
    if bbox_nwse is not None:
        r_n, r_w, r_s, r_e = bbox_nwse
        if r_w > r_e:
            shifted = ((da[lon_dim] - r_w) % 360.0) + r_w
            da = da.assign_coords({lon_dim: shifted}).sortby(lon_dim)
        if extent_vals is None:
            if r_w > r_e:
                extent_vals = [float(r_w), float(r_e) + 360.0, float(r_s), float(r_n)]
            else:
                extent_vals = [float(r_w), float(r_e), float(r_s), float(r_n)]
    return da, extent_vals


def _parse_extent(spec):
    if not spec:
        return None
    if isinstance(spec, (list, tuple)) and len(spec) == 4:
        return [float(x) for x in spec]
    parts = [float(x) for x in str(spec).split(",")]
    if len(parts) != 4:
        raise UsageError("--extent expects lon_min,lon_max,lat_min,lat_max")
    return parts


def _parse_cities(spec):
    if not spec:
        return {}
    if isinstance(spec, dict):
        data = spec
    else:
        p = Path(spec)
        raw = p.read_text(encoding="utf-8") if p.exists() else str(spec)
        data = json.loads(raw)
    out = {}
    for name, val in data.items():
        if isinstance(val, dict):
            out[name] = (float(val["lat"]), float(val["lon"]))
        else:
            out[name] = (float(val[0]), float(val[1]))
    return out


def _parse_draw_boxes(specs):
    if not specs:
        return []
    boxes = []
    for spec in specs:
        if isinstance(spec, (list, tuple)) and len(spec) == 4:
            boxes.append([float(x) for x in spec])
        else:
            boxes.append(list(parse_bbox(spec)))
    return boxes


def _format_step(value):
    arr = np.asarray(value)
    if arr.dtype.kind == "m":
        days = arr / np.timedelta64(1, "D")
        return f"{int(days)}d" if float(days).is_integer() else str(value)
    return str(value)


def _calendar_bin_width(da, all_steps):
    period = da.attrs.get("aggregation_period")
    if isinstance(period, str) and period.strip():
        try:
            return parse_aggregation_period(period)
        except UsageError:
            pass
    step_arr = np.asarray(all_steps)
    if step_arr.dtype.kind == "M" and step_arr.size > 1:
        deltas = np.diff(step_arr).astype("timedelta64[D]").astype(float)
        if deltas.size:
            return float(np.median(deltas))
    return None


def _format_calendar_panel(step_value, width):
    if width is None:
        return format_plot_date(step_value)
    try:
        days = float(width.to("day").magnitude) if hasattr(width, "to") else float(width)
    except (TypeError, ValueError, AttributeError):
        return format_plot_date(step_value)
    if days <= 1.5:
        return format_plot_date(step_value)
    start = np.asarray(step_value, dtype="datetime64[D]") - np.timedelta64(
        int(round(days)) - 1, "D"
    )
    return format_plot_date_range(start, step_value)


def panel_title(da, sdim, step_value, all_steps):
    """Human panel label: calendar range, forecast valid window, or ``<sdim>=…``."""
    step_arr = np.asarray(all_steps)
    value_arr = np.asarray(step_value)
    if step_arr.dtype.kind == "m" and "time" in da.coords and getattr(da["time"], "ndim", 1) == 0:
        fallback = f"{sdim}={_format_step(step_value)}"
        try:
            time_val = np.asarray(da["time"].values)
            start = time_val + np.asarray(step_value)
            dt = None
            interval = da.attrs.get(DATA_INTERVAL_ATTR)
            if isinstance(interval, str) and interval.strip():
                try:
                    seconds = float(parse_aggregation_period(interval).to("second").magnitude)
                    dt = np.timedelta64(int(round(seconds)), "s")
                except (TypeError, ValueError, UsageError):
                    dt = None
            if dt is None:
                dt = step_arr[1] - step_arr[0] if step_arr.size > 1 else np.timedelta64(1, "D")
            end = start + dt
            return f"{format_plot_date(start)} until {format_plot_date(end)}"
        except Exception:  # noqa: BLE001
            return fallback
    if value_arr.dtype.kind == "M" or hasattr(step_value, "calendar"):
        return _format_calendar_panel(step_value, _calendar_bin_width(da, all_steps))
    return f"{sdim}={_format_step(step_value)}"


def timeseries_axis(da, sdim):
    """X coord and xlabel. Forecast ``step`` + scalar init → valid times."""
    if sdim != "step" or "time" not in da.coords:
        return da[sdim].values, sdim
    time_coord = da["time"]
    if getattr(time_coord, "ndim", 1) != 0:
        return da[sdim].values, sdim
    steps = np.asarray(da["step"].values)
    if steps.dtype.kind != "m":
        return da[sdim].values, sdim
    init = np.asarray(time_coord.values)
    if init.dtype.kind != "M":
        return da[sdim].values, sdim
    return (init + steps).astype("datetime64[ns]"), "Valid time"


def _axis_label(text):
    if text is None:
        return text
    s = str(text).strip()
    if not s:
        return s
    known = {
        "lon": "Longitude",
        "lat": "Latitude",
        "longitude": "Longitude",
        "latitude": "Latitude",
        "valid time": "Valid time",
        "calendar day": "Calendar day",
        "time": "Time",
        "step": "Step",
        "forecast step": "Forecast step",
    }
    key = s.lower()
    if key in known:
        return known[key]
    return s[:1].upper() + s[1:] if s else s


def _geojson_lines(_extent):
    """Country outlines from bundled Natural Earth."""
    raw = json.loads(files("weather_skills_core.data").joinpath("countries.geojson").read_text())
    xs, ys = [], []

    def _add_ring(ring):
        if not ring:
            return
        xs.extend(float(p[0]) for p in ring)
        ys.extend(float(p[1]) for p in ring)
        xs.append(None)
        ys.append(None)

    def _walk(geom):
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        if gtype == "Polygon":
            for ring in coords:
                _add_ring(ring)
        elif gtype == "MultiPolygon":
            for poly in coords:
                for ring in poly:
                    _add_ring(ring)
        elif gtype == "LineString":
            _add_ring(coords)
        elif gtype == "MultiLineString":
            for line in coords:
                _add_ring(line)

    for feat in raw.get("features") or []:
        geom = feat.get("geometry") or {}
        _walk(geom)
    if not xs:
        return [], []
    return xs, ys


def _prepare_field(ds, spec_input: dict, geo: dict, style: str):
    variable = spec_input.get("variable") or auto_variable(ds)
    if not variable or variable not in ds:
        raise UsageError(f"no usable variable. Available: {list(ds.data_vars)}")
    ds = to_standard_units(ds, variables=[variable])
    ds = precip_for_display(ds, variable)
    da = ds[variable]
    try:
        overrides = parse_index(spec_input.get("index"))
    except ValueError as exc:
        raise UsageError(str(exc)) from None
    bbox_raw = geo.get("bbox")
    if isinstance(bbox_raw, str):
        bbox_nwse = parse_bbox(bbox_raw)
    elif bbox_raw is not None:
        bbox_nwse = tuple(bbox_raw)
    else:
        bbox_nwse = None
    mask_geojson = geo.get("mask_geojson")
    region_polygon = polygon_from_geojson(mask_geojson) if mask_geojson else None
    extent = _parse_extent(geo.get("extent"))

    if style == "timeseries":
        da = apply_index(da, overrides, list_dims=())
        sdim = "step" if "step" in da.dims else cf_dim(da, "time")
        if sdim is None:
            raise UsageError(f"timeseries needs 'step' or 'time'; got {list(da.dims)}.")
        reduce_dims = [d for d in da.dims if d != sdim]
        reduced = da.mean(reduce_dims, keep_attrs=True) if reduce_dims else da
        return {"da": _plain(reduced), "sdim": sdim}

    lat_dim = cf_dim(da, "latitude")
    lon_dim = cf_dim(da, "longitude")
    if lat_dim is None or lon_dim is None:
        raise UsageError(f"{style} requires lat/lon coords; got {list(da.dims)}.")
    if lat_dim not in da.dims or lon_dim not in da.dims:
        raise UsageError(
            f"{style} needs lat/lon as dimensions, but {lat_dim!r}/"
            f"{lon_dim!r} are non-dimension coordinates (dims: {list(da.dims)})"
        )
    native_step_dim = _step_dim(da)
    native_steps = list(da[native_step_dim].values) if native_step_dim else None
    list_dims = (native_step_dim,) if native_step_dim else ()
    da = apply_index(da, overrides, list_dims=list_dims)
    panel_dim = _step_dim(da)
    for dim in da.dims:
        if dim not in (panel_dim, "number", lat_dim, lon_dim):
            panel_desc = repr(panel_dim) if panel_dim else "step/time"
            raise UsageError(
                f"dimension {dim!r} remains after selection; {style} "
                f"panels only the {panel_desc} dimension — select a position "
                f"from {dim!r} with --index"
            )
    wrapped_bbox = bbox_nwse is not None and bbox_nwse[1] > bbox_nwse[3]
    da, extent = _subset_spatial(da, lat_dim, lon_dim, bbox_nwse, region_polygon, extent)
    if da.sizes[lat_dim] == 0 or da.sizes[lon_dim] == 0:
        raise UsageError("selection produced an empty grid; nothing to plot.")
    if "number" in da.dims:
        da = da.mean("number", keep_attrs=True)
    da = _plain(da)
    if not wrapped_bbox:
        da = ensure_normalized_longitude(da, lon_dim)
    if extent is None:
        extent = _pad_cell_extent(da[lat_dim].values, da[lon_dim].values)
    return {
        "da": da,
        "lat_dim": lat_dim,
        "lon_dim": lon_dim,
        "extent": extent,
        "native_step_dim": native_step_dim,
        "native_steps": native_steps,
        "variable": variable,
    }


def _coloraxis(scale, label, n_panels, vmin, vmax):
    cmin = scale.get("cmin") if vmin is None else vmin
    cmax = scale.get("cmax") if vmax is None else vmax
    colorbar = {
        "title": {"text": label or ""},
        "orientation": "v" if n_panels <= 1 else "h",
        "y": 0.5 if n_panels <= 1 else -0.12,
        "x": 1.02 if n_panels <= 1 else 0.5,
        "len": 0.8 if n_panels <= 1 else 0.7,
    }
    bounds = scale.get("bounds")
    if bounds:
        colorbar["tickvals"] = bounds
        colorbar["tickmode"] = "array"
    axis = {"colorscale": scale["colorscale"], "colorbar": colorbar, "showscale": True}
    if cmin is not None:
        axis["cmin"] = cmin
    if cmax is not None:
        axis["cmax"] = cmax
    return axis


def _apply_plotly_patch(fig, patch: dict | None):
    if not patch:
        return fig
    layout = patch.get("layout")
    if layout:
        fig.update_layout(**layout)
    annotations = patch.get("annotations")
    if annotations:
        existing = list(fig.layout.annotations or ())
        fig.update_layout(annotations=list(existing) + list(annotations))
    shapes = patch.get("shapes")
    if shapes:
        existing = list(fig.layout.shapes or ())
        fig.update_layout(shapes=list(existing) + list(shapes))
    return fig


def _compile_timeseries(prepared, spec, fontsize):
    import plotly.graph_objects as go

    da = prepared["da"]
    sdim = prepared["sdim"]
    xvals, default_xlabel = timeseries_axis(da, sdim)
    qty = variable_label_for_display(da, include_units=False)
    xlabel = spec.get("xlabel")
    if xlabel is None:
        xlabel = "" if np.asarray(xvals).dtype.kind in "Mm" else _axis_label(default_xlabel)
    else:
        xlabel = _axis_label(xlabel)
    ylabel = spec.get("ylabel") or variable_label_for_display(da)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=_as_plotly_x(xvals),
            y=np.asarray(da.values, dtype=float),
            mode="lines+markers",
            name=qty,
            marker={"size": 8},
            line={"width": 2},
        )
    )
    legend = spec.get("legend")
    showlegend = legend not in (None, "none", "off")
    fig.update_layout(
        template=register_template(fontsize),
        title=spec.get("title") or f"{qty} (timeseries)",
        xaxis_title=xlabel,
        yaxis_title=_axis_label(ylabel) if ylabel else ylabel,
        showlegend=showlegend,
        autosize=spec.get("layout", {}).get("autosize", True),
    )
    return fig


def _as_plotly_x(values):
    arr = np.asarray(values)
    if arr.dtype.kind == "M":
        return np.datetime_as_string(arr, unit="s").tolist()
    if arr.dtype.kind == "m":
        return (arr / np.timedelta64(1, "D")).astype(float).tolist()
    return arr.tolist()


def _compile_heatmap(prepared, spec, fontsize):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    da = prepared["da"]
    lat_dim = prepared["lat_dim"]
    lon_dim = prepared["lon_dim"]
    extent = prepared["extent"]
    sdim = _step_dim(da)
    if sdim is None or da.sizes.get(sdim, 1) == 1:
        if sdim and sdim in da.dims:
            da = da.squeeze(sdim, drop=True)
        steps = [None]
        sdim = None
    else:
        steps = list(da[sdim].values)
    n = len(steps)
    facet = spec.get("layout", {}).get("facet") or {}
    max_columns = int(facet.get("max_columns") or DEFAULT_MAX_COLUMNS)
    nrows, ncols = panel_shape(
        n,
        rows=facet.get("rows"),
        columns=facet.get("columns"),
        max_columns=max_columns,
    )
    user_titles = list(spec.get("subplot_titles") or [])
    if len(user_titles) > n:
        raise UsageError(
            f"--subplot-title was passed {len(user_titles)} time(s) but this figure "
            f"has {n} panel(s)"
        )
    titles = []
    title_steps = prepared.get("native_steps") if prepared.get("native_step_dim") == sdim else steps
    for i, step in enumerate(steps):
        if i < len(user_titles):
            titles.append(user_titles[i])
        elif step is not None:
            titles.append(panel_title(da, sdim, step, title_steps))
        else:
            titles.append("")

    scale = resolve_colorscale(
        da,
        (spec.get("style") or {}).get("colormap"),
        stretch=spec.get("vmin") is not None or spec.get("vmax") is not None,
    )
    label = spec.get("cbar_label") or variable_label_for_display(da)
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.04,
        vertical_spacing=0.12 if nrows > 1 else 0.08,
        subplot_titles=titles + [""] * (nrows * ncols - n),
    )
    lon = np.asarray(da[lon_dim].values, dtype=float)
    lat = np.asarray(da[lat_dim].values, dtype=float)
    overlays = spec.get("geo", {}).get("overlays", "auto")
    geo_x = geo_y = None
    if overlays not in (None, False, "none", "off"):
        geo_x, geo_y = _geojson_lines(extent)
    cities = _parse_cities(spec.get("geo", {}).get("cities"))
    boxes = _parse_draw_boxes(spec.get("geo", {}).get("draw_boxes"))
    xlabel = _axis_label(spec.get("xlabel") or "Longitude")
    ylabel = _axis_label(spec.get("ylabel") or "Latitude")

    for i, _step in enumerate(steps):
        row, col = divmod(i, ncols)
        row += 1
        col += 1
        slab = da if sdim is None else da.isel({sdim: i})
        slab = slab.transpose(lat_dim, lon_dim)
        z = np.asarray(slab.values, dtype=float)
        fig.add_trace(
            go.Heatmap(
                x=lon,
                y=lat,
                z=z,
                coloraxis="coloraxis",
                hoverongaps=False,
            ),
            row=row,
            col=col,
        )
        if geo_x:
            fig.add_trace(
                go.Scatter(
                    x=geo_x,
                    y=geo_y,
                    mode="lines",
                    line={"color": "#444444", "width": 0.6},
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
        if cities:
            fig.add_trace(
                go.Scatter(
                    x=[lon_ for _, lon_ in cities.values()],
                    y=[lat_ for lat_, _ in cities.values()],
                    mode="markers+text",
                    marker={"color": "black", "size": 8},
                    text=list(cities.keys()),
                    textposition="top left",
                    hoverinfo="text",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
        fig.update_xaxes(range=[extent[0], extent[1]], row=row, col=col)
        xref = "x" if i == 0 else f"x{i + 1}"
        fig.update_yaxes(
            range=[extent[2], extent[3]],
            scaleanchor=xref,
            scaleratio=1,
            constrain="domain",
            row=row,
            col=col,
        )
        show_x = row == nrows
        show_y = col == 1
        fig.update_xaxes(title_text=xlabel if show_x else "", row=row, col=col)
        fig.update_yaxes(title_text=ylabel if show_y else "", row=row, col=col)

    for j in range(n, nrows * ncols):
        row, col = divmod(j, ncols)
        fig.update_xaxes(visible=False, row=row + 1, col=col + 1)
        fig.update_yaxes(visible=False, row=row + 1, col=col + 1)

    sw, sh = _figsize_from_extent(*extent)
    figsize = spec.get("layout", {}).get("figsize")
    autosize = spec.get("layout", {}).get("autosize", True)
    layout_kw = {
        "template": register_template(fontsize),
        "title": spec.get("title") or None,
        "coloraxis": _coloraxis(scale, label, n, spec.get("vmin"), spec.get("vmax")),
        "showlegend": False,
        "autosize": autosize and figsize is None,
    }
    if figsize is not None:
        dpi = spec.get("style", {}).get("dpi") or DEFAULT_DPI
        layout_kw["width"] = int(float(figsize[0]) * dpi)
        layout_kw["height"] = int(float(figsize[1]) * dpi)
        layout_kw["autosize"] = False
    else:
        dpi = spec.get("style", {}).get("dpi") or DEFAULT_DPI
        layout_kw["width"] = int(sw * ncols * dpi)
        layout_kw["height"] = int(sh * nrows * dpi)
    fig.update_layout(**layout_kw)

    shapes = []
    for box in boxes:
        n_, w, s_, e = box
        shapes.append(
            {
                "type": "rect",
                "xref": "x",
                "yref": "y",
                "x0": w,
                "x1": e,
                "y0": s_,
                "y1": n_,
                "line": {"color": "black", "width": 1.5},
                "fillcolor": "rgba(0,0,0,0)",
            }
        )
    if shapes:
        fig.update_layout(shapes=shapes)
    return fig, scale, (nrows, ncols), n


def compile_figure(spec: dict, datasets: dict):
    """Compile ``spec`` against ``datasets`` ``{id: Dataset}``.

    Returns ``(fig, resolved_spec)``. ``resolved_spec`` has defaults filled in.
    """
    spec = overlay_spec({"version": 1, "layout": {}, "style": {}, "geo": {}}, spec)
    traces = spec.get("traces") or [{"type": "heatmap", "input": "a"}]
    trace0 = traces[0]
    style = trace0.get("type") or "heatmap"
    input_id = trace0.get("input") or "a"
    inputs = spec.get("inputs") or []
    spec_input = next((i for i in inputs if i.get("id") == input_id), None)
    if spec_input is None:
        spec_input = inputs[0] if inputs else {"id": input_id}
    ds = datasets.get(input_id) or datasets.get("a")
    if ds is None and len(datasets) == 1:
        ds = next(iter(datasets.values()))
    if ds is None:
        raise UsageError("plot spec has no Dataset for the requested input")
    if "path" not in spec_input:
        spec_input = {**spec_input, "id": spec_input.get("id", input_id)}
    fontsize = int((spec.get("style") or {}).get("fontsize") or DEFAULT_FONTSIZE)
    prepared = _prepare_field(ds, spec_input, spec.get("geo") or {}, style)
    if style == "timeseries":
        fig = _compile_timeseries(prepared, spec, fontsize)
        scale = None
        nrows = ncols = n = 1
    elif style in ("heatmap", "contour"):
        fig, scale, (nrows, ncols), n = _compile_heatmap(prepared, spec, fontsize)
    else:
        raise UsageError(f"plotly compiler does not yet support style {style!r}")

    if spec.get("annotations"):
        fig.update_layout(
            annotations=list(fig.layout.annotations or ()) + list(spec["annotations"])
        )
    if spec.get("shapes") and style == "timeseries":
        fig.update_layout(shapes=list(fig.layout.shapes or ()) + list(spec["shapes"]))
    fig = _apply_plotly_patch(fig, spec.get("plotly"))

    resolved = overlay_spec(spec, {})
    resolved["traces"] = traces
    resolved.setdefault("inputs", inputs or [spec_input])
    resolved["style"] = {
        **(resolved.get("style") or {}),
        "template": "weather_skills",
        "fontsize": fontsize,
        "colormap": (scale or {}).get("name") or (resolved.get("style") or {}).get("colormap"),
    }
    if style in ("heatmap", "contour"):
        resolved["layout"] = {
            **(resolved.get("layout") or {}),
            "facet": {
                **((resolved.get("layout") or {}).get("facet") or {}),
                "rows": nrows,
                "columns": ncols,
                "max_columns": (resolved.get("layout") or {})
                .get("facet", {})
                .get("max_columns", DEFAULT_MAX_COLUMNS),
                "n_panels": n,
            },
        }
        if prepared.get("extent"):
            resolved.setdefault("geo", {})["extent"] = prepared["extent"]
        if spec_input.get("variable") or prepared.get("variable"):
            resolved["inputs"] = [
                {**spec_input, "variable": spec_input.get("variable") or prepared.get("variable")}
            ]
    return fig, resolved
