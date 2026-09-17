"""Compile a weather-skills plot spec + Datasets into a matplotlib figure."""

from __future__ import annotations

import json
import sys
from importlib.resources import files
from pathlib import Path

import numpy as np

from weather_skills_core.cf import auto_variable, cf_dim
from weather_skills_core.errors import UsageError
from weather_skills_core.figure import (
    add_shared_colorbar,
    apply_date_ticks,
    apply_style,
    axis_label,
    colorbar_size_kwargs,
    colorbar_spec,
    format_plot_date,
    format_plot_date_range,
    resolve_axis_label,
)
from weather_skills_core.plot_spec import apply_index, overlay_spec, panel_shape, parse_index
from weather_skills_core.plot_style import (
    DEFAULT_FONTSIZE,
    DEFAULT_MAX_COLUMNS,
    aggregation_days,
    along_dim,
    mpl_cmap_norm,
    normalize_template,
    parse_band,
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


def plain(da):
    if getattr(da.pint, "units", None) is not None:
        return da.pint.dequantify()
    return da


def step_dim(da):
    for cand in ("step", "time", "valid_time"):
        if cand in da.dims:
            return cand
    cf = cf_dim(da, "time")
    return cf if cf and cf in da.dims else None


def pad_cell_extent(lat_vals, lon_vals):
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


def figsize_from_extent(lon_min, lon_max, lat_min, lat_max, base_height=5.0):
    lat_range = abs(lat_max - lat_min)
    lon_range = abs(lon_max - lon_min)
    if lat_range == 0 or lon_range == 0:
        return base_height, base_height
    height = base_height
    width = height * lon_range / lat_range
    return max(width, 2.0), height


def subset_spatial(da, lat_dim, lon_dim, bbox_nwse, region_polygon, extent_vals):
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


def slice_bbox_mask(da, lat_dim, lon_dim, bbox, polygon, label):
    """Slice a lat/lon field to ``bbox`` / polygon; error if the grid is empty."""
    if bbox is None and polygon is None:
        return da
    da, _ = subset_spatial(da, lat_dim, lon_dim, bbox, polygon, None)
    if polygon is not None:
        import shapely

        lon_grid, lat_grid = np.meshgrid(da[lon_dim].values, da[lat_dim].values)
        if not bool(shapely.contains_xy(polygon, lon_grid, lat_grid).any()):
            print(
                f"Warning: --mask-geojson polygon does not intersect {label}; "
                "its panels will be entirely empty.",
                file=sys.stderr,
            )
    if da.sizes.get(lat_dim, 0) == 0 or da.sizes.get(lon_dim, 0) == 0:
        raise UsageError(
            f"selection produced an empty grid on {label} "
            "(no cells remain after --bbox/--mask-geojson); nothing to plot."
        )
    return da


def extent_from_da(da, lat_dim, lon_dim, bbox=None):
    """Lon/lat extent from ``bbox`` (NWSE) or half-cell padding around ``da``."""
    if bbox is not None:
        r_n, r_w, r_s, r_e = bbox
        if r_w > r_e:
            return [float(r_w), float(r_e) + 360.0, float(r_s), float(r_n)]
        return [float(r_w), float(r_e), float(r_s), float(r_n)]
    return pad_cell_extent(da[lat_dim].values, da[lon_dim].values)


def is_cftime_axis(values):
    arr = np.asarray(values)
    return (
        getattr(arr.dtype, "kind", None) == "O"
        and arr.size > 0
        and hasattr(arr.flat[0], "calendar")
    )


def axis_kind(values):
    kind = getattr(np.asarray(values).dtype, "kind", None)
    if kind == "M":
        return "datetime"
    if kind == "m":
        return "timedelta"
    if is_cftime_axis(values):
        return "datetime"
    return None


def parse_extent(spec):
    if not spec:
        return None
    if isinstance(spec, (list, tuple)) and len(spec) == 4:
        return [float(x) for x in spec]
    parts = [float(x) for x in str(spec).split(",")]
    if len(parts) != 4:
        raise UsageError("--extent expects lon_min,lon_max,lat_min,lat_max")
    return parts


def parse_cities(spec):
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


def parse_draw_boxes(specs):
    if not specs:
        return []
    boxes = []
    for spec in specs:
        if isinstance(spec, (list, tuple)) and len(spec) == 4:
            boxes.append(tuple(float(x) for x in spec))
        else:
            boxes.append(tuple(parse_bbox(spec)))
    return boxes


def format_step(value):
    """Lead-time or calendar label: ``+3d`` or ``1 Jan '26``."""
    arr = np.asarray(value)
    if arr.dtype.kind == "M":
        return format_plot_date(value)
    if arr.dtype.kind == "m":
        days = int(arr.astype("timedelta64[D]").astype("int64").reshape(-1)[0])
        return f"+{days}d"
    if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day"):
        return format_plot_date(value)
    return str(value)


def calendar_bin_width(da, all_steps):
    """Days in a left-labeled multi-day calendar bin, or None for a single date."""
    days = aggregation_days(da)
    if days is not None and days >= 2:
        return float(days)
    arr = np.asarray(all_steps)
    if arr.size < 2:
        return None
    if arr.dtype.kind == "M":
        try:
            diffs = np.diff(arr.astype("datetime64[ns]").astype("int64"))
        except (TypeError, ValueError):
            return None
        positive = diffs[diffs > 0]
        if positive.size == 0:
            return None
        median_ns = float(np.median(positive))
        if median_ns < 2 * 86_400_000_000_000:
            return None
        return median_ns / 86_400_000_000_000
    if is_cftime_axis(arr):
        ordered = np.sort(arr)
        deltas = [
            abs((ordered[i + 1] - ordered[i]).total_seconds()) for i in range(ordered.size - 1)
        ]
        if not deltas:
            return None
        seconds = float(np.median(deltas))
        if seconds < 2 * 86_400:
            return None
        return seconds / 86_400
    return None


def format_calendar_panel(value, bin_width=None):
    """``14 Sept '26``, or ``4–10 Aug '26`` for multi-day left-edge bins."""
    import datetime as _dt

    if bin_width is None:
        return format_plot_date(value)
    try:
        if hasattr(bin_width, "to"):
            days = float(bin_width.to("day").magnitude)
        elif hasattr(bin_width, "total_seconds"):
            days = float(bin_width.total_seconds()) / 86_400
        else:
            days = float(bin_width)
    except (TypeError, ValueError, AttributeError):
        return format_plot_date(value)
    if days <= 1.5:
        return format_plot_date(value)

    offset_days = int(round(days)) - 1
    if hasattr(value, "calendar"):
        try:
            end = value + _dt.timedelta(days=offset_days)
        except (TypeError, ValueError):
            return format_plot_date(value)
        return format_plot_date_range(value, end)

    try:
        start = np.asarray(value, dtype="datetime64[D]")
        end = start + np.timedelta64(offset_days, "D")
        return format_plot_date_range(start, end)
    except (TypeError, ValueError):
        return format_step(value)


def panel_title(da, sdim, step_value, all_steps):
    """Human panel label: calendar range, forecast valid window, or ``<sdim>=…``."""
    step_arr = np.asarray(all_steps)
    value_arr = np.asarray(step_value)
    if step_arr.dtype.kind == "m" and "time" in da.coords and getattr(da["time"], "ndim", 1) == 0:
        fallback = f"{sdim}={format_step(step_value)}"
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
        return format_calendar_panel(step_value, calendar_bin_width(da, all_steps))
    return f"{sdim}={format_step(step_value)}"


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


def geojson_lines(_extent):
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
    extent = parse_extent(geo.get("extent"))

    if style == "timeseries":
        da = apply_index(da, overrides, list_dims=())
        sdim = "step" if "step" in da.dims else cf_dim(da, "time")
        if sdim is None:
            raise UsageError(f"timeseries needs 'step' or 'time'; got {list(da.dims)}.")
        along = along_dim(da, spec_input.get("along"))
        reduce_req = spec_input.get("reduce") or []
        if isinstance(reduce_req, str):
            reduce_req = [reduce_req]
        applicable = [d for d in reduce_req if d in da.dims]
        if applicable:
            da = da.mean(applicable, keep_attrs=True)
        extras = [d for d in da.dims if d != sdim]
        if along == sdim:
            raise UsageError(
                f"along {spec_input.get('along')!r} is the time axis "
                f"({sdim!r}); pass a non-time dim such as number."
            )
        if along:
            extras = [d for d in extras if d != along]
        if extras:
            if reduce_req or along:
                hint = extras[0]
                raise UsageError(
                    f"variable still has non-time dims {extras} after reduce. "
                    f"Pass reduce for each leftover dim, or along {hint}."
                )
            da = da.mean(extras, keep_attrs=True)
            along = None
        if along:
            da = da.transpose(sdim, along)
        return {
            "da": plain(da),
            "sdim": sdim,
            "along": along,
            "align": spec_input.get("align"),
            "band": spec_input.get("band"),
        }

    lat_dim = cf_dim(da, "latitude")
    lon_dim = cf_dim(da, "longitude")
    if lat_dim is None or lon_dim is None:
        raise UsageError(f"{style} requires lat/lon coords; got {list(da.dims)}.")
    if lat_dim not in da.dims or lon_dim not in da.dims:
        raise UsageError(
            f"{style} needs lat/lon as dimensions, but {lat_dim!r}/"
            f"{lon_dim!r} are non-dimension coordinates (dims: {list(da.dims)})"
        )
    native_step_dim = step_dim(da)
    native_steps = list(da[native_step_dim].values) if native_step_dim else None
    list_dims = (native_step_dim,) if native_step_dim else ()
    da = apply_index(da, overrides, list_dims=list_dims)
    panel_dim = step_dim(da)
    for dim in da.dims:
        if dim not in (panel_dim, "number", lat_dim, lon_dim):
            panel_desc = repr(panel_dim) if panel_dim else "step/time"
            raise UsageError(
                f"dimension {dim!r} remains after selection; {style} "
                f"panels only the {panel_desc} dimension — select a position "
                f"from {dim!r} with --index"
            )
    wrapped_bbox = bbox_nwse is not None and bbox_nwse[1] > bbox_nwse[3]
    da, extent = subset_spatial(da, lat_dim, lon_dim, bbox_nwse, region_polygon, extent)
    if da.sizes[lat_dim] == 0 or da.sizes[lon_dim] == 0:
        raise UsageError("selection produced an empty grid; nothing to plot.")
    if "number" in da.dims:
        da = da.mean("number", keep_attrs=True)
    da = plain(da)
    if not wrapped_bbox:
        da = ensure_normalized_longitude(da, lon_dim)
    if extent is None:
        extent = pad_cell_extent(da[lat_dim].values, da[lon_dim].values)
    return {
        "da": da,
        "lat_dim": lat_dim,
        "lon_dim": lon_dim,
        "extent": extent,
        "native_step_dim": native_step_dim,
        "native_steps": native_steps,
        "variable": variable,
    }


def _title_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return str(value.get("text") or "")
    return str(value)


def as_plot_x(values):
    """Matplotlib x values: datetime64 stays; timedeltas become days."""
    arr = np.asarray(values)
    if arr.dtype.kind == "m":
        return arr / np.timedelta64(1, "D")
    return arr


def draw_geo_lines(ax, geo_x, geo_y, *, lw=0.6):
    if not geo_x:
        return
    x = [np.nan if v is None else v for v in geo_x]
    y = [np.nan if v is None else v for v in geo_y]
    ax.plot(x, y, color="#444444", linewidth=lw, zorder=3)


def _apply_patch(fig, axes, patch: dict | None, *, map_extent=None):
    """Apply spec ``patch`` (title, annotations, shapes, font, colorbar size)."""
    if not patch:
        return fig
    layout = patch.get("layout") or {}
    title = _title_text(layout.get("title") or patch.get("title"))
    if title:
        fig.suptitle(title)
    font = layout.get("font") or {}
    if font.get("size"):
        apply_style(int(font["size"]))
    anns = list(patch.get("annotations") or layout.get("annotations") or [])
    shapes = list(patch.get("shapes") or layout.get("shapes") or [])
    ax0 = np.ravel(axes)[0]
    for ann in anns:
        if not isinstance(ann, dict) or not ann.get("text"):
            continue
        x = float(ann.get("x") if ann.get("x") is not None else 0.5)
        y = float(ann.get("y") if ann.get("y") is not None else 0.5)
        xref = str(ann.get("xref") or "")
        transform = ax0.transAxes if "domain" in xref or xref == "paper" else None
        ax0.text(
            x,
            y,
            str(ann["text"]),
            transform=transform or ax0.transData,
            zorder=7,
        )
    from matplotlib.patches import Rectangle

    for shape in shapes:
        if not isinstance(shape, dict) or shape.get("type") not in (None, "rect"):
            continue
        x0, x1 = float(shape.get("x0", 0)), float(shape.get("x1", 0))
        y0, y1 = float(shape.get("y0", 0)), float(shape.get("y1", 0))
        ax0.add_patch(
            Rectangle(
                (min(x0, x1), min(y0, y1)),
                abs(x1 - x0),
                abs(y1 - y0),
                fill=False,
                edgecolor="black",
                linewidth=1.5,
                zorder=6,
            )
        )
    return fig


def _compile_timeseries(prepared, spec, fontsize, *, template="weather_skills"):
    import matplotlib.pyplot as plt

    apply_style(fontsize, template=template, chart="line")
    da = prepared["da"]
    sdim = prepared["sdim"]
    align = prepared.get("align") or spec.get("align")
    xvals, default_xlabel = timeseries_axis(da, sdim)
    if align in ("dayofyear", "day_of_year", "day-of-year"):
        try:
            xvals = da[sdim].dt.dayofyear.values
        except (TypeError, AttributeError) as exc:
            raise UsageError("align=dayofyear needs a calendar-date time axis.") from exc
        default_xlabel = "calendar day"
    qty = variable_label_for_display(da, include_units=False)
    xlabel = spec.get("xlabel")
    if xlabel is None:
        xlabel = (
            ""
            if np.asarray(xvals).dtype.kind in "Mm"
            and align
            not in (
                "dayofyear",
                "day_of_year",
                "day-of-year",
            )
            else axis_label(default_xlabel)
        )
    else:
        xlabel = resolve_axis_label(xlabel, default_xlabel)
    ylabel = resolve_axis_label(spec.get("ylabel"), variable_label_for_display(da))
    figsize = spec.get("layout", {}).get("figsize") or (10.0, 5.0)
    fig, ax = plt.subplots(figsize=tuple(figsize), layout="constrained")
    xplot = as_plot_x(xvals)
    yarr = np.asarray(da.values, dtype=float)
    band = parse_band(prepared.get("band") or spec.get("band"))
    color = "C0"
    if yarr.ndim == 2 and band is not None:
        low = np.nanpercentile(yarr, band[0], axis=1)
        high = np.nanpercentile(yarr, band[1], axis=1)
        mean = np.nanmean(yarr, axis=1)
        ax.fill_between(xplot, low, high, color=color, alpha=0.25, linewidth=0, zorder=1)
        ax.plot(xplot, mean, color=color, linewidth=2, label=qty, zorder=3)
    elif yarr.ndim == 2:
        for j in range(yarr.shape[1]):
            ax.plot(
                xplot,
                yarr[:, j],
                color=color,
                linewidth=1.0,
                alpha=0.35,
                label=qty if j == 0 else "_nolegend_",
            )
    else:
        ax.plot(xplot, yarr, marker="o", markersize=8, linewidth=2, color=color, label=qty)
    if np.asarray(xvals).dtype.kind == "M":
        apply_date_ticks(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    legend = spec.get("legend")
    if legend not in (None, "none", "off"):
        ax.legend()
    fig.suptitle(spec.get("title") or f"{qty} (timeseries)")
    fig._ws_tight = (
        spec.get("layout", {}).get("autosize", True)
        and spec.get("layout", {}).get("figsize") is None
    )
    return fig


def _compile_heatmap(prepared, spec, fontsize, *, trace_type="heatmap", template="weather_skills"):
    import matplotlib.pyplot as plt

    apply_style(fontsize, template=template, chart="map")
    da = prepared["da"]
    lat_dim = prepared["lat_dim"]
    lon_dim = prepared["lon_dim"]
    extent = prepared["extent"]
    sdim = step_dim(da)
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
    if spec.get("vmin") is not None:
        scale["cmin"] = spec["vmin"]
    if spec.get("vmax") is not None:
        scale["cmax"] = spec["vmax"]
    label = spec.get("cbar_label") or variable_label_for_display(da)
    cmap, norm = mpl_cmap_norm(scale)
    lon = np.asarray(da[lon_dim].values, dtype=float)
    lat = np.asarray(da[lat_dim].values, dtype=float)
    overlays = spec.get("geo", {}).get("overlays", "auto")
    geo_x = geo_y = None
    if overlays not in (None, False, "none", "off"):
        geo_x, geo_y = geojson_lines(extent)
    cities = parse_cities(spec.get("geo", {}).get("cities"))
    boxes = parse_draw_boxes(spec.get("geo", {}).get("draw_boxes"))
    xlabel = resolve_axis_label(spec.get("xlabel"), "Longitude")
    ylabel = resolve_axis_label(spec.get("ylabel"), "Latitude")

    sw, sh = figsize_from_extent(*extent)
    user_figsize = spec.get("layout", {}).get("figsize")
    if user_figsize is not None:
        fig_w, fig_h = float(user_figsize[0]), float(user_figsize[1])
    else:
        fig_w, fig_h = sw * ncols, sh * nrows
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(fig_w, fig_h), squeeze=False, layout="constrained"
    )
    mappable = None
    for i, _step in enumerate(steps):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        slab = da if sdim is None else da.isel({sdim: i})
        slab = slab.transpose(lat_dim, lon_dim)
        z = np.asarray(slab.values, dtype=float)
        if trace_type == "contour":
            filled = ax.contourf(lon, lat, z, cmap=cmap, norm=norm, levels=12)
            ax.contour(lon, lat, z, levels=filled.levels, colors="black", linewidths=0.4)
            mappable = filled
        else:
            mappable = ax.pcolormesh(lon, lat, z, cmap=cmap, norm=norm, shading="nearest")
        draw_geo_lines(ax, geo_x, geo_y)
        if cities:
            ax.scatter(
                [lon_ for _, lon_ in cities.values()],
                [lat_ for lat_, _ in cities.values()],
                c="black",
                s=64,
                zorder=5,
            )
            for name, (clat, clon) in cities.items():
                ax.annotate(
                    name, (clon, clat), textcoords="offset points", xytext=(-4, 4), ha="right"
                )
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal", adjustable="box")
        if titles[i]:
            ax.set_title(titles[i])
        if row == nrows - 1:
            ax.set_xlabel(xlabel)
        if col == 0:
            ax.set_ylabel(ylabel)
        for box in boxes:
            n_, w, s_, e = box
            from matplotlib.patches import Rectangle

            ax.add_patch(
                Rectangle(
                    (w, s_), e - w, n_ - s_, fill=False, edgecolor="black", linewidth=1.5, zorder=6
                )
            )

    for j in range(n, nrows * ncols):
        row, col = divmod(j, ncols)
        axes[row, col].set_visible(False)
        axes[row, col].set_axis_off()

    visible = [ax for ax in axes.ravel() if ax.get_visible()]
    ticks = scale.get("bounds")
    add_shared_colorbar(
        fig,
        mappable,
        visible,
        label=label,
        location="right" if n <= 1 else "bottom",
        ticks=ticks,
        **colorbar_size_kwargs(spec),
    )
    if spec.get("title"):
        fig.suptitle(spec["title"])
    fig._ws_tight = spec.get("layout", {}).get("autosize", True) and user_figsize is None
    return fig, scale, (nrows, ncols), n, axes


def compile_figure(spec: dict, datasets: dict):
    """Compile ``spec`` against ``datasets`` ``{id: Dataset}``.

    Returns ``(fig, resolved_spec)``. ``resolved_spec`` has defaults filled in.
    ``fig`` is a matplotlib Figure.
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
    spec_input = dict(spec_input)
    for key in ("along", "reduce", "align", "band"):
        if spec_input.get(key) is None:
            if trace0.get(key) is not None:
                spec_input[key] = trace0[key]
            elif spec.get(key) is not None:
                spec_input[key] = spec[key]
    ds = datasets.get(input_id) or datasets.get("a")
    if ds is None and len(datasets) == 1:
        ds = next(iter(datasets.values()))
    if ds is None:
        raise UsageError("plot spec has no Dataset for the requested input")
    if "path" not in spec_input:
        spec_input = {**spec_input, "id": spec_input.get("id", input_id)}
    fontsize = int((spec.get("style") or {}).get("fontsize") or DEFAULT_FONTSIZE)
    template = normalize_template((spec.get("style") or {}).get("template"))
    prepared = _prepare_field(ds, spec_input, spec.get("geo") or {}, style)
    axes = None
    if style == "timeseries":
        fig = _compile_timeseries(prepared, spec, fontsize, template=template)
        scale = None
        nrows = ncols = n = 1
        axes = fig.axes
    elif style in ("heatmap", "contour"):
        fig, scale, (nrows, ncols), n, axes = _compile_heatmap(
            prepared, spec, fontsize, trace_type=style, template=template
        )
    else:
        raise UsageError(f"plot compiler does not yet support style {style!r}")

    _apply_patch(fig, axes, spec.get("patch"), map_extent=prepared.get("extent"))
    if spec.get("annotations") and axes is not None:
        _apply_patch(fig, axes, {"annotations": spec["annotations"]})
    if spec.get("shapes") and style == "timeseries":
        _apply_patch(fig, axes, {"shapes": spec["shapes"]})

    resolved = overlay_spec(spec, {})
    resolved["traces"] = traces
    resolved.setdefault("inputs", inputs or [spec_input])
    resolved["style"] = {
        **(resolved.get("style") or {}),
        "template": template,
        "fontsize": fontsize,
        "colormap": (scale or {}).get("name") or (resolved.get("style") or {}).get("colormap"),
    }
    layout = resolved.setdefault("layout", {})
    layout.setdefault("shared_colorscale", True)
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
        cbar = colorbar_spec(spec)
        if cbar:
            resolved["layout"]["colorbar"] = cbar
    return fig, resolved
