"""Lon/lat figures: heatmap, contour, quiver, layers, compare/verify grids.

Cartopy supplies the axes; Natural Earth overlays are drawn here. Spatial
subsetting (bbox/mask/extent) lives here because only geographic figures need it.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib.resources import files
from pathlib import Path

import numpy as np

from weather_skills_core.cf import auto_variable, cf_dim
from weather_skills_core.display_labels import resolve_input_labels
from weather_skills_core.errors import UsageError
from weather_skills_core.plot.figure import (
    add_shared_colorbar,
    apply_style_then_rc,
    apply_suptitle,
    colorbar_mpl_kwargs,
    facet_figure,
    finish_figure,
    format_plot_date,
    format_plot_date_range,
    mesh_kwargs,
    quiver_kwargs,
    resolve_axis_label,
    resolve_figsize,
    scatter_kwargs,
    settle_figure,
    wrap_axes_title,
)
from weather_skills_core.plot.spec import (
    apply_index,
    fold_layer_options,
    panel_shape,
    parse_index,
    trace_at,
)
from weather_skills_core.plot.theme import (
    DEFAULT_FONTSIZE,
    DISCRETE_PRECIP_NAMES,
    aggregation_days,
    mpl_cmap_norm,
    parse_colormap_spec,
    resolve_colorscale,
    resolve_mpl_cmap_name,
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
    units_equal,
    variable_label_for_display,
    variable_units,
)

# Natural Earth scale vs map span (max of the lon/lat extent in degrees).
# Admin-1 (states / provinces / counties) is only readable on country-scale
# views; a multi-country or basin map would be a thicket of province lines.
ADMIN1_MAX_SPAN_DEG = 20.0
HIRES_MAX_SPAN_DEG = 45.0
MIDRES_MAX_SPAN_DEG = 90.0

# Slate water fill (Lake Victoria, Turkana, …). Saturated CHC precip blues
# (#50a5f5 / #1e6eeb) look like rainfall; this grey-blue does not.
LAKE_FACECOLOR = "#708090"
ADMIN1_STYLE = {"facecolor": "none", "edgecolor": "0.45", "linewidth": 0.4, "zorder": 3}
LAKES_STYLE = {
    "facecolor": LAKE_FACECOLOR,
    "edgecolor": LAKE_FACECOLOR,
    "linewidth": 0.4,
    "zorder": 3.5,
}
BORDERS_STYLE = {"facecolor": "none", "edgecolor": "0.15", "linewidth": 0.8, "zorder": 4}
COAST_STYLE = {"facecolor": "none", "edgecolor": "black", "linewidth": 0.8, "zorder": 4}


def extent_span_deg(extent):
    lon_min, lon_max, lat_min, lat_max = extent
    return max(abs(lon_max - lon_min), abs(lat_max - lat_min))


def boundary_layers(extent):
    """Natural Earth scale and whether to overlay admin-1 for this view."""
    span = extent_span_deg(extent)
    if span > MIDRES_MAX_SPAN_DEG:
        return {"scale": "110m", "admin1": False}
    if span > HIRES_MAX_SPAN_DEG:
        return {"scale": "50m", "admin1": False}
    return {"scale": "10m", "admin1": span <= ADMIN1_MAX_SPAN_DEG}


def extent_clip_geom(extent):
    """Shapely clip geometry for ``lon_min,lon_max,lat_min,lat_max``.

    Antimeridian views store a continuous unwrapped lon (e.g. 170..190) which
    is split back into ``[-180, 180]`` pieces for Natural Earth intersection.
    """
    from shapely.geometry import box

    lon_min, lon_max, lat_min, lat_max = extent
    if lon_max > 180.0:
        return box(lon_min, lat_min, 180.0, lat_max).union(
            box(-180.0, lat_min, lon_max - 360.0, lat_max)
        )
    if lon_min > lon_max:
        return box(lon_min, lat_min, 180.0, lat_max).union(box(-180.0, lat_min, lon_max, lat_max))
    return box(lon_min, lat_min, lon_max, lat_max)


def unwrap_geoms(geoms, lon_min):
    """Shift western-hemisphere pieces so they match an unwrapped lon axis."""
    import numpy as np
    import shapely

    def shift(coords):
        out = np.asarray(coords).copy()
        out[:, 0] = np.where(out[:, 0] < lon_min, out[:, 0] + 360.0, out[:, 0])
        return out

    return [shapely.transform(g, shift) for g in geoms if g is not None and not g.is_empty]


def clip_ne_geoms(resolution, category, name, clip_geom):
    """Natural Earth geometries intersecting ``clip_geom`` (eager download)."""
    import cartopy.io.shapereader as shpreader

    path = shpreader.natural_earth(resolution=resolution, category=category, name=name)
    geoms = []
    for geom in shpreader.Reader(path).geometries():
        if geom is None or geom.is_empty:
            continue
        try:
            if not geom.intersects(clip_geom):
                continue
            clipped = geom.intersection(clip_geom)
        except Exception:  # noqa: BLE001
            clipped = geom
        if clipped is None or clipped.is_empty:
            continue
        if clipped.geom_type == "GeometryCollection":
            geoms.extend(g for g in clipped.geoms if g is not None and not g.is_empty)
        else:
            geoms.append(clipped)
    return geoms


def _bundled_country_geoms(clip_geom):
    """Country outlines from the bundled Natural Earth extract (no cartopy)."""
    import shapely
    from shapely.geometry import shape

    raw = json.loads(files("weather_skills_core.data").joinpath("countries.geojson").read_text())
    geoms = []
    for feat in raw.get("features") or []:
        geom = feat.get("geometry")
        if not geom:
            continue
        try:
            poly = shape(geom)
            if not poly.intersects(clip_geom):
                continue
            geoms.append(poly.intersection(clip_geom))
        except Exception:  # noqa: BLE001
            continue
    return [shapely.boundary(g) for g in geoms if g is not None and not g.is_empty]


def load_geo_overlays(extent):
    """Scale-appropriate coastline / border / filled-lake / admin-1 overlays.

    Returns a list of ``(geometries, matplotlib style)`` layers, each clipped to
    the map extent so a country-scale view does not draw the rest of the world.
    Download or clip failures warn and skip that layer — the map still renders.
    """
    if extent is None:
        return []
    spec = boundary_layers(extent)
    try:
        clip = extent_clip_geom(extent)
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: geographic overlays unavailable ({exc}); skipping.", file=sys.stderr)
        return []
    lon_min, lon_max = extent[0], extent[1]
    layers = []

    def add(category, name, style, resolution=None):
        res = resolution or spec["scale"]
        try:
            geoms = clip_ne_geoms(res, category, name, clip)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: {name} overlay unavailable ({exc}); skipping.", file=sys.stderr)
            return
        if lon_max > 180.0:
            geoms = unwrap_geoms(geoms, lon_min)
        if geoms:
            layers.append((geoms, style))

    try:
        import cartopy.io.shapereader  # noqa: F401
    except ImportError:
        try:
            geoms = _bundled_country_geoms(clip)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: geographic overlays unavailable ({exc}); skipping.", file=sys.stderr)
            return []
        if lon_max > 180.0:
            geoms = unwrap_geoms(geoms, lon_min)
        return [(geoms, BORDERS_STYLE)] if geoms else []

    if spec["admin1"]:
        add("cultural", "admin_1_states_provinces", ADMIN1_STYLE, resolution="10m")
    add("physical", "lakes", LAKES_STYLE)
    add("cultural", "admin_0_boundary_lines_land", BORDERS_STYLE)
    add("physical", "coastline", COAST_STYLE)
    return layers


def draw_geo_overlays(ax, overlays, crs=None):
    """Draw ``load_geo_overlays`` layers on ``ax``.

    A cartopy GeoAxes takes the geometries directly; a plain Axes gets them as
    line collections so both map paths share one set of overlays.
    """
    if not overlays:
        return
    if crs is not None and hasattr(ax, "add_geometries"):
        for geoms, style in overlays:
            ax.add_geometries(geoms, crs, **style)
        return
    from matplotlib.collections import LineCollection, PathCollection
    from matplotlib.path import Path as MplPath

    for geoms, style in overlays:
        segments, rings = [], []
        for geom in geoms:
            for part in getattr(geom, "geoms", [geom]):
                if part.geom_type in ("Polygon",):
                    rings.append(list(part.exterior.coords))
                    segments.extend(list(hole.coords) for hole in part.interiors)
                elif part.geom_type in ("LineString", "LinearRing"):
                    segments.append(list(part.coords))
        face = style.get("facecolor", "none")
        if rings and face not in (None, "none"):
            ax.add_collection(
                PathCollection(
                    [MplPath(ring) for ring in rings],
                    facecolors=face,
                    edgecolors=style.get("edgecolor", face),
                    linewidths=style.get("linewidth", 0.5),
                    zorder=style.get("zorder", 3),
                )
            )
        else:
            segments.extend(rings)
        if segments:
            ax.add_collection(
                LineCollection(
                    segments,
                    colors=style.get("edgecolor", "#444444"),
                    linewidths=style.get("linewidth", 0.6),
                    zorder=style.get("zorder", 3),
                )
            )


def draw_box_outlines(ax, boxes, transform=None):
    """Outline each N/W/S/E box in black (split antimeridian spans into two)."""
    from matplotlib.patches import Rectangle

    kw = {"fill": False, "edgecolor": "black", "linewidth": 1.5, "zorder": 6}
    if transform is not None:
        kw["transform"] = transform
    for north, west, south, east in boxes:
        height = north - south
        if west <= east:
            ax.add_patch(Rectangle((west, south), east - west, height, **kw))
        else:
            # Antimeridian: west..180 and -180..east
            ax.add_patch(Rectangle((west, south), 180.0 - west, height, **kw))
            ax.add_patch(Rectangle((-180.0, south), east + 180.0, height, **kw))


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


def _facet_spacing(wspace, hspace):
    """Return ``(wspace, hspace)`` or ``None`` when neither gap is set."""
    if wspace is None and hspace is None:
        return None
    return (
        None if wspace is None else float(wspace),
        None if hspace is None else float(hspace),
    )


def _figsize_with_spacing(size, nrows, ncols, spacing):
    """Grow a default canvas so GridSpec gaps are reserved, not stolen from panels."""
    if spacing is None:
        return tuple(size)
    width, height = float(size[0]), float(size[1])
    wspace, hspace = spacing
    if wspace and ncols > 1:
        width *= 1.0 + wspace * (ncols - 1) / ncols
    if hspace and nrows > 1:
        height *= 1.0 + hspace * (nrows - 1) / nrows
    return width, height


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


WIND_ROSE_SECTORS = 16
WIND_SPEED_EDGES_MS = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
WIND_SPEED_COLORS = [
    "#c6dbef",
    "#6baed6",
    "#2171b5",
    "#08306b",
    "#fd8d3c",
    "#d94801",
    "#7f2704",
]
_UV_NAME_PAIRS = (
    ("u10", "v10"),
    ("u100", "v100"),
    ("10u", "10v"),
    ("uas", "vas"),
    ("ua", "va"),
    ("u", "v"),
    ("eastward_wind", "northward_wind"),
    ("10m_u_component_of_wind", "10m_v_component_of_wind"),
    ("100m_u_component_of_wind", "100m_v_component_of_wind"),
    ("u_component_of_wind", "v_component_of_wind"),
    ("uwind", "vwind"),
    ("uwnd", "vwnd"),
)
_SAMPLE_DIM_NAMES = {"step", "number", "point_id", "station_id", "valid_time"}
QUIVER_CMAP = "YlGn"
QUIVER_SCALE = 100.0
QUIVER_STEP = 1
QUIVER_TARGET_SPACING_DEG = 1.5
QUIVER_ARROW_LEN_SPACING = 1.5
QUIVER_KEY_MS = (5.0, 10.0)
_LAYER_KINDS = frozenset({"heatmap", "scatter", "quiver", "outline", "mask"})
_ZARR_LAYER_KINDS = frozenset({"heatmap", "scatter", "quiver"})
_LAYER_OPTION_KEYS = frozenset(
    {
        "variable",
        "colormap",
        "index",
        "u_variable",
        "v_variable",
        "quiver_scale",
        "quiver_step",
        "vmin",
        "vmax",
    }
)
_KIND_ZORDER = {"heatmap": 1.0, "quiver": 5.0, "scatter": 6.0, "outline": 7.0}


class LayerSpec:
    """One ``--layer KIND:PATH[::k=v]`` entry. The decorator may set ``.ds``."""

    def __init__(self, kind, path, options, raw):
        self.kind = kind
        self.path = Path(path)
        self.options = options
        self.raw = raw
        self.ds = None

    def zarr_paths(self):
        if self.kind in _ZARR_LAYER_KINDS:
            return [self.path]
        return []

    def __str__(self):
        return self.raw

    def __repr__(self):
        return f"LayerSpec({self.raw!r})"


def _parse_layer_options(blob):
    """Parse ``k=v,k=v``; tokens without ``=`` continue the previous value (for ``index=step=0,1,2``)."""
    options = {}
    current = None
    for token in blob.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" in token:
            key, _, val = token.partition("=")
            key = key.strip()
            if not key:
                raise UsageError(f"--layer option {token!r} has an empty key")
            if key not in _LAYER_OPTION_KEYS:
                raise UsageError(
                    f"unknown --layer option {key!r}; "
                    f"expected one of {', '.join(sorted(_LAYER_OPTION_KEYS))}"
                )
            if key in options:
                raise UsageError(f"--layer option {key!r} is given more than once")
            current = key
            options[key] = val.strip()
        else:
            if current is None:
                raise UsageError(f"--layer option {token!r} appears before any key=value")
            options[current] = f"{options[current]},{token}"
    return options


def parse_layer(value):
    """Argparse converter for ``KIND:PATH`` or ``KIND:PATH::k=v[,k=v...]``."""
    if not value or not str(value).strip():
        raise argparse.ArgumentTypeError("--layer spec is empty")
    raw = str(value).strip()
    if "::" in raw:
        head, _, opt_blob = raw.partition("::")
    else:
        head, opt_blob = raw, ""
    if ":" not in head:
        raise argparse.ArgumentTypeError(
            f"--layer {raw!r} must be KIND:PATH (e.g. heatmap:/tmp/a.zarr)"
        )
    kind, _, path = head.partition(":")
    kind = kind.strip().lower()
    path = path.strip()
    if kind not in _LAYER_KINDS:
        raise argparse.ArgumentTypeError(
            f"unknown --layer kind {kind!r}; expected one of {', '.join(sorted(_LAYER_KINDS))}"
        )
    if not path:
        raise argparse.ArgumentTypeError(f"--layer {raw!r} is missing a path")
    try:
        options = _parse_layer_options(opt_blob) if opt_blob else {}
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    return LayerSpec(kind, path, options, raw)


def _subset_points(da, bbox_nwse, region_polygon):
    """Filter station / point samples to ``--bbox`` / ``--mask-geojson``."""
    import numpy as np
    import xarray as xr

    lat_name = cf_dim(da, "latitude")
    lon_name = cf_dim(da, "longitude")
    if lat_name is None or lon_name is None:
        raise UsageError(
            f"--bbox/--mask-geojson need latitude/longitude coordinates; got dims {list(da.dims)}"
        )
    lat = np.asarray(da[lat_name].values)
    lon = np.asarray(da[lon_name].values)
    keep = np.ones(np.broadcast(lat, lon).shape, dtype=bool)
    lat_b, lon_b = np.broadcast_arrays(lat, lon)
    if bbox_nwse is not None:
        r_n, r_w, r_s, r_e = bbox_nwse
        keep &= (lat_b >= r_s) & (lat_b <= r_n)
        if r_w > r_e:
            keep &= (lon_b >= r_w) | (lon_b <= r_e)
        else:
            keep &= (lon_b >= r_w) & (lon_b <= r_e)
    if region_polygon is not None:
        import shapely

        keep &= shapely.contains_xy(region_polygon, lon_b, lat_b)
        if not bool(keep.any()):
            print(
                "Warning: --mask-geojson polygon does not intersect the points; "
                "the rose will be empty.",
                file=sys.stderr,
            )
    keep_da = xr.DataArray(keep, dims=da[lat_name].dims)
    return da.where(keep_da, drop=True)


def _prepare_gridded_map(
    da, overrides, bbox_nwse, mask_geojson, extent, *, style, region_polygon=None
):
    """Index, bbox, and mask a lat/lon field for a map panel. Returns a tuple.

    ``(da, lat_dim, lon_dim, extent_vals, wrap_lon, native_step_dim, native_steps)``.
    """
    lat_dim = cf_dim(da, "latitude")
    lon_dim = cf_dim(da, "longitude")
    if lat_dim is None or lon_dim is None:
        raise UsageError(f"{style} requires lat/lon coords; got {list(da.dims)}.")
    if lat_dim not in da.dims or lon_dim not in da.dims:
        raise UsageError(
            f"{style} needs lat/lon as dimensions, but {lat_dim!r}/"
            f"{lon_dim!r} are non-dimension coordinates here (dims: "
            f"{list(da.dims)}); station data has no 2D grid to plot."
        )
    native_step_dim = step_dim(da)
    native_steps = list(da[native_step_dim].values) if native_step_dim else None
    list_dims = (native_step_dim,) if native_step_dim else ()
    da = apply_index(da, overrides, list_dims=list_dims)
    for spatial_dim in (lat_dim, lon_dim):
        if spatial_dim in overrides and spatial_dim not in da.dims:
            raise UsageError(
                f"--index removed the {spatial_dim!r} dimension; {style} needs a 2D lat/lon grid"
            )
    panel_dim = step_dim(da)
    for dim in da.dims:
        if dim not in (panel_dim, "number", lat_dim, lon_dim):
            panel_desc = repr(panel_dim) if panel_dim else "step/time"
            raise UsageError(
                f"dimension {dim!r} remains after selection; {style} "
                f"panels only the {panel_desc} dimension — select a position "
                f"from {dim!r} with --index"
            )
    if panel_dim is not None and da.sizes[panel_dim] == 0:
        raise UsageError(f"dimension {panel_dim!r} has size 0; nothing to plot.")
    extent_vals = parse_extent(extent)
    if region_polygon is None and mask_geojson:
        region_polygon = polygon_from_geojson(mask_geojson)
    wrapped_bbox = bbox_nwse is not None and bbox_nwse[1] > bbox_nwse[3]
    da, extent_vals = subset_spatial(da, lat_dim, lon_dim, bbox_nwse, region_polygon, extent_vals)
    if da.sizes[lat_dim] == 0 or da.sizes[lon_dim] == 0:
        raise UsageError(
            "selection produced an empty grid (no cells remain after "
            "--index/--bbox selection); nothing to plot."
        )
    return da, lat_dim, lon_dim, extent_vals, not wrapped_bbox, native_step_dim, native_steps


def _parse_colormap(spec):
    if spec is None:
        return spec
    parsed = parse_colormap_spec(spec)
    if parsed.get("colors"):
        from matplotlib.colors import LinearSegmentedColormap

        return LinearSegmentedColormap.from_list(parsed.get("name") or "custom", parsed["colors"])
    return resolve_mpl_cmap_name(parsed.get("cmap") or parsed.get("name") or spec)


def _flag_values(da):
    """Sorted CF ``flag_values``, or None."""
    import numpy as np

    raw = da.attrs.get("flag_values")
    if raw is None:
        return None
    values = np.asarray(raw, dtype=float).ravel()
    if values.size < 2:
        return None
    return np.sort(values)


def _discrete_flag_scale(da, colormap):
    """ListedColormap + BoundaryNorm for CF flag fields, or None."""
    import numpy as np
    from matplotlib.colors import BoundaryNorm, ListedColormap

    values = _flag_values(da)
    if values is None:
        return None
    meanings = da.attrs.get("flag_meanings")
    labels = None
    if isinstance(meanings, str) and meanings.strip():
        parts = meanings.split()
        raw = np.asarray(da.attrs.get("flag_values"), dtype=float).ravel()
        if parts and len(parts) == raw.size:
            labels = [parts[i] for i in np.argsort(raw)]
    colors = None
    parsed = parse_colormap_spec(colormap) if colormap else {}
    if parsed.get("colors") and len(parsed["colors"]) == values.size:
        colors = parsed["colors"]
    if colors is None:
        if values.size == 3:
            colors = ["#d73027", "#f0f0f0", "#1a9850"]
        else:
            from matplotlib import colormaps

            tab = colormaps["tab10"](np.linspace(0, 1, values.size))
            colors = [tuple(c) for c in tab]
    mids = (values[:-1] + values[1:]) / 2.0
    bounds = np.concatenate(([values[0] - 0.5], mids, [values[-1] + 0.5]))
    cmap = ListedColormap(colors)
    return cmap, BoundaryNorm(bounds, cmap.N), values, labels


def _heatmap_scale(da, colormap, *, stretch=False, registry=None):
    """Return ``(cmap, norm)``. ``norm`` is set for a discrete class scale.

    ``stretch=True`` (user ``--vmin`` / ``--vmax``) keeps the colors but drops
    ``BoundaryNorm`` so the colorbar can use arbitrary limits.

    Names, comma lists, and ``theme.colormap`` objects all go through
    ``resolve_colorscale`` so a nested precip window, a CHC palette, a
    user-registry name, or a custom discrete ``{colors, bounds}`` object
    resolve the same way.
    """
    scale = resolve_colorscale(da, colormap, stretch=stretch, registry=registry)
    if scale.get("bounds") and not stretch:
        return mpl_cmap_norm(scale)
    colors = scale.get("colors")
    if colors:
        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list(scale.get("name") or "custom", list(colors))
        extras = {}
        if scale.get("under"):
            extras["under"] = scale["under"]
        if scale.get("over"):
            extras["over"] = scale["over"]
        if extras:
            cmap = cmap.with_extremes(**extras)
        return cmap, None
    return resolve_mpl_cmap_name(scale.get("cmap") or scale.get("name") or "rocket"), None


def _layer_optional_float(spec, key):
    raw = spec.options.get(key)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise UsageError(f"--layer option {key}={raw!r} is not a number") from exc


def _resolve_color_limits(da, vmin=None, vmax=None, *, norm=None, flag="--vmin/--vmax"):
    """Resolve colorbar limits. User limits drop a discrete ``BoundaryNorm``.

    When both limits are omitted, diverging data is recentered on zero.
    """
    import numpy as np
    from matplotlib.colors import BoundaryNorm

    user_set = vmin is not None or vmax is not None
    if user_set and isinstance(norm, BoundaryNorm):
        norm = None
    if norm is not None and not user_set:
        return None, None, norm

    data_min = float(da.min(skipna=True).values)
    data_max = float(da.max(skipna=True).values)
    if not np.isfinite(data_min) or not np.isfinite(data_max):
        data_min, data_max = 0.0, 1.0
    lo = data_min if vmin is None else float(vmin)
    hi = data_max if vmax is None else float(vmax)
    if not user_set and hi > 0 and lo < 0:
        m = max(abs(hi), abs(lo))
        lo, hi = -m, m
    if lo > hi:
        raise UsageError(f"{flag}: lower limit {lo} is greater than upper limit {hi}")
    if lo == hi:
        pad = abs(lo) * 0.05 if lo != 0 else 1.0
        lo, hi = lo - pad, hi + pad
    return lo, hi, None


def _cbar_extend_for_limits(da, vmin, vmax):
    """``colorbar(extend=...)`` when data sits outside the user limits."""
    import numpy as np

    if vmin is None or vmax is None:
        return None
    data_min = float(da.min(skipna=True).values)
    data_max = float(da.max(skipna=True).values)
    lo = np.isfinite(data_min) and data_min < vmin
    hi = np.isfinite(data_max) and data_max > vmax
    if lo and hi:
        return "both"
    if lo:
        return "min"
    if hi:
        return "max"
    return None


def _cbar_boundary_kwargs(norm, cmap=None):
    """Colorbar kwargs for a BoundaryNorm scale (ticks, spacing, optional extend)."""
    from matplotlib.colors import BoundaryNorm

    if not isinstance(norm, BoundaryNorm):
        return {}
    kw = {"spacing": "uniform", "ticks": list(norm.boundaries)}
    if getattr(cmap, "name", None) in DISCRETE_PRECIP_NAMES:
        kw["extend"] = "both"
    return kw


def _variable_label(da):
    """Colorbar / axis label from CF ``long_name`` (then GRIB_name, then the name)."""
    return variable_label_for_display(da)


def _resolve_subplot_titles(overrides, n_panels):
    """Return user panel titles; extra flags are an error, fewer fall back to auto."""
    titles = list(overrides or [])
    if len(titles) > n_panels:
        raise UsageError(
            f"--subplot-title was passed {len(titles)} time(s) but this figure "
            f"has {n_panels} panel(s)"
        )
    return titles


def _set_panel_title(ax, index, auto, subplot_titles):
    """Apply ``--subplot-title`` when given for this panel; otherwise ``auto``."""
    text = subplot_titles[index] if index < len(subplot_titles) else auto
    if text:
        ax.set_title(wrap_axes_title(ax, text))


def _apply_geo_axis_labels(ax, xlabel, ylabel, *, xlabel_on=True, ylabel_on=True):
    """Lon/lat names; matplotlib places them relative to the colorbar slot."""
    xlab = resolve_axis_label(xlabel, "Longitude")
    ylab = resolve_axis_label(ylabel, "Latitude")
    ax.set_xlabel(xlab if xlabel_on else "")
    ax.set_ylabel(ylab if ylabel_on else "")


def _wind_component_role(da):
    """``'u'`` / ``'v'`` from CF ``standard_name``, or None."""
    sn = da.attrs.get("standard_name")
    if not isinstance(sn, str) or not sn.strip():
        return None
    key = sn.strip().lower()
    if "eastward" in key and "wind" in key:
        return "u"
    if "northward" in key and "wind" in key:
        return "v"
    return None


def _infer_uv_partner(name, *, want_v):
    """Guess the complementary u/v variable name, or None."""
    pairs = dict(_UV_NAME_PAIRS)
    inv = {v: u for u, v in _UV_NAME_PAIRS}
    if want_v:
        if name in pairs:
            return pairs[name]
        swapped = name.replace("eastward", "northward").replace("u_component", "v_component")
        if swapped != name:
            return swapped
        if name.startswith("u"):
            return "v" + name[1:]
        return None
    if name in inv:
        return inv[name]
    swapped = name.replace("northward", "eastward").replace("v_component", "u_component")
    if swapped != name:
        return swapped
    if name.startswith("v"):
        return "u" + name[1:]
    return None


def _resolve_uv(ds, u_variable, v_variable):
    """Eastward/northward variable names from flags, CF attrs, or common names."""
    names = list(ds.data_vars)
    if u_variable and u_variable not in ds:
        raise UsageError(f"--u_variable {u_variable!r} is not in the data (have {names})")
    if v_variable and v_variable not in ds:
        raise UsageError(f"--v_variable {v_variable!r} is not in the data (have {names})")
    if u_variable and v_variable:
        return u_variable, v_variable
    if u_variable:
        partner = _infer_uv_partner(u_variable, want_v=True)
        if partner and partner in ds:
            return u_variable, partner
        raise UsageError(
            f"--u_variable {u_variable!r} is set but no northward partner was found; "
            "pass --v_variable"
        )
    if v_variable:
        partner = _infer_uv_partner(v_variable, want_v=False)
        if partner and partner in ds:
            return partner, v_variable
        raise UsageError(
            f"--v_variable {v_variable!r} is set but no eastward partner was found; "
            "pass --u_variable"
        )
    u_cf, v_cf = [], []
    for name in names:
        role = _wind_component_role(ds[name])
        if role == "u":
            u_cf.append(name)
        elif role == "v":
            v_cf.append(name)
    if len(u_cf) == 1 and len(v_cf) == 1:
        return u_cf[0], v_cf[0]
    present = set(names)
    matches = [(u, v) for u, v in _UV_NAME_PAIRS if u in present and v in present]
    if matches:
        return matches[0]
    raise UsageError(
        "u/v plot needs eastward (u) and northward (v) wind components; "
        f"could not auto-detect them in {names}. Pass --u_variable and --v_variable."
    )


def _speed_units_display(da):
    raw = variable_units(da)
    if not raw:
        return "m/s"
    if units_equal(raw, "m s-1"):
        return "m/s"
    return raw


def _wind_speed_da(u_da, v_da):
    """Speed from eastward/northward components, with a Wind speed label."""
    import numpy as np
    import xarray as xr

    u_da = plain(u_da)
    v_da = plain(v_da)
    speed = xr.apply_ufunc(np.hypot, u_da, v_da, keep_attrs=False)
    units = variable_units(u_da) or "m s-1"
    speed.name = "speed"
    speed.attrs.update(long_name="Wind speed", units=units, standard_name="wind_speed")
    return speed


def _wind_speed_cbar_label(u_da):
    units_disp = _speed_units_display(u_da)
    blob = " ".join(str(u_da.attrs.get(key) or "") for key in ("long_name", "GRIB_name")).lower()
    if "anomal" in blob:
        return f"Wind speed anomaly [{units_disp}]"
    return f"Wind speed [{units_disp}]"


def _mean_axis_spacing(values, axis):
    """Mean absolute spacing along one axis of a 1-D or 2-D coordinate."""
    import numpy as np

    values = np.asarray(values, dtype=float)
    if values.ndim == 0 or values.shape[axis] < 2:
        return None
    delta = np.diff(values, axis=axis)
    delta = delta[np.isfinite(delta)]
    if delta.size == 0:
        return None
    return float(np.mean(np.abs(delta)))


def _native_spacing_deg(lat, lon):
    """Finest mean lat/lon spacing in degrees, or None if it cannot be measured."""
    import numpy as np

    lat = np.asarray(lat)
    lon = np.asarray(lon)
    if lat.ndim == 1 and lon.ndim == 1:
        spacings = [_mean_axis_spacing(lat, 0), _mean_axis_spacing(lon, 0)]
    else:
        spacings = [
            _mean_axis_spacing(lat, 0),
            _mean_axis_spacing(lon, 1 if lon.ndim > 1 else 0),
        ]
    candidates = [s for s in spacings if s is not None and s > 0]
    return min(candidates) if candidates else None


def _quiver_step(lat, lon, requested=None, target_spacing=QUIVER_TARGET_SPACING_DEG):
    """Stride for quiver arrows.

    ``plot_wind_and_sst_anomaly`` uses ``quiver_step=1`` on the native S2S
    ~1.5° grid. When ``requested`` is set, use that. Otherwise thin finer
    grids (GFS 0.25°, ERA5) to about 1.5° so basin maps match that look.
    """
    if requested is not None:
        if requested < 1:
            raise UsageError("--quiver_step must be >= 1")
        return int(requested)
    spacing = _native_spacing_deg(lat, lon)
    if spacing is None:
        return QUIVER_STEP
    return max(QUIVER_STEP, int(round(target_spacing / spacing)))


def _auto_quiver_scale(u, v, lon_span, spacing_deg, requested=None):
    """Matplotlib quiver ``scale`` (data units per axes-width).

    Larger scale → shorter arrows. ``requested`` (``--quiver_scale``) wins.
    Otherwise size a typical (95th-percentile) wind to about
    ``QUIVER_ARROW_LEN_SPACING`` times the subsampled grid spacing, as a
    fraction of the map width, so 10 m/s basin winds and small anomalies
    both stay readable.
    """
    if requested is not None:
        if requested <= 0:
            raise UsageError("--quiver_scale must be > 0")
        return float(requested)
    import numpy as np

    speed = np.hypot(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
    speed = speed[np.isfinite(speed)]
    if speed.size == 0 or lon_span <= 0 or spacing_deg is None or spacing_deg <= 0:
        return QUIVER_SCALE
    typical = float(np.percentile(speed, 95))
    if typical <= 0:
        return QUIVER_SCALE
    target_deg = QUIVER_ARROW_LEN_SPACING * float(spacing_deg)
    return typical * float(lon_span) / target_deg


def _subsample_quiver(lon, lat, u, v, step):
    """Native-grid u/v subsample, matching plot_wind_and_sst_anomaly."""
    import numpy as np

    lon = np.asarray(lon)
    lat = np.asarray(lat)
    u = np.asarray(u)
    v = np.asarray(v)
    if lon.ndim == 1 and lat.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
    step = max(1, int(step))
    return lon[::step, ::step], lat[::step, ::step], u[::step, ::step], v[::step, ::step]


def _label_key(value):
    import numpy as np

    arr = np.asarray(value)
    if arr.dtype.kind in ("M", "m"):
        return int(arr.astype("int64"))
    obj = arr.item() if getattr(arr, "shape", ()) == () else value
    if hasattr(obj, "calendar"):
        return (obj.calendar, str(obj))
    return obj


def _point_dim(ds):
    for name in ("station_id", "point_id"):
        if name in ds.dims:
            return name
    return None


def _combined_mask_polygon(mask_geojson, layers):
    paths = []
    flags = []
    if mask_geojson:
        paths.append(mask_geojson)
        flags.append("--mask-geojson")
    for spec in layers:
        if spec.kind == "mask":
            paths.append(spec.path)
            flags.append("--layer mask")
    if not paths:
        return None
    from shapely.ops import unary_union

    geoms = [polygon_from_geojson(path, flag=flag) for path, flag in zip(paths, flags, strict=True)]
    return geoms[0] if len(geoms) == 1 else unary_union(geoms)


def _layer_overrides(spec, default_index):
    raw = spec.options.get("index", default_index)
    if not raw:
        return {}
    try:
        return parse_index(raw)
    except ValueError as exc:
        raise UsageError(f"--layer {spec.kind}:{spec.path}: {exc}") from None


def _copy_layer(spec, options=None):
    out = LayerSpec(
        spec.kind, spec.path, options if options is not None else spec.options, spec.raw
    )
    out.ds = spec.ds
    return out


def _ensure_layer_dataset(spec):
    if spec.kind not in _ZARR_LAYER_KINDS:
        return
    if spec.ds is not None:
        return
    import xarray as xr

    if not spec.path.exists():
        raise UsageError(f"input not found: {spec.path}")
    spec.ds = xr.open_zarr(spec.path, consolidated=True)


def _layer_variable(ds, spec):
    variable = spec.options.get("variable") or auto_variable(ds)
    if not variable or variable not in ds:
        raise UsageError(
            f"--layer {spec.kind}:{spec.path}: no usable variable. Available: {list(ds.data_vars)}"
        )
    return variable


def _common_labels(driver_values, other_values, spec):
    other_keys = {_label_key(v) for v in other_values}
    common = [v for v in driver_values if _label_key(v) in other_keys]
    if not common:
        raise UsageError(
            f"no overlapping time bins between the panel axis and --layer {spec.kind}:{spec.path}; "
            "aggregate both inputs to a common resolution first, e.g. with the "
            "aggregate-temporal skill"
        )
    return common


def _align_panel_labels(driver_dim, driver_values, driver_kind, da, spec):
    """Return ``(panel_dim or None, labels or None)`` for this layer vs the driver."""
    other_dim = step_dim(da)
    if other_dim is None:
        return None, None
    other_values = list(da[other_dim].values)
    other_kind = axis_kind(da[other_dim].values)
    if driver_kind != other_kind or driver_kind is None or other_kind is None:
        driver_name = "forecast step" if driver_kind == "timedelta" else "calendar time"
        other_name = "forecast step" if other_kind == "timedelta" else "calendar time"
        if driver_kind == "timedelta" or other_kind == "timedelta":
            raise UsageError(
                f"--layer {spec.kind}:{spec.path} has a {other_name} axis ({other_dim!r}) but the "
                f"panel axis is a {driver_name} axis ({driver_dim!r}). Run the step-to-time skill "
                "on the forecast before overlaying observations."
            )
        raise UsageError(
            f"--layer {spec.kind}:{spec.path} time axis {other_dim!r} is not comparable to "
            f"panel axis {driver_dim!r}"
        )
    return other_dim, _common_labels(driver_values, other_values, spec)


def _extent_from_field(da, lat_dim, lon_dim):
    return pad_cell_extent(da[lat_dim].values, da[lon_dim].values)


def _extent_from_points(da):
    import numpy as np

    lat_name = cf_dim(da, "latitude")
    lon_name = cf_dim(da, "longitude")
    lats = np.asarray(da[lat_name].values, dtype=float)
    lons = np.asarray(da[lon_name].values, dtype=float)
    lats = lats[np.isfinite(lats)]
    lons = lons[np.isfinite(lons)]
    if lats.size == 0 or lons.size == 0:
        raise UsageError("scatter layer has no finite lat/lon coordinates")
    pad = 0.5
    return [
        float(lons.min()) - pad,
        float(lons.max()) + pad,
        float(lats.min()) - pad,
        float(lats.max()) + pad,
    ]


def _prep_heatmap_layer(spec, bbox_nwse, region_polygon, extent, *, registry=None):
    _ensure_layer_dataset(spec)
    ds = spec.ds
    variable = _layer_variable(ds, spec)
    ds = to_standard_units(ds, variables=[variable])
    ds = precip_for_display(ds, variable)
    da = ds[variable]
    overrides = _layer_overrides(spec, spec.options.get("index"))
    da, lat_dim, lon_dim, extent_vals, wrap_lon, native_step_dim, native_steps = (
        _prepare_gridded_map(
            da,
            overrides,
            bbox_nwse,
            None,
            extent,
            style="heatmap",
            region_polygon=region_polygon,
        )
    )
    if wrap_lon:
        da = ensure_normalized_longitude(da, lon_dim)
    if "number" in da.dims:
        da = da.mean("number", keep_attrs=True)
    user_vmin = _layer_optional_float(spec, "vmin")
    user_vmax = _layer_optional_float(spec, "vmax")
    user_vlim = user_vmin is not None or user_vmax is not None
    flag_scale = _discrete_flag_scale(da, spec.options.get("colormap"))
    if flag_scale is not None:
        if user_vlim:
            raise UsageError("--vmin/--vmax cannot be used with CF flag_values fields")
        cmap, norm, flag_ticks, flag_labels = flag_scale
        vmin = vmax = None
    else:
        cmap, norm = _heatmap_scale(
            da, spec.options.get("colormap"), stretch=user_vlim, registry=registry
        )
        flag_ticks = flag_labels = None
        vmin, vmax, norm = _resolve_color_limits(da, user_vmin, user_vmax, norm=norm)
    return {
        "kind": "heatmap",
        "spec": spec,
        "da": da,
        "lat_dim": lat_dim,
        "lon_dim": lon_dim,
        "cmap": cmap,
        "norm": norm,
        "vmin": vmin,
        "vmax": vmax,
        "vlim_user": user_vlim,
        "flag_ticks": flag_ticks,
        "flag_labels": flag_labels,
        "wrap_lon": wrap_lon,
        "native_step_dim": native_step_dim,
        "native_steps": native_steps,
        "panel_dim": step_dim(da),
        "cbar_label": _variable_label(da),
        "variable": variable,
        "units": variable_units(da),
        "zorder": _KIND_ZORDER["heatmap"],
        "draw": spec.options.get("draw"),
        "contour": spec.options.get("contour"),
        "mesh": spec.options.get("mesh"),
    }


def _prep_scatter_layer(spec, bbox_nwse, region_polygon, *, registry=None):
    _ensure_layer_dataset(spec)
    ds = spec.ds
    point_dim = _point_dim(ds)
    if point_dim is None:
        raise UsageError(
            f"--layer scatter:{spec.path} needs a station_id or point_id dimension "
            f"(got dims {list(ds.dims)})"
        )
    variable = _layer_variable(ds, spec)
    ds = to_standard_units(ds, variables=[variable])
    ds = precip_for_display(ds, variable)
    da = ds[variable]
    overrides = _layer_overrides(spec, spec.options.get("index"))
    panel_dim = step_dim(da)
    da = apply_index(da, overrides, list_dims=(panel_dim,) if panel_dim else ())
    if bbox_nwse is not None or region_polygon is not None:
        da = _subset_points(da, bbox_nwse, region_polygon)
    extra = [d for d in da.dims if d not in (panel_dim, point_dim) and d is not None]
    extra = [d for d in extra if d in da.dims]
    if extra:
        if extra == ["number"] or (len(extra) == 1 and extra[0] == "number"):
            da = da.mean("number", keep_attrs=True)
        else:
            raise UsageError(
                f"--layer scatter:{spec.path} still has dimension(s) {extra}; "
                "select a position with index= or reduce them first"
            )
    user_vmin = _layer_optional_float(spec, "vmin")
    user_vmax = _layer_optional_float(spec, "vmax")
    user_vlim = user_vmin is not None or user_vmax is not None
    cmap, norm = _heatmap_scale(
        da, spec.options.get("colormap"), stretch=user_vlim, registry=registry
    )
    vmin, vmax, norm = _resolve_color_limits(da, user_vmin, user_vmax, norm=norm)
    return {
        "kind": "scatter",
        "spec": spec,
        "da": da,
        "ds": ds,
        "point_dim": point_dim,
        "cmap": cmap,
        "norm": norm,
        "vmin": vmin,
        "vmax": vmax,
        "vlim_user": user_vlim,
        "panel_dim": step_dim(da),
        "cbar_label": _variable_label(da),
        "variable": variable,
        "units": variable_units(da),
        "zorder": _KIND_ZORDER["scatter"],
    }


def _prep_quiver_layer(spec, bbox_nwse, region_polygon, extent):
    _ensure_layer_dataset(spec)
    ds = spec.ds
    u_name, v_name = _resolve_uv(ds, spec.options.get("u_variable"), spec.options.get("v_variable"))
    ds = to_standard_units(ds, variables=[u_name, v_name])
    u_da = ds[u_name]
    v_da = ds[v_name]
    u_units = variable_units(u_da)
    v_units = variable_units(v_da)
    if u_units and v_units and not units_equal(u_units, v_units):
        raise UsageError(f"u units {u_units!r} do not match v units {v_units!r}")
    overrides = _layer_overrides(spec, spec.options.get("index"))
    u_da, lat_dim, lon_dim, extent_vals, wrap_lon, native_step_dim, native_steps = (
        _prepare_gridded_map(
            u_da,
            overrides,
            bbox_nwse,
            None,
            extent,
            style="quiver",
            region_polygon=region_polygon,
        )
    )
    v_da, *_ = _prepare_gridded_map(
        v_da,
        overrides,
        bbox_nwse,
        None,
        extent,
        style="quiver",
        region_polygon=region_polygon,
    )
    if wrap_lon:
        u_da = ensure_normalized_longitude(u_da, lon_dim)
        v_da = ensure_normalized_longitude(v_da, lon_dim)
    if "number" in u_da.dims:
        u_da = u_da.mean("number", keep_attrs=True)
        v_da = v_da.mean("number", keep_attrs=True)
    speed = _wind_speed_da(u_da, v_da)
    cmap = (
        _parse_colormap(spec.options.get("colormap"))
        if spec.options.get("colormap")
        else QUIVER_CMAP
    )
    qscale = spec.options.get("quiver_scale")
    qstep = spec.options.get("quiver_step")
    user_vmin = _layer_optional_float(spec, "vmin")
    user_vmax = _layer_optional_float(spec, "vmax")
    user_vlim = user_vmin is not None or user_vmax is not None
    vmin, vmax, _ = _resolve_color_limits(speed, user_vmin, user_vmax)
    return {
        "kind": "quiver",
        "spec": spec,
        "speed": speed,
        "u_da": u_da,
        "v_da": v_da,
        "lat_dim": lat_dim,
        "lon_dim": lon_dim,
        "cmap": cmap,
        "norm": None,
        "vmin": vmin,
        "vmax": vmax,
        "vlim_user": user_vlim,
        "wrap_lon": wrap_lon,
        "native_step_dim": native_step_dim,
        "native_steps": native_steps,
        "panel_dim": step_dim(speed),
        "cbar_label": _wind_speed_cbar_label(u_da),
        "variable": "speed",
        "units": variable_units(u_da),
        "quiver_scale": float(qscale) if qscale is not None else None,
        "quiver_step": int(qstep) if qstep is not None else None,
        "zorder": _KIND_ZORDER["quiver"],
        "draw_mesh": False,
        "quiver": spec.options.get("quiver"),
        "mesh": spec.options.get("mesh"),
    }


def _prep_outline_layer(spec):
    return {
        "kind": "outline",
        "spec": spec,
        "polygon": polygon_from_geojson(spec.path, flag="--layer outline"),
        "panel_dim": None,
        "zorder": _KIND_ZORDER["outline"],
    }


def _layer_field(prepared):
    if prepared["kind"] == "quiver":
        return prepared.get("speed")
    return prepared.get("da")


def _sel_layer(prepared, dim, labels):
    if prepared["kind"] == "heatmap":
        prepared["da"] = prepared["da"].sel({dim: labels})
    elif prepared["kind"] == "scatter":
        prepared["da"] = prepared["da"].sel({dim: labels})
    elif prepared["kind"] == "quiver":
        prepared["speed"] = prepared["speed"].sel({dim: labels})
        prepared["u_da"] = prepared["u_da"].sel({dim: labels})
        prepared["v_da"] = prepared["v_da"].sel({dim: labels})


def _squeeze_layer_dim(prepared, dim):
    field = _layer_field(prepared)
    if field is None or dim not in getattr(field, "dims", ()):
        return
    if field.sizes[dim] != 1:
        return
    if prepared["kind"] == "heatmap":
        prepared["da"] = prepared["da"].squeeze(dim, drop=True)
    elif prepared["kind"] == "scatter":
        prepared["da"] = prepared["da"].squeeze(dim, drop=True)
    elif prepared["kind"] == "quiver":
        prepared["speed"] = prepared["speed"].squeeze(dim, drop=True)
        prepared["u_da"] = prepared["u_da"].squeeze(dim, drop=True)
        prepared["v_da"] = prepared["v_da"].squeeze(dim, drop=True)
    prepared["panel_dim"] = None


def _select_panel(prepared, label):
    """Return a copy of ``prepared`` reduced to one panel label, or the original if static."""
    dim = prepared.get("panel_dim")
    if dim is None or label is None:
        return prepared
    out = dict(prepared)
    if prepared["kind"] == "heatmap":
        out["da"] = prepared["da"].sel({dim: label})
    elif prepared["kind"] == "scatter":
        out["da"] = prepared["da"].sel({dim: label})
    elif prepared["kind"] == "quiver":
        out["speed"] = prepared["speed"].sel({dim: label})
        out["u_da"] = prepared["u_da"].sel({dim: label})
        out["v_da"] = prepared["v_da"].sel({dim: label})
    out["panel_dim"] = None
    return out


def _draw_heatmap_on_ax(ax, prepared, transform):
    da = plain(prepared["da"])
    lat_dim, lon_dim = prepared["lat_dim"], prepared["lon_dim"]
    slab = da.transpose(lat_dim, lon_dim)
    if prepared.get("draw") == "contour":
        # Same layer, same scale — values interpolated between grid points
        # rather than drawn as cell rectangles.
        opts = dict(prepared.get("contour") or {})
        lines = opts.pop("lines", True)
        filled = ax.contourf(
            slab[lon_dim],
            slab[lat_dim],
            slab.values,
            cmap=prepared["cmap"],
            norm=prepared["norm"],
            vmin=prepared["vmin"],
            vmax=prepared["vmax"],
            levels=opts.pop("levels", 12),
            transform=transform,
            zorder=prepared["zorder"],
            **opts,
        )
        if lines is not False:
            ax.contour(
                slab[lon_dim],
                slab[lat_dim],
                slab.values,
                levels=filled.levels,
                colors="black",
                linewidths=0.4,
                transform=transform,
                zorder=prepared["zorder"] + 0.1,
            )
        return filled
    return ax.pcolormesh(
        slab[lon_dim],
        slab[lat_dim],
        slab.values,
        **{
            "cmap": prepared["cmap"],
            "norm": prepared["norm"],
            "vmin": prepared["vmin"],
            "vmax": prepared["vmax"],
            "transform": transform,
            "zorder": prepared["zorder"],
            **mesh_kwargs({"mesh": prepared.get("mesh")}),
        },
    )


def _draw_scatter_on_ax(ax, prepared, transform):
    da = plain(prepared["da"])
    lat_name = cf_dim(da, "latitude")
    lon_name = cf_dim(da, "longitude")
    return ax.scatter(
        da[lon_name].values,
        da[lat_name].values,
        c=da.values,
        cmap=prepared["cmap"],
        norm=prepared["norm"],
        vmin=prepared["vmin"],
        vmax=prepared["vmax"],
        s=30,
        transform=transform,
        zorder=prepared["zorder"],
        edgecolors="k",
        linewidths=0.3,
    )


def _draw_quiver_on_ax(ax, prepared, transform, scale, step, mpl_spec=None):
    u_da = plain(prepared["u_da"])
    v_da = plain(prepared["v_da"])
    lat_dim, lon_dim = prepared["lat_dim"], prepared["lon_dim"]
    u_slab = u_da.transpose(lat_dim, lon_dim)
    v_slab = v_da.transpose(lat_dim, lon_dim)
    mesh = None
    if prepared.get("draw_mesh"):
        speed = plain(prepared["speed"]).transpose(lat_dim, lon_dim)
        mesh = ax.pcolormesh(
            speed[lon_dim],
            speed[lat_dim],
            speed.values,
            cmap=prepared["cmap"],
            vmin=prepared["vmin"],
            vmax=prepared["vmax"],
            transform=transform,
            zorder=1.0,
            **mesh_kwargs(
                {"mesh": prepared["mesh"]} if prepared.get("mesh") is not None else trace_at(mpl_spec)
            ),
        )
    lon_q, lat_q, u_q, v_q = _subsample_quiver(
        u_slab[lon_dim].values,
        u_slab[lat_dim].values,
        u_slab.values,
        v_slab.values,
        step,
    )
    quiv = ax.quiver(
        lon_q,
        lat_q,
        u_q,
        v_q,
        **{
            "transform": transform,
            "scale": scale,
            "color": "k",
            "zorder": prepared["zorder"],
            **quiver_kwargs(
                {"quiver": prepared["quiver"]}
                if prepared.get("quiver") is not None
                else trace_at(mpl_spec)
            ),
        },
    )
    return mesh, quiv


def _draw_outline_on_ax(ax, prepared, crs):
    ax.add_geometries(
        [prepared["polygon"]],
        crs,
        facecolor="none",
        edgecolor="black",
        linewidth=1.2,
        zorder=prepared["zorder"],
    )


def _scale_groups(prepared_layers, shared_scale, independent_scale):
    """Return True if heatmap/scatter layers should share one color scale."""
    data = [p for p in prepared_layers if p["kind"] in ("heatmap", "scatter")]
    if len(data) < 2:
        return False
    if independent_scale:
        return False
    if shared_scale:
        return True
    variables = {p["variable"] for p in data}
    units = {p["units"] for p in data if p["units"]}
    return len(variables) == 1 and len(units) <= 1


def _apply_shared_scale(prepared_layers):
    data = [p for p in prepared_layers if p["kind"] in ("heatmap", "scatter")]
    if not data:
        return
    user = [p for p in data if p.get("vlim_user")]
    if user:
        limits = {(p["vmin"], p["vmax"]) for p in user}
        if len(limits) > 1:
            raise UsageError(
                "shared-scale layers disagree on vmin/vmax; "
                "use --independent-scale or one set of limits"
            )
        cmap, norm = user[0]["cmap"], user[0]["norm"]
        vmin, vmax = user[0]["vmin"], user[0]["vmax"]
    else:
        cmap, norm = data[0]["cmap"], data[0]["norm"]
        if norm is None:
            vmins = [p["vmin"] for p in data if p["vmin"] is not None]
            vmaxs = [p["vmax"] for p in data if p["vmax"] is not None]
            vmin = min(vmins) if vmins else None
            vmax = max(vmaxs) if vmaxs else None
            if vmin is not None and vmax is not None and vmax > 0 and vmin < 0:
                m = max(abs(vmax), abs(vmin))
                vmin, vmax = -m, m
        else:
            vmin = vmax = None
    for p in data:
        p["cmap"] = cmap
        p["norm"] = norm
        p["vmin"] = vmin
        p["vmax"] = vmax


def _plot_layers(
    layers,
    bbox_nwse,
    mask_geojson,
    extent,
    cities,
    title,
    fontsize,
    draw_boxes,
    rows,
    columns,
    variable,
    colormap,
    index,
    u_variable,
    v_variable,
    quiver_scale,
    quiver_step,
    shared_scale,
    independent_scale,
    layer_labels=None,
    xlabel=None,
    ylabel=None,
    figsize=None,
    vmin=None,
    vmax=None,
    subplot_titles=None,
    cbar_label=None,
    mpl_spec=None,
    template="weather_skills",
    registry=None,
    wspace=None,
    hspace=None,
):
    """Stack ``--layer`` entries on shared Cartopy panels.

    The renderer applies its own map chrome, so a caller cannot hand it the
    line-chart theme by mistake.
    """
    import cartopy.crs as ccrs

    from weather_skills_core.plot.figure import apply_style_then_rc

    apply_style_then_rc(mpl_spec or {}, chart="map", fontsize=fontsize, template=template)
    import numpy as np

    if shared_scale and independent_scale:
        raise UsageError("--shared-scale and --independent-scale are mutually exclusive")

    label_slots = resolve_input_labels(layer_labels, len(layers), input_flag="--layer")

    inherited = []
    for spec in layers:
        opts = dict(spec.options)
        if "variable" not in opts and variable:
            opts["variable"] = variable
        if "colormap" not in opts and colormap:
            opts["colormap"] = colormap
        if "index" not in opts and index:
            opts["index"] = index
        if "u_variable" not in opts and u_variable:
            opts["u_variable"] = u_variable
        if "v_variable" not in opts and v_variable:
            opts["v_variable"] = v_variable
        if "quiver_scale" not in opts and quiver_scale is not None:
            opts["quiver_scale"] = str(quiver_scale)
        if "quiver_step" not in opts and quiver_step is not None:
            opts["quiver_step"] = str(quiver_step)
        if "vmin" not in opts and vmin is not None:
            opts["vmin"] = str(vmin)
        if "vmax" not in opts and vmax is not None:
            opts["vmax"] = str(vmax)
        inherited.append(_copy_layer(spec, opts))

    region_polygon = _combined_mask_polygon(mask_geojson, inherited)
    extent_vals = parse_extent(extent)
    prepared = []
    for i, spec in enumerate(inherited):
        if spec.kind == "mask":
            continue
        if spec.kind == "heatmap":
            item = _prep_heatmap_layer(spec, bbox_nwse, region_polygon, extent, registry=registry)
        elif spec.kind == "scatter":
            item = _prep_scatter_layer(spec, bbox_nwse, region_polygon, registry=registry)
        elif spec.kind == "quiver":
            item = _prep_quiver_layer(spec, bbox_nwse, region_polygon, extent)
        elif spec.kind == "outline":
            item = _prep_outline_layer(spec)
        else:
            raise UsageError(f"unknown --layer kind {spec.kind!r}")
        label_override = label_slots[i]
        if label_override:
            item["cbar_label"] = label_override
        elif cbar_label:
            item["cbar_label"] = cbar_label
        item["zorder"] = item["zorder"] + i * 0.01
        prepared.append(item)

    if not prepared:
        raise UsageError("--layer needs at least one heatmap, scatter, quiver, or outline")

    has_heatmap = any(p["kind"] == "heatmap" for p in prepared)
    for p in prepared:
        if p["kind"] == "quiver":
            p["draw_mesh"] = not has_heatmap

    driver = next((p for p in prepared if p.get("panel_dim")), None)
    if driver is None:
        steps = [None]
        sdim = None
        title_da = None
        title_steps = [None]
    else:
        sdim = driver["panel_dim"]
        title_da = _layer_field(driver)
        driver_values = list(title_da[sdim].values)
        driver_kind = axis_kind(title_da[sdim].values)
        aligned = driver_values
        for p in prepared:
            if p is driver:
                continue
            field = _layer_field(p)
            if field is None:
                continue
            other_dim, labels = _align_panel_labels(sdim, aligned, driver_kind, field, p["spec"])
            if other_dim is None:
                continue
            aligned = labels
            p["panel_dim"] = other_dim
        if not aligned:
            raise UsageError("no overlapping time bins across --layer inputs")
        for p in prepared:
            field = _layer_field(p)
            dim = p.get("panel_dim")
            if field is None or dim is None or dim not in field.dims:
                continue
            _sel_layer(p, dim, aligned)
        title_da = _layer_field(driver)
        steps = list(title_da[sdim].values) if sdim in title_da.dims else aligned
        if sdim is not None and title_da.sizes.get(sdim, 1) == 1:
            for p in prepared:
                _squeeze_layer_dim(p, p.get("panel_dim"))
            steps = [None]
            sdim = None
        native = driver.get("native_steps")
        native_dim = driver.get("native_step_dim")
        title_steps = native if native is not None and native_dim == sdim else steps

    for p in prepared:
        dim = p.get("panel_dim")
        field = _layer_field(p)
        if sdim is None and dim and field is not None and dim in field.dims:
            raise UsageError(
                f"--layer {p['spec'].kind}:{p['spec'].path} still has {dim!r}; "
                "select a position with index= (the other layers have no panel axis)"
            )

    if extent_vals is None:
        if bbox_nwse is not None:
            r_n, r_w, r_s, r_e = bbox_nwse
            extent_vals = [float(r_w), float(r_e), float(r_s), float(r_n)]
        else:
            extent_vals = None
            for p in prepared:
                if p["kind"] in ("heatmap", "quiver"):
                    src = p["da"] if p["kind"] == "heatmap" else p["speed"]
                    extent_vals = _extent_from_field(src, p["lat_dim"], p["lon_dim"])
                    break
            if extent_vals is None:
                scatter = next((p for p in prepared if p["kind"] == "scatter"), None)
                if scatter is not None:
                    extent_vals = _extent_from_points(scatter["da"])
                else:
                    raise UsageError("could not determine map extent; pass --extent or --bbox")

    wrap_lon = True
    for p in prepared:
        if "wrap_lon" in p:
            wrap_lon = p["wrap_lon"]
            break

    share = _scale_groups(prepared, shared_scale, independent_scale)
    if share:
        _apply_shared_scale(prepared)

    num_steps = len(steps)
    subplot_titles = _resolve_subplot_titles(subplot_titles, num_steps)
    nrows, ncols = panel_shape(num_steps, rows=rows, columns=columns)
    sw, sh = figsize_from_extent(*extent_vals)
    spacing = _facet_spacing(wspace, hspace)
    default_size = (sw * ncols, sh * nrows)
    if spacing is not None and figsize is None:
        default_size = _figsize_with_spacing(default_size, nrows, ncols, spacing)
    fig, axes = facet_figure(
        nrows,
        ncols,
        figsize=resolve_figsize(figsize, default_size),
        sharex=True,
        sharey=True,
        subplot_kws={"projection": ccrs.PlateCarree()},
        wspace=None if spacing is None else spacing[0],
        hspace=None if spacing is None else spacing[1],
        despine=False,
    )
    axes = np.array(axes).reshape(nrows, ncols).flatten()
    drawn = {
        "rows": nrows,
        "columns": ncols,
        "n_panels": num_steps,
        "wspace": None if spacing is None else spacing[0],
        "hspace": None if spacing is None else spacing[1],
        "extent": list(extent_vals) if extent_vals is not None else None,
        "colormap": next(
            (getattr(p.get("cmap"), "name", None) for p in prepared if p.get("cmap") is not None),
            None,
        ),
    }
    overlays = load_geo_overlays(extent_vals)
    cities_map = parse_cities(cities)
    boxes = draw_boxes or []
    transform = ccrs.PlateCarree()

    quiver_meta = None
    for p in prepared:
        if p["kind"] == "quiver":
            step = _quiver_step(
                p["u_da"][p["lat_dim"]].values, p["u_da"][p["lon_dim"]].values, p.get("quiver_step")
            )
            native_spacing = _native_spacing_deg(
                p["u_da"][p["lat_dim"]].values, p["u_da"][p["lon_dim"]].values
            )
            arrow_spacing = None if native_spacing is None else native_spacing * step
            lon_span = abs(extent_vals[1] - extent_vals[0])
            scale = _auto_quiver_scale(
                p["u_da"].values,
                p["v_da"].values,
                lon_span,
                arrow_spacing,
                requested=p.get("quiver_scale"),
            )
            quiver_meta = (p, scale, step)
            break

    last_by_group = {}
    last_quiv = None
    for i, s in enumerate(steps):
        ax = axes[i]
        if wrap_lon:
            ax.set_extent(extent_vals, crs=transform)
        else:
            ax.set_xlim(extent_vals[0], extent_vals[1])
            ax.set_ylim(extent_vals[2], extent_vals[3])
        for p in prepared:
            slab = _select_panel(p, s)
            if slab["kind"] == "heatmap":
                last_by_group.setdefault(id(p) if not share else "shared", None)
                artist = _draw_heatmap_on_ax(ax, slab, transform)
                last_by_group["shared" if share else id(p)] = (artist, p)
            elif slab["kind"] == "scatter":
                artist = _draw_scatter_on_ax(ax, slab, transform)
                last_by_group["shared" if share else id(p)] = (artist, p)
            elif slab["kind"] == "quiver":
                _, scale, step = quiver_meta
                mesh, quiv = _draw_quiver_on_ax(ax, slab, transform, scale, step, mpl_spec=mpl_spec)
                last_quiv = quiv
                if mesh is not None:
                    last_by_group[id(p)] = (mesh, p)
            elif slab["kind"] == "outline":
                _draw_outline_on_ax(ax, slab, transform)
        draw_geo_overlays(ax, overlays, transform)
        ax.gridlines(draw_labels=False, alpha=0)
        _apply_geo_axis_labels(
            ax,
            xlabel,
            ylabel,
            xlabel_on=(i // ncols == nrows - 1),
            ylabel_on=(i % ncols == 0),
        )
        for city, (lat, lon) in cities_map.items():
            ax.plot(lon, lat, marker="o", color="k", markersize=6, transform=transform, zorder=8)
            ax.text(
                lon - 2.0,
                lat + 0.5,
                city,
                transform=transform,
                zorder=8,
            )
        if boxes:
            draw_box_outlines(ax, boxes, transform)
        auto = (
            panel_title(title_da, sdim, s, title_steps)
            if s is not None and title_da is not None
            else None
        )
        _set_panel_title(ax, i, auto, subplot_titles)

    for j in range(num_steps, len(axes)):
        axes[j].set_visible(False)

    if last_quiv is not None:
        last = axes[num_steps - 1]
        qlayer = next(p for p in prepared if p["kind"] == "quiver")
        units_disp = _speed_units_display(qlayer["u_da"])
        y_key = 0.18
        for u_ref in QUIVER_KEY_MS:
            last.quiverkey(
                last_quiv,
                1.18,
                y_key,
                u_ref,
                f"{u_ref:g} {units_disp}",
                labelpos="E",
                coordinates="axes",
            )
            y_key -= 0.10

    apply_suptitle(fig, title, mpl_spec)

    visible = [ax for ax in axes if ax.get_visible()]
    size_kw = colorbar_mpl_kwargs(mpl_spec or {})
    for mappable, p in last_by_group.values():
        kw = dict(_cbar_boundary_kwargs(p.get("norm"), p.get("cmap")))
        if p.get("flag_ticks") is not None:
            kw["ticks"] = p["flag_ticks"]
        kw.update(size_kw)
        field = p.get("da") if p.get("da") is not None else p.get("speed")
        extend = None
        if field is not None:
            extend = _cbar_extend_for_limits(field, p.get("vmin"), p.get("vmax"))
        if extend and "extend" not in kw:
            kw["extend"] = extend
        cbar = add_shared_colorbar(
            fig,
            mappable,
            visible,
            p.get("cbar_label") or _variable_label(p.get("da")),
            **kw,
        )
        if (
            cbar is not None
            and p.get("flag_labels") is not None
            and "labels" not in size_kw
            and "ticks" not in size_kw
        ):
            cbar.set_ticklabels(p["flag_labels"])
    settle_figure(fig)
    return fig, drawn


def _contour_levels(vmin, vmax, n=10, norm=None):
    """Shared isoline edges for every contour panel (and a constant-field pad)."""
    import numpy as np

    boundaries = getattr(norm, "boundaries", None) if norm is not None else None
    if boundaries is not None:
        return list(boundaries)
    if vmin is None or vmax is None or not np.isfinite(vmin) or not np.isfinite(vmax):
        return n
    if vmin == vmax:
        pad = abs(vmin) * 0.05 if vmin != 0 else 1.0
        return np.linspace(vmin - pad, vmax + pad, n + 1)
    return np.linspace(vmin, vmax, n + 1)


# A single-input map kind is a one-layer figure: `--kind heatmap` and
# `--layer heatmap:x.zarr` take the same path so they cannot drift apart.
KIND_TO_LAYER = {"heatmap": "heatmap", "contour": "heatmap", "quiver": "quiver"}
MAP_STYLES = frozenset(KIND_TO_LAYER) | {"layer"}


def _inherit_layer_options(options: dict, spec: dict, spec_input: dict | None = None) -> dict:
    """Fill omitted layer knobs from figure-level ``theme.colormap`` / vmin / vmax.

    A ``::colormap=`` / ``::vmin=`` suffix on ``--layer`` still wins. Same
    defaults a single-input ``--kind heatmap`` already copies onto its
    synthetic layer — without this, ``--layer heatmap:… --colormap RdBu_r``
    dropped the palette and fell through to ``rocket``.
    """
    spec_input = spec_input or {}
    out = dict(options)
    for key, value in (
        ("variable", spec_input.get("variable")),
        ("index", spec_input.get("index")),
        ("colormap", (spec.get("theme") or {}).get("colormap") or spec_input.get("colormap")),
        ("vmin", spec.get("vmin")),
        ("vmax", spec.get("vmax")),
    ):
        if value is not None and key not in out:
            out[key] = value
    return out


def layers_from_spec(spec: dict, datasets: dict) -> list:
    """Build the ``LayerSpec`` list a map spec describes.

    An explicit ``layers`` list is used as given; otherwise ``traces[0].kind``
    is turned into the single layer that draws it.
    """
    trace = trace_at(spec)
    style = trace.get("kind") or "heatmap"
    inputs = [i for i in (spec.get("inputs") or []) if isinstance(i, dict)]
    by_id = {str(i.get("id")): i for i in inputs}

    def dataset_for(input_id, path):
        if input_id and input_id in datasets:
            return datasets[input_id]
        if path:
            for ds in datasets.values():
                from weather_skills_core.decorator import INPUT_PATH_ATTR

                if str(getattr(ds, "attrs", {}).get(INPUT_PATH_ATTR) or "") == str(path):
                    return ds
        return next(iter(datasets.values()), None)

    raw_layers = spec.get("layers") or []
    if raw_layers:
        built = []
        for item in raw_layers:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or "").strip().lower()
            path = item.get("path")
            if not kind or not path:
                raise UsageError("spec layers[] entries need kind and path")
            spec_input = by_id.get(str(item.get("input") or ""), {})
            options = _inherit_layer_options(fold_layer_options(item), spec, spec_input)
            layer = LayerSpec(kind, path, options, f"{kind}:{path}")
            if kind in _ZARR_LAYER_KINDS:
                layer.ds = dataset_for(item.get("input"), path)
            built.append(layer)
        return built

    if style not in KIND_TO_LAYER:
        raise UsageError(f"{style!r} is not a map kind; expected one of {sorted(MAP_STYLES)}")
    input_id = str(trace.get("input") or (inputs[0].get("id") if inputs else "a"))
    spec_input = by_id.get(input_id, inputs[0] if inputs else {})
    options = _inherit_layer_options({}, spec, spec_input)
    for key, value in (
        ("u_variable", trace.get("u_variable")),
        ("v_variable", trace.get("v_variable")),
        ("quiver_scale", (trace.get("quiver") or {}).get("scale")),
        ("quiver_step", trace.get("quiver_step")),
    ):
        if value is not None:
            options[key] = value
    if style == "contour":
        options["draw"] = "contour"
        if trace.get("contour") is not None:
            options["contour"] = trace["contour"]
    if trace.get("mesh") is not None:
        options["mesh"] = trace["mesh"]
    layer = LayerSpec(
        KIND_TO_LAYER[style],
        spec_input.get("path") or "",
        options,
        f"{style}:{spec_input.get('path') or ''}",
    )
    layer.ds = dataset_for(input_id, spec_input.get("path"))
    return [layer]


def compile_map(spec: dict, datasets: dict, *, fontsize, template="weather_skills", registry=None):
    """Compile any map spec (single kind or layered) to a ``CompiledFigure``."""
    from weather_skills_core.plot.figure import CompiledFigure

    fig, drawn = compile_map_figure(
        spec, datasets, fontsize=fontsize, template=template, registry=registry
    )
    return CompiledFigure(fig, spec, tight=False, map_drawn=drawn)


def compile_map_figure(
    spec: dict, datasets: dict, *, fontsize, template="weather_skills", registry=None
):
    """Compile any map spec (single kind or layered) to a matplotlib Figure."""
    geo = spec.get("geo") or {}
    layout = spec.get("layout") or {}
    facet = layout.get("facet") or {}
    trace = trace_at(spec)
    shared = layout.get("shared_colorscale")
    bbox = geo.get("bbox")
    labels = []
    by_id = {str(item.get("id")): item.get("label") for item in (spec.get("inputs") or [])}
    raw_layers = spec.get("layers") or []
    if raw_layers:
        for item in raw_layers:
            labels.append(by_id.get(str(item.get("input") or "")))
    else:
        labels = [item.get("label") for item in (spec.get("inputs") or [])]
    theme = spec.get("theme") or {}
    first_input = (spec.get("inputs") or [{}])[0] if spec.get("inputs") else {}
    fig, drawn = _plot_layers(
        layers_from_spec(spec, datasets),
        tuple(bbox) if bbox is not None else None,
        geo.get("mask_geojson"),
        geo.get("extent"),
        geo.get("cities"),
        spec.get("title"),
        fontsize,
        geo.get("draw_boxes"),
        facet.get("rows"),
        facet.get("columns"),
        first_input.get("variable"),
        theme.get("colormap") or first_input.get("colormap"),
        first_input.get("index"),
        trace.get("u_variable"),
        trace.get("v_variable"),
        (trace.get("quiver") or {}).get("scale"),
        trace.get("quiver_step"),
        shared is True,
        shared is False,
        layer_labels=labels or None,
        xlabel=spec.get("xlabel"),
        ylabel=spec.get("ylabel"),
        figsize=layout.get("figsize"),
        vmin=spec.get("vmin"),
        vmax=spec.get("vmax"),
        subplot_titles=spec.get("subplot_titles"),
        cbar_label=spec.get("cbar_label"),
        mpl_spec=spec,
        template=template,
        registry=registry,
        wspace=facet.get("wspace"),
        hspace=facet.get("hspace"),
    )
    return fig, drawn


HITS_COLORS = ["#d73027", "#f0f0f0", "#1a9850"]
ERROR_DIVERGING_COLORS = [
    "#053061",
    "#2166ac",
    "#4393c3",
    "#92c5de",
    "#d1e5f0",
    "#ffffff",
    "#fddbc7",
    "#f4a582",
    "#d6604d",
    "#b2182b",
    "#67001f",
]
MAE_FROM_WHITE_COLORS = [
    "#ffffff",
    "#fddbc7",
    "#f4a582",
    "#d6604d",
    "#b2182b",
    "#67001f",
]


def scale_from_da(da, colormap=None, *, stretch=False, label=None, vmin=None, vmax=None):
    """``resolve_colorscale`` plus optional user limits and a colorbar label."""
    parsed = parse_colormap_spec(colormap) if colormap is not None else {}
    if parsed.get("bounds") and vmin is None and vmax is None:
        stretch = False
    scale = resolve_colorscale(
        da, colormap, stretch=stretch or vmin is not None or vmax is not None
    )
    if vmin is not None:
        scale["cmin"] = vmin
    if vmax is not None:
        scale["cmax"] = vmax
    if label:
        scale["label"] = label
    return scale


def hits_scale(*, label="event"):
    return {
        "name": "hits",
        "colors": HITS_COLORS,
        "bounds": [-1.5, -0.5, 0.5, 1.5],
        "tickvals": [-1, 0, 1],
        "ticktext": ["disagree", "below", "hit"],
        "cmin": -1.5,
        "cmax": 1.5,
        "label": label,
    }


def error_scale(da, metric, *, label=None):
    """Bias (diverging, white at 0) or MAE (white→warm) colormap."""
    vals = np.asarray(da.values, dtype=float)
    finite = vals[np.isfinite(vals)]
    if metric == "bias":
        if finite.size == 0:
            lo, hi = -1.0, 1.0
        else:
            lo = float(np.nanmin(finite))
            hi = float(np.nanmax(finite))
            m = max(abs(lo), abs(hi), 1e-6)
            lo, hi = -m, m
        return {
            "name": "verify_bias",
            "colors": ERROR_DIVERGING_COLORS,
            "cmin": lo,
            "cmax": hi,
            "label": label or "bias",
        }
    if finite.size == 0:
        lo, hi = 0.0, 1.0
    else:
        lo = 0.0
        hi = float(np.nanmax(finite)) or 1.0
    return {
        "name": "verify_mae",
        "colors": MAE_FROM_WHITE_COLORS,
        "cmin": lo,
        "cmax": hi,
        "label": label or "mae",
    }


def _colorbar_column_groups(axes, n_scales):
    """Split a grid's columns into groups so bottom colorbars sit side by side."""
    import numpy as np

    grid = np.atleast_2d(axes)
    ncols = grid.shape[1]
    if n_scales <= 1 or ncols < n_scales:
        visible = [ax for ax in grid.ravel() if ax.get_visible()]
        return [visible]
    sizes = [ncols // n_scales] * n_scales
    for i in range(ncols % n_scales):
        sizes[i] += 1
    groups = []
    start = 0
    for size in sizes:
        block = grid[:, start : start + size]
        groups.append([ax for ax in block.ravel() if ax.get_visible()])
        start += size
    return groups


def heatmap_cell(da, lat_dim, lon_dim, *, scale="field"):
    slab = da.transpose(lat_dim, lon_dim)
    return {
        "kind": "heatmap",
        "x": np.asarray(slab[lon_dim].values, dtype=float),
        "y": np.asarray(slab[lat_dim].values, dtype=float),
        "z": np.asarray(slab.values, dtype=float),
        "scale": scale,
    }


def scatter_cell(lons, lats, values, *, scale="field"):
    return {
        "kind": "scatter",
        "x": np.asarray(lons, dtype=float),
        "y": np.asarray(lats, dtype=float),
        "c": np.asarray(values, dtype=float),
        "scale": scale,
    }


def blank_cell(text="n/a"):
    return {"kind": "blank", "text": text}


def compile_grid(
    cells,
    *,
    extent,
    title=None,
    col_titles=None,
    row_titles=None,
    fontsize=DEFAULT_FONTSIZE,
    figsize=None,
    scales=None,
    overlays=True,
    xlabel="Longitude",
    cell_notes=None,
    template="weather_skills",
    spec=None,
):
    """Compile a 2-D grid of heatmap/scatter/blank cells.

    ``cells`` is a list of rows; each row is a list of dicts or None.
    A cell dict is ``{"kind": "heatmap", "x", "y", "z", "scale"}`` or
    ``{"kind": "scatter", "x", "y", "c", "scale"}`` or
    ``{"kind": "blank", "text": "n/a"}``.
    ``scales`` maps scale id → dict from ``resolve_colorscale``.
    """
    apply_style_then_rc(spec or {}, chart="map", fontsize=fontsize, template=template)
    trace = trace_at(spec)
    nrows = len(cells)
    ncols = max((len(row) for row in cells), default=1)
    geo_layers = load_geo_overlays(extent) if overlays else []
    sw, sh = (5.0, 4.0)
    if extent is not None:
        sw, sh = figsize_from_extent(*extent)
    facet = ((spec or {}).get("layout") or {}).get("facet") or {}
    spacing = _facet_spacing(facet.get("wspace"), facet.get("hspace"))
    if figsize is not None:
        fig_w, fig_h = float(figsize[0]), float(figsize[1])
    else:
        fig_w, fig_h = max(sw * ncols, 6.0), max(sh * nrows, 4.0)
        if spacing is not None:
            fig_w, fig_h = _figsize_with_spacing((fig_w, fig_h), nrows, ncols, spacing)
    fig, axes = facet_figure(
        nrows,
        ncols,
        figsize=(fig_w, fig_h),
        wspace=None if spacing is None else spacing[0],
        hspace=None if spacing is None else spacing[1],
        despine=False,
    )
    mappables = {}
    for r, row in enumerate(cells):
        for c in range(ncols):
            ax = axes[r, c]
            cell = row[c] if c < len(row) else None
            if cell is None:
                cell = blank_cell()
            kind = cell.get("kind") or "heatmap"
            scale_id = cell.get("scale") or "field"
            scale = scales.get(scale_id) or {}
            cmap, norm = mpl_cmap_norm(scale) if scale else (None, None)
            if extent is not None:
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])
                ax.set_aspect("equal", adjustable="box")
            if r == 0 and col_titles and c < len(col_titles):
                ax.set_title(wrap_axes_title(ax, col_titles[c]))
            if r == nrows - 1:
                ax.set_xlabel(xlabel)
            if c == 0 and row_titles and r < len(row_titles):
                ax.set_ylabel(row_titles[r])
            if kind == "blank":
                ax.text(
                    0.5,
                    0.5,
                    cell.get("text") or "n/a",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="#888888",
                )
            elif kind == "scatter":
                sk = scatter_kwargs(trace.get("scatter") or {})
                mappable = ax.scatter(
                    cell["x"],
                    cell["y"],
                    **{
                        "c": cell.get("c"),
                        "cmap": cmap,
                        "norm": norm,
                        "s": 64,
                        "edgecolors": "#333",
                        "linewidths": 0.4,
                        "zorder": 4,
                        **sk,
                    },
                )
                if scale_id not in mappables:
                    mappables[scale_id] = mappable
                draw_geo_overlays(ax, geo_layers)
            else:
                mk = mesh_kwargs(trace)
                mappable = ax.pcolormesh(
                    np.asarray(cell["x"], dtype=float),
                    np.asarray(cell["y"], dtype=float),
                    np.asarray(cell["z"], dtype=float),
                    **{"cmap": cmap, "norm": norm, "shading": "nearest", **mk},
                )
                if scale_id not in mappables:
                    mappables[scale_id] = mappable
                draw_geo_overlays(ax, geo_layers)
            note = None
            if cell_notes and r < len(cell_notes) and c < len(cell_notes[r]):
                note = cell_notes[r][c]
            if note:
                ax.text(
                    0.03,
                    0.97,
                    note,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=max(8, int(fontsize * 0.6)),
                    color="#333333",
                )

    visible = [ax for ax in axes.ravel() if ax.get_visible()]
    n_scales = len(mappables)
    extra_cbar = colorbar_mpl_kwargs(spec or {})
    shared_location = extra_cbar.get(
        "location", "right" if n_scales > 1 or len(visible) <= 1 else "bottom"
    )
    column_groups = (
        _colorbar_column_groups(axes, n_scales)
        if shared_location == "bottom" and n_scales > 1
        else None
    )
    for i, (scale_id, mappable) in enumerate(mappables.items()):
        scale = scales.get(scale_id) or {}
        cbar_kw = dict(extra_cbar)
        location = cbar_kw.pop("location", shared_location)
        ticks = cbar_kw.pop("ticks", None)
        labels = cbar_kw.pop("labels", None)
        if ticks is None:
            ticks = scale.get("tickvals") or scale.get("bounds")
        targets = column_groups[i] if column_groups and i < len(column_groups) else visible
        cbar = add_shared_colorbar(
            fig,
            mappable,
            targets,
            label=scale.get("label") or "",
            location=location,
            ticks=ticks,
            labels=labels,
            **cbar_kw,
        )
        if cbar is not None and labels is None and scale.get("ticktext"):
            cbar.set_ticklabels(list(scale["ticktext"]))
    apply_suptitle(fig, title, spec)
    finish_figure(fig, spec or {}, axes)
    settle_figure(fig)
    from weather_skills_core.plot.figure import CompiledFigure

    return CompiledFigure(fig, spec or {}, tight=False)
