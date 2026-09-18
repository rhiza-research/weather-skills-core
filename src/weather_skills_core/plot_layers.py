"""Layered map compilation: heatmap, scatter, quiver and outline on shared axes.

One renderer for every map in the stack. ``plot --style heatmap|contour|quiver``
is a single-layer figure and ``plot --layer`` is the multi-layer case, so the
same field draws identically whichever way it was asked for. Cartopy supplies
the axes; :mod:`weather_skills_core.plot_geo` supplies the overlays.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from weather_skills_core.cf import auto_variable, cf_dim
from weather_skills_core.display_labels import dataset_display_label, resolve_input_labels
from weather_skills_core.errors import UsageError
from weather_skills_core.figure import (
    add_shared_colorbar,
    resolve_axis_label,
    resolve_figsize,
)
from weather_skills_core.plot_compile import (
    axis_kind,
    figsize_from_extent,
    pad_cell_extent,
    panel_title,
    parse_cities,
    parse_extent,
    plain,
    step_dim,
    subset_spatial,
)
from weather_skills_core.plot_geo import (
    draw_box_outlines,
    draw_geo_overlays,
    load_geo_overlays,
)
from weather_skills_core.plot_mpl import colorbar_mpl_kwargs, mesh_kwargs, quiver_kwargs
from weather_skills_core.plot_spec import apply_index, panel_shape, parse_index, trace_at
from weather_skills_core.plot_style import (
    DISCRETE_PRECIP_NAMES,
    mpl_cmap_norm,
    resolve_colorscale,
)
from weather_skills_core.standard_utils import ensure_normalized_longitude, polygon_from_geojson
from weather_skills_core.units import (
    precip_for_display,
    to_standard_units,
    units_equal,
    variable_label_for_display,
    variable_units,
)

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
        "u-variable",
        "v-variable",
        "quiver-scale",
        "quiver-step",
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
                raise ValueError(f"--layer option {token!r} has an empty key")
            if key not in _LAYER_OPTION_KEYS:
                raise ValueError(
                    f"unknown --layer option {key!r}; "
                    f"expected one of {', '.join(sorted(_LAYER_OPTION_KEYS))}"
                )
            if key in options:
                raise ValueError(f"--layer option {key!r} is given more than once")
            current = key
            options[key] = val.strip()
        else:
            if current is None:
                raise ValueError(f"--layer option {token!r} appears before any key=value")
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
    if spec is None or "," not in spec:
        return spec
    from matplotlib.colors import LinearSegmentedColormap

    parts = [p.strip() for p in spec.split(",") if p.strip()]
    return LinearSegmentedColormap.from_list("custom", parts)


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
    if colormap and "," in colormap:
        parts = [p.strip() for p in colormap.split(",") if p.strip()]
        if len(parts) == values.size:
            colors = parts
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


def _heatmap_scale(da, colormap, *, stretch=False):
    """Return ``(cmap, norm)``. ``norm`` is set for the default precip scale.

    ``stretch=True`` (user ``--vmin`` / ``--vmax``) keeps the CHC colors
    but drops ``BoundaryNorm`` so the colorbar can use arbitrary limits.

    A comma list builds a colormap directly; anything else goes through
    ``resolve_colorscale`` so a named CHC palette (``chirps_total``, ``spi``, …)
    resolves the same way whether it came from ``--colormap`` or from a dumped
    sidecar, rather than being handed to matplotlib as an unknown name.
    """
    if colormap and "," in str(colormap):
        return _parse_colormap(colormap), None
    scale = resolve_colorscale(da, colormap, stretch=stretch)
    if stretch:
        colors = scale.get("colors")
        if colors:
            return _parse_colormap(",".join(colors)), None
        return scale.get("cmap") or scale.get("name"), None
    if scale.get("bounds"):
        return mpl_cmap_norm(scale)
    return scale.get("cmap") or scale.get("name") or "rocket", None


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
    if index < len(subplot_titles):
        ax.set_title(subplot_titles[index])
    elif auto:
        ax.set_title(auto)


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
        raise UsageError(f"--u-variable {u_variable!r} is not in the data (have {names})")
    if v_variable and v_variable not in ds:
        raise UsageError(f"--v-variable {v_variable!r} is not in the data (have {names})")
    if u_variable and v_variable:
        return u_variable, v_variable
    if u_variable:
        partner = _infer_uv_partner(u_variable, want_v=True)
        if partner and partner in ds:
            return u_variable, partner
        raise UsageError(
            f"--u-variable {u_variable!r} is set but no northward partner was found; "
            "pass --v-variable"
        )
    if v_variable:
        partner = _infer_uv_partner(v_variable, want_v=False)
        if partner and partner in ds:
            return partner, v_variable
        raise UsageError(
            f"--v-variable {v_variable!r} is set but no eastward partner was found; "
            "pass --u-variable"
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
        f"could not auto-detect them in {names}. Pass --u-variable and --v-variable."
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
            raise UsageError("--quiver-step must be >= 1")
        return int(requested)
    spacing = _native_spacing_deg(lat, lon)
    if spacing is None:
        return QUIVER_STEP
    return max(QUIVER_STEP, int(round(target_spacing / spacing)))


def _auto_quiver_scale(u, v, lon_span, spacing_deg, requested=None):
    """Matplotlib quiver ``scale`` (data units per axes-width).

    Larger scale → shorter arrows. ``requested`` (``--quiver-scale``) wins.
    Otherwise size a typical (95th-percentile) wind to about
    ``QUIVER_ARROW_LEN_SPACING`` times the subsampled grid spacing, as a
    fraction of the map width, so 10 m/s basin winds and small anomalies
    both stay readable.
    """
    if requested is not None:
        if requested <= 0:
            raise UsageError("--quiver-scale must be > 0")
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


def _prep_heatmap_layer(spec, bbox_nwse, region_polygon, extent):
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
        cmap, norm = _heatmap_scale(da, spec.options.get("colormap"), stretch=user_vlim)
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


def _prep_scatter_layer(spec, bbox_nwse, region_polygon):
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
    cmap, norm = _heatmap_scale(da, spec.options.get("colormap"), stretch=user_vlim)
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
    u_name, v_name = _resolve_uv(ds, spec.options.get("u-variable"), spec.options.get("v-variable"))
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
    qscale = spec.options.get("quiver-scale")
    qstep = spec.options.get("quiver-step")
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
            **mesh_kwargs(trace_at(mpl_spec)),
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
            **quiver_kwargs(trace_at(mpl_spec)),
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
):
    """Stack ``--layer`` entries on shared Cartopy panels.

    The renderer applies its own map chrome, so a caller cannot hand it the
    line-chart theme by mistake.
    """
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    from weather_skills_core.plot_mpl import apply_style_then_rc

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
        if "u-variable" not in opts and u_variable:
            opts["u-variable"] = u_variable
        if "v-variable" not in opts and v_variable:
            opts["v-variable"] = v_variable
        if "quiver-scale" not in opts and quiver_scale is not None:
            opts["quiver-scale"] = str(quiver_scale)
        if "quiver-step" not in opts and quiver_step is not None:
            opts["quiver-step"] = str(quiver_step)
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
            item = _prep_heatmap_layer(spec, bbox_nwse, region_polygon, extent)
        elif spec.kind == "scatter":
            item = _prep_scatter_layer(spec, bbox_nwse, region_polygon)
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
        elif spec.ds is not None and spec.kind in {"heatmap", "scatter", "quiver"}:
            item["cbar_label"] = dataset_display_label(spec.ds, item.get("cbar_label") or spec.path)
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
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=resolve_figsize(figsize, (sw * ncols, sh * nrows)),
        sharex=True,
        sharey=True,
        subplot_kw={"projection": ccrs.PlateCarree()},
        layout="compressed",
    )
    axes = np.array(axes).reshape(nrows, ncols).flatten()
    # What the renderer actually chose, for the resolved spec / sidecar.
    fig._ws_map = {
        "rows": nrows,
        "columns": ncols,
        "n_panels": num_steps,
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

    if title:
        fig.suptitle(title)

    visible = [ax for ax in axes if ax.get_visible()]
    size_kw = colorbar_mpl_kwargs(mpl_spec or {})
    for mappable, p in last_by_group.values():
        kw = dict(_cbar_boundary_kwargs(p.get("norm"), p.get("cmap")))
        kw.update(size_kw)
        if p.get("flag_ticks") is not None:
            kw["ticks"] = p["flag_ticks"]
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
        if cbar is not None and p.get("flag_labels") is not None:
            cbar.set_ticklabels(p["flag_labels"])
    return fig


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


# A single-input map style is a one-layer figure: `--style heatmap` and
# `--layer heatmap:x.zarr` take the same path so they cannot drift apart.
STYLE_TO_LAYER_KIND = {"heatmap": "heatmap", "contour": "heatmap", "quiver": "quiver"}
MAP_STYLES = frozenset(STYLE_TO_LAYER_KIND) | {"layer"}


def layers_from_spec(spec: dict, datasets: dict) -> list:
    """Build the ``LayerSpec`` list a map spec describes.

    An explicit ``layers`` list is used as given; otherwise ``traces[0].type``
    is turned into the single layer that draws it.
    """
    trace = trace_at(spec)
    style = trace.get("type") or "heatmap"
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
            layer = LayerSpec(kind, path, dict(item.get("options") or {}), f"{kind}:{path}")
            if kind in _ZARR_LAYER_KINDS:
                layer.ds = dataset_for(item.get("input"), path)
            built.append(layer)
        return built

    if style not in STYLE_TO_LAYER_KIND:
        raise UsageError(f"{style!r} is not a map style; expected one of {sorted(MAP_STYLES)}")
    input_id = str(trace.get("input") or (inputs[0].get("id") if inputs else "a"))
    spec_input = by_id.get(input_id, inputs[0] if inputs else {})
    options = {}
    for key, value in (
        ("variable", spec_input.get("variable")),
        ("index", spec_input.get("index")),
        ("colormap", (spec.get("style") or {}).get("colormap") or spec_input.get("colormap")),
        ("vmin", spec.get("vmin")),
        ("vmax", spec.get("vmax")),
        ("u-variable", trace.get("u_variable")),
        ("v-variable", trace.get("v_variable")),
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
        STYLE_TO_LAYER_KIND[style],
        spec_input.get("path") or "",
        options,
        f"{style}:{spec_input.get('path') or ''}",
    )
    layer.ds = dataset_for(input_id, spec_input.get("path"))
    return [layer]


def compile_map_figure(spec: dict, datasets: dict, *, fontsize, template="weather_skills"):
    """Compile any map spec (single style or layered) to a matplotlib Figure."""
    geo = spec.get("geo") or {}
    layout = spec.get("layout") or {}
    facet = layout.get("facet") or {}
    trace = trace_at(spec)
    shared = layout.get("shared_colorscale")
    bbox = geo.get("bbox")
    return _plot_layers(
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
        None,
        None,
        None,
        trace.get("u_variable"),
        trace.get("v_variable"),
        None,
        None,
        shared is True,
        shared is False,
        xlabel=spec.get("xlabel"),
        ylabel=spec.get("ylabel"),
        figsize=layout.get("figsize"),
        subplot_titles=spec.get("subplot_titles"),
        cbar_label=spec.get("cbar_label"),
        mpl_spec=spec,
        template=template,
    )
