"""Standard-dataset validation, CF dim detection, bbox helpers."""

from weather_skills_core.errors import DataError, UsageError
from weather_skills_core.types import Types

GRIDDED = Types.GRIDDED
FORECAST = Types.FORECAST
STATION = Types.STATION
ANY = Types.ANY

TYPES = (GRIDDED, FORECAST, STATION, ANY)

_LAT_NAMES = ("latitude", "lat", "y")
_LON_NAMES = ("longitude", "lon", "x")


def parse_bbox(bbox: str) -> tuple:
    """Parse N/W/S/E into four floats."""
    try:
        north, west, south, east = (float(x) for x in bbox.split("/"))
    except ValueError:
        raise UsageError("--bbox must be four decimal degrees N/W/S/E.") from None
    return north, west, south, east


def detect_spatial_dims(ds) -> tuple:
    """Lat/lon dim names via CF metadata only."""
    try:
        import cf_xarray  # noqa: F401 -- registers the .cf accessor

        return ds.cf["latitude"].name, ds.cf["longitude"].name
    except KeyError:
        raise UsageError(
            f"could not identify lat/lon coords via CF metadata in {list(ds.coords)}. "
            "Stamp CF standard_name/axis attrs on the coordinates."
        ) from None


def detect_time_dim(ds) -> str:
    """Time dim name via CF metadata only."""
    try:
        import cf_xarray  # noqa: F401 -- registers the .cf accessor

        name = ds.cf["time"].name
        if name in ds.dims:
            return name
    except KeyError:
        pass
    raise UsageError(
        f"could not identify a time dim via CF metadata in {list(ds.dims)}. "
        "Stamp CF standard_name/axis attrs on the time coordinate."
    )


def detect_type(ds) -> str:
    """Classify as station, forecast, or gridded."""
    if "station_id" in ds.dims:
        return STATION
    if "step" in ds.dims and "time" in ds.coords and ds["time"].ndim == 0:
        return FORECAST
    return GRIDDED


def validate_type(ds, expected, name: str = "dataset") -> str:
    """Validate ds matches a Types constant, union, list/tuple of types, or another dataset."""
    if hasattr(expected, "dims") and hasattr(expected, "coords"):
        allowed = [detect_type(expected)]
    elif isinstance(expected, str):
        allowed = [expected]
    else:
        allowed = list(expected)
    unknown = [t for t in allowed if t not in TYPES]
    if unknown:
        raise ValueError(f"unknown type(s) {unknown}; valid types: {list(TYPES)}")
    actual = detect_type(ds)
    if ANY not in allowed and actual not in allowed:
        raise UsageError(
            f"input {name} is a {actual} standard dataset, but this skill expects "
            f"{' or '.join(allowed)}: {_shape_detail(ds, allowed)}"
        )
    if actual == STATION and (ANY not in allowed or STATION in allowed):
        for coord in ("latitude", "longitude"):
            if coord not in ds.coords:
                raise UsageError(
                    f"input {name} is a station dataset but has no {coord!r} "
                    f"coordinate (coords: {list(ds.coords)})."
                )
            if tuple(ds[coord].dims) != ("station_id",):
                raise UsageError(
                    f"input {name} is a station dataset but its {coord!r} "
                    f"coordinate has dims {list(ds[coord].dims)}, expected "
                    "('station_id',)."
                )
    elif actual in (GRIDDED, FORECAST) and ANY not in allowed:
        detect_spatial_dims(ds)
    return actual


def _shape_detail(ds, allowed) -> str:
    """Short mismatch description for error messages."""
    details = []
    if FORECAST in allowed and "step" not in ds.dims:
        details.append("no 'step' dim")
    if FORECAST in allowed and "step" in ds.dims:
        if "time" not in ds.coords:
            details.append("no scalar 'time' coord")
        elif ds["time"].ndim != 0:
            details.append("'time' is a dim, not a scalar init coord")
    if STATION in allowed and "station_id" not in ds.dims:
        details.append("no 'station_id' dim")
    if GRIDDED in allowed and "station_id" in ds.dims:
        details.append("has a 'station_id' dim")
    if GRIDDED in allowed and "step" in ds.dims:
        details.append("has a 'step' dim")
    detail = "; ".join(details) if details else "shape does not match"
    return f"{detail} (dims: {list(ds.dims)})"


def stamp_cf_attrs(ds, *, long_names: dict | None = None, overwrite: bool = False):
    """Stamp CF attrs on lat/lon/time. overwrite replaces; else setdefault (aliases ok)."""
    long_names = long_names or {}
    if overwrite:
        stamps = {
            "latitude": {"standard_name": "latitude", "units": "degrees_north", "axis": "Y"},
            "longitude": {"standard_name": "longitude", "units": "degrees_east", "axis": "X"},
            "time": {"standard_name": "time", "axis": "T"},
        }
        for name, attrs in stamps.items():
            if name in ds.coords:
                ds[name].attrs.update(attrs)
                if name in long_names:
                    ds[name].attrs.setdefault("long_name", long_names[name])
        return ds
    for name in _LAT_NAMES:
        if name in ds.coords:
            ds[name].attrs.setdefault("standard_name", "latitude")
            ds[name].attrs.setdefault("units", "degrees_north")
            ds[name].attrs.setdefault("axis", "Y")
            break
    for name in _LON_NAMES:
        if name in ds.coords:
            ds[name].attrs.setdefault("standard_name", "longitude")
            ds[name].attrs.setdefault("units", "degrees_east")
            ds[name].attrs.setdefault("axis", "X")
            break
    if "time" in ds.coords:
        ds["time"].attrs.setdefault("standard_name", "time")
        ds["time"].attrs.setdefault("axis", "T")
    return ds


def stamp_cf_coords(ds, *, long_names: dict | None = None):
    """Overwrite CF attrs on latitude/longitude/time (alias for stamp_cf_attrs)."""
    return stamp_cf_attrs(ds, long_names=long_names, overwrite=True)


def udunits_error(units, *, catch: tuple = (ValueError,)):
    """Return cf_units parse error, or None if units parse."""
    import cf_units

    try:
        cf_units.Unit(units)
    except catch as exc:
        return exc
    return None


def cf_axes_missing(ds, axes: tuple = ("X", "Y", "T")) -> list:
    """CF axis letters among axes that do not resolve."""
    import cf_xarray  # noqa: F401 -- registers the .cf accessor

    missing = []
    for axis in axes:
        try:
            resolved = ds.cf.axes.get(axis)
        except Exception:  # noqa: BLE001 -- an unresolvable axis is the finding, not a failure
            resolved = None
        if not resolved:
            missing.append(axis)
    return missing


def cf_dim(obj, cf_name: str):
    """Resolved CF coord name, or None."""
    import cf_xarray  # noqa: F401 -- registers the .cf accessor

    try:
        return obj.cf[cf_name].name
    except KeyError:
        return None


def auto_variable(ds):
    """First real data var (skip grid-mapping containers)."""
    mapping_targets = {
        ds[d].attrs.get("grid_mapping") for d in ds.data_vars if ds[d].attrs.get("grid_mapping")
    }
    candidates = [
        v
        for v in ds.data_vars
        if "grid_mapping_name" not in ds[v].attrs and v not in mapping_targets
    ]
    if not candidates:
        return None
    multidim = [v for v in candidates if len(ds[v].dims) >= 2]
    return (multidim or candidates)[0]


def lat_slice(lat_vals, north, south) -> slice:
    """sel slice for ascending or descending latitude."""
    if lat_vals.size and lat_vals[0] > lat_vals[-1]:
        return slice(north, south)
    return slice(south, north)


def polygon_from_geojson(path, *, flag: str = "--mask-geojson"):
    """Unioned shapely polygon from a GeoJSON path."""
    import json
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        raise UsageError(f"{flag} file not found: {path}")
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise UsageError(f"could not read {flag} {path}: {exc}") from None

    if not isinstance(data, dict):
        raise UsageError(f"{flag} {path} has no usable geometry.")
    if data.get("type") == "FeatureCollection":
        features = data.get("features", [])
        if not isinstance(features, list):
            raise UsageError(f"{flag} {path}: 'features' is not a list.")
        geoms = []
        for feature in features:
            if not isinstance(feature, dict):
                raise UsageError(f"{flag} {path}: a feature is not a JSON object.")
            if feature.get("geometry"):
                geoms.append(feature["geometry"])
    elif data.get("type") == "Feature":
        geoms = [data["geometry"]] if data.get("geometry") else []
    else:
        geoms = [data]

    if not geoms:
        raise UsageError(f"{flag} {path} has no usable geometry.")

    from shapely.errors import GeometryTypeError
    from shapely.geometry import shape
    from shapely.ops import unary_union

    try:
        return unary_union([shape(g) for g in geoms])
    except (GeometryTypeError, KeyError, AttributeError, ValueError, TypeError) as exc:
        raise UsageError(f"{flag} {path} has no usable geometry ({exc}).") from None


def normalize_longitude(ds, lon_dim: str = "longitude"):
    """Map lon to [-180, 180) and sort ascending."""
    import numpy as np

    attrs = dict(ds[lon_dim].attrs)
    lon = ((ds[lon_dim] + 180) % 360) - 180
    lon.attrs = attrs
    ds = ds.assign_coords({lon_dim: lon})
    _, first = np.unique(ds[lon_dim].values, return_index=True)
    if len(first) < ds.sizes[lon_dim]:
        ds = ds.isel({lon_dim: np.sort(first)})
    return ds.sortby(lon_dim)


def stamp_cf_dsg(ds, var_attrs: dict, *, station_id_long_name: str, name_long_name: str):
    """Stamp CF timeSeries DSG attrs on a station dataset."""
    ds["latitude"].attrs.update(
        standard_name="latitude", long_name="station latitude", units="degrees_north", axis="Y"
    )
    ds["longitude"].attrs.update(
        standard_name="longitude", long_name="station longitude", units="degrees_east", axis="X"
    )
    ds["time"].attrs.update(standard_name="time", long_name="time", axis="T")
    ds["station_id"].attrs.update(cf_role="timeseries_id", long_name=station_id_long_name)
    if "name" in ds.coords or "name" in ds.variables:
        ds["name"].attrs.update(long_name=name_long_name)

    for var in ds.data_vars:
        ds[var].attrs.update({"coordinates": "latitude longitude time", **var_attrs[var]})
    return ds


def verify_cf_dsg(ds) -> None:
    """Raise DataError if CF DSG geometry does not resolve."""
    import cf_xarray  # noqa: F401 -- registers the .cf accessor

    problems = []
    cf_roles = ds.cf.cf_roles
    if "station_id" not in cf_roles.get("timeseries_id", []):
        problems.append(f"cf_role timeseries_id did not resolve to station_id (got {cf_roles})")
    for name in ("latitude", "longitude", "time"):
        try:
            ds.cf[name]
        except KeyError:
            problems.append(f"cf-xarray could not resolve the {name} coordinate")
    if problems:
        raise DataError(
            "CF-1.13 DSG verification failed before write:\n  - " + "\n  - ".join(problems)
        )


def latitude_weights(lats):
    """Cosine latitude weights (normalized to mean 1)."""
    import numpy as np
    import xarray as xr

    if not isinstance(lats, xr.DataArray):
        lats = xr.DataArray(lats)
    weights = np.cos(np.deg2rad(lats))
    return weights / weights.mean()


def clip_by_geometry(ds, geometry, *, lat_dim: str | None = None, lon_dim: str | None = None, drop: bool = True):
    """Clip gridded or station dataset to a shapely geometry (NaN outside; optional drop)."""
    import numpy as np
    import shapely
    import xarray as xr

    if geometry is None:
        return ds
    if "station_id" in ds.dims:
        for name in ("latitude", "longitude"):
            if name not in ds.coords:
                raise UsageError(
                    f"station clip requires {name!r} coord (coords: {list(ds.coords)})"
                )
        mask = shapely.contains_xy(
            geometry,
            np.asarray(ds["longitude"].values),
            np.asarray(ds["latitude"].values),
        )
        if not mask.any():
            raise DataError("geometry selects no stations")
        keep = xr.DataArray(mask, dims="station_id")
        return ds.isel(station_id=np.nonzero(mask)[0]) if drop else ds.where(keep, drop=False)

    if lat_dim is None or lon_dim is None:
        lat_dim, lon_dim = detect_spatial_dims(ds)
    lon_vals = np.asarray(ds[lon_dim].values)
    if lon_vals.size and float(np.nanmax(lon_vals)) > 180.0:
        ds = ds.assign_coords({lon_dim: ((ds[lon_dim] + 180) % 360 - 180)}).sortby(lon_dim)
    lon2d, lat2d = np.meshgrid(ds[lon_dim].values, ds[lat_dim].values)
    mask = shapely.contains_xy(geometry, lon2d, lat2d)
    mask_da = xr.DataArray(
        mask, dims=(lat_dim, lon_dim), coords={lat_dim: ds[lat_dim], lon_dim: ds[lon_dim]}
    )
    if not bool(mask_da.any()):
        raise DataError("geometry selects no grid cells")
    out = ds.where(mask_da)
    if drop:
        out = out.dropna(lat_dim, how="all").dropna(lon_dim, how="all")
    return out


def roll_and_agg(ds, window: int, dim: str, method: str = "mean", *, align: str = "left",
                 stride=None, min_periods: int | None = None):
    """N-step rolling aggregation (sheerwater-style), then optional stride subsample.

    ``window`` is in axis steps (days for daily ``time``, or steps for ``step``).
    ``align`` is left/right/center label placement. ``stride`` is an int step,
    or a ``stride_dates`` string when ``dim`` is datetime64.
    """
    import numpy as np
    import warnings

    from weather_skills_core.dates import stride_dates

    if window < 1:
        raise UsageError(f"rolling window must be >= 1; got {window}")
    if window == 1 and stride is None:
        return ds
    if min_periods is None:
        min_periods = window
    if method == "sum" and min_periods < window:
        warnings.warn(
            f"min_periods={min_periods} < window={window} with method=sum; "
            "partial windows are incomplete totals.",
            stacklevel=2,
        )
    dtype = ds[dim].dtype
    if not (np.issubdtype(dtype, np.datetime64) or np.issubdtype(dtype, np.timedelta64)):
        raise UsageError(
            f"rolling aggregation requires datetime64 or timedelta64 on {dim!r}; "
            f"got dtype {dtype}"
        )
    rolled = ds.rolling({dim: window}, min_periods=min_periods, center=False)
    fn = {"mean": rolled.mean, "sum": rolled.sum, "max": rolled.max, "min": rolled.min}.get(method)
    if fn is None:
        raise UsageError(f"unsupported rolling method {method!r}")
    out = fn(skipna=True)
    out = out.isel({dim: slice(window - 1, None)})
    if align == "center":
        shift = np.timedelta64((window - 1) // 2, "D")
    elif align == "right":
        shift = np.timedelta64(0, "D")
    elif align == "left":
        shift = np.timedelta64(window - 1, "D")
    else:
        raise UsageError(f"align must be left/right/center; got {align!r}")
    out = out.assign_coords({dim: out[dim] - shift})
    if stride is None:
        return out
    if isinstance(stride, int):
        if stride < 1:
            raise UsageError(f"stride must be >= 1; got {stride}")
        return out.isel({dim: slice(None, None, stride)})
    if not np.issubdtype(out[dim].dtype, np.datetime64):
        raise UsageError("string --stride requires a datetime64 time axis")
    times = stride_dates(out[dim].values[0], out[dim].values[-1], stride=str(stride))
    return out.sel({dim: times})


def bbox_subset(ds, bbox, *, lat_dim: str | None = None, lon_dim: str | None = None):
    """Subset a gridded dataset to an N/W/S/E bbox."""
    import numpy as np

    if isinstance(bbox, str):
        north, west, south, east = parse_bbox(bbox)
    else:
        north, west, south, east = bbox
    if lat_dim is None or lon_dim is None:
        lat_dim, lon_dim = detect_spatial_dims(ds)

    lon_vals = np.asarray(ds[lon_dim].values)
    if lon_vals.size and float(np.nanmax(lon_vals)) > 180.0:
        ds = ds.assign_coords({lon_dim: ((ds[lon_dim] + 180) % 360 - 180)}).sortby(lon_dim)
        lon_vals = np.asarray(ds[lon_dim].values)
    if lon_vals.size == 0:
        raise UsageError("lon axis has length 0; cannot subset.")
    if lon_vals.size == 1:
        lon_ascending = True
    else:
        lon_diffs = np.diff(lon_vals)
        if (lon_diffs > 0).all():
            lon_ascending = True
        elif (lon_diffs < 0).all():
            lon_ascending = False
        else:
            raise UsageError(
                "lon axis is non-monotonic; cannot infer slice orientation. "
                "Re-sort the input or pre-process before subsetting."
            )

    lat_vals = np.asarray(ds[lat_dim].values)
    if lat_vals.size == 0:
        raise UsageError("lat axis has length 0; cannot subset.")
    if lat_vals.size == 1:
        lat_slice = None
    else:
        diffs = np.diff(lat_vals)
        if (diffs > 0).all():
            lat_slice = slice(south, north)
        elif (diffs < 0).all():
            lat_slice = slice(north, south)
        else:
            raise UsageError(
                "lat axis is non-monotonic; cannot infer slice orientation. "
                "Re-sort the input or pre-process before subsetting."
            )
    if lat_slice is not None:
        ds = ds.sel({lat_dim: lat_slice})

    if west <= east:
        lon_slice = slice(west, east) if lon_ascending else slice(east, west)
        ds = ds.sel({lon_dim: lon_slice})
    else:
        import xarray as xr

        if lon_ascending:
            wings = [ds.sel({lon_dim: slice(None, east)}), ds.sel({lon_dim: slice(west, None)})]
        else:
            wings = [ds.sel({lon_dim: slice(None, west)}), ds.sel({lon_dim: slice(east, None)})]
        ds = xr.concat(
            wings,
            dim=lon_dim,
            data_vars="minimal",
            coords="minimal",
            compat="override",
            join="exact",
        )

    if ds.sizes.get(lat_dim, 0) == 0 or ds.sizes.get(lon_dim, 0) == 0:
        bbox_str = f"{north}/{west}/{south}/{east}"
        if west > east:
            raise DataError(
                f"--bbox {bbox_str} crosses the antimeridian (west {west} > east {east}) "
                "but selects no grid cells; check the N/S extent and that west/east "
                "bracket the intended dateline-crossing span."
            )
        raise DataError(
            f"--bbox {bbox_str} selects no grid cells; check the extent and N/W/S/E order."
        )
    return ds
