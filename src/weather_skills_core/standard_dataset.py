"""Ontology for the weather-skills standard dataset.

Dims and types skills declare via ``Dataset(...)`` on arguments, plus checks
that a Zarr matches those requirements.
"""

from __future__ import annotations

from dataclasses import dataclass

from weather_skills_core.errors import UsageError

# --- dimension vocabulary -------------------------------------------------

LAT = "lat"
LON = "lon"
TIME = "time"
INIT_TIME = "init_time"
PREDICTION_TIMEDELTA = "prediction_timedelta"
MEMBER = "member"
VERTICAL = "vertical"
DAY_OF_YEAR = "day_of_year"
POINT_ID = "point_id"
X = "x"
Y = "y"

# Ontology dims skills declare on Dataset(...)
DIMS = frozenset(
    {
        LAT,
        LON,
        TIME,
        INIT_TIME,
        PREDICTION_TIMEDELTA,
        MEMBER,
        VERTICAL,
        DAY_OF_YEAR,
        POINT_ID,
        X,
        Y,
    }
)

# Types → required ontology dims (AND)
TYPE_DIMS: dict[str, frozenset[str]] = {
    "spatial": frozenset({LAT, LON}),
    "observations": frozenset({LAT, LON, TIME}),
    "forecast": frozenset({LAT, LON, INIT_TIME, PREDICTION_TIMEDELTA}),
    "vertical_forecast": frozenset({LAT, LON, INIT_TIME, PREDICTION_TIMEDELTA, VERTICAL}),
    "ensemble_forecast": frozenset({LAT, LON, INIT_TIME, PREDICTION_TIMEDELTA, MEMBER}),
    "point_obs": frozenset({POINT_ID, TIME}),
}

# Type name synonyms → primary key in TYPE_DIMS
TYPE_ALIASES: dict[str, str] = {
    "space": "spatial",
    "obs": "observations",
    "analysis": "observations",
    "retrieval": "observations",
    "field": "observations",
    "data": "observations",
    "station": "point_obs",
}

# Dataset / CF names → preferred ontology name (lat/lon preferred; x/y also ontology dims).
ALIASES: dict[str, str] = {
    "lat": "lat",
    "latitude": "lat",
    "lon": "lon",
    "longitude": "lon",
    "time": "time",
    "init_time": "init_time",
    "prediction_timedelta": "prediction_timedelta",
    "step": "prediction_timedelta",
    "lead_time": "prediction_timedelta",
    "member": "member",
    "number": "member",
    "realization": "member",
    "vertical": "vertical",
    "level": "vertical",
    "pressure": "vertical",
    "height": "vertical",
    "altitude": "vertical",
    "lev": "vertical",
    "isobaricInhPa": "vertical",
    "point_id": "point_id",
    "station_id": "point_id",
    "day_of_year": "day_of_year",
    "doy": "day_of_year",
}

ANY = "any"


def names_for(preferred: str) -> tuple[str, ...]:
    """Dataset names that map to ``preferred`` in ``ALIASES`` (preferred first)."""
    others = sorted(k for k, v in ALIASES.items() if v == preferred and k != preferred)
    return (preferred, *others)


@dataclass(frozen=True)
class IoSpec:
    """Required ontology dims for one Zarr path.

    Built by :class:`~weather_skills_core.Dataset` (or :func:`normalize_io_spec`).
    Each alternative is a set of dims that must all be present; the dataset
    may match any one. ``None`` means no dim check (``"any"``).

    Opaque files use ``pathlib.Path``, not ``Dataset``.
    """

    alternatives: tuple[frozenset[str], ...] | None  # None = any (no dim check)


def required_dims_for(name: str) -> frozenset[str]:
    """Ontology dims required by one type or dim name (``TYPE_ALIASES`` → ``TYPE_DIMS``, else ``DIMS``/``ALIASES``)."""
    primary = TYPE_ALIASES.get(name, name)
    if primary in TYPE_DIMS:
        return TYPE_DIMS[primary]
    if name in DIMS:
        return frozenset({name})
    dim = ALIASES.get(name, name)
    if dim in DIMS:
        return frozenset({dim})
    raise ValueError(
        f"unknown type or dimension {name!r}; expected a dimension {sorted(DIMS)}, "
        f"a type {sorted(TYPE_DIMS)}, or a type alias {sorted(TYPE_ALIASES)}"
    )


def and_group(spec) -> frozenset[str]:
    """Flatten a str or AND-tuple of ontology names into one required dim-set."""
    if isinstance(spec, str):
        return required_dims_for(spec)
    if isinstance(spec, tuple):
        out: set[str] = set()
        for part in spec:
            if not isinstance(part, str):
                raise TypeError(f"invalid AND-group entry {part!r}")
            out |= required_dims_for(part)
        return frozenset(out)
    raise ValueError(f"expected str or tuple for AND-group, got {type(spec).__name__}")


def parse_alternatives(spec) -> tuple[frozenset[str], ...]:
    """Parse a Zarr IO entry body into OR-alternatives: str / tuple=AND / list=OR → :attr:`IoSpec.alternatives`."""
    if isinstance(spec, str):
        return (required_dims_for(spec),)
    if isinstance(spec, tuple):
        return (and_group(spec),)
    if isinstance(spec, list):
        if not spec:
            raise ValueError("OR list must not be empty")
        alts: list[frozenset[str]] = []
        for item in spec:
            if isinstance(item, list):
                raise TypeError(f"nested OR lists are not allowed; got {item!r}")
            alts.extend(parse_alternatives(item))
        return tuple(alts)
    raise ValueError(f"invalid IO entry {spec!r}; expected str, list (OR), or tuple (AND)")


def normalize_io_spec(raw) -> IoSpec:
    """Parse a Dataset declaration body into a :class:`IoSpec`."""
    if isinstance(raw, str) and raw == ANY:
        return IoSpec(alternatives=None)
    return IoSpec(alternatives=parse_alternatives(raw))


def has_dim(ds, dim: str) -> bool:
    """True if ``ds`` has the ontology dim (ALIASES / CF / special lat-lon rules)."""
    if dim == LAT:
        try:
            lat_dim, _ = detect_spatial_dims(ds)
            return lat_dim in ds.dims
        except UsageError:
            return False
    if dim == LON:
        try:
            _, lon_dim = detect_spatial_dims(ds)
            return lon_dim in ds.dims
        except UsageError:
            return False
    if dim == TIME:
        try:
            name = detect_time_dim(ds)
            return name in ds.dims
        except UsageError:
            return False
    if dim == INIT_TIME:
        if "init_time" in ds.dims or "init_time" in ds.coords:
            return True
        # Classic forecast: scalar time coord + lead-time dim
        if "time" in ds.coords and "time" not in ds.dims and ds["time"].ndim == 0:
            return any(ALIASES.get(n) == PREDICTION_TIMEDELTA for n in ds.dims)
        return False
    if dim == PREDICTION_TIMEDELTA:
        return any(ALIASES.get(n) == PREDICTION_TIMEDELTA for n in ds.dims)
    if dim == MEMBER:
        return any(ALIASES.get(n) == MEMBER for n in ds.dims)
    if dim == VERTICAL:
        return any(ALIASES.get(n) == VERTICAL for n in ds.dims)
    if dim == DAY_OF_YEAR:
        return any(ALIASES.get(n) == DAY_OF_YEAR for n in ds.dims)
    if dim == POINT_ID:
        point = next((n for n in names_for(POINT_ID) if n in ds.dims), None)
        if point is None:
            return False
        lat = next(
            (n for n in names_for("lat") if n in ds.coords and tuple(ds[n].dims) == (point,)), None
        )
        lon = next(
            (n for n in names_for("lon") if n in ds.coords and tuple(ds[n].dims) == (point,)), None
        )
        return lat is not None and lon is not None
    if dim == X:
        return "x" in ds.dims
    if dim == Y:
        return "y" in ds.dims
    return False


def missing_dims(
    ds,
    required: frozenset[str],
    *,
    dims: str | None = None,
    time_dim: str | None = None,
) -> list[str]:
    missing = []
    for d in sorted(required):
        if d in (LAT, LON) and dims is not None:
            try:
                lat_dim, lon_dim = detect_spatial_dims(ds, dims)
                if d == LAT and lat_dim in ds.dims:
                    continue
                if d == LON and lon_dim in ds.dims:
                    continue
                missing.append(d)
            except UsageError:
                missing.append(d)
            continue
        if d == TIME and time_dim is not None:
            try:
                detect_time_dim(ds, time_dim)
                continue
            except UsageError:
                missing.append(d)
                continue
        if not has_dim(ds, d):
            missing.append(d)
    return missing


def validate_dims(
    ds,
    alternatives: tuple[frozenset[str], ...] | None,
    name: str,
    *,
    dims: str | None = None,
    time_dim: str | None = None,
) -> frozenset[str]:
    """Return the first matching dim-set, or raise; ``alternatives is None`` = unconstrained."""
    if alternatives is None:
        return frozenset()
    for alt in alternatives:
        if not missing_dims(ds, alt, dims=dims, time_dim=time_dim):
            return alt
    parts = []
    for alt in alternatives:
        miss = missing_dims(ds, alt, dims=dims, time_dim=time_dim)
        parts.append(f"{{{', '.join(sorted(alt))}}} missing {miss}")
    raise UsageError(
        f"{name} does not satisfy required dimensions ({' OR '.join(parts)}); dims={list(ds.dims)}"
    )


def detect_type(ds) -> str:
    """Classify ``ds`` as a primary key in ``TYPE_DIMS`` (or ``ensemble_forecast``)."""
    if has_dim(ds, POINT_ID):
        return "point_obs"
    if has_dim(ds, PREDICTION_TIMEDELTA) and has_dim(ds, INIT_TIME):
        if has_dim(ds, MEMBER):
            return "ensemble_forecast"
        if has_dim(ds, VERTICAL):
            return "vertical_forecast"
        return "forecast"
    return "observations"


def validate_input(
    ds, allowed, name: str, *, dims: str | None = None, time_dim: str | None = None
) -> str:
    """Check ``ds`` against an IoSpec or Dataset declaration; return ``detect_type``."""
    if isinstance(allowed, IoSpec):
        spec = allowed
    elif isinstance(allowed, (str, list, tuple)):
        spec = normalize_io_spec(allowed)
    else:
        raise TypeError(f"invalid validate_input allowed={allowed!r}")
    validate_dims(ds, spec.alternatives, name, dims=dims, time_dim=time_dim)
    return detect_type(ds)


def detect_spatial_dims(ds, override: str | None = None) -> tuple:
    """Find lat/lon dim names: optional ``LAT,LON`` override, then CF, then ALIASES."""
    if override:
        parts = [s.strip() for s in override.split(",")]
        if len(parts) != 2:
            raise UsageError("--dims must be two comma-separated names: LAT,LON.")
        lat_dim, lon_dim = parts
        if lat_dim not in ds.dims or lon_dim not in ds.dims:
            raise UsageError(f"--dims names not in dataset dims {list(ds.dims)}")
        return lat_dim, lon_dim
    try:
        import cf_xarray  # noqa: F401 -- registers the .cf accessor

        return ds.cf["latitude"].name, ds.cf["longitude"].name
    except KeyError:
        pass
    lat_dim = next((n for n in names_for("lat") if n in ds.dims), None)
    lon_dim = next((n for n in names_for("lon") if n in ds.dims), None)
    if lat_dim is None or lon_dim is None:
        raise UsageError(
            f"could not identify lat/lon coords via CF metadata or name "
            f"heuristics in {list(ds.coords)}. Pass --dims to override."
        )
    return lat_dim, lon_dim


def detect_time_dim(ds, override: str | None = None) -> str:
    """Find the time dim: optional name override, then CF, then literal ``time``."""
    if override:
        if override not in ds.dims:
            raise UsageError(f"--time-dim {override!r} not in dataset dims {list(ds.dims)}")
        return override
    try:
        import cf_xarray  # noqa: F401 -- registers the .cf accessor

        name = ds.cf["time"].name
        if name in ds.dims:
            return name
    except KeyError:
        pass
    if "time" in ds.dims:
        return "time"
    raise UsageError(
        f"could not identify a time dim via CF metadata or name heuristics in "
        f"{list(ds.dims)}. Pass --time-dim to override."
    )
