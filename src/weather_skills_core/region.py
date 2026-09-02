"""Resolve a place name to a bbox and a GeoJSON Feature.

Lookup order (no network until a later step):

1. Bundled Natural Earth countries — ISO3 or country name.
2. Bundled Natural Earth groupings — continent / UN / World Bank labels
   (``East Africa``, ``Sub-Saharan Africa``, …).
3. Bundled custom forecast boxes — ``Kenya OND region``, ``Indian Ocean basin``, ….
4. geoBoundaries admin-1 / admin-2 — ``kenya-nairobi`` or ``KEN-nairobi``.
5. OSM Nominatim — landmarks only, via :func:`geocode_nominatim`.

Rebuild the country file with ``tools/build_countries.py``.
"""

from __future__ import annotations

import json
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from functools import lru_cache
from importlib.resources import files

from weather_skills_core.errors import DataError, UsageError

_GB_API = "https://www.geoboundaries.org/api/current/gbOpen/{iso3}/ADM{level}/"
_NOMINATIM_SEARCH = "https://nominatim.openstreetmap.org/search"
_HTTP_TIMEOUT = 60
_USER_AGENT = (
    "weather-skills-core/resolve-region (https://github.com/rhiza-research/weather-skills)"
)
_REGION_FIELDS = ("subregion", "continent", "region_un", "region_wb")
_DIRECTIONAL_PREFIXES = (
    ("eastern_", "east_"),
    ("western_", "west_"),
    ("northern_", "north_"),
    ("southern_", "south_"),
    ("middle_", "central_"),
)

# Keys and values are already cleaned (see :func:`clean_region_name`).
_COUNTRY_ALIASES = {
    "ch-in": "china",
    "people's_republic_of_china": "china",
    "cabo_verde": "cape_verde",
    "central_african_rep": "central_african_republic",
    "congo,_dem_rep_of_the": "democratic_republic_of_the_congo",
    "congo,_rep_of_the": "republic_of_the_congo",
    "czech_republic": "czechia",
    "cote_d'ivoire": "ivory_coast",
    "east_timor": "timor-leste",
    "swaziland": "eswatini",
    "bahamas,_the": "the_bahamas",
    "gambia,_the": "the_gambia",
    "gambia": "the_gambia",
    "korea,_north": "north_korea",
    "korea,_south": "south_korea",
    "macedonia": "north_macedonia",
    "marshall_is": "marshall_islands",
    "micronesia,_fed_states_of": "federated_states_of_micronesia",
    "burma": "myanmar",
    "solomon_is": "solomon_islands",
    "st_kitts_and_nevis": "saint_kitts_and_nevis",
    "st_lucia": "saint_lucia",
    "st_vincent_and_the_grenadines": "saint_vincent_and_the_grenadines",
    "virgin_islands,_u.s.": "virgin_islands",
    "united_states": "united_states_of_america",
    "falkland_islands_(uk)": "falkland_islands",
    "gaza_strip": "palestine",
    "west_bank": "palestine",
}

# Forecast / briefing boxes that are not Natural Earth groupings or admin units.
# ``bbox`` is (N, W, S, E). Aliases are passed through :func:`clean_region_name`.
_CUSTOM_REGIONS = {
    "kenya_ond_region": {
        "name": "Kenya OND region",
        "bbox": (1.0, 36.5, -3.0, 39.0),
        "iso3": "KEN",
        "country": "Kenya",
        "aliases": (
            "Kenya OND",
            "OND Kenya",
            "Central-Eastern Kenya",
            "Central Eastern Kenya",
            "Central and Eastern Kenya",
            "CE Kenya",
        ),
    },
    "indian_ocean_basin": {
        "name": "Indian Ocean basin",
        "bbox": (30.0, 20.0, -40.0, 120.0),
        "iso3": None,
        "country": None,
        "aliases": (
            "Indian Ocean",
            "Indian Ocean Basin",
            "Indian Ocean basin region",
            "IOB",
        ),
    },
}


class _AdminMissing(Exception):
    """geoBoundaries has no ADM layer for this country/level."""


# --- names -----------------------------------------------------------------


def clean_region_name(name) -> str:
    """Normalize a place name for matching: lowercase, underscores, ASCII, aliases."""
    if name in (None, "none", "None", "", "_", "-", "-_", " "):
        return "no_region"
    text = str(name).lower().replace(" ", "_").strip()
    text = text.replace("&", "and")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return _COUNTRY_ALIASES.get(text, text)


def _is_iso3_token(cleaned: str) -> bool:
    return len(cleaned) == 3 and cleaned.isalpha()


# --- bbox -----------------------------------------------------------------


def _iter_coords(coords):
    """Yield (lon, lat) pairs by walking a nested GeoJSON coordinate array."""
    if (
        isinstance(coords, (list, tuple))
        and len(coords) >= 2
        and isinstance(coords[0], (int, float))
        and isinstance(coords[1], (int, float))
    ):
        yield coords[0], coords[1]
        return
    for item in coords:
        yield from _iter_coords(item)


def _lon_bounds(lons):
    """``(W, E)`` on the circle. Antimeridian crossings return ``W > E`` (RFC 7946)."""
    values = sorted(set(lons))
    n = len(values)
    if n == 1:
        return values[0], values[0]

    max_gap = -1.0
    gap_index = -1
    for i in range(n - 1):
        gap = values[i + 1] - values[i]
        if gap > max_gap:
            max_gap, gap_index = gap, i
    wrap_gap = (values[0] + 360.0) - values[n - 1]
    wrap_is_largest = wrap_gap > max_gap
    if wrap_is_largest:
        max_gap = wrap_gap
    if 360.0 - max_gap >= 350.0:
        return -180.0, 180.0
    if wrap_is_largest:
        return values[0], values[n - 1]
    return values[gap_index + 1], values[gap_index]


def bbox_from_geometry(geometry) -> tuple[float, float, float, float]:
    """Return ``(N, W, S, E)`` from a GeoJSON geometry object."""
    lons = []
    min_lat = float("inf")
    max_lat = float("-inf")
    for lon, lat in _iter_coords(geometry["coordinates"]):
        lons.append(lon)
        min_lat = min(min_lat, lat)
        max_lat = max(max_lat, lat)
    west, east = _lon_bounds(lons)
    return max_lat, west, min_lat, east


def bbox_from_feature(feature: dict) -> tuple[float, float, float, float]:
    """Return ``(N, W, S, E)``. ``properties.bbox`` wins when present (Nominatim / regions)."""
    props = feature.get("properties") or {}
    bbox = props.get("bbox")
    if bbox is not None:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise DataError("feature properties.bbox must be [N, W, S, E].")
        return float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    geometry = feature.get("geometry")
    if not isinstance(geometry, dict) or "coordinates" not in geometry:
        raise DataError("feature has no geometry to derive a bbox from.")
    return bbox_from_geometry(geometry)


def _bbox_rectangle(north: float, west: float, south: float, east: float) -> dict:
    """GeoJSON Polygon covering a simple (non-wrapped) N/W/S/E box."""
    return {
        "type": "Polygon",
        "coordinates": [
            [[west, south], [east, south], [east, north], [west, north], [west, south]]
        ],
    }


# --- GeoJSON features -------------------------------------------------------


def _feature(geometry, *, iso3, name, region_name, level, country, **extra) -> dict:
    return {
        "type": "Feature",
        "properties": {
            "iso3": iso3,
            "name": name,
            "region_name": region_name,
            "level": level,
            "country": country,
            **extra,
        },
        "geometry": geometry,
    }


def _polygon_parts(geometry: dict) -> list:
    kind, coords = geometry.get("type"), geometry.get("coordinates")
    if kind == "Polygon" and coords:
        return [coords]
    if kind == "MultiPolygon" and coords:
        return list(coords)
    return []


# --- HTTP -------------------------------------------------------------------


def _http_json(url: str, *, what: str, missing=None):
    """GET ``url`` and parse JSON. ``missing`` is raised on HTTP 404 when given."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404 and missing is not None:
            raise missing from None
        raise DataError(f"{what} failed: HTTP {exc.code} {exc.reason}") from None
    except urllib.error.URLError as exc:
        raise DataError(f"{what} failed: {exc.reason}") from None
    if raw.startswith(b"version https://git-lfs.github.com"):
        raise DataError(
            f"{what} returned a Git LFS pointer instead of JSON; cannot download {url}."
        )
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DataError(f"{what} returned invalid JSON from {url}: {exc}") from None


# --- bundled countries ------------------------------------------------------


@lru_cache(maxsize=1)
def _country_indexes() -> tuple[dict[str, dict], dict[str, dict]]:
    """``(by_iso3, by_clean_name)`` onto bundled country features."""
    path = files("weather_skills_core.data").joinpath("countries.geojson")
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_iso3: dict[str, dict] = {}
    by_clean: dict[str, dict] = {}
    for feature in payload["features"]:
        iso3 = feature["properties"]["iso3"]
        by_iso3[iso3] = feature
        cleaned = clean_region_name(feature["properties"].get("name", ""))
        if cleaned != "no_region":
            by_clean[cleaned] = feature
    # Alias keys so ``united_states-california`` splits on the short name.
    for alias, canonical in _COUNTRY_ALIASES.items():
        if canonical in by_clean:
            by_clean[alias] = by_clean[canonical]
    return by_iso3, by_clean


def _directional_alias_keys(cleaned: str) -> list[str]:
    extra = []
    for long, short in _DIRECTIONAL_PREFIXES:
        if cleaned.startswith(long):
            extra.append(short + cleaned[len(long) :])
        elif cleaned.startswith(short):
            extra.append(long + cleaned[len(short) :])
    return extra


@lru_cache(maxsize=1)
def _region_indexes() -> dict[str, dict]:
    """Cleaned query → multi-country grouping from bundled region fields."""
    by_iso3, _ = _country_indexes()
    by_key: dict[str, dict] = {}
    for field in _REGION_FIELDS:
        buckets: dict[str, list] = {}
        for feature in by_iso3.values():
            label = feature["properties"].get(field)
            if label:
                buckets.setdefault(label, []).append(feature)
        for label, members in buckets.items():
            key = clean_region_name(label)
            if key not in by_key and key != "no_region":
                by_key[key] = {"name": label, "members": members}
    for key, spec in list(by_key.items()):
        for alias in _directional_alias_keys(key):
            by_key.setdefault(alias, spec)
    return by_key


@lru_cache(maxsize=1)
def _custom_indexes() -> dict[str, dict]:
    """Cleaned query → bundled custom forecast box."""
    by_key: dict[str, dict] = {}
    for key, spec in _CUSTOM_REGIONS.items():
        by_key[key] = spec
        for alias in (spec["name"], *spec.get("aliases", ())):
            cleaned = clean_region_name(alias)
            if cleaned != "no_region":
                by_key.setdefault(cleaned, spec)
    return by_key


def _slim_custom(spec: dict) -> dict:
    north, west, south, east = spec["bbox"]
    return _feature(
        _bbox_rectangle(north, west, south, east),
        iso3=spec.get("iso3"),
        name=spec["name"],
        region_name=clean_region_name(spec["name"]),
        level="custom",
        country=spec.get("country"),
        bbox=[north, west, south, east],
    )


def _slim_country(feature: dict) -> dict:
    props = feature["properties"]
    return _feature(
        feature["geometry"],
        iso3=props["iso3"],
        name=props["name"],
        region_name=clean_region_name(props["name"]),
        level="country",
        country=props["name"],
    )


def _slim_region(spec: dict) -> dict:
    parts = []
    for feature in spec["members"]:
        parts.extend(_polygon_parts(feature["geometry"]))
    if not parts:
        raise DataError(f"Natural Earth region {spec['name']!r} has no polygon geometry.")
    geometry = (
        {"type": "Polygon", "coordinates": parts[0]}
        if len(parts) == 1
        else {"type": "MultiPolygon", "coordinates": parts}
    )
    north, west, south, east = bbox_from_geometry(geometry)
    return _feature(
        geometry,
        iso3=None,
        name=spec["name"],
        region_name=clean_region_name(spec["name"]),
        level="region",
        country=None,
        bbox=[north, west, south, east],
    )


def _match_bundled(cleaned: str) -> dict | None:
    """ISO3, country name, NE grouping, or custom box. Country names win (South Africa)."""
    by_iso3, by_clean = _country_indexes()
    if _is_iso3_token(cleaned):
        match = by_iso3.get(cleaned.upper())
        if match is not None:
            return _slim_country(match)
    match = by_clean.get(cleaned)
    if match is not None:
        return _slim_country(match)
    named = _region_indexes().get(cleaned)
    if named is not None:
        return _slim_region(named)
    custom = _custom_indexes().get(cleaned)
    if custom is not None:
        return _slim_custom(custom)
    return None


# --- geoBoundaries admin ----------------------------------------------------


def _gb_download_url(meta) -> str | None:
    if isinstance(meta, list):
        meta = meta[0] if meta else None
    if not isinstance(meta, dict):
        return None
    return meta.get("simplifiedGeometryGeoJSON") or meta.get("gjDownloadURL") or None


def _load_admin_geojson(iso3: str, level: int) -> dict:
    """Fetch one country's geoBoundaries ``gbOpen`` FeatureCollection. Overridable in tests."""
    api_url = _GB_API.format(iso3=iso3, level=level)
    missing = _AdminMissing(f"no geoBoundaries gbOpen ADM{level} layer for {iso3} ({api_url})")
    meta = _http_json(
        api_url, what=f"geoBoundaries ADM{level} metadata for {iso3}", missing=missing
    )
    download = _gb_download_url(meta)
    if not download:
        raise _AdminMissing(f"geoBoundaries ADM{level} metadata for {iso3} has no GeoJSON URL")
    payload = _http_json(
        download,
        what=f"geoBoundaries ADM{level} GeoJSON for {iso3}",
        missing=_AdminMissing(f"geoBoundaries ADM{level} GeoJSON for {iso3} not found"),
    )
    if not isinstance(payload, dict) or payload.get("type") != "FeatureCollection":
        raise DataError(f"geoBoundaries ADM{level} for {iso3} is not a GeoJSON FeatureCollection")
    return payload


@lru_cache(maxsize=32)
def _admin_collection(iso3: str, level: int) -> dict:
    return _load_admin_geojson(iso3, level)


def _features_by_clean_name(collection: dict | None) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for feature in (collection or {}).get("features") or []:
        name = clean_region_name((feature.get("properties") or {}).get("shapeName", ""))
        if name != "no_region":
            grouped.setdefault(name, []).append(feature)
    return grouped


def _one(hits: list[dict] | None) -> dict | None:
    return hits[0] if hits is not None and len(hits) == 1 else None


def _admin_indexes(iso3: str) -> tuple[dict, dict]:
    grouped = []
    for level in (1, 2):
        try:
            grouped.append(_features_by_clean_name(_admin_collection(iso3, level)))
        except _AdminMissing:
            grouped.append({})
    return grouped[0], grouped[1]


def _slim_admin(country: dict, feature: dict, level: int, region_name: str) -> dict:
    props = feature.get("properties") or {}
    country_name = country["properties"]["name"]
    return _feature(
        feature["geometry"],
        iso3=country["properties"]["iso3"],
        name=props.get("shapeName") or region_name,
        region_name=region_name,
        level=f"admin_{level}",
        country=country_name,
    )


def _find_admin(country: dict, remainder: str) -> dict | None:
    iso3 = country["properties"]["iso3"]
    keyed = f"{clean_region_name(country['properties']['name'])}-{remainder}"
    by1, by2 = _admin_indexes(iso3)

    hit = _one(by1.get(remainder))
    if hit is not None:
        return _slim_admin(country, hit, 1, keyed)
    hit = _one(by2.get(remainder))
    if hit is not None:
        return _slim_admin(country, hit, 2, keyed)

    for name in sorted(by1, key=len, reverse=True):
        prefix = f"{name}-"
        if remainder.startswith(prefix):
            hit = _one(by2.get(remainder[len(prefix) :]))
            if hit is not None:
                return _slim_admin(country, hit, 2, keyed)
    return None


def _split_subnational(cleaned: str) -> tuple[dict, str] | None:
    """``(country_feature, remainder)`` for ``kenya-nairobi`` / ``KEN-nairobi``."""
    if "-" not in cleaned:
        return None
    by_iso3, by_clean = _country_indexes()
    head, rest = cleaned.split("-", 1)
    if _is_iso3_token(head) and rest:
        country = by_iso3.get(head.upper())
        if country is not None:
            return country, rest
    for name in sorted(by_clean, key=len, reverse=True):
        prefix = f"{name}-"
        if cleaned.startswith(prefix) and cleaned[len(prefix) :]:
            return by_clean[name], cleaned[len(prefix) :]
    return None


# --- Nominatim --------------------------------------------------------------


def should_geocode(query: str) -> bool:
    """True when a failed :func:`lookup_region` may fall through to Nominatim.

    ISO3-shaped tokens, hierarchical admin keys, Natural Earth regions, and
    bundled custom forecast boxes stay off Nominatim even when the lookup
    itself failed (typo in the unit).
    """
    text = query.strip()
    if not text:
        return False
    cleaned = clean_region_name(text)
    if _is_iso3_token(cleaned) or cleaned in _region_indexes() or cleaned in _custom_indexes():
        return False
    return _split_subnational(cleaned) is None


def _load_nominatim(query: str) -> list:
    """GET Nominatim search. Overridable in tests."""
    params = urllib.parse.urlencode(
        {"q": query, "format": "jsonv2", "limit": "1", "polygon_geojson": "1"}
    )
    payload = _http_json(f"{_NOMINATIM_SEARCH}?{params}", what=f"Nominatim search for {query!r}")
    if not isinstance(payload, list):
        raise DataError(f"Nominatim search for {query!r} did not return a JSON list.")
    return payload


@lru_cache(maxsize=32)
def _nominatim_collection(query: str) -> tuple:
    return tuple(_load_nominatim(query))


def _parse_nominatim_bbox(raw) -> tuple[float, float, float, float]:
    """Nominatim ``boundingbox`` is ``[south, north, west, east]`` strings."""
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise DataError("Nominatim result is missing a 4-value boundingbox.")
    try:
        south, north, west, east = (float(v) for v in raw)
    except (TypeError, ValueError) as exc:
        raise DataError(f"Nominatim boundingbox {raw!r} is not numeric.") from exc
    return north, west, south, east


def _nominatim_geometry(hit: dict, north, west, south, east) -> dict:
    geo = hit.get("geojson")
    if isinstance(geo, dict) and geo.get("type") in {"Polygon", "MultiPolygon"}:
        return geo
    if west <= east:
        return _bbox_rectangle(north, west, south, east)
    try:
        return {"type": "Point", "coordinates": [float(hit["lon"]), float(hit["lat"])]}
    except (TypeError, ValueError, KeyError):
        return {"type": "Point", "coordinates": [west, (south + north) / 2.0]}


def geocode_nominatim(query: str) -> dict:
    """Resolve leftover free text through OSM Nominatim (limit=1)."""
    text = query.strip()
    if not text:
        raise UsageError("Nominatim query must be a non-empty place name.")
    hits = _nominatim_collection(text)
    if not hits:
        raise DataError(
            f"{query!r} is not a known ISO3 code, country name, named region, "
            "or sub-national region, and Nominatim found no matching place. "
            "Pass an ISO3 code, country-admin1, a named region "
            "(e.g. East Africa, Kenya OND region), "
            "or a more specific landmark (e.g. 'Mount Kenya, Kenya')."
        )
    hit = hits[0]
    if not isinstance(hit, dict):
        raise DataError(f"Nominatim search for {query!r} returned a malformed hit.")
    north, west, south, east = _parse_nominatim_bbox(hit.get("boundingbox"))
    display = hit.get("display_name") or hit.get("name") or text
    return _feature(
        _nominatim_geometry(hit, north, west, south, east),
        iso3=None,
        name=hit.get("name") or display,
        region_name=clean_region_name(text),
        level="nominatim",
        country=None,
        display_name=display,
        bbox=[north, west, south, east],
    )


# --- public lookup ----------------------------------------------------------


def lookup_region(query: str) -> dict:
    """Resolve ISO3, country, NE region, custom box, or ``country-admin`` to a Feature."""
    text = query.strip()
    if not text:
        raise UsageError(
            "region query must be a non-empty ISO3 code, country name, "
            "named region (e.g. East Africa, Kenya OND region), or sub-national region "
            "(e.g. kenya-nairobi)."
        )

    cleaned = clean_region_name(text)
    match = _match_bundled(cleaned)
    if match is not None:
        return match

    split = _split_subnational(cleaned)
    if split is not None:
        country, remainder = split
        found = _find_admin(country, remainder)
        if found is not None:
            return found
        iso3 = country["properties"]["iso3"]
        country_name = country["properties"]["name"]
        raise DataError(
            f"{query!r} is not a known admin-1 or admin-2 region of {country_name} "
            f"({iso3}) in geoBoundaries gbOpen. Pass country-admin1 or "
            "country-admin1-admin2 after cleaning "
            "(e.g. kenya-nairobi, united_states_of_america-california-los_angeles)."
        )

    raise DataError(
        f"{query!r} is not a known ISO3 code, country name, named region "
        "(e.g. East Africa, Kenya OND region), or sub-national region "
        "(country-admin1 / country-admin1-admin2)."
    )


def resolve_region(query: str):
    """Resolve a query to ``((N, W, S, E), GeoDataFrame)``. Needs ``[geo]`` extra."""
    import geopandas as gpd

    match = lookup_region(query)
    gdf = gpd.GeoDataFrame.from_features(
        {"type": "FeatureCollection", "features": [match]},
        crs="EPSG:4326",
    )
    return bbox_from_feature(match), gdf
