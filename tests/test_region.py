"""Tests for region resolution (ISO3, country name, admin-1/admin-2)."""

import pytest

from weather_skills_core import region as region_mod
from weather_skills_core.errors import DataError, UsageError
from weather_skills_core.region import (
    bbox_from_feature,
    bbox_from_geometry,
    clean_region_name,
    geocode_nominatim,
    lookup_region,
    resolve_region,
    should_geocode,
)

_NAIROBI = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {
                "shapeName": "Nairobi",
                "shapeGroup": "KEN",
                "shapeType": "ADM1",
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [36.6, -1.4],
                        [37.0, -1.4],
                        [37.0, -1.1],
                        [36.6, -1.1],
                        [36.6, -1.4],
                    ]
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {
                "shapeName": "Elgeyo-Marakwet",
                "shapeGroup": "KEN",
                "shapeType": "ADM1",
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[35.0, 0.5], [35.5, 0.5], [35.5, 1.0], [35.0, 1.0], [35.0, 0.5]]],
            },
        },
    ],
}

_WESTLANDS = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {
                "shapeName": "Westlands",
                "shapeGroup": "KEN",
                "shapeType": "ADM2",
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [36.7, -1.3],
                        [36.9, -1.3],
                        [36.9, -1.2],
                        [36.7, -1.2],
                        [36.7, -1.3],
                    ]
                ],
            },
        }
    ],
}


@pytest.fixture(autouse=True)
def _clear_admin_cache():
    region_mod._admin_collection.cache_clear()
    region_mod._nominatim_collection.cache_clear()
    yield
    region_mod._admin_collection.cache_clear()
    region_mod._nominatim_collection.cache_clear()


def _fake_admin(iso3, level):
    if iso3 != "KEN":
        raise AssertionError(f"unexpected fetch {iso3} admin-{level}")
    if level == 1:
        return _NAIROBI
    if level == 2:
        return _WESTLANDS
    raise AssertionError(f"unexpected admin level {level}")


def test_clean_region_name():
    """Aliases, ASCII folding, and spaces-to-underscores land on the lookup key."""
    assert clean_region_name("United States") == "united_states_of_america"
    assert clean_region_name("South Korea") == "south_korea"
    assert clean_region_name("São Tomé") == "sao_tome"
    assert clean_region_name("Côte d'Ivoire") == "ivory_coast"


def test_resolve_kenya_iso3():
    """ISO3 KEN returns Kenya's bbox and a one-row GeoDataFrame."""
    bbox, gdf = resolve_region("KEN")
    n, w, s, e = bbox
    assert n > s and w < e
    assert 4.0 < n < 6.0
    assert 33.0 < w < 35.0
    assert -5.5 < s < -4.0
    assert 41.0 < e < 43.0
    assert list(gdf["iso3"]) == ["KEN"]
    assert list(gdf["name"]) == ["Kenya"]


def test_resolve_by_name_case_insensitive():
    """Country name and ISO3 resolve to the same bbox."""
    bbox_code, _ = resolve_region("KEN")
    bbox_name, gdf = resolve_region("Kenya")
    assert bbox_code == bbox_name
    assert list(gdf["iso3"]) == ["KEN"]


def test_resolve_unknown():
    """An unknown ISO3 is a DataError, not a Nominatim fallthrough."""
    with pytest.raises(DataError, match="not a known"):
        resolve_region("ZZZ")


def test_lookup_empty_is_usage_error():
    """Whitespace-only queries are UsageError."""
    with pytest.raises(UsageError, match="non-empty"):
        lookup_region("  ")


def test_bbox_wraps_antimeridian():
    """A polygon that crosses 180° reports W > E instead of a full-width box."""
    geometry = {
        "type": "Polygon",
        "coordinates": [[[170.0, 0.0], [175.0, 0.0], [-170.0, 0.0], [170.0, 0.0]]],
    }
    _n, w, _s, e = bbox_from_geometry(geometry)
    assert w > e
    assert w == 170.0
    assert e == -170.0


def test_lookup_admin1_name(monkeypatch):
    """country-admin1 resolves Nairobi as admin_1 with its polygon bbox."""
    monkeypatch.setattr("weather_skills_core.region._load_admin_geojson", _fake_admin)

    feature = lookup_region("kenya-nairobi")
    assert feature["properties"]["level"] == "admin_1"
    assert feature["properties"]["name"] == "Nairobi"
    assert feature["properties"]["iso3"] == "KEN"
    assert feature["properties"]["region_name"] == "kenya-nairobi"

    n, w, s, e = bbox_from_geometry(feature["geometry"])
    assert n == pytest.approx(-1.1)
    assert s == pytest.approx(-1.4)
    assert w == pytest.approx(36.6)
    assert e == pytest.approx(37.0)


def test_lookup_admin1_iso3_prefix(monkeypatch):
    """KEN-nairobi is the same admin_1 lookup as kenya-nairobi."""
    monkeypatch.setattr("weather_skills_core.region._load_admin_geojson", _fake_admin)

    feature = lookup_region("KEN-nairobi")
    assert feature["properties"]["region_name"] == "kenya-nairobi"
    assert feature["properties"]["level"] == "admin_1"


def test_lookup_admin2(monkeypatch):
    """country-admin1-admin2 resolves Westlands as admin_2."""
    monkeypatch.setattr("weather_skills_core.region._load_admin_geojson", _fake_admin)

    feature = lookup_region("kenya-nairobi-westlands")
    assert feature["properties"]["level"] == "admin_2"
    assert feature["properties"]["name"] == "Westlands"
    assert feature["properties"]["region_name"] == "kenya-nairobi-westlands"


def test_lookup_admin2_without_parent(monkeypatch):
    """An admin-2 name is found even without its parent admin-1 in the key."""
    monkeypatch.setattr("weather_skills_core.region._load_admin_geojson", _fake_admin)

    feature = lookup_region("kenya-westlands")
    assert feature["properties"]["level"] == "admin_2"
    assert feature["properties"]["name"] == "Westlands"


def test_lookup_hyphenated_admin1(monkeypatch):
    """A hyphen inside the admin-1 name (Elgeyo-Marakwet) is not a new hierarchy level."""
    monkeypatch.setattr("weather_skills_core.region._load_admin_geojson", _fake_admin)

    feature = lookup_region("kenya-elgeyo-marakwet")
    assert feature["properties"]["level"] == "admin_1"
    assert feature["properties"]["name"] == "Elgeyo-Marakwet"


def test_lookup_unknown_admin_unit(monkeypatch):
    """A country-prefixed key that matches no admin unit is a DataError."""
    monkeypatch.setattr("weather_skills_core.region._load_admin_geojson", _fake_admin)

    with pytest.raises(DataError, match="admin-1 or admin-2"):
        lookup_region("kenya-not-a-county")


def test_hyphenated_country_is_not_split_into_admin():
    """Guinea-Bissau is a country, not country 'Guinea' plus admin 'Bissau'."""
    feature = lookup_region("Guinea-Bissau")
    assert feature["properties"]["level"] == "country"
    assert feature["properties"]["iso3"] == "GNB"


_MOUNT_KENYA_POLYGON = {
    "type": "Polygon",
    "coordinates": [
        [
            [37.2, -0.25],
            [37.4, -0.25],
            [37.4, -0.05],
            [37.2, -0.05],
            [37.2, -0.25],
        ]
    ],
}


def _mount_kenya_hit(*, geojson=None):
    return [
        {
            "display_name": "Mount Kenya, Kenya",
            "name": "Mount Kenya",
            "boundingbox": ["-0.25", "-0.05", "37.2", "37.4"],
            "lat": "-0.15",
            "lon": "37.3",
            "geojson": geojson,
        }
    ]


def test_should_geocode_landmarks_not_admin_keys():
    """Landmarks may hit Nominatim; ISO3, admin keys, NE regions, and custom boxes must not."""
    assert should_geocode("Mount Kenya") is True
    assert should_geocode("Mount Kenya, Kenya") is True
    assert should_geocode("kenya-nairbi") is False
    assert should_geocode("KEN-nairbi") is False
    assert should_geocode("ZZZ") is False
    assert should_geocode("  ") is False
    assert should_geocode("East Africa") is False
    assert should_geocode("Eastern Africa") is False
    assert should_geocode("Western Africa") is False
    assert should_geocode("Sub-Saharan Africa") is False
    assert should_geocode("Kenya OND region") is False
    assert should_geocode("Kenya OND") is False
    assert should_geocode("CE Kenya") is False
    assert should_geocode("Indian Ocean") is False
    assert should_geocode("Indian Ocean basin") is False
    assert should_geocode("IOB") is False


def test_lookup_ne_eastern_africa_not_nominatim(monkeypatch):
    """East Africa / Eastern Africa dissolve bundled countries, never Nominatim."""

    def _fail_nominatim(query):
        raise AssertionError(f"Nominatim should not run for named region; got {query!r}")

    monkeypatch.setattr("weather_skills_core.region._load_nominatim", _fail_nominatim)

    kenya = lookup_region("KEN")
    kn, kw, ks, ke = bbox_from_feature(kenya)
    east = lookup_region("East Africa")
    eastern = lookup_region("Eastern Africa")
    assert east["properties"]["name"] == "Eastern Africa"
    assert east["properties"]["level"] == "region"
    assert east["properties"]["region_name"] == "eastern_africa"
    assert east["geometry"]["type"] == "MultiPolygon"
    n, w, s, e = bbox_from_feature(east)
    assert bbox_from_feature(eastern) == (n, w, s, e)
    assert s <= ks <= kn <= n
    assert w <= kw <= ke <= e
    # UN-style Eastern Africa includes Madagascar, not a Ugandan POI.
    assert s < -20
    assert n > 10


def test_lookup_kenya_ond_region_not_nominatim(monkeypatch):
    """Kenya OND region is a bundled forecast box, never Nominatim."""

    def _fail_nominatim(query):
        raise AssertionError(f"Nominatim should not run for named region; got {query!r}")

    monkeypatch.setattr("weather_skills_core.region._load_nominatim", _fail_nominatim)

    kenya = lookup_region("KEN")
    kn, kw, ks, ke = bbox_from_feature(kenya)
    feature = lookup_region("Kenya OND region")
    props = feature["properties"]
    assert props["name"] == "Kenya OND region"
    assert props["level"] == "custom"
    assert props["iso3"] == "KEN"
    assert props["country"] == "Kenya"
    n, w, s, e = bbox_from_feature(feature)
    assert (n, w, s, e) == (1.0, 36.5, -3.0, 39.0)
    assert ks <= s < n <= kn
    assert kw <= w < e <= ke
    assert feature["geometry"]["type"] == "Polygon"

    aliases = (
        lookup_region("Kenya OND"),
        lookup_region("OND Kenya"),
        lookup_region("Central-Eastern Kenya"),
        lookup_region("CE Kenya"),
    )
    for other in aliases:
        assert bbox_from_feature(other) == (n, w, s, e)
        assert other["properties"]["name"] == "Kenya OND region"


def test_lookup_indian_ocean_basin_not_nominatim(monkeypatch):
    """Indian Ocean is the conventional basin box, never a Nominatim centroid."""

    def _fail_nominatim(query):
        raise AssertionError(f"Nominatim should not run for Indian Ocean; got {query!r}")

    monkeypatch.setattr("weather_skills_core.region._load_nominatim", _fail_nominatim)

    feature = lookup_region("Indian Ocean basin")
    props = feature["properties"]
    assert props["name"] == "Indian Ocean basin"
    assert props["level"] == "custom"
    assert props["iso3"] is None
    n, w, s, e = bbox_from_feature(feature)
    assert (n, w, s, e) == (30.0, 20.0, -40.0, 120.0)
    assert feature["geometry"]["type"] == "Polygon"

    aliases = (
        lookup_region("Indian Ocean"),
        lookup_region("Indian Ocean Basin"),
        lookup_region("Indian Ocean basin region"),
        lookup_region("IOB"),
    )
    for other in aliases:
        assert bbox_from_feature(other) == (n, w, s, e)
        assert other["properties"]["name"] == "Indian Ocean basin"


def test_south_africa_is_the_country_not_the_subregion():
    """South Africa matches the country, not a Natural Earth subregion of that name."""
    feature = lookup_region("South Africa")
    assert feature["properties"]["level"] == "country"
    assert feature["properties"]["iso3"] == "ZAF"


def test_lookup_ne_western_africa():
    """West Africa is the UN Western Africa grouping and contains Ghana."""
    ghana = lookup_region("GHA")
    gn, gw, gs, ge = bbox_from_feature(ghana)
    west = lookup_region("West Africa")
    assert west["properties"]["name"] == "Western Africa"
    n, w, s, e = bbox_from_feature(west)
    assert s <= gs <= gn <= n
    assert w <= gw <= ge <= e


def test_geocode_nominatim_polygon(monkeypatch):
    """A Nominatim polygon hit is stored as geometry with bbox on properties."""
    monkeypatch.setattr(
        "weather_skills_core.region._load_nominatim",
        lambda query: _mount_kenya_hit(geojson=_MOUNT_KENYA_POLYGON),
    )

    feature = geocode_nominatim("Mount Kenya")
    props = feature["properties"]
    assert props["level"] == "nominatim"
    assert props["name"] == "Mount Kenya"
    assert props["display_name"] == "Mount Kenya, Kenya"
    assert props["bbox"] == pytest.approx([-0.05, 37.2, -0.25, 37.4])
    n, w, s, e = bbox_from_feature(feature)
    assert (n, w, s, e) == pytest.approx((-0.05, 37.2, -0.25, 37.4))
    assert feature["geometry"]["type"] == "Polygon"


def test_geocode_nominatim_point_uses_bbox_rectangle(monkeypatch):
    """A Nominatim Point is replaced by a rectangle from the boundingbox."""
    monkeypatch.setattr(
        "weather_skills_core.region._load_nominatim",
        lambda query: _mount_kenya_hit(geojson={"type": "Point", "coordinates": [37.3, -0.15]}),
    )

    feature = geocode_nominatim("Mount Kenya")
    assert feature["geometry"]["type"] == "Polygon"
    n, w, s, e = bbox_from_feature(feature)
    assert (n, w, s, e) == pytest.approx((-0.05, 37.2, -0.25, 37.4))


def test_geocode_nominatim_empty(monkeypatch):
    """Zero Nominatim hits is a DataError."""
    monkeypatch.setattr("weather_skills_core.region._load_nominatim", lambda query: [])

    with pytest.raises(DataError, match="Nominatim found no matching place"):
        geocode_nominatim("Not A Real Volcano 12345")


def test_bbox_from_feature_prefers_stored_bbox():
    """properties.bbox wins over deriving a box from a Point geometry."""
    feature = {
        "type": "Feature",
        "properties": {"bbox": [1.0, 2.0, -1.0, 4.0]},
        "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
    }
    assert bbox_from_feature(feature) == (1.0, 2.0, -1.0, 4.0)
