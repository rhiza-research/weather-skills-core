"""Dataset typing, CF dims, bbox, and CF helpers used by skills."""

import numpy as np
import pytest
from conftest import make_forecast, make_gridded, make_station

from weather_skills_core import dataset
from weather_skills_core.errors import UsageError


def test_parse_bbox():
    assert dataset.parse_bbox("1/2/3/4") == (1.0, 2.0, 3.0, 4.0)
    assert dataset.parse_bbox("-1/-20/-5/-10") == (-1.0, -20.0, -5.0, -10.0)
    with pytest.raises(UsageError, match="N/W/S/E"):
        dataset.parse_bbox("1/2/3")


def test_detect_types():
    assert dataset.detect_type(make_gridded()) == dataset.GRIDDED
    assert dataset.detect_type(make_forecast()) == dataset.FORECAST
    assert dataset.detect_type(make_station()) == dataset.STATION


def test_cf_only_spatial_dims():
    assert dataset.detect_spatial_dims(make_gridded()) == ("latitude", "longitude")
    ds = make_gridded().rename({"latitude": "row", "longitude": "col"})
    for name in ("row", "col"):
        ds[name].attrs.clear()
    with pytest.raises(UsageError, match="CF metadata"):
        dataset.detect_spatial_dims(ds)

    stamped = make_gridded().rename({"latitude": "yy", "longitude": "xx"})
    stamped["yy"].attrs.update(standard_name="latitude", units="degrees_north")
    stamped["xx"].attrs.update(standard_name="longitude", units="degrees_east")
    assert dataset.detect_spatial_dims(stamped) == ("yy", "xx")


def test_validate_type():
    from weather_skills_core import Types, validate_type

    assert validate_type(make_gridded(), "gridded", "in.zarr") == "gridded"
    validate_type(make_forecast(), "any", "in.zarr")
    assert validate_type(make_forecast(), Types.FORECAST) == "forecast"
    assert validate_type(make_gridded(), make_gridded()) == "gridded"
    with pytest.raises(UsageError, match="no 'step' dim"):
        validate_type(make_gridded(), "forecast", "in.zarr")
    with pytest.raises(UsageError, match="forecast"):
        validate_type(make_gridded(), Types.FORECAST)
    with pytest.raises(UsageError, match="gridded"):
        validate_type(make_forecast(), make_gridded())


def test_bbox_subset_and_antimeridian():
    ds = make_gridded(lats=(1.0, 2.0, 3.0), lons=(10.0, 11.0, 12.0, 13.0))
    sub = dataset.bbox_subset(ds, (2.5, 10.5, 0.5, 12.5))
    assert list(sub["latitude"].values) == [1.0, 2.0]
    assert list(sub["longitude"].values) == [11.0, 12.0]

    world = make_gridded(lons=tuple(float(x) for x in range(-170, 180, 10)))
    crossed = dataset.bbox_subset(world, (90.0, 170.0, -90.0, -170.0))
    lons = list(crossed["longitude"].values)
    assert all(lon >= 170 or lon <= -170 for lon in lons)
    assert 0.0 not in lons


def test_stamp_cf_attrs():
    ds = make_gridded()
    for name in ("latitude", "longitude", "time"):
        ds[name].attrs.clear()
    out = dataset.stamp_cf_attrs(ds)
    assert out["latitude"].attrs["standard_name"] == "latitude"
    assert out["longitude"].attrs["axis"] == "X"


def test_stamp_and_verify_cf_dsg():
    ds = make_station()
    for name in ("latitude", "longitude", "time", "station_id"):
        ds[name].attrs.clear()
    stamped = dataset.stamp_cf_dsg(
        ds,
        {"precip": {"units": "mm", "standard_name": "lwe_thickness_of_precipitation_amount"}},
        station_id_long_name="station",
        name_long_name="name",
    )
    dataset.verify_cf_dsg(stamped)


def test_normalize_longitude():
    ds = make_gridded(lons=(0.0, 90.0, 180.0, 270.0))
    out = dataset.normalize_longitude(ds)
    assert list(out["longitude"].values) == [-180.0, -90.0, 0.0, 90.0]


def test_cf_dim_and_lat_slice():
    ds = make_gridded()
    assert dataset.cf_dim(ds, "latitude") == "latitude"
    bare = ds.rename({"latitude": "row"})
    bare["row"].attrs.clear()
    assert dataset.cf_dim(bare, "latitude") is None
    assert dataset.lat_slice(np.array([1.0, 2.0, 3.0]), 2.5, 0.5) == slice(0.5, 2.5)
    assert dataset.lat_slice(np.array([3.0, 2.0, 1.0]), 2.5, 0.5) == slice(2.5, 0.5)


def test_auto_variable():
    ds = make_gridded()
    assert dataset.auto_variable(ds) == "precip"
    ds["crs"] = 0
    ds["crs"].attrs["grid_mapping_name"] = "latitude_longitude"
    ds["precip"].attrs["grid_mapping"] = "crs"
    assert dataset.auto_variable(ds) == "precip"


def test_polygon_from_geojson(tmp_path):
    path = tmp_path / "box.geojson"
    path.write_text(
        '{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,1],[0,0]]]}'
    )
    geom = dataset.polygon_from_geojson(path)
    assert geom.bounds == (0.0, 0.0, 1.0, 1.0)
    with pytest.raises(UsageError, match="--mask-geojson"):
        dataset.polygon_from_geojson(tmp_path / "missing.geojson")


def test_udunits_error():
    assert dataset.udunits_error("mm") is None
    assert dataset.udunits_error("not-a-unit") is not None


def test_latitude_weights():
    w = dataset.latitude_weights(np.array([0.0, 60.0]))
    assert float(w.mean()) == pytest.approx(1.0)
    assert float(w[0]) > float(w[1])


def test_roll_and_agg_left_align():
    ds = make_gridded(n_time=10, start="2026-01-01", fill=1.0)
    out = dataset.roll_and_agg(ds, 3, "time", "sum", align="left")
    # First full window labeled at 2026-01-01 (left); length 10-2=8.
    assert out.sizes["time"] == 8
    assert np.datetime_as_string(out["time"].values[0], unit="D") == "2026-01-01"
    assert float(out["precip"].isel(time=0, latitude=0, longitude=0)) == pytest.approx(3.0)


def test_clip_by_geometry():
    from shapely.geometry import box

    ds = make_gridded(lats=(0.0, 1.0, 2.0), lons=(10.0, 11.0, 12.0))
    out = dataset.clip_by_geometry(ds, box(10.5, 0.5, 11.5, 1.5), drop=True)
    assert list(out["latitude"].values) == [1.0]
    assert list(out["longitude"].values) == [11.0]
