from datetime import date

import numpy as np
import pytest
import xarray as xr
from conftest import make_gridded, make_station

from weather_skills_core import standard_utils as utils
from weather_skills_core.errors import DataError, UsageError


def test_parse_date_absolute():
    assert utils.parse_date("2026-01-15") == date(2026, 1, 15)


def test_parse_date_rejects_relative():
    for value in ("now", "today", "latest", "now-3d", "latest-1w"):
        with pytest.raises(UsageError, match="YYYY-MM-DD"):
            utils.parse_date(value)


def test_parse_date_rejects_compact():
    with pytest.raises(UsageError, match="YYYY-MM-DD"):
        utils.parse_date("20260115")


def test_parse_range_ok():
    assert utils.parse_range("2026-01-01", "2026-01-07") == (date(2026, 1, 1), date(2026, 1, 7))


def test_parse_range_reversed():
    with pytest.raises(UsageError, match="reversed"):
        utils.parse_range("2026-01-10", "2026-01-01")


def test_parse_bbox_valid():
    assert utils.parse_bbox("1/2/3/4") == (1.0, 2.0, 3.0, 4.0)


def test_parse_bbox_negative_values():
    assert utils.parse_bbox("-1/32/-5/42") == (-1.0, 32.0, -5.0, 42.0)


@pytest.mark.parametrize("value", ["1/2/3", "1/2/3/4/5", "a/b/c/d", ""])
def test_parse_bbox_malformed(value):
    with pytest.raises(UsageError, match="N/W/S/E"):
        utils.parse_bbox(value)


def test_bbox_subset_ascending_latitude():
    ds = make_gridded(lats=(1.0, 2.0, 3.0), lons=(10.0, 11.0, 12.0, 13.0))
    sub = utils.bbox_subset(ds, (2.5, 10.5, 0.5, 12.5))
    assert list(sub["latitude"].values) == [1.0, 2.0]
    assert list(sub["longitude"].values) == [11.0, 12.0]


def test_bbox_subset_descending_latitude_same_bbox():
    ds = make_gridded(lats=(3.0, 2.0, 1.0))
    sub = utils.bbox_subset(ds, (2.5, 10.5, 0.5, 12.5))
    assert list(sub["latitude"].values) == [2.0, 1.0]


def test_bbox_subset_lon_0_360_normalized():
    ds = make_gridded(lons=(0.0, 90.0, 180.0, 270.0, 359.0))
    sub = utils.bbox_subset(ds, (3.0, -95.0, 1.0, -85.0))
    assert list(sub["longitude"].values) == [-90.0]


def test_bbox_subset_antimeridian_keeps_wings_drops_interior():
    ds = make_gridded(lons=(-179.0, -100.0, 0.0, 100.0, 179.0))
    sub = utils.bbox_subset(ds, (3.0, 170.0, 1.0, -170.0))
    assert list(sub["longitude"].values) == [-179.0, 179.0]


def test_bbox_subset_single_row_latitude_passes_through():
    ds = make_gridded(lats=(1.0,))
    sub = utils.bbox_subset(ds, (60.0, 10.5, 50.0, 12.5))
    assert list(sub["latitude"].values) == [1.0]


def test_bbox_subset_non_monotonic_latitude_rejected():
    ds = make_gridded(lats=(1.0, 3.0, 2.0))
    with pytest.raises(UsageError, match="lat axis is non-monotonic"):
        utils.bbox_subset(ds, (3.0, 10.0, 1.0, 13.0))


def test_bbox_subset_non_monotonic_longitude_rejected():
    ds = make_gridded(lons=(10.0, 12.0, 11.0, 13.0))
    with pytest.raises(UsageError, match="lon axis is non-monotonic"):
        utils.bbox_subset(ds, (3.0, 10.0, 1.0, 13.0))


def test_bbox_subset_empty_longitude_axis_rejected():
    ds = make_gridded(lons=())
    with pytest.raises(UsageError, match="lon axis has length 0"):
        utils.bbox_subset(ds, (3.0, 10.0, 1.0, 13.0))


def test_bbox_subset_descending_longitude_contiguous_span():
    ds = make_gridded(lons=(13.0, 12.0, 11.0, 10.0))
    sub = utils.bbox_subset(ds, (2.5, 10.5, 0.5, 12.5))
    assert list(sub["longitude"].values) == [12.0, 11.0]


def test_bbox_subset_antimeridian_preserves_integer_dtype():
    ds = make_gridded(lons=(-179.0, -100.0, 0.0, 100.0, 179.0))
    ds["count"] = (("time", "latitude", "longitude"), np.ones((2, 3, 5), dtype=np.int32))
    sub = utils.bbox_subset(ds, (3.0, 170.0, 1.0, -170.0))
    assert sub["count"].dtype == np.int32
    assert list(sub["longitude"].values) == [-179.0, 179.0]


def test_bbox_subset_antimeridian_descending_longitude_keeps_native_order():
    ds = make_gridded(lons=(179.0, 100.0, 0.0, -100.0, -179.0))
    sub = utils.bbox_subset(ds, (3.0, 170.0, 1.0, -170.0))
    assert list(sub["longitude"].values) == [179.0, -179.0]


def test_bbox_subset_antimeridian_leaves_non_longitude_variables_alone():
    ds = make_gridded(lons=(-179.0, 0.0, 179.0))
    ds["tavg"] = (("time",), np.array([5, 6], dtype=np.int16))
    sub = utils.bbox_subset(ds, (3.0, 170.0, 1.0, -170.0))
    assert sub["tavg"].dims == ("time",)
    assert sub["tavg"].dtype == np.int16
    assert list(sub["tavg"].values) == [5, 6]


def test_bbox_subset_empty_result_is_data_error():
    ds = make_gridded()
    with pytest.raises(DataError, match="selects no grid cells"):
        utils.bbox_subset(ds, (60.0, 10.0, 50.0, 13.0))


def test_bbox_subset_empty_antimeridian_result_names_the_crossing():
    ds = make_gridded(lons=(-10.0, 0.0, 10.0))
    with pytest.raises(DataError, match="antimeridian"):
        utils.bbox_subset(ds, (3.0, 170.0, 1.0, -170.0))


def test_bbox_subset_string_bbox_accepted():
    sub = utils.bbox_subset(make_gridded(), "2.5/10.5/0.5/12.5")
    assert list(sub["latitude"].values) == [1.0, 2.0]


def test_bbox_subset_explicit_dims():
    ds = make_gridded().rename({"latitude": "yy", "longitude": "xx"})
    sub = utils.bbox_subset(ds, (2.5, 10.5, 0.5, 12.5), lat_dim="yy", lon_dim="xx")
    assert list(sub["yy"].values) == [1.0, 2.0]


def test_bbox_subset_data_selected_matches_coords():
    ds = make_gridded()
    sub = utils.bbox_subset(ds, (2.5, 10.5, 0.5, 12.5))
    assert sub["precip"].shape == (2, 2, 2)
    assert isinstance(sub, xr.Dataset)


def test_bbox_subset_stations_keeps_in_box():
    ds = make_station()
    sub = utils.bbox_subset(ds, (0.5, 35.0, -0.5, 37.5))
    assert list(sub.station_id.values) == ["TA0001"]
    assert sub.sizes["time"] == 2


def test_bbox_subset_stations_unsorted_lat_lon():
    ds = make_station(n_station=3)
    ds = ds.assign_coords(
        latitude=("station_id", [1.28, -4.04, 0.51]),
        longitude=("station_id", [36.82, 39.67, 34.77]),
    )
    sub = utils.bbox_subset(ds, (-3.9187, 39.5667, -4.1550, 39.7639))
    assert list(sub.station_id.values) == ["TA0001"]


def test_bbox_subset_stations_empty_is_data_error():
    with pytest.raises(DataError, match="no stations"):
        utils.bbox_subset(make_station(), (60.0, 10.0, 50.0, 13.0))


def test_bbox_subset_point_id_dim():
    ds = make_station().rename({"station_id": "point_id"})
    sub = utils.bbox_subset(ds, (0.5, 35.0, -0.5, 37.5))
    assert list(sub.point_id.values) == ["TA0001"]


def test_bbox_subset_stations_antimeridian():
    ds = make_station(n_station=3)
    ds = ds.assign_coords(
        latitude=("station_id", [0.0, 0.0, 0.0]),
        longitude=("station_id", [170.0, 0.0, -175.0]),
    )
    sub = utils.bbox_subset(ds, (1.0, 165.0, -1.0, -170.0))
    assert list(sub.station_id.values) == ["TA0000", "TA0002"]


def test_lat_slice_ascending():
    assert utils.lat_slice(np.array([1.0, 2.0, 3.0]), 3.0, 1.0) == slice(1.0, 3.0)


def test_lat_slice_descending():
    assert utils.lat_slice(np.array([3.0, 2.0, 1.0]), 3.0, 1.0) == slice(3.0, 1.0)


def test_lat_slice_empty_axis_defaults_to_ascending():
    assert utils.lat_slice(np.array([]), 3.0, 1.0) == slice(1.0, 3.0)


def test_lat_slice_single_value():
    assert utils.lat_slice(np.array([2.0]), 3.0, 1.0) == slice(1.0, 3.0)


_polygon_from_geojson_square = {
    "type": "Polygon",
    "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]],
}
_polygon_from_geojson_east_square = {
    "type": "Polygon",
    "coordinates": [[[2, 0], [2, 1], [3, 1], [3, 0], [2, 0]]],
}


def _polygon_from_geojson_write(tmp_path, payload):
    import json

    p = tmp_path / "mask.geojson"
    p.write_text(json.dumps(payload))
    return p


def test_polygon_from_geojson_feature_collection_unions_all_features(tmp_path):
    payload = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": _polygon_from_geojson_square},
            {"type": "Feature", "geometry": _polygon_from_geojson_east_square},
            {"type": "Feature", "geometry": None},
        ],
    }
    poly = utils.polygon_from_geojson(_polygon_from_geojson_write(tmp_path, payload))
    assert poly.area == pytest.approx(2.0)


def test_polygon_from_geojson_single_feature(tmp_path):
    payload = {"type": "Feature", "geometry": _polygon_from_geojson_square}
    poly = utils.polygon_from_geojson(_polygon_from_geojson_write(tmp_path, payload))
    assert poly.area == pytest.approx(1.0)


def test_polygon_from_geojson_bare_geometry(tmp_path):
    poly = utils.polygon_from_geojson(
        _polygon_from_geojson_write(tmp_path, _polygon_from_geojson_square)
    )
    assert poly.area == pytest.approx(1.0)


def test_polygon_from_geojson_missing_file(tmp_path):
    with pytest.raises(UsageError, match="--mask-geojson file not found"):
        utils.polygon_from_geojson(tmp_path / "nope.geojson")


def test_polygon_from_geojson_unreadable_json(tmp_path):
    p = tmp_path / "mask.geojson"
    p.write_text("{not json")
    with pytest.raises(UsageError, match="could not read --mask-geojson"):
        utils.polygon_from_geojson(p)


def test_polygon_from_geojson_no_usable_geometry(tmp_path):
    payload = {"type": "FeatureCollection", "features": []}
    with pytest.raises(UsageError, match="has no usable geometry"):
        utils.polygon_from_geojson(_polygon_from_geojson_write(tmp_path, payload))


def test_polygon_from_geojson_top_level_array_is_no_usable_geometry(tmp_path):
    with pytest.raises(UsageError, match="has no usable geometry"):
        utils.polygon_from_geojson(
            _polygon_from_geojson_write(tmp_path, [_polygon_from_geojson_square])
        )


def test_polygon_from_geojson_top_level_scalar_is_no_usable_geometry(tmp_path):
    with pytest.raises(UsageError, match="has no usable geometry"):
        utils.polygon_from_geojson(_polygon_from_geojson_write(tmp_path, "Polygon"))


def test_polygon_from_geojson_flag_names_the_source_flag(tmp_path):
    with pytest.raises(UsageError, match="--clip-geojson file not found"):
        utils.polygon_from_geojson(tmp_path / "nope.geojson", flag="--clip-geojson")


def test_polygon_from_geojson_non_list_features_value_raises_usage_error(tmp_path):
    payload = {"type": "FeatureCollection", "features": {"not": "a list"}}
    with pytest.raises(UsageError, match="'features' is not a list"):
        utils.polygon_from_geojson(_polygon_from_geojson_write(tmp_path, payload))


def test_polygon_from_geojson_non_object_feature_entry_raises_usage_error(tmp_path):
    payload = {"type": "FeatureCollection", "features": ["not-an-object"]}
    with pytest.raises(UsageError, match="a feature is not a JSON object"):
        utils.polygon_from_geojson(_polygon_from_geojson_write(tmp_path, payload))


def test_polygon_from_geojson_unknown_geometry_type_raises_usage_error_naming_the_flag(tmp_path):
    payload = {"type": "Bogus", "coordinates": [0, 0]}
    with pytest.raises(UsageError, match="--mask-geojson.*has no usable geometry"):
        utils.polygon_from_geojson(_polygon_from_geojson_write(tmp_path, payload))


def test_polygon_from_geojson_geometry_missing_coordinates_raises_usage_error(tmp_path):
    payload = {"type": "Feature", "geometry": {"type": "Point"}}
    with pytest.raises(UsageError, match="has no usable geometry"):
        utils.polygon_from_geojson(_polygon_from_geojson_write(tmp_path, payload))


def test_polygon_from_geojson_malformed_coordinates_raise_usage_error_not_a_traceback(tmp_path):
    payload = {"type": "Point", "coordinates": "nope"}
    with pytest.raises(UsageError, match="has no usable geometry"):
        utils.polygon_from_geojson(_polygon_from_geojson_write(tmp_path, payload))


def test_normalize_longitude_0_360_axis_wraps_and_sorts():
    ds = make_gridded(lons=(0.0, 90.0, 180.0, 270.0))
    out = utils.normalize_longitude(ds)
    assert list(out["longitude"].values) == [-180.0, -90.0, 0.0, 90.0]


def test_normalize_longitude_values_follow_their_cells():
    ds = make_gridded(lons=(0.0, 90.0, 180.0, 270.0))
    ds["precip"][:, :, 3] = 7.0
    out = utils.normalize_longitude(ds)
    assert float(out["precip"].sel(longitude=-90.0).isel(time=0, latitude=0)) == 7.0


def test_normalize_longitude_already_normalized_axis_is_unchanged():
    ds = make_gridded(lons=(-90.0, 0.0, 90.0))
    out = utils.normalize_longitude(ds)
    assert list(out["longitude"].values) == [-90.0, 0.0, 90.0]


def test_normalize_longitude_custom_dim_name():
    ds = make_gridded(lons=(0.0, 270.0)).rename({"longitude": "lon"})
    out = utils.normalize_longitude(ds, lon_dim="lon")
    assert list(out["lon"].values) == [-90.0, 0.0]


def test_normalize_longitude_longitude_attrs_preserved_across_the_wrap():
    ds = make_gridded(lons=(0.0, 90.0, 180.0, 270.0))
    ds["longitude"].attrs = {"standard_name": "longitude", "units": "degrees_east", "axis": "X"}
    out = utils.normalize_longitude(ds)
    assert out["longitude"].attrs == {
        "standard_name": "longitude",
        "units": "degrees_east",
        "axis": "X",
    }


def test_normalize_longitude_duplicate_endpoint_is_dropped_and_axis_stays_sorted():
    ds = make_gridded(lons=(0.0, 90.0, 180.0, 270.0, 360.0))
    out = utils.normalize_longitude(ds)
    lons = list(out["longitude"].values)
    assert lons == [-180.0, -90.0, 0.0, 90.0]
    assert len(lons) == len(set(lons))


def test_normalize_longitude_duplicate_drop_keeps_the_first_occurrence():
    ds = make_gridded(lons=(0.0, 90.0, 180.0, 270.0, 360.0))
    ds["precip"][:, :, 0] = 5.0
    ds["precip"][:, :, 4] = 9.0
    out = utils.normalize_longitude(ds)
    assert float(out["precip"].sel(longitude=0.0).isel(time=0, latitude=0)) == 5.0


def test_normalize_longitude_accepts_dataarray():
    ds = make_gridded(lons=(0.0, 90.0, 180.0, 270.0))
    out = utils.normalize_longitude(ds["precip"])
    assert list(out["longitude"].values) == [-180.0, -90.0, 0.0, 90.0]


def test_ensure_normalized_longitude_is_noop_when_already_180():
    ds = make_gridded(lons=(-90.0, 0.0, 90.0))
    assert utils.ensure_normalized_longitude(ds) is ds


def test_ensure_normalized_longitude_wraps_0_360():
    ds = make_gridded(lons=(0.0, 90.0, 180.0, 270.0))
    out = utils.ensure_normalized_longitude(ds)
    assert list(out["longitude"].values) == [-180.0, -90.0, 0.0, 90.0]


def test_ensure_normalized_longitude_wraps_station_coord():
    ds = make_station(n_station=2)
    ds = ds.assign_coords(longitude=("station_id", [10.0, 270.0]))
    out = utils.ensure_normalized_longitude(ds, lon_dim="longitude")
    assert list(out["longitude"].values) == [10.0, -90.0]


@pytest.mark.parametrize(
    "text",
    [
        "429 Client Error: Too Many Requests",
        "API request failed with status code 500",
        "502 Bad Gateway",
        "503 Service Unavailable",
        "504 Gateway Timeout",
        "read timed out",
        "ConnectTimeout: request timeout",
        "Connection reset by peer",
    ],
)
def test_is_transient_transient_markers(text):
    assert utils.is_transient(Exception(text)) is True


@pytest.mark.parametrize("text", ["404 Not Found", "401 Unauthorized", "invalid parameter", ""])
def test_is_transient_non_transient(text):
    assert utils.is_transient(Exception(text)) is False


def test_is_transient_case_insensitive():
    assert utils.is_transient(Exception("Timed Out while reading")) is True


@pytest.mark.parametrize(
    "text",
    [
        "order 14290 failed",
        "processed 50000 records",
        "HTTPSConnectionPool(host='x'): Max retries exceeded (404 Not Found)",
    ],
)
def test_is_transient_permanent_lookalikes_are_not_transient(text):
    assert utils.is_transient(Exception(text)) is False


@pytest.mark.parametrize(
    "text",
    [
        "Failed to establish a new connection: Connection refused",
        "('Connection aborted.', RemoteDisconnected())",
        "HTTPSConnectionPool(host='x'): Read timed out",
    ],
)
def test_is_transient_genuine_connection_and_timeout_failures_are_transient(text):
    assert utils.is_transient(Exception(text)) is True


def test_require_env_returns_values_in_order(monkeypatch):
    monkeypatch.setenv("WSC_TEST_USER", "u")
    monkeypatch.setenv("WSC_TEST_PASS", "p")
    assert utils.require_env("WSC_TEST_USER", "WSC_TEST_PASS") == ("u", "p")


def test_require_env_default_message_names_only_the_missing(monkeypatch):
    monkeypatch.setenv("WSC_TEST_USER", "u")
    monkeypatch.delenv("WSC_TEST_PASS", raising=False)
    with pytest.raises(UsageError) as excinfo:
        utils.require_env("WSC_TEST_USER", "WSC_TEST_PASS")
    assert str(excinfo.value) == "missing required env var(s): WSC_TEST_PASS"


def test_require_env_all_missing_listed_in_order(monkeypatch):
    monkeypatch.delenv("WSC_TEST_USER", raising=False)
    monkeypatch.delenv("WSC_TEST_PASS", raising=False)
    with pytest.raises(UsageError) as excinfo:
        utils.require_env("WSC_TEST_USER", "WSC_TEST_PASS")
    assert str(excinfo.value) == "missing required env var(s): WSC_TEST_USER, WSC_TEST_PASS"


def test_require_env_empty_value_counts_as_missing(monkeypatch):
    monkeypatch.setenv("WSC_TEST_USER", "")
    with pytest.raises(UsageError, match="WSC_TEST_USER"):
        utils.require_env("WSC_TEST_USER")


def test_require_env_message_override(monkeypatch):
    monkeypatch.delenv("WSC_TEST_USER", raising=False)
    with pytest.raises(UsageError) as excinfo:
        utils.require_env("WSC_TEST_USER", message="WSC_TEST_USER and WSC_TEST_PASS must be set.")
    assert str(excinfo.value) == "WSC_TEST_USER and WSC_TEST_PASS must be set."


def test_require_env_usage_error_exits_2(monkeypatch):
    monkeypatch.delenv("WSC_TEST_USER", raising=False)
    with pytest.raises(UsageError) as excinfo:
        utils.require_env("WSC_TEST_USER")
    assert excinfo.value.exit_code == 2


def test_grid_spacing_median_spacing():
    assert utils.grid_spacing([0.0, 0.25, 0.5, 0.75]) == pytest.approx(0.25)


def test_spacing_is_finer_rejects_true_halving():
    assert utils.spacing_is_finer(0.025, 0.05)
    assert utils.spacing_is_finer(0.05, 0.1)


def test_spacing_is_finer_allows_lateral_and_float_noise():
    assert not utils.spacing_is_finer(0.05, 0.05)
    assert not utils.spacing_is_finer(0.05, 0.05000000000000044)
    assert not utils.spacing_is_finer(0.0499, 0.05)
    assert not utils.spacing_is_finer(0.0501, 0.05)
    assert not utils.spacing_is_finer(0.05, 0.0499)


def test_pick_time_dim_prefers_time():
    ds = make_gridded()
    assert utils.pick_time_dim(ds) == "time"


def test_pick_time_dim_override():
    ds = make_gridded()
    assert utils.pick_time_dim(ds, "time") == "time"


def test_pick_time_dim_missing_override():
    with pytest.raises(UsageError, match="not in dims"):
        utils.pick_time_dim(make_gridded(), "nope")


def test_dataset_label_from_source_attr():
    ds = make_gridded()
    ds.attrs["weather_skills_source"] = "ecmwf-s2s"
    assert utils.dataset_label(ds, "fallback") == "ecmwf-s2s"


def test_dataset_label_fallback():
    assert utils.dataset_label(make_gridded(), "input 1") == "input 1"


def test_apply_write_encoding_time_and_fill():
    import numpy as np

    ds = make_gridded()
    utils.apply_write_encoding(
        ds, time_units="days since 1970-01-01", time_calendar="standard", fills={"precip": np.nan}
    )
    assert ds["time"].encoding["units"] == "days since 1970-01-01"
    assert ds["time"].encoding["calendar"] == "standard"
    assert np.isnan(ds["precip"].encoding["_FillValue"])


def test_verify_cf_decode_ok_on_stamped_grid():
    from weather_skills_core.cf import stamp_cf_attrs

    ds = stamp_cf_attrs(make_gridded())
    utils.verify_cf_decode(ds)


def test_verify_cf_decode_raises_when_unstamped():
    with pytest.raises(DataError, match="did not resolve"):
        utils.verify_cf_decode(make_gridded())


def test_normalize_step_coord_casts_timedelta_us_to_ns():
    import numpy as np
    import xarray as xr

    steps = (np.arange(1, 4) * np.timedelta64(7, "D")).astype("timedelta64[us]")
    ds = xr.Dataset({"tp": (("step",), np.ones(3), {"units": "mm"})}, coords={"step": steps})
    out = utils.normalize_step_coord(ds)
    assert out["step"].dtype == np.dtype("timedelta64[ns]")
    assert out["step"].values[0] == np.timedelta64(7, "D")


def test_normalize_latlon_coords_rounds_and_casts_float32():
    import numpy as np

    ds = make_gridded(
        lats=(5.9749990996248385, -1.2750010213),
        lons=(33.0, 36.800000000000004),
    )
    out = utils.normalize_latlon_coords(ds)
    assert out["latitude"].dtype == np.float32
    assert out["longitude"].dtype == np.float32
    np.testing.assert_array_equal(
        out["latitude"].values,
        np.round(np.array([5.9749990996248385, -1.2750010213]), 5).astype(np.float32),
    )
    np.testing.assert_array_equal(
        out["longitude"].values,
        np.round(np.array([33.0, 36.800000000000004]), 5).astype(np.float32),
    )
    assert out["latitude"].attrs == ds["latitude"].attrs


def test_normalize_latlon_coords_keeps_half_cell_offset():
    import numpy as np

    kenya = make_gridded(lats=(-1.275,), lons=(36.80,))
    chirps = make_gridded(lats=(-1.275,), lons=(36.825,))
    k = utils.normalize_latlon_coords(kenya)
    c = utils.normalize_latlon_coords(chirps)
    np.testing.assert_array_equal(k["latitude"].values, c["latitude"].values)
    assert not np.array_equal(k["longitude"].values, c["longitude"].values)


def test_normalize_latlon_coords_point_obs_and_aliases():
    import numpy as np

    ds = make_station()
    ds = ds.assign_coords(latitude=ds["latitude"].values + 1e-8)
    out = utils.normalize_latlon_coords(ds)
    assert out["latitude"].dtype == np.float32
    assert out["longitude"].dtype == np.float32

    aliased = make_gridded().rename({"latitude": "lat", "longitude": "lon"})
    out_a = utils.normalize_latlon_coords(aliased)
    assert out_a["lat"].dtype == np.float32
    assert out_a["lon"].dtype == np.float32


def test_normalize_latlon_coords_idempotent():
    first = utils.normalize_latlon_coords(make_gridded())
    second = utils.normalize_latlon_coords(first)
    assert second is first


def test_normalize_step_coord_noop_when_already_ns():
    import numpy as np
    import xarray as xr

    steps = np.arange(1, 4) * np.timedelta64(1, "D")
    ds = xr.Dataset(
        {"tp": (("step",), np.ones(3))}, coords={"step": steps.astype("timedelta64[ns]")}
    )
    out = utils.normalize_step_coord(ds)
    assert out["step"].dtype == np.dtype("timedelta64[ns]")


def test_clip_by_geometry_grid_drop():
    from shapely.geometry import box

    ds = make_gridded()
    out = utils.clip_by_geometry(ds, box(10.5, 0.5, 12.5, 2.5), drop=True)
    assert list(out.latitude.values) == [1.0, 2.0]
    assert list(out.longitude.values) == [11.0, 12.0]


def test_clip_by_geometry_empty_raises():
    from shapely.geometry import box

    with pytest.raises(DataError, match="no grid cells"):
        utils.clip_by_geometry(make_gridded(), box(50, 50, 51, 51), drop=True)


def test_clip_by_geometry_stations_drop():
    from shapely.geometry import box

    ds = make_station()
    out = utils.clip_by_geometry(ds, box(36.5, -0.5, 37.5, 0.5), drop=True)
    assert list(out.station_id.values) == ["TA0001"]


def test_clip_by_geometry_stations_empty_raises():
    from shapely.geometry import box

    with pytest.raises(DataError, match="no stations"):
        utils.clip_by_geometry(make_station(), box(50, 50, 51, 51), drop=True)


def test_stride_dates_daily():
    times = utils.stride_dates("2026-01-01", "2026-01-03", stride="day")
    assert len(times) == 3


def test_stride_dates_weekdays():
    times = utils.stride_dates("2026-01-05", "2026-01-11", stride="Monday")
    assert len(times) == 1
    assert str(times[0])[:10] == "2026-01-05"


def test_roll_and_agg_mean_window():
    ds = make_gridded(n_time=5, fill=2.0)
    out = utils.roll_and_agg(ds, 3, "time", method="mean", align="right")
    assert out.sizes["time"] == 3
    np.testing.assert_allclose(out["precip"].values, 2.0)


def test_roll_and_agg_rejects_sum():
    with pytest.raises(UsageError, match="unsupported rolling method"):
        utils.roll_and_agg(make_gridded(n_time=5), 3, "time", method="sum")
