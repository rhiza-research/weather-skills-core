import numpy as np
import pytest
from conftest import make_forecast, make_gridded, make_station, make_vertical_forecast

from weather_skills_core import standard_dataset as dataset
from weather_skills_core.errors import UsageError


def test_detect_spatial_dims_canonical_names():
    assert dataset.detect_spatial_dims(make_gridded()) == ("latitude", "longitude")


def test_detect_spatial_dims_alias_names():
    ds = make_gridded().rename({"latitude": "lat", "longitude": "lon"})
    assert dataset.detect_spatial_dims(ds) == ("lat", "lon")


def test_detect_spatial_dims_override_wins():
    ds = make_gridded().rename({"latitude": "yy", "longitude": "xx"})
    assert dataset.detect_spatial_dims(ds, "yy,xx") == ("yy", "xx")


def test_detect_spatial_dims_override_names_must_exist():
    with pytest.raises(UsageError, match="not in dataset dims"):
        dataset.detect_spatial_dims(make_gridded(), "a,b")


def test_detect_spatial_dims_override_must_be_two_names():
    with pytest.raises(UsageError, match="LAT,LON"):
        dataset.detect_spatial_dims(make_gridded(), "onlyone")


def test_detect_spatial_dims_unidentifiable():
    ds = make_gridded().rename({"latitude": "row", "longitude": "col"})
    with pytest.raises(UsageError, match="Pass --dims"):
        dataset.detect_spatial_dims(ds)


def test_detect_spatial_dims_cf_attrs_resolve_nonstandard_names():
    ds = make_gridded().rename({"latitude": "yy", "longitude": "xx"})
    ds["yy"].attrs.update(standard_name="latitude", units="degrees_north")
    ds["xx"].attrs.update(standard_name="longitude", units="degrees_east")
    assert dataset.detect_spatial_dims(ds) == ("yy", "xx")


def test_detect_time_dim_literal_time():
    assert dataset.detect_time_dim(make_gridded()) == "time"


def test_detect_time_dim_override():
    ds = make_gridded().rename({"time": "t"})
    assert dataset.detect_time_dim(ds, "t") == "t"


def test_detect_time_dim_override_must_exist():
    with pytest.raises(UsageError, match="not in dataset dims"):
        dataset.detect_time_dim(make_gridded(), "t")


def test_detect_time_dim_unidentifiable():
    ds = make_gridded().rename({"time": "record"}).drop_vars("record")
    with pytest.raises(UsageError, match="Pass --time-dim"):
        dataset.detect_time_dim(ds)


def test_detect_type_gridded():
    assert dataset.detect_type(make_gridded()) == "observations"


def test_detect_type_forecast():
    assert dataset.detect_type(make_forecast(n_number=0)) == "forecast"
    assert dataset.detect_type(make_forecast()) == "ensemble_forecast"


def test_detect_type_vertical_forecast():
    ds = make_vertical_forecast()
    assert dataset.detect_type(ds) == "vertical_forecast"
    assert dataset.validate_input(ds, "vertical_forecast", "in.zarr") == "vertical_forecast"
    with pytest.raises(UsageError, match="vertical"):
        dataset.validate_input(make_forecast(n_number=0), "vertical_forecast", "in.zarr")


def test_detect_type_point_obs():
    assert dataset.detect_type(make_station()) == "point_obs"


def test_detect_type_station_alias_same_as_point_obs():
    assert dataset.parse_alternatives("station") == dataset.parse_alternatives("point_obs")


def test_detect_type_step_with_time_dim_is_not_forecast():
    ds = make_forecast()
    ds = ds.drop_vars("time")
    ds = ds.expand_dims(time=np.array(["2026-01-01"], dtype="datetime64[ns]"))
    assert dataset.detect_type(ds) == "observations"


def test_parse_io_spec_canonical_observations():
    assert dataset.parse_alternatives("observations") == (frozenset({"lat", "lon", "time"}),)


def test_parse_io_spec_obs_aliases_same_as_observations():
    expected = dataset.parse_alternatives("observations")
    for alias in ("obs", "analysis", "retrieval", "field", "data"):
        assert dataset.parse_alternatives(alias) == expected


def test_parse_io_spec_or_list():
    alts = dataset.parse_alternatives(["forecast", "ensemble_forecast"])
    assert len(alts) == 2
    assert frozenset({"lat", "lon", "init_time", "prediction_timedelta"}) in alts
    assert frozenset({"lat", "lon", "init_time", "prediction_timedelta", "member"}) in alts


def test_parse_io_spec_and_tuple():
    assert dataset.parse_alternatives(("lat", "lon", "time")) == (
        frozenset({"lat", "lon", "time"}),
    )


def test_parse_io_spec_spatial_alias():
    assert dataset.parse_alternatives("spatial") == (frozenset({"lat", "lon"}),)
    assert dataset.parse_alternatives("space") == (frozenset({"lat", "lon"}),)


def test_parse_io_spec_vertical_forecast():
    assert dataset.parse_alternatives("vertical_forecast") == (
        frozenset({"lat", "lon", "init_time", "prediction_timedelta", "vertical"}),
    )
    assert dataset.parse_alternatives("vertical") == (frozenset({"vertical"}),)


def test_parse_io_spec_single_dim():
    assert dataset.parse_alternatives("time") == (frozenset({"time"}),)
    assert dataset.parse_alternatives("lat") == (frozenset({"lat"}),)


def test_validate_input_matching_type_passes():
    assert dataset.validate_input(make_gridded(), "observations", "in.zarr") == "observations"


def test_validate_input_legacy_data_alias():
    assert dataset.validate_input(make_gridded(), "data", "in.zarr") == "observations"


def test_validate_input_list_of_alternatives():
    assert (
        dataset.validate_input(make_forecast(n_number=0), ["observations", "forecast"], "in.zarr")
        == "forecast"
    )


def test_validate_input_or_canonical_list():
    assert (
        dataset.validate_input(make_forecast(), ["forecast", "ensemble_forecast"], "in.zarr")
        == "ensemble_forecast"
    )


def test_validate_input_dim_only_spatial():
    assert dataset.validate_input(make_gridded(), "spatial", "in.zarr") == "observations"
    assert dataset.validate_input(make_gridded(), "space", "in.zarr") == "observations"


def test_validate_input_gridded_rejected_when_forecast_expected():
    with pytest.raises(UsageError, match="prediction_timedelta|init_time"):
        dataset.validate_input(make_gridded(), "forecast", "in.zarr")


def test_validate_input_forecast_rejected_when_point_obs_expected():
    with pytest.raises(UsageError, match="point_id"):
        dataset.validate_input(make_forecast(), "point_obs", "in.zarr")


def test_validate_input_error_names_the_input():
    with pytest.raises(UsageError, match="my/path.zarr"):
        dataset.validate_input(make_gridded(), "point_obs", "my/path.zarr")


def test_validate_input_point_obs_missing_latitude_coord():
    ds = make_station().drop_vars("latitude")
    with pytest.raises(UsageError, match="point_id"):
        dataset.validate_input(ds, "point_obs", "in.zarr")


def test_validate_input_point_obs_latitude_on_wrong_dim():
    ds = make_station()
    ds = ds.assign_coords(latitude=("time", np.zeros(ds.sizes["time"])))
    with pytest.raises(UsageError, match="point_id"):
        dataset.validate_input(ds, "point_obs", "in.zarr")


def test_validate_input_point_obs_accepts_lat_lon_aliases():
    ds = make_station().rename({"latitude": "lat", "longitude": "lon"})
    assert dataset.validate_input(ds, "point_obs", "in.zarr") == "point_obs"


def test_validate_input_unknown_declared_type_is_a_programming_error():
    with pytest.raises(ValueError, match="unknown type or dimension"):
        dataset.validate_input(make_gridded(), "grid", "in.zarr")


def test_validate_input_dims_override_validates_undetectable_gridded():
    ds = make_gridded().rename({"latitude": "yy", "longitude": "xx"})
    with pytest.raises(UsageError, match="lat|lon"):
        dataset.validate_input(ds, "observations", "in.zarr")
    assert dataset.validate_input(ds, "observations", "in.zarr", dims="yy,xx") == "observations"


def test_validate_input_dims_override_names_must_exist():
    ds = make_gridded().rename({"latitude": "yy", "longitude": "xx"})
    with pytest.raises(UsageError, match="lat|lon"):
        dataset.validate_input(ds, "observations", "in.zarr", dims="a,b")


def test_validate_input_time_dim_override_must_exist():
    with pytest.raises(UsageError, match="time"):
        dataset.validate_input(make_gridded(), "observations", "in.zarr", time_dim="t")
    ds = make_gridded().rename({"time": "t"})
    assert dataset.validate_input(ds, "observations", "in.zarr", time_dim="t") == "observations"


def test_validate_input_any_accepts_all():
    for ds in (make_gridded(), make_forecast(), make_station()):
        dataset.validate_input(ds, "any", "in.zarr")
