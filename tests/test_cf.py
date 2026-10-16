import numpy as np
import pytest
import xarray as xr
from conftest import make_gridded, make_station

from weather_skills_core import cf as dataset
from weather_skills_core.errors import DataError


def test_stamp_cf_attrs_canonical_names():
    """Canonical lat/lon/time get CF attrs."""
    ds = dataset.stamp_cf_attrs(make_gridded())
    assert ds["latitude"].attrs == {
        "standard_name": "latitude",
        "units": "degrees_north",
        "axis": "Y",
    }
    assert ds["longitude"].attrs == {
        "standard_name": "longitude",
        "units": "degrees_east",
        "axis": "X",
    }
    assert ds["time"].attrs == {"standard_name": "time", "axis": "T"}


def test_stamp_cf_attrs_alias_names():
    """lat/lon aliases get CF standard_names."""
    ds = dataset.stamp_cf_attrs(make_gridded().rename({"latitude": "lat", "longitude": "lon"}))
    assert ds["lat"].attrs["standard_name"] == "latitude"
    assert ds["lon"].attrs["standard_name"] == "longitude"


def test_stamp_cf_attrs_setdefault_preserves_source_values():
    """Existing CF values are not overwritten."""
    ds = make_gridded()
    ds["latitude"].attrs["units"] = "degree_north"
    ds["time"].attrs["standard_name"] = "forecast_reference_time"
    dataset.stamp_cf_attrs(ds)
    assert ds["latitude"].attrs["units"] == "degree_north"
    assert ds["latitude"].attrs["axis"] == "Y"
    assert ds["time"].attrs["standard_name"] == "forecast_reference_time"


def test_stamp_cf_attrs_missing_coords_are_skipped():
    """Unknown dim names are left unstamped."""
    ds = make_gridded().rename({"latitude": "row", "longitude": "col"}).drop_vars("time")
    out = dataset.stamp_cf_attrs(ds)
    assert out["row"].attrs == {}
    assert out["col"].attrs == {}


def test_stamp_cf_attrs_returns_dataset():
    """stamp_cf_attrs returns the same dataset."""
    ds = make_gridded()
    assert dataset.stamp_cf_attrs(ds) is ds


def test_stamp_cf_coords_overwrites_prior_values():
    """stamp_cf_coords overwrites wrong CF attrs."""
    ds = make_gridded()
    ds["latitude"].attrs.update(standard_name="wrong", units="wrong", axis="Z")
    dataset.stamp_cf_coords(ds)
    assert ds["latitude"].attrs == {
        "standard_name": "latitude",
        "units": "degrees_north",
        "axis": "Y",
    }
    assert ds["longitude"].attrs == {
        "standard_name": "longitude",
        "units": "degrees_east",
        "axis": "X",
    }


def test_stamp_cf_coords_time_gets_no_units():
    """Time is stamped without units."""
    ds = dataset.stamp_cf_coords(make_gridded())
    assert ds["time"].attrs == {"standard_name": "time", "axis": "T"}


def test_stamp_cf_coords_long_names_applied_with_setdefault():
    """Caller long_names fill only missing long_name."""
    ds = make_gridded()
    ds["latitude"].attrs["long_name"] = "source latitude"
    dataset.stamp_cf_coords(
        ds, long_names={"latitude": "Latitude", "longitude": "Longitude", "time": "Time"}
    )
    assert ds["latitude"].attrs["long_name"] == "source latitude"
    assert ds["longitude"].attrs["long_name"] == "Longitude"
    assert ds["time"].attrs["long_name"] == "Time"


def test_stamp_cf_coords_no_long_name_by_default():
    """long_name is not added unless requested."""
    ds = dataset.stamp_cf_coords(make_gridded())
    assert "long_name" not in ds["latitude"].attrs


def test_stamp_cf_coords_missing_coords_are_skipped():
    """Present coords still stamp when time is missing."""
    ds = make_gridded().drop_vars("time")
    out = dataset.stamp_cf_coords(ds)
    assert out["latitude"].attrs["axis"] == "Y"


def test_stamp_cf_coords_alias_names_are_not_stamped():
    # Only the canonical post-rename names are asserted; a fetcher stamps
    # after renaming to latitude/longitude.
    """Pre-rename aliases are not stamped."""
    ds = dataset.stamp_cf_coords(make_gridded().rename({"latitude": "lat"}))
    assert ds["lat"].attrs == {}


def test_stamp_cf_coords_returns_dataset():
    """stamp_cf_coords returns the same dataset."""
    ds = make_gridded()
    assert dataset.stamp_cf_coords(ds) is ds


def test_cf_dim_resolves_stamped_coord():
    """cf_dim finds a renamed coord by standard_name."""
    ds = make_gridded().rename({"latitude": "yy"})
    ds["yy"].attrs.update(standard_name="latitude", units="degrees_north")
    assert dataset.cf_dim(ds, "latitude") == "yy"


def test_cf_dim_unresolvable_returns_none():
    """An unstamped rename does not resolve."""
    ds = make_gridded().rename({"latitude": "yy"})
    assert dataset.cf_dim(ds, "latitude") is None


def test_cf_dim_works_on_dataarrays():
    """cf_dim works on a DataArray."""
    ds = dataset.stamp_cf_attrs(make_gridded())
    assert dataset.cf_dim(ds["precip"], "longitude") == "longitude"


def test_auto_variable_picks_the_first_data_var():
    """The first data var is chosen."""
    assert dataset.auto_variable(make_gridded()) == "precip"


def test_auto_variable_skips_grid_mapping_container_and_targets():
    """grid_mapping containers are skipped."""
    ds = make_gridded()
    ds["crs"] = xr.DataArray(0, attrs={"grid_mapping_name": "latitude_longitude"})
    ds["precip"].attrs["grid_mapping"] = "crs"
    assert dataset.auto_variable(ds) == "precip"
    # A var NAMED by another var's grid_mapping attr is skipped even
    # without its own grid_mapping_name attr.
    ds2 = make_gridded()
    ds2["other"] = xr.DataArray(0)
    ds2["precip"].attrs["grid_mapping"] = "other"
    assert dataset.auto_variable(ds2) == "precip"


def test_auto_variable_prefers_multidim_vars():
    """A multidimensional var wins over a leading scalar."""
    ds = make_gridded()
    ds = ds[["precip"]]
    ds["scalar_first"] = xr.DataArray(1.0)
    ds = ds[["scalar_first", "precip"]]
    assert dataset.auto_variable(ds) == "precip"


def test_auto_variable_falls_back_to_one_dim_candidate():
    """A 1-D series is used when it is the only candidate."""
    ds = make_gridded()
    ds["series"] = ("time", np.ones(ds.sizes["time"]))
    ds = ds[["series"]]
    assert dataset.auto_variable(ds) == "series"


def test_auto_variable_no_candidates_returns_none():
    """An empty dataset has no auto variable."""
    ds = make_gridded()[[]]
    assert dataset.auto_variable(ds) is None


def _stamped_dsg(ds=None, var_attrs=None):
    ds = ds if ds is not None else make_station()
    var_attrs = (
        var_attrs
        if var_attrs is not None
        else {
            "precip": {
                "standard_name": "lwe_thickness_of_precipitation_amount",
                "long_name": "daily precipitation total",
                "units": "mm",
                "cell_methods": "time: sum",
            }
        }
    )
    return dataset.stamp_cf_dsg(
        ds,
        var_attrs,
        station_id_long_name="GHCN station identifier",
        name_long_name="station name",
    )


def test_stamp_cf_dsg_coordinate_attrs():
    """DSG coords get CF and timeseries_id role."""
    ds = _stamped_dsg()
    assert ds["latitude"].attrs == {
        "standard_name": "latitude",
        "long_name": "station latitude",
        "units": "degrees_north",
        "axis": "Y",
    }
    assert ds["longitude"].attrs == {
        "standard_name": "longitude",
        "long_name": "station longitude",
        "units": "degrees_east",
        "axis": "X",
    }
    assert ds["time"].attrs == {"standard_name": "time", "long_name": "time", "axis": "T"}
    assert ds["station_id"].attrs == {
        "cf_role": "timeseries_id",
        "long_name": "GHCN station identifier",
    }


def test_stamp_cf_dsg_data_variable_attrs_follow_the_coordinates_attr():
    """coordinates is injected first on the data var."""
    ds = _stamped_dsg()
    assert ds["precip"].attrs == {
        "coordinates": "latitude longitude time",
        "standard_name": "lwe_thickness_of_precipitation_amount",
        "long_name": "daily precipitation total",
        "units": "mm",
        "cell_methods": "time: sum",
    }
    # The load-bearing DSG attr is injected first; the caller's attrs
    # follow in their own insertion order.
    assert list(ds["precip"].attrs) == [
        "coordinates",
        "standard_name",
        "long_name",
        "units",
        "cell_methods",
    ]


def test_stamp_cf_dsg_var_attrs_without_standard_name():
    # A variable whose unit family backs no CF standard_name entry is
    # stamped without one (units + long_name alone is CF-valid).
    """A var without standard_name is still stamped with coordinates."""
    ds = _stamped_dsg(
        var_attrs={"precip": {"units": "mm", "long_name": "precip", "cell_methods": "time: mean"}}
    )
    assert "standard_name" not in ds["precip"].attrs
    assert ds["precip"].attrs["coordinates"] == "latitude longitude time"


def test_stamp_cf_dsg_optional_name_coord():
    """An optional name coord gets long_name."""
    ds = make_station()
    ds = ds.assign_coords(name=("station_id", ["a", "b", "c"]))
    _stamped_dsg(ds=ds)
    assert ds["name"].attrs == {"long_name": "station name"}


def test_stamp_cf_dsg_name_absent_is_skipped():
    """No name coord is not an error."""
    ds = _stamped_dsg()
    assert "name" not in ds.variables


def test_stamp_cf_dsg_missing_var_attrs_entry_raises_keyerror():
    """Missing var_attrs for the data var is KeyError."""
    with pytest.raises(KeyError):
        _stamped_dsg(var_attrs={})


def test_stamp_cf_dsg_returns_dataset():
    """stamp_cf_dsg returns the same dataset."""
    ds = make_station()
    assert _stamped_dsg(ds=ds) is ds


def test_verify_cf_dsg_stamped_dataset_passes():
    """A stamped station cube verifies."""
    ds = make_station()
    dataset.stamp_cf_dsg(
        ds,
        {"precip": {"units": "mm", "long_name": "precip"}},
        station_id_long_name="id",
        name_long_name="name",
    )
    dataset.verify_cf_dsg(ds)


def test_verify_cf_dsg_unstamped_dataset_lists_every_problem():
    """Unstamped DSG lists every missing CF axis/role."""
    ds = make_station()
    ds = ds.rename({"latitude": "row", "longitude": "col", "time": "record"})
    with pytest.raises(DataError) as excinfo:
        dataset.verify_cf_dsg(ds)
    message = str(excinfo.value)
    assert message.startswith("CF-1.13 DSG verification failed before write:")
    assert "cf_role timeseries_id did not resolve to station_id" in message
    for name in ("latitude", "longitude", "time"):
        assert f"cf-xarray could not resolve the {name} coordinate" in message


def test_verify_cf_dsg_missing_cf_role_alone():
    """Dropping cf_role alone fails verification."""
    ds = make_station()
    dataset.stamp_cf_dsg(
        ds,
        {"precip": {"units": "mm"}},
        station_id_long_name="id",
        name_long_name="name",
    )
    del ds["station_id"].attrs["cf_role"]
    with pytest.raises(DataError, match="timeseries_id did not resolve"):
        dataset.verify_cf_dsg(ds)


@pytest.fixture
def _require_pint():
    pytest.importorskip("pint")
    pytest.importorskip("pint_xarray")


@pytest.mark.parametrize("units", ["mm", "degC", "kg m-3", "mm day-1", "1"])
def test_udunits_error_valid_units_return_none(units, _require_pint):
    """Known unit strings are valid."""
    assert dataset.udunits_error(units) is None


def test_udunits_error_invalid_units_return_the_exception(_require_pint):
    """Garbage units return UndefinedUnitError."""
    from pint import UndefinedUnitError

    exc = dataset.udunits_error("definitely ! not a unit")
    assert isinstance(exc, UndefinedUnitError)
    assert "definitely" in str(exc)


def test_udunits_error_blank_units_pass_through(_require_pint):
    # None / "" pass through without raising; rejecting blanks is the caller's guard.
    """None and empty string are not parse errors."""
    assert dataset.udunits_error(None) is None
    assert dataset.udunits_error("") is None


def test_udunits_error_catch_widens_the_converted_failures(_require_pint):
    """catch= controls whether pint errors are converted or raised."""
    from pint import UndefinedUnitError

    class Boom(Exception):
        pass

    # Default catch converts pint parse errors; a narrow catch re-raises.
    assert dataset.udunits_error("degC", catch=(Exception,)) is None
    exc = dataset.udunits_error("definitely ! not a unit", catch=(Exception,))
    assert isinstance(exc, UndefinedUnitError)
    with pytest.raises(UndefinedUnitError):
        dataset.udunits_error("definitely ! not a unit", catch=(Boom,))


def test_cf_axes_missing_all_resolved():
    """A fully stamped grid misses no axes."""
    ds = dataset.stamp_cf_attrs(make_gridded())
    assert dataset.cf_axes_missing(ds) == []


def test_cf_axes_missing_partially_stamped_dataset_misses_x_and_y():
    # Axis resolution keys on the CF attrs (an unrenamed bare `time` name
    # resolves the "time" coordinate, not the "T" axis), so only the
    # stamped coord resolves.
    """Only a T axis attr leaves X and Y missing."""
    ds = make_gridded().rename({"latitude": "row", "longitude": "col"})
    ds["time"].attrs["axis"] = "T"
    assert dataset.cf_axes_missing(ds) == ["X", "Y"]


def test_cf_axes_missing_all_missing_on_unstamped_dataset():
    """An unstamped grid misses X, Y, and T."""
    assert dataset.cf_axes_missing(make_gridded()) == ["X", "Y", "T"]


def test_cf_axes_missing_custom_axes():
    """axes= limits which missing axes are reported."""
    ds = make_gridded()
    ds["time"].attrs["axis"] = "T"
    assert dataset.cf_axes_missing(ds, axes=("T",)) == []
    assert dataset.cf_axes_missing(ds, axes=("Y",)) == ["Y"]


def _with_crs_and_bounds():
    ds = make_gridded(n_time=2)
    ds = ds.assign_coords(
        spatial_ref=0,
        time_bounds=(("time", "nv"), np.zeros((2, 2), dtype="datetime64[ns]")),
        valid_time=("time", ds["time"].values),
    )
    ds["time"].attrs["bounds"] = "time_bounds"
    ds["precip"].attrs.update(grid_mapping="spatial_ref", coordinates="valid_time spatial_ref")
    return ds


def test_bounds_of_finds_present_bounds_only():
    ds = _with_crs_and_bounds()
    assert dataset.bounds_of(ds, ["time", "latitude", "missing"]) == {"time": "time_bounds"}


def test_rename_cf_refs_rewrites_attrs_and_encoding():
    ds = _with_crs_and_bounds()
    ds["precip"].encoding["coordinates"] = "valid_time"
    ds = ds.rename({"time_bounds": "time_bnds", "valid_time": "vt"})
    dataset.rename_cf_refs(ds, {"time_bounds": "time_bnds", "valid_time": "vt"})
    assert ds["time"].attrs["bounds"] == "time_bnds"
    assert ds["precip"].attrs["coordinates"] == "vt spatial_ref"
    assert ds["precip"].encoding["coordinates"] == "vt"


def test_rename_cf_refs_keeps_extended_grid_mapping_colons():
    ds = _with_crs_and_bounds()
    ds["precip"].attrs["grid_mapping"] = "spatial_ref: latitude longitude"
    dataset.rename_cf_refs(ds, {"latitude": "lat", "longitude": "lon"})
    assert ds["precip"].attrs["grid_mapping"] == "spatial_ref: lat lon"


def test_drop_cf_refs_removes_tokens_and_empty_attrs():
    ds = _with_crs_and_bounds().drop_vars(["spatial_ref", "valid_time"])
    dataset.drop_cf_refs(ds, ["spatial_ref", "valid_time"])
    assert "grid_mapping" not in ds["precip"].attrs
    assert "coordinates" not in ds["precip"].attrs


def test_drop_cf_refs_removes_extended_grid_mapping_whole():
    ds = _with_crs_and_bounds()
    ds["precip"].attrs["grid_mapping"] = "spatial_ref: latitude longitude"
    dataset.drop_cf_refs(ds, ["spatial_ref"])
    assert "grid_mapping" not in ds["precip"].attrs


def test_dangling_cf_refs_lists_missing_names():
    ds = _with_crs_and_bounds()
    attrs = {"coordinates": "valid_time gone", "grid_mapping": "spatial_ref", "units": "mm"}
    assert dataset.dangling_cf_refs(attrs, ds) == ["gone"]


def test_fill_missing_attrs_does_not_restore_dangling_refs():
    from weather_skills_core.standard_utils import fill_missing_data_var_attrs

    src = _with_crs_and_bounds()
    dst = src.drop_vars("spatial_ref").copy(deep=True)
    del dst["precip"].attrs["grid_mapping"]
    del dst["precip"].attrs["units"]
    out = fill_missing_data_var_attrs(src, dst)
    assert "grid_mapping" not in out["precip"].attrs
    assert out["precip"].attrs["units"] == "mm day-1"
