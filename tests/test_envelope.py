from typing import ClassVar

import numpy as np
import pytest
import xarray as xr
from conftest import make_forecast, make_gridded, make_series, make_station

from weather_skills_core import envelope, types
from weather_skills_core.errors import DataError, UsageError


class TestParseBbox:
    def test_valid(self):
        assert envelope.parse_bbox("1/2/3/4") == (1.0, 2.0, 3.0, 4.0)

    def test_negative_values(self):
        assert envelope.parse_bbox("-1/32/-5/42") == (-1.0, 32.0, -5.0, 42.0)

    @pytest.mark.parametrize("value", ["1/2/3", "1/2/3/4/5", "a/b/c/d", ""])
    def test_malformed(self, value):
        with pytest.raises(UsageError, match="N/W/S/E"):
            envelope.parse_bbox(value)


class TestDetectSpatialDims:
    def test_canonical_names(self):
        assert envelope.detect_spatial_dims(make_gridded()) == ("latitude", "longitude")

    def test_alias_names(self):
        ds = make_gridded().rename({"latitude": "lat", "longitude": "lon"})
        assert envelope.detect_spatial_dims(ds) == ("lat", "lon")

    def test_override_wins(self):
        ds = make_gridded().rename({"latitude": "yy", "longitude": "xx"})
        assert envelope.detect_spatial_dims(ds, "yy,xx") == ("yy", "xx")

    def test_override_names_must_exist(self):
        with pytest.raises(UsageError, match="not in dataset dims"):
            envelope.detect_spatial_dims(make_gridded(), "a,b")

    def test_override_must_be_two_names(self):
        with pytest.raises(UsageError, match="LAT,LON"):
            envelope.detect_spatial_dims(make_gridded(), "onlyone")

    def test_unidentifiable(self):
        ds = make_gridded().rename({"latitude": "row", "longitude": "col"})
        with pytest.raises(UsageError, match="Pass --dims"):
            envelope.detect_spatial_dims(ds)

    def test_cf_attrs_resolve_nonstandard_names(self):
        ds = make_gridded().rename({"latitude": "yy", "longitude": "xx"})
        ds["yy"].attrs.update(standard_name="latitude", units="degrees_north")
        ds["xx"].attrs.update(standard_name="longitude", units="degrees_east")
        assert envelope.detect_spatial_dims(ds) == ("yy", "xx")


class TestDetectTimeDim:
    def test_literal_time(self):
        assert envelope.detect_time_dim(make_gridded()) == "time"

    def test_override(self):
        ds = make_gridded().rename({"time": "t"})
        assert envelope.detect_time_dim(ds, "t") == "t"

    def test_override_must_exist(self):
        with pytest.raises(UsageError, match="not in dataset dims"):
            envelope.detect_time_dim(make_gridded(), "t")

    def test_unidentifiable(self):
        ds = make_gridded().rename({"time": "record"}).drop_vars("record")
        with pytest.raises(UsageError, match="Pass --time-dim"):
            envelope.detect_time_dim(ds)


class TestDetectType:
    def test_gridded(self):
        assert envelope.detect_type(make_gridded()) == envelope.GRIDDED

    def test_forecast(self):
        assert envelope.detect_type(make_forecast()) == envelope.FORECAST

    def test_station(self):
        assert envelope.detect_type(make_station()) == envelope.STATION

    def test_series(self):
        assert envelope.detect_type(make_series()) == envelope.SERIES

    def test_unidentifiable_grid_is_a_series_until_dims_names_it(self):
        ds = make_gridded().rename({"latitude": "yy", "longitude": "xx"})
        assert envelope.detect_type(ds) == envelope.SERIES
        assert envelope.detect_type(ds, "yy,xx") == envelope.GRIDDED

    def test_forecast_collapsed_over_lat_lon_is_a_series(self):
        # What `reduce --dim latitude --dim longitude` leaves of a forecast:
        # step and the scalar init time survive, the spatial axes do not.
        # Classifying that a forecast made it unreadable by every skill, since
        # the forecast structural check cannot pass without spatial dims.
        ds = xr.Dataset(
            {"tp": (("number", "step"), np.zeros((3, 8)))},
            coords={
                "number": np.arange(3),
                "step": np.arange(8),
                "time": np.datetime64("2026-01-01"),
            },
        )
        assert envelope.detect_type(ds) == envelope.SERIES
        assert envelope.validate_input(ds, types.ALL, "in") == envelope.SERIES

    def test_forecast_with_one_spatial_axis_selected_away_is_a_series(self):
        # What `select --dim latitude --index 0` leaves: the newly-scalar
        # latitude coord is dropped, so only one spatial axis remains.
        ds = make_forecast().isel(latitude=0).drop_vars("latitude")
        assert envelope.detect_type(ds) == envelope.SERIES
        assert envelope.validate_input(ds, types.ALL, "in") == envelope.SERIES

    def test_forecast_keeps_its_spatial_dims(self):
        assert envelope.detect_type(make_forecast()) == envelope.FORECAST
        ds = make_forecast().rename({"latitude": "yy", "longitude": "xx"})
        assert envelope.detect_type(ds) == envelope.SERIES
        assert envelope.detect_type(ds, "yy,xx") == envelope.FORECAST

    def test_step_with_time_dim_is_not_forecast(self):
        # A forecast envelope is a step dim plus a SCALAR time coord; a step
        # dim alongside a time dim does not classify as forecast.
        ds = make_forecast()
        ds = ds.drop_vars("time")
        ds = ds.expand_dims(time=np.array(["2026-01-01"], dtype="datetime64[ns]"))
        assert envelope.detect_type(ds) == envelope.GRIDDED


class TestValidateInput:
    def test_matching_type_passes(self):
        assert envelope.validate_input(make_gridded(), types.GRIDDED, "in.zarr") == types.GRIDDED

    def test_all_accepts_every_type(self):
        for ds, expected in (
            (make_gridded(), types.GRIDDED),
            (make_forecast(), types.FORECAST),
            (make_station(), types.STATION),
            (make_series(), types.SERIES),
        ):
            assert envelope.validate_input(ds, types.ALL, "in.zarr") == expected

    def test_list_of_alternatives(self):
        assert (
            envelope.validate_input(make_forecast(), [types.GRIDDED, types.FORECAST], "in.zarr")
            == types.FORECAST
        )

    def test_gridded_rejected_when_forecast_expected(self):
        with pytest.raises(UsageError, match="no 'step' dim"):
            envelope.validate_input(make_gridded(), types.FORECAST, "in.zarr")

    def test_forecast_rejected_when_station_expected(self):
        with pytest.raises(UsageError, match="no 'station_id' dim"):
            envelope.validate_input(make_forecast(), types.STATION, "in.zarr")

    def test_error_names_the_input(self):
        with pytest.raises(UsageError, match="my/path.zarr"):
            envelope.validate_input(make_gridded(), types.STATION, "my/path.zarr")

    def test_station_missing_latitude_coord(self):
        ds = make_station().drop_vars("latitude")
        with pytest.raises(UsageError, match="'latitude'"):
            envelope.validate_input(ds, types.STATION, "in.zarr")

    def test_station_latitude_on_wrong_dim(self):
        ds = make_station()
        ds = ds.assign_coords(latitude=("time", np.zeros(ds.sizes["time"])))
        with pytest.raises(UsageError, match="station_id"):
            envelope.validate_input(ds, types.STATION, "in.zarr")

    def test_unknown_declared_type_is_a_programming_error(self):
        with pytest.raises(ValueError, match="unknown envelope type"):
            envelope.validate_input(make_gridded(), "grid", "in.zarr")

    def test_dims_override_validates_undetectable_gridded(self):
        # Unnamed, the grid's axes are unidentifiable, so it does not classify
        # as gridded at all; naming them with --dims is what makes it one.
        ds = make_gridded().rename({"latitude": "yy", "longitude": "xx"})
        with pytest.raises(UsageError, match="pass --dims"):
            envelope.validate_input(ds, types.GRIDDED, "in.zarr")
        assert envelope.validate_input(ds, types.GRIDDED, "in.zarr", dims="yy,xx") == types.GRIDDED

    def test_dims_override_names_must_exist(self):
        ds = make_gridded().rename({"latitude": "yy", "longitude": "xx"})
        with pytest.raises(UsageError, match="not in dataset dims"):
            envelope.validate_input(ds, types.GRIDDED, "in.zarr", dims="a,b")

    def test_time_dim_override_must_exist(self):
        with pytest.raises(UsageError, match="not in dataset dims"):
            envelope.validate_input(make_gridded(), types.GRIDDED, "in.zarr", time_dim="t")
        ds = make_gridded().rename({"time": "t"})
        assert envelope.validate_input(ds, types.GRIDDED, "in.zarr", time_dim="t") == types.GRIDDED

    def test_forecast_with_unnamed_axes_is_accepted_as_the_series_it_is(self):
        # A forecast owes identifiable spatial dims, so a store without them
        # is a series -- read as one rather than rejected. --dims names the
        # axes and makes it a forecast again.
        ds = make_forecast().rename({"latitude": "yy", "longitude": "xx"})
        assert envelope.validate_input(ds, types.ALL, "in.zarr") == types.SERIES
        assert envelope.validate_input(ds, types.ALL, "in.zarr", dims="yy,xx") == types.FORECAST

    def test_all_still_runs_the_time_dim_check(self):
        with pytest.raises(UsageError, match="not in dataset dims"):
            envelope.validate_input(make_gridded(), types.ALL, "in.zarr", time_dim="t")

    def test_all_still_runs_the_station_coord_check(self):
        ds = make_station().drop_vars("latitude")
        with pytest.raises(UsageError, match="'latitude'"):
            envelope.validate_input(ds, types.ALL, "in.zarr")

    def test_series_rejected_when_gridded_expected(self):
        with pytest.raises(UsageError, match="no identifiable lat/lon coords"):
            envelope.validate_input(make_series(), types.GRIDDED, "in.zarr")

    def test_gridded_rejected_when_series_expected(self):
        with pytest.raises(UsageError, match="has lat/lon coords"):
            envelope.validate_input(make_gridded(), types.SERIES, "in.zarr")

    def test_series_still_runs_the_time_dim_check(self):
        with pytest.raises(UsageError, match="not in dataset dims"):
            envelope.validate_input(make_series(), types.SERIES, "in.zarr", time_dim="t")


class TestValidateType:
    def test_single_expected_type(self):
        assert envelope.validate_type(make_gridded(), types.GRIDDED) == types.GRIDDED

    def test_tuple_of_expected_types(self):
        assert envelope.validate_type(make_forecast(), types.ALL) == types.FORECAST

    def test_reference_dataset_form(self):
        assert envelope.validate_type(make_gridded(), make_gridded()) == types.GRIDDED

    def test_mismatch_is_a_data_error(self):
        with pytest.raises(DataError, match="expected a station envelope, got gridded"):
            envelope.validate_type(make_gridded(), types.STATION)

    def test_reference_dataset_mismatch_names_both_shapes(self):
        with pytest.raises(DataError, match="expected a gridded envelope, got series"):
            envelope.validate_type(make_series(), make_gridded())

    def test_unknown_expected_type_is_a_programming_error(self):
        with pytest.raises(ValueError, match="unknown envelope type"):
            envelope.validate_type(make_gridded(), "grid")


class TestValidateTypeDims:
    """The ``dims`` argument makes validate_type classify as the decorator does."""

    @staticmethod
    def renamed():
        return make_gridded().rename({"latitude": "yy", "longitude": "xx"})

    def test_override_classifies_both_sides(self):
        # The defect this closes: without the override, both sides of the
        # assertion classified series where the decorator saw gridded, so the
        # claim compared two shapes nobody else read.
        ds = self.renamed()
        assert envelope.validate_type(ds, ds) == types.SERIES
        assert envelope.validate_type(ds, ds, "yy,xx") == types.GRIDDED
        assert envelope.validate_type(ds, types.GRIDDED, "yy,xx") == types.GRIDDED

    def test_drift_on_overridden_axes_is_caught(self):
        ds = self.renamed()
        with pytest.raises(DataError, match="expected a gridded envelope, got series"):
            envelope.validate_type(ds.mean(dim=["yy", "xx"]), ds, "yy,xx")

    def test_names_absent_from_the_output_leave_it_a_series(self):
        # The output side of a --dims run: a store that collapsed the named
        # axes no longer has them, which is a shape, not a usage error.
        assert envelope.detect_type(make_series(), "yy,xx") == types.SERIES

    def test_override_does_not_hide_canonical_axes(self):
        # A transform that renames its axes to canonical names preserves the
        # shape: --dims named the input's axes, the output's own names
        # identify the output's.
        assert envelope.validate_type(make_gridded(), self.renamed(), "yy,xx") == types.GRIDDED


class TestBboxSubset:
    def test_ascending_latitude(self):
        ds = make_gridded(lats=(1.0, 2.0, 3.0), lons=(10.0, 11.0, 12.0, 13.0))
        sub = envelope.bbox_subset(ds, (2.5, 10.5, 0.5, 12.5))
        assert list(sub["latitude"].values) == [1.0, 2.0]
        assert list(sub["longitude"].values) == [11.0, 12.0]

    def test_descending_latitude_same_bbox(self):
        ds = make_gridded(lats=(3.0, 2.0, 1.0))
        sub = envelope.bbox_subset(ds, (2.5, 10.5, 0.5, 12.5))
        assert list(sub["latitude"].values) == [2.0, 1.0]

    def test_lon_0_360_normalized(self):
        ds = make_gridded(lons=(0.0, 90.0, 180.0, 270.0, 359.0))
        sub = envelope.bbox_subset(ds, (3.0, -95.0, 1.0, -85.0))
        assert list(sub["longitude"].values) == [-90.0]

    def test_antimeridian_keeps_wings_drops_interior(self):
        ds = make_gridded(lons=(-179.0, -100.0, 0.0, 100.0, 179.0))
        sub = envelope.bbox_subset(ds, (3.0, 170.0, 1.0, -170.0))
        assert list(sub["longitude"].values) == [-179.0, 179.0]

    def test_single_row_latitude_passes_through(self):
        ds = make_gridded(lats=(1.0,))
        sub = envelope.bbox_subset(ds, (60.0, 10.5, 50.0, 12.5))
        assert list(sub["latitude"].values) == [1.0]

    def test_non_monotonic_latitude_rejected(self):
        ds = make_gridded(lats=(1.0, 3.0, 2.0))
        with pytest.raises(UsageError, match="lat axis is non-monotonic"):
            envelope.bbox_subset(ds, (3.0, 10.0, 1.0, 13.0))

    def test_non_monotonic_longitude_rejected(self):
        ds = make_gridded(lons=(10.0, 12.0, 11.0, 13.0))
        with pytest.raises(UsageError, match="lon axis is non-monotonic"):
            envelope.bbox_subset(ds, (3.0, 10.0, 1.0, 13.0))

    def test_empty_longitude_axis_rejected(self):
        ds = make_gridded(lons=())
        with pytest.raises(UsageError, match="lon axis has length 0"):
            envelope.bbox_subset(ds, (3.0, 10.0, 1.0, 13.0))

    def test_descending_longitude_contiguous_span(self):
        ds = make_gridded(lons=(13.0, 12.0, 11.0, 10.0))
        sub = envelope.bbox_subset(ds, (2.5, 10.5, 0.5, 12.5))
        assert list(sub["longitude"].values) == [12.0, 11.0]

    def test_antimeridian_preserves_integer_dtype(self):
        ds = make_gridded(lons=(-179.0, -100.0, 0.0, 100.0, 179.0))
        ds["count"] = (
            ("time", "latitude", "longitude"),
            np.ones((2, 3, 5), dtype=np.int32),
        )
        sub = envelope.bbox_subset(ds, (3.0, 170.0, 1.0, -170.0))
        assert sub["count"].dtype == np.int32
        assert list(sub["longitude"].values) == [-179.0, 179.0]

    def test_antimeridian_descending_longitude_keeps_native_order(self):
        ds = make_gridded(lons=(179.0, 100.0, 0.0, -100.0, -179.0))
        sub = envelope.bbox_subset(ds, (3.0, 170.0, 1.0, -170.0))
        assert list(sub["longitude"].values) == [179.0, -179.0]

    def test_antimeridian_leaves_non_longitude_variables_alone(self):
        ds = make_gridded(lons=(-179.0, 0.0, 179.0))
        ds["tavg"] = (("time",), np.array([5, 6], dtype=np.int16))
        sub = envelope.bbox_subset(ds, (3.0, 170.0, 1.0, -170.0))
        assert sub["tavg"].dims == ("time",)
        assert sub["tavg"].dtype == np.int16
        assert list(sub["tavg"].values) == [5, 6]

    def test_empty_result_is_data_error(self):
        ds = make_gridded()
        with pytest.raises(DataError, match="selects no grid cells"):
            envelope.bbox_subset(ds, (60.0, 10.0, 50.0, 13.0))

    def test_empty_antimeridian_result_names_the_crossing(self):
        ds = make_gridded(lons=(-10.0, 0.0, 10.0))
        with pytest.raises(DataError, match="antimeridian"):
            envelope.bbox_subset(ds, (3.0, 170.0, 1.0, -170.0))

    def test_string_bbox_accepted(self):
        sub = envelope.bbox_subset(make_gridded(), "2.5/10.5/0.5/12.5")
        assert list(sub["latitude"].values) == [1.0, 2.0]

    def test_explicit_dims(self):
        ds = make_gridded().rename({"latitude": "yy", "longitude": "xx"})
        sub = envelope.bbox_subset(ds, (2.5, 10.5, 0.5, 12.5), lat_dim="yy", lon_dim="xx")
        assert list(sub["yy"].values) == [1.0, 2.0]

    def test_data_selected_matches_coords(self):
        ds = make_gridded()
        sub = envelope.bbox_subset(ds, (2.5, 10.5, 0.5, 12.5))
        assert sub["precip"].shape == (2, 2, 2)
        assert isinstance(sub, xr.Dataset)


class TestStampCfAttrs:
    def test_canonical_names(self):
        ds = envelope.stamp_cf_attrs(make_gridded())
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

    def test_alias_names(self):
        ds = envelope.stamp_cf_attrs(make_gridded().rename({"latitude": "lat", "longitude": "lon"}))
        assert ds["lat"].attrs["standard_name"] == "latitude"
        assert ds["lon"].attrs["standard_name"] == "longitude"

    def test_setdefault_preserves_source_values(self):
        ds = make_gridded()
        ds["latitude"].attrs["units"] = "degree_north"
        ds["time"].attrs["standard_name"] = "forecast_reference_time"
        envelope.stamp_cf_attrs(ds)
        assert ds["latitude"].attrs["units"] == "degree_north"
        assert ds["latitude"].attrs["axis"] == "Y"
        assert ds["time"].attrs["standard_name"] == "forecast_reference_time"

    def test_missing_coords_are_skipped(self):
        ds = make_gridded().rename({"latitude": "row", "longitude": "col"}).drop_vars("time")
        out = envelope.stamp_cf_attrs(ds)
        assert out["row"].attrs == {}
        assert out["col"].attrs == {}

    def test_returns_dataset(self):
        ds = make_gridded()
        assert envelope.stamp_cf_attrs(ds) is ds


class TestStampCfCoords:
    def test_overwrites_prior_values(self):
        ds = make_gridded()
        ds["latitude"].attrs.update(standard_name="wrong", units="wrong", axis="Z")
        envelope.stamp_cf_coords(ds)
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

    def test_time_gets_no_units(self):
        ds = envelope.stamp_cf_coords(make_gridded())
        assert ds["time"].attrs == {"standard_name": "time", "axis": "T"}

    def test_long_names_applied_with_setdefault(self):
        ds = make_gridded()
        ds["latitude"].attrs["long_name"] = "source latitude"
        envelope.stamp_cf_coords(
            ds, long_names={"latitude": "Latitude", "longitude": "Longitude", "time": "Time"}
        )
        assert ds["latitude"].attrs["long_name"] == "source latitude"
        assert ds["longitude"].attrs["long_name"] == "Longitude"
        assert ds["time"].attrs["long_name"] == "Time"

    def test_no_long_name_by_default(self):
        ds = envelope.stamp_cf_coords(make_gridded())
        assert "long_name" not in ds["latitude"].attrs

    def test_missing_coords_are_skipped(self):
        ds = make_gridded().drop_vars("time")
        out = envelope.stamp_cf_coords(ds)
        assert out["latitude"].attrs["axis"] == "Y"

    def test_alias_names_are_not_stamped(self):
        # Only the canonical post-rename names are asserted; a fetcher stamps
        # after renaming to latitude/longitude.
        ds = envelope.stamp_cf_coords(make_gridded().rename({"latitude": "lat"}))
        assert ds["lat"].attrs == {}

    def test_returns_dataset(self):
        ds = make_gridded()
        assert envelope.stamp_cf_coords(ds) is ds


class TestCfDim:
    def test_resolves_stamped_coord(self):
        ds = make_gridded().rename({"latitude": "yy"})
        ds["yy"].attrs.update(standard_name="latitude", units="degrees_north")
        assert envelope.cf_dim(ds, "latitude") == "yy"

    def test_unresolvable_returns_none(self):
        ds = make_gridded().rename({"latitude": "yy"})
        assert envelope.cf_dim(ds, "latitude") is None

    def test_works_on_dataarrays(self):
        ds = envelope.stamp_cf_attrs(make_gridded())
        assert envelope.cf_dim(ds["precip"], "longitude") == "longitude"


class TestAutoVariable:
    def test_picks_the_first_data_var(self):
        assert envelope.auto_variable(make_gridded()) == "precip"

    def test_skips_grid_mapping_container_and_targets(self):
        ds = make_gridded()
        ds["crs"] = xr.DataArray(0, attrs={"grid_mapping_name": "latitude_longitude"})
        ds["precip"].attrs["grid_mapping"] = "crs"
        assert envelope.auto_variable(ds) == "precip"
        # A var NAMED by another var's grid_mapping attr is skipped even
        # without its own grid_mapping_name attr.
        ds2 = make_gridded()
        ds2["other"] = xr.DataArray(0)
        ds2["precip"].attrs["grid_mapping"] = "other"
        assert envelope.auto_variable(ds2) == "precip"

    def test_prefers_multidim_vars(self):
        ds = make_gridded()
        ds = ds[["precip"]]
        ds["scalar_first"] = xr.DataArray(1.0)
        ds = ds[["scalar_first", "precip"]]
        assert envelope.auto_variable(ds) == "precip"

    def test_falls_back_to_one_dim_candidate(self):
        ds = make_gridded()
        ds["series"] = ("time", np.ones(ds.sizes["time"]))
        ds = ds[["series"]]
        assert envelope.auto_variable(ds) == "series"

    def test_no_candidates_returns_none(self):
        ds = make_gridded()[[]]
        assert envelope.auto_variable(ds) is None


class TestLatSlice:
    def test_ascending(self):
        assert envelope.lat_slice(np.array([1.0, 2.0, 3.0]), 3.0, 1.0) == slice(1.0, 3.0)

    def test_descending(self):
        assert envelope.lat_slice(np.array([3.0, 2.0, 1.0]), 3.0, 1.0) == slice(3.0, 1.0)

    def test_empty_axis_defaults_to_ascending(self):
        assert envelope.lat_slice(np.array([]), 3.0, 1.0) == slice(1.0, 3.0)

    def test_single_value(self):
        assert envelope.lat_slice(np.array([2.0]), 3.0, 1.0) == slice(1.0, 3.0)


class TestPolygonFromGeojson:
    square: ClassVar[dict] = {
        "type": "Polygon",
        "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]],
    }
    east_square: ClassVar[dict] = {
        "type": "Polygon",
        "coordinates": [[[2, 0], [2, 1], [3, 1], [3, 0], [2, 0]]],
    }

    def write(self, tmp_path, payload):
        import json

        p = tmp_path / "mask.geojson"
        p.write_text(json.dumps(payload))
        return p

    def test_feature_collection_unions_all_features(self, tmp_path):
        payload = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "geometry": self.square},
                {"type": "Feature", "geometry": self.east_square},
                {"type": "Feature", "geometry": None},
            ],
        }
        poly = envelope.polygon_from_geojson(self.write(tmp_path, payload))
        assert poly.area == pytest.approx(2.0)

    def test_single_feature(self, tmp_path):
        payload = {"type": "Feature", "geometry": self.square}
        poly = envelope.polygon_from_geojson(self.write(tmp_path, payload))
        assert poly.area == pytest.approx(1.0)

    def test_bare_geometry(self, tmp_path):
        poly = envelope.polygon_from_geojson(self.write(tmp_path, self.square))
        assert poly.area == pytest.approx(1.0)

    def test_missing_file(self, tmp_path):
        with pytest.raises(UsageError, match="--mask-geojson file not found"):
            envelope.polygon_from_geojson(tmp_path / "nope.geojson")

    def test_unreadable_json(self, tmp_path):
        p = tmp_path / "mask.geojson"
        p.write_text("{not json")
        with pytest.raises(UsageError, match="could not read --mask-geojson"):
            envelope.polygon_from_geojson(p)

    def test_no_usable_geometry(self, tmp_path):
        payload = {"type": "FeatureCollection", "features": []}
        with pytest.raises(UsageError, match="has no usable geometry"):
            envelope.polygon_from_geojson(self.write(tmp_path, payload))

    def test_top_level_array_is_no_usable_geometry(self, tmp_path):
        with pytest.raises(UsageError, match="has no usable geometry"):
            envelope.polygon_from_geojson(self.write(tmp_path, [self.square]))

    def test_top_level_scalar_is_no_usable_geometry(self, tmp_path):
        with pytest.raises(UsageError, match="has no usable geometry"):
            envelope.polygon_from_geojson(self.write(tmp_path, "Polygon"))

    def test_flag_names_the_source_flag(self, tmp_path):
        with pytest.raises(UsageError, match="--clip-geojson file not found"):
            envelope.polygon_from_geojson(tmp_path / "nope.geojson", flag="--clip-geojson")

    def test_non_list_features_value_raises_usage_error(self, tmp_path):
        payload = {"type": "FeatureCollection", "features": {"not": "a list"}}
        with pytest.raises(UsageError, match="'features' is not a list"):
            envelope.polygon_from_geojson(self.write(tmp_path, payload))

    def test_non_object_feature_entry_raises_usage_error(self, tmp_path):
        payload = {"type": "FeatureCollection", "features": ["not-an-object"]}
        with pytest.raises(UsageError, match="a feature is not a JSON object"):
            envelope.polygon_from_geojson(self.write(tmp_path, payload))

    def test_unknown_geometry_type_raises_usage_error_naming_the_flag(self, tmp_path):
        payload = {"type": "Bogus", "coordinates": [0, 0]}
        with pytest.raises(UsageError, match="--mask-geojson.*has no usable geometry"):
            envelope.polygon_from_geojson(self.write(tmp_path, payload))

    def test_geometry_missing_coordinates_raises_usage_error(self, tmp_path):
        payload = {"type": "Feature", "geometry": {"type": "Point"}}
        with pytest.raises(UsageError, match="has no usable geometry"):
            envelope.polygon_from_geojson(self.write(tmp_path, payload))

    def test_malformed_coordinates_raise_usage_error_not_a_traceback(self, tmp_path):
        # A string where a coordinate array is expected makes shape() raise a
        # TypeError; it must convert to a flag-named UsageError.
        payload = {"type": "Point", "coordinates": "nope"}
        with pytest.raises(UsageError, match="has no usable geometry"):
            envelope.polygon_from_geojson(self.write(tmp_path, payload))


class TestNormalizeLongitude:
    def test_0_360_axis_wraps_and_sorts(self):
        ds = make_gridded(lons=(0.0, 90.0, 180.0, 270.0))
        out = envelope.normalize_longitude(ds)
        assert list(out["longitude"].values) == [-180.0, -90.0, 0.0, 90.0]

    def test_values_follow_their_cells(self):
        ds = make_gridded(lons=(0.0, 90.0, 180.0, 270.0))
        ds["precip"][:, :, 3] = 7.0  # the 270 column
        out = envelope.normalize_longitude(ds)
        assert float(out["precip"].sel(longitude=-90.0).isel(time=0, latitude=0)) == 7.0

    def test_already_normalized_axis_is_unchanged(self):
        ds = make_gridded(lons=(-90.0, 0.0, 90.0))
        out = envelope.normalize_longitude(ds)
        assert list(out["longitude"].values) == [-90.0, 0.0, 90.0]

    def test_custom_dim_name(self):
        ds = make_gridded(lons=(0.0, 270.0)).rename({"longitude": "lon"})
        out = envelope.normalize_longitude(ds, lon_dim="lon")
        assert list(out["lon"].values) == [-90.0, 0.0]

    def test_longitude_attrs_preserved_across_the_wrap(self):
        ds = make_gridded(lons=(0.0, 90.0, 180.0, 270.0))
        ds["longitude"].attrs = {"standard_name": "longitude", "units": "degrees_east", "axis": "X"}
        out = envelope.normalize_longitude(ds)
        assert out["longitude"].attrs == {
            "standard_name": "longitude",
            "units": "degrees_east",
            "axis": "X",
        }

    def test_duplicate_endpoint_is_dropped_and_axis_stays_sorted(self):
        # 0.0 and 360.0 both wrap onto 0.0; the duplicate is dropped and the
        # axis remains a valid, ascending index.
        ds = make_gridded(lons=(0.0, 90.0, 180.0, 270.0, 360.0))
        out = envelope.normalize_longitude(ds)
        lons = list(out["longitude"].values)
        assert lons == [-180.0, -90.0, 0.0, 90.0]
        assert len(lons) == len(set(lons))

    def test_duplicate_drop_keeps_the_first_occurrence(self):
        # The 0.0 column carries a distinct value from the 360.0 column; the
        # first occurrence (input order) is the one kept.
        ds = make_gridded(lons=(0.0, 90.0, 180.0, 270.0, 360.0))
        ds["precip"][:, :, 0] = 5.0  # the original 0.0 column
        ds["precip"][:, :, 4] = 9.0  # the original 360.0 column
        out = envelope.normalize_longitude(ds)
        assert float(out["precip"].sel(longitude=0.0).isel(time=0, latitude=0)) == 5.0


class TestStampCfDsg:
    def stamped(self, ds=None, var_attrs=None):
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
        return envelope.stamp_cf_dsg(
            ds,
            var_attrs,
            station_id_long_name="GHCN station identifier",
            name_long_name="station name",
        )

    def test_coordinate_attrs(self):
        ds = self.stamped()
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

    def test_data_variable_attrs_follow_the_coordinates_attr(self):
        ds = self.stamped()
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

    def test_var_attrs_without_standard_name(self):
        # A variable whose unit family backs no CF standard_name entry is
        # stamped without one (units + long_name alone is CF-valid).
        ds = self.stamped(
            var_attrs={
                "precip": {"units": "mm", "long_name": "precip", "cell_methods": "time: mean"}
            }
        )
        assert "standard_name" not in ds["precip"].attrs
        assert ds["precip"].attrs["coordinates"] == "latitude longitude time"

    def test_optional_name_coord(self):
        ds = make_station()
        ds = ds.assign_coords(name=("station_id", ["a", "b", "c"]))
        self.stamped(ds=ds)
        assert ds["name"].attrs == {"long_name": "station name"}

    def test_name_absent_is_skipped(self):
        ds = self.stamped()
        assert "name" not in ds.variables

    def test_missing_var_attrs_entry_raises_keyerror(self):
        with pytest.raises(KeyError):
            self.stamped(var_attrs={})

    def test_returns_dataset(self):
        ds = make_station()
        assert self.stamped(ds=ds) is ds


class TestVerifyCfDsg:
    def test_stamped_dataset_passes(self):
        ds = make_station()
        envelope.stamp_cf_dsg(
            ds,
            {"precip": {"units": "mm", "long_name": "precip"}},
            station_id_long_name="id",
            name_long_name="name",
        )
        envelope.verify_cf_dsg(ds)

    def test_unstamped_dataset_lists_every_problem(self):
        ds = make_station()
        ds = ds.rename({"latitude": "row", "longitude": "col", "time": "record"})
        with pytest.raises(DataError) as excinfo:
            envelope.verify_cf_dsg(ds)
        message = str(excinfo.value)
        assert message.startswith("CF-1.13 DSG verification failed before write:")
        assert "cf_role timeseries_id did not resolve to station_id" in message
        for name in ("latitude", "longitude", "time"):
            assert f"cf-xarray could not resolve the {name} coordinate" in message

    def test_missing_cf_role_alone(self):
        ds = make_station()
        envelope.stamp_cf_dsg(
            ds,
            {"precip": {"units": "mm"}},
            station_id_long_name="id",
            name_long_name="name",
        )
        del ds["station_id"].attrs["cf_role"]
        with pytest.raises(DataError, match="timeseries_id did not resolve"):
            envelope.verify_cf_dsg(ds)


class TestUdunitsError:
    @pytest.mark.parametrize("units", ["mm", "degC", "kg m-3", "mm day-1", "1"])
    def test_valid_units_return_none(self, units):
        assert envelope.udunits_error(units) is None

    def test_invalid_units_return_the_exception(self):
        exc = envelope.udunits_error("definitely ! not a unit")
        assert isinstance(exc, ValueError)
        assert "not a unit" in str(exc)

    def test_blank_units_pass_through(self):
        # cf_units.Unit(None) and Unit("") return an "unknown" unit without
        # raising; rejecting blanks is the caller's guard.
        assert envelope.udunits_error(None) is None
        assert envelope.udunits_error("") is None

    def test_catch_widens_the_converted_failures(self):
        class Boom(Exception):
            pass

        # With the default catch, only ValueError converts; a wider catch
        # returns whatever cf_units raised.
        assert envelope.udunits_error("degC", catch=(Exception,)) is None
        exc = envelope.udunits_error("definitely ! not a unit", catch=(Exception,))
        assert isinstance(exc, ValueError)
        with pytest.raises(ValueError):
            envelope.udunits_error("definitely ! not a unit", catch=(Boom,))


class TestCfAxesMissing:
    def test_all_resolved(self):
        ds = envelope.stamp_cf_attrs(make_gridded())
        assert envelope.cf_axes_missing(ds) == []

    def test_partially_stamped_dataset_misses_x_and_y(self):
        # Axis resolution keys on the CF attrs (an unrenamed bare `time` name
        # resolves the "time" coordinate, not the "T" axis), so only the
        # stamped coord resolves.
        ds = make_gridded().rename({"latitude": "row", "longitude": "col"})
        ds["time"].attrs["axis"] = "T"
        assert envelope.cf_axes_missing(ds) == ["X", "Y"]

    def test_all_missing_on_unstamped_dataset(self):
        assert envelope.cf_axes_missing(make_gridded()) == ["X", "Y", "T"]

    def test_custom_axes(self):
        ds = make_gridded()
        ds["time"].attrs["axis"] = "T"
        assert envelope.cf_axes_missing(ds, axes=("T",)) == []
        assert envelope.cf_axes_missing(ds, axes=("Y",)) == ["Y"]
