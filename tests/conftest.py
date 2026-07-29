import numpy as np
import pytest
import xarray as xr


def make_gridded(
    n_time=2,
    lats=(1.0, 2.0, 3.0),
    lons=(10.0, 11.0, 12.0, 13.0),
    name="precip",
    fill=1.0,
    start="2026-01-01",
):
    times = np.arange(np.datetime64(start), np.datetime64(start) + np.timedelta64(n_time, "D"))
    data = np.full((n_time, len(lats), len(lons)), fill)
    return xr.Dataset(
        {name: (("time", "latitude", "longitude"), data)},
        coords={
            "time": times.astype("datetime64[ns]"),
            "latitude": list(lats),
            "longitude": list(lons),
        },
    )


def make_series(n_time=4, name="precip", start="2026-01-01"):
    """A series envelope: a time axis and no spatial coords, as collapsing lat/lon leaves."""
    times = np.arange(np.datetime64(start), np.datetime64(start) + np.timedelta64(n_time, "D"))
    return xr.Dataset(
        {name: (("time",), np.ones(n_time))},
        coords={"time": times.astype("datetime64[ns]")},
    )


def make_forecast(n_number=3, n_step=4):
    data = np.ones((n_number, n_step, 2, 2))
    return xr.Dataset(
        {"tp": (("number", "step", "latitude", "longitude"), data)},
        coords={
            "number": np.arange(n_number),
            "step": np.array([np.timedelta64(i, "D") for i in range(n_step)]),
            "time": np.datetime64("2026-01-01", "ns"),
            "latitude": [0.0, 1.0],
            "longitude": [10.0, 11.0],
        },
    )


def make_station(n_station=3, n_time=2):
    ids = [f"TA{i:04d}" for i in range(n_station)]
    times = np.arange(
        np.datetime64("2026-01-01"), np.datetime64("2026-01-01") + np.timedelta64(n_time, "D")
    )
    return xr.Dataset(
        {"precip": (("time", "station_id"), np.ones((n_time, n_station)))},
        coords={
            "time": times.astype("datetime64[ns]"),
            "station_id": ids,
            "latitude": ("station_id", np.linspace(-1.0, 1.0, n_station)),
            "longitude": ("station_id", np.linspace(36.0, 38.0, n_station)),
        },
    )


@pytest.fixture
def gridded_store(tmp_path):
    path = tmp_path / "in.zarr"
    make_gridded().to_zarr(path, mode="w", consolidated=True)
    return path
