"""Tests for the simplified @weather_skill decorator."""

from datetime import date
from pathlib import Path

import pytest
import xarray as xr

from weather_skills_core import EntryOverride, Types, weather_skill
from conftest import make_gridded


def test_fetcher_writes_gridded(tmp_path):
    out = tmp_path / "out.zarr"

    @weather_skill(
        name="toy-fetch",
        version="0.1.0",
        outputs=[Types.GRIDDED],
        required_args=("start_time", "end_time", "variable"),
        optional_args=("bbox",),
        check_cache=True,
    )
    @weather_skill.argument("--workers", type=int, default=1)
    def fetch(start_time, end_time, variable, bbox, workers):
        assert isinstance(start_time, date)
        assert end_time >= start_time
        assert variable == ["precip"]
        assert bbox is None
        assert workers == 1
        return make_gridded()

    fetch(
        [
            "--start",
            "2026-01-01",
            "--end",
            "2026-01-02",
            "-v",
            "precip",
            "-o",
            str(out),
        ]
    )
    assert out.exists()
    ds = xr.open_zarr(out, consolidated=True)
    assert "precip" in ds
    assert "weather_skills_history" in ds.attrs


def test_latest_passthrough_and_entry_override(tmp_path):
    out = tmp_path / "out.zarr"

    @weather_skill(
        name="toy-latest",
        version="0.1.0",
        outputs=[Types.GRIDDED],
        required_args=("start_time", "end_time"),
    )
    def fetch(start_time, end_time):
        assert start_time == "latest"
        assert isinstance(end_time, date)
        return make_gridded(), EntryOverride(
            args={"start_time": "2026-01-01", "end_time": end_time.isoformat()}
        )

    fetch(["--start", "latest", "--end", "2026-01-10", "-o", str(out)])
    ds = xr.open_zarr(out, consolidated=True)
    import json

    history = json.loads(ds.attrs["weather_skills_history"])
    assert history[0]["args"]["start_time"] == "2026-01-01"


def test_transform_cache_hit(tmp_path, gridded_store, capsys):
    out = tmp_path / "clipped.zarr"

    @weather_skill(
        name="toy-clip",
        version="0.1.0",
        inputs=[Types.ANY],
        outputs=[Types.ANY],
        required_args=("bbox",),
    )
    def clip(ds, bbox):
        return ds

    argv = ["-i", str(gridded_store), "-o", str(out), "--bbox", "3/10/1/13"]
    clip(argv)
    assert out.exists()
    clip(argv)
    err = capsys.readouterr().err
    assert "cache hit" in err


def test_check_cache_false_rewrites(tmp_path, gridded_store):
    out = tmp_path / "out.zarr"
    calls = []

    @weather_skill(
        name="toy-clip",
        version="0.1.0",
        inputs=[Types.GRIDDED],
        outputs=[Types.GRIDDED],
        required_args=("bbox",),
        check_cache=True,
    )
    def clip(ds, bbox):
        calls.append(1)
        return ds

    argv = ["-i", str(gridded_store), "-o", str(out), "--bbox", "3/10/1/13"]
    clip(argv)
    clip([*argv, "--no-check-cache"])
    assert len(calls) == 2


def test_rejects_offset_dates(tmp_path):
    @weather_skill(
        name="toy",
        version="0.1.0",
        outputs=[Types.GRIDDED],
        required_args=("time",),
    )
    def fetch(time):
        return make_gridded()

    with pytest.raises(SystemExit) as exc:
        fetch(["--time", "latest-2w", "-o", str(tmp_path / "o.zarr")])
    assert exc.value.code == 2


def test_signature_must_include_catalog_args():
    with pytest.raises(ValueError, match="missing arg"):

        @weather_skill(
            name="bad",
            version="0.1.0",
            outputs=[Types.GRIDDED],
            required_args=("bbox",),
        )
        def fetch():
            return make_gridded()


def test_check_cache_not_in_signature():
    with pytest.raises(ValueError, match="check_cache"):

        @weather_skill(name="bad", version="0.1.0", outputs=[Types.GRIDDED])
        def fetch(check_cache=True):
            return make_gridded()


def test_custom_argument(tmp_path):
    out = tmp_path / "out.zarr"

    @weather_skill(
        name="toy",
        version="0.1.0",
        outputs=[Types.GRIDDED],
        exclude_args=("workers",),
    )
    @weather_skill.argument("--workers", type=int, default=2)
    def fetch(workers):
        assert workers == 4
        return make_gridded()

    fetch(["--workers", "4", "-o", str(out)])
    assert out.exists()


def test_negative_bbox(tmp_path, gridded_store):
    out = tmp_path / "out.zarr"

    @weather_skill(
        name="toy-clip",
        version="0.1.0",
        inputs=[Types.ANY],
        outputs=[Types.ANY],
        required_args=("bbox",),
        check_cache=False,
    )
    def clip(ds, bbox):
        assert bbox[0] == -10.0
        return ds

    clip(["-i", str(gridded_store), "-o", str(out), "--bbox", "-10/-20/-30/-5"])
