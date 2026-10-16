"""Decorator behavior for the stacked @weather_skill.argument API."""

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from conftest import make_forecast, make_gridded
from PIL import Image

from weather_skills_core import Dataset
from weather_skills_core.decorator import argv_has_option, weather_skill
from weather_skills_core.provenance import (
    HISTORY_ATTR,
    load_figure_history,
    load_history,
    stamp_zarr,
)
from weather_skills_core.standard_args import rewrite_negative_values


def test_rewrite_negative_values():
    opts = {"--bbox", "--point", "-o"}
    assert rewrite_negative_values(["--bbox", "-10/20/-20/30"], opts) == ["--bbox=-10/20/-20/30"]
    assert rewrite_negative_values(["--point", "-1.28,36.82,NBO"], opts) == [
        "--point=-1.28,36.82,NBO"
    ]
    assert rewrite_negative_values(["--point", "-.5,36"], opts) == ["--point=-.5,36"]
    # A real flag after a value option is left for argparse to report.
    assert rewrite_negative_values(["--point", "-o", "x"], opts) == ["--point", "-o", "x"]
    # Options not taking a value are untouched.
    assert rewrite_negative_values(["--flag", "-1,2"], opts) == ["--flag", "-1,2"]


def test_negative_point_parses_without_equals(tmp_path):
    seen = {}

    @weather_skill(name="pts", version="1.0.0", output=False)
    @weather_skill.argument("--point", action="append")
    @weather_skill.argument("--bbox")
    def pts(point, bbox, **kwargs):
        seen.update(point=point, bbox=bbox)

    pts(["--point", "-1.28,36.82,NBO", "--point", "-4.04,39.67", "--bbox", "-3/39/-5/40"])
    assert seen == {"point": ["-1.28,36.82,NBO", "-4.04,39.67"], "bbox": (-3.0, 39.0, -5.0, 40.0)}


def test_parser_basic_flags():

    @weather_skill(name="s", version="1.0.0")
    @weather_skill.argument("-i", "--input", type=Dataset("observations"), required=True)
    @weather_skill.argument("--bbox")
    def skill(ds, output, bbox, **kwargs):
        return ds

    dests = {a.dest for a in skill.parser._actions if a.dest != "help"}
    assert dests == {"input", "output", "bbox"}


def test_argv_has_option():
    flags = ("--probe-latest",)
    assert argv_has_option(["--probe-latest"], flags) is True
    assert argv_has_option(["--probe-latest", "final"], flags) is True
    assert argv_has_option(["--probe-latest=final"], flags) is True
    assert argv_has_option(["--start-time", "2026-01-01"], flags) is False
    assert argv_has_option(["--probe-latest-extra"], flags) is False


def test_parser_probe_latest_skips_required_output_and_flags(capsys):
    @weather_skill(name="s", version="1.0.0")
    @weather_skill.argument("--start-time", required=True)
    @weather_skill.argument("--probe-latest", nargs="?", const="", default=None, probe=True)
    def skill(output, start_time, probe_latest, **kwargs):
        print("none")

    skill(["--probe-latest"])
    assert capsys.readouterr().out.strip() == "none"
    out_action = next(a for a in skill.parser._actions if a.dest == "output")
    start_action = next(a for a in skill.parser._actions if a.dest == "start_time")
    assert out_action.required is True
    assert start_action.required is True


def test_probe_skips_required_input_and_does_not_write(tmp_path):
    out = tmp_path / "out.zarr"

    @weather_skill(name="s", version="1.0.0")
    @weather_skill.argument("-i", "--input", type=Dataset("observations"), required=True)
    @weather_skill.argument("--ping", action="store_true", probe=True)
    def skill(ds, output, ping, **kwargs):
        assert ping is True
        assert ds is None
        return make_gridded()

    result = skill(["--ping", "-o", str(out)])
    assert result is not None
    assert not out.exists()


def test_probe_equals_form_skips_required_flags(capsys):
    @weather_skill(name="s", version="1.0.0")
    @weather_skill.argument("--start-time", required=True)
    @weather_skill.argument("--ident", nargs="?", const="", default=None, probe=True)
    def skill(output, start_time, ident, **kwargs):
        print(ident)

    skill(["--ident=final"])
    assert capsys.readouterr().out.strip() == "final"


def test_missing_output_still_required_without_probe():
    @weather_skill(name="s", version="1.0.0")
    @weather_skill.argument("--ping", action="store_true", probe=True)
    def skill(output, ping, **kwargs):
        return make_gridded()

    with pytest.raises(SystemExit):
        skill([])


def test_parser_dates_range():

    @weather_skill(name="s", version="1.0.0")
    @weather_skill.argument("--start-time", required=True)
    @weather_skill.argument("--end-time", required=True)
    def skill(output, start_time, end_time, **kwargs):
        return make_gridded()

    dests = {a.dest for a in skill.parser._actions if a.dest != "help"}
    assert "start_time" in dests and "end_time" in dests


def test_parser_variable_append():

    @weather_skill(name="s", version="1.0.0")
    @weather_skill.argument("--variable", "-v", action="append", required=True)
    def skill(output, variable, **kwargs):
        return make_gridded()

    action = next(a for a in skill.parser._actions if a.dest == "variable")
    assert action.required is True
    assert action.option_strings == ["--variable", "-v"]


def test_parser_extra_argument():

    @weather_skill(name="s", version="1.0.0")
    @weather_skill.argument("--smoothing", "-s", type=int, default=1)
    def skill(output, smoothing, **kwargs):
        return make_gridded()

    action = next(a for a in skill.parser._actions if a.dest == "smoothing")
    assert action.default == 1


def test_parser_requires_kwargs():
    with pytest.raises(TypeError, match="\\*\\*kwargs"):

        @weather_skill(name="s", version="1.0.0")
        def skill(ds):
            return ds


def test_parser_argument_order_matches_source():

    @weather_skill(name="s", version="1.0.0")
    @weather_skill.argument("--date", required=True)
    @weather_skill.argument("--bbox", help="Study area.")
    def skill(output, date, bbox, **kwargs):
        return make_gridded()

    dests = [a.dest for a in skill.parser._actions if a.dest in ("date", "bbox")]
    assert dests == ["date", "bbox"]


def test_parser_canonical_bbox_help_appended():

    @weather_skill(name="s", version="1.0.0")
    @weather_skill.argument("--bbox", help="Study area.")
    def skill(output, bbox, **kwargs):
        return make_gridded()

    action = next(a for a in skill.parser._actions if a.dest == "bbox")
    assert "Study area." in action.help
    assert "N/W/S/E" in action.help


def test_parser_name_and_version_are_keyword_only():
    with pytest.raises(TypeError):
        weather_skill("s", "1.0.0")


def test_parser_dataset_and_comma_and():
    d = Dataset("lat, lon")
    assert d.io_spec.alternatives == (frozenset({"lat", "lon"}),)
    d2 = Dataset(["forecast", "ensemble_forecast"])
    assert len(d2.io_spec.alternatives) == 2


def test_run_loop_copy_dataset(tmp_path):
    src = tmp_path / "in.zarr"
    out = tmp_path / "out.zarr"
    make_gridded().to_zarr(src, mode="w", consolidated=True)

    @weather_skill(name="copy", version="0.1.0")
    @weather_skill.argument("-i", "--input", type=Dataset("observations"), required=True)
    def copy(ds, output, **kwargs):
        return ds

    copy(["-i", str(src), "-o", str(out)])
    assert out.exists()
    assert load_history(out)[-1]["skill"] == "copy"


def test_run_loop_bbox_passed_as_tuple(tmp_path):
    from datetime import date

    src = tmp_path / "in.zarr"
    out = tmp_path / "out.zarr"
    make_gridded().to_zarr(src, mode="w", consolidated=True)
    seen = {}

    @weather_skill(name="s", version="0.1.0")
    @weather_skill.argument("-i", "--input", type=Dataset("observations"), required=True)
    @weather_skill.argument("--bbox", required=True)
    @weather_skill.argument("--start-time", required=True)
    @weather_skill.argument("--end-time", required=True)
    def skill(ds, output, bbox, start_time, end_time, **kwargs):
        seen["bbox"] = bbox
        seen["start_time"] = start_time
        seen["end_time"] = end_time
        return ds

    skill(
        [
            "-i",
            str(src),
            "-o",
            str(out),
            "--bbox",
            "10/20/0/30",
            "--start-time",
            "2026-01-01",
            "--end-time",
            "2026-01-10",
        ]
    )
    assert seen["bbox"] == (10.0, 20.0, 0.0, 30.0)
    assert seen["start_time"] == date(2026, 1, 1)
    assert isinstance(seen["start_time"], date)
    assert seen["end_time"] == date(2026, 1, 10)


def test_run_loop_start_after_end_exits(tmp_path):
    out = tmp_path / "out.zarr"

    @weather_skill(name="s", version="0.1.0")
    @weather_skill.argument("--start-time", required=True)
    @weather_skill.argument("--end-time", required=True)
    def skill(output, start_time, end_time, **kwargs):
        return make_gridded()

    with pytest.raises(SystemExit) as exc:
        skill(["-o", str(out), "--start-time", "2026-01-10", "--end-time", "2026-01-01"])
    assert exc.value.code == 2


def test_run_loop_date_parsed(tmp_path):
    out = tmp_path / "out.zarr"
    seen = {}

    @weather_skill(name="s", version="0.1.0")
    @weather_skill.argument("--date", required=True)
    def skill(output, date, **kwargs):
        seen["date"] = date.isoformat()
        return make_gridded()

    skill(["-o", str(out), "--date", "2026-01-15"])
    assert seen["date"] == "2026-01-15"
    assert load_history(out)[-1]["args"]["date"] == "2026-01-15"


def test_run_loop_two_inputs(tmp_path):
    a = tmp_path / "a.zarr"
    b = tmp_path / "b.zarr"
    out = tmp_path / "out.zarr"
    make_gridded().to_zarr(a, mode="w", consolidated=True)
    make_gridded(fill=2.0).to_zarr(b, mode="w", consolidated=True)

    @weather_skill(name="s", version="1.0.0")
    @weather_skill.argument(
        "-i", "--input", type=Dataset("observations"), action="append", required=True
    )
    def skill(ds, output, **kwargs):
        seen["n"] = len(ds)
        return ds[0]

    seen = {}
    skill(["-i", str(a), "-i", str(b), "-o", str(out)])
    assert seen["n"] == 2
    assert out.exists()


def test_run_loop_path_input_not_dataset(tmp_path):
    raw = tmp_path / "raw.bin"
    raw.write_bytes(b"abc")
    out = tmp_path / "out.zarr"

    @weather_skill(name="wrap", version="0.1.0")
    @weather_skill.argument("-i", "--input", type=Path, required=True)
    def wrap(input, output, **kwargs):
        assert input == raw
        return make_gridded()

    wrap(["-i", str(raw), "-o", str(out)])
    assert out.exists()


def test_run_loop_figure_output(tmp_path):
    src = tmp_path / "in.zarr"
    out = tmp_path / "plot.png"
    make_gridded().to_zarr(src, mode="w", consolidated=True)

    @weather_skill(name="plot", version="0.1.0")
    @weather_skill.argument("-i", "--input", type=Dataset("observations"), required=True)
    def plot(ds, output, **kwargs):
        Image.new("RGB", (8, 8), color=(1, 2, 3)).save(output)
        return output

    plot(["-i", str(src), "-o", str(out)])
    assert out.exists()
    assert load_figure_history(out)[-1]["skill"] == "plot"


def test_run_loop_figure_wrong_path_exits(tmp_path):
    out = tmp_path / "plot.png"
    wrong = tmp_path / "other.png"

    @weather_skill(name="plot", version="0.1.0")
    def plot(output, **kwargs):
        Image.new("RGB", (4, 4)).save(wrong)
        return wrong

    with pytest.raises(SystemExit) as exc:
        plot(["-o", str(out)])
    assert exc.value.code == 1


def test_run_loop_output_kwarg_single(tmp_path):
    src = tmp_path / "in.zarr"
    out = tmp_path / "out.zarr"
    make_gridded().to_zarr(src, mode="w", consolidated=True)
    seen = {}

    @weather_skill(name="copy", version="0.1.0")
    @weather_skill.argument("-i", "--input", type=Dataset("observations"), required=True)
    def copy(ds, output, **kwargs):
        seen["output"] = output
        return ds

    copy(["-i", str(src), "-o", str(out)])
    assert seen["output"] == out


def test_run_loop_refuses_plain_file_output(tmp_path, capsys):
    src = tmp_path / "in.zarr"
    out = tmp_path / "notes.txt"
    make_gridded().to_zarr(src, mode="w", consolidated=True)
    out.write_text("keep me")

    @weather_skill(name="copy", version="0.1.0")
    @weather_skill.argument("-i", "--input", type=Dataset("observations"), required=True)
    def copy(ds, **kwargs):
        return ds

    with pytest.raises(SystemExit) as exc:
        copy(["-i", str(src), "-o", str(out)])
    assert exc.value.code == 2
    assert "existing file, not a Zarr store" in capsys.readouterr().err
    assert out.read_text() == "keep me"


def test_run_loop_no_artifact(tmp_path):
    src = tmp_path / "in.zarr"
    make_gridded().to_zarr(src, mode="w", consolidated=True)

    @weather_skill(name="inspect", version="0.1.0", output=False)
    @weather_skill.argument("-i", "--input", type=Dataset("observations"), required=True)
    def inspect(ds, **kwargs):
        return {"n": int(ds.sizes["time"])}

    assert inspect(["-i", str(src)]) == {"n": 2}


def test_run_loop_rejects_manual_output_flag():
    with pytest.raises(ValueError, match="do not declare -o/--output"):

        @weather_skill(name="s", version="0.1.0")
        @weather_skill.argument("-o", "--output", type=Path, required=True)
        def skill(output, **kwargs):
            return make_gridded()


def test_run_loop_multi_output_writes_both(tmp_path):
    src = tmp_path / "in.zarr"
    a = tmp_path / "a.zarr"
    b = tmp_path / "b.zarr"
    make_gridded().to_zarr(src, mode="w", consolidated=True)

    @weather_skill(name="split", version="0.1.0")
    @weather_skill.argument("-i", "--input", type=Dataset("observations"), required=True)
    def split(ds, output, **kwargs):
        assert output == [a, b]
        return (ds, ds)

    split(["-i", str(src), "-o", str(a), "-o", str(b)])
    assert a.exists() and b.exists()


def test_run_loop_output_count_mismatch_exits(tmp_path):
    src = tmp_path / "in.zarr"
    a = tmp_path / "a.zarr"
    b = tmp_path / "b.zarr"
    make_gridded().to_zarr(src, mode="w", consolidated=True)

    @weather_skill(name="copy", version="0.1.0")
    @weather_skill.argument("-i", "--input", type=Dataset("observations"), required=True)
    def copy(ds, output, **kwargs):
        return ds

    with pytest.raises(SystemExit) as exc:
        copy(["-i", str(src), "-o", str(a), "-o", str(b)])
    assert exc.value.code == 1


def test_run_loop_any_accepts_shapes(tmp_path):
    src = tmp_path / "in.zarr"
    out = tmp_path / "out.zarr"
    make_gridded().to_zarr(src, mode="w", consolidated=True)

    @weather_skill(name="s", version="0.1.0")
    @weather_skill.argument("-i", "--input", type=Dataset("any"), required=True)
    def skill(ds, output, **kwargs):
        return ds

    skill(["-i", str(src), "-o", str(out)])
    assert out.exists()


def test_run_loop_variadic_inputs(tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f"{i}.zarr"
        make_gridded(fill=float(i)).to_zarr(p, mode="w", consolidated=True)
        paths.append(p)
    out = tmp_path / "out.zarr"
    seen = {}

    @weather_skill(name="cat", version="0.1.0")
    @weather_skill.argument("-i", "--input", type=Dataset("any"), action="append", required=True)
    def cat(ds, output, **kwargs):
        seen["n"] = len(ds)
        return ds[0]

    argv = [token for p in paths for token in ("-i", str(p))] + ["-o", str(out)]
    cat(argv)
    assert seen["n"] == 3


def test_run_loop_multi_input_records_join_dag_and_commit(tmp_path):
    from weather_skills_core.decorator import build_history
    from weather_skills_core.provenance import origin_skill

    a = tmp_path / "a.zarr"
    b = tmp_path / "b.zarr"
    out = tmp_path / "out.zarr"
    a_hist = [{"skill": "chirps-fetch", "version": "0.1.0", "args": {}, "input": None}]
    b_hist = [{"skill": "dynamical-fetch", "version": "0.1.0", "args": {}, "input": None}]
    ds_a = make_gridded(fill=1.0)
    stamp_zarr(ds_a, a_hist)
    ds_a.to_zarr(a, mode="w", consolidated=True)
    ds_b = make_gridded(fill=2.0)
    stamp_zarr(ds_b, b_hist)
    ds_b.to_zarr(b, mode="w", consolidated=True)

    @weather_skill(name="cat", version="0.1.0")
    @weather_skill.argument("-i", "--input", type=Dataset("any"), action="append", required=True)
    def cat(ds, output, **kwargs):
        return ds[0]

    cat(["-i", str(a), "-i", str(b), "-o", str(out)])
    history = load_history(out)
    assert [step["skill"] for step in history] == ["cat"]
    parents = history[0]["input"]
    assert [p["basename"] for p in parents] == ["a.zarr", "b.zarr"]
    assert parents[0]["history"] == a_hist
    assert parents[1]["history"] == b_hist
    assert origin_skill(history) == "chirps-fetch"
    assert isinstance(history[0].get("commit"), str) and history[0]["commit"]

    # Direct helper: a join must not prepend the first parent as a linear spine.
    class _Args:
        extra = True

    joined = build_history(
        "cat",
        "0.1.0",
        _Args(),
        {},
        [a, b],
        [a_hist, b_hist],
        {"output"},
        revision={"commit": "deadbeef"},
    )
    assert [step["skill"] for step in joined] == ["cat"]
    assert joined[0]["commit"] == "deadbeef"


def test_run_loop_negative_bbox_latitude(tmp_path):
    out = tmp_path / "out.zarr"
    seen = {}

    @weather_skill(name="s", version="0.1.0")
    @weather_skill.argument("--bbox", required=True)
    def skill(output, bbox, **kwargs):
        seen["bbox"] = bbox
        return make_gridded()

    skill(["-o", str(out), "--bbox", "-10/20/-20/30"])
    assert seen["bbox"] == (-10.0, 20.0, -20.0, 30.0)


def test_run_loop_history_args_json(tmp_path):
    out = tmp_path / "out.zarr"

    @weather_skill(name="s", version="0.1.0")
    @weather_skill.argument("--variable", "-v", action="append", required=True)
    def skill(output, variable, **kwargs):
        return make_gridded()

    skill(["-o", str(out), "-v", "precip", "-v", "temp"])
    entry = load_history(out)[-1]
    assert entry["args"]["variable"] == ["precip", "temp"]
    assert "output" not in entry["args"]
    json.dumps(entry)


def test_run_loop_attrs_merge_from_input(tmp_path):
    src = tmp_path / "in.zarr"
    out = tmp_path / "out.zarr"
    ds = make_gridded()
    ds.attrs["weather_skills_source"] = "test-src"
    ds.to_zarr(src, mode="w", consolidated=True)

    @weather_skill(name="copy", version="0.1.0")
    @weather_skill.argument("-i", "--input", type=Dataset("observations"), required=True)
    def copy(ds, output, **kwargs):
        return ds.assign(precip=ds["precip"] * 2)

    copy(["-i", str(src), "-o", str(out)])
    written = xr.open_zarr(out, consolidated=True)
    assert written.attrs.get("weather_skills_source") == "test-src"
    assert HISTORY_ATTR in written.attrs


def test_run_loop_write_normalizes_step_and_fills_stripped_units(tmp_path):
    src = tmp_path / "in.zarr"
    out = tmp_path / "out.zarr"
    ds = make_forecast(n_number=0, n_step=3)
    ds = ds.assign_coords(step=ds["step"].values.astype("timedelta64[us]"))
    ds.to_zarr(src, mode="w", consolidated=True)

    @weather_skill(name="strip", version="0.1.0")
    @weather_skill.argument("-i", "--input", type=Dataset("forecast"), required=True)
    def strip(ds, output, **kwargs):
        out_ds = ds.copy(deep=True)
        for name in out_ds.data_vars:
            out_ds[name].attrs.pop("units", None)
        return out_ds

    strip(["-i", str(src), "-o", str(out)])
    written = xr.open_zarr(out, consolidated=True)
    assert written["step"].dtype == np.dtype("timedelta64[ns]")
    assert "units" in written["tp"].attrs
    assert written["tp"].attrs["units"] in ("mm day-1", "millimeter / day")


def test_run_loop_write_stamps_amount_standard_name(tmp_path):
    src = tmp_path / "in.zarr"
    out = tmp_path / "out.zarr"
    ds = make_gridded(name="tp", units="kg m-2")
    ds["tp"].attrs["standard_name"] = "lwe_precipitation_rate"
    ds.to_zarr(src, mode="w", consolidated=True)

    @weather_skill(name="copy", version="0.1.0")
    @weather_skill.argument("-i", "--input", type=Dataset("observations"), required=True)
    def copy(ds, output, **kwargs):
        return ds

    copy(["-i", str(src), "-o", str(out)])
    written = xr.open_zarr(out, consolidated=True)
    assert written["tp"].attrs["standard_name"] == "precipitation_amount"  # CF: kg m-2 is a mass


def test_run_loop_write_snaps_latlon_to_float32(tmp_path):
    src = tmp_path / "in.zarr"
    out = tmp_path / "out.zarr"
    ds = make_gridded(lats=(5.9749990996248385, -1.2750010213), lons=(33.0, 36.825))
    ds.to_zarr(src, mode="w", consolidated=True)

    @weather_skill(name="copy", version="0.1.0")
    @weather_skill.argument("-i", "--input", type=Dataset("observations"), required=True)
    def copy(ds, output, **kwargs):
        return ds

    copy(["-i", str(src), "-o", str(out)])
    written = xr.open_zarr(out, consolidated=True)
    assert written["latitude"].dtype == np.float32
    assert written["longitude"].dtype == np.float32
    np.testing.assert_array_equal(
        written["latitude"].values,
        np.round(np.array([5.9749990996248385, -1.2750010213]), 5).astype(np.float32),
    )
    np.testing.assert_array_equal(
        written["longitude"].values,
        np.round(np.array([33.0, 36.825]), 5).astype(np.float32),
    )


def test_run_loop_none_return_skips_write(tmp_path):
    out = tmp_path / "out.txt"

    @weather_skill(name="compose", version="0.1.0")
    def compose(output, **kwargs):
        output.write_text("ok")

    compose(["-o", str(out)])
    assert out.read_text() == "ok"


class _LayerHolder:
    """Minimal zarr_paths() holder for decorator tests."""

    def __init__(self, path):
        self.path = Path(path)
        self.ds = None
        self.raw = f"heatmap:{path}"

    def zarr_paths(self):
        return [self.path]

    def __str__(self):
        return self.raw


def test_zarr_paths_holder_is_opened_and_hashed(tmp_path):
    src = tmp_path / "in.zarr"
    out = tmp_path / "out.zarr"
    make_gridded().to_zarr(src, mode="w", consolidated=True)
    seen = {}

    @weather_skill(name="layered", version="0.1.0")
    @weather_skill.argument("--layer", action="append", type=_LayerHolder)
    def layered(output, layer, **kwargs):
        seen["ds"] = layer[0].ds
        return layer[0].ds

    layered(["--layer", str(src), "-o", str(out)])
    assert seen["ds"] is not None
    assert "precip" in seen["ds"]
    history = load_history(out)
    assert history[-1]["skill"] == "layered"
    assert history[-1]["input"]["basename"] == "in.zarr"


def test_zarr_paths_dedupes_identical_paths(tmp_path):
    src = tmp_path / "in.zarr"
    out = tmp_path / "out.zarr"
    make_gridded().to_zarr(src, mode="w", consolidated=True)
    seen = {}

    @weather_skill(name="layered", version="0.1.0")
    @weather_skill.argument("--layer", action="append", type=_LayerHolder)
    def layered(output, layer, **kwargs):
        seen["layers"] = layer
        return layer[0].ds

    layered(["--layer", str(src), "--layer", str(src), "-o", str(out)])
    assert seen["layers"][0].ds is seen["layers"][1].ds
    history = load_history(out)
    inp = history[-1]["input"]
    assert inp["basename"] == "in.zarr"


class _NamedInput:
    """``PATH=NAME`` holder, like a plotting skill's ``-i``."""

    def __init__(self, value):
        self.path, _, self.name = value.rpartition("=")
        self.ds = None

    def zarr_paths(self):
        return [Path(self.path)]

    def __str__(self):
        return f"{self.path}={self.name}"


def test_history_records_flags_and_input_args(tmp_path):
    a = tmp_path / "a.zarr"
    b = tmp_path / "b.zarr"
    out = tmp_path / "out.zarr"
    make_gridded().to_zarr(a, mode="w", consolidated=True)
    make_gridded(fill=2.0).to_zarr(b, mode="w", consolidated=True)

    @weather_skill(name="named", version="0.1.0")
    @weather_skill.argument("-i", "--input", dest="inputs", action="append", type=_NamedInput)
    @weather_skill.argument("--obs", type=Dataset("any"))
    @weather_skill.argument("--max-lead", "-m")
    def named(output, inputs, obs, **kwargs):
        return inputs[0].ds

    named(["-i", f"{a}=A", "-i", f"{b}=B", "-i", f"{a}=C", "--obs", str(b), "-o", str(out)])
    entry = load_history(out)[-1]
    # Only dests whose option is not --dest are mapped.
    assert entry["flags"] == {"inputs": "--input"}
    assert entry["args"]["inputs"] == [f"{a}=A", f"{b}=B", f"{a}=C"]
    assert [p["basename"] for p in entry["input"]] == ["b.zarr", "a.zarr"]
    # Dataset dests open first; a repeated path reuses its input index.
    assert entry["input_args"] == [
        {"dest": "obs", "item": None, "path": str(b), "input": 0},
        {"dest": "inputs", "item": 0, "path": str(a), "input": 1},
        {"dest": "inputs", "item": 1, "path": str(b), "input": 0},
        {"dest": "inputs", "item": 2, "path": str(a), "input": 1},
    ]


def test_history_omits_flags_when_options_match_dests(tmp_path):
    src = tmp_path / "in.zarr"
    out = tmp_path / "out.zarr"
    make_gridded().to_zarr(src, mode="w", consolidated=True)

    @weather_skill(name="plain", version="0.1.0")
    @weather_skill.argument("-i", "--input", type=Dataset("any"), required=True)
    @weather_skill.argument("--max-lead")
    def plain(ds, output, **kwargs):
        return ds

    plain(["-i", str(src), "-o", str(out)])
    entry = load_history(out)[-1]
    assert "flags" not in entry
    assert entry["input_args"] == [{"dest": "input", "item": None, "path": str(src), "input": 0}]


def test_float64_input_joins_float32_output(tmp_path):
    """A pre-float32 Zarr still aligns cell-for-cell with a new float32 one."""
    lats = tuple(np.round(np.arange(-4.975, 5.0, 0.05), 3))
    old = tmp_path / "old.zarr"
    new = tmp_path / "new.zarr"
    out = tmp_path / "out.zarr"
    make_gridded(lats=lats).to_zarr(old, mode="w", consolidated=True)
    seen = {}

    @weather_skill(name="passthrough", version="0.1.0")
    @weather_skill.argument("-i", "--input", type=Dataset("observations"), required=True)
    def passthrough(ds, output, **kwargs):
        return ds

    @weather_skill(name="pair", version="0.1.0")
    @weather_skill.argument(
        "-i", "--input", action="append", type=Dataset("observations"), required=True
    )
    def pair(ds, output, **kwargs):
        seen["joined"] = xr.align(ds[0], ds[1], join="inner")[0]
        return ds[0]

    passthrough(["-i", str(old), "-o", str(new)])
    pair(["-i", str(new), "-i", str(old), "-o", str(out)])
    assert seen["joined"].sizes["latitude"] == len(lats)
