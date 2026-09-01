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
from weather_skills_core.provenance import HISTORY_ATTR, load_figure_history, load_history
from weather_skills_core.standard_args import rewrite_bbox_argv


def test_rewrite_bbox_equals_form():
    assert rewrite_bbox_argv(["--bbox", "-10/20/-20/30"]) == ["--bbox=-10/20/-20/30"]


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
    assert written["tp"].attrs["standard_name"] == "lwe_thickness_of_precipitation_amount"


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
