import argparse
import json
import sys
from dataclasses import fields
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from conftest import make_forecast, make_gridded, make_series, make_station

from weather_skills_core import (
    DATE_GRAMMAR,
    DataError,
    EntryOverride,
    RunContext,
    UsageError,
    envelope,
    provenance,
    set_source,
    types,
    weather_skill,
)
from weather_skills_core.decorator import rewrite_bbox_argv


class FakeFigure:
    """Stand-in for a matplotlib Figure; captures the savefig call."""

    def __init__(self):
        self.saved = None

    def savefig(self, path, metadata=None, **kwargs):
        self.saved = {"path": path, "metadata": metadata, "kwargs": kwargs}
        Path(path).write_bytes(b"\x89PNG fake")


def history_of(store):
    ds = xr.open_zarr(store, consolidated=True)
    return json.loads(ds.attrs["weather_skills_history"])


def make_identity_skill(calls, **declaration):
    declaration.setdefault("input_type", types.GRIDDED)
    declaration.setdefault("output_type", types.GRIDDED)

    @weather_skill("identity", "0.1.0", **declaration)
    def identity(ds, args):
        """Copy the input envelope unchanged."""
        # vars() records the delivered namespace as a dict for comparison, and
        # raises on a plain dict, so the recording pins the delivery type too.
        calls.append(vars(args))
        return ds.copy()

    return identity


class TestParserConstruction:
    def test_help_contains_version_epilog_and_description(self, capsys):
        skill = make_identity_skill([])
        with pytest.raises(SystemExit) as exc:
            skill(["--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "skill version: 0.1.0" in out
        assert "Copy the input envelope unchanged." in out

    def test_standard_flags(self):
        skill = make_identity_skill(
            [], variable=types.SINGLE, title=True, dims=True, time_dim="time", bbox=types.OPTIONAL
        )
        args = skill.parser.parse_args(
            ["-i", "a", "-o", "b", "-v", "precip", "--title", "T", "--dims", "y,x"]
        )
        assert args.variable == "precip"
        assert args.title == "T"
        assert args.dims == "y,x"
        assert args.time_dim == "time"  # declared default
        assert args.bbox is None

    def test_repeatable_variable(self):
        skill = make_identity_skill([], variable=types.REPEAT)
        args = skill.parser.parse_args(["-i", "a", "-o", "b", "-v", "x", "-v", "y"])
        assert args.variable == ["x", "y"]

    def test_extra_args_kwargs_pass_through_to_argparse(self):
        skill = make_identity_skill(
            [],
            extra_args=[
                ("--method", {"choices": ["mean", "std"], "required": True}),
                ("--dim", {"action": "append"}),
                ("--factor", "-f", {"type": int}),
            ],
        )
        args = skill.parser.parse_args(
            ["-i", "a", "-o", "b", "--method", "std", "--dim", "x", "--dim", "y", "-f", "3"]
        )
        assert (args.method, args.dim, args.factor) == ("std", ["x", "y"], 3)
        with pytest.raises(SystemExit) as exc:
            skill.parser.parse_args(["-i", "a", "-o", "b", "--method", "median"])
        assert exc.value.code == 2

    def test_extra_args_entry_needs_no_kwargs(self):
        skill = make_identity_skill([], extra_args=[("--cc",)])
        assert skill.parser.parse_args(["-i", "a", "-o", "b"]).cc is None

    def test_extra_args_positional_is_a_bare_name(self):
        @weather_skill("resolve-region", "0.1.0", extra_args=[("code", {"metavar": "CODE"})])
        def resolve_region(args):
            """Resolve a country code."""

        assert resolve_region.parser.parse_args(["KEN"]).code == "KEN"

    def test_dashed_positional_keeps_its_dashes(self, tmp_path, gridded_store):
        # argparse rewrites dashes to underscores for optionals only, so a
        # dashed positional's dest -- and the key its value arrives under --
        # keeps the dash.
        calls = []
        skill = make_identity_skill(calls, extra_args=[("target-grid",)])
        args = skill.parser.parse_args(["-i", "a", "-o", "b", "ref.zarr"])
        assert getattr(args, "target-grid") == "ref.zarr"
        skill(["-i", str(gridded_store), "-o", str(tmp_path / "o.zarr"), "ref.zarr"])
        assert calls == [{"target-grid": "ref.zarr"}]

    def test_extra_args_dest_keyword_decouples_dest_from_flag(self):
        skill = make_identity_skill([], extra_args=[("--from", {"dest": "sender"})])
        assert skill.parser.parse_args(["-i", "a", "-o", "b", "--from", "x"]).sender == "x"

    def test_extra_args_entry_must_be_a_tuple(self):
        with pytest.raises(ValueError, match="tuple of add_argument arguments"):
            make_identity_skill([], extra_args=["--factor"])

    def test_extra_args_duplicate_dests_rejected(self):
        with pytest.raises(ValueError, match="more than once"):
            make_identity_skill([], extra_args=[("--factor",), ("--factor", "-f")])

    def test_extra_args_reserved_dest_collisions_rejected(self):
        with pytest.raises(ValueError, match="collide"):
            make_identity_skill([], start_time=True, end_time=True, extra_args=[("--start-time",)])
        with pytest.raises(ValueError, match="collide"):
            make_identity_skill([], date=True, extra_args=[("--date",)])
        with pytest.raises(ValueError, match="collide"):
            make_identity_skill([], bbox=types.OPTIONAL, extra_args=[("--bbox",)])

    def test_extra_args_dest_allowed_when_toggle_off(self, tmp_path, gridded_store):
        # Without the corresponding toggle the name is not resolved by the
        # decorator, so an extra arg may use it.
        calls = []
        skill = make_identity_skill(calls, extra_args=[("--date",)])
        skill(["-i", str(gridded_store), "-o", str(tmp_path / "o.zarr"), "--date", "x"])
        assert calls == [{"date": "x"}]

    def test_input_help_on_single_input_flag(self, capsys):
        skill = make_identity_skill([], input_help="Gridded Zarr to copy.")
        with pytest.raises(SystemExit):
            skill(["--help"])
        assert "Gridded Zarr to copy." in capsys.readouterr().out

    def test_input_help_on_repeated_input_flag(self, capsys):
        @weather_skill(
            "difference",
            "0.1.0",
            input_type=[types.ALL, types.ALL],
            output_type=types.GRIDDED,
            input_help="Input Zarr; pass exactly twice, minuend then subtrahend.",
        )
        def difference(ds_a, ds_b, args):
            """A - B."""

        with pytest.raises(SystemExit):
            difference(["--help"])
        assert "minuend" in capsys.readouterr().out

    def test_input_help_on_variadic_input_flag(self, capsys):
        @weather_skill(
            "concat",
            "0.1.0",
            input_type=types.ALL,
            output_type=types.GRIDDED,
            variadic_input=True,
            input_help="Input Zarr; repeat in concatenation order.",
        )
        def concat(datasets, args):
            """Concatenate."""

        with pytest.raises(SystemExit):
            concat(["--help"])
        assert "concatenation order" in capsys.readouterr().out

    def test_input_help_on_named_input_flags(self, capsys):
        @weather_skill(
            "plot-mediogram",
            "0.1.0",
            input_type=[types.ALL, types.ALL],
            output_type=types.PNG,
            input_names=["forecast", "mclimate"],
            input_help=["Forecast ensemble Zarr.", "M-climate ensemble Zarr."],
        )
        def plot_mediogram(forecast_ds, mclimate_ds, args):
            """Mediogram."""

        with pytest.raises(SystemExit):
            plot_mediogram(["--help"])
        out = capsys.readouterr().out
        assert "Forecast ensemble Zarr." in out
        assert "M-climate ensemble Zarr." in out

    def test_input_help_none_entry_skips_one_flag(self, capsys):
        @weather_skill(
            "plot-mediogram",
            "0.1.0",
            input_type=[types.ALL, types.ALL],
            output_type=types.PNG,
            input_names=["forecast", "mclimate"],
            input_help=[None, "M-climate ensemble Zarr."],
        )
        def plot_mediogram(forecast_ds, mclimate_ds, args):
            """Mediogram."""

        with pytest.raises(SystemExit):
            plot_mediogram(["--help"])
        assert "M-climate ensemble Zarr." in capsys.readouterr().out

    def test_input_help_declaration_errors(self):
        with pytest.raises(ValueError, match="input_help requires"):
            weather_skill("x", "0.1.0", output_type=types.GRIDDED, input_help="h")(lambda: None)
        with pytest.raises(ValueError, match="one help string per input flag"):
            weather_skill(
                "x",
                "0.1.0",
                input_type=[types.ALL, types.ALL],
                output_type=types.PNG,
                input_names=["a", "b"],
                input_help="one string for two flags",
            )(lambda a, b: None)
        with pytest.raises(ValueError, match="one help string per input flag"):
            weather_skill(
                "x",
                "0.1.0",
                input_type=[types.ALL, types.ALL],
                output_type=types.PNG,
                input_names=["a", "b"],
                input_help=["only one"],
            )(lambda a, b: None)
        with pytest.raises(ValueError, match="single help string"):
            weather_skill(
                "x",
                "0.1.0",
                input_type=types.ALL,
                output_type=types.GRIDDED,
                input_help=["list", "form"],
            )(lambda ds: None)

    def test_missing_required_input_exits_2(self):
        skill = make_identity_skill([])
        with pytest.raises(SystemExit) as exc:
            skill(["-o", "b"])
        assert exc.value.code == 2

    def test_declaration_errors(self):
        with pytest.raises(ValueError, match="output_type"):
            weather_skill("x", "0.1.0", output_type="zar")(lambda: None)
        with pytest.raises(ValueError, match="together"):
            weather_skill("x", "0.1.0", output_type=types.GRIDDED, start_time=True)(lambda: None)
        with pytest.raises(ValueError, match="history label"):
            weather_skill("x", "0.1.0", input_type=[types.ALL, types.ALL], output_type=types.PNG)(
                lambda a, b: None
            )

    def test_unknown_input_type_is_declaration_error(self):
        with pytest.raises(ValueError, match="unknown envelope type"):
            weather_skill("x", "0.1.0", input_type="grided", output_type=types.GRIDDED)(
                lambda ds: None
            )
        with pytest.raises(ValueError, match="unknown envelope type"):
            weather_skill(
                "x", "0.1.0", input_type=(types.GRIDDED, "forecst"), output_type=types.GRIDDED
            )(lambda ds: None)
        with pytest.raises(ValueError, match="unknown envelope type"):
            weather_skill(
                "x", "0.1.0", input_type=[types.GRIDDED, "statoin"], output_type=types.GRIDDED
            )(lambda a, b: None)


class TestTypeUnions:
    """The shape of the declaration says how many inputs there are."""

    def store(self, tmp_path, ds, name):
        path = tmp_path / name
        ds.to_zarr(path, mode="w", consolidated=True)
        return str(path)

    def test_tuple_is_one_input_accepting_any_member(self, tmp_path):
        calls = []
        skill = make_identity_skill(calls, input_type=(types.GRIDDED, types.FORECAST))
        for i, ds in enumerate((make_gridded(), make_forecast())):
            skill(
                ["-i", self.store(tmp_path, ds, f"in{i}.zarr"), "-o", str(tmp_path / f"o{i}.zarr")]
            )
        assert len(calls) == 2

    def test_tuple_union_rejects_a_non_member(self, tmp_path, capsys):
        skill = make_identity_skill([], input_type=(types.GRIDDED, types.FORECAST))
        with pytest.raises(SystemExit) as exc:
            skill(
                [
                    "-i",
                    self.store(tmp_path, make_station(), "st.zarr"),
                    "-o",
                    str(tmp_path / "o.zarr"),
                ]
            )
        assert exc.value.code == 2
        assert "expects gridded or forecast" in capsys.readouterr().err

    def test_list_is_one_input_per_entry(self, tmp_path):
        seen = []

        @weather_skill(
            "pair", "0.1.0", input_type=[types.GRIDDED, types.FORECAST], output_type=types.GRIDDED
        )
        def pair(a, b, args):
            """Two inputs, one declared type each."""
            seen.append((envelope.detect_type(a), envelope.detect_type(b)))
            return a.copy()

        pair(
            [
                "-i",
                self.store(tmp_path, make_gridded(), "a.zarr"),
                "-i",
                self.store(tmp_path, make_forecast(), "b.zarr"),
                "-o",
                str(tmp_path / "o.zarr"),
            ]
        )
        assert seen == [(types.GRIDDED, types.FORECAST)]

    def test_list_entry_may_itself_be_a_union(self, tmp_path):
        skill = weather_skill(
            "pair",
            "0.1.0",
            input_type=[types.ALL, (types.GRIDDED, types.FORECAST)],
            output_type=types.GRIDDED,
        )(lambda a, b: None)
        assert len(skill.parser.parse_args(["-i", "x", "-i", "y", "-o", "z"]).input) == 2

    def test_output_type_rejects_a_list(self):
        with pytest.raises(ValueError, match="not a list"):
            weather_skill("x", "0.1.0", output_type=[types.GRIDDED, types.FORECAST])(lambda: None)

    def test_pipe_string_is_not_a_union(self):
        with pytest.raises(ValueError, match="unknown envelope type"):
            weather_skill("x", "0.1.0", input_type="gridded|forecast", output_type=types.GRIDDED)(
                lambda ds: None
            )

    def test_comma_string_is_not_two_inputs(self):
        with pytest.raises(ValueError, match="unknown envelope type"):
            weather_skill("x", "0.1.0", input_type="gridded,gridded", output_type=types.GRIDDED)(
                lambda a, b: None
            )


class TestDateGrammarHelp:
    """Core owns the date grammar, so core puts it in the help."""

    def help_text(self, skill, capsys):
        with pytest.raises(SystemExit):
            skill(["--help"])
        return " ".join(capsys.readouterr().out.split())

    def test_appended_to_start_and_date(self, capsys):
        grammar = " ".join(DATE_GRAMMAR.split())
        starter = make_identity_skill([], start_time=True, end_time=True)
        assert grammar in self.help_text(starter, capsys)
        dated = make_identity_skill([], date=True)
        assert grammar in self.help_text(dated, capsys)

    def test_end_cross_references_instead_of_repeating(self, capsys):
        skill = make_identity_skill([], start_time=True, end_time=True)
        text = self.help_text(skill, capsys)
        # Exactly once per --help: --end points at --start rather than
        # printing the grammar a second time.
        assert text.count(" ".join(DATE_GRAMMAR.split())) == 1
        assert "Same date grammar as --start." in text

    def test_a_skill_note_precedes_the_grammar_without_restating_it(self, capsys):
        # The smallest spelling for source-specific content: override `help`
        # with the flag's own sentence; core appends the grammar to it.
        skill = make_identity_skill(
            [],
            start_time={"help": "Start date. 'latest' is the current UTC date here."},
            end_time=True,
        )
        text = self.help_text(skill, capsys)
        note = "Start date. 'latest' is the current UTC date here."
        assert f"{note} {' '.join(DATE_GRAMMAR.split())}" in text
        assert text.count(" ".join(DATE_GRAMMAR.split())) == 1


class TestToggleDictForm:
    def test_variable_help_required_and_choices(self, capsys):
        skill = make_identity_skill(
            [],
            variable={
                "mode": types.SINGLE,
                "help": "CMIP6 variable_id (e.g. tas, pr).",
                "required": True,
                "choices": ["tas", "pr"],
            },
        )
        with pytest.raises(SystemExit):
            skill(["--help"])
        assert "CMIP6 variable_id" in capsys.readouterr().out
        args = skill.parser.parse_args(["-i", "a", "-o", "b", "-v", "tas"])
        assert args.variable == "tas"
        with pytest.raises(SystemExit) as exc:
            skill.parser.parse_args(["-i", "a", "-o", "b"])
        assert exc.value.code == 2  # required
        with pytest.raises(SystemExit) as exc:
            skill.parser.parse_args(["-i", "a", "-o", "b", "-v", "nope"])
        assert exc.value.code == 2  # choices

    def test_variable_repeat_mode_in_dict_form(self):
        skill = make_identity_skill([], variable={"mode": types.REPEAT, "help": "Repeatable."})
        args = skill.parser.parse_args(["-i", "a", "-o", "b", "-v", "x", "-v", "y"])
        assert args.variable == ["x", "y"]

    def test_bbox_dict_help_and_optional_mode(self, tmp_path, gridded_store, capsys):
        calls = []
        skill = make_identity_skill(
            calls, bbox={"mode": types.OPTIONAL, "help": "Custom bbox help."}
        )
        with pytest.raises(SystemExit):
            skill(["--help"])
        assert "Custom bbox help." in capsys.readouterr().out
        skill(["-i", str(gridded_store), "-o", str(tmp_path / "o.zarr")])
        assert calls == [{"bbox": None}]

    def test_bbox_dict_required_mode_still_rewrites_argv(self, tmp_path, gridded_store):
        calls = []
        skill = make_identity_skill(calls, input_type=types.ALL, bbox={"mode": types.REQUIRED})
        skill(["-i", str(gridded_store), "-o", str(tmp_path / "o.zarr"), "--bbox", "-1/32/-5/42"])
        assert calls[0]["bbox"] == (-1.0, 32.0, -5.0, 42.0)

    def test_workers_dict_default_help_and_choices(self, capsys):
        skill = make_identity_skill(
            [], workers={"default": 2, "help": "Parallel legs.", "choices": [1, 2, 4]}
        )
        with pytest.raises(SystemExit):
            skill(["--help"])
        assert "Parallel legs." in capsys.readouterr().out
        assert skill.parser.parse_args(["-i", "a", "-o", "b"]).workers == 2
        with pytest.raises(SystemExit) as exc:
            skill.parser.parse_args(["-i", "a", "-o", "b", "--workers", "3"])
        assert exc.value.code == 2

    def test_start_end_dict_help_and_optional(self, tmp_path, gridded_store, capsys):
        skill = make_identity_skill(
            [],
            start_time={"help": "Range start (analysis datasets).", "required": False},
            end_time={"help": "Range end (analysis datasets).", "required": False},
        )
        with pytest.raises(SystemExit):
            skill(["--help"])
        assert "analysis datasets" in capsys.readouterr().out
        calls = []
        skill = make_identity_skill(
            calls,
            start_time={"required": False},
            end_time={"required": False},
        )
        out = tmp_path / "o.zarr"
        skill(["-i", str(gridded_store), "-o", str(out)])
        assert calls == [{"start_time": None, "end_time": None}]
        # No resolved dates recorded when the optional window is omitted.
        recorded = history_of(out)[-1]["args"]
        assert recorded["start"] is None
        assert recorded["end"] is None

    def test_lone_optional_start_exits_2(self, tmp_path, gridded_store, capsys):
        skill = make_identity_skill(
            [],
            start_time={"required": False},
            end_time={"required": False},
        )
        with pytest.raises(SystemExit) as exc:
            skill(
                [
                    "-i",
                    str(gridded_store),
                    "-o",
                    str(tmp_path / "o.zarr"),
                    "--start",
                    "2026-01-01",
                ]
            )
        assert exc.value.code == 2
        assert "given together" in capsys.readouterr().err

    def test_optional_window_still_resolves_when_given(self, tmp_path, gridded_store):
        calls = []
        skill = make_identity_skill(
            calls,
            start_time={"required": False},
            end_time={"required": False},
        )
        out = tmp_path / "o.zarr"
        skill(
            [
                "-i",
                str(gridded_store),
                "-o",
                str(out),
                "--start",
                "2026-01-01",
                "--end",
                "2026-01-05",
            ]
        )
        assert calls[0]["start_time"] == date(2026, 1, 1)
        assert history_of(out)[-1]["args"]["end"] == "2026-01-05"

    def test_date_context_labels_resolution_log(self, tmp_path, capsys):
        @weather_skill(
            "f",
            "0.1.0",
            output_type=types.GRIDDED,
            date={"context": "single forecast init date"},
        )
        def fetch(args):
            """Fetch one init."""
            return make_gridded()

        fetch(["--date", "now-1d", "-o", str(tmp_path / "o.zarr")])
        assert "(single forecast init date)" in capsys.readouterr().err

    def test_date_dict_optional_passes_none(self, tmp_path):
        calls = []

        @weather_skill("f", "0.1.0", output_type=types.GRIDDED, date={"required": False})
        def fetch(args):
            """Fetch."""
            calls.append(args.date)
            return make_gridded()

        out = tmp_path / "o.zarr"
        fetch(["-o", str(out)])
        assert calls == [None]
        assert history_of(out)[0]["args"]["date"] is None

    def test_declaration_errors(self):
        def declare(**kwargs):
            return weather_skill("x", "0.1.0", output_type=types.GRIDDED, **kwargs)(lambda: None)

        with pytest.raises(ValueError, match="unknown keys"):
            declare(start_time={"metavar": "S"}, end_time=True)
        with pytest.raises(ValueError, match="unknown keys"):
            declare(date={"mode": types.SINGLE})
        with pytest.raises(ValueError, match="context"):
            # `context` is a date-only key; start_time does not accept it.
            declare(start_time={"context": "x"}, end_time=True)
        with pytest.raises(ValueError, match="mode must be one of"):
            declare(bbox={"help": "no mode"})
        with pytest.raises(ValueError, match="mode must be one of"):
            declare(variable={"mode": "many"})
        with pytest.raises(ValueError, match="unknown keys"):
            declare(workers={"defualt": 4})
        with pytest.raises(ValueError, match="agree on required"):
            declare(start_time={"required": False}, end_time=True)
        with pytest.raises(ValueError, match="together"):
            declare(start_time={"required": False})


class TestMutexGroups:
    def make_select_skill(self, calls, required=True):
        return make_identity_skill(
            calls,
            extra_args=[("--index", {"type": int}), ("--value",)],
            mutex_groups={"selector": {"args": ("index", "value"), "required": required}},
        )

    def test_one_member_parses_and_reaches_function(self, tmp_path, gridded_store):
        calls = []
        skill = self.make_select_skill(calls)
        skill(["-i", str(gridded_store), "-o", str(tmp_path / "o.zarr"), "--index", "3"])
        assert calls == [{"index": 3, "value": None}]

    def test_two_members_exit_2(self, tmp_path, gridded_store, capsys):
        skill = self.make_select_skill([])
        with pytest.raises(SystemExit) as exc:
            skill(
                [
                    "-i",
                    str(gridded_store),
                    "-o",
                    str(tmp_path / "o.zarr"),
                    "--index",
                    "3",
                    "--value",
                    "x",
                ]
            )
        assert exc.value.code == 2
        assert "not allowed with" in capsys.readouterr().err

    def test_required_group_with_no_member_exits_2(self, tmp_path, gridded_store, capsys):
        skill = self.make_select_skill([])
        with pytest.raises(SystemExit) as exc:
            skill(["-i", str(gridded_store), "-o", str(tmp_path / "o.zarr")])
        assert exc.value.code == 2
        assert "is required" in capsys.readouterr().err

    def test_optional_group_allows_no_member(self, tmp_path, gridded_store):
        calls = []
        skill = self.make_select_skill(calls, required=False)
        skill(["-i", str(gridded_store), "-o", str(tmp_path / "o.zarr")])
        assert calls == [{"index": None, "value": None}]

    def test_sequence_shorthand_is_optional_group(self, tmp_path, gridded_store):
        skill = make_identity_skill(
            [],
            extra_args=[("--index", {"type": int}), ("--value",)],
            mutex_groups={"selector": ("index", "value")},
        )
        with pytest.raises(SystemExit) as exc:
            skill.parser.parse_args(["-i", "a", "-o", "b", "--index", "1", "--value", "x"])
        assert exc.value.code == 2
        args = skill.parser.parse_args(["-i", "a", "-o", "b"])
        assert args.index is None

    def test_usage_renders_group_brackets(self):
        skill = self.make_select_skill([])
        assert "(--index INDEX | --value VALUE)" in skill.parser.format_usage()

    def test_aliases_work_inside_group(self, tmp_path, gridded_store):
        calls = []
        skill = make_identity_skill(
            calls,
            extra_args=[
                ("--factor", "-f", {"type": float}),
                ("--target-resolution", {"type": float}),
                ("--reference-grid",),
            ],
            mutex_groups={
                "target": {
                    "args": ("factor", "target_resolution", "reference_grid"),
                    "required": True,
                }
            },
        )
        skill(["-i", str(gridded_store), "-o", str(tmp_path / "o.zarr"), "-f", "2.0"])
        assert calls[0]["factor"] == 2.0
        with pytest.raises(SystemExit):
            skill.parser.parse_args(["-i", "a", "-o", "b", "-f", "2", "--reference-grid", "g"])

    def test_declaration_errors(self):
        def declare(**kwargs):
            return weather_skill("x", "0.1.0", **kwargs)(lambda: None)

        with pytest.raises(ValueError, match="not an extra_args dest"):
            declare(extra_args=[("--a", {"type": int})], mutex_groups={"g": ("a", "b")})
        with pytest.raises(ValueError, match="at least two"):
            declare(extra_args=[("--a", {"type": int})], mutex_groups={"g": ("a",)})
        with pytest.raises(ValueError, match="both mutex groups"):
            declare(
                extra_args=[("--a", {"type": int}), ("--b", {"type": int}), ("--c", {"type": int})],
                mutex_groups={"g1": ("a", "b"), "g2": ("a", "c")},
            )
        with pytest.raises(ValueError, match="positional"):
            declare(
                extra_args=[("a",), ("--b", {"type": int})],
                mutex_groups={"g": ("a", "b")},
            )
        with pytest.raises(ValueError, match="on the group"):
            declare(
                extra_args=[("--a", {"required": True}), ("--b", {"type": int})],
                mutex_groups={"g": ("a", "b")},
            )
        with pytest.raises(ValueError, match="unknown keys"):
            declare(
                extra_args=[("--a", {"type": int}), ("--b", {"type": int})],
                mutex_groups={"g": {"args": ("a", "b"), "exclusive": True}},
            )
        with pytest.raises(ValueError, match="under 'args'"):
            declare(
                extra_args=[("--a", {"type": int}), ("--b", {"type": int})],
                mutex_groups={"g": {"required": True}},
            )


class TestBboxArgv:
    def test_rewrite(self):
        assert rewrite_bbox_argv(["--bbox", "-1/32/-5/42", "-o", "x"]) == [
            "--bbox=-1/32/-5/42",
            "-o",
            "x",
        ]

    def test_negative_north_parses_end_to_end(self, tmp_path, gridded_store):
        seen = {}

        @weather_skill(
            "clip", "0.1.0", input_type=types.ALL, output_type=types.GRIDDED, bbox=types.REQUIRED
        )
        def clip(ds, args):
            """Clip."""
            seen["bbox"] = args.bbox
            return ds.copy()

        clip(["-i", str(gridded_store), "-o", str(tmp_path / "o.zarr"), "--bbox", "-1/32/-5/42"])
        assert seen["bbox"] == (-1.0, 32.0, -5.0, 42.0)


class TestArgumentDelivery:
    def test_arguments_arrive_as_a_namespace(self, tmp_path, gridded_store):
        seen = {}

        @weather_skill(
            "deliver",
            "0.1.0",
            input_type=types.ALL,
            output_type=types.GRIDDED,
            bbox=types.REQUIRED,
            variable=types.SINGLE,
            extra_args=[("--factor", {"type": float})],
        )
        def deliver(ds, args):
            """Read every argument off the namespace."""
            seen["type"] = type(args)
            seen["values"] = (args.bbox, args.variable, args.factor)
            return ds.copy()

        deliver(
            [
                "-i",
                str(gridded_store),
                "-o",
                str(tmp_path / "o.zarr"),
                "--bbox",
                "3/10/1/13",
                "-v",
                "precip",
                "--factor",
                "2.5",
            ]
        )
        assert seen["type"] is argparse.Namespace
        assert seen["values"] == ((3.0, 10.0, 1.0, 13.0), "precip", 2.5)

    def test_delivery_does_not_touch_the_recorded_entry(self, tmp_path, gridded_store):
        # The namespace is built from the resolved params, never by writing
        # them back onto the parsed args: the entry records the raw --bbox
        # string and the raw date tokens. A 4-tuple here would be sorted by
        # _order_lists, giving two different spatial subsets one cache key.
        @weather_skill(
            "record",
            "0.1.0",
            input_type=types.ALL,
            output_type=types.GRIDDED,
            bbox=types.REQUIRED,
            start_time=True,
            end_time=True,
        )
        def record(ds, args):
            """Touch every resolved value, then return the input."""
            assert args.bbox == (3.0, 10.0, 1.0, 13.0)
            assert args.start_time == date(2026, 1, 1)
            return ds.copy()

        out = tmp_path / "o.zarr"
        record(
            [
                "-i",
                str(gridded_store),
                "-o",
                str(out),
                "--bbox",
                "3/10/1/13",
                "--start",
                "2026-01-01",
                "--end",
                "2026-01-05",
            ]
        )
        assert history_of(out)[0]["args"] == {
            "bbox": "3/10/1/13",
            "start": "2026-01-01",
            "end": "2026-01-05",
        }


class TestCacheShortCircuit:
    def test_second_run_skips_function_and_store(self, tmp_path, gridded_store, capsys):
        calls = []
        skill = make_identity_skill(calls)
        out = tmp_path / "out.zarr"
        skill(["-i", str(gridded_store), "-o", str(out)])
        assert len(calls) == 1
        first_history = history_of(out)
        skill(["-i", str(gridded_store), "-o", str(out)])
        assert len(calls) == 1
        assert "Cache hit" in capsys.readouterr().err
        assert history_of(out) == first_history

    def test_changed_extra_arg_reruns(self, tmp_path, gridded_store):
        calls = []
        skill = make_identity_skill(calls, extra_args=[("--factor", {"type": int})])
        out = tmp_path / "out.zarr"
        skill(["-i", str(gridded_store), "-o", str(out), "--factor", "1"])
        skill(["-i", str(gridded_store), "-o", str(out), "--factor", "2"])
        assert len(calls) == 2

    def test_modified_input_reruns_with_hash(self, tmp_path, gridded_store):
        calls = []
        skill = make_identity_skill(calls)
        out = tmp_path / "out.zarr"
        skill(["-i", str(gridded_store), "-o", str(out)])
        make_gridded(fill=5.0).to_zarr(gridded_store, mode="w", consolidated=True)
        skill(["-i", str(gridded_store), "-o", str(out)])
        assert len(calls) == 2

    def test_modified_input_still_hits_without_hash_compare(self, tmp_path, gridded_store):
        calls = []
        skill = make_identity_skill(calls, hash_input=False)
        out = tmp_path / "out.zarr"
        skill(["-i", str(gridded_store), "-o", str(out)])
        # The stamped entry still carries a hash even though the check skips it.
        assert "hash" in history_of(out)[-1]["input"]
        make_gridded(fill=5.0).to_zarr(gridded_store, mode="w", consolidated=True)
        skill(["-i", str(gridded_store), "-o", str(out)])
        assert len(calls) == 1

    def test_workers_excluded_from_cache_key(self, tmp_path, gridded_store):
        calls = []
        skill = make_identity_skill(calls, workers=4)
        out = tmp_path / "out.zarr"
        skill(["-i", str(gridded_store), "-o", str(out), "--workers", "2"])
        skill(["-i", str(gridded_store), "-o", str(out), "--workers", "8"])
        assert len(calls) == 1
        assert "workers" not in history_of(out)[-1]["args"]

    def test_list_args_are_ordered_without_a_hook(self, tmp_path, gridded_store):
        # Core orders list-valued entry args, so flag order alone cannot cause
        # a cache miss. No normalize_args hook involved.
        calls = []
        skill = make_identity_skill(calls, variable=types.REPEAT)
        out = tmp_path / "out.zarr"
        skill(["-i", str(gridded_store), "-o", str(out), "-v", "b", "-v", "a"])
        skill(["-i", str(gridded_store), "-o", str(out), "-v", "a", "-v", "b"])
        assert len(calls) == 1
        assert history_of(out)[-1]["args"]["variable"] == ["a", "b"]

    def test_list_args_are_ordered_but_never_deduped(self, tmp_path, gridded_store):
        # A value given twice is recorded twice: ordering is a canonical
        # spelling of what was asked for, not a rewrite of it.
        calls = []
        skill = make_identity_skill(calls, variable=types.REPEAT)
        out = tmp_path / "out.zarr"
        skill(["-i", str(gridded_store), "-o", str(out), "-v", "b", "-v", "a", "-v", "a"])
        assert history_of(out)[-1]["args"]["variable"] == ["a", "a", "b"]

    def test_ordering_runs_after_the_normalize_hook(self, tmp_path, gridded_store):
        # The guarantee has to hold for a list the hook itself produced, so
        # ordering is last. A hook that needs to keep an order records a
        # string instead, which ordering leaves alone.
        calls = []

        def normalize(args):
            args["ordered"] = ",".join(args.get("variable") or ())
            args["variable"] = ["z", "y"]
            return args

        skill = make_identity_skill(calls, variable=types.REPEAT, normalize_args=normalize)
        out = tmp_path / "out.zarr"
        skill(["-i", str(gridded_store), "-o", str(out), "-v", "b", "-v", "a"])
        recorded = history_of(out)[-1]["args"]
        assert recorded["variable"] == ["y", "z"]
        assert recorded["ordered"] == "b,a"

    def test_mixed_type_list_keeps_its_given_order(self, tmp_path, gridded_store):
        # Sorting has no total order across types; recording it unchanged
        # beats crashing the provenance stamp.
        calls = []

        def normalize(args):
            args["mixed"] = [2, "a"]
            return args

        skill = make_identity_skill(calls, normalize_args=normalize)
        out = tmp_path / "out.zarr"
        skill(["-i", str(gridded_store), "-o", str(out)])
        assert history_of(out)[-1]["args"]["mixed"] == [2, "a"]

    def test_preserve_order_exempts_a_dest_from_the_sort(self, tmp_path, gridded_store):
        # An argument whose order changes the output must keep the order given:
        # sorting it would hand two different requests one cache key.
        calls = []
        skill = make_identity_skill(
            calls,
            extra_args=[("--index", {"action": "append"})],
            preserve_order=("index",),
        )
        out = tmp_path / "a.zarr"
        skill(["-i", str(gridded_store), "-o", str(out), "--index", "2", "--index", "0"])
        assert history_of(out)[-1]["args"]["index"] == ["2", "0"]

    def test_preserved_order_keeps_two_orderings_distinct(self, tmp_path, gridded_store):
        calls = []
        skill = make_identity_skill(
            calls,
            extra_args=[("--index", {"action": "append"})],
            preserve_order=("index",),
        )
        out = tmp_path / "a.zarr"
        skill(["-i", str(gridded_store), "-o", str(out), "--index", "2", "--index", "0"])
        skill(["-i", str(gridded_store), "-o", str(out), "--index", "0", "--index", "2"])
        assert len(calls) == 2  # recomputed, not a cache hit
        assert history_of(out)[-1]["args"]["index"] == ["0", "2"]

    def test_preserve_order_leaves_other_lists_sorted(self, tmp_path, gridded_store):
        calls = []
        skill = make_identity_skill(
            calls,
            variable=types.REPEAT,
            extra_args=[("--index", {"action": "append"})],
            preserve_order=("index",),
        )
        out = tmp_path / "a.zarr"
        skill(
            ["-i", str(gridded_store), "-o", str(out)]
            + ["--index", "2", "--index", "0", "-v", "b", "-v", "a"]
        )
        recorded = history_of(out)[-1]["args"]
        assert recorded["index"] == ["2", "0"]
        assert recorded["variable"] == ["a", "b"]

    def test_preserve_order_rejects_an_unknown_dest(self):
        # A typo here would silently sort an order-significant argument, which
        # is the bug the declaration exists to prevent.
        with pytest.raises(ValueError, match="preserve_order names"):
            make_identity_skill(
                [], extra_args=[("--index", {"action": "append"})], preserve_order=("indices",)
            )

    def test_normalize_args_tuple_still_hits(self, tmp_path, gridded_store):
        # A tuple from the normalize hook stamps as a JSON list; the compared
        # entry must go through the same canonicalization or the store would
        # never match its own cache key again.
        calls = []

        def normalize(args):
            if args.get("variable"):
                args["variable"] = tuple(sorted(set(args["variable"])))
            return args

        skill = make_identity_skill(calls, variable=types.REPEAT, normalize_args=normalize)
        out = tmp_path / "out.zarr"
        skill(["-i", str(gridded_store), "-o", str(out), "-v", "b", "-v", "a"])
        skill(["-i", str(gridded_store), "-o", str(out), "-v", "a", "-v", "b"])
        assert len(calls) == 1
        assert history_of(out)[-1]["args"]["variable"] == ["a", "b"]

    def test_exclude_args(self, tmp_path, gridded_store):
        calls = []
        skill = make_identity_skill(
            calls, extra_args=[("--verbose", {"action": "store_true"})], exclude_args=("verbose",)
        )
        out = tmp_path / "out.zarr"
        skill(["-i", str(gridded_store), "-o", str(out)])
        skill(["-i", str(gridded_store), "-o", str(out), "--verbose"])
        assert len(calls) == 1

    def test_completeness_probe_rejects_transform_hit(self, tmp_path, gridded_store):
        calls = []
        skill = make_identity_skill(calls, completeness_probe=lambda p: False)
        out = tmp_path / "out.zarr"
        argv = ["-i", str(gridded_store), "-o", str(out)]
        skill(argv)
        skill(argv)
        assert len(calls) == 2

    def test_completeness_probe_accepts_transform_hit(self, tmp_path, gridded_store):
        calls = []
        probed = []
        skill = make_identity_skill(calls, completeness_probe=lambda p: probed.append(p) or True)
        out = tmp_path / "out.zarr"
        argv = ["-i", str(gridded_store), "-o", str(out)]
        skill(argv)
        skill(argv)
        assert len(calls) == 1
        # The probe ran on the output store, only for the second run's check.
        assert probed == [out]

    def test_completeness_probe_on_multi_input_check(self, tmp_path):
        a = tmp_path / "a.zarr"
        b = tmp_path / "b.zarr"
        make_gridded(fill=1.0).to_zarr(a, mode="w", consolidated=True)
        make_gridded(fill=2.0).to_zarr(b, mode="w", consolidated=True)
        calls = []

        @weather_skill(
            "difference",
            "0.1.0",
            input_type=[types.ALL, types.ALL],
            output_type=types.GRIDDED,
            completeness_probe=lambda p: False,
        )
        def difference(ds_a, ds_b, args):
            """A - B."""
            calls.append(1)
            return ds_a.copy()

        argv = ["-i", str(a), "-i", str(b), "-o", str(tmp_path / "out.zarr")]
        difference(argv)
        difference(argv)
        assert len(calls) == 2

    def test_chain_appends_on_upstream(self, tmp_path, gridded_store):
        first = make_identity_skill([])
        second = make_identity_skill([])
        mid = tmp_path / "mid.zarr"
        out = tmp_path / "out.zarr"
        first(["-i", str(gridded_store), "-o", str(mid)])
        second(["-i", str(mid), "-o", str(out)])
        chain = history_of(out)
        assert len(chain) == 2
        assert chain[0] == history_of(mid)[0]
        assert chain[-1]["input"]["basename"] == "mid.zarr"


class TestValidationOverrides:
    @pytest.fixture
    def renamed_store(self, tmp_path):
        path = tmp_path / "renamed.zarr"
        ds = make_gridded().rename({"latitude": "yy", "longitude": "xx"})
        ds.to_zarr(path, mode="w", consolidated=True)
        return path

    def make_clip_skill(self, calls):
        @weather_skill(
            "clip-region",
            "0.1.0",
            input_type=types.GRIDDED,
            output_type=types.GRIDDED,
            bbox=types.REQUIRED,
            dims=True,
        )
        def clip_region(ds, args):
            """Clip."""
            from weather_skills_core.envelope import bbox_subset, detect_spatial_dims

            calls.append(args.dims)
            lat_dim, lon_dim = detect_spatial_dims(ds, args.dims)
            return bbox_subset(ds, args.bbox, lat_dim=lat_dim, lon_dim=lon_dim)

        return clip_region

    def test_undetectable_dims_rejected_without_override(self, tmp_path, renamed_store, capsys):
        skill = self.make_clip_skill([])
        with pytest.raises(SystemExit) as exc:
            skill(["-i", str(renamed_store), "-o", str(tmp_path / "o.zarr"), "--bbox", "3/10/1/13"])
        assert exc.value.code == 2
        assert "pass --dims" in capsys.readouterr().err

    def test_dims_override_validates_typed_gridded_input(self, tmp_path, renamed_store):
        calls = []
        skill = self.make_clip_skill(calls)
        out = tmp_path / "o.zarr"
        skill(
            [
                "-i",
                str(renamed_store),
                "-o",
                str(out),
                "--bbox",
                "2.5/10.5/0.5/12.5",
                "--dims",
                "yy,xx",
            ]
        )
        assert calls == ["yy,xx"]
        written = xr.open_zarr(out, consolidated=True)
        assert written.sizes["yy"] == 2

    def test_dims_override_naming_absent_dims_exits_2(self, tmp_path, renamed_store, capsys):
        skill = self.make_clip_skill([])
        with pytest.raises(SystemExit) as exc:
            skill(
                [
                    "-i",
                    str(renamed_store),
                    "-o",
                    str(tmp_path / "o.zarr"),
                    "--bbox",
                    "3/10/1/13",
                    "--dims",
                    "a,b",
                ]
            )
        assert exc.value.code == 2
        assert "not in dataset dims" in capsys.readouterr().err

    def test_validate_type_classifies_as_the_run_does(self, tmp_path, renamed_store):
        seen = []

        @weather_skill(
            "assert-shape", "0.1.0", input_type=types.ALL, output_type=types.ALL, dims=True
        )
        def assert_shape(ds, args):
            """Copy the input, asserting the copy preserves its shape."""
            out = ds.copy()
            seen.append(envelope.validate_type(out, ds, args.dims))
            return out

        assert_shape(["-i", str(renamed_store), "-o", str(tmp_path / "o.zarr"), "--dims", "yy,xx"])
        assert seen == [types.GRIDDED]

    def test_shape_drift_under_a_dims_override_is_caught(self, tmp_path, renamed_store, capsys):
        # The assertion has to bite where drift is likeliest: on axes only
        # --dims can name. Classifying the reference without the override makes
        # both sides a series and the claim vacuous.
        @weather_skill("collapse", "0.1.0", input_type=types.ALL, output_type=types.ALL, dims=True)
        def collapse(ds, args):
            """Collapse the named spatial axes, then claim the shape is preserved."""
            out = ds.mean(dim=["yy", "xx"])
            envelope.validate_type(out, ds, args.dims)
            return out

        with pytest.raises(SystemExit) as exc:
            collapse(["-i", str(renamed_store), "-o", str(tmp_path / "o.zarr"), "--dims", "yy,xx"])
        assert exc.value.code == 1
        assert "expected a gridded envelope, got series" in capsys.readouterr().err

    def test_renaming_the_axes_is_not_drift(self, tmp_path, renamed_store):
        # The other half: --dims names the INPUT's axes, so a transform that
        # writes canonical names is shape-preserving and must not raise.
        @weather_skill(
            "canonicalize", "0.1.0", input_type=types.ALL, output_type=types.ALL, dims=True
        )
        def canonicalize(ds, args):
            """Rename the named spatial axes to their canonical names."""
            out = ds.rename({"yy": "latitude", "xx": "longitude"})
            envelope.validate_type(out, ds, args.dims)
            return out

        out = tmp_path / "o.zarr"
        canonicalize(["-i", str(renamed_store), "-o", str(out), "--dims", "yy,xx"])
        assert set(xr.open_zarr(out, consolidated=True).sizes) == {"time", "latitude", "longitude"}

    def test_the_override_is_only_what_the_run_was_given(self, tmp_path, renamed_store):
        seen = []

        @weather_skill("classify", "0.1.0", input_type=types.ALL, output_type=types.ALL, dims=True)
        def classify(ds, args):
            """Record how the run classifies its input."""
            seen.append(envelope.detect_type(ds, args.dims))
            return ds.copy()

        classify(["-i", str(renamed_store), "-o", str(tmp_path / "o.zarr"), "--dims", "yy,xx"])
        classify(["-i", str(renamed_store), "-o", str(tmp_path / "o2.zarr")])
        assert seen == [types.GRIDDED, types.SERIES]

    def test_time_dim_override_validated(self, tmp_path, gridded_store, capsys):
        calls = []
        skill = make_identity_skill(calls, time_dim=True)
        with pytest.raises(SystemExit) as exc:
            skill(["-i", str(gridded_store), "-o", str(tmp_path / "o.zarr"), "--time-dim", "t"])
        assert exc.value.code == 2
        assert "not in dataset dims" in capsys.readouterr().err
        skill(["-i", str(gridded_store), "-o", str(tmp_path / "o.zarr"), "--time-dim", "time"])
        assert calls == [{"time_dim": "time"}]


class TestAllInputType:
    def test_one_input_not_one_per_member(self, tmp_path, gridded_store):
        # types.ALL is a tuple: one input allowing any of the three types,
        # not three inputs. A list is what declares one entry per input.
        calls = []
        skill = make_identity_skill(calls, input_type=types.ALL)
        skill(["-i", str(gridded_store), "-o", str(tmp_path / "o.zarr")])
        assert calls == [{}]

    def test_every_envelope_shape_accepted(self, tmp_path):
        skill = make_identity_skill([], input_type=types.ALL, output_type=types.ALL)
        shapes = (make_gridded(), make_forecast(), make_station(), make_series())
        for i, ds in enumerate(shapes):
            store = tmp_path / f"in{i}.zarr"
            ds.to_zarr(store, mode="w", consolidated=True)
            skill(["-i", str(store), "-o", str(tmp_path / f"out{i}.zarr")])

    def test_forecast_with_unnamed_axes_reads_as_a_series(self, tmp_path):
        # A forecast owes identifiable spatial dims; a store without them is a
        # series and an ALL skill reads it as one.
        store = tmp_path / "renamed.zarr"
        make_forecast().rename({"latitude": "yy", "longitude": "xx"}).to_zarr(
            store, mode="w", consolidated=True
        )
        skill = make_identity_skill([], input_type=types.ALL, output_type=types.ALL)
        skill(["-i", str(store), "-o", str(tmp_path / "o.zarr")])

    def test_spatially_collapsed_forecast_is_readable(self, tmp_path):
        # The store `reduce --dim latitude --dim longitude` leaves: step and
        # the scalar init time, no spatial axes. It passes the ALL union check
        # on the way out, so every ALL skill must be able to read it back.
        store = tmp_path / "collapsed.zarr"
        make_forecast().mean(dim=["latitude", "longitude"]).to_zarr(
            store, mode="w", consolidated=True
        )
        skill = make_identity_skill([], input_type=types.ALL, output_type=types.ALL)
        skill(["-i", str(store), "-o", str(tmp_path / "o.zarr")])

    def test_series_output_passes_the_union_check(self, tmp_path):
        # A skill declaring the ALL union must be able to WRITE a series, not
        # just read one: collapsing lat/lon is what produces the shape.
        store = tmp_path / "in.zarr"
        make_gridded().to_zarr(store, mode="w", consolidated=True)
        calls = []

        @weather_skill("collapse", "0.1.0", input_type=types.ALL, output_type=types.ALL)
        def collapse(ds, args):
            """Mean over the spatial dims, leaving a series."""
            calls.append(1)
            return ds.mean(dim=["latitude", "longitude"])

        out = tmp_path / "o.zarr"
        collapse(["-i", str(store), "-o", str(out)])
        assert calls == [1]
        assert set(xr.open_zarr(out, consolidated=True).sizes) == {"time"}

    def test_time_dim_check_still_runs(self, tmp_path, gridded_store, capsys):
        skill = make_identity_skill([], input_type=types.ALL, time_dim=True)
        with pytest.raises(SystemExit) as exc:
            skill(["-i", str(gridded_store), "-o", str(tmp_path / "o.zarr"), "--time-dim", "t"])
        assert exc.value.code == 2
        assert "not in dataset dims" in capsys.readouterr().err

    def test_time_dim_default_is_validated_like_a_given_value(self, tmp_path, capsys):
        # A string time_dim is an argparse DEFAULT, so the dest holds it on
        # every run and validation cannot tell it from a user-given --time-dim.
        # A forecast, whose `time` is a scalar coord rather than a dim, is
        # therefore rejected -- declare time_dim=True unless the skill really
        # does require that dim on every input.
        store = tmp_path / "fc.zarr"
        make_forecast().to_zarr(store, mode="w", consolidated=True)
        defaulted = make_identity_skill([], input_type=types.ALL, time_dim="time")
        with pytest.raises(SystemExit) as exc:
            defaulted(["-i", str(store), "-o", str(tmp_path / "o.zarr")])
        assert exc.value.code == 2
        assert "--time-dim 'time' not in dataset dims" in capsys.readouterr().err

        toggled = make_identity_skill([], input_type=types.ALL, time_dim=True)
        toggled(["-i", str(store), "-o", str(tmp_path / "o2.zarr")])

    def test_time_dim_toggle_resolves_lazily_in_the_body(self, tmp_path, capsys):
        # What a skill does instead of declaring a default: leave the dest None
        # and resolve when the axis is actually needed, so an input without one
        # reaches the body and an explicit --time-dim stays authoritative.
        seen = []

        @weather_skill(
            "resolve", "0.1.0", input_type=types.ALL, output_type=types.ALL, time_dim=True
        )
        def resolve(ds, args):
            """Resolve the time axis in the body."""
            seen.append(envelope.detect_time_dim(ds, args.time_dim))
            return ds.copy()

        detectable = tmp_path / "detectable.zarr"
        make_gridded().to_zarr(detectable, mode="w", consolidated=True)
        resolve(["-i", str(detectable), "-o", str(tmp_path / "a.zarr")])
        assert seen == ["time"]

        renamed = tmp_path / "renamed.zarr"
        make_gridded().rename({"time": "t"}).to_zarr(renamed, mode="w", consolidated=True)
        resolve(["-i", str(renamed), "-o", str(tmp_path / "b.zarr"), "--time-dim", "t"])
        assert seen == ["time", "t"]

        undetectable = tmp_path / "undetectable.zarr"
        ds = make_gridded().rename({"time": "record"}).drop_vars("record")
        ds.to_zarr(undetectable, mode="w", consolidated=True)
        with pytest.raises(SystemExit) as exc:
            resolve(["-i", str(undetectable), "-o", str(tmp_path / "c.zarr")])
        assert exc.value.code == 2
        assert "Pass --time-dim to override" in capsys.readouterr().err


class TestShapePreservingTransform:
    def test_end_to_end(self, tmp_path, gridded_store):
        calls = []
        skill = make_identity_skill(calls, input_type=types.ALL, output_type=types.ALL)
        out = tmp_path / "out.zarr"
        skill(["-i", str(gridded_store), "-o", str(out)])
        skill(["-i", str(gridded_store), "-o", str(out)])
        assert len(calls) == 1  # cache applies as for any zarr output
        assert history_of(out)[-1]["skill"] == "identity"

    def test_overlap_guard_applies(self, gridded_store):
        skill = make_identity_skill([], output_type=types.ALL)
        with pytest.raises(SystemExit) as exc:
            skill(["-i", str(gridded_store), "-o", str(gridded_store)])
        assert exc.value.code == 2

    def test_same_sentinel_is_gone(self):
        with pytest.raises(ValueError, match="unknown output_type"):
            weather_skill("x", "0.1.0", input_type=types.ALL, output_type="same")(lambda ds: None)

    def test_body_asserts_preservation_and_exits_1_on_drift(self, tmp_path, gridded_store, capsys):
        # The skill states the claim "same" used to imply, and a transform that
        # silently changes the shape is caught where it happens.
        @weather_skill("collapse", "0.1.0", input_type=types.ALL, output_type=types.ALL)
        def collapse(ds, args):
            """Return a series while claiming to preserve the input's shape."""
            out = ds.mean(dim=["latitude", "longitude"])
            envelope.validate_type(out, ds)
            return out

        with pytest.raises(SystemExit) as exc:
            collapse(["-i", str(gridded_store), "-o", str(tmp_path / "o.zarr")])
        assert exc.value.code == 1
        assert "expected a gridded envelope, got series" in capsys.readouterr().err


class TestUnionOutputType:
    def make_fetcher(self, build, **declaration):
        @weather_skill(
            "shape-fetch",
            "0.1.0",
            output_type=(types.GRIDDED, types.FORECAST),
            **declaration,
        )
        def fetch(args):
            """Fetch a dataset whose shape the source decides."""
            return build()

        return fetch

    def test_member_shapes_validate_and_write(self, tmp_path):
        out = tmp_path / "g.zarr"
        self.make_fetcher(make_gridded)(["-o", str(out)])
        assert xr.open_zarr(out, consolidated=True).sizes["time"] == 2
        out = tmp_path / "f.zarr"
        self.make_fetcher(make_forecast)(["-o", str(out)])
        assert "step" in xr.open_zarr(out, consolidated=True).sizes

    def test_cache_applies(self, tmp_path):
        ran = []

        def build():
            ran.append(1)
            return make_gridded()

        fetch = self.make_fetcher(build)
        out = tmp_path / "o.zarr"
        fetch(["-o", str(out)])
        fetch(["-o", str(out)])
        assert len(ran) == 1

    def test_non_member_shape_exits_1(self, tmp_path, capsys):
        fetch = self.make_fetcher(make_station)
        with pytest.raises(SystemExit) as exc:
            fetch(["-o", str(tmp_path / "o.zarr")])
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "returned a station envelope" in err
        assert "gridded or forecast" in err

    def test_streaming_pieces_validated(self, tmp_path):
        @weather_skill(
            "s",
            "0.1.0",
            output_type=(types.GRIDDED, types.FORECAST),
            streaming=True,
        )
        def fetch(args):
            """Stream."""
            yield make_gridded(n_time=1, start="2026-01-01")
            yield make_station()

        out = tmp_path / "o.zarr"
        with pytest.raises(SystemExit) as exc:
            fetch(["-o", str(out)])
        assert exc.value.code == 1
        # The mid-stream failure rolled back the partial store.
        assert not out.exists()

    def test_transform_union_validates(self, tmp_path, gridded_store):
        @weather_skill(
            "t",
            "0.1.0",
            input_type=types.ALL,
            output_type=(types.GRIDDED, types.FORECAST),
        )
        def transform(ds, args):
            """Transform."""
            return ds.copy()

        out = tmp_path / "o.zarr"
        transform(["-i", str(gridded_store), "-o", str(out)])
        assert history_of(out)[-1]["skill"] == "t"

    def test_declaration_errors(self):
        with pytest.raises(ValueError, match="only zarr envelope types"):
            weather_skill("x", "0.1.0", output_type=(types.GRIDDED, types.PNG))(lambda: None)
        with pytest.raises(ValueError, match="only zarr envelope types"):
            weather_skill("x", "0.1.0", output_type=(types.GRIDDED, "grid"))(lambda: None)
        with pytest.raises(ValueError, match="at least one"):
            weather_skill("x", "0.1.0", output_type=())(lambda: None)


class TestCacheDisabled:
    def test_always_recomputes_and_stamps(self, tmp_path, gridded_store, capsys):
        calls = []
        skill = make_identity_skill(calls, cache=False)
        out = tmp_path / "out.zarr"
        skill(["-i", str(gridded_store), "-o", str(out)])
        skill(["-i", str(gridded_store), "-o", str(out)])
        assert len(calls) == 2
        assert "Cache hit" not in capsys.readouterr().err
        chain = history_of(out)
        assert chain[-1]["skill"] == "identity"
        assert "hash" in chain[-1]["input"]

    def test_fetcher_always_recomputes(self, tmp_path):
        calls = []

        @weather_skill("toy-fetch", "0.1.0", output_type=types.GRIDDED, cache=False)
        def fetch(args):
            """Fetch a toy dataset."""
            calls.append(vars(args))
            return set_source(make_gridded(), "toy")

        out = tmp_path / "out.zarr"
        fetch(["-o", str(out)])
        fetch(["-o", str(out)])
        assert len(calls) == 2
        assert history_of(out)[0]["input"] is None

    def test_deferred_hash_still_stamped(self, tmp_path, gridded_store):
        skill = make_identity_skill([], cache=False, hash_input=False)
        out = tmp_path / "out.zarr"
        skill(["-i", str(gridded_store), "-o", str(out)])
        assert "hash" in history_of(out)[-1]["input"]

    def test_requires_zarr_output(self):
        with pytest.raises(ValueError, match="cache=False"):
            weather_skill("x", "0.1.0", input_type=types.ALL, output_type=types.PNG, cache=False)(
                lambda ds: None
            )
        with pytest.raises(ValueError, match="cache=False"):
            weather_skill("x", "0.1.0", cache=False)(lambda: None)


class TestFetcherMode:
    def make_fetcher(self, calls, **declaration):
        @weather_skill("toy-fetch", "0.1.0", output_type=types.GRIDDED, **declaration)
        def fetch(args):
            """Fetch a toy dataset."""
            calls.append(vars(args))
            return set_source(make_gridded(), "toy")

        return fetch

    def test_entry_input_none_and_source_stamped(self, tmp_path):
        fetch = self.make_fetcher([])
        out = tmp_path / "out.zarr"
        fetch(["-o", str(out)])
        chain = history_of(out)
        assert chain == [{"skill": "toy-fetch", "version": "0.1.0", "args": {}, "input": None}]
        assert xr.open_zarr(out, consolidated=True).attrs["weather_skills_source"] == "toy"

    def test_cache_hit_on_rerun(self, tmp_path):
        calls = []
        fetch = self.make_fetcher(calls)
        out = tmp_path / "out.zarr"
        fetch(["-o", str(out)])
        fetch(["-o", str(out)])
        assert len(calls) == 1

    def test_completeness_probe_failure_refetches(self, tmp_path):
        calls = []
        fetch = self.make_fetcher(calls, completeness_probe=lambda p: False)
        out = tmp_path / "out.zarr"
        fetch(["-o", str(out)])
        fetch(["-o", str(out)])
        assert len(calls) == 2

    def test_resolved_dates_recorded_and_passed(self, tmp_path):
        calls = []
        resolver_calls = []

        def resolver(args):
            resolver_calls.append(1)
            return date(2026, 6, 30)

        fetch = self.make_fetcher(calls, start_time=True, end_time=True, latest_resolver=resolver)
        out = tmp_path / "out.zarr"
        fetch(["--start", "latest-3w", "--end", "latest", "-o", str(out)])
        assert resolver_calls == [1]
        assert calls[0]["start_time"] == date(2026, 6, 10)
        assert calls[0]["end_time"] == date(2026, 6, 30)
        args = history_of(out)[0]["args"]
        assert args["start"] == "2026-06-10"
        assert args["end"] == "2026-06-30"

    def test_resolver_runs_once_for_window_and_date(self, tmp_path):
        calls = []
        resolver_calls = []

        def resolver(args):
            resolver_calls.append(1)
            return date(2026, 6, 30)

        fetch = self.make_fetcher(
            calls, start_time=True, end_time=True, date=True, latest_resolver=resolver
        )
        fetch(
            [
                "--start",
                "latest-1w",
                "--end",
                "latest",
                "--date",
                "latest",
                "-o",
                str(tmp_path / "o.zarr"),
            ]
        )
        assert resolver_calls == [1]
        assert calls[0]["end_time"] == date(2026, 6, 30)
        assert calls[0]["date"] == date(2026, 6, 30)

    def test_resolver_not_called_for_absolute_dates(self, tmp_path):
        resolver_calls = []
        fetch = self.make_fetcher(
            [],
            start_time=True,
            end_time=True,
            latest_resolver=lambda args: resolver_calls.append(1) or date(2026, 6, 30),
        )
        fetch(["--start", "2026-01-01", "--end", "2026-01-05", "-o", str(tmp_path / "o.zarr")])
        assert resolver_calls == []

    def test_relative_resolution_logged(self, tmp_path, capsys):
        fetch = self.make_fetcher([], start_time=True, end_time=True)
        fetch(["--start", "now-1w", "--end", "now", "-o", str(tmp_path / "o.zarr")])
        err = capsys.readouterr().err
        assert '"now-1w".."now"' in err
        assert "7 days" in err

    def test_singular_date(self, tmp_path):
        calls = []

        @weather_skill("init-fetch", "0.1.0", output_type=types.GRIDDED, date=True)
        def fetch(args):
            """Fetch one init."""
            calls.append(vars(args))
            return make_gridded()

        fetch(["--date", "2026-02-03", "-o", str(tmp_path / "o.zarr")])
        assert calls[0]["date"] == date(2026, 2, 3)
        assert history_of(tmp_path / "o.zarr")[0]["args"]["date"] == "2026-02-03"


class TestExitCodes:
    def test_bad_date_token_exits_2(self, tmp_path, capsys):
        fetch = TestFetcherMode().make_fetcher([], start_time=True, end_time=True)
        with pytest.raises(SystemExit) as exc:
            fetch(["--start", "now+3d", "--end", "now", "-o", str(tmp_path / "o.zarr")])
        assert exc.value.code == 2
        assert "Error: invalid date value" in capsys.readouterr().err

    def test_missing_input_exits_2(self, tmp_path, capsys):
        skill = make_identity_skill([])
        with pytest.raises(SystemExit) as exc:
            skill(["-i", str(tmp_path / "nope.zarr"), "-o", str(tmp_path / "o.zarr")])
        assert exc.value.code == 2
        assert "not found" in capsys.readouterr().err

    def test_overlap_guard_exits_2(self, gridded_store, capsys):
        skill = make_identity_skill([])
        with pytest.raises(SystemExit) as exc:
            skill(["-i", str(gridded_store), "-o", str(gridded_store)])
        assert exc.value.code == 2
        assert "overlaps" in capsys.readouterr().err

    def test_nested_output_exits_2(self, gridded_store):
        skill = make_identity_skill([])
        with pytest.raises(SystemExit) as exc:
            skill(["-i", str(gridded_store), "-o", str(gridded_store / "sub.zarr")])
        assert exc.value.code == 2

    def test_data_error_from_function_exits_1(self, tmp_path, gridded_store):
        @weather_skill("boom", "0.1.0", input_type=types.ALL, output_type=types.GRIDDED)
        def boom(ds, args):
            """Fail with a data error."""
            raise DataError("no data in the requested window")

        with pytest.raises(SystemExit) as exc:
            boom(["-i", str(gridded_store), "-o", str(tmp_path / "o.zarr")])
        assert exc.value.code == 1

    def test_usage_error_from_function_exits_2(self, tmp_path, gridded_store):
        @weather_skill("boom", "0.1.0", input_type=types.ALL, output_type=types.GRIDDED)
        def boom(ds, args):
            """Fail with a usage error."""
            raise UsageError("bad arguments")

        with pytest.raises(SystemExit) as exc:
            boom(["-i", str(gridded_store), "-o", str(tmp_path / "o.zarr")])
        assert exc.value.code == 2

    def test_envelope_mismatch_exits_2(self, tmp_path, gridded_store, capsys):
        skill = make_identity_skill([], input_type=types.STATION)
        with pytest.raises(SystemExit) as exc:
            skill(["-i", str(gridded_store), "-o", str(tmp_path / "o.zarr")])
        assert exc.value.code == 2
        assert "station_id" in capsys.readouterr().err

    def test_unreadable_zarr_exits_2(self, tmp_path):
        bad = tmp_path / "plain"
        bad.mkdir()
        (bad / "junk.txt").write_text("hello")
        skill = make_identity_skill([])
        with pytest.raises(SystemExit) as exc:
            skill(["-i", str(bad), "-o", str(tmp_path / "o.zarr")])
        assert exc.value.code == 2

    def test_validate_args_hook_runs_before_cache(self, tmp_path, gridded_store, capsys):
        def validate(args):
            if not args.to_name.strip():
                raise UsageError("--to-name must be a non-empty variable name.")

        skill = make_identity_skill([], extra_args=[("--to-name",)], validate_args=validate)
        with pytest.raises(SystemExit) as exc:
            skill(["-i", str(gridded_store), "-o", str(tmp_path / "o.zarr"), "--to-name", " "])
        assert exc.value.code == 2
        assert "non-empty" in capsys.readouterr().err


class TestEmptyStringValues:
    def make_fetcher(self, calls, **declaration):
        @weather_skill("toy-fetch", "0.1.0", output_type=types.GRIDDED, **declaration)
        def fetch(args):
            """Fetch a toy dataset."""
            calls.append(vars(args))
            return set_source(make_gridded(), "toy")

        return fetch

    def test_empty_start_and_end_exit_2(self, tmp_path, capsys):
        calls = []
        fetch = self.make_fetcher(calls, start_time=True, end_time=True)
        out = tmp_path / "o.zarr"
        with pytest.raises(SystemExit) as exc:
            fetch(["--start", "", "--end", "", "-o", str(out)])
        assert exc.value.code == 2
        assert "invalid date value" in capsys.readouterr().err
        assert calls == []
        assert not out.exists()

    def test_lone_empty_start_with_end_exits_2(self, tmp_path, capsys):
        calls = []
        fetch = self.make_fetcher(calls, start_time=True, end_time=True)
        with pytest.raises(SystemExit) as exc:
            fetch(["--start", "", "--end", "2026-01-02", "-o", str(tmp_path / "o.zarr")])
        assert exc.value.code == 2
        assert "invalid date value" in capsys.readouterr().err
        assert calls == []

    def test_empty_optional_start_and_end_exit_2(self, tmp_path, gridded_store):
        # An explicit empty value on an optional window is malformed, not an
        # omission.
        calls = []
        skill = make_identity_skill(
            calls,
            start_time={"required": False},
            end_time={"required": False},
        )
        with pytest.raises(SystemExit) as exc:
            skill(
                [
                    "-i",
                    str(gridded_store),
                    "-o",
                    str(tmp_path / "o.zarr"),
                    "--start",
                    "",
                    "--end",
                    "",
                ]
            )
        assert exc.value.code == 2
        assert calls == []

    def test_empty_date_exits_2(self, tmp_path, capsys):
        calls = []
        fetch = self.make_fetcher(calls, date=True)
        with pytest.raises(SystemExit) as exc:
            fetch(["--date", "", "-o", str(tmp_path / "o.zarr")])
        assert exc.value.code == 2
        assert "invalid date value" in capsys.readouterr().err
        assert calls == []

    def test_empty_required_bbox_exits_2(self, tmp_path, capsys):
        calls = []
        fetch = self.make_fetcher(calls, bbox=types.REQUIRED)
        with pytest.raises(SystemExit) as exc:
            fetch(["--bbox", "", "-o", str(tmp_path / "o.zarr")])
        assert exc.value.code == 2
        assert "N/W/S/E" in capsys.readouterr().err
        assert calls == []

    def test_malformed_bbox_exits_before_latest_discovery(self, tmp_path, capsys):
        resolver_calls = []

        def resolver(args):
            resolver_calls.append(1)
            return date(2026, 6, 30)

        fetch = self.make_fetcher(
            [],
            bbox=types.OPTIONAL,
            start_time=True,
            end_time=True,
            latest_resolver=resolver,
        )
        with pytest.raises(SystemExit) as exc:
            fetch(
                [
                    "--bbox",
                    "1/2/3",
                    "--start",
                    "latest-1w",
                    "--end",
                    "latest",
                    "-o",
                    str(tmp_path / "o.zarr"),
                ]
            )
        assert exc.value.code == 2
        assert "N/W/S/E" in capsys.readouterr().err
        assert resolver_calls == []


class TestUnprefixedErrors:
    def make_skill(self, exc):
        @weather_skill("submit-feedback", "0.1.0", extra_args=[("--body",)])
        def skill(args):
            """Submit a feedback body."""
            raise exc

        return skill

    def test_unprefixed_data_error_exits_1_with_exact_message(self, capsys):
        skill = self.make_skill(
            DataError("Body too long: 120 characters over the limit.", prefix=False)
        )
        with pytest.raises(SystemExit) as exc:
            skill(["--body", "x"])
        assert exc.value.code == 1
        assert capsys.readouterr().err == "Body too long: 120 characters over the limit.\n"

    def test_unprefixed_usage_error_exits_2_with_exact_message(self, capsys):
        skill = self.make_skill(UsageError("retry with a shorter body", prefix=False))
        with pytest.raises(SystemExit) as exc:
            skill(["--body", "x"])
        assert exc.value.code == 2
        assert capsys.readouterr().err == "retry with a shorter body\n"

    def test_default_keeps_error_prefix(self, capsys):
        skill = self.make_skill(DataError("hard failure"))
        with pytest.raises(SystemExit) as exc:
            skill(["--body", "x"])
        assert exc.value.code == 1
        assert capsys.readouterr().err == "Error: hard failure\n"

    def test_unprefixed_error_in_zarr_mode(self, tmp_path, gridded_store, capsys):
        @weather_skill("identity", "0.1.0", input_type=types.GRIDDED, output_type=types.GRIDDED)
        def skill(ds, args):
            """Copy the input envelope unchanged."""
            raise DataError("Body too long: 1 character over the limit.", prefix=False)

        with pytest.raises(SystemExit) as exc:
            skill(["-i", str(gridded_store), "-o", str(tmp_path / "out.zarr")])
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert err.endswith("Body too long: 1 character over the limit.\n")
        assert "Error:" not in err


class TestMultiInput:
    def test_fixed_two_inputs(self, tmp_path):
        a = tmp_path / "a.zarr"
        b = tmp_path / "b.zarr"
        make_gridded(fill=1.0).to_zarr(a, mode="w", consolidated=True)
        make_gridded(fill=2.0).to_zarr(b, mode="w", consolidated=True)
        received = []

        @weather_skill(
            "difference", "0.1.0", input_type=[types.ALL, types.ALL], output_type=types.GRIDDED
        )
        def difference(ds_a, ds_b, args):
            """A - B."""
            received.extend([ds_a, ds_b])
            out = ds_a.copy()
            out["precip"] = ds_a["precip"] - ds_b["precip"]
            return out

        out = tmp_path / "out.zarr"
        difference(["-i", str(a), "-i", str(b), "-o", str(out)])
        assert len(received) == 2
        chain = history_of(out)
        assert len(chain) == 1
        inputs = chain[-1]["input"]
        assert [i["basename"] for i in inputs] == ["a.zarr", "b.zarr"]
        assert all({"basename", "hash", "history"} <= set(i) for i in inputs)

    def test_wrong_input_count_exits_2(self, tmp_path, gridded_store):
        @weather_skill(
            "difference", "0.1.0", input_type=[types.ALL, types.ALL], output_type=types.GRIDDED
        )
        def difference(ds_a, ds_b, args):
            """A - B."""

        with pytest.raises(SystemExit) as exc:
            difference(["-i", str(gridded_store), "-o", str(tmp_path / "o.zarr")])
        assert exc.value.code == 2

    def test_trunk_is_first_inputs_chain(self, tmp_path, gridded_store):
        upstreamed = tmp_path / "up.zarr"
        make_identity_skill([])(["-i", str(gridded_store), "-o", str(upstreamed)])

        @weather_skill(
            "concat", "0.1.0", input_type=types.ALL, output_type=types.GRIDDED, variadic_input=True
        )
        def concat(datasets, args):
            """Concatenate."""
            return datasets[0].copy()

        out = tmp_path / "out.zarr"
        concat(["-i", str(upstreamed), "-i", str(gridded_store), "-o", str(out)])
        chain = history_of(out)
        # Trunk = first input's chain + the merge entry.
        assert chain[:-1] == history_of(upstreamed)
        assert chain[-1]["skill"] == "concat"

    def test_variadic_requires_two_inputs(self, tmp_path, gridded_store):
        @weather_skill(
            "concat", "0.1.0", input_type=types.ALL, output_type=types.GRIDDED, variadic_input=True
        )
        def concat(datasets, args):
            """Concatenate."""

        with pytest.raises(SystemExit) as exc:
            concat(["-i", str(gridded_store), "-o", str(tmp_path / "o.zarr")])
        assert exc.value.code == 2

    def test_multi_input_cache_hit_and_branch_miss(self, tmp_path):
        a = tmp_path / "a.zarr"
        b = tmp_path / "b.zarr"
        make_gridded(fill=1.0).to_zarr(a, mode="w", consolidated=True)
        make_gridded(fill=2.0).to_zarr(b, mode="w", consolidated=True)
        calls = []

        @weather_skill(
            "difference", "0.1.0", input_type=[types.ALL, types.ALL], output_type=types.GRIDDED
        )
        def difference(ds_a, ds_b, args):
            """A - B."""
            calls.append(1)
            return ds_a.copy()

        out = tmp_path / "out.zarr"
        argv = ["-i", str(a), "-i", str(b), "-o", str(out)]
        difference(argv)
        difference(argv)
        assert len(calls) == 1
        # Modifying one branch in place forces a recompute (per-input hash).
        make_gridded(fill=9.0).to_zarr(b, mode="w", consolidated=True)
        difference(argv)
        assert len(calls) == 2


class TestReferenceInputs:
    def test_reference_change_forces_miss(self, tmp_path, gridded_store):
        ref = tmp_path / "grid.zarr"
        make_gridded(fill=1.0).to_zarr(ref, mode="w", consolidated=True)
        calls = []
        skill = make_identity_skill(
            calls,
            input_type=types.ALL,
            hash_input=False,
            extra_args=[("--reference-grid",)],
            reference_args=("reference_grid",),
        )
        out = tmp_path / "out.zarr"
        argv = [
            "-i",
            str(gridded_store),
            "-o",
            str(out),
            "--reference-grid",
            str(ref),
        ]
        skill(argv)
        skill(argv)
        assert len(calls) == 1
        assert history_of(out)[-1]["reference_inputs"][0]["basename"] == "grid.zarr"
        make_gridded(fill=7.0).to_zarr(ref, mode="w", consolidated=True)
        skill(argv)
        assert len(calls) == 2

    def test_missing_reference_exits_2(self, tmp_path, gridded_store):
        skill = make_identity_skill(
            [],
            extra_args=[("--reference-grid",)],
            reference_args=("reference_grid",),
        )
        with pytest.raises(SystemExit) as exc:
            skill(
                [
                    "-i",
                    str(gridded_store),
                    "-o",
                    str(tmp_path / "o.zarr"),
                    "--reference-grid",
                    str(tmp_path / "nope.zarr"),
                ]
            )
        assert exc.value.code == 2


class TestStreaming:
    def make_streamer(self, pieces, **declaration):
        @weather_skill(
            "stream-fetch",
            "0.1.0",
            output_type=types.GRIDDED,
            streaming=True,
            **declaration,
        )
        def fetch(args):
            """Stream a toy dataset per period."""
            for piece in pieces():
                yield set_source(piece, "toy") if hasattr(piece, "attrs") else piece

        return fetch

    def test_appends_and_restamps(self, tmp_path):
        def pieces():
            yield make_gridded(n_time=1, start="2026-01-01")
            yield make_gridded(n_time=1, start="2026-01-02")

        fetch = self.make_streamer(pieces)
        out = tmp_path / "out.zarr"
        fetch(["-o", str(out)])
        ds = xr.open_zarr(out, consolidated=True)
        assert ds.sizes["time"] == 2
        assert history_of(out)[0]["skill"] == "stream-fetch"
        assert ds.attrs["weather_skills_source"] == "toy"

    def test_second_run_is_cache_hit(self, tmp_path):
        ran = []

        def pieces():
            ran.append(1)
            yield make_gridded(n_time=1)

        fetch = self.make_streamer(pieces)
        out = tmp_path / "out.zarr"
        fetch(["-o", str(out)])
        fetch(["-o", str(out)])
        assert len(ran) == 1

    def test_midstream_failure_rolls_back_partial_store(self, tmp_path, capsys):
        def pieces():
            yield make_gridded(n_time=1)
            raise RuntimeError("transfer failed")

        fetch = self.make_streamer(pieces)
        out = tmp_path / "out.zarr"
        with pytest.raises(RuntimeError, match="transfer failed"):
            fetch(["-o", str(out)])
        assert not out.exists()
        assert "Removed partial store" in capsys.readouterr().err

    def test_prior_complete_store_survives_pre_write_failure(self, tmp_path):
        def good():
            yield make_gridded(n_time=1)

        fetch = self.make_streamer(good)
        out = tmp_path / "out.zarr"
        fetch(["-o", str(out)])

        def bad():
            raise RuntimeError("nothing fetched")
            yield

        fetch_bad = self.make_streamer(bad, completeness_probe=lambda p: False)
        with pytest.raises(RuntimeError):
            fetch_bad(["-o", str(out)])
        # The failure happened before any write, so the prior store is intact.
        assert out.exists()

    def test_no_yield_exits_1(self, tmp_path, capsys):
        def pieces():
            return
            yield

        fetch = self.make_streamer(pieces)
        with pytest.raises(SystemExit) as exc:
            fetch(["-o", str(tmp_path / "o.zarr")])
        assert exc.value.code == 1
        assert "produced no data" in capsys.readouterr().err

    def test_entry_override_rewrites_effective_end(self, tmp_path):
        def pieces():
            yield EntryOverride({"end": "2026-01-05"})
            yield make_gridded(n_time=1)

        fetch = self.make_streamer(pieces, start_time=True, end_time=True)
        out = tmp_path / "out.zarr"
        fetch(["--start", "2026-01-01", "--end", "2026-01-31", "-o", str(out)])
        args = history_of(out)[0]["args"]
        assert args["start"] == "2026-01-01"
        assert args["end"] == "2026-01-05"

    def test_entry_override_after_final_dataset_is_applied(self, tmp_path):
        def pieces():
            yield make_gridded(n_time=1, start="2026-01-01")
            yield make_gridded(n_time=1, start="2026-01-02")
            yield EntryOverride({"end": "2026-01-02"})

        fetch = self.make_streamer(pieces, start_time=True, end_time=True)
        out = tmp_path / "out.zarr"
        fetch(["--start", "2026-01-01", "--end", "2026-01-31", "-o", str(out)])
        # Both the consolidated and unconsolidated readers see the override.
        args = history_of(out)[0]["args"]
        assert args["end"] == "2026-01-02"
        from weather_skills_core import provenance

        assert provenance.load_history(out)[0]["args"]["end"] == "2026-01-02"
        # The re-stamp touched only the history attr.
        ds = xr.open_zarr(out, consolidated=True)
        assert ds.sizes["time"] == 2
        assert ds.attrs["weather_skills_source"] == "toy"


class TestEntryOverrideStandardMode:
    def test_tuple_return_rewrites_entry(self, tmp_path):
        @weather_skill(
            "effective", "0.1.0", output_type=types.GRIDDED, start_time=True, end_time=True
        )
        def fetch(args):
            """Fetch with an effective end."""
            return make_gridded(), EntryOverride({"end": "2026-01-02"})

        out = tmp_path / "out.zarr"
        fetch(["--start", "2026-01-01", "--end", "2026-01-31", "-o", str(out)])
        assert history_of(out)[0]["args"]["end"] == "2026-01-02"


class TestPngMode:
    def test_single_input_key_and_software(self, tmp_path, gridded_store):
        fig = FakeFigure()

        @weather_skill("plot", "0.1.0", input_type=types.ALL, output_type=types.PNG, title=True)
        def plot(ds, args):
            """Plot."""
            return fig

        out = tmp_path / "plot.png"
        plot(["-i", str(gridded_store), "-o", str(out), "--title", "T"])
        metadata = fig.saved["metadata"]
        assert set(metadata) == {"weather_skills_history", "Software"}
        chain = json.loads(metadata["weather_skills_history"])
        assert chain[-1]["skill"] == "plot"
        assert chain[-1]["args"]["title"] == "T"
        assert chain[-1]["input"]["basename"] == "in.zarr"
        assert "hash" in chain[-1]["input"]
        assert fig.saved["kwargs"]["dpi"] == 150
        assert out.exists()

    def test_two_inputs_suffixed_keys(self, tmp_path, gridded_store):
        other = tmp_path / "b.zarr"
        make_gridded(fill=3.0).to_zarr(other, mode="w", consolidated=True)
        fig = FakeFigure()

        @weather_skill(
            "plot-compare",
            "0.1.0",
            input_type=[types.ALL, types.ALL],
            output_type=types.PNG,
            history_labels=["a", "b"],
        )
        def plot_compare(ds_a, ds_b, args):
            """Compare."""
            return fig

        plot_compare(["-i", str(gridded_store), "-i", str(other), "-o", str(tmp_path / "p.png")])
        assert set(fig.saved["metadata"]) == {
            "weather_skills_history_a",
            "weather_skills_history_b",
            "Software",
        }

    def test_named_inputs_default_labels(self, tmp_path, gridded_store):
        other = tmp_path / "mc.zarr"
        make_gridded(fill=3.0).to_zarr(other, mode="w", consolidated=True)
        fig = FakeFigure()

        @weather_skill(
            "plot-mediogram",
            "0.1.0",
            input_type=[types.ALL, types.ALL],
            output_type=types.PNG,
            input_names=["forecast", "mclimate"],
        )
        def plot_mediogram(forecast_ds, mclimate_ds, args):
            """Mediogram."""
            return fig

        plot_mediogram(
            [
                "--forecast",
                str(gridded_store),
                "--mclimate",
                str(other),
                "-o",
                str(tmp_path / "p.png"),
            ]
        )
        metadata = fig.saved["metadata"]
        assert "weather_skills_history_forecast" in metadata
        assert "weather_skills_history_mclimate" in metadata
        # Path strings do not leak into the recorded args.
        chain = json.loads(metadata["weather_skills_history_forecast"])
        assert "forecast" not in chain[-1]["args"]
        assert "output" not in chain[-1]["args"]

    def test_savefig_kwargs_extend(self, tmp_path, gridded_store):
        fig = FakeFigure()

        @weather_skill(
            "plot",
            "0.1.0",
            input_type=types.ALL,
            output_type=types.PNG,
            savefig_kwargs={"bbox_inches": "tight"},
        )
        def plot(ds, args):
            """Plot."""
            return fig

        plot(["-i", str(gridded_store), "-o", str(tmp_path / "p.png")])
        assert fig.saved["kwargs"] == {"dpi": 150, "bbox_inches": "tight"}

    def test_no_cache_always_renders(self, tmp_path, gridded_store):
        calls = []

        @weather_skill("plot", "0.1.0", input_type=types.ALL, output_type=types.PNG)
        def plot(ds, args):
            """Plot."""
            calls.append(1)
            return FakeFigure()

        argv = ["-i", str(gridded_store), "-o", str(tmp_path / "p.png")]
        plot(argv)
        plot(argv)
        assert len(calls) == 2

    def test_output_dir_exits_2(self, tmp_path, gridded_store, capsys):
        calls = []

        @weather_skill("plot", "0.1.0", input_type=types.ALL, output_type=types.PNG)
        def plot(ds, args):
            """Plot."""
            calls.append(1)
            return FakeFigure()

        out = tmp_path / "p.png"
        out.mkdir()
        with pytest.raises(SystemExit) as exc:
            plot(["-i", str(gridded_store), "-o", str(out)])
        assert exc.value.code == 2
        assert "is a directory" in capsys.readouterr().err
        assert calls == []

    def test_no_input_is_declaration_error(self):
        with pytest.raises(ValueError, match="at least one declared zarr input"):
            weather_skill("plot", "0.1.0", output_type=types.PNG)(lambda: None)

    def test_duplicate_history_labels_is_declaration_error(self):
        with pytest.raises(ValueError, match="unique"):
            weather_skill(
                "plot-compare",
                "0.1.0",
                input_type=[types.ALL, types.ALL],
                output_type=types.PNG,
                history_labels=["a", "a"],
            )(lambda x, y: None)

    def test_upstream_chain_embedded(self, tmp_path, gridded_store):
        mid = tmp_path / "mid.zarr"
        make_identity_skill([])(["-i", str(gridded_store), "-o", str(mid)])
        fig = FakeFigure()

        @weather_skill("plot", "0.1.0", input_type=types.ALL, output_type=types.PNG)
        def plot(ds, args):
            """Plot."""
            return fig

        plot(["-i", str(mid), "-o", str(tmp_path / "p.png")])
        chain = json.loads(fig.saved["metadata"]["weather_skills_history"])
        assert [e["skill"] for e in chain] == ["identity", "plot"]


class TestOutputMessages:
    def test_cache_hit_names_the_skill(self, tmp_path, gridded_store, capsys):
        # The line always names the skill, so a pipeline running several can
        # be read back to the one that was skipped.
        @weather_skill("clip-region", "0.1.0", input_type=types.ALL, output_type=types.GRIDDED)
        def clip_region(ds, args):
            """Clip."""
            return ds.copy()

        out = tmp_path / "o.zarr"
        argv = ["-i", str(gridded_store), "-o", str(out)]
        clip_region(argv)
        clip_region(argv)
        assert "skipping clip-region." in capsys.readouterr().err

    def test_cache_hit_defaults_to_skill_name(self, tmp_path, gridded_store, capsys):
        skill = make_identity_skill([])
        out = tmp_path / "o.zarr"
        argv = ["-i", str(gridded_store), "-o", str(out)]
        skill(argv)
        skill(argv)
        assert "skipping identity." in capsys.readouterr().err

    def test_zarr_detail_is_the_output_sizes(self, tmp_path, gridded_store, capsys):
        skill = make_identity_skill([], input_type=types.ALL)
        out = tmp_path / "out.zarr"
        skill(["-i", str(gridded_store), "-o", str(out)])
        err = capsys.readouterr().err
        # A plain dict, not the Frozen(...) repr the skills used to emit.
        assert f"Wrote: {out} ({{'time': 2, 'latitude': 3, 'longitude': 4}})" in err

    def test_streaming_detail_is_the_append_total(self, tmp_path, capsys):
        @weather_skill("s", "0.1.0", output_type=types.GRIDDED, streaming=True)
        def fetch(args):
            """Stream."""
            yield make_gridded(n_time=1, start="2026-01-01")
            yield make_gridded(n_time=1, start="2026-01-02")

        out = tmp_path / "o.zarr"
        fetch(["-o", str(out)])
        assert f"Wrote: {out} (time=2)" in capsys.readouterr().err

    def test_png_has_no_detail(self, tmp_path, gridded_store, capsys):
        @weather_skill("plot", "0.1.0", input_type=types.ALL, output_type=types.PNG)
        def plot(ds, args):
            """Plot."""
            return FakeFigure()

        out = tmp_path / "p.png"
        plot(["-i", str(gridded_store), "-o", str(out)])
        assert f"Wrote: {out}\n" in capsys.readouterr().err

    def test_a_skill_prints_its_own_extra_line(self, tmp_path, gridded_store, capsys):
        # What the four custom-text skills do now: their own stderr line,
        # which lands before the decorator's Wrote: line.
        @weather_skill("rename", "0.1.0", input_type=types.ALL, output_type=types.GRIDDED)
        def rename(ds, args):
            """Rename."""
            print("Renaming variable 'precip' -> 'rain'", file=sys.stderr)
            return ds.copy()

        rename(["-i", str(gridded_store), "-o", str(tmp_path / "o.zarr")])
        err = capsys.readouterr().err
        assert err.index("Renaming variable") < err.index("Wrote:")

    def test_unexpected_extra_return_is_type_error(self, tmp_path, gridded_store):
        @weather_skill("bad", "0.1.0", input_type=types.ALL, output_type=types.GRIDDED)
        def bad(ds, args):
            """Bad return."""
            return ds.copy(), "not a marker"

        with pytest.raises(TypeError, match="unexpected extra return value"):
            bad(["-i", str(gridded_store), "-o", str(tmp_path / "o.zarr")])

    def test_two_extras_rejected(self, tmp_path, gridded_store):
        @weather_skill("bad", "0.1.0", input_type=types.ALL, output_type=types.GRIDDED)
        def bad(ds, args):
            """Bad return."""
            return ds.copy(), EntryOverride({"x": 1}), EntryOverride({"y": 2})

        with pytest.raises(TypeError, match="exactly one EntryOverride"):
            bad(["-i", str(gridded_store), "-o", str(tmp_path / "o.zarr")])

    def test_png_rejects_entry_override(self, tmp_path, gridded_store):
        @weather_skill("plot", "0.1.0", input_type=types.ALL, output_type=types.PNG)
        def plot(ds, args):
            """Plot."""
            return FakeFigure(), EntryOverride({"x": 1})

        with pytest.raises(TypeError, match="unexpected extra return value"):
            plot(["-i", str(gridded_store), "-o", str(tmp_path / "p.png")])


class TestNoArtifactMode:
    def test_cli_only(self, capsys):
        received = {}

        @weather_skill(
            "resolve-region",
            "0.1.0",
            extra_args=[("code",), ("--geojson",)],
        )
        def resolve_region(args):
            """Resolve a country code to a bbox."""
            received.update(code=args.code, geojson=args.geojson)
            print("1/2/3/4")

        resolve_region(["KEN"])
        assert received == {"code": "KEN", "geojson": None}
        assert capsys.readouterr().out == "1/2/3/4\n"

    def test_no_output_flag(self):
        @weather_skill("email-report", "0.1.0", extra_args=[("--to",)])
        def email_report(args):
            """Send a report."""

        with pytest.raises(SystemExit) as exc:
            email_report.parser.parse_args(["-o", "x"])
        assert exc.value.code == 2

    def test_help_epilog(self, capsys):
        @weather_skill("submit-feedback", "0.1.0")
        def submit_feedback(args):
            """Submit feedback."""

        with pytest.raises(SystemExit):
            submit_feedback(["--help"])
        assert "skill version: 0.1.0" in capsys.readouterr().out

    def test_exit_1_on_data_error(self):
        @weather_skill("submit-feedback", "0.1.0")
        def submit_feedback(args):
            """Submit feedback."""
            raise DataError("server rejected the submission; retry")

        with pytest.raises(SystemExit) as exc:
            submit_feedback([])
        assert exc.value.code == 1


class TestWriteTail:
    def test_encoding_cleared_and_hook_applies_after(self, tmp_path, gridded_store):
        applied = []

        def write_encoding(ds):
            applied.append(sorted(str(v) for v in ds.variables if ds[v].encoding))
            ds["time"].encoding["units"] = "days since 1970-01-01 00:00:00"

        skill = make_identity_skill([], write_encoding=write_encoding)
        skill(["-i", str(gridded_store), "-o", str(tmp_path / "o.zarr")])
        # At hook time every encoding had been cleared.
        assert applied == [[]]

    def test_input_attrs_carried(self, tmp_path):
        src = tmp_path / "in.zarr"
        ds = make_gridded()
        ds.attrs["Conventions"] = "CF-1.13"
        ds.to_zarr(src, mode="w", consolidated=True)

        @weather_skill("fresh", "0.1.0", input_type=types.ALL, output_type=types.GRIDDED)
        def fresh(ds, args):
            """Return a dataset built from scratch (no attrs of its own)."""
            return xr.Dataset({"precip": ds["precip"]})

        out = tmp_path / "o.zarr"
        fresh(["-i", str(src), "-o", str(out)])
        assert xr.open_zarr(out, consolidated=True).attrs["Conventions"] == "CF-1.13"

    def test_rhiza_attrs_are_opaque(self, tmp_path, capsys):
        # A store carrying only rhiza_* attrs has no history: the input is
        # opaque, the chain starts fresh, and the attrs ride along unchanged.
        src = tmp_path / "in.zarr"
        ds = make_gridded()
        ds.attrs["rhiza_source"] = "chirps"
        ds.attrs["rhiza_history"] = json.dumps(
            [{"skill": "chirps-fetch", "version": "0.1.0", "args": {}, "input": None}]
        )
        ds.to_zarr(src, mode="w", consolidated=True)
        skill = make_identity_skill([], input_type=types.ALL)
        out = tmp_path / "o.zarr"
        skill(["-i", str(src), "-o", str(out)])
        assert "treating input as opaque" in capsys.readouterr().err
        assert [e["skill"] for e in history_of(out)] == ["identity"]
        attrs = xr.open_zarr(out, consolidated=True).attrs
        assert attrs["rhiza_source"] == "chirps"
        assert "weather_skills_source" not in attrs

    def test_existing_output_replaced(self, tmp_path, gridded_store):
        out = tmp_path / "o.zarr"
        make_gridded(fill=9.0, name="other").to_zarr(out, mode="w", consolidated=True)
        skill = make_identity_skill([])
        skill(["-i", str(gridded_store), "-o", str(out)])
        written = xr.open_zarr(out, consolidated=True)
        assert list(written.data_vars) == ["precip"]

    def test_existing_output_regular_file_replaced(self, tmp_path, gridded_store):
        out = tmp_path / "o.zarr"
        out.write_text("not a store")
        skill = make_identity_skill([])
        skill(["-i", str(gridded_store), "-o", str(out)])
        assert out.is_dir()
        assert list(xr.open_zarr(out, consolidated=True).data_vars) == ["precip"]

    def test_existing_output_dangling_symlink_replaced(self, tmp_path, gridded_store):
        out = tmp_path / "o.zarr"
        out.symlink_to(tmp_path / "never-created.zarr")
        skill = make_identity_skill([])
        skill(["-i", str(gridded_store), "-o", str(out)])
        assert out.is_dir()
        assert not out.is_symlink()
        assert list(xr.open_zarr(out, consolidated=True).data_vars) == ["precip"]

    def test_existing_output_symlink_to_dir_unlinked_not_followed(self, tmp_path, gridded_store):
        target = tmp_path / "target.zarr"
        make_gridded(fill=9.0, name="other").to_zarr(target, mode="w", consolidated=True)
        out = tmp_path / "o.zarr"
        out.symlink_to(target)
        skill = make_identity_skill([])
        skill(["-i", str(gridded_store), "-o", str(out)])
        # The link itself was replaced by a real store; the target survives.
        assert out.is_dir()
        assert not out.is_symlink()
        assert list(xr.open_zarr(target, consolidated=True).data_vars) == ["other"]

    def test_streaming_existing_output_dangling_symlink_replaced(self, tmp_path):
        @weather_skill("s", "0.1.0", output_type=types.GRIDDED, streaming=True)
        def fetch(args):
            """Stream."""
            yield make_gridded(n_time=1)

        out = tmp_path / "o.zarr"
        out.symlink_to(tmp_path / "never-created.zarr")
        fetch(["-o", str(out)])
        assert out.is_dir()
        assert not out.is_symlink()
        assert xr.open_zarr(out, consolidated=True).sizes["time"] == 1

    def test_streaming_existing_output_regular_file_replaced(self, tmp_path):
        @weather_skill("s", "0.1.0", output_type=types.GRIDDED, streaming=True)
        def fetch(args):
            """Stream."""
            yield make_gridded(n_time=1)

        out = tmp_path / "o.zarr"
        out.write_text("not a store")
        fetch(["-o", str(out)])
        assert out.is_dir()
        assert xr.open_zarr(out, consolidated=True).sizes["time"] == 1

    def test_failed_write_removes_partial_store(self, tmp_path, gridded_store, capsys):
        @weather_skill("bad-write", "0.1.0", input_type=types.ALL, output_type=types.GRIDDED)
        def bad_write(ds, args):
            """Return a dataset zarr cannot serialize."""
            out = ds.copy()
            # Object dtype fails during the write, after the store directory
            # (and its root metadata) already exists on disk.
            out["junk"] = (("time",), np.array([object(), object()], dtype=object))
            return out

        out = tmp_path / "o.zarr"
        with pytest.raises(ValueError, match="serialize"):
            bad_write(["-i", str(gridded_store), "-o", str(out)])
        assert not out.exists()
        assert "Removed partial store" in capsys.readouterr().err

    def test_failed_write_reruns_cleanly(self, tmp_path, gridded_store):
        # The rollback re-raises the original failure; the second run is a
        # clean miss (no truncated store to mistake for a cache hit).
        calls = []

        @weather_skill("bad-write", "0.1.0", input_type=types.ALL, output_type=types.GRIDDED)
        def bad_write(ds, args):
            """Return a dataset zarr cannot serialize."""
            calls.append(1)
            out = ds.copy()
            out["junk"] = (("time",), np.array([object(), object()], dtype=object))
            return out

        out = tmp_path / "o.zarr"
        argv = ["-i", str(gridded_store), "-o", str(out)]
        with pytest.raises(ValueError, match="serialize"):
            bad_write(argv)
        with pytest.raises(ValueError, match="serialize"):
            bad_write(argv)
        assert len(calls) == 2

    def test_wrote_message(self, tmp_path, gridded_store, capsys):
        skill = make_identity_skill([])
        skill(["-i", str(gridded_store), "-o", str(tmp_path / "o.zarr")])
        assert "Wrote:" in capsys.readouterr().err

    def test_opaque_input_warning(self, tmp_path, gridded_store, capsys):
        skill = make_identity_skill([])
        skill(["-i", str(gridded_store), "-o", str(tmp_path / "o.zarr")])
        assert "treating input as opaque" in capsys.readouterr().err


class TestInputPathAttr:
    def test_single_input_carries_its_path(self, tmp_path, gridded_store):
        seen = []

        @weather_skill("identity", "0.1.0", input_type=types.ALL, output_type=types.GRIDDED)
        def identity(ds, args):
            """Copy the input envelope unchanged."""
            seen.append(provenance.input_path(ds))
            return ds.copy()

        out = tmp_path / "o.zarr"
        identity(["-i", str(gridded_store), "-o", str(out)])
        assert seen == [gridded_store]
        # The path is not an argument, so it cannot reach the recorded args.
        assert "input_path" not in history_of(out)[-1]["args"]

    def test_variadic_inputs_carry_their_own_paths_in_repeat_order(self, tmp_path):
        a = tmp_path / "a.zarr"
        b = tmp_path / "b.zarr"
        make_gridded(fill=1.0).to_zarr(a, mode="w", consolidated=True)
        make_gridded(fill=2.0).to_zarr(b, mode="w", consolidated=True)
        seen = []

        @weather_skill(
            "concat",
            "0.1.0",
            input_type=types.ALL,
            output_type=types.GRIDDED,
            variadic_input=True,
        )
        def concat(datasets, args):
            """Concatenate."""
            seen.extend(provenance.input_path(ds) for ds in datasets)
            return datasets[0].copy()

        concat(["-i", str(b), "-i", str(a), "-o", str(tmp_path / "o.zarr")])
        assert seen == [b, a]

    def test_named_inputs_carry_their_own_paths(self, tmp_path, gridded_store):
        other = tmp_path / "mc.zarr"
        make_gridded(fill=3.0).to_zarr(other, mode="w", consolidated=True)
        seen = []

        @weather_skill(
            "plot-mediogram",
            "0.1.0",
            input_type=[types.ALL, types.ALL],
            output_type=types.PNG,
            input_names=["forecast", "mclimate"],
        )
        def plot_mediogram(forecast_ds, mclimate_ds, args):
            """Mediogram."""
            seen.extend(provenance.input_path(ds) for ds in (forecast_ds, mclimate_ds))
            return FakeFigure()

        plot_mediogram(
            ["--mclimate", str(other), "--forecast", str(gridded_store), "-o", str(tmp_path / "p")]
        )
        assert seen == [gridded_store, other]

    def test_identical_data_at_different_paths_hashes_the_same(self, tmp_path):
        # attrs live inside the store and hash_zarr hashes the store's bytes,
        # so a surviving input path would make the content hash vary with the
        # local directory layout. The decorator merges the first input's attrs
        # into the result, so the attr does reach the write path.
        stores = []
        for d in ("dir_a", "dir_b"):
            store = tmp_path / d / "in.zarr"
            store.parent.mkdir()
            make_gridded().to_zarr(store, mode="w", consolidated=True)
            stores.append(store)
        skill = make_identity_skill([], input_type=types.ALL)
        outs = []
        for i, store in enumerate(stores):
            out = tmp_path / f"out{i}.zarr"
            skill(["-i", str(store), "-o", str(out)])
            outs.append(out)

        for out in outs:
            attrs = xr.open_zarr(out, consolidated=True).attrs
            assert provenance.INPUT_PATH_ATTR not in attrs
            assert not any(str(tmp_path) in str(v) for v in attrs.values())
        assert provenance.hash_zarr(outs[0]) == provenance.hash_zarr(outs[1])

    def test_streaming_write_strips_the_attr(self, tmp_path, gridded_store):
        @weather_skill(
            "stream", "0.1.0", input_type=types.ALL, output_type=types.GRIDDED, streaming=True
        )
        def stream(ds, args):
            """Yield the input back one period at a time."""
            for i in range(ds.sizes["time"]):
                yield ds.isel(time=slice(i, i + 1))

        out = tmp_path / "o.zarr"
        stream(["-i", str(gridded_store), "-o", str(out)])
        assert provenance.INPUT_PATH_ATTR not in xr.open_zarr(out, consolidated=True).attrs


class TestPostWrite:
    def test_runs_after_write_with_output_path(self, tmp_path, gridded_store):
        seen = {}

        def verify(path):
            # The store is fully written by the time the hook runs.
            seen["sizes"] = dict(xr.open_zarr(path, consolidated=True).sizes)
            seen["path"] = path

        skill = make_identity_skill([], post_write=verify)
        out = tmp_path / "o.zarr"
        skill(["-i", str(gridded_store), "-o", str(out)])
        assert seen["path"] == out
        assert seen["sizes"]["time"] == 2

    def test_cache_hit_skips_post_write(self, tmp_path, gridded_store):
        ran = []
        skill = make_identity_skill([], post_write=lambda p: ran.append(p))
        out = tmp_path / "o.zarr"
        argv = ["-i", str(gridded_store), "-o", str(out)]
        skill(argv)
        skill(argv)
        assert len(ran) == 1

    def test_failure_maps_to_usual_exit_codes(self, tmp_path, gridded_store, capsys):
        def verify(path):
            raise DataError("written store failed verification")

        skill = make_identity_skill([], post_write=verify)
        out = tmp_path / "o.zarr"
        with pytest.raises(SystemExit) as exc:
            skill(["-i", str(gridded_store), "-o", str(out)])
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "Error: written store failed verification" in err
        # A failed run never claims success.
        assert "Wrote:" not in err
        # The written store is left in place for inspection.
        assert out.exists()

    def test_streaming_runs_after_final_append(self, tmp_path):
        seen = {}

        def verify(path):
            seen["time"] = xr.open_zarr(path, consolidated=True).sizes["time"]

        @weather_skill("s", "0.1.0", output_type=types.GRIDDED, streaming=True, post_write=verify)
        def fetch(args):
            """Stream."""
            yield make_gridded(n_time=1, start="2026-01-01")
            yield make_gridded(n_time=1, start="2026-01-02")

        fetch(["-o", str(tmp_path / "o.zarr")])
        assert seen["time"] == 2

    def test_png_receives_png_path(self, tmp_path, gridded_store):
        seen = {}

        @weather_skill(
            "plot",
            "0.1.0",
            input_type=types.ALL,
            output_type=types.PNG,
            post_write=lambda p: seen.update(path=p),
        )
        def plot(ds, args):
            """Plot."""
            return FakeFigure()

        out = tmp_path / "p.png"
        plot(["-i", str(gridded_store), "-o", str(out)])
        assert seen["path"] == out
        assert out.exists()

    def test_context_opt_in(self, tmp_path):
        # The body stashes a fetch-discovered value; the post-write hook
        # verifies the written store against it.
        seen = {}

        def verify(path, context):
            seen["calendar"] = context.state["source_calendar"]
            seen["path"] = path

        @weather_skill("f", "0.1.0", output_type=types.GRIDDED, post_write=verify)
        def fetch(args, context):
            """Fetch."""
            context.state["source_calendar"] = "noleap"
            return make_gridded()

        out = tmp_path / "o.zarr"
        fetch(["-o", str(out)])
        assert seen == {"calendar": "noleap", "path": out}

    def test_requires_artifact_output(self):
        with pytest.raises(ValueError, match="post_write"):
            weather_skill("x", "0.1.0", post_write=lambda p: None)(lambda: None)


class TestRunContext:
    def test_function_opt_in_receives_context(self, tmp_path, gridded_store):
        seen = {}

        @weather_skill(
            "ctx",
            "0.1.0",
            input_type=types.ALL,
            output_type=types.GRIDDED,
            start_time=True,
            end_time=True,
        )
        def skill(ds, args, context):
            """Ctx."""
            seen["context"] = context
            return ds.copy()

        out = tmp_path / "o.zarr"
        skill(
            [
                "-i",
                str(gridded_store),
                "-o",
                str(out),
                "--start",
                "2026-01-01",
                "--end",
                "2026-01-05",
            ]
        )
        ctx = seen["context"]
        assert isinstance(ctx, RunContext)
        assert ctx.args.start == "2026-01-01"
        assert ctx.args.end == "2026-01-05"
        assert ctx.start_time == date(2026, 1, 1)
        assert ctx.state == {}
        # The context carries only these three; everything else is on args.
        assert {f.name for f in fields(RunContext)} == {"args", "start_time", "state"}

    def test_start_time_is_none_without_the_toggle(self, tmp_path):
        seen = {}

        @weather_skill("f", "0.1.0", output_type=types.GRIDDED, date=True)
        def fetch(args, context):
            """Fetch one init."""
            seen["context"] = context
            return make_gridded()

        fetch(["--date", "2026-02-03", "-o", str(tmp_path / "o.zarr")])
        assert seen["context"].start_time is None
        assert seen["context"].args.date == "2026-02-03"

    def test_kwargs_function_does_not_receive_context(self, tmp_path, gridded_store):
        # A **params catch-all is not an opt-in; only a named context param is.
        calls = []
        skill = make_identity_skill(calls)
        skill(["-i", str(gridded_store), "-o", str(tmp_path / "o.zarr")])
        assert "context" not in calls[0]

    def test_context_never_enters_recorded_args(self, tmp_path, gridded_store):
        @weather_skill("ctx", "0.1.0", input_type=types.ALL, output_type=types.GRIDDED)
        def skill(ds, args, context):
            """Ctx."""
            return ds.copy()

        out = tmp_path / "o.zarr"
        skill(["-i", str(gridded_store), "-o", str(out)])
        assert "context" not in history_of(out)[-1]["args"]

    def test_hooks_share_state_with_function(self, tmp_path):
        events = []

        def validate(args, context):
            context.state["token"] = "abc"
            events.append("validate")

        def probe(path, context):
            events.append(("probe", context.state["token"]))
            return False

        def encode(ds, context):
            events.append(("encode", context.state["token"], context.state["body"]))

        @weather_skill(
            "f",
            "0.1.0",
            output_type=types.GRIDDED,
            validate_args=validate,
            completeness_probe=probe,
            write_encoding=encode,
        )
        def fetch(args, context):
            """Fetch."""
            context.state["body"] = "ran"
            return make_gridded()

        out = tmp_path / "o.zarr"
        fetch(["-o", str(out)])
        assert events == ["validate", ("encode", "abc", "ran")]
        events.clear()
        # Second run: the probe rejects the hit and sees the fresh run's state.
        fetch(["-o", str(out)])
        assert events == ["validate", ("probe", "abc"), ("encode", "abc", "ran")]

    def test_state_is_fresh_per_run(self, tmp_path, gridded_store):
        seen = []

        def validate(args, context):
            seen.append(dict(context.state))
            context.state["x"] = 1

        skill = make_identity_skill([], validate_args=validate)
        argv = ["-i", str(gridded_store), "-o", str(tmp_path / "o.zarr")]
        skill(argv)
        skill(argv)
        assert seen == [{}, {}]

    def test_latest_resolver_opt_in(self, tmp_path):
        def resolver(args, context):
            context.state["resolved"] = context.state.get("resolved", 0) + 1
            return date(2026, 6, 30)

        seen = {}

        @weather_skill(
            "f",
            "0.1.0",
            output_type=types.GRIDDED,
            start_time=True,
            end_time=True,
            latest_resolver=resolver,
        )
        def fetch(args, context):
            """Fetch."""
            seen["resolved"] = context.state["resolved"]
            return make_gridded()

        fetch(["--start", "latest-1w", "--end", "latest", "-o", str(tmp_path / "o.zarr")])
        assert seen["resolved"] == 1

    def test_normalize_args_opt_in(self, tmp_path, gridded_store):
        def validate(args, context):
            context.state["marker"] = "resolved-in-validate"

        def normalize(raw, context):
            raw["marker"] = context.state["marker"]
            return raw

        skill = make_identity_skill([], validate_args=validate, normalize_args=normalize)
        out = tmp_path / "o.zarr"
        skill(["-i", str(gridded_store), "-o", str(out)])
        assert history_of(out)[-1]["args"]["marker"] == "resolved-in-validate"

    def test_hooks_without_context_param_keep_plain_shapes(self, tmp_path, gridded_store):
        # A one-positional-arg hook (no context param) is called with the
        # plain single-argument shape.
        validated = []
        skill = make_identity_skill([], validate_args=validated.append)
        skill(["-i", str(gridded_store), "-o", str(tmp_path / "o.zarr")])
        assert len(validated) == 1
        assert validated[0].output == str(tmp_path / "o.zarr")

    def test_context_dest_collision_is_declaration_error(self):
        with pytest.raises(ValueError, match="context"):

            @weather_skill("x", "0.1.0", output_type=types.GRIDDED, extra_args=[("--context",)])
            def f(args, context):
                """F."""

    def test_extra_arg_named_context_without_opt_in(self, tmp_path, gridded_store):
        # Without a named context param, an extra arg dest "context" is untouched.
        calls = []
        skill = make_identity_skill(calls, extra_args=[("--context",)])
        skill(["-i", str(gridded_store), "-o", str(tmp_path / "o.zarr"), "--context", "v"])
        assert calls[0]["context"] == "v"


class TestFunctionParams:
    def test_bbox_and_np_data_roundtrip(self, tmp_path, gridded_store):
        received = {}

        @weather_skill(
            "clip",
            "0.1.0",
            input_type=types.ALL,
            output_type=types.GRIDDED,
            bbox=types.REQUIRED,
            dims=True,
        )
        def clip(ds, args):
            """Clip to a bbox."""
            received.update(bbox=args.bbox, dims=args.dims)
            from weather_skills_core import envelope

            return envelope.bbox_subset(ds, args.bbox)

        out = tmp_path / "o.zarr"
        clip(["-i", str(gridded_store), "-o", str(out), "--bbox", "2.5/10.5/0.5/12.5"])
        assert received["bbox"] == (2.5, 10.5, 0.5, 12.5)
        assert received["dims"] is None
        written = xr.open_zarr(out, consolidated=True)
        assert written.sizes["latitude"] == 2
        assert written.sizes["longitude"] == 2
        # The recorded args keep the raw bbox string.
        assert history_of(out)[-1]["args"]["bbox"] == "2.5/10.5/0.5/12.5"

    def test_optional_bbox_defaults_to_none(self, tmp_path, gridded_store):
        received = {}
        skill_calls = []

        @weather_skill(
            "subset", "0.1.0", input_type=types.ALL, output_type=types.GRIDDED, bbox=types.OPTIONAL
        )
        def subset(ds, args):
            """Subset."""
            received["bbox"] = args.bbox
            skill_calls.append(1)
            return ds.copy()

        subset(["-i", str(gridded_store), "-o", str(tmp_path / "o.zarr")])
        assert received["bbox"] is None

    def test_workers_passed_but_not_recorded(self, tmp_path, gridded_store):
        received = {}

        @weather_skill("w", "0.1.0", input_type=types.ALL, output_type=types.GRIDDED, workers=4)
        def w(ds, args):
            """Workers."""
            received["workers"] = args.workers
            return ds.copy()

        w(["-i", str(gridded_store), "-o", str(tmp_path / "o.zarr"), "--workers", "2"])
        assert received["workers"] == 2

    def test_workers_below_one_exits_2(self, tmp_path, gridded_store):
        skill = make_identity_skill([], workers=4)
        with pytest.raises(SystemExit) as exc:
            skill(["-i", str(gridded_store), "-o", str(tmp_path / "o.zarr"), "--workers", "0"])
        assert exc.value.code == 2

    def test_numpy_values_survive_roundtrip(self, tmp_path, gridded_store):
        @weather_skill(
            "scale",
            "0.1.0",
            input_type=types.ALL,
            output_type=types.GRIDDED,
            extra_args=[("--factor", {"type": float})],
        )
        def scale(ds, args):
            """Scale."""
            out = ds.copy()
            out["precip"] = ds["precip"] * args.factor
            return out

        out = tmp_path / "o.zarr"
        scale(["-i", str(gridded_store), "-o", str(out), "--factor", "2.5"])
        written = xr.open_zarr(out, consolidated=True)
        assert float(written["precip"].values.max()) == pytest.approx(2.5)
        assert np.isfinite(written["precip"].values).all()
