"""Cross-check standard_parameters() against the decorator's actual parser construction."""

import pytest

from weather_skills_core import standard_parameters, types, weather_skill

# A declaration enabling every standard toggle through its dict form (where one
# exists), so the parser carries every standard flag with observable overrides.
DICT_HELP = "override help"
DICT_CHOICES = {
    "start": ["2026-01-01", "2026-01-02"],
    "end": ["2026-01-03"],
    "date": ["2026-01-04"],
    "bbox": ["1/2/3/4"],
    "variable": ["precip", "tmax"],
    "workers": [1, 2, 4],
}


@weather_skill(
    "probe-all-toggles",
    "0.0.0",
    output_type=types.GRIDDED,
    start_time={"required": False, "help": DICT_HELP, "choices": DICT_CHOICES["start"]},
    end_time={"required": False, "help": DICT_HELP, "choices": DICT_CHOICES["end"]},
    date={"required": False, "help": DICT_HELP, "choices": DICT_CHOICES["date"]},
    bbox={"mode": types.OPTIONAL, "help": DICT_HELP, "choices": DICT_CHOICES["bbox"]},
    variable={
        "mode": types.REPEAT,
        "required": True,
        "help": DICT_HELP,
        "choices": DICT_CHOICES["variable"],
    },
    workers={
        "default": 2,
        "required": False,
        "help": DICT_HELP,
        "choices": DICT_CHOICES["workers"],
    },
    title=True,
    dims=True,
    time_dim="time",
)
def probe_all_toggles(args):
    """Probe declaration; never executed."""


@weather_skill(
    "probe-io",
    "0.0.0",
    input_type=types.ALL,
    output_type=types.ALL,
    variable=types.SINGLE,
)
def probe_io(ds, args):
    """Probe declaration; never executed."""


@weather_skill(
    "probe-variadic",
    "0.0.0",
    input_type=types.ALL,
    output_type=types.ALL,
    variadic_input=True,
)
def probe_variadic(datasets, args):
    """Probe declaration; never executed."""


def action_by_dest(parser, dest):
    matches = [a for a in parser._actions if a.dest == dest]
    assert len(matches) == 1, f"expected exactly one action for dest {dest!r}, got {matches}"
    return matches[0]


def by_name(name):
    matches = [p for p in standard_parameters() if p.name == name]
    assert len(matches) == 1
    return matches[0]


class TestToggleSurface:
    def test_every_toggle_matches_the_built_parser(self):
        parser = probe_all_toggles.parser
        for param in standard_parameters():
            if param.kind != "toggle":
                continue
            action = action_by_dest(parser, param.dest)
            assert tuple(action.option_strings) == param.flags, param.name
            if param.type_name is None:
                assert action.type is None, param.name
            else:
                assert action.type is not None and action.type.__name__ == param.type_name

    def test_dict_form_help_lands_where_declared_accepted(self):
        parser = probe_all_toggles.parser
        for param in standard_parameters():
            if param.kind != "toggle":
                continue
            action = action_by_dest(parser, param.dest)
            if param.accepts_help:
                # The declared help leads; a date flag then carries the
                # decorator-appended grammar.
                assert action.help.startswith(DICT_HELP), param.name
            else:
                # No dict form: the decorator-owned help (or none) applies.
                assert action.help != DICT_HELP, param.name

    def test_dict_form_choices_land_where_declared_accepted(self):
        parser = probe_all_toggles.parser
        for param in standard_parameters():
            if param.kind != "toggle":
                continue
            action = action_by_dest(parser, param.dest)
            if param.accepts_choices:
                assert action.choices == DICT_CHOICES[param.dest], param.name
            else:
                assert action.choices is None, param.name

    def test_variable_repeat_mode_is_the_append_arity(self):
        param = by_name("variable")
        assert param.arity == "single_or_append"
        repeat_action = action_by_dest(probe_all_toggles.parser, "variable")
        assert type(repeat_action).__name__ == "_AppendAction"
        single_action = action_by_dest(probe_io.parser, "variable")
        assert type(single_action).__name__ == "_StoreAction"

    def test_single_arity_toggles_store_one_value(self):
        parser = probe_all_toggles.parser
        for param in standard_parameters():
            if param.kind != "toggle" or param.arity != "single":
                continue
            action = action_by_dest(parser, param.dest)
            assert type(action).__name__ == "_StoreAction", param.name

    def test_required_dict_key_rejected_where_declared_unaccepted(self):
        # bbox requiredness is the mode key; a dict "required" must raise at
        # declaration time, matching accepts_required=False.
        assert by_name("bbox").accepts_required is False
        with pytest.raises(ValueError):
            weather_skill(
                "bad-bbox",
                "0.0.0",
                output_type=types.GRIDDED,
                bbox={"mode": types.OPTIONAL, "required": True},
            )

    def test_required_dict_key_honored_where_declared_accepted(self):
        parser = probe_all_toggles.parser
        for param in standard_parameters():
            if param.kind != "toggle" or not param.accepts_required:
                continue
            action = action_by_dest(parser, param.dest)
            declared_required = {"start": False, "end": False, "date": False}.get(param.dest, None)
            if declared_required is not None:
                assert action.required is declared_required, param.name
        # variable declared required=True above; workers required=False.
        assert action_by_dest(parser, "variable").required is True
        assert action_by_dest(parser, "workers").required is False


class TestIoSurface:
    def test_input_and_output_flags_match(self):
        parser = probe_io.parser
        for param in standard_parameters():
            if param.kind != "io":
                continue
            action = action_by_dest(parser, param.dest)
            assert tuple(action.option_strings) == param.flags, param.name

    def test_input_arity_covers_the_variadic_append_form(self):
        assert by_name("input").arity == "single_or_append"
        single = action_by_dest(probe_io.parser, "input")
        assert type(single).__name__ == "_StoreAction"
        variadic = action_by_dest(probe_variadic.parser, "input")
        assert type(variadic).__name__ == "_AppendAction"

    def test_output_is_a_single_store(self):
        assert by_name("output").arity == "single"
        action = action_by_dest(probe_io.parser, "output")
        assert type(action).__name__ == "_StoreAction"


class TestCatalogShape:
    def test_every_parameter_named_once(self):
        names = [p.name for p in standard_parameters()]
        assert len(names) == len(set(names))

    def test_dests_and_flags_are_unique(self):
        params = standard_parameters()
        dests = [p.dest for p in params]
        assert len(dests) == len(set(dests))
        flags = [f for p in params for f in p.flags]
        assert len(flags) == len(set(flags))
