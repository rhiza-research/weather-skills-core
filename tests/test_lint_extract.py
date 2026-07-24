"""AST extraction of skill declarations: shapes, toggles, version, PEP 723."""

import os
import textwrap
from pathlib import Path

import pytest

from weather_skills_core.lint.extract import (
    extract_script,
    extract_skill,
    normalize_requirement_name,
)

FIXTURES = Path(__file__).parent / "fixtures" / "lint"


def write_script(tmp_path, body):
    skill_dir = tmp_path / "some-skill"
    scripts = skill_dir / "scripts"
    scripts.mkdir(parents=True)
    script = scripts / "some_skill.py"
    script.write_text(textwrap.dedent(body))
    return script, skill_dir


class TestDeclarationExtraction:
    def test_clean_fixture_extracts_the_full_surface(self):
        skill_dir = FIXTURES / "clean_tree" / "skills" / "clean-skill"
        decl = extract_script(skill_dir / "scripts" / "clean_skill.py", skill_dir)
        assert decl.error is None
        assert decl.name == "clean-skill"
        assert decl.version_constant and decl.version_passed
        assert decl.has_input and decl.has_output
        assert decl.input_arity == "single"
        assert decl.toggles["bbox"] == "optional"
        shape = decl.extra_args["smoothing"]
        assert shape.flags == ("--smoothing",)
        assert shape.type_name == "int"
        assert shape.arity == "single"
        assert not shape.dynamic and not shape.dynamic_keys
        assert decl.pep723_deps is not None
        assert "weather-skills-core" in {normalize_requirement_name(d) for d in decl.pep723_deps}

    def test_dict_spec_flag_aliases_repeat_positional(self, tmp_path):
        script, skill_dir = write_script(
            tmp_path,
            '''
            """Doc."""
            from weather_skills_core import weather_skill

            _SKILL_VERSION = "0.1.0"


            @weather_skill(
                "some-skill",
                _SKILL_VERSION,
                extra_args={
                    "sender": {"flag": "--from", "required": True},
                    "item": {"aliases": ["-x"], "repeat": True},
                    "attach": {"nargs": "*", "default": []},
                    "code": {"positional": True},
                    "verbose": {"action": "store_true"},
                },
            )
            def some_skill(sender, item, attach, code, verbose):
                """Doc."""
            ''',
        )
        decl = extract_script(script, skill_dir)
        assert decl.extra_args["sender"].flags == ("--from",)
        assert decl.extra_args["sender"].required is True
        assert decl.extra_args["item"].flags == ("--item", "-x")
        assert decl.extra_args["item"].arity == "append"
        assert decl.extra_args["attach"].nargs == "*"
        assert decl.extra_args["code"].positional and decl.extra_args["code"].flags == ()
        assert decl.extra_args["verbose"].arity == "store_true"

    def test_bare_type_tuple_and_constraint_set_specs(self, tmp_path):
        script, skill_dir = write_script(
            tmp_path,
            '''
            """Doc."""
            from weather_skills_core import weather_skill

            _SKILL_VERSION = "0.1.0"


            @weather_skill(
                "some-skill",
                _SKILL_VERSION,
                extra_args={
                    "factor": int,
                    "flagged": bool,
                    "mode": ("fast", "slow"),
                    "level": {int, range(0, 3)},
                },
            )
            def some_skill(factor, flagged, mode, level):
                """Doc."""
            ''',
        )
        decl = extract_script(script, skill_dir)
        assert decl.extra_args["factor"].type_name == "int"
        assert decl.extra_args["flagged"].arity == "store_true"
        assert decl.extra_args["mode"].choices == ("fast", "slow")
        level = decl.extra_args["level"]
        assert level.type_name == "int"
        assert level.choices == (0, 1, 2)

    def test_non_literal_values_recorded_as_dynamic_with_a_note(self, tmp_path):
        script, skill_dir = write_script(
            tmp_path,
            '''
            """Doc."""
            from weather_skills_core import weather_skill

            _SKILL_VERSION = "0.1.0"
            DEFAULT = 4
            OPTIONS = ["a", "b"]


            @weather_skill(
                "some-skill",
                _SKILL_VERSION,
                workers={"default": DEFAULT, "help": f"threads ({DEFAULT})"},
                extra_args={
                    "pick": {"choices": OPTIONS, "help": "static"},
                },
            )
            def some_skill(workers, pick):
                """Doc."""
            ''',
        )
        decl = extract_script(script, skill_dir)
        assert decl.toggle_enabled("workers")
        pick = decl.extra_args["pick"]
        assert pick.choices is None
        assert pick.dynamic_keys == ("choices",)
        assert any("'pick'" in note and "dynamic" in note for note in decl.notes)

    def test_variadic_and_named_inputs(self, tmp_path):
        script, skill_dir = write_script(
            tmp_path,
            '''
            """Doc."""
            from weather_skills_core import weather_skill

            _SKILL_VERSION = "0.1.0"


            @weather_skill(
                "some-skill",
                _SKILL_VERSION,
                input_type=["any", "any"],
                input_names=["forecast", "mclimate"],
                output_type="png",
            )
            def some_skill(a, b):
                """Doc."""
            ''',
        )
        decl = extract_script(script, skill_dir)
        assert decl.has_input and decl.input_arity == "append"
        assert decl.input_names == ["forecast", "mclimate"]

    def test_extraction_is_ast_only_and_never_imports_the_script(self):
        # The alpha fixture imports a module that does not exist anywhere;
        # extraction succeeds because the script is parsed, not imported.
        skill_dir = FIXTURES / "multi_tree" / "skills" / "alpha"
        decl = extract_script(skill_dir / "scripts" / "alpha.py", skill_dir)
        assert decl.error is None
        assert decl.name == "alpha"


class TestMalformedDeclarationHardening:
    def test_non_sequence_input_type_literal_notes_instead_of_raising(self, tmp_path):
        script, skill_dir = write_script(
            tmp_path,
            '''
            """Doc."""
            from weather_skills_core import weather_skill

            _SKILL_VERSION = "0.1.0"


            @weather_skill("some-skill", _SKILL_VERSION, input_type=5)
            def some_skill(ds):
                """Doc."""
            ''',
        )
        decl = extract_script(script, skill_dir)
        assert decl.error is None
        assert decl.has_input
        assert any("input_type" in note and "arity unknown" in note for note in decl.notes)

    def test_non_string_flag_and_string_aliases_are_noted_and_ignored(self, tmp_path):
        script, skill_dir = write_script(
            tmp_path,
            '''
            """Doc."""
            from weather_skills_core import weather_skill

            _SKILL_VERSION = "0.1.0"


            @weather_skill(
                "some-skill",
                _SKILL_VERSION,
                extra_args={
                    "bad_flag": {"flag": 7},
                    "bad_aliases": {"aliases": "-x"},
                    "bad_choices": {"choices": 3},
                },
            )
            def some_skill(bad_flag, bad_aliases, bad_choices):
                """Doc."""
            ''',
        )
        decl = extract_script(script, skill_dir)
        assert decl.error is None
        # A non-string flag falls back to the derived default; string aliases
        # are dropped rather than spread into single characters.
        assert decl.extra_args["bad_flag"].flags == ("--bad-flag",)
        assert decl.extra_args["bad_aliases"].flags == ("--bad-aliases",)
        assert decl.extra_args["bad_choices"].choices is None
        assert any("'flag' is not a string" in note for note in decl.notes)
        assert any("'aliases' is not a list" in note for note in decl.notes)
        assert any("'choices' is not a list" in note for note in decl.notes)

    def test_extra_args_name_reference_marks_dynamic(self, tmp_path):
        script, skill_dir = write_script(
            tmp_path,
            '''
            """Doc."""
            from weather_skills_core import weather_skill

            _SKILL_VERSION = "0.1.0"
            SHARED = {"pick": {"type": int}}


            @weather_skill("some-skill", _SKILL_VERSION, extra_args=SHARED)
            def some_skill(pick):
                """Doc."""
            ''',
        )
        decl = extract_script(script, skill_dir)
        assert decl.error is None
        assert decl.extra_args == {}
        assert decl.extra_args_dynamic is True
        assert any("reverse check is suppressed" in note for note in decl.notes)

    def test_extra_args_kwargs_merge_marks_dynamic(self, tmp_path):
        script, skill_dir = write_script(
            tmp_path,
            '''
            """Doc."""
            from weather_skills_core import weather_skill

            _SKILL_VERSION = "0.1.0"
            BASE = {"pick": {"type": int}}


            @weather_skill(
                "some-skill",
                _SKILL_VERSION,
                extra_args={**BASE, "extra": {"type": str}},
            )
            def some_skill(pick, extra):
                """Doc."""
            ''',
        )
        decl = extract_script(script, skill_dir)
        assert decl.error is None
        assert decl.extra_args_dynamic is True
        assert "extra" in decl.extra_args  # the literal entry is still extracted

    def test_multiple_decorated_functions_are_noted(self, tmp_path):
        script, skill_dir = write_script(
            tmp_path,
            '''
            """Doc."""
            from weather_skills_core import weather_skill

            _SKILL_VERSION = "0.1.0"


            @weather_skill("first", _SKILL_VERSION)
            def first(ds):
                """Doc."""


            @weather_skill("second", _SKILL_VERSION)
            def second(ds):
                """Doc."""
            ''',
        )
        decl = extract_script(script, skill_dir)
        assert decl.error is None
        assert decl.name == "first"
        assert any("only the first" in note and "second" in note for note in decl.notes)

    def test_duplicate_pep723_blocks_are_noted(self, tmp_path):
        script, skill_dir = write_script(
            tmp_path,
            '''
            # /// script
            # dependencies = ["weather-skills-core"]
            # ///

            # /// script
            # dependencies = ["cftime"]
            # ///
            """Doc."""
            from weather_skills_core import weather_skill

            _SKILL_VERSION = "0.1.0"


            @weather_skill("some-skill", _SKILL_VERSION)
            def some_skill(ds):
                """Doc."""
            ''',
        )
        decl = extract_script(script, skill_dir)
        assert decl.error is None
        assert decl.pep723_deps == ["weather-skills-core"]  # the first block wins
        assert any("PEP 723 script blocks found" in note for note in decl.notes)


class TestExtractionErrors:
    def test_syntax_error_reported_per_script(self):
        skill_dir = FIXTURES / "errors_tree" / "skills" / "broken-syntax"
        decls = extract_skill(skill_dir)
        assert len(decls) == 1
        assert "does not parse" in decls[0].error

    def test_missing_decorator_reported(self):
        skill_dir = FIXTURES / "errors_tree" / "skills" / "no-decorator"
        decls = extract_skill(skill_dir)
        assert len(decls) == 1
        assert "no @weather_skill decorator call" in decls[0].error

    def test_helper_scripts_ignored_when_a_decorated_script_exists(self, tmp_path):
        _script, skill_dir = write_script(
            tmp_path,
            '''
            """Doc."""
            from weather_skills_core import weather_skill

            _SKILL_VERSION = "0.1.0"


            @weather_skill("some-skill", _SKILL_VERSION)
            def some_skill():
                """Doc."""
            ''',
        )
        (skill_dir / "scripts" / "helper.py").write_text('"""Helper, no decorator."""\n')
        decls = extract_skill(skill_dir)
        assert len(decls) == 1
        assert decls[0].name == "some-skill"

    def test_no_scripts_directory(self, tmp_path):
        empty = tmp_path / "empty-skill"
        empty.mkdir()
        decls = extract_skill(empty)
        assert len(decls) == 1
        assert "no scripts/*.py" in decls[0].error

    def test_non_utf8_script_reported_per_script(self, tmp_path):
        script, skill_dir = write_script(tmp_path, '"""Doc."""\n')
        script.write_bytes(b'"""Doc."""\nname = "caf\xe9"\n')  # latin-1, not UTF-8
        decl = extract_script(script, skill_dir)
        assert decl.error is not None
        assert "not valid UTF-8" in decl.error

    @pytest.mark.skipif(os.geteuid() == 0, reason="permission bits do not bind root")
    def test_unreadable_script_reported_per_script(self, tmp_path):
        script, skill_dir = write_script(tmp_path, '"""Doc."""\n')
        script.chmod(0o000)
        try:
            decl = extract_script(script, skill_dir)
        finally:
            script.chmod(0o644)
        assert decl.error is not None
        assert "could not read the script" in decl.error

    @pytest.mark.skipif(os.geteuid() == 0, reason="permission bits do not bind root")
    def test_unlistable_scripts_directory_reported_per_skill(self, tmp_path):
        script, skill_dir = write_script(tmp_path, '"""Doc."""\n')
        scripts_dir = script.parent
        scripts_dir.chmod(0o000)
        try:
            decls = extract_skill(skill_dir)
        finally:
            scripts_dir.chmod(0o755)
        assert len(decls) == 1
        assert "could not list the scripts directory" in decls[0].error


class TestVersionAndDeps:
    def test_missing_constant_and_literal_version(self):
        no_constant = FIXTURES / "version_tree" / "skills" / "no-constant"
        decl = extract_skill(no_constant)[0]
        assert decl.version_constant is False
        literal = FIXTURES / "version_tree" / "skills" / "literal-version"
        decl = extract_skill(literal)[0]
        assert decl.version_constant is True
        assert decl.version_passed is False

    def test_missing_block_and_missing_core_dep(self):
        no_block = FIXTURES / "dep_tree" / "skills" / "no-block"
        assert extract_skill(no_block)[0].pep723_deps is None
        missing_core = FIXTURES / "dep_tree" / "skills" / "missing-core"
        deps = extract_skill(missing_core)[0].pep723_deps
        assert deps == ["cftime"]

    def test_requirement_name_normalization(self):
        assert normalize_requirement_name("weather_skills.core[extra]>=1.0") == (
            "weather-skills-core"
        )
        assert (
            normalize_requirement_name(
                "weather-skills-core @ git+https://github.com/rhiza-research/weather-skills-core"
            )
            == "weather-skills-core"
        )
        assert normalize_requirement_name("   ") is None
