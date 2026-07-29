"""The weather-skills-core CLI: argument handling, output formats, exit codes."""

import json
from pathlib import Path

import pytest

from weather_skills_core.cli import main

FIXTURES = Path(__file__).parent / "fixtures" / "lint"


class TestExitCodes:
    def test_findings_still_exit_zero(self, capsys):
        assert main(["lint", str(FIXTURES / "shadow_tree")]) == 0
        assert "WSK101" in capsys.readouterr().out

    def test_clean_run_exits_zero(self, capsys):
        assert main(["lint", str(FIXTURES / "clean_tree")]) == 0
        assert "no findings" in capsys.readouterr().out

    def test_unlintable_path_is_a_usage_error(self, tmp_path, capsys):
        assert main(["lint", str(tmp_path)]) == 2
        assert "does not match any skill layout" in capsys.readouterr().err

    def test_bad_against_value_is_a_usage_error(self, capsys):
        code = main(["lint", str(FIXTURES / "clean_tree"), "--against", "/no/such/path"])
        assert code == 2
        assert "--against /no/such/path" in capsys.readouterr().err

    def test_missing_subcommand_is_a_usage_error(self):
        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code == 2

    def test_strict_exits_one_at_or_above_the_threshold(self, capsys):
        assert main(["lint", str(FIXTURES / "shadow_tree"), "--strict", "warning"]) == 1
        capsys.readouterr()
        # Warnings only: an error-level threshold does not trip.
        assert main(["lint", str(FIXTURES / "shadow_tree"), "--strict", "error"]) == 0


_PEP723 = '# /// script\n# dependencies = ["weather-skills-core"]\n# ///\n'


def _clean_shared_flag_tree(root):
    """A skills/ tree of two otherwise-clean skills sharing one same-shape flag.

    The only finding this tree can produce is WSK201 (the shared one-off
    ``--method``), and only when WSK201 is opted in -- so it isolates the
    WSK201/selection behavior from every other rule.
    """
    skills = root / "skills"
    for name in ("alpha", "beta"):
        scripts_dir = skills / name / "scripts"
        scripts_dir.mkdir(parents=True)
        (scripts_dir / f"{name}.py").write_text(
            _PEP723
            + "from weather_skills_core import types, weather_skill\n"
            + '_SKILL_VERSION = "0.1.0"\n'
            + f"@weather_skill({name!r}, _SKILL_VERSION, input_type=types.ALL, "
            + "output_type=types.ALL, extra_args=[('--method', {'type': str, 'help': 'x'})])\n"
            + f"def {name}(ds, args):\n    return ds\n"
        )
        (skills / name / "SKILL.md").write_text(
            "# skill\n\n## Usage\n\n### Arguments\n"
            "- `--method` — the method.\n"
            "- `--input`, `-i` — input Zarr.\n"
            "- `--output`, `-o` — output Zarr.\n"
        )
    return skills


class TestRuleSelection:
    def test_default_run_omits_wsk201(self, tmp_path, capsys):
        skills = _clean_shared_flag_tree(tmp_path)
        assert main(["lint", str(skills)]) == 0
        assert "WSK201" not in capsys.readouterr().out

    def test_extend_select_surfaces_wsk201(self, tmp_path, capsys):
        skills = _clean_shared_flag_tree(tmp_path)
        assert main(["lint", str(skills), "--extend-select", "WSK201"]) == 0
        assert "WSK201" in capsys.readouterr().out

    def test_unknown_selector_exits_two_naming_it(self, tmp_path, capsys):
        skills = _clean_shared_flag_tree(tmp_path)
        assert main(["lint", str(skills), "--select", "WSK999"]) == 2
        assert "WSK999" in capsys.readouterr().err

    def test_extend_select_with_strict_warning_trips_on_wsk201(self, tmp_path, capsys):
        skills = _clean_shared_flag_tree(tmp_path)
        # WSK201 is a warning: the default run is clean (exit 0 even at
        # --strict warning); opting WSK201 in trips --strict warning but not
        # --strict error (the finding is below the error threshold).
        assert main(["lint", str(skills), "--strict", "warning"]) == 0
        capsys.readouterr()
        assert main(["lint", str(skills), "--extend-select", "WSK201", "--strict", "warning"]) == 1
        capsys.readouterr()
        assert main(["lint", str(skills), "--extend-select", "WSK201", "--strict", "error"]) == 0


class TestDefaultPath:
    def test_no_argument_lints_the_current_directory(self, capsys, monkeypatch):
        monkeypatch.chdir(FIXTURES / "clean_tree" / "skills" / "clean-skill")
        assert main(["lint"]) == 0
        out = capsys.readouterr().out
        assert "clean-skill — score 100/100" in out


class TestTextFormat:
    def test_grouped_by_skill_with_score_lines(self, capsys):
        main(["lint", str(FIXTURES / "multi_tree")])
        out = capsys.readouterr().out
        assert "alpha — score" in out
        assert "beta — score" in out
        assert "Aggregate score:" in out

    def test_skipped_rules_visible_in_text(self, capsys):
        main(["lint", str(FIXTURES / "clean_tree" / "skills" / "clean-skill")])
        out = capsys.readouterr().out
        assert "Skipped: WSK201" in out
        assert "Skipped: WSK202" in out


class TestJsonFormat:
    def test_stable_schema(self, capsys):
        main(["lint", str(FIXTURES / "multi_tree"), "--format", "json"])
        payload = json.loads(capsys.readouterr().out)
        assert set(payload) == {"findings", "score", "skipped_rules", "notes"}
        assert set(payload["score"]) == {"aggregate", "per_skill"}
        finding = payload["findings"][0]
        assert set(finding) == {"rule", "severity", "skill", "flag", "file", "message"}
        # per_skill keys are the collision-proof relative script path, not the
        # display name (two scripts in one skill directory can share a name).
        assert payload["score"]["per_skill"].keys() == {
            "alpha/scripts/alpha.py",
            "beta/scripts/beta.py",
            "gamma/scripts/gamma.py",
        }

    def test_skipped_rules_visible_in_json(self, capsys):
        main(
            [
                "lint",
                str(FIXTURES / "clean_tree" / "skills" / "clean-skill"),
                "--format",
                "json",
            ]
        )
        payload = json.loads(capsys.readouterr().out)
        assert [s["rule"] for s in payload["skipped_rules"]] == ["WSK201", "WSK202"]
        assert payload["findings"] == []
        assert payload["score"]["per_skill"] == {"clean-skill/scripts/clean_skill.py": 100}
