"""The weather-skills-core CLI: argument handling, output formats, exit codes."""

import json
from pathlib import Path

import pytest

from weather_skills_core.linting.cli import main

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "linting"


def test_exit_codes_findings_still_exit_zero(capsys):
    assert main(["lint", str(FIXTURES / "shadow_tree")]) == 0
    assert "WSK101" in capsys.readouterr().out


def test_exit_codes_clean_run_exits_zero(capsys):
    assert main(["lint", str(FIXTURES / "clean_tree")]) == 0
    assert "no findings" in capsys.readouterr().out


def test_exit_codes_unlintable_path_is_a_usage_error(tmp_path, capsys):
    assert main(["lint", str(tmp_path)]) == 2
    assert "does not match any skill layout" in capsys.readouterr().err


def test_exit_codes_bad_against_value_is_a_usage_error(capsys):
    code = main(["lint", str(FIXTURES / "clean_tree"), "--against", "/no/such/path"])
    assert code == 2
    assert "--against /no/such/path" in capsys.readouterr().err


def test_exit_codes_missing_subcommand_is_a_usage_error():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_exit_codes_strict_exits_one_at_or_above_the_threshold(capsys):
    assert main(["lint", str(FIXTURES / "shadow_tree"), "--strict", "warning"]) == 1
    capsys.readouterr()
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
            + "from weather_skills_core import Dataset, weather_skill\n"
            + '_SKILL_VERSION = "0.1.0"\n'
            + f"@weather_skill(name={name!r}, version=_SKILL_VERSION)\n"
            + "@weather_skill.argument('-i', '--input', type=Dataset('observations'), required=True, dest='ds')\n"
            + "@weather_skill.argument('--method', type=str, help='x')\n"
            + f"def {name}(ds, output, method, **kwargs):\n    return ds\n"
        )
        (skills / name / "SKILL.md").write_text(
            "# skill\n\n## Usage\n\n### Arguments\n- `--method` — the method.\n- `--input`, `-i` — input Zarr.\n- `--output`, `-o` — output Zarr.\n"
        )
    return skills


def test_rule_selection_default_run_omits_wsk201(tmp_path, capsys):
    skills = _clean_shared_flag_tree(tmp_path)
    assert main(["lint", str(skills)]) == 0
    assert "WSK201" not in capsys.readouterr().out


def test_rule_selection_extend_select_surfaces_wsk201(tmp_path, capsys):
    skills = _clean_shared_flag_tree(tmp_path)
    assert main(["lint", str(skills), "--extend-select", "WSK201"]) == 0
    assert "WSK201" in capsys.readouterr().out


def test_rule_selection_unknown_selector_exits_two_naming_it(tmp_path, capsys):
    skills = _clean_shared_flag_tree(tmp_path)
    assert main(["lint", str(skills), "--select", "WSK999"]) == 2
    assert "WSK999" in capsys.readouterr().err


def test_rule_selection_extend_select_with_strict_warning_trips_on_wsk201(tmp_path, capsys):
    skills = _clean_shared_flag_tree(tmp_path)
    assert main(["lint", str(skills), "--strict", "warning"]) == 0
    capsys.readouterr()
    assert main(["lint", str(skills), "--extend-select", "WSK201", "--strict", "warning"]) == 1
    capsys.readouterr()
    assert main(["lint", str(skills), "--extend-select", "WSK201", "--strict", "error"]) == 0


def test_default_path_no_argument_lints_the_current_directory(capsys, monkeypatch):
    monkeypatch.chdir(FIXTURES / "clean_tree" / "skills" / "clean-skill")
    assert main(["lint"]) == 0
    out = capsys.readouterr().out
    assert "clean-skill — score 100/100" in out


def test_text_format_grouped_by_skill_with_score_lines(capsys):
    main(["lint", str(FIXTURES / "multi_tree")])
    out = capsys.readouterr().out
    assert "alpha — score" in out
    assert "beta — score" in out
    assert "Aggregate score:" in out


def test_text_format_skipped_rules_visible_in_text(capsys):
    main(["lint", str(FIXTURES / "clean_tree" / "skills" / "clean-skill")])
    out = capsys.readouterr().out
    assert "Skipped: WSK201" in out
    assert "Skipped: WSK202" in out


def test_json_format_stable_schema(capsys):
    main(["lint", str(FIXTURES / "multi_tree"), "--format", "json"])
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"findings", "score", "skipped_rules", "notes"}
    assert set(payload["score"]) == {"aggregate", "per_skill"}
    finding = payload["findings"][0]
    assert set(finding) == {"rule", "severity", "skill", "flag", "file", "message"}
    assert payload["score"]["per_skill"].keys() == {
        "alpha/scripts/alpha.py",
        "beta/scripts/beta.py",
        "gamma/scripts/gamma.py",
    }


def test_json_format_skipped_rules_visible_in_json(capsys):
    main(["lint", str(FIXTURES / "clean_tree" / "skills" / "clean-skill"), "--format", "json"])
    payload = json.loads(capsys.readouterr().out)
    assert [s["rule"] for s in payload["skipped_rules"]] == ["WSK201", "WSK202"]
    assert payload["findings"] == []
    assert payload["score"]["per_skill"] == {"clean-skill/scripts/clean_skill.py": 100}
