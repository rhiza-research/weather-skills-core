"""The six lint rules and the score rubric, on the committed fixture trees."""

from pathlib import Path

import pytest

from weather_skills_core.errors import UsageError
from weather_skills_core.linting.rules import default_rule_set
from weather_skills_core.linting.run import resolve_rule_set, run_lint

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "linting"


def findings_for(report, skill):
    return [f for f in report.findings if f.skill == skill]


def rules_for(report, skill):
    return sorted({f.rule for f in findings_for(report, skill)})


def score_of(report, skill):
    return next(s["score"] for s in report.skills if s["name"] == skill)


def test_clean_skill_no_findings_and_maximal_score():
    report = run_lint(FIXTURES / "clean_tree", [])
    assert report.findings == []
    assert report.aggregate == 100
    assert score_of(report, "clean-skill") == 100


def test_clean_skill_out_of_tree_single_skill_skips_cross_rules():
    report = run_lint(FIXTURES / "clean_tree" / "skills" / "clean-skill", [])
    assert report.findings == []
    skipped = {s["rule"] for s in report.skipped_rules}
    assert skipped == {"WSK201", "WSK202"}
    assert all("no corpus beyond the target" in s["reason"] for s in report.skipped_rules)


def _shadow_rule_single_skill(tmp_path, *, name, decorator_head, argument_decorators):
    skill = tmp_path / name
    scripts_dir = skill / "scripts"
    scripts_dir.mkdir(parents=True)
    head = decorator_head.rstrip(", ")
    ws = f"@weather_skill(name={name!r}, version=_SKILL_VERSION"
    if head:
        ws += f", {head}"
    ws += ")\n"
    (scripts_dir / "run.py").write_text(
        _PEP723
        + "from weather_skills_core import Dataset, weather_skill\n"
        + '_SKILL_VERSION = "0.1.0"\n'
        + ws
        + argument_decorators
        + "def run(**kwargs):\n    pass\n"
    )
    (skill / "SKILL.md").write_text(_manifest([]))
    return skill


def test_shadow_rule_non_canonical_bbox_fires_wsk101():
    report = run_lint(FIXTURES / "shadow_tree", [])
    shadow_findings = [f for f in report.findings if f.rule == "WSK101"]
    assert {f.flag for f in shadow_findings} == {"-b"}
    assert all(f.severity == "warning" for f in shadow_findings)


def test_shadow_rule_remediation_names_canonical_form():
    report = run_lint(FIXTURES / "shadow_tree", [])
    by_flag = {f.flag: f.message for f in report.findings if f.rule == "WSK101"}
    assert "canonical" in by_flag["-b"]
    assert "--bbox" in by_flag["-b"]


def test_shadow_rule_non_shadowing_argument_does_not_fire():
    report = run_lint(FIXTURES / "shadow_tree", [])
    assert not [f for f in report.findings if f.flag == "--period"]


def test_shadow_rule_canonical_specials_do_not_fire_wsk101():
    report = run_lint(FIXTURES / "shadow_tree", [])
    assert not [f for f in report.findings if f.flag in ("--date", "--input", "--output")]


def test_shadow_rule_freeform_input_do_not_fire_wsk101(tmp_path):
    skill = _shadow_rule_single_skill(
        tmp_path,
        name="io-ok",
        decorator_head="",
        argument_decorators="@weather_skill.argument('-i', '--input', type=Dataset('observations'), required=True)\n",
    )
    report = run_lint(skill, [])
    shadow = [f for f in report.findings if f.rule == "WSK101"]
    assert shadow == []


def test_shadow_rule_non_canonical_bbox_in_isolation_fires_wsk101(tmp_path):
    skill = _shadow_rule_single_skill(
        tmp_path,
        name="bad-bbox",
        decorator_head="",
        argument_decorators="@weather_skill.argument('-b', dest='bbox', help='x')\n",
    )
    report = run_lint(skill, [])
    shadow = [f for f in report.findings if f.rule == "WSK101"]
    assert {f.flag for f in shadow} == {"-b"}


def test_shadow_rule_canonical_variable_does_not_fire_wsk101(tmp_path):
    skill = _shadow_rule_single_skill(
        tmp_path,
        name="ok-variable",
        decorator_head="",
        argument_decorators="@weather_skill.argument('--variable', '-v', help='x')\n",
    )
    report = run_lint(skill, [])
    shadow = [f for f in report.findings if f.rule == "WSK101"]
    assert shadow == []


def test_cross_skill_rules_same_shape_duplicate_fires_wsk201_on_every_holder():
    report = run_lint(FIXTURES / "multi_tree", [], extend_select=["WSK201"])
    dupes = [f for f in report.findings if f.rule == "WSK201" and f.flag == "--method"]
    assert {f.skill for f in dupes} == {"alpha", "beta", "gamma"}
    alpha = next(f for f in dupes if f.skill == "alpha")
    assert "beta (target)" in alpha.message and "gamma (target)" in alpha.message
    assert "propose promoting" in alpha.message


def test_cross_skill_rules_divergent_shape_fires_wsk202_naming_the_difference():
    report = run_lint(FIXTURES / "multi_tree", [])
    divergences = [f for f in report.findings if f.rule == "WSK202"]
    gamma_method = next(f for f in divergences if f.skill == "gamma" and f.flag == "--method")
    assert "choices" in gamma_method.message
    alpha_method = next(f for f in divergences if f.skill == "alpha" and f.flag == "--method")
    assert "gamma (target)" in alpha_method.message
    assert "beta" not in alpha_method.message


def test_cross_skill_rules_type_divergence_detected():
    report = run_lint(FIXTURES / "multi_tree", [])
    window = [f for f in report.findings if f.rule == "WSK202" and f.flag == "--window"]
    assert {f.skill for f in window} == {"alpha", "gamma"}
    assert "type int vs float" in next(f for f in window if f.skill == "alpha").message


def test_cross_skill_rules_upward_discovery_reports_findings_only_for_the_target():
    report = run_lint(FIXTURES / "multi_tree" / "skills" / "alpha", [], extend_select=["WSK201"])
    assert {f.skill for f in report.findings} == {"alpha"}
    assert [s["name"] for s in report.skills] == ["alpha"]
    assert not report.skipped_rules
    dupe = next(f for f in report.findings if f.rule == "WSK201")
    tree = str((FIXTURES / "multi_tree" / "skills").resolve())
    assert f"beta ({tree})" in dupe.message


def test_skill_md_rule_drift_fires_in_both_directions():
    report = run_lint(FIXTURES / "drift_tree", [])
    drift = [f for f in findings_for(report, "drift-skill") if f.rule == "WSK301"]
    messages = {f.flag: f.message for f in drift}
    assert "--window" in messages and "not mentioned in SKILL.md" in messages["--window"]
    assert "--nonexistent" in messages
    assert "does not declare it" in messages["--nonexistent"]


def test_skill_md_rule_missing_manifest_is_its_own_finding():
    report = run_lint(FIXTURES / "drift_tree", [])
    missing = [f for f in findings_for(report, "no-manifest") if f.rule == "WSK301"]
    assert len(missing) == 1
    assert "SKILL.md is missing" in missing[0].message


def test_skill_md_rule_documented_skill_produces_no_drift():
    report = run_lint(FIXTURES / "clean_tree", [])
    assert not [f for f in report.findings if f.rule == "WSK301"]


def test_version_rule_missing_constant_and_unpassed_constant_fire():
    report = run_lint(FIXTURES / "version_tree", [])
    assert rules_for(report, "no-constant") == ["WSK401"]
    assert "no module-level _SKILL_VERSION" in findings_for(report, "no-constant")[0].message
    assert rules_for(report, "literal-version") == ["WSK401"]
    assert "not passed" in findings_for(report, "literal-version")[0].message


def test_version_rule_conformant_version_does_not_fire():
    report = run_lint(FIXTURES / "clean_tree", [])
    assert not [f for f in report.findings if f.rule == "WSK401"]


def test_core_dep_rule_missing_block_and_missing_dependency_fire():
    report = run_lint(FIXTURES / "dep_tree", [])
    assert rules_for(report, "no-block") == ["WSK402"]
    assert "no PEP 723" in findings_for(report, "no-block")[0].message
    assert rules_for(report, "missing-core") == ["WSK402"]
    assert "do not declare weather-skills-core" in findings_for(report, "missing-core")[0].message


def test_core_dep_rule_declared_core_dep_does_not_fire():
    report = run_lint(FIXTURES / "clean_tree", [])
    assert not [f for f in report.findings if f.rule == "WSK402"]


def test_analysis_failures_unanalyzable_scripts_fire_wsk001_and_others_still_lint():
    report = run_lint(FIXTURES / "errors_tree", [])
    assert rules_for(report, "broken-syntax") == ["WSK001"]
    assert rules_for(report, "no-decorator") == ["WSK001"]
    assert findings_for(report, "good-skill") == []
    assert score_of(report, "broken-syntax") == 0
    assert score_of(report, "no-decorator") == 0
    assert score_of(report, "good-skill") == 100


_PEP723 = '# /// script\n# dependencies = ["weather-skills-core"]\n# ///\n'


def _script(skill_name, func_name, argument_decorators):
    return (
        _PEP723
        + "from weather_skills_core import Dataset, weather_skill\n"
        + '_SKILL_VERSION = "0.1.0"\n'
        + f"@weather_skill(name={skill_name!r}, version=_SKILL_VERSION)\n"
        + "@weather_skill.argument('-i', '--input', type=Dataset('observations'), required=True, dest='ds')\n"
        + argument_decorators
        + f"def {func_name}(ds, output, **kwargs):\n    return ds\n"
    )


def _manifest(flags):
    lines = ["# multi-skill", "", "## Usage", "", "### Arguments"]
    lines += [f"- `{flag}` — description." for flag in flags]
    lines += ["- `--input`, `-i` — input Zarr.", "- `--output`, `-o` — output Zarr."]
    return "\n".join(lines) + "\n"


def make_multi_script_skill(tmp_path, *, scripts, manifest_flags):
    """A single skill directory holding several decorated scripts and a manifest."""
    skill = tmp_path / "multi-skill"
    scripts_dir = skill / "scripts"
    scripts_dir.mkdir(parents=True)
    for filename, source in scripts.items():
        (scripts_dir / filename).write_text(source)
    (skill / "SKILL.md").write_text(_manifest(manifest_flags))
    return skill


def test_multi_script_skill_a_skills_own_scripts_are_not_a_corpus(tmp_path):
    skill = make_multi_script_skill(
        tmp_path,
        scripts={
            "one.py": _script(
                "one", "one", "@weather_skill.argument('--shared', type=int, help='x')\n"
            ),
            "two.py": _script(
                "two", "two", "@weather_skill.argument('--shared', type=int, help='x')\n"
            ),
        },
        manifest_flags=["--shared"],
    )
    report = run_lint(skill, [])
    assert not [f for f in report.findings if f.rule in ("WSK201", "WSK202")]
    assert {s["rule"] for s in report.skipped_rules} == {"WSK201", "WSK202"}


def test_multi_script_skill_wsk301_reverse_check_unions_every_script(tmp_path):
    skill = make_multi_script_skill(
        tmp_path,
        scripts={
            "one.py": _script(
                "one", "one", "@weather_skill.argument('--foo', type=int, help='x')\n"
            ),
            "two.py": _script(
                "two", "two", "@weather_skill.argument('--bar', type=int, help='x')\n"
            ),
        },
        manifest_flags=["--foo", "--bar"],
    )
    report = run_lint(skill, [])
    assert [f for f in report.findings if f.rule == "WSK301"] == []


def test_multi_script_skill_wsk301_reverse_check_fires_once_for_an_undeclared_documented_flag(
    tmp_path,
):
    skill = make_multi_script_skill(
        tmp_path,
        scripts={
            "one.py": _script(
                "one", "one", "@weather_skill.argument('--foo', type=int, help='x')\n"
            ),
            "two.py": _script(
                "two", "two", "@weather_skill.argument('--bar', type=int, help='x')\n"
            ),
        },
        manifest_flags=["--foo", "--bar", "--ghost"],
    )
    report = run_lint(skill, [])
    ghost = [f for f in report.findings if f.rule == "WSK301" and f.flag == "--ghost"]
    assert len(ghost) == 1
    assert "does not declare it in any script" in ghost[0].message


def test_multi_script_skill_findings_and_scores_key_by_script_not_display_name(tmp_path):
    skill = make_multi_script_skill(
        tmp_path,
        scripts={
            "one.py": _script(
                "dup", "one", "@weather_skill.argument('-b', dest='bbox', help='x')\n"
            ),
            "two.py": _script(
                "dup", "two", "@weather_skill.argument('--clean', type=int, help='x')\n"
            ),
        },
        manifest_flags=["-b", "--clean"],
    )
    report = run_lint(skill, [])
    assert {s["key"] for s in report.skills} == {
        "multi-skill/scripts/one.py",
        "multi-skill/scripts/two.py",
    }
    assert {s["name"] for s in report.skills} == {"dup"}
    shadow = [f for f in report.findings if f.rule == "WSK101"]
    assert len(shadow) == 1 and shadow[0].file.endswith("one.py")


def test_multi_script_skill_dynamic_arguments_suppresses_the_wsk301_reverse_check(tmp_path):
    skill = tmp_path / "multi-skill"
    scripts_dir = skill / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "one.py").write_text(
        _PEP723
        + "from weather_skills_core import Dataset, weather_skill\n"
        + '_SKILL_VERSION = "0.1.0"\n'
        + "EXTRA = {'type': int}\n"
        + "@weather_skill(name='one', version=_SKILL_VERSION)\n"
        + "@weather_skill.argument('-i', '--input', type=Dataset('observations'), required=True, dest='ds')\n"
        + "@weather_skill.argument('--foo', **EXTRA)\n"
        + "def one(ds, output, **kwargs):\n    return ds\n"
    )
    (skill / "SKILL.md").write_text(_manifest(["--foo", "--bar", "--baz"]))
    report = run_lint(skill, [])
    reverse = [
        f for f in report.findings if f.rule == "WSK301" and "does not declare it" in f.message
    ]
    assert reverse == []
    assert any("reverse check is suppressed" in note for note in report.notes)


def test_resolve_rule_set_default_excludes_wsk201_and_keeps_the_rest():
    active = resolve_rule_set()
    assert "WSK201" not in active
    assert active == default_rule_set()
    assert {"WSK001", "WSK101", "WSK202", "WSK301", "WSK401", "WSK402"} <= active


def test_resolve_rule_set_select_replaces_the_default_set():
    assert resolve_rule_set(select=["WSK101"]) == {"WSK101"}


def test_resolve_rule_set_select_is_repeatable():
    assert resolve_rule_set(select=["WSK101", "WSK301"]) == {"WSK101", "WSK301"}


def test_resolve_rule_set_extend_select_unions_onto_the_default():
    assert resolve_rule_set(extend_select=["WSK201"]) == default_rule_set() | {"WSK201"}


def test_resolve_rule_set_extend_select_unions_onto_select():
    assert resolve_rule_set(select=["WSK101"], extend_select=["WSK201"]) == {"WSK101", "WSK201"}


def test_resolve_rule_set_ignore_subtracts_from_the_default_set():
    assert resolve_rule_set(ignore=["WSK202"]) == default_rule_set() - {"WSK202"}


def test_resolve_rule_set_ignore_of_an_inactive_rule_is_a_silent_no_op():
    assert resolve_rule_set(ignore=["WSK201"]) == default_rule_set()


def test_resolve_rule_set_ignore_is_applied_last():
    assert resolve_rule_set(extend_select=["WSK201"], ignore=["WSK201"]) == default_rule_set()


def test_resolve_rule_set_full_code_matches_only_itself():
    assert resolve_rule_set(select=["WSK201"]) == {"WSK201"}


def test_resolve_rule_set_prefix_selects_a_whole_category():
    assert resolve_rule_set(select=["WSK2"]) == {"WSK201", "WSK202"}


def test_resolve_rule_set_short_prefix_selects_all():
    assert resolve_rule_set(select=["WSK"]) == set(default_rule_set()) | {"WSK201"}


def test_resolve_rule_set_unknown_selector_raises_naming_it():
    with pytest.raises(UsageError) as exc:
        resolve_rule_set(select=["WSK999"])
    assert "WSK999" in str(exc.value)


def test_resolve_rule_set_unknown_ignore_selector_also_raises():
    with pytest.raises(UsageError) as exc:
        resolve_rule_set(ignore=["NOPE"])
    assert "NOPE" in str(exc.value)


def test_selection_end_to_end_default_run_excludes_wsk201():
    report = run_lint(FIXTURES / "multi_tree", [])
    assert not [f for f in report.findings if f.rule == "WSK201"]
    assert [f for f in report.findings if f.rule == "WSK202"]


def test_selection_end_to_end_extend_select_brings_wsk201_back():
    report = run_lint(FIXTURES / "multi_tree", [], extend_select=["WSK201"])
    assert [f for f in report.findings if f.rule == "WSK201"]


def test_selection_end_to_end_select_wsk101_runs_only_that_rule():
    report = run_lint(FIXTURES / "shadow_tree", [], select=["WSK101"])
    assert {f.rule for f in report.findings} == {"WSK101"}


def test_selection_end_to_end_ignore_removes_wsk202_from_the_default_set():
    report = run_lint(FIXTURES / "multi_tree", [], ignore=["WSK202"])
    assert not [f for f in report.findings if f.rule == "WSK202"]


def test_selection_end_to_end_prefix_select_runs_wsk201_and_wsk202():
    report = run_lint(FIXTURES / "multi_tree", [], select=["WSK2"])
    assert {f.rule for f in report.findings} <= {"WSK201", "WSK202"}
    assert [f for f in report.findings if f.rule == "WSK201"]
    assert [f for f in report.findings if f.rule == "WSK202"]


def test_selection_end_to_end_unknown_selector_raises_usage_error():
    with pytest.raises(UsageError) as exc:
        run_lint(FIXTURES / "multi_tree", [], select=["WSK999"])
    assert "WSK999" in str(exc.value)


def test_score_rubric_warning_only_rule_scores_half_of_that_rule():
    report = run_lint(FIXTURES / "shadow_tree", [])
    assert score_of(report, "shadow-skill") == 88


def test_score_rubric_cross_rules_excluded_from_the_denominator_when_skipped():
    report = run_lint(FIXTURES / "clean_tree" / "skills" / "clean-skill", [])
    assert score_of(report, "clean-skill") == 100
    assert {s["rule"] for s in report.skipped_rules} == {"WSK201", "WSK202"}


def test_score_rubric_aggregate_is_the_mean_of_per_skill_scores():
    report = run_lint(FIXTURES / "errors_tree", [])
    scores = [s["score"] for s in report.skills]
    assert report.aggregate == round(sum(scores) / len(scores))
