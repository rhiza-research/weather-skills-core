"""Layout detection, corpus resolution, and the --against fetch paths."""

import shutil
import subprocess
from pathlib import Path

import pytest

from weather_skills_core.errors import UsageError
from weather_skills_core.linting import corpus as corpus_module
from weather_skills_core.linting.corpus import resolve_skill_dirs, sibling_skills
from weather_skills_core.linting.run import run_lint

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "linting"
DECORATED_SCRIPT = "from weather_skills_core import weather_skill\n_SKILL_VERSION = '0.1.0'\n@weather_skill(name='s', version=_SKILL_VERSION)\ndef s():\n    ...\n"


def make_skill(parent, name, *, skill_md=True, decorated=True):
    """A skill directory under ``parent`` with a script and optional SKILL.md."""
    skill = parent / name
    scripts = skill / "scripts"
    scripts.mkdir(parents=True)
    body = DECORATED_SCRIPT if decorated else "x = 1\n"
    (scripts / f"{name.replace('-', '_')}.py").write_text(body)
    if skill_md:
        (skill / "SKILL.md").write_text(f"# {name}\n")
    return skill


def test_layout_detection_single_skill_dir():
    skill = FIXTURES / "clean_tree" / "skills" / "clean-skill"
    dirs, single = resolve_skill_dirs(skill)
    assert dirs == [skill.resolve()] and single


def test_layout_detection_scripts_dir_resolves_to_its_skill():
    scripts = FIXTURES / "clean_tree" / "skills" / "clean-skill" / "scripts"
    dirs, single = resolve_skill_dirs(scripts)
    assert dirs == [scripts.parent.resolve()] and single


def test_layout_detection_skills_tree_passed_directly():
    tree = FIXTURES / "multi_tree" / "skills"
    dirs, single = resolve_skill_dirs(tree)
    assert [d.name for d in dirs] == ["alpha", "beta", "gamma"] and (not single)


def test_layout_detection_repo_root_with_a_skills_tree():
    dirs, single = resolve_skill_dirs(FIXTURES / "multi_tree")
    assert [d.name for d in dirs] == ["alpha", "beta", "gamma"] and (not single)


def test_layout_detection_non_layout_is_a_usage_error(tmp_path):
    with pytest.raises(UsageError, match="does not match any skill layout"):
        resolve_skill_dirs(tmp_path)


def test_layout_detection_file_path_is_a_usage_error():
    target = FIXTURES / "clean_tree" / "skills" / "clean-skill" / "SKILL.md"
    with pytest.raises(UsageError, match="not a directory"):
        resolve_skill_dirs(target)


def test_layout_detection_sibling_discovery():
    alpha = FIXTURES / "multi_tree" / "skills" / "alpha"
    assert [d.name for d in sibling_skills(alpha)] == ["beta", "gamma"]


def test_layout_detection_skills_tree_wins_over_a_root_that_also_has_scripts(tmp_path):
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "tool.py").write_text(DECORATED_SCRIPT)
    make_skill(root / "skills", "alpha")
    make_skill(root / "skills", "beta")
    dirs, single = resolve_skill_dirs(root)
    assert [d.name for d in dirs] == ["alpha", "beta"] and (not single)


def test_layout_detection_non_skill_dir_with_scripts_does_not_shadow_the_skills_tree(tmp_path):
    root = tmp_path / "repo"
    examples = root / "examples"
    (examples / "scripts").mkdir(parents=True)
    (examples / "scripts" / "demo.py").write_text("x = 1\n")
    make_skill(root / "skills", "alpha")
    dirs, single = resolve_skill_dirs(root)
    assert [d.name for d in dirs] == ["alpha"] and (not single)


def test_layout_detection_bare_scripts_dir_without_skill_md_or_decorator_is_not_a_skill(tmp_path):
    skill = make_skill(tmp_path, "bare", skill_md=False, decorated=False)
    with pytest.raises(UsageError, match="does not match any skill layout"):
        resolve_skill_dirs(skill)


def test_layout_detection_decorated_script_without_skill_md_is_a_skill(tmp_path):
    skill = make_skill(tmp_path, "decorated", skill_md=False, decorated=True)
    dirs, single = resolve_skill_dirs(skill)
    assert dirs == [skill.resolve()] and single


def test_against_local_path_local_tree_joins_the_corpus_with_its_label():
    target = FIXTURES / "multi_tree" / "skills" / "alpha"
    against = str(FIXTURES / "against_tree")
    report = run_lint(target, [against], extend_select=["WSK201"])
    dupe = next(f for f in report.findings if f.rule == "WSK201" and f.flag == "--method")
    assert f"remote-skill (--against {against})" in dupe.message
    divergence = next(f for f in report.findings if f.rule == "WSK202" and f.flag == "--method")
    assert "remote-skill" in divergence.message
    assert {f.skill for f in report.findings} == {"alpha"}


def test_against_local_path_missing_local_path_that_is_not_a_github_ref_is_a_usage_error():
    with pytest.raises(UsageError, match="not an existing local path"):
        run_lint(FIXTURES / "clean_tree", ["/no/such/path/anywhere"])


def test_against_local_path_against_the_targets_own_tree_is_excluded():
    tree = FIXTURES / "multi_tree"
    report = run_lint(tree, [str(tree)], extend_select=["WSK201"])
    assert any("is the lint target itself" in note for note in report.notes)
    dupes = [f for f in report.findings if f.rule == "WSK201" and f.flag == "--method"]
    for f in dupes:
        assert f"{f.skill} (" not in f.message


def make_git_repo(tmp_path, source_tree):
    """A local git repository holding the given fixture tree, for file:// clones."""
    repo = tmp_path / "remote-repo"
    shutil.copytree(source_tree, repo)
    env_args = ["-c", "user.email=fixture@example.invalid", "-c", "user.name=Fixture"]
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(
        ["git", *env_args, "-C", str(repo), "commit", "--quiet", "-m", "fixture"], check=True
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "uploadpack.allowAnySHA1InWant", "true"], check=True
    )
    return repo


def head_sha(repo):
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def test_against_git_hub_github_reference_fetched_via_clone(tmp_path, monkeypatch):
    repo = make_git_repo(tmp_path, FIXTURES / "against_tree")
    monkeypatch.setattr(corpus_module, "github_clone_url", lambda reference: f"file://{repo}")
    report = run_lint(
        FIXTURES / "multi_tree" / "skills" / "alpha",
        ["fixture-org/fixture-repo"],
        extend_select=["WSK201"],
    )
    dupe = next(f for f in report.findings if f.rule == "WSK201" and f.flag == "--method")
    assert "remote-skill (--against fixture-org/fixture-repo)" in dupe.message


def test_against_git_hub_no_clone_left_behind(tmp_path, monkeypatch):
    repo = make_git_repo(tmp_path, FIXTURES / "against_tree")
    created = []
    real_fetch = corpus_module._fetch_github

    def spy_fetch(reference, dest):
        created.append(dest)
        return real_fetch(reference, dest)

    monkeypatch.setattr(corpus_module, "github_clone_url", lambda reference: f"file://{repo}")
    monkeypatch.setattr(corpus_module, "_fetch_github", spy_fetch)
    run_lint(FIXTURES / "multi_tree" / "skills" / "alpha", ["fixture-org/fixture-repo"])
    assert created and all(not dest.exists() for dest in created)


def test_against_git_hub_unreachable_reference_is_a_usage_error_naming_it(tmp_path, monkeypatch):
    monkeypatch.setattr(
        corpus_module, "github_clone_url", lambda reference: f"file://{tmp_path}/does-not-exist"
    )
    with pytest.raises(UsageError, match="fixture-org/fixture-repo"):
        run_lint(FIXTURES / "clean_tree", ["fixture-org/fixture-repo"])


def test_against_git_hub_commit_sha_reference_fetched_by_sha(tmp_path, monkeypatch):
    repo = make_git_repo(tmp_path, FIXTURES / "against_tree")
    sha = head_sha(repo)
    monkeypatch.setattr(corpus_module, "github_clone_url", lambda reference: f"file://{repo}")
    report = run_lint(
        FIXTURES / "multi_tree" / "skills" / "alpha",
        [f"fixture-org/fixture-repo@{sha}"],
        extend_select=["WSK201"],
    )
    dupe = next(f for f in report.findings if f.rule == "WSK201" and f.flag == "--method")
    assert f"remote-skill (--against fixture-org/fixture-repo@{sha})" in dupe.message


def test_against_git_hub_sha_clone_leaves_no_checkout_behind(tmp_path, monkeypatch):
    repo = make_git_repo(tmp_path, FIXTURES / "against_tree")
    sha = head_sha(repo)
    created = []
    real_fetch = corpus_module._fetch_github

    def spy_fetch(reference, dest):
        created.append(dest)
        return real_fetch(reference, dest)

    monkeypatch.setattr(corpus_module, "github_clone_url", lambda reference: f"file://{repo}")
    monkeypatch.setattr(corpus_module, "_fetch_github", spy_fetch)
    run_lint(FIXTURES / "multi_tree" / "skills" / "alpha", [f"fixture-org/fixture-repo@{sha}"])
    assert created and all(not dest.exists() for dest in created)


def test_against_git_hub_malformed_reference_is_a_usage_error():
    with pytest.raises(UsageError, match="not an existing local path"):
        run_lint(FIXTURES / "clean_tree", ["no-slash-and-no-such-path"])
