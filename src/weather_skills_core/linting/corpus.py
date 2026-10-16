"""Lint-target layout detection and cross-skill corpus resolution.

The lint target resolves to one or more skill directories by layout
auto-detection. The cross-skill corpus is the target's own tree, plus upward
discovery when the target is a single skill inside an enclosing skills tree
(siblings are context only: findings are reported for the target alone), plus
every ``--against`` value -- a local path or a GitHub repository reference.

A GitHub reference is fetched shallowly (blob-filtered and sparse) into a
temporary directory that only the declaration files are read from and that is
removed when the lint run ends; no clone is retained and no credentials are
used (public repositories only). A branch or tag revision is fetched with
``git clone --branch``; a commit SHA is fetched with ``git init`` + ``git
fetch --depth 1 origin <sha>`` + ``git checkout FETCH_HEAD``, because ``git
clone --branch`` accepts only branch and tag names.
"""

import os
import re
import subprocess
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from weather_skills_core.errors import UsageError
from weather_skills_core.linting.extract import SkillDeclaration, extract_skill

#: Ceiling on any one git subprocess. The fetches are shallow and
#: blob-filtered, so a healthy clone finishes well inside this; expiry is a
#: usage error naming the reference rather than an indefinite hang.
GIT_TIMEOUT_SECONDS = 300

_GITHUB_REF_RE = re.compile(
    r"^(?:https?://github\.com/)?"
    r"(?P<org>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+?)"
    r"(?:\.git)?(?:@(?P<rev>[^@/]+))?$"
)

#: A revision token that can only be a commit SHA lookup: full 40-hex is
#: fetched by SHA directly; 7-39 hex is tried as a branch/tag first (hex
#: branch names exist) with the SHA fetch as the fallback.
_HEX_REV_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")

#: A ``@weather_skill`` decorator application, matched on source text (not the
#: AST) so a script that does not parse still marks its directory as a skill.
_DECORATOR_RE = re.compile(r"@\s*(?:[A-Za-z_][\w.]*\.)?weather_skill\s*\(")


@dataclass
class CorpusSkill:
    """One declaration in the corpus, labeled with where it came from."""

    decl: SkillDeclaration
    source: str  # "target", the enclosing tree path, or "--against <ref>"
    is_target: bool


def _is_skill_dir(path: Path) -> bool:
    """A skill directory holds ``scripts/*.py`` plus evidence of skill-ness.

    The evidence is a ``SKILL.md`` manifest or a script applying the
    ``@weather_skill`` decorator; a bare ``scripts/*.py`` directory (an
    ``examples/`` folder, a tools directory) is not a skill and must not
    shadow a real ``skills/`` tree during layout detection.
    """
    scripts_dir = path / "scripts"
    if not scripts_dir.is_dir() or not any(scripts_dir.glob("*.py")):
        return False
    if (path / "SKILL.md").is_file():
        return True
    for script in scripts_dir.glob("*.py"):
        try:
            source = script.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if _DECORATOR_RE.search(source):
            return True
    return False


def _skill_children(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return [child for child in sorted(path.iterdir()) if _is_skill_dir(child)]


def resolve_skill_dirs(path: Path) -> tuple[list[Path], bool]:
    """Auto-detect the layout at ``path`` and return ``(skill dirs, single_skill)``.

    Recognized layouts, checked in order: a repo root holding a ``skills/``
    tree; a directory of skill directories (a ``skills/`` tree passed
    directly); a single skill directory (``scripts/*.py`` plus a SKILL.md or
    a ``@weather_skill``-decorated script); a ``scripts`` directory (the
    skill is its parent). A ``skills/`` tree wins over the other layouts, so
    a repo root that also carries its own ``scripts/*.py`` (or non-skill
    directories that do) resolves to the tree rather than misdetecting as
    one skill. Anything else raises :class:`UsageError` (exit 2).
    """
    p = path.resolve()
    if not p.is_dir():
        raise UsageError(
            f"{path} is not a directory; lint a skill directory, a scripts directory, "
            "a skills/ tree, or a repo root holding one."
        )
    try:
        nested = _skill_children(p / "skills")
        if nested:
            return nested, False
        children = _skill_children(p)
        if children:
            return children, False
        if _is_skill_dir(p):
            return [p], True
        if p.name == "scripts" and any(p.glob("*.py")):
            return [p.parent], True
    except OSError as exc:
        raise UsageError(f"could not scan {path} for skill layouts: {exc}") from exc
    raise UsageError(
        f"{path} does not match any skill layout (no scripts/*.py alongside a SKILL.md "
        "or a @weather_skill script, no */scripts/*.py, no skills/*)."
    )


def sibling_skills(skill_dir: Path) -> list[Path]:
    """Skill directories that share the target's enclosing tree (upward discovery)."""
    parent = skill_dir.resolve().parent
    try:
        return [child for child in _skill_children(parent) if child != skill_dir.resolve()]
    except OSError as exc:
        raise UsageError(f"could not scan {parent} for sibling skills: {exc}") from exc


def github_clone_url(reference: str) -> str:
    """The public HTTPS clone URL for a GitHub repository reference."""
    m = _GITHUB_REF_RE.match(reference)
    if m is None:
        raise UsageError(f"--against {reference}: not a GitHub repository reference.")
    return f"https://github.com/{m['org']}/{m['repo']}.git"


def _run_git(args: list[str], reference: str) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            ["git", *args],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            check=False,
        )
    except FileNotFoundError as exc:
        raise UsageError(
            f"--against {reference}: git is not installed or not on PATH; "
            "a GitHub reference needs the git binary."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise UsageError(
            f"--against {reference}: git {args[0]} timed out after {GIT_TIMEOUT_SECONDS} seconds."
        ) from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        raise UsageError(
            f"--against {reference}: could not fetch the repository"
            + (f" ({detail[-1]})" if detail else "")
        )
    return result


def _widen_when_no_skills_tree(dest: Path, reference: str) -> None:
    """Drop the ``skills/``-only sparse checkout when the repo has no such tree.

    A repository without a ``skills/`` directory falls back to a full (still
    shallow) checkout so layout detection can run on it.
    """
    if not (dest / "skills").is_dir():
        _run_git(["-C", str(dest), "sparse-checkout", "disable"], reference)


def _fetch_sha(reference: str, url: str, sha: str, dest: Path) -> None:
    """Shallow-fetch one commit by SHA (``git clone --branch`` cannot take one).

    ``git init`` + ``git fetch --depth 1 origin <sha>`` transfers just that
    commit; the sparse checkout is narrowed to ``skills/`` before ``git
    checkout FETCH_HEAD`` so only the skill declarations are materialized.
    """
    _run_git(["init", "--quiet", str(dest)], reference)
    _run_git(["-C", str(dest), "remote", "add", "origin", url], reference)
    _run_git(
        ["-C", str(dest), "fetch", "--quiet", "--depth", "1", "--filter=blob:none", "origin", sha],
        reference,
    )
    _run_git(["-C", str(dest), "sparse-checkout", "set", "skills"], reference)
    _run_git(["-C", str(dest), "checkout", "--quiet", "FETCH_HEAD"], reference)
    _widen_when_no_skills_tree(dest, reference)


def _fetch_github(reference: str, dest: Path) -> None:
    """Shallow-fetch just the declaration files of a public GitHub repository.

    ``git clone --depth 1 --filter=blob:none --sparse`` transfers the named
    commit only, with file contents fetched on demand; the sparse checkout is
    then narrowed to ``skills/`` so only the skill declarations are
    materialized (widened again when the repo has no ``skills/`` tree).

    The revision may be a branch name, a tag name, or a commit SHA. ``git
    clone --branch`` accepts only branch and tag names, so a full 40-hex
    revision goes straight to the fetch-by-SHA path; a 7-39-hex revision is
    tried as a branch/tag first (hex-named branches exist) and falls back to
    the SHA fetch, which requires the full 40-hex form on GitHub.
    """
    m = _GITHUB_REF_RE.match(reference)
    rev = m["rev"] if m else None
    url = github_clone_url(reference)
    if rev and len(rev) == 40 and _HEX_REV_RE.match(rev):
        _fetch_sha(reference, url, rev, dest)
        return
    clone_args = ["clone", "--quiet", "--depth", "1", "--filter=blob:none", "--sparse"]
    if rev:
        clone_args += ["--branch", rev]
    clone_args += [url, str(dest)]
    try:
        _run_git(clone_args, reference)
    except UsageError:
        # git clone removes the directory it created when it fails, so the
        # fallback fetch starts from a clean destination.
        if not (rev and _HEX_REV_RE.match(rev)):
            raise
        _fetch_sha(reference, url, rev, dest)
        return
    _run_git(["-C", str(dest), "sparse-checkout", "set", "skills"], reference)
    _widen_when_no_skills_tree(dest, reference)


def resolve_against(value: str, stack: ExitStack) -> list[Path]:
    """Skill directories for one ``--against`` value.

    An existing local path is layout-detected in place. Anything else must be
    a GitHub repository reference (``org/repo``, ``org/repo@rev``, or an
    ``https://github.com/...`` URL), fetched into a temporary directory
    registered on ``stack`` for removal when the lint run ends.
    """
    local = Path(value)
    if local.exists():
        dirs, _ = resolve_skill_dirs(local)
        return dirs
    if _GITHUB_REF_RE.match(value) and "/" in value:
        tmpdir = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="wsk-against-")))
        _fetch_github(value, tmpdir / "repo")
        dirs, _ = resolve_skill_dirs(tmpdir / "repo")
        return dirs
    raise UsageError(
        f"--against {value}: not an existing local path or a GitHub repository "
        "reference (org/repo[@rev])."
    )


def build_corpus(
    target_path: Path, against: list[str], stack: ExitStack
) -> tuple[list[CorpusSkill], list[str]]:
    """Resolve the full corpus for a lint run.

    Returns the corpus (target declarations first) and corpus notes (context
    skills that could not be analyzed and were excluded). Target declarations
    keep their extraction errors -- those become per-skill findings.
    """
    notes: list[str] = []
    target_dirs, single_skill = resolve_skill_dirs(target_path)
    corpus: list[CorpusSkill] = [
        CorpusSkill(decl=decl, source="target", is_target=True)
        for skill_dir in target_dirs
        for decl in extract_skill(skill_dir)
    ]

    context_dirs: list[tuple[Path, str]] = []
    if single_skill:
        tree = target_dirs[0].resolve().parent
        context_dirs += [(sibling, str(tree)) for sibling in sibling_skills(target_dirs[0])]
    for value in against:
        context_dirs += [
            (skill_dir, f"--against {value}") for skill_dir in resolve_against(value, stack)
        ]

    # An --against value resolving back to the target's own tree (or a skill
    # directory both siblings and --against contributed) would make every
    # skill collide with its own duplicate; compare resolved paths and keep
    # each context directory at most once, never a target directory.
    target_set = {skill_dir.resolve() for skill_dir in target_dirs}
    seen: set[Path] = set()
    deduped: list[tuple[Path, str]] = []
    for skill_dir, source in context_dirs:
        resolved = skill_dir.resolve()
        if resolved in target_set:
            notes.append(
                f"corpus skill {skill_dir.name} ({source}) is the lint target itself "
                "and was excluded"
            )
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append((skill_dir, source))

    for skill_dir, source in deduped:
        for decl in extract_skill(skill_dir):
            if decl.error is not None:
                notes.append(
                    f"corpus skill {decl.display_name} ({source}) could not be analyzed "
                    f"and was excluded: {decl.error}"
                )
                continue
            corpus.append(CorpusSkill(decl=decl, source=source, is_target=False))
    return corpus, notes
