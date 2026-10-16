"""Lint-run orchestration and the deterministic score rubric.

Scoring: each skill's score is the mean of its per-rule scores, on a 0-100
scale. A rule with no findings for the skill scores 1.0; otherwise the rule
scores by its worst finding's severity -- ``error`` 0.0, ``warning`` 0.5,
``info`` 0.8. The rules in the denominator are the active per-skill rules plus, when
a corpus beyond the target exists, the active cross-skill rules; rules that
did not run (skipped for lack of a corpus, or not in the resolved rule set)
are excluded from the denominator entirely (never silently scored as clean). A
skill whose script could not be analyzed (WSK001) scores 0. The aggregate is
the mean of the per-skill scores. Scores inform; maintainers decide.
"""

from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

from weather_skills_core.errors import UsageError
from weather_skills_core.linting.corpus import build_corpus
from weather_skills_core.linting.rules import (
    CROSS_SKILL_RULES,
    PER_SKILL_RULES,
    RULES,
    Finding,
    default_rule_set,
    expand_selector,
    lint_corpus,
)

SEVERITY_RULE_SCORE = {"error": 0.0, "warning": 0.5, "info": 0.8}


def resolve_rule_set(
    select: list[str] | None = None,
    extend_select: list[str] | None = None,
    ignore: list[str] | None = None,
) -> set[str]:
    """Resolve the active rule set with ruff's select/extend-select/ignore semantics.

    ``base`` is the codes named by ``select`` if any were given, else the
    default set; then ``extend_select`` is unioned on; then ``ignore`` is
    subtracted. Each selector is a full rule code (``WSK201``) or a category
    prefix (``WSK2`` matches every ``WSK2xx``); all three arguments are lists
    (repeatable flags). A selector matching no known rule code raises
    :class:`weather_skills_core.errors.UsageError` (exit 2 at the CLI); an
    ``ignore`` selector for a rule that is valid but not in the active set is a
    silent no-op.
    """

    def _expand(selectors: list[str] | None, flag: str) -> set[str]:
        codes: set[str] = set()
        for selector in selectors or []:
            matched = expand_selector(selector)
            if not matched:
                raise UsageError(
                    f"unknown lint rule selector {selector!r} for {flag}: "
                    "matches no known rule code."
                )
            codes |= matched
        return codes

    base = _expand(select, "--select") if select else default_rule_set()
    active = base | _expand(extend_select, "--extend-select")
    active -= _expand(ignore, "--ignore")
    return active


@dataclass
class LintReport:
    findings: list[Finding]
    skills: list[dict]  # per target skill: name, file, score, ordered as linted
    aggregate: int
    skipped_rules: list[dict]  # {"rule", "title", "reason"}
    notes: list[str] = field(default_factory=list)


def skill_score(findings: list[Finding], corpus_available: bool, active_rules: set[str]) -> int:
    """The rubric score for one skill's findings, over the active rules only."""
    if any(f.rule == "WSK001" for f in findings):
        return 0
    candidates = PER_SKILL_RULES + (CROSS_SKILL_RULES if corpus_available else ())
    applicable = [rule_id for rule_id in candidates if rule_id in active_rules]
    if not applicable:
        return 100
    per_rule = []
    for rule_id in applicable:
        rule_findings = [f for f in findings if f.rule == rule_id]
        if not rule_findings:
            per_rule.append(1.0)
        else:
            per_rule.append(min(SEVERITY_RULE_SCORE[f.severity] for f in rule_findings))
    return round(100 * sum(per_rule) / len(per_rule))


def run_lint(
    target: Path,
    against: list[str],
    select: list[str] | None = None,
    extend_select: list[str] | None = None,
    ignore: list[str] | None = None,
) -> LintReport:
    """Lint ``target`` (with optional ``--against`` corpora) and build the report.

    ``select``/``extend_select``/``ignore`` are the ruff-style rule selectors
    (see :func:`resolve_rule_set`); with none given, the default rule set runs.

    Raises :class:`weather_skills_core.errors.UsageError` for an unlintable
    target path, an unresolvable ``--against`` value, or an unknown rule
    selector (exit 2 at the CLI).
    """
    active_rules = resolve_rule_set(select, extend_select, ignore)
    with ExitStack() as stack:
        corpus, notes = build_corpus(target, against, stack)
        # A cross-skill comparison needs at least two distinct skill
        # directories: a single skill's own sibling scripts are not a corpus,
        # so a lone skill (however many scripts) has no corpus regardless of
        # declaration count.
        distinct_dirs = {cs.decl.skill_dir.resolve() for cs in corpus if cs.decl.error is None}
        corpus_available = len(distinct_dirs) >= 2
        findings = lint_corpus(corpus, corpus_available, active_rules)

    skills = []
    scores = []
    for cs in corpus:
        if not cs.is_target:
            continue
        # Group findings by the script path, not the display name: two scripts
        # in one skill directory can share a name, and the file path is their
        # collision-proof identity (it is what _finding records).
        skill_findings = [f for f in findings if f.file == str(cs.decl.script)]
        score = skill_score(skill_findings, corpus_available, active_rules)
        skills.append(
            {
                "name": cs.decl.display_name,
                "key": cs.decl.key,
                "file": str(cs.decl.script),
                "score": score,
            }
        )
        scores.append(score)

    # Surface each target script's extraction notes (a dynamic argument, a
    # skipped second decorated function, a duplicate PEP 723 block) into the
    # report, where render prints them; without this they stay in the buried
    # per-declaration notes field and never reach the reader.
    for cs in corpus:
        if cs.is_target and cs.decl.error is None:
            for note in cs.decl.notes:
                notes.append(f"{cs.decl.key}: {note}")

    skipped = []
    if not corpus_available:
        for rule_id in CROSS_SKILL_RULES:
            skipped.append(
                {
                    "rule": rule_id,
                    "title": RULES[rule_id].title,
                    "reason": (
                        "no corpus beyond the target; lint within a skills tree or pass --against"
                    ),
                }
            )

    aggregate = round(sum(scores) / len(scores)) if scores else 0
    return LintReport(
        findings=findings,
        skills=skills,
        aggregate=aggregate,
        skipped_rules=skipped,
        notes=notes,
    )
