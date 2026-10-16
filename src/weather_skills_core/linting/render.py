"""Text and JSON renderers for a lint report.

The JSON schema is stable: ``findings`` (list of ``{rule, severity, skill,
flag, file, message}``), ``score`` (``{"aggregate": int, "per_skill":
{key: int}}`` where ``key`` is the collision-proof relative script path, not
the display name -- two scripts in one skill directory can share a name),
``skipped_rules`` (list of ``{rule, title, reason}``), and ``notes`` (list of
strings). New keys may be added; existing keys keep their meaning.
"""

import json
from dataclasses import asdict

from weather_skills_core.linting.run import LintReport


def render_text(report: LintReport) -> str:
    lines = []
    for skill in report.skills:
        lines.append(f"{skill['name']} — score {skill['score']}/100")
        skill_findings = [f for f in report.findings if f.file == skill["file"]]
        if not skill_findings:
            lines.append("  no findings")
        for finding in skill_findings:
            where = f" [{finding.flag}]" if finding.flag else ""
            lines.append(f"  {finding.rule} {finding.severity}{where} {finding.message}")
        lines.append("")
    for skipped in report.skipped_rules:
        lines.append(f"Skipped: {skipped['rule']} ({skipped['title']}) — {skipped['reason']}")
    for note in report.notes:
        lines.append(f"Note: {note}")
    if report.skipped_rules or report.notes:
        lines.append("")
    lines.append(
        f"Aggregate score: {report.aggregate}/100 "
        f"({len(report.skills)} skill{'s' if len(report.skills) != 1 else ''}, "
        f"{len(report.findings)} finding{'s' if len(report.findings) != 1 else ''})"
    )
    return "\n".join(lines)


def render_json(report: LintReport) -> str:
    payload = {
        "findings": [asdict(f) for f in report.findings],
        "score": {
            "aggregate": report.aggregate,
            "per_skill": {skill["key"]: skill["score"] for skill in report.skills},
        },
        "skipped_rules": report.skipped_rules,
        "notes": report.notes,
    }
    return json.dumps(payload, indent=2)
