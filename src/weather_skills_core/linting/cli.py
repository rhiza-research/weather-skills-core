"""The ``weather-skills-core`` console entry point.

Dispatches subcommands; ``lint`` is the only one. Exit codes: 0 whether or
not the lint produced findings (the linter is advisory), 2 for usage errors
(an unlintable path, an unresolvable ``--against`` reference), and 1 only
when ``--strict`` is given and a finding at or above the chosen severity
exists.
"""

import argparse
import sys
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="weather-skills-core",
        description="Tools for the weather-skills ecosystem.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    lint_parser = subparsers.add_parser(
        "lint",
        help="Lint skill declarations for ecosystem conformance (advisory).",
        description=(
            "Lint a skill, a scripts directory, a skills/ tree, or a repo root "
            "against the weather-skills conventions. Advisory: findings never "
            "make the exit code nonzero unless --strict is given."
        ),
    )
    lint_parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Lint target (default: the current directory, auto-detected).",
    )
    lint_parser.add_argument(
        "--against",
        action="append",
        default=[],
        metavar="PATH_OR_REPO",
        help=(
            "Additional corpus for the cross-skill rules: a local path or a public "
            "GitHub repository reference (org/repo[@rev], where rev is a branch, tag, "
            "or full commit SHA). Repeatable."
        ),
    )
    lint_parser.add_argument(
        "--select",
        action="append",
        metavar="CODE",
        help=(
            "Replace the default rule set with these rules. A full code (WSK201) "
            "or a category prefix (WSK2 matches every WSK2xx). Repeatable."
        ),
    )
    lint_parser.add_argument(
        "--extend-select",
        action="append",
        metavar="CODE",
        help=(
            "Add these rules to the set being run (the default set, or --select "
            "if given). Codes or category prefixes. Repeatable. Use "
            "--extend-select WSK201 to survey shared one-off flags."
        ),
    )
    lint_parser.add_argument(
        "--ignore",
        action="append",
        metavar="CODE",
        help=(
            "Remove these rules from the set being run (applied last). Codes or "
            "category prefixes. Repeatable."
        ),
    )
    lint_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text).",
    )
    lint_parser.add_argument(
        "--strict",
        choices=("error", "warning", "info"),
        help=(
            "Exit 1 when any finding at or above this severity exists. Off by "
            "default; nothing in the ecosystem depends on it."
        ),
    )

    args = parser.parse_args(argv)
    from weather_skills_core.errors import UsageError
    from weather_skills_core.linting.render import render_json, render_text
    from weather_skills_core.linting.run import run_lint

    try:
        report = run_lint(
            Path(args.path),
            args.against,
            select=args.select,
            extend_select=args.extend_select,
            ignore=args.ignore,
        )
    except UsageError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(render_json(report) if args.format == "json" else render_text(report))
    if args.strict:
        threshold = ("error", "warning", "info").index(args.strict)
        severities = ("error", "warning", "info")
        if any(severities.index(f.severity) <= threshold for f in report.findings):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
