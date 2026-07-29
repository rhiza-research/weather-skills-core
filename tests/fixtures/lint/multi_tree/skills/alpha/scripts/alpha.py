# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "weather-skills-core @ git+https://github.com/rhiza-research/weather-skills-core",
# ]
# ///
"""Lint fixture: one-off flags shared with sibling skills. Never executed.

The unresolvable import below proves declaration extraction is AST-only: the
linter must analyze this script without importing it (an import would fail).
"""

import fixture_module_that_must_never_be_imported  # noqa: F401

from weather_skills_core import types, weather_skill

_SKILL_VERSION = "0.1.0"


@weather_skill(
    "alpha",
    _SKILL_VERSION,
    input_type=types.ALL,
    output_type=types.ALL,
    extra_args=[
        ("--method", {"choices": ["mean", "sum"], "required": True, "help": "Reducer."}),
        ("--window", {"type": int, "help": "Window width."}),
    ],
)
def alpha(ds, args):
    """Lint fixture; never executed."""
    return ds


if __name__ == "__main__":
    alpha()
