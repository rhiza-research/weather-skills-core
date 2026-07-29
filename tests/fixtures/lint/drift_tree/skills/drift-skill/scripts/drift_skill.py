# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "weather-skills-core @ git+https://github.com/rhiza-research/weather-skills-core",
# ]
# ///
"""Lint fixture: SKILL.md drift in both directions. Never executed."""

from weather_skills_core import types, weather_skill

_SKILL_VERSION = "0.1.0"


@weather_skill(
    "drift-skill",
    _SKILL_VERSION,
    input_type=types.ALL,
    output_type=types.ALL,
    extra_args=[("--window", {"type": int, "help": "Declared but not documented (WSK301)."})],
)
def drift_skill(ds, args):
    """Lint fixture; never executed."""
    return ds


if __name__ == "__main__":
    drift_skill()
