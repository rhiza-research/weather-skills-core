# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "weather-skills-core @ git+https://github.com/rhiza-research/weather-skills-core",
# ]
# ///
"""Lint fixture: a conformant skill next to unanalyzable siblings. Never executed."""

from weather_skills_core import types, weather_skill

_SKILL_VERSION = "0.1.0"


@weather_skill(
    "good-skill",
    _SKILL_VERSION,
    input_type=types.ALL,
    output_type=types.ALL,
)
def good_skill(ds, args):
    """Lint fixture; never executed."""
    return ds


if __name__ == "__main__":
    good_skill()
