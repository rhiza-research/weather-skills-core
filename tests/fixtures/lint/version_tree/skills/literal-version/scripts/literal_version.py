# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "weather-skills-core @ git+https://github.com/rhiza-research/weather-skills-core",
# ]
# ///
"""Lint fixture: _SKILL_VERSION defined but a literal passed instead (WSK401)."""

from weather_skills_core import types, weather_skill

_SKILL_VERSION = "0.1.0"


@weather_skill(
    "literal-version",
    "0.2.0",
    input_type=types.ALL,
    output_type=types.ALL,
)
def literal_version(ds, args):
    """Lint fixture; never executed."""
    return ds


if __name__ == "__main__":
    literal_version()
