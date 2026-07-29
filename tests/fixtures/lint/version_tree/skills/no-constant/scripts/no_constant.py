# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "weather-skills-core @ git+https://github.com/rhiza-research/weather-skills-core",
# ]
# ///
"""Lint fixture: no _SKILL_VERSION constant (WSK401). Never executed."""

from weather_skills_core import types, weather_skill


@weather_skill(
    "no-constant",
    "0.1.0",
    input_type=types.ALL,
    output_type=types.ALL,
)
def no_constant(ds, args):
    """Lint fixture; never executed."""
    return ds


if __name__ == "__main__":
    no_constant()
