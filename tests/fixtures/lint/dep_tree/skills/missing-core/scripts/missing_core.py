# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "cftime",
# ]
# ///
"""Lint fixture: PEP 723 block without weather-skills-core (WSK402). Never executed."""

from weather_skills_core import types, weather_skill

_SKILL_VERSION = "0.1.0"


@weather_skill(
    "missing-core",
    _SKILL_VERSION,
    input_type=types.ALL,
    output_type=types.ALL,
)
def missing_core(ds, args):
    """Lint fixture; never executed."""
    return ds


if __name__ == "__main__":
    missing_core()
