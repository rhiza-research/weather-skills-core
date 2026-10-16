# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "weather-skills-core @ git+https://github.com/rhiza-research/weather-skills-core",
# ]
# ///
"""Lint fixture. Never executed."""

from pathlib import Path

from weather_skills_core import Dataset, weather_skill

_SKILL_VERSION = "0.1.0"


@weather_skill(
    name="gamma",
    version=_SKILL_VERSION,
)
@weather_skill.argument("-i", "--input", type=Dataset("observations"), required=True, dest="ds")
@weather_skill.argument("--method", type=str, help="Aggregation method.", choices=["sum", "mean"])
@weather_skill.argument("--window", type=float, help="Window width.")
def gamma(ds, **kwargs):
    """Lint fixture; never executed."""
    return ds


if __name__ == "__main__":
    gamma()
