# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "weather-skills-core @ git+https://github.com/rhiza-research/weather-skills-core",
# ]
# ///
"""Lint fixture: an --against corpus skill sharing --method at a divergent shape."""

from weather_skills_core import types, weather_skill

_SKILL_VERSION = "0.1.0"


@weather_skill(
    "remote-skill",
    _SKILL_VERSION,
    input_type=types.ALL,
    output_type=types.ALL,
    extra_args=[("--method", {"choices": ["p10", "p90"], "required": True, "help": "Percentile."})],
)
def remote_skill(ds, args):
    """Lint fixture; never executed."""
    return ds


if __name__ == "__main__":
    remote_skill()
