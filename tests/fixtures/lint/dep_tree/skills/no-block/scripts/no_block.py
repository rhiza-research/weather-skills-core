"""Lint fixture: no PEP 723 script block (WSK402). Never executed."""

from weather_skills_core import types, weather_skill

_SKILL_VERSION = "0.1.0"


@weather_skill(
    "no-block",
    _SKILL_VERSION,
    input_type=types.ALL,
    output_type=types.ALL,
)
def no_block(ds, args):
    """Lint fixture; never executed."""
    return ds


if __name__ == "__main__":
    no_block()
