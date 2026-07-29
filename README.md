# weather-skills-core

Core library for weather skills. It provides the `@weather_skill` decorator,
which owns everything around a skill's domain logic: CLI construction from a
declaration, input reading, envelope validation, the relative-date grammar,
provenance stamping (`weather_skills_history`), the cache-hit short-circuit,
and output writing (Zarr, streaming Zarr appends, or PNG figures).

A skill declares its surface and keeps only its domain logic:

```python
from weather_skills_core import types, weather_skill

_SKILL_VERSION = "0.1.0"


@weather_skill(
    "my-fancy-skill",
    _SKILL_VERSION,
    input_type=[types.FORECAST, types.STATION],
    output_type=types.FORECAST,
    input_names=["forecast", "stations"],
    start_time=True,
    end_time=True,
    extra_args=[
        ("--corr-coefficient", {"type": int, "required": True}),
        ("--interpolation-factor", {"type": int, "choices": [0, 1, 2]}),
    ],
)
def my_fancy_skill(forecast_ds, station_ds, args):
    """Correct a forecast against station observations."""
    window = forecast_ds.sel(time=slice(args.start_time, args.end_time))
    ...
    return corrected_ds
```

The wrapped function receives the opened input dataset(s) positionally, then
one namespace holding every argument under its dest, and returns the output;
the decorator does everything else, including skipping the call entirely on a
cache hit. `skills/weather-skill-authoring/SKILL.md` is the authoring guide.

## Install

```
uv add "weather-skills-core @ git+https://github.com/rhiza-research/weather-skills-core"
```

In a PEP 723 single-file script, add the same
`weather-skills-core @ git+https://github.com/rhiza-research/weather-skills-core`
entry to the inline `dependencies` list.

## Development

```
uv sync
uv run pytest
uv run ruff format --check .
uv run ruff check .
uv run pre-commit run --all-files
```
