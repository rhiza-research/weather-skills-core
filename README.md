# weather-skills-core

Core library for weather skills. It provides the `@weather_skill` decorator,
which owns everything around a skill's domain logic: CLI construction from a
declaration, input reading, envelope validation, the relative-date grammar,
provenance stamping (`weather_skills_history`), the cache-hit short-circuit,
and output writing (Zarr, streaming Zarr appends, or PNG figures).

A skill declares its surface and keeps only its domain logic:

```python
@weather_skill(
    "my-fancy-skill",
    "0.1.0",
    input_type="forecast, station",
    output_type="forecast",
    start_time=True,
    end_time=True,
    extra_args={"corr_coefficient": int, "interpolation_factor": {int, range(0, 2)}},
)
def my_fancy_skill(
    forecast_ds, station_ds, start_time, end_time, corr_coefficient, interpolation_factor
): ...
```

The wrapped function receives the input dataset(s) and resolved arguments and
returns the output; the decorator does everything else, including skipping the
call entirely on a cache hit.

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
