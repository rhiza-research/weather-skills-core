# weather-skills-core

`@weather_skill` owns CLI, I/O, cache, provenance, and writes. Skills keep domain logic.

```python
from weather_skills_core import Types, weather_skill

@weather_skill(
    name="my-skill",
    version="0.1.0",
    outputs=[Types.GRIDDED],
    required_args=("start_time", "end_time", "variable"),
    optional_args=("bbox",),
    check_cache=True,
)
@weather_skill.argument("--workers", type=int, default=1)
def my_skill(start_time, end_time, variable, bbox, workers):
    ...
```

## Install

```
uv add "weather-skills-core @ git+https://github.com/rhiza-research/weather-skills-core"
```

## Development

```
uv sync
uv run pytest
uv run ruff check .
```
