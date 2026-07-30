---
name: weather-skill-authoring
description: How to write a weather skill on @weather_skill. Use when creating or reviewing a skill.
---

# weather-skill-authoring

Read first: [STANDARD_DATASET.md](references/STANDARD_DATASET.md), [CONVENTIONS.md](references/CONVENTIONS.md).

## Skeleton

```python
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "weather-skills-core @ git+https://github.com/rhiza-research/weather-skills-core",
# ]
# ///
from weather_skills_core import Types, weather_skill

_SKILL_VERSION = "0.1.0"

@weather_skill(
    name="my-skill",
    version=_SKILL_VERSION,
    inputs=[Types.GRIDDED],   # omit for fetchers
    outputs=[Types.GRIDDED],  # omit for no-artifact
    required_args=("start_time", "end_time"),
    optional_args=("bbox",),
    required_env=(),          # e.g. ("API_KEY",)
    check_cache=True,         # CLI --check-cache/--no-check-cache
)
@weather_skill.argument("--workers", type=int, default=1)
def my_skill(ds, start_time, end_time, bbox, workers):
    """CLI description."""
    return ds

if __name__ == "__main__":
    my_skill()
```

## Declaration

| Kwarg | Meaning |
|---|---|
| `inputs` / `outputs` | Lists of `Types.*` (or unions as tuples). Variadic inputs use argparse-style `+`/`*`: `inputs=[Types.ANY + "+"]` (≥1), `inputs=[Types.ANY + "*"]` (≥0), or `inputs=[Types.ANY + "+2"]` (≥2); skill gets one list as the first positional |
| `required_args` / `optional_args` | From catalog: `time`, `start_time`, `end_time`, `bbox`, `variable` |
| `required_env` | Env vars checked before run |
| `exclude_args` | Dests omitted from cache key |
| `check_cache` | Default for CLI cache flag (not a skill kwarg) |

Custom flags: stack `@weather_skill.argument(...)` (argparse `add_argument` API).
Fetchers set the CF global `source` attr on the returned Dataset themselves.

## Variables

Omit `--variable` → act on the **whole dataset**. Narrow with the `select` skill first when only one variable matters. Do not auto-pick a “main” variable.

## Dates

`--time` / `--start` / `--end` accept `YYYY-MM-DD` or `latest`. Skills resolve `latest` themselves. No offset grammar.

## Returns

- `xarray.Dataset` → zarr write
- object with `savefig` → PNG
- `str`/`Path` → already written
- optional `EntryOverride(args={...})` to rewrite provenance

## Helpers

`validate_type(ds, Types.FORECAST)` or `validate_type(ds1, ds2)` — assert shape compliance (or match another dataset).

## Errors

`UsageError` → exit 2. `DataError` / `SkillError` → exit 1.
