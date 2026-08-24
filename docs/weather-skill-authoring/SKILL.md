---
name: weather-skill-authoring
description: Playbook for writing a weather skill on the @weather_skill decorator. Covers the declaration surface, dim-ontology IO, stacked argument decorators, provenance, and script layout.
---

# weather-skill-authoring

A skill is `skills/<name>/` with a **SKILL.md** and `scripts/<name>.py`
(hyphens → underscores). Do not name the script after a stdlib module
(`inspect.py`, `json.py`): `uv run` puts that directory on `sys.path` and
shadows the real module.

The `@weather_skill` decorator owns the CLI, Dataset input opening,
standard-dataset validation, provenance, and output writing. The script body
is domain logic only.

## References

One question per file:

- `references/CONVENTIONS.md` — what should this flag be called?
- `references/STANDARD_DATASET.md` — what dims does this Zarr need?
- `references/UNITS.md` — rate or amount, and which attrs to stamp?

## Declaration

```python
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "weather-skills-core @ git+https://github.com/rhiza-research/weather-skills-core",
# ]
# ///
from weather_skills_core import Dataset, weather_skill

_SKILL_VERSION = "0.1.0"


@weather_skill(
    name="my-skill",
    version=_SKILL_VERSION,
)
@weather_skill.argument("-i", "--input", type=Dataset("spatial"), required=True)
@weather_skill.argument("--bbox", required=True)
@weather_skill.argument("--start-time", required=True)
@weather_skill.argument("--end-time", required=True)
@weather_skill.argument("--variable", "-v", action="append")
@weather_skill.argument("--smoothing", type=int, help="Window width.")
def my_skill(ds, output, bbox, start_time, end_time, variable, smoothing, **kwargs):
    """Shown as the CLI description."""
    return ds  # or Path(...)


if __name__ == "__main__":
    my_skill()
```

`@weather_skill.argument(...)` mirrors
`argparse.ArgumentParser.add_argument`. Stack one decorator per flag. The skill
function **must** accept `**kwargs`. **Every** declared flag is injected as a
keyword argument (named parameter or via kwargs) — Dataset inputs, Path outputs,
and custom flags alike.

## Dataset inputs

Use `type=Dataset(...)` for Zarr inputs. The decorator opens the path, checks
required dims, quantifies units, and injects the opened dataset as `ds` (a
list if you used `action="append"`). Grammar:

| Form | Meaning |
| --- | --- |
| `Dataset("forecast")` | named type → required dims |
| `Dataset("lat, lon")` | AND of ontology dims |
| `Dataset(("lat", "lon"))` | same AND as a tuple |
| `Dataset(["spatial", "point_obs"])` | OR of alternatives |
| `Dataset("any")` | any Zarr; skip dim checks |

Opaque files (GeoJSON, …) use `type=Path`, not `Dataset`. Flag names are
free-form (`-i/--input`, `--forecast`, …). Multi-input: `action="append"`
(repeat `-i` once per Zarr) or separate Dataset args.

## Outputs

The decorator owns ``-o/--output`` (repeatable). Do not declare it yourself.
It injects ``output`` as a ``Path`` (one path) or ``list[Path]`` (several).

- Return an ``xr.Dataset`` → decorator stamps provenance and ``to_zarr(output)``.
- Return a ``Path`` (plots) → decorator stamps that file (must match an ``--output``).
- Return a sequence → one write per ``--output``; counts must match.
- Return ``None`` → skill already wrote; decorator skips write.
- Inspect-only skills: ``@weather_skill(..., output=False)``.

There is no output dim check; output shape is whatever the skill returns.

## Standard arguments

Shared by many weather skills. Declare them with the canonical flags below;
the decorator adds help, parses CLI strings, and injects kwargs. The skill body
must **not** re-parse those values (no `bbox.split("/")`, no
`date.fromisoformat` on `bbox` / `date` / `start_time` / `end_time`). Format for
APIs with `.isoformat()` if needed, or apply a spatial subset in the skill
itself. Other `@weather_skill.argument` flags still arrive as kwargs; they just
use ordinary argparse typing (`type=`, `action=`, …) without this extra
conversion.

| Argument | Flag | What you get |
| --- | --- | --- |
| `bbox` | `--bbox` | Bounding box `(N, W, S, E)` floats |
| `date` | `--date` | `datetime.date` |
| `start_time` | `--start-time` | Range start as `datetime.date` |
| `end_time` | `--end-time` | Range end as `datetime.date` |
| `variable` | `--variable` / `-v` | Variable name(s) |

When both `start_time` and `end_time` are set, start must be ≤ end. Named
places are not a decorator flag: compose with the resolve-region skill and
pass the printed `N/W/S/E` as `--bbox`. `--geojson` / `--mask-geojson` stay
skill-specific.

Absolute dates are `datetime.date`. Relative tokens (`latest`, `now-3d`,
"the last two weeks") are not parsed by the decorator — compose with the
resolve-time skill and pass the printed `--start-time`/`--end-time` or
`--date`. For the latest day a fetcher has published, run that skill with
`--probe-latest`.

## Skill shapes

| Kind | Typical args | Return |
| --- | --- | --- |
| Fetcher | decorator `-o` | Dataset |
| Transform | Dataset input(s) + decorator `-o` | Dataset |
| Figure | Dataset input(s) + decorator `-o` | Path (write PNG yourself) |
| Inspect | Dataset or Path input; `output=False` | anything (stdout) |

## Units

Most skills accept precip **rates** (`mm day-1`) and **amounts** (`mm`). The
exception is `convert-to-totals` / `rate_to_total`, which multiply a rate by
`aggregation_period` and therefore refuse amounts. See `references/UNITS.md`.

## Provenance

The decorator appends a `weather_skills_history` entry (skill name, version,
args, input basename+hash). Path write targets and Dataset path strings are
omitted from the args blob. PNG/JPEG figures with an intact chain get a corner
mark.

## Layout

Keep the script as domain logic. Put version in `_SKILL_VERSION`. Declare
`weather-skills-core` in the PEP 723 block. Document every flag in SKILL.md.
