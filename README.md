# weather-skills-core

**Note.** This repository is intended for weather-skills developers. If you
are a weather-skills user, call the skills directly in your favorite AI
interface — start at [weather-skills.org](https://weather-skills.org).

Weather skills turn scientific weather-data pipelines into fixed, reviewable
tools that are meant for agentic AI. You write each step in ordinary Python:
fetch a forecast or satellite rainfall field, clip it to a country, convert
units, make a map. Wrapping the function with `@weather_skill` gives it a
command line, shared flags such as `--bbox` and `--start-time`, and a common
Zarr layout so one step's output is the next step's input. Each output file
carries provenance: a record of which skill, version, and arguments produced
it, so someone else can rerun or check the chain. The skill is the script you
checked in; nothing is generated at runtime unless you want it to be.

![Composable weather skills: fetchers write a standard Zarr, transforms take Zarr to Zarr, and a figure skill writes a PNG](docs/composable-skills.svg)

```bash
# Fetch an ensemble forecast over Kenya
uv run ecmwf-fetch.py --bbox 5/34/-5/42 --date 2026-01-15 -o forecast.zarr

# Clip precip observations and compare against the forecast
uv run clip-region.py -i imerg.zarr -o kenya.zarr --bbox 5/34/-5/42
uv run plot-compare.py -i kenya.zarr -i forecast.zarr -o compare.png
```

## Writing a skill

Stack `@weather_skill.argument(...)` decorators to declare the CLI; they take
the same kwargs as `argparse.add_argument`. The decorator parses argv, opens
Zarr inputs, calls your function, and writes or stamps outputs.

```python
from weather_skills_core import Dataset, weather_skill


@weather_skill(name="clip-region", version="0.1.0")
@weather_skill.argument("-i", "--input", type=Dataset("spatial"), required=True)
@weather_skill.argument("--bbox", required=True)
def clip_region(ds, output, bbox, **kwargs):
    north, west, south, east = bbox  # already (N, W, S, E) floats
    clipped = ds.sel(
        lat=slice(south, north),
        lon=slice(west, east),
    )
    return clipped  # decorator stamps provenance and writes -o/--output
```

`ds` is the opened Zarr and `bbox` is already parsed. Every skill gets
`-o/--output` automatically — do not declare it. The user passes
`-o kenya.zarr`; the function receives that destination as `output`.
Returning `clipped` is enough: the decorator writes the Zarr there and
stamps provenance.

### Function parameters

The decorator always calls your function as `fn(**params)` — keyword arguments
only. Each CLI flag becomes one of those names:

| Where it comes from | Parameter name |
| --- | --- |
| `--start-time` | `start_time` (hyphens become underscores) |
| `--input` | `ds` (opened Dataset; a list if you passed several) |
| `-o/--output` | `output` (always added; do not declare this flag) |

The skill function must accept `**kwargs`. The decorator may pass keys you
did not list as named parameters, and without `**kwargs` the skill refuses to
load. Bind the values you use as named parameters and let `**kwargs` absorb
the rest.

### Zarr inputs

`type=Dataset(...)` means the CLI takes a path string, and your function
receives an opened `xarray.Dataset` as `ds` that already passed the dimension
check and has pint units attached. Use it for weather-skills Zarr stores.

Opaque files (GeoJSON, a PNG you read) use `type=Path`. Those stay
paths; the decorator does not open them.

### Outputs

The decorator always adds `-o/--output`. Do not declare that flag. It is
required (unless you pass `output=False`) and repeatable. A single
`-o kenya.zarr` arrives as `output: Path`; repeating the flag
(`-o a.zarr -o b.zarr`) arrives as `output: list[Path]`.

That path is where the user wants the result. For a Dataset return, leave
the writing to the decorator: return the xarray object and it will
`to_zarr` to `--output` after stamping provenance. For a figure, write the
PNG to `output` yourself and return that same Path so provenance can be
stamped on the file you created.

What you return decides how the path is filled. The number of returned
artifacts must match the number of `--output` paths.

| Return | What the decorator does |
| --- | --- |
| `xr.Dataset` | stamp provenance and `to_zarr` to `--output` |
| `Path` | stamp that file (it must be the `--output` path; typical for a PNG you already wrote) |
| a sequence | one write per `--output` |
| `None` | skip write — the skill already wrote the file |

For skills that only print to stdout and write no file, pass `output=False`:
`@weather_skill(..., output=False)`.

The next example adds a custom flag alongside the standard ones:

```python
@weather_skill(name="my-fancy-skill", version="0.1.0")
@weather_skill.argument("-i", "--input", type=Dataset("any"), required=True)
@weather_skill.argument("--bbox")
@weather_skill.argument("--start-time", required=True)
@weather_skill.argument("--end-time", required=True)
@weather_skill.argument("--corr-coefficient", type=int)
def my_fancy_skill(ds, output, bbox, start_time, end_time, corr_coefficient, **kwargs):
    ...
    return result_ds
```

`--corr-coefficient` arrives as `corr_coefficient`, the same way `--bbox`
arrives as `bbox`. Standard flags get extra parsing (see below); custom flags
use ordinary argparse `type=` / `action=`.

## Why dimensions are standardized

Weather data comes from many sources with different shapes and names. Skills
share one contract so a forecast fetch, an observation clip, and a model
comparison can plug together without custom glue.

We follow [CF conventions](https://cfconventions.org/) for coordinates and
metadata, and expose a small set of **standard dimensions**. `Dataset(...)`
declares what an input must have, in one of two ways: a dimension name
(`lat`, `time`, `member`, …), or a **type** — a named bundle of those
dimensions. `forecast` means the Zarr must have `lat`, `lon`, `init_time`,
and `prediction_timedelta`; `spatial` means `lat` and `lon`. The decorator
checks that before your function runs.

Incoming datasets often use other names for the same axes (`step` for
`prediction_timedelta`, `number` for `member`). Those count. Declare the
ontology name or a type, not every possible name.

## Dataset inputs

| Form | Meaning | Example |
| --- | --- | --- |
| String type or dim | That type or dim | `Dataset("spatial")` |
| Comma string | All of these (AND) | `Dataset("lat, lon")` |
| Tuple | All of these (AND) | `Dataset(("lat", "lon", "member"))` |
| List | Any one of these (OR) | `Dataset(["forecast", "ensemble_forecast"])` |
| `"any"` | Any Zarr (no dim check) | `Dataset("any")` |

Pass several Zarrs by repeating the flag with `action="append"`
(`-i a.zarr -i b.zarr`). `--input` still arrives as `ds`, now a list.
Give each input its own flag when the roles differ (`--forecast` vs `--obs`).

```python
@weather_skill(name="concat", version="0.1.0")
@weather_skill.argument("-i", "--input", type=Dataset("any"), action="append", required=True)
def concat(ds, output, **kwargs):
    # uv run concat.py -i a.zarr -i b.zarr -o stacked.zarr
    return xr.concat(ds, dim="member")


@weather_skill(name="compare", version="0.1.0")
@weather_skill.argument("--forecast", type=Dataset("forecast"), required=True)
@weather_skill.argument("--obs", type=Dataset("observations"), required=True)
def compare(forecast, obs, output, **kwargs):
    # uv run compare.py --forecast fc.zarr --obs imerg.zarr -o diff.zarr
    return forecast - obs
```

### Dimensions

| Name | Meaning |
| --- | --- |
| `lat` | Latitude (regular grid) |
| `lon` | Longitude (regular grid) |
| `time` | Valid time |
| `init_time` | Forecast initialization time |
| `prediction_timedelta` | Forecast lead time |
| `member` | Ensemble member |
| `vertical` | Vertical level (pressure, height, …) |
| `day_of_year` | Day of year |
| `point_id` | Station or point id |
| `x`, `y` | Coordinates for irregular gridded data (e.g. projected meshes) |

### Types

A type is a shortcut for a required set of dimensions.
`Dataset("forecast")` is the same check as
`Dataset("lat, lon, init_time, prediction_timedelta")`. Some types have extra
names (`space` for `spatial`, `station` for `point_obs`); those are listed in
the first column.

| Type | Required dimensions |
| --- | --- |
| `spatial`, `space` | `lat` + `lon` |
| `observations`, `obs`, `analysis`, `retrieval`, `field`, `data` | `lat` + `lon` + `time` |
| `forecast` | `lat` + `lon` + `init_time` + `prediction_timedelta` |
| `vertical_forecast` | forecast dims + `vertical` |
| `ensemble_forecast` | forecast dims + `member` |
| `point_obs`, `station` | `point_id` + `time` |
| `any` | any Zarr (no dimension check) |

See
[`docs/weather-skill-authoring/references/STANDARD_DATASET.md`](docs/weather-skill-authoring/references/STANDARD_DATASET.md)
for the full contract.

## Standard flags

A few shared names get extra help text, parsing, and checks so every skill
that uses them behaves the same way. Declare them with the canonical flags
below; do **not** re-parse in the skill body (no `bbox.split("/")`, no
`date.fromisoformat` on these).

`--bbox` arrives as `(north, west, south, east)` floats. Named places are not
a decorator flag. First call resolve-region to print the `N/W/S/E` bbox, then
pass that to `--bbox`. `--start-time` / `--end-time` arrive as
`datetime.date`, with a check that start is not after end. Relative dates
(`latest`, `now-3d`) are not parsed here. First call resolve-time to print
the `--start-time`/`--end-time` or `--date` flags (calendar math against
UTC today / `--as-of`), then pass those through. Fetcher skills also
declare `--probe-latest` (latest available `YYYY-MM-DD` or `none` on
stdout; no `-o`) — call the fetcher directly for that; do not expect
resolve-time to probe.

| Parameter | Flag | What you get |
| --- | --- | --- |
| `bbox` | `--bbox` | `(N, W, S, E)` floats |
| `date` | `--date` | `datetime.date` (`YYYY-MM-DD`) |
| `start_time` | `--start-time` | Range start as `datetime.date` |
| `end_time` | `--end-time` | Range end as `datetime.date` |
| `variable` | `--variable` / `-v` | Variable name(s); `action="append"` for several |

## Units

Units live on each data variable as a CF `units` string. The decorator
**quantifies** with pint when it opens a Zarr and **dequantifies** before it
writes, so the skill body sees pint quantities and the file stores a plain
string. Those strings must be **pintable** — parseable by pint / UDUNITS
(`mm day-1`, `degree_Celsius`, `kg m-2 s-1`). Known kinds (temp, precip)
must carry pintable units; other variables may include units optionally.

A few kinds also have a **standard unit** that skills convert to for
display and comparison:

| Kind | Standard units |
| --- | --- |
| temp | `degree_Celsius` |
| precip (rate) | `mm day-1` |
| precip (amount) | `mm` |

For accumulated variables such as precipitation, fetch writes a **rate**
(`mm day-1`), not a period total. A **total** (`mm`) is the rate multiplied
by a stamped `aggregation_period` (`convert-to-totals` / `rate_to_total`).
Most skills open rates and totals alike. Those totals helpers refuse
inputs that are already amounts, so multiplying by the period cannot
double-count.

See
[`docs/weather-skill-authoring/references/UNITS.md`](docs/weather-skill-authoring/references/UNITS.md)
for the full units contract.

## Install

```
uv add "weather-skills-core @ git+https://github.com/rhiza-research/weather-skills-core"
```

## Development

This repository is the core library (`weather_skills_core`) and the tools
for writing skills. The skills themselves — fetchers, clips, plots — live
in companion collections such as
[weather-skills](https://github.com/rhiza-research/weather-skills).

To work on this library:

```
uv sync
uv run pytest
uv run ruff format --check .
uv run ruff check .
uv run pre-commit run --all-files
```

Country polygons and Natural Earth region labels (continent, UN subregion,
World Bank region, …) live in
`src/weather_skills_core/data/countries.geojson`. `resolve-region` groups
those features at runtime, so names like `East Africa` need no sidecar.
A few briefing boxes that are not Natural Earth labels (e.g.
`Kenya OND region`) are listed in `region.py` as custom rectangles.
Rebuild the file from upstream Natural Earth 110m admin-0
(`uv run python tools/build_countries.py --help` for the contract):

```
uv run python tools/build_countries.py
```

To write a new skill, follow
[`docs/weather-skill-authoring/SKILL.md`](docs/weather-skill-authoring/SKILL.md)
and check the script with `weather-skills-core lint`. Pull requests that
improve the decorator, the standard dataset contract, or the docs are
welcome.

The longer-term aim is an open registry of shared weather skills on the
same dataset and provenance model, with a public place to discover and
publish them.
