# Standard dataset

A weather-skills Zarr is a [CF-compliant](https://cfconventions.org/)
store with a small, shared dimension vocabulary. Skill A can feed skill
B because both agree what `forecast` and `lat` mean.

Declare what an input must have with `type=Dataset(...)`. The decorator
opens the path, checks those dimensions, attaches pint units, and
injects the opened `xarray.Dataset` as `ds`.

Units policy: [UNITS.md](UNITS.md). Flag names: [CONVENTIONS.md](CONVENTIONS.md).

## Dims are the vocabulary, types are nicknames

A **dimension** is one axis (`lat`, `time`, `member`). A **type** is a
named bundle of dimensions. These two declarations are the same check:

```python
Dataset("forecast")
Dataset("lat, lon, init_time, prediction_timedelta")
```

Prefer the type when the cube is one of the usual shapes. Spell dims
when you need a custom AND (`lat` + `member` but no time).

## Dimensions

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
| `x`, `y` | Irregular / projected grid axes |

Declare the **ontology** name. Incoming files often use a CF or GRIB
synonym; those still count (see [Names on disk](#names-on-disk)).

## Types

| Type (and aliases) | Required dimensions |
| --- | --- |
| `spatial`, `space` | `lat` + `lon` |
| `observations`, `obs`, `analysis`, `retrieval`, `field`, `data` | `lat` + `lon` + `time` |
| `forecast` | `lat` + `lon` + `init_time` + `prediction_timedelta` |
| `vertical_forecast` | forecast + `vertical` |
| `ensemble_forecast` | forecast + `member` |
| `point_obs`, `station` | `point_id` + `time` |
| `any` | any Zarr (skip the dim check) |

`any` still opens a Zarr. Opaque files and figures use `pathlib.Path`,
not `Dataset`.

## Names on disk

| Ontology | Also accepted |
| --- | --- |
| `lat` | `latitude` |
| `lon` | `longitude` |
| `prediction_timedelta` | `step`, `lead_time` |
| `member` | `number`, `realization` |
| `vertical` | `level`, `pressure`, `height`, `altitude`, `lev`, `isobaricInhPa` |
| `point_id` | `station_id` |
| `day_of_year` | `doy` |

Declare the ontology name, not every synonym. A file whose dim is
called `step` still satisfies `Dataset("forecast")`.

## Declaring `Dataset(...)`

| Form | Meaning | Example |
| --- | --- | --- |
| String type or dim | That type, or that one dim | `Dataset("spatial")`, `Dataset("time")` |
| Comma string | All of these (AND) | `Dataset("lat, lon")` |
| Tuple | All of these (AND) | `Dataset(("lat", "lon", "member"))` |
| List | Any one of these (OR) | `Dataset(["forecast", "ensemble_forecast"])` |
| `"any"` | Skip the dim check | `Dataset("any")` |

```python
from weather_skills_core import Dataset, weather_skill


@weather_skill(name="clip-region", version=_SKILL_VERSION)
@weather_skill.argument("-i", "--input", type=Dataset("spatial"), required=True)
def clip_region(ds, output, **kwargs):
    return ds
```

Several Zarrs: `action="append"` and repeat `-i` once per store.
`--input` still arrives as `ds`, now a list. `nargs="+"` is not the
same — a second `-i` replaces the first. Separate Dataset flags when
the roles differ (`--forecast` vs `--obs`).

The decorator owns `-o/--output`. There is no output dim check — the
returned cube is whatever the skill produced. Return count must match
the number of `--output` paths.

## Provenance

Every writing skill appends one step to `weather_skills_history`.

| Attr | Who sets it | Meaning |
| --- | --- | --- |
| `weather_skills_source` | fetchers (optional) | Where the data came from, e.g. `chirps` |
| `weather_skills_history` | every writing skill | JSON list of `{skill, version, args, input}` |

`input` is `{basename, hash}` of the upstream Zarr. Path write targets
and Dataset path strings are omitted from `args`. Plots store the same
JSON in file metadata. When the chain is intact, PNG/JPEG figures also
get a corner `weather-skills provenance verified` mark; HTML figures
get metadata only.

## Writing Zarr

- CF-compliant, `consolidated=True`
- Missing values are NaN
- Clear `.encoding` before `to_zarr`

The decorator stamps history and writes when you return an
`xr.Dataset`. Fetchers should also stamp native cell geometry
(`data_interval` or CF `{dim}_bounds`) — see [UNITS.md](UNITS.md).
