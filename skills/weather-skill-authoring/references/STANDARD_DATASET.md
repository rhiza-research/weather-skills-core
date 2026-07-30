# WeatherSkills standard dataset

CF-compliant Zarr that skills read and write. Types: `gridded`, `forecast`, `station` (`any` skips shape checks).

## Shapes

**Gridded** — CF latitude/longitude + CF time (or forecast `step` + scalar `time` init).

**Forecast** — `step` dim + scalar `time` init coord.

**Station** — `station_id` + 1-D `latitude`/`longitude` on that dim + `time`.

Dims are identified via cf-xarray CF attrs only.

## Provenance and source

| Attr | Meaning |
|---|---|
| `weather_skills_history` | JSON array of `{skill, version, args, input}` |
| `source` | Optional CF global attribute naming the data origin (set by fetchers) |

Write with `consolidated=True`. Clear `.encoding` before `to_zarr`. Missing data = NaN.
