# CLI conventions

Same meaning, same flag. Agents chain skills by copying a printed
`--bbox` or `--start-time` onto the next command — that only works if
every skill spells those flags the same way.

This file is the naming contract. Dims and Zarr shapes live in
[STANDARD_DATASET.md](STANDARD_DATASET.md). Units live in
[UNITS.md](UNITS.md).

## Three kinds of flags

| Kind | Who owns it | Examples |
| --- | --- | --- |
| Decorator-owned | `@weather_skill` adds it. Do not declare it. | `-o` / `--output` |
| Canonical special | You declare it with the exact flags below. The decorator parses the string. | `--bbox`, `--date`, `--start-time`, `--end-time`, `--variable` / `-v` |
| Everything else | Ordinary `argparse`. Reuse an existing name when the meaning matches. | `--input` / `-i`, `--geojson`, `--title` |

The linter treats the specials as a closed set (WSK101). Input and
output *paths* are free-form: `-i/--input` is the usual single-Zarr
name, but `--forecast` vs `--obs` is better when the roles differ.

## Canonical specials

Declare these with `@weather_skill.argument(...)`. The decorator adds
help, parses the CLI string, and injects a Python value. **Do not
re-parse** in the skill body — no `bbox.split("/")`, no
`date.fromisoformat` on these.

| Flag | CLI string | What the function receives |
| --- | --- | --- |
| `--bbox` | `N/W/S/E` decimal degrees | `(N, W, S, E)` floats |
| `--date` | `YYYY-MM-DD` | `datetime.date` |
| `--start-time` | `YYYY-MM-DD` (inclusive) | `datetime.date` |
| `--end-time` | `YYYY-MM-DD` (inclusive) | `datetime.date` |
| `--variable` / `-v` | catalog field name | string, or a list if `action="append"` |

When both `--start-time` and `--end-time` are set, the decorator checks
`start_time <= end_time`. Mutual exclusion of `--date` vs the range
flags is skill-owned when you need it.

## Place and time are other skills

`--bbox` and the date flags take **absolute** values only. Named places
and relative dates are not decorator flags.

```text
"Kenya" / "East Africa" / "Mt Kenya"
        →  resolve-region  →  printed N/W/S/E  →  --bbox 5/34/-5/42

"last two weeks" / "latest" / "now-3d"
        →  resolve-time    →  printed --date or --start-time/--end-time
```

- **resolve-region** turns an ISO3 code, a Natural Earth multi-country
  region (`East Africa`), or a `country-admin…` key into a bbox (and
  optional GeoJSON). Anything else falls through to OSM Nominatim
  (`limit=1`) for landmarks.
- **resolve-time** is calendar math against UTC today (or `--as-of`).
  It does not probe a product or clip to what is on disk.

Consumer skills take `--bbox`. Polygon clipping is skill-specific
(`--geojson` / `--mask-geojson`).

## Inputs and outputs

| Concept | Usual flag | Notes |
| --- | --- | --- |
| Zarr input | `-i` / `--input` | `type=Dataset(...)`. Arrives as `ds` (a list if `action="append"`). Repeat `-i` once per Zarr; do not use `nargs="+"` for that pattern. Name the flag by role when inputs differ (`--forecast`, `--obs`). |
| Opaque file | whatever fits | GeoJSON, PNG, … — `type=Path`, not `Dataset`. |
| Output path | `-o` / `--output`, repeatable | Owned by the decorator (`output=True` default). Count must match returned artifacts. |

## Fetchers

Two extra conventions apply only to skills that pull from a catalog.

### `--probe-latest`

Every fetcher implements this. It prints one line on
stdout — `YYYY-MM-DD` or `none` (no realtime cap, e.g. CMIP6) — and
exits. No `-o`. Do not GET full fields: HEAD, a directory listing, a
time coordinate, or a tiny catalog query.

Optional IDENT selects a product:

```text
--probe-latest
--probe-latest final
--probe-latest noaa-gfs-forecast
```

Agents call the fetcher directly. To end a rolling window on that day,
pass the printed date as resolve-time `--as-of`. Latest-available is
not a YAML field and not a decorator date flag.

### `metadata.variables`

On the fetcher's SKILL.md, list the exact `-v` tokens, **most-used
first**. Names are catalog-specific (`tp` on ECMWF S2S,
`total_precipitation` on ARCO, `precipitation_surface` on dynamical).
Open catalogs and closed ones with many fields list the usual first
choices, not every field the source can serve.

```yaml
metadata:
  catalog-group: fetchers
  variables:
    - tp
    - t2m
```

## Shared extras (not canonical)

Reuse these names when the meaning matches. They are ordinary argparse
flags — the decorator does not parse them specially, and the linter
does not treat them as canonical specials.

| Flag | Meaning |
| --- | --- |
| `--geojson` / `--mask-geojson` | Boundary polygon (skill-specific) |
| `--calendar` | CF calendar name |
| `--align-on` | `date` or `year` |
| `--workers` | Parallelism |
| `--title` | Plot title |
| `--probe-latest` | Fetchers only (see above) |

## How the linter checks this

`weather-skills-core lint` is advisory; findings do not block a run.

| Rule | What it catches |
| --- | --- |
| WSK101 | A non-canonical spelling of a special (`-b` for bbox). Use `--bbox`. |
| WSK201 | The same one-off flag on several skills (survey; off by default). Promote it or rename it. |
| WSK202 | The same flag name with a different shape across skills (error). |
