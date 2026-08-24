# Units

Skills combine, convert, and plot without guessing from a variable
name. Every data variable that is a **known kind** (temperature,
precipitation) carries a pint-parseable CF `units` attr. Other
variables may include units optionally.

On disk the attr is a string (`mm day-1`). In the skill body it is a
pint quantity: the decorator **quantifies** on open and **dequantifies**
before write. Figure labels use the variable `long_name` (then
`GRIB_name`, then the variable name) plus a short unit spelling
(`mm/day`, `°C`) via `variable_label_for_display` /
`format_units_for_display` — they do not change the on-disk string.
Implementation: `weather_skills_core.units`.

Dims and types: [STANDARD_DATASET.md](STANDARD_DATASET.md).

## Standard kinds

| Kind | Standard units | Notes |
| --- | --- | --- |
| temp | `degree_Celsius` | Existing CF `standard_name` is kept |
| precip (rate) | `mm day-1` | `lwe_precipitation_rate` |
| precip (amount) | `mm` | Rate × period via totals utilities |

Mass precip flux (`kg m-2 s-1`) converts to a depth rate using liquid
water density (1000 kg m⁻³). Custom pint durations: `pentad` (5 day)
and `dekad` (10 day); week and month come from pint itself.

`to_standard_units` converts a recognized temp/precip variable to the
table above and stamps that kind's `standard_name` when set. It leaves
the **variable name** unchanged. Value conversion is skill-owned — the
decorator does not call it for you.

## Precipitation: rate vs amount

Fetch writes **rates** (`mm day-1`), not period totals. An **amount**
(`mm`) is a rate multiplied by a stamped `aggregation_period`.

```text
fetch                 →  rate + native spacing (data_interval or bounds)
aggregate-temporal    →  rate + aggregation_period + coverage + cell_methods
convert-to-totals     →  amount (mm)     ← last step before a plot
```

Most skills open rates and amounts alike. The exception is
`convert-to-totals` / `rate_to_total`: multiplying an amount by the
period would double-count, so those refuse precip totals (amount units,
or `cell_methods` with `sum`).

## Native spacing vs aggregation

Two different clocks. Do not mix them up.

**Native cell geometry** is what the source actually sampled. Fetch
stamps exactly one of:

- `data_interval` — uniform spacing as a pint string (`30 minute`,
  `1 day`)
- `{dim}_bounds` — CF `(N, 2)` start/end per sample, when spacing is
  irregular

Never both. Deaccumulate uses the same pair.

**Aggregation** is a later window. Only `aggregate-temporal` stamps:

| Attr | Where | Meaning |
| --- | --- | --- |
| `aggregation_period` | data variable (pint string) | Length of each aggregated interval (`7 day`, `21 day`) |
| `aggregation_coverage` | time/step coordinate, 0–1 | Fraction of native samples present in that interval |
| `cell_methods` | data variable | The operation (`time: mean`, `time: sum`) |

`data_interval` is not the aggregation window. Convert-to-totals
multiplies rates by `aggregation_period`, never by `data_interval` or
bound widths. CF bounds are irregular native geometry, not a per-sample
period.

A weekly-mean rate therefore carries `time: mean`,
`aggregation_period = "7 day"`, and the original `data_interval` when
the native axis was uniform.

CLI labels map to `aggregation_period`: `daily` → `1 day`, `weekly` →
`7 day`, `dekadal` → `1 dekad`, `monthly` → `1 month`. Custom pint
durations (`21 day`) are also valid `--period` values.

## What the decorator does

1. After opening a Zarr, `quantify_dataset` attaches pint units from
   each data variable's `units` attr (when present).
2. Known kinds with `units_required` (today: temp, precip) must have
   parseable units. Other variables may omit them.
3. Before writing, `dequantify_dataset` strips pint so stored attrs
   stay plain strings. The write path also normalizes GRIB unit
   strings, stamps precip-amount CF names when units are amounts
   (including leftover rate-like `long_name` / `GRIB_name`), casts
   `step` to `timedelta64[ns]`, and copies data-var attrs the skill
   stripped from the first input (same variable names).

## `convert-to-totals`

Terminal step: rate × `aggregation_period` → amount (`mm`). It also
rewrites leftover rate display names containing `rate` or `flux` to
`Total precipitation`. It requires:

- a stamped `aggregation_period` (run `aggregate-temporal` first —
  native-only cubes will not convert)
- coverage at or above `--min-coverage` (default 1.0). Aggregation
  keeps incomplete bins and only stamps coverage; this flag drops them
- non-overlapping intervals (sample spacing ≥ `aggregation_period`)
- rate inputs — precip totals are refused

Overlapping series (rolling `--window`, or 21-day bins labelled 10 days
apart) are refused. Run `select` on `time` or `step` first. A singleton
axis (one aggregated bin) skips the overlap check.

## How a variable is classified

`classify_variable` picks a kind in this order:

1. CF `standard_name`
2. Variable-name hints (`t2m`, `tp`, `precip`, …). For precip hints,
   amount vs rate comes from `units` when that fingerprint is clear
   (`kg m-2` / `mm` → amount, `kg m-2 s-1` / `mm day-1` → rate). If
   units are present but not convertible to a precip rate or amount,
   the name hint is ignored (`precipitation_quality_index_surface`
   with `units="1"` is not precip).

Units alone do **not** classify a variable. A bare `kg m-2 s-1` field
is not treated as precip.

## Helpers

| Function | Role |
| --- | --- |
| `units_equal` | Spelling-independent equality (`mm/day` ≈ `mm day-1`) |
| `format_units_for_display` | Short figure unit spelling (`mm/day`, `°C`) |
| `variable_label_for_display` | Figure label: `long_name` → `GRIB_name` → name, plus units |
| `convert_dataarray` / `convert_values` | Explicit unit ↔ unit |
| `to_standard_units` | Temp / precip → standard display units |
| `stamp_data_interval` | Uniform `data_interval` or CF `{dim}_bounds` on fetch / deaccumulate |
| `precip_amounts_to_rates` | Amount → `mm day-1` (deaccumulate amount vars on `step`, else ÷ interval) |
| `stamp_precip_amounts` | Amount units → amount CF `standard_name`; rewrite rate display names |
| `rate_to_total` | Rate × period → amount (refuses precip totals) |
| `parse_aggregation_period` | Parse an `aggregation_period` / duration string |
| `filter_min_coverage` | Drop aggregated intervals below a coverage threshold |

## Author checklist

- Temp and precip need a udunits-parseable `units` attr.
- Fetch writes accumulated variables as rates (`mm day-1`).
- After fetch: `data_interval` **or** CF `{dim}_bounds`, and no
  `aggregation_period`.
- After `aggregate-temporal`: `aggregation_period` +
  `aggregation_coverage` + `cell_methods`.
- Convert to amounts with `convert-to-totals` only when you need
  totals for display — and never feed those totals back into
  rate-math skills.
