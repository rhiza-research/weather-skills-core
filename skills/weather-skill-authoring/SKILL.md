---
name: weather-skill-authoring
description: The playbook for writing a weather skill on the weather_skills_core @weather_skill decorator. Covers the envelope contract, the declaration surface for all five skill classes (transform, fetcher, streaming fetcher, plot, no-artifact), the date grammar, provenance and caching, units rules, error handling, credentials, versioning, and the PEP 723 script layout. Use when creating a new skill, converting an existing one onto the decorator, or reviewing a skill for conformance.
---

# weather-skill-authoring

How to write a weather skill. A skill is a directory `skills/<name>/` holding a
**SKILL.md manifest** and a single-file **`scripts/<name>.py`** script whose CLI,
input reading, envelope validation, provenance, caching, and output writing are
owned by the `@weather_skill` decorator from `weather_skills_core`. The script
body holds only domain logic.

## Read these first

- `references/ENVELOPE.md` — the artifact contract: envelope shapes, the
  `weather_skills_history` schema, CF compliance, write rules. This is the
  authoritative copy.
- `references/CONVENTIONS.md` — canonical CLI flag names and the
  relative-or-absolute date grammar. A flag that does the same thing on
  different skills has the same name; match the table. This is the
  authoritative copy.
- `CONTRIBUTING.md` in the forecasting-skills repo — that repo's own
  publish/version-bump/CI/branch-protection flow. Relevant only when
  publishing a skill into forecasting-skills; not part of the core authoring
  contract.

## The five skill classes

| Class | Declaration shape | Function returns |
| --- | --- | --- |
| Transform | `input_type` + zarr `output_type` (or `"same"`) | a Dataset |
| Fetcher | no `input_type`, zarr `output_type`, `source=` | a Dataset |
| Streaming fetcher | fetcher + `streaming=True` | a generator of per-period Datasets |
| Plot | `input_type` + `output_type="png"` | a matplotlib Figure |
| No-artifact | no `output_type` | anything (ignored) |

## The envelope contract

Every zarr input and output is a weather-skills envelope: a CF-compliant Zarr
store plus the `weather_skills_history` provenance attr, in one of three
shapes — `gridded`, `forecast`, `station`. Declare each input's shape in
`input_type` (use `any` to opt out of shape validation); the decorator
validates on open and exits 2 with a message naming the offending dim.

`references/ENVELOPE.md` is authoritative for the rest: the exact dims and
coords of each shape, the CF-compliance requirement, the write rules
(`consolidated=True`, NaN for missing data, per-variable `encoding` cleared on
write and not part of the contract), and the `weather_skills_history` schema.

## Declaring a skill

The script is a PEP 723 single file. Skeleton:

```python
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "weather-skills-core @ git+https://github.com/rhiza-research/weather-skills-core",
# ]
# ///
"""Module docstring: what the script is. Not read by the decorator."""

from weather_skills_core import weather_skill

# Auto-populated by the version-bump CI workflow. Do not edit manually.
_SKILL_VERSION = "0.1.0"


@weather_skill("my-skill", _SKILL_VERSION, ...)
def my_skill(ds, ...):
    """Docstring shown as the CLI description."""
    ...


if __name__ == "__main__":
    my_skill()
```

The **function docstring** is the `--help` description: the decorator builds
the parser with `description=fn.__doc__` and never reads the module
docstring. When a skill's `--help` description is its full module docstring
(the standalone-script pattern of `description=__doc__`), that full text must
live in the function docstring — a shortened function docstring shortens
`--help`.

Declaration surface (all keyword-only after `name`, `version`):

- `input_type` — `None`, one type, or a comma string / list with one type per
  input. Inputs arrive as `--input`/`-i` (repeated for several), or via
  `input_names=["forecast", "mclimate"]` for dedicated flags, or
  `variadic_input=True` for two-or-more `--input` repeats (the function then
  receives one list of datasets).
- `input_help` — help text for the input flag(s). With `input_names`, a list
  giving one help string per named flag
  (`input_help=["Forecast ensemble Zarr.", "M-climate ensemble Zarr."]`);
  otherwise a single string shown on `--input`/`-i`, replacing the
  decorator's default help in the repeated-input cases.
- `output_type` — `None`, a zarr envelope type, a tuple/set of zarr envelope
  types (a union), `"same"`, or `"png"`. `"same"` declares a shape-preserving
  transform: the output is whatever envelope type the input carries. Use it
  (instead of hard-coding one zarr type) when `input_type` admits several
  shapes (`"gridded|forecast"`, `"any"`) and the skill preserves whichever
  came in. It requires at least one declared zarr input and writes through
  the zarr path exactly like an explicit zarr type. A union (e.g.
  `("gridded", "forecast")`, for a fetcher whose source decides the shape)
  VALIDATES rather than selects: the returned dataset's detected shape must
  be one of the members, checked before the write (a mismatch exits 1); the
  declaration never coerces the output toward any member. A single-type
  declaration stays unchecked.
- Standard flags, enabled by toggles and passed as keyword arguments:
  `start_time`/`end_time` (`--start`/`--end`), `date` (`--date`), `bbox`
  (`"required"` or `"optional"`; the function receives a parsed
  `(N, W, S, E)` tuple), `variable` (`"single"` or `"repeat"`), `workers`
  (pass the default int), `title`, `dims`, `time_dim`.
- Toggle dict form: `start_time`/`end_time`/`date`, `bbox`, `variable`, and
  `workers` also accept a dict overriding the flag's argparse surface —
  `help` replaces the decorator-owned help text, `required` overrides
  requiredness (`--start`/`--end`/`--date` default to required; with
  `"required": False` an omitted value reaches the function as `None` and no
  resolved date is recorded), and `choices` constrains the accepted values.
  The string/int forms become the dict's `mode`/`default` key —
  `bbox={"mode": "optional", ...}`, `variable={"mode": "repeat", ...}`,
  `workers={"default": 4, ...}` — and `date` additionally accepts `context`:
  the parenthetical label on the resolved-date stderr line (default
  `"single date"`; e.g. `date={"context": "single forecast init date"}`
  logs `resolved "latest" -> 2026-07-14 (single forecast init date)`).
  Prefer a dict-form standard toggle over redeclaring the same flag under
  `extra_args`: the toggle keeps the date grammar, the bbox parse/argv
  rewrite, and the resolved-provenance behavior.
- `extra_args` — dest name to a bare type (`int`; `bool` makes a store-true
  flag), a tuple of literal string choices (`("mean", "std")`), a constraint
  set combining a type with a value domain (`{int, range(0, 2)}` derives
  `choices`; the set must name the element type), or an argparse-keyword
  dict (supports `positional`, `flag`, `aliases`, `repeat`, and any argparse
  keyword such as `help`). A dest may not reuse a name the decorator
  resolves and passes itself (`start_time`/`end_time`/`date`/`bbox`/
  `input_paths`/`context`).
- `mutex_groups` — named groups of mutually exclusive `extra_args` (see
  below).
- `input_paths=True` — the function also receives an `input_paths` keyword
  argument: the CLI-given input path(s) as a list of `pathlib.Path`, in
  input order. Use it for diagnostics and messages that name the inputs; the
  datasets still arrive positionally, and the paths never enter the recorded
  provenance args. This is the supported way to learn an input's path — do
  not fish it out of `ds.encoding`.
- Hooks and cache behavior: `latest_resolver`, `source`, `streaming`,
  `cache`, `hash_input`, `completeness_probe`, `validate_args`,
  `normalize_args`, `exclude_args`, `reference_args`, `history_labels`,
  `write_encoding`, `post_write`, `append_dim`, `savefig_kwargs`,
  `cache_hit_label`.
- `post_write` — `callable(path)` run after the artifact is written (zarr,
  streaming, or PNG; requires an artifact `output_type`), receiving the
  output path. Use it for read-back verification of the written store (a
  calendar-coercion check, a CF decode check on the bytes on disk). Raise a
  `SkillError` (`DataError`/`UsageError`) to fail the run with the usual
  exit codes; the hook runs before the `Wrote:` line, so a failed run never
  claims success, and a cache hit skips it (nothing was written).

### Mutually exclusive groups

`mutex_groups` maps a group name to a sequence of `extra_args` dests (an
optional group) or to `{"args": (...), "required": True}`. The decorator
builds a real argparse mutually exclusive group per entry, so usage renders
the `(--a | --b)` bracketing and argparse enforces at-most-one (exactly-one
when required):

```python
@weather_skill(
    "downscale", _SKILL_VERSION,
    input_type="gridded", output_type="gridded",
    extra_args={
        "factor": {"type": float, "aliases": ["-f"]},
        "target_resolution": {"type": float},
        "reference_grid": {},
    },
    mutex_groups={
        "target": {"args": ("factor", "target_resolution", "reference_grid"),
                   "required": True},
    },
)
```

Members must be non-positional `extra_args` entries that do not set their own
`required` (requiredness belongs to the group); a dest may belong to at most
one group, and a group needs at least two members. Declare groups here —
never assemble them by reaching into `wrapper.parser._actions` after
decoration.

### The run context

Hooks and the wrapped function often need to share run-scoped values: a
lazily opened remote store used by both the `latest` resolver and the body, a
requested-variable list the completeness probe checks, a fetch-discovered
calendar the post-write hook verifies. The run context is the supported
channel for all of it — never module-level globals (a module-scope `_STATE`
dict leaks state across calls in the same process and hides the data flow).

Opt in by naming a `context` parameter on any hook (`latest_resolver`,
`validate_args`, `normalize_args`, `completeness_probe`, `write_encoding`,
`post_write`) or on the function itself; the decorator then also passes
`context=`, a `RunContext` carrying:

- `args` — the parsed argparse namespace;
- `input_paths` / `output_path` — the CLI paths as `pathlib.Path`;
- `start_time` / `end_time` / `date` — the resolved absolute dates (`None`
  before resolution or when the toggle is off);
- `state` — a mutable dict reserved for the skill, empty at the start of
  every run and shared across the hooks and the function within that run.

The opt-in is the literal parameter name: a `**kwargs` catch-all does not opt
in, and callables without the parameter keep their plain call shapes. An
`extra_args` dest may not be named `context` when the function opts in.

```python
def _open_remote(context):
    """Open the remote store at most once per run, memoized in the context."""
    if "ds" not in context.state:
        import xarray as xr

        context.state["ds"] = xr.open_zarr(_STORE_URL, chunks=None)
    return context.state["ds"]


def _latest(args, context):
    """Newest date with available data, from the opened remote store."""
    ds = _open_remote(context)
    ...
    return newest_date  # a datetime.date


def _remember_request(args, context):
    """Stash the requested variable pre-cache-check for the completeness probe."""
    context.state["req_variable"] = args.variable


def _store_is_complete(out, context):
    """Corner-read probe: True when the requested variable reads back."""
    import xarray as xr

    variable = context.state["req_variable"]
    ...


@weather_skill(
    "my-fetch",
    _SKILL_VERSION,
    output_type="gridded",
    source="my-source",
    start_time=True,
    end_time=True,
    variable="single",
    latest_resolver=_latest,
    validate_args=_remember_request,
    completeness_probe=_store_is_complete,
)
def fetch(start_time, end_time, variable, context):
    """Fetch and write a weather-skills envelope Zarr."""
    ds = _open_remote(context)
    ...
```

The function receives the opened input dataset(s) positionally, then the
resolved parameters as keyword arguments. Raise
`weather_skills_core.UsageError` for usage/validation failures (exit 2) and
`weather_skills_core.DataError` for data-availability or hard failures
(exit 1). Never call `sys.exit` from the body.

Defer heavy imports (`xarray`, `numpy`, plotting, client libraries) into the
function body so `--help` and cache hits stay cheap; `weather_skills_core`
itself defers them.

### Worked example: transform

```python
@weather_skill(
    "clip-region",
    _SKILL_VERSION,
    input_type="gridded",
    output_type="gridded",
    bbox="required",
    dims=True,
    hash_input=False,  # cheap cache check; hash computed only on a miss
    cache_hit_label="clip",  # cache-hit line reads "skipping clip."
)
def clip_region(ds, bbox, dims):
    """Spatially subset a gridded weather-skills envelope Zarr."""
    from weather_skills_core.envelope import bbox_subset, detect_spatial_dims

    lat_dim, lon_dim = detect_spatial_dims(ds, dims)
    return bbox_subset(ds, bbox, lat_dim=lat_dim, lon_dim=lon_dim)
```

A typed `input_type="gridded"` composes with `dims=True`: when the caller
passes `--dims LAT,LON`, input validation checks that the overridden names
exist on the dataset instead of running CF/heuristic detection, so an input
with nonstandard dim names validates and reaches the body (the same holds
for `--time-dim`). Overrides participate only in typed validation; an input
declared `any` skips all shape checks.

The decorator writes the returned Dataset: it carries the first input's attrs
forward, stamps the provenance chain, clears encodings, replaces whatever
occupied the output path, and removes a partial store when the write fails,
so a truncated store is never mistaken for a complete cache. Do not open or
write zarr yourself.

### Worked example: fetcher with a `latest` resolver

```python
def _latest(args):
    """Newest date with available data. One bounded discovery call."""
    import xarray as xr

    ...
    return newest_date  # a datetime.date


def _store_is_complete(out):
    """Corner-read probe: True when a candidate cache hit actually reads back."""
    import xarray as xr

    ...


@weather_skill(
    "oisst-fetch",
    _SKILL_VERSION,
    output_type="gridded",
    source="oisst",
    start_time=True,
    end_time=True,
    bbox="optional",
    latest_resolver=_latest,
    completeness_probe=_store_is_complete,
)
def fetch(start_time, end_time, bbox):
    """Fetch daily SST and write a weather-skills envelope Zarr."""
    import xarray as xr

    ...
    return ds
```

`start_time`/`end_time` arrive as resolved `datetime.date` objects. The
resolver runs lazily and at most once, only when a token references `latest`;
an all-absolute invocation performs zero network before the cache check.

### Worked example: streaming fetcher

```python
from weather_skills_core import EntryOverride


def _set_write_encoding(ds):
    """Controlled write encodings, applied after the decorator's encoding clear."""
    import numpy as np

    ds["time"].encoding.update(units="days since 1970-01-01 00:00:00", calendar="standard")
    ds["sst"].encoding["_FillValue"] = np.float32("nan")


@weather_skill(
    "oisst-fetch",
    _SKILL_VERSION,
    output_type="gridded",
    source="oisst",
    start_time=True,
    end_time=True,
    bbox="optional",
    streaming=True,
    append_dim="time",
    write_encoding=_set_write_encoding,
)
def fetch(start_time, end_time, bbox):
    """Fetch daily SST, one period per yield, bounded memory."""
    days = plan_days(start_time, end_time)
    if days and days[-1] != end_time:
        # Trailing days not yet published: record the effective window.
        yield EntryOverride({"end": days[-1].isoformat()})
    for day in days:
        yield fetch_one_day(day, bbox)
```

Yield one Dataset per period. The decorator writes the first with
`mode="w"` and appends the rest along `append_dim`, re-stamping provenance on
every append, and removes a partial store on any mid-stream failure. Yield an
`EntryOverride` at any point to rewrite the recorded args; the persisted
chain reflects every override, including one yielded after the final dataset
(the decorator re-stamps the written store).

### Worked example: plot

```python
@weather_skill(
    "plot-compare",
    _SKILL_VERSION,
    input_type=["any", "any"],
    output_type="png",
    history_labels=["a", "b"],
    title=True,
    savefig_kwargs={"bbox_inches": "tight"},
)
def plot_compare(ds_a, ds_b, title):
    """Render two inputs as stacked heatmap rows."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2)
    ...
    return fig
```

Return the Figure; the decorator saves it with each input branch's full
history embedded in the PNG metadata (`weather_skills_history` for a single
input; `weather_skills_history_<label>` per declared label otherwise, plus a
`Software` key). Plot skills have no cache: they always render.

### Worked example: no-artifact

```python
@weather_skill(
    "resolve-region",
    _SKILL_VERSION,
    extra_args={"code": {"positional": True, "metavar": "CODE"}, "geojson": str},
)
def resolve_region(code, geojson):
    """Resolve an ISO 3166-1 alpha-3 country code to an N/W/S/E bbox."""
    print("12.0/33.9/-4.7/41.9")  # stdout is load-bearing: callers consume it
```

No provenance, no cache, no output flag — the decorator provides the CLI and
the version epilog. Keep stdout limited to the machine-consumed result; all
diagnostics go to stderr.

## The date grammar, from the author's side

You never parse date tokens. Declare `start_time`/`end_time` (or `date`) and
the decorator applies the full relative-or-absolute grammar defined in
`references/CONVENTIONS.md` (absolute dates, `now`/`today`/`latest`, `-N{d,w}`
offsets, inclusive endpoints, the duration idiom). Malformed tokens, future
offsets, and reversed ranges exit 2 before any network call; relative
resolutions print a stderr line with the resolved dates. Your only obligation
is the `latest_resolver` callable for sources that support `latest` — one
bounded discovery call returning a `datetime.date`.

## Provenance and caching

The decorator computes the provenance entry — skill, version, the recorded
args, and the input reference(s) — **before** your function runs; on a cache
hit it returns without calling you or touching the store. What you control:

- The recorded args are the argparse namespace minus input/output path
  strings, with resolved absolute dates (never relative tokens) and
  `--workers` excluded. Use `normalize_args` to canonicalize (sort a repeated
  `--variable`, coerce types) so flag order cannot cause spurious misses, and
  `exclude_args` for any other pure-concurrency or presentation knob.
- `cache=False` removes the cache check entirely: the function runs and the
  output is rewritten on every invocation, with the provenance entry still
  built and stamped. Declare it when a meaningful cache key does not exist
  or the recompute is cheaper than the check; it is valid only on zarr
  output types (PNG and no-artifact skills have no cache to disable).
- `hash_input=False` defers the input content hash until after a cheap cache
  check (the stamped entry still carries the hash). Keep the default when a
  modified same-named input must force a recompute.
- `reference_args` names arg dests holding secondary reference stores
  (a reference grid, a distribution reference); their content hashes enter
  the cache key as `reference_inputs`.
- `completeness_probe` guards cache hits — fetcher and transform alike —
  against a truncated prior store: the probe receives the output store's
  path and is invoked only after the entry matches, so make it a cheap
  corner-element read of the output, not a metadata check. Build it with
  `weather_skills_core.provenance.make_completeness_probe` (see Library
  helpers) rather than hand-writing the open, corner-read, and
  except-everything-to-False steps.
- `validate_args` runs before the cache check — an invalid argument must
  never report a cache hit.
- `EntryOverride` and the cache: the entry is the cache key and is computed
  BEFORE the function runs, so a store stamped with overridden args never
  matches the pre-override entry a rerun builds — every rerun recomputes. An
  entry carrying fetch-discovered values (a resolved grid label, a catalog
  data version) must therefore have those values resolved BEFORE the cache
  check for reruns to hit: resolve them in `validate_args` and write the
  resolved value back onto the namespace (the pattern the relative-date
  grammar itself uses), so the normal entry includes them on both the first
  run and the rerun. Reserve `EntryOverride` for values that only exist
  after the work (an effective end discovered mid-fetch), accepting that
  reruns miss.

Everything that shapes the on-disk chain — entry fields, chain append on the
first input's trunk, per-branch histories for multi-input entries, the
`weather_skills_source` stamp, PNG metadata keys — the decorator does for you;
`references/ENVELOPE.md` is authoritative for the `weather_skills_history`
schema those pieces produce. One author-facing consequence:
`weather_skills_history` is the only provenance attr the decorator reads, so
an input without it (whatever other attrs it carries) is opaque, and the chain
starts fresh at your skill's entry.

### Raw-string parsers and the schema validator

A skill that reads `weather_skills_history` values itself (a provenance
inspector reading zarr attrs or PNG tEXt keys) uses the functions exported by
`weather_skills_core.provenance` instead of reimplementing them:

- `parse_chain(raw)` — strict: returns the chain list, or raises
  `ValueError` with the message `"value is not valid JSON"` or
  `"value is not a JSON array"` (schema checkers such as
  `provenance --check` record the raised message as a violation).
- `coerce_chain(raw, label)` — lenient: returns the chain list, or `None`
  for a value that is not a JSON array, after a one-line stderr warning
  naming `label` (the artifact basename or key being read) and pointing at
  `provenance --check`. A valid array passes through unchanged, even when
  its entries are imperfect.
- `validate_chain(chain, loc)` — validates a parsed chain against the entry
  schema and returns `(violations, notes)`, both lists of location-prefixed
  strings rooted at `loc`. Violations cover a non-array chain, non-object
  entries, and missing or mistyped required entry keys
  (`skill`/`version`/`args`/`input`), recursing into a multi-input entry's
  nested per-branch `history`; unknown/extra keys land in `notes` and do not
  fail validation.

## Library helpers for skill bodies

Beyond the decorator, `weather_skills_core` exports helpers for the
mechanisms skills otherwise copy from each other. The division of labor is
fixed: the per-skill constants — variable names, units strings, attr dicts,
error-message wording — stay in the skill; the helper provides the mechanism,
parameterized by them. Do not reimplement any of these in a skill body.

### CF stamping and validation (`weather_skills_core.envelope`)

- `stamp_cf_attrs(ds)` — non-destructive gridded stamping: sets CF
  `standard_name`/`units`/`axis` on the first latitude-named coord
  (`latitude`/`lat`/`y`), the first longitude-named coord
  (`longitude`/`lon`/`x`), and a `time` coord (`standard_name`/`axis` only),
  every attr via `setdefault` so source-provided values win. Returns `ds`.
  For fetchers whose source coords may already carry correct CF metadata.
  Precondition: the alias matching treats `y`/`x` as latitude/longitude
  names, so it assumes geographic coordinates — a dataset with projected
  `x`/`y` coords (meters, not degrees) must be renamed to its real
  geographic coords first, or must not go through this helper at all.
- `stamp_cf_coords(ds, *, long_names=None)` — the overwriting counterpart:
  `update`s the same attrs onto the canonical `latitude`/`longitude`/`time`
  names (post-rename), replacing prior values; coords absent from the dataset
  are skipped. `long_names` optionally maps a coord name to a `long_name`
  applied with `setdefault`. For fetchers that assert coordinate metadata.
  Global attrs and data-variable attrs stay in the skill — they are
  per-source constants.
- `stamp_cf_dsg(ds, var_attrs, *, station_id_long_name, name_long_name)` —
  station timeSeries DSG stamping: fixed coordinate attrs
  (lat/lon/time + `cf_role="timeseries_id"` on `station_id`, the two
  long-name parameters naming the station identifier and the optional `name`
  coord), then per data variable the load-bearing
  `coordinates="latitude longitude time"` attr followed by that variable's
  entry in `var_attrs` (a mapping of data-variable name to its attr dict:
  `units`, `long_name`, `cell_methods`, any `standard_name`). Build and
  udunits-validate the `var_attrs` values in the skill; a data variable
  missing from the mapping raises `KeyError`.
- `verify_cf_dsg(ds)` — pre-write DSG check: raises `DataError` listing every
  problem when cf-xarray does not resolve `station_id` under the
  `timeseries_id` role or any of the latitude/longitude/time coordinates.
- `cf_axes_missing(ds, axes=("X", "Y", "T"))` — returns the axis letters
  cf-xarray cannot resolve from the CF attrs, each axis probed independently
  and a resolution failure counted as missing rather than raised. Use it for
  write-side checks (on the dataset about to be written) and post-write
  checks (on the reopened store); the failure message is yours.
- `udunits_error(units, *, catch=(ValueError,))` — parses `units` with
  `cf_units.Unit` and returns the parse failure (or None when it parses);
  `catch` is the exception tuple converted to a returned value. Raise your
  own `DataError` from the result — the message wording is a per-skill
  constant. `cf_units.Unit(None)`/`Unit("")` return an "unknown" unit rather
  than raising, so a missing/blank-units guard also belongs to the caller.

### Envelope geometry and selection (`weather_skills_core.envelope`)

- `cf_dim(obj, cf_name)` — the coord name cf-xarray resolves for a CF key on
  a Dataset or DataArray, or None; a bare lookup with no heuristic fallback
  and no error.
- `auto_variable(ds)` — the no-flag variable auto-pick: first data var that
  is not a CF grid-mapping (CRS) container nor named by another var's
  `grid_mapping` attr, preferring one with >= 2 dims; None when no candidate
  remains.
- `lat_slice(lat_vals, north, south)` — a `.sel` slice that follows the
  latitude axis's own ascending or descending order.
- `polygon_from_geojson(path, flag="--mask-geojson")` — the unioned shapely
  polygon from a GeoJSON file (FeatureCollection, Feature, or bare geometry);
  raises `UsageError` naming `flag` when the file is missing, unreadable, or
  has no usable geometry. `shapely` stays in the skill's inline deps.
- `normalize_longitude(ds, lon_dim="longitude")` — maps a 0..360 longitude
  axis onto [-180, 180) and sorts ascending, so N/W/S/E bboxes with negative
  west/east select correctly.

### Dates (`weather_skills_core.dates`)

- `np_to_date(value)` — numpy datetime64 to `datetime.date`, truncating any
  time-of-day.
- `today_utc(args=None)` — the current UTC date, shaped as a
  `latest_resolver`: pass `latest_resolver=today_utc` for sources with no
  cheap day-precise discovery, where `latest` means today and a thin
  not-yet-published tail is a normal partial window.

### Completeness probes (`weather_skills_core.provenance`)

- `make_completeness_probe(variables=None, *, check_time=None)` — builds the
  `completeness_probe=` callable: opens the candidate store consolidated and
  corner-reads one element of each probed variable, returning False on any
  store-read failure — an unreadable store, an unknown name, an empty
  dimension, an undecodable chunk. `variables` is None to probe every data
  variable present (an empty store is incomplete), a name or list of required
  names, or a callable receiving the run context (e.g.
  `lambda context: context.args.variable`) resolved at probe time; an
  exception the callable raises propagates — a skill bug, not a cache miss.
  `check_time` names a coordinate that must be present, non-NaT, and strictly
  increasing, and moves the corner read to the LAST index along it — the
  slice an interrupted append would be missing. `check_time` REQUIRES a
  datetime64 dimension coordinate: a store whose named coordinate is a
  scalar/auxiliary coordinate or holds cftime/object/numeric values makes
  the probe raise `ValueError` (a misconfigured probe must fail loudly, not
  read as a permanent miss that recomputes a complete store on every run).
  For such stores — a non-standard model calendar decoded to cftime, a
  forecast envelope's scalar init `time` — probe by variables alone or write
  a bespoke probe. Write a bespoke probe only when a store needs checks this
  cannot express.

### Runtime checks (`weather_skills_core.util`)

- `is_transient(exc)` — True when an error's text carries an HTTP 429/5xx
  status or a timeout/connection marker; the retry policy (attempt count,
  backoff) stays in the skill.
- `require_env(*names, message=None)` — returns the named environment
  variables' values in order, raising `UsageError` (exit 2) when any is unset
  or empty — with the default message naming only the missing variables, or
  with `message` verbatim. Hand the values straight to the auth library;
  never print or echo them.

## Units

Units are the single most error-prone surface. For any skill that produces or
relabels data variables:

- **Pass the source's units through verbatim by default.**
- **Remap only** when the source value is a valid unit spelled in a form
  udunits will not accept — relabel to the conformant spelling of the *same*
  unit. Never remap a unit that already parses.
- **Never convert numeric values** to land in a different unit. The one
  principled exception is a documented integer storage encoding with no unit
  of its own (e.g. "tenths of a mm"); declare it as a value conversion.
- Validate every output data-variable unit with a real udunits check —
  `weather_skills_core.envelope.udunits_error` wraps the `cf_units.Unit`
  parse and hands you the failure to word your own error around. A missing
  or empty unit is invalid — drop the variable with a note or fail, never
  write `units=None` (blank values do not fail the parse, so guard them
  explicitly).
- `standard_name` must match the unit family; verify the exact string against
  the current CF standard-name table before stamping it, and omit it when no
  verified entry cleanly applies (that is CF-valid).

Unit *conversion* is its own skill (`unit-convert`); do not fold conversions
into fetchers or transforms.

## The source-to-output transform declaration

In a fetcher, declare every divergence between the raw source and the written
output in one labeled comment block near the top of the script: every unit
remap (with the same-unit-made-to-comply reason), variable rename, value
conversion, and standard_name/long_name assignment. Pass-through is the
unstated default; a reader must be able to reconstruct the entire
source-to-output delta from the block alone.

## Errors: reactive, never proactive

The user decides what to fetch or compute. Never refuse a request because it
looks big: no pre-flight size estimates, no cell-count thresholds, no
"large/slow" warnings. (A *required* `--bbox` for a source whose global query
is genuinely unbounded is a missing-argument error, not a size guard.)

Handle real failures reactively with one-line, actionable messages that tell
the calling agent what to change, classified where the remedies differ:

- provider-rejected-oversized — "reduce `--bbox` / shorten the window;
  retrying unchanged will not help";
- availability (outside the served range, not yet published) — distinct from
  transport;
- transport (network/timeout) — distinct from availability;
- auth — see Credentials.

Raise `UsageError`/`DataError` with the message; never let a known failure
mode reach the user as a raw traceback.

### Unprefixed failures

The decorator prints a raised `UsageError`/`DataError` as `Error: <message>`.
Raising with `prefix=False` prints exactly the given message, with no
`Error: ` prefix; the exit code is unchanged (2 for `UsageError`, 1 for
`DataError`):

```python
raise DataError(f"Body too long: {over} characters over the limit.", prefix=False)
```

Two surfaces legitimately need this — and both still raise instead of
calling `sys.exit` (the never-`sys.exit`-from-the-body rule holds):

- exit-code-as-product programs, where the exit code is the skill's result
  and the printed line is a report rather than an error (`provenance
  --check` exits 0/1/2 for valid/absent/invalid);
- machine-consumed retry signals, where a caller parses the stderr text
  verbatim (submit-feedback's over-budget retry contract: stderr starts
  `Body too long: ...`).

Everything else keeps the default prefix.

## Credentials

For a credentialed source: read the credential from the environment with a
presence check and exit with a clear "set `<ENV_VAR>`" message when unset;
hand the value straight to the auth library or an HTTP header; never print,
log, or echo it anywhere, including in error messages. Classify auth failures
(HTTP 401/403, login-library errors) into a one-line actionable message
without echoing the key; a per-item auth failure mid-run is fatal and
surfaced, not silently dropped. Declare the required env var in the SKILL.md
frontmatter metadata so the runner knows it is needed.

## What the decorator does for you

Do not re-implement these in a skill body:

- CLI construction, the `--bbox` negative-north argv rewrite, the
  `skill version:` epilog, exit-code mapping.
- Input open, envelope validation, the input/output overlap guard.
- Date-grammar parsing, `latest` memoization, the resolved-dates stderr line.
- The cache key, the cache-hit short-circuit, cache-completeness probing.
- Provenance: entry construction, chain append, multi-input branch histories,
  PNG metadata.
- Writing: encoding clear (set controlled write encodings via
  `write_encoding`, which runs after the clear), `consolidated=True`,
  streaming first-write/append ordering, partial-store rollback on failure,
  and the `post_write` invocation (after the write, before the `Wrote:`
  line) — verify the written artifact there, never by re-opening the store
  from `__main__`.

## Decorator-owned stderr lines

These lines are printed by the decorator; a skill body never re-prints its
own version of any of them:

- the resolved-dates line for relative date tokens
  (`resolved "now-1w".."now" -> ... (7 days; ...)`);
- `Cache hit: <output> already matches requested params; skipping <label>.`
  — `<label>` defaults to the skill name; set `cache_hit_label` to change
  the word (e.g. `cache_hit_label="clip"`);
- `Wrote: <output> (<detail>)` — the default detail is the output's sizes
  for a standard zarr skill, `<append_dim>=<total>` for a streaming skill,
  and nothing for a PNG skill. To add or replace detail, return
  `weather_skills_core.WroteSummary("...")` alongside the output (a tuple:
  `return ds, WroteSummary("variable 'precip' -> 'rain'")`; combinable with
  an `EntryOverride`), or yield it from a streaming generator. The text is
  appended after the default detail unless `replace=True`;
- the opaque-input warnings (`no upstream weather_skills_history ...`), the
  incomplete-store re-fetch note, the partial-store removal note, and the
  malformed-history note.

Everything else the body wants to say goes to stderr under its own wording
(stdout stays reserved for load-bearing results).

## Versioning

`_SKILL_VERSION` sits at the top of the script and is passed to the decorator
so it lands in the epilog and every provenance entry. CI owns it: the
version-bump workflow updates `_SKILL_VERSION` and the SKILL.md
`metadata.version` in lockstep on merge, and a consistency check fails the PR
when they disagree. Never edit either by hand, and keep the constant's
one-line assignment shape so the bump tooling's regex continues to match.

## Script and lockfile layout

- One file: `skills/<name>/scripts/<name>.py`, runnable with
  `uv run --script`.
- Dependencies go in the PEP 723 inline header, including
  `weather-skills-core @ git+https://github.com/rhiza-research/weather-skills-core`.
  No `uv add`, no shared helper module in the skills repo.
- The core library declares `cftime` (its zarr reads decode model calendars
  such as `360_day`/`noleap`) plus `xarray`/`zarr`/`numpy`/`cf-xarray` — but
  NOT `pandas`, `pint`, or `matplotlib`. The inline header must keep every
  package the script body itself imports; do not drop a dependency on the
  assumption that core carries it.
- A repo-side dependency guard that scans script bodies for `open_zarr`
  calls (a `check_cftime_deps`-style check) cannot see the reads the
  decorator performs on the script's behalf, so it will not flag a missing
  `cftime`; core's own `cftime` dependency is what covers calendar decoding
  on those reads.
- Each script has a sibling `<name>.py.lock`, regenerated with
  `uv lock --script` when the inline dependencies change. Never hand-edit a
  `.py.lock`.

## Where tests live

Skill behavior is tested in the weather-skills-core repo — the grammar,
envelope, provenance, and decorator suites — never in forecasting-skills.
Do not add unit tests, a `tests/` directory, doctests, self-test modes, or
CI test steps to a skills repo; its check surface is ruff, inline-dep
validation, and one `--help` invocation per script. If a change seems to need
a test to be correct, add the test to weather-skills-core (extending the core
if the behavior belongs there) or raise it with the maintainer.

## SKILL.md (the skill's own docs)

- Describe **current behavior** only — no "previously", "used to", or
  "no longer".
- Examples use realistic, bounded selections, with no narration about why the
  example was chosen; state the real cost model once in a performance note
  and let the examples be examples.
- Document the reactive error catalog and, for a credentialed source, the
  missing/wrong-key behavior; keep the runner's required-env metadata block.

## Creation checklist

Before calling a skill done, confirm:

- [ ] The declaration matches CONVENTIONS.md flag names exactly; new concepts
      are added to that file in the same PR.
- [ ] The body holds domain logic only — nothing from "What the decorator
      does for you" is re-implemented.
- [ ] Heavy imports are deferred into the function body; `--help` runs
      without them.
- [ ] Failures raise `UsageError`/`DataError` with one-line actionable
      messages, classified by remedy; no proactive size guard anywhere.
- [ ] Units: verbatim pass-through or a declared same-unit compliance remap,
      udunits-validated; fetchers carry the source-to-output transform block.
- [ ] (Credentialed) no credential value is ever printed or echoed; auth
      failures classified; required env declared in frontmatter metadata.
- [ ] `write_encoding` sets any controlled time units/calendar and
      `_FillValue`; nothing else touches `.encoding`.
- [ ] Run-scoped values shared between hooks and the body go through the run
      context's `state`, never module-level globals.
- [ ] Written-store verification lives in `post_write`, not in `__main__`
      after the decorated call.
- [ ] Cache declaration is deliberate: `cache`, `hash_input`,
      `normalize_args`, `exclude_args`, `reference_args`,
      `completeness_probe` each considered; fetch-discovered entry values are
      resolved before the cache check, with `EntryOverride` reserved for
      values that only exist after the work.
- [ ] `_SKILL_VERSION` untouched by hand; PEP 723 header carries the core git
      dependency; `<name>.py.lock` present.
- [ ] No tests in the skills repo; new behavior is covered in
      weather-skills-core.
- [ ] SKILL.md: current-behavior only, bounded examples, reactive-error
      catalog documented.

## Linting your skill

weather-skills-core ships a conformance linter that checks a skill's
declaration against the ecosystem's conventions. It is not published to PyPI,
so run it from the git repository (or from inside your skill directory with
no path argument):

```bash
uvx --from git+https://github.com/rhiza-research/weather-skills-core weather-skills-core lint skills/my-skill
```

From a checkout of weather-skills-core:

```bash
uv run weather-skills-core lint skills/my-skill
```

The target is layout-auto-detected: a skill directory, a `scripts/`
directory, a `skills/` tree, or a repo root holding one. Declarations are
read from the scripts by AST — the linter never imports or runs a skill —
and the standard flag surface comes from `standard_parameters()`, a
maintained description of the decorator's CLI surface that
`test_standard_parameters.py` verifies against the parser the decorator
actually builds, so drift between the two fails the test suite.

### The corpus model

The cross-skill rules (duplicate and divergent one-off flags) compare your
skill against a corpus:

- linting a whole `skills/` tree uses the tree itself;
- linting one skill inside a `skills/` tree discovers its siblings upward —
  they join the corpus as context, and findings are reported only for your
  skill;
- `--against <path-or-repo>` adds more corpora: a local path or a public
  GitHub repository reference (`org/repo`, `org/repo@rev`, or an
  `https://github.com/...` URL), where `rev` is a branch, tag, or full commit
  SHA, fetched shallowly for its declarations and not retained. Each git
  subprocess is bounded by a 5-minute timeout; expiry is a usage error naming
  the reference rather than an indefinite hang. Repeat the flag to lint
  against several skill sets at once.

With no corpus beyond the target (a standalone skill and no `--against`),
the cross-skill rules report a skipped state — visible in both output
formats — and are excluded from the score; they are never silently scored
as clean.

### Rule catalog

| ID | Severity | Checks | Remediation |
| --- | --- | --- | --- |
| WSK001 | error | The script parses and contains a `@weather_skill` call. | Fix the syntax error or declare the skill through the decorator. |
| WSK101 | warning | No `extra_args` entry shadows a standard flag or dest (`--input`, `--output`, `--start`, `--end`, `--date`, `--bbox`, `--variable`, `--workers`, `--title`, `--dims`, `--time-dim`). | Declare the standard toggle instead; its dict form covers help/required/choices overrides. |
| WSK201 | warning | A one-off flag name is not also declared by another corpus skill. **Advisory: off by default** (CONVENTIONS.md wants a flag doing the same job to share a name, so a shared flag is usually the desired consistency, not a defect; the genuinely-bad case — a shared name with a divergent shape — is WSK202). Opt in with `--extend-select WSK201` to survey shared flags that might be promoted to a core standard parameter. | Rename it, or propose promoting it to a weather-skills-core standard parameter. Findings name each skill and corpus holding the collision. |
| WSK202 | error | Skills sharing a one-off flag name agree on its shape (type, arity, nargs, choices). | Align the declarations or rename the flags. Values that are not literals in the source are recorded as dynamic and skipped from the comparison. |
| WSK301 | warning | SKILL.md and the declaration agree: every declared flag is mentioned in SKILL.md, and every flag the Arguments section documents is declared. A missing SKILL.md is its own finding. | Update the Arguments section or the declaration. |
| WSK401 | error | `_SKILL_VERSION` exists and is passed as the decorator's version argument. | Define the constant at module top and pass it (CI bumps it). |
| WSK402 | error | The PEP 723 block declares weather-skills-core. | Add the core dependency to the inline script block. |

Rule IDs are stable: new rules take new numbers and existing rules are never
renumbered. Severity semantics: `error` breaks an ecosystem contract,
`warning` is a conformance divergence worth fixing, `info` is an advisory
note. Severity is display and `--strict` metadata only — it does not decide
which rules run.

### Rule selection

The **default rule set** is every rule above except WSK201: WSK001, WSK101,
WSK202, WSK301, WSK401, and WSK402 run when you pass no selection flag (this
is what CI invokes). WSK201 is off by default.

Three repeatable flags choose the rules to run, with the same semantics as
ruff's `select`/`extend-select`/`ignore`:

- `--select CODE` **replaces** the default set with exactly the rules you
  name. `--select WSK101` runs only WSK101.
- `--extend-select CODE` **adds** to the set (the default set, or `--select`
  if you also gave it). `--extend-select WSK201` runs the default set plus
  WSK201.
- `--ignore CODE` **removes** rules, applied last. `--ignore WSK202` runs the
  default set without WSK202.

Resolution order is: base (the `--select` rules if any were given, else the
default set), then union `--extend-select`, then subtract `--ignore`.

Each selector is a full rule code (`WSK201`) or a category prefix that matches
every rule in the band: `WSK2` matches WSK201 and WSK202, `WSK` matches all.
A selector that matches no known rule code is a usage error (exit 2) naming
the bad selector. An `--ignore` selector that is valid but not in the active
set is a silent no-op.

### Score and output

Each skill scores 0-100: the mean over the applicable rules, where a rule
with no findings scores 1.0 and a rule with findings scores by its worst
severity (error 0.0, warning 0.5, info 0.8). Every applicable rule weighs
equally in the mean; a rule that did not run — skipped for lack of a corpus,
or left out of the resolved rule set — leaves the denominator entirely and is
never silently scored as clean. An unanalyzable script scores 0. The
aggregate is the mean of the per-skill scores.

`--format json` emits a stable schema (`findings`, `score`, `skipped_rules`,
`notes`) for tooling.

The linter is advisory: it exits 0 whether or not findings exist (2 for
usage errors), and the score informs — maintainers decide. A shadowed or
duplicated flag can be the right call for a particular skill; the linter's
job is to make the divergence visible, not to block it. Callers that want a
gate can opt in with `--strict <severity>`; nothing in the ecosystem depends
on it.

## Updating this playbook

This is a living document. When the skill paradigm shifts — a new declaration
parameter, a refined units case, a different error classification — update
the relevant section here in the same change that establishes it, so the next
skill inherits the lesson. Each rule reads as a current-behavior statement,
not a history of how it changed.
