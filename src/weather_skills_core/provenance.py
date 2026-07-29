"""Provenance chain handling for weather-skill artifacts.

The canonical provenance attr is ``weather_skills_history``: a JSON-encoded
append-only array of entries, ordered oldest first along the pipeline. Each
entry carries ``skill``, ``version``, ``args`` (the argparse namespace minus
input/output path strings, with resolved absolute dates), and ``input``:

- fetchers: ``None`` (no upstream zarr);
- single-input transformers: ``{"basename": ..., "hash": ...}`` where ``hash``
  is a sha256 over the upstream zarr's stored bytes (the hash may be deferred
  until after a cheap cache check);
- multi-input transformers: a list of ``{"basename", "hash", "history"}``
  dicts in input order, where ``history`` is that input's full chain;
- ``reference_inputs``: an optional sibling key on the entry listing
  ``{"basename", "hash"}`` for secondary reference stores (e.g. a reference
  grid) whose content must enter the cache key.
"""

import hashlib
import json
import sys
from pathlib import Path

HISTORY_ATTR = "weather_skills_history"
SOURCE_ATTR = "weather_skills_source"
#: Set on every input dataset the decorator opens, holding the path it was
#: opened from, and stripped by :func:`stamp_zarr` before any write.
INPUT_PATH_ATTR = "weather_skills_input_path"

DEFAULT_SOFTWARE = "forecasting-skills"


def hash_zarr(zarr_path: Path) -> str:
    """Stable content hash of a zarr's stored bytes. Walks the zarr dir
    deterministically and hashes relative-path bytes + each file's
    content. Returns sha256 hex digest."""
    zarr_path = Path(zarr_path)
    h = hashlib.sha256()
    for p in sorted(zarr_path.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(zarr_path)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def parse_chain(raw: str) -> list:
    """Strictly parse a raw ``weather_skills_history`` value into a chain list.

    Raises :class:`ValueError` with the message ``"value is not valid JSON"``
    when the value does not decode, or ``"value is not a JSON array"`` when it
    decodes to anything but an array. Schema checkers record the raised
    message as a violation; lenient render paths use :func:`coerce_chain`.
    """
    try:
        chain = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        raise ValueError("value is not valid JSON") from None
    if not isinstance(chain, list):
        raise ValueError("value is not a JSON array")  # noqa: TRY004 -- ValueError is the observable contract asserted by callers/tests
    return chain


def coerce_chain(raw: str, label: str) -> list | None:
    """Leniently parse a raw ``weather_skills_history`` value for render paths.

    A value that is present but not a JSON array (non-JSON, or a JSON
    object/scalar) is malformed under the ``weather_skills_history`` array
    contract; return ``None`` after a one-line stderr warning naming
    ``label`` (the artifact basename or key being read) and pointing at
    ``provenance --check``, so the caller omits the branch. A valid array
    (including an empty one) passes through unchanged, even when its entries
    are imperfect.
    """
    try:
        chain = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        chain = None
    if not isinstance(chain, list):
        print(
            f"ignoring malformed weather_skills_history on {label}; "
            "run `provenance --check` for details",
            file=sys.stderr,
        )
        return None
    return chain


_ENTRY_KNOWN_KEYS = {"skill", "version", "args", "input"}
_INPUT_ITEM_KNOWN_KEYS = {"basename", "hash", "history"}


def _validate_input(value, loc: str, violations: list, notes: list) -> None:
    """Validate an entry's ``input`` field against the array contract.

    ``input`` is one of: ``null``; a ``{basename, hash}`` dict; or an array of
    ``{basename, hash}`` dicts, each of which may also carry a nested
    ``history`` chain (recursively validated). Appends violations and notes in
    place.
    """
    if value is None:
        return

    def _check_item(item, item_loc: str) -> None:
        if not isinstance(item, dict):
            violations.append(f"{item_loc}: input entry is not an object")
            return
        if "basename" not in item:
            violations.append(f"{item_loc}: missing required key 'basename'")
        elif not isinstance(item["basename"], str):
            violations.append(f"{item_loc}.basename: must be a string")
        if "hash" not in item:
            violations.append(f"{item_loc}: missing required key 'hash'")
        elif not isinstance(item["hash"], str):
            violations.append(f"{item_loc}.hash: must be a string")
        if "history" in item:
            _validate_chain(item["history"], f"{item_loc}.history", violations, notes)
        for key in item:
            if key not in _INPUT_ITEM_KNOWN_KEYS:
                notes.append(f"{item_loc}: unknown key {key!r}")

    if isinstance(value, list):
        for j, item in enumerate(value):
            _check_item(item, f"{loc}[{j}]")
        return
    if isinstance(value, dict):
        _check_item(value, loc)
        return
    violations.append(f"{loc}: must be null, an object, or an array of objects")


def _validate_chain(chain, loc: str, violations: list, notes: list) -> None:
    """Validate one chain (an array of entries) against the schema, in place.

    Records every violation with its location into ``violations``; records
    unknown/extra keys (which do not fail validation) into ``notes``. Recurses
    into a multi-input entry's ``input[*].history``.
    """
    if not isinstance(chain, list):
        violations.append(f"{loc}: value is not a JSON array")
        return
    for i, entry in enumerate(chain):
        eloc = f"{loc}[{i}]"
        if not isinstance(entry, dict):
            violations.append(f"{eloc}: entry is not an object")
            continue
        if "skill" not in entry:
            violations.append(f"{eloc}: missing required key 'skill'")
        elif not isinstance(entry["skill"], str):
            violations.append(f"{eloc}.skill: must be a string")
        elif not entry["skill"]:
            violations.append(f"{eloc}.skill: must be a non-empty string")
        if "version" not in entry:
            violations.append(f"{eloc}: missing required key 'version'")
        elif not isinstance(entry["version"], str):
            violations.append(f"{eloc}.version: must be a string")
        if "args" not in entry:
            violations.append(f"{eloc}: missing required key 'args'")
        elif not isinstance(entry["args"], dict):
            violations.append(f"{eloc}.args: must be an object")
        if "input" not in entry:
            violations.append(f"{eloc}: missing required key 'input'")
        else:
            _validate_input(entry["input"], f"{eloc}.input", violations, notes)
        for key in entry:
            if key not in _ENTRY_KNOWN_KEYS:
                notes.append(f"{eloc}: unknown key {key!r}")


def validate_chain(chain, loc: str) -> tuple[list, list]:
    """Validate a parsed ``weather_skills_history`` chain against the entry schema.

    Returns ``(violations, notes)``, both lists of location-prefixed strings.
    Violations cover a non-array chain, non-object entries, missing or
    mistyped required entry keys (``skill``/``version``/``args``/``input``),
    and a malformed ``input`` value; a multi-input entry's nested per-branch
    ``history`` is validated recursively, its findings located under
    ``<loc>[i].input[j].history``. Unknown/extra keys land in ``notes`` and do
    not fail validation. ``loc`` prefixes every location (typically the attr
    or tEXt key name the chain was read from).
    """
    violations: list = []
    notes: list = []
    _validate_chain(chain, loc, violations, notes)
    return violations, notes


def load_history(zarr_path: Path) -> list:
    """Read an artifact's provenance chain, tolerating absence and malformation.

    Only the ``weather_skills_history`` attr is read; a store carrying no such
    attr has no history. A not-yet-existing or unreadable store is a silent
    miss (empty chain). A present-but-non-array value is malformed under the
    ``weather_skills_history`` contract; it is treated as no history with a
    one-line stderr warning pointing at ``provenance --check``. The coercion
    is array-level only: an array whose individual entries are imperfect is
    passed through unchanged.
    """
    zarr_path = Path(zarr_path)
    try:
        import xarray as xr

        with xr.open_zarr(zarr_path, consolidated=False) as ds:
            raw = ds.attrs.get(HISTORY_ATTR)
    except (OSError, KeyError, ValueError):
        # A not-yet-existing or unreadable output during a cache check is a
        # miss; OSError covers missing stores and filesystem-level failures
        # such as PermissionError alike.
        return []
    if not raw:
        return []
    parsed = coerce_chain(raw, str(zarr_path))
    return [] if parsed is None else parsed


def input_ref(path: Path, *, include_hash: bool = True) -> dict:
    """Build a single-input ``input`` value: ``{basename[, hash]}``.

    With ``include_hash=False`` the (expensive) content hash is omitted so a
    cheap cache pre-check can run first; complete the entry with a hashed ref
    on a miss before stamping.
    """
    path = Path(path)
    ref = {"basename": path.name}
    if include_hash:
        ref["hash"] = hash_zarr(path)
    return ref


def multi_input_ref(paths, histories) -> list:
    """Build a multi-input ``input`` value: per-input ``{basename, hash, history}``.

    ``histories`` holds each input's full chain (``[]`` when the input had
    none), in the same order as ``paths``.
    """
    paths = [Path(p) for p in paths]
    return [
        {"basename": p.name, "hash": hash_zarr(p), "history": h}
        for p, h in zip(paths, histories, strict=True)
    ]


def reference_ref(paths) -> list:
    """Build a ``reference_inputs`` sibling value: per-reference ``{basename, hash}``."""
    return [{"basename": Path(p).name, "hash": hash_zarr(Path(p))} for p in paths]


def build_entry(skill: str, version: str, args: dict, input, reference_inputs=None) -> dict:
    """Assemble a provenance entry.

    ``input`` is ``None`` for a fetcher, an :func:`input_ref` dict for a
    single-input skill, or a :func:`multi_input_ref` list. ``reference_inputs``
    (a :func:`reference_ref` list), when given, is attached as a sibling key.
    """
    entry = {"skill": skill, "version": version, "args": args, "input": input}
    if reference_inputs:
        entry["reference_inputs"] = reference_inputs
    return entry


def _chained_input_match(last_input, entry_input, *, compare_hash: bool) -> bool:
    """Compare the recorded ``input`` of the output's last entry to the candidate's.

    A recorded input of the wrong shape (not the dict or list the entry
    schema prescribes) compares as a mismatch, never a traceback.
    """
    if isinstance(entry_input, list):
        # Multi-input: per-item basename + hash + history, in order.
        if not isinstance(last_input, list) or len(last_input) != len(entry_input):
            return False
        return all(
            isinstance(li, dict)
            and li.get("basename") == ei["basename"]
            and li.get("hash") == ei["hash"]
            and li.get("history") == ei["history"]
            for li, ei in zip(last_input, entry_input, strict=True)
        )
    if last_input is not None and not isinstance(last_input, dict):
        return False
    last_input = last_input or {}
    entry_input = entry_input or {}
    if last_input.get("basename") != entry_input.get("basename"):
        return False
    return not (compare_hash and last_input.get("hash") != entry_input.get("hash"))


def cache_hit(
    out: Path,
    entry: dict,
    upstream: list | None = None,
    *,
    fetcher: bool = False,
    compare_hash: bool = True,
    completeness_probe=None,
) -> bool:
    """Return True when the store at ``out`` was produced by this same entry.

    Two chain positions are supported:

    - ``fetcher=True``: the candidate entry is the chain's FIRST entry
      (``history[0]``); ``skill``/``version``/``args``/``input`` are compared
      wholesale.
    - chained (default): the candidate entry is the chain's LAST entry on top
      of ``upstream`` (the input's chain); the output chain must be exactly
      ``upstream + [entry]``. The recorded input is compared by basename (and
      by content hash unless ``compare_hash=False``, for skills that defer the
      expensive hash until after this check). ``reference_inputs`` is always
      compared, so an in-place change to a secondary reference forces a
      recompute; entries without references compare equal on absence.

    ``completeness_probe``, honored in both positions, is a
    ``callable(Path) -> bool`` invoked on ``out`` only after the entry
    matches; a False result rejects the hit (a partial prior write can leave
    a matching history attr over truncated arrays) and prints a stderr note.
    """
    out = Path(out)
    if not out.exists():
        return False

    def probe_rejects(action):
        if completeness_probe is not None and not completeness_probe(out):
            print(
                f"Note: {out} matches the request but is an incomplete/unreadable "
                f"store (likely a prior interrupted write); {action}.",
                file=sys.stderr,
            )
            return True
        return False

    history = load_history(out)
    if fetcher:
        if not history:
            return False
        existing = history[0]
        # A malformed chain entry (anything but an object) is a miss, never a
        # traceback.
        if not isinstance(existing, dict):
            return False
        matches = (
            existing.get("skill") == entry["skill"]
            and existing.get("version") == entry["version"]
            and existing.get("args") == entry["args"]
            and existing.get("input") == entry["input"]
        )
        if not matches:
            return False
        return not probe_rejects("re-fetching")

    upstream = upstream or []
    if len(history) != len(upstream) + 1:
        return False
    if history[:-1] != upstream:
        return False
    last = history[-1]
    # A malformed tail entry (anything but an object) is a miss, never a
    # traceback.
    if not isinstance(last, dict):
        return False
    matches = (
        last.get("skill") == entry["skill"]
        and last.get("version") == entry["version"]
        and last.get("args") == entry["args"]
        and _chained_input_match(last.get("input"), entry.get("input"), compare_hash=compare_hash)
        and last.get("reference_inputs") == entry.get("reference_inputs")
    )
    if not matches:
        return False
    return not probe_rejects("recomputing")


def _open_for_probe(xr, path):
    """Open a store for the completeness probe, tolerating absent consolidated metadata.

    A store written without consolidated metadata is valid, but
    ``open_zarr(consolidated=True)`` raises rather than falling back; retry the
    unconsolidated open before the caller concludes a miss. A genuinely broken
    or missing store fails both opens and surfaces to the probe's miss handler.
    """
    try:
        return xr.open_zarr(path, consolidated=True)
    except (FileNotFoundError, KeyError, ValueError):
        return xr.open_zarr(path, consolidated=False)


class _UnsupportedTimeCoordError(ValueError):
    """``check_time`` named a coordinate the probe cannot check.

    Raised (as a :class:`ValueError`) by probes built with
    :func:`make_completeness_probe` when the named coordinate is not a
    datetime64 dimension coordinate; the probe re-raises it past its
    store-read exception handling so a misconfigured probe fails loudly
    instead of reading as a permanent cache miss.
    """


def make_completeness_probe(variables=None, *, check_time: str | None = None):
    """Build a completeness probe for the decorator's ``completeness_probe=`` slot.

    The returned probe -- ``callable(path, *, context=None) -> bool`` -- cheaply
    verifies that a candidate cache-hit store is fully written and readable: a
    mid-run failure can leave a partial Zarr whose root attrs (and so its
    ``weather_skills_history``) were written before the array data landed, and
    honoring such a store as a cache hit would hand the caller a broken output.
    The probe opens the store with ``consolidated=True``, falling back to
    ``consolidated=False`` when the store carries no consolidated metadata (a
    valid unconsolidated store must not read as a permanent miss), and
    corner-reads one element of each probed variable, forcing a real chunk
    read (a truncated store can keep intact metadata while a chunk is missing,
    so a metadata-only check is not enough). Any store-read failure -- an
    unreadable store, an unknown probed variable, an empty dimension, an
    undecodable chunk -- returns False (a cache miss that forces a recompute),
    never an exception. Probe misconfiguration is the exception to that: a
    raising ``variables`` callable and an unsupported ``check_time``
    coordinate (below) propagate, because a misconfigured probe that returned
    False would silently recompute a complete store on every run.

    ``variables`` declares which data variables must read back:

    - ``None`` (or any falsy value): every data variable present in the store
      is probed; a store with no data variables is incomplete;
    - a ``str``: that single variable must be present;
    - an iterable of names: all must be present;
    - a callable: invoked with the run context at probe time and returning any
      of the above -- e.g. ``lambda context: context.args.variable`` for a
      probe keyed to the requested ``--variable`` value(s). The callable is
      resolved before the store is opened, and an exception it raises
      propagates -- it is a skill bug, not a cache miss.

    ``check_time``, when given, names a coordinate that must be present,
    non-empty, free of NaT (a half-written append leaves unfilled slots), and
    strictly increasing; each probed variable's corner read then uses the LAST
    index along that dimension (the slice an interrupted append would be
    missing) instead of the first.

    ``check_time`` REQUIRES a datetime64 dimension coordinate: the named
    coordinate must be a dimension coordinate (its only dim is itself) holding
    datetime64 values. The store's actual representation is only knowable at
    probe time, so the factory cannot reject a mismatch up front; instead the
    probe raises :class:`ValueError` when the coordinate is absent from the
    store, or is present but is a scalar/auxiliary coordinate, or holds a
    non-datetime64 representation (cftime/object/numeric values) -- each a
    misconfiguration, distinct from genuine incompleteness. Stores with such
    time coordinates (e.g. a
    non-standard model calendar decoded to cftime, or a forecast envelope's
    scalar init ``time``) cannot use ``check_time``; probe them by variables
    alone or write a bespoke probe.
    """

    def probe(path, *, context=None) -> bool:
        import xarray as xr

        # A callable spec is resolved outside the store-read try block: an
        # exception it raises is a skill bug and must propagate, not read as
        # a cache miss that silently recomputes on every run.
        spec = variables(context) if callable(variables) else variables
        if isinstance(spec, str):
            wanted = {spec}
        elif spec:
            wanted = set(spec)
        else:
            wanted = None  # probe every data variable present in the store
        try:
            with _open_for_probe(xr, path) as ds:
                probed = set(ds.data_vars) if wanted is None else wanted
                if not probed or not probed.issubset(ds.data_vars):
                    return False
                if check_time is not None:
                    import numpy as np

                    if check_time not in ds.coords:
                        raise _UnsupportedTimeCoordError(
                            f"check_time={check_time!r} names a coordinate absent from the "
                            f"store (coords present: {list(ds.coords)}); a missing time "
                            "coordinate is a probe misconfiguration, not incompleteness"
                        )
                    time_var = ds[check_time]
                    if time_var.dims != (check_time,):
                        raise _UnsupportedTimeCoordError(
                            f"check_time={check_time!r} requires a dimension "
                            f"coordinate, but the store's {check_time!r} has dims "
                            f"{list(time_var.dims)}"
                        )
                    if not np.issubdtype(time_var.dtype, np.datetime64):
                        raise _UnsupportedTimeCoordError(
                            f"check_time={check_time!r} requires datetime64 values, "
                            f"but the store's {check_time!r} has dtype "
                            f"{time_var.dtype} (cftime/object/numeric time "
                            "representations are not supported)"
                        )
                    if ds.sizes.get(check_time, 0) == 0:
                        return False
                    time_vals = time_var.values
                    if np.isnat(time_vals).any():
                        return False
                    zero = np.timedelta64(0, "ns")
                    if time_vals.size > 1 and not np.all(np.diff(time_vals) > zero):
                        return False
                for name in sorted(probed):
                    var = ds[name]
                    corner = {d: (-1 if d == check_time else 0) for d in var.dims}
                    var.isel(corner).compute()
        except _UnsupportedTimeCoordError:
            raise
        except Exception:  # noqa: BLE001 -- a store-read failure is a cache miss, not an error
            return False
        return True

    return probe


def stamp_zarr(ds, history: list) -> None:
    """Stamp a dataset for writing: history attr, input-path strip, encoding clear.

    Serializes ``history`` (the full chain, oldest first) onto
    ``weather_skills_history`` with sorted keys, drops :data:`INPUT_PATH_ATTR`,
    and clears every variable's ``encoding`` -- per-variable encoding is not
    part of the envelope contract, so re-writes must not carry the input's
    codecs. ``weather_skills_source`` is the fetcher's own to set, with
    :func:`set_source`; it is left untouched here, so it survives the write and
    propagates to whatever a transform carries forward.
    Skills that need controlled write encodings (time units/calendar,
    ``_FillValue``) set them after this call so the clear cannot drop them.
    Other pre-existing attrs are left untouched.

    The input-path strip is load-bearing for caching, not tidiness: attrs live
    inside the store and :func:`hash_zarr` hashes the store's bytes, so a path
    left in the attrs would make the content hash vary with the local
    directory layout. The decorator merges the first input's attrs into the
    result before stamping, so the attr reaches here on every transform.
    """
    ds.attrs[HISTORY_ATTR] = json.dumps(history, sort_keys=True)
    ds.attrs.pop(INPUT_PATH_ATTR, None)
    for v in ds.variables:
        ds[v].encoding = {}


def set_source(ds, source: str):
    """Name the data product this dataset came from, returning ``ds``.

    Sets ``weather_skills_source``: the identity of the SOURCE, not of the
    skill that fetched it (``ecmwf-s2s``, not ``ecmwf-fetch``), so a rename of
    the script cannot silently rewrite provenance. A fetcher calls this in its
    body, which lets the value be one discovered at run time. The attr rides on
    the dataset, so a transform carrying its input's attrs forward propagates
    it down the pipeline.
    """
    ds.attrs[SOURCE_ATTR] = source
    return ds


def input_path(ds) -> Path:
    """The path the decorator opened this input dataset from.

    Set on every dataset the decorator opens and stripped before write, so a
    skill can name its inputs in messages and labels without the path reaching
    the written store. Raises :class:`KeyError` for a dataset the decorator
    did not open.
    """
    return Path(ds.attrs[INPUT_PATH_ATTR])


def restamp_zarr(zarr_path: Path, history: list) -> None:
    """Rewrite the history attr on an already-written store, in place.

    Updates ``weather_skills_history`` on the root group and re-consolidates
    the metadata so consolidated and unconsolidated readers agree. Chunk data
    and every other attr are untouched.
    """
    import zarr

    group = zarr.open_group(str(zarr_path), mode="r+", use_consolidated=False)
    group.attrs[HISTORY_ATTR] = json.dumps(history, sort_keys=True)
    zarr.consolidate_metadata(str(zarr_path))


def png_metadata(chains, software: str = DEFAULT_SOFTWARE) -> dict:
    """Build the ``savefig(metadata=...)`` dict for a plot skill's PNG output.

    ``chains`` is a list of ``(label, chain)`` pairs, one per input branch,
    where ``chain`` is that branch's full history (upstream + the plot entry).
    A single unlabeled input (``label`` None) uses the key
    ``weather_skills_history``; labeled inputs use suffixed keys
    (``weather_skills_history_<label>``, e.g. ``_a``/``_b`` or
    ``_forecast``/``_mclimate``). A ``Software`` key is always added.
    """
    metadata = {}
    for label, chain in chains:
        key = HISTORY_ATTR if label is None else f"{HISTORY_ATTR}_{label}"
        metadata[key] = json.dumps(chain, sort_keys=True)
    metadata["Software"] = software
    return metadata
