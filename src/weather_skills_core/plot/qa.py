"""Stdout QA for a written plot PNG: pixel hash + whether plotted data is finite."""

from __future__ import annotations

import hashlib
import sys


def hash_png_pixels(path) -> str:
    """sha256 of RGB pixels. Provenance tEXt does not change this; a corner mark does."""
    from PIL import Image

    with Image.open(path) as img:
        payload = img.convert("RGB").tobytes()
    return hashlib.sha256(payload).hexdigest()


def _iter_datasets(datasets):
    if datasets is None:
        return
    items = datasets.values() if isinstance(datasets, dict) else datasets
    if hasattr(datasets, "data_vars") and not isinstance(datasets, dict):
        items = (datasets,)
    for ds in items:
        if ds is None:
            continue
        if hasattr(ds, "data_vars"):
            yield ds
        inner = getattr(ds, "datasets", None)
        if inner:
            yield from _iter_datasets(inner)
        one = getattr(ds, "ds", None)
        if one is not None and one is not ds:
            yield from _iter_datasets(one)


def _numeric_values(da):
    import numpy as np

    if hasattr(da, "pint") and getattr(da.pint, "units", None) is not None:
        da = da.pint.dequantify()
    values = np.asarray(getattr(da, "values", da))
    if values.dtype.kind not in "biufc":
        return None
    return values


def data_var_counts(datasets) -> list[tuple[str, int, int]]:
    """``(name, n_finite, n)`` for each numeric data variable."""
    import numpy as np

    rows = []
    seen: set[int] = set()
    for ds in _iter_datasets(datasets):
        key = id(ds)
        if key in seen:
            continue
        seen.add(key)
        for name, da in ds.data_vars.items():
            values = _numeric_values(da)
            if values is None:
                continue
            n = int(values.size)
            n_finite = int(np.isfinite(values).sum()) if n else 0
            rows.append((str(name), n_finite, n))
    return rows


def data_status(datasets) -> tuple[str, str]:
    """Return ``(status, detail)``.

    ``status`` is ``not null`` when any numeric data variable has a finite
    value, ``NULL`` when every such variable is all-NaN / empty, or
    ``not checked`` when there is nothing to scan.
    """
    rows = data_var_counts(datasets)
    if not rows:
        return "not checked", "no numeric data variables"
    detail = ", ".join(f"{name} {finite}/{n} finite" for name, finite, n in rows)
    if any(finite > 0 for _name, finite, _n in rows):
        return "not null", detail
    return "NULL", detail


def report_figure(path, datasets=None, *, file=None) -> str:
    """Print ``plot hash`` and ``data:`` lines. Returns the pixel hash."""
    dest = sys.stdout if file is None else file
    digest = hash_png_pixels(path)
    status, detail = data_status(datasets)
    print(f"plot hash: {digest}", file=dest)
    print(f"data: {status} ({detail})", file=dest)
    if status == "NULL":
        print(
            "data is all NaN — inspect-zarr the input before treating this figure as valid",
            file=dest,
        )
    return digest
