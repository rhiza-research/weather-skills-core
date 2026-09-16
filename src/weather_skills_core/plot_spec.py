"""Weather-skills plot spec: dumpable JSON that compiles to a Plotly figure."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from weather_skills_core.errors import UsageError
from weather_skills_core.plot_style import (
    DEFAULT_MAX_COLUMNS,
    SPEC_VERSION,
    deep_merge,
)

_INDEX_INT_RE = re.compile(r"[+-]?[0-9]+")


class PlotSpec:
    """Resolved (or partial) plot configuration. Decorator may set ``.ds``."""

    def __init__(self, data: dict, path: Path | None = None):
        if not isinstance(data, dict):
            raise UsageError("plot spec must be a JSON object")
        self.data = data
        self.path = Path(path) if path is not None else None
        self.ds = None
        self.datasets = None

    def zarr_paths(self):
        paths = []
        for item in self.data.get("inputs") or []:
            raw = item.get("path") if isinstance(item, dict) else None
            if raw:
                paths.append(Path(raw))
        return paths

    def to_dict(self) -> dict:
        return copy.deepcopy(self.data)

    def __str__(self):
        return str(self.path) if self.path is not None else "<inline spec>"

    def __repr__(self):
        return f"PlotSpec({self.path!r})"


def panel_shape(n, rows=None, columns=None, *, max_columns: int = DEFAULT_MAX_COLUMNS):
    """``(nrows, ncols)`` for ``n`` heatmap panels.

    Default: up to ``max_columns`` columns, extra rows as needed (leftover
    cells stay blank). When ``rows`` and/or ``columns`` are set, leftover
    cells stay blank the same way; both given must yield a grid large
    enough to hold ``n``.
    """
    if rows is not None and rows < 1:
        raise UsageError(f"--rows must be a positive integer; got {rows}")
    if columns is not None and columns < 1:
        raise UsageError(f"--columns must be a positive integer; got {columns}")
    if rows is None and columns is None:
        ncols = min(max_columns, max(n, 1))
        nrows = (n + ncols - 1) // ncols if n else 1
        return nrows, ncols
    if rows is not None and columns is not None:
        product = rows * columns
        if product < n:
            raise UsageError(
                f"--rows {rows} × --columns {columns} = {product} panels, "
                f"but the data has {n}; the grid must hold at least {n}"
            )
        return rows, columns
    if columns is not None:
        nrows = (n + columns - 1) // columns if n else 1
        return nrows, columns
    ncols = (n + rows - 1) // rows if n else 1
    return rows, ncols


def parse_index(spec):
    """Parse ``--index`` into ``{dim: int | list[int]}`` (e.g. ``step=0,1,2``)."""
    if not spec or not str(spec).strip():
        return {}
    values = {}
    current = None
    for token in str(spec).split(","):
        if "=" in token:
            key, _, raw = token.partition("=")
            current = key.strip()
            if not current:
                raise ValueError(f"--index token {token.strip()!r} has an empty dimension name")
            if current in values:
                raise ValueError(f"--index dimension {current!r} is given more than once")
            values[current] = []
            raw = raw.strip()
            if not raw:
                raise ValueError(f"--index value for {current!r} is empty")
        else:
            raw = token.strip()
            if not raw:
                raise ValueError("--index spec has an empty token (stray comma)")
            if current is None:
                raise ValueError(f"--index token {raw!r} appears before any 'dim=' assignment")
        if not _INDEX_INT_RE.fullmatch(raw):
            raise ValueError(f"--index value {raw!r} for {current!r} is not an integer")
        pos = int(raw)
        if pos in values[current]:
            raise ValueError(f"--index position {pos} is repeated for dimension {current!r}")
        values[current].append(pos)
    return {k: v[0] if len(v) == 1 else v for k, v in values.items()}


def apply_index(da, overrides, *, list_dims=()):
    """Select integer positions from ``overrides`` (``--index``).

    A dim in ``list_dims`` may keep several positions; any other dim must be a
    single position (the dim is then dropped). Pass ``list_dims=None`` to allow
    a list on every dim.
    """
    if not overrides:
        return da
    allow_all_lists = list_dims is None
    list_dims = () if list_dims is None else list_dims
    for dim, idx in overrides.items():
        if dim not in da.dims:
            raise UsageError(
                f"--index dimension {dim!r} is not in the data (dims: {list(da.dims)})"
            )
        if isinstance(idx, list) and not allow_all_lists and dim not in list_dims:
            panel_desc = ", ".join(repr(d) for d in list_dims) if list_dims else "step/time"
            raise UsageError(
                f"--index list selection on {dim!r} is only supported "
                f"on the panel dimension ({panel_desc}); give a single position"
            )
    for dim, idx in overrides.items():
        size = da.sizes[dim]
        positions = idx if isinstance(idx, list) else [idx]
        seen = {}
        for pos in positions:
            if not -size <= pos < size:
                raise UsageError(
                    f"--index position {pos} is out of range for dimension {dim!r} (size {size})"
                )
            norm = pos % size
            if norm in seen:
                raise UsageError(
                    f"--index positions {seen[norm]} and {pos} address "
                    f"the same element of dimension {dim!r} (size {size})"
                )
            seen[norm] = pos
        da = da.isel({dim: idx}, drop=True)
    return da


def load_spec(value) -> PlotSpec:
    """Load a spec from a path, inline JSON, or existing mapping/PlotSpec."""
    if isinstance(value, PlotSpec):
        return value
    if isinstance(value, dict):
        return PlotSpec(value)
    raw = str(value).strip()
    if not raw:
        raise UsageError("plot spec is empty")
    path = Path(raw)
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        return PlotSpec(data, path)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise UsageError(f"plot spec is not a file or JSON object: {exc}") from exc
    if not isinstance(data, dict):
        raise UsageError("plot spec JSON must be an object")
    return PlotSpec(data)


def dump_spec(spec: dict | PlotSpec, path=None) -> str:
    """Serialize a spec (optionally write ``path``; ``-`` means stdout)."""
    data = spec.to_dict() if isinstance(spec, PlotSpec) else copy.deepcopy(spec)
    data.setdefault("version", SPEC_VERSION)
    text = json.dumps(data, indent=2, default=str) + "\n"
    if path is None:
        return text
    if str(path) == "-":
        import sys

        sys.stdout.write(text)
        return text
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return text


def overlay_spec(base: dict, overlay: dict | None) -> dict:
    """Deep-merge ``overlay`` onto ``base`` (overlay wins)."""
    if not overlay:
        return copy.deepcopy(base)
    return deep_merge(base, overlay)


def spec_from_flags(
    *,
    input_path=None,
    variable=None,
    style="heatmap",
    colormap=None,
    title=None,
    subplot_titles=None,
    xlabel=None,
    ylabel=None,
    cbar_label=None,
    index=None,
    extent=None,
    cities=None,
    fontsize=None,
    figsize=None,
    legend=None,
    bbox=None,
    mask_geojson=None,
    draw_boxes=None,
    rows=None,
    columns=None,
    vmin=None,
    vmax=None,
    plotly_patch=None,
) -> dict:
    """Build a (possibly partial) spec from CLI flags."""
    inputs = []
    if input_path is not None:
        item = {"id": "a", "path": str(input_path)}
        if variable:
            item["variable"] = variable
        if index:
            item["index"] = index
        inputs.append(item)
    facet = {}
    if rows is not None:
        facet["rows"] = rows
    if columns is not None:
        facet["columns"] = columns
    if style in ("heatmap", "contour"):
        facet.setdefault("max_columns", DEFAULT_MAX_COLUMNS)
    spec = {
        "version": SPEC_VERSION,
        "inputs": inputs,
        "layout": {
            "facet": facet,
            "shared_colorscale": True,
            "autosize": True,
        },
        "traces": [{"type": style, "input": "a"}],
        "style": {},
        "geo": {},
        "annotations": [],
        "shapes": [],
    }
    if colormap:
        spec["style"]["colormap"] = colormap
    if fontsize is not None:
        spec["style"]["fontsize"] = fontsize
    if title is not None:
        spec["title"] = title
    if subplot_titles:
        spec["subplot_titles"] = list(subplot_titles)
    if xlabel is not None:
        spec["xlabel"] = xlabel
    if ylabel is not None:
        spec["ylabel"] = ylabel
    if cbar_label is not None:
        spec["cbar_label"] = cbar_label
    if legend is not None:
        spec["legend"] = legend
    if vmin is not None:
        spec["vmin"] = vmin
    if vmax is not None:
        spec["vmax"] = vmax
    if figsize is not None:
        spec["layout"]["figsize"] = list(figsize)
        spec["layout"]["autosize"] = False
    if extent is not None:
        spec["geo"]["extent"] = extent
    if cities:
        spec["geo"]["cities"] = cities
    if bbox is not None:
        spec["geo"]["bbox"] = bbox
    if mask_geojson:
        spec["geo"]["mask_geojson"] = str(mask_geojson)
    if draw_boxes:
        spec["geo"]["draw_boxes"] = list(draw_boxes)
    if plotly_patch:
        spec["plotly"] = plotly_patch
    return spec


def sidecar_path(output: Path) -> Path:
    """``out.png`` → ``out.plot.json``."""
    output = Path(output)
    return output.with_name(output.stem + ".plot.json")


def spec_inputs_from_datasets(datasets) -> list[dict]:
    """Build ``inputs`` entries from decorator-opened Datasets (stamped paths)."""
    from weather_skills_core.decorator import INPUT_PATH_ATTR

    if isinstance(datasets, dict):
        items = list(datasets.items())
    else:
        items = [(chr(ord("a") + i), ds) for i, ds in enumerate(datasets)]
    inputs = []
    for key, ds in items:
        item = {"id": str(key)}
        path = getattr(ds, "attrs", {}).get(INPUT_PATH_ATTR)
        if path:
            item["path"] = str(path)
        inputs.append(item)
    return inputs
