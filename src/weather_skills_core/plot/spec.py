"""Weather-skills plot spec: dumpable JSON that compiles to a matplotlib figure."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from weather_skills_core.errors import UsageError
from weather_skills_core.plot.style import (
    COLORMAP_SPEC_KEYS,
    DEFAULT_MAX_COLUMNS,
    SPEC_VERSION,
    deep_merge,
    parse_colormap_spec,
)

_INDEX_INT_RE = re.compile(r"[+-]?[0-9]+")
_NON_ZARR_SUFFIXES = {".geojson", ".json", ".shp", ".gpkg", ".kml"}

# Artist option blocks. Their contents are checked against the matplotlib
# allowlists in plot_mpl; here we only fix where they may appear.
ARTIST_BLOCKS = frozenset(
    {"line", "mesh", "contour", "scatter", "bar", "quiver", "windrose", "fill", "box", "mediogram"}
)

# The one home for every knob. A knob named here may not be read anywhere else.
TOP_KEYS = frozenset(
    {
        "version",
        "skill",
        "inputs",
        "traces",
        "layers",
        "layout",
        "style",
        "geo",
        "axes",
        "annotations",
        "shapes",
        "title",
        "subplot_titles",
        "xlabel",
        "ylabel",
        "cbar_label",
        "legend",
        "vmin",
        "vmax",
        "weather_skills_history",
    }
)
LAYOUT_KEYS = frozenset(
    {
        "figsize",
        "autosize",
        "dpi",
        "facecolor",
        "facet",
        "colorbar",
        "shared_colorscale",
        "subplots",
    }
)
FACET_KEYS = frozenset({"rows", "columns", "max_columns", "n_panels"})
COLORBAR_KEYS = frozenset(
    {
        "extend",
        "extendfrac",
        "extendrect",
        "drawedges",
        "orientation",
        "location",
        "pad",
        "spacing",
        "format",
        "len",
        "shrink",
        "thickness",
        "ticks",
        "labels",
    }
)
STYLE_KEYS = frozenset({"template", "colormap", "fontsize", "rc"})
GEO_KEYS = frozenset(
    {"extent", "bbox", "cities", "mask_geojson", "draw_boxes", "overlays", "lat", "lon"}
)
INPUT_KEYS = frozenset({"id", "path", "variable", "index", "label", "colormap", "role"})
TRACE_KEYS = (
    frozenset(
        {
            "type",
            "input",
            "style",
            "x",
            "y",
            "path",
            "along",
            "along_color",
            "reduce",
            "align",
            "band",
            "pair_on",
            "u_variable",
            "v_variable",
            "x_variable",
            "y_variable",
            "metric",
            "leads",
        }
    )
    | ARTIST_BLOCKS
)
LAYER_KEYS = frozenset({"kind", "path", "options", "input", "raw"})

_SECTIONS = {
    "layout": LAYOUT_KEYS,
    "style": STYLE_KEYS,
    "geo": GEO_KEYS,
}

# Keys that used to be read from a second location. Naming the canonical path
# in the error is the whole point: an agent editing a sidecar gets told where
# the knob moved instead of watching its edit silently do nothing.
RELOCATED = {
    "patch": "merge your edits into the spec itself (or pass --patch on the CLI)",
    "layered": "traces[0].type = 'layer'",
    "rc": "style.rc",
    "facecolor": "layout.facecolor",
    "colorbar": "layout.colorbar",
    "along": "traces[].along",
    "along_color": "traces[].along_color",
    "reduce": "traces[].reduce",
    "align": "traces[].align",
    "band": "traces[].band",
    "u_variable": "traces[].u_variable",
    "v_variable": "traces[].v_variable",
    "x_variable": "traces[].x_variable",
    "y_variable": "traces[].y_variable",
    "pair_on": "traces[].pair_on",
    "layout.title": "title",
    "layout.axes": "axes",
    "layout.annotations": "annotations",
    "layout.shapes": "shapes",
    "layout.coloraxis": "layout.colorbar",
    "layout.rows": "layout.facet.rows",
    "layout.columns": "layout.facet.columns",
    "layout.metric": "traces[].metric",
    "layout.leads": "traces[].leads",
    "style.dpi": "layout.dpi",
    "style.max_columns": "layout.facet.max_columns",
    "style.colormap_a": "inputs[0].colormap",
    "style.colormap_b": "inputs[1].colormap",
    **{key: f"traces[].{key}" for key in sorted(ARTIST_BLOCKS)},
}


def _check_keys(obj, allowed, loc, *, relocated_prefix=""):
    """Raise on any key of ``obj`` outside ``allowed``, naming its new home."""
    if not isinstance(obj, dict):
        raise UsageError(f"plot spec {loc} must be an object")
    for key in obj:
        if key in allowed:
            continue
        moved = RELOCATED.get(f"{relocated_prefix}{key}") or RELOCATED.get(key)
        where = f"{loc}.{key}" if loc else key
        if moved:
            raise UsageError(f"plot spec {where} moved to {moved}")
        raise UsageError(
            f"plot spec {where} is not a known key; allowed here: {', '.join(sorted(allowed))}"
        )


def normalize_spec(data: dict) -> dict:
    """Validate a spec against the canonical schema, returning it unchanged.

    Every knob has exactly one home (see ``TOP_KEYS`` and friends). An unknown
    key is an error rather than a silent no-op, and a key that used to live
    somewhere else reports the path it moved to.
    """
    if not isinstance(data, dict):
        raise UsageError("plot spec must be a JSON object")
    _check_keys(data, TOP_KEYS, "")
    for section, allowed in _SECTIONS.items():
        block = data.get(section)
        if block is None:
            continue
        _check_keys(block, allowed, section, relocated_prefix=f"{section}.")
    facet = (data.get("layout") or {}).get("facet")
    if facet is not None:
        _check_keys(facet, FACET_KEYS, "layout.facet")
    colorbar = (data.get("layout") or {}).get("colorbar")
    if colorbar is not None:
        _check_keys(colorbar, COLORBAR_KEYS, "layout.colorbar")
        ticks, labels = colorbar.get("ticks"), colorbar.get("labels")
        if labels is not None and ticks is None:
            raise UsageError("plot spec layout.colorbar.labels requires layout.colorbar.ticks")
        if ticks is not None and labels is not None and len(list(ticks)) != len(list(labels)):
            raise UsageError(
                f"plot spec layout.colorbar.labels has {len(list(labels))} entries "
                f"but ticks has {len(list(ticks))}"
            )
    colormap = (data.get("style") or {}).get("colormap")
    if isinstance(colormap, dict):
        _check_keys(colormap, COLORMAP_SPEC_KEYS, "style.colormap")
        parse_colormap_spec(colormap)
    for i, item in enumerate(data.get("inputs") or []):
        if isinstance(item, dict):
            _check_keys(item, INPUT_KEYS, f"inputs[{i}]")
            if isinstance(item.get("colormap"), dict):
                parse_colormap_spec(item["colormap"])
    for i, item in enumerate(data.get("traces") or []):
        if isinstance(item, dict):
            _check_keys(item, TRACE_KEYS, f"traces[{i}]")
    for i, item in enumerate(data.get("layers") or []):
        if isinstance(item, dict):
            _check_keys(item, LAYER_KEYS, f"layers[{i}]")
    return data


def trace_at(spec: dict | None, index: int = 0) -> dict:
    """The ``traces[index]`` object, or ``{}``. The one home for trace knobs."""
    traces = (spec or {}).get("traces") or []
    if not isinstance(traces, list) or index >= len(traces):
        return {}
    trace = traces[index]
    return trace if isinstance(trace, dict) else {}


def _zarr_spec_path(raw):
    """Return a Zarr ``Path`` from a spec field, or ``None`` for ids / GeoJSON."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if "/" not in text and "\\" not in text and not text.endswith(".zarr"):
        return None
    path = Path(text)
    if path.suffix.lower() in _NON_ZARR_SUFFIXES:
        return None
    return path


class PlotSpec:
    """Resolved (or partial) plot configuration. Decorator may set ``.ds``."""

    def __init__(self, data: dict, path: Path | None = None):
        if not isinstance(data, dict):
            raise UsageError("plot spec must be a JSON object")
        self.data = normalize_spec(data)
        self.path = Path(path) if path is not None else None
        self.ds = None
        self.datasets = None

    def zarr_paths(self):
        paths = []
        seen = set()

        def add(raw):
            path = _zarr_spec_path(raw)
            if path is None:
                return
            key = str(path)
            if key in seen:
                return
            seen.add(key)
            paths.append(path)

        for item in self.data.get("inputs") or []:
            if isinstance(item, dict):
                add(item.get("path"))
        for item in self.data.get("layers") or []:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or "").lower()
            if kind in {"outline", "mask"}:
                continue
            add(item.get("path"))
        for trace in self.data.get("traces") or []:
            if isinstance(trace, dict):
                add(trace.get("x"))
                add(trace.get("y"))
                add(trace.get("path"))
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
    if isinstance(spec, dict):
        return spec
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
    # Inline JSON first. Path.is_file() stats the whole string as a filename, and
    # Linux NAME_MAX (~255 bytes) raises OSError 36 instead of returning False.
    if raw[:1] in "{[":
        return _plot_spec_from_json(raw)
    path = Path(raw)
    try:
        is_file = path.is_file()
    except OSError:
        is_file = False
    if is_file:
        data = json.loads(path.read_text(encoding="utf-8"))
        return PlotSpec(data, path)
    try:
        return _plot_spec_from_json(raw)
    except UsageError as exc:
        raise UsageError(f"plot spec is not a file or JSON object: {exc}") from exc


def _plot_spec_from_json(raw: str) -> PlotSpec:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise UsageError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise UsageError("plot spec JSON must be an object")
    return PlotSpec(data)


def prune_nulls(value):
    """Drop keys whose value is null. A null in this schema means "unset", so a
    sidecar reads better showing only what was actually resolved."""
    if isinstance(value, dict):
        return {k: prune_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [prune_nulls(v) for v in value]
    return value


def dump_spec(spec: dict | PlotSpec, path=None) -> str:
    """Serialize a spec (optionally write ``path``; ``-`` means stdout)."""
    data = spec.to_dict() if isinstance(spec, PlotSpec) else copy.deepcopy(spec)
    data.setdefault("version", SPEC_VERSION)
    normalize_spec(data)
    data = prune_nulls(data)
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


# One knob name, one canonical spec path. overlay_flags / spec_from_flags use
# this table to build JSON; figure skills no longer expose these as CLI flags.
# An int segment indexes a list.
FLAG_TO_SPEC = {
    "title": ("title",),
    "subplot_titles": ("subplot_titles",),
    "xlabel": ("xlabel",),
    "ylabel": ("ylabel",),
    "cbar_label": ("cbar_label",),
    "legend": ("legend",),
    "vmin": ("vmin",),
    "vmax": ("vmax",),
    "axes": ("axes",),
    "annotations": ("annotations",),
    "shapes": ("shapes",),
    "colormap": ("style", "colormap"),
    "colormap_bounds": ("style", "colormap", "bounds"),
    "colormap_under": ("style", "colormap", "under"),
    "colormap_over": ("style", "colormap", "over"),
    "fontsize": ("style", "fontsize"),
    "template": ("style", "template"),
    "rc": ("style", "rc"),
    "figsize": ("layout", "figsize"),
    "dpi": ("layout", "dpi"),
    "facecolor": ("layout", "facecolor"),
    "subplots": ("layout", "subplots"),
    "colorbar": ("layout", "colorbar"),
    "cbar_ticks": ("layout", "colorbar", "ticks"),
    "cbar_labels": ("layout", "colorbar", "labels"),
    "rows": ("layout", "facet", "rows"),
    "columns": ("layout", "facet", "columns"),
    "max_columns": ("layout", "facet", "max_columns"),
    "shared_colorscale": ("layout", "shared_colorscale"),
    # plot-compare / plot-compare-forecasts name the panel-column count --panels.
    "panels": ("layout", "facet", "columns"),
    "extent": ("geo", "extent"),
    "bbox": ("geo", "bbox"),
    "cities": ("geo", "cities"),
    "mask_geojson": ("geo", "mask_geojson"),
    "draw_boxes": ("geo", "draw_boxes"),
    "lat": ("geo", "lat"),
    "lon": ("geo", "lon"),
    "input_path": ("inputs", 0, "path"),
    "variable": ("inputs", 0, "variable"),
    "index": ("inputs", 0, "index"),
    "label": ("inputs", 0, "label"),
    # plot-compare's two sides are inputs[0] and inputs[1].
    "variable_a": ("inputs", 0, "variable"),
    "variable_b": ("inputs", 1, "variable"),
    "colormap_a": ("inputs", 0, "colormap"),
    "colormap_b": ("inputs", 1, "colormap"),
    "trace_type": ("traces", 0, "type"),
    "trace_style": ("traces", 0, "style"),
    "along": ("traces", 0, "along"),
    "along_color": ("traces", 0, "along_color"),
    "reduce": ("traces", 0, "reduce"),
    "align": ("traces", 0, "align"),
    "band": ("traces", 0, "band"),
    "pair_on": ("traces", 0, "pair_on"),
    "u_variable": ("traces", 0, "u_variable"),
    "v_variable": ("traces", 0, "v_variable"),
    "x_variable": ("traces", 0, "x_variable"),
    "y_variable": ("traces", 0, "y_variable"),
    "metric": ("traces", 0, "metric"),
    "leads": ("traces", 0, "leads"),
    "quiver_scale": ("traces", 0, "quiver", "scale"),
    "quiver_step": ("traces", 0, "quiver", "step"),
}

# Flags whose value is normalized on the way into the spec (JSON-safe types).
_FLAG_COERCE = {
    "figsize": lambda v: [float(v[0]), float(v[1])],
    "bbox": lambda v: list(v),
    "draw_boxes": lambda v: [list(b) for b in v],
    "subplot_titles": lambda v: list(v),
    "mask_geojson": str,
    "input_path": str,
    "reduce": lambda v: [v] if isinstance(v, str) else list(v),
    "band": lambda v: list(v) if isinstance(v, (list, tuple)) else v,
    "colormap_bounds": lambda v: [float(x) for x in v],
    "cbar_ticks": lambda v: [float(x) for x in v],
    "cbar_labels": lambda v: [str(x) for x in v],
}

# CLI flags that fold into style.colormap instead of replacing a string name.
_COLORMAP_FOLD = {
    "colormap_bounds": "bounds",
    "colormap_under": "under",
    "colormap_over": "over",
}


def _flag_path(flag):
    try:
        return FLAG_TO_SPEC[flag]
    except KeyError:
        raise UsageError(f"{flag!r} is not a known plot flag; add it to FLAG_TO_SPEC") from None


def _dig(spec, path, *, create):
    """Walk ``path[:-1]``, returning the container that holds ``path[-1]``."""
    node = spec
    for i, segment in enumerate(path[:-1]):
        nxt = path[i + 1]
        blank = [] if isinstance(nxt, int) else {}
        if isinstance(segment, int):
            if not isinstance(node, list):
                return None
            while create and len(node) <= segment:
                node.append({})
            if segment >= len(node):
                return None
            node = node[segment]
        else:
            if not isinstance(node, dict):
                return None
            if segment not in node or node[segment] is None:
                if not create:
                    return None
                node[segment] = blank
            node = node[segment]
    return node


def _colormap_object(value) -> dict:
    """Coerce a stored ``style.colormap`` value into a dict."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    return parse_colormap_spec(value)


def spec_set(spec: dict, flag: str, value) -> dict:
    """Write ``value`` at ``flag``'s canonical path, creating containers.

    ``colormap-bounds`` / ``under`` / ``over`` fold into the single
    ``style.colormap`` object so a string name becomes
    ``{name, bounds, …}`` instead of crashing when the path walks into a string.
    """
    if flag in _COLORMAP_FOLD:
        field = _COLORMAP_FOLD[flag]
        coerce = _FLAG_COERCE.get(flag)
        style = spec.setdefault("style", {})
        obj = _colormap_object(style.get("colormap"))
        obj[field] = coerce(value) if coerce else value
        style["colormap"] = obj
        return spec
    if flag == "colormap":
        style = spec.setdefault("style", {})
        current = style.get("colormap")
        incoming = parse_colormap_spec(value)
        keep = isinstance(current, dict) and any(
            current.get(key) is not None for key in ("bounds", "under", "over", "colors")
        )
        if keep:
            merged = dict(current)
            for key, item in incoming.items():
                if item is not None:
                    merged[key] = item
            style["colormap"] = merged
            return spec
        raw = value.strip() if isinstance(value, str) else None
        if raw is not None and "," not in raw and not raw.startswith("{"):
            style["colormap"] = raw
            return spec
        style["colormap"] = incoming if incoming else value
        return spec
    path = _flag_path(flag)
    holder = _dig(spec, path, create=True)
    coerce = _FLAG_COERCE.get(flag)
    holder[path[-1]] = coerce(value) if coerce else value
    return spec


def spec_get(spec: dict | None, flag: str, default=None):
    """Read ``flag`` from its canonical path in ``spec``."""
    path = _flag_path(flag)
    holder = _dig(spec or {}, path, create=False)
    if holder is None:
        return default
    key = path[-1]
    if isinstance(key, int):
        if not isinstance(holder, list) or key >= len(holder):
            return default
        found = holder[key]
    else:
        if not isinstance(holder, dict):
            return default
        found = holder.get(key)
    return default if found is None else found


def overlay_flags(spec: dict | None, **flags) -> dict:
    """Copy ``spec`` with every set flag written to its canonical path."""
    out = copy.deepcopy(spec) if spec else {}
    for flag, value in flags.items():
        if value is None or value is False or value == () or value == []:
            continue
        spec_set(out, flag, value)
    return out


def resolve_flags(spec: dict | None, **cli) -> dict:
    """Merge named knobs over a spec: a set value wins, else the spec's value."""
    return {
        flag: value if value is not None and value != () and value != [] else spec_get(spec, flag)
        for flag, value in cli.items()
    }


def params_from_spec(spec: dict | None) -> dict:
    """Read every FLAG_TO_SPEC knob from ``spec`` (``None`` if unset)."""
    return {flag: spec_get(spec, flag) for flag in FLAG_TO_SPEC}


def spec_from_flags(**flags) -> dict:
    """Build a spec from named knobs, with the structural defaults filled in."""
    trace_type = flags.pop("trace_type", None) or flags.pop("style", None) or "heatmap"
    spec = {
        "version": SPEC_VERSION,
        "inputs": [{"id": "a"}],
        "layout": {"shared_colorscale": True, "autosize": True, "facet": {}},
        "traces": [{"type": trace_type, "input": "a"}],
        "style": {},
        "geo": {},
        "annotations": [],
        "shapes": [],
    }
    if trace_type in ("heatmap", "contour"):
        spec["layout"]["facet"]["max_columns"] = DEFAULT_MAX_COLUMNS
    spec = overlay_flags(spec, **flags)
    if flags.get("figsize") is not None:
        spec["layout"]["autosize"] = False
    if not spec["inputs"][0].get("path"):
        spec["inputs"] = []
    return normalize_spec(spec)


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


SPEC_ARGUMENT_HELP = (
    "Plot spec JSON (path or inline). Figure options (title, style, colormap, "
    "layout, variable, …) belong here, not as CLI flags. A default run writes "
    "sidecar *.plot.json; edit and pass back. Inputs listed in the spec are "
    "opened for provenance; dataset flags are optional when the spec has paths."
)

# Former figure CLI flags → canonical spec path. Used only to hint when an
# agent still passes them as argparse unknowns.
PLOT_CLI_TO_SPEC = {
    "--title": "title",
    "--subplot-title": "subplot_titles",
    "--xlabel": "xlabel",
    "--ylabel": "ylabel",
    "--cbar-label": "cbar_label",
    "--cbar-ticks": "layout.colorbar.ticks",
    "--cbar-labels": "layout.colorbar.labels",
    "--legend": "legend",
    "--vmin": "vmin",
    "--vmax": "vmax",
    "--colormap": "style.colormap",
    "--colormap-bounds": "style.colormap.bounds",
    "--colormap-under": "style.colormap.under",
    "--colormap-over": "style.colormap.over",
    "--colormap-a": "inputs[0].colormap",
    "--colormap-b": "inputs[1].colormap",
    "--fontsize": "style.fontsize",
    "--theme": "style.template",
    "--figsize": "layout.figsize",
    "--rows": "layout.facet.rows",
    "--columns": "layout.facet.columns",
    "--panels": "layout.facet.columns",
    "--subplots": "layout.subplots",
    "--extent": "geo.extent",
    "--bbox": "geo.bbox",
    "--cities": "geo.cities",
    "--mask-geojson": "geo.mask_geojson",
    "--draw-box": "geo.draw_boxes",
    "--lat": "geo.lat",
    "--lon": "geo.lon",
    "--variable": "inputs[].variable",
    "-v": "inputs[].variable",
    "--variable-a": "inputs[0].variable",
    "--variable-b": "inputs[1].variable",
    "--index": "inputs[].index",
    "--label": "inputs[].label",
    "--style": "traces[].type",
    "--along": "traces[].along",
    "--along-color": "traces[].along_color",
    "--reduce": "traces[].reduce",
    "--align-day-of-year": "traces[].align",
    "--band": "traces[].band",
    "--pair-on": "traces[].pair_on",
    "--u-variable": "traces[].u_variable",
    "--v-variable": "traces[].v_variable",
    "--x-variable": "traces[].x_variable",
    "--y-variable": "traces[].y_variable",
    "--quiver-scale": "traces[].quiver.scale",
    "--quiver-step": "traces[].quiver.step",
    "--shared-scale": "layout.shared_colorscale",
    "--independent-scale": "layout.shared_colorscale",
    "--trace": "traces[].line",
    "--lead": "traces[].leads",
}


def hint_moved_plot_flags(message: str) -> str:
    """Append spec-path hints when argparse rejected a former plotting flag."""
    found = []
    for flag, path in PLOT_CLI_TO_SPEC.items():
        if re.search(rf"(?:^|[\s]){re.escape(flag)}(?:\s|=|$)", message):
            found.append((flag, path))
    if not found:
        return message
    lines = [message.rstrip(), "Plotting options belong in --spec, not CLI flags:"]
    for flag, path in found:
        lines.append(f"  {flag} → {path}")
    lines.append(
        'Example: --spec \'{"title": "S2S precip", "layout": {"facet": {"columns": 4}}}\''
    )
    return "\n".join(lines)


def patch_parser_for_spec_flags(parser):
    """Make ``parser.error`` name the spec home of a former plotting flag."""
    orig = parser.error

    def error(message):
        orig(hint_moved_plot_flags(message))

    parser.error = error
    return parser
DUMP_SPEC_ARGUMENT_HELP = (
    "Where to write the resolved plot spec. Default: <output-stem>.plot.json. "
    "Use '-' for stdout, 'none' to skip."
)


def parse_plot_spec(value):
    """Argparse converter for a plot spec path or inline JSON object."""
    import argparse

    try:
        return load_spec(value)
    except UsageError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def dump_spec_dest(value):
    """Normalize ``--dump-spec``: ``None`` (default sidecar), ``False`` (skip), or a path."""
    if value is None or value is False:
        return value
    if str(value).lower() in {"none", "off", "false"}:
        return False
    return value


def opened_datasets_from_spec(spec: PlotSpec | None) -> list:
    """Datasets the decorator opened from ``spec.zarr_paths()``."""
    if spec is None:
        return []
    if spec.datasets:
        return list(spec.datasets)
    if spec.ds is not None:
        return [spec.ds]
    return []


def named_datasets_from_spec(spec: PlotSpec | None) -> dict:
    """``{input id: Dataset}`` in spec ``inputs`` order."""
    opened = opened_datasets_from_spec(spec)
    if not opened:
        return {}
    inputs = [i for i in ((spec.data.get("inputs") if spec else None) or []) if isinstance(i, dict)]
    named = {}
    for i, ds in enumerate(opened):
        key = inputs[i].get("id") if i < len(inputs) else None
        named[str(key or chr(ord("a") + i))] = ds
    return named


def datasets_from_cli_or_spec(
    cli,
    spec,
    *,
    min_count=1,
    exactly=None,
    flag="-i/--input",
):
    """CLI dataset list wins when given; otherwise datasets opened from ``--spec``."""
    if cli is None:
        datasets = []
    elif isinstance(cli, (list, tuple)):
        datasets = [item for item in cli if item is not None]
    else:
        datasets = [cli]
    if not datasets:
        datasets = opened_datasets_from_spec(spec)
    n = len(datasets)
    if exactly is not None:
        if n != exactly:
            raise UsageError(
                f"pass {flag} {exactly} time(s), or --spec with {exactly} input path(s); got {n}"
            )
        return datasets
    if n < min_count:
        raise UsageError(f"pass {flag}, or --spec with input paths")
    return datasets


def spec_role_datasets(named: dict, prefix: str) -> list:
    """Datasets whose spec id is ``prefix`` or ``prefix`` + an integer (``forecast1``)."""
    items = []
    for key, ds in named.items():
        raw = str(key)
        if raw == prefix:
            items.append((0, ds))
        elif raw.startswith(prefix):
            suffix = raw[len(prefix) :]
            if suffix.isdigit():
                items.append((int(suffix), ds))
    items.sort(key=lambda item: item[0])
    return [ds for _i, ds in items]


def spec_input_labels(spec_data: dict | None) -> list | None:
    """Per-input ``label`` values, or ``None`` when the spec has none."""
    labels = []
    found = False
    for item in (spec_data or {}).get("inputs") or []:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        labels.append(label)
        if label:
            found = True
    return labels if found else None
