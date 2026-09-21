"""Plot spec, matplotlib compiler, and figure helpers.

The ``plot`` extra (matplotlib, seaborn, cartopy) is required to compile a
figure. ``import weather_skills_core.plot`` does not load matplotlib.

=====================  ========================================================
Module                 Role
=====================  ========================================================
:mod:`.spec`           Canonical JSON spec (``FLAG_TO_SPEC``, overlay/resolve)
:mod:`.theme`          Colormaps, user theme files, cmap/norm
:mod:`.figure`         Dates, colorbar, artist allowlists, PNG save
:mod:`.maps`           Lon/lat figures (heatmap, quiver, layers, grids)
:mod:`.charts`         1-D / categorical figures (lines, xy, windrose, mediogram)
=====================  ========================================================
"""

from __future__ import annotations

from weather_skills_core.plot.spec import PlotSpec, dump_spec, load_spec

__all__ = ["PlotSpec", "compile", "dump_spec", "export", "load_spec"]


def compile(spec, datasets, *, theme_registry=None):
    """Compile ``spec`` against ``datasets`` ``{id: Dataset}``.

    Returns a :class:`~weather_skills_core.plot.figure.CompiledFigure`.
    """
    from weather_skills_core.errors import UsageError
    from weather_skills_core.plot.figure import (
        colorbar_spec,
        finish_figure,
        resolve_axes_block,
    )
    from weather_skills_core.plot.spec import (
        MAP_KINDS,
        SPEC_VERSION,
        TRACE_KINDS,
        normalize_spec,
        overlay_spec,
        trace_at,
    )
    from weather_skills_core.plot.theme import (
        DEFAULT_FONTSIZE,
        DEFAULT_MAX_COLUMNS,
        normalize_template,
    )

    if hasattr(spec, "data"):
        spec = spec.data
    spec = overlay_spec({"version": SPEC_VERSION, "layout": {}, "theme": {}, "geo": {}}, spec)
    spec = normalize_spec(spec)
    traces = spec.get("traces") or [{"kind": "heatmap", "input": "a"}]
    trace0 = traces[0] if traces else {}
    kind = trace0.get("kind") or "heatmap"
    if spec.get("layers"):
        kind = "layer"
    fontsize = int((spec.get("theme") or {}).get("fontsize") or DEFAULT_FONTSIZE)
    template = normalize_template((spec.get("theme") or {}).get("template"))
    inputs = spec.get("inputs") or []

    if kind in MAP_KINDS:
        from weather_skills_core.plot.maps import compile_map

        compiled = compile_map(
            spec, datasets, fontsize=fontsize, template=template, registry=theme_registry
        )
    elif kind == "timeseries":
        from weather_skills_core.plot.charts import compile_timeseries

        compiled = compile_timeseries(spec, datasets, fontsize=fontsize, template=template)
    elif kind == "xy":
        from weather_skills_core.plot.charts import compile_xy

        compiled = compile_xy(spec, datasets, fontsize=fontsize, template=template)
    elif kind == "windrose":
        from weather_skills_core.plot.charts import compile_windrose

        compiled = compile_windrose(spec, datasets, fontsize=fontsize, template=template)
    elif kind == "grid":
        raise UsageError(
            "traces[].kind 'grid' is compiled with maps.compile_grid; "
            "the skill supplies the cell matrix"
        )
    elif kind == "mediogram":
        raise UsageError(
            "traces[].kind 'mediogram' is compiled with charts.compile_mediogram; "
            "the skill supplies the ensemble arrays"
        )
    else:
        raise UsageError(
            f"unknown traces[].kind {kind!r}; expected one of {', '.join(sorted(TRACE_KINDS))}"
        )

    fig = compiled.fig
    finish_figure(fig, spec, getattr(fig, "axes", None))
    drawn = compiled.map_drawn or {}
    resolved = overlay_spec(spec, {})
    resolved["traces"] = traces
    resolved["theme"] = {
        **(resolved.get("theme") or {}),
        "template": template,
        "fontsize": fontsize,
        "colormap": (resolved.get("theme") or {}).get("colormap") or drawn.get("colormap"),
    }
    layout = resolved.setdefault("layout", {})
    layout.setdefault("shared_colorscale", True)
    facet_in = layout.get("facet") or {}
    layout["facet"] = {
        "max_columns": facet_in.get("max_columns", DEFAULT_MAX_COLUMNS),
        **{k: drawn[k] for k in ("rows", "columns", "n_panels") if drawn.get(k) is not None},
        **{k: facet_in[k] for k in ("wspace", "hspace") if facet_in.get(k) is not None},
    }
    if drawn.get("extent"):
        resolved.setdefault("geo", {})["extent"] = drawn["extent"]
    cbar = colorbar_spec(spec)
    if cbar:
        layout["colorbar"] = cbar
    resolved["axes"] = resolve_axes_block(spec)
    along = trace_at(resolved).get("along")
    if along:
        from weather_skills_core.plot.theme import parse_along_color

        resolved["traces"][0]["along_color"] = parse_along_color(
            resolved["traces"][0].get("along_color")
        )
    if not resolved.get("inputs"):
        resolved["inputs"] = inputs
    compiled.spec = resolved
    if compiled.tight is True:
        compiled.tight = (
            spec.get("layout", {}).get("autosize", True)
            and spec.get("layout", {}).get("figsize") is None
        )
    return compiled


def export(compiled, output, *, datasets=None, dump_spec_path=None, spec=None):
    """Write ``output`` PNG. Dump the resolved spec only when ``dump_spec_path`` is set.

    Figure skills dump with ``maybe_emit_spec`` *before* compile and skip this
    function. ``dump_spec_path`` is ``None``/``False`` (skip), ``"-"`` (stdout),
    or a path. No ``*.plot.json`` sidecar is written by default.
    """
    import json
    import sys
    from pathlib import Path

    from weather_skills_core.errors import UsageError
    from weather_skills_core.plot.figure import CompiledFigure, attach_figure_spec, export_png
    from weather_skills_core.plot.spec import dump_spec

    if not isinstance(compiled, CompiledFigure):
        raise TypeError("export() expects a CompiledFigure from compile()")
    output = Path(output)
    suffix = output.suffix.lower()
    if suffix in {".html", ".htm"}:
        raise UsageError("plot output is PNG; HTML export is not supported")
    resolved_spec = compiled.spec
    layout = resolved_spec.get("layout") or {}
    tight = compiled.tight
    if layout.get("figsize"):
        tight = False
    export_png(compiled.fig, output, tight=tight)

    spec_out = attach_figure_spec(resolved_spec, spec)
    if datasets:
        for ds in datasets.values() if isinstance(datasets, dict) else datasets:
            raw = getattr(ds, "attrs", {}).get("weather_skills_history")
            if not raw:
                continue
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except json.JSONDecodeError:
                    continue
            spec_out["weather_skills_history"] = raw
            break
    if dump_spec_path in (None, False):
        return output
    text = dump_spec(spec_out, None if str(dump_spec_path) == "-" else dump_spec_path)
    if str(dump_spec_path) == "-":
        sys.stdout.write(text)
    else:
        print(f"Wrote spec: {dump_spec_path}", file=sys.stderr)
    return output
