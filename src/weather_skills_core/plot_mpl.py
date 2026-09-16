"""Rasterize a Plotly figure to a matplotlib Figure (Agg PNG, no browser).

Plotly.js cannot emit PNG without Chrome/Kaleido. Weather-skills keeps Plotly as
the figure IR (HTML, spec iteration) and draws the traces we emit with matplotlib.
"""

from __future__ import annotations

import base64
import json
import math
import re
from collections import defaultdict

import numpy as np

from weather_skills_core.figure import (
    DEFAULT_DPI,
    DEFAULT_FONTSIZE,
    add_shared_colorbar,
    apply_date_ticks,
    apply_style,
)

_RGB_RE = re.compile(
    r"rgba?\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)(?:\s*,\s*([0-9.]+))?\s*\)"
)


def _figure_payload(fig) -> dict:
    """JSON dict of a Plotly figure. Numeric arrays may still use ``bdata`` encoding."""
    if hasattr(fig, "to_json"):
        return json.loads(fig.to_json())
    if hasattr(fig, "to_dict"):
        return fig.to_dict()
    return dict(fig)


def _plotly_array(value):
    """Decode a Plotly trace array (plain list or ``{dtype, bdata, shape}``)."""
    if value is None:
        return None
    if isinstance(value, dict) and "bdata" in value:
        dtype = value.get("dtype") or "f8"
        arr = np.frombuffer(base64.b64decode(value["bdata"]), dtype=dtype).copy()
        shape = value.get("shape")
        if shape:
            if isinstance(shape, str):
                shape = tuple(int(p) for p in shape.replace(" ", "").split(",") if p)
            arr = arr.reshape(shape)
        return arr
    return value


def _axis_num(ref) -> int:
    if not ref:
        return 1
    text = str(ref).split()[0]
    text = text.replace("axis", "")
    if text in {"x", "y"}:
        return 1
    try:
        return int(text[1:])
    except (TypeError, ValueError):
        return 1


def _axis_key(kind: str, idx: int) -> str:
    return f"{kind}axis" if idx == 1 else f"{kind}axis{idx}"


def _layout_axis(layout: dict, kind: str, idx: int) -> dict:
    raw = layout.get(_axis_key(kind, idx))
    return raw if isinstance(raw, dict) else {}


def _title_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return str(value.get("text") or "")
    return str(value)


def _to_mpl_color(color):
    if color is None:
        return "black"
    if isinstance(color, (int, float)):
        v = float(max(0.0, min(1.0, color)))
        return (v, v, v)
    raw = str(color).strip()
    match = _RGB_RE.fullmatch(raw)
    if match:
        r, g, b = (float(match.group(i)) for i in range(1, 4))
        if r > 1.0 or g > 1.0 or b > 1.0:
            r, g, b = r / 255.0, g / 255.0, b / 255.0
        a = float(match.group(4)) if match.group(4) is not None else 1.0
        return (r, g, b, a)
    return raw


def _unique_scale_colors(colorscale) -> list:
    colors = []
    for item in colorscale or []:
        col = _to_mpl_color(item[1])
        if not colors or colors[-1] != col:
            colors.append(col)
    return colors


def _cmap_norm(axis: dict, values=None):
    """Return ``(cmap, norm)`` for a Plotly coloraxis / colorscale dict."""
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, Normalize

    colorscale = axis.get("colorscale") if axis else None
    cmin = axis.get("cmin") if axis else None
    cmax = axis.get("cmax") if axis else None
    if cmin is None or cmax is None:
        finite = None
        if values is not None:
            arr = np.asarray(values, dtype=float)
            finite = arr[np.isfinite(arr)]
        if finite is not None and finite.size:
            if cmin is None:
                cmin = float(np.nanmin(finite))
            if cmax is None:
                cmax = float(np.nanmax(finite))
        else:
            cmin = 0.0 if cmin is None else cmin
            cmax = 1.0 if cmax is None else cmax
    if cmax == cmin:
        cmax = cmin + 1.0

    if isinstance(colorscale, str):
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap(colorscale.lower())
        return cmap, Normalize(vmin=cmin, vmax=cmax)

    tickvals = ((axis.get("colorbar") or {}).get("tickvals")) if axis else None
    unique = _unique_scale_colors(colorscale) if isinstance(colorscale, list) else []
    if tickvals and len(unique) == len(tickvals) - 1:
        from matplotlib.colors import ListedColormap

        bounds = [float(v) for v in tickvals]
        cmap = ListedColormap(unique).with_extremes(under=unique[0], over=unique[-1])
        return cmap, BoundaryNorm(bounds, cmap.N)

    if isinstance(colorscale, list) and colorscale:
        stops = [(float(item[0]), _to_mpl_color(item[1])) for item in colorscale]
        cmap = LinearSegmentedColormap.from_list("plotly", stops)
        return cmap, Normalize(vmin=cmin, vmax=cmax)

    import matplotlib.pyplot as plt

    return plt.get_cmap("viridis"), Normalize(vmin=cmin, vmax=cmax)


def _coloraxis(layout: dict, name) -> dict:
    if not name:
        name = "coloraxis"
    raw = layout.get(name)
    return raw if isinstance(raw, dict) else {}


def _trace_coloraxis(trace: dict) -> str | None:
    name = trace.get("coloraxis")
    if name:
        return name
    marker = trace.get("marker") or {}
    return marker.get("coloraxis")


def _as_xy(values):
    values = _plotly_array(values)
    if values is None:
        return np.array([])
    if isinstance(values, np.ndarray) and values.dtype.kind in "fiu":
        return np.asarray(values, dtype=float)
    seq = list(values)
    if not seq:
        return np.array([])
    sample = next((v for v in seq if v is not None), None)
    if isinstance(sample, str) and len(sample) >= 10 and sample[4:5] == "-":
        from matplotlib.dates import datestr2num

        out = []
        for v in seq:
            if v is None:
                out.append(np.nan)
            else:
                out.append(datestr2num(str(v)))
        return np.asarray(out, dtype=float)
    out = []
    for v in seq:
        out.append(np.nan if v is None else v)
    try:
        return np.asarray(out, dtype=float)
    except (TypeError, ValueError):
        return np.asarray(out, dtype=object)


def _is_datetime_x(values) -> bool:
    if not values:
        return False
    sample = next((v for v in values if v is not None), None)
    return isinstance(sample, str) and len(sample) >= 10 and sample[4:5] == "-"


def _grid_shape(layout: dict, n_axes: int) -> tuple[int, int]:
    grid = layout.get("grid") or {}
    nrows = int(grid.get("rows") or 0)
    ncols = int(grid.get("columns") or 0)
    if nrows and ncols:
        return nrows, ncols
    if n_axes <= 1:
        return 1, 1
    ncols = int(math.ceil(math.sqrt(n_axes)))
    nrows = int(math.ceil(n_axes / ncols))
    return nrows, ncols


def _anchor_ha_va(position: str | None) -> tuple[str, str]:
    raw = (position or "top right").lower()
    va = "bottom" if "top" in raw else "top" if "bottom" in raw else "center"
    ha = "left" if "right" in raw else "right" if "left" in raw else "center"
    return ha, va


def _draw_heatmap(ax, trace, cmap, norm):
    z = np.asarray(_plotly_array(trace.get("z")), dtype=float)
    x = np.asarray(_plotly_array(trace.get("x")), dtype=float)
    y = np.asarray(_plotly_array(trace.get("y")), dtype=float)
    if z.ndim != 2 or x.size == 0 or y.size == 0:
        return None
    cmap = cmap.with_extremes(bad=(0.0, 0.0, 0.0, 0.0))
    return ax.pcolormesh(x, y, z, cmap=cmap, norm=norm, shading="nearest")


def _draw_contour(ax, trace, cmap, norm):
    z = np.asarray(_plotly_array(trace.get("z")), dtype=float)
    x = np.asarray(_plotly_array(trace.get("x")), dtype=float)
    y = np.asarray(_plotly_array(trace.get("y")), dtype=float)
    if z.ndim != 2 or x.size == 0 or y.size == 0:
        return None
    filled = ax.contourf(x, y, z, cmap=cmap, norm=norm, levels=12)
    line = trace.get("line") or {}
    ax.contour(
        x,
        y,
        z,
        levels=filled.levels,
        colors=[_to_mpl_color(line.get("color") or "black")],
        linewidths=float(line.get("width") or 0.4),
    )
    return filled


def _draw_scatter(ax, trace, layout, cmap=None, norm=None):
    xraw = _plotly_array(trace.get("x"))
    yraw = _plotly_array(trace.get("y"))
    xraw = list(xraw) if xraw is not None else []
    yraw = list(yraw) if yraw is not None else []
    x = _as_xy(xraw)
    y = _as_xy(yraw)
    mode = str(trace.get("mode") or "lines")
    line = trace.get("line") or {}
    marker = trace.get("marker") or {}
    color = _to_mpl_color(line.get("color") or marker.get("color") or "#222222")
    width = float(line.get("width") or 2)
    opacity = float(trace.get("opacity") or 1.0)
    label = trace.get("name") if trace.get("showlegend") else None
    mappable = None
    if "lines" in mode:
        ax.plot(
            x,
            y,
            color=color,
            linewidth=width,
            alpha=opacity,
            label=label,
            zorder=3,
        )
        label = None
    if "markers" in mode:
        cvals = _plotly_array(marker.get("color"))
        kwargs = {
            "alpha": opacity,
            "label": label,
            "s": float(marker.get("size") or 8) ** 2,
            "zorder": 4,
        }
        edge = (marker.get("line") or {}).get("color")
        if edge:
            kwargs["edgecolors"] = _to_mpl_color(edge)
            kwargs["linewidths"] = float((marker.get("line") or {}).get("width") or 0.4)
        if isinstance(cvals, (list, tuple, np.ndarray)) and cmap is not None:
            mappable = ax.scatter(
                x, y, c=np.asarray(cvals, dtype=float), cmap=cmap, norm=norm, **kwargs
            )
        else:
            if cvals is not None and not isinstance(cvals, (list, tuple, np.ndarray)):
                kwargs["color"] = _to_mpl_color(cvals)
            else:
                kwargs["color"] = color
            ax.scatter(x, y, **kwargs)
    if "text" in mode:
        texts = list(trace.get("text") or [])
        ha, va = _anchor_ha_va(trace.get("textposition"))
        for xi, yi, t in zip(x, y, texts, strict=False):
            if t is None or (isinstance(xi, float) and not np.isfinite(xi)):
                continue
            ax.annotate(
                str(t),
                (xi, yi),
                textcoords="offset points",
                xytext=(-4, 4),
                ha=ha,
                va=va,
                fontsize=max(8, int(layout.get("font", {}).get("size") or DEFAULT_FONTSIZE) - 2),
            )
    if _is_datetime_x(xraw):
        apply_date_ticks(ax)
    return mappable


def _draw_bar(ax, trace):
    xraw = _plotly_array(trace.get("x"))
    y = np.asarray(_plotly_array(trace.get("y")), dtype=float)
    xraw = list(xraw) if xraw is not None else []
    color = _to_mpl_color((trace.get("marker") or {}).get("color") or "#4c78a8")
    label = trace.get("name") if trace.get("showlegend") else None
    if _is_datetime_x(xraw):
        x = _as_xy(xraw)
        ax.bar(x, y, color=color, label=label, width=0.8)
        apply_date_ticks(ax)
        return
    ax.bar(range(len(y)), y, color=color, label=label)
    if xraw:
        ax.set_xticks(range(len(xraw)))
        ax.set_xticklabels([str(v) for v in xraw], rotation=30, ha="right")


def _draw_boxes(ax, traces: list[dict]):
    by_group = defaultdict(lambda: defaultdict(list))
    group_meta = {}
    x_order = []
    for trace in traces:
        xs = _plotly_array(trace.get("x"))
        ys = _plotly_array(trace.get("y"))
        xs = [] if xs is None else list(xs)
        ys = [] if ys is None else list(ys)
        og = trace.get("offsetgroup") or trace.get("name") or "box"
        color = _to_mpl_color(
            trace.get("marker_color") or (trace.get("marker") or {}).get("color") or "C0"
        )
        group_meta[og] = {
            "color": color,
            "name": trace.get("name") if trace.get("showlegend") else None,
        }
        for x, y in zip(xs, ys, strict=False):
            if x not in x_order:
                x_order.append(x)
            by_group[og][x].append(y)
    groups = list(by_group)
    n_groups = max(len(groups), 1)
    width = 0.8 / n_groups
    positions_all = []
    for g_i, og in enumerate(groups):
        offset = (g_i - (n_groups - 1) / 2) * width
        positions = [i + offset for i in range(len(x_order))]
        data = [by_group[og].get(x, [np.nan]) for x in x_order]
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=width * 0.85,
            patch_artist=True,
            manage_ticks=False,
        )
        meta = group_meta.get(og) or {}
        for patch in bp["boxes"]:
            patch.set_facecolor(meta.get("color") or "white")
            patch.set_edgecolor("black")
        if meta.get("name"):
            bp["boxes"][0].set_label(meta["name"])
        positions_all.extend(positions)
    ax.set_xticks(range(len(x_order)))
    ax.set_xticklabels([str(x) for x in x_order], rotation=30, ha="right")
    if positions_all:
        pad = width
        ax.set_xlim(min(positions_all) - pad, max(positions_all) + pad)


def _apply_axis(ax, layout: dict, idx: int, *, equal_aspect: bool):
    xaxis = _layout_axis(layout, "x", idx)
    yaxis = _layout_axis(layout, "y", idx)
    if xaxis.get("visible") is False and yaxis.get("visible") is False:
        ax.set_visible(False)
        ax.set_axis_off()
        return
    xr = xaxis.get("range")
    yr = yaxis.get("range")
    if xr and len(xr) == 2:
        ax.set_xlim(float(xr[0]), float(xr[1]))
    if yr and len(yr) == 2:
        ax.set_ylim(float(yr[0]), float(yr[1]))
    xlabel = _title_text(xaxis.get("title"))
    ylabel = _title_text(yaxis.get("title"))
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if equal_aspect or yaxis.get("scaleanchor"):
        ax.set_aspect("equal", adjustable="box")


def _add_shapes(axes, layout: dict, ncols: int):
    from matplotlib.patches import Rectangle

    for shape in layout.get("shapes") or []:
        if not isinstance(shape, dict) or shape.get("type") not in (None, "rect"):
            continue
        idx = _axis_num(shape.get("xref") or "x")
        row, col = divmod(idx - 1, ncols)
        if row >= axes.shape[0] or col >= axes.shape[1]:
            continue
        ax = axes[row, col]
        x0, x1 = float(shape.get("x0", 0)), float(shape.get("x1", 0))
        y0, y1 = float(shape.get("y0", 0)), float(shape.get("y1", 0))
        line = shape.get("line") or {}
        ax.add_patch(
            Rectangle(
                (min(x0, x1), min(y0, y1)),
                abs(x1 - x0),
                abs(y1 - y0),
                fill=False,
                edgecolor=_to_mpl_color(line.get("color") or "black"),
                linewidth=float(line.get("width") or 1.5),
                zorder=6,
            )
        )


def _add_annotations(axes, layout: dict, ncols: int):
    for ann in layout.get("annotations") or []:
        if not isinstance(ann, dict):
            continue
        text = ann.get("text")
        if not text:
            continue
        xref = str(ann.get("xref") or "")
        yref = str(ann.get("yref") or "")
        if "domain" not in xref or "domain" not in yref:
            continue
        idx = _axis_num(xref)
        row, col = divmod(idx - 1, ncols)
        if row >= axes.shape[0] or col >= axes.shape[1]:
            continue
        ax = axes[row, col]
        x = float(ann.get("x") if ann.get("x") is not None else 0.5)
        y = float(ann.get("y") if ann.get("y") is not None else 0.5)
        font = ann.get("font") or {}
        color = _to_mpl_color(font.get("color") or "#333333")
        size = font.get("size")
        yanchor = str(ann.get("yanchor") or "center")
        if yanchor == "bottom" or y >= 1.0:
            ax.set_title(str(text))
            continue
        ha = {"left": "left", "right": "right"}.get(str(ann.get("xanchor") or ""), "center")
        va = {"top": "top", "bottom": "bottom"}.get(yanchor, "center")
        ax.text(
            x,
            y,
            str(text),
            transform=ax.transAxes,
            ha=ha,
            va=va,
            color=color,
            fontsize=size,
            zorder=7,
        )


def plotly_to_mpl(fig, *, width=None, height=None):
    """Build a matplotlib Figure from a Plotly figure (heatmap/contour/scatter/bar/box)."""
    import matplotlib

    try:
        matplotlib.use("Agg")
    except Exception:
        pass
    import matplotlib.pyplot as plt

    payload = _figure_payload(fig)
    layout = payload.get("layout") or {}
    traces = list(payload.get("data") or [])
    fontsize = int((layout.get("font") or {}).get("size") or DEFAULT_FONTSIZE)
    apply_style(fontsize)

    n_axes = 1
    for tr in traces:
        n_axes = max(n_axes, _axis_num(tr.get("xaxis")))
    for key in layout:
        if str(key).startswith("xaxis"):
            n_axes = max(n_axes, _axis_num(str(key).replace("axis", "")))
    nrows, ncols = _grid_shape(layout, n_axes)

    dpi = DEFAULT_DPI
    width = width or layout.get("width") or int(8 * DEFAULT_DPI)
    height = height or layout.get("height") or int(5 * DEFAULT_DPI)
    fig_mpl, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(int(width) / dpi, int(height) / dpi),
        dpi=dpi,
        squeeze=False,
        layout="constrained",
    )

    by_subplot = defaultdict(list)
    for tr in traces:
        by_subplot[_axis_num(tr.get("xaxis"))].append(tr)

    mappables = {}
    equal_by_axis = {}
    for idx in range(1, nrows * ncols + 1):
        yaxis = _layout_axis(layout, "y", idx)
        equal_by_axis[idx] = bool(yaxis.get("scaleanchor"))

    for idx, group in by_subplot.items():
        row, col = divmod(idx - 1, ncols)
        if row >= nrows or col >= ncols:
            continue
        ax = axes[row, col]
        boxes = [tr for tr in group if tr.get("type") == "box"]
        if boxes:
            _draw_boxes(ax, boxes)
        for tr in group:
            kind = tr.get("type") or "scatter"
            cname = _trace_coloraxis(tr)
            axis = _coloraxis(layout, cname) if cname else {}
            marker_color = (tr.get("marker") or {}).get("color")
            values = _plotly_array(tr.get("z") if kind in {"heatmap", "contour"} else marker_color)
            cmap = norm = None
            if cname or kind in {"heatmap", "contour"}:
                cmap, norm = _cmap_norm(axis or tr, values)
            mappable = None
            if kind == "heatmap":
                mappable = _draw_heatmap(ax, tr, cmap, norm)
            elif kind == "contour":
                mappable = _draw_contour(ax, tr, cmap, norm)
            elif kind == "bar":
                _draw_bar(ax, tr)
            elif kind == "box":
                continue
            else:
                mappable = _draw_scatter(ax, tr, layout, cmap=cmap, norm=norm)
            if mappable is not None and cname and cname not in mappables:
                mappables[cname] = (mappable, axis)

        _apply_axis(ax, layout, idx, equal_aspect=equal_by_axis.get(idx, False))

    for idx in range(1, nrows * ncols + 1):
        if idx in by_subplot:
            continue
        row, col = divmod(idx - 1, ncols)
        if row < nrows and col < ncols:
            _apply_axis(axes[row, col], layout, idx, equal_aspect=False)

    _add_shapes(axes, layout, ncols)
    _add_annotations(axes, layout, ncols)

    title = _title_text(layout.get("title"))
    if title:
        fig_mpl.suptitle(title)

    visible = [ax for ax in axes.ravel() if ax.get_visible()]
    for _name, (mappable, axis) in mappables.items():
        colorbar = axis.get("colorbar") or {}
        label = _title_text(colorbar.get("title"))
        ticks = colorbar.get("tickvals")
        ticktext = colorbar.get("ticktext")
        location = "right" if len(visible) <= 1 and len(mappables) <= 1 else "bottom"
        if colorbar.get("orientation") == "h":
            location = "bottom"
        elif colorbar.get("orientation") == "v":
            location = "right"
        cbar = add_shared_colorbar(
            fig_mpl,
            mappable,
            visible or axes.ravel(),
            label=label,
            location=location,
            ticks=ticks,
        )
        if cbar is not None and ticktext:
            cbar.set_ticklabels(list(ticktext))

    if layout.get("showlegend"):
        handles, labels = [], []
        for ax in visible:
            h, lab = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(lab)
        if handles:
            fig_mpl.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4))

    return fig_mpl
