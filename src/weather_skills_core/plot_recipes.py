"""Plotly compilers for multi-panel recipes (compare, verify, timeseries, mediogram)."""

from __future__ import annotations

import numpy as np

from weather_skills_core.plot_compile import (
    _as_plotly_x,
    _coloraxis,
    _figsize_from_extent,
    _geojson_lines,
)
from weather_skills_core.plot_style import (
    DEFAULT_DPI,
    DEFAULT_FONTSIZE,
    discrete_colorscale,
    register_template,
    resolve_colorscale,
)

HITS_COLORS = ["#d73027", "#f0f0f0", "#1a9850"]
ERROR_DIVERGING_COLORS = [
    "#053061",
    "#2166ac",
    "#4393c3",
    "#92c5de",
    "#d1e5f0",
    "#ffffff",
    "#fddbc7",
    "#f4a582",
    "#d6604d",
    "#b2182b",
    "#67001f",
]
MAE_FROM_WHITE_COLORS = [
    "#ffffff",
    "#fddbc7",
    "#f4a582",
    "#d6604d",
    "#b2182b",
    "#67001f",
]


def _layout_size(figsize, *, default_in, dpi=DEFAULT_DPI):
    if figsize is not None:
        return int(float(figsize[0]) * dpi), int(float(figsize[1]) * dpi), False
    return int(default_in[0] * dpi), int(default_in[1] * dpi), True


def plotly_color(color):
    """Map matplotlib-style grayscale numbers / names to a Plotly color."""
    if color is None:
        return None
    if isinstance(color, (int, float)):
        v = float(color)
        c = int(round(max(0.0, min(1.0, v)) * 255))
        return f"rgb({c},{c},{c})"
    raw = str(color).strip()
    try:
        v = float(raw)
    except ValueError:
        return raw
    if 0.0 <= v <= 1.0:
        c = int(round(v * 255))
        return f"rgb({c},{c},{c})"
    return raw


def scale_from_da(da, colormap=None, *, stretch=False, label=None, vmin=None, vmax=None):
    """``resolve_colorscale`` plus optional user limits and a colorbar label."""
    scale = resolve_colorscale(
        da, colormap, stretch=stretch or vmin is not None or vmax is not None
    )
    if vmin is not None:
        scale["cmin"] = vmin
    if vmax is not None:
        scale["cmax"] = vmax
    if label:
        scale["label"] = label
    return scale


def hits_scale(*, label="event"):
    return {
        "name": "hits",
        "colorscale": discrete_colorscale(HITS_COLORS),
        "cmin": -1.5,
        "cmax": 1.5,
        "bounds": [-1, 0, 1],
        "ticktext": ["disagree", "below", "hit"],
        "label": label,
    }


def error_scale(da, metric, *, label=None):
    """Bias (diverging, white at 0) or MAE (white→warm) Plotly colorscale."""
    vals = np.asarray(da.values, dtype=float)
    finite = vals[np.isfinite(vals)]
    if metric == "bias":
        if finite.size == 0:
            lo, hi = -1.0, 1.0
        else:
            lo = float(np.nanmin(finite))
            hi = float(np.nanmax(finite))
            m = max(abs(lo), abs(hi), 1e-6)
            lo, hi = -m, m
        return {
            "name": "verify_bias",
            "colorscale": discrete_colorscale(ERROR_DIVERGING_COLORS),
            "cmin": lo,
            "cmax": hi,
            "label": label or "bias",
        }
    if finite.size == 0:
        lo, hi = 0.0, 1.0
    else:
        lo = 0.0
        hi = float(np.nanmax(finite)) or 1.0
    return {
        "name": "verify_mae",
        "colorscale": discrete_colorscale(MAE_FROM_WHITE_COLORS),
        "cmin": lo,
        "cmax": hi,
        "label": label or "mae",
    }


def heatmap_cell(da, lat_dim, lon_dim, *, coloraxis="coloraxis"):
    slab = da.transpose(lat_dim, lon_dim)
    return {
        "kind": "heatmap",
        "x": np.asarray(slab[lon_dim].values, dtype=float),
        "y": np.asarray(slab[lat_dim].values, dtype=float),
        "z": np.asarray(slab.values, dtype=float),
        "coloraxis": coloraxis,
    }


def scatter_cell(lons, lats, values, *, coloraxis="coloraxis"):
    return {
        "kind": "scatter",
        "x": np.asarray(lons, dtype=float),
        "y": np.asarray(lats, dtype=float),
        "c": np.asarray(values, dtype=float),
        "coloraxis": coloraxis,
    }


def blank_cell(text="n/a"):
    return {"kind": "blank", "text": text}


def _place_coloraxes(coloraxes, nrows, ncols):
    layout = {}
    names = list(coloraxes)
    n_axes = len(names)
    n_panels = max(nrows * ncols, 1)
    for i, name in enumerate(names):
        scale = coloraxes[name]
        axis = _coloraxis(scale, scale.get("label") or "", n_panels, None, None)
        if n_axes > 1 and not (scale.get("colorbar") or {}):
            axis["colorbar"]["orientation"] = "v"
            axis["colorbar"]["x"] = 1.02
            axis["colorbar"]["len"] = max(0.28, 0.85 / n_axes)
            axis["colorbar"]["y"] = 1.0 - (i + 0.5) / n_axes
            axis["colorbar"]["yanchor"] = "middle"
        layout[name] = axis
    return layout


def compile_heatmap_grid(
    cells,
    *,
    extent,
    title=None,
    col_titles=None,
    row_titles=None,
    fontsize=DEFAULT_FONTSIZE,
    figsize=None,
    coloraxes=None,
    overlays=True,
    xlabel="Longitude",
    cell_notes=None,
):
    """Compile a 2-D grid of heatmap/scatter/blank cells.

    ``cells`` is a list of rows; each row is a list of dicts or None.
    A cell dict is ``{"kind": "heatmap", "x", "y", "z", "coloraxis"}`` or
    ``{"kind": "scatter", "x", "y", "c", "coloraxis"}`` or
    ``{"kind": "blank", "text": "n/a"}``.
    ``coloraxes`` maps Plotly axis name → scale dict from ``resolve_colorscale``.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    nrows = len(cells)
    ncols = max((len(row) for row in cells), default=1)
    titles = []
    for r in range(nrows):
        for c in range(ncols):
            if r == 0 and col_titles and c < len(col_titles):
                titles.append(col_titles[c])
            else:
                titles.append("")
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.04,
        vertical_spacing=0.08 if nrows > 1 else 0.06,
        subplot_titles=titles,
    )
    geo_x = geo_y = None
    if overlays and extent is not None:
        geo_x, geo_y = _geojson_lines(extent)
    coloraxes = coloraxes or {}
    layout_coloraxes = _place_coloraxes(coloraxes, nrows, ncols)

    for r, row in enumerate(cells):
        row_i = r + 1
        for c in range(ncols):
            col_i = c + 1
            i = r * ncols + c
            xref = "x" if i == 0 else f"x{i + 1}"
            if extent is not None:
                fig.update_xaxes(range=[extent[0], extent[1]], row=row_i, col=col_i)
                fig.update_yaxes(
                    range=[extent[2], extent[3]],
                    scaleanchor=xref,
                    scaleratio=1,
                    constrain="domain",
                    row=row_i,
                    col=col_i,
                )
            ylabel = ""
            if row_titles and c == 0 and r < len(row_titles):
                ylabel = row_titles[r]
            fig.update_xaxes(title_text=xlabel if r == nrows - 1 else "", row=row_i, col=col_i)
            fig.update_yaxes(title_text=ylabel, row=row_i, col=col_i)
            cell = row[c] if c < len(row) else None
            if cell is None:
                cell = blank_cell()
            kind = cell.get("kind") or "heatmap"
            caxis = cell.get("coloraxis") or "coloraxis"
            if kind == "blank":
                fig.add_annotation(
                    text=cell.get("text") or "n/a",
                    row=row_i,
                    col=col_i,
                    x=0.5,
                    y=0.5,
                    xref="x domain",
                    yref="y domain",
                    showarrow=False,
                    font={"color": "#888888", "size": max(10, int(fontsize * 0.8))},
                )
            elif kind == "scatter":
                fig.add_trace(
                    go.Scatter(
                        x=list(cell["x"]),
                        y=list(cell["y"]),
                        mode="markers",
                        marker={
                            "color": list(cell.get("c") if cell.get("c") is not None else "#222"),
                            "coloraxis": caxis,
                            "size": 8,
                            "line": {"width": 0.4, "color": "#333"},
                        },
                        showlegend=False,
                    ),
                    row=row_i,
                    col=col_i,
                )
            else:
                fig.add_trace(
                    go.Heatmap(
                        x=np.asarray(cell["x"], dtype=float),
                        y=np.asarray(cell["y"], dtype=float),
                        z=np.asarray(cell["z"], dtype=float),
                        coloraxis=caxis,
                        hoverongaps=False,
                    ),
                    row=row_i,
                    col=col_i,
                )
            if geo_x and kind != "blank":
                fig.add_trace(
                    go.Scatter(
                        x=geo_x,
                        y=geo_y,
                        mode="lines",
                        line={"color": "#444444", "width": 0.5},
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                    row=row_i,
                    col=col_i,
                )
            note = None
            if cell_notes and r < len(cell_notes) and c < len(cell_notes[r]):
                note = cell_notes[r][c]
            if note:
                fig.add_annotation(
                    text=note,
                    row=row_i,
                    col=col_i,
                    x=0.03,
                    y=0.97,
                    xref="x domain",
                    yref="y domain",
                    xanchor="left",
                    yanchor="top",
                    showarrow=False,
                    font={"size": max(8, int(fontsize * 0.6)), "color": "#333333"},
                )

    sw, sh = (5.0, 4.0)
    if extent is not None:
        sw, sh = _figsize_from_extent(*extent)
    width, height, autosize = _layout_size(
        figsize, default_in=(max(sw * ncols, 6.0), max(sh * nrows, 4.0))
    )
    layout_kw = {
        "template": register_template(fontsize),
        "title": title or None,
        "showlegend": False,
        "width": width,
        "height": height,
        "autosize": autosize and figsize is None,
    }
    layout_kw.update(layout_coloraxes)
    fig.update_layout(**layout_kw)
    return fig


def compile_line_figure(
    series,
    *,
    title=None,
    xlabel="",
    ylabels=None,
    fontsize=DEFAULT_FONTSIZE,
    figsize=None,
    subplots=False,
    kinds=None,
    styles=None,
):
    """``series`` is a list of ``(x, y, label)``. ``y`` may be 1-D or 2-D (along)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n = len(series)
    kinds = kinds or ["line"] * n
    styles = styles or [{} for _ in series]
    ylabels = ylabels or [""] * n
    if subplots:
        fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.06)
        width, height, autosize = _layout_size(figsize, default_in=(10.0, max(2.8 * n, 4.0)))
    else:
        fig = go.Figure()
        width, height, autosize = _layout_size(figsize, default_in=(10.0, 6.0))
    for i, ((xvals, yvals, label), kind, style) in enumerate(
        zip(series, kinds, styles, strict=True)
    ):
        row = i + 1 if subplots else None
        color = plotly_color(style.get("color"))
        width_pt = style.get("lw") or style.get("linewidth") or style.get("width") or 2
        yarr = np.asarray(yvals, dtype=float)
        xplot = _as_plotly_x(xvals)
        traces_y = [yarr] if yarr.ndim == 1 else [yarr[:, j] for j in range(yarr.shape[1])]
        for j, yy in enumerate(traces_y):
            name = label if j == 0 else None
            showlegend = j == 0 and not subplots
            if kind == "bar":
                tr = go.Bar(
                    x=xplot,
                    y=yy,
                    name=name,
                    marker={"color": color},
                    showlegend=showlegend,
                )
            else:
                markers = style.get("marker")
                mode = "lines"
                if markers not in (None, "None", "none", "null"):
                    mode = "lines+markers"
                tr = go.Scatter(
                    x=xplot,
                    y=yy,
                    mode=mode,
                    name=name,
                    line={"width": width_pt, "color": color},
                    marker={"size": float(style.get("markersize") or 6)},
                    opacity=float(style.get("alpha") or 1),
                    showlegend=showlegend,
                )
            if subplots:
                fig.add_trace(tr, row=row, col=1)
            else:
                fig.add_trace(tr)
        if subplots:
            fig.update_yaxes(title_text=ylabels[i], row=row, col=1)
            fig.update_xaxes(title_text=xlabel if i == n - 1 else "", row=row, col=1)
    if not subplots:
        fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabels[0] if ylabels else "")
    fig.update_layout(
        template=register_template(fontsize),
        title=title or None,
        width=width,
        height=height,
        autosize=autosize and figsize is None,
        legend={"orientation": "h", "y": -0.2},
        barmode="group",
    )
    return fig


def compile_mediogram(
    fc,
    mc,
    tick_labels,
    *,
    title=None,
    xlabel="Forecast step",
    ylabel="",
    fontsize=DEFAULT_FONTSIZE,
    figsize=None,
):
    """ECMWF-style two-layer boxes: forecast (cyan) vs m-climate (red)."""
    import plotly.graph_objects as go

    n_steps = fc.shape[1]
    fig = go.Figure()
    for i in range(n_steps):
        fig.add_trace(
            go.Box(
                y=np.asarray(fc[:, i], dtype=float),
                name="forecast",
                legendgroup="forecast",
                showlegend=i == 0,
                marker_color="cyan",
                line={"color": "black"},
                offsetgroup="fc",
                x=[tick_labels[i]] * fc.shape[0],
            )
        )
        fig.add_trace(
            go.Box(
                y=np.asarray(mc[:, i], dtype=float),
                name="m-climate",
                legendgroup="m-climate",
                showlegend=i == 0,
                marker_color="red",
                line={"color": "black"},
                offsetgroup="mc",
                x=[tick_labels[i]] * mc.shape[0],
            )
        )
    fig.add_trace(
        go.Scatter(
            x=list(tick_labels),
            y=np.mean(fc, axis=0),
            mode="lines",
            name="forecast mean",
            line={"color": "black", "width": 1.5},
        )
    )
    width, height, autosize = _layout_size(figsize, default_in=(10.0, 5.0))
    fig.update_layout(
        template=register_template(fontsize),
        title=title or None,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        boxmode="group",
        width=width,
        height=height,
        autosize=autosize and figsize is None,
        legend={"orientation": "h", "y": -0.25},
    )
    return fig
