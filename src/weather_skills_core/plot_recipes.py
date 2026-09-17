"""Matplotlib compilers for multi-panel recipes (compare, verify, timeseries, mediogram)."""

from __future__ import annotations

import numpy as np

from weather_skills_core.figure import add_shared_colorbar, apply_date_ticks
from weather_skills_core.plot_compile import (
    as_plot_x,
    draw_geo_lines,
    figsize_from_extent,
    geojson_lines,
)
from weather_skills_core.plot_mpl import (
    LEGEND_KEYS,
    apply_style_then_rc,
    bar_kwargs,
    box_kwargs,
    colorbar_mpl_kwargs,
    finish_figure,
    line_kwargs,
    mesh_kwargs,
    pick,
    scatter_kwargs,
)
from weather_skills_core.plot_style import (
    DEFAULT_FONTSIZE,
    mpl_cmap_norm,
    mpl_color,
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
        "colors": HITS_COLORS,
        "bounds": [-1.5, -0.5, 0.5, 1.5],
        "tickvals": [-1, 0, 1],
        "ticktext": ["disagree", "below", "hit"],
        "cmin": -1.5,
        "cmax": 1.5,
        "label": label,
    }


def error_scale(da, metric, *, label=None):
    """Bias (diverging, white at 0) or MAE (white→warm) colormap."""
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
            "colors": ERROR_DIVERGING_COLORS,
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
        "colors": MAE_FROM_WHITE_COLORS,
        "cmin": lo,
        "cmax": hi,
        "label": label or "mae",
    }


def heatmap_cell(da, lat_dim, lon_dim, *, scale="field"):
    slab = da.transpose(lat_dim, lon_dim)
    return {
        "kind": "heatmap",
        "x": np.asarray(slab[lon_dim].values, dtype=float),
        "y": np.asarray(slab[lat_dim].values, dtype=float),
        "z": np.asarray(slab.values, dtype=float),
        "scale": scale,
    }


def scatter_cell(lons, lats, values, *, scale="field"):
    return {
        "kind": "scatter",
        "x": np.asarray(lons, dtype=float),
        "y": np.asarray(lats, dtype=float),
        "c": np.asarray(values, dtype=float),
        "scale": scale,
    }


def blank_cell(text="n/a"):
    return {"kind": "blank", "text": text}


def compile_heatmap_grid(
    cells,
    *,
    extent,
    title=None,
    col_titles=None,
    row_titles=None,
    fontsize=DEFAULT_FONTSIZE,
    figsize=None,
    scales=None,
    overlays=True,
    xlabel="Longitude",
    cell_notes=None,
    template="weather_skills",
    spec=None,
):
    """Compile a 2-D grid of heatmap/scatter/blank cells.

    ``cells`` is a list of rows; each row is a list of dicts or None.
    A cell dict is ``{"kind": "heatmap", "x", "y", "z", "scale"}`` or
    ``{"kind": "scatter", "x", "y", "c", "scale"}`` or
    ``{"kind": "blank", "text": "n/a"}``.
    ``scales`` maps scale id → dict from ``resolve_colorscale``.
    """
    import matplotlib.pyplot as plt

    apply_style_then_rc(spec or {}, chart="map", fontsize=fontsize, template=template)
    nrows = len(cells)
    ncols = max((len(row) for row in cells), default=1)
    geo_x = geo_y = None
    if overlays and extent is not None:
        geo_x, geo_y = geojson_lines(extent)
    sw, sh = (5.0, 4.0)
    if extent is not None:
        sw, sh = figsize_from_extent(*extent)
    if figsize is not None:
        fig_w, fig_h = float(figsize[0]), float(figsize[1])
        tight = False
    else:
        fig_w, fig_h = max(sw * ncols, 6.0), max(sh * nrows, 4.0)
        tight = True
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(fig_w, fig_h), squeeze=False, layout="constrained"
    )
    mappables = {}
    for r, row in enumerate(cells):
        for c in range(ncols):
            ax = axes[r, c]
            cell = row[c] if c < len(row) else None
            if cell is None:
                cell = blank_cell()
            kind = cell.get("kind") or "heatmap"
            scale_id = cell.get("scale") or "field"
            scale = scales.get(scale_id) or {}
            cmap, norm = mpl_cmap_norm(scale) if scale else (None, None)
            if extent is not None:
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])
                ax.set_aspect("equal", adjustable="box")
            if r == 0 and col_titles and c < len(col_titles):
                ax.set_title(col_titles[c])
            if r == nrows - 1:
                ax.set_xlabel(xlabel)
            if c == 0 and row_titles and r < len(row_titles):
                ax.set_ylabel(row_titles[r])
            if kind == "blank":
                ax.text(
                    0.5,
                    0.5,
                    cell.get("text") or "n/a",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="#888888",
                )
            elif kind == "scatter":
                sk = scatter_kwargs((spec or {}).get("scatter") or cell.get("scatter") or {})
                mappable = ax.scatter(
                    cell["x"],
                    cell["y"],
                    **{
                        "c": cell.get("c"),
                        "cmap": cmap,
                        "norm": norm,
                        "s": 64,
                        "edgecolors": "#333",
                        "linewidths": 0.4,
                        "zorder": 4,
                        **sk,
                    },
                )
                if scale_id not in mappables:
                    mappables[scale_id] = mappable
                draw_geo_lines(ax, geo_x, geo_y, lw=0.5)
            else:
                mk = mesh_kwargs(spec or {}, cell)
                mappable = ax.pcolormesh(
                    np.asarray(cell["x"], dtype=float),
                    np.asarray(cell["y"], dtype=float),
                    np.asarray(cell["z"], dtype=float),
                    **{"cmap": cmap, "norm": norm, "shading": "nearest", **mk},
                )
                if scale_id not in mappables:
                    mappables[scale_id] = mappable
                draw_geo_lines(ax, geo_x, geo_y, lw=0.5)
            note = None
            if cell_notes and r < len(cell_notes) and c < len(cell_notes[r]):
                note = cell_notes[r][c]
            if note:
                ax.text(
                    0.03,
                    0.97,
                    note,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=max(8, int(fontsize * 0.6)),
                    color="#333333",
                )

    visible = [ax for ax in axes.ravel() if ax.get_visible()]
    n_scales = len(mappables)
    extra_cbar = colorbar_mpl_kwargs(spec or {})
    for scale_id, mappable in mappables.items():
        scale = scales.get(scale_id) or {}
        cbar_kw = dict(extra_cbar)
        location = cbar_kw.pop(
            "location", "right" if n_scales > 1 or len(visible) <= 1 else "bottom"
        )
        cbar = add_shared_colorbar(
            fig,
            mappable,
            visible,
            label=scale.get("label") or "",
            location=location,
            ticks=scale.get("tickvals") or scale.get("bounds"),
            **cbar_kw,
        )
        if cbar is not None and scale.get("ticktext"):
            cbar.set_ticklabels(list(scale["ticktext"]))
    if title:
        fig.suptitle(title)
    fig._ws_tight = tight
    finish_figure(fig, spec or {}, axes)
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
    template="weather_skills",
    spec=None,
):
    """``series`` is a list of ``(x, y, label)``. ``y`` may be 1-D or 2-D (along)."""
    import matplotlib.pyplot as plt

    apply_style_then_rc(spec or {}, chart="line", fontsize=fontsize, template=template)
    n = len(series)
    kinds = kinds or ["line"] * n
    styles = styles or [{} for _ in series]
    ylabels = ylabels or [""] * n
    if figsize is not None:
        fig_w, fig_h = float(figsize[0]), float(figsize[1])
        tight = False
    elif subplots:
        fig_w, fig_h = 10.0, max(2.8 * n, 4.0)
        tight = True
    else:
        fig_w, fig_h = 10.0, 6.0
        tight = True
    if subplots:
        fig, axes = plt.subplots(
            n, 1, sharex=True, figsize=(fig_w, fig_h), squeeze=False, layout="constrained"
        )
    else:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), layout="constrained")
        axes = np.array([[ax]])
    for i, ((xvals, yvals, label), kind, style) in enumerate(
        zip(series, kinds, styles, strict=True)
    ):
        ax = axes[i if subplots else 0, 0]
        plot_ax = ax
        if style.get("twin") in (True, "y", "twinx"):
            plot_ax = ax.twinx()
        color = mpl_color(style.get("color")) or f"C{i % 10}"
        width_pt = style.get("lw") or style.get("linewidth") or style.get("width") or 2
        yarr = np.asarray(yvals, dtype=float)
        xplot = as_plot_x(xvals)
        band = style.get("band")
        markers = style.get("marker")
        use_marker = markers not in (None, "None", "none", "null")
        alpha = float(style.get("alpha") or 1)
        lk = line_kwargs(style.get("line") or {}, loc=f"styles[{i}].line")
        bk = bar_kwargs(style.get("bar") or {}, loc=f"styles[{i}].bar")
        if yarr.ndim == 2 and band:
            lo_q, hi_q = band
            low = np.nanpercentile(yarr, lo_q, axis=1)
            high = np.nanpercentile(yarr, hi_q, axis=1)
            mean = np.nanmean(yarr, axis=1)
            plot_ax.fill_between(
                xplot,
                low,
                high,
                color=color,
                alpha=float(style.get("band_alpha") or 0.25),
                linewidth=0,
                zorder=float(style.get("zorder") or 1),
                label="_nolegend_",
            )
            plot_ax.plot(
                xplot,
                mean,
                **{
                    "color": color,
                    "linewidth": width_pt,
                    "marker": "o" if use_marker else None,
                    "markersize": float(style.get("markersize") or 6),
                    "alpha": alpha,
                    "label": label,
                    "zorder": float(style.get("zorder") or 2) + 1,
                    **lk,
                },
            )
        else:
            traces_y = [yarr] if yarr.ndim == 1 else [yarr[:, j] for j in range(yarr.shape[1])]
            member_alpha = alpha if yarr.ndim == 1 else float(style.get("alpha") or 0.35)
            for j, yy in enumerate(traces_y):
                name = label if j == 0 else "_nolegend_"
                if kind == "bar":
                    plot_ax.bar(
                        np.arange(len(yy)) if np.asarray(xplot).dtype.kind == "O" else xplot,
                        yy,
                        **{"color": color, "label": name, "alpha": alpha, **bk},
                    )
                else:
                    plot_ax.plot(
                        xplot,
                        yy,
                        **{
                            "color": color,
                            "linewidth": (
                                width_pt if yarr.ndim == 1 else min(float(width_pt), 1.2)
                            ),
                            "marker": "o" if use_marker else None,
                            "markersize": float(style.get("markersize") or 6),
                            "alpha": member_alpha if yarr.ndim == 2 else alpha,
                            "label": (
                                name
                                if (j == 0 and not subplots)
                                else (name if j == 0 else "_nolegend_")
                            ),
                            "zorder": float(style.get("zorder") or 2),
                            **lk,
                        },
                    )
        if np.asarray(xvals).dtype.kind == "M":
            apply_date_ticks(ax)
        if subplots:
            ax.set_ylabel(ylabels[i])
            if i == n - 1:
                ax.set_xlabel(xlabel)
        elif i == 0:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabels[0] if ylabels else "")
    if not subplots:
        axes[0, 0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=min(n, 4))
    if title:
        fig.suptitle(title)
    fig._ws_tight = tight
    finish_figure(fig, spec or {}, axes)
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
    template="weather_skills",
    spec=None,
):
    """ECMWF-style two-layer boxes: forecast (cyan) vs m-climate (red)."""
    import matplotlib.pyplot as plt

    apply_style_then_rc(spec or {}, chart="line", fontsize=fontsize, template=template)
    opts = (spec or {}).get("mediogram") or {}
    n_steps = fc.shape[1]
    if figsize is not None:
        fig_w, fig_h = float(figsize[0]), float(figsize[1])
        tight = False
    else:
        fig_w, fig_h = 10.0, 5.0
        tight = True
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), layout="constrained")
    width = float(opts.get("width", 0.35))
    positions = np.arange(n_steps)
    bp_fc = ax.boxplot(
        [np.asarray(fc[:, i], dtype=float) for i in range(n_steps)],
        positions=positions - width / 2,
        widths=width * 0.9,
        patch_artist=True,
        manage_ticks=False,
    )
    bp_mc = ax.boxplot(
        [np.asarray(mc[:, i], dtype=float) for i in range(n_steps)],
        positions=positions + width / 2,
        widths=width * 0.9,
        patch_artist=True,
        manage_ticks=False,
    )
    fc_style = {"facecolor": "cyan", "edgecolor": "black", **box_kwargs(opts.get("forecast") or {})}
    mc_style = {"facecolor": "red", "edgecolor": "black", **box_kwargs(opts.get("mclimate") or {})}
    for patch in bp_fc["boxes"]:
        patch.set_facecolor(fc_style.get("facecolor", "cyan"))
        patch.set_edgecolor(fc_style.get("edgecolor", "black"))
        if fc_style.get("linewidth") is not None:
            patch.set_linewidth(fc_style["linewidth"])
        if fc_style.get("alpha") is not None:
            patch.set_alpha(fc_style["alpha"])
        if fc_style.get("linestyle") is not None:
            patch.set_linestyle(fc_style["linestyle"])
        if fc_style.get("hatch") is not None:
            patch.set_hatch(fc_style["hatch"])
        if fc_style.get("zorder") is not None:
            patch.set_zorder(fc_style["zorder"])
    for patch in bp_mc["boxes"]:
        patch.set_facecolor(mc_style.get("facecolor", "red"))
        patch.set_edgecolor(mc_style.get("edgecolor", "black"))
        if mc_style.get("linewidth") is not None:
            patch.set_linewidth(mc_style["linewidth"])
        if mc_style.get("alpha") is not None:
            patch.set_alpha(mc_style["alpha"])
        if mc_style.get("linestyle") is not None:
            patch.set_linestyle(mc_style["linestyle"])
        if mc_style.get("hatch") is not None:
            patch.set_hatch(mc_style["hatch"])
        if mc_style.get("zorder") is not None:
            patch.set_zorder(mc_style["zorder"])
    bp_fc["boxes"][0].set_label("forecast")
    bp_mc["boxes"][0].set_label("m-climate")
    mean_kw = {
        "color": "black",
        "linewidth": 1.5,
        "label": "forecast mean",
        **line_kwargs(opts.get("mean") or {}, loc="mediogram.mean"),
    }
    ax.plot(positions, np.mean(fc, axis=0), **mean_kw)
    ax.set_xticks(positions)
    ax.set_xticklabels(list(tick_labels), rotation=30, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    legend = opts.get("legend")
    if legend in (False, "off", "none"):
        pass
    elif isinstance(legend, dict):
        ax.legend(
            **{
                "loc": "upper center",
                "bbox_to_anchor": (0.5, -0.18),
                "ncol": 3,
                **pick(legend, LEGEND_KEYS, loc="mediogram.legend"),
            }
        )
    else:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    if title:
        fig.suptitle(title)
    fig._ws_tight = tight
    finish_figure(fig, spec or {}, ax)
    return fig
