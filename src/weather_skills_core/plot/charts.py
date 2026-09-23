"""1-D and categorical figures: timeseries, xy, windrose, mediogram."""

from __future__ import annotations

import sys

import numpy as np

from weather_skills_core.cf import auto_variable, cf_dim
from weather_skills_core.errors import UsageError
from weather_skills_core.plot.figure import (
    LEGEND_KEYS,
    apply_date_ticks,
    apply_style_then_rc,
    apply_suptitle,
    axis_label,
    bar_kwargs,
    box_kwargs,
    facet_figure,
    fill_kwargs,
    finish_figure,
    line_kwargs,
    pick,
    resolve_axis_label,
    resolve_figsize,
    settle_figure,
    windrose_kwargs,
    wrap_axes_title,
)
from weather_skills_core.plot.maps import (
    _SAMPLE_DIM_NAMES,
    WIND_ROSE_SECTORS,
    WIND_SPEED_COLORS,
    WIND_SPEED_EDGES_MS,
    _parse_colormap,
    _resolve_uv,
    _speed_units_display,
    _subset_points,
    _variable_label,
    pad_cell_extent,
    parse_extent,
    plain,
    step_dim,
    subset_spatial,
    timeseries_axis,
)
from weather_skills_core.plot.spec import apply_index, parse_index, resolve_bar_mode, trace_at
from weather_skills_core.plot.theme import (
    ALONG_COLOR_CYCLE,
    DEFAULT_FONTSIZE,
    along_dim,
    along_member_label,
    mpl_color,
    parse_along_color,
    parse_band,
)
from weather_skills_core.standard_utils import (
    ensure_normalized_longitude,
    parse_bbox,
    polygon_from_geojson,
)
from weather_skills_core.units import (
    precip_for_display,
    to_standard_units,
    units_equal,
    variable_label_for_display,
    variable_units,
)


def _bar_x_numeric(xplot):
    """Numeric x for ``ax.bar``: matplotlib dates, else float, else 0..n-1."""
    arr = np.asarray(xplot)
    if arr.dtype.kind == "M":
        import matplotlib.dates as mdates

        return np.asarray(mdates.date2num(arr), dtype=float)
    if arr.dtype.kind == "O":
        return np.arange(len(arr), dtype=float)
    return np.asarray(arr, dtype=float)


def _bar_unit_width(xnum):
    xnum = np.asarray(xnum, dtype=float)
    if xnum.size < 2:
        return 1.0
    diffs = np.diff(np.sort(xnum))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    return float(np.median(diffs)) if diffs.size else 1.0


def _relative_bar_width(data_width, x):
    """Seaborn ``barplot`` width is a fraction of the native x spacing."""
    unit = _bar_unit_width(x)
    if unit <= 0:
        return 0.8
    return float(data_width) / unit


def _sns_line(ax, x, y, **kwargs):
    """One line via ``sns.lineplot``, without aggregation or a confidence band."""
    import seaborn as sns

    sns.lineplot(
        x=np.asarray(x),
        y=np.asarray(y, dtype=float),
        ax=ax,
        estimator=None,
        errorbar=None,
        sort=False,
        legend=False,
        **kwargs,
    )


def _sns_scatter(ax, x, y, **kwargs):
    import seaborn as sns

    sns.scatterplot(
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        ax=ax,
        legend=False,
        **kwargs,
    )


def _sns_bars(ax, x, y, *, bottom=None, **kwargs):
    """Bars via seaborn. Stacked bars use ``seaborn.objects.Bar`` baselines."""
    import seaborn as sns
    from matplotlib.patches import Rectangle

    label = kwargs.pop("label", None)
    alpha = kwargs.pop("alpha", None)
    color = kwargs.pop("color", None)
    width = float(kwargs.pop("width", 0.8))
    zorder = kwargs.pop("zorder", None)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_before = len(ax.patches)
    if bottom is None:
        sns.barplot(
            x=x,
            y=y,
            ax=ax,
            color=color,
            width=_relative_bar_width(width, x),
            native_scale=True,
            errorbar=None,
            saturation=1,
            legend=False,
            **kwargs,
        )
    else:
        import pandas as pd
        import seaborn.objects as so

        base = np.asarray(bottom, dtype=float)
        frame = pd.DataFrame(
            {
                "x": x,
                "y": base + np.nan_to_num(y, nan=0.0),
                "base": base,
            }
        )
        mark = {"width": width}
        if color is not None:
            mark["color"] = color
        if alpha is not None:
            mark["alpha"] = alpha
        so.Plot(frame, x="x", y="y").add(so.Bar(**mark), baseline="base").on(ax).plot()
    new = [p for p in ax.patches[n_before:] if isinstance(p, Rectangle)]
    if label and new:
        new[0].set_label(label)
    if alpha is not None:
        for patch in new:
            patch.set_alpha(alpha)
    if zorder is not None:
        for patch in new:
            patch.set_zorder(zorder)
    for key, value in kwargs.items():
        setter_name = f"set_{key}"
        for patch in new:
            setter = getattr(patch, setter_name, None)
            if setter is not None:
                setter(value)
    return new


def _style_box_patches(patches, style):
    for patch in patches:
        if style.get("facecolor") is not None:
            patch.set_facecolor(style["facecolor"])
        if style.get("edgecolor") is not None:
            patch.set_edgecolor(style["edgecolor"])
        if style.get("linewidth") is not None:
            patch.set_linewidth(style["linewidth"])
        if style.get("alpha") is not None:
            patch.set_alpha(style["alpha"])
        if style.get("linestyle") is not None:
            patch.set_linestyle(style["linestyle"])
        if style.get("hatch") is not None:
            patch.set_hatch(style["hatch"])
        if style.get("zorder") is not None:
            patch.set_zorder(style["zorder"])


def _sns_boxes(ax, samples, positions, *, width, color):
    """One ``sns.boxplot`` series at numeric ``positions`` (already offset)."""
    import pandas as pd
    import seaborn as sns

    rows_x = []
    rows_y = []
    for pos, column in zip(positions, samples, strict=True):
        values = np.asarray(column, dtype=float)
        rows_x.extend([float(pos)] * len(values))
        rows_y.extend(values.tolist())
    n_before = len(ax.patches)
    sns.boxplot(
        data=pd.DataFrame({"x": rows_x, "y": rows_y}),
        x="x",
        y="y",
        ax=ax,
        color=color,
        width=float(width),
        native_scale=True,
        saturation=1,
        legend=False,
    )
    return list(ax.patches[n_before:])


def leftover_dims(da, time_dim, *, along=None, reduce=None):
    """Dims that remain after ``reduce`` / ``along`` besides the time axis."""
    reduce_req = reduce or []
    if isinstance(reduce_req, str):
        reduce_req = [reduce_req]
    extras = [d for d in da.dims if d != time_dim]
    if along:
        extras = [d for d in extras if d != along]
    extras = [d for d in extras if d not in reduce_req]
    return extras


def require_reduced_or_along(da, time_dim, *, along=None, reduce=None):
    """Never silently average leftover dims; require ``reduce`` or ``along``."""
    extras = leftover_dims(da, time_dim, along=along, reduce=reduce)
    if extras:
        hint = extras[0]
        raise UsageError(
            f"variable still has non-time dims {extras}. Pass --reduce "
            f"(traces[].reduce) for each leftover dim, or --along {hint} "
            f"(traces[].along) to draw one line per {hint} value."
        )
    return da


def compile_lines(
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
    apply_style_then_rc(spec or {}, chart="line", fontsize=fontsize, template=template)
    n = len(series)
    kinds = kinds or ["line"] * n
    styles = styles or [{} for _ in series]
    ylabels = ylabels or [""] * n
    if figsize is not None:
        fig_w, fig_h = float(figsize[0]), float(figsize[1])
    elif subplots:
        fig_w, fig_h = 10.0, max(2.8 * n, 4.0)
    else:
        fig_w, fig_h = 10.0, 6.0
    if subplots:
        fig, axes = facet_figure(
            n, 1, figsize=(fig_w, fig_h), sharex=True, despine=True
        )
    else:
        fig, axes = facet_figure(1, 1, figsize=(fig_w, fig_h), despine=True)
    along_modes = [
        parse_along_color(
            style.get("along_color") or (trace_at(spec, i).get("along_color") if spec else None)
        )
        for i, style in enumerate(styles)
    ]
    any_cycle = any(
        mode == ALONG_COLOR_CYCLE and np.asarray(yvals).ndim == 2
        for (_, yvals, _), mode in zip(series, along_modes, strict=True)
    )
    palette_i = 0
    legend_n = n
    bar_mode = resolve_bar_mode(spec)
    bar_slots = {}
    for i, kind in enumerate(kinds):
        if kind == "bar":
            bar_slots.setdefault(i if subplots else 0, []).append(i)
    bar_pos = {
        i: (k, len(idxs)) for idxs in bar_slots.values() for k, i in enumerate(idxs)
    }
    stack_bottom = {}
    for i, ((xvals, yvals, label), kind, style, along_color) in enumerate(
        zip(series, kinds, styles, along_modes, strict=True)
    ):
        ax = axes[i if subplots else 0, 0]
        plot_ax = ax
        if style.get("twin") in (True, "y", "twinx"):
            plot_ax = ax.twinx()
        width_pt = style.get("lw") or style.get("linewidth") or style.get("width") or 2
        yarr = np.asarray(yvals, dtype=float)
        xplot = as_plot_x(xvals)
        band = style.get("band")
        markers = style.get("marker")
        use_marker = markers not in (None, "None", "none", "null")
        alpha = float(style.get("alpha") or 1)
        lk = line_kwargs(style.get("line") or {}, loc=f"styles[{i}].line")
        bk = bar_kwargs(style.get("bar") or {}, loc=f"styles[{i}].bar")
        cycle = along_color == ALONG_COLOR_CYCLE and yarr.ndim == 2
        if cycle and band:
            raise UsageError("along_color cycle cannot be combined with a percentile band.")
        if cycle and (style.get("color") is not None or "color" in lk or "c" in lk):
            raise UsageError(
                "along_color cycle cannot set a single color; omit color to cycle, "
                "or use along_color same."
            )
        if along_color == ALONG_COLOR_CYCLE and yarr.ndim != 2 and style.get("along_color"):
            raise UsageError("along_color requires an along (2-D) series.")
        color = mpl_color(style.get("color"))
        if color is None:
            color = f"C{(palette_i if any_cycle else i) % 10}"
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
            _sns_line(
                plot_ax,
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
            palette_i += 1
        else:
            traces_y = [yarr] if yarr.ndim == 1 else [yarr[:, j] for j in range(yarr.shape[1])]
            member_alpha = alpha if (yarr.ndim == 1 or cycle) else float(style.get("alpha") or 0.35)
            along_labels = style.get("along_labels") or []
            if cycle:
                legend_n += max(0, len(traces_y) - 1)
            for j, yy in enumerate(traces_y):
                if cycle:
                    member = along_labels[j] if j < len(along_labels) else str(j)
                    name = f"{label} {member}" if n > 1 else member
                    member_color = f"C{palette_i % 10}"
                    palette_i += 1
                else:
                    name = label if j == 0 else "_nolegend_"
                    member_color = color
                    if j == 0:
                        palette_i += 1
                if kind == "bar":
                    slot, n_bar = bar_pos[i]
                    xnum = _bar_x_numeric(xplot)
                    unit = _bar_unit_width(xnum)
                    bar_kw = dict(bk)
                    user_width = bar_kw.pop("width", None)
                    ax_i = i if subplots else 0
                    if bar_mode == "stacked":
                        width = float(user_width) if user_width is not None else 0.8 * unit
                        bottom = stack_bottom.get(ax_i)
                        if bottom is None or len(bottom) != len(yy):
                            bottom = np.zeros(len(yy), dtype=float)
                        _sns_bars(
                            plot_ax,
                            xnum,
                            yy,
                            bottom=bottom,
                            **{
                                "color": member_color,
                                "label": name,
                                "alpha": alpha,
                                "width": width,
                                **bar_kw,
                            },
                        )
                        stack_bottom[ax_i] = bottom + np.nan_to_num(
                            np.asarray(yy, dtype=float), nan=0.0
                        )
                    elif bar_mode == "grouped" and n_bar > 1:
                        width = (
                            float(user_width)
                            if user_width is not None
                            else (0.8 * unit) / n_bar
                        )
                        offset = (slot - (n_bar - 1) / 2.0) * width
                        _sns_bars(
                            plot_ax,
                            xnum + offset,
                            yy,
                            **{
                                "color": member_color,
                                "label": name,
                                "alpha": alpha,
                                "width": width,
                                **bar_kw,
                            },
                        )
                    else:
                        width = float(user_width) if user_width is not None else 0.8 * unit
                        _sns_bars(
                            plot_ax,
                            xnum,
                            yy,
                            **{
                                "color": member_color,
                                "label": name,
                                "alpha": alpha,
                                "width": width,
                                **bar_kw,
                            },
                        )
                else:
                    _sns_line(
                        plot_ax,
                        xplot,
                        yy,
                        **{
                            "color": member_color,
                            "linewidth": (
                                width_pt if (yarr.ndim == 1 or cycle) else min(float(width_pt), 1.2)
                            ),
                            "marker": "o" if use_marker else None,
                            "markersize": float(style.get("markersize") or 6),
                            "alpha": member_alpha if yarr.ndim == 2 else alpha,
                            "label": name,
                            "zorder": float(style.get("zorder") or 2),
                            **lk,
                        },
                    )
            if subplots and cycle:
                plot_ax.legend(loc="best")
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
        axes[0, 0].legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=min(max(legend_n, 1), 4)
        )
    apply_suptitle(fig, title, spec)
    finish_figure(fig, spec or {}, axes)
    settle_figure(fig)
    from weather_skills_core.plot.figure import CompiledFigure

    return CompiledFigure(fig, spec or {}, tight=False)


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
    apply_style_then_rc(spec or {}, chart="line", fontsize=fontsize, template=template)
    opts = trace_at(spec).get("mediogram") or {}
    n_steps = fc.shape[1]
    if figsize is not None:
        fig_w, fig_h = float(figsize[0]), float(figsize[1])
    else:
        fig_w, fig_h = 10.0, 5.0
    fig, axes = facet_figure(1, 1, figsize=(fig_w, fig_h), despine=True)
    ax = axes[0, 0]
    width = float(opts.get("width", 0.35))
    positions = np.arange(n_steps)
    fc_style = {"facecolor": "cyan", "edgecolor": "black", **box_kwargs(opts.get("forecast") or {})}
    mc_style = {"facecolor": "red", "edgecolor": "black", **box_kwargs(opts.get("mclimate") or {})}
    box_width = width * 0.9
    fc_patches = _sns_boxes(
        ax,
        [np.asarray(fc[:, i], dtype=float) for i in range(n_steps)],
        positions - width / 2,
        width=box_width,
        color=fc_style.get("facecolor", "cyan"),
    )
    mc_patches = _sns_boxes(
        ax,
        [np.asarray(mc[:, i], dtype=float) for i in range(n_steps)],
        positions + width / 2,
        width=box_width,
        color=mc_style.get("facecolor", "red"),
    )
    _style_box_patches(fc_patches, fc_style)
    _style_box_patches(mc_patches, mc_style)
    if fc_patches:
        fc_patches[0].set_label("forecast")
    if mc_patches:
        mc_patches[0].set_label("m-climate")
    mean_kw = {
        "color": "black",
        "linewidth": 1.5,
        "label": "forecast mean",
        **line_kwargs(opts.get("mean") or {}, loc="mediogram.mean"),
    }
    _sns_line(ax, positions, np.mean(fc, axis=0), **mean_kw)
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
    apply_suptitle(fig, title, spec)
    finish_figure(fig, spec or {}, ax)
    settle_figure(fig)
    from weather_skills_core.plot.figure import CompiledFigure

    return CompiledFigure(fig, spec or {}, tight=False)


def _prepare_field(ds, spec_input: dict, geo: dict, style: str):
    variable = spec_input.get("variable") or auto_variable(ds)
    if not variable or variable not in ds:
        raise UsageError(f"no usable variable. Available: {list(ds.data_vars)}")
    ds = to_standard_units(ds, variables=[variable])
    ds = precip_for_display(ds, variable)
    da = ds[variable]
    overrides = parse_index(spec_input.get("index"))
    bbox_raw = geo.get("bbox")
    if isinstance(bbox_raw, str):
        bbox_nwse = parse_bbox(bbox_raw)
    elif bbox_raw is not None:
        bbox_nwse = tuple(bbox_raw)
    else:
        bbox_nwse = None
    mask_geojson = geo.get("mask_geojson")
    region_polygon = polygon_from_geojson(mask_geojson) if mask_geojson else None
    extent = parse_extent(geo.get("extent"))

    if style == "timeseries":
        da = apply_index(da, overrides, list_dims=())
        sdim = "step" if "step" in da.dims else cf_dim(da, "time")
        if sdim is None:
            raise UsageError(f"timeseries needs 'step' or 'time'; got {list(da.dims)}.")
        along = along_dim(da, spec_input.get("along"))
        reduce_req = spec_input.get("reduce") or []
        if isinstance(reduce_req, str):
            reduce_req = [reduce_req]
        applicable = [d for d in reduce_req if d in da.dims]
        if applicable:
            da = da.mean(applicable, keep_attrs=True)
        if along == sdim:
            raise UsageError(
                f"traces[].along {spec_input.get('along')!r} is the time axis "
                f"({sdim!r}); pass a non-time dim such as number."
            )
        require_reduced_or_along(da, sdim, along=along)
        along_labels = None
        if along:
            da = da.transpose(sdim, along)
            along_labels = [along_member_label(v) for v in da[along].values]
        return {
            "da": plain(da),
            "sdim": sdim,
            "along": along,
            "along_labels": along_labels,
            "align": spec_input.get("align"),
            "band": spec_input.get("band"),
        }

    lat_dim = cf_dim(da, "latitude")
    lon_dim = cf_dim(da, "longitude")
    if lat_dim is None or lon_dim is None:
        raise UsageError(f"{style} requires lat/lon coords; got {list(da.dims)}.")
    if lat_dim not in da.dims or lon_dim not in da.dims:
        raise UsageError(
            f"{style} needs lat/lon as dimensions, but {lat_dim!r}/"
            f"{lon_dim!r} are non-dimension coordinates (dims: {list(da.dims)})"
        )
    native_step_dim = step_dim(da)
    native_steps = list(da[native_step_dim].values) if native_step_dim else None
    list_dims = (native_step_dim,) if native_step_dim else ()
    da = apply_index(da, overrides, list_dims=list_dims)
    panel_dim = step_dim(da)
    for dim in da.dims:
        if dim not in (panel_dim, "number", lat_dim, lon_dim):
            panel_desc = repr(panel_dim) if panel_dim else "step/time"
            raise UsageError(
                f"dimension {dim!r} remains after selection; {style} "
                f"panels only the {panel_desc} dimension — select a position "
                f"from {dim!r} with --index"
            )
    wrapped_bbox = bbox_nwse is not None and bbox_nwse[1] > bbox_nwse[3]
    da, extent = subset_spatial(da, lat_dim, lon_dim, bbox_nwse, region_polygon, extent)
    if da.sizes[lat_dim] == 0 or da.sizes[lon_dim] == 0:
        raise UsageError("selection produced an empty grid; nothing to plot.")
    if "number" in da.dims:
        da = da.mean("number", keep_attrs=True)
    da = plain(da)
    if not wrapped_bbox:
        da = ensure_normalized_longitude(da, lon_dim)
    if extent is None:
        extent = pad_cell_extent(da[lat_dim].values, da[lon_dim].values)
    return {
        "da": da,
        "lat_dim": lat_dim,
        "lon_dim": lon_dim,
        "extent": extent,
        "native_step_dim": native_step_dim,
        "native_steps": native_steps,
        "variable": variable,
    }


def as_plot_x(values):
    """Matplotlib x values: datetime64 stays; timedeltas become days."""
    arr = np.asarray(values)
    if arr.dtype.kind == "m":
        return arr / np.timedelta64(1, "D")
    return arr


def _compile_timeseries(prepared, spec, fontsize, *, template="weather_skills"):
    apply_style_then_rc(spec, chart="line", fontsize=fontsize, template=template)
    da = prepared["da"]
    sdim = prepared["sdim"]
    align = prepared.get("align")
    xvals, default_xlabel = timeseries_axis(da, sdim)
    if align in ("dayofyear", "day_of_year", "day-of-year"):
        try:
            xvals = da[sdim].dt.dayofyear.values
        except (TypeError, AttributeError) as exc:
            raise UsageError("align=dayofyear needs a calendar-date time axis.") from exc
        default_xlabel = "calendar day"
    qty = variable_label_for_display(da, include_units=False)
    xlabel = spec.get("xlabel")
    if xlabel is None:
        xlabel = (
            ""
            if np.asarray(xvals).dtype.kind in "Mm"
            and align
            not in (
                "dayofyear",
                "day_of_year",
                "day-of-year",
            )
            else axis_label(default_xlabel)
        )
    else:
        xlabel = resolve_axis_label(xlabel, default_xlabel)
    ylabel = resolve_axis_label(spec.get("ylabel"), variable_label_for_display(da))
    figsize = spec.get("layout", {}).get("figsize") or (10.0, 5.0)
    fig, axes = facet_figure(1, 1, figsize=tuple(figsize), despine=True)
    ax = axes[0, 0]
    xplot = as_plot_x(xvals)
    yarr = np.asarray(da.values, dtype=float)
    band = parse_band(prepared.get("band"))
    trace0 = trace_at(spec)
    along_color_raw = trace0.get("along_color")
    if along_color_raw and not prepared.get("along"):
        raise UsageError("traces[].along_color requires traces[].along")
    along_color = parse_along_color(along_color_raw)
    if along_color == ALONG_COLOR_CYCLE and band is not None:
        raise UsageError("traces[].along_color cycle cannot be combined with traces[].band")
    lk = line_kwargs(trace0.get("line") or {}, loc="traces[0].line")
    if along_color == ALONG_COLOR_CYCLE and ("color" in lk or "c" in lk):
        raise UsageError(
            "traces[].along_color cycle cannot set traces[].line.color; "
            "omit color to cycle, or use along_color same."
        )
    color = lk.get("color") or lk.get("c") or "C0"
    fill_kw = {"color": color, "alpha": 0.25, "linewidth": 0, "zorder": 1, **fill_kwargs(trace0)}
    cycle = along_color == ALONG_COLOR_CYCLE and yarr.ndim == 2
    along_labels = prepared.get("along_labels") or []
    if yarr.ndim == 2 and band is not None:
        low = np.nanpercentile(yarr, band[0], axis=1)
        high = np.nanpercentile(yarr, band[1], axis=1)
        mean = np.nanmean(yarr, axis=1)
        ax.fill_between(xplot, low, high, **fill_kw)
        _sns_line(ax, xplot, mean, **{"color": color, "linewidth": 2, "label": qty, "zorder": 3, **lk})
    elif yarr.ndim == 2:
        for j in range(yarr.shape[1]):
            member = along_labels[j] if j < len(along_labels) else str(j)
            line_kw = {
                "linewidth": 2.0 if cycle else 1.0,
                "alpha": 1.0 if cycle else 0.35,
                "label": member if cycle else (qty if j == 0 else "_nolegend_"),
                **lk,
            }
            if cycle:
                line_kw["color"] = f"C{j % 10}"
            else:
                line_kw["color"] = color
            _sns_line(ax, xplot, yarr[:, j], **line_kw)
    else:
        _sns_line(
            ax,
            xplot,
            yarr,
            **{
                "marker": "o",
                "markersize": 8,
                "linewidth": 2,
                "color": color,
                "label": qty,
                **lk,
            },
        )
    if np.asarray(xvals).dtype.kind == "M":
        apply_date_ticks(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    legend = spec.get("legend")
    if cycle and legend in (None, "on", "true", "yes"):
        ax.legend()
    elif legend not in (None, "none", "off"):
        loc = legend if isinstance(legend, str) else None
        ax.legend(**({"loc": loc} if loc and loc not in ("on", "true", "yes") else {}))
    apply_suptitle(fig, spec.get("title") or f"{qty} (timeseries)", spec)
    settle_figure(fig)
    from weather_skills_core.plot.figure import CompiledFigure

    return CompiledFigure(fig, spec, tight=False)


def compile_timeseries(spec: dict, datasets: dict, *, fontsize, template="weather_skills"):
    """Single-input timeseries from a plot spec."""

    traces = spec.get("traces") or [{"kind": "timeseries", "input": "a"}]
    trace0 = traces[0]
    inputs = spec.get("inputs") or []
    input_id = trace0.get("input") or "a"
    spec_input = next((i for i in inputs if i.get("id") == input_id), None)
    if spec_input is None:
        spec_input = inputs[0] if inputs else {"id": input_id}
    spec_input = {
        **spec_input,
        **{k: trace0[k] for k in ("along", "reduce", "align", "band") if k in trace0},
    }
    ds = datasets.get(input_id) or datasets.get("a")
    if ds is None and len(datasets) == 1:
        ds = next(iter(datasets.values()))
    if ds is None:
        raise UsageError("plot spec has no Dataset for the requested input")
    prepared = _prepare_field(ds, spec_input, spec.get("geo") or {}, "timeseries")
    return _compile_timeseries(prepared, spec, fontsize, template=template)


def _calendar_year(value) -> int:
    """Calendar year from a datetime-like sample (numpy, cftime, or datetime)."""
    import numpy as np

    if hasattr(value, "year"):
        return int(value.year)
    arr = np.asarray(value)
    if arr.dtype.kind == "M":
        return int(arr.astype("datetime64[Y]").astype(int) + 1970)
    raise UsageError(f"--pair-on year needs datetime samples; got {value!r}")


def _pair_key(value, pair_on: str):
    """Hashable alignment key for one sample along ``--pair-on``."""
    import numpy as np

    if pair_on == "year":
        return _calendar_year(value)
    arr = np.asarray(value)
    if arr.dtype.kind == "M":
        return str(np.datetime64(arr, "D"))
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return value


def _xy_1d(ds, variable, overrides, bbox_nwse, region_polygon, role: str):
    """Reduce one input to a 1D series plus pairing-axis values."""
    import numpy as np

    variable = variable or auto_variable(ds)
    if not variable or variable not in ds:
        raise UsageError(
            f"{role} has no usable variable {variable!r}. Available: {list(ds.data_vars)}"
        )
    try:
        ds = to_standard_units(ds, variables=[variable])
    except UsageError:
        # Totals (mm) or anomalies can still carry a rate/temp standard_name
        # that classify_variable would force into an incompatible target.
        pass
    ds = precip_for_display(ds, variable)
    da = apply_index(plain(ds[variable]), overrides, list_dims=())
    lat_dim = cf_dim(da, "latitude")
    lon_dim = cf_dim(da, "longitude")
    if (bbox_nwse is not None or region_polygon is not None) and lat_dim and lon_dim:
        if lat_dim in da.dims and lon_dim in da.dims:
            da, _ = subset_spatial(da, lat_dim, lon_dim, bbox_nwse, region_polygon, None)
    sdim = "step" if "step" in da.dims else cf_dim(da, "time")
    if sdim is None:
        if da.ndim == 1:
            sdim = da.dims[0]
        else:
            raise UsageError(f"{role} needs a time/step axis to pair samples; got {list(da.dims)}.")
    reduce_dims = [d for d in da.dims if d != sdim]
    reduced = da.mean(reduce_dims, keep_attrs=True) if reduce_dims else da
    axis_vals, _ = timeseries_axis(reduced, sdim)
    values = np.asarray(plain(reduced).values, dtype=float)
    return reduced, np.asarray(axis_vals), values


def _pair_xy(x_axis, x_vals, y_axis, y_vals, pair_on: str):
    """Inner-join two 1D series on time, calendar year, or position."""
    import numpy as np

    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    if pair_on == "index":
        if x_vals.size != y_vals.size:
            raise UsageError(
                f"--pair-on index needs the same number of samples "
                f"(--x has {x_vals.size}, --y has {y_vals.size})."
            )
        keys = list(range(x_vals.size))
        return x_vals, y_vals, keys

    x_keys = [_pair_key(v, pair_on) for v in np.ravel(x_axis)]
    y_keys = [_pair_key(v, pair_on) for v in np.ravel(y_axis)]
    x_map: dict = {}
    for i, key in enumerate(x_keys):
        if key in x_map:
            raise UsageError(
                f"--pair-on {pair_on} has duplicate {key!r} on --x; "
                "aggregate or select so each key appears once."
            )
        x_map[key] = i
    y_map: dict = {}
    for i, key in enumerate(y_keys):
        if key in y_map:
            raise UsageError(
                f"--pair-on {pair_on} has duplicate {key!r} on --y; "
                "aggregate or select so each key appears once."
            )
        y_map[key] = i
    shared = [key for key in x_keys if key in y_map]
    if not shared:
        raise UsageError(f"--pair-on {pair_on} found no matching samples between --x and --y.")
    x_out = np.array([x_vals[x_map[k]] for k in shared], dtype=float)
    y_out = np.array([y_vals[y_map[k]] for k in shared], dtype=float)
    return x_out, y_out, shared


def _plot_xy(
    x_ds,
    y_ds,
    x_variable,
    y_variable,
    pair_on,
    overrides,
    bbox_nwse,
    mask_geojson,
    title,
    xlabel,
    ylabel,
    fontsize,
    figsize=None,
):
    """Scatter --x against --y after reducing each input to 1D and pairing samples."""
    region_polygon = polygon_from_geojson(mask_geojson) if mask_geojson else None
    x_da, x_axis, x_raw = _xy_1d(x_ds, x_variable, overrides, bbox_nwse, region_polygon, "--x")
    y_da, y_axis, y_raw = _xy_1d(y_ds, y_variable, overrides, bbox_nwse, region_polygon, "--y")
    x_vals, y_vals, keys = _pair_xy(x_axis, x_raw, y_axis, y_raw, pair_on)
    finite = np.isfinite(x_vals) & np.isfinite(y_vals)
    x_vals, y_vals = x_vals[finite], y_vals[finite]
    keys = [k for k, keep in zip(keys, finite, strict=True) if keep]
    if x_vals.size == 0:
        raise UsageError("xy scatter has no finite paired samples to plot.")

    fig, axes = facet_figure(
        1, 1, figsize=resolve_figsize(figsize, (8, 6)), despine=True
    )
    ax = axes[0, 0]
    _sns_scatter(ax, x_vals, y_vals, s=36, zorder=3)
    if pair_on == "year" or (pair_on == "time" and x_vals.size <= 25):
        for xv, yv, key in zip(x_vals, y_vals, keys, strict=True):
            ax.annotate(
                str(key),
                (xv, yv),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=max(8, int(round(fontsize * 0.55))),
            )
    ax.set_xlabel(resolve_axis_label(xlabel, _variable_label(x_da)))
    ax.set_ylabel(resolve_axis_label(ylabel, _variable_label(y_da)))
    x_qty = variable_label_for_display(x_da, include_units=False)
    y_qty = variable_label_for_display(y_da, include_units=False)
    ax.set_title(wrap_axes_title(ax, title or f"{y_qty} vs {x_qty}"))
    ax.grid(True, alpha=0.3)
    settle_figure(fig)
    return fig


def _is_sample_dim(da, dim):
    """True if ``dim`` is flattened into wind-rose samples rather than indexed."""
    if dim in _SAMPLE_DIM_NAMES:
        return True
    for cf_name in ("latitude", "longitude", "time"):
        if cf_dim(da, cf_name) == dim:
            return True
    return False


def _flat_numeric(da):
    """Raveled float samples, stripping a pint wrapper if present."""
    import numpy as np

    if getattr(da.pint, "units", None) is not None:
        da = da.pint.dequantify()
    return np.asarray(da.values, dtype=float).reshape(-1)


def _uv_to_speed_fromdir(u, v):
    """Speed and meteorological FROM direction in degrees (0=N, 90=E)."""
    import numpy as np

    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    speed = np.hypot(u, v)
    fromdir = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0
    return speed, fromdir


def _speed_edges(speed, units):
    """Speed-bin edges. Standard 2 m/s classes when units are m/s, else 6 linear bins."""
    import numpy as np

    vmax = float(np.nanmax(speed)) if speed.size else 0.0
    ms = bool(units) and units_equal(units, "m s-1")
    if ms:
        return np.asarray([*WIND_SPEED_EDGES_MS, np.inf], dtype=float)
    if not np.isfinite(vmax) or vmax <= 0:
        return np.array([0.0, 1.0], dtype=float)
    return np.linspace(0.0, vmax, 7)


def _speed_bin_labels(edges):
    import numpy as np

    labels = []
    n = len(edges) - 1
    for i in range(n):
        lo = float(edges[i])
        hi = edges[i + 1]
        if np.isinf(hi):
            labels.append(f"≥{lo:g}")
        else:
            labels.append(f"{lo:g}–{hi:g}")
    return labels


def _speed_colors(n, colormap):
    import numpy as np
    from matplotlib import colormaps
    from matplotlib.colors import LinearSegmentedColormap

    if n < 1:
        return []
    if colormap is None:
        cmap = LinearSegmentedColormap.from_list("windrose", WIND_SPEED_COLORS)
    else:
        parsed = _parse_colormap(colormap)
        cmap = colormaps[parsed] if isinstance(parsed, str) else parsed
    if n == 1:
        return [cmap(0.5)]
    return [cmap(x) for x in np.linspace(0.0, 1.0, n)]


def _wind_rose_hist(speed, direction, speed_edges, nsector=WIND_ROSE_SECTORS):
    """2D histogram ``(nsector, nspeed)``. Sector 0 is North-centered."""
    import numpy as np

    offset = 180.0 / nsector
    shifted = (np.asarray(direction, dtype=float) + offset) % 360.0
    dir_edges = np.linspace(0.0, 360.0, nsector + 1)
    hist, _, _ = np.histogram2d(shifted, speed, bins=[dir_edges, speed_edges])
    return hist


def _windrose(
    speed,
    direction,
    *,
    title,
    fontsize,
    units_disp,
    colormap,
    units,
    figsize=None,
    legend=None,
    ylabel=None,
    mpl_spec=None,
):
    """Polar stacked-bar wind rose; radial axis is frequency percent."""
    from matplotlib.patches import Patch

    wr = windrose_kwargs(trace_at(mpl_spec))
    nsector = int(wr.pop("nsector", WIND_ROSE_SECTORS))
    theta_zero = wr.pop("theta_zero_location", "N")
    theta_dir = wr.pop("theta_direction", -1)
    edgecolor = wr.pop("edgecolor", "white")
    linewidth = wr.pop("linewidth", 0.4)
    zorder = wr.pop("zorder", 2)
    speed_edges = _speed_edges(speed, units)
    hist = _wind_rose_hist(speed, direction, speed_edges, nsector=nsector)
    while hist.shape[1] > 1 and float(hist[:, -1].sum()) == 0:
        hist = hist[:, :-1]
        speed_edges = speed_edges[:-1]
    total = float(hist.sum())
    if total <= 0:
        raise UsageError("windrose has no finite u/v samples to plot.")
    freq = 100.0 * hist / total
    n_speed = freq.shape[1]
    colors = _speed_colors(n_speed, colormap)
    unit_suffix = f" {units_disp}" if units_disp else ""
    legend_labels = [f"{lab}{unit_suffix}" for lab in _speed_bin_labels(speed_edges)]
    width = 2.0 * np.pi / nsector
    theta = np.arange(nsector) * width
    fig, axes = facet_figure(
        1,
        1,
        figsize=resolve_figsize(figsize, (8.5, 7.0)),
        subplot_kws={"projection": "polar"},
        despine=False,
    )
    ax = axes[0, 0]
    ax.set_theta_zero_location(str(theta_zero))
    ax.set_theta_direction(theta_dir)
    bottom = np.zeros(nsector)
    for i in range(n_speed):
        ax.bar(
            theta,
            freq[:, i],
            width=width,
            bottom=bottom,
            color=colors[i],
            edgecolor=edgecolor,
            linewidth=linewidth,
            align="center",
            zorder=zorder,
        )
        bottom += freq[:, i]
    ax.set_thetagrids(
        [0, 45, 90, 135, 180, 225, 270, 315],
        ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
    )
    ax.set_ylim(0, max(float(bottom.max()) * 1.08, 1.0))
    ax.set_ylabel(resolve_axis_label(ylabel, "Frequency (%)"))
    handles = [
        Patch(facecolor=colors[i], edgecolor="white", label=legend_labels[i])
        for i in range(n_speed)
    ]
    _place_legend(ax, legend, default="outside right", handles=handles, title="Wind speed")
    apply_suptitle(fig, title, mpl_spec)
    settle_figure(fig)
    return fig


def _plot_windrose(
    ds,
    u_variable,
    v_variable,
    variable,
    overrides,
    bbox_nwse,
    mask_geojson,
    title,
    fontsize,
    colormap,
    figsize=None,
    legend=None,
    ylabel=None,
    mpl_spec=None,
):
    """Flatten u/v samples into one meteorological-from wind rose."""
    import numpy as np

    if variable:
        print(
            "Warning: --variable is ignored for --kind windrose; "
            "use --u-variable/--v-variable or auto-detection.",
            file=sys.stderr,
        )
    u_name, v_name = _resolve_uv(ds, u_variable, v_variable)
    ds = to_standard_units(ds, variables=[u_name, v_name])
    u_da = ds[u_name]
    v_da = ds[v_name]
    u_da = apply_index(u_da, overrides, list_dims=None)
    v_da = apply_index(v_da, overrides, list_dims=None)
    extra = [d for d in u_da.dims if not _is_sample_dim(u_da, d)]
    if extra:
        raise UsageError(
            f"dimension {extra[0]!r} remains after selection; windrose "
            "flattens space/time/ensemble into samples — select a position "
            f"from {extra[0]!r} with --index"
        )
    region_polygon = polygon_from_geojson(mask_geojson) if mask_geojson else None
    if bbox_nwse is not None or region_polygon is not None:
        lat_dim = cf_dim(u_da, "latitude")
        lon_dim = cf_dim(u_da, "longitude")
        if lat_dim and lon_dim and lat_dim in u_da.dims and lon_dim in u_da.dims:
            u_da, _ = subset_spatial(u_da, lat_dim, lon_dim, bbox_nwse, region_polygon, None)
            v_da, _ = subset_spatial(v_da, lat_dim, lon_dim, bbox_nwse, region_polygon, None)
        else:
            u_da = _subset_points(u_da, bbox_nwse, region_polygon)
            v_da = _subset_points(v_da, bbox_nwse, region_polygon)
    u_vals = _flat_numeric(u_da)
    v_vals = _flat_numeric(v_da)
    if u_vals.size != v_vals.size:
        raise UsageError(
            f"u {u_name!r} and v {v_name!r} have different sizes after selection "
            f"({u_vals.size} vs {v_vals.size}); they must share coordinates"
        )
    valid = np.isfinite(u_vals) & np.isfinite(v_vals)
    u_vals, v_vals = u_vals[valid], v_vals[valid]
    if u_vals.size == 0:
        raise UsageError("windrose has no finite u/v samples to plot.")
    u_units = variable_units(u_da)
    v_units = variable_units(v_da)
    if u_units and v_units and not units_equal(u_units, v_units):
        raise UsageError(f"u units {u_units!r} do not match v units {v_units!r}")
    speed, direction = _uv_to_speed_fromdir(u_vals, v_vals)
    return _windrose(
        speed,
        direction,
        title=title,
        fontsize=fontsize,
        units_disp=_speed_units_display(u_da),
        colormap=colormap,
        units=u_units,
        figsize=figsize,
        legend=legend,
        ylabel=ylabel,
        mpl_spec=mpl_spec,
    )


def _place_legend(ax, loc, *, default="none", **extra):
    """Draw an axes or figure legend. Outside placements stay on-canvas."""
    resolved = default if loc is None else loc
    if resolved in (None, "none", "off"):
        return None
    extra = {"frameon": False, **extra}
    if isinstance(resolved, str) and resolved.startswith("outside"):
        loc_name = "outside right" if resolved in ("outside", "outside right") else resolved
        return ax.figure.legend(loc=loc_name, **extra)
    if resolved == "below":
        return ax.figure.legend(loc="outside lower center", **extra)
    return ax.legend(loc=resolved, **extra)


def compile_xy(spec: dict, datasets: dict, *, fontsize, template="weather_skills"):
    """Scatter two 1-D series from a plot spec."""
    from weather_skills_core.plot.figure import CompiledFigure
    from weather_skills_core.plot.spec import parse_index, trace_at

    apply_style_then_rc(spec, chart="line", fontsize=fontsize, template=template)
    trace = trace_at(spec)
    geo = spec.get("geo") or {}
    inputs = spec.get("inputs") or []
    x_ds = datasets.get("x") or datasets.get("X")
    y_ds = datasets.get("y") or datasets.get("Y")
    if x_ds is None or y_ds is None:
        if len(datasets) >= 2 and x_ds is None:
            items = list(datasets.values())
            x_ds, y_ds = items[0], items[1]
        elif len(datasets) == 1:
            only = next(iter(datasets.values()))
            x_ds = x_ds or only
            y_ds = y_ds or only
    if x_ds is None or y_ds is None:
        raise UsageError("kind xy needs two datasets (or one dataset and x_variable + y_variable)")
    x_variable = trace.get("x_variable") or (inputs[0].get("variable") if inputs else None)
    y_variable = trace.get("y_variable")
    if len(inputs) > 1 and y_variable is None:
        y_variable = inputs[1].get("variable")
    pair_on = trace.get("pair_on") or "time"
    index = inputs[0].get("index") if inputs else None
    overrides = parse_index(index) if index else {}
    bbox = geo.get("bbox")
    if isinstance(bbox, str):
        bbox = parse_bbox(bbox)
    elif bbox is not None:
        bbox = tuple(bbox)
    fig = _plot_xy(
        x_ds,
        y_ds,
        x_variable,
        y_variable,
        pair_on,
        overrides,
        bbox,
        geo.get("mask_geojson"),
        spec.get("title"),
        spec.get("xlabel"),
        spec.get("ylabel"),
        fontsize,
        figsize=(spec.get("layout") or {}).get("figsize"),
    )
    return CompiledFigure(fig, spec, tight=False)


def compile_windrose(spec: dict, datasets: dict, *, fontsize, template="weather_skills"):
    """Wind rose from u/v in a plot spec."""
    from weather_skills_core.plot.figure import CompiledFigure
    from weather_skills_core.plot.spec import parse_index, trace_at

    apply_style_then_rc(spec, chart="line", fontsize=fontsize, template=template)
    trace = trace_at(spec)
    geo = spec.get("geo") or {}
    inputs = spec.get("inputs") or []
    input_id = trace.get("input") or "a"
    ds = datasets.get(input_id) or datasets.get("a")
    if ds is None and len(datasets) == 1:
        ds = next(iter(datasets.values()))
    if ds is None:
        raise UsageError("plot spec has no Dataset for the requested input")
    spec_input = next((i for i in inputs if i.get("id") == input_id), None) or (
        inputs[0] if inputs else {}
    )
    index = spec_input.get("index")
    overrides = parse_index(index) if index else {}
    bbox = geo.get("bbox")
    if isinstance(bbox, str):
        bbox = parse_bbox(bbox)
    elif bbox is not None:
        bbox = tuple(bbox)
    fig = _plot_windrose(
        ds,
        trace.get("u_variable"),
        trace.get("v_variable"),
        spec_input.get("variable"),
        overrides,
        bbox,
        geo.get("mask_geojson"),
        spec.get("title"),
        fontsize,
        (spec.get("theme") or {}).get("colormap"),
        figsize=(spec.get("layout") or {}).get("figsize"),
        legend=spec.get("legend"),
        ylabel=spec.get("ylabel"),
        mpl_spec=spec,
    )
    return CompiledFigure(fig, spec, tight=False)
