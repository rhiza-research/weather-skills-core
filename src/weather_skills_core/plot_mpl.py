"""JSON-safe matplotlib options applied from a plot spec.

Values must be JSON types (str, number, bool, null, list, dict). No Python
callables, no ``eval``. Unknown keys on artist/axes option objects are errors
so agents see a path instead of a silent no-op. ``rc`` keys are matplotlib
rcParam names; backend/interactive keys are rejected.
"""

from __future__ import annotations

from weather_skills_core.errors import UsageError

_JSON_TYPES = (str, int, float, bool, list, dict, type(None))

_RC_FORBIDDEN_PREFIXES = (
    "backend",
    "interactive",
    "tk.",
    "webagg.",
    "nbagg.",
    "macosx.",
    "pdf.use14corefonts",
)

LINE_KEYS = frozenset(
    {
        "alpha",
        "antialiased",
        "aa",
        "color",
        "c",
        "dash_capstyle",
        "dash_joinstyle",
        "dashes",
        "drawstyle",
        "fillstyle",
        "gapcolor",
        "label",
        "linestyle",
        "ls",
        "linewidth",
        "lw",
        "marker",
        "markeredgecolor",
        "mec",
        "markeredgewidth",
        "mew",
        "markerfacecolor",
        "mfc",
        "markerfacecoloralt",
        "mfcalt",
        "markersize",
        "ms",
        "markevery",
        "rasterized",
        "solid_capstyle",
        "solid_joinstyle",
        "visible",
        "zorder",
    }
)

MESH_KEYS = frozenset(
    {
        "alpha",
        "antialiased",
        "edgecolors",
        "hatch",
        "linewidth",
        "linewidths",
        "rasterized",
        "shading",
        "vmin",
        "vmax",
        "zorder",
    }
)

CONTOUR_KEYS = frozenset(
    {
        "alpha",
        "antialiased",
        "colors",
        "extend",
        "hatches",
        "levels",
        "linestyles",
        "linewidths",
        "nchunk",
        "rasterized",
        "vmin",
        "vmax",
        "zorder",
    }
)

SCATTER_KEYS = frozenset(
    {
        "alpha",
        "cmap",
        "edgecolors",
        "linewidths",
        "marker",
        "s",
        "vmin",
        "vmax",
        "zorder",
        "rasterized",
    }
)

BAR_KEYS = frozenset(
    {
        "align",
        "alpha",
        "color",
        "edgecolor",
        "hatch",
        "linewidth",
        "width",
        "zorder",
        "rasterized",
    }
)

QUIVER_KEYS = frozenset(
    {
        "alpha",
        "angles",
        "color",
        "headaxislength",
        "headlength",
        "headwidth",
        "minlength",
        "minshaft",
        "pivot",
        "scale",
        "scale_units",
        "units",
        "width",
        "zorder",
        "rasterized",
    }
)

LEGEND_KEYS = frozenset(
    {
        "alignment",
        "bbox_to_anchor",
        "borderaxespad",
        "columnspacing",
        "draggable",
        "edgecolor",
        "facecolor",
        "fontsize",
        "framealpha",
        "frameon",
        "handlelength",
        "labelcolor",
        "loc",
        "markerscale",
        "ncol",
        "shadow",
        "title",
        "title_fontsize",
    }
)

GRID_KEYS = frozenset(
    {
        "alpha",
        "axis",
        "color",
        "linestyle",
        "linewidth",
        "which",
        "visible",
        "zorder",
    }
)

TICK_KEYS = frozenset(
    {
        "axis",
        "which",
        "direction",
        "length",
        "width",
        "color",
        "pad",
        "labelsize",
        "labelcolor",
        "labelrotation",
        "rotation",
        "bottom",
        "top",
        "left",
        "right",
        "labelbottom",
        "labeltop",
        "labelleft",
        "labelright",
        "grid_alpha",
        "grid_color",
        "grid_linestyle",
        "grid_linewidth",
    }
)

TEXT_KEYS = frozenset(
    {
        "alpha",
        "backgroundcolor",
        "bbox",
        "clip_on",
        "color",
        "fontfamily",
        "fontstyle",
        "fontsize",
        "fontweight",
        "ha",
        "horizontalalignment",
        "ma",
        "rotation",
        "rotation_mode",
        "va",
        "verticalalignment",
        "wrap",
        "zorder",
        "transform",
    }
)

ANNOTATE_KEYS = TEXT_KEYS | frozenset(
    {
        "arrowprops",
        "annotation_clip",
        "textcoords",
        "xycoords",
        "xytext",
    }
)

ARROW_KEYS = frozenset(
    {
        "arrowstyle",
        "connectionstyle",
        "color",
        "ec",
        "edgecolor",
        "fc",
        "facecolor",
        "linestyle",
        "linewidth",
        "lw",
        "mutation_scale",
        "relpos",
        "shrinkA",
        "shrinkB",
        "alpha",
    }
)

COLORBAR_EXTRA_KEYS = frozenset(
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
    }
)

RECT_KEYS = frozenset(
    {
        "alpha",
        "angle",
        "edgecolor",
        "facecolor",
        "fill",
        "hatch",
        "linestyle",
        "linewidth",
        "zorder",
    }
)

WINDROSE_KEYS = frozenset(
    {
        "edgecolor",
        "linewidth",
        "nsector",
        "theta_direction",
        "theta_zero_location",
        "zorder",
    }
)

FILL_KEYS = frozenset(
    {
        "alpha",
        "color",
        "edgecolor",
        "facecolor",
        "hatch",
        "interpolate",
        "linestyle",
        "linewidth",
        "step",
        "zorder",
    }
)

BOX_KEYS = frozenset(
    {
        "alpha",
        "edgecolor",
        "facecolor",
        "hatch",
        "linestyle",
        "linewidth",
        "zorder",
    }
)

_ANNOTATION_META = frozenset(
    {"text", "s", "x", "y", "xy", "xref", "axes", "panel", "transform", "showarrow"}
)

# Object-form axes.xlabel / axes.ylabel: text plus position knobs.
AXIS_LABEL_META = frozenset({"text", "loc", "pad", "coords"})
AXIS_LABEL_KEYS = AXIS_LABEL_META | TEXT_KEYS

# Dumped on every resolved spec so agents can see the axis knobs.
AXES_TEMPLATE = {
    "xscale": None,
    "yscale": None,
    "xlim": None,
    "ylim": None,
    "xlabel": None,
    "ylabel": None,
    "title": None,
    "aspect": None,
    "facecolor": None,
    "grid": None,
    "spines": None,
    "xticks": None,
    "yticks": None,
    "xticklabels": None,
    "yticklabels": None,
    "tick_params": None,
    "xlocator": None,
    "ylocator": None,
    "xformatter": None,
    "yformatter": None,
    "legend": None,
    "twinx": None,
    "twiny": None,
}


def assert_json_value(value, loc: str) -> None:
    """Reject callables and other non-JSON types."""
    if not isinstance(value, _JSON_TYPES):
        raise UsageError(
            f"{loc} must be JSON (str/number/bool/list/object/null); got {type(value).__name__}"
        )
    if isinstance(value, list):
        for i, item in enumerate(value):
            assert_json_value(item, f"{loc}[{i}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            assert_json_value(item, f"{loc}.{key}")


def pick(options: dict | None, allowed: frozenset, *, loc: str) -> dict:
    """Return ``options`` keys in ``allowed``; error on anything else."""
    if not options:
        return {}
    if not isinstance(options, dict):
        raise UsageError(f"{loc} must be an object")
    unknown = [k for k in options if k not in allowed]
    if unknown:
        raise UsageError(
            f"{loc} has unknown key(s) {unknown}; allowed: {', '.join(sorted(allowed))}"
        )
    out = {}
    for key, value in options.items():
        assert_json_value(value, f"{loc}.{key}")
        out[key] = _tuples(value)
    return out


def _tuples(value):
    """Turn JSON lists of numbers into tuples where matplotlib wants xy pairs."""
    if isinstance(value, list) and value and all(isinstance(v, (int, float)) for v in value):
        if len(value) in (2, 4):
            return tuple(value)
        return value
    if isinstance(value, list) and value and all(isinstance(v, list) for v in value):
        return [tuple(v) if len(v) == 2 else v for v in value]
    return value


def apply_rc(rc: dict | None) -> None:
    """Apply ``spec.rc`` / ``style.rc`` as matplotlib rcParams."""
    if not rc:
        return
    if not isinstance(rc, dict):
        raise UsageError("rc must be an object of matplotlib rcParam names")
    import matplotlib as mpl

    for key, value in rc.items():
        name = str(key)
        lower = name.lower()
        if any(lower.startswith(p) or lower == p.rstrip(".") for p in _RC_FORBIDDEN_PREFIXES):
            raise UsageError(f"rc key {name!r} is not allowed (backend/interactive)")
        assert_json_value(value, f"rc.{name}")
        try:
            mpl.rcParams[name] = value
        except KeyError as exc:
            raise UsageError(f"unknown matplotlib rcParam {name!r}") from exc
        except ValueError as exc:
            raise UsageError(f"invalid rcParam {name!r}: {exc}") from exc


def _pair(value, loc: str):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    raise UsageError(f"{loc} must be [min, max]")


def _apply_locator(axis, spec) -> None:
    if spec is None:
        return
    from matplotlib.ticker import AutoLocator, LogLocator, MaxNLocator, MultipleLocator, NullLocator

    if isinstance(spec, dict):
        kind = str(spec.get("type") or spec.get("name") or "auto").lower()
        extra = {k: v for k, v in spec.items() if k not in ("type", "name")}
    else:
        kind = str(spec).lower()
        extra = {}
    if kind in ("auto", "autoLocator"):
        axis.set_major_locator(AutoLocator())
    elif kind in ("log", "loglocator"):
        axis.set_major_locator(
            LogLocator(**{k: extra[k] for k in extra if k in {"base", "numticks", "subs"}})
        )
    elif kind in ("maxn", "maxlocator", "maxN"):
        nbins = extra.get("nbins", extra.get("n", 7))
        axis.set_major_locator(MaxNLocator(nbins=int(nbins)))
    elif kind in ("null", "nulllocator", "none"):
        axis.set_major_locator(NullLocator())
    elif kind in ("multiple", "multiplelocator"):
        base = extra.get("base", extra.get("step", 1))
        axis.set_major_locator(MultipleLocator(float(base)))
    else:
        raise UsageError(f"unknown locator {kind!r}; use auto, log, maxn, null, or multiple")


def _apply_formatter(axis, spec) -> None:
    if spec is None:
        return
    from matplotlib.ticker import FuncFormatter, LogFormatter, PercentFormatter, ScalarFormatter

    if isinstance(spec, dict):
        kind = str(spec.get("type") or spec.get("name") or "scalar").lower()
        extra = {k: v for k, v in spec.items() if k not in ("type", "name")}
    else:
        kind = str(spec).lower()
        extra = {}
    if kind in ("scalar", "scalarformatter"):
        axis.set_major_formatter(ScalarFormatter())
    elif kind in ("log", "logformatter"):
        axis.set_major_formatter(LogFormatter())
    elif kind in ("percent", "percentformatter"):
        xmax = float(extra.get("xmax", 100))
        axis.set_major_formatter(PercentFormatter(xmax=xmax))
    elif kind in ("date", "dateformatter"):
        from weather_skills_core.figure import apply_date_ticks

        if getattr(axis, "axis_name", "x") != "x":
            raise UsageError("date formatter is only supported on the x axis")
        apply_date_ticks(axis.axes)
    elif kind in ("format", "strformat") and extra.get("fmt"):
        fmt = str(extra["fmt"])
        axis.set_major_formatter(FuncFormatter(lambda x, _p, f=fmt: f.format(x)))
    elif kind in ("dayofyear", "doy", "calendar_day"):
        import datetime as dt

        from weather_skills_core.figure import format_plot_date

        def _doy(x, _p):
            day = int(round(float(x)))
            if day < 1 or day > 366:
                return ""
            if day == 366:
                return format_plot_date(dt.date(2023, 12, 31), year=False)
            return format_plot_date(dt.date(2023, 1, 1) + dt.timedelta(days=day - 1), year=False)

        axis.set_major_formatter(FuncFormatter(_doy))
    else:
        raise UsageError(
            f"unknown formatter {kind!r}; use scalar, log, percent, date, format, or dayofyear"
        )


def _tick_locs_and_labels(raw, loc: str):
    """Parse ``xticks``/``yticks``: a list of values, or ``{values, labels, minor}``."""
    if isinstance(raw, dict):
        values = raw.get("values")
        if values is None:
            values = raw.get("ticks")
        if values is None:
            values = raw.get("locs")
        labels = raw.get("labels") if raw.get("labels") is not None else raw.get("ticklabels")
        minor = bool(raw.get("minor"))
    elif isinstance(raw, list):
        values, labels, minor = raw, None, False
    else:
        raise UsageError(f"{loc} must be a list of values or {{values, labels}}")
    if values is not None and not isinstance(values, list):
        raise UsageError(f"{loc}.values must be a list")
    if labels is not None and not isinstance(labels, list):
        raise UsageError(f"{loc}.labels must be a list")
    return values, labels, minor


def _apply_tick_values(ax, which: str, raw, loc: str) -> None:
    values, labels, minor = _tick_locs_and_labels(raw, loc)
    setter = ax.set_xticks if which == "x" else ax.set_yticks
    label_setter = ax.set_xticklabels if which == "x" else ax.set_yticklabels
    if values is not None:
        setter(values, minor=minor)
    if labels is not None:
        label_setter(labels, minor=minor)


def _apply_axis_label(ax, which: str, raw, loc: str) -> None:
    """Apply ``axes.xlabel`` / ``axes.ylabel`` as a string or ``{text, loc, pad, coords, …}``.

    Omit ``text`` to keep the already-drawn label and only change position or
    font (the windrose ``Frequency (%)`` case). ``coords`` is ``[x, y]`` in
    axes fraction.
    """
    axis = ax.xaxis if which == "x" else ax.yaxis
    setter = ax.set_xlabel if which == "x" else ax.set_ylabel
    if isinstance(raw, str):
        setter(raw)
        return
    if not isinstance(raw, dict):
        raise UsageError(f"{loc} must be a string or object")
    opts = pick(raw, AXIS_LABEL_KEYS, loc=loc)
    text = opts.pop("text", None)
    pad = opts.pop("pad", None)
    label_loc = opts.pop("loc", None)
    coords = opts.pop("coords", None)
    text_kw = {k: v for k, v in opts.items() if k in TEXT_KEYS}
    if text is not None or pad is not None or label_loc is not None or text_kw:
        kw = dict(text_kw)
        if pad is not None:
            kw["labelpad"] = pad
        if label_loc is not None:
            kw["loc"] = str(label_loc)
        if text is None:
            text = axis.get_label().get_text()
        setter(text, **kw)
    if coords is not None:
        if not isinstance(coords, (list, tuple)) or len(coords) != 2:
            raise UsageError(f"{loc}.coords must be [x, y] in axes fraction")
        try:
            x, y = float(coords[0]), float(coords[1])
        except (TypeError, ValueError) as exc:
            raise UsageError(f"{loc}.coords must be [x, y] in axes fraction") from exc
        axis.set_label_coords(x, y)


def apply_axes(ax, opts: dict | None, *, skip_legend: bool = False) -> None:
    """Apply ``spec.axes`` to one matplotlib Axes."""
    if not opts:
        return
    if not isinstance(opts, dict):
        raise UsageError("axes must be an object (or a list of objects, one per panel)")
    if opts.get("xscale"):
        ax.set_xscale(str(opts["xscale"]))
    if opts.get("yscale"):
        ax.set_yscale(str(opts["yscale"]))
    if opts.get("xlabel") is not None:
        _apply_axis_label(ax, "x", opts["xlabel"], "axes.xlabel")
    if opts.get("ylabel") is not None:
        _apply_axis_label(ax, "y", opts["ylabel"], "axes.ylabel")
    if opts.get("title") is not None:
        ax.set_title(opts["title"])
    if opts.get("xlim") is not None:
        ax.set_xlim(*_pair(opts["xlim"], "axes.xlim"))
    if opts.get("ylim") is not None:
        ax.set_ylim(*_pair(opts["ylim"], "axes.ylim"))
    if opts.get("aspect") is not None:
        ax.set_aspect(opts["aspect"])
    if opts.get("facecolor") is not None:
        ax.set_facecolor(opts["facecolor"])
    grid = opts.get("grid")
    if grid is False:
        ax.grid(False)
    elif grid is True:
        ax.grid(True)
    elif isinstance(grid, dict):
        ax.grid(**pick(grid, GRID_KEYS, loc="axes.grid"))
    spines = opts.get("spines")
    if isinstance(spines, dict):
        for name, val in spines.items():
            if name not in ax.spines:
                raise UsageError(f"unknown spine {name!r}; this axes has {list(ax.spines)}")
            if val is False:
                ax.spines[name].set_visible(False)
            elif val is True:
                ax.spines[name].set_visible(True)
            elif isinstance(val, dict):
                if "visible" in val:
                    ax.spines[name].set_visible(bool(val["visible"]))
                if "color" in val:
                    ax.spines[name].set_color(val["color"])
                if "linewidth" in val:
                    ax.spines[name].set_linewidth(val["linewidth"])
            else:
                raise UsageError(f"axes.spines.{name} must be bool or object")
    ticks = opts.get("tick_params")
    if isinstance(ticks, dict):
        kw = pick(ticks, TICK_KEYS, loc="axes.tick_params")
        if "rotation" in kw and "labelrotation" not in kw:
            kw["labelrotation"] = kw.pop("rotation")
        ax.tick_params(**kw)
    locator = opts.get("locator") if isinstance(opts.get("locator"), dict) else {}
    _apply_locator(ax.xaxis, opts.get("xlocator") or locator.get("x"))
    _apply_locator(ax.yaxis, opts.get("ylocator") or locator.get("y"))
    formatter = opts.get("formatter") if isinstance(opts.get("formatter"), dict) else {}
    _apply_formatter(ax.xaxis, opts.get("xformatter") or formatter.get("x"))
    _apply_formatter(ax.yaxis, opts.get("yformatter") or formatter.get("y"))
    if opts.get("xticks") is not None:
        _apply_tick_values(ax, "x", opts["xticks"], "axes.xticks")
    if opts.get("yticks") is not None:
        _apply_tick_values(ax, "y", opts["yticks"], "axes.yticks")
    if opts.get("xticklabels") is not None:
        ax.set_xticklabels(opts["xticklabels"])
    if opts.get("yticklabels") is not None:
        ax.set_yticklabels(opts["yticklabels"])
    for twin_key, factory in (("twinx", ax.twinx), ("twiny", ax.twiny)):
        twin = opts.get(twin_key)
        if not twin:
            continue
        tax = factory()
        if isinstance(twin, dict):
            nested = {k: v for k, v in twin.items() if k not in ("twinx", "twiny")}
            apply_axes(tax, nested, skip_legend=skip_legend)
    if skip_legend:
        return
    legend = opts.get("legend")
    if legend in (False, "off", "none"):
        handle = ax.get_legend()
        if handle is not None:
            handle.remove()
    elif legend is True:
        ax.legend()
    elif isinstance(legend, dict):
        ax.legend(**pick(legend, LEGEND_KEYS, loc="axes.legend"))


def _visible_axes(fig, axes=None):
    import numpy as np

    if axes is None:
        axes = fig.axes
    if hasattr(axes, "ravel"):
        axes = list(np.ravel(axes))
    elif not isinstance(axes, (list, tuple)):
        axes = [axes]
    return [
        ax
        for ax in axes
        if getattr(ax, "get_visible", lambda: True)() and ax.get_label() != "<colorbar>"
    ]


def apply_annotation(ax, ann: dict, loc: str = "annotations") -> None:
    """``ax.annotate`` when arrows/xytext are set, else ``ax.text``."""
    if not isinstance(ann, dict):
        raise UsageError(f"{loc} item must be an object")
    text = ann.get("text") if ann.get("text") is not None else ann.get("s")
    if text is None or str(text) == "":
        return
    if "xy" in ann:
        xy = tuple(ann["xy"])
        if len(xy) != 2:
            raise UsageError(f"{loc}.xy must be [x, y]")
    else:
        xy = (
            float(ann["x"]) if ann.get("x") is not None else 0.0,
            float(ann["y"]) if ann.get("y") is not None else 0.0,
        )
    xref = str(ann.get("xref") or ann.get("xycoords") or "")
    kw = {}
    for key in ANNOTATE_KEYS:
        if key in ann:
            kw[key] = ann[key]
    if "arrowprops" in kw:
        if not isinstance(kw["arrowprops"], dict):
            raise UsageError(f"{loc}.arrowprops must be an object")
        kw["arrowprops"] = pick(kw["arrowprops"], ARROW_KEYS, loc=f"{loc}.arrowprops")
    if "xytext" in kw:
        kw["xytext"] = tuple(kw["xytext"]) if isinstance(kw["xytext"], list) else kw["xytext"]
    if "bbox" in kw and isinstance(kw["bbox"], dict):
        kw["bbox"] = dict(kw["bbox"])
    transform_name = str(ann.get("transform") or "")
    if "domain" in xref or xref in ("paper", "figure") or transform_name in ("axes", "figure"):
        kw["transform"] = (
            ax.transAxes
            if transform_name != "figure" and xref != "figure"
            else ax.figure.transFigure
        )
        kw.pop("xycoords", None)
    extra = {k: v for k, v in ann.items() if k not in ANNOTATE_KEYS | _ANNOTATION_META}
    if extra:
        raise UsageError(f"{loc} has unknown key(s) {sorted(extra)}")
    if "arrowprops" in kw or "xytext" in kw:
        annotate_kw = dict(kw)
        if "xycoords" in annotate_kw:
            annotate_kw.pop("transform", None)
        ax.annotate(str(text), xy=xy, **annotate_kw)
    else:
        text_kw = {k: v for k, v in kw.items() if k in TEXT_KEYS}
        ax.text(xy[0], xy[1], str(text), **text_kw)


def apply_shape(ax, shape: dict, loc: str = "shapes") -> None:
    """Add a rect, hline/vline, span, line, or circle from JSON."""
    if not isinstance(shape, dict):
        raise UsageError(f"{loc} item must be an object")
    kind = str(shape.get("type") or shape.get("kind") or "rect").lower()
    if kind in ("rect", "rectangle", "box"):
        from matplotlib.patches import Rectangle

        if "x0" in shape:
            x0, x1 = float(shape["x0"]), float(shape["x1"])
            y0, y1 = float(shape["y0"]), float(shape["y1"])
            xy, w, h = (min(x0, x1), min(y0, y1)), abs(x1 - x0), abs(y1 - y0)
        else:
            xy = tuple(shape.get("xy") or (0, 0))
            w = float(shape.get("width", 0))
            h = float(shape.get("height", 0))
        style = pick(
            {k: v for k, v in shape.items() if k in RECT_KEYS},
            RECT_KEYS,
            loc=loc,
        )
        style.setdefault("fill", False)
        style.setdefault("edgecolor", "black")
        style.setdefault("linewidth", 1.5)
        style.setdefault("zorder", 6)
        ax.add_patch(Rectangle(xy, w, h, **style))
        return
    if kind in ("hline", "axhline"):
        ax.axhline(
            float(shape.get("y", 0)),
            **pick({k: v for k, v in shape.items() if k in LINE_KEYS}, LINE_KEYS, loc=loc),
        )
        return
    if kind in ("vline", "axvline"):
        ax.axvline(
            float(shape.get("x", 0)),
            **pick({k: v for k, v in shape.items() if k in LINE_KEYS}, LINE_KEYS, loc=loc),
        )
        return
    if kind in ("hspan", "axhspan"):
        ax.axhspan(
            float(shape["ymin"]),
            float(shape["ymax"]),
            **pick(
                {
                    k: v
                    for k, v in shape.items()
                    if k in RECT_KEYS | {"alpha", "zorder", "color", "facecolor"}
                },
                RECT_KEYS | {"alpha", "zorder", "color", "facecolor"},
                loc=loc,
            ),
        )
        return
    if kind in ("vspan", "axvspan"):
        ax.axvspan(
            float(shape["xmin"]),
            float(shape["xmax"]),
            **pick(
                {
                    k: v
                    for k, v in shape.items()
                    if k in RECT_KEYS | {"alpha", "zorder", "color", "facecolor"}
                },
                RECT_KEYS | {"alpha", "zorder", "color", "facecolor"},
                loc=loc,
            ),
        )
        return
    if kind in ("line", "segment"):
        x = shape.get("x") or [shape.get("x0"), shape.get("x1")]
        y = shape.get("y") or [shape.get("y0"), shape.get("y1")]
        ax.plot(
            x, y, **pick({k: v for k, v in shape.items() if k in LINE_KEYS}, LINE_KEYS, loc=loc)
        )
        return
    if kind in ("circle", "ellipse"):
        from matplotlib.patches import Circle, Ellipse

        xy = tuple(shape.get("xy") or (float(shape.get("x", 0)), float(shape.get("y", 0))))
        style = pick({k: v for k, v in shape.items() if k in RECT_KEYS}, RECT_KEYS, loc=loc)
        if kind == "ellipse" or "width" in shape:
            ax.add_patch(
                Ellipse(
                    xy,
                    float(shape.get("width", 1)),
                    float(shape.get("height", shape.get("width", 1))),
                    **style,
                )
            )
        else:
            ax.add_patch(Circle(xy, float(shape.get("radius", 1)), **style))
        return
    raise UsageError(
        f"{loc}.type {kind!r} is unknown; use rect, hline, vline, hspan, vspan, line, or circle"
    )


def apply_annotations_and_shapes(fig, spec: dict, axes=None) -> None:
    visible = _visible_axes(fig, axes)
    if not visible:
        return
    layout = spec.get("layout") or {}
    patch = spec.get("patch") or {}
    anns = list(
        spec.get("annotations") or layout.get("annotations") or patch.get("annotations") or []
    )
    if isinstance(patch.get("layout"), dict):
        anns = anns or list(patch["layout"].get("annotations") or [])
    shapes = list(spec.get("shapes") or layout.get("shapes") or patch.get("shapes") or [])
    if isinstance(patch.get("layout"), dict):
        shapes = shapes or list(patch["layout"].get("shapes") or [])
    for i, ann in enumerate(anns):
        if not isinstance(ann, dict):
            continue
        idx = int(ann.get("axes") if ann.get("axes") is not None else ann.get("panel") or 0)
        if idx < 0 or idx >= len(visible):
            raise UsageError(
                f"annotations[{i}] axes index {idx} is out of range (0..{len(visible) - 1})"
            )
        apply_annotation(visible[idx], ann, loc=f"annotations[{i}]")
    for i, shape in enumerate(shapes):
        if not isinstance(shape, dict):
            continue
        idx = int(shape.get("axes") if shape.get("axes") is not None else shape.get("panel") or 0)
        if idx < 0 or idx >= len(visible):
            raise UsageError(
                f"shapes[{i}] axes index {idx} is out of range (0..{len(visible) - 1})"
            )
        apply_shape(visible[idx], shape, loc=f"shapes[{i}]")


def apply_axes_from_spec(fig, spec: dict, axes=None) -> None:
    opts = spec.get("axes")
    if opts is None:
        opts = (spec.get("layout") or {}).get("axes")
    if opts is None:
        return
    visible = _visible_axes(fig, axes)
    if isinstance(opts, list):
        for ax, one in zip(visible, opts, strict=False):
            apply_axes(ax, one)
        return
    for ax in visible:
        apply_axes(ax, opts)


def colorbar_mpl_kwargs(spec: dict | None) -> dict:
    """Extra matplotlib colorbar kwargs (extend, pad, orientation, …)."""
    from weather_skills_core.figure import colorbar_size_kwargs, colorbar_spec

    cbar = colorbar_spec(spec) or {}
    kw = colorbar_size_kwargs(colorbar=cbar)
    extra = pick(
        {k: v for k, v in cbar.items() if k in COLORBAR_EXTRA_KEYS},
        COLORBAR_EXTRA_KEYS,
        loc="layout.colorbar",
    )
    kw.update(extra)
    return kw


def line_kwargs(style: dict | None, *, loc: str = "line") -> dict:
    if not style:
        return {}
    raw = style.get("line") if isinstance(style.get("line"), dict) else style
    return pick({k: v for k, v in raw.items() if k in LINE_KEYS}, LINE_KEYS, loc=loc)


def mesh_kwargs(spec: dict, trace: dict | None = None) -> dict:
    raw = (trace or {}).get("mesh") or spec.get("mesh") or {}
    return pick(raw, MESH_KEYS, loc="mesh") if raw else {}


def contour_kwargs(spec: dict, trace: dict | None = None) -> dict:
    raw = dict((trace or {}).get("contour") or spec.get("contour") or {})
    raw.pop("lines", None)
    return pick(raw, CONTOUR_KEYS, loc="contour") if raw else {}


def quiver_kwargs(spec: dict, trace: dict | None = None) -> dict:
    raw = (trace or {}).get("quiver") or spec.get("quiver") or {}
    return pick(raw, QUIVER_KEYS, loc="quiver") if raw else {}


def windrose_kwargs(spec: dict) -> dict:
    raw = spec.get("windrose") or {}
    return pick(raw, WINDROSE_KEYS, loc="windrose") if raw else {}


def scatter_kwargs(style: dict | None, *, loc: str = "scatter") -> dict:
    if not style:
        return {}
    raw = style.get("scatter") if isinstance(style.get("scatter"), dict) else style
    return pick({k: v for k, v in raw.items() if k in SCATTER_KEYS}, SCATTER_KEYS, loc=loc)


def bar_kwargs(style: dict | None, *, loc: str = "bar") -> dict:
    if not style:
        return {}
    raw = style.get("bar") if isinstance(style.get("bar"), dict) else style
    return pick({k: v for k, v in raw.items() if k in BAR_KEYS}, BAR_KEYS, loc=loc)


def fill_kwargs(spec: dict | None, *, loc: str = "fill") -> dict:
    raw = (spec or {}).get("fill") if isinstance(spec, dict) else None
    return pick(raw, FILL_KEYS, loc=loc) if raw else {}


def box_kwargs(style: dict | None, *, loc: str = "box") -> dict:
    if not style:
        return {}
    raw = style.get("box") if isinstance(style.get("box"), dict) else style
    return pick({k: v for k, v in raw.items() if k in BOX_KEYS}, BOX_KEYS, loc=loc)


def resolve_axes_block(spec: dict | None) -> dict | list:
    """``axes`` for a dumped spec: template keys plus any user values."""

    def _one(user):
        out = dict(AXES_TEMPLATE)
        if isinstance(user, dict):
            out.update(user)
        return out

    user = (spec or {}).get("axes")
    if user is None:
        user = ((spec or {}).get("layout") or {}).get("axes")
    if isinstance(user, list):
        return [_one(item) for item in user]
    return _one(user)


def attach_figure_spec(resolved: dict, spec: dict | None = None) -> dict:
    """Fill matplotlib layout knobs on a dumped spec (axes, annotations, shapes).

    Every plot sidecar gets the same ``axes`` template so locators, tick lists,
    spines, and the rest are editable without a per-skill special case.
    """
    src = spec or {}
    out = dict(resolved)
    if src.get("axes") is not None:
        out["axes"] = src["axes"]
    out["axes"] = resolve_axes_block(out)
    for key in ("annotations", "shapes"):
        if src.get(key):
            out[key] = list(src[key])
        else:
            out.setdefault(key, list(out.get(key) or []))
    for key in ("line", "mesh", "contour", "fill", "quiver", "windrose", "mediogram"):
        if src.get(key) is not None:
            out[key] = src[key]
    layout = dict(out.get("layout") or {})
    spec_layout = src.get("layout") or {}
    if spec_layout.get("facecolor") is not None:
        layout["facecolor"] = spec_layout["facecolor"]
    else:
        layout.setdefault("facecolor", layout.get("facecolor"))
    if spec_layout.get("dpi") is not None:
        layout["dpi"] = spec_layout["dpi"]
    else:
        layout.setdefault("dpi", None)
    if spec_layout.get("colorbar") is not None:
        layout["colorbar"] = spec_layout["colorbar"]
    out["layout"] = layout
    style = dict(out.get("style") or {})
    rc = (src.get("style") or {}).get("rc") or src.get("rc")
    if rc:
        style["rc"] = rc
    out["style"] = style
    return out


def apply_style_then_rc(spec: dict, *, chart: str, fontsize, template: str) -> None:
    from weather_skills_core.figure import apply_style

    apply_style(fontsize, template=template, chart=chart)
    style = spec.get("style") or {}
    apply_rc(style.get("rc") or spec.get("rc"))


def finish_figure(fig, spec: dict, axes=None) -> None:
    """Post-draw: per-axes options, annotations, shapes, figure facecolor."""
    apply_axes_from_spec(fig, spec, axes)
    apply_annotations_and_shapes(fig, spec, axes)
    facecolor = (spec.get("layout") or {}).get("facecolor") or spec.get("facecolor")
    if facecolor:
        fig.patch.set_facecolor(facecolor)
    dpi = (spec.get("layout") or {}).get("dpi") or (spec.get("style") or {}).get("dpi")
    if dpi:
        fig.set_dpi(float(dpi))
