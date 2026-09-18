"""Figure construction helpers: dates, colorbar, allowlists, PNG save."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from weather_skills_core.errors import UsageError

DEFAULT_FONTSIZE = 16
DEFAULT_DPI = 150

# Sept not Sep — the usual meteorological short form.
_MONTHS = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sept",
    "Oct",
    "Nov",
    "Dec",
)


def _ymd(value):
    """Return ``(year, month, day)`` from a datetime-like, or None."""
    if value is None:
        return None
    if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day"):
        try:
            year, month, day = int(value.year), int(value.month), int(value.day)
            if 1 <= month <= 12 and 1 <= day <= 31:
                return year, month, day
        except (TypeError, ValueError):
            pass
    try:
        import numpy as np

        arr = np.asarray(value)
        if arr.dtype.kind == "M":
            sample = np.asarray(arr.reshape(-1)[0]).astype("datetime64[D]")
            text = str(np.datetime_as_string(sample, unit="D"))
            return int(text[0:4]), int(text[5:7]), int(text[8:10])
    except (TypeError, ValueError, IndexError):
        pass
    text = str(value)
    if len(text) >= 10 and text[4:5] == "-" and text[7:8] == "-":
        try:
            return int(text[0:4]), int(text[5:7]), int(text[8:10])
        except ValueError:
            return None
    return None


def format_plot_date(value, *, year=True):
    """Figure date: ``14 Sept '26``. Set ``year=False`` for day-of-year ticks."""
    ymd = _ymd(value)
    if ymd is None:
        return str(value)
    y, month, day = ymd
    mon = _MONTHS[month - 1]
    if not year:
        return f"{day} {mon}"
    return f"{day} {mon} '{y % 100:02d}"


def axis_label(text):
    """Sentence-case an axis label; map lon/lat shorthand to Longitude/Latitude."""
    if text is None:
        return text
    s = str(text).strip()
    if not s:
        return s
    known = {
        "lon": "Longitude",
        "lat": "Latitude",
        "longitude": "Longitude",
        "latitude": "Latitude",
        "valid time": "Valid time",
        "calendar day": "Calendar day",
        "time": "Time",
        "step": "Step",
        "forecast step": "Forecast step",
    }
    key = s.lower()
    if key in known:
        return known[key]
    if s[:1].islower():
        return s[:1].upper() + s[1:]
    return s


def resolve_axis_label(override, default):
    """Use ``override`` verbatim when set; otherwise sentence-case ``default``."""
    if override is not None and str(override).strip() != "":
        return str(override)
    return axis_label(default)


def is_datetime_axis(values):
    """True when ``values`` are calendar dates (datetime64 or cftime)."""
    import numpy as np

    arr = np.asarray(values)
    if arr.dtype.kind == "M":
        return True
    if arr.size == 0:
        return False
    first = arr.reshape(-1)[0]
    return hasattr(first, "year") and hasattr(first, "month")


def resolve_time_axis_label(override, default, values):
    """Axis label for a 1D time axis. Datetime ticks already name the axis."""
    if override is not None and str(override).strip() != "":
        return str(override)
    if is_datetime_axis(values):
        return ""
    return axis_label(default)


def format_plot_date_range(start, end):
    """Inclusive range: ``1–7 Sept '26``, ``28 Aug–3 Sept '26``, ``28 Dec '25–3 Jan '26``."""
    a = _ymd(start)
    b = _ymd(end)
    if a is None or b is None:
        return f"{format_plot_date(start)}–{format_plot_date(end)}"
    ay, am, ad = a
    by, bm, bd = b
    a_mon, b_mon = _MONTHS[am - 1], _MONTHS[bm - 1]
    a_yr, b_yr = f"'{ay % 100:02d}", f"'{by % 100:02d}"
    if ay == by and am == bm:
        return f"{ad}–{bd} {a_mon} {a_yr}"
    if ay == by:
        return f"{ad} {a_mon}–{bd} {b_mon} {b_yr}"
    return f"{ad} {a_mon} {a_yr}–{bd} {b_mon} {b_yr}"


def apply_date_ticks(ax):
    """Show datetime x ticks as ``14 Sept '26``, never midnight timestamps."""
    import matplotlib.dates as mdates
    from matplotlib.ticker import FuncFormatter

    def _fmt(x, _pos):
        return format_plot_date(mdates.num2date(x))

    ax.xaxis.set_major_formatter(FuncFormatter(_fmt))


def parse_figsize(value):
    """Argparse converter for ``W,H`` or ``WxH`` inches."""
    if value is None:
        return None
    raw = str(value).strip().lower().replace("×", "x")
    if not raw:
        raise argparse.ArgumentTypeError("--figsize must be W,H inches (e.g. 10,6 or 10x6)")
    sep = "x" if "x" in raw and "," not in raw else ","
    parts = [p.strip() for p in raw.split(sep)]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--figsize must be W,H inches (e.g. 10,6 or 10x6)")
    try:
        width, height = float(parts[0]), float(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(
            "--figsize must be W,H inches (e.g. 10,6 or 10x6)"
        ) from None
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("--figsize width and height must be positive")
    return (width, height)


def parse_number_list(value):
    """Argparse converter for comma-separated floats (bounds / colorbar ticks)."""
    if isinstance(value, (list, tuple)):
        parts = [str(v).strip() for v in value if str(v).strip()]
    else:
        parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected comma-separated numbers")
    try:
        return [float(p) for p in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated numbers, got {value!r}"
        ) from None


def parse_label_list(value):
    """Argparse converter for comma-separated colorbar tick labels."""
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [p.strip() for p in str(value).split(",")]


def resolve_figsize(requested, default):
    """Honor an explicit ``--figsize``; otherwise use ``default``."""
    return tuple(requested) if requested is not None else tuple(default)


def apply_style(fontsize=DEFAULT_FONTSIZE, *, template="weather_skills", chart="line"):
    """Seaborn chrome plus a single ``--fontsize``.

    ``template`` is ``weather_skills`` (seaborn ``deep``) or ``colorblind``.
    ``chart`` is ``line`` (whitegrid) or ``map`` (ticks, no background grid).
    """
    import matplotlib as mpl

    from weather_skills_core.plot.theme import seaborn_palette_name, seaborn_style_name

    palette = seaborn_palette_name(template)
    style = seaborn_style_name(chart)
    try:
        import seaborn as sns
    except ImportError:
        sns = None
    if sns is not None:
        sns.set_theme(style=style, palette=palette, context="notebook")
        if chart == "map":
            mpl.rcParams["axes.grid"] = False
    fs = int(fontsize)
    tick = max(8, int(round(fs * 0.85)))
    legend = max(8, int(round(fs * 0.9)))
    mpl.rcParams.update(
        {
            "font.size": fs,
            "axes.titlesize": fs,
            "axes.labelsize": fs,
            "xtick.labelsize": tick,
            "ytick.labelsize": tick,
            "legend.fontsize": legend,
            "figure.titlesize": fs,
        }
    )


def _format_cbar_tick(value):
    """Short numeric tick: ``10`` not ``10.0``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:g}"


def colorbar_spec(obj: dict | None) -> dict | None:
    """Return the ``layout.colorbar`` size dict, or None.

    ``len`` / ``shrink`` is the long-side fraction (0–1); ``thickness`` is the
    short side in pixels (>1) or figure fraction (≤1).
    """
    if not isinstance(obj, dict):
        return None
    layout = obj.get("layout")
    if isinstance(layout, dict) and isinstance(layout.get("colorbar"), dict):
        return dict(layout["colorbar"])
    return None


def colorbar_size_kwargs(spec=None, *, colorbar=None) -> dict:
    """Matplotlib ``colorbar()`` kwargs from ``len``/``shrink`` and ``thickness``."""
    cbar = colorbar if colorbar is not None else colorbar_spec(spec)
    if not cbar:
        return {}
    kw = {}
    length = cbar.get("len") if cbar.get("len") is not None else cbar.get("shrink")
    if length is not None:
        kw["shrink"] = float(length)
    thickness = cbar.get("thickness")
    if thickness is not None:
        thick = float(thickness)
        if thick > 1:
            kw["aspect"] = max(4.0, 20.0 * (30.0 / thick))
        elif thick > 0:
            kw["fraction"] = thick
    return kw


def apply_colorbar_size(fig, spec=None, *, colorbar=None):
    """Resize existing colorbar axes from ``len``/``shrink`` and ``thickness``."""
    cbar = colorbar if colorbar is not None else colorbar_spec(spec)
    if not cbar:
        return
    ax = next((a for a in fig.axes if a.get_label() == "<colorbar>"), None)
    if ax is None:
        return
    length = cbar.get("len") if cbar.get("len") is not None else cbar.get("shrink")
    thickness = cbar.get("thickness")
    if length is None and thickness is None:
        return
    fig.canvas.draw()
    pos = ax.get_position()
    horizontal = pos.width >= pos.height
    fig_w, fig_h = fig.get_size_inches()
    dpi = float(fig.dpi or DEFAULT_DPI)
    x0, y0, width, height = pos.x0, pos.y0, pos.width, pos.height
    if length is not None:
        frac = float(length)
        if 0 < frac <= 1:
            if horizontal:
                new_w = width * frac
                x0 = x0 + (width - new_w) / 2
                width = new_w
            else:
                new_h = height * frac
                y0 = y0 + (height - new_h) / 2
                height = new_h
    if thickness is not None:
        thick = float(thickness)
        if thick > 1:
            if horizontal:
                height = thick / (dpi * fig_h)
            else:
                width = thick / (dpi * fig_w)
        elif thick > 0:
            if horizontal:
                height = thick
            else:
                width = thick
    ax.set_in_layout(False)
    ax.set_position([x0, y0, width, height])


def add_shared_colorbar(fig, mappable, axes, label="", *, location=None, **kwargs):
    """Attach a colorbar in a matplotlib-reserved slot (not a figure-fraction box).

    One map gets a right-hand bar; several maps share a bottom bar. Discrete
    ``ticks`` (BoundaryNorm bounds) are all labeled.
    """
    import numpy as np

    if mappable is None:
        return None
    if hasattr(axes, "ravel"):
        axes = [ax for ax in np.ravel(axes) if getattr(ax, "get_visible", lambda: True)()]
    elif not isinstance(axes, (list, tuple)):
        axes = [axes]
    if not axes:
        return None
    if location is None:
        location = "right" if len(axes) == 1 else "bottom"
    ticks = kwargs.get("ticks")
    tick_list = list(ticks) if ticks is not None else None
    labels = kwargs.pop("labels", None)
    # Discrete class bars need the full axes span so every bound can be labeled.
    shrink = kwargs.pop("shrink", None)
    if shrink is None:
        shrink = 1.0 if tick_list and len(tick_list) >= 6 else 0.8
    pad = kwargs.pop("pad", 0.08)
    if "location" in kwargs:
        location = kwargs.pop("location")
    cbar = fig.colorbar(
        mappable,
        ax=axes,
        location=location,
        shrink=shrink,
        pad=pad,
        **kwargs,
    )
    if label:
        cbar.set_label(label)
    if tick_list is not None:
        cbar.set_ticks(tick_list)
        if labels is not None:
            if len(list(labels)) != len(tick_list):
                from weather_skills_core.errors import UsageError

                raise UsageError(
                    f"colorbar labels has {len(list(labels))} entries "
                    f"but ticks has {len(tick_list)}"
                )
            cbar.set_ticklabels(list(labels))
        else:
            cbar.set_ticklabels([_format_cbar_tick(t) for t in tick_list])
    elif labels is not None:
        from weather_skills_core.errors import UsageError

        raise UsageError("colorbar labels requires ticks")
    return cbar


def save_figure(fig, path, *, pad_inches=None, tight=True, dpi=None):
    """Write a PNG. Default tight-crops chrome; ``tight=False`` keeps ``figsize``."""
    import matplotlib.pyplot as plt

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    kw = {"dpi": DEFAULT_DPI if dpi is None else dpi}
    if tight:
        kw["bbox_inches"] = "tight"
        if pad_inches is not None:
            kw["pad_inches"] = pad_inches
    fig.savefig(output, **kw)
    plt.close(fig)
    return output


@dataclass
class CompiledFigure:
    """A compiled matplotlib figure plus the resolved spec used to export it."""

    fig: object
    spec: dict
    tight: bool = True
    map_drawn: dict = field(default_factory=dict)


def export_png(fig, path, *, scale=1, tight=True) -> Path:
    """Write a PNG via matplotlib Agg. Pass ``tight=False`` to keep ``--figsize``."""
    return save_figure(fig, path, tight=tight, dpi=DEFAULT_DPI * float(scale or 1))


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
        "mode",
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
    """Apply ``theme.rc`` as matplotlib rcParams."""
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
        from weather_skills_core.plot.figure import apply_date_ticks

        if getattr(axis, "axis_name", "x") != "x":
            raise UsageError("date formatter is only supported on the x axis")
        apply_date_ticks(axis.axes)
    elif kind in ("format", "strformat") and extra.get("fmt"):
        fmt = str(extra["fmt"])
        axis.set_major_formatter(FuncFormatter(lambda x, _p, f=fmt: f.format(x)))
    elif kind in ("dayofyear", "doy", "calendar_day"):
        import datetime as dt

        from weather_skills_core.plot.figure import format_plot_date

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
    anns = list(spec.get("annotations") or [])
    shapes = list(spec.get("shapes") or [])
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
    cbar = colorbar_spec(spec) or {}
    kw = colorbar_size_kwargs(colorbar=cbar)
    extra = pick(
        {k: v for k, v in cbar.items() if k in COLORBAR_EXTRA_KEYS},
        COLORBAR_EXTRA_KEYS,
        loc="layout.colorbar",
    )
    kw.update(extra)
    if cbar.get("ticks") is not None:
        kw["ticks"] = list(cbar["ticks"])
    if cbar.get("labels") is not None:
        kw["labels"] = list(cbar["labels"])
    return kw


def line_kwargs(style: dict | None, *, loc: str = "line") -> dict:
    if not style:
        return {}
    raw = style.get("line") if isinstance(style.get("line"), dict) else style
    return pick({k: v for k, v in raw.items() if k in LINE_KEYS}, LINE_KEYS, loc=loc)


def mesh_kwargs(trace: dict | None) -> dict:
    raw = (trace or {}).get("mesh") or {}
    return pick(raw, MESH_KEYS, loc="traces[].mesh") if raw else {}


def contour_kwargs(trace: dict | None) -> dict:
    raw = dict((trace or {}).get("contour") or {})
    raw.pop("lines", None)
    return pick(raw, CONTOUR_KEYS, loc="traces[].contour") if raw else {}


def quiver_kwargs(trace: dict | None) -> dict:
    raw = (trace or {}).get("quiver") or {}
    return pick(raw, QUIVER_KEYS, loc="traces[].quiver") if raw else {}


def windrose_kwargs(trace: dict | None) -> dict:
    raw = (trace or {}).get("windrose") or {}
    return pick(raw, WINDROSE_KEYS, loc="traces[].windrose") if raw else {}


def scatter_kwargs(style: dict | None, *, loc: str = "scatter") -> dict:
    if not style:
        return {}
    raw = style.get("scatter") if isinstance(style.get("scatter"), dict) else style
    return pick({k: v for k, v in raw.items() if k in SCATTER_KEYS}, SCATTER_KEYS, loc=loc)


def bar_kwargs(style: dict | None, *, loc: str = "bar") -> dict:
    if not style:
        return {}
    raw = style.get("bar") if isinstance(style.get("bar"), dict) else style
    mpl_keys = BAR_KEYS - {"mode"}
    return pick({k: v for k, v in raw.items() if k in mpl_keys}, mpl_keys, loc=loc)


def fill_kwargs(trace: dict | None, *, loc: str = "traces[].fill") -> dict:
    raw = (trace or {}).get("fill") if isinstance(trace, dict) else None
    return pick(raw, FILL_KEYS, loc=loc) if raw else {}


def box_kwargs(style: dict | None, *, loc: str = "box") -> dict:
    if not style:
        return {}
    raw = style.get("box") if isinstance(style.get("box"), dict) else style
    return pick({k: v for k, v in raw.items() if k in BOX_KEYS}, BOX_KEYS, loc=loc)


def resolve_axes_block(spec: dict | None) -> dict | list:
    """``axes`` for a dumped spec: only the keys that carry a value.

    The full set of editable knobs lives in ``AXES_TEMPLATE`` and is documented
    in ``docs/plotting.md``; stamping all of them as nulls onto every dumped
    spec buried the handful of values that were actually set.
    """

    def _one(user):
        if not isinstance(user, dict):
            return {}
        return {k: v for k, v in user.items() if v is not None}

    user = (spec or {}).get("axes")
    if isinstance(user, list):
        return [_one(item) for item in user]
    return _one(user)


def attach_figure_spec(resolved: dict, spec: dict | None = None) -> dict:
    """Fill matplotlib layout knobs on a dumped spec (axes, annotations, shapes).

    Sidecars dump only the axes keys that were set. The full editable catalog is
    ``AXES_TEMPLATE``.
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
    layout = dict(out.get("layout") or {})
    spec_layout = src.get("layout") or {}
    for key in ("facecolor", "dpi", "colorbar"):
        if spec_layout.get(key) is not None:
            layout[key] = spec_layout[key]
    out["layout"] = layout
    theme = dict(out.get("theme") or {})
    rc = (src.get("theme") or {}).get("rc")
    if rc:
        theme["rc"] = rc
    out["theme"] = theme
    return out


def apply_style_then_rc(spec: dict, *, chart: str, fontsize, template: str) -> None:
    apply_style(fontsize, template=template, chart=chart)
    apply_rc((spec.get("theme") or {}).get("rc"))


def finish_figure(fig, spec: dict, axes=None) -> None:
    """Post-draw: per-axes options, annotations, shapes, figure facecolor."""
    apply_axes_from_spec(fig, spec, axes)
    apply_annotations_and_shapes(fig, spec, axes)
    layout = spec.get("layout") or {}
    if layout.get("facecolor"):
        fig.patch.set_facecolor(layout["facecolor"])
    if layout.get("dpi"):
        fig.set_dpi(float(layout["dpi"]))
