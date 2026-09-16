"""Shared matplotlib chrome for weather-skills figures.

Matplotlib places artists; callers set style and data only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

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


def resolve_figsize(requested, default):
    """Honor an explicit ``--figsize``; otherwise use ``default``."""
    return tuple(requested) if requested is not None else tuple(default)


def apply_style(fontsize=DEFAULT_FONTSIZE):
    """Set rcParams from a single ``--fontsize``. Ticks use matplotlib's usual ratio."""
    import matplotlib as mpl

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
    # Discrete class bars need the full axes span so every bound can be labeled.
    shrink = kwargs.pop("shrink", None)
    if shrink is None:
        shrink = 1.0 if tick_list and len(tick_list) >= 6 else 0.8
    cbar = fig.colorbar(
        mappable,
        ax=axes,
        location=location,
        shrink=shrink,
        pad=0.08,
        **kwargs,
    )
    if label:
        cbar.set_label(label)
    if tick_list is not None:
        cbar.set_ticks(tick_list)
        cbar.set_ticklabels([_format_cbar_tick(t) for t in tick_list])
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
