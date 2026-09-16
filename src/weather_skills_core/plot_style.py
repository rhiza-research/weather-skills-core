"""Plotly template, named colormaps, and user style-file overlays."""

from __future__ import annotations

import json
import os
from pathlib import Path

from weather_skills_core.errors import UsageError
from weather_skills_core.figure import DEFAULT_DPI, DEFAULT_FONTSIZE
from weather_skills_core.units import (
    parse_aggregation_period,
    variable_units,
)

SPEC_VERSION = 1
DEFAULT_MAX_COLUMNS = 4
DEFAULT_TEMPLATE = "weather_skills"

# CHIRPS-GEFS / Early Warning eXplorer rainfall-total classes (mm).
# Under (<2) is white; over (>2500) is pale pink.
PRECIP_COLORS = [
    "#ffffff",
    "#c7ffbb",
    "#75f676",
    "#1bb61d",
    "#b8edfb",
    "#50a5f8",
    "#1e6eec",
    "#dcdcff",
    "#a08bff",
    "#7060de",
    "#fff8ad",
    "#ff9d00",
    "#ff1400",
    "#a30005",
    "#e58d8b",
    "#ffe5e4",
]
PRECIP_BOUNDS = [2, 5, 10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2500]
PRECIP_SHORT_BOUNDS = [0.5, 1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200]
PRECIP_LONG_MIN_DAYS = 5

PRECIP_ANOMALY_COLORS = [
    "#c00006",
    "#ff3300",
    "#ff9d00",
    "#ffe772",
    "#7a5044",
    "#b68c80",
    "#f2dcd1",
    "#ffffff",
    "#c7ffbb",
    "#75f676",
    "#1bb61c",
    "#9bd1f5",
    "#2583f5",
    "#dcdcff",
    "#8070ee",
]
PRECIP_ANOMALY_BOUNDS = [-500, -300, -200, -100, -50, -25, -10, 10, 25, 50, 100, 200, 300, 500]

_STYLE_ENV = "WEATHER_SKILLS_PLOT_STYLE"
_USER_STYLE_CANDIDATES = (
    Path.home() / ".config" / "weather-skills" / "plot.toml",
    Path.home() / ".config" / "weather-skills" / "plot.json",
)


def default_style() -> dict:
    """Built-in style: template name, font, facet cap, colormap aliases."""
    return {
        "template": DEFAULT_TEMPLATE,
        "fontsize": DEFAULT_FONTSIZE,
        "max_columns": DEFAULT_MAX_COLUMNS,
        "dpi": DEFAULT_DPI,
        "colormap": None,
        "colormaps": {
            "chirps_total": {"colors": PRECIP_COLORS, "bounds": PRECIP_BOUNDS},
            "chirps_short": {"colors": PRECIP_COLORS, "bounds": PRECIP_SHORT_BOUNDS},
            "chirps_anom": {"colors": PRECIP_ANOMALY_COLORS, "bounds": PRECIP_ANOMALY_BOUNDS},
            "viridis": {"colorscale": "Viridis"},
        },
    }


def deep_merge(base: dict, overlay: dict) -> dict:
    """Return a new dict; nested dicts merge, other values replace."""
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_style_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".toml":
        import tomllib

        data = tomllib.loads(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise UsageError(f"plot style file {path} must contain a JSON/TOML object")
    return data.get("plot", data)


def load_user_style(path=None) -> dict:
    """Merge built-in style with optional user file (later wins)."""
    style = default_style()
    if path is not None:
        style = deep_merge(style, _load_style_file(Path(path)))
        return style
    env = os.environ.get(_STYLE_ENV)
    if env:
        return deep_merge(style, _load_style_file(Path(env)))
    for candidate in _USER_STYLE_CANDIDATES:
        if candidate.is_file():
            return deep_merge(style, _load_style_file(candidate))
    return style


def weather_skills_template(fontsize: int = DEFAULT_FONTSIZE):
    """Plotly template used as the default figure chrome."""
    import plotly.graph_objects as go

    fs = int(fontsize)
    return go.layout.Template(
        layout=go.Layout(
            font={"size": fs, "color": "#222222", "family": "Helvetica, Arial, sans-serif"},
            title={"font": {"size": fs}},
            paper_bgcolor="white",
            plot_bgcolor="white",
            coloraxis={
                "colorbar": {
                    "outlinewidth": 0,
                    "ticks": "outside",
                    "tickfont": {"size": max(8, int(round(fs * 0.85)))},
                    "title": {"font": {"size": fs}},
                }
            },
            xaxis={"showgrid": False, "zeroline": False, "automargin": True},
            yaxis={"showgrid": False, "zeroline": False, "automargin": True},
            margin={"l": 64, "r": 48, "t": 72, "b": 72},
            hovermode="closest",
        )
    )


def register_template(fontsize: int = DEFAULT_FONTSIZE) -> str:
    """Register ``weather_skills`` on Plotly's template map; return the name."""
    import plotly.io as pio

    pio.templates[DEFAULT_TEMPLATE] = weather_skills_template(fontsize)
    return DEFAULT_TEMPLATE


def aggregation_days(da) -> float | None:
    """Return stamped ``aggregation_period`` in days, or None."""
    period = da.attrs.get("aggregation_period")
    if not (isinstance(period, str) and period.strip()):
        return None
    try:
        return float(parse_aggregation_period(period).to("day").magnitude)
    except UsageError:
        return None


def is_precip(da) -> bool:
    from weather_skills_core.units import classify_variable

    kind = classify_variable(
        da.name or "",
        units=variable_units(da),
        standard_name=da.attrs.get("standard_name"),
    )
    return kind in ("precip", "precip_amount")


def is_precip_anomaly(da) -> bool:
    import numpy as np

    if not is_precip(da):
        return False
    name = f"{da.name or ''} {da.attrs.get('long_name') or ''}".lower()
    if "anomal" in name:
        return True
    sample = np.asarray(da.values, dtype=float).ravel()
    finite = sample[np.isfinite(sample)]
    return bool(finite.size) and bool(np.nanmin(finite) < 0)


def named_precip_scale(da) -> tuple[str, list[str], list[float]]:
    """Return ``(name, colors, bounds)`` for the default precip palette."""
    if is_precip_anomaly(da):
        return "chirps_anom", list(PRECIP_ANOMALY_COLORS), list(PRECIP_ANOMALY_BOUNDS)
    days = aggregation_days(da)
    if days is not None and days < PRECIP_LONG_MIN_DAYS:
        return "chirps_short", list(PRECIP_COLORS), list(PRECIP_SHORT_BOUNDS)
    return "chirps_total", list(PRECIP_COLORS), list(PRECIP_BOUNDS)


def parse_colormap_spec(spec: str | None) -> dict:
    """Parse a colormap name or comma-separated color list."""
    if spec is None or not str(spec).strip():
        return {}
    raw = str(spec).strip()
    if "," in raw:
        colors = [p.strip() for p in raw.split(",") if p.strip()]
        if len(colors) < 2:
            raise UsageError("--colormap comma list needs at least two colors")
        return {"name": "custom", "colors": colors}
    return {"name": raw}


def discrete_colorscale(colors: list[str], bounds: list[float] | None = None) -> list:
    """Plotly colorscale: interior colors as flat classes; ends are under/over."""
    if bounds is not None and len(colors) >= len(bounds) + 1:
        interior = colors[1 : 1 + (len(bounds) - 1)]
    else:
        interior = colors
    n = max(len(interior), 1)
    scale = []
    for i, color in enumerate(interior):
        t0 = i / n
        t1 = (i + 1) / n
        scale.append([t0, color])
        scale.append([min(t1, 1.0), color])
    if scale and scale[-1][0] < 1.0:
        scale.append([1.0, interior[-1]])
    return scale


def resolve_colorscale(da, colormap: str | None, *, stretch: bool = False) -> dict:
    """Pick a Plotly colorscale dict: name, colorscale, optional bounds/ticks."""
    parsed = parse_colormap_spec(colormap)
    if parsed.get("colors"):
        return {
            "name": parsed["name"],
            "colorscale": discrete_colorscale(parsed["colors"]),
            "bounds": None,
        }
    named = parsed.get("name")
    if named in ("chirps_total", "chirps_short", "chirps_anom"):
        registry = default_style()["colormaps"][named]
        colors = list(registry["colors"])
        bounds = list(registry["bounds"])
        if stretch:
            return {"name": named, "colorscale": discrete_colorscale(colors), "bounds": None}
        return {
            "name": named,
            "colorscale": discrete_colorscale(colors, bounds),
            "bounds": bounds,
            "cmin": bounds[0],
            "cmax": bounds[-1],
        }
    if named:
        return {"name": named, "colorscale": named, "bounds": None}
    if da is not None and is_precip(da):
        name, colors, bounds = named_precip_scale(da)
        if stretch:
            return {"name": name, "colorscale": discrete_colorscale(colors), "bounds": None}
        return {
            "name": name,
            "colorscale": discrete_colorscale(colors, bounds),
            "bounds": bounds,
            "cmin": bounds[0],
            "cmax": bounds[-1],
        }
    return {"name": "viridis", "colorscale": "Viridis", "bounds": None}
