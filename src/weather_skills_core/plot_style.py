"""Named colormaps, matplotlib cmap/norm, and user style-file overlays."""

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
TEMPLATES = ("weather_skills", "colorblind")
SEABORN_SEQUENTIAL = "rocket"


def _rgb(*rows: tuple[int, int, int]) -> list[str]:
    """``(r, g, b)`` 0–255 → ``#rrggbb``. Source values are CHC IDL palettes."""
    return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in rows]


# CHC ``ppt_total_cmap.pro`` (Will Turner, 6 Feb 2018). 17 colors / 16 interior
# breaks: null/negative white, 0–2 mm white, then the published classes;
# over (>2500 mm) is pale pink. Data-dependent IDL min/max ends are under/over.
PRECIP_COLORS = _rgb(
    (255, 255, 255),  # null / negative
    (255, 255, 255),  # 0–2 mm
    (200, 255, 190),
    (120, 245, 115),
    (30, 180, 30),
    (180, 240, 250),
    (80, 165, 245),
    (30, 110, 235),
    (220, 220, 255),
    (160, 140, 255),
    (112, 96, 220),
    (255, 250, 170),
    (255, 160, 0),
    (255, 20, 0),
    (165, 0, 0),
    (230, 140, 140),
    (255, 230, 230),
)
PRECIP_BOUNDS = [0, 2, 5, 10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2500]
PRECIP_SHORT_BOUNDS = [0.5, 1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200]
PRECIP_LONG_MIN_DAYS = 5

# CHC ``ppt_anomaly_cmap.pro`` (Will Turner, 8 Feb 2018).
PRECIP_ANOMALY_COLORS = _rgb(
    (192, 0, 0),
    (255, 50, 0),
    (255, 160, 0),
    (255, 232, 120),
    (120, 80, 70),
    (180, 140, 130),
    (240, 220, 210),
    (255, 255, 255),
    (200, 255, 190),
    (120, 245, 115),
    (30, 180, 30),
    (150, 210, 250),
    (40, 130, 240),
    (220, 220, 255),
    (128, 112, 235),
)
PRECIP_ANOMALY_BOUNDS = [-500, -300, -200, -100, -50, -25, -10, 10, 25, 50, 100, 200, 300, 500]

# CHC ``ppt_poa_cmap.pro`` (percent of normal). Missing gray is NaN, not a class.
PRECIP_POA_COLORS = _rgb(
    (225, 190, 180),
    (192, 0, 0),
    (255, 50, 0),
    (255, 160, 0),
    (255, 232, 120),
    (255, 255, 255),
    (200, 255, 190),
    (120, 245, 115),
    (30, 180, 30),
    (150, 210, 250),
    (40, 130, 240),
)
PRECIP_POA_BOUNDS = [30, 45, 60, 75, 90, 110, 125, 150, 200, 300]

# CHC ``ppt_spp_cmap.pro`` (seasonal rainfall performance probability classes).
PRECIP_SPP_COLORS = _rgb(
    (220, 220, 220),
    (255, 255, 255),
    (255, 232, 120),
    (255, 160, 0),
    (255, 50, 0),
    (192, 0, 0),
    (200, 255, 190),
    (150, 245, 140),
    (55, 210, 60),
    (15, 160, 15),
    (180, 240, 250),
    (120, 185, 250),
    (40, 130, 240),
    (20, 100, 210),
)
PRECIP_SPP_BOUNDS = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5]

# CHC ``spi_cmap.pro``. Missing gray is NaN; outer bounds are ±2.5.
SPI_COLORS = _rgb(
    (115, 0, 0),
    (231, 0, 0),
    (255, 170, 0),
    (255, 211, 123),
    (255, 255, 0),
    (255, 255, 255),
    (189, 235, 255),
    (115, 178, 255),
    (0, 113, 255),
    (0, 77, 173),
    (173, 0, 231),
)
SPI_BOUNDS = [-2.5, -2.0, -1.5, -1.2, -0.7, -0.5, 0.5, 0.7, 1.2, 1.5, 2.0, 2.5]

# CHC ``rank_cmap.pro`` (missing gray omitted; bounds depend on ``n_seasons``).
RANK_COLORS = _rgb(
    (115, 0, 0),
    (231, 0, 0),
    (255, 170, 0),
    (255, 255, 255),
    (189, 235, 255),
    (0, 113, 255),
    (0, 0, 85),
)

DISCRETE_PRECIP_NAMES = frozenset(
    {
        "chirps_total",
        "chirps_short",
        "chirps_anom",
        "ppt_total",
        "ppt_short",
        "ppt_anomaly",
        "ppt_anom",
        "ppt_poa",
        "ppt_spp",
        "spi",
    }
)

_STYLE_ENV = "WEATHER_SKILLS_PLOT_STYLE"
_USER_STYLE_CANDIDATES = (
    Path.home() / ".config" / "weather-skills" / "plot.toml",
    Path.home() / ".config" / "weather-skills" / "plot.json",
)


def seaborn_palette_name(template: str | None) -> str:
    """Seaborn qualitative palette for ``template``."""
    name = (template or DEFAULT_TEMPLATE).strip().lower().replace("-", "_")
    if name in ("colorblind", "seaborn_colorblind", "colourblind"):
        return "colorblind"
    return "deep"


def seaborn_style_name(chart: str | None) -> str:
    """Seaborn axes style: whitegrid for 1-D, ticks for maps."""
    if (chart or "line") == "map":
        return "ticks"
    return "whitegrid"


def normalize_template(template: str | None) -> str:
    """``weather_skills`` or ``colorblind``."""
    if seaborn_palette_name(template) == "colorblind":
        return "colorblind"
    return DEFAULT_TEMPLATE


def along_dim(da, along: str | None) -> str | None:
    """Resolve ``along`` to a dim on ``da``, including ontology aliases (member/number)."""
    from weather_skills_core.standard_dataset import ALIASES, names_for

    if not along:
        return None
    if along in da.dims:
        return along
    preferred = ALIASES.get(along, along)
    return next((name for name in names_for(preferred) if name in da.dims), None)


def parse_band(value) -> tuple[float, float] | None:
    """Parse ``--band`` / spec ``band`` as two percentiles, default ``10,90``."""
    if value is None or value is False or value == "":
        return None
    if value is True:
        return (10.0, 90.0)
    if isinstance(value, dict):
        q = value.get("q") or value.get("percentiles")
        if q is None:
            lo = value.get("low", 10)
            hi = value.get("high", 90)
            value = (lo, hi)
        else:
            value = q
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise UsageError("--band must be two percentiles, e.g. 10,90")
        lo, hi = float(value[0]), float(value[1])
    else:
        raw = str(value).strip().lower().replace("q", "")
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if len(parts) != 2:
            raise UsageError("--band must be two percentiles, e.g. 10,90")
        lo, hi = float(parts[0]), float(parts[1])
    if not 0 <= lo < hi <= 100:
        raise UsageError(f"--band percentiles must satisfy 0 ≤ low < high ≤ 100; got {lo},{hi}")
    return (lo, hi)


def default_style() -> dict:
    """Built-in style: seaborn template, font, facet cap, colormap aliases."""
    return {
        "template": DEFAULT_TEMPLATE,
        "palette": "deep",
        "fontsize": DEFAULT_FONTSIZE,
        "max_columns": DEFAULT_MAX_COLUMNS,
        "dpi": DEFAULT_DPI,
        "colormap": None,
        "colormaps": {
            "chirps_total": {"colors": PRECIP_COLORS, "bounds": PRECIP_BOUNDS},
            "ppt_total": {"colors": PRECIP_COLORS, "bounds": PRECIP_BOUNDS},
            "chirps_short": {"colors": PRECIP_COLORS, "bounds": PRECIP_SHORT_BOUNDS},
            "ppt_short": {"colors": PRECIP_COLORS, "bounds": PRECIP_SHORT_BOUNDS},
            "chirps_anom": {"colors": PRECIP_ANOMALY_COLORS, "bounds": PRECIP_ANOMALY_BOUNDS},
            "ppt_anomaly": {"colors": PRECIP_ANOMALY_COLORS, "bounds": PRECIP_ANOMALY_BOUNDS},
            "ppt_anom": {"colors": PRECIP_ANOMALY_COLORS, "bounds": PRECIP_ANOMALY_BOUNDS},
            "ppt_poa": {"colors": PRECIP_POA_COLORS, "bounds": PRECIP_POA_BOUNDS},
            "ppt_spp": {"colors": PRECIP_SPP_COLORS, "bounds": PRECIP_SPP_BOUNDS},
            "spi": {"colors": SPI_COLORS, "bounds": SPI_BOUNDS},
            "rocket": {"cmap": "rocket"},
            "viridis": {"cmap": "viridis"},
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


def mpl_color(color):
    """Map grayscale numbers / names to a matplotlib color."""
    if color is None:
        return None
    if isinstance(color, (int, float)):
        v = float(max(0.0, min(1.0, color)))
        return (v, v, v)
    raw = str(color).strip()
    try:
        v = float(raw)
    except ValueError:
        return raw
    if 0.0 <= v <= 1.0:
        return (v, v, v)
    return raw


def mpl_cmap_norm(scale: dict):
    """Return ``(cmap, norm)`` for a scale dict from ``resolve_colorscale``."""
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, Normalize

    colors = scale.get("colors") if scale else None
    bounds = scale.get("bounds") if scale else None
    cmin = scale.get("cmin") if scale else None
    cmax = scale.get("cmax") if scale else None
    cmap_name = str((scale or {}).get("name") or "discrete")
    if bounds and colors and len(colors) == len(bounds) - 1:
        cmap = ListedColormap(list(colors), name=cmap_name).with_extremes(
            bad=(0.0, 0.0, 0.0, 0.0)
        )
        return cmap, BoundaryNorm([float(b) for b in bounds], cmap.N)
    if bounds and colors and len(colors) >= len(bounds) + 1:
        under, over = colors[0], colors[-1]
        interior = colors[1 : 1 + (len(bounds) - 1)]
        cmap = ListedColormap(list(interior), name=cmap_name).with_extremes(
            under=under, over=over, bad=(0.0, 0.0, 0.0, 0.0)
        )
        return cmap, BoundaryNorm([float(b) for b in bounds], cmap.N)
    if colors:
        cmap = LinearSegmentedColormap.from_list(scale.get("name") or "custom", list(colors))
        cmap = cmap.with_extremes(bad=(0.0, 0.0, 0.0, 0.0))
        if cmin is None:
            cmin = 0.0
        if cmax is None or cmax == cmin:
            cmax = cmin + 1.0
        return cmap, Normalize(vmin=cmin, vmax=cmax)
    import matplotlib.pyplot as plt

    name = str((scale or {}).get("cmap") or (scale or {}).get("name") or SEABORN_SEQUENTIAL).lower()
    if name in ("rocket", "mako", "flare", "crest"):
        try:
            import seaborn as sns

            cmap = sns.color_palette(name, as_cmap=True)
        except ImportError:
            cmap = plt.get_cmap("viridis")
    else:
        cmap = plt.get_cmap(name)
    return cmap, Normalize(vmin=cmin, vmax=cmax)


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


def is_precip_poa(da) -> bool:
    """Percent-of-normal precip (CHC ``ppt_poa``)."""
    name = f"{da.name or ''} {da.attrs.get('long_name') or ''}".lower()
    tokens = name.replace("_", " ").replace("-", " ").split()
    if "percent of" in name or "pct of" in name or "poa" in tokens:
        return True
    units = (variable_units(da) or "").strip().lower()
    return units in {"%", "percent"}


def is_spi(da) -> bool:
    name = f"{da.name or ''} {da.attrs.get('long_name') or ''}".lower()
    tokens = name.replace("_", " ").replace("-", " ").split()
    return "spi" in tokens or "standardized precipitation" in name


def named_precip_scale(da) -> tuple[str, list[str], list[float]]:
    """Return ``(name, colors, bounds)`` for the default precip palette."""
    if is_spi(da):
        return "spi", list(SPI_COLORS), list(SPI_BOUNDS)
    if is_precip_poa(da):
        return "ppt_poa", list(PRECIP_POA_COLORS), list(PRECIP_POA_BOUNDS)
    if is_precip_anomaly(da):
        return "chirps_anom", list(PRECIP_ANOMALY_COLORS), list(PRECIP_ANOMALY_BOUNDS)
    days = aggregation_days(da)
    if days is not None and days < PRECIP_LONG_MIN_DAYS:
        return "chirps_short", list(PRECIP_COLORS), list(PRECIP_SHORT_BOUNDS)
    return "chirps_total", list(PRECIP_COLORS), list(PRECIP_BOUNDS)


def rank_colorscale(n_seasons: int) -> dict:
    """CHC ``rank_cmap.pro`` classes for a climatology of ``n_seasons`` years."""
    n = int(n_seasons)
    if n < 4:
        raise UsageError("--colormap ppt_rank needs n_seasons ≥ 4")
    bounds = [-0.5, 1.5, 2.5, 3.5, n - 2.5, n - 1.5, n - 0.5, n + 0.5]
    return {
        "name": "ppt_rank",
        "colors": list(RANK_COLORS),
        "bounds": bounds,
        "cmin": bounds[0],
        "cmax": bounds[-1],
    }


def _discrete_scale(name: str, registry: dict, *, stretch: bool) -> dict:
    colors = list(registry["colors"])
    bounds = list(registry["bounds"])
    if stretch:
        return {"name": name, "colors": colors, "bounds": None}
    return {
        "name": name,
        "colors": colors,
        "bounds": bounds,
        "cmin": bounds[0],
        "cmax": bounds[-1],
    }


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


def resolve_colorscale(da, colormap: str | None, *, stretch: bool = False) -> dict:
    """Pick a colormap dict: name, colors and/or cmap, optional bounds."""
    parsed = parse_colormap_spec(colormap)
    if parsed.get("colors"):
        return {"name": parsed["name"], "colors": parsed["colors"], "bounds": None}
    named = parsed.get("name")
    registry = default_style()["colormaps"]
    entry = registry.get(named) if named else None
    if entry and entry.get("colors") and entry.get("bounds"):
        return _discrete_scale(named, entry, stretch=stretch)
    if named:
        return {"name": named, "cmap": named.lower(), "bounds": None}
    if da is not None and (is_precip(da) or is_spi(da) or is_precip_poa(da)):
        name, colors, bounds = named_precip_scale(da)
        return _discrete_scale(name, {"colors": colors, "bounds": bounds}, stretch=stretch)
    return {"name": SEABORN_SEQUENTIAL, "cmap": SEABORN_SEQUENTIAL, "bounds": None}
