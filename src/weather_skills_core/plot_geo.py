"""Geographic overlays for weather-skills maps: one loader, one drawer.

Every map in the stack — single heatmap, contour, quiver, layered map, and the
compare / verify grids — draws its coastlines, borders, lakes and admin-1 lines
through this module, so the same field renders with the same geography whatever
skill asked for it.

Natural Earth (via cartopy) is the preferred source and picks a resolution from
the map span. When cartopy or the Natural Earth download is unavailable, the
bundled ``countries.geojson`` supplies country outlines instead; a download or
clip failure warns and skips that layer rather than failing the figure.
"""

from __future__ import annotations

import json
import sys
from importlib.resources import files

# Natural Earth scale vs map span (max of the lon/lat extent in degrees).
# Admin-1 (states / provinces / counties) is only readable on country-scale
# views; a multi-country or basin map would be a thicket of province lines.
ADMIN1_MAX_SPAN_DEG = 20.0
HIRES_MAX_SPAN_DEG = 45.0
MIDRES_MAX_SPAN_DEG = 90.0

# KMD-style water fill (Lake Victoria, Turkana, …) drawn on top of the heatmap.
LAKE_FACECOLOR = "#4da6ff"
ADMIN1_STYLE = {"facecolor": "none", "edgecolor": "0.45", "linewidth": 0.4, "zorder": 3}
LAKES_STYLE = {
    "facecolor": LAKE_FACECOLOR,
    "edgecolor": LAKE_FACECOLOR,
    "linewidth": 0.4,
    "zorder": 3.5,
}
BORDERS_STYLE = {"facecolor": "none", "edgecolor": "0.15", "linewidth": 0.8, "zorder": 4}
COAST_STYLE = {"facecolor": "none", "edgecolor": "black", "linewidth": 0.8, "zorder": 4}


def extent_span_deg(extent):
    lon_min, lon_max, lat_min, lat_max = extent
    return max(abs(lon_max - lon_min), abs(lat_max - lat_min))


def boundary_layers(extent):
    """Natural Earth scale and whether to overlay admin-1 for this view."""
    span = extent_span_deg(extent)
    if span > MIDRES_MAX_SPAN_DEG:
        return {"scale": "110m", "admin1": False}
    if span > HIRES_MAX_SPAN_DEG:
        return {"scale": "50m", "admin1": False}
    return {"scale": "10m", "admin1": span <= ADMIN1_MAX_SPAN_DEG}


def extent_clip_geom(extent):
    """Shapely clip geometry for ``lon_min,lon_max,lat_min,lat_max``.

    Antimeridian views store a continuous unwrapped lon (e.g. 170..190) which
    is split back into ``[-180, 180]`` pieces for Natural Earth intersection.
    """
    from shapely.geometry import box

    lon_min, lon_max, lat_min, lat_max = extent
    if lon_max > 180.0:
        return box(lon_min, lat_min, 180.0, lat_max).union(
            box(-180.0, lat_min, lon_max - 360.0, lat_max)
        )
    if lon_min > lon_max:
        return box(lon_min, lat_min, 180.0, lat_max).union(box(-180.0, lat_min, lon_max, lat_max))
    return box(lon_min, lat_min, lon_max, lat_max)


def unwrap_geoms(geoms, lon_min):
    """Shift western-hemisphere pieces so they match an unwrapped lon axis."""
    import numpy as np
    import shapely

    def shift(coords):
        out = np.asarray(coords).copy()
        out[:, 0] = np.where(out[:, 0] < lon_min, out[:, 0] + 360.0, out[:, 0])
        return out

    return [shapely.transform(g, shift) for g in geoms if g is not None and not g.is_empty]


def clip_ne_geoms(resolution, category, name, clip_geom):
    """Natural Earth geometries intersecting ``clip_geom`` (eager download)."""
    import cartopy.io.shapereader as shpreader

    path = shpreader.natural_earth(resolution=resolution, category=category, name=name)
    geoms = []
    for geom in shpreader.Reader(path).geometries():
        if geom is None or geom.is_empty:
            continue
        try:
            if not geom.intersects(clip_geom):
                continue
            clipped = geom.intersection(clip_geom)
        except Exception:  # noqa: BLE001
            clipped = geom
        if clipped is None or clipped.is_empty:
            continue
        if clipped.geom_type == "GeometryCollection":
            geoms.extend(g for g in clipped.geoms if g is not None and not g.is_empty)
        else:
            geoms.append(clipped)
    return geoms


def _bundled_country_geoms(clip_geom):
    """Country outlines from the bundled Natural Earth extract (no cartopy)."""
    import shapely
    from shapely.geometry import shape

    raw = json.loads(files("weather_skills_core.data").joinpath("countries.geojson").read_text())
    geoms = []
    for feat in raw.get("features") or []:
        geom = feat.get("geometry")
        if not geom:
            continue
        try:
            poly = shape(geom)
            if not poly.intersects(clip_geom):
                continue
            geoms.append(poly.intersection(clip_geom))
        except Exception:  # noqa: BLE001
            continue
    return [shapely.boundary(g) for g in geoms if g is not None and not g.is_empty]


def load_geo_overlays(extent):
    """Scale-appropriate coastline / border / filled-lake / admin-1 overlays.

    Returns a list of ``(geometries, matplotlib style)`` layers, each clipped to
    the map extent so a country-scale view does not draw the rest of the world.
    Download or clip failures warn and skip that layer — the map still renders.
    """
    if extent is None:
        return []
    spec = boundary_layers(extent)
    try:
        clip = extent_clip_geom(extent)
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: geographic overlays unavailable ({exc}); skipping.", file=sys.stderr)
        return []
    lon_min, lon_max = extent[0], extent[1]
    layers = []

    def add(category, name, style, resolution=None):
        res = resolution or spec["scale"]
        try:
            geoms = clip_ne_geoms(res, category, name, clip)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: {name} overlay unavailable ({exc}); skipping.", file=sys.stderr)
            return
        if lon_max > 180.0:
            geoms = unwrap_geoms(geoms, lon_min)
        if geoms:
            layers.append((geoms, style))

    try:
        import cartopy.io.shapereader  # noqa: F401
    except ImportError:
        try:
            geoms = _bundled_country_geoms(clip)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: geographic overlays unavailable ({exc}); skipping.", file=sys.stderr)
            return []
        if lon_max > 180.0:
            geoms = unwrap_geoms(geoms, lon_min)
        return [(geoms, BORDERS_STYLE)] if geoms else []

    if spec["admin1"]:
        add("cultural", "admin_1_states_provinces", ADMIN1_STYLE, resolution="10m")
    add("physical", "lakes", LAKES_STYLE)
    add("cultural", "admin_0_boundary_lines_land", BORDERS_STYLE)
    add("physical", "coastline", COAST_STYLE)
    return layers


def draw_geo_overlays(ax, overlays, crs=None):
    """Draw ``load_geo_overlays`` layers on ``ax``.

    A cartopy GeoAxes takes the geometries directly; a plain Axes gets them as
    line collections so both map paths share one set of overlays.
    """
    if not overlays:
        return
    if crs is not None and hasattr(ax, "add_geometries"):
        for geoms, style in overlays:
            ax.add_geometries(geoms, crs, **style)
        return
    from matplotlib.collections import LineCollection, PathCollection
    from matplotlib.path import Path as MplPath

    for geoms, style in overlays:
        segments, rings = [], []
        for geom in geoms:
            for part in getattr(geom, "geoms", [geom]):
                if part.geom_type in ("Polygon",):
                    rings.append(list(part.exterior.coords))
                    segments.extend(list(hole.coords) for hole in part.interiors)
                elif part.geom_type in ("LineString", "LinearRing"):
                    segments.append(list(part.coords))
        face = style.get("facecolor", "none")
        if rings and face not in (None, "none"):
            ax.add_collection(
                PathCollection(
                    [MplPath(ring) for ring in rings],
                    facecolors=face,
                    edgecolors=style.get("edgecolor", face),
                    linewidths=style.get("linewidth", 0.5),
                    zorder=style.get("zorder", 3),
                )
            )
        else:
            segments.extend(rings)
        if segments:
            ax.add_collection(
                LineCollection(
                    segments,
                    colors=style.get("edgecolor", "#444444"),
                    linewidths=style.get("linewidth", 0.6),
                    zorder=style.get("zorder", 3),
                )
            )


def draw_box_outlines(ax, boxes, transform=None):
    """Outline each N/W/S/E box in black (split antimeridian spans into two)."""
    from matplotlib.patches import Rectangle

    kw = {"fill": False, "edgecolor": "black", "linewidth": 1.5, "zorder": 6}
    if transform is not None:
        kw["transform"] = transform
    for north, west, south, east in boxes:
        height = north - south
        if west <= east:
            ax.add_patch(Rectangle((west, south), east - west, height, **kw))
        else:
            # Antimeridian: west..180 and -180..east
            ax.add_patch(Rectangle((west, south), 180.0 - west, height, **kw))
            ax.add_patch(Rectangle((-180.0, south), east + 180.0, height, **kw))
