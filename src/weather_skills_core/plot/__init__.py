"""Plot spec, matplotlib compiler, and figure chrome.

The ``plot`` extra (matplotlib, seaborn, cartopy) is required to compile a
figure. Import the submodule you need; this package does not pull matplotlib
on ``import weather_skills_core.plot``.

=====================  ========================================================
Module                 Role
=====================  ========================================================
:mod:`.spec`           Canonical JSON spec (``FLAG_TO_SPEC``, overlay/resolve)
:mod:`.style`          Colormaps, user style files, cmap/norm
:mod:`.figure`         Dates, figsize, colorbar chrome, PNG save
:mod:`.mpl`            Artist allowlists, axes/annotations/shapes, rc
:mod:`.geo`            Cartopy Natural Earth overlays
:mod:`.layers`         One map renderer (heatmap/contour/quiver/scatter)
:mod:`.recipes`        Compare/verify grids, line figures, mediograms
:mod:`.compile`        ``compile_figure`` plus map/timeseries helpers
:mod:`.export`         PNG + sidecar write
=====================  ========================================================

Deprecated top-level modules (``weather_skills_core.plot_spec``, …,
``weather_skills_core.figure``) re-export these submodules.
"""
