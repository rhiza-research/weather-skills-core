"""Tests for weather_skills_core.figure."""

import argparse

import pytest

from weather_skills_core.figure import (
    DEFAULT_FONTSIZE,
    add_shared_colorbar,
    apply_style,
    axis_label,
    format_plot_date,
    format_plot_date_range,
    parse_figsize,
    resolve_axis_label,
    resolve_figsize,
    resolve_time_axis_label,
    save_figure,
)


def test_parse_figsize():
    assert parse_figsize("10,6") == (10.0, 6.0)
    assert parse_figsize("8x5") == (8.0, 5.0)
    assert parse_figsize(None) is None
    with pytest.raises(argparse.ArgumentTypeError, match="W,H"):
        parse_figsize("wide")
    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        parse_figsize("0,4")


def test_resolve_figsize():
    assert resolve_figsize(None, (10, 6)) == (10, 6)
    assert resolve_figsize((8.0, 4.0), (10, 6)) == (8.0, 4.0)


def test_axis_label_capitalizes():
    assert axis_label("lon") == "Longitude"
    assert axis_label("valid time") == "Valid time"
    assert axis_label("total precipitation [mm]") == "Total precipitation [mm]"
    assert axis_label("Latitude") == "Latitude"


def test_resolve_axis_label_override_is_verbatim():
    import numpy as np

    assert resolve_axis_label("lon (E)", "Longitude") == "lon (E)"
    assert resolve_axis_label(None, "lon") == "Longitude"
    assert resolve_axis_label("", "Latitude") == "Latitude"
    times = np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[ns]")
    assert resolve_time_axis_label(None, "Valid time", times) == ""
    assert resolve_time_axis_label("Lead time", "Valid time", times) == "Lead time"
    assert resolve_time_axis_label(None, "step", np.array([1, 2, 3])) == "Step"


def test_format_plot_date():
    import datetime as dt

    import numpy as np

    assert format_plot_date(dt.date(2026, 9, 14)) == "14 Sept '26"
    assert format_plot_date(np.datetime64("2026-01-01")) == "1 Jan '26"
    assert format_plot_date(dt.date(2026, 10, 1), year=False) == "1 Oct"
    assert format_plot_date_range(dt.date(2026, 8, 4), dt.date(2026, 8, 10)) == "4–10 Aug '26"
    assert format_plot_date_range(dt.date(2026, 8, 28), dt.date(2026, 9, 3)) == (
        "28 Aug–3 Sept '26"
    )
    assert format_plot_date_range(dt.date(2025, 12, 28), dt.date(2026, 1, 3)) == (
        "28 Dec '25–3 Jan '26"
    )


def test_apply_style_sets_rcparams():
    import matplotlib as mpl

    apply_style(16)
    assert mpl.rcParams["axes.labelsize"] == 16
    assert mpl.rcParams["xtick.labelsize"] == max(8, int(round(DEFAULT_FONTSIZE * 0.85)))
    assert mpl.rcParams["figure.titlesize"] == 16


def test_add_shared_colorbar_labels_every_discrete_tick():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import BoundaryNorm, ListedColormap

    bounds = [2, 5, 10, 25, 50, 75, 100, 150, 200, 300]
    cmap = ListedColormap(["#ccc"] * (len(bounds) - 1))
    norm = BoundaryNorm(bounds, cmap.N)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    mesh = axes[0].pcolormesh(np.arange(4).reshape(2, 2), cmap=cmap, norm=norm)
    cbar = add_shared_colorbar(fig, mesh, axes, "precip", ticks=bounds, spacing="uniform")
    fig.canvas.draw()
    labels = [t.get_text() for t in cbar.ax.get_xticklabels() if t.get_visible() and t.get_text()]
    assert labels == ["2", "5", "10", "25", "50", "75", "100", "150", "200", "300"]
    plt.close(fig)


def test_add_shared_colorbar_and_save(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    apply_style(16)
    fig, ax = plt.subplots(figsize=(6, 4))
    mesh = ax.pcolormesh(np.arange(4).reshape(2, 2))
    cbar = add_shared_colorbar(fig, mesh, ax, "value")
    assert cbar is not None
    assert cbar.ax.get_ylabel() == "value" or cbar.ax.get_xlabel() == "value"
    out = save_figure(fig, tmp_path / "fig.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_save_figure_keeps_canvas_when_not_tight(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1])
    out = save_figure(fig, tmp_path / "sized.png", tight=False)
    img = mpimg.imread(out)
    assert img.shape[1] == 7 * 150
    assert img.shape[0] == 5 * 150
