"""Tests for weather_skills_core.plot.figure."""

import argparse

import pytest

from weather_skills_core.plot.figure import (
    DEFAULT_FONTSIZE,
    add_shared_colorbar,
    apply_style,
    axis_label,
    format_plot_date,
    format_plot_date_range,
    parse_figsize,
    parse_panel_spacing,
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


def test_parse_panel_spacing():
    assert parse_panel_spacing("0.25") == (0.25, 0.25)
    assert parse_panel_spacing("0.4,0.2") == (0.4, 0.2)
    assert parse_panel_spacing("0.4x0.2") == (0.4, 0.2)
    assert parse_panel_spacing(None) is None
    with pytest.raises(argparse.ArgumentTypeError, match="W or W,H"):
        parse_panel_spacing("wide")
    with pytest.raises(argparse.ArgumentTypeError, match=">= 0"):
        parse_panel_spacing("-0.1")


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


def test_add_shared_colorbar_custom_tick_labels():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import BoundaryNorm, ListedColormap

    bounds = [0, 10, 50, 100]
    cmap = ListedColormap(["#fff", "#080", "#040"])
    norm = BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots(figsize=(4, 3))
    mesh = ax.pcolormesh(np.arange(4).reshape(2, 2), cmap=cmap, norm=norm)
    cbar = add_shared_colorbar(
        fig, mesh, ax, "precip", ticks=bounds, labels=["dry", "low", "wet", "flood"]
    )
    fig.canvas.draw()
    texts = [t.get_text() for t in cbar.ax.get_yticklabels() if t.get_text()]
    if not texts:
        texts = [t.get_text() for t in cbar.ax.get_xticklabels() if t.get_text()]
    assert texts == ["dry", "low", "wet", "flood"]
    plt.close(fig)


def test_apply_suptitle_honors_layout_y():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from weather_skills_core.errors import UsageError
    from weather_skills_core.plot.figure import apply_suptitle, suptitle_kwargs

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    apply_suptitle(fig, "Rain", {"layout": {"suptitle": {"y": 1.05}}})
    fig.canvas.draw()
    assert fig._suptitle.get_position()[1] == pytest.approx(1.05)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    apply_suptitle(fig, "Rain", None)
    fig.canvas.draw()
    assert fig._suptitle.get_position()[1] == pytest.approx(0.98)
    plt.close(fig)

    with pytest.raises(UsageError, match="layout.suptitle.y must be a number"):
        suptitle_kwargs({"layout": {"suptitle": {"y": "up"}}})


def test_add_shared_colorbar_labelpad():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(4, 3))
    mesh = ax.pcolormesh(np.arange(4).reshape(2, 2))
    cbar = add_shared_colorbar(fig, mesh, ax, "value", labelpad=20, labelsize=22, ticksize=15)
    fig.canvas.draw()
    axis = (
        cbar.ax.yaxis if getattr(cbar, "orientation", "vertical") == "vertical" else cbar.ax.xaxis
    )
    assert axis.get_label_text() == "value"
    assert axis.labelpad == 20
    assert axis.label.get_size() == 22
    tick_sizes = [t.get_fontsize() for t in axis.get_ticklabels() if t.get_text()]
    assert tick_sizes and all(size == 15 for size in tick_sizes)
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


def test_facet_panel_titles_do_not_overlap():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from weather_skills_core.plot.figure import (
        apply_suptitle,
        facet_figure,
        settle_figure,
        wrap_axes_title,
    )

    fig, axes = facet_figure(1, 2, figsize=(8, 4), despine=True)
    caption = "ECMWF ENS mean (init 2026-08-01), IRPS interpolated"
    for ax in axes.flat:
        ax.set_title(wrap_axes_title(ax, caption))
    apply_suptitle(fig, "Ghana August 2026 total precipitation")
    settle_figure(fig)
    renderer = fig.canvas.get_renderer()
    left, right = (ax.title.get_window_extent(renderer) for ax in axes.flat)
    assert left.x1 <= right.x0 + 1.0
    sup = fig._suptitle.get_window_extent(renderer)
    assert sup.ymin >= max(left.ymax, right.ymax) - 2.0
    plt.close(fig)


def test_crowded_colorbar_labels_rotate_without_dropping_ticks():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from weather_skills_core.plot.figure import facet_figure, settle_figure

    bounds = [0, 1, 2, 5, 10, 15, 20, 30, 40, 50, 75, 100, 500, 2000]
    fig, axes = facet_figure(1, 2, figsize=(5, 3), despine=True)
    mesh = axes[0, 0].pcolormesh(np.arange(4).reshape(2, 2))
    add_shared_colorbar(fig, mesh, axes.ravel(), "precip", ticks=bounds, location="bottom")
    settle_figure(fig)
    cbar = next(ax for ax in fig.axes if ax.get_label() == "<colorbar>")
    labels = [tick for tick in cbar.get_xticklabels() if tick.get_text()]
    assert [tick.get_text() for tick in labels] == [str(b) for b in bounds]
    assert all(tick.get_rotation() == 45 for tick in labels)
    plt.close(fig)
