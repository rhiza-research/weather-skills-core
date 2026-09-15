"""Tests for weather_skills_core.figure."""

import argparse

import pytest

from weather_skills_core.figure import (
    DEFAULT_FONTSIZE,
    add_shared_colorbar,
    apply_style,
    parse_figsize,
    resolve_figsize,
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


def test_apply_style_sets_rcparams():
    import matplotlib as mpl

    apply_style(16)
    assert mpl.rcParams["axes.labelsize"] == 16
    assert mpl.rcParams["xtick.labelsize"] == max(8, int(round(DEFAULT_FONTSIZE * 0.85)))
    assert mpl.rcParams["figure.titlesize"] == 16


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
