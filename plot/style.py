#!/usr/bin/env python3
"""Common plotting style for figures."""

from __future__ import annotations

from functools import lru_cache
import numpy as np


@lru_cache(maxsize=1)
def get_mpl():
    """Cached matplotlib imports."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.tri import LinearTriInterpolator
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.ticker import MaxNLocator

    return mpl, plt, mtri, LinearTriInterpolator, GridSpec, TwoSlopeNorm, MaxNLocator


def apply_paper_style(dpi: int = 350) -> None:
    """Apply serif fonts (Times-like) and set DPI."""
    mpl, *_ = get_mpl()
    mpl.rcParams.update(
        {
            # Font settings - serif to match LaTeX
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "DejaVu Serif",
                "Bitstream Vera Serif",
                "Computer Modern Roman",
            ],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            # Use LaTeX-style math
            "mathtext.fontset": "cm",  # Computer Modern for math
            # DPI settings
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            # Clean look
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            # White background
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def add_colorbar(
    fig, ax, mappable, label: str = "$|v|$ (m/s)", size: float = 0.12, pad: float = 0.04
):
    """
    Add compact colorbar to the right of axes.

    Args:
        fig: Figure object
        ax: Axes object
        mappable: The image/contour to create colorbar for
        label: Colorbar label (supports LaTeX math)
        size: Width of colorbar in inches
        pad: Padding between axes and colorbar
    """
    _, plt, _, _, _, _, MaxNLocator = get_mpl()
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    cax = make_axes_locatable(ax).append_axes("right", size=size, pad=pad)
    cb = fig.colorbar(mappable, cax=cax)
    cb.locator = MaxNLocator(nbins=4)
    cb.update_ticks()
    cb.outline.set_edgecolor("black")
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(colors="black", labelsize=8, pad=1)
    cb.set_label(label, fontsize=9, color="black")
    return cb


def robust_vminmax(x: np.ndarray, qlo: float = 0.01, qhi: float = 0.99):
    """Compute robust min/max from quantiles, ignoring NaN/inf."""
    x = np.asarray(x).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.quantile(x, [qlo, qhi])
    if not (np.isfinite(vmin) and np.isfinite(vmax) and vmin < vmax):
        vmin, vmax = x.min(), max(x.max(), x.min() + 1.0)
    return float(vmin), float(vmax)


# Standard colormaps
CMAP_VELOCITY = "inferno"
CMAP_ERROR = "coolwarm"

# Standard DPI
PUBLICATION_DPI = 350
