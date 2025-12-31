#!/usr/bin/env python3
"""Shared Matplotlib styling for figures.

Usage:
    from meshgraphnet.utils.plot_style import paper_rcparams, apply_style, savefig
    apply_style("paper")
    fig, ax = plt.subplots(...)
    ...
    savefig(fig, out_path)  # handles bbox/pad/dpi + optional formats
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, Tuple

import matplotlib as mpl


def paper_rcparams(
    *,
    base_fontsize: float = 9.0,
    font_family: str = "DejaVu Sans",
) -> dict:
    """Return rcParams for figures."""
    fs = float(base_fontsize)
    return {
        # Fonts
        "font.family": "sans-serif",
        "font.sans-serif": [font_family],
        "font.size": fs,
        "axes.titlesize": fs,
        "axes.labelsize": fs,
        "xtick.labelsize": fs - 1,
        "ytick.labelsize": fs - 1,
        "legend.fontsize": fs - 1,
        # Lines/markers
        "lines.linewidth": 1.4,
        "lines.markersize": 3.5,
        # Axes aesthetics
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.axisbelow": True,
        # Legend
        "legend.frameon": False,
        # Figure export
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        # Slight padding avoids cutting long axis labels in compact figures.
        "savefig.pad_inches": 0.06,
        # Make PDF/SVG text editable in vector outputs
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }


def apply_style(
    style: str = "paper",
    *,
    base_fontsize: float = 9.0,
    font_family: str = "DejaVu Sans",
) -> None:
    """Apply a named style. Currently supports: 'paper' or 'default'."""
    if style == "default":
        mpl.rcdefaults()
        return
    if style != "paper":
        raise ValueError(f"Unknown style={style!r}. Use 'paper' or 'default'.")
    mpl.rcParams.update(
        paper_rcparams(base_fontsize=base_fontsize, font_family=font_family)
    )


@contextmanager
def style_context(
    style: str = "paper",
    *,
    base_fontsize: float = 9.0,
    font_family: str = "DejaVu Sans",
) -> Iterator[None]:
    """Context manager to apply style temporarily."""
    with mpl.rc_context(
        paper_rcparams(base_fontsize=base_fontsize, font_family=font_family)
    ):
        if style == "default":
            with mpl.rc_context():
                yield
        elif style == "paper":
            yield
        else:
            raise ValueError(f"Unknown style={style!r}. Use 'paper' or 'default'.")


def savefig(
    fig,
    out_path: Path,
    *,
    formats: Sequence[str] = ("png",),
) -> None:
    """Save a figure with tight bbox and multiple formats.

    If out_path has a suffix, it is ignored and replaced by each format.
    """
    out_path = Path(out_path)
    stem = out_path.with_suffix("")  # remove suffix if present
    for fmt in formats:
        fmt = fmt.lstrip(".").lower()
        fig.savefig(stem.with_suffix(f".{fmt}"))
