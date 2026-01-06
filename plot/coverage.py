#!/usr/bin/env python3
"""
Coverage reliability and size efficiency plots.

Creates 1x2 grid:
- Left: Coverage reliability (empirical vs target confidence)
- Right: Area efficiency (normalized prediction set area, log-scale)

Uses serif fonts matching paper styling.

Data source (from plot/tables.py):
    - Coverage: Computed on test/eval split (held-out data)
    - size_norm: Prediction set volume / ground truth domain range product
      (interpretable as fraction of total domain; lower = tighter bounds)

Recommended CSV files (matching conformal/RESULTS.md):
    - Cylinder: paper/tables_generated_big_xgboost_sigcap098/cylinder_table.csv
      (uses conformal/_out_cylinder_200k_big_inregime_xgbq_physfull)
    - Flag: paper/tables_generated_big_xgboost_sigcap098/flag_table.csv
      (uses conformal/_out_flag_200k_big_inregime_xgboost_sigcap098)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from plot.style import get_mpl, apply_paper_style, PUBLICATION_DPI


# Method display names and colors
METHOD_CONFIG = {
    "l2": {"label": "L2", "color": "#E69F00", "marker": "o"},
    "linf": {"label": r"L$_\infty$ Box", "color": "#56B4E9", "marker": "^"},
    "mah": {"label": "Mahalanobis", "color": "#0072B2", "marker": "s"},
    "adapt": {"label": "Adaptive", "color": "#D55E00", "marker": "D"},
    "cw_adapt": {"label": "CW-Adaptive", "color": "#009E73", "marker": "v"},
}


def load_table_csv(csv_path: Path) -> list[dict]:
    """Load conformal results table CSV."""
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def extract_method_data(rows: list[dict], method: str, dim: int = 2):
    """Extract alpha, coverage, size_norm, and width for a given method.
    
    Args:
        rows: CSV rows
        method: Method name (l2, linf, mah, adapt)
        dim: Output dimension (2 for CylinderFlow, 3 for Flag)
    
    Returns:
        alpha, coverage, size_norm, width arrays
    """
    sub = [r for r in rows if r["method"] == method]
    sub.sort(key=lambda x: float(x["alpha"]))
    alpha = np.array([float(r["alpha"]) for r in sub])
    coverage = np.array([float(r["coverage"]) for r in sub])
    size_norm = np.array([float(r["size_norm"]) for r in sub])
    # Width = V^(1/d) - more interpretable linear-scale metric
    width = size_norm ** (1.0 / dim)
    return alpha, coverage, size_norm, width


def fig_coverage_efficiency(
    csv_path: Path,
    out_png: Path,
    dataset_name: str = "",
    *,
    dim: int = 2,
    dpi: int = PUBLICATION_DPI,
) -> None:
    """
    Create 1x2 coverage reliability and width efficiency plot.

    Left panel: Coverage Reliability
        - X-axis: Target confidence (1-α)
        - Y-axis: Empirical coverage
        - Perfect calibration line (dashed)
        - ±2% tolerance band (shaded)
        - All four methods plotted

    Right panel: Width Efficiency
        - X-axis: Target confidence (1-α)
        - Y-axis: Normalized width = V^(1/d) (log scale)
        - Lower is better (tighter prediction sets)
        - Width is more interpretable than volume for comparing methods

    Args:
        csv_path: Path to conformal results CSV (e.g., cylinder_table.csv)
        out_png: Output PNG path
        dataset_name: Dataset name for title (e.g., "Cylinder", "Flag")
        dim: Output dimension (2 for CylinderFlow, 3 for Flag)
        dpi: Output resolution
    """
    _, plt, *_ = get_mpl()
    apply_paper_style(dpi)

    # Load data
    rows = load_table_csv(csv_path)

    # Figure setup: 1x2 grid (extra bottom margin for footnote)
    fig, (ax_cov, ax_eff) = plt.subplots(1, 2, figsize=(7.0, 2.5), facecolor="white")
    fig.subplots_adjust(wspace=0.35, left=0.09, right=0.98, bottom=0.20, top=0.88)

    # === Left panel: Coverage Reliability ===
    # Perfect calibration line and tolerance band
    xs = np.linspace(0.68, 0.97, 100)
    ax_cov.plot(
        xs, xs, color="0.3", linestyle="--", linewidth=1.2, label="Perfect Calibration"
    )
    ax_cov.fill_between(
        xs,
        xs - 0.02,
        xs + 0.02,
        color="#D55E00",
        alpha=0.15,
        label=r"$\pm$2% Tolerance",
    )

    # Determine available methods
    available_methods = list(set(r["method"] for r in rows))
    plot_methods = [m for m in METHOD_CONFIG.keys() if m in available_methods]

    # Plot each method
    for method in plot_methods:
        cfg = METHOD_CONFIG[method]
        alpha, coverage, _, _ = extract_method_data(rows, method, dim=dim)
        x = 1.0 - alpha  # Target confidence
        ax_cov.plot(
            x,
            coverage,
            marker=cfg["marker"],
            color=cfg["color"],
            linewidth=1.5,
            markersize=6,
            label=cfg["label"],
        )

    ax_cov.set_xlabel(r"$1 - \alpha$", fontsize=9)
    ax_cov.set_ylabel("Empirical Coverage", fontsize=9)
    ax_cov.set_title("Coverage Reliability", fontsize=10, fontweight="medium")
    ax_cov.set_xlim(0.68, 0.97)
    ax_cov.set_ylim(0.68, 0.97)
    # Use 0.1 intervals for more compact vertical display
    ax_cov.set_xticks([0.7, 0.8, 0.9])
    ax_cov.set_yticks([0.7, 0.8, 0.9])
    ax_cov.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax_cov.grid(True, alpha=0.3, linewidth=0.5)

    # === Right panel: Width Efficiency ===
    # Width = V^(1/d) is a linear-scale metric that shows Adaptive's advantage clearly.
    for method in plot_methods:
        cfg = METHOD_CONFIG[method]
        alpha, _, _, width = extract_method_data(rows, method, dim=dim)
        x = 1.0 - alpha  # Target confidence
        ax_eff.plot(
            x,
            width,
            marker=cfg["marker"],
            color=cfg["color"],
            linewidth=1.5,
            markersize=6,
            label=cfg["label"],
        )

    ax_eff.set_xlabel(r"$1 - \alpha$", fontsize=9)
    ax_eff.set_ylabel(r"Width $W = V^{1/d}$$^\dagger$", fontsize=9)
    ax_eff.set_title("Width Efficiency", fontsize=10, fontweight="medium")
    ax_eff.set_yscale("log")
    ax_eff.set_xlim(0.68, 0.97)
    # Use 0.1 intervals for more compact display
    ax_eff.set_xticks([0.7, 0.8, 0.9])
    ax_eff.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax_eff.grid(True, alpha=0.3, linewidth=0.5, which="both")

    # Add footnote matching paper tables
    fig.text(
        0.5,
        0.01,
        r"$^\dagger$Width $W = V^{1/d}$ normalized by L2 baseline; lower = tighter prediction sets",
        ha="center",
        fontsize=7,
        style="italic",
        color="0.4",
    )

    # Save
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_png, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.02
    )
    plt.close(fig)
    print(f"Wrote: {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate coverage reliability and width efficiency plots",
        epilog="""
Recommended usage:
  Cylinder (d=2):
    --csv paper/tables_final/cylinder_table.csv --dim 2
  Flag (d=3):
    --csv paper/tables_final/flag_table.csv --dim 3
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--csv", required=True, help="Conformal results CSV file")
    ap.add_argument(
        "--out_png",
        default="paper/figures_generated/coverage_efficiency.png",
        help="Output PNG path",
    )
    ap.add_argument("--dataset", default="", help="Dataset name for title")
    ap.add_argument("--dim", type=int, default=2, help="Output dimension (2 for Cylinder, 3 for Flag)")
    args = ap.parse_args()

    fig_coverage_efficiency(
        Path(args.csv),
        Path(args.out_png),
        dataset_name=args.dataset,
        dim=args.dim,
    )


if __name__ == "__main__":
    main()
