#!/usr/bin/env python3
"""
Generate publication-quality plots matching paper Figures 8 & 9.

Creates coverage reliability and area efficiency plots for conformal prediction
experiments, matching the exact style from the paper.

Usage:
    python plot_results.py -d cylinder_medium_noise
    python plot_results.py -d flag_medium
"""

import argparse
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path


def apply_paper_style():
    """Apply publication-quality styling matching paper figures."""
    mpl.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "stixsans",
            "axes.linewidth": 1.0,
            "grid.alpha": 0.3,
            "lines.linewidth": 2.0,
        }
    )


def plot_geometry_comparison(
    dataset_name, results_dir="results/conformal_preds", output_dir="results/figures"
):
    """
    Plot coverage reliability and area efficiency (paper Figures 8 & 9 style).

    Left: Coverage vs confidence with ±2% tolerance bands
    Right: Normalized area on log scale showing efficiency
    """

    # Load comparison JSON and adaptive results
    comp_file = Path(results_dir) / f"{dataset_name}_alpha_sweep_compare.json"
    adaptive_file = (
        Path(results_dir) / f"{dataset_name}_boosted_enhanced_alpha_sweep.json"
    )

    if not comp_file.exists():
        print(f"ERROR: {comp_file} not found")
        return

    with open(comp_file) as f:
        comp_data = json.load(f)

    # Load adaptive if available
    adaptive_data = None
    if adaptive_file.exists():
        with open(adaptive_file) as f:
            adaptive_data = json.load(f)

    apply_paper_style()

    # Extract data for all geometries
    alphas = [entry["alpha"] for entry in comp_data]
    confidences = [1 - a for a in alphas]  # 1 - alpha for x-axis

    # Standard geometries
    l2_cov = [entry["l2"]["coverage"] for entry in comp_data]
    l2_area = [entry["l2"]["normalized_width"] for entry in comp_data]

    mah_cov = [entry["mahalanobis"]["coverage"] for entry in comp_data]
    mah_area = [entry["mahalanobis"]["normalized_width"] for entry in comp_data]

    box_cov = [entry["box"]["coverage"] for entry in comp_data]
    box_area = [entry["box"]["normalized_width"] for entry in comp_data]

    # Adaptive if available
    if adaptive_data:
        adaptive_cov = [entry["empirical_coverage"] for entry in adaptive_data]
        adaptive_area = [
            entry["width_stats"]["normalized_width"] for entry in adaptive_data
        ]

    # Create figure matching paper style
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # LEFT PLOT: Coverage Reliability
    # Perfect calibration line
    ax1.plot(
        confidences,
        confidences,
        "k--",
        linewidth=2,
        alpha=0.6,
        label="Perfect Calibration",
        zorder=1,
    )

    # ±2% tolerance band
    ax1.fill_between(
        confidences,
        np.array(confidences) - 0.02,
        np.array(confidences) + 0.02,
        alpha=0.15,
        color="red",
        label="±2% Tolerance",
        zorder=0,
    )

    # Geometry curves
    ax1.plot(
        confidences,
        l2_cov,
        "o-",
        label="L2",
        linewidth=2,
        markersize=6,
        color="#E67E22",
    )
    ax1.plot(
        confidences,
        box_cov,
        "^-",
        label="L∞ Box",
        linewidth=2,
        markersize=6,
        color="#27AE60",
    )
    ax1.plot(
        confidences,
        mah_cov,
        "s-",
        label="Mahalanobis",
        linewidth=2,
        markersize=6,
        color="#3498DB",
    )

    if adaptive_data:
        ax1.plot(
            confidences,
            adaptive_cov,
            "d-",
            label="Adaptive Scaling",
            linewidth=2,
            markersize=6,
            color="#C0392B",
        )

    ax1.set_xlabel(r"$1 - \alpha$", fontsize=12)
    ax1.set_ylabel("Empirical Coverage", fontsize=12)
    ax1.set_title("Coverage Reliability", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9, loc="lower right", framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle=":")
    ax1.set_xlim([min(confidences) - 0.02, max(confidences) + 0.02])

    # RIGHT PLOT: Area Efficiency (log scale)
    ax2.semilogy(
        confidences,
        l2_area,
        "o-",
        label="L2",
        linewidth=2,
        markersize=6,
        color="#E67E22",
    )
    ax2.semilogy(
        confidences,
        box_area,
        "^-",
        label="L∞ Box",
        linewidth=2,
        markersize=6,
        color="#27AE60",
    )
    ax2.semilogy(
        confidences,
        mah_area,
        "s-",
        label="Mahalanobis",
        linewidth=2,
        markersize=6,
        color="#3498DB",
    )

    if adaptive_data:
        ax2.semilogy(
            confidences,
            adaptive_area,
            "d-",
            label="Adaptive Scaling",
            linewidth=2,
            markersize=6,
            color="#C0392B",
        )

    ax2.set_xlabel(r"$1 - \alpha$", fontsize=12)
    ax2.set_ylabel("Norm. Area (log)", fontsize=12)
    ax2.set_title("Area Efficiency", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax2.grid(True, alpha=0.3, which="both", linestyle=":")

    plt.tight_layout(pad=2.0)

    # Save PNG only
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_png = Path(output_dir) / f"{dataset_name}_coverage_analysis.png"

    plt.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="white")

    print(f"Saved: {output_png}")

    plt.close()


def plot_adaptive_comparison(
    dataset_name, results_dir="results/conformal_preds", output_dir="results/figures"
):
    """Plot comparison of basic vs enhanced adaptive CP."""

    # Load JSONs
    basic_file = Path(results_dir) / f"{dataset_name}_boosted_basic_alpha_sweep.json"
    enhanced_file = (
        Path(results_dir) / f"{dataset_name}_boosted_enhanced_alpha_sweep.json"
    )

    if not basic_file.exists() or not enhanced_file.exists():
        print(f"ERROR: Adaptive results not found")
        return

    with open(basic_file) as f:
        basic_data = json.load(f)
    with open(enhanced_file) as f:
        enhanced_data = json.load(f)

    # Extract data
    alphas = [entry["alpha"] for entry in basic_data]

    basic_cov = [entry["empirical_coverage"] for entry in basic_data]
    basic_area = [
        entry["width_stats"].get("normalized_width", entry["mean_area"])
        for entry in basic_data
    ]

    enhanced_cov = [entry["empirical_coverage"] for entry in enhanced_data]
    enhanced_area = [
        entry["width_stats"].get("normalized_width", entry["mean_area"])
        for entry in enhanced_data
    ]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Coverage
    ax1.plot(alphas, basic_cov, "o-", label="Basic (p=5)", linewidth=2, markersize=8)
    ax1.plot(
        alphas, enhanced_cov, "s-", label="Enhanced (p=17)", linewidth=2, markersize=8
    )
    targets = [1 - a for a in alphas]
    ax1.plot(alphas, targets, "k--", label="Target", linewidth=2, alpha=0.5)

    ax1.set_xlabel("Miscoverage Level (alpha)", fontsize=12)
    ax1.set_ylabel("Empirical Coverage", fontsize=12)
    ax1.set_title("Adaptive CP: Coverage Comparison", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Area Reduction
    area_reduction = [(b - e) / b * 100 for b, e in zip(basic_area, enhanced_area)]

    ax2.plot(alphas, area_reduction, "o-", linewidth=2, markersize=8, color="green")
    ax2.axhline(y=0, color="k", linestyle="--", linewidth=1, alpha=0.5)

    ax2.set_xlabel("Miscoverage Level (alpha)", fontsize=12)
    ax2.set_ylabel("Area Reduction (%)", fontsize=12)
    ax2.set_title("Enhanced vs Basic Features", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save PNG only
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_png = Path(output_dir) / f"{dataset_name}_adaptive_comparison.png"
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_png}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot conformal prediction results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Plot geometry comparison (L2, Mahalanobis, Box)
    python plot_results.py comparison -d cylinder_medium_noise
    
    # Plot adaptive CP (basic vs enhanced features)
    python plot_results.py adaptive -d cylinder_medium_noise
    
    # Plot all
    python plot_results.py all -d cylinder_medium_noise
    
    # Flag dataset
    python plot_results.py all -d flag_medium
        """,
    )

    parser.add_argument(
        "--mode",
        "-m",
        choices=["comparison", "adaptive", "all"],
        default="all",
        help="What to plot (default: all)",
    )

    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        required=True,
        help="Dataset name (e.g., cylinder_medium_noise, flag_medium)",
    )

    parser.add_argument(
        "--results-dir",
        "-r",
        type=str,
        default="results/conformal_preds",
        help="Results directory (default: results/conformal_preds)",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="results/figures",
        help="Output directory for plots (default: results/figures)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print(f"PLOTTING CONFORMAL PREDICTION RESULTS: {args.dataset}")
    print("=" * 70)
    print()

    if args.mode in ["comparison", "all"]:
        print("Creating geometry comparison plot...")
        plot_geometry_comparison(args.dataset, args.results_dir, args.output_dir)

    if args.mode in ["adaptive", "all"]:
        print("Creating adaptive CP comparison plot...")
        plot_adaptive_comparison(args.dataset, args.results_dir, args.output_dir)

    print()
    print("=" * 70)
    print(f"Plots saved to: {args.output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
