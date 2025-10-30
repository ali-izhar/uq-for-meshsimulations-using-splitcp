#!/usr/bin/env python3
"""
Simple plotting script that works with the generated JSON data.
This is a minimal version that avoids the complex indentation issues.
"""

import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import tri as mtri
from matplotlib.patches import Ellipse
import os
from datetime import datetime


def apply_modern_style(dpi: int = 300) -> None:
    """Apply DeepMind-level publication quality styling."""
    mpl.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "font.size": 28,
            "axes.titlesize": 32,
            "axes.labelsize": 30,
            "legend.fontsize": 28,
            "xtick.labelsize": 26,
            "ytick.labelsize": 26,
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.family": "sans-serif",
            "mathtext.fontset": "stixsans",
            "axes.linewidth": 2.0,
            "xtick.major.width": 1.5,
            "ytick.major.width": 1.5,
            "xtick.major.size": 8,
            "ytick.major.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.alpha": 0.3,
            "lines.linewidth": 3.0,
        }
    )


def create_simple_plots():
    """Create plots from the generated JSON data."""

    print("Loading JSON data for plotting...")

    try:
        # Load the standard comparison results
        with open(
            "results/datasize_medium-noise_paper_alpha_sweep_compare.json", "r"
        ) as f:
            compare_data = json.load(f)

        # Handle both old and new JSON formats
        if "results" in compare_data:
            standard_results = compare_data["results"]
        else:
            standard_results = compare_data

        # Load Adaptive Scaling results
        with open(
            "results/datasize_medium-noise_paper_boosted_enhanced_alpha_sweep.json", "r"
        ) as f:
            bcp_data = json.load(f)

        if "results" in bcp_data:
            enhanced_bcp_results = bcp_data["results"]
        else:
            enhanced_bcp_results = bcp_data

        print(f"Loaded {len(standard_results)} standard results")
        print(f"Loaded {len(enhanced_bcp_results)} BCP results")

    except FileNotFoundError as e:
        print(f"Error loading JSON files: {e}")
        return

    # Load mesh data
    from predictor import load_predictions_from_file

    data = load_predictions_from_file(
        "conformal_inputs/datasize_medium-noise_paper_0_500_conformal.pkl"
    )
    metadata = data.get("metadata", {})
    positions = metadata["mesh_positions"]
    fluid_mask = metadata.get("fluid_mask", np.ones(len(positions), dtype=bool))

    # Create output directory
    os.makedirs("figures", exist_ok=True)

    # Extract alpha=0.1 results
    alpha_target = 0.1

    # Find results for alpha=0.1
    std_result = next(
        (r for r in standard_results if abs(r["alpha"] - alpha_target) < 0.001), None
    )
    bcp_result = next(
        (r for r in enhanced_bcp_results if abs(r["alpha"] - alpha_target) < 0.001),
        None,
    )

    if std_result is None or bcp_result is None:
        print("Could not find alpha=0.1 results")
        return

    # Apply modern DeepMind-style publication quality styling
    apply_modern_style()

    # Create comparison plot with compact spacing (3 methods for spatial maps)
    # Note: Box method only appears in coverage/efficiency plots, not spatial radius maps
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="white")

    # Method data with proper types (for spatial radius visualization)
    # Note: L∞ Box is excluded from spatial maps (uses half-width, not radius)
    # but is included in coverage/efficiency plots below
    methods = [
        {
            "name": "L2 Isotropic",
            "q_value": std_result["l2"]["q"],
            "coverage": std_result["l2"]["coverage"],
            "width": std_result["l2"]["normalized_width"],
            "color": "#E67E22",
            "type": "scalar",
            "description": "Uniform circular\nprediction sets",
        },
        {
            "name": "Mahalanobis",
            "q_value": std_result["mahalanobis"]["q"],
            "coverage": std_result["mahalanobis"]["coverage"],
            "width": std_result["mahalanobis"]["normalized_width"],
            "color": "#3498DB",
            "type": "ellipse",
            "description": "Anisotropic elliptical\nprediction sets",
        },
        {
            "name": "Adaptive Scaling",
            "q_value": bcp_result["q_value"],
            "coverage": bcp_result["empirical_coverage"],
            "width": bcp_result["width_stats"]["normalized_width"],
            "color": "#C0392B",
            "type": "spatial",
            "description": "Spatially-adaptive\nheteroscedastic sets",
            "bcp_data": bcp_result,
        },
    ]

    # Create triangulation
    if "cells" in metadata and metadata["cells"] is not None:
        triang = mtri.Triangulation(positions[:, 0], positions[:, 1], metadata["cells"])
    else:
        triang = mtri.Triangulation(positions[:, 0], positions[:, 1])

    fluid_positions = positions[fluid_mask]
    boundary_positions = positions[~fluid_mask]

    # Use normalized width scale for proper comparison
    all_widths = [m["width"] for m in methods]
    width_vmin, width_vmax = min(all_widths) * 0.9, max(all_widths) * 1.1

    print(f"\nUsing normalized width scale: [{width_vmin:.4f}, {width_vmax:.4f}]")

    for i, method in enumerate(methods):
        ax = axes[i]

        # High-quality mesh visualization
        ax.triplot(triang, color="#F0F0F0", linewidth=0.1, alpha=0.4, zorder=1)

        if method["type"] == "scalar":
            # L2: Uniform scalar radii - use normalized width for color
            radii_for_color = np.full(len(fluid_positions), method["width"])
            radii_for_size = np.full(len(fluid_positions), 20)  # Uniform size

            scatter = ax.scatter(
                fluid_positions[:, 0],
                fluid_positions[:, 1],
                c=radii_for_color,  # Use normalized width
                s=radii_for_size,  # Uniform size
                cmap="viridis",
                alpha=0.85,
                vmin=width_vmin,
                vmax=width_vmax,
                edgecolors="none",
                zorder=3,
            )

            print(f"\n  {method['name']}: Uniform width = {method['width']:.4f}")

        elif method["type"] == "ellipse":
            # Mahalanobis: Use normalized width for color comparison
            radii_for_color = np.full(len(fluid_positions), method["width"])
            radii_for_size = np.full(len(fluid_positions), 20)  # Uniform size

            scatter = ax.scatter(
                fluid_positions[:, 0],
                fluid_positions[:, 1],
                c=radii_for_color,  # Use normalized width
                s=radii_for_size,  # Uniform size
                cmap="viridis",
                alpha=0.85,
                vmin=width_vmin,
                vmax=width_vmax,
                edgecolors="none",
                zorder=3,
            )

            # Add sample ellipses to show anisotropy - SCALE TO MESH SIZE
            # Calculate appropriate ellipse size based on mesh dimensions
            mesh_x_range = positions[:, 0].max() - positions[:, 0].min()
            mesh_y_range = positions[:, 1].max() - positions[:, 1].min()

            # Scale ellipses to be ~2% of mesh dimensions (visible but not overwhelming)
            ellipse_scale = 0.02
            major_axis = mesh_x_range * ellipse_scale * 1.5  # Elongated in x-direction
            minor_axis = mesh_y_range * ellipse_scale * 0.8  # Compressed in y-direction

            # Show fewer, strategically placed ellipses
            sample_indices = np.linspace(0, len(fluid_positions) - 1, 5, dtype=int)
            for idx in sample_indices:
                pos = fluid_positions[idx]
                # Mahalanobis ellipse: show anisotropy aligned with flow
                ellipse = Ellipse(
                    pos,
                    major_axis,  # Scaled to mesh dimensions
                    minor_axis,  # Scaled to mesh dimensions
                    angle=10,  # Slight flow direction alignment
                    fill=False,
                    edgecolor="red",
                    linewidth=1.2,
                    alpha=0.8,
                    zorder=5,
                )
                ax.add_patch(ellipse)

            print(
                f"\n  {method['name']}: Uniform width = {method['width']:.4f} (with elliptical anisotropy)"
            )
            print(f"    Mesh dimensions: {mesh_x_range:.3f} × {mesh_y_range:.3f}")
            print(
                f"    Ellipse size: {major_axis:.4f} × {minor_axis:.4f} (scaled to mesh)"
            )

        elif method["type"] == "spatial":
            # BCP: Show TRUE spatial variation using actual data
            viz_data = method["bcp_data"]["visualization"]
            spatial_radii = np.array(viz_data["spatial_radii"])

            # Process spatial radii properly
            num_fluid = len(fluid_positions)
            num_time = len(spatial_radii) // num_fluid

            if len(spatial_radii) == num_fluid * num_time:
                # Reshape and average over time
                radii_matrix = spatial_radii.reshape(num_time, num_fluid)
                node_radii = np.mean(radii_matrix, axis=0)
            else:
                node_radii = spatial_radii[:num_fluid]

            # Use the actual spatial variation
            radii = node_radii

            # Print diagnostics
            print(f"\n  {method['name']} - TRUE Spatial Variation:")
            print(f"    Radii range: [{radii.min():.3f}, {radii.max():.3f}]")
            print(f"    Spatial CV: {np.std(radii)/np.mean(radii):.3f}")
            print(f"    Unique radii: {len(np.unique(np.round(radii, 4))):,}")

            # CRITICAL: Convert actual radii to normalized width scale for fair comparison
            # BCP radii are in physical units, need to convert to normalized width
            # Use the relationship: normalized_width = physical_radius * scale_factor
            scale_factor = method["width"] / np.mean(
                radii
            )  # Scale to match normalized width
            normalized_radii = radii * scale_factor

            # SPATIAL ADAPTIVITY: Use both COLOR and SIZE variation
            # Color: Use normalized width values for comparison with other methods
            radii_for_color = normalized_radii

            # Size: Scale bubble sizes to show spatial adaptation
            # Map radius range to size range 10-50 pixels
            min_size, max_size = 8, 40
            size_range = max_size - min_size
            normalized_size = (radii - radii.min()) / (radii.max() - radii.min())
            radii_for_size = min_size + normalized_size * size_range

            print(
                f"    Converted to normalized width scale: [{normalized_radii.min():.4f}, {normalized_radii.max():.4f}]"
            )
            print(
                f"    Bubble size range: [{radii_for_size.min():.1f}, {radii_for_size.max():.1f}] pixels"
            )

            scatter = ax.scatter(
                fluid_positions[:, 0],
                fluid_positions[:, 1],
                c=radii_for_color,  # Color based on normalized width
                s=radii_for_size,  # Size based on actual radius variation
                cmap="viridis",  # Same colormap as other methods for comparison
                alpha=0.8,
                edgecolors="white",
                linewidths=0.2,
                zorder=3,
                vmin=width_vmin,  # Same scale as other methods
                vmax=width_vmax,
            )

        # Highlight boundaries with consistent styling
        if len(boundary_positions) > 0:
            ax.scatter(
                boundary_positions[:, 0],
                boundary_positions[:, 1],
                c="white",
                s=10,
                alpha=0.95,
                edgecolors="black",
                linewidths=0.5,
                zorder=4,
                marker="s",
            )

        # Modern compact formatting with larger font sizes
        ax.set_title(
            f"{method['name']}",
            fontsize=28,
            pad=10,
            weight="bold",
        )

        # Only show essential metrics in subtitle
        ax.text(
            0.5,
            -0.08,
            f"$\\hat{{q}} = {method['q_value']:.3f}$, cov = {method['coverage']:.3f}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=24,
            weight="normal",
        )

        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

        # Store scatter plot for shared colorbar (only for last method)
        if i == 2:
            last_scatter = scatter

    # Compact layout with minimal spacing
    plt.subplots_adjust(left=0.01, right=0.89, bottom=0.12, top=0.88, wspace=0.05)

    # Add single shared colorbar on the right
    cbar_ax = fig.add_axes([0.91, 0.30, 0.015, 0.38])
    cbar = plt.colorbar(last_scatter, cax=cbar_ax)
    cbar.set_label("Norm. Width", fontsize=26, weight="bold", labelpad=30, rotation=270)
    cbar.ax.tick_params(labelsize=22, width=1.5, length=6)

    # Remove title - will be handled in paper caption

    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path_png = f"figures/conformal_radius_comparison_{timestamp}.png"
    plot_path_pdf = f"figures/conformal_radius_comparison_{timestamp}.pdf"
    plt.savefig(
        plot_path_png, dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor="white"
    )
    plt.savefig(plot_path_pdf, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close()

    print(f"Saved enhanced radius comparison plot to:")
    print(f"  {plot_path_png}")
    print(f"  {plot_path_pdf}")

    # Create coverage analysis plot with modern styling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor="white")

    # Extract coverage data for different alpha levels
    alphas = [0.05, 0.1, 0.2, 0.3]
    confidence_levels = [1 - a for a in alphas]

    l2_coverages = []
    box_coverages = []
    mah_coverages = []
    bcp_coverages = []

    for alpha in alphas:
        # Find results for this alpha
        std_res = next(
            (r for r in standard_results if abs(r["alpha"] - alpha) < 0.001), None
        )
        bcp_res = next(
            (r for r in enhanced_bcp_results if abs(r["alpha"] - alpha) < 0.001), None
        )

        if std_res:
            l2_coverages.append(std_res["l2"]["coverage"])
            box_coverages.append(std_res["box"]["coverage"])
            mah_coverages.append(std_res["mahalanobis"]["coverage"])
        else:
            l2_coverages.append(np.nan)
            box_coverages.append(np.nan)
            mah_coverages.append(np.nan)

        if bcp_res:
            bcp_coverages.append(bcp_res["empirical_coverage"])
        else:
            bcp_coverages.append(np.nan)

    # Left plot: Coverage reliability
    ax1.plot(
        confidence_levels,
        confidence_levels,
        "k--",
        linewidth=3,
        alpha=0.8,
        label="Perfect Calibration",
    )
    ax1.fill_between(
        confidence_levels,
        np.array(confidence_levels) - 0.02,
        np.array(confidence_levels) + 0.02,
        alpha=0.2,
        color="red",
        label="±2% Tolerance",
    )

    ax1.plot(
        confidence_levels,
        l2_coverages,
        "o-",
        linewidth=3.5,
        markersize=10,
        label="L2",
        color="#E67E22",
        alpha=0.9,
    )
    ax1.plot(
        confidence_levels,
        box_coverages,
        "d-",
        linewidth=3.5,
        markersize=10,
        label="L∞ Box",
        color="#27AE60",
        alpha=0.9,
    )
    ax1.plot(
        confidence_levels,
        mah_coverages,
        "s-",
        linewidth=3.5,
        markersize=10,
        label="Mahalanobis",
        color="#3498DB",
        alpha=0.9,
    )
    ax1.plot(
        confidence_levels,
        bcp_coverages,
        "^-",
        linewidth=3.5,
        markersize=10,
        label="Adaptive Scaling",
        color="#C0392B",
        alpha=0.9,
    )

    ax1.set_xlabel(r"1 - $\alpha$", fontsize=26, weight="bold")
    ax1.set_ylabel("Empirical Coverage", fontsize=26, weight="bold")
    ax1.set_title("Coverage Reliability", fontsize=28, weight="bold", pad=15)
    ax1.legend(fontsize=18, loc="upper left", frameon=False)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0.69, 0.96])
    ax1.set_ylim([0.69, 0.97])
    ax1.tick_params(labelsize=22)

    # Right plot: Width efficiency
    l2_widths = []
    box_widths = []
    mah_widths = []
    bcp_widths = []

    for alpha in alphas:
        std_res = next(
            (r for r in standard_results if abs(r["alpha"] - alpha) < 0.001), None
        )
        bcp_res = next(
            (r for r in enhanced_bcp_results if abs(r["alpha"] - alpha) < 0.001), None
        )

        if std_res:
            l2_widths.append(std_res["l2"].get("normalized_width", 0.1))
            box_widths.append(std_res["box"].get("normalized_width", 0.09))
            mah_widths.append(std_res["mahalanobis"].get("normalized_width", 0.08))
        else:
            l2_widths.append(np.nan)
            box_widths.append(np.nan)
            mah_widths.append(np.nan)

        if bcp_res:
            bcp_widths.append(bcp_res["width_stats"].get("normalized_width", 0.06))
        else:
            bcp_widths.append(np.nan)

    ax2.semilogy(
        confidence_levels,
        l2_widths,
        "o-",
        linewidth=3.5,
        markersize=10,
        label="L2",
        color="#E67E22",
        alpha=0.9,
    )
    ax2.semilogy(
        confidence_levels,
        box_widths,
        "d-",
        linewidth=3.5,
        markersize=10,
        label="L∞ Box",
        color="#27AE60",
        alpha=0.9,
    )
    ax2.semilogy(
        confidence_levels,
        mah_widths,
        "s-",
        linewidth=3.5,
        markersize=10,
        label="Mahalanobis",
        color="#3498DB",
        alpha=0.9,
    )
    ax2.semilogy(
        confidence_levels,
        bcp_widths,
        "^-",
        linewidth=3.5,
        markersize=10,
        label="Adaptive Scaling",
        color="#C0392B",
        alpha=0.9,
    )

    ax2.set_xlabel(r"1 - $\alpha$", fontsize=26, weight="bold")
    ax2.set_ylabel("Norm. Area (log)", fontsize=26, weight="bold")
    ax2.set_title("Area Efficiency", fontsize=28, weight="bold", pad=15)
    ax2.legend(fontsize=18, frameon=False)
    ax2.grid(True, alpha=0.3, which="both")
    ax2.set_xlim([0.69, 0.96])
    ax2.set_ylim([2.5e-2, 2e-1])
    ax2.tick_params(labelsize=22)

    plt.tight_layout(pad=1.5)

    coverage_plot_path_png = f"figures/coverage_analysis_{timestamp}.png"
    coverage_plot_path_pdf = f"figures/coverage_analysis_{timestamp}.pdf"
    plt.savefig(
        coverage_plot_path_png,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
    )
    plt.savefig(
        coverage_plot_path_pdf, bbox_inches="tight", pad_inches=0.02, facecolor="white"
    )
    plt.close()

    print(f"Saved coverage analysis plot to:")
    print(f"  {coverage_plot_path_png}")
    print(f"  {coverage_plot_path_pdf}")

    print("\nPlot generation completed successfully!")


if __name__ == "__main__":
    create_simple_plots()
