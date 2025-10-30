#!/usr/bin/env python3
"""
Visualization utilities for standard conformal prediction geometries.

Extracts visualization data (mesh metadata, ellipse parameters, spatial coverage)
for plotting ℓ2, ℓ∞, and Mahalanobis prediction sets.
"""

import numpy as np
from typing import Dict, Optional, TYPE_CHECKING
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from conformal.predictor import ConformalPredictor


def extract_visualization_data(
    predictor: "ConformalPredictor",
    prepared_data: Dict,
    nonconformity: str,
    alpha: float,
) -> Dict:
    """
    Extract visualization data for plotting conformal prediction sets.

    Geometry-specific data:
        - ℓ2: scalar radius
        - Mahalanobis: ellipse parameters (eigenvalues, eigenvectors, axes, angle)
        - ℓ∞ box: half-width

    Args:
        predictor: Calibrated conformal predictor
        prepared_data: Data dict with fluid_mask, metadata, etc.
        nonconformity: Geometry type {'l2', 'mahalanobis', 'box'}
        alpha: Confidence level

    Returns:
        Dictionary with visualization data
    """
    data = prepared_data["data"]
    fluid_mask = prepared_data["fluid_mask"]

    viz_data = {
        "method": nonconformity,
        "alpha": alpha,
        "q_value": predictor.q_value if hasattr(predictor, "q_value") else None,
        "prediction_set_type": "scalar",
    }

    # Extract mesh metadata
    metadata = data.get("metadata", {})
    if "mesh_positions" in metadata:
        positions = metadata["mesh_positions"]
        viz_data["mesh_positions"] = positions.tolist()
        viz_data["fluid_mask"] = fluid_mask.tolist()

        if "node_types" in metadata:
            viz_data["node_types"] = metadata["node_types"].tolist()

        if "cells" in metadata and metadata["cells"] is not None:
            viz_data["mesh_cells"] = metadata["cells"].tolist()

        # Wall distances for spatial analysis
        try:
            wall_distances = compute_wall_distances(
                positions, metadata.get("node_types")
            )
            viz_data["wall_distances"] = wall_distances.tolist()
        except Exception as e:
            print(f"Warning: Could not compute wall distances: {e}")

    # Geometry-specific parameters
    if nonconformity == "mahalanobis" and hasattr(predictor, "_covariance_matrix"):
        try:
            cov_matrix = predictor._covariance_matrix
            cov_inv = predictor._covariance_inv

            # Ellipse parameters from eigendecomposition
            eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)

            viz_data.update(
                {
                    "prediction_set_type": "ellipse",
                    "covariance_matrix": cov_matrix.tolist(),
                    "eigenvalues": eigenvals.tolist(),
                    "eigenvectors": eigenvecs.tolist(),
                    "ellipse_axes": (predictor.q_value * np.sqrt(eigenvals)).tolist(),
                    "ellipse_angle": float(
                        np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                    ),
                }
            )
        except Exception as e:
            print(f"Warning: Could not extract Mahalanobis ellipse parameters: {e}")

    elif nonconformity == "l2":
        viz_data["prediction_set_type"] = "circle"
        viz_data["circle_radius"] = predictor.q_value

    elif nonconformity == "box":
        viz_data["prediction_set_type"] = "box"
        viz_data["box_half_width"] = predictor.q_value

    # Calibration residuals (sampled for efficiency)
    if (
        hasattr(predictor, "calibration_residuals")
        and predictor.calibration_residuals is not None
    ):
        residuals = predictor.calibration_residuals
        if len(residuals) > 1000:
            sample_idx = np.linspace(0, len(residuals) - 1, 1000, dtype=int)
            sampled_residuals = residuals[sample_idx]
        else:
            sampled_residuals = residuals
        viz_data["calibration_residuals_sample"] = sampled_residuals.tolist()

    # Spatial coverage analysis
    if "mesh_positions" in viz_data:
        try:
            coverage_spatial = compute_spatial_coverage_analysis(
                predictor, prepared_data, positions, fluid_mask
            )
            viz_data["spatial_coverage"] = coverage_spatial
        except Exception as e:
            print(f"Warning: Could not compute spatial coverage: {e}")

    return viz_data


def compute_wall_distances(
    positions: np.ndarray, node_types: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute distance from each node to nearest wall node.

    Args:
        positions: Node coordinates [N, 2]
        node_types: Optional node type encoding [N]

    Returns:
        Wall distances [N]
    """
    if node_types is None:
        # Fallback: identify boundary from domain edges
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()

        boundary_threshold = 0.05 * min(x_max - x_min, y_max - y_min)
        boundary_mask = (
            (positions[:, 0] <= x_min + boundary_threshold)
            | (positions[:, 0] >= x_max - boundary_threshold)
            | (positions[:, 1] <= y_min + boundary_threshold)
            | (positions[:, 1] >= y_max - boundary_threshold)
        )
        wall_positions = positions[boundary_mask]
    else:
        # Identify wall nodes from type encoding
        unique_types = np.unique(node_types)
        majority_type = np.bincount(node_types).argmax()
        wall_candidates = [t for t in unique_types if t > 0 and t != majority_type]
        wall_type = min(wall_candidates) if wall_candidates else unique_types[0]

        wall_positions = positions[node_types == wall_type]

    if len(wall_positions) == 0:
        return np.full(len(positions), 0.1)

    # k-d tree for efficient nearest-neighbor query
    tree = cKDTree(wall_positions)
    distances, _ = tree.query(positions, k=1)

    # Clip minimum distance for numerical stability
    min_dist = (
        np.percentile(distances[distances > 0], 5) if np.any(distances > 0) else 1e-3
    )
    return np.maximum(distances, min_dist)


def compute_spatial_coverage_analysis(
    predictor: "ConformalPredictor",
    prepared_data: Dict,
    positions: np.ndarray,
    fluid_mask: np.ndarray,
    n_bins: int = 6,
) -> Dict:
    """
    Analyze coverage as a function of wall distance.

    Bins fluid nodes by distance to nearest wall and computes empirical
    coverage in each bin to identify spatial variations in coverage.

    Args:
        predictor: Calibrated conformal predictor
        prepared_data: Data dict with test indices
        positions: Node coordinates [N, 2]
        fluid_mask: Boolean mask [N] for fluid nodes
        n_bins: Number of distance bins

    Returns:
        Dict with binned coverage statistics
    """
    # Get test data
    test_indices = prepared_data["test_indices"]
    test_predictions = prepared_data["predictions"][test_indices]
    test_ground_truth = prepared_data["ground_truth"][test_indices]

    # Compute coverage on test set
    test_scores = predictor.compute_nonconformity_scores(
        test_predictions, test_ground_truth, fluid_mask
    )
    covered = test_scores <= predictor.q_value

    # Reshape to [T, N_fluid]
    num_test_steps = len(test_indices)
    num_fluid = np.sum(fluid_mask)
    coverage_mesh = covered.reshape(num_test_steps, num_fluid)
    node_coverage = coverage_mesh.mean(axis=0)  # Per-node coverage

    # Wall distances for fluid nodes
    node_types = prepared_data["data"].get("metadata", {}).get("node_types")
    wall_distances = compute_wall_distances(positions, node_types)
    fluid_wall_distances = wall_distances[fluid_mask]

    # Bin by wall distance
    distance_bins = np.linspace(
        np.percentile(fluid_wall_distances, 10),
        np.percentile(fluid_wall_distances, 90),
        n_bins + 1,
    )

    bin_centers = (distance_bins[:-1] + distance_bins[1:]) / 2
    bin_coverage = []
    bin_counts = []

    for i in range(n_bins):
        in_bin = (fluid_wall_distances >= distance_bins[i]) & (
            fluid_wall_distances < distance_bins[i + 1]
        )
        if np.any(in_bin):
            bin_coverage.append(float(node_coverage[in_bin].mean()))
            bin_counts.append(int(in_bin.sum()))
        else:
            bin_coverage.append(None)
            bin_counts.append(0)

    return {
        "distance_bins": distance_bins.tolist(),
        "bin_centers": bin_centers.tolist(),
        "bin_coverage": bin_coverage,
        "bin_counts": bin_counts,
        "node_coverage": node_coverage.tolist(),
    }
