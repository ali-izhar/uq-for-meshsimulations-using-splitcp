#!/usr/bin/env python3
"""
Utilities for loading and processing JSON results for visualization.

This module provides functions to load the enhanced JSON files saved by run_conformal.py
and extract the visualization data needed for plotting different conformal prediction
geometries (circles, ellipses, spatial variation).
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
import glob
import os
from datetime import datetime


def load_conformal_results(json_file: str) -> Dict:
    """
    Load conformal prediction results from enhanced JSON file.

    Args:
        json_file: Path to JSON file saved by run_conformal.py

    Returns:
        Dictionary containing results and metadata
    """
    with open(json_file, "r") as f:
        data = json.load(f)

    return data


def find_latest_results(results_dir: str, pattern: str = "*alpha_sweep*.json") -> str:
    """
    Find the most recent results file matching the pattern.

    Args:
        results_dir: Directory containing JSON results
        pattern: Glob pattern to match files

    Returns:
        Path to the most recent matching file
    """
    files = glob.glob(os.path.join(results_dir, pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} found in {results_dir}")

    # Sort by modification time, return newest
    return max(files, key=os.path.getmtime)


def extract_method_results(data: Dict, alpha: float, method: str) -> Optional[Dict]:
    """
    Extract results for a specific method and alpha from JSON data.

    Args:
        data: Loaded JSON data
        alpha: Target alpha value
        method: Method name ("l2", "mahalanobis", "boosted_l2")

    Returns:
        Dictionary with method results or None if not found
    """
    results = data.get("results", [])

    for result in results:
        if (
            abs(result.get("alpha", 0) - alpha) < 0.001
            and result.get("nonconformity_method") == method
        ):
            return result

    return None


def get_visualization_data(result: Dict) -> Dict:
    """
    Extract visualization data from a result dictionary.

    Args:
        result: Single result dictionary from JSON

    Returns:
        Visualization data dictionary
    """
    return result.get("visualization", {})


def create_prediction_set_data(viz_data: Dict) -> Tuple[str, np.ndarray, Dict]:
    """
    Create prediction set data for visualization from JSON visualization data.

    Args:
        viz_data: Visualization data from JSON

    Returns:
        Tuple of (prediction_set_type, data_array, metadata)
        - prediction_set_type: "scalar", "ellipse", "spatial", "box"
        - data_array: Numpy array with appropriate data for visualization
        - metadata: Dictionary with additional parameters
    """
    pred_type = viz_data.get("prediction_set_type", "scalar")
    q_value = viz_data.get("q_value", 1.0)

    if pred_type == "scalar":
        # L2: uniform radius for all fluid nodes
        fluid_mask = np.array(viz_data.get("fluid_mask", []))
        num_fluid = np.sum(fluid_mask) if len(fluid_mask) > 0 else 1000
        radii = np.full(num_fluid, q_value)
        metadata = {"radius": q_value}
        return pred_type, radii, metadata

    elif pred_type == "ellipse":
        # Mahalanobis: ellipse parameters
        ellipse_axes = np.array(viz_data.get("ellipse_axes", [q_value, q_value]))
        ellipse_angle = viz_data.get("ellipse_angle", 0.0)

        metadata = {
            "axes": ellipse_axes,
            "angle": ellipse_angle,
            "covariance_matrix": np.array(
                viz_data.get("covariance_matrix", [[1, 0], [0, 1]])
            ),
            "eigenvalues": np.array(viz_data.get("eigenvalues", [1, 1])),
            "eigenvectors": np.array(viz_data.get("eigenvectors", [[1, 0], [0, 1]])),
        }
        return pred_type, ellipse_axes, metadata

    elif pred_type == "spatial":
        # BCP: spatially-varying radii
        spatial_radii = np.array(viz_data.get("spatial_radii", [q_value] * 1000))

        metadata = {
            "base_q": q_value,
            "predicted_scales": np.array(viz_data.get("predicted_scales", [])),
            "scale_statistics": viz_data.get("scale_statistics", {}),
        }
        return pred_type, spatial_radii, metadata

    elif pred_type == "box":
        # Box: per-component intervals
        half_width = viz_data.get("box_half_width", q_value)
        fluid_mask = np.array(viz_data.get("fluid_mask", []))
        num_fluid = np.sum(fluid_mask) if len(fluid_mask) > 0 else 1000

        box_widths = np.full(num_fluid, half_width)
        metadata = {"half_width": half_width}
        return pred_type, box_widths, metadata

    else:
        # Fallback to scalar
        return "scalar", np.array([q_value]), {"radius": q_value}


def get_mesh_data(viz_data: Dict) -> Dict:
    """
    Extract mesh geometry data from visualization data.

    Args:
        viz_data: Visualization data from JSON

    Returns:
        Dictionary with mesh data
    """
    mesh_data = {}

    if "mesh_positions" in viz_data:
        mesh_data["positions"] = np.array(viz_data["mesh_positions"])

    if "fluid_mask" in viz_data:
        mesh_data["fluid_mask"] = np.array(viz_data["fluid_mask"], dtype=bool)

    if "node_types" in viz_data:
        mesh_data["node_types"] = np.array(viz_data["node_types"])

    if "mesh_cells" in viz_data:
        mesh_data["cells"] = np.array(viz_data["mesh_cells"])

    if "wall_distances" in viz_data:
        mesh_data["wall_distances"] = np.array(viz_data["wall_distances"])

    return mesh_data


def get_coverage_analysis_data(viz_data: Dict) -> Optional[Dict]:
    """
    Extract spatial coverage analysis data.

    Args:
        viz_data: Visualization data from JSON

    Returns:
        Dictionary with coverage analysis or None if not available
    """
    spatial_coverage = viz_data.get("spatial_coverage")
    if spatial_coverage is None:
        return None

    return {
        "distance_bins": np.array(spatial_coverage["distance_bins"]),
        "bin_centers": np.array(spatial_coverage["bin_centers"]),
        "bin_coverage": [c for c in spatial_coverage["bin_coverage"] if c is not None],
        "bin_counts": spatial_coverage["bin_counts"],
        "node_coverage": np.array(spatial_coverage["node_coverage"]),
    }


def compare_methods_from_json(json_files: List[str], alpha: float = 0.1) -> Dict:
    """
    Compare multiple conformal prediction methods from JSON files.

    Args:
        json_files: List of JSON file paths
        alpha: Alpha value to extract

    Returns:
        Dictionary with comparison data
    """
    comparison = {
        "alpha": alpha,
        "methods": {},
        "mesh_data": None,  # Will use mesh data from first available file
    }

    for json_file in json_files:
        data = load_conformal_results(json_file)

        # Handle both old format (direct list) and new enhanced format
        if "results" in data:
            # New enhanced format: {"results": [...], "metadata": {...}}
            results = data["results"]
        elif "result" in data:
            # Single result format: {"result": {...}, "metadata": {...}}
            results = [data["result"]]
        elif isinstance(data, list):
            # Old format: direct list of results
            results = data
        else:
            # Fallback: treat as single result
            results = [data]

        for result in results:
            if abs(result.get("alpha", 0) - alpha) < 0.001:
                method = result.get("nonconformity_method", "unknown")
                viz_data = result.get("visualization", {})

                # Extract prediction set data
                pred_type, data_array, metadata = create_prediction_set_data(viz_data)

                comparison["methods"][method] = {
                    "q_value": result.get("q_value"),
                    "coverage": result.get("empirical_coverage"),
                    "width_stats": result.get("width_stats", {}),
                    "prediction_set_type": pred_type,
                    "data": data_array,
                    "metadata": metadata,
                    "visualization_data": viz_data,
                }

                # Use mesh data from first available file
                if comparison["mesh_data"] is None:
                    comparison["mesh_data"] = get_mesh_data(viz_data)

    return comparison


# Example usage functions
def load_standard_comparison(results_dir: str, alpha: float = 0.1) -> Dict:
    """
    Load standard comparison (L2, Mahalanobis, BCP) from results directory.

    Args:
        results_dir: Directory containing JSON results
        alpha: Alpha value to compare

    Returns:
        Comparison dictionary
    """
    # Look for common result files
    patterns = [
        "*alpha_sweep_compare*.json",  # Multi-method comparison
        "*boosted_alpha_sweep*.json",  # BCP results
        "*alpha_sweep*.json",  # Standard methods
    ]

    json_files = []
    for pattern in patterns:
        files = glob.glob(os.path.join(results_dir, pattern))
        if files:
            json_files.append(max(files, key=os.path.getmtime))  # Get newest

    if not json_files:
        raise FileNotFoundError(
            f"No conformal prediction results found in {results_dir}"
        )

    return compare_methods_from_json(json_files, alpha)


if __name__ == "__main__":
    # Example usage
    results_dir = "./results"

    try:
        # Load latest comparison
        comparison = load_standard_comparison(results_dir, alpha=0.1)

        print("Available methods:")
        for method, data in comparison["methods"].items():
            print(
                f"  {method}: q={data['q_value']:.3f}, coverage={data['coverage']:.3f}"
            )
            print(f"    Prediction set type: {data['prediction_set_type']}")

        print(f"\nMesh data available: {comparison['mesh_data'] is not None}")
        if comparison["mesh_data"]:
            positions = comparison["mesh_data"].get("positions")
            if positions is not None:
                print(f"  Mesh nodes: {len(positions)}")
                print(
                    f"  Fluid nodes: {np.sum(comparison['mesh_data'].get('fluid_mask', []))}"
                )

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run conformal analysis with --save flag first to generate JSON files.")
