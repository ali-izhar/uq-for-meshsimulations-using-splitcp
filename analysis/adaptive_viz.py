#!/usr/bin/env python3
"""
Visualization utilities for spatially adaptive conformal prediction.

Extracts spatial uncertainty patterns and metadata for plotting adaptive
prediction sets from the BoostedConformalPredictor.
"""

import numpy as np
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from conformal.adaptive import BoostedConformalPredictor


def extract_adaptive_visualization_data(
    predictor: "BoostedConformalPredictor",
    prepared_data: Dict,
    test_predictions: np.ndarray,
    test_ground_truth: np.ndarray,
    alpha: float,
) -> Dict:
    """
    Extract spatially-varying radii and metadata for visualization.

    Args:
        predictor: Calibrated adaptive conformal predictor
        prepared_data: Data dictionary with fluid_mask, metadata, etc.
        test_predictions: Test predictions [T, N, d]
        test_ground_truth: Test ground truth [T, N, d]
        alpha: Confidence level

    Returns:
        Dictionary with visualization data including spatial_radii
    """
    data = prepared_data["data"]
    fluid_mask = prepared_data["fluid_mask"]

    viz_data = {
        "method": "adaptive_l2",
        "alpha": alpha,
        "q_value": predictor.q_value if hasattr(predictor, "q_value") else None,
        "prediction_set_type": "spatial",
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

    # Compute spatially-varying radii using meta-model
    try:
        fluid_positions = (
            positions[fluid_mask] if "mesh_positions" in metadata else None
        )

        if fluid_positions is not None:
            # Get test velocities
            test_velocities = data.get("velocities", np.zeros_like(test_predictions))
            if len(test_velocities) > len(test_predictions):
                test_velocities = test_velocities[prepared_data["test_indices"]]

            # Build features for test data (fluid nodes only)
            test_pred_masked = test_predictions[:, fluid_mask, :]
            test_vel_masked = test_velocities[:, fluid_mask, :]

            features = predictor._build_features(
                test_pred_masked, test_vel_masked, metadata, fluid_mask
            )

            # Predict local scales using meta-model
            if hasattr(predictor, "_model") and predictor._model is not None:
                try:
                    predicted_scales = predictor._model.predict(features)

                    # Convert from log space if needed
                    if predictor._use_log:
                        s_hat = np.exp(predicted_scales)
                    else:
                        s_hat = predicted_scales

                    s_hat = np.maximum(s_hat, predictor._eps)

                    # Apply clipping and shrinkage
                    if (
                        hasattr(predictor, "_clip_lo_")
                        and predictor._clip_lo_ is not None
                    ):
                        s_hat = np.clip(s_hat, predictor._clip_lo_, predictor._clip_hi_)

                    if predictor._shrink_lambda > 0.0 and hasattr(
                        predictor, "_global_med_res_"
                    ):
                        if predictor._global_med_res_ is not None:
                            s_hat = (
                                (1.0 - predictor._shrink_lambda) * s_hat
                                + predictor._shrink_lambda * predictor._global_med_res_
                            )

                    # Convert to local radii: r(x) = q * ŝ(x)
                    spatial_radii = predictor.q_value * s_hat

                    # Average over time steps for visualization
                    if len(spatial_radii.shape) > 1:
                        spatial_radii = spatial_radii.reshape(
                            -1, len(fluid_positions)
                        ).mean(axis=0)

                    viz_data["spatial_radii"] = spatial_radii.tolist()
                    viz_data["predicted_scales"] = (
                        predicted_scales.tolist()
                        if len(predicted_scales.shape) == 1
                        else predicted_scales.reshape(-1, len(fluid_positions))
                        .mean(axis=0)
                        .tolist()
                    )
                    viz_data["scale_diagnostics"] = {
                        "use_log_scale": predictor._use_log,
                        "q_value": predictor.q_value,
                        "radii_range": [
                            float(spatial_radii.min()),
                            float(spatial_radii.max()),
                        ],
                    }

                except Exception as e:
                    print(
                        f"Warning: Could not compute spatial radii from meta-model: {e}"
                    )
                    print(
                        "Visualization data incomplete - meta-model not properly trained"
                    )
            else:
                print("Warning: Meta-model not available for spatial radii extraction")

    except Exception as e:
        print(f"Warning: Could not extract spatial radii: {e}")
        print(
            "Visualization data incomplete - use actual trained model for spatial uncertainty"
        )

    # Add feature importance if available
    if hasattr(predictor, "_model") and predictor._model is not None:
        try:
            if hasattr(predictor._model, "feature_importances_"):
                viz_data["feature_importances"] = (
                    predictor._model.feature_importances_.tolist()
                )
        except:
            pass

    # Add scale statistics
    if hasattr(predictor, "_scale_stats"):
        viz_data["scale_statistics"] = {
            k: float(v) if np.isscalar(v) else v.tolist() if hasattr(v, "tolist") else v
            for k, v in predictor._scale_stats.items()
        }

    return viz_data


# REMOVED: simulate_adaptive_spatial_variation
# This was a fallback that created fake spatial patterns. For production/publication,
# visualizations should only use actual meta-model predictions from real experiments.
