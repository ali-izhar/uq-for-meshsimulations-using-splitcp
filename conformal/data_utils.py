#!/usr/bin/env python3
"""
Data loading and preparation utilities for conformal prediction.

Handles loading MeshGraphNet predictions, scale correction, data splitting,
and preparation for conformal calibration.

Reference:
    Paper §5.1 (Experimental Setup) for data split strategy.
"""

import numpy as np
import pickle
from typing import Dict, Optional


def load_predictions_from_file(
    file_path: str, apply_scale_correction: bool = True
) -> Dict:
    """
    Load MeshGraphNet predictions from pickle file.

    Handles NumPy version compatibility issues and applies optional scale
    correction to fix extraction artifacts.

    Args:
        file_path: Path to pickle file with predictions/ground_truth
        apply_scale_correction: Fix scale mismatch from reverse-engineering

    Returns:
        Dict with 'predictions', 'ground_truth', 'metadata', etc.
    """
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
    except ModuleNotFoundError as e:
        # Handle NumPy version compatibility (numpy._core in NumPy>=2.0)
        if "numpy._core" in str(e):
            import sys
            import types
            import numpy as np_module

            target_core = getattr(np_module, "core", np_module)
            sys.modules.setdefault("numpy._core", target_core)

            # Map common submodules
            for submod in ["multiarray", "numerictypes"]:
                try:
                    sys.modules.setdefault(
                        f"numpy._core.{submod}", getattr(target_core, submod)
                    )
                except:
                    pass

            # Provide dummy overrides module if needed
            if "numpy._core.overrides" not in sys.modules:
                sys.modules["numpy._core.overrides"] = types.ModuleType("overrides")

            with open(file_path, "rb") as f:
                data = pickle.load(f)
        else:
            raise

    if apply_scale_correction:
        data = normalize_extracted_data(data)

    return data


def normalize_extracted_data(data: Dict) -> Dict:
    """
    Fix scale mismatch between reverse-engineered predictions and ground truth.

    MeshGraphNet extraction creates predictions by integrating velocities,
    while ground truth uses raw dataset values. This causes scale inconsistency
    that artificially widens intervals. Solution: normalize both to unit variance.

    Args:
        data: Raw extracted data

    Returns:
        Data with consistent scaling
    """
    predictions = data["predictions"].copy()
    ground_truth = data["ground_truth"].copy()

    print("  Fixing scale mismatch in extracted data...")
    print(f"    Original prediction std: {np.std(predictions):.4f}")
    print(f"    Original ground truth std: {np.std(ground_truth):.4f}")

    # Normalize both to unit variance (most robust approach)
    predictions_norm = (predictions - predictions.mean()) / predictions.std()
    ground_truth_norm = (ground_truth - ground_truth.mean()) / ground_truth.std()

    print(
        f"    After normalization: pred_std={predictions_norm.std():.4f}, gt_std={ground_truth_norm.std():.4f}"
    )

    # Create corrected data dictionary
    corrected = data.copy()
    corrected["predictions"] = predictions_norm
    corrected["ground_truth"] = ground_truth_norm
    corrected["original_predictions"] = predictions  # Keep for reference
    corrected["original_ground_truth"] = ground_truth
    corrected["scale_correction_applied"] = True

    return corrected


def extract_fluid_mask(data: Dict, use_fluid_mask: bool) -> Optional[np.ndarray]:
    """
    Extract fluid mask from metadata if available.

    Args:
        data: Loaded predictions data
        use_fluid_mask: Whether to extract mask

    Returns:
        Boolean mask [N] for fluid nodes, or None
    """
    if use_fluid_mask and "metadata" in data and "fluid_mask" in data["metadata"]:
        return data["metadata"]["fluid_mask"]
    return None


def prepare_conformal_data(
    predictions_file: str,
    calib_ratio: float,
    use_fluid_mask: bool,
    random_seed: int,
    aux_ratio: float = 0.0,
) -> Dict:
    """
    Load and prepare data for conformal prediction with train/cal/test split.

    Implements the data splitting strategy from the paper: auxiliary set for
    learning score components (boosted CP), calibration set for quantile,
    and held-out test set for evaluation.

    Args:
        predictions_file: Path to pickle file
        calib_ratio: Fraction for calibration (of non-aux data)
        use_fluid_mask: Whether to mask wall nodes
        random_seed: Random seed for reproducibility
        aux_ratio: Fraction for auxiliary set (0 = no auxiliary split)

    Returns:
        Dict with data, splits, and masks

    Reference:
        Paper §5.1 (Three-way data split for valid boosted CP)
    """
    from .utils import create_timestep_split, create_threeway_timestep_split

    # Load data
    data = load_predictions_from_file(predictions_file)
    predictions = data["predictions"]
    ground_truth = data["ground_truth"]
    num_steps, num_nodes, _ = predictions.shape

    # Extract fluid mask
    fluid_mask = extract_fluid_mask(data, use_fluid_mask)
    if fluid_mask is not None:
        fluid_count = np.sum(fluid_mask)
        print(
            f"  Using fluid mask: {fluid_count}/{num_nodes} nodes ({100*fluid_count/num_nodes:.1f}%)"
        )

    # Create timestep split
    if aux_ratio > 0.0:
        aux_idx, calib_idx, test_idx = create_threeway_timestep_split(
            num_steps, aux_ratio, calib_ratio, random_seed
        )
        print(
            f"  Split: {len(aux_idx)} aux, {len(calib_idx)} calib, {len(test_idx)} test"
        )
    else:
        aux_idx = np.array([], dtype=int)
        calib_idx, test_idx = create_timestep_split(num_steps, calib_ratio, random_seed)
        print(f"  Split: {len(calib_idx)} calib, {len(test_idx)} test")

    return {
        "data": data,
        "predictions": predictions,
        "ground_truth": ground_truth,
        "num_steps": num_steps,
        "num_nodes": num_nodes,
        "fluid_mask": fluid_mask,
        "aux_indices": aux_idx,
        "calib_indices": calib_idx,
        "test_indices": test_idx,
    }


def compute_normalized_width_stats(width_stats: Dict, data: Dict) -> Dict:
    """
    Compute interval widths normalized by ground truth scale.

    Denormalizes widths from unit-variance space back to original scale,
    then expresses as fraction of ground truth range for interpretability.

    Args:
        width_stats: Width statistics in normalized space
        data: Data dict with original scale information

    Returns:
        Dict with normalized width metrics
    """
    normalized_stats = width_stats.copy()

    if "original_ground_truth" in data:
        original_gt_std = data["original_ground_truth"].std()
        original_gt_range = np.ptp(data["original_ground_truth"])  # Peak-to-peak

        # Current width is in normalized units (std=1)
        normalized_width = width_stats["mean_width"]

        # Convert to physical units, then express as fraction of range
        physical_width = normalized_width * original_gt_std
        range_normalized = physical_width / original_gt_range

        normalized_stats.update(
            {
                "normalized_width": range_normalized,
                "normalized_area": width_stats["area"],
                "original_scale_factor": original_gt_std,
                "interpretation": f"Width = {100*range_normalized:.2f}% of ground truth range",
            }
        )
    else:
        normalized_stats.update(
            {
                "normalized_width": width_stats["mean_width"],
                "normalized_area": width_stats["area"],
                "original_scale_factor": 1.0,
                "interpretation": "No scale correction applied",
            }
        )

    return normalized_stats
