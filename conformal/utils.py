#!/usr/bin/env python3
"""
Utility functions for conformal prediction: validation and data splitting.

Reference:
    Paper §5.1 for train-auxiliary-calibration-test split strategy.
"""

import numpy as np
from typing import Tuple


def validate_alpha(alpha: float) -> None:
    """
    Validate miscoverage level α ∈ (0,1).

    Raises:
        ValueError: If α not in valid range
    """
    if not (0 < alpha < 1):
        raise ValueError(f"Alpha must be in (0,1), got {alpha}")


def validate_calib_ratio(calib_ratio: float) -> None:
    """
    Validate calibration ratio ∈ (0,1).

    Raises:
        ValueError: If ratio not in valid range
    """
    if not (0 < calib_ratio < 1):
        raise ValueError(f"Calibration ratio must be in (0,1), got {calib_ratio}")


def validate_data_shapes(predictions: np.ndarray, ground_truth: np.ndarray) -> None:
    """
    Validate prediction and ground truth array shapes.

    Expected: [T, N, d] where T=timesteps, N=nodes, d ∈ {2,3}

    Raises:
        ValueError: If shapes incompatible or unexpected
    """
    if predictions.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} != ground_truth {ground_truth.shape}"
        )

    if len(predictions.shape) != 3 or predictions.shape[2] not in [2, 3]:
        raise ValueError(
            f"Expected shape [num_steps, num_nodes, 2|3], got {predictions.shape}"
        )


def validate_sufficient_data(num_steps: int, calib_ratio: float) -> None:
    """
    Ensure sufficient data for reliable conformal prediction.

    Minimum requirements:
        - Calibration: ≥10 timesteps for stable quantiles
        - Test: ≥5 timesteps for coverage evaluation

    Raises:
        ValueError: If insufficient data
    """
    calib_steps = int(num_steps * calib_ratio)
    test_steps = num_steps - calib_steps

    if calib_steps < 10:
        raise ValueError(
            f"Calibration set too small ({calib_steps} steps). Need ≥10 for stable quantiles."
        )

    if test_steps < 5:
        raise ValueError(
            f"Test set too small ({test_steps} steps). Need ≥5 for coverage evaluation."
        )


def validate_conformal_inputs(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    alpha: float,
    calib_ratio: float,
    num_steps: int,
) -> None:
    """
    Comprehensive input validation for conformal prediction.

    Validates: alpha range, calibration ratio, array shapes, data sufficiency.

    Raises:
        ValueError: If any validation fails
    """
    validate_alpha(alpha)
    validate_calib_ratio(calib_ratio)
    validate_data_shapes(predictions, ground_truth)
    validate_sufficient_data(num_steps, calib_ratio)


def create_timestep_split(
    num_steps: int,
    calib_ratio: float,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Random split of timesteps into calibration and test sets.

    Simple exchangeable splitting strategy: randomly permute timesteps,
    then allocate first calib_ratio fraction to calibration.

    Args:
        num_steps: Total number of timesteps
        calib_ratio: Fraction for calibration (rest goes to test)
        random_seed: Random seed for reproducibility

    Returns:
        (calibration_indices, test_indices)

    Reference:
        Paper §5.1 (Data Splitting Strategy)
    """
    np.random.seed(random_seed)
    indices = np.arange(num_steps)
    np.random.shuffle(indices)

    calib_size = int(num_steps * calib_ratio)
    return indices[:calib_size], indices[calib_size:]


def create_threeway_timestep_split(
    num_steps: int,
    aux_ratio: float,
    calib_ratio: float,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Three-way split: auxiliary, calibration, test.

    Required for valid boosted/adaptive conformal prediction where auxiliary
    set is used to learn score components without information leakage.

    Args:
        num_steps: Total timesteps
        aux_ratio: Fraction for auxiliary set
        calib_ratio: Fraction for calibration set (of total, not remainder)
        random_seed: Random seed

    Returns:
        (aux_indices, calib_indices, test_indices)

    Raises:
        ValueError: If aux_ratio + calib_ratio ≥ 1

    Reference:
        Paper §4.4: Auxiliary set prevents information leakage in boosted CP
    """
    if (
        not (0 < aux_ratio < 1)
        or not (0 < calib_ratio < 1)
        or aux_ratio + calib_ratio >= 1
    ):
        raise ValueError(
            f"Invalid ratios: aux={aux_ratio}, calib={calib_ratio}. Need 0<each<1 and sum<1."
        )

    np.random.seed(random_seed)
    indices = np.arange(num_steps)
    np.random.shuffle(indices)

    aux_size = int(num_steps * aux_ratio)
    calib_size = int(num_steps * calib_ratio)

    aux_idx = indices[:aux_size]
    calib_idx = indices[aux_size : aux_size + calib_size]
    test_idx = indices[aux_size + calib_size :]

    return aux_idx, calib_idx, test_idx
