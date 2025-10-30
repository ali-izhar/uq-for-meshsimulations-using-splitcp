#!/usr/bin/env python3
"""
Nonconformity Score Functions for Mesh-Based Conformal Prediction

Implements the four prediction-set geometries described in the paper:
    - ℓ2 disks: {‖y−ŷ‖₂ ≤ q}
    - Joint ℓ∞ boxes: {max(|y−ŷ|) ≤ q} (componentwise intervals)
    - Mahalanobis ellipses: {(y−ŷ)ᵀΣ⁻¹(y−ŷ) ≤ q²}
    - Spatially adaptive scaling (see adaptive.py)

These scores form the foundation of split conformal prediction, where calibration
yields quantile q at rank ⌈(m+1)(1−α)⌉ for finite-sample coverage guarantee:
    P(Y ∈ C_α(X)) ≥ 1 - α

Reference:
    See paper.tex §3 (Conformal Prediction Framework) for mathematical details.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict

# Constants
DEFAULT_REGULARIZATION_EPS = 1e-8


@dataclass
class ConformityScoreResult:
    """
    Container for conformity score computation results.

    Attributes:
        scores: Nonconformity scores for calibration set [m]
        metadata: Optional dict containing geometry-specific data (e.g., covariance for Mahalanobis)
    """

    scores: np.ndarray
    metadata: Dict = field(default_factory=dict)

    def get_covariance_matrix(self) -> Optional[np.ndarray]:
        """Get covariance matrix Σ (Mahalanobis geometry only)."""
        return self.metadata.get("covariance_matrix")

    def get_covariance_inverse(self) -> Optional[np.ndarray]:
        """Get inverse covariance matrix Σ⁻¹ (Mahalanobis geometry only)."""
        return self.metadata.get("covariance_inverse")


def _validate_inputs(
    predictions: np.ndarray, ground_truth: np.ndarray, mask: Optional[np.ndarray]
) -> None:
    """Validate input shapes for score computation."""
    if predictions.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} != ground_truth {ground_truth.shape}"
        )
    if mask is not None and mask.shape[0] != predictions.shape[1]:
        raise ValueError(
            f"Mask length {mask.shape[0]} != num_nodes {predictions.shape[1]}"
        )


# --------------------------------------------------------------
# ------------------ L2 Disk Sets ------------------------------
# --------------------------------------------------------------


def compute_l2_scores(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> ConformityScoreResult:
    """
    Compute ℓ2 nonconformity scores: s(x,y) = ‖y − ŷ(x)‖₂

    Corresponds to disk prediction sets C_α(x) = {y: ‖y − ŷ(x)‖₂ ≤ q_α}
    where q_α is the calibrated quantile. This is the baseline geometry.

    Args:
        predictions: Model predictions ŷ [T, N, d] where d ∈ {2,3}
        ground_truth: True values y [T, N, d]
        mask: Optional boolean mask [N] for fluid nodes (excludes walls)

    Returns:
        ConformityScoreResult with scores [T*N] or [T*N_fluid]

    Raises:
        ValueError: If input shapes are incompatible

    Reference:
        Paper §4.1 (ℓ2 Disk Sets)
    """
    _validate_inputs(predictions, ground_truth, mask)

    # Compute residual vectors: e_i = y_i - ŷ_i
    residuals = ground_truth - predictions  # [T, N, d]
    scores = np.linalg.norm(residuals, axis=-1)  # [T, N]

    if mask is not None:
        scores = scores[:, mask]  # [T, N_fluid]

    return ConformityScoreResult(scores.reshape(-1))


# --------------------------------------------------------------
# ------------------ Mahalanobis Ellipse Sets ------------------
# --------------------------------------------------------------


def compute_mahalanobis_scores(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> ConformityScoreResult:
    """
    Compute Mahalanobis nonconformity scores: s(x,y) = √[(y−ŷ)ᵀΣ⁻¹(y−ŷ)]

    Corresponds to ellipse prediction sets C_α(x) = {y: (y−ŷ)ᵀΣ⁻¹(y−ŷ) ≤ q_α²}
    where Σ is the empirical covariance of residuals on calibration set.
    Adapts to correlation structure between velocity components.

    Args:
        predictions: Model predictions ŷ [T, N, d] where d ∈ {2,3}
        ground_truth: True values y [T, N, d]
        mask: Optional boolean mask [N] for fluid nodes

    Returns:
        ConformityScoreResult with scores and covariance matrices {Σ, Σ⁻¹}

    Raises:
        ValueError: If input shapes incompatible or insufficient data (need ≥3 samples)

    Reference:
        Paper §4.2 (Mahalanobis Ellipse Sets)
    """
    _validate_inputs(predictions, ground_truth, mask)

    # Compute residual vectors: e_i = y_i - ŷ_i
    residuals = ground_truth - predictions  # [T, N, d]

    if mask is not None:
        residuals = residuals[:, mask, :]  # [T, N_fluid, d]

    # Flatten to [m, d] for covariance estimation
    residuals_flat = residuals.reshape(-1, residuals.shape[-1])

    if residuals_flat.shape[0] < 3:
        raise ValueError(
            f"Insufficient data for covariance estimation: need ≥3 samples, got {residuals_flat.shape[0]}"
        )

    # Estimate covariance Σ with regularization for numerical stability
    cov = np.cov(residuals_flat, rowvar=False) + DEFAULT_REGULARIZATION_EPS * np.eye(
        residuals_flat.shape[-1]
    )

    # Compute Σ⁻¹ with fallback to pseudo-inverse
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)

    # Compute Mahalanobis distances: √(eᵀ Σ⁻¹ e)
    scores = np.sqrt(np.einsum("ij,jk,ik->i", residuals_flat, cov_inv, residuals_flat))

    return ConformityScoreResult(
        scores, metadata={"covariance_matrix": cov, "covariance_inverse": cov_inv}
    )


# --------------------------------------------------------------
# ------------------ Joint ℓ∞ Box Sets -------------------------
# --------------------------------------------------------------


def compute_box_scores(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> ConformityScoreResult:
    """
    Compute joint ℓ∞ nonconformity scores: s(x,y) = max_j |y_j − ŷ_j(x)|

    Corresponds to box prediction sets C_α(x) = {y: max_j |y_j − ŷ_j(x)| ≤ q_α}
    which gives componentwise intervals [ŷ_j − q, ŷ_j + q] for each dimension.
    Tightest geometry when residuals are uncorrelated.

    Args:
        predictions: Model predictions ŷ [T, N, d] where d ∈ {2,3}
        ground_truth: True values y [T, N, d]
        mask: Optional boolean mask [N] for fluid nodes

    Returns:
        ConformityScoreResult with scores [T*N] or [T*N_fluid]

    Raises:
        ValueError: If input shapes are incompatible

    Reference:
        Paper §4.3 (Joint ℓ∞ Box Sets)
    """
    _validate_inputs(predictions, ground_truth, mask)

    # Compute absolute residuals and take max over components
    abs_residuals = np.abs(ground_truth - predictions)  # [T, N, d]
    scores = np.max(abs_residuals, axis=-1)  # [T, N]

    if mask is not None:
        scores = scores[:, mask]  # [T, N_fluid]

    return ConformityScoreResult(scores.reshape(-1))


# --------------------------------------------------------------
# ------------------ Conformal Quantile ------------------------
# --------------------------------------------------------------


def compute_conformal_quantile(
    scores: np.ndarray, alpha: float, conservative: bool = True, margin: float = 0.01
) -> float:
    """
    Compute conformal quantile q_α at rank k = ⌈(m+1)(1−α)⌉

    For split conformal prediction with m calibration scores, the quantile
    at this rank ensures finite-sample coverage: P(Y ∈ C_α(X)) ≥ 1 - α

    Args:
        scores: Calibration nonconformity scores [m]
        alpha: Miscoverage level α ∈ (0,1), target coverage = 1 - α
        conservative: If True, reduce α by margin for practical conservativeness
        margin: Margin to subtract from α (default 0.01 → 1% extra coverage)

    Returns:
        Quantile value q_α

    Raises:
        ValueError: If scores empty or α not in (0,1)

    Reference:
        Paper §3 (Split Conformal Prediction), Equation (2)
        Vovk et al. (2005): finite-sample validity proof
    """
    if scores.size == 0:
        raise ValueError("Cannot compute quantile from empty scores")
    if not (0 < alpha < 1):
        raise ValueError(f"Alpha must be in (0,1), got {alpha}")

    m = scores.size

    # Apply conservative margin if requested
    alpha_eff = max(0.001, alpha - margin) if conservative else alpha

    # Compute rank: k = ⌈(m+1)(1−α)⌉
    k = int(np.ceil((m + 1) * (1.0 - alpha_eff)))
    k = np.clip(k, 1, m)  # Ensure valid index

    # Use partition for O(m) quantile computation
    return float(np.partition(scores, k - 1)[k - 1])


# Registry of nonconformity score functions
# Maps geometry name → score function for strategy pattern
SCORE_FUNCTIONS = {
    "l2": compute_l2_scores,  # §4.1: ℓ2 disk sets
    "mahalanobis": compute_mahalanobis_scores,  # §4.2: Mahalanobis ellipses
    "box": compute_box_scores,  # §4.3: Joint ℓ∞ boxes
}
