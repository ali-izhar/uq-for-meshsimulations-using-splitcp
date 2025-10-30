#!/usr/bin/env python3
"""Core conformal prediction functionality for MeshGraphNet predictions."""

import numpy as np
import pickle
from typing import Dict, List, Optional, Tuple

from .conformity import compute_conformal_quantile, SCORE_FUNCTIONS

# Constants
SUPPORTED_NONCONFORMITY_MODES = tuple(SCORE_FUNCTIONS.keys())
COVERAGE_WARNING_THRESHOLD = 0.02


# Import utilities from extracted modules
from .data_utils import load_predictions_from_file
from .data_utils import prepare_conformal_data as _prepare_conformal_data
from .data_utils import (
    compute_normalized_width_stats as _compute_normalized_width_stats,
)
from .utils import validate_conformal_inputs as _validate_conformal_inputs
from .utils import validate_alpha as _validate_alpha


class ConformalPredictor:
    """
    Conformal predictor for MeshGraphNet acceleration predictions.

    Works directly with extracted predictions to provide uncertainty quantification
    without requiring the original model or dataset.
    """

    def __init__(self, nonconformity: str = "l2"):
        """Initialize the conformal predictor.

        Args:
            nonconformity: 'l2' for vector L2-norm balls; 'box' for per-component boxes; 'mahalanobis' for ellipsoids.
        """
        self.calibrated = False
        self.q_value = None
        self.calibration_residuals = None
        if nonconformity not in SUPPORTED_NONCONFORMITY_MODES:
            raise ValueError(
                f"nonconformity must be one of {SUPPORTED_NONCONFORMITY_MODES}"
            )
        self.nonconformity = nonconformity
        self._covariance_matrix = None
        self._covariance_inv = None
        # When set via auxiliary split, use fixed covariance for Mahalanobis
        self._use_fixed_mahalanobis_cov = False

    @property
    def is_ready(self) -> bool:
        """Check if predictor is calibrated and ready for predictions."""
        return self.calibrated and self.q_value is not None

    def _set_calibrated_state(
        self, q_value: float, calibration_scores: np.ndarray
    ) -> None:
        """
        Set the calibrated state of the predictor.

        Internal method for optimized workflows that bypass the full calibrate() method.

        Args:
            q_value: Calibrated quantile value
            calibration_scores: Nonconformity scores used for calibration
        """
        self.q_value = q_value
        self.calibration_residuals = calibration_scores
        self.calibrated = True

    def compute_nonconformity_scores(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute nonconformity scores based on the selected method.

        Args:
            predictions: Model predictions [num_steps, num_nodes, 2]
            ground_truth: Ground truth values [num_steps, num_nodes, 2]
            mask: Optional boolean mask for valid nodes [num_nodes] (e.g., fluid nodes only)

        Returns:
            Array of nonconformity scores

        Raises:
            ValueError: If unknown nonconformity method or invalid inputs
        """
        if self.nonconformity not in SCORE_FUNCTIONS:
            raise ValueError(f"Unknown nonconformity method: {self.nonconformity}")

        # Special handling for Mahalanobis when a fixed covariance has been set
        if (
            self.nonconformity == "mahalanobis"
            and self._use_fixed_mahalanobis_cov
            and self._covariance_inv is not None
        ):
            return self._compute_mahalanobis_scores_with_cov(
                predictions, ground_truth, mask
            )

        # Default: use strategy function (will estimate covariance from provided data)
        score_function = SCORE_FUNCTIONS[self.nonconformity]
        result = score_function(predictions, ground_truth, mask)

        if self.nonconformity == "mahalanobis":
            # Store last estimated covariance for diagnostics if using data-driven mode
            self._covariance_matrix = result.get_covariance_matrix()
            self._covariance_inv = result.get_covariance_inverse()

        return result.scores

    def _compute_mahalanobis_scores_with_cov(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute Mahalanobis scores using a precomputed covariance inverse.

        Requires self._covariance_inv to be set (from auxiliary split).
        """
        if self._covariance_inv is None:
            raise ValueError(
                "Mahalanobis covariance not set. Call set_mahalanobis_cov_from_data() first."
            )

        if predictions.shape != ground_truth.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} != ground_truth {ground_truth.shape}"
            )

        residual_vectors = ground_truth - predictions  # [T, N, d]
        if mask is not None:
            if mask.shape[0] != predictions.shape[1]:
                raise ValueError(
                    f"Mask length {mask.shape[0]} != num_nodes {predictions.shape[1]}"
                )
            residual_vectors = residual_vectors[:, mask, :]

        flat_res = residual_vectors.reshape(-1, residual_vectors.shape[-1])
        scores = np.sqrt(
            np.einsum("ij,jk,ik->i", flat_res, self._covariance_inv, flat_res)
        )
        return scores

    def set_mahalanobis_cov_from_data(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        mask: Optional[np.ndarray] = None,
        regularization_eps: float = 1e-8,
    ) -> None:
        """Estimate and freeze Mahalanobis covariance from provided data (auxiliary split)."""
        if predictions.shape != ground_truth.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} != ground_truth {ground_truth.shape}"
            )
        residual_vectors = ground_truth - predictions
        if mask is not None:
            if mask.shape[0] != predictions.shape[1]:
                raise ValueError(
                    f"Mask length {mask.shape[0]} != num_nodes {predictions.shape[1]}"
                )
            residual_vectors = residual_vectors[:, mask, :]
        flat_res = residual_vectors.reshape(-1, residual_vectors.shape[-1])
        if flat_res.shape[0] < 3:
            raise ValueError(
                f"Insufficient data for Mahalanobis: need at least 3 samples, got {flat_res.shape[0]}"
            )
        cov = np.cov(flat_res, rowvar=False)
        cov = cov + regularization_eps * np.eye(cov.shape[0], dtype=float)
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)
        self._covariance_matrix = cov
        self._covariance_inv = cov_inv
        self._use_fixed_mahalanobis_cov = True

    def _compute_conformal_quantile(
        self,
        scores: np.ndarray,
        alpha: float,
        conservative: bool = True,
        margin: float = 0.01,
    ) -> float:
        """
        Compute the conformal quantile using finite-sample correction.

        Wrapper around the imported stateless function.

        Args:
            scores: Array of nonconformity scores
            alpha: Miscoverage level
            conservative: If True, ensure coverage ≥ 1-α (recommended)
            margin: Conservative margin to subtract from alpha
        """
        return compute_conformal_quantile(
            scores, alpha, conservative=conservative, margin=margin
        )

    def _print_calibration_diagnostics(
        self, scores: np.ndarray, q: float, alpha: float
    ) -> None:
        """
        Print diagnostic information about calibration.

        Args:
            scores: Array of nonconformity scores
            q: Calibrated quantile value
            alpha: Miscoverage level
        """
        m = scores.size
        k = int(np.ceil((m + 1) * (1.0 - alpha)))
        target_coverage = 1 - alpha
        empirical_coverage = np.sum(scores <= q) / max(1, len(scores))

        print(f"  Calibration: m={m}, k={k}, alpha={alpha:.3f}")

        # Score-specific diagnostics with consistent formatting
        score_stats_msg = f"mean={np.mean(scores):.4f}, std={np.std(scores):.4f}"
        score_type_labels = {
            "l2": "Residual-norm",
            "mahalanobis": "Mahalanobis score",
            "box": "Abs-residual",
        }

        label = score_type_labels[self.nonconformity]
        print(f"  {label} stats: {score_stats_msg}")

        # Q-value interpretation varies by method
        q_label = "radius" if self.nonconformity in ("l2", "mahalanobis") else "value"
        print(f"  Calibrated q-value ({q_label}): {q:.4f}")

        # Coverage diagnostics
        print(f"  Empirical coverage on calibration set: {empirical_coverage:.4f}")
        print(f"  Target coverage: {target_coverage:.4f}")

        if abs(empirical_coverage - target_coverage) > COVERAGE_WARNING_THRESHOLD:
            print(
                f"  WARNING: Gap between empirical ({empirical_coverage:.4f}) "
                f"and target ({target_coverage:.4f}) coverage!"
            )

    def calibrate(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        alpha: float,
        mask: Optional[np.ndarray] = None,
    ) -> float:
        """
        Calibrate the conformal predictor using calibration data.

        WARNING: Ensure this data is SEPARATE from evaluation data to maintain
        conformal prediction validity. Use run_conformal_analysis() for automatic
        proper splitting, or manually ensure train/test separation.

        Args:
            predictions: Model predictions [num_steps, num_nodes, 2]
            ground_truth: Ground truth values [num_steps, num_nodes, 2]
            alpha: Miscoverage level (target coverage = 1 - alpha)
            mask: Optional boolean mask for valid nodes [num_nodes] (e.g., fluid nodes only)

        Returns:
            Calibrated quantile value q
        """
        # Validate inputs (conformity functions will also validate, but catch early)
        _validate_alpha(alpha)

        # Compute nonconformity scores using the appropriate method
        scores = self.compute_nonconformity_scores(predictions, ground_truth, mask)

        # Compute conformal quantile
        q = self._compute_conformal_quantile(scores, alpha)

        # Print diagnostic information
        self._print_calibration_diagnostics(scores, q, alpha)

        # Store calibration results
        self.q_value = q
        self.calibration_residuals = scores
        self.calibrated = True

        return q

    def evaluate_coverage(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> float:
        """
        Evaluate empirical coverage for selected nonconformity.

        WARNING: This data should be SEPARATE from calibration data to get valid
        coverage estimates. Using the same data will give artificially perfect results.

        Args:
            predictions: Model predictions [num_steps, num_nodes, 2]
            ground_truth: Ground truth values [num_steps, num_nodes, 2]
            mask: Optional boolean mask for valid nodes [num_nodes] (e.g., fluid nodes only)

        Returns:
            Empirical coverage rate (fraction of points within prediction sets)
        """
        if not self.is_ready:
            raise ValueError("Predictor must be calibrated before evaluating coverage")

        # Special case for Mahalanobis: ensure covariance matrix is available
        if self.nonconformity == "mahalanobis" and self._covariance_inv is None:
            raise ValueError(
                "Mahalanobis mode requires calibration first to estimate covariance"
            )

        # Compute nonconformity scores using the same method as calibration
        scores = self.compute_nonconformity_scores(predictions, ground_truth, mask)

        # Check coverage: fraction of scores within the calibrated threshold
        covered = scores <= self.q_value
        return float(np.mean(covered)) if covered.size > 0 else 0.0

    # Intervals are meaningful only for 'box'; for 'l2' return NaNs
    def predict_intervals(
        self, predictions: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate prediction intervals (only meaningful for 'box' nonconformity).

        Args:
            predictions: Model predictions [num_steps, num_nodes, 2]
            mask: Optional boolean mask for valid nodes [num_nodes] (e.g., fluid nodes only)

        Returns:
            Prediction intervals [num_steps, num_nodes, 2, 2] where last dim is [lower, upper]
            Returns NaN arrays for non-box methods

        Raises:
            ValueError: If predictor not calibrated
        """
        if not self.is_ready:
            raise ValueError("Predictor must be calibrated before generating intervals")

        if self.nonconformity != "box":
            # Intervals only meaningful for box method
            return np.full((*predictions.shape, 2), np.nan, dtype=float)

        # Construct componentwise intervals: [ŷ - q, ŷ + q]
        intervals = np.zeros((*predictions.shape, 2))
        intervals[..., 0] = predictions - self.q_value  # Lower bounds
        intervals[..., 1] = predictions + self.q_value  # Upper bounds

        # Apply simple node mask if provided
        if mask is not None:
            if mask.shape[0] != predictions.shape[1]:
                raise ValueError(
                    f"Mask length {mask.shape[0]} != num_nodes {predictions.shape[1]}"
                )
            # Set intervals to NaN for masked-out nodes (across all timesteps)
            intervals[:, ~mask, :, :] = np.nan

        return intervals

    def _calculate_area_and_axes(self) -> Tuple[float, List[float]]:
        """
        Calculate area and axis widths based on nonconformity type.

        Returns:
            Tuple of (area, axis_widths)
        """
        diameter = 2.0 * float(self.q_value)

        if self.nonconformity == "l2":
            # L2 ball: circular area
            area = float(np.pi * (self.q_value**2))
            axis_widths = [diameter, diameter]

        elif self.nonconformity == "mahalanobis":
            # Mahalanobis ellipse: depends on covariance structure
            if self._covariance_matrix is None:
                # Fallback to circular if no covariance available
                area = float(np.pi * (self.q_value**2))
                axis_widths = [diameter, diameter]
            else:
                det = float(np.linalg.det(self._covariance_matrix))
                area = float(np.pi * (self.q_value**2) * np.sqrt(max(det, 0.0)))
                eigvals = np.linalg.eigvalsh(self._covariance_matrix)
                eigvals = np.clip(eigvals, 0.0, None)
                axis_widths = [
                    float(2.0 * self.q_value * np.sqrt(ev)) for ev in eigvals
                ]
        else:  # box
            # Box area: (2q)^2 in 2D
            area = float((2.0 * self.q_value) ** 2)
            axis_widths = [diameter, diameter]

        return area, axis_widths

    def calculate_interval_widths(self) -> Dict[str, float]:
        """
        Calculate statistics about prediction set widths.

        Returns:
            Dictionary with width statistics including area and axis widths
        """
        if not self.is_ready:
            raise ValueError("Predictor must be calibrated before computing widths")

        # Diameter convention (consistent across all methods)
        diameter = 2.0 * float(self.q_value)

        # Calculate method-specific area and axis widths
        area, axis_widths = self._calculate_area_and_axes()

        return {
            "mean_width": diameter,
            "median_width": diameter,
            "std_width": 0.0,
            "min_width": diameter,
            "max_width": diameter,
            "q25_width": diameter,
            "q75_width": diameter,
            "area": area,
            "axis_widths": axis_widths,
        }

    def get_calibration_stats(self) -> Dict[str, float]:
        """Get statistics about the calibration residuals."""
        if not self.is_ready:
            raise ValueError("Predictor not yet calibrated")

        residuals = self.calibration_residuals
        stats = {
            "mean_residual": np.mean(residuals),
            "median_residual": np.median(residuals),
            "std_residual": np.std(residuals),
            "min_residual": np.min(residuals),
            "max_residual": np.max(residuals),
            "q_value": self.q_value,
            "num_residuals": len(residuals),
            "theoretical_coverage": 1.0
            - (np.sum(residuals <= self.q_value) / len(residuals)),
        }

        return stats


def run_conformal_analysis(
    predictions_file: str,
    alpha: float = 0.1,
    calib_ratio: float = 0.5,
    use_fluid_mask: bool = True,
    nonconformity: str = "l2",
    random_seed: int = 42,
    normalize_widths: bool = False,
    aux_ratio: float = 0.0,
) -> Dict:
    """
    Run complete conformal prediction analysis on extracted predictions.

    WARNING: This implementation treats (node, time) pairs as exchangeable units,
    which is an approximation for temporal mesh data.

    Args:
        predictions_file: Path to predictions pickle file
        alpha: Miscoverage level (target coverage = 1 - alpha)
        calib_ratio: Fraction of data to use for calibration (default: 0.5)
        use_fluid_mask: Whether to use fluid node mask for analysis
        nonconformity: Nonconformity score type ('l2', 'mahalanobis', 'box')
        random_seed: Random seed for reproducible splits
        normalize_widths: Whether to compute normalized width statistics

    Returns:
        Dictionary with comprehensive analysis results

    Raises:
        ValueError: If inputs are invalid or data is insufficient
    """
    # Prepare data using shared helper (eliminates duplication)
    prepared = _prepare_conformal_data(
        predictions_file, calib_ratio, use_fluid_mask, random_seed, aux_ratio=aux_ratio
    )

    # Validate inputs
    _validate_conformal_inputs(
        prepared["predictions"],
        prepared["ground_truth"],
        alpha,
        calib_ratio,
        prepared["num_steps"],
    )

    # Initialize predictor and perform split conformal prediction
    predictor = ConformalPredictor(nonconformity=nonconformity)

    # Split data using simple timestep indices
    calib_predictions = prepared["predictions"][prepared["calib_indices"]]
    calib_ground_truth = prepared["ground_truth"][prepared["calib_indices"]]
    test_predictions = prepared["predictions"][prepared["test_indices"]]
    test_ground_truth = prepared["ground_truth"][prepared["test_indices"]]

    # If Mahalanobis with auxiliary split, estimate and freeze covariance on auxiliary data
    if nonconformity == "mahalanobis" and prepared["aux_indices"].size > 0:
        aux_predictions = prepared["predictions"][prepared["aux_indices"]]
        aux_ground_truth = prepared["ground_truth"][prepared["aux_indices"]]
        predictor.set_mahalanobis_cov_from_data(
            aux_predictions, aux_ground_truth, prepared["fluid_mask"]
        )

    # Calibration phase: compute quantile from calibration data
    q_value = predictor.calibrate(
        calib_predictions, calib_ground_truth, alpha, prepared["fluid_mask"]
    )

    # Evaluation phase: assess coverage on test data
    coverage = predictor.evaluate_coverage(
        test_predictions, test_ground_truth, prepared["fluid_mask"]
    )

    # Compute prediction set statistics
    width_stats = predictor.calculate_interval_widths()

    # Optionally compute normalized width statistics
    if normalize_widths:
        width_stats = _compute_normalized_width_stats(width_stats, prepared["data"])

    calib_stats = predictor.get_calibration_stats()

    # Visualization data (use analysis.predictor_viz if needed)
    visualization_data = None

    # Compile comprehensive results with visualization data
    results = {
        # Core conformal prediction results
        "alpha": alpha,
        "target_coverage": 1 - alpha,
        "empirical_coverage": coverage,
        "q_value": q_value,
        "nonconformity_method": nonconformity,
        # Statistical analysis
        "calibration_stats": calib_stats,
        "width_stats": width_stats,
        # Data split information
        "num_calib_steps": len(prepared["calib_indices"]),
        "num_test_steps": len(prepared["test_indices"]),
        "calib_ratio": calib_ratio,
        "random_seed": random_seed,
        # Dataset metadata
        "num_steps": prepared["num_steps"],
        "num_nodes": prepared["num_nodes"],
        "use_fluid_mask": use_fluid_mask,
        "model_name": prepared["data"].get("model_name", "unknown"),
        "delta_t": prepared["data"].get("delta_t", 0.01),
        # Coverage validation
        "coverage_gap": abs(coverage - (1 - alpha)),
        "valid_coverage": abs(coverage - (1 - alpha)) <= COVERAGE_WARNING_THRESHOLD,
        # Visualization data for plotting
        "visualization": visualization_data,
    }

    return results


def run_alpha_sweep(
    predictions_file: str,
    alphas: List[float],
    calib_ratio: float = 0.5,
    use_fluid_mask: bool = True,
    nonconformity: str = "l2",
    random_seed: int = 42,
    normalize_widths: bool = False,
    conservative_margin: float = 0.01,
    aux_ratio: float = 0.0,
) -> List[Dict]:
    """
    Efficiently run conformal prediction analysis across multiple alpha values.

    OPTIMIZED: Computes nonconformity scores only ONCE, then varies quantiles.
    This provides massive speedup for multiple alphas.

    Args:
        predictions_file: Path to predictions pickle file
        alphas: List of miscoverage levels to evaluate
        calib_ratio: Fraction of data to use for calibration
        use_fluid_mask: Whether to use fluid node mask for analysis
        nonconformity: Nonconformity score type
        random_seed: Random seed for reproducible splits

    Returns:
        List of results dictionaries for each alpha

    Raises:
        ValueError: If any alpha is invalid or data is insufficient
    """
    # Validate alpha values efficiently
    for alpha in alphas:
        _validate_alpha(alpha)

    print(f"Running OPTIMIZED alpha sweep for {len(alphas)} values: {alphas}")

    # Prepare data using shared helper (eliminates duplication)
    prepared = _prepare_conformal_data(
        predictions_file, calib_ratio, use_fluid_mask, random_seed, aux_ratio=aux_ratio
    )

    # Validate inputs once
    _validate_conformal_inputs(
        prepared["predictions"],
        prepared["ground_truth"],
        alphas[0],
        calib_ratio,
        prepared["num_steps"],
    )

    # Split data using simple timestep indices
    calib_predictions = prepared["predictions"][prepared["calib_indices"]]
    calib_ground_truth = prepared["ground_truth"][prepared["calib_indices"]]
    test_predictions = prepared["predictions"][prepared["test_indices"]]
    test_ground_truth = prepared["ground_truth"][prepared["test_indices"]]

    # Compute scores only ONCE for all alphas
    print(f"  Computing {nonconformity} scores once for all alphas...")
    predictor = ConformalPredictor(nonconformity=nonconformity)

    # If Mahalanobis with auxiliary split, estimate and freeze covariance on auxiliary data
    if nonconformity == "mahalanobis" and prepared["aux_indices"].size > 0:
        aux_predictions = prepared["predictions"][prepared["aux_indices"]]
        aux_ground_truth = prepared["ground_truth"][prepared["aux_indices"]]
        predictor.set_mahalanobis_cov_from_data(
            aux_predictions, aux_ground_truth, prepared["fluid_mask"]
        )

    # Compute calibration scores once (needed for all quantiles)
    calib_scores = predictor.compute_nonconformity_scores(
        calib_predictions, calib_ground_truth, prepared["fluid_mask"]
    )

    # Compute test scores once (needed for all coverage evaluations)
    test_scores = predictor.compute_nonconformity_scores(
        test_predictions, test_ground_truth, prepared["fluid_mask"]
    )

    print(f"  Calibration scores: {len(calib_scores)} samples")
    print(f"  Test scores: {len(test_scores)} samples")

    results = []
    for alpha in alphas:
        print(f"  Computing quantile for alpha = {alpha:.3f}...")

        # Compute quantile for this alpha (fast: O(m) partition)
        q_value = compute_conformal_quantile(
            calib_scores, alpha, conservative=True, margin=conservative_margin
        )

        # Evaluate coverage using precomputed test scores (fast: O(m) comparison)
        covered = test_scores <= q_value
        coverage = float(np.mean(covered)) if covered.size > 0 else 0.0

        # Set predictor state for width calculations
        predictor._set_calibrated_state(q_value, calib_scores)

        # Compute statistics
        width_stats = predictor.calculate_interval_widths()

        # Optionally compute normalized width statistics
        if normalize_widths:
            width_stats = _compute_normalized_width_stats(width_stats, prepared["data"])

        # Compile results for this alpha
        result = {
            # Core conformal prediction results
            "alpha": alpha,
            "target_coverage": 1 - alpha,
            "empirical_coverage": coverage,
            "q_value": q_value,
            "nonconformity_method": nonconformity,
            # Statistical analysis
            "width_stats": width_stats,
            # Data split information (shared across alphas)
            "num_calib_steps": len(prepared["calib_indices"]),
            "num_test_steps": len(prepared["test_indices"]),
            "calib_ratio": calib_ratio,
            "random_seed": random_seed,
            # Dataset metadata
            "num_nodes": prepared["num_nodes"],
            "use_fluid_mask": use_fluid_mask,
            "model_name": prepared["data"].get("model_name", "unknown"),
            # Coverage validation
            "coverage_gap": abs(coverage - (1 - alpha)),
            "valid_coverage": abs(coverage - (1 - alpha)) <= COVERAGE_WARNING_THRESHOLD,
        }

        results.append(result)

    return results


def run_alpha_sweep_compare(
    predictions_file: str,
    alphas: List[float],
    calib_ratio: float = 0.5,
    use_fluid_mask: bool = True,
    modes: List[str] = None,
    random_seed: int = 42,
    normalize_widths: bool = False,
    conservative_margin: float = 0.01,
    aux_ratio: float = 0.0,
) -> List[Dict]:
    """
    Run conformal analysis comparing multiple nonconformity modes across alpha values.

    Efficiently computes results for all mode-alpha combinations using shared
    data splits to ensure fair comparison.

    Args:
        predictions_file: Path to predictions pickle file
        alphas: List of miscoverage levels to evaluate
        calib_ratio: Fraction of data to use for calibration
        use_fluid_mask: Whether to use fluid node mask for analysis
        modes: List of nonconformity modes to compare (default: all three)
        random_seed: Random seed for reproducible splits

    Returns:
        List of comparison dictionaries, one per alpha value

    Raises:
        ValueError: If inputs are invalid
    """
    if modes is None:
        modes = list(SUPPORTED_NONCONFORMITY_MODES)

    # Validate inputs efficiently
    for alpha in alphas:
        _validate_alpha(alpha)

    for mode in modes:
        if mode not in SUPPORTED_NONCONFORMITY_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be one of {SUPPORTED_NONCONFORMITY_MODES}"
            )

    print(f"Running comparison across {len(modes)} modes and {len(alphas)} alphas")
    print(f"Modes: {modes}")
    print(f"Alphas: {alphas}")

    # Prepare data using shared helper (eliminates duplication)
    prepared = _prepare_conformal_data(
        predictions_file, calib_ratio, use_fluid_mask, random_seed, aux_ratio=aux_ratio
    )

    # Validate inputs once
    _validate_conformal_inputs(
        prepared["predictions"],
        prepared["ground_truth"],
        alphas[0],
        calib_ratio,
        prepared["num_steps"],
    )

    # Split data using simple timestep indices
    calib_predictions = prepared["predictions"][prepared["calib_indices"]]
    calib_ground_truth = prepared["ground_truth"][prepared["calib_indices"]]
    test_predictions = prepared["predictions"][prepared["test_indices"]]
    test_ground_truth = prepared["ground_truth"][prepared["test_indices"]]

    # Precompute scores for all modes (massive speedup)
    print("  Precomputing scores for all modes...")
    mode_scores = {}

    for mode in modes:
        print(f"    Computing {mode} scores once...")
        predictor = ConformalPredictor(nonconformity=mode)

        # If Mahalanobis with auxiliary split, estimate and freeze covariance on auxiliary data
        if mode == "mahalanobis" and prepared["aux_indices"].size > 0:
            aux_predictions = prepared["predictions"][prepared["aux_indices"]]
            aux_ground_truth = prepared["ground_truth"][prepared["aux_indices"]]
            predictor.set_mahalanobis_cov_from_data(
                aux_predictions, aux_ground_truth, prepared["fluid_mask"]
            )

        # Compute calibration and test scores once per mode
        calib_scores = predictor.compute_nonconformity_scores(
            calib_predictions, calib_ground_truth, prepared["fluid_mask"]
        )
        test_scores = predictor.compute_nonconformity_scores(
            test_predictions, test_ground_truth, prepared["fluid_mask"]
        )

        mode_scores[mode] = {
            "calib_scores": calib_scores,
            "test_scores": test_scores,
            "predictor": predictor,  # Keep for width calculations
        }

    rows: List[Dict] = []
    for alpha in alphas:
        row: Dict = {"alpha": float(alpha), "target_coverage": 1 - float(alpha)}

        for mode in modes:
            print(f"    alpha={alpha:.3f}, mode={mode} (using precomputed scores)")

            # Get precomputed scores (no recomputation!)
            calib_scores = mode_scores[mode]["calib_scores"]
            test_scores = mode_scores[mode]["test_scores"]
            predictor = mode_scores[mode]["predictor"]

            # Compute quantile for this alpha (fast: O(m) partition)
            q_value = compute_conformal_quantile(
                calib_scores, alpha, conservative=True, margin=conservative_margin
            )

            # Evaluate coverage using precomputed test scores (fast: O(m) comparison)
            covered = test_scores <= q_value
            coverage = float(np.mean(covered)) if covered.size > 0 else 0.0

            # Set predictor state for width calculations
            predictor._set_calibrated_state(q_value, mode_scores[mode]["calib_scores"])

            # Get width statistics
            width_stats = predictor.calculate_interval_widths()

            # Optionally compute normalized width statistics
            if normalize_widths:
                width_stats = _compute_normalized_width_stats(
                    width_stats, prepared["data"]
                )

            # Store results for this mode-alpha combination
            # Store results with normalized widths if requested
            result_dict = {
                "coverage": float(coverage),
                "q": float(q_value),
                "mean_width": float(width_stats["mean_width"]),
                "area": float(width_stats["area"]),
                "valid_coverage": abs(coverage - (1 - alpha))
                <= COVERAGE_WARNING_THRESHOLD,
            }

            # Add normalized width if computed
            if normalize_widths and "normalized_width" in width_stats:
                result_dict["normalized_width"] = float(width_stats["normalized_width"])

            row[mode] = result_dict

        rows.append(row)

    return rows
