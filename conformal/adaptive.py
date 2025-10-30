#!/usr/bin/env python3
"""
Spatially Adaptive Conformal Prediction for Mesh-Based Simulations

Implements locally-scaled split conformal prediction where a meta-model learns
to predict spatially varying uncertainty scales from domain-aware features.
This enables adaptive prediction sets that capture heterogeneous uncertainty
patterns across the mesh topology.

Method:
    1. Train meta-model to predict scale ŝ(x) from features (auxiliary set)
    2. Compute normalized scores: u_i = s_i / ŝ(x_i) on calibration set
    3. Find quantile q from normalized scores
    4. Test-time radius: r(x) = q * ŝ(x)

Preserves marginal coverage P(Y ∈ C_α(X)) ≥ 1-α while achieving spatially
adaptive sets that shrink in low-uncertainty regions (smooth flow) and expand
in high-uncertainty regions (boundaries, wake zones).

Reference:
    See paper.tex §4.4 (Spatially Adaptive Scaling) for mathematical framework.
    Feature engineering detailed in features.py.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


class BoostedConformalPredictor:
    """
    Boosted-CP predictor for 2D vector targets (velocity). Uses L2-residuals
    as the base nonconformity and learns a scale function via a meta-model.

    The meta-model is intentionally simple (Gradient Boosting) and trained on
    hand-crafted features readily available in the predictions pickle, so this
    module has zero dependency on the training code path.
    """

    def __init__(
        self,
        random_state: int = 42,
        use_log_scale: bool = True,
        clip_lower_quantile: float = 0.02,
        clip_upper_quantile: float = 0.99,
        shrinkage_lambda: float = 0.05,
        model_kind: str = "gbr_median",
        quantile_method: str = "order",  # 'order' (finite-sample) or 'linear' (numpy) or 'harrell-davis'
        trim_frac: float = 0.0,  # optional tail trimming on u (0 keeps validity)
        model_n_jobs: int = 4,
        coverage_margin: float = 0.005,  # inflate quantile level by this amount
        use_enhanced_features: bool = False,  # Enable physics-informed features
        enable_expensive_features: bool = True,  # Enable computationally expensive features
    ) -> None:
        # Lazy import to avoid hard dependency if sklearn is absent
        try:
            from sklearn.ensemble import GradientBoostingRegressor  # type: ignore
            from sklearn.ensemble import RandomForestRegressor  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "BoostedCP requires scikit-learn. Please `pip install scikit-learn`."
            ) from exc

        # Choose a robust meta-model for scale prediction (median-focused by default)
        mk = (model_kind or "gbr_median").lower()
        if mk == "gbr_median":
            # Gradient boosting in quantile mode (median)
            self._model = GradientBoostingRegressor(
                loss="quantile",
                alpha=0.5,
                n_estimators=400,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.9,
                random_state=random_state,
            )
        elif mk == "rf":
            self._model = RandomForestRegressor(
                n_estimators=600,
                max_depth=None,
                min_samples_leaf=10,
                max_features="sqrt",
                n_jobs=int(model_n_jobs),
                random_state=random_state,
            )
        else:
            # Fallback GBR in squared loss mode
            self._model = GradientBoostingRegressor(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                random_state=random_state,
            )

        self._fitted: bool = False
        self._q_ratio: Optional[float] = None
        self._eps: float = 1e-8

        # Robustification controls
        self._use_log: bool = bool(use_log_scale)
        self._clip_lo_q: float = float(clip_lower_quantile)
        self._clip_hi_q: float = float(clip_upper_quantile)
        self._shrink_lambda: float = float(shrinkage_lambda)
        self._q_method: str = quantile_method
        self._trim_frac: float = float(trim_frac)
        self._model_n_jobs: int = int(model_n_jobs)
        self._cov_margin: float = float(coverage_margin)
        self._use_enhanced: bool = bool(use_enhanced_features)
        self._enable_expensive: bool = bool(enable_expensive_features)

        # Enhanced feature engine (lazy initialization)
        self._feature_engine = None

        # Learned at calibration time
        self._clip_lo_: Optional[float] = None
        self._clip_hi_: Optional[float] = None
        self._global_med_res_: Optional[float] = None
        self._q_ratio: Optional[float] = None
        self._fitted: bool = False

    # -------------------------------
    # Feature engineering
    # -------------------------------
    def _build_features(
        self,
        predictions: np.ndarray,  # [T, N_active, 2] - already masked
        velocities: np.ndarray,  # [T, N_active, 2] - already masked
        metadata: Optional[Dict],
        mask: Optional[
            np.ndarray
        ] = None,  # [N_total] - original mask for metadata indexing
    ) -> np.ndarray:
        """
        Build per-node-time tabular features φ_i.

        CRITICAL FIX: Properly handle enhanced features with masked data.
        Enhanced features need full mesh metadata but work on masked predictions.
        """
        if self._use_enhanced and metadata is not None:
            # Use sophisticated physics-informed features
            if self._feature_engine is None:
                try:
                    from .features import PhysicsFeatures

                    # Use "full" mode for all 17 features from paper's 6 categories
                    self._feature_engine = PhysicsFeatures(mode="full")
                    print("Using enhanced physics-informed features for BCP (p=17)")
                except ImportError as e:
                    print(
                        f"Enhanced features not available ({e}), falling back to basic features"
                    )
                    self._use_enhanced = False
                    return self._build_basic_features(
                        predictions, velocities, metadata, mask
                    )

            # CRITICAL FIX: Enhanced features need full mesh data, then extract for active nodes
            T, N_active, _ = predictions.shape

            # Get full mesh metadata
            full_positions = metadata.get("mesh_positions")  # [N_total, 2]
            full_node_types = metadata.get("node_types")  # [N_total]

            if full_positions is None or full_node_types is None:
                print(
                    "  Missing mesh metadata for enhanced features, falling back to basic"
                )
                return self._build_basic_features(predictions, velocities, metadata)

            N_total = len(full_positions)

            # Reconstruct full predictions/velocities with zeros for non-active nodes
            # Handle both 2D (cylinder) and 3D (flag) velocity data
            vel_dim = predictions.shape[2]  # Get actual velocity dimension
            full_predictions = np.zeros((T, N_total, vel_dim))
            full_velocities = np.zeros((T, N_total, vel_dim))

            if mask is not None:
                full_predictions[:, mask, :] = predictions
                full_velocities[:, mask, :] = velocities
            else:
                full_predictions = predictions
                full_velocities = velocities

            # Compute enhanced features on full mesh
            full_features = self._feature_engine.compute_enhanced_features(
                full_predictions, full_velocities, metadata
            )  # [T*N_total, F]

            # Extract features only for active nodes
            if mask is not None:
                # Create mask for flattened (T*N_total) array
                mask_2d = np.broadcast_to(mask[None, :], (T, N_total))  # [T, N_total]
                mask_flat = mask_2d.reshape(-1)  # [T*N_total]
                active_features = full_features[mask_flat]  # [T*N_active, F]
            else:
                active_features = full_features

            print(f"  Enhanced BCP features: {active_features.shape[1]} dimensions")
            return active_features
        else:
            # Use basic feature set (original implementation)
            return self._build_basic_features(predictions, velocities, metadata, mask)

    def _build_basic_features(
        self,
        predictions: np.ndarray,  # [T, N_active, 2] - already masked
        velocities: np.ndarray,  # [T, N_active, 2] - already masked
        metadata: Optional[Dict],
        mask: Optional[
            np.ndarray
        ] = None,  # [N_total] - original mask for spatial features
    ) -> np.ndarray:
        """
        Basic feature engineering (fixed for masked data).

        CRITICAL: Now expects already-masked data for consistent shapes.
        """
        T, N_active, _ = predictions.shape

        # Basic kinematics features (local magnitudes)
        pred_speed = np.linalg.norm(predictions, axis=-1)  # [T, N_active]
        vel_speed = np.linalg.norm(velocities, axis=-1)  # [T, N_active]

        # Time feature (normalized index)
        time_feat = (np.arange(T, dtype=np.float32) / max(1, T - 1)).reshape(T, 1)
        time_feat = np.repeat(time_feat, N_active, axis=1)  # [T, N_active]

        # Assemble features
        feats: List[np.ndarray] = [
            pred_speed[..., None],  # [T, N_active, 1]
            vel_speed[..., None],  # [T, N_active, 1]
            time_feat[..., None],  # [T, N_active, 1]
        ]

        # Handle spatial features - need to extract only for active nodes
        if metadata is not None:
            pos = metadata.get("mesh_positions", None)  # [N_total, 2]
            node_type = metadata.get("node_types", None)  # [N_total]

            # CRITICAL FIX: Add spatial features for basic BCP
            if pos is not None and mask is not None:
                # Extract positions for active (fluid) nodes only
                fluid_positions = pos[mask]  # [N_active, 2]

                # Add spatial coordinates as features
                x_coords = np.broadcast_to(
                    fluid_positions[:, 0][None, :], (T, N_active)
                )  # [T, N_active]
                y_coords = np.broadcast_to(
                    fluid_positions[:, 1][None, :], (T, N_active)
                )  # [T, N_active]

                # Add to features for spatial variation
                feats.extend(
                    [
                        x_coords[..., None],  # [T, N_active, 1]
                        y_coords[..., None],  # [T, N_active, 1]
                    ]
                )

                # Add wall distance if node types available
                if node_type is not None:
                    try:
                        from scipy.spatial import cKDTree

                        # Find wall nodes
                        unique_types = np.unique(node_type)
                        majority_type = np.bincount(node_type).argmax()
                        wall_candidates = [
                            t for t in unique_types if t > 0 and t != majority_type
                        ]

                        if wall_candidates:
                            wall_type = min(wall_candidates)
                            wall_mask = node_type == wall_type
                            wall_positions = pos[wall_mask]

                            if len(wall_positions) > 0:
                                # Compute wall distances for fluid nodes
                                tree = cKDTree(wall_positions)
                                wall_distances, _ = tree.query(fluid_positions)

                                # Broadcast to time dimension
                                wall_dist_feat = np.broadcast_to(
                                    wall_distances[None, :], (T, N_active)
                                )
                                feats.append(
                                    wall_dist_feat[..., None]
                                )  # [T, N_active, 1]

                    except Exception as e:
                        print(f"  Warning: Could not compute wall distances: {e}")
                        pass

        # Concatenate along feature axis and flatten [T, N_active, F]
        X = np.concatenate(feats, axis=-1).reshape(T * N_active, -1)
        print(f"  BCP features shape: {X.shape} for {T}×{N_active} masked data")
        return X

    # -------------------------------
    # Calibration (fit + quantile on normalized residuals)
    # -------------------------------
    def calibrate(
        self,
        calib_predictions: np.ndarray,  # [Tc, N, 2]
        calib_ground_truth: np.ndarray,  # [Tc, N, 2]
        calib_velocities: np.ndarray,  # [Tc, N, 2]
        metadata: Optional[Dict],
        alpha: float,
        mask: Optional[np.ndarray] = None,  # [N]
    ) -> float:
        # CRITICAL FIX: Apply mask consistently to ALL data structures

        # Apply mask to predictions/ground_truth BEFORE processing
        if mask is not None:
            calib_predictions_masked = calib_predictions[:, mask, :]  # [Tc, N_fluid, 2]
            calib_ground_truth_masked = calib_ground_truth[
                :, mask, :
            ]  # [Tc, N_fluid, 2]
            calib_velocities_masked = calib_velocities[:, mask, :]  # [Tc, N_fluid, 2]
        else:
            calib_predictions_masked = calib_predictions
            calib_ground_truth_masked = calib_ground_truth
            calib_velocities_masked = calib_velocities

        # Residuals (L2 norm) per node-time - now consistent shapes
        res = np.linalg.norm(
            calib_ground_truth_masked - calib_predictions_masked, axis=-1
        )  # [Tc, N_fluid]

        # Build features on masked data to ensure shape consistency
        X = self._build_features(
            calib_predictions_masked, calib_velocities_masked, metadata, mask
        )  # [Tc*N_fluid, F] - consistent with residuals

        y = res.reshape(-1)  # [Tc*N_fluid] - now matches X shape

        # Train meta-model on (log-)residuals
        if self._use_log:
            y_target = np.log(np.maximum(y, self._eps))
        else:
            y_target = y
        self._model.fit(X, y_target)
        self._fitted = True

        # Predict on calibration and form normalized residuals u = s / s_hat
        raw_pred = self._model.predict(X)
        s_hat = np.exp(raw_pred) if self._use_log else raw_pred
        s_hat = np.maximum(s_hat, self._eps)

        # Winsorize and shrink scales to avoid heavy tails
        lo = float(np.quantile(s_hat, self._clip_lo_q))
        hi = float(np.quantile(s_hat, self._clip_hi_q))
        self._clip_lo_, self._clip_hi_ = lo, hi
        s_hat = np.clip(s_hat, lo, hi)

        med_res = float(np.median(y))
        self._global_med_res_ = med_res
        if self._shrink_lambda > 0.0:
            s_hat = (1.0 - self._shrink_lambda) * s_hat + self._shrink_lambda * med_res
        u = y / s_hat

        # Compute conformal quantile on u
        if self._trim_frac > 0.0:
            t = int(max(0, np.floor(self._trim_frac * u.size)))
            u_sorted = np.sort(u)
            u_eff = u_sorted[: max(1, u.size - t)]
        else:
            u_eff = u

        # Effective sample size for logging
        m_eff = u_eff.size
        k_val: Optional[int] = None

        # Slightly conservative quantile to ensure coverage ≥ target
        p_target = min(max(1.0 - alpha + self._cov_margin, 0.0), 0.999999)

        if self._q_method == "order":
            k_val = int(np.ceil((m_eff + 1) * p_target))
            k_val = min(max(k_val, 1), m_eff)
            q_ratio = float(np.partition(u_eff, k_val - 1)[k_val - 1])
        elif self._q_method in ("hd", "harrell-davis", "harrell_davis"):
            q_ratio = float(self._harrell_davis_quantile(u_eff, p_target))
        else:
            # Numpy's interpolated quantile (may reduce conservatism slightly)
            q_ratio = float(np.quantile(u_eff, p_target))
        self._q_ratio = q_ratio
        self.q_value = q_ratio  # Add q_value attribute for compatibility

        # Debug prints
        k_str = str(k_val) if k_val is not None else "n/a"
        print(
            f"  BoostedCP calibration: m={m_eff}, k={k_str}, method={self._q_method}, alpha={alpha:.3f}"
        )
        print(
            f"  Residual stats: mean={np.mean(y):.4f}, std={np.std(y):.4f}; "
            f"u-quantile (q_ratio)={q_ratio:.4f}"
        )

        # DIAGNOSTIC: Check if meta-model is learning spatial variation
        scale_stats = {
            "mean_scale": float(np.mean(s_hat)),
            "std_scale": float(np.std(s_hat)),
            "min_scale": float(np.min(s_hat)),
            "max_scale": float(np.max(s_hat)),
            "scale_cv": (
                float(np.std(s_hat) / np.mean(s_hat)) if np.mean(s_hat) > 0 else 0.0
            ),
        }
        print(
            f"  Scale prediction stats: mean={scale_stats['mean_scale']:.4f}, "
            f"std={scale_stats['std_scale']:.4f}, cv={scale_stats['scale_cv']:.4f}"
        )

        if scale_stats["scale_cv"] < 0.01:
            print("  WARNING: Meta-model predicting nearly constant scales (CV < 1%)")
            print("  This means BCP will behave like standard conformal prediction")

        # Store scale stats for later analysis
        self._scale_stats = scale_stats

        return q_ratio

    # -------------------------------
    # Auxiliary-first protocol (fit on aux, quantile on calib)
    # -------------------------------
    def fit_scale_on_aux(
        self,
        aux_predictions: np.ndarray,
        aux_ground_truth: np.ndarray,
        aux_velocities: np.ndarray,
        metadata: Optional[Dict],
        mask: Optional[np.ndarray] = None,
    ) -> None:
        """Fit the meta-model for scale prediction on the auxiliary split only.

        This method estimates the scale model and records clipping/shrinkage
        statistics using ONLY auxiliary labels, ensuring A2 independence.
        """
        # Apply mask
        if mask is not None:
            pred_m = aux_predictions[:, mask, :]
            gt_m = aux_ground_truth[:, mask, :]
            vel_m = aux_velocities[:, mask, :]
        else:
            pred_m = aux_predictions
            gt_m = aux_ground_truth
            vel_m = aux_velocities

        # Residuals and features on auxiliary
        res = np.linalg.norm(gt_m - pred_m, axis=-1)  # [Ta, N_fluid]
        X = self._build_features(pred_m, vel_m, metadata, mask)
        y = res.reshape(-1)

        # Train meta-model on (log-)residuals
        y_target = np.log(np.maximum(y, self._eps)) if self._use_log else y
        self._model.fit(X, y_target)
        self._fitted = True

        # Predict scales on aux to set clipping/shrinkage baselines
        raw_pred = self._model.predict(X)
        s_hat = np.exp(raw_pred) if self._use_log else raw_pred
        s_hat = np.maximum(s_hat, self._eps)

        lo = float(np.quantile(s_hat, self._clip_lo_q))
        hi = float(np.quantile(s_hat, self._clip_hi_q))
        self._clip_lo_, self._clip_hi_ = lo, hi

        med_res = float(np.median(y))
        self._global_med_res_ = med_res

    def calibrate_quantile_with_frozen_model(
        self,
        calib_predictions: np.ndarray,
        calib_ground_truth: np.ndarray,
        calib_velocities: np.ndarray,
        metadata: Optional[Dict],
        alpha: float,
        mask: Optional[np.ndarray] = None,
    ) -> float:
        """Compute conformal quantile on calibration with the scale model FROZEN.

        Requires prior fit_scale_on_aux(). Does not refit or update clipping/shrinkage.
        """
        if not self._fitted:
            raise ValueError("Scale model not fitted. Call fit_scale_on_aux() first.")

        if mask is not None:
            pred_m = calib_predictions[:, mask, :]
            gt_m = calib_ground_truth[:, mask, :]
            vel_m = calib_velocities[:, mask, :]
        else:
            pred_m = calib_predictions
            gt_m = calib_ground_truth
            vel_m = calib_velocities

        # Residuals on calibration
        res = np.linalg.norm(gt_m - pred_m, axis=-1)
        y = res.reshape(-1)

        # Features and predicted scales on calibration
        X = self._build_features(pred_m, vel_m, metadata, mask)
        raw_pred = self._model.predict(X)
        s_hat = np.exp(raw_pred) if self._use_log else raw_pred
        s_hat = np.maximum(s_hat, self._eps)

        # Apply previously computed clipping and shrinkage from aux
        if self._clip_lo_ is not None and self._clip_hi_ is not None:
            s_hat = np.clip(s_hat, self._clip_lo_, self._clip_hi_)
        if self._shrink_lambda > 0.0 and self._global_med_res_ is not None:
            s_hat = (
                1.0 - self._shrink_lambda
            ) * s_hat + self._shrink_lambda * self._global_med_res_

        u = y / s_hat

        # Quantile
        m_eff = u.size
        p_target = min(max(1.0 - alpha + self._cov_margin, 0.0), 0.999999)
        if self._q_method == "order":
            k_val = int(np.ceil((m_eff + 1) * p_target))
            k_val = min(max(k_val, 1), m_eff)
            q_ratio = float(np.partition(u, k_val - 1)[k_val - 1])
        elif self._q_method in ("hd", "harrell-davis", "harrell_davis"):
            q_ratio = float(self._harrell_davis_quantile(u, p_target))
        else:
            q_ratio = float(np.quantile(u, p_target))

        self._q_ratio = q_ratio
        self.q_value = q_ratio
        print(
            f"  BoostedCP (frozen) calibration: m={m_eff}, method={self._q_method}, alpha={alpha:.3f}, q_ratio={q_ratio:.4f}"
        )
        return q_ratio

    # Harrell–Davis quantile estimator (weighted average of order stats)
    def _harrell_davis_quantile(self, samples: np.ndarray, p: float) -> float:
        s = np.asarray(samples, dtype=float)
        n = s.size
        if n == 0:
            return float("nan")
        s_sorted = np.sort(s)
        # a = p*(n+1), b = (1-p)*(n+1)
        a = p * (n + 1)
        b = (1.0 - p) * (n + 1)
        try:
            # Prefer scipy if available for numerical stability
            from scipy.stats import beta as _beta  # type: ignore

            xs = np.arange(1, n + 1, dtype=float) / n
            xs_prev = np.arange(0, n, dtype=float) / n
            cdf_hi = _beta.cdf(xs, a, b)
            cdf_lo = _beta.cdf(xs_prev, a, b)
            w = cdf_hi - cdf_lo
        except Exception:
            # Fallback: simple linear quantile if SciPy is unavailable
            return float(np.quantile(s_sorted, p))

        # Normalize weights (just in case of numeric drift)
        w = np.maximum(w, 0.0)
        w_sum = np.sum(w)
        if w_sum <= 0:
            return float(np.quantile(s_sorted, p))
        w = w / w_sum
        return float(np.dot(w, s_sorted))

    # -------------------------------
    # Evaluation
    # -------------------------------
    def predict_radii(
        self,
        test_predictions: np.ndarray,
        test_velocities: np.ndarray,
        metadata: Optional[Dict],
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if not self._fitted or self._q_ratio is None:
            raise ValueError("Call calibrate() before predict_radii().")

        # CRITICAL FIX: Apply mask consistently like in calibrate()
        if mask is not None:
            test_predictions_masked = test_predictions[:, mask, :]  # [T, N_fluid, 2]
            test_velocities_masked = test_velocities[:, mask, :]  # [T, N_fluid, 2]
        else:
            test_predictions_masked = test_predictions
            test_velocities_masked = test_velocities

        # Build features on masked data (consistent with calibration)
        Xtest = self._build_features(
            test_predictions_masked, test_velocities_masked, metadata, mask
        )

        raw_pred = self._model.predict(Xtest)
        s_hat = np.exp(raw_pred) if self._use_log else raw_pred
        s_hat = np.maximum(s_hat, self._eps)

        # Apply calibrated clipping and shrinkage
        if self._clip_lo_ is not None and self._clip_hi_ is not None:
            s_hat = np.clip(s_hat, self._clip_lo_, self._clip_hi_)
        if self._shrink_lambda > 0.0 and self._global_med_res_ is not None:
            s_hat = (
                1.0 - self._shrink_lambda
            ) * s_hat + self._shrink_lambda * self._global_med_res_
        r_flat = self._q_ratio * s_hat  # radii in physical units (per node-time)

        # Reshape back to original shape
        T, N_original = test_predictions.shape[:2]
        if mask is None:
            return r_flat.reshape(T, N_original)
        else:
            # Place masked radii into full array
            T, N_active = test_predictions_masked.shape[:2]
            R = np.zeros((T, N_original), dtype=np.float32)
            mask_2d = np.broadcast_to(mask[None, :], (T, N_original))
            R[mask_2d] = r_flat
            return R

    def evaluate(
        self,
        test_predictions: np.ndarray,
        test_ground_truth: np.ndarray,
        test_velocities: np.ndarray,
        metadata: Optional[Dict],
        mask: Optional[np.ndarray] = None,
    ) -> Dict:
        # Compute per-node radii
        radii = self.predict_radii(test_predictions, test_velocities, metadata, mask)

        # Residual norms
        res = np.linalg.norm(test_ground_truth - test_predictions, axis=-1)  # [T,N]
        if mask is not None:
            res = res[:, mask]
            radii = radii[:, mask]

        # Coverage (fraction of residuals within radius)
        covered = res <= radii
        coverage = float(np.mean(covered)) if covered.size > 0 else 0.0

        # Width (mean diameter) and area (mean area of disks)
        mean_width = float(np.mean(2.0 * radii))
        mean_area = float(np.mean(np.pi * (radii**2)))

        return {
            "empirical_coverage": coverage,
            "mean_width": mean_width,
            "mean_area": mean_area,
            "q_ratio": float(self._q_ratio if self._q_ratio is not None else np.nan),
        }


# ----------------------------------------------------------------------------
# Convenience runners that mirror the existing conformal API
# ----------------------------------------------------------------------------
def _random_split_indices(
    num_steps: int, calib_ratio: float, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(num_steps)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    csz = int(num_steps * calib_ratio)
    return idx[:csz], idx[csz:]


def run_boosted_conformal_analysis(
    predictions_file: str,
    alpha: float = 0.1,
    calib_ratio: float = 0.5,
    use_fluid_mask: bool = True,
    random_seed: int = 42,
    predictor_kwargs: Optional[Dict] = None,
    aux_ratio: float = 0.0,
) -> Dict:
    """
    Run BCP analysis with proper integration to the new optimized architecture.

    Uses the same data loading, scale correction, and splitting logic as
    the standard conformal prediction for consistency.
    """
    # Import the optimized data preparation (avoid duplication)
    from .data_utils import prepare_conformal_data
    from .utils import validate_conformal_inputs

    # Use the same optimized data preparation pipeline (with optional auxiliary split)
    prepared = prepare_conformal_data(
        predictions_file, calib_ratio, use_fluid_mask, random_seed, aux_ratio=aux_ratio
    )

    # Validate inputs using the same validation logic
    validate_conformal_inputs(
        prepared["predictions"],
        prepared["ground_truth"],
        alpha,
        calib_ratio,
        prepared["num_steps"],
    )

    # Extract data with scale correction applied
    predictions = prepared["predictions"]  # Scale-corrected
    ground_truth = prepared["ground_truth"]  # Scale-corrected
    velocities = prepared["data"].get("velocities", np.zeros_like(predictions))
    metadata = prepared["data"].get("metadata", None)

    # Assemble splits
    aux_predictions = (
        predictions[prepared["aux_indices"]]
        if prepared["aux_indices"].size > 0
        else None
    )
    aux_ground_truth = (
        ground_truth[prepared["aux_indices"]]
        if prepared["aux_indices"].size > 0
        else None
    )
    aux_velocities = (
        velocities[prepared["aux_indices"]]
        if prepared["aux_indices"].size > 0
        else None
    )

    calib_predictions = predictions[prepared["calib_indices"]]
    calib_ground_truth = ground_truth[prepared["calib_indices"]]
    calib_velocities = velocities[prepared["calib_indices"]]

    test_predictions = predictions[prepared["test_indices"]]
    test_ground_truth = ground_truth[prepared["test_indices"]]
    test_velocities = velocities[prepared["test_indices"]]

    # Initialize BCP predictor
    predictor = BoostedConformalPredictor(**(predictor_kwargs or {}))

    # Fit meta-model on auxiliary if provided, then compute quantile on calibration with the model frozen
    if prepared["aux_indices"].size > 0:
        predictor.fit_scale_on_aux(
            aux_predictions,
            aux_ground_truth,
            aux_velocities,
            metadata,
            prepared["fluid_mask"],
        )
        q_ratio = predictor.calibrate_quantile_with_frozen_model(
            calib_predictions,
            calib_ground_truth,
            calib_velocities,
            metadata,
            alpha,
            prepared["fluid_mask"],
        )
    else:
        # Single-split behavior (warn about potential leakage)
        print(
            "WARNING: aux_ratio=0.0; BCP scales learned on calibration data. Consider setting aux_ratio>0."
        )
        q_ratio = predictor.calibrate(
            calib_predictions,
            calib_ground_truth,
            calib_velocities,
            metadata,
            alpha,
            prepared["fluid_mask"],
        )

    # Evaluate using scale-corrected data
    eval_stats = predictor.evaluate(
        test_predictions,
        test_ground_truth,
        test_velocities,
        metadata,
        prepared["fluid_mask"],
    )

    # CRITICAL FIX: Compute actual spatially-varying width statistics
    # Get the actual per-node radii to compute proper statistics
    test_radii = predictor.predict_radii(
        test_predictions, test_velocities, metadata, prepared["fluid_mask"]
    )

    # Apply mask if needed
    if prepared["fluid_mask"] is not None:
        test_radii = test_radii[:, prepared["fluid_mask"]]

    # Compute width statistics from actual spatially-varying radii
    test_widths = 2.0 * test_radii  # Convert radii to diameters
    width_stats = {
        "mean_width": float(np.mean(test_widths)),
        "median_width": float(np.median(test_widths)),
        "std_width": float(np.std(test_widths)),  # Now shows actual variation!
        "min_width": float(np.min(test_widths)),
        "max_width": float(np.max(test_widths)),
        "q25_width": float(np.percentile(test_widths, 25)),
        "q75_width": float(np.percentile(test_widths, 75)),
        "area": eval_stats["mean_area"],
        # Add spatial variation diagnostics
        "radius_std": float(np.std(test_radii)),
        "radius_cv": (
            float(np.std(test_radii) / np.mean(test_radii))
            if np.mean(test_radii) > 0
            else 0.0
        ),
        "spatial_variation": {
            "min_radius": float(np.min(test_radii)),
            "max_radius": float(np.max(test_radii)),
            "radius_range": float(np.max(test_radii) - np.min(test_radii)),
            "num_unique_radii": len(np.unique(np.round(test_radii, 6))),
        },
    }

    # Add normalized width statistics for paper comparison
    from .data_utils import compute_normalized_width_stats

    width_stats = compute_normalized_width_stats(width_stats, prepared["data"])

    # Compose results with consistent field names (match standard CP format)
    return {
        "alpha": alpha,
        "target_coverage": 1 - alpha,
        "empirical_coverage": eval_stats["empirical_coverage"],
        "q_value": q_ratio,  # For consistency with standard CP
        "width_stats": width_stats,  # For consistency with standard CP
        "calibration_stats": {  # Dummy stats for consistency
            "mean_residual": eval_stats.get("mean_residual", 0.0),
            "std_residual": eval_stats.get("std_residual", 0.0),
            "min_residual": eval_stats.get("min_residual", 0.0),
            "max_residual": eval_stats.get("max_residual", 0.0),
            "q_value": q_ratio,
            "num_residuals": (
                len(prepared["calib_indices"]) * np.sum(prepared["fluid_mask"])
                if prepared["fluid_mask"] is not None
                else len(prepared["calib_indices"]) * prepared["num_nodes"]
            ),
        },
        "mean_width": eval_stats["mean_width"],
        "mean_area": eval_stats["mean_area"],
        "q_ratio": q_ratio,
        "num_calib_steps": len(prepared["calib_indices"]),
        "num_test_steps": len(prepared["test_indices"]),
        "num_nodes": prepared["num_nodes"],
        "use_fluid_mask": use_fluid_mask,  # Add missing field
        "model_name": prepared["data"].get("model_name", "unknown"),
        "delta_t": prepared["data"].get("delta_t", 0.01),
        "nonconformity_method": "boosted_l2",
        "coverage_gap": abs(eval_stats["empirical_coverage"] - (1 - alpha)),
        "valid_coverage": abs(eval_stats["empirical_coverage"] - (1 - alpha)) <= 0.02,
        # Spatial visualization data (optional, extracted separately if needed)
        "visualization": None,  # Use analysis.adaptive_viz for visualization
    }


def run_boosted_alpha_sweep(
    predictions_file: str,
    alphas: List[float],
    calib_ratio: float = 0.5,
    use_fluid_mask: bool = True,
    random_seed: int = 42,
    predictor_kwargs: Optional[Dict] = None,
    aux_ratio: float = 0.0,
    parallel: bool = True,  # Enable parallel execution for efficiency
    max_workers: int = 4,
) -> List[Dict]:
    """
    OPTIMIZED: Run BCP analysis across multiple alphas efficiently with parallelization.

    Args:
        predictions_file: Path to predictions pickle file
        alphas: List of alpha values to evaluate
        calib_ratio: Calibration ratio
        use_fluid_mask: Whether to use fluid mask
        random_seed: Random seed for reproducibility
        predictor_kwargs: BCP hyperparameters
        parallel: Whether to run alphas in parallel (default: True)
        max_workers: Maximum parallel workers (default: 4)
    """
    if parallel and len(alphas) > 1:
        print(f"Running BCP analysis for {len(alphas)} alphas in PARALLEL...")

        # Parallel execution using ProcessPoolExecutor
        from concurrent.futures import ProcessPoolExecutor, as_completed

        results = [None] * len(alphas)  # Pre-allocate to maintain order

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all alpha tasks
            future_to_index = {
                executor.submit(
                    run_boosted_conformal_analysis,
                    predictions_file,
                    alpha,
                    calib_ratio,
                    use_fluid_mask,
                    random_seed,
                    predictor_kwargs,
                    aux_ratio,
                ): i
                for i, alpha in enumerate(alphas)
            }

            # Collect results as they complete
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                    print(f"  Success: Completed alpha = {alphas[index]:.3f}")
                except Exception as e:
                    print(f"  Error: Failed alpha = {alphas[index]:.3f}: {e}")
                    # Create dummy result to maintain structure
                    results[index] = {
                        "alpha": alphas[index],
                        "target_coverage": 1 - alphas[index],
                        "empirical_coverage": float("nan"),
                        "error": str(e),
                    }

        # Filter out failed results
        results = [r for r in results if r is not None and "error" not in r]

    else:
        # Sequential execution (fallback or single alpha)
        print(f"Running BCP analysis for {len(alphas)} alphas SEQUENTIALLY...")
        results: List[Dict] = []
        for a in alphas:
            print(f"  Running BoostedCP analysis for alpha = {a:.3f}...")
            results.append(
                run_boosted_conformal_analysis(
                    predictions_file,
                    a,
                    calib_ratio,
                    use_fluid_mask,
                    random_seed,
                    predictor_kwargs,
                    aux_ratio,
                )
            )

    return results
