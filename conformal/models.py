#!/usr/bin/env python3
"""Conformal prediction model fitting utilities."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


def fit_covariance(
    residuals_flat: np.ndarray, eps: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit covariance Σ from residuals (M,D) with ridge eps*I. Returns (Σ, Σ^{-1})."""
    r = np.asarray(residuals_flat, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError(f"residuals_flat must be (M,D), got {r.shape}")
    if r.shape[0] < 2:
        raise ValueError("Need at least 2 samples to estimate covariance")
    if eps <= 0:
        raise ValueError("eps must be > 0")

    # Center residuals to remove bias
    rc = r - np.mean(r, axis=0, keepdims=True)
    Sigma = np.cov(rc, rowvar=False, bias=False)
    Sigma = np.asarray(Sigma, dtype=np.float64)
    D = Sigma.shape[0]
    Sigma = Sigma + float(eps) * np.eye(D, dtype=np.float64)
    Sigma_inv = np.linalg.inv(Sigma)
    return Sigma, Sigma_inv


class SigmaModel:
    def __init__(self, Sigma: np.ndarray, Sigma_inv: np.ndarray, eps: float):
        self.Sigma = np.asarray(Sigma, dtype=np.float64)
        self.Sigma_inv = np.asarray(Sigma_inv, dtype=np.float64)
        self.eps = float(eps)

    def save(self, outdir: Union[str, Path]) -> None:
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "Sigma.npy", self.Sigma)
        np.save(out / "Sigma_inv.npy", self.Sigma_inv)
        (out / "Sigma_meta.json").write_text(json.dumps({"eps": self.eps}, indent=2))

    @staticmethod
    def load(indir: Union[str, Path]) -> "SigmaModel":
        p = Path(indir)
        Sigma = np.load(p / "Sigma.npy")
        Sigma_inv = np.load(p / "Sigma_inv.npy")
        meta = json.loads((p / "Sigma_meta.json").read_text())
        return SigmaModel(
            Sigma=Sigma, Sigma_inv=Sigma_inv, eps=float(meta.get("eps", 1e-6))
        )


class SigmaPredictor:
    """Interface for σ(x) models."""

    def predict_log_sigma(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_sigma(self, X: np.ndarray, sigma_min: float = 1e-6) -> np.ndarray:
        log_s = self.predict_log_sigma(X)
        s = np.exp(log_s)
        return np.maximum(s, float(sigma_min))

    def save(self, path: Union[str, Path]) -> None:
        with open(Path(path), "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Union[str, Path]) -> "SigmaPredictor":
        with open(Path(path), "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, SigmaPredictor):
            raise TypeError(f"Loaded object is not a SigmaPredictor: {type(obj)}")
        return obj


@dataclass
class CappedSigmaPredictor(SigmaPredictor):
    """
    Wrap a SigmaPredictor and cap σ(x) at sigma_cap (after exponentiation).

    This is useful when a small fraction of samples have extremely large predicted
    σ(x), which can dominate mean volume/area metrics (because size scales as σ(x)^D).
    The cap is applied consistently during calibration and evaluation, so thresholds
    re-adjust to preserve coverage as much as possible.
    """

    base: SigmaPredictor
    sigma_cap: float

    def predict_log_sigma(self, X: np.ndarray) -> np.ndarray:
        log_s = self.base.predict_log_sigma(X)
        s = np.exp(log_s)
        cap = float(self.sigma_cap)
        s2 = np.minimum(s, cap)
        return np.log(np.maximum(s2, 1e-300))

    def predict_sigma(self, X: np.ndarray, sigma_min: float = 1e-6) -> np.ndarray:
        s = self.base.predict_sigma(X, sigma_min=sigma_min)
        return np.minimum(s, float(self.sigma_cap))


class ComponentwiseSigmaPredictor:
    """
    Componentwise adaptive: learns σ_i(x) for each output dimension i.

    Instead of scalar σ(x), this learns a vector σ(x) ∈ ℝ^d where each component
    has its own spatial scale function. This captures per-component heteroscedasticity
    without assuming correlations between components.

    Score: ||r ⊘ σ(x)||₂ = √(Σᵢ (rᵢ/σᵢ(x))²)

    This produces axis-aligned ellipsoids with spatially-varying semi-axes,
    which is more expressive than scalar adaptive (spheres) but simpler and
    more robust than full Σ(x) (general ellipsoids).
    """

    def __init__(self, component_models: list, d: int, sigma_caps: Optional[list] = None):
        """
        Args:
            component_models: List of d SigmaPredictor instances, one per component
            d: Output dimension (2 for CylinderFlow, 3 for Flag)
            sigma_caps: Optional list of d cap values for each component
        """
        self.component_models = component_models
        self.d = d
        self.sigma_caps = sigma_caps

    def predict_sigma_vec(self, X: np.ndarray, sigma_min: float = 1e-6) -> np.ndarray:
        """
        Predict per-component sigma values.

        Args:
            X: Features array (M, F)
            sigma_min: Minimum sigma value

        Returns:
            sigma_vec: Array of shape (M, d) with per-component scales
        """
        M = X.shape[0]
        sigma_vec = np.zeros((M, self.d), dtype=np.float64)
        for i, model in enumerate(self.component_models):
            sigma_vec[:, i] = model.predict_sigma(X, sigma_min=sigma_min)
            if self.sigma_caps is not None and self.sigma_caps[i] is not None:
                sigma_vec[:, i] = np.minimum(sigma_vec[:, i], self.sigma_caps[i])
        return sigma_vec

    def compute_scores(self, residuals: np.ndarray, X: np.ndarray, sigma_min: float = 1e-6) -> np.ndarray:
        """
        Compute componentwise adaptive scores: ||r ⊘ σ(x)||₂

        Args:
            residuals: Array of shape (M, d)
            X: Features array (M, F)
            sigma_min: Minimum sigma value

        Returns:
            scores: Array of shape (M,)
        """
        sigma_vec = self.predict_sigma_vec(X, sigma_min=sigma_min)
        normalized = residuals / sigma_vec  # (M, d)
        return np.linalg.norm(normalized, axis=-1)  # (M,)

    def compute_width(self, q: float, X: np.ndarray, sigma_min: float = 1e-6) -> np.ndarray:
        """
        Compute effective width for each sample: q * (∏ᵢ σᵢ(x))^{1/d}

        This is the geometric mean of the semi-axes, giving a comparable
        "effective radius" metric to other methods.

        Args:
            q: Calibrated quantile threshold
            X: Features array (M, F)
            sigma_min: Minimum sigma value

        Returns:
            widths: Array of shape (M,)
        """
        sigma_vec = self.predict_sigma_vec(X, sigma_min=sigma_min)
        # Geometric mean of component sigmas
        geom_mean = np.exp(np.mean(np.log(sigma_vec), axis=-1))
        return q * geom_mean

    def save(self, path: Union[str, Path]) -> None:
        with open(Path(path), "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Union[str, Path]) -> "ComponentwiseSigmaPredictor":
        with open(Path(path), "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, ComponentwiseSigmaPredictor):
            raise TypeError(f"Loaded object is not a ComponentwiseSigmaPredictor: {type(obj)}")
        return obj


def fit_componentwise_sigma_model(
    X: np.ndarray,
    residuals: np.ndarray,
    method: str = "xgboost",
    lam: float = 1e-3,
    y_eps: float = 1e-8,
    seed: int = 0,
    sigma_cap_quantile: Optional[float] = None,
) -> ComponentwiseSigmaPredictor:
    """
    Fit componentwise σ(x) models: one model per output dimension.

    Args:
        X: Features array (M, F)
        residuals: Residuals array (M, d)
        method: Model type ('xgboost', 'sklearn_hgb', etc.)
        lam: Regularization for ridge
        y_eps: Small constant added before log
        seed: Random seed
        sigma_cap_quantile: Optional quantile for capping (e.g., 0.98)

    Returns:
        ComponentwiseSigmaPredictor with d fitted models
    """
    X = np.asarray(X, dtype=np.float64)
    res = np.asarray(residuals, dtype=np.float64)
    if res.ndim != 2:
        raise ValueError(f"residuals must be (M, d), got {res.shape}")
    M, d = res.shape
    if X.shape[0] != M:
        raise ValueError(f"X rows {X.shape[0]} != residuals rows {M}")

    component_models = []
    sigma_caps = []

    for i in range(d):
        # Fit model for component i using |r_i| as target
        abs_ri = np.abs(res[:, i])
        model = fit_sigma_model(
            X=X,
            y_l2=abs_ri,  # Note: using |r_i|, not ||r||
            method=method,
            lam=lam,
            y_eps=y_eps,
            seed=seed + i,  # Different seed per component
        )
        component_models.append(model)

        # Compute cap if requested
        if sigma_cap_quantile is not None:
            sigma_pred = model.predict_sigma(X, sigma_min=1e-6)
            cap = float(np.quantile(sigma_pred, sigma_cap_quantile))
            sigma_caps.append(cap)
        else:
            sigma_caps.append(None)

    return ComponentwiseSigmaPredictor(
        component_models=component_models,
        d=d,
        sigma_caps=sigma_caps if sigma_cap_quantile is not None else None,
    )


@dataclass
class RidgeLogSigma(SigmaPredictor):
    """Ridge regression on log(||r||2) with closed-form solution."""

    w: np.ndarray  # (F,)
    b: float
    lam: float

    def predict_log_sigma(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return (X @ self.w + self.b).astype(np.float64)


@dataclass
class SklearnLogSigma(SigmaPredictor):
    """Wrap an sklearn regressor that predicts log sigma."""

    model: Any

    def predict_log_sigma(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return np.asarray(self.model.predict(X), dtype=np.float64)


def fit_sigma_model(
    X: np.ndarray,
    y_l2: np.ndarray,
    method: str = "auto",
    lam: float = 1e-3,
    y_eps: float = 1e-8,
    seed: int = 0,
) -> SigmaPredictor:
    """Fit σ(x) on aux: regress log(||r||2 + y_eps) from features X."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y_l2, dtype=np.float64).reshape(-1)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X rows {X.shape[0]} != y {y.shape[0]}")
    if X.shape[0] < 10:
        raise ValueError("Need at least 10 samples to fit sigma model")
    if y_eps <= 0:
        raise ValueError("y_eps must be > 0")

    y_log = np.log(y + float(y_eps))

    method = method.lower()
    if method not in (
        "auto",
        "ridge",
        "sklearn_hgb",
        "sklearn_hgb_quantile",
        "xgboost",
    ):
        raise ValueError(f"Unknown sigma model method: {method}")

    # -------- Gradient-boosted trees on log residual scale
    # Fit log(||r||) with a robust median/quantile loss.
    # We implement this via sklearn (Hist)GradientBoosting.
    if method in ("auto", "sklearn_hgb", "sklearn_hgb_quantile"):
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor

            # Note: quantile loss support depends on sklearn version; we fall back to
            # GradientBoostingRegressor if needed.

            if method == "sklearn_hgb_quantile":
                try:
                    model = HistGradientBoostingRegressor(
                        loss="quantile",
                        quantile=0.5,
                        max_depth=6,
                        learning_rate=0.08,
                        max_iter=350,
                        random_state=int(seed),
                    )
                except TypeError:
                    # Older sklearn: use classic GradientBoostingRegressor quantile loss
                    from sklearn.ensemble import GradientBoostingRegressor

                    model = GradientBoostingRegressor(
                        loss="quantile",
                        alpha=0.5,
                        learning_rate=0.05,
                        n_estimators=600,
                        max_depth=3,
                        random_state=int(seed),
                    )
            else:
                model = HistGradientBoostingRegressor(
                    max_depth=6,
                    learning_rate=0.08,
                    max_iter=250,
                    random_state=int(seed),
                )

            model.fit(X.astype(np.float32), y_log.astype(np.float32))
            return SklearnLogSigma(model=model)
        except Exception:
            if method in ("sklearn_hgb", "sklearn_hgb_quantile"):
                raise
            # fall through to ridge

    if method == "xgboost":
        # Optional: allow using xgboost if installed (not a hard dependency).
        try:
            import xgboost as xgb  # type: ignore
        except Exception as e:
            raise ImportError(
                "sigma_model=xgboost requested, but xgboost is not installed. "
                "Install it (e.g., `uv pip install xgboost`) or use sklearn_hgb_quantile."
            ) from e

        # Paper-style robust scale learning: median (τ=0.5) regression on log(||r||).
        # In xgboost>=2 this is available as reg:quantileerror with quantile_alpha.
        try:
            model = xgb.XGBRegressor(
                n_estimators=900,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                objective="reg:quantileerror",
                quantile_alpha=0.5,
                random_state=int(seed),
                n_jobs=1,  # single-threaded for bit-reproducibility across machines
            )
        except TypeError:
            # Fallback for older xgboost: pseudo-Huber is a reasonable robust surrogate.
            model = xgb.XGBRegressor(
                n_estimators=900,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                objective="reg:pseudohubererror",
                random_state=int(seed),
                n_jobs=1,  # single-threaded for bit-reproducibility across machines
            )
        model.fit(X.astype(np.float32), y_log.astype(np.float32))
        return SklearnLogSigma(model=model)

    # Ridge (closed-form with bias term)
    Xc = X - np.mean(X, axis=0, keepdims=True)
    yc = y_log - float(np.mean(y_log))
    F = X.shape[1]
    A = Xc.T @ Xc + float(lam) * np.eye(F)
    w = np.linalg.solve(A, Xc.T @ yc)
    b = float(np.mean(y_log) - np.mean(X, axis=0) @ w)
    return RidgeLogSigma(w=w, b=b, lam=float(lam))
