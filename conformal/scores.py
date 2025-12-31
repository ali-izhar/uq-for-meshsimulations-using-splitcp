#!/usr/bin/env python3
"""Conformal prediction score computation."""

from __future__ import annotations

import numpy as np


def residuals(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    r = np.asarray(gt) - np.asarray(pred)
    if r.ndim != 3:
        raise ValueError(f"Expected residuals shape (T,N,D), got {r.shape}")
    return r


def l2_scores(res: np.ndarray) -> np.ndarray:
    return np.linalg.norm(res, axis=-1)  # (T,N)


def linf_scores(res: np.ndarray) -> np.ndarray:
    return np.max(np.abs(res), axis=-1)  # (T,N)


def mahalanobis_scores(res_flat: np.ndarray, sigma_inv: np.ndarray) -> np.ndarray:
    """Compute Mahalanobis distances. Inputs: res_flat (M,D), sigma_inv (D,D). Returns (M,)."""
    r = np.asarray(res_flat)
    if r.ndim != 2:
        raise ValueError(f"res_flat must be (M,D), got {r.shape}")
    Sinv = np.asarray(sigma_inv)
    if Sinv.shape != (r.shape[1], r.shape[1]):
        raise ValueError(
            f"sigma_inv shape {Sinv.shape} incompatible with D={r.shape[1]}"
        )
    d2 = np.einsum("bi,ij,bj->b", r, Sinv, r)
    d2 = np.maximum(d2, 0.0)
    return np.sqrt(d2)
