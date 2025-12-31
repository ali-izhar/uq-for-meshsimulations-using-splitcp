#!/usr/bin/env python3
"""Conformal quantile computation."""

from __future__ import annotations

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Split conformal quantile using the finite-sample rank:
      q = k-th smallest where k = ceil((n+1)*(1-alpha))
    """
    s = np.asarray(scores).reshape(-1)
    if s.size == 0:
        raise ValueError("scores is empty")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1), got {alpha}")

    n = int(s.size)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = max(1, min(k, n))
    # k-th smallest => index k-1
    return float(np.partition(s, k - 1)[k - 1])
