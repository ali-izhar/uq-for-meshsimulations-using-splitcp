"""
Conformal Prediction for Mesh-Based Simulations

This module implements the uncertainty quantification framework described in:
"Uncertainty Quantification Using Conformal Prediction for Mesh-Based Simulations"
by Mabtoul, Ali, and Ho (2025).

The framework transforms deterministic GNN outputs into statistically rigorous
prediction sets with finite-sample coverage guarantees. Supports four prediction-set
geometries on unstructured meshes:
    - ℓ2 disks
    - Joint ℓ∞ boxes
    - Mahalanobis ellipses
    - Spatially adaptive scaling

Key properties:
    - Post-hoc (no model retraining required)
    - Distribution-free with finite-sample guarantees
    - Works with any exchangeable data under split conformal prediction

Reference:
    Paper: See paper.tex in repository root for full mathematical details.
    Datasets: CylinderFlow (2D) and Flag (3D) from Pfaff et al. (2021).
"""

from .conformity import *
from .predictor import *
from .features import *
from .adaptive import *

__all__ = [
    "conformity",
    "predictor",
    "features",
    "adaptive",
]
