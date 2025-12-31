#!/usr/bin/env python3
"""Rollout `.pkl` helpers (load + a few small utilities)."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class RolloutKeys:
    pred: str
    gt: str
    mesh_pos: str = "mesh_pos"
    faces: str = "faces"


def load_rollouts(path: Union[str, Path]) -> List[Dict[str, Any]]:
    with open(Path(path), "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, list) or (len(obj) > 0 and not isinstance(obj[0], dict)):
        raise TypeError(f"Unexpected rollout type: {type(obj)}")
    return obj


def infer_keys(traj: Dict[str, Any]) -> RolloutKeys:
    if "pred_velocity" in traj and "gt_velocity" in traj:
        return RolloutKeys(pred="pred_velocity", gt="gt_velocity")
    if "pred_pos" in traj and "gt_pos" in traj:
        return RolloutKeys(pred="pred_pos", gt="gt_pos")
    raise KeyError(f"Cannot infer pred/gt keys from: {sorted(traj.keys())}")


def mesh_xy_and_tris(traj: Dict[str, Any], t: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    keys = infer_keys(traj)
    mesh_pos = np.asarray(traj[keys.mesh_pos])
    faces = np.asarray(traj[keys.faces])
    if mesh_pos.ndim != 3:
        raise ValueError(f"mesh_pos must be (T,N,2), got {mesh_pos.shape}")
    if faces.ndim != 3:
        raise ValueError(f"faces must be (T,F,3), got {faces.shape}")
    t = int(np.clip(t, 0, mesh_pos.shape[0] - 1))
    xy = mesh_pos[t]
    tris = faces[t].astype(np.int32)
    return xy, tris


def boundary_node_mask(tris: np.ndarray, n_nodes: int) -> np.ndarray:
    """Boundary nodes from triangle connectivity (edges that appear once)."""
    from collections import defaultdict

    edge_count = defaultdict(int)
    for a, b, c in np.asarray(tris, dtype=np.int64):
        for u, v in ((a, b), (b, c), (c, a)):
            if u > v:
                u, v = v, u
            edge_count[(u, v)] += 1
    mask = np.zeros((int(n_nodes),), dtype=bool)
    for (u, v), cnt in edge_count.items():
        if cnt == 1:
            mask[int(u)] = True
            mask[int(v)] = True
    return mask


def get_pred_gt(traj: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, RolloutKeys]:
    keys = infer_keys(traj)
    pred = np.asarray(traj[keys.pred])
    gt = np.asarray(traj[keys.gt])
    if pred.shape != gt.shape:
        raise ValueError(f"pred shape {pred.shape} != gt {gt.shape}")
    if pred.ndim != 3:
        raise ValueError(f"pred/gt must be (T,N,D), got {pred.shape}")
    return pred, gt, keys


def residuals(traj: Dict[str, Any]) -> np.ndarray:
    pred, gt, _ = get_pred_gt(traj)
    return gt - pred


def l2_error(traj: Dict[str, Any]) -> np.ndarray:
    """Per-node error magnitude: (T,N)."""
    r = residuals(traj)
    return np.linalg.norm(r, axis=-1)


def component(traj: Dict[str, Any], *, which: str, idx: int = 0) -> np.ndarray:
    """
    Extract a scalar field from pred/gt/residual.
    which in {"pred","gt","err"}.
    returns (T,N)
    """
    pred, gt, _ = get_pred_gt(traj)
    if idx < 0 or idx >= pred.shape[-1]:
        raise ValueError(f"idx out of range for D={pred.shape[-1]}")
    if which == "pred":
        return pred[..., idx]
    if which == "gt":
        return gt[..., idx]
    if which == "err":
        return (gt - pred)[..., idx]
    raise ValueError(which)


def summarize_over_nodes(
    x_tn: np.ndarray, qs=(0.25, 0.5, 0.75)
) -> Dict[str, np.ndarray]:
    """
    Summaries over nodes for each timestep.
    x_tn: (T,N)
    returns dict of arrays length T.
    """
    x = np.asarray(x_tn)
    if x.ndim != 2:
        raise ValueError(f"Expected (T,N), got {x.shape}")
    out = {
        "mean": np.mean(x, axis=1),
        "std": np.std(x, axis=1),
    }
    for q in qs:
        out[f"q{int(100*q):02d}"] = np.quantile(x, q, axis=1)
    return out
