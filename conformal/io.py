#!/usr/bin/env python3
"""Rollout `.pkl` helpers (load + a few small utilities)."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union


RolloutTraj = Dict[str, Any]


def load_rollouts(path: Union[str, Path]) -> List[RolloutTraj]:
    """Load a MeshGraphNet rollout .pkl (expects `list[dict]`)."""
    p = Path(path)
    with open(p, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected list in rollout pkl, got {type(data)} from {p}")
    return data


def infer_pred_gt_keys(traj: RolloutTraj) -> Tuple[str, str]:
    """Infer (pred_key, gt_key) from known MeshGraphNet rollout schemas."""
    candidates = [
        ("pred_velocity", "gt_velocity"),
        ("pred_pos", "gt_pos"),
        ("pred_world_pos", "gt_world_pos"),
    ]
    keys = set(traj.keys())
    for pk, gk in candidates:
        if pk in keys and gk in keys:
            return pk, gk
    raise KeyError(f"Could not infer pred/gt keys. Available keys: {sorted(keys)}")


def get_mesh_pos(traj: RolloutTraj, key: str = "mesh_pos") -> Any:
    if key not in traj:
        raise KeyError(
            f"Missing '{key}' in rollout trajectory. Keys: {sorted(traj.keys())}"
        )
    return traj[key]
