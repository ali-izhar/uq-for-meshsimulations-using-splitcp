#!/usr/bin/env python3
"""Feature construction utilities for conformal prediction."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional, Tuple


def build_default_features(
    mesh_pos: np.ndarray, pred_vec: np.ndarray, t_index: np.ndarray
) -> np.ndarray:
    """Default σ(x) features: [mesh_x, mesh_y, pred..., ||pred||2, t_norm]. Returns X with shape (T*N, F)."""
    return build_features(
        mesh_pos=mesh_pos,
        pred_vec=pred_vec,
        t_index=t_index,
        faces=None,
        feature_set="default",
    )


def default_time_index(T: int) -> np.ndarray:
    if T <= 1:
        return np.zeros((T,), dtype=np.float32)
    return (np.arange(T, dtype=np.float32) / float(T - 1)).astype(np.float32)


FeatureSet = Literal["default", "physics_basic", "physics_full"]


@dataclass(frozen=True)
class FeatureMeta:
    feature_set: str


def _validate_shapes(
    mesh_pos: np.ndarray, pred_vec: np.ndarray, t_index: np.ndarray
) -> Tuple[int, int, int]:
    if mesh_pos.ndim != 3 or mesh_pos.shape[-1] != 2:
        raise ValueError(f"mesh_pos must be (T,N,2), got {mesh_pos.shape}")
    if pred_vec.ndim != 3:
        raise ValueError(f"pred_vec must be (T,N,D), got {pred_vec.shape}")
    if mesh_pos.shape[:2] != pred_vec.shape[:2]:
        raise ValueError(
            f"mesh_pos (T,N)={mesh_pos.shape[:2]} != pred_vec (T,N)={pred_vec.shape[:2]}"
        )
    T, N, D = pred_vec.shape
    t = np.asarray(t_index).reshape(-1)
    if t.size != T:
        raise ValueError(f"t_index must have length T={T}, got {t.size}")
    return int(T), int(N), int(D)


def _faces_to_edges_and_boundary(
    faces: np.ndarray, num_nodes: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (edges[E,2], boundary_mask[N]) from triangular faces[F,3]."""
    f = np.asarray(faces)
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"faces must be (F,3), got {f.shape}")
    # all directed edges from faces, then canonicalize to undirected (min,max)
    e01 = f[:, [0, 1]]
    e12 = f[:, [1, 2]]
    e20 = f[:, [2, 0]]
    edges = np.concatenate([e01, e12, e20], axis=0).astype(np.int64, copy=False)
    edges = np.sort(edges, axis=1)
    # unique edges + counts => boundary edges appear once
    edges_u, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = edges_u[counts == 1]
    boundary_mask = np.zeros((int(num_nodes),), dtype=bool)
    if boundary_edges.size > 0:
        boundary_mask[boundary_edges.reshape(-1)] = True
    return edges_u.astype(np.int32), boundary_mask


def _mesh_degree_norm(faces: np.ndarray, num_nodes: int) -> np.ndarray:
    deg = np.bincount(
        np.asarray(faces, dtype=np.int64).reshape(-1), minlength=int(num_nodes)
    ).astype(np.float32)
    m = float(np.max(deg)) if deg.size else 1.0
    if m <= 0:
        m = 1.0
    return deg / m


def _mesh_quality_static(
    pos0: np.ndarray, faces: np.ndarray, num_nodes: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-node (log aspect ratio, min angle / 60deg) using t=0 positions as static mesh quality."""
    p = np.asarray(pos0, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError(f"pos0 must be (N,2), got {p.shape}")
    # gather triangle vertices
    a = p[f[:, 0]]
    b = p[f[:, 1]]
    c = p[f[:, 2]]
    ab = b - a
    bc = c - b
    ca = a - c
    lab = np.linalg.norm(ab, axis=1)
    lbc = np.linalg.norm(bc, axis=1)
    lca = np.linalg.norm(ca, axis=1)
    per = lab + lbc + lca
    # area = 0.5*|cross(ab, ac)|
    ac = c - a
    area2 = np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0])
    area = 0.5 * area2
    # avoid degenerates
    valid = (area > 1e-14) & (lab > 1e-14) & (lbc > 1e-14) & (lca > 1e-14)
    aspect = np.ones((f.shape[0],), dtype=np.float64)
    aspect[valid] = (per[valid] ** 2) / (12.0 * np.sqrt(3.0) * area[valid])
    aspect_log = np.log(np.maximum(aspect, 1.0 + 1e-6)).astype(np.float32)

    # angles via cosine law
    def ang(opp, adj1, adj2):
        cosv = (adj1**2 + adj2**2 - opp**2) / (2.0 * adj1 * adj2 + 1e-30)
        return np.arccos(np.clip(cosv, -1.0, 1.0))

    angA = ang(lbc, lab, lca)
    angB = ang(lca, lab, lbc)
    angC = ang(lab, lbc, lca)
    min_ang = np.minimum(angA, np.minimum(angB, angC))
    min_ang_norm = (min_ang / (np.pi / 3.0)).astype(np.float32)  # /60deg

    # accumulate to nodes
    ar_node = np.zeros((int(num_nodes),), dtype=np.float32)
    ma_node = np.zeros((int(num_nodes),), dtype=np.float32)
    cnt = np.zeros((int(num_nodes),), dtype=np.float32)
    for k in range(3):
        idx = f[:, k]
        np.add.at(ar_node, idx, aspect_log)
        np.add.at(ma_node, idx, min_ang_norm)
        np.add.at(cnt, idx, 1.0)
    cnt = np.maximum(cnt, 1.0)
    ar_node /= cnt
    ma_node /= cnt
    return ar_node, ma_node


def _boundary_distance_per_t(
    mesh_pos: np.ndarray, boundary_mask: np.ndarray
) -> np.ndarray:
    """Compute per-timestep distance to nearest boundary node using broadcasting (no SciPy). Returns (T,N)."""
    mp = np.asarray(mesh_pos, dtype=np.float32)
    T, N, _ = mp.shape
    b_idx = np.flatnonzero(boundary_mask)
    if b_idx.size == 0:
        return np.full((T, N), 0.0, dtype=np.float32)
    # cap boundary nodes for speed if extremely large (shouldn't happen here)
    bpos = mp[:, b_idx, :]  # (T,B,2)
    # squared distances (T,N,B) -> min over B
    # Use chunking over B to control memory.
    B = int(bpos.shape[1])
    chunk = 256
    best = np.full((T, N), np.inf, dtype=np.float32)
    for s in range(0, B, chunk):
        e = min(B, s + chunk)
        diff = mp[:, :, None, :] - bpos[:, None, s:e, :]  # (T,N,chunk,2)
        d2 = np.sum(diff * diff, axis=-1)  # (T,N,chunk)
        best = np.minimum(best, np.min(d2, axis=-1))
    return np.sqrt(np.maximum(best, 0.0))


def _edge_grad_log(
    pred_vec: np.ndarray, mesh_pos: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    """Approx gradient magnitude via neighbor differences along mesh edges. Returns log(grad+eps) as (T,N)."""
    pred = np.asarray(pred_vec, dtype=np.float32)
    pos = np.asarray(mesh_pos, dtype=np.float32)
    T, N, D = pred.shape
    e = np.asarray(edges, dtype=np.int64)
    i = e[:, 0]
    j = e[:, 1]
    # (T,E,D)
    dv = pred[:, i, :] - pred[:, j, :]
    dp = pos[:, i, :] - pos[:, j, :]
    dist = np.linalg.norm(dp, axis=-1)  # (T,E)
    mag = np.linalg.norm(dv, axis=-1) / np.maximum(dist, 1e-6)  # (T,E)
    # scatter mean to nodes
    out = np.zeros((T, N), dtype=np.float32)
    cnt = np.zeros((N,), dtype=np.float32)
    cnt_i = np.bincount(i, minlength=N).astype(np.float32)
    cnt_j = np.bincount(j, minlength=N).astype(np.float32)
    cnt = np.maximum(cnt_i + cnt_j, 1.0)
    for t in range(T):
        tmp = np.zeros((N,), dtype=np.float32)
        np.add.at(tmp, i, mag[t])
        np.add.at(tmp, j, mag[t])
        out[t] = tmp / cnt
    return np.log(np.maximum(out, 1e-8))


def build_features(
    *,
    mesh_pos: np.ndarray,
    pred_vec: np.ndarray,
    t_index: np.ndarray,
    faces: Optional[np.ndarray],
    feature_set: FeatureSet = "default",
) -> np.ndarray:
    """
    Feature builder for σ(x).

    - default: [x,y, pred..., ||pred||2, t_norm]
    - physics_basic/full: add mesh/topology-derived features computed from `faces`.

    Notes:
    - Uses only rollout-available quantities: mesh_pos (T,N,2), faces (F,3), pred_vec (T,N,D).
    - No SciPy dependency; nearest-boundary distance is computed via numpy broadcasting.
    """
    feature_set = str(feature_set).lower()
    T, N, D = _validate_shapes(mesh_pos, pred_vec, t_index)

    t = np.asarray(t_index).reshape(-1).astype(np.float32)
    t = np.clip(t, 0.0, 1.0)
    t_feat = np.repeat(t[:, None], N, axis=1)[:, :, None]  # (T,N,1)
    pred_norm = np.linalg.norm(
        np.asarray(pred_vec, dtype=np.float32), axis=-1, keepdims=True
    )  # (T,N,1)

    base = [
        np.asarray(mesh_pos, dtype=np.float32),
        np.asarray(pred_vec, dtype=np.float32),
        pred_norm,
        t_feat,
    ]

    if feature_set == "default" or faces is None:
        X = np.concatenate(base, axis=-1)
        return X.reshape(T * N, -1)

    # canonicalize faces (F,3)
    f = np.asarray(faces)
    if f.ndim == 3:
        f = f[0]
    edges, boundary_mask = _faces_to_edges_and_boundary(f, num_nodes=N)
    degree = _mesh_degree_norm(f, num_nodes=N)  # (N,)
    aspect_log, min_ang = _mesh_quality_static(
        mesh_pos[0], f, num_nodes=N
    )  # (N,), (N,)
    d_bnd = _boundary_distance_per_t(mesh_pos, boundary_mask)  # (T,N)
    d_bnd_log = np.log(np.maximum(d_bnd, 1e-6))
    bnd_ind = boundary_mask.astype(np.float32)[None, :]
    bnd_ind = np.repeat(bnd_ind, T, axis=0)  # (T,N)
    grad_pred = _edge_grad_log(pred_vec, mesh_pos, edges)  # (T,N)
    re_local = np.log1p((pred_norm[..., 0] * np.maximum(d_bnd, 1e-6)) / 1e-6)  # (T,N)

    extra = [
        d_bnd[..., None],
        d_bnd_log[..., None],
        np.repeat(degree[None, :], T, axis=0)[..., None],
        np.repeat(aspect_log[None, :], T, axis=0)[..., None],
        np.repeat(min_ang[None, :], T, axis=0)[..., None],
        bnd_ind[..., None],
        grad_pred[..., None],
        re_local[..., None],
    ]

    if feature_set == "physics_basic":
        # Keep a small, robust subset (distance-to-boundary + degree + boundary flag)
        keep = [extra[0], extra[1], extra[2], extra[5]]
        X = np.concatenate(base + keep, axis=-1)
        return X.reshape(T * N, -1)

    if feature_set == "physics_full":
        X = np.concatenate(base + extra, axis=-1)
        return X.reshape(T * N, -1)

    raise ValueError(f"Unknown feature_set: {feature_set}")
