#!/usr/bin/env python3
"""Conformal prediction pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from conformal.features import build_features, default_time_index
from conformal.io import infer_pred_gt_keys, load_rollouts, get_mesh_pos
from conformal.models import SigmaModel, SigmaPredictor, fit_covariance, fit_sigma_model
from conformal.quantiles import conformal_quantile
from conformal.scores import mahalanobis_scores


def flatten_rollouts(
    rollout_pkl: Union[str, Path],
    pred_gt_keys: Optional[Tuple[str, str]] = None,
    max_samples: Optional[int] = None,
    seed: int = 0,
    return_ids: bool = False,
    feature_set: str = "default",
) -> Dict[str, np.ndarray]:
    """Load rollout .pkl and flatten to per-node-per-timestep samples: pred/gt/res (M,D), scores (M,), features X (M,F)."""
    rollouts = load_rollouts(rollout_pkl)
    pred_list = []
    gt_list = []
    X_list = []
    traj_id_list = []
    t_idx_list = []
    node_idx_list = []

    for traj_i, traj in enumerate(rollouts):
        pk, gk = pred_gt_keys or infer_pred_gt_keys(traj)
        pred = np.asarray(traj[pk])
        gt = np.asarray(traj[gk])
        mesh_pos = np.asarray(get_mesh_pos(traj))
        faces = traj.get("faces", None)
        if pred.shape != gt.shape:
            raise ValueError(
                f"pred shape {pred.shape} != gt shape {gt.shape} in {rollout_pkl}"
            )
        if mesh_pos.shape[:2] != pred.shape[:2]:
            raise ValueError(
                f"mesh_pos shape {mesh_pos.shape} incompatible with pred {pred.shape} in {rollout_pkl}"
            )

        T = pred.shape[0]
        t = default_time_index(T)
        X = build_features(
            mesh_pos=mesh_pos,
            pred_vec=pred,
            t_index=t,
            faces=faces,
            feature_set=feature_set,
        )

        pred_list.append(pred.reshape(-1, pred.shape[-1]))
        gt_list.append(gt.reshape(-1, gt.shape[-1]))
        X_list.append(X)
        if return_ids:
            # Each sample corresponds to (timestep, node). We keep:
            # - traj_id: which rollout trajectory it came from
            # - t_idx: integer timestep index within the rollout window
            N = int(pred.shape[1])
            traj_id_list.append(np.full(T * N, traj_i, dtype=np.int32))
            t_idx_list.append(np.repeat(np.arange(T, dtype=np.int32), N))
            node_idx_list.append(np.tile(np.arange(N, dtype=np.int32), T))

    pred_all = np.concatenate(pred_list, axis=0)
    gt_all = np.concatenate(gt_list, axis=0)
    X_all = np.concatenate(X_list, axis=0)
    if return_ids:
        traj_id_all = np.concatenate(traj_id_list, axis=0)
        t_idx_all = np.concatenate(t_idx_list, axis=0)
        node_idx_all = np.concatenate(node_idx_list, axis=0)

    res_all = gt_all - pred_all
    l2_all = np.linalg.norm(res_all, axis=-1)
    linf_all = np.max(np.abs(res_all), axis=-1)

    if max_samples is not None and max_samples > 0 and pred_all.shape[0] > max_samples:
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(pred_all.shape[0], size=int(max_samples), replace=False)
        pred_all = pred_all[idx]
        gt_all = gt_all[idx]
        X_all = X_all[idx]
        res_all = res_all[idx]
        l2_all = l2_all[idx]
        linf_all = linf_all[idx]
        if return_ids:
            traj_id_all = traj_id_all[idx]
            t_idx_all = t_idx_all[idx]
            node_idx_all = node_idx_all[idx]

    out = {
        "pred": pred_all,
        "gt": gt_all,
        "res": res_all,
        "l2": l2_all,
        "linf": linf_all,
        "X": X_all,
    }
    if return_ids:
        out["traj_id"] = traj_id_all
        out["t_idx"] = t_idx_all
        out["node_idx"] = node_idx_all
    return out


def _group_max(scores: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Compute max(score) within each group defined by integer keys."""
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    k = np.asarray(keys).reshape(-1)
    if s.shape[0] != k.shape[0]:
        raise ValueError(f"scores {s.shape} and keys {k.shape} must have same length")
    if s.size == 0:
        raise ValueError("scores empty")
    order = np.argsort(k, kind="mergesort")
    k2 = k[order]
    s2 = s[order]
    start = np.flatnonzero(np.r_[True, k2[1:] != k2[:-1]])
    return np.maximum.reduceat(s2, start)


@dataclass
class ConformalArtifacts:
    sigma: SigmaModel
    sigma_model_path: Path
    thresholds: Dict[str, Dict[str, float]]  # method -> alpha_str -> q


def run_conformal_pipeline(
    aux_pkl: Union[str, Path],
    cal_pkl: Union[str, Path],
    outdir: Union[str, Path],
    alphas: List[float],
    eval_pkl: Optional[Union[str, Path]] = None,
    test_pkl: Optional[Union[str, Path]] = None,
    sigma_eps: float = 1e-6,
    sigma_model: str = "auto",
    sigma_model_lam: float = 1e-3,
    sigma_min: float = 1e-6,
    sigma_cap_quantile: Optional[float] = None,
    cal_grouping: str = "none",
    timeblock_size: int = 10,
    nodeblock_size: int = 64,
    feature_set: str = "default",
    reuse_aux_from: Optional[Union[str, Path]] = None,
    max_aux_samples_cov: Optional[int] = None,
    max_aux_samples_sigma: Optional[int] = None,
    max_cal_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    seed: int = 0,
) -> Dict:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    # Backwards-compatible naming: historically this code used "test_pkl" for the
    # held-out evaluation rollouts. We now prefer "eval_pkl".
    if eval_pkl is None and test_pkl is None:
        raise ValueError("Must provide eval_pkl (preferred) or test_pkl (deprecated).")
    if eval_pkl is None:
        eval_pkl = test_pkl
    assert eval_pkl is not None
    if max_eval_samples is None:
        max_eval_samples = max_test_samples

    # ---------- AUX: fit (or reuse) Sigma and sigma(x)
    aux_counts_cov = None
    aux_counts_sigma = None
    if reuse_aux_from:
        reuse_dir = Path(reuse_aux_from)
        sigma = SigmaModel.load(reuse_dir)
        sigma_pred = SigmaPredictor.load(reuse_dir / "sigma_model.pkl")
        # best-effort: read counts from existing summary.json
        summ = reuse_dir / "summary.json"
        if summ.exists():
            try:
                sd = json.loads(summ.read_text())
                c = sd.get("counts", {})
                aux_counts_cov = c.get("aux_samples_cov", None)
                aux_counts_sigma = c.get("aux_samples_sigma", None)
            except Exception:
                pass
        # Save/copy into this outdir so it is self-contained
        sigma.save(out)
        sigma_model_path = out / "sigma_model.pkl"
        sigma_pred.save(sigma_model_path)
    else:
        aux = flatten_rollouts(
            aux_pkl, max_samples=max_aux_samples_cov, seed=seed, feature_set=feature_set
        )
        Sigma, Sigma_inv = fit_covariance(aux["res"], eps=sigma_eps)
        sigma = SigmaModel(Sigma=Sigma, Sigma_inv=Sigma_inv, eps=sigma_eps)
        sigma.save(out)
        aux_counts_cov = int(aux["res"].shape[0])

        aux_for_sigma = flatten_rollouts(
            aux_pkl,
            max_samples=max_aux_samples_sigma,
            seed=seed + 1,
            feature_set=feature_set,
        )
        aux_counts_sigma = int(aux_for_sigma["res"].shape[0])
        sigma_pred = fit_sigma_model(
            X=aux_for_sigma["X"],
            y_l2=aux_for_sigma["l2"],
            method=sigma_model,
            lam=sigma_model_lam,
            seed=seed,
        )

        # Optional robustness: cap extreme sigma values to avoid heavy tails dominating
        # volume/area metrics (size scales as (q*sigma)^D). The cap is learned from the
        # auxiliary sigma-fit sample distribution and applied consistently in cal+eval.
        if sigma_cap_quantile is not None:
            qcap = float(sigma_cap_quantile)
            if not (0.0 < qcap < 1.0):
                raise ValueError("sigma_cap_quantile must be in (0,1).")
            sigma_hat = sigma_pred.predict_sigma(
                aux_for_sigma["X"], sigma_min=sigma_min
            )
            cap = float(np.quantile(sigma_hat, qcap))
            from conformal.models import CappedSigmaPredictor

            sigma_pred = CappedSigmaPredictor(base=sigma_pred, sigma_cap=cap)
        sigma_model_path = out / "sigma_model.pkl"
        sigma_pred.save(sigma_model_path)

    # ---------- CAL: compute quantiles
    cal_grouping = str(cal_grouping).lower()
    if cal_grouping not in (
        "none",
        "traj_max",
        "timeblock_max",
        "timeblock_nodeblock_max",
    ):
        raise ValueError(f"Unknown cal_grouping: {cal_grouping}")
    cal = flatten_rollouts(
        cal_pkl,
        max_samples=max_cal_samples,
        seed=seed + 2,
        return_ids=(cal_grouping != "none"),
        feature_set=feature_set,
    )
    cal_l2 = cal["l2"]
    cal_linf = cal["linf"]
    cal_mah = mahalanobis_scores(cal["res"], sigma.Sigma_inv)
    cal_sigma_x = sigma_pred.predict_sigma(cal["X"], sigma_min=sigma_min)
    cal_adapt = cal_l2 / cal_sigma_x

    if cal_grouping != "none":
        traj_id = np.asarray(cal["traj_id"], dtype=np.int64).reshape(-1)
        if cal_grouping == "traj_max":
            keys = traj_id
        elif cal_grouping == "timeblock_max":
            # (trajectory, timeblock) grouping where timeblock = floor(t_idx / B)
            B = int(timeblock_size)
            if B <= 0:
                raise ValueError("timeblock_size must be > 0")
            t_idx = np.asarray(cal["t_idx"], dtype=np.int64).reshape(-1)
            blk = t_idx // B
            # pack into a single int64 key
            keys = (traj_id << 32) | (blk & 0xFFFFFFFF)
        else:
            # (trajectory, timeblock, nodeblock): max within a small spatiotemporal block.
            # This avoids extreme conservatism when there are only a few trajectories.
            B = int(timeblock_size)
            if B <= 0:
                raise ValueError("timeblock_size must be > 0")
            NB = int(nodeblock_size)
            if NB <= 0:
                raise ValueError("nodeblock_size must be > 0")
            t_idx = np.asarray(cal["t_idx"], dtype=np.int64).reshape(-1)
            node_idx = np.asarray(cal["node_idx"], dtype=np.int64).reshape(-1)
            tblk = t_idx // B
            nblk = node_idx // NB
            # pack into int64: [traj_id (high 24 bits-ish)] [tblk (20 bits-ish)] [nblk (20 bits-ish)]
            keys = (traj_id << 40) | ((tblk & 0xFFFFF) << 20) | (nblk & 0xFFFFF)

        cal_l2 = _group_max(cal_l2, keys)
        cal_linf = _group_max(cal_linf, keys)
        cal_mah = _group_max(cal_mah, keys)
        cal_adapt = _group_max(cal_adapt, keys)

    thresholds: Dict[str, Dict[str, float]] = {
        "l2": {},
        "linf": {},
        "mah": {},
        "adapt": {},
    }
    for a in alphas:
        a_str = str(a)
        thresholds["l2"][a_str] = conformal_quantile(cal_l2, alpha=a)
        thresholds["linf"][a_str] = conformal_quantile(cal_linf, alpha=a)
        thresholds["mah"][a_str] = conformal_quantile(cal_mah, alpha=a)
        thresholds["adapt"][a_str] = conformal_quantile(cal_adapt, alpha=a)

    (out / "thresholds.json").write_text(json.dumps(thresholds, indent=2))

    # ---------- EVAL: report empirical coverage on held-out evaluation rollouts
    ev = flatten_rollouts(
        eval_pkl,
        max_samples=max_eval_samples,
        seed=seed + 3,
        feature_set=feature_set,
    )
    ev_l2 = ev["l2"]
    ev_linf = ev["linf"]
    ev_mah = mahalanobis_scores(ev["res"], sigma.Sigma_inv)
    ev_sigma_x = sigma_pred.predict_sigma(ev["X"], sigma_min=sigma_min)

    results = {
        # Prefer "eval" terminology; include "test" for backwards compatibility.
        "paths": {
            "aux": str(aux_pkl),
            "cal": str(cal_pkl),
            "eval": str(eval_pkl),
            "test": str(eval_pkl),
        },
        "alphas": alphas,
        "cal_grouping": {"mode": cal_grouping, "timeblock_size": int(timeblock_size)},
        "feature_set": str(feature_set),
        "sigma_cap_quantile": (
            float(sigma_cap_quantile) if sigma_cap_quantile is not None else None
        ),
        "counts": {
            "aux_samples_cov": aux_counts_cov,
            "aux_samples_sigma": aux_counts_sigma,
            "cal_samples": int(cal["res"].shape[0]),
            "cal_groups": int(cal_l2.shape[0]) if cal_grouping != "none" else None,
            # Prefer "eval_samples"; include "test_samples" for backwards compatibility.
            "eval_samples": int(ev["res"].shape[0]),
            "test_samples": int(ev["res"].shape[0]),
        },
        "thresholds": thresholds,
        "coverage": {"l2": {}, "linf": {}, "mah": {}, "adapt": {}},
        "avg_radius": {"l2": {}, "linf": {}, "mah": {}, "adapt": {}},
    }

    for a in alphas:
        a_str = str(a)

        q_l2 = thresholds["l2"][a_str]
        q_linf = thresholds["linf"][a_str]
        q_mah = thresholds["mah"][a_str]
        q_adapt = thresholds["adapt"][a_str]

        cov_l2 = float(np.mean(ev_l2 <= q_l2))
        cov_linf = float(np.mean(ev_linf <= q_linf))
        cov_mah = float(np.mean(ev_mah <= q_mah))
        cov_adapt = float(np.mean(ev_l2 <= (q_adapt * ev_sigma_x)))

        results["coverage"]["l2"][a_str] = cov_l2
        results["coverage"]["linf"][a_str] = cov_linf
        results["coverage"]["mah"][a_str] = cov_mah
        results["coverage"]["adapt"][a_str] = cov_adapt

        # Radius summaries (Mah is a radius in whitened space; included for completeness)
        results["avg_radius"]["l2"][a_str] = float(q_l2)
        results["avg_radius"]["linf"][a_str] = float(q_linf)
        results["avg_radius"]["mah"][a_str] = float(q_mah)
        results["avg_radius"]["adapt"][a_str] = float(np.mean(q_adapt * ev_sigma_x))

    (out / "summary.json").write_text(json.dumps(results, indent=2))
    (out / "features_meta.json").write_text(
        json.dumps({"feature_set": str(feature_set)}, indent=2)
    )
    return results
