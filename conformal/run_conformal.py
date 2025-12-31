#!/usr/bin/env python3
"""
Run split conformal prediction on MeshGraphNet rollout .pkl files.

Inputs: 3 rollout pickle files (auxiliary, calibration, evaluation).
Outputs: Σ / Σ^{-1}, sigma(x) model, thresholds, summary metrics.
"""

import argparse
import json
from pathlib import Path
import sys

# Allow running as a script: `python conformal/run_conformal.py ...`
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conformal.pipeline import run_conformal_pipeline


def main():
    ap = argparse.ArgumentParser(
        description="Conformal prediction for MeshGraphNet rollouts"
    )
    ap.add_argument("--aux_pkl", type=str, required=True, help="Auxiliary rollout .pkl")
    ap.add_argument(
        "--cal_pkl", type=str, required=True, help="Calibration rollout .pkl"
    )
    ap.add_argument(
        "--eval_pkl",
        type=str,
        default="",
        help="Evaluation rollout .pkl (held-out; used only for reporting).",
    )
    ap.add_argument(
        "--test_pkl",
        type=str,
        default="",
        help="DEPRECATED alias for --eval_pkl (kept for backwards compatibility).",
    )
    ap.add_argument("--outdir", type=str, required=True, help="Output directory")
    ap.add_argument(
        "--reuse_aux_from",
        type=str,
        default="",
        help="Optional: reuse Sigma.npy/Sigma_inv.npy/sigma_model.pkl from an existing conformal output dir (skips refitting aux objects).",
    )
    ap.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.1, 0.05],
        help="Miscoverage levels (default: 0.1 0.05)",
    )

    ap.add_argument(
        "--cov_eps",
        type=float,
        default=1e-6,
        help="Ridge added to covariance diag (default: 1e-6)",
    )
    ap.add_argument(
        "--sigma_model",
        type=str,
        default="auto",
        choices=["auto", "ridge", "sklearn_hgb", "sklearn_hgb_quantile", "xgboost"],
        help="Sigma(x) model type",
    )
    ap.add_argument(
        "--sigma_lam",
        type=float,
        default=1e-3,
        help="Ridge lambda (only for ridge sigma model)",
    )
    ap.add_argument("--sigma_min", type=float, default=1e-6, help="Minimum sigma floor")
    ap.add_argument(
        "--sigma_cap_quantile",
        type=float,
        default=0.0,
        help="Optional: cap predicted sigma(x) at this quantile of auxiliary sigma predictions "
        "(0 disables; recommended: 0.999 to prevent heavy-tail sigma from dominating size).",
    )

    ap.add_argument(
        "--feature_set",
        type=str,
        default="default",
        choices=["default", "physics_basic", "physics_full"],
        help="Feature set used for sigma(x) learning and adaptive CP.",
    )

    ap.add_argument(
        "--cal_grouping",
        type=str,
        default="none",
        choices=["none", "traj_max", "timeblock_max", "timeblock_nodeblock_max"],
        help="Blockwise calibration quantile: group calibration scores and take max within each group before computing the conformal quantile.",
    )
    ap.add_argument(
        "--timeblock_size",
        type=int,
        default=10,
        help="Block size in timesteps when cal_grouping=timeblock_max (default: 10).",
    )
    ap.add_argument(
        "--nodeblock_size",
        type=int,
        default=64,
        help="Node block size when cal_grouping=timeblock_nodeblock_max (default: 64).",
    )

    ap.add_argument(
        "--max_aux_cov",
        type=int,
        default=0,
        help="Max samples for covariance fit (0=all)",
    )
    ap.add_argument(
        "--max_aux_sigma",
        type=int,
        default=200000,
        help="Max samples for sigma fit (0=all)",
    )
    ap.add_argument(
        "--max_cal",
        type=int,
        default=0,
        help="Max samples for calibration quantiles (0=all)",
    )
    ap.add_argument(
        "--max_eval",
        type=int,
        default=0,
        help="Max samples for evaluation reporting (0=all)",
    )
    ap.add_argument(
        "--max_test",
        type=int,
        default=0,
        help="DEPRECATED alias for --max_eval (0=all).",
    )
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for subsampling")
    args = ap.parse_args()

    eval_pkl = args.eval_pkl or args.test_pkl
    if not eval_pkl:
        raise SystemExit(
            "Must provide --eval_pkl (preferred) or --test_pkl (deprecated)."
        )
    max_eval = args.max_eval or args.max_test or None

    results = run_conformal_pipeline(
        aux_pkl=args.aux_pkl,
        cal_pkl=args.cal_pkl,
        eval_pkl=eval_pkl,
        outdir=args.outdir,
        alphas=args.alphas,
        sigma_eps=args.cov_eps,
        sigma_model=args.sigma_model,
        sigma_model_lam=args.sigma_lam,
        sigma_min=args.sigma_min,
        sigma_cap_quantile=(args.sigma_cap_quantile or None),
        cal_grouping=args.cal_grouping,
        timeblock_size=args.timeblock_size,
        nodeblock_size=args.nodeblock_size,
        feature_set=args.feature_set,
        reuse_aux_from=(args.reuse_aux_from or None),
        max_aux_samples_cov=(args.max_aux_cov or None),
        max_aux_samples_sigma=(args.max_aux_sigma or None),
        max_cal_samples=(args.max_cal or None),
        max_eval_samples=max_eval,
        seed=args.seed,
    )

    print(json.dumps(results["coverage"], indent=2))
    print(f"Wrote: {Path(args.outdir) / 'summary.json'}")


if __name__ == "__main__":
    main()
