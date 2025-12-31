#!/usr/bin/env python3
"""Exchangeability diagnostics from a rollout `.pkl` (no conformal scaling).

What it reports:
- Temporal dependence: autocorr of RMSE(t); block KS shift (early vs late).
- Spatial dependence: Moran's I on per-node error magnitude (typically across time).

These are diagnostics (not guarantees).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

from meshgraphnet.utils.rollout_io import get_pred_gt, load_rollouts
from meshgraphnet.utils.plot_style import apply_style, savefig


def _edges_from_tris(tris: np.ndarray) -> np.ndarray:
    """Unique undirected edges from (F,3) triangles. Returns (E,2) int."""
    tris = np.asarray(tris, dtype=np.int64)
    e = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e.sort(axis=1)
    e = np.unique(e, axis=0)
    return e.astype(np.int32)


def _morans_i(x: np.ndarray, edges: np.ndarray) -> float:
    """Moran's I for scalar field x over an unweighted adjacency from edges."""
    x = np.asarray(x, dtype=float).reshape(-1)
    n = x.size
    if n < 3:
        return float("nan")
    x0 = x - x.mean()
    denom = float(np.sum(x0**2))
    if denom == 0.0:
        return float("nan")
    i, j = edges[:, 0], edges[:, 1]
    w_sum = float(edges.shape[0] * 2)  # symmetric weights
    num = float(np.sum(x0[i] * x0[j]) + np.sum(x0[j] * x0[i]))
    return (n / w_sum) * (num / denom)


def _autocorr(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    if lag <= 0 or lag >= len(x):
        return float("nan")
    x0 = x - x.mean()
    a = x0[:-lag]
    b = x0[lag:]
    denom = float(np.sqrt(np.sum(a**2) * np.sum(b**2)))
    if denom == 0:
        return float("nan")
    return float(np.sum(a * b) / denom)


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """KS statistic without scipy (two-sample)."""
    a = np.sort(np.asarray(a, dtype=float).reshape(-1))
    b = np.sort(np.asarray(b, dtype=float).reshape(-1))
    if a.size == 0 or b.size == 0:
        return float("nan")
    data = np.unique(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, data, side="right") / a.size
    cdf_b = np.searchsorted(b, data, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _acf_curve(x: np.ndarray, max_lag: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).reshape(-1)
    max_lag = int(max(1, min(max_lag, len(x) - 1)))
    lags = np.arange(1, max_lag + 1, dtype=int)
    vals = np.asarray([_autocorr(x, int(l)) for l in lags], dtype=float)
    return lags, vals


def main():
    ap = argparse.ArgumentParser(
        description="Exchangeability diagnostics from rollout pkl"
    )
    ap.add_argument("--rollout_pkl", required=True)
    ap.add_argument("--traj_idx", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--style", default="paper", choices=["paper", "default"])
    ap.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        help="Output formats (e.g. png pdf). Default: png",
    )
    ap.add_argument("--base_fontsize", type=float, default=9.0)
    ap.add_argument("--lags", type=int, nargs="+", default=[1, 2, 5, 10])
    ap.add_argument(
        "--acf_max_lag", type=int, default=50, help="Plot ACF curve up to this lag."
    )
    ap.add_argument(
        "--morans_stride",
        type=int,
        default=1,
        help="Compute Moran's I every k timesteps (stride). Use >1 if slow.",
    )
    ap.add_argument(
        "--spatial_timesteps",
        type=int,
        nargs="+",
        default=[0, 10, 20, 50, 100, 200],
        help="Also report Moran's I at these selected timesteps in the summary CSV.",
    )
    args = ap.parse_args()

    apply_style(args.style, base_fontsize=float(args.base_fontsize))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traj = load_rollouts(args.rollout_pkl)[int(args.traj_idx)]
    pred, gt, _ = get_pred_gt(traj)
    T = pred.shape[0]

    # Temporal series: RMSE(t)
    rmse = np.sqrt(np.mean((pred - gt) ** 2, axis=(1, 2)))

    # Temporal autocorrelation
    ac = {f"acf_lag{lag}": _autocorr(rmse, int(lag)) for lag in args.lags}
    lags, acf_curve = _acf_curve(rmse, int(args.acf_max_lag))

    # Block shift: early vs late RMSE(t) distribution
    third = max(1, T // 3)
    ks = _ks_statistic(rmse[:third], rmse[-third:])

    # Spatial: Moran's I on error magnitude at selected timesteps (using triangle edges)
    err_mag = np.linalg.norm(gt - pred, axis=-1)  # (T,N)
    # mesh connectivity is typically constant; build edges once.
    tris0 = np.asarray(traj["faces"])[0].astype(np.int32)
    edges = _edges_from_tris(tris0)

    # Moran's I over time (stride for speed)
    stride = int(max(1, args.morans_stride))
    ts = np.arange(0, T, stride, dtype=int)
    morans_series = np.asarray([_morans_i(err_mag[t], edges) for t in ts], dtype=float)
    morans: Dict[int, float] = {}
    for t in args.spatial_timesteps:
        tt = int(np.clip(t, 0, T - 1))
        morans[t] = _morans_i(err_mag[tt], edges)

    # Write summary CSV
    out_csv = out_dir / "exchangeability_summary.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["rollout", Path(args.rollout_pkl).name])
        w.writerow(["traj_idx", args.traj_idx])
        w.writerow(["ks_rmse_early_vs_late", ks])
        for k, v in ac.items():
            w.writerow([k, v])
        for t, v in morans.items():
            w.writerow([f"moransI_errmag_t{t}", v])

    # Plot RMSE(t)
    fig, ax = plt.subplots(figsize=(3.6, 1.55))
    ax.plot(rmse, color="black")
    ax.set_xlabel("timestep")
    ax.set_ylabel("RMSE")
    fig.tight_layout(pad=0.15)
    out_png = out_dir / "rmse_over_time.png"
    savefig(fig, out_png, formats=list(args.formats))
    plt.close(fig)

    # Plot ACF curve
    fig, ax = plt.subplots(figsize=(3.6, 1.55))
    ax.plot(
        lags, acf_curve, marker="o", markersize=2.5, linewidth=1.0, color="tab:blue"
    )
    ax.axhline(0.0, color="0.2", linewidth=0.8)
    ax.set_xlabel("lag")
    ax.set_ylabel("ACF")
    # Titles are intentionally omitted.
    fig.tight_layout(pad=0.15)
    out_acf = out_dir / "rmse_acf.png"
    savefig(fig, out_acf, formats=list(args.formats))
    plt.close(fig)

    # Plot early vs late RMSE histogram (visualize shift behind KS)
    a = rmse[:third]
    b = rmse[-third:]
    fig, ax = plt.subplots(figsize=(3.6, 1.55))
    bins = max(10, int(np.sqrt(T)))
    ax.hist(a, bins=bins, alpha=0.55, label="early", density=True, color="tab:blue")
    ax.hist(b, bins=bins, alpha=0.55, label="late", density=True, color="tab:orange")
    ax.set_xlabel("RMSE(t)")
    ax.set_ylabel("density")
    # Titles are intentionally omitted; KS lives in caption/text and CSV.
    ax.legend(loc="best")
    fig.tight_layout(pad=0.15)
    out_hist = out_dir / "rmse_early_vs_late_hist.png"
    savefig(fig, out_hist, formats=list(args.formats))
    plt.close(fig)

    # Plot Moran's I over time
    fig, ax = plt.subplots(figsize=(3.6, 1.55))
    ax.plot(
        ts, morans_series, marker="o", markersize=2.0, linewidth=1.0, color="tab:green"
    )
    ax.axhline(0.0, color="0.2", linewidth=0.8)
    ax.set_xlabel("timestep")
    ax.set_ylabel("Moran's I")
    # Titles are intentionally omitted.
    fig.tight_layout(pad=0.15)
    out_mi = out_dir / "moransI_over_time.png"
    savefig(fig, out_mi, formats=list(args.formats))
    plt.close(fig)

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_png.with_suffix('')}")
    print(f"Wrote {out_acf.with_suffix('')}")
    print(f"Wrote {out_hist.with_suffix('')}")
    print(f"Wrote {out_mi.with_suffix('')}")


if __name__ == "__main__":
    main()
