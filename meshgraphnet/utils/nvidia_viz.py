#!/usr/bin/env python3
"""NVIDIA/PhysicsNeMo-style rollout snapshot (PNG).

Creates a snapshot from an existing rollout `.pkl`:
- Prediction
- Ground truth
- Absolute error

No training/inference; uses stored rollout arrays only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import tri as mtri
from matplotlib.ticker import MaxNLocator

from meshgraphnet.utils.plot_style import apply_style, savefig
from meshgraphnet.utils.rollout_io import get_pred_gt, load_rollouts


def _clamp_int(x: int, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(x))))


def _scalar_field(
    pred_tnd: np.ndarray, gt_tnd: np.ndarray, mode: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pred_scalar, gt_scalar, abs_err) for a single timestep."""
    if mode == "component0":
        p = pred_tnd[:, 0]
        g = gt_tnd[:, 0]
    elif mode == "component1":
        p = pred_tnd[:, 1]
        g = gt_tnd[:, 1]
    elif mode == "magnitude":
        p = np.linalg.norm(pred_tnd, axis=-1)
        g = np.linalg.norm(gt_tnd, axis=-1)
    else:
        raise ValueError(f"Unknown mode={mode!r}")
    e = np.abs(g - p)
    return p, g, e


def main() -> None:
    ap = argparse.ArgumentParser(
        description="NVIDIA-style rollout visualization (snapshot)"
    )
    ap.add_argument("--rollout_pkl", required=True)
    ap.add_argument("--traj_idx", type=int, default=0)
    ap.add_argument("--timestep", type=int, default=0)
    ap.add_argument(
        "--mode",
        choices=["magnitude", "component0", "component1"],
        default="magnitude",
        help="What scalar to visualize from vector fields (CylinderFlow velocity).",
    )
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--style", default="paper", choices=["paper", "default"])
    ap.add_argument("--base_fontsize", type=float, default=9.0)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument(
        "--quantiles",
        nargs=2,
        type=float,
        default=(0.01, 0.99),
        metavar=("QLOW", "QHIGH"),
        help="Robust color limits for pred/gt.",
    )
    args = ap.parse_args()

    apply_style(args.style, base_fontsize=float(args.base_fontsize))

    traj = load_rollouts(args.rollout_pkl)[int(args.traj_idx)]
    pred, gt, _ = get_pred_gt(traj)  # (T,N,D)
    T = pred.shape[0]
    t = _clamp_int(int(args.timestep), 0, T - 1)

    xy = np.asarray(traj["mesh_pos"])[t]  # (N,2)
    tris = np.asarray(traj["faces"])[t].astype(np.int32)  # (F,3)
    triang = mtri.Triangulation(xy[:, 0], xy[:, 1], tris)

    p, g, e = _scalar_field(pred[t], gt[t], args.mode)

    # Robust shared limits for pred/gt
    q0, q1 = float(np.clip(args.quantiles[0], 0.0, 1.0)), float(
        np.clip(args.quantiles[1], 0.0, 1.0)
    )
    if q1 < q0:
        q0, q1 = q1, q0
    pg = np.concatenate([p.ravel(), g.ravel()])
    vmin = float(np.quantile(pg, q0))
    vmax = float(np.quantile(pg, q1))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = float(np.min(pg)), float(np.max(pg))

    # Error limits
    emax = float(np.quantile(e, 0.99))
    if not np.isfinite(emax) or emax <= 0:
        emax = float(np.max(e)) if np.max(e) > 0 else 1.0

    # NVIDIA-like dark look (caption carries context; no titles)
    fig, axes = plt.subplots(
        nrows=1, ncols=3, figsize=(6.8, 2.2), constrained_layout=False
    )
    fig.patch.set_facecolor("black")
    for ax in axes:
        ax.set_facecolor("black")
        ax.set_aspect("equal")
        ax.set_axis_off()

    plt.rcParams["image.cmap"] = "inferno"
    im0 = axes[0].tripcolor(triang, p, vmin=vmin, vmax=vmax)
    im1 = axes[1].tripcolor(triang, g, vmin=vmin, vmax=vmax)
    plt.rcParams["image.cmap"] = "magma"
    im2 = axes[2].tripcolor(triang, e, vmin=0.0, vmax=emax)

    # Subtle mesh overlay (thin, unobtrusive)
    for ax in axes:
        ax.triplot(triang, color=(0.0, 0.0, 0.0, 0.28), linewidth=0.25)

    # Single colorbar (error only). Pred/GT share vmin/vmax but the paper caption can
    # describe the normalization; the quantitative scale of interest is the error.
    cax = fig.add_axes([0.92, 0.20, 0.014, 0.60])  # abs error bar
    cb = fig.colorbar(im2, cax=cax)
    cb.locator = MaxNLocator(nbins=4)
    cb.update_ticks()
    cb.outline.set_edgecolor("white")
    cb.ax.yaxis.set_ticks_position("right")
    cb.ax.yaxis.set_label_position("right")
    cb.ax.tick_params(colors="white", labelsize=8, pad=1)
    plt.setp(cb.ax.get_yticklabels(), color="white")

    # Tight manual layout for dark figure
    fig.subplots_adjust(left=0.02, right=0.90, bottom=0.06, top=0.98, wspace=0.03)

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=int(args.dpi), facecolor="black", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
