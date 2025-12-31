#!/usr/bin/env python3
"""Compare multiple rollout `.pkl` files (PNG only).

Rows = items you pass, cols = {Prediction, Ground Truth}.
No inference/training; reads existing `.pkl`s only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import tri as mtri
from matplotlib.patches import Rectangle

from meshgraphnet.utils.rollout_io import infer_keys, load_rollouts


def _clamp_int(x: int, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(x))))


def _extract_series(rollout_pkl: str, traj_idx: int, component: int):
    traj = load_rollouts(rollout_pkl)[traj_idx]
    keys = infer_keys(traj)

    pred = np.asarray(traj[keys.pred])  # (T,N,D)
    gt = np.asarray(traj[keys.gt])  # (T,N,D)
    mesh_pos = np.asarray(traj["mesh_pos"])  # (T,N,2)
    faces = np.asarray(traj["faces"])  # (T,F,3)

    if pred.ndim != 3:
        raise ValueError(f"{rollout_pkl}: expected pred (T,N,D), got {pred.shape}")
    if gt.shape != pred.shape:
        raise ValueError(f"{rollout_pkl}: gt shape {gt.shape} != pred {pred.shape}")
    if component < 0 or component >= pred.shape[-1]:
        raise ValueError(
            f"{rollout_pkl}: component {component} out of range for D={pred.shape[-1]}"
        )

    return keys, pred[..., component], gt[..., component], mesh_pos, faces


@dataclass
class _Item:
    label: str
    pred: np.ndarray  # (T,N)
    gt: np.ndarray  # (T,N)
    mesh_pos: np.ndarray  # (T,N,2)
    faces: np.ndarray  # (T,F,3)


def main():
    ap = argparse.ArgumentParser(
        description="Compare multiple rollout pkls (vortex-shedding style)"
    )
    ap.add_argument(
        "--item",
        action="append",
        nargs=2,
        metavar=("LABEL", "PKL"),
        required=True,
        help="Add an item to compare: label + rollout_pkl. Repeat to add multiple.",
    )
    ap.add_argument("--traj_idx", type=int, default=0)
    ap.add_argument("--component", type=int, default=0)
    ap.add_argument("--timestep", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--out_png", required=True)
    ap.add_argument(
        "--clim",
        choices=["per_frame", "global"],
        default="global",
        help="Color limits: global (default; easier comparisons) or per_frame (PhysicsNeMo exact).",
    )
    ap.add_argument(
        "--clim_q",
        nargs=2,
        type=float,
        default=(0.01, 0.99),
        metavar=("QLOW", "QHIGH"),
        help="Quantiles for global clim (robust).",
    )
    args = ap.parse_args()

    items: List[_Item] = []
    first_keys = None
    for label, pkl in args.item:
        keys, pred, gt, mesh_pos, faces = _extract_series(
            pkl, traj_idx=int(args.traj_idx), component=int(args.component)
        )
        if first_keys is None:
            first_keys = keys
        items.append(
            _Item(label=str(label), pred=pred, gt=gt, mesh_pos=mesh_pos, faces=faces)
        )

    # Single frame index (timestep in the rollout)
    step = int(args.timestep)

    # Global clim across all items and both pred/gt for this timestep
    clim: Optional[Tuple[float, float]] = None
    if args.clim == "global":
        q0, q1 = float(args.clim_q[0]), float(args.clim_q[1])
        q0 = float(np.clip(q0, 0.0, 1.0))
        q1 = float(np.clip(q1, 0.0, 1.0))
        if q1 < q0:
            q0, q1 = q1, q0
        vals = []
        for it in items:
            tmax = it.pred.shape[0] - 1
            t = min(step, tmax)
            vals.append(it.pred[t].ravel())
            vals.append(it.gt[t].ravel())
        allv = np.concatenate(vals)
        vmin = float(np.quantile(allv, q0))
        vmax = float(np.quantile(allv, q1))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin = float(np.min(allv))
            vmax = float(np.max(allv))
        clim = (vmin, vmax)

    # PhysicsNeMo-style figure setup
    plt.rcParams["image.cmap"] = "inferno"
    fig, axes = plt.subplots(nrows=len(items), ncols=2, figsize=(16, 4.5 * len(items)))
    fig.set_facecolor("black")
    if len(items) == 1:
        axes = np.asarray(axes)[None, :]  # (1,2)

    for r in range(len(items)):
        for c in range(2):
            axes[r, c].set_facecolor("black")

    for r, it in enumerate(items):
        t = min(step, it.pred.shape[0] - 1)
        xy = it.mesh_pos[t]
        tris = it.faces[t].astype(np.int32)
        triang = mtri.Triangulation(xy[:, 0], xy[:, 1], tris)

        y_pred = it.pred[t]
        y_gt = it.gt[t]

        axp = axes[r, 0]
        axg = axes[r, 1]
        axp.cla()
        axg.cla()
        for ax in (axp, axg):
            ax.set_aspect("equal")
            ax.set_axis_off()
            ax.add_patch(Rectangle((0, 0), 1.4, 0.4, facecolor="navy"))

        if clim is None:
            axp.tripcolor(
                triang, y_pred, vmin=float(np.min(y_pred)), vmax=float(np.max(y_pred))
            )
            axg.tripcolor(
                triang, y_gt, vmin=float(np.min(y_gt)), vmax=float(np.max(y_gt))
            )
        else:
            axp.tripcolor(triang, y_pred, vmin=clim[0], vmax=clim[1])
            axg.tripcolor(triang, y_gt, vmin=clim[0], vmax=clim[1])

        axp.triplot(triang, "ko-", ms=0.5, lw=0.3)
        axg.triplot(triang, "ko-", ms=0.5, lw=0.3)

        axp.set_title(f"{it.label} | Prediction", color="white")
        axg.set_title(f"{it.label} | Ground Truth", color="white")

    fig.subplots_adjust(
        left=0.03, bottom=0.03, right=0.97, top=0.97, wspace=0.06, hspace=0.18
    )

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=int(args.dpi), facecolor="black", bbox_inches="tight")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
