#!/usr/bin/env python3
"""Error accumulation from rollout `.pkl` files (RMSE over time + summary).

If `--traj_idx=-1` (default), aggregates across *all* trajectories in each rollout file and
plots mean ± IQR bands (dense, more informative than a single trajectory).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from meshgraphnet.utils.rollout_io import get_pred_gt, infer_keys, load_rollouts
from meshgraphnet.utils.plot_style import apply_style, savefig


def rmse_t(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """RMSE per timestep over nodes+components. pred/gt: (T,N,D) -> (T,)."""
    return np.sqrt(np.mean((pred - gt) ** 2, axis=(1, 2)))


def summarize_horizons(
    rmse: np.ndarray, horizons=(1, 10, 20, 50, 100, 200)
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    T = len(rmse)
    for h in horizons:
        if h < T:
            out[f"rmse@{h}"] = float(np.mean(rmse[1 : h + 1]))
    out["rmse@final"] = float(rmse[-1])
    out["auc_rmse"] = float(np.mean(rmse))  # average over time
    return out


def main():
    ap = argparse.ArgumentParser(description="Error accumulation from rollout pkls")
    ap.add_argument("--rollout_pkl", action="append", required=True)
    ap.add_argument(
        "--traj_idx",
        type=int,
        default=-1,
        help="Trajectory index inside each pkl. Use -1 to aggregate all trajectories (default).",
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--style", default="paper", choices=["paper", "default"])
    ap.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        help="Output formats (e.g. png pdf). Default: png",
    )
    ap.add_argument("--base_fontsize", type=float, default=9.0)
    ap.add_argument(
        "--plot_all_trajectories",
        action="store_true",
        help="Overlay per-trajectory RMSE(t) as faint lines (when --traj_idx=-1).",
    )
    ap.add_argument(
        "--max_traj_lines",
        type=int,
        default=200,
        help="If overlaying all trajectories, cap the number of lines per rollout (deterministic).",
    )
    args = ap.parse_args()

    apply_style(args.style, base_fontsize=float(args.base_fontsize))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # For plotting: per rollout, store (name, rmse_matrix [K,T], mean[T], q25[T], q75[T])
    plot_items: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    summaries: List[Dict[str, str]] = []

    for pkl in args.rollout_pkl:
        rollouts = load_rollouts(pkl)
        if args.traj_idx >= 0:
            traj_ids = [int(args.traj_idx)]
        else:
            traj_ids = list(range(len(rollouts)))

        rmse_list: List[np.ndarray] = []
        keys_gt = None
        for ti in traj_ids:
            traj = rollouts[ti]
            pred, gt, keys = get_pred_gt(traj)
            keys_gt = keys.gt
            r = rmse_t(pred, gt)
            rmse_list.append(r)

            s = summarize_horizons(r)
            s_row = {"rollout": Path(pkl).name, "traj_idx": str(ti), "field": keys.gt}
            s_row.update({k: f"{v:.6g}" for k, v in s.items()})
            summaries.append(s_row)

        # stack with NaN padding to max T for this rollout
        maxT = max(len(r) for r in rmse_list)
        mat = np.full((len(rmse_list), maxT), np.nan, dtype=float)
        for i, r in enumerate(rmse_list):
            mat[i, : len(r)] = r
        mean = np.nanmean(mat, axis=0)
        q25 = np.nanquantile(mat, 0.25, axis=0)
        q75 = np.nanquantile(mat, 0.75, axis=0)
        name = Path(pkl).name
        plot_items.append((name, mat, mean, q25, q75))

    # CSV: rmse time series
    ts_csv = out_dir / "rmse_over_time.csv"
    maxT = max(mat.shape[1] for _, mat, _, _, _ in plot_items)
    with ts_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestep"] + [n for n, *_ in plot_items])
        for t in range(maxT):
            row = [t]
            for _, mat, _, _, _ in plot_items:
                # write mean over trajectories at timestep t
                row.append("" if t >= mat.shape[1] else float(np.nanmean(mat[:, t])))
            w.writerow(row)

    # CSV: summary metrics
    summary_csv = out_dir / "summary.csv"
    fieldnames = sorted({k for d in summaries for k in d.keys()})
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summaries)

    # Plot: one subplot per rollout (mean + IQR band)
    # Use a horizontal layout (1 row × N cols) for paper-friendly comparison.
    n = len(plot_items)
    fig_w = max(3.6, 3.2 * n)
    fig_h = 2.25
    fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(fig_w, fig_h), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (name, mat, mean, q25, q75) in zip(axes, plot_items):
        if args.plot_all_trajectories and args.traj_idx < 0:
            # Overlay raw trajectories (rich plot). If too many, plot a deterministic subset.
            k = mat.shape[0]
            cap = int(max(1, args.max_traj_lines))
            if k > cap:
                idx = np.linspace(0, k - 1, num=cap, dtype=int)
                mat_plot = mat[idx]
            else:
                mat_plot = mat
            for r in mat_plot:
                ax.plot(r, color="tab:blue", alpha=0.08, linewidth=0.7)
        ax.fill_between(
            np.arange(len(mean)),
            q25,
            q75,
            alpha=0.22,
            color="tab:blue",
            label="IQR (25–75%)",
            linewidth=0.0,
        )
        ax.plot(mean, label="mean RMSE", color="black", linewidth=1.4)
        # Titles are intentionally omitted; use the legend/caption in the paper.
        ax.set_xlabel("timestep")
        ax.legend(loc="upper left")
    axes[0].set_ylabel("RMSE")
    fig.tight_layout(pad=0.2, w_pad=0.8)
    out_png = out_dir / "rmse_over_time.png"
    savefig(fig, out_png, formats=list(args.formats))
    plt.close(fig)

    print(f"Wrote {ts_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {out_png.with_suffix('')}")


if __name__ == "__main__":
    main()
