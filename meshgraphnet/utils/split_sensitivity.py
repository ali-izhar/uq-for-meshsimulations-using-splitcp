#!/usr/bin/env python3
"""Summarize split-sensitivity rollouts (across seeds) from `rollouts_sensitivity/`.

Produces *dense* plots by default by aggregating across **all trajectories** in each rollout file.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from meshgraphnet.utils.rollout_io import get_pred_gt, load_rollouts
from meshgraphnet.utils.plot_style import apply_style, savefig


_RE = re.compile(
    r"rollout_(?P<dataset>cylinder|flag)_(?P<split>auxiliary|calibration|test)_seed(?P<seed>\d+)\.pkl$"
)


def _rmse_t(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((pred - gt) ** 2, axis=(1, 2)))


def _violin_with_points(
    ys_by_group: List[List[float]],
    group_labels: List[str],
    *,
    title: str,
    ylabel: str,
    out_path: Path,
    formats: List[str],
) -> None:
    # Slightly wider to prevent ylabel clipping in single-column layouts.
    fig, ax = plt.subplots(figsize=(3.6, 2.15))
    parts = ax.violinplot(ys_by_group, showmeans=True, showextrema=False)
    for pc in parts.get("bodies", []):
        pc.set_alpha(0.55)
        pc.set_facecolor("0.6")
        pc.set_edgecolor("0.25")
        pc.set_linewidth(0.8)
    if "cmeans" in parts:
        parts["cmeans"].set_color("0.15")
        parts["cmeans"].set_linewidth(1.2)
    rng = np.random.default_rng(0)
    for i, ys in enumerate(ys_by_group, start=1):
        xs = i + rng.uniform(-0.10, 0.10, size=len(ys))
        ax.scatter(xs, ys, s=10, alpha=0.65, color="0.1", linewidths=0.0)
    ax.set_xticks(range(1, len(group_labels) + 1), group_labels)
    ax.set_ylabel(ylabel, labelpad=4)
    # Titles are intentionally omitted (paper captions should carry context).
    fig.tight_layout(pad=0.2)
    savefig(fig, out_path, formats=formats)
    plt.close(fig)


def _seed_split_heatmap(
    mat: np.ndarray,
    *,
    seeds: List[str],
    splits: List[str],
    title: str,
    cbar_label: str,
    out_path: Path,
    formats: List[str],
) -> None:
    # Compact heatmap; height scales mildly with number of seeds.
    h = max(1.6, 0.32 * len(seeds) + 0.9)
    fig, ax = plt.subplots(figsize=(3.6, h))
    im = ax.imshow(mat, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.03)
    cbar.set_label(cbar_label)
    ax.set_xticks(range(len(splits)), splits)
    ax.set_yticks(range(len(seeds)), seeds)
    ax.set_xlabel("split")
    ax.set_ylabel("seed")
    # Titles are intentionally omitted.
    fig.tight_layout(pad=0.2)
    savefig(fig, out_path, formats=formats)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Split sensitivity summary from rollouts_sensitivity/"
    )
    ap.add_argument("--rollouts_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--traj_idx",
        type=int,
        default=-1,
        help="Trajectory index inside each pkl. Use -1 to aggregate all trajectories (default).",
    )
    ap.add_argument("--style", default="paper", choices=["paper", "default"])
    ap.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        help="Output formats (e.g. png pdf). Default: png",
    )
    ap.add_argument("--base_fontsize", type=float, default=9.0)
    args = ap.parse_args()

    apply_style(args.style, base_fontsize=float(args.base_fontsize))

    rollouts_dir = Path(args.rollouts_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    grouped_points_final: Dict[Tuple[str, str], List[float]] = {}
    grouped_points_auc: Dict[Tuple[str, str], List[float]] = {}
    grouped_seed_means_final: Dict[Tuple[str, str, str], List[float]] = {}
    grouped_seed_means_auc: Dict[Tuple[str, str, str], List[float]] = {}

    for pkl in sorted(rollouts_dir.glob("rollout_*_seed*.pkl")):
        m = _RE.match(pkl.name)
        if not m:
            continue
        dataset = m.group("dataset")
        split = m.group("split")
        seed = m.group("seed")

        rollouts = load_rollouts(pkl)
        if args.traj_idx >= 0:
            traj_ids = [int(args.traj_idx)]
        else:
            traj_ids = list(range(len(rollouts)))

        for ti in traj_ids:
            traj = rollouts[ti]
            pred, gt, _ = get_pred_gt(traj)
            r = _rmse_t(pred, gt)
            final = float(r[-1])
            auc = float(np.mean(r))

            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "seed": seed,
                    "traj_idx": str(ti),
                    "rmse_final": f"{final:.6g}",
                    "rmse_auc": f"{auc:.6g}",
                    "timesteps": str(len(r)),
                    "file": pkl.name,
                }
            )
            grouped_points_final.setdefault((dataset, split), []).append(final)
            grouped_points_auc.setdefault((dataset, split), []).append(auc)
            grouped_seed_means_final.setdefault((dataset, split, seed), []).append(
                final
            )
            grouped_seed_means_auc.setdefault((dataset, split, seed), []).append(auc)

    out_csv = out_dir / "split_sensitivity.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)

    datasets = sorted({d for d, _ in grouped_points_final.keys()})
    split_order = ["auxiliary", "calibration", "test"]

    # Dense plots: per dataset, distribution across (seed, traj) points + heatmap of per-seed means.
    for dataset in datasets:
        splits = [s for (d, s) in grouped_points_final.keys() if d == dataset]
        splits = [s for s in split_order if s in splits]
        if not splits:
            continue

        # 1) Violin + jittered scatter (final RMSE)
        data_final = [grouped_points_final[(dataset, s)] for s in splits]
        _violin_with_points(
            data_final,
            splits,
            title="",
            ylabel="final-step RMSE",
            out_path=out_dir / f"{dataset}_split_sensitivity_final_rmse.png",
            formats=list(args.formats),
        )

        # 2) Violin + jittered scatter (AUC RMSE)
        data_auc = [grouped_points_auc[(dataset, s)] for s in splits]
        _violin_with_points(
            data_auc,
            splits,
            title="",
            ylabel="AUC RMSE (mean$_t$ RMSE(t))",
            out_path=out_dir / f"{dataset}_split_sensitivity_auc_rmse.png",
            formats=list(args.formats),
        )

        # 3) Heatmap: seed × split of mean(metric over trajectories)
        seeds = sorted(
            {seed for (d, _s, seed) in grouped_seed_means_final.keys() if d == dataset}
        )
        if seeds:
            mat_final = np.full((len(seeds), len(splits)), np.nan, dtype=float)
            mat_auc = np.full((len(seeds), len(splits)), np.nan, dtype=float)
            for si, seed in enumerate(seeds):
                for sj, split in enumerate(splits):
                    vals_f = grouped_seed_means_final.get((dataset, split, seed), [])
                    vals_a = grouped_seed_means_auc.get((dataset, split, seed), [])
                    if vals_f:
                        mat_final[si, sj] = float(np.mean(vals_f))
                    if vals_a:
                        mat_auc[si, sj] = float(np.mean(vals_a))
            _seed_split_heatmap(
                mat_final,
                seeds=seeds,
                splits=splits,
                title="",
                cbar_label="mean final RMSE",
                out_path=out_dir / f"{dataset}_seed_split_heatmap_final_rmse.png",
                formats=list(args.formats),
            )
            _seed_split_heatmap(
                mat_auc,
                seeds=seeds,
                splits=splits,
                title="",
                cbar_label="mean AUC RMSE",
                out_path=out_dir / f"{dataset}_seed_split_heatmap_auc_rmse.png",
                formats=list(args.formats),
            )

        # 4) Final vs AUC scatter (all points), colored by split
        fig, ax = plt.subplots(figsize=(3.6, 2.6))
        color = {
            "auxiliary": "tab:blue",
            "calibration": "tab:orange",
            "test": "tab:green",
        }
        for split in splits:
            x = np.asarray(grouped_points_auc[(dataset, split)], dtype=float)
            y = np.asarray(grouped_points_final[(dataset, split)], dtype=float)
            ax.scatter(
                x,
                y,
                s=12,
                alpha=0.65,
                label=split,
                color=color.get(split, "black"),
                linewidths=0.0,
            )
        ax.set_xlabel("AUC RMSE (mean_t RMSE(t))")
        ax.set_ylabel("final-step RMSE")
        # Titles are intentionally omitted.
        ax.legend(loc="best")
        fig.tight_layout(pad=0.2)
        savefig(
            fig,
            out_dir / f"{dataset}_final_vs_auc_scatter.png",
            formats=list(args.formats),
        )
        plt.close(fig)

    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
