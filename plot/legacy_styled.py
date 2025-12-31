#!/usr/bin/env python3
"""
Supplementary diagnostic plots.

Usage:
    PYTHONPATH=. python plot/legacy_styled.py morans_i
    PYTHONPATH=. python plot/legacy_styled.py early_late
    PYTHONPATH=. python plot/legacy_styled.py split_heatmap
    PYTHONPATH=. python plot/legacy_styled.py final_vs_auc
    PYTHONPATH=. python plot/legacy_styled.py all
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from plot.style import apply_paper_style, get_mpl, PUBLICATION_DPI


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _load_rollouts(pkl_path: str) -> list:
    """Load rollout trajectories from pickle file."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _get_pred_gt(traj: dict) -> tuple:
    """Extract prediction and ground truth arrays from trajectory."""
    if "pred_velocity" in traj:
        return traj["pred_velocity"], traj["gt_velocity"]
    elif "pred_pos" in traj:
        return traj["pred_pos"], traj["gt_pos"]
    raise KeyError("Cannot find velocity or pos keys in trajectory")


def _rmse_t(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Compute RMSE at each timestep."""
    err = pred - gt
    return np.sqrt((err**2).sum(axis=-1).mean(axis=-1))


def _edges_from_tris(faces: np.ndarray) -> set:
    """Extract unique edges from triangle faces."""
    edges = set()
    for tri in faces:
        for i in range(3):
            e = tuple(sorted([tri[i], tri[(i + 1) % 3]]))
            edges.add(e)
    return edges


def _morans_i(values: np.ndarray, edges: set, n_nodes: int) -> float:
    """Compute Moran's I spatial autocorrelation."""
    mean_val = values.mean()
    denom = ((values - mean_val) ** 2).sum()
    if denom < 1e-12:
        return 0.0

    numer = 0.0
    W = 0
    for i, j in edges:
        if i < n_nodes and j < n_nodes:
            numer += (values[i] - mean_val) * (values[j] - mean_val)
            W += 1

    if W == 0:
        return 0.0
    return (n_nodes / W) * (numer / denom)


def _autocorr(x: np.ndarray, max_lag: int = 50) -> np.ndarray:
    """Compute autocorrelation function."""
    x = x - x.mean()
    n = len(x)
    acf = np.correlate(x, x, mode="full")[n - 1 :]
    if acf[0] > 0:
        acf = acf / acf[0]
    return acf[: max_lag + 1]


# ---------------------------------------------------------------------------
# Moran's I Time Series (comparative)
# ---------------------------------------------------------------------------


def fig_morans_i_timeseries(
    cylinder_pkls: list,
    flag_pkls: list,
    out_png: str = "assets/morans_i_timeseries.png",
):
    """
    Create a comparative Moran's I over time plot for both datasets.
    Shows spatial autocorrelation of error magnitude across ALL trajectories
    from ALL provided pkl files, with median line and IQR bands.
    """
    _, plt, *_ = get_mpl()
    apply_paper_style()

    fig, ax = plt.subplots(figsize=(4.0, 2.5))

    datasets = [
        ("CylinderFlow", cylinder_pkls, "#1f77b4", "o"),
        ("Flag", flag_pkls, "#d62728", "s"),
    ]

    for name, pkl_paths, color, marker in datasets:
        # Collect Moran's I from ALL trajectories across ALL pkl files
        all_morans = []
        timesteps = None
        n_total = 0

        for pkl_path in pkl_paths:
            rollouts = _load_rollouts(pkl_path)
            n_total += len(rollouts)

            for traj in rollouts:
                pred, gt = _get_pred_gt(traj)
                pos = (
                    traj["mesh_pos"][0]
                    if traj["mesh_pos"].ndim == 3
                    else traj["mesh_pos"]
                )
                faces = traj["faces"][0] if traj["faces"].ndim == 3 else traj["faces"]

                edges = _edges_from_tris(faces)
                n_nodes = pos.shape[0]
                T = pred.shape[0]

                # Sample timesteps (use same for all trajectories)
                if timesteps is None:
                    timesteps = np.linspace(0, T - 1, min(30, T)).astype(int)

                morans = []
                for t in timesteps:
                    err_mag = np.linalg.norm(pred[t] - gt[t], axis=-1)
                    morans.append(_morans_i(err_mag, edges, n_nodes))
                all_morans.append(morans)

        # Stack and compute statistics
        all_morans = np.array(all_morans)  # (n_traj, n_timesteps)
        median = np.median(all_morans, axis=0)
        q25 = np.percentile(all_morans, 25, axis=0)
        q75 = np.percentile(all_morans, 75, axis=0)

        # Plot IQR band
        ax.fill_between(timesteps, q25, q75, alpha=0.2, color=color, linewidth=0)

        # Plot median line with markers
        ax.plot(
            timesteps,
            median,
            marker=marker,
            markersize=3,
            color=color,
            label=f"{name} (n={n_total})",
            linewidth=1.2,
        )

    ax.set_xlabel("timestep")
    ax.set_ylabel("Moran's I (|error|)")
    ax.set_ylim(0, 1.05)
    ax.legend(framealpha=0.9, fontsize=7)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=PUBLICATION_DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote: {out_png}")


# ---------------------------------------------------------------------------
# Early vs Late Histogram (comparative)
# ---------------------------------------------------------------------------


def fig_early_late_hist(
    cylinder_pkls: list,
    flag_pkls: list,
    out_png: str = "assets/early_late_hist.png",
):
    """
    Create a 1x2 plot showing early vs late RMSE distributions for both datasets.
    Aggregates RMSE from ALL trajectories across ALL provided pkl files.
    """
    _, plt, *_ = get_mpl()
    apply_paper_style()

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.2))

    datasets = [
        ("CylinderFlow", cylinder_pkls),
        ("Flag", flag_pkls),
    ]

    for idx, (name, pkl_paths) in enumerate(datasets):
        ax = axes[idx]

        # Collect RMSE from ALL trajectories across ALL pkl files
        all_early = []
        all_late = []
        n_total = 0

        for pkl_path in pkl_paths:
            rollouts = _load_rollouts(pkl_path)
            n_total += len(rollouts)

            for traj in rollouts:
                pred, gt = _get_pred_gt(traj)
                rmse = _rmse_t(pred, gt)
                T = len(rmse)
                all_early.extend(rmse[: T // 2])
                all_late.extend(rmse[T // 2 :])

        all_early = np.array(all_early)
        all_late = np.array(all_late)
        all_rmse = np.concatenate([all_early, all_late])

        bins = np.linspace(0, np.percentile(all_rmse, 99), 30)
        ax.hist(
            all_early,
            bins=bins,
            alpha=0.6,
            label="Early",
            color="#1f77b4",
            density=True,
        )
        ax.hist(
            all_late, bins=bins, alpha=0.6, label="Late", color="#d62728", density=True
        )

        ax.set_xlabel("RMSE(t)")
        ax.set_ylabel("Density" if idx == 0 else "")
        ax.set_title(f"{name} (n={n_total})", fontsize=9)
        ax.legend(fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=PUBLICATION_DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote: {out_png}")


# ---------------------------------------------------------------------------
# Split Heatmap (seed × split)
# ---------------------------------------------------------------------------


def fig_split_heatmap(
    sensitivity_csv: str,
    out_png: str = "assets/split_heatmap.png",
):
    """
    Create a 1x2 heatmap showing seed × split AUC RMSE for both datasets.
    """
    _, plt, *_ = get_mpl()
    apply_paper_style()

    df = pd.read_csv(sensitivity_csv)

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))

    for idx, dataset in enumerate(["cylinder", "flag"]):
        ax = axes[idx]
        sub = df[df["dataset"] == dataset].copy()

        if sub.empty:
            ax.set_visible(False)
            continue

        # Pivot to seed × split
        pivot = sub.pivot_table(
            values="rmse_auc", index="seed", columns="split", aggfunc="mean"
        )

        # Reorder columns
        col_order = ["auxiliary", "calibration", "test"]
        pivot = pivot[[c for c in col_order if c in pivot.columns]]

        im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([c[:3] for c in pivot.columns], fontsize=7)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=7)
        ax.set_xlabel("Split")
        ax.set_ylabel("Seed" if idx == 0 else "")
        ax.set_title(dataset.capitalize(), fontsize=9)

        # Add values
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                color = "white" if val > pivot.values.max() * 0.6 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=color,
                )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="AUC RMSE")

    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=PUBLICATION_DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote: {out_png}")


# ---------------------------------------------------------------------------
# Final vs AUC Scatter
# ---------------------------------------------------------------------------


def fig_final_vs_auc(
    sensitivity_csv: str,
    out_png: str = "assets/final_vs_auc.png",
):
    """
    Create a 1x2 scatter plot showing final RMSE vs AUC RMSE for both datasets.
    Uses ALL trajectory data from split sensitivity analysis.
    """
    _, plt, *_ = get_mpl()
    apply_paper_style()

    df = pd.read_csv(sensitivity_csv)

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))

    colors = {"auxiliary": "#1f77b4", "calibration": "#ff7f0e", "test": "#2ca02c"}
    split_labels = {"auxiliary": "Aux", "calibration": "Cal", "test": "Eval"}
    dataset_titles = {"cylinder": "CylinderFlow", "flag": "Flag"}

    for idx, dataset in enumerate(["cylinder", "flag"]):
        ax = axes[idx]
        sub = df[df["dataset"] == dataset].copy()

        if sub.empty:
            ax.set_visible(False)
            continue

        n_total = len(sub)
        n_clipped = 0

        # For Flag, clip outliers at 98.9th percentile of AUC RMSE
        # (consistent with violin plot which uses same threshold)
        if dataset == "flag":
            clip_val = np.percentile(sub["rmse_auc"], 98.9)
            n_clipped = (sub["rmse_auc"] > clip_val).sum()
            sub = sub[sub["rmse_auc"] <= clip_val]

        for split in ["auxiliary", "calibration", "test"]:
            split_data = sub[sub["split"] == split]
            if not split_data.empty:
                ax.scatter(
                    split_data["rmse_auc"],
                    split_data["rmse_final"],
                    c=colors.get(split, "gray"),
                    label=split_labels[split],
                    s=18,
                    alpha=0.6,
                    edgecolors="none",
                )

        ax.set_xlabel("AUC RMSE")
        ax.set_ylabel("Final RMSE" if idx == 0 else "")
        ax.set_title(f"{dataset_titles[dataset]} (n={n_total})", fontsize=9)
        ax.legend(fontsize=6, markerscale=1.0, loc="upper left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add note about clipping for Flag
        if dataset == "flag" and n_clipped > 0:
            ax.text(
                0.98,
                0.98,
                f"({n_clipped} outliers clipped)",
                transform=ax.transAxes,
                fontsize=6,
                ha="right",
                va="top",
                color="0.5",
                style="italic",
            )

        # Add diagonal reference
        lims = [0, max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, "--", color="gray", linewidth=0.8, alpha=0.6, zorder=0)

    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=PUBLICATION_DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote: {out_png}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate styled legacy plots")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # morans_i
    p2 = subparsers.add_parser("morans_i", help="Moran's I over time")
    p2.add_argument(
        "--cylinder_pkls",
        nargs="+",
        default=[
            "meshgraphnet/rollouts_200k_big/rollout_cylinder_auxiliary_200k.pkl",
            "meshgraphnet/rollouts_200k_big/rollout_cylinder_calibration_200k.pkl",
            "meshgraphnet/rollouts_200k_big/rollout_cylinder_test_200k.pkl",
        ],
    )
    p2.add_argument(
        "--flag_pkls",
        nargs="+",
        default=[
            "meshgraphnet/rollouts_200k_big/rollout_flag_auxiliary_200k.pkl",
            "meshgraphnet/rollouts_200k_big/rollout_flag_calibration_200k.pkl",
            "meshgraphnet/rollouts_200k_big/rollout_flag_test_200k.pkl",
        ],
    )
    p2.add_argument("--out_png", default="assets/morans_i_timeseries.png")

    # early_late
    p3 = subparsers.add_parser("early_late", help="Early vs late RMSE histograms")
    p3.add_argument(
        "--cylinder_pkls",
        nargs="+",
        default=[
            "meshgraphnet/rollouts_200k_big/rollout_cylinder_auxiliary_200k.pkl",
            "meshgraphnet/rollouts_200k_big/rollout_cylinder_calibration_200k.pkl",
            "meshgraphnet/rollouts_200k_big/rollout_cylinder_test_200k.pkl",
        ],
    )
    p3.add_argument(
        "--flag_pkls",
        nargs="+",
        default=[
            "meshgraphnet/rollouts_200k_big/rollout_flag_auxiliary_200k.pkl",
            "meshgraphnet/rollouts_200k_big/rollout_flag_calibration_200k.pkl",
            "meshgraphnet/rollouts_200k_big/rollout_flag_test_200k.pkl",
        ],
    )
    p3.add_argument("--out_png", default="assets/early_late_hist.png")

    # split_heatmap
    p4 = subparsers.add_parser("split_heatmap", help="Seed × split heatmap")
    p4.add_argument(
        "--csv",
        default="meshgraphnet/_artifacts/split_sensitivity_dense/split_sensitivity.csv",
    )
    p4.add_argument("--out_png", default="assets/split_heatmap.png")

    # final_vs_auc
    p5 = subparsers.add_parser("final_vs_auc", help="Final vs AUC scatter")
    p5.add_argument(
        "--csv",
        default="meshgraphnet/_artifacts/split_sensitivity_dense/split_sensitivity.csv",
    )
    p5.add_argument("--out_png", default="assets/final_vs_auc.png")

    # all
    p6 = subparsers.add_parser("all", help="Generate all styled legacy plots")

    args = parser.parse_args()

    if args.mode == "morans_i":
        fig_morans_i_timeseries(
            args.cylinder_pkls, args.flag_pkls, out_png=args.out_png
        )
    elif args.mode == "early_late":
        fig_early_late_hist(args.cylinder_pkls, args.flag_pkls, out_png=args.out_png)
    elif args.mode == "split_heatmap":
        fig_split_heatmap(args.csv, out_png=args.out_png)
    elif args.mode == "final_vs_auc":
        fig_final_vs_auc(args.csv, out_png=args.out_png)
    elif args.mode == "all":
        # Generate all
        fig_morans_i_timeseries(
            "meshgraphnet/rollouts_200k_big/rollout_cylinder_auxiliary_200k.pkl",
            "meshgraphnet/rollouts_200k_big/rollout_flag_auxiliary_200k.pkl",
        )
        fig_early_late_hist(
            "meshgraphnet/rollouts_200k_big/rollout_cylinder_auxiliary_200k.pkl",
            "meshgraphnet/rollouts_200k_big/rollout_flag_auxiliary_200k.pkl",
        )
        if Path(
            "meshgraphnet/_artifacts/split_sensitivity_dense/split_sensitivity.csv"
        ).exists():
            fig_split_heatmap(
                "meshgraphnet/_artifacts/split_sensitivity_dense/split_sensitivity.csv"
            )
            fig_final_vs_auc(
                "meshgraphnet/_artifacts/split_sensitivity_dense/split_sensitivity.csv"
            )
        else:
            print("Skipping split_heatmap and final_vs_auc (CSV not found)")


if __name__ == "__main__":
    main()
