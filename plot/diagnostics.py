#!/usr/bin/env python3
"""Diagnostic plots for meshgraphnet rollouts.

Plots:
  - fig_acf: Autocorrelation of RMSE(t) showing temporal dependence
  - fig_error_accumulation: RMSE over time grid
  - fig_batch_dependence: 1x3 boxplots of exchangeability diagnostics
  - fig_split_sensitivity: Violin plots of AUC RMSE across splits/seeds
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from plot.style import get_mpl, apply_paper_style, robust_vminmax, PUBLICATION_DPI


# -----------------------------------------------------------------------------
# Data loading helpers (from meshgraphnet.utils)
# -----------------------------------------------------------------------------


def _load_rollouts(pkl_path: Path) -> list:
    """Load rollouts from pickle file."""
    import pickle

    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _get_pred_gt(traj: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Extract prediction and ground truth from trajectory dict."""
    # Try velocity keys first (CylinderFlow), then position keys (Flag)
    for pred_key, gt_key in [
        ("pred_velocity", "gt_velocity"),
        ("pred|velocity", "gt|velocity"),
        ("pred_pos", "gt_pos"),
        ("pred|pos", "gt|pos"),
    ]:
        if pred_key in traj:
            return np.asarray(traj[pred_key]), np.asarray(traj[gt_key])
    raise KeyError(
        f"Cannot find pred/gt keys in trajectory. Available: {list(traj.keys())}"
    )


def _rmse_t(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """RMSE per timestep. pred/gt: (T, N, D) -> (T,)."""
    return np.sqrt(np.mean((pred - gt) ** 2, axis=(1, 2)))


def _autocorr(x: np.ndarray, lag: int) -> float:
    """Autocorrelation at given lag."""
    x = np.asarray(x, dtype=float).ravel()
    if lag <= 0 or lag >= len(x):
        return float("nan")
    x0 = x - x.mean()
    a, b = x0[:-lag], x0[lag:]
    denom = float(np.sqrt(np.sum(a**2) * np.sum(b**2)))
    return float(np.sum(a * b) / denom) if denom > 0 else float("nan")


def _edges_from_tris(tris: np.ndarray) -> np.ndarray:
    """Unique undirected edges from (F, 3) triangles."""
    tris = np.asarray(tris, dtype=np.int64)
    e = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e.sort(axis=1)
    return np.unique(e, axis=0).astype(np.int32)


def _morans_i(x: np.ndarray, edges: np.ndarray) -> float:
    """Moran's I for scalar field x over mesh edges."""
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 3:
        return float("nan")
    x0 = x - x.mean()
    denom = float(np.sum(x0**2))
    if denom == 0.0:
        return float("nan")
    i, j = edges[:, 0], edges[:, 1]
    w_sum = float(edges.shape[0] * 2)
    num = float(np.sum(x0[i] * x0[j]) + np.sum(x0[j] * x0[i]))
    return (n / w_sum) * (num / denom)


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample KS statistic (no scipy)."""
    a = np.sort(np.asarray(a, dtype=float).ravel())
    b = np.sort(np.asarray(b, dtype=float).ravel())
    if a.size == 0 or b.size == 0:
        return float("nan")
    data = np.unique(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, data, side="right") / a.size
    cdf_b = np.searchsorted(b, data, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


# -----------------------------------------------------------------------------
# Figure 7: ACF of RMSE(t) - Comparative
# -----------------------------------------------------------------------------


def _compute_acf_curves_from_pkls(
    rollout_pkls: List[Path], max_lag: int
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Compute ACF curves across all trajectories in multiple rollout files.

    Returns (lags, list_of_acf_arrays) where each array is one trajectory's ACF.
    """
    all_acfs = []
    min_len = max_lag + 1
    for pkl in rollout_pkls:
        rollouts = _load_rollouts(pkl)
        for traj in rollouts:
            pred, gt = _get_pred_gt(traj)
            rmse = _rmse_t(pred, gt)
            if len(rmse) < max_lag + 1:
                continue  # Skip short trajectories
            acf = np.array(
                [1.0] + [_autocorr(rmse, int(l)) for l in range(1, max_lag + 1)]
            )
            all_acfs.append(acf)
    lags = np.arange(0, max_lag + 1, dtype=int)
    return lags, all_acfs


def fig_acf_comparative(
    cylinder_pkls: List[Path],
    flag_pkls: List[Path],
    out_png: Path,
    max_lag: int = 50,
    *,
    dpi: int = PUBLICATION_DPI,
) -> None:
    """
    Create comparative ACF plot showing temporal dependence for both datasets.

    Shows median ACF across all trajectories with IQR bands (25-75%).
    Aggregates across ALL provided rollout files for maximum statistical power.
    Strong positive autocorrelation at small lags indicates that node-level
    errors are not exchangeable across time.
    """
    mpl, plt, *_ = get_mpl()
    apply_paper_style(dpi)

    # Compute ACF for both datasets (all trajectories from all pkls)
    lags_cyl, acfs_cyl = _compute_acf_curves_from_pkls(cylinder_pkls, max_lag)
    lags_flag, acfs_flag = _compute_acf_curves_from_pkls(flag_pkls, max_lag)

    # Stack into matrices
    mat_cyl = np.vstack(acfs_cyl)  # (n_traj, max_lag+1)
    mat_flag = np.vstack(acfs_flag)

    # Compute median and IQR (more robust than mean/std)
    med_cyl = np.nanmedian(mat_cyl, axis=0)
    q25_cyl = np.nanquantile(mat_cyl, 0.25, axis=0)
    q75_cyl = np.nanquantile(mat_cyl, 0.75, axis=0)

    med_flag = np.nanmedian(mat_flag, axis=0)
    q25_flag = np.nanquantile(mat_flag, 0.25, axis=0)
    q75_flag = np.nanquantile(mat_flag, 0.75, axis=0)

    # Colors
    color_cyl = "#1f77b4"  # Blue
    color_flag = "#d62728"  # Red

    # Plot - sized to pair with split_sensitivity side-by-side
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    # Cylinder - IQR band + median
    ax.fill_between(
        lags_cyl,
        q25_cyl,
        q75_cyl,
        alpha=0.25,
        color=color_cyl,
        linewidth=0,
    )
    ax.plot(
        lags_cyl,
        med_cyl,
        "o-",
        color=color_cyl,
        markersize=3,
        linewidth=1.2,
        label=f"CylinderFlow (n={len(acfs_cyl)})",
    )

    # Flag - IQR band + median
    ax.fill_between(
        lags_flag,
        q25_flag,
        q75_flag,
        alpha=0.25,
        color=color_flag,
        linewidth=0,
    )
    ax.plot(
        lags_flag,
        med_flag,
        "s-",
        color=color_flag,
        markersize=3,
        linewidth=1.2,
        label=f"Flag (n={len(acfs_flag)})",
    )

    # Reference line at 0
    ax.axhline(0.0, color="0.5", linewidth=0.6, linestyle="--", zorder=0)

    ax.set_xlabel("lag")
    ax.set_ylabel("ACF")
    ax.set_xlim(-1, max_lag + 1)
    ax.set_ylim(-0.15, 1.05)
    ax.legend(loc="lower left", fontsize=7, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote: {out_png}")


def fig_acf(
    rollout_pkl: Path,
    out_png: Path,
    traj_idx: int = 0,
    max_lag: int = 50,
    *,
    dpi: int = PUBLICATION_DPI,
) -> None:
    """
    Create single ACF plot showing temporal dependence in RMSE(t).

    For comparative plots, use fig_acf_comparative() instead.
    """
    mpl, plt, *_ = get_mpl()
    apply_paper_style(dpi)

    # Load data
    traj = _load_rollouts(rollout_pkl)[traj_idx]
    pred, gt = _get_pred_gt(traj)
    rmse = _rmse_t(pred, gt)

    # Compute ACF curve
    lags = np.arange(0, min(max_lag + 1, len(rmse)), dtype=int)
    acf = np.array([1.0] + [_autocorr(rmse, int(l)) for l in lags[1:]])

    # Plot
    fig, ax = plt.subplots(figsize=(3.3, 1.8))
    ax.plot(lags, acf, "o-", color="#1f77b4", markersize=4, linewidth=1.2)
    ax.axhline(0.0, color="0.4", linewidth=0.6, linestyle="--")
    ax.set_xlabel("lag")
    ax.set_ylabel("ACF")
    ax.set_xlim(-1, max_lag + 1)
    ax.set_ylim(-0.1, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote: {out_png}")


# -----------------------------------------------------------------------------
# Figure 9: Error Accumulation Grid (RMSE over time)
# -----------------------------------------------------------------------------


def fig_error_accumulation(
    rollout_pkls: List[Path],
    out_png: Path,
    labels: Optional[List[str]] = None,
    *,
    layout: str = "2x3",  # "2x3" for 2 datasets × 3 splits, or "1x3" for single row
    show_trajectories: bool = True,  # Overlay individual trajectory lines
    max_traj_lines: int = 50,  # Cap for performance
    dpi: int = PUBLICATION_DPI,
) -> None:
    """
    Create grid of RMSE-over-time plots with IQR bands.

    For CylinderFlow+Flag comparison:
      - Top row: CylinderFlow (auxiliary, calibration, test)
      - Bottom row: Flag (auxiliary, calibration, test)

    Each panel aggregates all trajectories: mean RMSE (black) + IQR band (blue).
    Optionally overlays individual trajectory lines (faint) for richness.
    """
    mpl, plt, *_ = get_mpl()
    apply_paper_style(dpi)

    n_plots = len(rollout_pkls)
    if layout == "2x3":
        nrows, ncols = 2, 3
        col_headers = ["Auxiliary", "Calibration", "Eval"]
        row_labels = ["CylinderFlow", "Flag"]
    elif layout == "1x3":
        nrows, ncols = 1, 3
        col_headers = ["Auxiliary", "Calibration", "Eval"]
        row_labels = None
    else:
        nrows, ncols = 1, n_plots
        col_headers = None
        row_labels = None

    # Figure sizing - add space for headers
    panel_w, panel_h = 2.0, 1.5
    fig_w = ncols * panel_w + 0.5
    fig_h = nrows * panel_h + 0.4

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharex=False)
    if nrows == 1:
        axes = axes.reshape(1, -1)

    # First pass: compute all data and find per-row y-limits (clip outliers)
    all_data = []
    for pkl in rollout_pkls:
        rollouts = _load_rollouts(pkl)
        rmse_list = []
        for traj in rollouts:
            pred, gt = _get_pred_gt(traj)
            rmse_list.append(_rmse_t(pred, gt))
        max_T = max(len(r) for r in rmse_list)
        mat = np.full((len(rmse_list), max_T), np.nan)
        for i, r in enumerate(rmse_list):
            mat[i, : len(r)] = r
        all_data.append(mat)

    # Compute per-row y-limits (99th percentile to avoid outlier compression)
    row_ylims = {}
    for idx, mat in enumerate(all_data):
        row = idx // ncols
        ymax = np.nanpercentile(mat, 99)
        if row not in row_ylims:
            row_ylims[row] = ymax
        else:
            row_ylims[row] = max(row_ylims[row], ymax)

    for idx, (pkl, mat) in enumerate(zip(rollout_pkls, all_data)):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        mean = np.nanmean(mat, axis=0)
        q25 = np.nanquantile(mat, 0.25, axis=0)
        q75 = np.nanquantile(mat, 0.75, axis=0)
        ts = np.arange(mat.shape[1])

        # Overlay individual trajectories (faint lines) for richness
        if show_trajectories:
            n_traj = mat.shape[0]
            step = max(1, n_traj // max_traj_lines)
            for i in range(0, n_traj, step):
                ax.plot(
                    ts, mat[i], color="#1f77b4", alpha=0.08, linewidth=0.5, zorder=1
                )

        # IQR band (on top of trajectories)
        ax.fill_between(ts, q25, q75, alpha=0.3, color="#1f77b4", linewidth=0, zorder=2)
        ax.plot(ts, mean, color="black", linewidth=1.2, zorder=3)

        # Set consistent y-limits per row (clipped at 99th percentile)
        ax.set_ylim(0, row_ylims[row] * 1.05)

        # Column headers (top row only)
        if row == 0 and col_headers:
            ax.set_title(col_headers[col], fontsize=9, fontweight="medium")

        # Styling
        if row == nrows - 1:
            ax.set_xlabel("timestep", fontsize=8)
        # Row-specific y-label with dataset name
        if col == 0 and row_labels:
            ax.set_ylabel(f"{row_labels[row]}\nRMSE", fontsize=9)
        elif col == 0:
            ax.set_ylabel("RMSE", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Legend (only on first)
        if idx == 0:
            ax.fill_between(
                [], [], [], alpha=0.3, color="#1f77b4", label="IQR (25–75%)"
            )
            ax.plot([], [], color="black", linewidth=1.2, label="mean RMSE")
            ax.legend(loc="upper left", fontsize=7, framealpha=0.9)

    # Hide unused axes
    for idx in range(n_plots, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    plt.tight_layout(pad=0.3, h_pad=0.4, w_pad=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote: {out_png}")


# -----------------------------------------------------------------------------
# Figure 14: Batch Dependence Diagnostics (1x3 boxplots)
# -----------------------------------------------------------------------------


def fig_batch_dependence(
    rollout_dirs: List[Path],
    out_png: Path,
    *,
    acf_lag: int = 1,
    morans_stride: int = 2,
    dpi: int = PUBLICATION_DPI,
) -> None:
    """
    Create 1x3 boxplot grid showing exchangeability diagnostics.

    Panels:
      - Left: ACF(lag=1) for RMSE(t) - temporal dependence
      - Middle: KS statistic (early vs late RMSE distribution) - drift
      - Right: Mean Moran's I of |r| over time - spatial dependence
    """
    mpl, plt, *_ = get_mpl()
    apply_paper_style(dpi)

    # Collect diagnostics per group
    acf_by_group: Dict[str, List[float]] = {}
    ks_by_group: Dict[str, List[float]] = {}
    morans_by_group: Dict[str, List[float]] = {}

    RE_BASE = re.compile(
        r"rollout_(?P<dataset>cylinder|flag)_(?P<split>auxiliary|calibration|test)"
    )

    for rollout_dir in rollout_dirs:
        for pkl in sorted(Path(rollout_dir).glob("rollout_*.pkl")):
            m = RE_BASE.search(pkl.stem)
            if not m:
                continue
            group = f"{m.group('dataset')}/{m.group('split')}"

            rollouts = _load_rollouts(pkl)
            for ti, traj in enumerate(rollouts):
                pred, gt = _get_pred_gt(traj)
                rmse = _rmse_t(pred, gt)
                T = len(rmse)

                # ACF
                acf1 = _autocorr(rmse, acf_lag)
                acf_by_group.setdefault(group, []).append(acf1)

                # KS (early vs late)
                third = max(1, T // 3)
                ks = _ks_statistic(rmse[:third], rmse[-third:])
                ks_by_group.setdefault(group, []).append(ks)

                # Moran's I (mean over time)
                err_mag = np.linalg.norm(gt - pred, axis=-1)
                tris = np.asarray(traj["faces"])[0].astype(np.int32)
                edges = _edges_from_tris(tris)
                morans_vals = [
                    _morans_i(err_mag[t], edges) for t in range(0, T, morans_stride)
                ]
                morans_by_group.setdefault(group, []).append(np.nanmean(morans_vals))

    if not acf_by_group:
        print("No matching rollouts found!")
        return

    # Organize by dataset and split for grouped display
    datasets = ["cylinder", "flag"]
    splits = ["auxiliary", "calibration", "test"]
    split_labels = ["Aux", "Cal", "Eval"]

    # Colors for datasets
    colors = {"cylinder": "#1f77b4", "flag": "#d62728"}  # Blue, Red
    dataset_labels = {"cylinder": "CylinderFlow", "flag": "Flag"}

    # Create 1x3 figure
    fig, axes = plt.subplots(1, 3, figsize=(6.0, 2.4))

    metrics = [
        (acf_by_group, f"ACF(lag={acf_lag})"),
        (ks_by_group, "KS statistic"),
        (morans_by_group, "Moran's I"),
    ]

    rng = np.random.default_rng(42)
    bar_width = 0.35

    for ax, (data_dict, ylabel) in zip(axes, metrics):
        x = np.arange(len(splits))

        for i, dataset in enumerate(datasets):
            offset = (i - 0.5) * bar_width
            data = [data_dict.get(f"{dataset}/{split}", []) for split in splits]

            # Calculate statistics for bars
            means = [np.mean(d) if d else 0 for d in data]
            stds = [np.std(d) if d else 0 for d in data]

            # Plot bars
            bars = ax.bar(
                x + offset,
                means,
                bar_width * 0.9,
                label=dataset_labels[dataset],
                color=colors[dataset],
                alpha=0.7,
                edgecolor=colors[dataset],
                linewidth=1.0,
            )

            # Add error bars (std)
            ax.errorbar(
                x + offset,
                means,
                yerr=stds,
                fmt="none",
                color="0.3",
                capsize=2,
                capthick=0.8,
                linewidth=0.8,
            )

            # Overlay scatter points with jitter
            for j, vals in enumerate(data):
                if vals:
                    xs = (
                        x[j]
                        + offset
                        + rng.uniform(-bar_width * 0.3, bar_width * 0.3, size=len(vals))
                    )
                    ax.scatter(
                        xs, vals, s=6, alpha=0.4, color="0.2", linewidths=0, zorder=3
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(split_labels, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=7)

    # Add legend to first panel only
    axes[0].legend(loc="lower left", fontsize=7, framealpha=0.9)

    plt.tight_layout(pad=0.4, w_pad=0.8)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote: {out_png}")


# -----------------------------------------------------------------------------
# Figure 15: Split Sensitivity Violin Plots
# -----------------------------------------------------------------------------


def fig_split_sensitivity(
    rollouts_dir: Path,
    out_png: Path,
    *,
    dpi: int = PUBLICATION_DPI,
) -> None:
    """
    Create 1x2 violin plot showing AUC RMSE across splits for Cylinder and Flag.

    Left: CylinderFlow, Right: Flag

    Each violin shows distribution of mean_t RMSE(t) across trajectories and seeds.
    """
    mpl, plt, *_ = get_mpl()
    apply_paper_style(dpi)

    RE_SENS = re.compile(
        r"rollout_(?P<dataset>cylinder|flag)_(?P<split>auxiliary|calibration|test)_seed(?P<seed>\d+)\.pkl$"
    )

    # Collect data
    data: Dict[str, Dict[str, List[float]]] = {
        "cylinder": {"auxiliary": [], "calibration": [], "test": []},
        "flag": {"auxiliary": [], "calibration": [], "test": []},
    }

    for pkl in sorted(Path(rollouts_dir).glob("rollout_*_seed*.pkl")):
        m = RE_SENS.match(pkl.name)
        if not m:
            continue
        dataset = m.group("dataset")
        split = m.group("split")

        rollouts = _load_rollouts(pkl)
        for traj in rollouts:
            pred, gt = _get_pred_gt(traj)
            auc = float(np.mean(_rmse_t(pred, gt)))
            data[dataset][split].append(auc)

    splits = ["auxiliary", "calibration", "test"]
    dataset_titles = {"cylinder": "CylinderFlow", "flag": "Flag"}

    # Create 1x2 figure (side by side)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.2))

    rng = np.random.default_rng(42)
    for idx, (ax, dataset) in enumerate(zip(axes, ["cylinder", "flag"])):
        ys_list = [data[dataset][s] for s in splits]

        if all(len(y) > 0 for y in ys_list):
            # For Flag, clip to 98.9th percentile for better visualization
            all_vals = np.concatenate(ys_list)
            if dataset == "flag":
                clip_val = np.percentile(all_vals, 98.9)
                ys_list_clipped = [np.clip(y, 0, clip_val) for y in ys_list]
                n_clipped = sum(v > clip_val for y in ys_list for v in y)
            else:
                ys_list_clipped = ys_list
                clip_val = None

            parts = ax.violinplot(ys_list_clipped, showmeans=True, showextrema=False)

            # Style violins
            for pc in parts.get("bodies", []):
                pc.set_alpha(0.4)
                pc.set_facecolor("0.6")
                pc.set_edgecolor("0.3")
                pc.set_linewidth(0.8)
            if "cmeans" in parts:
                parts["cmeans"].set_color("0.15")
                parts["cmeans"].set_linewidth(1.2)

            # Overlay scatter points (use original values, clip for display)
            for i, ys in enumerate(ys_list, start=1):
                ys_plot = np.array(ys)
                if clip_val:
                    ys_plot = np.clip(ys_plot, 0, clip_val)
                xs = i + rng.uniform(-0.12, 0.12, size=len(ys_plot))
                ax.scatter(
                    xs, ys_plot, s=10, alpha=0.45, color="0.25", linewidths=0, zorder=3
                )

        ax.set_title(dataset_titles[dataset], fontsize=9, fontweight="medium")
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["Aux", "Cal", "Eval"], fontsize=8)
        if idx == 0:
            ax.set_ylabel(r"AUC RMSE", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add note about clipping for Flag
        if dataset == "flag" and clip_val:
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

    plt.tight_layout(pad=0.3, w_pad=0.8)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote: {out_png}")


# -----------------------------------------------------------------------------
# Figure 8: Normalized Residuals over Time
# -----------------------------------------------------------------------------


def fig_normalized_residuals(
    cylinder_csv: Path,
    flag_csv: Path,
    out_png: Path,
    *,
    dpi: int = PUBLICATION_DPI,
) -> None:
    """
    Create 2-row plot of normalized residuals z = ||r||_2 / sigma(x) over time.

    Top: CylinderFlow, Bottom: Flag
    Shows quantile bands (10-90%, 25-75%) and median line.
    """
    mpl, plt, *_ = get_mpl()
    apply_paper_style(dpi)

    import pandas as pd

    fig, axes = plt.subplots(2, 1, figsize=(3.8, 2.8), sharex=False)

    datasets = [
        ("CylinderFlow", cylinder_csv, axes[0]),
        ("Flag", flag_csv, axes[1]),
    ]

    # Colors for quantile bands
    colors = {
        "q10": "#9467bd",  # purple
        "q25": "#ff7f0e",  # orange
        "q50": "#2ca02c",  # green (median)
        "q75": "#d62728",  # red
        "q90": "#7f7f7f",  # gray
    }

    for name, csv_path, ax in datasets:
        df = pd.read_csv(csv_path)
        t = df["t"].values

        # Skip first few timesteps if they have extreme values
        # (initialization artifacts)
        start_idx = 2 if df["mean"].iloc[1] > 100 else 0
        t = t[start_idx:]

        # Plot quantile bands
        ax.fill_between(
            t,
            df["q10"].values[start_idx:],
            df["q90"].values[start_idx:],
            alpha=0.15,
            color="#1f77b4",
            linewidth=0,
            label="10-90%",
        )
        ax.fill_between(
            t,
            df["q25"].values[start_idx:],
            df["q75"].values[start_idx:],
            alpha=0.25,
            color="#1f77b4",
            linewidth=0,
            label="25-75%",
        )

        # Plot median and mean
        ax.plot(
            t,
            df["q50"].values[start_idx:],
            color="#2ca02c",
            linewidth=1.2,
            label="median",
        )
        ax.plot(
            t,
            df["mean"].values[start_idx:],
            color="black",
            linewidth=1.2,
            linestyle="--",
            label="mean",
        )

        # Reference line at z=1 (ideal calibration)
        ax.axhline(1.0, color="0.6", linewidth=0.8, linestyle=":", zorder=0, alpha=0.7)

        ax.set_ylabel(r"$z = \|r\|_2 / \sigma(x)$", fontsize=9)
        ax.set_title(name, fontsize=9, fontweight="medium")
        ax.set_ylim(bottom=0)  # Start at 0
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Use fewer integer ticks for x-axis (compact figure)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5, integer=True))

        if name == "CylinderFlow":
            ax.legend(loc="upper right", fontsize=7, framealpha=0.9)

    axes[1].set_xlabel("timestep", fontsize=9)

    plt.tight_layout(pad=0.3, h_pad=0.5)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote: {out_png}")


# -----------------------------------------------------------------------------
# Figure: Combined Temporal Diagnostics (ACF + Normalized Residuals)
# -----------------------------------------------------------------------------


def fig_temporal_diagnostics(
    cylinder_pkls: List[Path],
    flag_pkls: List[Path],
    cylinder_resid_csv: Path,
    flag_resid_csv: Path,
    out_png: Path,
    *,
    max_lag: int = 50,
    dpi: int = PUBLICATION_DPI,
) -> None:
    """
    Create a 1x2 combined figure:
    - Left: ACF of RMSE(t) for both datasets
    - Right: Normalized residuals z = ||r||_2 / sigma(x) over time (stacked)
    """
    mpl, plt, *_ = get_mpl()
    apply_paper_style(dpi)
    import pandas as pd

    fig = plt.figure(figsize=(7.0, 2.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.28)

    # === Left panel: ACF comparative ===
    ax_acf = fig.add_subplot(gs[0])

    # Use existing ACF computation
    lags_cyl, acfs_cyl = _compute_acf_curves_from_pkls(cylinder_pkls, max_lag)
    lags_flag, acfs_flag = _compute_acf_curves_from_pkls(flag_pkls, max_lag)

    mat_cyl = np.vstack(acfs_cyl)
    mat_flag = np.vstack(acfs_flag)

    med_cyl = np.nanmedian(mat_cyl, axis=0)
    q25_cyl = np.nanquantile(mat_cyl, 0.25, axis=0)
    q75_cyl = np.nanquantile(mat_cyl, 0.75, axis=0)

    med_flag = np.nanmedian(mat_flag, axis=0)
    q25_flag = np.nanquantile(mat_flag, 0.25, axis=0)
    q75_flag = np.nanquantile(mat_flag, 0.75, axis=0)

    color_cyl, color_flag = "#1f77b4", "#d62728"

    ax_acf.fill_between(
        lags_cyl, q25_cyl, q75_cyl, alpha=0.2, color=color_cyl, linewidth=0
    )
    ax_acf.plot(
        lags_cyl,
        med_cyl,
        color=color_cyl,
        linewidth=1.5,
        marker="o",
        markersize=3,
        markevery=5,
        label=f"CylinderFlow (n={len(acfs_cyl)})",
    )

    ax_acf.fill_between(
        lags_flag, q25_flag, q75_flag, alpha=0.2, color=color_flag, linewidth=0
    )
    ax_acf.plot(
        lags_flag,
        med_flag,
        color=color_flag,
        linewidth=1.5,
        marker="s",
        markersize=3,
        markevery=5,
        label=f"Flag (n={len(acfs_flag)})",
    )

    ax_acf.axhline(0, color="0.5", linewidth=0.8, linestyle="--", zorder=0)
    ax_acf.set_xlabel("lag", fontsize=9)
    ax_acf.set_ylabel("ACF", fontsize=9)
    ax_acf.set_xlim(0, max_lag)
    ax_acf.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax_acf.spines["top"].set_visible(False)
    ax_acf.spines["right"].set_visible(False)
    ax_acf.set_title("(a) Autocorrelation of RMSE(t)", fontsize=9, fontweight="medium")

    # === Right panel: Normalized residuals (stacked) ===
    gs_right = gs[1].subgridspec(2, 1, hspace=0.35)
    ax_cyl = fig.add_subplot(gs_right[0])
    ax_flag = fig.add_subplot(gs_right[1])

    residual_datasets = [
        ("CylinderFlow", cylinder_resid_csv, ax_cyl),
        ("Flag", flag_resid_csv, ax_flag),
    ]

    for name, csv_path, ax in residual_datasets:
        df = pd.read_csv(csv_path)
        t = df["t"].values
        start_idx = 2 if df["mean"].iloc[1] > 100 else 0
        t = t[start_idx:]

        ax.fill_between(
            t,
            df["q10"].values[start_idx:],
            df["q90"].values[start_idx:],
            alpha=0.15,
            color="#1f77b4",
            linewidth=0,
        )
        ax.fill_between(
            t,
            df["q25"].values[start_idx:],
            df["q75"].values[start_idx:],
            alpha=0.25,
            color="#1f77b4",
            linewidth=0,
        )
        ax.plot(
            t,
            df["q50"].values[start_idx:],
            color="#2ca02c",
            linewidth=1.0,
            label="median",
        )
        ax.plot(
            t,
            df["mean"].values[start_idx:],
            color="black",
            linewidth=1.0,
            linestyle="--",
            label="mean",
        )
        ax.axhline(1.0, color="0.6", linewidth=0.8, linestyle=":", zorder=0, alpha=0.7)

        ax.set_ylabel(r"$z$", fontsize=8)
        ax.set_ylim(bottom=0, top=2.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5, integer=True))
        ax.tick_params(labelsize=7)

        # Dataset label
        ax.text(
            0.98,
            0.92,
            name,
            transform=ax.transAxes,
            fontsize=7,
            ha="right",
            va="top",
            fontweight="medium",
        )

    ax_flag.set_xlabel("timestep", fontsize=9)
    ax_cyl.set_title(
        "(b) Normalized residuals over time", fontsize=9, fontweight="medium"
    )
    ax_cyl.legend(loc="upper left", fontsize=6, framealpha=0.9, ncol=2)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.92)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote: {out_png}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate diagnostic figures")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ACF (single dataset)
    p_acf = sub.add_parser("acf", help="ACF of RMSE(t) for single dataset")
    p_acf.add_argument("--rollout_pkl", required=True)
    p_acf.add_argument("--out_png", default="paper/figures_generated/acf.png")
    p_acf.add_argument("--traj_idx", type=int, default=0)
    p_acf.add_argument("--max_lag", type=int, default=50)

    # ACF comparative (both datasets, multiple pkls each)
    p_acf_cmp = sub.add_parser(
        "acf_comparative", help="ACF comparison of Cylinder vs Flag (all trajectories)"
    )
    p_acf_cmp.add_argument(
        "--cylinder_pkls",
        nargs="+",
        required=True,
        help="All cylinder rollout pkl files",
    )
    p_acf_cmp.add_argument(
        "--flag_pkls", nargs="+", required=True, help="All flag rollout pkl files"
    )
    p_acf_cmp.add_argument(
        "--out_png", default="paper/figures_generated/acf_comparative.png"
    )
    p_acf_cmp.add_argument("--max_lag", type=int, default=50)

    # Error accumulation
    p_err = sub.add_parser("error_accumulation", help="RMSE over time grid")
    p_err.add_argument("--rollout_pkls", nargs="+", required=True)
    p_err.add_argument(
        "--out_png", default="paper/figures_generated/error_accumulation.png"
    )
    p_err.add_argument("--layout", default="2x3", choices=["2x3", "1x3", "1xN"])

    # Batch dependence
    p_batch = sub.add_parser("batch_dependence", help="Exchangeability boxplots")
    p_batch.add_argument("--rollout_dirs", nargs="+", required=True)
    p_batch.add_argument(
        "--out_png", default="paper/figures_generated/batch_dependence.png"
    )

    # Split sensitivity
    p_split = sub.add_parser(
        "split_sensitivity", help="Violin plots of split sensitivity"
    )
    p_split.add_argument("--rollouts_dir", required=True)
    p_split.add_argument(
        "--out_png", default="paper/figures_generated/split_sensitivity.png"
    )

    # Normalized residuals
    p_resid = sub.add_parser(
        "normalized_residuals", help="Normalized residuals z = ||r||/sigma over time"
    )
    p_resid.add_argument(
        "--cylinder_csv",
        required=True,
        help="CSV from conformal/_artifacts/normalized_residuals/cylinder_eval/",
    )
    p_resid.add_argument(
        "--flag_csv",
        required=True,
        help="CSV from conformal/_artifacts/normalized_residuals/flag_eval/",
    )
    p_resid.add_argument(
        "--out_png", default="paper/figures_generated/normalized_residuals.png"
    )

    # Combined temporal diagnostics (ACF + normalized residuals)
    p_temporal = sub.add_parser(
        "temporal_diagnostics", help="Combined ACF + normalized residuals figure"
    )
    p_temporal.add_argument(
        "--cylinder_pkls", nargs="+", required=True, help="Cylinder rollout pkl files"
    )
    p_temporal.add_argument(
        "--flag_pkls", nargs="+", required=True, help="Flag rollout pkl files"
    )
    p_temporal.add_argument(
        "--cylinder_resid_csv", required=True, help="Cylinder normalized_residuals.csv"
    )
    p_temporal.add_argument(
        "--flag_resid_csv", required=True, help="Flag normalized_residuals.csv"
    )
    p_temporal.add_argument(
        "--out_png", default="paper/figures_generated/temporal_diagnostics.png"
    )
    p_temporal.add_argument("--max_lag", type=int, default=50)

    args = ap.parse_args()

    if args.cmd == "acf":
        fig_acf(
            Path(args.rollout_pkl),
            Path(args.out_png),
            traj_idx=args.traj_idx,
            max_lag=args.max_lag,
        )
    elif args.cmd == "acf_comparative":
        fig_acf_comparative(
            [Path(p) for p in args.cylinder_pkls],
            [Path(p) for p in args.flag_pkls],
            Path(args.out_png),
            max_lag=args.max_lag,
        )
    elif args.cmd == "error_accumulation":
        fig_error_accumulation(
            [Path(p) for p in args.rollout_pkls],
            Path(args.out_png),
            layout=args.layout,
        )
    elif args.cmd == "batch_dependence":
        fig_batch_dependence(
            [Path(d) for d in args.rollout_dirs],
            Path(args.out_png),
        )
    elif args.cmd == "split_sensitivity":
        fig_split_sensitivity(
            Path(args.rollouts_dir),
            Path(args.out_png),
        )
    elif args.cmd == "normalized_residuals":
        fig_normalized_residuals(
            Path(args.cylinder_csv),
            Path(args.flag_csv),
            Path(args.out_png),
        )
    elif args.cmd == "temporal_diagnostics":
        fig_temporal_diagnostics(
            [Path(p) for p in args.cylinder_pkls],
            [Path(p) for p in args.flag_pkls],
            Path(args.cylinder_resid_csv),
            Path(args.flag_resid_csv),
            Path(args.out_png),
            max_lag=args.max_lag,
        )


if __name__ == "__main__":
    main()
