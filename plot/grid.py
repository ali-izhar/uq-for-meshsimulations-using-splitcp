#!/usr/bin/env python3
"""Creates a tight grid showing Truth/Pred/Error rows for two timesteps (columns)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from plot.style import (
    get_mpl,
    apply_paper_style,
    robust_vminmax,
    CMAP_VELOCITY,
    CMAP_ERROR,
    PUBLICATION_DPI,
)


def _load_traj(pkl: Path, idx: int = 0) -> dict:
    """Load trajectory from rollout pickle."""
    from conformal.io import load_rollouts

    rollouts = load_rollouts(str(pkl))
    if not rollouts:
        raise ValueError(f"No trajectories in {pkl}")
    return rollouts[idx]


def _triangulation(traj: dict, t: int = 0):
    """Build triangulation from trajectory."""
    _, _, mtri, *_ = get_mpl()
    faces = np.asarray(traj["faces"])
    mesh_pos = np.asarray(traj["mesh_pos"])
    tris = faces[min(t, len(faces) - 1)].astype(np.int32)
    pos = mesh_pos[min(t, len(mesh_pos) - 1)].astype(np.float64)
    return mtri.Triangulation(pos[:, 0], pos[:, 1], tris), pos


def _get_arrays(traj: dict):
    """Get pred and gt arrays from trajectory."""
    from conformal.io import infer_pred_gt_keys

    pk, gk = infer_pred_gt_keys(traj)
    pred = np.asarray(traj[pk], dtype=np.float32)
    gt = np.asarray(traj[gk], dtype=np.float32)
    return pred, gt


def _boundary_nodes(tris: np.ndarray) -> np.ndarray:
    """Find boundary nodes (edges appearing exactly once)."""
    edge_count = defaultdict(int)
    for a, b, c in tris:
        for u, v in ((a, b), (b, c), (c, a)):
            edge_count[(min(u, v), max(u, v))] += 1
    boundary = set()
    for (u, v), cnt in edge_count.items():
        if cnt == 1:
            boundary.update([u, v])
    return np.array(sorted(boundary), dtype=np.int32)


def _load_conformal(conformal_out: Path):
    """Load conformal prediction outputs including official coverage from test set."""
    from conformal.models import SigmaModel, SigmaPredictor, ComponentwiseSigmaPredictor

    thresholds = json.loads((conformal_out / "thresholds.json").read_text())
    sigma = SigmaModel.load(conformal_out)
    sigma_pred = SigmaPredictor.load(conformal_out / "sigma_model.pkl")

    # Load componentwise sigma predictor if available
    cw_sigma_pred = None
    cw_path = conformal_out / "componentwise_sigma_model.pkl"
    if cw_path.exists():
        cw_sigma_pred = ComponentwiseSigmaPredictor.load(cw_path)

    # Load official coverage values from summary.json (evaluated on test set)
    summary_path = conformal_out / "summary.json"
    coverage = None
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        coverage = summary.get("coverage", None)

    return thresholds, sigma, sigma_pred, cw_sigma_pred, coverage


def _read_feature_set(conformal_out: Path) -> str:
    """Read feature_set used for sigma(x)."""
    for f in ["features_meta.json", "summary.json"]:
        p = conformal_out / f
        if p.exists():
            try:
                return str(json.loads(p.read_text()).get("feature_set", "default"))
            except Exception:
                pass
    return "default"


def _conformal_radii(
    traj, thresholds, sigma, sigma_pred, cw_sigma_pred, alpha: float, feature_set: str
):
    """
    Compute conformal prediction effective radii for all methods.

    Returns effective radii in the same physical units (m/s or m) for fair comparison:

    - l2: constant = q_l2 (L2 norm quantile, same everywhere)
    - mah: constant = q_mah × det(Σ)^(1/2D) (Mahalanobis converted to volume-equivalent L2)
    - adapt: varying = q_adapt × σ(x) (scaled by learned local uncertainty)
    - cw_adapt: varying = q_cw × (∏ᵢ σᵢ(x))^(1/D) (geometric mean of per-component scales)

    The raw quantiles (q_l2, q_mah, q_adapt, q_cw) have different units/meanings:
    - q_l2: in physical units (m/s or m)
    - q_mah: dimensionless Mahalanobis distance
    - q_adapt: dimensionless scaling factor
    - q_cw: dimensionless scaling factor

    By computing effective radii, we convert all to the same physical units.
    """
    from conformal.io import infer_pred_gt_keys
    from conformal.features import build_features, default_time_index

    pk, _ = infer_pred_gt_keys(traj)
    pred = np.asarray(traj[pk]).astype(np.float32)
    mesh_pos = np.asarray(traj["mesh_pos"]).astype(np.float32)
    T, N, D = pred.shape

    a_str = str(alpha)
    q_l2 = float(thresholds["l2"][a_str])
    q_mah = float(thresholds["mah"][a_str])
    q_adapt = float(thresholds["adapt"][a_str])

    # Convert Mahalanobis quantile to volume-equivalent L2 radius
    det_sigma = float(np.linalg.det(sigma.Sigma))
    rad_mah_eff = float(q_mah * (det_sigma ** (1.0 / (2.0 * D))))

    # Build features and predict local uncertainty σ(x)
    faces = traj.get("faces", None)
    X = build_features(
        mesh_pos=mesh_pos,
        pred_vec=pred,
        t_index=default_time_index(T),
        faces=faces,
        feature_set=feature_set,
    )
    sigma_x = sigma_pred.predict_sigma(X).reshape(T, N).astype(np.float32)

    result = {
        "l2": np.full((T, N), q_l2, dtype=np.float32),
        "mah": np.full((T, N), rad_mah_eff, dtype=np.float32),
        "adapt": (q_adapt * sigma_x).astype(np.float32),
        "q_l2": q_l2,
        "q_mah": q_mah,
        "q_adapt": q_adapt,
    }

    # Componentwise adaptive if available
    if cw_sigma_pred is not None and "cw_adapt" in thresholds:
        q_cw = float(thresholds["cw_adapt"][a_str])
        # Get per-component sigma: (T*N, D) -> reshape to (T, N, D)
        sigma_vec = cw_sigma_pred.predict_sigma_vec(X).reshape(T, N, D).astype(np.float32)
        # Effective width = q × geometric_mean(σᵢ) = q × (∏ᵢ σᵢ)^(1/D)
        geom_mean = np.prod(sigma_vec, axis=-1) ** (1.0 / D)
        result["cw_adapt"] = (q_cw * geom_mean).astype(np.float32)
        result["q_cw"] = q_cw

    return result


def fig_temporal_grid(
    rollout_pkl: Path,
    out_png: Path,
    step_a: int = 50,
    step_b: int = 140,
    *,
    vel_cmap: str = CMAP_VELOCITY,
    err_cmap: str = CMAP_ERROR,
    dpi: int = PUBLICATION_DPI,
) -> None:
    """
    Create 3x2 grid: rows=Truth/Pred/Error, cols=step_a/step_b.

    Features:
    - Zero spacing between panels
    - Shared colorbars (velocity for Truth/Pred, diverging for Error)
    - Tight bounding box for exact dimensions
    """
    _, plt, _, _, GridSpec, TwoSlopeNorm, MaxNLocator = get_mpl()
    apply_paper_style(dpi)

    # Load data
    traj = _load_traj(rollout_pkl)
    tri, pos = _triangulation(traj)
    pred, gt = _get_arrays(traj)
    T = gt.shape[0]

    # Clip steps
    ta = min(max(step_a, 0), T - 1)
    tb = min(max(step_b, 0), T - 1)

    # Extract x-component of velocity
    def xcomp(arr, t):
        return arr[t, :, 0]

    # Compute consistent color limits
    vel_vals = np.concatenate(
        [xcomp(gt, ta), xcomp(gt, tb), xcomp(pred, ta), xcomp(pred, tb)]
    )
    vel_vmin, vel_vmax = robust_vminmax(vel_vals)

    err_vals = np.concatenate([xcomp(gt - pred, ta), xcomp(gt - pred, tb)])
    err_max = np.quantile(np.abs(err_vals[np.isfinite(err_vals)]), 0.98)

    # Domain limits
    xmin, xmax = pos[:, 0].min(), pos[:, 0].max()
    ymin, ymax = pos[:, 1].min(), pos[:, 1].max()
    data_aspect = (ymax - ymin) / max(xmax - xmin, 1e-9)

    # Figure size calculated to match data aspect exactly
    # 2 panel columns + colorbar, with exact proportions
    cbar_frac = 0.06  # Colorbar width as fraction of one panel
    panel_width = 3.0  # inches per panel
    total_panel_width = 2 * panel_width  # Two columns touching
    cbar_width = panel_width * cbar_frac

    # Add margins for external labels
    left_margin = 0.35  # inches for row labels
    top_margin = 0.25  # inches for column headers
    fig_width = left_margin + total_panel_width + cbar_width + 0.3

    panel_height = panel_width * data_aspect
    row_gap = 0.08  # Small gap between rows in inches
    fig_height = 3 * panel_height + 2 * row_gap + top_margin

    # Create figure
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")

    # Panel positions in figure coordinates [0, 1]
    left = left_margin / fig_width
    right = (left_margin + total_panel_width) / fig_width
    bottom = 0.0
    top = (fig_height - top_margin) / fig_height

    # GridSpec for the 2 panel columns (colorbars added separately)
    gap_frac = row_gap / panel_height * 0.33  # Small gap as fraction
    gs = GridSpec(
        3,
        2,
        figure=fig,
        width_ratios=[1, 1],
        height_ratios=[1, 1, 1],
        wspace=gap_frac,  # Match vertical gap
        hspace=gap_frac,  # Small vertical gap
        left=left,
        right=right,
        bottom=bottom,
        top=top,
    )

    # Panel data: [row][(data, label, step), ...]
    panels = [
        [(xcomp(gt, ta), "Truth", ta), (xcomp(gt, tb), "Truth", tb)],
        [(xcomp(pred, ta), "Pred", ta), (xcomp(pred, tb), "Pred", tb)],
        [(xcomp(gt - pred, ta), "Error", ta), (xcomp(gt - pred, tb), "Error", tb)],
    ]

    # Store mappables for colorbars
    vel_mappable = None
    err_mappable = None

    for row in range(3):
        for col in range(2):
            vals, label, step = panels[row][col]
            ax = fig.add_subplot(gs[row, col])

            # Clean axes - aspect handled by figure sizing
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_axis_off()
            ax.set_facecolor("white")

            # Plot
            if label == "Error":
                norm = TwoSlopeNorm(vcenter=0, vmin=-err_max, vmax=err_max)
                tpc = ax.tripcolor(
                    tri, vals, shading="gouraud", cmap=err_cmap, norm=norm
                )
                err_mappable = tpc
            else:
                tpc = ax.tripcolor(
                    tri,
                    vals,
                    shading="gouraud",
                    cmap=vel_cmap,
                    vmin=vel_vmin,
                    vmax=vel_vmax,
                )
                vel_mappable = tpc

            # Subtle mesh overlay
            ax.triplot(tri, color=(0, 0, 0, 0.12), linewidth=0.2)

            # Column headers (top row only) - outside the mesh
            if row == 0:
                ax.set_title(f"$t = {step}$", fontsize=10, pad=4, family="serif")

            # Row labels (left column only) - outside the mesh
            if col == 0:
                ax.text(
                    -0.08,
                    0.5,
                    label,
                    ha="right",
                    va="center",
                    rotation=90,
                    transform=ax.transAxes,
                    fontsize=10,
                    fontweight="medium",
                    family="serif",
                )

    # Add colorbars manually positioned to the right
    cbar_w = 0.015  # Width in figure coords
    cbar_pad = 0.01  # Pad from panels
    cbar_x = right + cbar_pad

    # Row heights in figure coords (accounting for top margin)
    row_h = top / 3.0
    gap_h = row_gap / fig_height

    # Velocity colorbar (spans top 2 rows)
    cax_vel = fig.add_axes([cbar_x, row_h + gap_h / 2, cbar_w, 2 * row_h - gap_h])
    cb_vel = fig.colorbar(vel_mappable, cax=cax_vel)
    cb_vel.locator = MaxNLocator(nbins=5)
    cb_vel.update_ticks()
    cb_vel.ax.tick_params(labelsize=7, pad=1)
    cb_vel.set_label("$v_x$ (m/s)", fontsize=8)
    cb_vel.outline.set_linewidth(0.5)

    # Error colorbar (bottom row)
    cax_err = fig.add_axes([cbar_x, 0, cbar_w, row_h - gap_h / 2])
    cb_err = fig.colorbar(err_mappable, cax=cax_err)
    cb_err.locator = MaxNLocator(nbins=4)
    cb_err.update_ticks()
    cb_err.ax.tick_params(labelsize=7, pad=1)
    cb_err.set_label("Error (m/s)", fontsize=8)
    cb_err.outline.set_linewidth(0.5)

    # Save with tight bounding box
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Wrote: {out_png}")


def fig_flag_row(
    rollout_pkl: Path,
    out_png: Path,
    step: int = 50,
    *,
    vel_cmap: str = CMAP_VELOCITY,
    err_cmap: str = CMAP_ERROR,
    dpi: int = PUBLICATION_DPI,
) -> None:
    """
    Create compact 1x3 row: Truth | Pred | Error for Flag dataset.

    Each panel has its own colorbar on the right.
    """
    _, plt, _, _, GridSpec, TwoSlopeNorm, MaxNLocator = get_mpl()
    apply_paper_style(dpi)

    # Load data
    traj = _load_traj(rollout_pkl)
    tri, pos = _triangulation(traj)
    pred, gt = _get_arrays(traj)
    T = gt.shape[0]
    t = min(max(step, 0), T - 1)

    # Extract x-component
    gt_vals = gt[t, :, 0]
    pred_vals = pred[t, :, 0]
    err_vals = gt_vals - pred_vals

    # Color limits
    vel_vmin, vel_vmax = robust_vminmax(np.concatenate([gt_vals, pred_vals]))
    err_max = np.quantile(np.abs(err_vals[np.isfinite(err_vals)]), 0.98)

    # Domain limits
    xmin, xmax = pos[:, 0].min(), pos[:, 0].max()
    ymin, ymax = pos[:, 1].min(), pos[:, 1].max()
    data_aspect = (ymax - ymin) / max(xmax - xmin, 1e-9)

    # Figure sizing: 3 panels + colorbars
    panel_w = 2.0  # Panel width in inches
    cbar_w = 0.08  # Colorbar width in inches
    cbar_gap = 0.03  # Gap between panel and colorbar
    panel_gap = 0.25  # Gap between panel+colorbar groups (for label)
    panel_height = panel_w * data_aspect
    # Total: 3 panels + 3 colorbars + 2 gaps between groups + right margin
    fig_width = 3 * panel_w + 3 * cbar_w + 3 * cbar_gap + 2 * panel_gap + 0.3
    fig_height = panel_height

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")

    # Panel data: (values, title, cmap, norm, cbar_label)
    panels = [
        (gt_vals, f"Truth, $t = {step}$", vel_cmap, None, "$v_x$ (m/s)"),
        (pred_vals, "Pred.", vel_cmap, None, "$v_x$ (m/s)"),
        (
            err_vals,
            "Error",
            err_cmap,
            TwoSlopeNorm(vcenter=0, vmin=-err_max, vmax=err_max),
            "(m/s)",
        ),
    ]

    # Width of one panel+colorbar+gap group
    group_w = panel_w + cbar_gap + cbar_w + panel_gap

    for i, (vals, title, cmap, norm, cbar_label) in enumerate(panels):
        # Calculate panel position in figure coords
        x_start = i * group_w / fig_width
        width = panel_w / fig_width
        ax = fig.add_axes([x_start, 0, width, 1])

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_axis_off()
        ax.set_facecolor("white")

        # Plot
        if norm:
            tpc = ax.tripcolor(tri, vals, shading="gouraud", cmap=cmap, norm=norm)
        else:
            tpc = ax.tripcolor(
                tri, vals, shading="gouraud", cmap=cmap, vmin=vel_vmin, vmax=vel_vmax
            )

        ax.triplot(tri, color=(0, 0, 0, 0.10), linewidth=0.15)

        # Title above panel
        ax.set_title(title, fontsize=10, family="serif", pad=2)

        # Colorbar - positioned with room for label
        cbar_x = x_start + width + cbar_gap / fig_width
        cax = fig.add_axes([cbar_x, 0.05, cbar_w / fig_width, 0.9])
        cb = fig.colorbar(tpc, cax=cax)
        cb.locator = MaxNLocator(nbins=4)
        cb.update_ticks()
        cb.ax.tick_params(labelsize=7, pad=1)
        cb.set_label(cbar_label, fontsize=8)
        cb.outline.set_linewidth(0.5)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_png, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.05
    )
    plt.close(fig)
    print(f"Wrote: {out_png}")


def fig_conformal_radii(
    rollout_pkl: Path,
    conformal_out: Path,
    out_png: Path,
    alpha: float = 0.1,
    step: int = 50,
    *,
    layout: str = "1x4",  # "1x4" for horizontal row, "2x2" for grid
    cmap: str = CMAP_VELOCITY,
    dpi: int = PUBLICATION_DPI,
) -> None:
    """
    Compare conformal prediction set radii (L2, Mahalanobis, Adaptive, CW-Adaptive).

    Panels: L2 Isotropic | Mahalanobis | Adaptive | CW-Adaptive (if available)

    Recommended configurations (from conformal/RESULTS.md):
        Cylinder:
            rollout: meshgraphnet/rollouts_200k_big/rollout_cylinder_test_200k.pkl
            conformal_out: conformal/_out_cylinder_200k_big_inregime_xgbq_physfull
        Flag:
            rollout: meshgraphnet/rollouts_200k_big/rollout_flag_test_200k.pkl
            conformal_out: conformal/_out_flag_200k_big_inregime_xgboost_sigcap098

    Note: Use test rollout for visualization (same split as coverage evaluation).
    The sigcap098 version for flag caps σ(x) at 98th percentile for stable bounds.

    Effective Radii (all in same physical units):
        - L2 Isotropic: constant radius = q_l2 (the L2 norm quantile)
        - Mahalanobis: constant radius = q_mah × det(Σ)^(1/2D)
          (Mahalanobis quantile scaled to volume-equivalent L2 radius)
        - Adaptive: varying radius = q_adapt × σ(x) at each node
          (adaptive quantile scaled by learned local uncertainty σ(x))
        - CW-Adaptive: varying radius = q_cw × (∏ᵢ σᵢ(x))^(1/D)
          (geometric mean of per-component scales)

    Visualization:
        - Colors show "Norm. Width" = radius / q_l2 (normalized by L2 Isotropic radius)
        - L2 Isotropic always appears as 1.0 (by definition, since we divide by q_l2)
        - Mahalanobis appears as constant = (q_mah × det(Σ)^(1/2D)) / q_l2
        - Adaptive shows spatial variation = (q_adapt × σ(x)) / q_l2
        - CW-Adaptive shows spatial variation = (q_cw × geom_mean(σᵢ)) / q_l2

    Colorbar scale:
        - vmin = 0.0 (fixed)
        - vmax = 98th percentile of all normalized radii (dynamic per dataset)
        - This means scale differs between datasets: if Mahalanobis/Adaptive produce
          much wider bounds than L2, vmax > 1.0; if similar, vmax ≈ 1.0

    Stats displayed below each panel:
        - w̄ = mean effective width (in physical units, directly comparable)
        - cov = empirical coverage from official test set evaluation (summary.json)
          Should be ≈1-α for well-calibrated methods

    Additional features:
        - White squares mark boundary nodes
        - Shared colorbar on right labeled "Norm. Width"
    """
    _, plt, mtri, _, GridSpec, _, MaxNLocator = get_mpl()
    apply_paper_style(dpi)

    # Load data
    traj = _load_traj(rollout_pkl)
    faces = np.asarray(traj["faces"])
    mesh_pos = np.asarray(traj["mesh_pos"])
    t0 = 0
    tris = faces[min(t0, len(faces) - 1)].astype(np.int32)
    pos = mesh_pos[min(t0, len(mesh_pos) - 1)].astype(np.float64)
    tri = mtri.Triangulation(pos[:, 0], pos[:, 1], tris)
    bnodes = _boundary_nodes(tris)

    # Load conformal outputs (including official coverage from test set)
    thresholds, sigma, sigma_pred, cw_sigma_pred, official_coverage = _load_conformal(conformal_out)
    feature_set = _read_feature_set(conformal_out)
    radii = _conformal_radii(traj, thresholds, sigma, sigma_pred, cw_sigma_pred, alpha, feature_set)

    T = radii["l2"].shape[0]
    t = min(max(step, 0), T - 1)

    # Use official coverage from summary.json (evaluated on test set)
    # These are the proper held-out coverage values
    a_str = str(alpha)
    if official_coverage and a_str in official_coverage.get("l2", {}):
        cov_l2 = official_coverage["l2"][a_str]
        cov_mah = official_coverage["mah"][a_str]
        cov_adapt = official_coverage["adapt"][a_str]
        cov_cw = official_coverage.get("cw_adapt", {}).get(a_str, None)
    else:
        # Fallback: compute on current trajectory (less accurate)
        from conformal.io import infer_pred_gt_keys

        pk, gk = infer_pred_gt_keys(traj)
        pred = np.asarray(traj[pk]).astype(np.float32)
        gt = np.asarray(traj[gk]).astype(np.float32)
        residuals = np.linalg.norm(gt - pred, axis=-1)
        cov_l2 = float(np.mean(residuals <= radii["l2"]))
        cov_mah = float(np.mean(residuals <= radii["mah"]))
        cov_adapt = float(np.mean(residuals <= radii["adapt"]))
        cov_cw = None

    # Get raw effective radii (all in same physical units)
    rad_l2 = radii["l2"][t]  # constant = q_l2
    rad_mah = radii["mah"][t]  # constant = q_mah * det(Sigma)^(1/2D)
    rad_adapt = radii["adapt"][t]  # varies = q_adapt * sigma(x)

    # Mean effective widths for display (all in same units now)
    mean_l2 = float(np.mean(rad_l2))
    mean_mah = float(np.mean(rad_mah))
    mean_adapt = float(np.mean(rad_adapt))

    # Normalize for visualization by L2 radius
    norm_factor = radii["q_l2"]
    rad_l2_norm = rad_l2 / norm_factor
    rad_mah_norm = rad_mah / norm_factor
    rad_adapt_norm = rad_adapt / norm_factor

    # Panel data: (normalized_radii, title, mean_width, cov)
    # First panel includes timestep and alpha; target coverage = 1-alpha
    # Coverage values are from official test set evaluation (summary.json)
    panels = [
        (
            rad_l2_norm,
            f"L2 Isotropic, $t = {t}$, $\\alpha = {alpha}$",
            mean_l2,
            cov_l2,
        ),
        (
            rad_mah_norm,
            "Mahalanobis",
            mean_mah,
            cov_mah,
        ),
        (
            rad_adapt_norm,
            "Adaptive",
            mean_adapt,
            cov_adapt,
        ),
    ]

    # Add CW-Adaptive panel if available
    if "cw_adapt" in radii:
        rad_cw = radii["cw_adapt"][t]
        mean_cw = float(np.mean(rad_cw))
        rad_cw_norm = rad_cw / norm_factor
        panels.append((
            rad_cw_norm,
            "CW-Adaptive",
            mean_cw,
            cov_cw if cov_cw is not None else 0.0,
        ))

    n_panels = len(panels)

    # Color limits for normalized radii (normalized by L2 Isotropic radius q_l2)
    # L2 panel is always 1.0; Mah/Adapt may be > 1.0 if they produce wider bounds
    # vmax is dynamic (98th percentile) so scale differs per dataset
    all_radii = np.concatenate([p[0] for p in panels])
    vmax = np.quantile(all_radii[np.isfinite(all_radii)], 0.98)
    vmin = 0.0

    # Domain limits
    xmin, xmax = pos[:, 0].min(), pos[:, 0].max()
    ymin, ymax = pos[:, 1].min(), pos[:, 1].max()
    data_aspect = (ymax - ymin) / max(xmax - xmin, 1e-9)

    # Layout configuration
    if layout == "2x2":
        # 2x2 grid layout (for cylinder)
        n_cols = 2
        n_rows = 2
        panel_w = 2.8
        panel_h = panel_w * data_aspect
        cbar_w = 0.15
        cbar_gap = 0.12
        h_gap = 0.08
        v_gap = max(0.65, 0.25 * panel_h)

        fig_width = n_cols * panel_w + h_gap + cbar_gap + cbar_w + 0.1
        fig_height = n_rows * panel_h + v_gap + 0.3
    else:
        # 1x4 horizontal layout (default, for flag)
        n_cols = n_panels
        n_rows = 1
        panel_w = 2.2 if n_panels == 3 else 1.8
        panel_h = panel_w * data_aspect
        cbar_w = 0.12
        cbar_gap = 0.08
        h_gap = 0.02
        v_gap = 0

        fig_width = n_panels * panel_w + (n_panels - 1) * h_gap + cbar_gap + cbar_w + 0.25
        fig_height = panel_h + 0.35

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")

    mappables = []
    axes_list = []

    for i, (vals, title, mean_width, cov) in enumerate(panels):
        if layout == "2x2":
            row = i // n_cols
            col = i % n_cols
            x_start = col * (panel_w + h_gap) / fig_width
            y_start = (1 - (row + 1) * panel_h / fig_height - row * v_gap / fig_height)
            y_start = max(0.12, y_start)
        else:
            # 1x4 layout
            x_start = (i * (panel_w + h_gap)) / fig_width
            y_start = 0.25 / fig_height  # Space for stats

        width_frac = panel_w / fig_width
        height_frac = panel_h / fig_height

        ax = fig.add_axes([x_start, y_start, width_frac, height_frac])
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_axis_off()
        ax.set_facecolor("white")

        # Plot radii
        tpc = ax.tripcolor(
            tri, vals, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax
        )
        mappables.append(tpc)
        axes_list.append(ax)

        # Mesh overlay
        ax.triplot(tri, color=(0, 0, 0, 0.08), linewidth=0.15)

        # Boundary nodes as white squares
        ax.scatter(
            pos[bnodes, 0],
            pos[bnodes, 1],
            s=3,
            c="white",
            marker="s",
            edgecolors="black",
            linewidths=0.2,
            zorder=5,
        )

        # Title above panel
        title_fontsize = 10 if layout == "2x2" else (10 if n_panels == 3 else 9)
        ax.set_title(title, fontsize=title_fontsize, family="serif", pad=3)

        # Stats below panel
        ax.text(
            0.5,
            -0.08,
            f"$\\bar{{W}} = {mean_width:.2f}$, cov = {cov:.2f}",
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontsize=9 if layout == "2x2" else 8,
            family="serif",
            clip_on=False,
            zorder=100,
        )

    # Shared colorbar on the right
    if layout == "2x2":
        cbar_x = (n_cols * panel_w + h_gap + cbar_gap) / fig_width
        cbar_bottom = axes_list[2].get_position().y0 if len(axes_list) > 2 else axes_list[0].get_position().y0
        cbar_top = axes_list[0].get_position().y1
        cbar_h = cbar_top - cbar_bottom
    else:
        total_panels_w = n_panels * panel_w + (n_panels - 1) * h_gap
        cbar_x = (total_panels_w + cbar_gap) / fig_width
        cbar_h = panel_h / fig_height
        cbar_bottom = 0.25 / fig_height

    cax = fig.add_axes([cbar_x, cbar_bottom, cbar_w / fig_width, cbar_h])
    cb = fig.colorbar(mappables[-1], cax=cax)
    cb.locator = MaxNLocator(nbins=5)
    cb.update_ticks()
    cb.ax.tick_params(labelsize=7 if layout == "1x4" else 8, pad=1 if layout == "1x4" else 2)
    cb.set_label("Norm. Width", fontsize=9 if layout == "1x4" else 10, rotation=270, labelpad=12 if layout == "1x4" else 14)
    cb.outline.set_linewidth(0.5)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_png, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.02
    )
    plt.close(fig)
    print(f"Wrote: {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate grid figures",
        epilog="""
Recommended radii mode configurations:
  Cylinder:
    --rollout_pkl meshgraphnet/rollouts_200k_big/rollout_cylinder_test_200k.pkl
    --conformal_out conformal/_out_cylinder_200k_big_inregime_xgbq_physfull
  Flag:
    --rollout_pkl meshgraphnet/rollouts_200k_big/rollout_flag_test_200k.pkl
    --conformal_out conformal/_out_flag_200k_big_inregime_xgboost_sigcap098
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--rollout_pkl",
        required=True,
        help="Rollout pickle file (use test split for radii mode)",
    )
    ap.add_argument("--out_png", default="paper/figures_generated/temporal_grid.png")
    ap.add_argument("--step_a", type=int, default=50, help="First timestep")
    ap.add_argument("--step_b", type=int, default=140, help="Second timestep")
    ap.add_argument(
        "--mode", choices=["3x2", "1x3", "radii"], default="3x2", help="Grid layout"
    )
    ap.add_argument(
        "--conformal_out",
        help="Conformal output dir (for radii mode). Use sigcap098 version for flag.",
    )
    ap.add_argument("--alpha", type=float, default=0.1, help="Alpha for radii mode")
    ap.add_argument("--layout", choices=["1x4", "2x2"], default="1x4", help="Layout for radii mode (1x4 for flag, 2x2 for cylinder)")
    args = ap.parse_args()

    if args.mode == "3x2":
        fig_temporal_grid(
            Path(args.rollout_pkl),
            Path(args.out_png),
            step_a=args.step_a,
            step_b=args.step_b,
        )
    elif args.mode == "1x3":
        fig_flag_row(
            Path(args.rollout_pkl),
            Path(args.out_png),
            step=args.step_a,
        )
    elif args.mode == "radii":
        if not args.conformal_out:
            ap.error("--conformal_out required for radii mode")
        fig_conformal_radii(
            Path(args.rollout_pkl),
            Path(args.conformal_out),
            Path(args.out_png),
            alpha=args.alpha,
            step=args.step_a,
            layout=args.layout,
        )


if __name__ == "__main__":
    main()
