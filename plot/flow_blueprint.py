#!/usr/bin/env python3
"""
Flow visualizations (4 panels) from rollout PKLs:

- Cylinder: speed+quiver, streamlines
- Flag: speed+quiver, streamlines
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from plot.style import (
    get_mpl,
    apply_paper_style,
    add_colorbar,
    robust_vminmax,
    CMAP_VELOCITY,
    PUBLICATION_DPI,
)


def _boundary_nodes(tris: np.ndarray) -> np.ndarray:
    """Return boundary node indices from triangles."""
    tris = np.asarray(tris, dtype=np.int64)
    edge_count: dict[tuple[int, int], int] = {}
    for a, b, c in tris:
        for u, v in ((a, b), (b, c), (c, a)):
            key = (min(u, v), max(u, v))
            edge_count[key] = edge_count.get(key, 0) + 1
    boundary = {n for (u, v), cnt in edge_count.items() if cnt == 1 for n in (u, v)}
    return np.array(sorted(boundary), dtype=np.int32)


def _flag_constraints(pos: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Find left-edge constraint points on flag boundary."""
    b = _boundary_nodes(tris)
    if b.size == 0:
        return np.array([], dtype=np.int32)
    xmin = pos[b, 0].min()
    tol = 0.01 * max(pos[b, 0].max() - pos[b, 0].min(), 1e-9)
    left = b[np.abs(pos[b, 0] - xmin) <= tol]
    if left.size < 2:
        return left
    yl = pos[left, 1]
    return np.array([left[yl.argmin()], left[yl.argmax()]], dtype=np.int32)


def _interior_mask(
    pos: np.ndarray, tris: np.ndarray, margin: float = 0.02
) -> np.ndarray:
    """Mask nodes away from boundary by margin * domain_diagonal."""
    b = _boundary_nodes(tris)
    if b.size == 0:
        return np.ones(pos.shape[0], dtype=bool)
    diag = max(
        np.hypot(pos[:, 0].max() - pos[:, 0].min(), pos[:, 1].max() - pos[:, 1].min()),
        1e-9,
    )
    radius = margin * diag
    # Compute distance to boundary using broadcasting (vectorized)
    pb = pos[b]
    # Chunked to avoid memory explosion on large meshes
    d_min = np.full(pos.shape[0], np.inf)
    for i in range(0, pos.shape[0], 2048):
        chunk = pos[i : i + 2048]
        d2 = ((chunk[:, None, :] - pb[None, :, :]) ** 2).sum(axis=2)
        d_min[i : i + 2048] = np.sqrt(d2.min(axis=1))
    return d_min >= radius


def _load_traj(pkl: Path, idx: int = 0) -> dict:
    """Load trajectory from rollout pickle."""
    from conformal.io import load_rollouts

    rollouts = load_rollouts(str(pkl))
    if not rollouts:
        raise ValueError(f"No trajectories in {pkl}")
    return rollouts[idx]


def _triangulation(traj: dict, t: int = 0):
    """Build Matplotlib triangulation from trajectory."""
    _, _, mtri, *_ = get_mpl()
    faces = np.asarray(traj["faces"])
    mesh_pos = np.asarray(traj["mesh_pos"])
    tris = faces[min(t, len(faces) - 1)].astype(np.int32)
    pos = mesh_pos[min(t, len(mesh_pos) - 1)].astype(np.float64)
    return mtri.Triangulation(pos[:, 0], pos[:, 1], tris), pos, tris


def _velocity(traj: dict, step: int, use_gt: bool = True):
    """Extract velocity field (u, v, t_used) from trajectory.
    
    For CylinderFlow (D=2): returns velocity directly from rollout.
    For Flag (D=3): returns finite-difference of position (p[t+1] - p[t]).
        Uses dt=1 per DeepMind convention; units are position-change-per-frame.
    """
    from conformal.io import infer_pred_gt_keys

    pk, gk = infer_pred_gt_keys(traj)
    arr = np.asarray(traj[gk if use_gt else pk], dtype=np.float32)
    D = arr.shape[-1]
    if D == 2:  # Cylinder: direct velocity (m/s)
        t = min(max(step, 0), arr.shape[0] - 1)
        return arr[t, :, 0].astype(np.float64), arr[t, :, 1].astype(np.float64), t
    if D == 3:  # Flag: finite-difference (dt=1 per DeepMind convention)
        T = arr.shape[0]
        if T < 2:
            raise ValueError("Need ≥2 timesteps for velocity")
        t = min(max(step, 0), T - 2)
        vel = arr[t + 1, :, :2] - arr[t, :, :2]  # dt=1, position-change-per-frame
        return vel[:, 0].astype(np.float64), vel[:, 1].astype(np.float64), t
    raise ValueError(f"Unsupported dim D={D}")


def fig_speed_quiver(
    traj: dict,
    out_png: Path,
    *,
    step: int,
    cmap: str = CMAP_VELOCITY,
    quiver_target: int = 220,
    cbar_label: str = "$|v|$ (m/s)",
    cbar_size: float = 0.12,
    axes_limits: tuple[float, float, float, float] | None = None,
) -> None:
    """Generate speed+quiver visualization."""
    _, plt, *_ = get_mpl()
    apply_paper_style()

    u, v, _ = _velocity(traj, step)
    triang, pos, tris = _triangulation(traj)
    speed = np.hypot(u, v)

    # Compute limits
    if axes_limits:
        xmin, xmax, ymin, ymax = axes_limits
    else:
        xmin, xmax = pos[:, 0].min(), pos[:, 0].max()
        ymin, ymax = pos[:, 1].min(), pos[:, 1].max()

    fig, ax = plt.subplots(figsize=(6.5, 2.0))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Speed field
    vmin, vmax = robust_vminmax(speed)
    tpc = ax.tripcolor(
        triang, speed, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax
    )
    ax.triplot(triang, color=(0, 0, 0, 0.16), linewidth=0.25)

    # Quiver arrows (downsampled, interior only)
    interior = _interior_mask(pos, tris)
    idx = np.where(interior)[0]
    if idx.size == 0:
        idx = np.arange(len(pos))
    idx = idx[:: max(1, len(idx) // quiver_target)]

    uu, vv = u[idx], v[idx]
    sp = np.hypot(uu, vv)
    q95 = max(np.quantile(sp[np.isfinite(sp)], 0.95), 1e-12) if sp.size else 1.0
    diag = np.hypot(xmax - xmin, ymax - ymin)
    base_len = 0.024 * diag
    length = base_len * np.clip(sp / q95, 0.25, 1.0)
    nz = sp > 1e-12
    u_plot, v_plot = np.zeros_like(uu), np.zeros_like(vv)
    u_plot[nz] = uu[nz] / sp[nz] * length[nz]
    v_plot[nz] = vv[nz] / sp[nz] * length[nz]

    ax.quiver(
        pos[idx, 0],
        pos[idx, 1],
        u_plot,
        v_plot,
        color=(1, 1, 1, 0.75),
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.00135,
        headwidth=2.8,
        headlength=3.2,
        headaxislength=2.9,
        pivot="mid",
        clip_on=True,
        zorder=4,
    )

    add_colorbar(fig, ax, tpc, label=cbar_label, size=cbar_size)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_png,
        dpi=PUBLICATION_DPI,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)


def fig_streamlines(
    traj: dict,
    out_png: Path,
    *,
    step: int,
    cmap: str = CMAP_VELOCITY,
    grid_nx: int = 500,
    grid_ny: int = 140,
    density: float = 1.2,
    lw_range: tuple[float, float] = (0.35, 2.0),
    show_constraints: bool = False,
    cbar_label: str = "$|v|$ (m/s)",
    cbar_size: float = 0.12,
    axes_limits: tuple[float, float, float, float] | None = None,
) -> None:
    """Generate streamlines visualization."""
    _, plt, _, LinearTriInterpolator, *_ = get_mpl()
    apply_paper_style()

    u, v, _ = _velocity(traj, step)
    triang, pos, tris = _triangulation(traj)

    # Interpolate onto regular grid
    Iu, Iv = LinearTriInterpolator(triang, u), LinearTriInterpolator(triang, v)
    if axes_limits:
        xmin, xmax, ymin, ymax = axes_limits
    else:
        xmin, xmax = pos[:, 0].min(), pos[:, 0].max()
        ymin, ymax = pos[:, 1].min(), pos[:, 1].max()

    X, Y = np.linspace(xmin, xmax, grid_nx), np.linspace(ymin, ymax, grid_ny)
    XX, YY = np.meshgrid(X, Y)
    UU, VV = np.ma.array(Iu(XX, YY)), np.ma.array(Iv(XX, YY))
    mask = np.ma.getmask(UU) | np.ma.getmask(VV)
    UU.mask = VV.mask = mask
    speed = np.hypot(UU, VV)

    smin, smax = robust_vminmax(np.asarray(speed))
    lw_min, lw_max = lw_range
    lw = lw_min + (lw_max - lw_min) * np.clip(
        (speed - smin) / max(smax - smin, 1e-9), 0, 1
    )

    fig, ax = plt.subplots(figsize=(6.5, 2.0))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    strm = ax.streamplot(
        X,
        Y,
        UU,
        VV,
        color=speed,
        linewidth=lw,
        cmap=cmap,
        norm=plt.Normalize(vmin=smin, vmax=smax),
        density=density,
        arrowsize=0.7,
        minlength=0.08,
    )
    ax.triplot(triang, color=(0, 0, 0, 0.14), linewidth=0.25)

    if show_constraints:
        cidx = _flag_constraints(pos, tris)
        if cidx.size:
            ax.scatter(
                pos[cidx, 0],
                pos[cidx, 1],
                s=32,
                c="#E24A33",
                edgecolors="white",
                linewidths=0.8,
                zorder=5,
            )

    add_colorbar(fig, ax, strm.lines, label=cbar_label, size=cbar_size)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_png,
        dpi=PUBLICATION_DPI,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cylinder_pkl", required=True)
    ap.add_argument("--flag_pkl", required=True)
    ap.add_argument("--out_dir", default="paper/figures_generated")
    ap.add_argument("--step", type=int, default=50)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    step = args.step

    cyl = _load_traj(Path(args.cylinder_pkl))
    flag = _load_traj(Path(args.flag_pkl))

    # Get cylinder limits for consistent axes
    _, pos_cyl, _ = _triangulation(cyl)
    cyl_limits = (
        pos_cyl[:, 0].min(),
        pos_cyl[:, 0].max(),
        pos_cyl[:, 1].min(),
        pos_cyl[:, 1].max(),
    )

    # Cylinder plots
    fig_speed_quiver(
        cyl,
        out_dir / "mesh_flow_colored.png",
        step=step,
        quiver_target=240,
        axes_limits=cyl_limits,
    )
    fig_streamlines(
        cyl,
        out_dir / "mesh_stream_truth.png",
        step=step,
        grid_nx=520,
        grid_ny=150,
        density=1.25,
        axes_limits=cyl_limits,
    )

    # Flag plots (position-change-per-frame, dt=1)
    fig_speed_quiver(
        flag,
        out_dir / f"flag_speed_quiver_step{step}.png",
        step=step,
        quiver_target=170,
        cbar_label=r"$|\Delta p|$ / frame",
    )
    fig_streamlines(
        flag,
        out_dir / f"flag_streamlines_step{step}.png",
        step=step,
        grid_nx=420,
        grid_ny=420,
        density=1.10,
        show_constraints=True,
        cbar_label=r"$|\Delta p|$ / frame",
    )

    print("Wrote:")
    for name in [
        "mesh_flow_colored.png",
        "mesh_stream_truth.png",
        f"flag_speed_quiver_step{step}.png",
        f"flag_streamlines_step{step}.png",
    ]:
        print(f"  - {out_dir / name}")


if __name__ == "__main__":
    main()
