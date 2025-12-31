#!/usr/bin/env python3
"""
Mesh topology visualization.

Shows:
- Triangular mesh structure
- Interior nodes
- Boundary nodes (highlighted)

Uses serif fonts matching paper styling.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from plot.style import get_mpl, apply_paper_style, PUBLICATION_DPI


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


def fig_mesh_topology(
    rollout_pkl: Path,
    out_png: Path,
    *,
    show_interior_nodes: bool = True,
    show_boundary_nodes: bool = True,
    dpi: int = PUBLICATION_DPI,
) -> None:
    """
    Create mesh topology visualization.

    Shows the triangular mesh structure with:
    - Mesh edges (light gray)
    - Interior nodes (small, semi-transparent)
    - Boundary nodes (highlighted in crimson)

    Args:
        rollout_pkl: Path to rollout pickle file
        out_png: Output PNG path
        show_interior_nodes: Whether to show interior mesh nodes
        show_boundary_nodes: Whether to highlight boundary nodes
        dpi: Output resolution
    """
    _, plt, mtri, *_ = get_mpl()
    apply_paper_style(dpi)

    # Load data
    from conformal.io import load_rollouts

    traj = load_rollouts(str(rollout_pkl))[0]
    faces = np.asarray(traj["faces"])
    mesh_pos = np.asarray(traj["mesh_pos"])

    # Use first timestep
    tris = faces[0].astype(np.int32)
    pos = mesh_pos[0].astype(np.float64)
    tri = mtri.Triangulation(pos[:, 0], pos[:, 1], tris)

    # Domain limits
    xmin, xmax = pos[:, 0].min(), pos[:, 0].max()
    ymin, ymax = pos[:, 1].min(), pos[:, 1].max()
    data_aspect = (ymax - ymin) / max(xmax - xmin, 1e-9)

    # Figure size based on aspect ratio
    fig_width = 6.0
    fig_height = fig_width * data_aspect

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor="white")

    # Draw mesh edges
    ax.triplot(tri, color="0.7", linewidth=0.3, alpha=0.8)

    # Draw interior nodes
    if show_interior_nodes:
        ax.scatter(
            pos[:, 0],
            pos[:, 1],
            s=1.5,
            c="0.3",
            alpha=0.5,
            linewidths=0,
            zorder=2,
        )

    # Highlight boundary nodes
    if show_boundary_nodes:
        bnodes = _boundary_nodes(tris)
        ax.scatter(
            pos[bnodes, 0],
            pos[bnodes, 1],
            s=6,
            c="#D55E00",  # crimson for boundary
            alpha=0.9,
            linewidths=0,
            zorder=3,
        )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # Save
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_png, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.02
    )
    plt.close(fig)
    print(f"Wrote: {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate mesh topology visualization",
        epilog="""
Example usage:
  python plot/mesh.py \\
    --rollout_pkl meshgraphnet/rollouts_200k_big/rollout_cylinder_test_200k.pkl \\
    --out_png paper/figures_generated/cylinder_mesh.png
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--rollout_pkl", required=True, help="Rollout pickle file")
    ap.add_argument(
        "--out_png",
        default="paper/figures_generated/mesh_topology.png",
        help="Output PNG path",
    )
    ap.add_argument(
        "--no_interior",
        action="store_true",
        help="Hide interior nodes",
    )
    ap.add_argument(
        "--no_boundary",
        action="store_true",
        help="Hide boundary nodes",
    )
    args = ap.parse_args()

    fig_mesh_topology(
        Path(args.rollout_pkl),
        Path(args.out_png),
        show_interior_nodes=not args.no_interior,
        show_boundary_nodes=not args.no_boundary,
    )


if __name__ == "__main__":
    main()
