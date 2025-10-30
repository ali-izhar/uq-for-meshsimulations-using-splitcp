"""
Lightweight plotting/animation script for MeshGraphNet experiments.

Purpose:
- Reuse existing plotting utilities in meshgraph/utils/plot.py
- Avoid training; only compose config and generate figures/animations

Usage examples:
  # Create standard plots from existing checkpoint/config
  python make_plots.py plots +datasize=medium +noise=paper

  # Create standard animations
  python make_plots.py animations +datasize=large

  # Create both
  python make_plots.py both +datasize=small

Notes:
- This script mirrors the visualization functionality in run_gnn.py,
  but without any training-related code paths.
"""

from __future__ import annotations

import argparse
import sys

import hydra
from omegaconf import DictConfig

# Import plotting helpers from the existing module
from utils.plot import (
    create_standard_plots,
    create_standard_animations,
    create_velocity_frames,
    create_mesh_topology_plot,
    create_mesh_flow_plot,
    create_mesh_streamplot,
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse CLI args, separating Hydra overrides (unknown to argparse).

    We follow the same pattern as run_gnn.py so users can pass +overrides.
    """
    parser = argparse.ArgumentParser(
        description="Create MeshGraphNet plots/animations from existing checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python make_plots.py plots +datasize=medium +noise=paper\n"
            "  python make_plots.py animations +datasize=large\n"
            "  python make_plots.py both +datasize=small\n"
        ),
    )

    parser.add_argument(
        "mode",
        choices=["plots", "animations", "both", "frames", "mesh"],
        help="Visualization mode: create plots, animations, both, static frames, or a mesh topology plot",
    )

    # Frame-specific options (used when mode == frames)
    parser.add_argument(
        "--start-step",
        type=int,
        default=0,
        help="Starting timestep for frames (default: 0)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of timesteps to consider for frames (default: 100)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Save every k-th step (default: 10)",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="horizontal",
        choices=["horizontal", "vertical"],
        help="Figure layout: horizontal (1x3) or vertical (3x1)",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Use compact research-style layout",
    )
    parser.add_argument(
        "--cmap-vel",
        type=str,
        default="cividis",
        help="Colormap for velocity panels",
    )
    parser.add_argument(
        "--cmap-err",
        type=str,
        default="coolwarm",
        help="Colormap for error panel",
    )
    parser.add_argument(
        "--vel-comp",
        type=int,
        default=0,
        choices=[0, 1],
        help="Velocity component to plot: 0=x, 1=y (default: 0)",
    )
    parser.add_argument(
        "--rollout",
        action="store_true",
        help="Use rollout predictions (default: single-step)",
    )
    parser.add_argument(
        "--use-test-traj",
        action="store_true",
        help="Use test trajectory (default: training trajectory)",
    )

    # Return known args + Hydra overrides
    return parser.parse_known_args()


def main() -> None:
    """Entry point: compose Hydra config and run the requested visualizations.

    We intentionally keep this script focused on visualization only.
    """
    # If no args passed, show help quickly
    if len(sys.argv) == 1:
        print("Usage: python make_plots.py [plots|animations|both] +overrides")
        return

    args, hydra_overrides = parse_args()

    # Compose config using Hydra (same config tree as run_gnn.py)
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with hydra.initialize(version_base=None, config_path="configs", job_name="viz"):
        cfg: DictConfig = hydra.compose(
            config_name="default", overrides=hydra_overrides
        )

    # Route by mode, using existing helpers
    if args.mode == "plots":
        print("Creating standard plots...")
        create_standard_plots(cfg)
    elif args.mode == "animations":
        print("Creating standard animations...")
        create_standard_animations(cfg)
    elif args.mode == "both":
        print("Creating both plots and animations...")
        create_standard_plots(cfg)
        create_standard_animations(cfg)
    elif args.mode == "frames":
        print(
            f"Creating static frames: start={args.start_step}, num={args.num_steps}, stride={args.stride}, "
            f"mode={'rollout' if args.rollout else 'single-step'}, test_traj={args.use_test_traj}"
        )
        saved = create_velocity_frames(
            cfg,
            start_step=args.start_step,
            num_steps=args.num_steps,
            stride=args.stride,
            single_step=(not args.rollout),
            use_test_traj=args.use_test_traj,
            layout=args.layout,
            compact=args.compact,
            cmap_vel=args.cmap_vel,
            cmap_err=args.cmap_err,
            velocity_component=args.vel_comp,
        )
        if saved:
            print(f"Saved {len(saved)} frame(s). Example: {saved[0]}")
        else:
            print("No frames saved (empty window or error).")
    elif args.mode == "mesh":
        print("Creating high-contrast mesh topology plot...")
        out = create_mesh_topology_plot(
            cfg,
            step=args.start_step,
            use_test_traj=args.use_test_traj,
            dpi=300,
        )
        print(f"Saved: {out}")
        print("Creating high-contrast mesh flow plot (with fluid colors)...")
        out2 = create_mesh_flow_plot(
            cfg,
            step=args.start_step,
            use_test_traj=args.use_test_traj,
            dpi=300,
        )
        print(f"Saved: {out2}")
        print("Creating mathematically precise streamplot (ground truth speed)...")
        out3 = create_mesh_streamplot(
            cfg,
            step=args.start_step,
            use_test_traj=args.use_test_traj,
            field="truth",
            dpi=300,
            density=1.2,
        )
        print(f"Saved: {out3}")
    else:
        print(f"Unknown mode: {args.mode}")

    print("Visualization complete!")


if __name__ == "__main__":
    main()
