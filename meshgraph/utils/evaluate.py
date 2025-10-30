#!/usr/bin/env python3
"""
Entry point for evaluation and analysis using Hydra configuration.

This script provides evaluation capabilities including noise effect analysis,
rollout stability comparison, and custom evaluation plots.

Usage:
    python utils/evaluate.py --analysis noise_comparison +datasize=large
    python utils/evaluate.py --analysis rollout_analysis +datasize=large
"""

import hydra
from omegaconf import DictConfig
import sys
import os
import argparse

# Add utils to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from plot import compare_noise_effects, plot_rollout_error  # noqa: E402


def noise_comparison_analysis(cfg: DictConfig) -> None:
    """Run noise effect comparison analysis."""
    print("Analyzing training noise effects on rollout stability...")

    # Extract base config from current settings
    base_config = []
    if hasattr(cfg, "datasize"):
        base_config.append(f"+datasize={cfg.datasize}")
    if hasattr(cfg, "dataset") and cfg.dataset:
        base_config.append(f"+dataset={cfg.dataset}")

    # Run comparison
    compare_noise_effects(
        base_config=base_config,
        start_step=50,
        num_steps=100,
        use_test_traj=True,
        output_name="noise_effect_analysis",
    )
    print("✓ Noise comparison analysis complete!")


def rollout_analysis(cfg: DictConfig) -> None:
    """Run detailed rollout error analysis."""
    print("Analyzing rollout error accumulation...")

    # Multiple window analyses
    windows = [
        (0, 100, "early_rollout"),
        (50, 150, "mid_rollout"),
        (100, 200, "late_rollout"),
    ]

    for start, window_size, name in windows:
        print(f"  Analyzing {name} (steps {start}-{start + window_size})...")
        plot_rollout_error(
            cfg, start_step=start, num_steps=window_size, use_test_traj=True
        )

    print("✓ Rollout analysis complete!")


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """
    Main evaluation dispatcher.

    Args:
        cfg: Hydra configuration loaded from configs/
    """
    parser = argparse.ArgumentParser(description="MeshGraphNet Evaluation Suite")
    parser.add_argument(
        "--analysis",
        type=str,
        default="noise_comparison",
        choices=["noise_comparison", "rollout_analysis", "both"],
        help="Type of analysis to run",
    )

    # Parse args (Hydra modifies sys.argv, so we need to handle this carefully)
    args, unknown = parser.parse_known_args()

    print(f"Running evaluation analysis: {args.analysis}")

    if args.analysis in ["noise_comparison", "both"]:
        noise_comparison_analysis(cfg)

    if args.analysis in ["rollout_analysis", "both"]:
        rollout_analysis(cfg)

    print("✓ All evaluation analyses complete!")


if __name__ == "__main__":
    main()
