#!/usr/bin/env python3
"""
Command-line interface for running conformal prediction experiments.

Simplified wrapper around conformal.run_conformal for ease of use with
results/mesh_preds/*.pkl files.

Usage Examples:
    # Compare all geometries (ℓ2, ℓ∞, Mahalanobis) at single α
    python cli.py cylinder_medium_noise --alpha 0.1 --compare

    # Alpha sweep with comparison
    python cli.py cylinder_medium_noise --sweep 0.05,0.1,0.15,0.2 --compare

    # Spatially adaptive CP with basic features (p=5)
    python cli.py cylinder_medium_noise -s 0.05,0.1,0.15,0.2 -ad -x 0.2

    # Spatially adaptive CP with full features (p=17, all 6 categories from Table 3)
    python cli.py cylinder_medium_noise -s 0.05,0.1,0.15,0.2 -ad -x 0.2 -f

    # Flag dataset (3D)
    python cli.py flag_medium --alpha 0.1 --compare

Reference:
    See paper.tex for experimental protocol and conformal/README.md for details.
"""

import argparse
import sys
import os
import subprocess
from pathlib import Path


def get_prediction_file(name: str) -> str:
    """
    Get full path to prediction file from short name.

    Args:
        name: Short name like 'cylinder_medium_noise' or 'flag_medium'

    Returns:
        Full path to .pkl file in results/mesh_preds/
    """
    # Map short names to actual files
    pred_dir = Path("results/mesh_preds")

    # Pattern matching for common names
    patterns = {
        "cylinder_medium_noise": "cylinder_medium_noise_0-500_predictions.pkl",
        "cylinder_medium": "cylinder_medium_0-500_predictions.pkl",
        "flag_medium": "flag_medium_0-100_predictions.pkl",
    }

    if name in patterns:
        filepath = pred_dir / patterns[name]
    elif name.endswith(".pkl"):
        # Direct filename provided
        filepath = pred_dir / name
    else:
        # Try to find matching file
        matches = list(pred_dir.glob(f"{name}*.pkl"))
        if len(matches) == 1:
            filepath = matches[0]
        elif len(matches) > 1:
            print(f"Multiple matches for '{name}':")
            for m in matches:
                print(f"  - {m.name}")
            sys.exit(1)
        else:
            print(f"No prediction file found matching '{name}' in {pred_dir}")
            print("\nAvailable files:")
            for f in sorted(pred_dir.glob("*.pkl")):
                print(f"  - {f.name}")
            sys.exit(1)

    if not filepath.exists():
        print(f"File not found: {filepath}")
        sys.exit(1)

    return str(filepath)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Conformal Prediction CLI for Mesh-Based Simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single alpha, compare all geometries (ℓ2, ℓ∞, Mahalanobis)
    python cli.py cylinder_medium_noise -a 0.1 -c
    
    # Alpha sweep across confidence levels
    python cli.py cylinder_medium_noise -s 0.05,0.1,0.15,0.2 -c
    
    # Spatially adaptive CP with basic features (p=5)
    python cli.py cylinder_medium_noise -s 0.05,0.1,0.15,0.2 -ad -x 0.2
    
    # Spatially adaptive CP with full features (p=17, all 6 categories from Table 3)
    python cli.py cylinder_medium_noise -s 0.05,0.1,0.15,0.2 -ad -x 0.2 -f
    
    # Flag dataset (3D simulation)
    python cli.py flag_medium -a 0.1 -c
    
    Note: Results are saved to JSON by default in results/conformal_preds/
        """,
    )

    # Positional argument for prediction file
    parser.add_argument(
        "predictions",
        type=str,
        help="Prediction file name (e.g., 'cylinder_medium_noise', 'flag_medium') or full .pkl filename",
    )

    # Alpha selection
    group_alpha = parser.add_mutually_exclusive_group(required=True)
    group_alpha.add_argument(
        "--alpha",
        "-a",
        type=float,
        help="Single miscoverage level α ∈ (0,1), target coverage = 1-α",
    )
    group_alpha.add_argument(
        "--sweep",
        "-s",
        type=str,
        help="Alpha sweep: comma-separated values (e.g., 0.05,0.1,0.15,0.2)",
    )

    # Analysis mode
    parser.add_argument(
        "--compare",
        "-c",
        action="store_true",
        help="Compare all geometries (ℓ2, ℓ∞, Mahalanobis)",
    )

    parser.add_argument(
        "--adaptive",
        "-ad",
        action="store_true",
        help="Run spatially adaptive conformal prediction",
    )

    parser.add_argument(
        "--full-features",
        "-f",
        dest="enhanced",
        action="store_true",
        help="Use full features (p=17) for adaptive CP (default: basic p=5)",
    )

    parser.add_argument(
        "--geometry",
        "-g",
        type=str,
        choices=["l2", "mahalanobis", "box"],
        default="l2",
        help="Single geometry to use (default: l2)",
    )

    # Data splitting
    parser.add_argument(
        "--calib-ratio",
        "-r",
        type=float,
        default=0.5,
        help="Calibration ratio (default: 0.5)",
    )

    parser.add_argument(
        "--aux",
        "-x",
        type=float,
        default=0.0,
        help="Auxiliary ratio for adaptive CP (default: 0.0, use 0.2-0.3)",
    )

    # Output options
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="results/conformal_preds",
        help="Output directory (default: results/conformal_preds)",
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable JSON saving (saved by default)",
    )

    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Disable fluid node masking (include walls)",
    )

    # Advanced options
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=4,
        help="Parallel workers for adaptive CP (default: 4)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    # Get prediction file path
    pred_file = get_prediction_file(args.predictions)
    print(f"Using predictions: {pred_file}")
    print(f"Output directory: {args.output}")
    print()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Build sys.argv for run_conformal.py
    conformal_args = [
        "run_conformal.py",
        "-f",
        pred_file,
        "-r",
        str(args.calib_ratio),
        "-o",
        args.output,
    ]

    if args.no_mask:
        conformal_args.append("--no-mask")

    if args.no_save:
        conformal_args.append("--no-save")

    if args.aux > 0:
        conformal_args.extend(["--aux", str(args.aux)])

    # Add alpha/sweep
    if args.alpha:
        conformal_args.extend(["-a", str(args.alpha)])
    elif args.sweep:
        conformal_args.extend(["-s", args.sweep])

    # Add mode flags
    if args.compare:
        conformal_args.append("--compare")

    if args.adaptive:
        conformal_args.extend(["--boosted", "--workers", str(args.workers)])
        if args.enhanced:
            conformal_args.append("--enhanced")
    elif not args.compare:
        # Single geometry mode
        conformal_args.extend(["--nonconformity", args.geometry])

    # Run conformal module as subprocess
    print("=" * 80)
    print("CONFORMAL PREDICTION ANALYSIS")
    print("=" * 80)
    print()

    # Build command to run conformal module
    cmd = [sys.executable, "-m", "conformal.run_conformal"] + conformal_args[1:]

    print(f"Running: {' '.join(cmd[2:])}")  # Skip python -m part
    print()

    # Run the command
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
