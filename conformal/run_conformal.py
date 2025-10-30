#!/usr/bin/env python3
"""
Command-Line Interface for Conformal Prediction Analysis

Main entry point for uncertainty quantification on mesh-based simulation predictions.
Implements the experimental framework from the paper with support for:

    - Four prediction-set geometries: ℓ2, joint ℓ∞, Mahalanobis, spatially adaptive
    - Alpha sweep analysis across confidence levels
    - Comparison mode for evaluating geometry efficiency
    - Three-way data split: auxiliary, calibration, test (for valid boosted CP)

Operates post-hoc on saved predictions without requiring the GNN model.

Reference:
    See paper.tex §5 (Experimental Evaluation) for experimental setup.
    Coverage guarantee: P(Y ∈ C_α(X)) ≥ 1 - α (§3, Theorem 1).
"""

import argparse
import os
import sys
import json
import numpy as np

from .predictor import (
    run_conformal_analysis,
    run_alpha_sweep,
    load_predictions_from_file,
    run_alpha_sweep_compare,
)
from .adaptive import (
    run_boosted_alpha_sweep,
    run_boosted_conformal_analysis,
)
from .data_utils import compute_normalized_width_stats


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Conformal Prediction Analysis for MeshGraphNet Predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single alpha analysis
    python run_conformal.py -f conformal_inputs/datasize_medium_0_500_conformal.pkl -a 0.1

    # Alpha sweep with normalized widths and plots
    python run_conformal.py -f conformal_inputs/datasize_medium_0_500_conformal.pkl -s 0.05,0.1,0.15,0.2 -n -p

    # Conservative analysis with comparison
    python run_conformal.py -f conformal_inputs/datasize_medium_0_500_conformal.pkl -s 0.1,0.2 -c -m 0.02 --compare

    # Quick analysis (uses defaults)
    python run_conformal.py -f conformal_inputs/datasize_medium_0_500_conformal.pkl
        """,
    )

    parser.add_argument(
        "-f",
        "--file",
        type=str,
        required=True,
        help="Path to predictions pickle file",
    )

    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=0.1,
        help="Miscoverage level (default: 0.1)",
    )

    parser.add_argument(
        "-s",
        "--sweep",
        type=str,
        help="Alpha sweep: comma-separated values (e.g., 0.05,0.1,0.2)",
    )

    parser.add_argument(
        "-r",
        "--ratio",
        type=float,
        default=0.5,
        help="Calibration ratio (default: 0.5)",
    )

    parser.add_argument(
        "--aux",
        "--aux-ratio",
        dest="aux_ratio",
        type=float,
        default=0.0,
        help="Auxiliary ratio for learning score components (default: 0.0)",
    )

    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Disable fluid node masking",
    )

    parser.add_argument("-p", "--plot", action="store_true", help="Generate plots")

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./results",
        help="Output directory (default: ./results)",
    )

    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save results to JSON (default: True)",
    )
    parser.add_argument(
        "--no-save",
        dest="save",
        action="store_false",
        help="Disable saving results to JSON",
    )

    parser.add_argument(
        "-n",
        "--normalize",
        action="store_true",
        help="Report normalized widths for literature comparison",
    )

    parser.add_argument(
        "-c",
        "--conservative",
        action="store_true",
        default=True,
        help="Use conservative quantiles (default: True)",
    )

    parser.add_argument(
        "-m",
        "--margin",
        type=float,
        default=0.01,
        help="Conservative margin (default: 0.01)",
    )

    parser.add_argument(
        "-t",
        "--type",
        type=str,
        choices=["l2", "mahalanobis", "box"],
        default="l2",
        help="Nonconformity: l2|mahalanobis|box (default: l2)",
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all nonconformity modes",
    )

    parser.add_argument(
        "-b",
        "--boosted",
        action="store_true",
        help="Run Boosted Conformal Prediction (meta-model scaling)",
    )

    parser.add_argument(
        "-e",
        "--enhanced",
        action="store_true",
        help="Use enhanced physics-informed features for BCP (improves performance)",
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        default=True,
        help="Run BCP alphas in parallel (default: True)",
    )

    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers for BCP (default: 4)",
    )
    parser.add_argument(
        "--cov-gap-tol",
        type=float,
        default=0.01,
        help="Coverage tolerance for selecting best configs (default: 0.01)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    # Check if predictions file exists
    if not os.path.exists(args.file):
        print(f"Error: Predictions file not found: {args.file}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load predictions to get metadata
    print(f"Loading predictions from: {args.file}")
    data = load_predictions_from_file(args.file)

    # Get model name and convert to clean naming convention
    old_model_name = data.get("model_name", "unknown")

    # Map old names to new clean names
    name_mapping = {
        "datasize_medium-noise_paper": "cylinder_medium_noise",
        "datasize_medium": "cylinder_medium",
        "train200_hd128": "flag_medium",
    }
    model_name = name_mapping.get(old_model_name, old_model_name)

    print(f"Model: {model_name}")
    print(f"Data shape: {data['predictions'].shape}")
    print(f"Timesteps: {data['predictions'].shape[0]}")
    print(f"Nodes: {data['predictions'].shape[1]}")

    # Determine if we should use fluid mask
    use_fluid_mask = not args.no_mask
    if use_fluid_mask and "metadata" in data and "fluid_mask" in data["metadata"]:
        fluid_nodes = np.sum(data["metadata"]["fluid_mask"])
        total_nodes = len(data["metadata"]["fluid_mask"])
        print(
            f"Fluid nodes: {fluid_nodes}/{total_nodes} ({fluid_nodes / total_nodes:.1%})"
        )
    else:
        print("No fluid mask available or disabled")

    if args.sweep and args.compare and args.boosted:
        # Combined comparison: l2/mahalanobis/box + boosted (best defaults)
        alphas = [float(x.strip()) for x in args.sweep.split(",")]
        print(f"\nRunning combined comparison over {len(alphas)} alphas...")

        rows_base = run_alpha_sweep_compare(
            args.file, alphas, args.ratio, use_fluid_mask, aux_ratio=args.aux_ratio
        )
        boosted_rows = run_boosted_alpha_sweep(
            args.file,
            alphas,
            args.ratio,
            use_fluid_mask,
            random_seed=42,
            parallel=args.parallel,
            max_workers=args.workers,
        )

        print("\n" + "=" * 110)
        print("NONCONFORMITY COMPARISON (WITH BOOSTED)")
        print("=" * 110)
        header = (
            f"{'Alpha':<8} {'Target':<8} | "
            f"{'L2_Cov':<8} {'L2_W':<10} {'L2_Area':<10} | "
            f"{'Mah_Cov':<8} {'Mah_W':<10} {'Mah_Area':<10} | "
            f"{'Box_Cov':<8} {'Box_W':<10} {'Box_Area':<10} | "
            f"{'Boost_Cov':<9} {'Boost_W':<10} {'Boost_Area':<11}"
        )
        print(header)
        print("-" * len(header))
        # rows_base and boosted_rows are aligned by alpha order
        for base, boost in zip(rows_base, boosted_rows):
            l2 = base.get("l2", {})
            mah = base.get("mahalanobis", {})
            box = base.get("box", {})
            print(
                f"{base['alpha']:<8.3f} {base['target_coverage']:<8.3f} | "
                f"{l2.get('coverage', float('nan')):<8.3f} {l2.get('mean_width', float('nan')):<10.4f} {l2.get('area', float('nan')):<10.4f} | "
                f"{mah.get('coverage', float('nan')):<8.3f} {mah.get('mean_width', float('nan')):<10.4f} {mah.get('area', float('nan')):<10.4f} | "
                f"{box.get('coverage', float('nan')):<8.3f} {box.get('mean_width', float('nan')):<10.4f} {box.get('area', float('nan')):<10.4f} | "
                f"{boost.get('empirical_coverage', float('nan')):<9.3f} {boost.get('mean_width', float('nan')):<10.4f} {boost.get('mean_area', float('nan')):<11.4f}"
            )

        print(f"\nAnalysis complete! Results saved to: {args.output}")

    elif args.boosted and args.sweep:
        # BoostedCP alpha sweep
        alphas = [float(x.strip()) for x in args.sweep.split(",")]
        print(f"\nRunning BoostedCP alpha sweep for {len(alphas)} values...")
        # Configure BCP hyperparameters based on CLI flags
        bcp_kwargs = {
            "use_enhanced_features": args.enhanced,
            "enable_expensive_features": args.enhanced,
        }

        results = run_boosted_alpha_sweep(
            args.file,
            alphas,
            args.ratio,
            use_fluid_mask,
            random_seed=42,
            predictor_kwargs=bcp_kwargs,
            aux_ratio=args.aux_ratio,
            parallel=args.parallel,
            max_workers=args.workers,
        )

        # Add normalized width support for BCP alpha sweep
        if args.normalize:
            data = load_predictions_from_file(args.file)
            for result in results:
                if "width_stats" in result:
                    result["width_stats"] = _compute_normalized_width_stats(
                        result["width_stats"], data
                    )

        print("\n" + "=" * 60)
        print("BOOSTED CONFORMAL RESULTS SUMMARY")
        print("=" * 60)
        if (
            args.normalize
            and results
            and "width_stats" in results[0]
            and "normalized_width" in results[0]["width_stats"]
        ):
            print(
                f"{'Alpha':<8} {'Target':<8} {'Empirical':<10} {'NormW':<10} {'MeanArea':<12} {'q_ratio':<10}"
            )
            print("-" * 65)
            for r in results:
                norm_width = r["width_stats"].get("normalized_width", r["mean_width"])
                print(
                    f"{r['alpha']:<8.3f} {r['target_coverage']:<8.3f} {r['empirical_coverage']:<10.3f} "
                    f"{norm_width:<10.4f} {r['mean_area']:<12.4f} {r['q_ratio']:<10.4f}"
                )
        else:
            print(
                f"{'Alpha':<8} {'Target':<8} {'Empirical':<10} {'MeanW':<10} {'MeanArea':<12} {'q_ratio':<10}"
            )
            print("-" * 60)
            for r in results:
                print(
                    f"{r['alpha']:<8.3f} {r['target_coverage']:<8.3f} {r['empirical_coverage']:<10.3f} "
                    f"{r['mean_width']:<10.4f} {r['mean_area']:<12.4f} {r['q_ratio']:<10.4f}"
                )

        if args.save:
            # Create descriptive filename based on BCP configuration
            bcp_type = "enhanced" if args.enhanced else "basic"
            results_file = os.path.join(
                args.output, f"{model_name}_boosted_{bcp_type}_alpha_sweep.json"
            )

            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj

            serializable_results = convert_numpy_types(results)

            with open(results_file, "w") as f:
                json.dump(serializable_results, f, indent=2)
            print(f"Saved BoostedCP results to: {results_file}")

        if args.plot:
            print("\nPlots for BoostedCP not implemented yet; skipping.")

        print(f"\nAnalysis complete! Results saved to: {args.output}")

    elif args.sweep and args.compare:
        # Comparison mode across nonconformity types
        alphas = [float(x.strip()) for x in args.sweep.split(",")]
        print(
            f"\nRunning comparison across nonconformity modes for {len(alphas)} alphas..."
        )
        rows = run_alpha_sweep_compare(
            args.file,
            alphas,
            args.ratio,
            use_fluid_mask,
            modes=None,
            random_seed=42,
            normalize_widths=args.normalize,
            conservative_margin=args.margin,
            aux_ratio=args.aux_ratio,
        )

        # Pretty print comparison table
        print("\n" + "=" * 100)
        print("NONCONFORMITY COMPARISON")
        print("=" * 100)

        if (
            args.normalize
            and rows
            and any(
                "normalized_width" in rows[0].get(mode, {})
                for mode in ["l2", "mahalanobis", "box"]
            )
        ):
            # Show comprehensive comparison with normalized widths
            header = (
                f"{'Alpha':<8} {'Target':<8} | "
                f"{'L2_Cov':<8} {'L2_q':<8} {'L2_NormW':<10} | "
                f"{'Mah_Cov':<8} {'Mah_q':<8} {'Mah_NormW':<10} | "
                f"{'Box_Cov':<8} {'Box_q':<8} {'Box_NormW':<10}"
            )
            print(header)
            print("-" * len(header))
            for r in rows:
                l2 = r.get("l2", {})
                mah = r.get("mahalanobis", {})
                box = r.get("box", {})
                print(
                    f"{r['alpha']:<8.3f} {r['target_coverage']:<8.3f} | "
                    f"{l2.get('coverage', float('nan')):<8.3f} {l2.get('q', float('nan')):<8.4f} {l2.get('normalized_width', l2.get('mean_width', float('nan'))):<10.4f} | "
                    f"{mah.get('coverage', float('nan')):<8.3f} {mah.get('q', float('nan')):<8.4f} {mah.get('normalized_width', mah.get('mean_width', float('nan'))):<10.4f} | "
                    f"{box.get('coverage', float('nan')):<8.3f} {box.get('q', float('nan')):<8.4f} {box.get('normalized_width', box.get('mean_width', float('nan'))):<10.4f}"
                )
        else:
            # Show raw widths and areas
            header = (
                f"{'Alpha':<8} {'Target':<8} | "
                f"{'L2_Cov':<8} {'L2_q':<10} {'L2_W':<10} {'L2_Area':<10} | "
                f"{'Mah_Cov':<8} {'Mah_q':<10} {'Mah_W':<10} {'Mah_Area':<10} | "
                f"{'Box_Cov':<8} {'Box_q':<10} {'Box_W':<10} {'Box_Area':<10}"
            )
            print(header)
            print("-" * len(header))
            for r in rows:
                l2 = r.get("l2", {})
                mah = r.get("mahalanobis", {})
                box = r.get("box", {})
                print(
                    f"{r['alpha']:<8.3f} {r['target_coverage']:<8.3f} | "
                    f"{l2.get('coverage', float('nan')):<8.3f} {l2.get('q', float('nan')):<10.4f} {l2.get('mean_width', float('nan')):<10.4f} {l2.get('area', float('nan')):<10.4f} | "
                    f"{mah.get('coverage', float('nan')):<8.3f} {mah.get('q', float('nan')):<10.4f} {mah.get('mean_width', float('nan')):<10.4f} {mah.get('area', float('nan')):<10.4f} | "
                    f"{box.get('coverage', float('nan')):<8.3f} {box.get('q', float('nan')):<10.4f} {box.get('mean_width', float('nan')):<10.4f} {box.get('area', float('nan')):<10.4f}"
                )

        if args.save:
            results_file = os.path.join(
                args.output, f"{model_name}_alpha_sweep_compare.json"
            )
            with open(results_file, "w") as f:
                json.dump(rows, f, indent=2)
            print(f"Saved comparison results to: {results_file}")

        if args.plot:
            print(
                "\nComparison plots are not implemented yet for multi-mode; skipping."
            )

        print(f"\nAnalysis complete! Results saved to: {args.output}")

    elif args.sweep:
        # Run alpha sweep analysis
        alphas = [float(x.strip()) for x in args.sweep.split(",")]
        print(f"\nRunning alpha sweep analysis for {len(alphas)} values...")
        print(f"Alpha values: {alphas}")

        results = run_alpha_sweep(
            args.file,
            alphas,
            args.ratio,
            use_fluid_mask,
            nonconformity=args.type,
            normalize_widths=args.normalize,
            conservative_margin=args.margin,
            aux_ratio=args.aux_ratio,
        )

        # Print summary
        print("\n" + "=" * 60)
        print("ALPHA SWEEP RESULTS SUMMARY")
        print("=" * 60)
        if (
            args.normalize
            and results
            and "normalized_width" in results[0]["width_stats"]
        ):
            print(
                f"{'Alpha':<8} {'Target':<8} {'Empirical':<10} {'Q-Value':<10} {'Width':<10} {'Norm_Width':<12}"
            )
            print("-" * 70)
            for result in results:
                ws = result["width_stats"]
                print(
                    f"{result['alpha']:<8.3f} {result['target_coverage']:<8.3f} "
                    f"{result['empirical_coverage']:<10.3f} {result['q_value']:<10.4f} "
                    f"{ws['mean_width']:<10.4f} {ws['normalized_width']:<12.4f}"
                )
        else:
            print(
                f"{'Alpha':<8} {'Target':<8} {'Empirical':<10} {'Q-Value':<10} {'Mean Width':<12}"
            )
            print("-" * 60)
            for result in results:
                print(
                    f"{result['alpha']:<8.3f} {result['target_coverage']:<8.3f} "
                    f"{result['empirical_coverage']:<10.3f} {result['q_value']:<10.4f} "
                    f"{result['width_stats']['mean_width']:<12.4f}"
                )

        # Save results if requested
        if args.save:
            # Create timestamped filename to prevent overrides
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(
                args.output, f"{model_name}_alpha_sweep_{timestamp}.json"
            )

            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj

            # Add comprehensive metadata for plotting
            enhanced_results = {
                "timestamp": timestamp,
                "run_configuration": {
                    "predictions_file": args.file,
                    "alphas": alphas,
                    "calib_ratio": args.ratio,
                    "use_fluid_mask": not args.no_mask,
                    "random_seed": 42,
                    "nonconformity_methods": (
                        ["l2", "mahalanobis", "box"] if args.compare else [args.type]
                    ),
                    "enhanced_features": args.enhanced if args.boosted else False,
                    "boosted": args.boosted,
                },
                "results": results,
                "metadata": {
                    "total_results": len(results),
                    "methods_analyzed": len(
                        set(r.get("nonconformity_method", "unknown") for r in results)
                    ),
                    "alpha_range": (
                        [min(alphas), max(alphas)] if len(alphas) > 1 else alphas
                    ),
                    "generated_by": "run_conformal.py",
                    "version": "enhanced_visualization_v1.0",
                },
            }

            serializable_results = convert_numpy_types(enhanced_results)

            with open(results_file, "w") as f:
                json.dump(serializable_results, f, indent=2)

            print(f"Enhanced results with visualization data saved to: {results_file}")

        print(f"\nAnalysis complete! Results saved to: {args.output}")

    else:
        # Single alpha analysis
        if args.boosted:
            print(f"\nRunning single BCP analysis for alpha = {args.alpha:.3f}...")
            # Configure BCP hyperparameters based on CLI flags
            bcp_kwargs = {
                "use_enhanced_features": args.enhanced,
                "enable_expensive_features": args.enhanced,
            }

            result = run_boosted_conformal_analysis(
                args.file,
                args.alpha,
                args.ratio,
                use_fluid_mask,
                random_seed=42,
                predictor_kwargs=bcp_kwargs,
                aux_ratio=args.aux_ratio,
            )

            # Add normalized width support for BCP
            if args.normalize and "width_stats" in result:
                data = load_predictions_from_file(args.file)
                result["width_stats"] = _compute_normalized_width_stats(
                    result["width_stats"], data
                )
        else:
            print(f"\nRunning single alpha analysis for alpha = {args.alpha:.3f}...")
            result = run_conformal_analysis(
                args.file,
                args.alpha,
                args.ratio,
                use_fluid_mask,
                nonconformity=args.type,
                normalize_widths=args.normalize,
                aux_ratio=args.aux_ratio,
            )

        # Print results
        print("\n" + "=" * 50)
        print("CONFORMAL PREDICTION ANALYSIS RESULTS")
        print("=" * 50)
        print(f"Alpha: {result['alpha']:.3f}")
        print(f"Target Coverage: {result['target_coverage']:.3f}")
        print(f"Empirical Coverage: {result['empirical_coverage']:.3f}")
        print(f"Calibrated Q-Value: {result['q_value']:.4f}")
        print(f"Calibration Steps: {result['num_calib_steps']}")
        print(f"Test Steps: {result['num_test_steps']}")
        print(f"Nodes: {result['num_nodes']}")
        print(f"Fluid Mask: {result['use_fluid_mask']}")

        print("\nInterval Width Statistics:")
        width_stats = result["width_stats"]
        print(f"  Mean: {width_stats['mean_width']:.4f}")
        print(f"  Median: {width_stats['median_width']:.4f}")
        print(f"  Std: {width_stats['std_width']:.4f}")
        print(f"  Q25: {width_stats['q25_width']:.4f}")
        print(f"  Q75: {width_stats['q75_width']:.4f}")
        print(f"  Area: {width_stats['area']:.4f}")

        # Show normalized statistics if requested
        if args.normalize and "normalized_width" in width_stats:
            print(f"\nNormalized Width Statistics (for literature comparison):")
            print(f"  Normalized Width: {width_stats['normalized_width']:.4f}")
            print(
                f"  Original Scale Factor: {width_stats['original_scale_factor']:.4f}"
            )
            print(f"  Interpretation: {width_stats['interpretation']}")

            # Compare with reference table
            reference_width = 0.1309  # From your reference table for alpha=0.10
            if abs(result["alpha"] - 0.1) < 0.01:  # If testing alpha=0.10
                ratio = width_stats["normalized_width"] / reference_width
                print(f"  Comparison with reference (alpha=0.10): {ratio:.1f}x wider")
                if ratio < 2.0:
                    print("  [OK] Comparable to literature results")
                elif ratio < 5.0:
                    print("  [WARN] Wider than literature, but reasonable")
                else:
                    print("  [ERROR] Much wider than literature results")

        print("\nCalibration Statistics:")
        calib_stats = result["calibration_stats"]
        print(f"  Mean Residual: {calib_stats['mean_residual']:.4f}")
        print(f"  Std Residual: {calib_stats['std_residual']:.4f}")
        print(f"  Min Residual: {calib_stats['min_residual']:.4f}")
        print(f"  Max Residual: {calib_stats['max_residual']:.4f}")

        # Save results if requested
        if args.save:
            # Create timestamped filename to prevent overrides
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(
                args.output, f"{model_name}_alpha_{args.alpha:.3f}_{timestamp}.json"
            )

            # Convert numpy types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj

            # Add comprehensive metadata for single alpha run
            enhanced_result = {
                "timestamp": timestamp,
                "run_configuration": {
                    "predictions_file": args.file,
                    "alpha": args.alpha,
                    "calib_ratio": args.ratio,
                    "use_fluid_mask": not args.no_mask,
                    "random_seed": 42,
                    "nonconformity_method": args.type,
                    "enhanced_features": args.enhanced if args.boosted else False,
                    "boosted": args.boosted,
                },
                "result": result,
                "metadata": {
                    "generated_by": "run_conformal.py",
                    "version": "enhanced_visualization_v1.0",
                    "analysis_type": "single_alpha",
                },
            }

            serializable_result = convert_numpy_types(enhanced_result)

            with open(results_file, "w") as f:
                json.dump(serializable_result, f, indent=2)

            print(
                f"\nEnhanced results with visualization data saved to: {results_file}"
            )

        print(f"\nAnalysis complete! Results saved to: {args.output}")


if __name__ == "__main__":
    main()
