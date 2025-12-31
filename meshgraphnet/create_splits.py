#!/usr/bin/env python3
"""Creates single or multiple alternative data splits for downstream conformal prediction."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _create_splits(
    dataset_name: str,
    num_trajectories: int,
    train_ratio: float,
    aux_ratio: float,
    cal_ratio: float,
    timesteps_per_traj: int,
    train_timesteps: int,
    aux_timesteps: int,
    cal_timesteps: int,
    test_timesteps: int,
    seed: int,
    *,
    fixed_train_ids: Optional[List[int]] = None,
    nontrain_timestep_range: Optional[Tuple[int, int]] = None,
    aux_n: Optional[int] = None,
    cal_n: Optional[int] = None,
    test_n: Optional[int] = None,
) -> Dict:
    """Create splits for a dataset."""
    np.random.seed(seed)

    if fixed_train_ids is None:
        n_train = max(1, int(num_trajectories * train_ratio))
    else:
        # Preserve an existing training trajectory assignment (e.g., to stay compatible
        # with a checkpoint trained on a particular filtered dataset).
        fixed_train_ids = sorted(set(int(x) for x in fixed_train_ids))
        if len(fixed_train_ids) == 0:
            raise ValueError("fixed_train_ids must be non-empty if provided.")
        if min(fixed_train_ids) < 0 or max(fixed_train_ids) >= num_trajectories:
            raise ValueError(
                f"fixed_train_ids must be within [0, {num_trajectories-1}] "
                f"(got min={min(fixed_train_ids)}, max={max(fixed_train_ids)})."
            )
        n_train = len(fixed_train_ids)
    if aux_n is not None:
        n_aux = int(aux_n)
    else:
        n_aux = max(1, int(num_trajectories * aux_ratio))

    if cal_n is not None:
        n_cal = int(cal_n)
    else:
        n_cal = max(1, int(num_trajectories * cal_ratio))

    # test_n defaults to "the remainder" (at least 1).
    remaining_after = num_trajectories - n_train - n_aux - n_cal
    if test_n is not None:
        n_test = int(test_n)
    else:
        n_test = max(1, remaining_after)

    # Sanity: we must have at least one test trajectory, and we can't exceed total count.
    # If the user overspecifies counts, fail fast (better than silently truncating).
    if n_aux < 1 or n_cal < 1 or n_test < 1 or n_train < 1:
        raise ValueError(
            f"Invalid split sizes: train={n_train}, aux={n_aux}, cal={n_cal}, test={n_test} "
            "(all must be >= 1)."
        )
    if n_train + n_aux + n_cal + n_test > num_trajectories:
        raise ValueError(
            f"Requested split sizes sum to {n_train + n_aux + n_cal + n_test}, "
            f"but num_trajectories={num_trajectories}."
        )

    if fixed_train_ids is None:
        all_traj_ids = np.arange(num_trajectories)
        np.random.shuffle(all_traj_ids)

        train_ids = sorted(all_traj_ids[:n_train].tolist())
        aux_ids = sorted(all_traj_ids[n_train : n_train + n_aux].tolist())
        cal_ids = sorted(
            all_traj_ids[n_train + n_aux : n_train + n_aux + n_cal].tolist()
        )
        test_ids = sorted(all_traj_ids[n_train + n_aux + n_cal :].tolist())
    else:
        train_ids = fixed_train_ids
        remaining = np.array(
            [i for i in range(num_trajectories) if i not in set(train_ids)]
        )
        np.random.shuffle(remaining)
        aux_ids = sorted(remaining[:n_aux].tolist())
        cal_ids = sorted(remaining[n_aux : n_aux + n_cal].tolist())
        test_ids = sorted(remaining[n_aux + n_cal : n_aux + n_cal + n_test].tolist())

    if nontrain_timestep_range is None:
        aux_start = train_timesteps
        cal_start = aux_start + aux_timesteps
        test_start = cal_start + cal_timesteps

        aux_range = (aux_start, aux_start + aux_timesteps)
        cal_range = (cal_start, cal_start + cal_timesteps)
        test_range = (test_start, test_start + test_timesteps)
    else:
        s, e = (int(nontrain_timestep_range[0]), int(nontrain_timestep_range[1]))
        if not (0 <= s < e <= timesteps_per_traj):
            raise ValueError(
                f"nontrain_timestep_range must be within [0, {timesteps_per_traj}] "
                f"and have start < end (got {(s, e)})."
            )
        aux_range = (s, e)
        cal_range = (s, e)
        test_range = (s, e)

    splits = {
        "train": {
            "trajectory_ids": train_ids,
            "timestep_range": (0, train_timesteps),
            "num_timesteps": train_timesteps,
            "num_trajectories": len(train_ids),
            "total_samples": len(train_ids) * train_timesteps,
        },
        "auxiliary": {
            "trajectory_ids": aux_ids,
            "timestep_range": aux_range,
            "num_timesteps": int(aux_range[1] - aux_range[0]),
            "num_trajectories": len(aux_ids),
            "total_samples": len(aux_ids) * int(aux_range[1] - aux_range[0]),
        },
        "calibration": {
            "trajectory_ids": cal_ids,
            "timestep_range": cal_range,
            "num_timesteps": int(cal_range[1] - cal_range[0]),
            "num_trajectories": len(cal_ids),
            "total_samples": len(cal_ids) * int(cal_range[1] - cal_range[0]),
        },
        "test": {
            "trajectory_ids": test_ids,
            "timestep_range": test_range,
            "num_timesteps": int(test_range[1] - test_range[0]),
            "num_trajectories": len(test_ids),
            "total_samples": len(test_ids) * int(test_range[1] - test_range[0]),
        },
        "metadata": {
            "dataset": dataset_name,
            "total_trajectories": num_trajectories,
            "timesteps_per_trajectory": timesteps_per_traj,
            "seed": seed,
            "fixed_train_ids_source": None,
            "nontrain_timestep_range": (
                list(nontrain_timestep_range) if nontrain_timestep_range else None
            ),
        },
    }

    return splits


def create_splits_cylinder(
    num_trajectories: int = 28,
    train_ratio: float = 0.64,
    aux_ratio: float = 0.18,
    cal_ratio: float = 0.11,
    timesteps_per_traj: int = 600,
    train_timesteps: int = 400,
    aux_timesteps: int = 120,
    cal_timesteps: int = 50,
    test_timesteps: int = 30,
    seed: int = 42,
    fixed_train_ids: Optional[List[int]] = None,
    nontrain_timestep_range: Optional[Tuple[int, int]] = None,
    aux_n: Optional[int] = None,
    cal_n: Optional[int] = None,
    test_n: Optional[int] = None,
) -> Dict:
    """Create splits for CylinderFlow dataset."""
    return _create_splits(
        dataset_name="cylinder_flow",
        num_trajectories=num_trajectories,
        train_ratio=train_ratio,
        aux_ratio=aux_ratio,
        cal_ratio=cal_ratio,
        timesteps_per_traj=timesteps_per_traj,
        train_timesteps=train_timesteps,
        aux_timesteps=aux_timesteps,
        cal_timesteps=cal_timesteps,
        test_timesteps=test_timesteps,
        seed=seed,
        fixed_train_ids=fixed_train_ids,
        nontrain_timestep_range=nontrain_timestep_range,
        aux_n=aux_n,
        cal_n=cal_n,
        test_n=test_n,
    )


def create_splits_flag(
    num_trajectories: int = 18,
    train_ratio: float = 0.67,
    aux_ratio: float = 0.22,
    cal_ratio: float = 0.11,
    timesteps_per_traj: int = 300,
    train_timesteps: int = 200,
    aux_timesteps: int = 60,
    cal_timesteps: int = 30,
    test_timesteps: int = 10,
    seed: int = 42,
    fixed_train_ids: Optional[List[int]] = None,
    nontrain_timestep_range: Optional[Tuple[int, int]] = None,
    aux_n: Optional[int] = None,
    cal_n: Optional[int] = None,
    test_n: Optional[int] = None,
) -> Dict:
    """Create splits for Flag dataset."""
    return _create_splits(
        dataset_name="flag_simple",
        num_trajectories=num_trajectories,
        train_ratio=train_ratio,
        aux_ratio=aux_ratio,
        cal_ratio=cal_ratio,
        timesteps_per_traj=timesteps_per_traj,
        train_timesteps=train_timesteps,
        aux_timesteps=aux_timesteps,
        cal_timesteps=cal_timesteps,
        test_timesteps=test_timesteps,
        seed=seed,
        fixed_train_ids=fixed_train_ids,
        nontrain_timestep_range=nontrain_timestep_range,
        aux_n=aux_n,
        cal_n=cal_n,
        test_n=test_n,
    )


def print_summary(splits: Dict) -> None:
    """Print split summary."""
    print("\n" + "=" * 60)
    print("DATA SPLIT SUMMARY")
    print("=" * 60)

    metadata = splits["metadata"]
    print(f"\nDataset: {metadata['dataset']}")
    print(f"Total trajectories: {metadata['total_trajectories']}")

    for split_name in ["train", "auxiliary", "calibration", "test"]:
        split = splits[split_name]
        print(f"\n{split_name.upper()}:")
        print(f"  Trajectories: {split['trajectory_ids']}")
        print(f"  Timestep range: {split['timestep_range']}")
        print(f"  Total samples: {split['total_samples']:,}")

    aux_samples = splits["auxiliary"]["total_samples"]
    # Minimal sanity check on auxiliary split size (trajectory × timestep count).
    # NOTE: downstream (auxiliary) learning is node×timestep, so this is only a coarse guardrail.
    min_aux = 480 if metadata["dataset"] == "cylinder_flow" else 180
    status = "OK" if aux_samples >= min_aux else "WARNING"
    print(f"\n{status} Auxiliary split: {aux_samples} samples (need >= {min_aux})")
    print("=" * 60 + "\n")


def create_alternative_splits(
    dataset: str, num_trajectories: int, seeds: List[int], output_dir: Path
) -> List[Dict]:
    """
    Create alternative splits with different seeds for sensitivity analysis.

    Args:
        dataset: 'cylinder' or 'flag'
        num_trajectories: Total number of trajectories
        seeds: List of random seeds to use
        output_dir: Directory to save split JSON files

    Returns:
        List of created split dictionaries
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"Creating Alternative Splits for {dataset.upper()}")
    print(f"{'=' * 70}")
    print(f"Trajectories: {num_trajectories}, Seeds: {seeds}")
    print(f"Output: {output_dir}\n")

    created_splits = []

    for seed in seeds:
        if dataset == "cylinder":
            splits = create_splits_cylinder(
                num_trajectories=num_trajectories, seed=seed
            )
        elif dataset == "flag":
            splits = create_splits_flag(num_trajectories=num_trajectories, seed=seed)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        output_file = output_dir / f"{dataset}_splits_seed{seed}.json"
        with open(output_file, "w") as f:
            json.dump(splits, f, indent=2)

        print_summary(splits)
        print(f"Saved to: {output_file}\n")

        created_splits.append(
            {"seed": seed, "file": str(output_file), "splits": splits}
        )

    summary_file = output_dir / f"{dataset}_splits_summary.json"
    summary = {
        "dataset": dataset,
        "num_trajectories": num_trajectories,
        "seeds": seeds,
        "splits": [
            {
                "seed": s["seed"],
                "file": s["file"],
                "train_trajectories": s["splits"]["train"]["trajectory_ids"],
                "aux_trajectories": s["splits"]["auxiliary"]["trajectory_ids"],
                "cal_trajectories": s["splits"]["calibration"]["trajectory_ids"],
                "test_trajectories": s["splits"]["test"]["trajectory_ids"],
            }
            for s in created_splits
        ],
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"{'=' * 70}")
    print(f"Created {len(seeds)} alternative splits")
    print(f"Summary: {summary_file}")
    print(f"{'=' * 70}\n")

    return created_splits


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create data splits for conformal prediction"
    )
    parser.add_argument(
        "--dataset", choices=["cylinder", "flag"], required=True, help="Dataset name"
    )
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=None,
        help="Total number of trajectories",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=None,
        help="Trajectory fraction for train split (default: dataset-specific).",
    )
    parser.add_argument(
        "--aux_ratio",
        type=float,
        default=None,
        help="Trajectory fraction for auxiliary split (default: dataset-specific).",
    )
    parser.add_argument(
        "--cal_ratio",
        type=float,
        default=None,
        help="Trajectory fraction for calibration split (default: dataset-specific).",
    )
    parser.add_argument(
        "--timesteps_per_traj",
        type=int,
        default=None,
        help="Total timesteps per trajectory in the raw dataset (default: dataset-specific).",
    )
    parser.add_argument(
        "--train_timesteps",
        type=int,
        default=None,
        help="Timesteps allocated to train window (default: dataset-specific).",
    )
    parser.add_argument(
        "--aux_timesteps",
        type=int,
        default=None,
        help="Timesteps allocated to auxiliary window (default: dataset-specific).",
    )
    parser.add_argument(
        "--cal_timesteps",
        type=int,
        default=None,
        help="Timesteps allocated to calibration window (default: dataset-specific).",
    )
    parser.add_argument(
        "--test_timesteps",
        type=int,
        default=None,
        help="Timesteps allocated to test/eval window (default: dataset-specific).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="splits.json",
        help="Output file path (for single split)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (for single split)"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", help="Multiple seeds for alternative splits"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for alternative splits (required if --seeds provided)",
    )
    parser.add_argument(
        "--keep_train_from",
        type=str,
        default=None,
        help="Path to an existing splits JSON; if provided, reuse its train.trajectory_ids "
        "and sample aux/cal/test from the remaining trajectories (useful to stay compatible "
        "with an already-trained checkpoint).",
    )
    parser.add_argument(
        "--nontrain_timestep_range",
        type=int,
        nargs=2,
        default=None,
        metavar=("START", "END"),
        help="If provided, use the same timestep range [START, END) for auxiliary, calibration, "
        "and test splits (disjointness is then enforced only by trajectory IDs). This often makes "
        "cal and eval more comparable and can move empirical coverage closer to nominal.",
    )
    parser.add_argument(
        "--aux_n",
        type=int,
        default=None,
        help="Explicit number of auxiliary trajectories (overrides --aux_ratio).",
    )
    parser.add_argument(
        "--cal_n",
        type=int,
        default=None,
        help="Explicit number of calibration trajectories (overrides --cal_ratio).",
    )
    parser.add_argument(
        "--test_n",
        type=int,
        default=None,
        help="Explicit number of test/eval trajectories (defaults to remainder).",
    )

    args = parser.parse_args()

    if args.num_trajectories is None:
        args.num_trajectories = 28 if args.dataset == "cylinder" else 18

    if args.seeds:
        if not args.output_dir:
            parser.error("--output_dir is required when using --seeds")

        create_alternative_splits(
            dataset=args.dataset,
            num_trajectories=args.num_trajectories,
            seeds=args.seeds,
            output_dir=Path(args.output_dir),
        )
    else:
        common_kwargs = dict(
            num_trajectories=args.num_trajectories,
            seed=args.seed,
        )
        # Optional overrides (only pass if explicitly provided).
        if args.train_ratio is not None:
            common_kwargs["train_ratio"] = args.train_ratio
        if args.aux_ratio is not None:
            common_kwargs["aux_ratio"] = args.aux_ratio
        if args.cal_ratio is not None:
            common_kwargs["cal_ratio"] = args.cal_ratio
        if args.timesteps_per_traj is not None:
            common_kwargs["timesteps_per_traj"] = args.timesteps_per_traj
        if args.train_timesteps is not None:
            common_kwargs["train_timesteps"] = args.train_timesteps
        if args.aux_timesteps is not None:
            common_kwargs["aux_timesteps"] = args.aux_timesteps
        if args.cal_timesteps is not None:
            common_kwargs["cal_timesteps"] = args.cal_timesteps
        if args.test_timesteps is not None:
            common_kwargs["test_timesteps"] = args.test_timesteps
        if args.nontrain_timestep_range is not None:
            common_kwargs["nontrain_timestep_range"] = tuple(
                args.nontrain_timestep_range
            )
        if args.aux_n is not None:
            common_kwargs["aux_n"] = args.aux_n
        if args.cal_n is not None:
            common_kwargs["cal_n"] = args.cal_n
        if args.test_n is not None:
            common_kwargs["test_n"] = args.test_n

        fixed_train_ids = None
        if args.keep_train_from:
            p = Path(args.keep_train_from)
            keep = json.loads(p.read_text())
            fixed_train_ids = keep.get("train", {}).get("trajectory_ids", None)
            if fixed_train_ids is None:
                raise SystemExit(
                    f"--keep_train_from file has no train.trajectory_ids: {p}"
                )
            common_kwargs["fixed_train_ids"] = fixed_train_ids

        if args.dataset == "cylinder":
            splits = create_splits_cylinder(**common_kwargs)
        else:
            splits = create_splits_flag(**common_kwargs)

        if args.keep_train_from:
            splits["metadata"]["fixed_train_ids_source"] = str(
                Path(args.keep_train_from)
            )

        print_summary(splits)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(splits, f, indent=2)
        print(f"Saved splits to {output_path}")


if __name__ == "__main__":
    main()
