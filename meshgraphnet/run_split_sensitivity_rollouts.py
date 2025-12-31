#!/usr/bin/env python3
"""Creates alternative data splits, filters datasets, runs rollouts to assess robustness to split variations."""

import argparse
import json
import pickle
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# if running inside a docker container, add the workspace root to the path
workspace_root = Path("/workspace")
if workspace_root.exists():
    sys.path.insert(0, str(workspace_root))

try:
    from create_splits import create_alternative_splits
except ImportError:
    create_alternative_splits = None


class SplitSensitivityRunner:
    """Split sensitivity analysis workflow."""

    def __init__(
        self,
        workspace: str = "/workspace",
        seeds: List[int] = None,
        datasets: List[str] = None,
    ):
        self.workspace = Path(workspace)
        self.seeds = seeds or [42, 123, 456, 789, 999]
        self.datasets = datasets or ["cylinder", "flag"]

        self.deepmind_dir = self.workspace / "deepmind-research"
        self.data_dir = self.workspace / "data"
        self.splits_dir = self.data_dir / "splits_alternative"
        self.results_dir = self.workspace / "rollouts_sensitivity"

        self.model_types = {"cylinder": "cfd", "flag": "cloth"}
        self.checkpoint_dirs = {
            "cylinder": self.workspace / "checkpoints_cylinder_ts",
            "flag": self.workspace / "checkpoints_flag_ts",
        }
        self.input_dirs = {
            "cylinder": self.data_dir / "cylinder_flow" / "cylinder_flow",
            "flag": self.data_dir / "flag_simple" / "flag_simple",
        }
        self.num_trajectories = {"cylinder": 28, "flag": 18}

        self.splits_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_command(self, cmd: List[str], cwd: Path = None, check: bool = True):
        """Run a shell command."""
        print(f"Running: {' '.join(cmd)}")
        if cwd:
            print(f"  (in {cwd})")

        try:
            return subprocess.run(
                cmd,
                cwd=str(cwd) if cwd else None,
                check=check,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {' '.join(cmd)}")
            print(f"  Return code: {e.returncode}")
            print(f"  stdout: {e.stdout}")
            print(f"  stderr: {e.stderr}")
            if check:
                raise
            return e

    def _get_expected_trajectories(self, rollout_path: Path) -> int:
        """Get expected number of trajectories based on dataset and split."""
        if "auxiliary" in str(rollout_path):
            return 3 if "flag" in str(rollout_path) else 5
        elif "calibration" in str(rollout_path):
            return 1 if "flag" in str(rollout_path) else 3
        else:  # test split
            return 1 if "flag" in str(rollout_path) else 3

    def _run_rollout_with_retry(
        self,
        cmd: List[str],
        rollout_dir: Path,
        rollout_path: Path,
        log_path: Path,
        split_name: str,
        max_retries: int = 2,
        timeout_minutes: int = 60,
    ) -> bool:
        """
        Run rollout with retry logic and verification.

        Returns:
            True if rollout succeeded and .pkl file was created, False otherwise
        """
        expected_trajectories = self._get_expected_trajectories(rollout_path)

        for attempt in range(max_retries + 1):
            if attempt > 0:
                print(f"      Retry attempt {attempt}/{max_retries}...")
                time.sleep(5)

            print(
                f"      Running... (attempt {attempt + 1}, timeout: {timeout_minutes} min)"
            )

            if attempt > 0 and log_path.exists():
                log_path.unlink()

            try:
                with open(log_path, "w") as log_file:
                    process = subprocess.Popen(
                        cmd,
                        cwd=str(rollout_dir),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )

                    start_time = time.time()
                    last_line_time = start_time
                    trajectories_seen = set()
                    stdout_lines = []
                    stdout_closed = threading.Event()

                    def read_stdout():
                        try:
                            for line in process.stdout:
                                stdout_lines.append(line)
                            stdout_closed.set()
                        except Exception as e:
                            log_file.write(f"\n--- Error reading stdout: {e} ---\n")
                            stdout_closed.set()

                    read_thread = threading.Thread(target=read_stdout, daemon=True)
                    read_thread.start()

                    while True:
                        if stdout_closed.is_set() and len(stdout_lines) == 0:
                            break

                        while len(stdout_lines) > 0:
                            line = stdout_lines.pop(0)
                            log_file.write(line)
                            log_file.flush()
                            last_line_time = time.time()

                            if "Rollout trajectory" in line:
                                try:
                                    parts = line.split("Rollout trajectory")
                                    if len(parts) > 1:
                                        traj_num = int(parts[1].strip())
                                        trajectories_seen.add(traj_num)
                                        print(
                                            f"        {line.strip()} "
                                            f"(seen {len(trajectories_seen)}/{expected_trajectories})"
                                        )
                                except:
                                    pass
                            elif "mse_" in line:
                                print(f"        {line.strip()}")
                                if len(trajectories_seen) >= expected_trajectories:
                                    print(
                                        f"        All {expected_trajectories} trajectories "
                                        f"completed, waiting for file save..."
                                    )
                            elif (
                                "ERROR" in line
                                or "Error" in line
                                or "Traceback" in line
                            ):
                                print(f"        WARNING: {line.strip()}")
                            elif (
                                "Killed" in line
                                or "killed" in line
                                or "SIGKILL" in line
                            ):
                                print(
                                    f"        WARNING: Process killed: {line.strip()}"
                                )

                        elapsed = time.time() - start_time
                        if elapsed > timeout_minutes * 60:
                            print(
                                f"      Timeout after {timeout_minutes} minutes, killing process..."
                            )
                            process.kill()
                            log_file.write(
                                f"\n--- TIMEOUT after {timeout_minutes} minutes ---\n"
                            )
                            return_code = process.wait()
                            break

                        if process.poll() is not None:
                            return_code = process.returncode
                            read_thread.join(timeout=5)
                            while len(stdout_lines) > 0:
                                log_file.write(stdout_lines.pop(0))
                                log_file.flush()
                            if return_code != 0:
                                log_file.write(
                                    f"\n--- Process terminated with code {return_code} ---\n"
                                )
                            break

                        if time.time() - last_line_time > 180:
                            print(
                                f"      No output for 3 minutes, checking process status..."
                            )
                            if process.poll() is None:
                                if (
                                    len(trajectories_seen) > 0
                                    and len(trajectories_seen) < expected_trajectories
                                ):
                                    print(
                                        f"      Only {len(trajectories_seen)}/{expected_trajectories} "
                                        f"trajectories seen, waiting 2 more minutes..."
                                    )
                                    time.sleep(120)
                                    if (
                                        process.poll() is None
                                        and time.time() - last_line_time > 300
                                    ):
                                        print(
                                            f"      Process hung (no output for 5+ minutes), killing..."
                                        )
                                        process.kill()
                                        return_code = process.wait()
                                        log_file.write(
                                            f"\n--- Process hung (only "
                                            f"{len(trajectories_seen)}/{expected_trajectories} "
                                            f"trajectories) ---\n"
                                        )
                                        break
                                elif len(trajectories_seen) >= expected_trajectories:
                                    print(
                                        f"      All {expected_trajectories} trajectories seen but "
                                        f"no file saved, waiting 3 more minutes..."
                                    )
                                    time.sleep(180)
                                    if (
                                        process.poll() is None
                                        and time.time() - last_line_time > 360
                                    ):
                                        print(
                                            f"      Process hung after all trajectories, killing..."
                                        )
                                        process.kill()
                                        return_code = process.wait()
                                        log_file.write(
                                            f"\n--- Process hung (no file save) ---\n"
                                        )
                                        break
                            else:
                                return_code = process.returncode
                                break

                        time.sleep(0.1)

                    if process.poll() is None:
                        print(
                            f"      Process still running, waiting up to 5 minutes..."
                        )
                        try:
                            return_code = process.wait(timeout=300)
                        except subprocess.TimeoutExpired:
                            print(f"      Process hung, killing...")
                            process.kill()
                            return_code = process.wait()
                            log_file.write(f"\n--- Process hung and was killed ---\n")
                    else:
                        return_code = process.returncode

                    read_thread.join(timeout=2)

                    if return_code != 0:
                        print(f"      Exit code: {return_code}")
                        log_file.write(
                            f"\n--- Process exited with code {return_code} ---\n"
                        )
                        if attempt < max_retries:
                            continue
                        return False

                    if rollout_path.exists():
                        file_size = rollout_path.stat().st_size
                        if file_size > 0:
                            print(
                                f"      Success! File size: {file_size / (1024*1024):.2f} MB"
                            )
                            return True
                        else:
                            print(f"      File created but empty ({file_size} bytes)")
                            rollout_path.unlink()
                            if attempt < max_retries:
                                continue
                            return False
                    else:
                        print(f"      .pkl file not created!")
                        if attempt < max_retries:
                            continue
                        return False

            except KeyboardInterrupt:
                print(f"      Interrupted by user")
                if "process" in locals() and process.poll() is None:
                    process.kill()
                return False
            except Exception as e:
                print(f"      Exception: {e}")
                with open(log_path, "a") as log_file:
                    log_file.write(
                        f"\n--- EXCEPTION ---\n{e}\n{traceback.format_exc()}\n"
                    )
                if attempt < max_retries:
                    continue
                return False

        return False

    def create_splits(self):
        """Create alternative splits."""
        print("\n" + "=" * 70)
        print("Step 1: Creating Alternative Splits")
        print("=" * 70)

        for dataset in self.datasets:
            print(f"\nCreating splits for {dataset}...")

            if create_alternative_splits:
                create_alternative_splits(
                    dataset=dataset,
                    num_trajectories=self.num_trajectories[dataset],
                    seeds=self.seeds,
                    output_dir=self.splits_dir,
                )
            else:
                cmd = (
                    [
                        "python",
                        str(self.workspace / "create_splits.py"),
                        "--dataset",
                        dataset,
                        "--num_trajectories",
                        str(self.num_trajectories[dataset]),
                        "--seeds",
                    ]
                    + [str(s) for s in self.seeds]
                    + ["--output_dir", str(self.splits_dir)]
                )
                self.run_command(cmd, cwd=self.workspace)

        print("\nAlternative splits created")

    def filter_trajectories(self):
        """Filter trajectories for each alternative split."""
        print("\n" + "=" * 70)
        print("Step 2: Filtering Trajectories")
        print("=" * 70)

        for dataset in self.datasets:
            print(f"\nFiltering {dataset} trajectories...")

            for seed in self.seeds:
                splits_file = self.splits_dir / f"{dataset}_splits_seed{seed}.json"
                output_dir = self.data_dir / f"{dataset}_flow_filtered_seed{seed}"

                if not splits_file.exists():
                    print(f"  Warning: {splits_file} not found, skipping seed {seed}")
                    continue

                print(f"  Seed {seed}: Filtering trajectories...")

                cmd = [
                    "python",
                    str(self.workspace / "filter_trajectories_tf1.py"),
                    "--splits_file",
                    str(splits_file),
                    "--input_dir",
                    str(self.input_dirs[dataset]),
                    "--output_dir",
                    str(output_dir),
                    "--dataset",
                    dataset,
                ]

                self.run_command(cmd, cwd=self.workspace)

        print("\nTrajectories filtered")

    def run_rollouts(self):
        """Run rollouts on all alternative splits."""
        print("\n" + "=" * 70)
        print("Step 3: Running Rollouts")
        print("=" * 70)

        rollout_dir = self.deepmind_dir

        if not rollout_dir.exists():
            print(f"Error: {rollout_dir} does not exist")
            return

        meshgraphnets_dir = rollout_dir / "meshgraphnets"
        if not meshgraphnets_dir.exists():
            print(f"Error: {meshgraphnets_dir} does not exist")
            return

        for dataset in self.datasets:
            model_type = self.model_types[dataset]
            checkpoint_dir = self.checkpoint_dirs[dataset]

            print(f"\nRunning rollouts for {dataset} (model: {model_type})...")

            for seed in self.seeds:
                data_dir = self.data_dir / f"{dataset}_flow_filtered_seed{seed}"

                if not data_dir.exists():
                    print(f"  Warning: {data_dir} not found, skipping seed {seed}")
                    continue

                print(f"\n  Seed {seed}:")

                num_rollouts = {
                    "auxiliary": 3 if dataset == "flag" else 5,
                    "calibration": 1 if dataset == "flag" else 3,
                    "test": 1 if dataset == "flag" else 3,
                }

                for split_name in ["auxiliary", "calibration", "test"]:
                    print(f"    - {split_name.capitalize()} split...")
                    rollout_path = (
                        self.results_dir
                        / f"rollout_{dataset}_{split_name}_seed{seed}.pkl"
                    )
                    log_path = (
                        self.results_dir
                        / f"rollout_{dataset}_{split_name}_seed{seed}.log"
                    )

                    cmd = [
                        "python",
                        "-m",
                        "meshgraphnets.run_model",
                        "--mode=eval",
                        f"--model={model_type}",
                        f"--checkpoint_dir={checkpoint_dir}",
                        f"--dataset_dir={data_dir}/{split_name}",
                        f"--rollout_path={rollout_path}",
                        f"--num_rollouts={num_rollouts[split_name]}",
                    ]

                    success = self._run_rollout_with_retry(
                        cmd, rollout_dir, rollout_path, log_path, split_name
                    )
                    if not success:
                        print(f"      Failed after retries - check {log_path}")

        print("\nRollouts complete")

    def save_summary(self):
        """Save summary of all runs."""
        summary = {
            "workspace": str(self.workspace),
            "seeds": self.seeds,
            "datasets": self.datasets,
            "splits_dir": str(self.splits_dir),
            "results_dir": str(self.results_dir),
            "rollouts": [],
        }

        for dataset in self.datasets:
            for seed in self.seeds:
                for split_name in ["auxiliary", "calibration", "test"]:
                    rollout_file = (
                        self.results_dir
                        / f"rollout_{dataset}_{split_name}_seed{seed}.pkl"
                    )
                    log_file = (
                        self.results_dir
                        / f"rollout_{dataset}_{split_name}_seed{seed}.log"
                    )

                    summary["rollouts"].append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "split": split_name,
                            "rollout_file": str(rollout_file),
                            "log_file": str(log_file),
                            "exists": rollout_file.exists(),
                        }
                    )

        summary_file = self.results_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nSummary saved to: {summary_file}")

    def _compute_mse(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Compute mean squared error."""
        return np.mean((pred - gt) ** 2)

    def _compute_metrics_from_rollout(
        self, rollout_data: List[Dict], field: str = "velocity"
    ) -> Optional[Dict[str, float]]:
        """
        Compute metrics from rollout data.

        Args:
            rollout_data: List of trajectory dictionaries from rollout
            field: Field to analyze ('velocity' for CFD, 'pos' for Flag)

        Returns:
            Dictionary with metrics or None
        """
        if not rollout_data:
            return None

        all_mse = []
        all_rmse = []
        available_keys = list(rollout_data[0].keys())

        field_to_keys = {
            "velocity": ("pred_velocity", "gt_velocity"),
            "pos": ("pred_pos", "gt_pos"),
            "world_pos": ("pred_world_pos", "gt_world_pos"),
        }

        pred_key, gt_key = field_to_keys.get(field, ("pred_velocity", "gt_velocity"))

        for traj in rollout_data:
            if pred_key not in traj:
                possible_keys = [
                    k
                    for k in traj.keys()
                    if "pred" in k.lower()
                    and ("pos" in k.lower() or "velocity" in k.lower())
                ]
                if possible_keys:
                    pred_key = possible_keys[0]
                else:
                    print(
                        f"Warning: {pred_key} not found. Available keys: {available_keys}"
                    )
                    continue

            if gt_key not in traj:
                possible_keys = [
                    k
                    for k in traj.keys()
                    if ("gt" in k.lower() or "target" in k.lower())
                    and ("pos" in k.lower() or "velocity" in k.lower())
                ]
                if possible_keys:
                    gt_key = possible_keys[0]
                else:
                    print(
                        f"Warning: {gt_key} not found. Available keys: {available_keys}"
                    )
                    continue

            pred = traj[pred_key]
            gt = traj[gt_key]

            if not (isinstance(pred, np.ndarray) and isinstance(gt, np.ndarray)):
                continue

            min_steps = min(len(pred), len(gt))
            pred = pred[:min_steps]
            gt = gt[:min_steps]

            for step in range(min_steps):
                step_mse = self._compute_mse(pred[step], gt[step])
                step_rmse = np.sqrt(step_mse)
                all_mse.append(step_mse)
                all_rmse.append(step_rmse)

        if not all_mse:
            return None

        return {
            "mean_mse": np.mean(all_mse),
            "std_mse": np.std(all_mse),
            "mean_rmse": np.mean(all_rmse),
            "std_rmse": np.std(all_rmse),
            "min_rmse": np.min(all_rmse),
            "max_rmse": np.max(all_rmse),
            "num_samples": len(all_mse),
        }

    def analyze_results(self):
        """Analyze rollout results and compute split sensitivity metrics."""
        summary_file = self.results_dir / "summary.json"

        # Try to load summary.json if it exists, otherwise use defaults
        if summary_file.exists():
            with open(summary_file, "r") as f:
                summary = json.load(f)
            datasets = self.datasets or summary.get("datasets", [])
            seeds = self.seeds or summary.get("seeds", [])
        else:
            print(f"Note: Summary file not found at {summary_file}")
            print(f"Using default datasets: {self.datasets}, seeds: {self.seeds}")
            datasets = self.datasets
            seeds = self.seeds

        print("\n" + "=" * 70)
        print("SPLIT SENSITIVITY ANALYSIS")
        print("=" * 70)
        print(f"\nDatasets: {datasets}")
        print(f"Seeds: {seeds}")
        print(f"Results directory: {self.results_dir}")

        field_map = {"cylinder": "velocity", "flag": "pos"}
        splits = ["auxiliary", "calibration", "test"]

        for dataset in datasets:
            print("\n" + "=" * 70)
            print(f"DATASET: {dataset.upper()}")
            print("=" * 70)

            field = field_map.get(dataset, "velocity")
            split_metrics = {split: [] for split in splits}

            for split in splits:
                print(f"\n{split.upper()} Split:")
                print("-" * 70)

                for seed in seeds:
                    rollout_file = (
                        self.results_dir / f"rollout_{dataset}_{split}_seed{seed}.pkl"
                    )

                    if not rollout_file.exists():
                        print(f"  Seed {seed}: File not found")
                        continue

                    try:
                        with open(rollout_file, "rb") as f:
                            rollout_data = pickle.load(f)

                        metrics = self._compute_metrics_from_rollout(
                            rollout_data, field=field
                        )

                        if metrics:
                            split_metrics[split].append(metrics)
                            print(
                                f"  Seed {seed:3d}: RMSE = {metrics['mean_rmse']:.6f} "
                                f"± {metrics['std_rmse']:.6f}"
                            )
                            print(
                                f"           Samples: {metrics['num_samples']}, "
                                f"Range: [{metrics['min_rmse']:.6f}, {metrics['max_rmse']:.6f}]"
                            )
                    except Exception as e:
                        print(f"  Seed {seed}: Error loading/analyzing - {e}")

                if split_metrics[split]:
                    mean_rmse_values = [m["mean_rmse"] for m in split_metrics[split]]

                    print(f"\n  Across {len(split_metrics[split])} seeds:")
                    print(
                        f"    Mean RMSE: {np.mean(mean_rmse_values):.6f} "
                        f"± {np.std(mean_rmse_values):.6f}"
                    )
                    print(
                        f"    RMSE Range: [{np.min(mean_rmse_values):.6f}, "
                        f"{np.max(mean_rmse_values):.6f}]"
                    )

                    cv = (
                        np.std(mean_rmse_values) / np.mean(mean_rmse_values) * 100
                        if np.mean(mean_rmse_values) > 0
                        else 0
                    )
                    print(f"    Coefficient of Variation: {cv:.2f}%")

            print("\n" + "=" * 70)
            print(f"SUMMARY: {dataset.upper()}")
            print("=" * 70)
            print(
                f"{'Split':<15} {'Mean RMSE':<15} {'Std RMSE':<15} "
                f"{'CV (%)':<10} {'Seeds':<10}"
            )
            print("-" * 70)

            for split in splits:
                if split_metrics[split]:
                    mean_rmse_values = [m["mean_rmse"] for m in split_metrics[split]]
                    mean_val = np.mean(mean_rmse_values)
                    std_val = np.std(mean_rmse_values)
                    cv = (std_val / mean_val * 100) if mean_val > 0 else 0
                    num_seeds = len(split_metrics[split])

                    print(
                        f"{split:<15} {mean_val:<15.6f} {std_val:<15.6f} "
                        f"{cv:<10.2f} {num_seeds:<10}"
                    )

        print("\n" + "=" * 70)
        print("INTERPRETATION")
        print("=" * 70)
        print("\nLow Coefficient of Variation (CV < 5%):")
        print("  Results are robust to split variations")
        print("\nModerate CV (5-15%):")
        print("  Some sensitivity to splits, but acceptable")
        print("\nHigh CV (>15%):")
        print("  High sensitivity - consider reporting variability")
        print("\n" + "=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run split sensitivity analysis for conformal prediction"
    )
    parser.add_argument(
        "--workspace", type=str, default="/workspace", help="Workspace directory"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456, 789, 999],
        help="Random seeds for alternative splits",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        choices=["cylinder", "flag"],
        help="Datasets to process (default: both)",
    )
    parser.add_argument(
        "--all_datasets",
        action="store_true",
        help="Process all datasets (cylinder and flag)",
    )
    parser.add_argument(
        "--skip_splits", action="store_true", help="Skip creating splits (use existing)"
    )
    parser.add_argument(
        "--skip_filter",
        action="store_true",
        help="Skip filtering trajectories (use existing)",
    )
    parser.add_argument(
        "--skip_rollouts", action="store_true", help="Skip running rollouts"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run analysis after rollouts (or run analysis only if --skip_rollouts)",
    )

    args = parser.parse_args()

    if args.all_datasets or args.datasets is None:
        datasets = ["cylinder", "flag"]
    else:
        datasets = args.datasets

    runner = SplitSensitivityRunner(
        workspace=args.workspace, seeds=args.seeds, datasets=datasets
    )

    if args.skip_splits and args.skip_filter and args.skip_rollouts:
        if args.analyze:
            runner.analyze_results()
        else:
            runner.save_summary()
    else:
        if not args.skip_splits:
            runner.create_splits()
        if not args.skip_filter:
            runner.filter_trajectories()
        if not args.skip_rollouts:
            runner.run_rollouts()
        runner.save_summary()

        if args.analyze:
            runner.analyze_results()
