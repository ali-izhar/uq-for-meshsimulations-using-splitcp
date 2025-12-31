#!/usr/bin/env python3
"""Compute simple rollout metrics from `.pkl` files (no inference).

Outputs:
- CSV with per-timestep RMSE
- PNG plot of RMSE vs timestep
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from meshgraphnet.utils.rollout_io import infer_keys, load_rollouts


def _rmse_t(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """RMSE over nodes+components per timestep. pred/gt: (T,N,D). returns (T,)."""
    err2 = (pred - gt) ** 2
    return np.sqrt(np.mean(err2, axis=(1, 2)))


def main():
    ap = argparse.ArgumentParser(description="Compute RMSE(t) from rollout pkls")
    ap.add_argument("--rollout_pkl", action="append", required=True)
    ap.add_argument("--traj_idx", type=int, default=0)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_png", required=True)
    args = ap.parse_args()

    series: List[Tuple[str, np.ndarray]] = []
    for pkl in args.rollout_pkl:
        traj = load_rollouts(pkl)[int(args.traj_idx)]
        keys = infer_keys(traj)
        pred = np.asarray(traj[keys.pred])
        gt = np.asarray(traj[keys.gt])
        T = min(pred.shape[0], gt.shape[0])
        rmse = _rmse_t(pred[:T], gt[:T])
        series.append((Path(pkl).name, rmse))

    # Write CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    maxT = max(len(r) for _, r in series)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestep"] + [name for name, _ in series])
        for t in range(maxT):
            row = [t]
            for _, rmse in series:
                row.append("" if t >= len(rmse) else float(rmse[t]))
            w.writerow(row)

    # Plot
    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    for name, rmse in series:
        plt.plot(rmse, label=name)
    plt.xlabel("timestep")
    plt.ylabel("RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
