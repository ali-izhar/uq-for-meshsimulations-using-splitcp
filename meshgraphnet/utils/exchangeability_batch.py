#!/usr/bin/env python3
"""Batch exchangeability diagnostics over many rollout `.pkl` files.

Why this exists:
- `meshgraphnet.utils.exchangeability` is a single-rollout/trajectory diagnostic.
- Reviewers often ask: "is this dependence pervasive or cherry-picked?"

This script computes the same core diagnostics across *all* trajectories for:
- `meshgraphnet/rollouts_200k/*.pkl` (base rollouts)
- `meshgraphnet/rollouts_sensitivity/*.pkl` (split sensitivity rollouts)

Outputs (in --out_dir):
- `exchangeability_batch.csv` (one row per (file, traj_idx))
- Summary plots:
  - `acf_lag1_by_group.png`
  - `ks_by_group.png`
  - `moransI_mean_by_group.png`
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from meshgraphnet.utils.rollout_io import get_pred_gt, load_rollouts
from meshgraphnet.utils.plot_style import apply_style, savefig


_RE_SENS = re.compile(
    r"rollout_(?P<dataset>cylinder|flag)_(?P<split>auxiliary|calibration|test)_seed(?P<seed>\d+)\.pkl$"
)
_RE_BASE = re.compile(
    r"rollout_(?P<dataset>cylinder|flag)_(?P<split>auxiliary|calibration|test)_200k\.pkl$"
)


def _edges_from_tris(tris: np.ndarray) -> np.ndarray:
    """Unique undirected edges from (F,3) triangles. Returns (E,2) int."""
    tris = np.asarray(tris, dtype=np.int64)
    e = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e.sort(axis=1)
    e = np.unique(e, axis=0)
    return e.astype(np.int32)


def _morans_i(x: np.ndarray, edges: np.ndarray) -> float:
    """Moran's I for scalar field x over an unweighted adjacency from edges."""
    x = np.asarray(x, dtype=float).reshape(-1)
    n = x.size
    if n < 3:
        return float("nan")
    x0 = x - x.mean()
    denom = float(np.sum(x0**2))
    if denom == 0.0:
        return float("nan")
    i, j = edges[:, 0], edges[:, 1]
    w_sum = float(edges.shape[0] * 2)  # symmetric weights
    num = float(np.sum(x0[i] * x0[j]) + np.sum(x0[j] * x0[i]))
    return (n / w_sum) * (num / denom)


def _autocorr(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    if lag <= 0 or lag >= len(x):
        return float("nan")
    x0 = x - x.mean()
    a = x0[:-lag]
    b = x0[lag:]
    denom = float(np.sqrt(np.sum(a**2) * np.sum(b**2)))
    if denom == 0:
        return float("nan")
    return float(np.sum(a * b) / denom)


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """KS statistic without scipy (two-sample)."""
    a = np.sort(np.asarray(a, dtype=float).reshape(-1))
    b = np.sort(np.asarray(b, dtype=float).reshape(-1))
    if a.size == 0 or b.size == 0:
        return float("nan")
    data = np.unique(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, data, side="right") / a.size
    cdf_b = np.searchsorted(b, data, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _infer_group(pkl_name: str) -> Tuple[str, str, str]:
    """Return (dataset, split, seed) where seed is '200k' for base rollouts."""
    m = _RE_SENS.match(pkl_name)
    if m:
        return m.group("dataset"), m.group("split"), m.group("seed")
    m = _RE_BASE.match(pkl_name)
    if m:
        return m.group("dataset"), m.group("split"), "200k"
    return "unknown", "unknown", "unknown"


def _boxplot_by_group(
    values: Dict[str, List[float]],
    *,
    title: str,
    ylabel: str,
    out_path: Path,
    formats: List[str],
) -> None:
    labels = list(values.keys())
    data = [values[k] for k in labels]
    # Only a handful of groups (dataset/split), keep compact.
    fig_w = max(3.35, 0.55 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, 2.1))
    ax.boxplot(data, tick_labels=labels, showfliers=True)
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylabel(ylabel)
    fig.tight_layout(pad=0.15)
    savefig(fig, out_path, formats=formats)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Batch exchangeability diagnostics over rollouts"
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--style", default="paper", choices=["paper", "default"])
    ap.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        help="Output formats (e.g. png pdf). Default: png",
    )
    ap.add_argument("--base_fontsize", type=float, default=9.0)
    ap.add_argument(
        "--rollouts_dirs",
        nargs="+",
        default=["meshgraphnet/rollouts_200k", "meshgraphnet/rollouts_sensitivity"],
        help="One or more dirs to scan for *.pkl rollouts.",
    )
    ap.add_argument("--acf_lags", type=int, nargs="+", default=[1, 2, 5, 10])
    ap.add_argument(
        "--morans_stride",
        type=int,
        default=2,
        help="Compute Moran's I every k timesteps and average (speed/robustness tradeoff).",
    )
    args = ap.parse_args()

    apply_style(args.style, base_fontsize=float(args.base_fontsize))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pkls: List[Path] = []
    for d in args.rollouts_dirs:
        pkls.extend(sorted(Path(d).glob("*.pkl")))
    if not pkls:
        raise SystemExit("No *.pkl rollouts found.")

    rows: List[Dict[str, str]] = []

    for pkl in pkls:
        dataset, split, seed = _infer_group(pkl.name)
        rollouts = load_rollouts(pkl)
        for ti, traj in enumerate(rollouts):
            pred, gt, _ = get_pred_gt(traj)
            T = int(pred.shape[0])

            rmse = np.sqrt(np.mean((pred - gt) ** 2, axis=(1, 2)))
            third = max(1, T // 3)
            ks = _ks_statistic(rmse[:third], rmse[-third:])

            # temporal dependence
            ac: Dict[str, float] = {
                f"acf_lag{lag}": _autocorr(rmse, int(lag)) for lag in args.acf_lags
            }

            # spatial dependence over time (mean Moran's I)
            err_mag = np.linalg.norm(gt - pred, axis=-1)  # (T,N)
            tris0 = np.asarray(traj["faces"])[0].astype(np.int32)
            edges = _edges_from_tris(tris0)
            stride = int(max(1, args.morans_stride))
            ts = range(0, T, stride)
            morans_vals = [_morans_i(err_mag[t], edges) for t in ts]
            morans_mean = (
                float(np.nanmean(morans_vals)) if morans_vals else float("nan")
            )

            row: Dict[str, str] = {
                "file": pkl.name,
                "dataset": dataset,
                "split": split,
                "seed": seed,
                "traj_idx": str(ti),
                "timesteps": str(T),
                "ks_rmse_early_vs_late": f"{ks:.6g}",
                "moransI_errmag_mean": f"{morans_mean:.6g}",
            }
            for k, v in ac.items():
                row[k] = f"{v:.6g}"
            rows.append(row)

    out_csv = out_dir / "exchangeability_batch.csv"
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Summary plots by group = dataset/split (aggregate across seeds and trajectories)
    def gkey(r: Dict[str, str]) -> str:
        return f"{r['dataset']}/{r['split']}"

    by_group_acf1: Dict[str, List[float]] = {}
    by_group_ks: Dict[str, List[float]] = {}
    by_group_mi: Dict[str, List[float]] = {}
    for r in rows:
        k = gkey(r)
        by_group_acf1.setdefault(k, []).append(float(r.get("acf_lag1", "nan")))
        by_group_ks.setdefault(k, []).append(
            float(r.get("ks_rmse_early_vs_late", "nan"))
        )
        by_group_mi.setdefault(k, []).append(float(r.get("moransI_errmag_mean", "nan")))

    for d in (by_group_acf1, by_group_ks, by_group_mi):
        for k in list(d.keys()):
            d[k] = [x for x in d[k] if np.isfinite(x)]

    _boxplot_by_group(
        dict(sorted(by_group_acf1.items())),
        title="Temporal dependence across rollouts: ACF lag 1 (RMSE(t))",
        ylabel="ACF(lag=1)",
        out_path=out_dir / "acf_lag1_by_group.png",
        formats=list(args.formats),
    )
    _boxplot_by_group(
        dict(sorted(by_group_ks.items())),
        title="Distribution shift across rollouts: KS(early vs late) on RMSE(t)",
        ylabel="KS statistic",
        out_path=out_dir / "ks_by_group.png",
        formats=list(args.formats),
    )
    _boxplot_by_group(
        dict(sorted(by_group_mi.items())),
        title="Spatial dependence across rollouts: mean Moran's I(|error|)",
        ylabel="mean Moran's I",
        out_path=out_dir / "moransI_mean_by_group.png",
        formats=list(args.formats),
    )

    print(f"Wrote {out_csv}")
    print(f"Wrote {(out_dir / 'acf_lag1_by_group').as_posix()}")
    print(f"Wrote {(out_dir / 'ks_by_group').as_posix()}")
    print(f"Wrote {(out_dir / 'moransI_mean_by_group').as_posix()}")


if __name__ == "__main__":
    main()
