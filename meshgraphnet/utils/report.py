#!/usr/bin/env python3
"""Create tables from `split_sensitivity.csv`.

This is a pure rollout-only summarizer:
- Uses all points in the CSV (typically seed × trajectory).
- Reports both final-step RMSE and AUC RMSE (mean_t RMSE(t)).
- Writes:
  - `summary_by_split.csv` and `summary_by_split.md`
  - `summary_by_seed.csv` and `summary_by_seed.md`
  - `outliers_topk.csv` and `outliers_topk.md`
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class Row:
    dataset: str
    split: str
    seed: str
    traj_idx: int
    rmse_final: float
    rmse_auc: float
    timesteps: int
    file: str


def _read_rows(csv_path: Path) -> List[Row]:
    rows: List[Row] = []
    with csv_path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for d in r:
            if not d:
                continue
            rows.append(
                Row(
                    dataset=str(d["dataset"]),
                    split=str(d["split"]),
                    seed=str(d["seed"]),
                    traj_idx=int(d["traj_idx"]),
                    rmse_final=float(d["rmse_final"]),
                    rmse_auc=float(d["rmse_auc"]),
                    timesteps=int(d.get("timesteps", "0") or "0"),
                    file=str(d.get("file", "")),
                )
            )
    return rows


def _quantiles(x: np.ndarray) -> Tuple[float, float, float]:
    q25 = float(np.quantile(x, 0.25))
    q50 = float(np.quantile(x, 0.50))
    q75 = float(np.quantile(x, 0.75))
    return q25, q50, q75


def _stats(x: Iterable[float]) -> Dict[str, float]:
    a = np.asarray(list(x), dtype=float)
    if a.size == 0:
        return {}
    mean = float(np.mean(a))
    std = float(np.std(a, ddof=0))
    q25, q50, q75 = _quantiles(a)
    iqr = q75 - q25
    cv = float(std / mean) if mean != 0.0 else float("nan")
    return {
        "n": float(a.size),
        "mean": mean,
        "median": q50,
        "std": std,
        "q25": q25,
        "q75": q75,
        "iqr": float(iqr),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "cv": cv,
    }


def _fmt(x: float) -> str:
    if np.isnan(x):
        return "nan"
    # compact but readable
    return f"{x:.6g}"


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _write_md_table(path: Path, headers: List[str], rows: List[List[str]]) -> None:
    # minimal markdown table
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Report from split_sensitivity.csv")
    ap.add_argument("--csv", required=True, help="Path to split_sensitivity.csv")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_rows(Path(args.csv))
    if not rows:
        raise SystemExit(f"No rows found in {args.csv}")

    # 1) Summary by dataset+split using all points (seed × trajectory)
    by_split_final: DefaultDict[Tuple[str, str], List[float]] = defaultdict(list)
    by_split_auc: DefaultDict[Tuple[str, str], List[float]] = defaultdict(list)
    timesteps_by_split: DefaultDict[Tuple[str, str], List[int]] = defaultdict(list)
    for r in rows:
        k = (r.dataset, r.split)
        by_split_final[k].append(r.rmse_final)
        by_split_auc[k].append(r.rmse_auc)
        timesteps_by_split[k].append(r.timesteps)

    summary_rows: List[Dict[str, str]] = []
    for (dataset, split), vals_f in sorted(by_split_final.items()):
        vals_a = by_split_auc[(dataset, split)]
        st_f = _stats(vals_f)
        st_a = _stats(vals_a)
        ts = np.asarray(timesteps_by_split[(dataset, split)], dtype=int)
        ts_med = float(np.median(ts)) if ts.size else float("nan")
        ts_min = float(np.min(ts)) if ts.size else float("nan")
        ts_max = float(np.max(ts)) if ts.size else float("nan")
        summary_rows.append(
            {
                "dataset": dataset,
                "split": split,
                "n_points": str(int(st_f["n"])),
                "timesteps_median": _fmt(ts_med),
                "timesteps_min": _fmt(ts_min),
                "timesteps_max": _fmt(ts_max),
                "final_mean": _fmt(st_f["mean"]),
                "final_median": _fmt(st_f["median"]),
                "final_iqr": _fmt(st_f["iqr"]),
                "final_max": _fmt(st_f["max"]),
                "final_cv": _fmt(st_f["cv"]),
                "auc_mean": _fmt(st_a["mean"]),
                "auc_median": _fmt(st_a["median"]),
                "auc_iqr": _fmt(st_a["iqr"]),
                "auc_max": _fmt(st_a["max"]),
                "auc_cv": _fmt(st_a["cv"]),
            }
        )

    out_csv1 = out_dir / "summary_by_split.csv"
    f1 = list(summary_rows[0].keys())
    _write_csv(out_csv1, f1, summary_rows)
    out_md1 = out_dir / "summary_by_split.md"
    _write_md_table(
        out_md1,
        headers=f1,
        rows=[[r[h] for h in f1] for r in summary_rows],
    )

    # 2) Summary by dataset+split aggregated to per-seed means first (sensitivity to seed)
    by_seed_final: DefaultDict[Tuple[str, str, str], List[float]] = defaultdict(list)
    by_seed_auc: DefaultDict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for r in rows:
        k = (r.dataset, r.split, r.seed)
        by_seed_final[k].append(r.rmse_final)
        by_seed_auc[k].append(r.rmse_auc)

    # reduce to seed means
    seed_means_final: DefaultDict[Tuple[str, str], List[float]] = defaultdict(list)
    seed_means_auc: DefaultDict[Tuple[str, str], List[float]] = defaultdict(list)
    for (dataset, split, _seed), v in by_seed_final.items():
        seed_means_final[(dataset, split)].append(float(np.mean(v)))
    for (dataset, split, _seed), v in by_seed_auc.items():
        seed_means_auc[(dataset, split)].append(float(np.mean(v)))

    seed_rows: List[Dict[str, str]] = []
    for (dataset, split), vals in sorted(seed_means_final.items()):
        st_f = _stats(vals)
        st_a = _stats(seed_means_auc[(dataset, split)])
        seed_rows.append(
            {
                "dataset": dataset,
                "split": split,
                "n_seeds": str(int(st_f["n"])),
                "seed_final_mean": _fmt(st_f["mean"]),
                "seed_final_median": _fmt(st_f["median"]),
                "seed_final_iqr": _fmt(st_f["iqr"]),
                "seed_final_max": _fmt(st_f["max"]),
                "seed_final_cv": _fmt(st_f["cv"]),
                "seed_auc_mean": _fmt(st_a["mean"]),
                "seed_auc_median": _fmt(st_a["median"]),
                "seed_auc_iqr": _fmt(st_a["iqr"]),
                "seed_auc_max": _fmt(st_a["max"]),
                "seed_auc_cv": _fmt(st_a["cv"]),
            }
        )

    out_csv2 = out_dir / "summary_by_seed.csv"
    f2 = list(seed_rows[0].keys())
    _write_csv(out_csv2, f2, seed_rows)
    out_md2 = out_dir / "summary_by_seed.md"
    _write_md_table(out_md2, headers=f2, rows=[[r[h] for h in f2] for r in seed_rows])

    # 3) Outliers: top-k rows by final and by AUC
    topk = int(args.topk)
    by_final = sorted(rows, key=lambda r: r.rmse_final, reverse=True)[:topk]
    by_auc = sorted(rows, key=lambda r: r.rmse_auc, reverse=True)[:topk]

    outlier_rows: List[Dict[str, str]] = []
    for kind, sel in [("rmse_final", by_final), ("rmse_auc", by_auc)]:
        for rank, r in enumerate(sel, start=1):
            outlier_rows.append(
                {
                    "metric": kind,
                    "rank": str(rank),
                    "dataset": r.dataset,
                    "split": r.split,
                    "seed": r.seed,
                    "traj_idx": str(r.traj_idx),
                    "value": _fmt(r.rmse_final if kind == "rmse_final" else r.rmse_auc),
                    "timesteps": str(r.timesteps),
                    "file": r.file,
                }
            )

    out_csv3 = out_dir / "outliers_topk.csv"
    f3 = list(outlier_rows[0].keys())
    _write_csv(out_csv3, f3, outlier_rows)
    out_md3 = out_dir / "outliers_topk.md"
    _write_md_table(
        out_md3, headers=f3, rows=[[r[h] for h in f3] for r in outlier_rows]
    )

    print(f"Wrote {out_csv1}")
    print(f"Wrote {out_md1}")
    print(f"Wrote {out_csv2}")
    print(f"Wrote {out_md2}")
    print(f"Wrote {out_csv3}")
    print(f"Wrote {out_md3}")


if __name__ == "__main__":
    main()
