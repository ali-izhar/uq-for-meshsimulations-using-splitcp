#!/usr/bin/env python3
"""
Generate LaTeX tables from pre-computed conformal prediction results.

Reads summary.json and Sigma.npy files - no rollout loading required.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHODS = ["l2", "linf", "mah", "adapt"]

METHOD_LABELS = {
    "l2": r"$\ell_2$ (disk)",
    "linf": r"Joint $\ell_\infty$ (box)",
    "mah": r"Mahalanobis",
    "adapt": r"Adaptive",
}

METHOD_LABELS_SHORT = {
    "l2": r"$\ell_2$",
    "linf": r"$\ell_\infty$",
    "mah": r"Mah.",
    "adapt": r"Adapt.",
}

DATASET_LABELS = {
    "cylinder": r"\textsc{CylinderFlow}",
    "flag": r"\textsc{Flag}",
}


def _unit_ball_volume(d: int) -> float:
    """Volume of d-dimensional unit ball."""
    return (math.pi ** (d / 2.0)) / math.gamma(d / 2.0 + 1.0)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class MethodResult:
    """Results for a single method at a single alpha."""

    coverage: float
    avg_radius: float
    width_norm: float  # Normalized width = (V/V_l2)^(1/d), comparable across methods


@dataclass
class DatasetResults:
    """All results for a dataset."""

    name: str
    dim: int
    sqrt_det_sigma: float
    eval_samples: int
    results: Dict[str, Dict[float, MethodResult]]


# Sample breakdown for captions (from summary.json eval_samples)
# Full rollout: CylinderFlow = 100×398×1972 = 78.5M, Flag = 100×198×1579 = 31.3M
# Conformal pipeline subsamples for efficiency
SAMPLE_INFO = {
    "cylinder": "subsampled from 100 trajectories $\\times$ 1972 nodes $\\times$ 398 timesteps",
    "flag": "subsampled from 100 trajectories $\\times$ 1579 nodes $\\times$ 198 timesteps",
}


# ---------------------------------------------------------------------------
# Load from summary.json + Sigma.npy
# ---------------------------------------------------------------------------


def load_dataset_results(conformal_out: str, name: str, dim: int) -> DatasetResults:
    """Load results from summary.json and Sigma.npy - fast, no rollout loading."""
    conf = Path(conformal_out)
    summary = json.loads((conf / "summary.json").read_text())

    # Load Sigma for Mahalanobis volume calculation
    sigma = np.load(conf / "Sigma.npy")
    sqrt_det = float(np.sqrt(np.abs(np.linalg.det(sigma))))

    unit_vol = _unit_ball_volume(dim)

    results: Dict[str, Dict[float, MethodResult]] = {}

    for method in METHODS:
        results[method] = {}
        for alpha_str, cov in summary["coverage"][method].items():
            alpha = float(alpha_str)
            avg_r = float(summary["avg_radius"][method][alpha_str])
            l2_r = float(summary["avg_radius"]["l2"][alpha_str])

            # Compute actual volumes
            l2_vol = unit_vol * (l2_r**dim)

            if method == "l2":
                vol = l2_vol
            elif method == "adapt":
                vol = unit_vol * (avg_r**dim)
            elif method == "linf":
                vol = (2 * avg_r) ** dim
            elif method == "mah":
                # Mahalanobis ellipsoid volume = unit_vol * r^d * sqrt(det(Sigma))
                vol = unit_vol * (avg_r**dim) * sqrt_det
            else:
                vol = l2_vol

            # Width = V^(1/d), normalized to L2 width
            # This is more comparable across methods than raw volume
            l2_width = l2_vol ** (1.0 / dim)
            width = vol ** (1.0 / dim)
            width_norm = width / l2_width

            results[method][alpha] = MethodResult(
                coverage=float(cov),
                avg_radius=avg_r,
                width_norm=float(width_norm),
            )

    eval_samples = summary.get("counts", {}).get("eval_samples", 0)
    return DatasetResults(
        name=name,
        dim=dim,
        sqrt_det_sigma=sqrt_det,
        eval_samples=eval_samples,
        results=results,
    )


# ---------------------------------------------------------------------------
# LaTeX Formatters
# ---------------------------------------------------------------------------


def _fmt_cov(val: float, best: float) -> str:
    """Format coverage, bold if best."""
    s = f"{val:.3f}"
    return rf"\textbf{{{s}}}" if abs(val - best) < 0.002 else s


def _fmt_width(val: float, best: float) -> str:
    """Format width, bold if best (smallest)."""
    s = f"{val:.2f}"
    return rf"\textbf{{{s}}}" if abs(val - best) / max(best, 1e-9) < 0.02 else s


def _find_best(data: DatasetResults, alphas: List[float]):
    """Find best coverage and width per alpha."""
    best_cov = {}
    best_width = {}
    for a in alphas:
        if a in data.results["l2"]:
            best_cov[a] = max(data.results[m][a].coverage for m in METHODS)
            best_width[a] = min(data.results[m][a].width_norm for m in METHODS)
    return best_cov, best_width


# ---------------------------------------------------------------------------
# LaTeX Tables - Compact with Improvement
# ---------------------------------------------------------------------------


def write_latex_dataset_table(
    data: DatasetResults,
    alphas: List[float],
    out_tex: Path,
) -> None:
    """
    Write table for single dataset with improvement row.
    Improvement shows best achievable (smallest size with valid coverage) vs L2.
    """
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    # Filter to available alphas
    alphas = [a for a in alphas if a in data.results["l2"]]
    best_cov, best_width = _find_best(data, alphas)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        rf"\caption{{{DATASET_LABELS[data.name]} (evaluation): Coverage and normalized width across confidence levels.}}",
        (
            rf"\label{{tab:{data.name}-main}}"
            if data.name == "cylinder"
            else rf"\label{{tab:{data.name}-conformal}}"
        ),
        r"\begin{tabular}{@{}l" + "cc" * len(alphas) + r"@{}}",
        r"\toprule",
    ]

    # Header row 1: alpha values
    header1 = " & " + " & ".join(
        rf"\multicolumn{{2}}{{c}}{{$\alpha={a:.2f}$}}" for a in alphas
    )
    lines.append(header1 + r" \\")

    # cmidrules
    cmidrules = "".join(
        rf"\cmidrule(lr){{{2+2*i}-{3+2*i}}}" for i in range(len(alphas))
    )
    lines.append(cmidrules)

    # Header row 2: Cov / Width
    header2 = "Method & " + " & ".join(["Cov & Width$^\\dagger$"] * len(alphas))
    lines.append(header2 + r" \\")
    lines.append(r"\midrule")

    # Data rows
    for i, method in enumerate(METHODS):
        row_color = r"\rowcolor{gray!8}" if i % 2 == 1 else ""
        cells = [METHOD_LABELS[method]]
        for a in alphas:
            r = data.results[method][a]
            cells.append(_fmt_cov(r.coverage, best_cov[a]))
            cells.append(_fmt_width(r.width_norm, best_width[a]))
        lines.append(row_color + " & ".join(cells) + r" \\")

    # Improvement row - find best method per alpha (smallest width)
    lines.append(r"\midrule")
    cells = [r"\textit{Improvement}"]
    for a in alphas:
        l2 = data.results["l2"][a]

        # Find method with smallest width at this alpha
        best_method = min(METHODS, key=lambda m: data.results[m][a].width_norm)
        best = data.results[best_method][a]

        # Coverage change
        cov_diff = (best.coverage - l2.coverage) * 100
        if abs(cov_diff) < 0.1:
            cov_str = "--"
        elif cov_diff >= 0:
            cov_str = rf"+{cov_diff:.1f}\%"
        else:
            cov_str = rf"{cov_diff:.1f}\%"

        # Width reduction (relative to L2)
        width_red = (1 - best.width_norm / l2.width_norm) * 100
        if abs(width_red) < 0.5:
            width_str = "--"
        elif width_red > 0:
            width_str = rf"$\downarrow${width_red:.1f}\%"
        else:
            width_str = rf"$\uparrow${-width_red:.1f}\%"

        cells.append(cov_str)
        cells.append(width_str)
    lines.append(" & ".join(cells) + r" \\")

    # Add sample info
    sample_info = SAMPLE_INFO.get(data.name, "")
    n_samples = f"{data.eval_samples:,}" if data.eval_samples else "N/A"

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"",
            r"\vspace{2pt}",
            rf"\parbox{{\linewidth}}{{\footnotesize $^\dagger$Width $= V^{{1/d}}$ normalized to $\ell_2$. Evaluated on {n_samples} samples ({sample_info}).}}",
            r"\end{table}",
        ]
    )
    out_tex.write_text("\n".join(lines) + "\n")
    print(f"Wrote: {out_tex}")


def write_latex_combined_compact(
    cyl: DatasetResults,
    flag: DatasetResults,
    alphas: List[float],
    out_tex: Path,
) -> None:
    """Write compact combined table for both datasets."""
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    alphas = [a for a in alphas if a in cyl.results["l2"] and a in flag.results["l2"]]

    cyl_best_cov, cyl_best_width = _find_best(cyl, alphas)
    flag_best_cov, flag_best_width = _find_best(flag, alphas)

    lines = [
        r"\begin{table*}[ht]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{Conformal prediction results: empirical coverage and normalized prediction set width.}",
        r"\label{tab:conformal-results}",
        r"\begin{tabular}{@{}ll" + "cc" * len(alphas) + r"@{}}",
        r"\toprule",
    ]

    # Header
    header1 = r"Dataset & Method"
    for a in alphas:
        header1 += rf" & \multicolumn{{2}}{{c}}{{$\alpha={a:.2f}$}}"
    lines.append(header1 + r" \\")

    cmidrules = "".join(
        rf"\cmidrule(lr){{{3+2*i}-{4+2*i}}}" for i in range(len(alphas))
    )
    lines.append(cmidrules)
    lines.append(" & ".join(["", ""] + ["Cov. & Width"] * len(alphas)) + r" \\")
    lines.append(r"\midrule")

    # CylinderFlow
    for i, method in enumerate(METHODS):
        ds_label = DATASET_LABELS["cylinder"] if i == 0 else ""
        row_color = r"\rowcolor{gray!8}" if i % 2 == 1 else ""
        cells = [row_color + ds_label, METHOD_LABELS_SHORT[method]]
        for a in alphas:
            r = cyl.results[method][a]
            cells.append(_fmt_cov(r.coverage, cyl_best_cov[a]))
            cells.append(_fmt_width(r.width_norm, cyl_best_width[a]))
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\midrule")

    # Flag
    for i, method in enumerate(METHODS):
        ds_label = DATASET_LABELS["flag"] if i == 0 else ""
        row_color = r"\rowcolor{gray!8}" if i % 2 == 1 else ""
        cells = [row_color + ds_label, METHOD_LABELS_SHORT[method]]
        for a in alphas:
            r = flag.results[method][a]
            cells.append(_fmt_cov(r.coverage, flag_best_cov[a]))
            cells.append(_fmt_width(r.width_norm, flag_best_width[a]))
        lines.append(" & ".join(cells) + r" \\")

    cyl_n = f"{cyl.eval_samples:,}" if cyl.eval_samples else "N/A"
    flag_n = f"{flag.eval_samples:,}" if flag.eval_samples else "N/A"

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"",
            r"\vspace{2pt}",
            rf"{{\footnotesize Width $= V^{{1/d}}$ normalized to $\ell_2$. Samples: CylinderFlow {cyl_n}, Flag {flag_n}.}}",
            r"\end{table*}",
        ]
    )
    out_tex.write_text("\n".join(lines) + "\n")
    print(f"Wrote: {out_tex}")


def write_latex_coverage_shortfall(
    cyl: DatasetResults,
    flag: DatasetResults,
    alphas: List[float],
    out_tex: Path,
) -> None:
    """Write coverage table showing shortfalls in red."""
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    alphas = [a for a in alphas if a in cyl.results["l2"] and a in flag.results["l2"]]

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\caption{Empirical coverage vs.\ nominal $(1-\alpha)$. Shortfalls in \textcolor{red}{red}.}",
        r"\label{tab:coverage-shortfall}",
        r"\begin{tabular}{@{}l" + "r" * len(alphas) * 2 + r"@{}}",
        r"\toprule",
    ]

    # Dataset headers
    header1 = r"Method"
    for ds in ["CylinderFlow", "Flag"]:
        header1 += rf" & \multicolumn{{{len(alphas)}}}{{c}}{{\textsc{{{ds}}}}}"
    lines.append(header1 + r" \\")

    cmidrule1 = rf"\cmidrule(lr){{2-{1+len(alphas)}}}"
    cmidrule2 = rf"\cmidrule(lr){{{2+len(alphas)}-{1+2*len(alphas)}}}"
    lines.append(cmidrule1 + cmidrule2)

    # Nominal coverage headers
    header2 = " & ".join([""] + [f"${int((1-a)*100)}\\%$" for a in alphas] * 2)
    lines.append(header2 + r" \\")
    lines.append(r"\midrule")

    for i, method in enumerate(METHODS):
        row_color = r"\rowcolor{gray!8}" if i % 2 == 1 else ""
        cells = [row_color + METHOD_LABELS_SHORT[method]]
        for ds_data in [cyl, flag]:
            for a in alphas:
                cov = ds_data.results[method][a].coverage
                target = 1 - a
                cov_pct = f"{cov*100:.1f}\\%"
                if cov >= target - 0.005:  # Small tolerance
                    cells.append(cov_pct)
                else:
                    cells.append(rf"\textit{{{cov_pct}}}")
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    out_tex.write_text("\n".join(lines) + "\n")
    print(f"Wrote: {out_tex}")


def write_latex_efficiency_ratio(
    cyl: DatasetResults,
    flag: DatasetResults,
    alphas: List[float],
    out_tex: Path,
) -> None:
    """Write table showing width ratio to smallest."""
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    alphas = [a for a in alphas if a in cyl.results["l2"] and a in flag.results["l2"]]

    # Best (smallest) width per dataset per alpha
    cyl_best = {a: min(cyl.results[m][a].width_norm for m in METHODS) for a in alphas}
    flag_best = {a: min(flag.results[m][a].width_norm for m in METHODS) for a in alphas}

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\caption{Relative prediction set width (ratio to smallest). 1.00 = best.}",
        r"\label{tab:efficiency-comparison}",
        r"\begin{tabular}{@{}l" + "r" * len(alphas) * 2 + r"@{}}",
        r"\toprule",
    ]

    header1 = r"Method"
    for ds in ["CylinderFlow", "Flag"]:
        header1 += rf" & \multicolumn{{{len(alphas)}}}{{c}}{{\textsc{{{ds}}}}}"
    lines.append(header1 + r" \\")

    cmidrule1 = rf"\cmidrule(lr){{2-{1+len(alphas)}}}"
    cmidrule2 = rf"\cmidrule(lr){{{2+len(alphas)}-{1+2*len(alphas)}}}"
    lines.append(cmidrule1 + cmidrule2)

    header2 = " & ".join([""] + [f"$\\alpha={a:.2f}$" for a in alphas] * 2)
    lines.append(header2 + r" \\")
    lines.append(r"\midrule")

    for i, method in enumerate(METHODS):
        row_color = r"\rowcolor{gray!8}" if i % 2 == 1 else ""
        cells = [row_color + METHOD_LABELS_SHORT[method]]

        for ds_data, ds_best in [(cyl, cyl_best), (flag, flag_best)]:
            for a in alphas:
                ratio = ds_data.results[method][a].width_norm / ds_best[a]
                if abs(ratio - 1.0) < 0.02:
                    cells.append(rf"\textbf{{1.00}}")
                else:
                    cells.append(f"{ratio:.2f}")
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    out_tex.write_text("\n".join(lines) + "\n")
    print(f"Wrote: {out_tex}")


def write_latex_split_sensitivity(out_tex: Path) -> None:
    """
    Generate split sensitivity table from rollout data.
    Uses robust IQR-based dispersion (IQR/median) instead of CV (std/mean).
    """
    import pandas as pd

    csv_path = Path(
        "meshgraphnet/_artifacts/split_sensitivity_dense/split_sensitivity.csv"
    )
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found, skipping split sensitivity table")
        return

    out_tex.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Surrogate rollout split sensitivity.} Dispersion (IQR/median, \%) over 5 seeds $\times$ 50 trajectories per split.}",
        r"\label{tab:mgn-split-sensitivity-seedcv}",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Dataset & Split & AUC RMSE (\%) & Final RMSE (\%) \\",
        r"\midrule",
    ]

    split_map = {
        "auxiliary": "Auxiliary",
        "calibration": "Calibration",
        "test": "Evaluation",
    }

    row_idx = 0
    for ds in ["cylinder", "flag"]:
        ds_label = r"\textsc{CylinderFlow}" if ds == "cylinder" else r"\textsc{Flag}"
        for split in ["auxiliary", "calibration", "test"]:
            sub = df[(df["dataset"] == ds) & (df["split"] == split)]
            if len(sub) == 0:
                continue

            # Robust dispersion: IQR / median * 100
            auc_median = sub["rmse_auc"].median()
            auc_iqr = sub["rmse_auc"].quantile(0.75) - sub["rmse_auc"].quantile(0.25)
            auc_disp = auc_iqr / auc_median * 100 if auc_median > 0 else 0

            final_median = sub["rmse_final"].median()
            final_iqr = sub["rmse_final"].quantile(0.75) - sub["rmse_final"].quantile(
                0.25
            )
            final_disp = final_iqr / final_median * 100 if final_median > 0 else 0

            row_color = r"\rowcolor{gray!8} " if row_idx % 2 == 1 else ""
            lines.append(
                f"{row_color}{ds_label} & {split_map[split]} & {auc_disp:.1f} & {final_disp:.1f} \\\\"
            )
            row_idx += 1

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"",
            r"\vspace{2pt}",
            r"{\footnotesize 5 seeds $\times$ 50 trajectories = 250 rollouts per cell. IQR-based metric robust to outliers.}",
            r"\end{table}",
        ]
    )
    out_tex.write_text("\n".join(lines) + "\n")
    print(f"Wrote: {out_tex}")


# ---------------------------------------------------------------------------
# CSV output for coverage plots
# ---------------------------------------------------------------------------


def write_csv_table(data: DatasetResults, alphas: List[float], out_csv: Path) -> None:
    """Write CSV table for coverage plots."""
    import csv
    
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    alphas = [a for a in alphas if a in data.results["l2"]]
    
    # Get L2 width for normalization
    l2_widths = {a: data.results["l2"][a].width_norm for a in alphas}
    
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "method", "alpha", "coverage", "avg_radius", "width_norm", "size_norm"])
        
        for method in METHODS:
            for a in alphas:
                r = data.results[method][a]
                # size_norm = width^d (volume), for backwards compatibility with coverage plot
                size_norm = r.width_norm ** data.dim
                writer.writerow([
                    data.name,
                    method,
                    a,
                    r.coverage,
                    r.avg_radius,
                    r.width_norm,
                    size_norm,
                ])
    
    print(f"Wrote: {out_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate tables from conformal results")

    ap.add_argument("--alphas", type=float, nargs="+", default=[0.30, 0.20, 0.10, 0.05])
    ap.add_argument("--out_dir", default="paper/tables_final")

    ap.add_argument("--cyl_conformal_out", required=True)
    ap.add_argument("--flag_conformal_out", required=True)

    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load from summary.json - instant
    print("Loading pre-computed results...")
    cyl = load_dataset_results(args.cyl_conformal_out, "cylinder", dim=2)
    flag = load_dataset_results(args.flag_conformal_out, "flag", dim=3)
    print(f"  CylinderFlow: {len(cyl.results['l2'])} alpha levels")
    print(f"  Flag: {len(flag.results['l2'])} alpha levels")

    # Generate all tables
    print("\nGenerating tables...")

    # Individual dataset tables with improvement
    write_latex_dataset_table(cyl, args.alphas, out_dir / "cylinder_table.tex")
    write_latex_dataset_table(flag, args.alphas, out_dir / "flag_table.tex")

    # Combined tables
    write_latex_combined_compact(cyl, flag, args.alphas, out_dir / "combined_table.tex")
    # Removed: coverage_compact table (info already in Tables 4 & 5)
    write_latex_efficiency_ratio(
        cyl, flag, args.alphas, out_dir / "efficiency_comparison.tex"
    )

    # Split sensitivity table (robust IQR-based)
    write_latex_split_sensitivity(out_dir / "meshgraphnet_split_sensitivity_seedcv.tex")

    # CSV files for coverage plots
    write_csv_table(cyl, args.alphas, out_dir / "cylinder_table.csv")
    write_csv_table(flag, args.alphas, out_dir / "flag_table.csv")

    print(f"\nAll tables written to {out_dir}/")


if __name__ == "__main__":
    main()
