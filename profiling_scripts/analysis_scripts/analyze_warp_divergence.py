#!/usr/bin/env python3
"""
analyze_warp_divergence.py
────────────────────────────────────────────────────────────────────────
Parses one .ncu-rep per ablation scenario (baseline / no-warp /
default-batch / etc.) and compares, per kernel:

  - Thread utilization: active threads per warp instruction (goal = 32),
    and non-predicated-off threads per warp instruction (catches
    divergence the compiler hid behind predication/selp rather than an
    actual branch).
  - Warp/branch divergence: % of branches with a uniform target across
    the warp (goal = 100%), plus an independently recomputed divergence
    % from raw branch-target counts as a cross-check.

This assumes the .ncu-rep files were collected with (at least) these
metrics -- i.e. the metric set added to ncu_metrics_RTX4090.sh earlier:
    smsp__thread_inst_executed_per_inst_executed.ratio
    smsp__thread_inst_executed_pred_on_per_inst_executed.ratio
    smsp__sass_average_branch_targets_threads_uniform.pct
    smsp__sass_branch_targets.sum
    smsp__sass_branch_targets_threads_divergent.sum
    smsp__sass_branch_targets_threads_uniform.sum
    gpu__time_duration.sum   (used as a launch weight for aggregation)

Any of these that are missing from the report are simply reported as NaN
for the metrics that depend on them -- the script does not require all of
them to be present.

How it works
────────────
For each .ncu-rep, shells out to:
    ncu --import <file> --csv --page raw
to dump one row per kernel launch with a column per collected metric,
then aggregates (duration-weighted mean, if duration is available) across
all launches of each simplified kernel name (Stage1, Stage2, ABMKernel,
cuco::insert, ...) within that scenario.

Usage
─────
    # Auto-detect scenario labels from filenames matching
    # "<label>-ncu-gpu-parallel-static-*.ncu-rep" (the naming convention
    # produced by ncu_metrics_RTX4090.sh):
    python analyze_warp_divergence.py --dir ./profile \
        --out-csv warp_divergence_comparison.csv \
        --out-plot warp_divergence_comparison.png

    # Or specify scenarios explicitly:
    python analyze_warp_divergence.py \
        --reports baseline=baseline-ncu-....ncu-rep \
                  no-warp=no-warp-ncu-....ncu-rep \
                  default-batch=default-batch-ncu-....ncu-rep \
        --out-csv warp_divergence_comparison.csv
"""

import argparse
import csv
import io
import re
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


METRIC_ACTIVE_THREADS   = "smsp__thread_inst_executed_per_inst_executed.ratio"
METRIC_PRED_ON_THREADS  = "smsp__thread_inst_executed_pred_on_per_inst_executed.ratio"
METRIC_BRANCH_UNIFORM_PCT = "smsp__sass_average_branch_targets_threads_uniform.pct"
METRIC_BRANCH_TARGETS   = "smsp__sass_branch_targets.sum"
METRIC_BRANCH_DIVERGENT = "smsp__sass_branch_targets_threads_divergent.sum"
METRIC_DURATION         = "gpu__time_duration.sum"

# Recognized ablation scenario naming convention:
#   <label>-ncu-gpu-parallel-static-pa-<years>y-<growth>p.ncu-rep
LABEL_FROM_FILENAME = re.compile(r"^(?P<label>.+?)-ncu-gpu-parallel-static-")


def find_ncu() -> str:
    ncu = shutil.which("ncu")
    if ncu is None:
        sys.exit("ERROR: 'ncu' not found on PATH. Load the CUDA/nsight-compute "
                 "module on the machine that has (or can re-read) these reports.")
    return ncu


def discover_reports(directory: Path):
    """Auto-detect {label: path} from files matching the ablation naming convention."""
    found = {}
    for p in sorted(directory.glob("*.ncu-rep")):
        m = LABEL_FROM_FILENAME.match(p.name)
        label = m.group("label") if m else p.stem
        found[label] = p
    return found


def dump_raw_csv(ncu_bin: str, rep_path: Path) -> str:
    cmd = [ncu_bin, "--import", str(rep_path), "--csv", "--page", "raw"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ncu --import failed for {rep_path.name}:\n{result.stderr}")
    return result.stdout


def clean_numeric(value: str):
    """ncu's --csv output can still contain thousands separators or 'N/A'."""
    if value is None:
        return None
    v = value.strip().strip('"')
    if v in ("", "N/A", "n/a"):
        return None
    v = v.replace(",", "")
    try:
        return float(v)
    except ValueError:
        return None


# ── Kernel name simplification (same scheme used for the nsys occupancy script) ──
NAME_PATTERNS = [
    (re.compile(r"kernelCallStage1(_warped)?", re.I), "Stage1"),
    (re.compile(r"kernelCallStage2", re.I), "Stage2"),
    (re.compile(r"kernelCallStage3", re.I), "Stage3"),
    (re.compile(r"kernelCallStage4", re.I), "Stage4"),
    (re.compile(r"ABMKernel", re.I), "ABMKernel"),
    (re.compile(r"ExtractSizes", re.I), "ExtractSizes"),
    (re.compile(r"cuco::[a-zA-Z_]+"), None),
    (re.compile(r"cub::[a-zA-Z_]+"), None),
    (re.compile(r"thrust::[a-zA-Z_:]+"), None),
]


def simplify_kernel_name(raw_name: str) -> str:
    for pattern, replacement in NAME_PATTERNS:
        m = pattern.search(raw_name)
        if m:
            return replacement if replacement is not None else m.group(0)
    short = raw_name.split("(")[0].split("<")[0]
    return short.split("::")[-1] if "::" in short else short


def parse_raw_csv(csv_text: str):
    """
    Parse ncu's `--csv --page raw` output into per-launch dicts.
    Robust to exact column ordering: metric columns are matched by name,
    not position.
    """
    reader = csv.reader(io.StringIO(csv_text))
    rows = list(reader)
    if not rows:
        return []
    header = rows[0]
    col_idx = {name: i for i, name in enumerate(header)}

    kernel_name_col = None
    for candidate in ("Kernel Name", "Kernel Name ", "Function Name"):
        if candidate in col_idx:
            kernel_name_col = col_idx[candidate]
            break
    if kernel_name_col is None:
        raise RuntimeError(
            f"Could not find a kernel-name column in ncu CSV header: {header}"
        )

    launches = []
    for row in rows[1:]:
        if len(row) != len(header):
            continue  # skip malformed/truncated lines
        record = {"kernel_name": row[kernel_name_col]}
        for metric in (METRIC_ACTIVE_THREADS, METRIC_PRED_ON_THREADS,
                       METRIC_BRANCH_UNIFORM_PCT, METRIC_BRANCH_TARGETS,
                       METRIC_BRANCH_DIVERGENT, METRIC_DURATION):
            if metric in col_idx:
                record[metric] = clean_numeric(row[col_idx[metric]])
            else:
                record[metric] = None
        launches.append(record)
    return launches


def weighted_mean(values_weights):
    """values_weights: list of (value, weight). Falls back to plain mean if
    weights are missing/zero."""
    pairs = [(v, w) for v, w in values_weights if v is not None]
    if not pairs:
        return None
    if all(w for _, w in pairs):
        total_w = sum(w for _, w in pairs)
        return sum(v * w for v, w in pairs) / total_w
    return statistics.mean(v for v, _ in pairs)


def aggregate_by_kernel(launches):
    groups = defaultdict(list)
    for rec in launches:
        kname = simplify_kernel_name(rec["kernel_name"])
        groups[kname].append(rec)

    summary = {}
    for kname, recs in groups.items():
        dur = [(r[METRIC_DURATION], 1.0) for r in recs]
        weights = [r[METRIC_DURATION] or 0.0 for r in recs]

        def wm(metric):
            vw = [(r[metric], w) for r, w in zip(recs, weights)]
            return weighted_mean(vw)

        active = wm(METRIC_ACTIVE_THREADS)
        pred_on = wm(METRIC_PRED_ON_THREADS)
        branch_uniform_pct = wm(METRIC_BRANCH_UNIFORM_PCT)
        branch_targets = sum(r[METRIC_BRANCH_TARGETS] or 0 for r in recs)
        branch_divergent = sum(r[METRIC_BRANCH_DIVERGENT] or 0 for r in recs)

        recomputed_divergence_pct = (
            branch_divergent / branch_targets * 100.0 if branch_targets else None
        )
        divergence_pct = (100.0 - branch_uniform_pct) if branch_uniform_pct is not None \
            else recomputed_divergence_pct

        summary[kname] = {
            "launches": len(recs),
            "active_threads_per_warp": round(active, 2) if active is not None else None,
            "thread_util_pct": round(active / 32.0 * 100.0, 1) if active is not None else None,
            "pred_on_threads_per_warp": round(pred_on, 2) if pred_on is not None else None,
            "pred_util_pct": round(pred_on / 32.0 * 100.0, 1) if pred_on is not None else None,
            "branch_uniform_pct": round(branch_uniform_pct, 1) if branch_uniform_pct is not None else None,
            "divergence_pct": round(divergence_pct, 1) if divergence_pct is not None else None,
            "divergence_pct_recomputed": round(recomputed_divergence_pct, 1) if recomputed_divergence_pct is not None else None,
        }
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=None,
                     help="Directory to auto-discover *.ncu-rep scenarios from")
    ap.add_argument("--reports", nargs="+", default=None,
                     help="Explicit label=path pairs, e.g. baseline=foo.ncu-rep")
    ap.add_argument("--out-csv", type=Path, default=Path("warp_divergence_comparison.csv"))
    ap.add_argument("--out-plot", type=Path, default=None,
                     help="Optional PNG path for a comparison bar chart")
    ap.add_argument("--baseline-label", default="baseline",
                     help="Which scenario label to compute deltas against")
    args = ap.parse_args()

    reports = {}
    if args.reports:
        for item in args.reports:
            label, _, path = item.partition("=")
            reports[label] = Path(path)
    elif args.dir:
        reports = discover_reports(args.dir)
    else:
        sys.exit("ERROR: supply --dir or --reports")

    if not reports:
        sys.exit("ERROR: no .ncu-rep files found/specified")

    ncu_bin = find_ncu()
    per_scenario = {}

    for label, path in reports.items():
        if not path.exists():
            print(f"  [skip] {label}: {path} not found")
            continue
        print(f"  [ok]   {label}: {path.name}")
        try:
            csv_text = dump_raw_csv(ncu_bin, path)
            launches = parse_raw_csv(csv_text)
        except RuntimeError as e:
            print(f"    ERROR: {e}")
            continue
        per_scenario[label] = aggregate_by_kernel(launches)

    if not per_scenario:
        sys.exit("ERROR: no reports parsed successfully")

    all_kernels = sorted({k for s in per_scenario.values() for k in s})
    labels = list(per_scenario.keys())
    baseline = args.baseline_label if args.baseline_label in per_scenario else None

    # ── Write comparison CSV ────────────────────────────────────────────
    fieldnames = ["Kernel", "Scenario", "Launches",
                  "ActiveThreadsPerWarp", "ThreadUtil_pct",
                  "PredOnThreadsPerWarp", "PredUtil_pct",
                  "BranchUniform_pct", "Divergence_pct", "Divergence_pct_recomputed",
                  "Delta_Divergence_vs_baseline_pp"]
    rows_out = []
    for kname in all_kernels:
        base_div = None
        if baseline and kname in per_scenario[baseline]:
            base_div = per_scenario[baseline][kname]["divergence_pct"]
        for label in labels:
            s = per_scenario[label].get(kname)
            if s is None:
                continue
            delta = None
            if base_div is not None and s["divergence_pct"] is not None:
                delta = round(s["divergence_pct"] - base_div, 1)
            rows_out.append({
                "Kernel": kname,
                "Scenario": label,
                "Launches": s["launches"],
                "ActiveThreadsPerWarp": s["active_threads_per_warp"],
                "ThreadUtil_pct": s["thread_util_pct"],
                "PredOnThreadsPerWarp": s["pred_on_threads_per_warp"],
                "PredUtil_pct": s["pred_util_pct"],
                "BranchUniform_pct": s["branch_uniform_pct"],
                "Divergence_pct": s["divergence_pct"],
                "Divergence_pct_recomputed": s["divergence_pct_recomputed"],
                "Delta_Divergence_vs_baseline_pp": delta,
            })

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nWrote {len(rows_out)} rows -> {args.out_csv}")

    # ── Console summary table ───────────────────────────────────────────
    print("\n=== Warp divergence (%) by kernel and scenario (lower = better) ===")
    header = f"{'Kernel':16s}" + "".join(f"{l:>16s}" for l in labels)
    print(header)
    for kname in all_kernels:
        line = f"{kname:16s}"
        for label in labels:
            s = per_scenario[label].get(kname)
            val = s["divergence_pct"] if s and s["divergence_pct"] is not None else float("nan")
            line += f"{val:16.1f}"
        print(line)

    print("\n=== Thread utilization (%, active threads/warp -- higher = better) ===")
    print(header)
    for kname in all_kernels:
        line = f"{kname:16s}"
        for label in labels:
            s = per_scenario[label].get(kname)
            val = s["thread_util_pct"] if s and s["thread_util_pct"] is not None else float("nan")
            line += f"{val:16.1f}"
        print(line)

    # ── Optional plot ───────────────────────────────────────────────────
    if args.out_plot:
        try:
            import numpy as np
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("\nmatplotlib/numpy not available -- skipping plot")
            return

        plot_kernels = [k for k in all_kernels
                        if any(per_scenario[l].get(k, {}).get("divergence_pct") is not None
                               for l in labels)]
        x = np.arange(len(plot_kernels))
        n = len(labels)
        bw = 0.8 / max(n, 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
        for i, label in enumerate(labels):
            div_vals = [per_scenario[label].get(k, {}).get("divergence_pct") or 0
                        for k in plot_kernels]
            util_vals = [per_scenario[label].get(k, {}).get("thread_util_pct") or 0
                         for k in plot_kernels]
            offset = (i - (n - 1) / 2) * bw
            ax1.bar(x + offset, div_vals, width=bw, label=label)
            ax2.bar(x + offset, util_vals, width=bw, label=label)

        ax1.set_xticks(x); ax1.set_xticklabels(plot_kernels, rotation=30, ha="right")
        ax1.set_ylabel("Warp divergence (%)"); ax1.set_title("Warp Divergence by Kernel")
        ax1.legend(fontsize=8)

        ax2.set_xticks(x); ax2.set_xticklabels(plot_kernels, rotation=30, ha="right")
        ax2.set_ylabel("Active threads / warp (%)"); ax2.set_title("Thread Utilization by Kernel")
        ax2.axhline(100, color="gray", linestyle="--", linewidth=1)
        ax2.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(args.out_plot, dpi=200)
        print(f"Saved comparison plot -> {args.out_plot}")


if __name__ == "__main__":
    main()
