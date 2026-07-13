#!/usr/bin/env python3
"""
nsys_paper_metrics.py
────────────────────────────────────────────────────────────────────────
Extracts paper-ready scaling/performance metrics from a sweep of
.nsys-rep files (years x growth-rate grid) and produces:

  1. {prefix}_summary.csv   -- one row per scenario:
       wall/kernel/memcpy/memset/other time + percentages,
       GPU_util_pct (SM Active), Memcpy_ovh_pct,
       achieved bandwidth (GB/s) and %-of-peak,
       total kernel launches.

  2. {prefix}_kernels.csv   -- one row per (scenario, kernel):
       time share %, launch count, avg grid size, avg block size.
       This is the long-format table behind the "where does time go as
       the graph grows" and "does grid size scale to saturate the GPU"
       plots.

  3. Two plots (PNG):
       {prefix}_time_decomposition.png  -- stacked bars, kernel/memcpy/
           memset/other %, faceted by years, x=growth. Shows whether the
           GPU-active fraction is growing or shrinking with scale --
           i.e. whether the NEXT bottleneck is algorithmic (kernels) or
           structural (host-side bookkeeping).
       {prefix}_grid_saturation.png     -- Stage1's grid size (and any
           other kernel you point it at) vs. scenario, with a horizontal
           reference line at the GPU's SM count. Turns "grid size fills
           the GPU" into an actual scaling curve instead of one data
           point.

Why these and not others
──────────────────────────
- Time decomposition is the direct systematic version of the by-hand
  Amdahl breakdown (kernel_ns + memcpy_ns + memset_ns vs wall_ns) already
  used earlier in this analysis -- this script just runs it over every
  scenario and turns it into one chart instead of nine manual queries.
- Grid-size-vs-SM-count directly supports a "does the warp-cooperative
  redesign actually saturate the GPU across scale" claim, which is a
  stronger and more falsifiable statement than reporting occupancy or
  utilization numbers alone.
- Kernel time-share evolution answers "is Stage1 BFS an increasing or
  decreasing fraction of the runtime as N grows" -- relevant for whether
  future optimization effort should stay focused on Stage1 or move
  elsewhere as problems scale up.
- Bandwidth-%-of-peak is included because it's a single, referee-legible
  number, but it is NOT the centerpiece -- for this workload the
  memcpy_ovh_pct trend matters more than the absolute bandwidth number,
  since these kernels are latency/occupancy-bound, not bandwidth-bound.

Usage
─────
    python nsys_paper_metrics.py --dir ./profile \
        --years 3 10 30 --growth 1 3 6 \
        --pattern "nsys-pa-{years}y-{growth}p.nsys-rep" \
        --gpu h100 \
        --grid-kernel Stage1 \
        --out-prefix pa_paper
"""

import argparse
import csv
import re
import shutil
import sqlite3
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


# ── GPU constants (peak bandwidth for the roofline-style %, SM count for
#    the grid-saturation reference line) ────────────────────────────────
GPU_SPECS = {
    "h100":    dict(name="NVIDIA H100 80GB HBM3", peak_bw_GBs=3350.0, sm_count=132),
    "rtx4090": dict(name="NVIDIA GeForce RTX 4090", peak_bw_GBs=1008.0, sm_count=128),
    "a100":    dict(name="NVIDIA A100 80GB", peak_bw_GBs=2039.0, sm_count=108),
}

SM_ACTIVE_NAMES = (
    "SM Active", "SMs Active",
    "SM Active [Throughput %]", "SMs Active [Throughput %]",
)

LABEL_PATTERN_DEFAULT = "nsys-pa-{years}y-{growth}p.nsys-rep"

# ── Kernel name simplification (same scheme used throughout this analysis) ──
NAME_PATTERNS = [
    (re.compile(r"kernelCallStage1(_warped)?", re.I), "Stage1"),
    (re.compile(r"kernelCallStage2", re.I), "Stage2"),
    (re.compile(r"kernelCallStage3", re.I), "Stage3"),
    (re.compile(r"kernelCallStage4", re.I), "Stage4"),
    (re.compile(r"ABMKernel", re.I), "ABMKernel"),
    (re.compile(r"ExtractSizes|extract_vector_sizes", re.I), "ExtractSizes"),
    (re.compile(r"cuco::[a-zA-Z_:]+"), None),
    (re.compile(r"cub::[a-zA-Z_:]+"), None),
    (re.compile(r"thrust::[a-zA-Z_:]+"), None),
]


def simplify_kernel_name(raw_name: str) -> str:
    for pattern, replacement in NAME_PATTERNS:
        m = pattern.search(raw_name)
        if m:
            return replacement if replacement is not None else m.group(0)
    short = raw_name.split("(")[0].split("<")[0]
    return short.split("::")[-1] if "::" in short else short


def find_nsys() -> str:
    nsys = shutil.which("nsys")
    if nsys is None:
        sys.exit("ERROR: 'nsys' not found on PATH. Load the CUDA/nsys module.")
    return nsys


def export_sqlite(nsys_bin: str, rep_path: Path, force: bool = False) -> Path:
    sqlite_path = rep_path.with_suffix(".sqlite")
    if sqlite_path.exists() and not force:
        return sqlite_path
    cmd = [nsys_bin, "export", "--type", "sqlite", "-o", str(sqlite_path),
           "--force-overwrite", "true", str(rep_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not sqlite_path.exists():
        raise RuntimeError(f"nsys export failed for {rep_path.name}:\n{result.stderr}")
    return sqlite_path


def table_exists(conn, name):
    return conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def sum_duration(conn, table):
    if not table_exists(conn, table):
        return 0
    row = conn.execute(f"SELECT COALESCE(SUM(end-start),0) FROM {table}").fetchone()
    return row[0] or 0


def sm_active_pct(conn):
    if table_exists(conn, "GPU_METRICS") and table_exists(conn, "TARGET_INFO_GPU_METRICS"):
        placeholders = ",".join("?" for _ in SM_ACTIVE_NAMES)
        row = conn.execute(
            f"SELECT metricId FROM TARGET_INFO_GPU_METRICS WHERE metricName IN ({placeholders}) LIMIT 1",
            SM_ACTIVE_NAMES,
        ).fetchone()
        if row is not None:
            avg_row = conn.execute(
                "SELECT AVG(value) FROM GPU_METRICS WHERE metricId = ?", (row[0],)
            ).fetchone()
            if avg_row and avg_row[0] is not None:
                return float(avg_row[0]), False
    return None, True


def analyze_report(rep_path: Path, nsys_bin: str, force_export: bool, gpu_spec: dict):
    sqlite_path = export_sqlite(nsys_bin, rep_path, force=force_export)
    conn = sqlite3.connect(str(sqlite_path))
    try:
        kernel_ns = sum_duration(conn, "CUPTI_ACTIVITY_KIND_KERNEL")
        memcpy_ns = sum_duration(conn, "CUPTI_ACTIVITY_KIND_MEMCPY")
        memset_ns = sum_duration(conn, "CUPTI_ACTIVITY_KIND_MEMSET")

        t0t1 = conn.execute(
            "SELECT MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ).fetchone()
        wall_ns = (t0t1[1] - t0t1[0]) if (t0t1 and t0t1[0] is not None) else 0
        gpu_ns = kernel_ns + memcpy_ns + memset_ns
        other_ns = max(wall_ns - gpu_ns, 0)

        util_pct, util_is_fallback = sm_active_pct(conn)

        total_bytes = 0
        if table_exists(conn, "CUPTI_ACTIVITY_KIND_MEMCPY"):
            row = conn.execute("SELECT COALESCE(SUM(bytes),0) FROM CUPTI_ACTIVITY_KIND_MEMCPY").fetchone()
            total_bytes = row[0] or 0
        achieved_bw_GBs = (total_bytes / 1e9) / (memcpy_ns / 1e9) if memcpy_ns > 0 else 0.0
        bw_util_pct = achieved_bw_GBs / gpu_spec["peak_bw_GBs"] * 100.0 if gpu_spec["peak_bw_GBs"] else None

        launches_total = 0
        if table_exists(conn, "CUPTI_ACTIVITY_KIND_KERNEL"):
            launches_total = conn.execute(
                "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL"
            ).fetchone()[0]

        summary = {
            "wall_s": wall_ns / 1e9,
            "kernel_s": kernel_ns / 1e9,
            "memcpy_s": memcpy_ns / 1e9,
            "memset_s": memset_ns / 1e9,
            "other_s": other_ns / 1e9,
            "kernel_pct": (kernel_ns / wall_ns * 100.0) if wall_ns else None,
            "memcpy_pct": (memcpy_ns / wall_ns * 100.0) if wall_ns else None,
            "memset_pct": (memset_ns / wall_ns * 100.0) if wall_ns else None,
            "other_pct": (other_ns / wall_ns * 100.0) if wall_ns else None,
            "gpu_util_pct": round(util_pct, 2) if util_pct is not None else None,
            "gpu_util_is_fallback": util_is_fallback,
            "memcpy_ovh_pct": round(memcpy_ns / gpu_ns * 100.0, 2) if gpu_ns else None,
            "achieved_bw_GBs": round(achieved_bw_GBs, 2),
            "bw_util_pct": round(bw_util_pct, 2) if bw_util_pct is not None else None,
            "total_launches": launches_total,
        }

        # ── per-kernel breakdown ──────────────────────────────────────
        kernel_rows = []
        if table_exists(conn, "CUPTI_ACTIVITY_KIND_KERNEL"):
            query = """
                SELECT s.value, (k.end-k.start) AS dur, k.gridX, k.gridY, k.gridZ,
                       k.blockX, k.blockY, k.blockZ
                FROM CUPTI_ACTIVITY_KIND_KERNEL k
                JOIN StringIds s ON s.id = k.shortName
            """
            groups = defaultdict(lambda: {"dur": 0, "launches": 0, "grid": [], "block": []})
            for name, dur, gx, gy, gz, bx, by, bz in conn.execute(query):
                kname = simplify_kernel_name(name)
                g = groups[kname]
                g["dur"] += dur
                g["launches"] += 1
                g["grid"].append(gx * gy * gz)
                g["block"].append(bx * by * bz)

            for kname, g in groups.items():
                kernel_rows.append({
                    "Kernel": kname,
                    "Launches": g["launches"],
                    "Total_s": round(g["dur"] / 1e9, 4),
                    "SharePct": round(g["dur"] / kernel_ns * 100.0, 2) if kernel_ns else None,
                    "AvgGrid": round(sum(g["grid"]) / len(g["grid"]), 1),
                    "AvgBlock": round(sum(g["block"]) / len(g["block"]), 1),
                })
            kernel_rows.sort(key=lambda r: r["Total_s"], reverse=True)
    finally:
        conn.close()

    return summary, kernel_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("."))
    ap.add_argument("--years", type=int, nargs="+", default=[3, 10, 30])
    ap.add_argument("--growth", type=int, nargs="+", default=[1, 3, 6])
    ap.add_argument("--pattern", type=str, default=LABEL_PATTERN_DEFAULT)
    ap.add_argument("--gpu", choices=sorted(GPU_SPECS), default="h100")
    ap.add_argument("--grid-kernel", default="Stage1",
                     help="Which simplified kernel name to plot grid-size-vs-SM-count for")
    ap.add_argument("--out-prefix", default="pa_paper")
    ap.add_argument("--force-export", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    nsys_bin = find_nsys()
    gpu_spec = GPU_SPECS[args.gpu]

    summary_rows = []
    kernel_rows_all = []

    for years in args.years:
        for growth in args.growth:
            fname = args.pattern.format(years=years, growth=growth)
            rep_path = args.dir / fname
            label = f"{years}e/{growth}%"

            if not rep_path.exists():
                print(f"  [skip] {label}: {rep_path} not found")
                continue

            print(f"  [ok]   {label}: {rep_path.name}")
            try:
                summary, kernels = analyze_report(rep_path, nsys_bin, args.force_export, gpu_spec)
            except Exception as e:
                print(f"    ERROR: {e}")
                continue

            if summary["gpu_util_is_fallback"]:
                print(f"    NOTE: {label} has no GPU Metrics collected; "
                      f"GPU_util_pct is the kernel-busy/wall-time approximation")

            summary_row = {"Label": label, "Years": years, "Growth": growth, **summary}
            del summary_row["gpu_util_is_fallback"]
            summary_rows.append(summary_row)

            for k in kernels:
                kernel_rows_all.append({"Label": label, "Years": years, "Growth": growth, **k})

    if not summary_rows:
        sys.exit("ERROR: no reports parsed successfully")

    # ── Write CSVs ───────────────────────────────────────────────────────
    summary_fields = ["Label", "Years", "Growth", "wall_s", "kernel_s", "memcpy_s",
                       "memset_s", "other_s", "kernel_pct", "memcpy_pct", "memset_pct",
                       "other_pct", "gpu_util_pct", "memcpy_ovh_pct", "achieved_bw_GBs",
                       "bw_util_pct", "total_launches"]
    summary_csv = f"{args.out_prefix}_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nWrote {len(summary_rows)} rows -> {summary_csv}")

    kernel_fields = ["Label", "Years", "Growth", "Kernel", "Launches", "Total_s",
                      "SharePct", "AvgGrid", "AvgBlock"]
    kernels_csv = f"{args.out_prefix}_kernels.csv"
    with open(kernels_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=kernel_fields)
        w.writeheader()
        w.writerows(kernel_rows_all)
    print(f"Wrote {len(kernel_rows_all)} rows -> {kernels_csv}")

    if args.no_plots:
        return

    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib/numpy not available -- skipping plots")
        return

    # ── Plot 1: time decomposition, faceted by years, x=growth ─────────
    years_list = sorted(set(r["Years"] for r in summary_rows))
    growth_list = sorted(set(r["Growth"] for r in summary_rows))
    fig, axes = plt.subplots(1, len(years_list), figsize=(4.2 * len(years_list), 5), sharey=True)
    if len(years_list) == 1:
        axes = [axes]

    components = [("kernel_pct", "Kernel", "#1976D2"),
                  ("memcpy_pct", "Memcpy", "#E65100"),
                  ("memset_pct", "Memset", "#9C27B0"),
                  ("other_pct", "Other (host-side)", "#B71C1C")]

    by_label = {r["Label"]: r for r in summary_rows}
    for ax, years in zip(axes, years_list):
        x = np.arange(len(growth_list))
        bottoms = np.zeros(len(growth_list))
        for field, name, color in components:
            vals = np.array([
                (by_label.get(f"{years}e/{g}%", {}).get(field) or 0) for g in growth_list
            ])
            ax.bar(x, vals, bottom=bottoms, color=color, label=name, edgecolor="white", linewidth=0.5)
            bottoms += vals
        ax.set_xticks(x)
        ax.set_xticklabels([f"{g}%" for g in growth_list])
        ax.set_title(f"{years}-year horizon", fontweight="bold")
        ax.set_xlabel("Growth rate")
        if ax is axes[0]:
            ax.set_ylabel("% of wall-clock time")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
    fig.suptitle("GPU-Time Decomposition Across Scale", fontweight="bold")
    fig.tight_layout()
    decomp_path = f"{args.out_prefix}_time_decomposition.png"
    fig.savefig(decomp_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {decomp_path}")

    # ── Plot 2: grid size vs SM count for the chosen kernel ─────────────
    target = args.grid_kernel
    by_scenario_kernel = {(r["Label"]): r for r in kernel_rows_all if r["Kernel"] == target}
    labels_ordered = [f"{y}e/{g}%" for y in years_list for g in growth_list
                      if f"{y}e/{g}%" in by_scenario_kernel]
    grids = [by_scenario_kernel[l]["AvgGrid"] for l in labels_ordered]

    if grids:
        fig2, ax2 = plt.subplots(figsize=(9, 5))
        x = np.arange(len(labels_ordered))
        ax2.bar(x, grids, color="#2E7D32")
        ax2.axhline(gpu_spec["sm_count"], color="#C62828", linestyle="--", linewidth=2,
                    label=f"{gpu_spec['name']} SM count ({gpu_spec['sm_count']})")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels_ordered, rotation=45, ha="right")
        ax2.set_ylabel("Average grid size (blocks)")
        ax2.set_title(f"{target} Grid Size vs. SM Count Across Scale", fontweight="bold")
        ax2.legend()
        fig2.tight_layout()
        grid_path = f"{args.out_prefix}_grid_saturation.png"
        fig2.savefig(grid_path, dpi=200, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved -> {grid_path}")
    else:
        print(f"\nNOTE: no kernel named '{target}' found in any report -- skipping grid-saturation plot. "
              f"Available kernel names: {sorted(set(r['Kernel'] for r in kernel_rows_all))}")


if __name__ == "__main__":
    main()
