#!/usr/bin/env python3
"""
parse_nsys_gpu_util.py
────────────────────────────────────────────────────────────────────────
Parses a sweep of .nsys-rep files (one per years x growth-rate scenario)
and produces a CSV of:
    Label, GPU_util_pct, Memcpy_ovh_pct

matching the format of pa_h100_gpu_util.csv:
    Label,GPU_util_pct,Memcpy_ovh_pct
    3e/1%,20.07,7.93
    ...

GPU_util_pct  = average "SM Active" GPU Metric over the whole capture
                (the "gross" utilization defined in NVIDIA's own writeup:
                https://developer.nvidia.com/blog/measuring-the-gpu-occupancy-of-multi-stream-workloads/)
                REQUIRES the .nsys-rep to have been collected with
                --gpu-metrics-device=<idx> (or --gpu-metrics-devices=all on
                newer nsys). If GPU Metrics were not collected, this script
                falls back to an approximate "kernel busy time / wall time"
                ratio and flags it as such in the output.

Memcpy_ovh_pct = memcpy+memset time / (memcpy+memset time + kernel time) * 100
                 i.e. what fraction of GPU-side work was data movement
                 rather than compute.

Expects files named like:  nsys-pa-<years>y-<growth>p.nsys-rep
  e.g. nsys-pa-3y-1p.nsys-rep, nsys-pa-30y-6p.nsys-rep

Usage
─────
    python parse_nsys_gpu_util.py --dir ./profile \
        --years 3 10 30 --growth 1 3 5 6 \
        --pattern "nsys-pa-{years}y-{growth}p.nsys-rep" \
        --out pa_h100_gpu_util.csv

If your files use a different naming convention, override --pattern using
the same {years}/{growth} placeholders.
"""

import argparse
import csv
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path


SM_ACTIVE_NAMES = (
    "SM Active", "SMs Active",
    "SM Active [Throughput %]", "SMs Active [Throughput %]",
)


def find_nsys() -> str:
    nsys = shutil.which("nsys")
    if nsys is None:
        sys.exit("ERROR: 'nsys' not found on PATH. Load the CUDA/nsys module "
                 "on the machine that collected these reports (this script "
                 "must run where nsys is installed).")
    return nsys


def export_sqlite(nsys_bin: str, rep_path: Path, force: bool = False) -> Path:
    """Export a .nsys-rep to .sqlite (reuses existing export unless --force)."""
    sqlite_path = rep_path.with_suffix(".sqlite")
    if sqlite_path.exists() and not force:
        return sqlite_path
    cmd = [nsys_bin, "export", "--type", "sqlite", "-o", str(sqlite_path),
           "--force-overwrite", "true", str(rep_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not sqlite_path.exists():
        raise RuntimeError(
            f"nsys export failed for {rep_path.name}:\n{result.stderr}"
        )
    return sqlite_path


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def gpu_util_pct(conn: sqlite3.Connection):
    """Return (gpu_util_pct, is_approximate)."""
    if table_exists(conn, "GPU_METRICS") and table_exists(conn, "TARGET_INFO_GPU_METRICS"):
        placeholders = ",".join("?" for _ in SM_ACTIVE_NAMES)
        row = conn.execute(
            f"SELECT metricId FROM TARGET_INFO_GPU_METRICS "
            f"WHERE metricName IN ({placeholders}) LIMIT 1",
            SM_ACTIVE_NAMES,
        ).fetchone()
        if row is not None:
            metric_id = row[0]
            avg_row = conn.execute(
                "SELECT AVG(value) FROM GPU_METRICS WHERE metricId = ?",
                (metric_id,),
            ).fetchone()
            if avg_row is not None and avg_row[0] is not None:
                return float(avg_row[0]), False

    # Fallback: approximate utilization as kernel-busy-time / wall-time.
    # This ignores overlap between concurrent kernels/streams and is only
    # a rough proxy -- true SM Active requires --gpu-metrics-device(s).
    if not table_exists(conn, "CUPTI_ACTIVITY_KIND_KERNEL"):
        return None, True
    row = conn.execute(
        "SELECT SUM(end - start), MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL"
    ).fetchone()
    if row is None or row[0] is None or row[1] is None:
        return None, True
    busy_ns, t0, t1 = row
    wall_ns = t1 - t0
    if wall_ns <= 0:
        return None, True
    return busy_ns / wall_ns * 100.0, True


def memcpy_overhead_pct(conn: sqlite3.Connection):
    def summed(table):
        if not table_exists(conn, table):
            return 0
        row = conn.execute(f"SELECT COALESCE(SUM(end - start), 0) FROM {table}").fetchone()
        return row[0] or 0

    memop_ns = summed("CUPTI_ACTIVITY_KIND_MEMCPY") + summed("CUPTI_ACTIVITY_KIND_MEMSET")
    kernel_ns = summed("CUPTI_ACTIVITY_KIND_KERNEL")
    total = memop_ns + kernel_ns
    if total == 0:
        return None
    return memop_ns / total * 100.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("."),
                     help="Directory containing the .nsys-rep files")
    ap.add_argument("--years", type=int, nargs="+", default=[3, 10, 30])
    ap.add_argument("--growth", type=int, nargs="+", default=[1, 3, 5, 6])
    ap.add_argument("--pattern", type=str, default="nsys-pa-{years}y-{growth}p.nsys-rep",
                     help="Filename pattern with {years}/{growth} placeholders")
    ap.add_argument("--out", type=Path, default=Path("pa_h100_gpu_util.csv"))
    ap.add_argument("--force-export", action="store_true",
                     help="Re-export .sqlite even if one already exists")
    args = ap.parse_args()

    nsys_bin = find_nsys()
    rows_out = []

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
                sqlite_path = export_sqlite(nsys_bin, rep_path, force=args.force_export)
            except RuntimeError as e:
                print(f"    ERROR exporting: {e}")
                continue

            conn = sqlite3.connect(str(sqlite_path))
            try:
                util, approx = gpu_util_pct(conn)
                ovh = memcpy_overhead_pct(conn)
            finally:
                conn.close()

            if util is None:
                print(f"    WARNING: no kernel/GPU-metric data found for {label}")
                util = float("nan")
            elif approx:
                print(f"    NOTE: {label} has no GPU Metrics collected "
                      f"(no --gpu-metrics-device at profile time); "
                      f"GPU_util_pct is an approximate kernel-busy/wall-time ratio")

            rows_out.append({
                "Label": label,
                "GPU_util_pct": round(util, 2) if util == util else "",  # NaN check
                "Memcpy_ovh_pct": round(ovh, 2) if ovh is not None else "",
            })

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Label", "GPU_util_pct", "Memcpy_ovh_pct"])
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\nWrote {len(rows_out)} rows -> {args.out}")


if __name__ == "__main__":
    main()
