#!/usr/bin/env python3
"""
parse_nsys_occupancy.py
────────────────────────────────────────────────────────────────────────
Parses a sweep of .nsys-rep files and produces a per-kernel occupancy CSV
matching the format of pa_h100_occupancy.csv:

    Label,Kernel,Launches,Total_ms,Avg_ms,Block,Grid,Regs,Shmem_static_B,
    Shmem_dyn_B,Lmem_B,ThOcc_pct,Limiter,pct_GPU

Everything except ThOcc_pct/Limiter comes directly from nsys's
CUPTI_ACTIVITY_KIND_KERNEL table (launch parameters + durations) -- nsys
captures these at collection time regardless of whether GPU Metrics were
enabled.

ThOcc_pct / Limiter (theoretical occupancy) are NOT stored by nsys anywhere
-- nsys's GUI computes this live from launch params + device limits, but
doesn't persist it to the sqlite export. This script recomputes it using
the standard CUDA occupancy-calculator formula:

    warps_per_block   = ceil(threads_per_block / 32)
    regs_per_warp     = round_up(regs_per_thread * 32, REG_ALLOC_UNIT)
    blocks_by_reg     = floor(MAX_REGS_PER_SM / regs_per_warp / warps_per_block)
    blocks_by_thread  = floor(MAX_THREADS_PER_SM / threads_per_block)
    blocks_by_shmem   = floor(MAX_SHMEM_PER_SM / round_up(shmem_per_block, SHMEM_ALLOC_UNIT))
    blocks_by_hw      = MAX_BLOCKS_PER_SM
    active_blocks     = min(blocks_by_reg, blocks_by_thread, blocks_by_shmem, blocks_by_hw)
    ThOcc_pct         = active_blocks * warps_per_block / MAX_WARPS_PER_SM * 100

Verified against pa_h100_occupancy.csv: this formula reproduces every
ThOcc_pct value in that reference file exactly (register- and
thread-limited rows all match to 1 decimal place) using the H100 (Hopper,
compute capability 9.0) constants below.

CAVEAT: the "Limiter" label for library kernels that declare very small
static/dynamic shared memory (e.g. cuCollections' cuco::insert) may not
match a tool that accounts for compiler-inserted/reserved shared memory
beyond the declared static amount -- in the reference data, two such rows
are labeled "shmem" as the limiter even though the register-limit alone
already produces the same active-block count. Where multiple limits tie,
this script reports whichever is smallest and picks "reg" as a tiebreak
before "shmem"/"thr"/"blk"; treat the Limiter column as indicative, and
cross-check against `ncu --metrics launch__occupancy_theoretical` for any
kernel where the exact limiter matters for your analysis.

Expects files named like:  nsys-pa-<years>y-<growth>p.nsys-rep

Usage
─────
    python parse_nsys_occupancy.py --dir ./profile \
        --years 3 10 30 --growth 1 3 5 6 \
        --pattern "nsys-pa-{years}y-{growth}p.nsys-rep" \
        --gpu h100 \
        --out pa_h100_occupancy.csv

    python parse_nsys_occupancy.py --dir ./profile \
        --years 30 --growth 6 \
        --pattern "nsys-pa-{years}y-{growth}p.nsys-rep" \
        --gpu h100 \
        --out pa_30y6p_h100_occupancy.csv
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


# ── GPU architecture occupancy constants ────────────────────────────────
# (compute-capability-level limits used by the CUDA occupancy calculator)
GPU_LIMITS = {
    "h100": dict(   # Hopper, compute capability 9.0
        max_threads_per_sm=2048,
        max_warps_per_sm=64,
        max_blocks_per_sm=32,
        max_regs_per_sm=65536,
        reg_alloc_unit=256,     # registers rounded up to this per warp
        max_shmem_per_sm=233472,  # 228 KiB
        shmem_alloc_unit=128,
        warp_size=32,
    ),
    "rtx4090": dict(  # Ada Lovelace, compute capability 8.9
        max_threads_per_sm=1536,
        max_warps_per_sm=48,
        max_blocks_per_sm=24,
        max_regs_per_sm=65536,
        reg_alloc_unit=256,
        max_shmem_per_sm=102400,  # 100 KiB
        shmem_alloc_unit=128,
        warp_size=32,
    ),
    "a100": dict(   # Ampere, compute capability 8.0
        max_threads_per_sm=2048,
        max_warps_per_sm=64,
        max_blocks_per_sm=32,
        max_regs_per_sm=65536,
        reg_alloc_unit=256,
        max_shmem_per_sm=167936,  # 164 KiB
        shmem_alloc_unit=128,
        warp_size=32,
    ),
}


def round_up(x: int, unit: int) -> int:
    return ((x + unit - 1) // unit) * unit


def theoretical_occupancy(threads_per_block, regs_per_thread, shmem_per_block, limits):
    warp_size = limits["warp_size"]
    warps_per_block = max(1, -(-threads_per_block // warp_size))  # ceil div

    # Register limit
    if regs_per_thread > 0:
        regs_per_warp = round_up(regs_per_thread * warp_size, limits["reg_alloc_unit"])
        blocks_by_reg = (limits["max_regs_per_sm"] // regs_per_warp) // warps_per_block
    else:
        blocks_by_reg = limits["max_blocks_per_sm"]

    # Thread limit
    blocks_by_thread = limits["max_threads_per_sm"] // max(threads_per_block, 1)

    # Shared-memory limit
    if shmem_per_block > 0:
        shmem_alloc = round_up(shmem_per_block, limits["shmem_alloc_unit"])
        blocks_by_shmem = limits["max_shmem_per_sm"] // shmem_alloc
    else:
        blocks_by_shmem = limits["max_blocks_per_sm"]

    # Hardware block limit
    blocks_by_hw = limits["max_blocks_per_sm"]

    candidates = [
        ("reg", blocks_by_reg),
        ("thr", blocks_by_thread),
        ("shmem", blocks_by_shmem),
        ("blk", blocks_by_hw),
    ]
    limiter, active_blocks = min(candidates, key=lambda kv: kv[1])
    active_blocks = max(0, min(active_blocks, limits["max_blocks_per_sm"]))

    occ_pct = active_blocks * warps_per_block / limits["max_warps_per_sm"] * 100.0
    occ_pct = min(occ_pct, 100.0)
    return round(occ_pct, 1), limiter


# ── Kernel name simplification ──────────────────────────────────────────
NAME_PATTERNS = [
    (re.compile(r"kernelCallStage1(_warped)?", re.I), "Stage1"),
    (re.compile(r"kernelCallStage2", re.I), "Stage2"),
    (re.compile(r"kernelCallStage3", re.I), "Stage3"),
    (re.compile(r"kernelCallStage4", re.I), "Stage4"),
    (re.compile(r"ABMKernel", re.I), "ABMKernel"),
    (re.compile(r"ExtractSizes", re.I), "ExtractSizes"),
    (re.compile(r"cuco::[a-zA-Z_]+"), None),   # keep the matched cuco::xxx as-is
    (re.compile(r"cub::[a-zA-Z_]+"), None),    # keep the matched cub::xxx as-is
    (re.compile(r"thrust::[a-zA-Z_:]+"), None),
]


def simplify_kernel_name(raw_name: str) -> str:
    for pattern, replacement in NAME_PATTERNS:
        m = pattern.search(raw_name)
        if m:
            return replacement if replacement is not None else m.group(0)
    # Fallback: strip template args / namespaces, keep first identifier chunk
    short = raw_name.split("(")[0]
    short = short.split("<")[0]
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


def extract_kernel_rows(conn: sqlite3.Connection):
    """Yield raw per-launch kernel records from the sqlite export."""
    query = """
        SELECT
            s.value            AS raw_name,
            k.registersPerThread,
            k.staticSharedMemory,
            k.dynamicSharedMemory,
            k.localMemoryPerThread,
            k.blockX, k.blockY, k.blockZ,
            k.gridX, k.gridY, k.gridZ,
            (k.end - k.start)  AS dur_ns
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id = k.shortName
    """
    try:
        cur = conn.execute(query)
    except sqlite3.OperationalError as e:
        # localMemoryPerThread may not exist in some schema versions
        if "localMemoryPerThread" in str(e):
            query = query.replace("k.localMemoryPerThread,", "0 AS localMemoryPerThread,")
            cur = conn.execute(query)
        else:
            raise
    return cur.fetchall()


def process_report(rep_path: Path, nsys_bin: str, limits, force_export: bool):
    sqlite_path = export_sqlite(nsys_bin, rep_path, force=force_export)
    conn = sqlite3.connect(str(sqlite_path))
    try:
        raw_rows = extract_kernel_rows(conn)
    finally:
        conn.close()

    # Group launches by (simplified kernel name, block/grid/reg/shmem signature)
    groups = defaultdict(lambda: {"launches": 0, "dur_ns": 0})
    for row in raw_rows:
        (raw_name, regs, shmem_static, shmem_dyn, lmem,
         bx, by, bz, gx, gy, gz, dur_ns) = row
        kname = simplify_kernel_name(raw_name)
        block = bx * by * bz
        grid = gx * gy * gz
        key = (kname, block, grid, regs, shmem_static, shmem_dyn, lmem or 0)
        groups[key]["launches"] += 1
        groups[key]["dur_ns"] += dur_ns

    total_ns = sum(g["dur_ns"] for g in groups.values()) or 1

    out_rows = []
    for (kname, block, grid, regs, shmem_static, shmem_dyn, lmem), stats in groups.items():
        total_ms = stats["dur_ns"] / 1e6
        avg_ms = total_ms / stats["launches"] if stats["launches"] else 0.0
        occ_pct, limiter = theoretical_occupancy(
            block, regs, shmem_static + shmem_dyn, limits
        )
        out_rows.append({
            "Kernel": kname,
            "Launches": stats["launches"],
            "Total_ms": round(total_ms, 4),
            "Avg_ms": round(avg_ms, 5),
            "Block": block,
            "Grid": grid,
            "Regs": regs,
            "Shmem_static_B": shmem_static,
            "Shmem_dyn_B": shmem_dyn,
            "Lmem_B": lmem,
            "ThOcc_pct": occ_pct,
            "Limiter": limiter,
            "pct_GPU": round(stats["dur_ns"] / total_ns * 100.0, 2),
        })

    out_rows.sort(key=lambda r: r["Total_ms"], reverse=True)
    return out_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("."))
    ap.add_argument("--years", type=int, nargs="+", default=[3, 10, 30])
    ap.add_argument("--growth", type=int, nargs="+", default=[1, 3, 5, 6])
    ap.add_argument("--pattern", type=str, default="nsys-pa-{years}y-{growth}p.nsys-rep")
    ap.add_argument("--gpu", choices=sorted(GPU_LIMITS), default="h100")
    ap.add_argument("--out", type=Path, default=Path("pa_h100_occupancy.csv"))
    ap.add_argument("--force-export", action="store_true")
    args = ap.parse_args()

    nsys_bin = find_nsys()
    limits = GPU_LIMITS[args.gpu]
    all_rows = []

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
                rows = process_report(rep_path, nsys_bin, limits, args.force_export)
            except RuntimeError as e:
                print(f"    ERROR: {e}")
                continue

            for r in rows:
                r["Label"] = label
            all_rows.extend(rows)

    fieldnames = ["Label", "Kernel", "Launches", "Total_ms", "Avg_ms", "Block",
                  "Grid", "Regs", "Shmem_static_B", "Shmem_dyn_B", "Lmem_B",
                  "ThOcc_pct", "Limiter", "pct_GPU"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
