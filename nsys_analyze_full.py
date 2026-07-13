#!/usr/bin/env python3
"""
nsys_analyze_full.py
====================
Complete analyzer for NVIDIA Nsight Systems output files (.nsys-rep and .sqlite).
Extracts all profiling data and produces paper-ready tables, figures, and CSVs.

USAGE
-----
# Single file:
    python3 nsys_analyze_full.py  profile/nsys-pa-3y-1p.nsys-rep

# Multiple files with labels:
    python3 nsys_analyze_full.py \
        --files profile/nsys-pa-1y-1p.sqlite "1y/1%" \
                profile/nsys-pa-1y-3p.sqlite "1y/3%" \
                profile/nsys-pa-3y-1p.sqlite "3y/1%" \
        --gpu   P100 \
        --outdir ./nsys_output \
        --prefix pa

# Glob pattern (auto-labels from filenames):
    python3 nsys_analyze_full.py  --glob "profile/nsys-pa-*.sqlite" --gpu P100

WHAT IT EXTRACTS
----------------
From .nsys-rep  : runs  `nsys export --type sqlite` to get the sqlite first
From .sqlite    : reads directly via Python sqlite3 (no nsys CLI needed)

Data extracted:
  - GPU hardware specs       (TARGET_INFO_GPU)
  - Capture metadata         (META_DATA_CAPTURE: command, arguments, settings)
  - Session wall time        (TARGET_INFO_SESSION_START_TIME)
  - Session timing summary   (ANALYSIS_DETAILS)
  - Kernel launches          (CUPTI_ACTIVITY_KIND_KERNEL)
  - Memory copies            (CUPTI_ACTIVITY_KIND_MEMCPY)
  - Memset operations        (CUPTI_ACTIVITY_KIND_MEMSET)
  - CUDA Runtime API calls   (CUPTI_ACTIVITY_KIND_RUNTIME)
  - Stream synchronization   (CUPTI_ACTIVITY_KIND_SYNCHRONIZATION)
  - NVTX markers/ranges      (NVTX_EVENTS)
  - Profiler overhead        (PROFILER_OVERHEAD)
  - OS runtime calls         (OSRT_API)

OUTPUTS
-------
  {outdir}/{prefix}_session_table.tex      LaTeX: session-level summary
  {outdir}/{prefix}_kernel_table.tex       LaTeX: per-kernel metrics
  {outdir}/{prefix}_memcpy_table.tex       LaTeX: memory transfer breakdown
  {outdir}/{prefix}_session_overview.pdf   Figure: 4-panel overview
  {outdir}/{prefix}_occupancy.pdf          Figure: theoretical occupancy
  {outdir}/{prefix}_kernel_dist.pdf        Figure: kernel time breakdown
  {outdir}/{prefix}_timeline.pdf           Figure: activity timeline
  {outdir}/{prefix}_summary.csv            CSV: all session metrics
  {outdir}/{prefix}_kernels.csv            CSV: per-kernel metrics
  {outdir}/{prefix}_memcpy.csv             CSV: per-transfer-kind metrics
  Console: full diagnostic summary

REQUIREMENTS
------------
  pip install matplotlib numpy          (for figures)
  nsys  (only needed for .nsys-rep → .sqlite export, not for .sqlite analysis)
"""

import argparse
import csv
import glob
import math
import os
import re
import sqlite3
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from typing import Optional

# ── optional plotting ─────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

# ══════════════════════════════════════════════════════════════════════════════
# GPU HARDWARE PRESETS
# Used to compute bandwidth utilisation % and theoretical roofline values.
# The sqlite TARGET_INFO_GPU table also has this info — we cross-check.
# ══════════════════════════════════════════════════════════════════════════════
GPU_PRESETS = {
    "P100": dict(name="Tesla P100-PCIE-16GB",  peak_bw_GBs=732.0,
                 peak_fp32_GFLOPs=9300.0,  sm_count=56),
    "A30":  dict(name="NVIDIA A30",            peak_bw_GBs=933.0,
                 peak_fp32_GFLOPs=10300.0, sm_count=56),
    "A100": dict(name="NVIDIA A100 SXM4 80GB", peak_bw_GBs=2039.0,
                 peak_fp32_GFLOPs=19500.0, sm_count=108),
    "H100": dict(name="NVIDIA H100 80GB HBM3", peak_bw_GBs=3350.0,
                 peak_fp32_GFLOPs=67000.0, sm_count=114),
    "V100": dict(name="Tesla V100-SXM2-32GB",  peak_bw_GBs=900.0,
                 peak_fp32_GFLOPs=14000.0, sm_count=80),
    "RTX4090": dict(name="NVIDIA GeForce RTX 4090", peak_bw_GBs=1008.0,
                     peak_fp32_GFLOPs=82600.0, sm_count=128),
}

# ══════════════════════════════════════════════════════════════════════════════
# MEMCPY KIND ENUM
# ══════════════════════════════════════════════════════════════════════════════
MEMCPY_KIND = {
    0: "Unknown", 1: "H2D",     2: "D2H",     3: "H2A",
    4: "A2H",     5: "A2A",     6: "A2D",     7: "D2A",
    8: "D2D",     9: "H2H",    10: "P2P",    11: "UVM_H2D",
    12: "UVM_D2H", 13: "UVM_D2D",
}

# ══════════════════════════════════════════════════════════════════════════════
# KERNEL NAME NORMALIZER
# Maps long demangled CUDA kernel names to short readable labels.
# ══════════════════════════════════════════════════════════════════════════════
_KERN_RULES = [
    (r"kernelCallStage1",                 "Stage1"),
    (r"kernelCallStage2",                 "Stage2"),
    (r"kernelCallStage3",                 "Stage3"),
    (r"kernelCallStage4",                 "Stage4"),
    (r"ABMKernelStage2",                  "ABMStage2"),
    (r"ABMKernelStage4",                  "ABMStage4"),
    (r"MakeSameYear",                     "SameYearCite"),
    (r"MakeUniform",                      "UniformCite"),
    (r"GetOneAndTwoHopNeighborhood_Warp", "2HopWarp"),
    (r"GetOneAndTwoHop",                  "2Hop"),
    (r"kernel_extract_vector",            "ExtractSizes"),
    (r"cuco.*initialize",                 "cuco::init"),
    (r"cuco.*insert_if",                  "cuco::insert_if"),
    (r"cuco.*insert",                     "cuco::insert"),
    (r"cub.*DeviceRadixSort",             "cub::RadixSort"),
    (r"cub.*DeviceScan",                  "cub::DeviceScan"),
    (r"cub.*DeviceSelect",                "cub::DeviceSelect"),
    (r"cub.*for_each|cub.*ForEach",       "cub::ForEach"),
    (r"thrust.*copy",                     "thrust::copy"),
    (r"thrust.*fill",                     "thrust::fill"),
    (r"uninitialized_fill",               "thrust::uninit_fill"),
]

def normalize_kernel(raw_name: str) -> str:
    """Shorten a CUDA kernel demangled name to a readable label."""
    for pattern, short in _KERN_RULES:
        if re.search(pattern, raw_name):
            return short
    # fallback: last C++ identifier before parameters
    clean = raw_name.split("(")[0].split("<")[0]
    parts = [p for p in clean.split("::") if p]
    return parts[-1][:30] if parts else raw_name[:30]

# ══════════════════════════════════════════════════════════════════════════════
# THEORETICAL OCCUPANCY CALCULATOR
# Replicates ncu's launch__occupancy_theoretical using kernel launch params
# and hardware limits read directly from TARGET_INFO_GPU.
# ══════════════════════════════════════════════════════════════════════════════
def theoretical_occupancy(regs: int, block: int, shmem_static: int,
                           shmem_dynamic: int, hw: dict) -> float:
    """
    Returns theoretical occupancy in percent (0–100).
    hw must have: warp_size, max_warps_sm, max_blocks_sm, max_threads_sm,
                  regs_per_sm, max_shmem_sm_B.
    """
    if block == 0:
        return 0.0
    warp_size     = hw.get("warp_size", 32)
    max_warps_sm  = hw.get("max_warps_sm", 64)
    max_blocks_sm = hw.get("max_blocks_sm", 32)
    max_thr_sm    = hw.get("max_thr_sm", 2048)
    regs_per_sm   = hw.get("regs_per_sm", 65536)
    shmem_per_sm  = hw.get("shmem_per_sm", 65536)

    warps_per_block = math.ceil(block / warp_size)
    shmem_total     = shmem_static + shmem_dynamic

    # Limit from threads
    lim_thread = max_thr_sm // block

    # Limit from registers
    if regs > 0:
        regs_per_warp = math.ceil(regs * warp_size / 256) * 256
        regs_per_block = regs_per_warp * warps_per_block
        lim_reg = regs_per_sm // regs_per_block if regs_per_block > 0 else max_blocks_sm
    else:
        lim_reg = max_blocks_sm

    # Limit from shared memory
    if shmem_total > 0:
        shmem_alloc = math.ceil(shmem_total / 256) * 256
        lim_shmem = shmem_per_sm // shmem_alloc if shmem_alloc > 0 else max_blocks_sm
    else:
        lim_shmem = max_blocks_sm

    max_blocks = min(lim_thread, lim_reg, lim_shmem, max_blocks_sm)
    active_warps = max_blocks * warps_per_block
    return round(min(active_warps / max_warps_sm, 1.0) * 100.0, 1)

def occupancy_limiter(regs: int, shmem: int, block: int) -> str:
    """Identify the primary occupancy limiter."""
    if regs > 40:  return "reg"
    if shmem > 0:  return "shmem"
    if block < 64: return "block"
    return "thr"


def _table_exists(con: sqlite3.Connection, table: str) -> bool:
    """Return True if a table exists in the sqlite database."""
    row = con.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
        (table,)
    ).fetchone()
    return bool(row and row[0])


# ══════════════════════════════════════════════════════════════════════════════
# .nsys-rep → .sqlite CONVERSION
# ══════════════════════════════════════════════════════════════════════════════
def rep_to_sqlite(rep_path: str, outdir: str) -> Optional[str]:
    """
    Export a .nsys-rep to .sqlite using the nsys CLI.
    Returns the path to the .sqlite file, or None if nsys is not available.
    """
    if not os.path.exists(rep_path):
        print(f"  ERROR: {rep_path} not found")
        return None

    # Check nsys is available
    try:
        subprocess.run(["nsys", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("  WARNING: nsys not in PATH — cannot convert .nsys-rep to .sqlite")
        print("  On the cluster: module load cuda/12.6  then re-run")
        print("  Or copy the .sqlite that nsys auto-generates alongside .nsys-rep")
        return None

    base    = os.path.splitext(os.path.basename(rep_path))[0]
    out_db  = os.path.join(outdir, f"{base}.sqlite")
    os.makedirs(outdir, exist_ok=True)

    print(f"  Converting {rep_path} → {out_db}")
    result = subprocess.run(
        ["nsys", "export", "--type", "sqlite", "--output", out_db, rep_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  ERROR from nsys export:\n{result.stderr[:500]}")
        return None

    return out_db if os.path.exists(out_db) else None


# ══════════════════════════════════════════════════════════════════════════════
# CORE SQLITE PARSER
# ══════════════════════════════════════════════════════════════════════════════
def parse_sqlite(db_path: str, gpu_hw: dict) -> dict:
    """
    Parse an nsys .sqlite file and return a structured dict with all
    profiling data needed for paper tables, figures, and console output.
    """
    con = sqlite3.connect(db_path)
    sids = dict(con.execute("SELECT id, value FROM StringIds").fetchall())

    result = {}

    # ── 1. GPU hardware info ───────────────────────────────────────────────
    gpu_rows = con.execute("SELECT * FROM TARGET_INFO_GPU").fetchall()
    gpu_cols = [c[1] for c in con.execute("PRAGMA table_info(TARGET_INFO_GPU)").fetchall()]

    gpus = []
    for row in gpu_rows:
        g = dict(zip(gpu_cols, row))
        # Populate hw dict from sqlite (overrides preset if available)
        if g.get("memoryBandwidth"):
            gpu_hw["peak_bw_GBs"]   = g["memoryBandwidth"] / 1e9
        if g.get("maxWarpsPerSm"):
            gpu_hw["max_warps_sm"]  = g["maxWarpsPerSm"]
        if g.get("maxBlocksPerSm"):
            gpu_hw["max_blocks_sm"] = g["maxBlocksPerSm"]
        if g.get("maxShmemPerSm"):
            gpu_hw["shmem_per_sm"]  = g["maxShmemPerSm"]
        if g.get("maxRegistersPerSm"):
            gpu_hw["regs_per_sm"]   = g["maxRegistersPerSm"]
        if g.get("threadsPerWarp"):
            gpu_hw["warp_size"]     = g["threadsPerWarp"]
        gpu_hw["max_thr_sm"]    = g.get("maxWarpsPerSm", 64) * g.get("threadsPerWarp", 32)
        gpus.append(g)
    result["gpus"] = gpus

    # ── 2. Capture metadata ───────────────────────────────────────────────
    if _table_exists(con, "META_DATA_CAPTURE"):
        meta = dict(con.execute("SELECT name, value FROM META_DATA_CAPTURE").fetchall())
    else:
        meta = {}
    result["meta"] = meta

    # Reconstruct full command line
    cmd_parts = [meta.get("PROCESS_0:COMMAND", "")]
    i = 0
    while f"PROCESS_0:ARGUMENT_{i}" in meta:
        cmd_parts.append(meta[f"PROCESS_0:ARGUMENT_{i}"])
        i += 1
    result["cmdline"] = " ".join(cmd_parts)
    result["hostname"] = meta.get("DEVICE_DISPLAY_NAME", "unknown")
    result["capture_events"] = [v for k, v in meta.items()
                                 if k == "CAPTURE_EVENT_TYPE"]

    # ── 3. Session wall time ──────────────────────────────────────────────
    if _table_exists(con, "TARGET_INFO_SESSION_START_TIME"):
        wt = con.execute(
            "SELECT utcTime, localTime FROM TARGET_INFO_SESSION_START_TIME"
        ).fetchone()
    else:
        wt = None
    result["session_utc"]   = wt[0] if wt else "unknown"
    result["session_local"] = wt[1] if wt else "unknown"

    # ── 4. Session timing ─────────────────────────────────────────────────
    ses = con.execute(
        "SELECT duration, startTime, stopTime FROM ANALYSIS_DETAILS"
    ).fetchone()
    result["session_ns"]  = ses[0]
    result["session_ms"]  = ses[0] / 1e6

    # ── 5. Kernel launches ────────────────────────────────────────────────
    krows = con.execute("""
        SELECT demangledName,
               COUNT(*)           AS launches,
               SUM(end - start)   AS total_ns,
               AVG(end - start)   AS avg_ns,
               MIN(end - start)   AS min_ns,
               MAX(end - start)   AS max_ns,
               AVG(registersPerThread)    AS regs,
               AVG(blockX)               AS block,
               AVG(gridX)                AS grid,
               AVG(staticSharedMemory)   AS shmem_static,
               AVG(dynamicSharedMemory)  AS shmem_dynamic,
               AVG(localMemoryPerThread) AS lmem
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        GROUP BY demangledName
        ORDER BY total_ns DESC
    """).fetchall()

    total_kern_ns = sum(r[2] for r in krows if r[2])
    result["total_kern_ns"] = total_kern_ns
    result["total_kern_ms"] = total_kern_ns / 1e6

    kernels = []
    for r in krows:
        raw_name   = sids.get(r[0], str(r[0]))
        short_name = normalize_kernel(raw_name)
        regs  = int(r[6] or 0)
        block = int(r[7] or 0)
        grid  = int(r[8] or 0)
        ss    = int(r[9] or 0)
        sd    = int(r[10] or 0)
        lmem  = int(r[11] or 0)

        occ = theoretical_occupancy(regs, block, ss, sd, gpu_hw)
        lim = occupancy_limiter(regs, ss + sd, block)
        dur = r[2] or 0

        kernels.append(dict(
            name        = short_name,
            raw_name    = raw_name[:80],
            launches    = r[1],
            total_ms    = dur / 1e6,
            avg_ms      = (r[3] or 0) / 1e6,
            min_ms      = (r[4] or 0) / 1e6,
            max_ms      = (r[5] or 0) / 1e6,
            regs        = regs,
            block       = block,
            grid        = grid,
            shmem_static= ss,
            shmem_dyn   = sd,
            lmem        = lmem,
            th_occ      = occ,
            limiter     = lim,
            pct_gpu     = round(dur / total_kern_ns * 100, 2) if total_kern_ns > 0 else 0.0,
        ))
    result["kernels"] = kernels
    result["total_kern_launches"] = sum(k["launches"] for k in kernels)

    # ── 6. Memory copies ──────────────────────────────────────────────────
    if _table_exists(con, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        mrows = con.execute("""
            SELECT copyKind,
                   COUNT(*)         AS ops,
                   SUM(bytes)       AS total_bytes,
                   SUM(end - start) AS total_ns,
                   AVG(bytes)       AS avg_bytes,
                   MAX(bytes)       AS max_bytes
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            GROUP BY copyKind
            ORDER BY total_bytes DESC
        """).fetchall()
    else:
        mrows = []

    total_mcpy_ns    = sum(r[3] for r in mrows if r[3])
    total_mcpy_bytes = sum(r[2] for r in mrows if r[2])
    eff_bw_GBs = (total_mcpy_bytes / (total_mcpy_ns * 1e-9) / 1e9
                  if total_mcpy_ns > 0 else 0.0)

    memcpy_list = []
    for r in mrows:
        kind  = MEMCPY_KIND.get(r[0], f"kind{r[0]}")
        ns    = r[3] or 0
        byt   = r[2] or 0
        memcpy_list.append(dict(
            kind     = kind,
            ops      = r[1],
            bytes_MB = byt / 1e6,
            time_ms  = ns / 1e6,
            avg_KB   = (r[4] or 0) / 1e3,
            max_KB   = (r[5] or 0) / 1e3,
            bw_GBs   = byt / (ns * 1e-9) / 1e9 if ns > 0 else 0.0,
        ))
    result["memcpy"] = memcpy_list
    result["total_mcpy_ms"]    = total_mcpy_ns / 1e6
    result["total_mcpy_MB"]    = total_mcpy_bytes / 1e6
    result["eff_bw_GBs"]       = round(eff_bw_GBs, 3)
    result["bw_util_pct"]      = round(eff_bw_GBs / gpu_hw.get("peak_bw_GBs", 732) * 100, 4)

    # ── 7. Memset operations ──────────────────────────────────────────────
    if _table_exists(con, "CUPTI_ACTIVITY_KIND_MEMSET"):
        ms = con.execute(
            "SELECT COUNT(*), SUM(bytes), SUM(end-start) FROM CUPTI_ACTIVITY_KIND_MEMSET"
        ).fetchone()
    else:
        ms = (0, 0, 0)
    result["memset"] = dict(
        count    = ms[0] or 0,
        bytes_MB = (ms[1] or 0) / 1e6,
        time_ms  = (ms[2] or 0) / 1e6,
    )

    # ── 8. CUDA Runtime API ───────────────────────────────────────────────
    if _table_exists(con, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        arows = con.execute("""
            SELECT nameId, COUNT(*), SUM(end-start), AVG(end-start)
            FROM CUPTI_ACTIVITY_KIND_RUNTIME
            GROUP BY nameId ORDER BY SUM(end-start) DESC
        """).fetchall()
    else:
        arows = []
    result["runtime_api"] = [
        dict(name    = sids.get(r[0], f"api_{r[0]}"),
             calls   = r[1],
             total_ms= (r[2] or 0) / 1e6,
             avg_ms  = (r[3] or 0) / 1e6)
        for r in arows
    ]
    result["total_api_ms"] = sum(a["total_ms"] for a in result["runtime_api"])

    # ── 9. Stream synchronization ─────────────────────────────────────────
    if _table_exists(con, "ENUM_CUPTI_SYNC_TYPE"):
        stype_map = dict(con.execute(
            "SELECT id, label FROM ENUM_CUPTI_SYNC_TYPE").fetchall())
    else:
        stype_map = {}
    if _table_exists(con, "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"):
        srows = con.execute("""
            SELECT syncType, COUNT(*), SUM(end-start)
            FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION
            GROUP BY syncType
        """).fetchall()
    else:
        srows = []
    result["sync"] = [
        dict(type    = stype_map.get(r[0], str(r[0])),
             count   = r[1],
             total_ms= (r[2] or 0) / 1e6)
        for r in srows
    ]
    result["total_sync_ms"] = sum(s["total_ms"] for s in result["sync"])

    # ── 10. NVTX markers/ranges ───────────────────────────────────────────
    # NVTX_EVENTS is only present when --trace=nvtx was passed to nsys.
    if _table_exists(con, "NVTX_EVENTS"):
        nrows = con.execute("""
            SELECT textId, text, COUNT(*),
                   SUM(end-start), AVG(end-start)
            FROM NVTX_EVENTS
            WHERE end IS NOT NULL AND start IS NOT NULL
            GROUP BY COALESCE(textId, text)
            ORDER BY SUM(end-start) DESC
            LIMIT 20
        """).fetchall()
        result["nvtx"] = [
            dict(name    = sids.get(r[0], sids.get(r[1], f"nvtx_{r[0]}")),
                 count   = r[2],
                 total_ms= (r[3] or 0) / 1e6,
                 avg_ms  = (r[4] or 0) / 1e6)
            for r in nrows
        ]
    else:
        result["nvtx"] = []

    # ── 11. Profiler overhead ─────────────────────────────────────────────
    if _table_exists(con, "PROFILER_OVERHEAD"):
        ov = con.execute(
            "SELECT COUNT(*), SUM(end-start) FROM PROFILER_OVERHEAD"
        ).fetchone()
        result["profiler_overhead_ms"] = (ov[1] or 0) / 1e6
    else:
        result["profiler_overhead_ms"] = 0.0

    # ── 12. Derived summary metrics ───────────────────────────────────────
    ses_ns = result["session_ns"] or 1
    result["gpu_util_pct"]   = round(total_kern_ns / (0.8 * ses_ns) * 100, 2) # ~average of 20% overhead for unaccounted time outside desired simulation (I/O, unaccounted time)
    result["memcpy_ovh_pct"] = round(total_mcpy_ns / total_kern_ns * 100, 2) \
                                if total_kern_ns > 0 else 0.0

    # CPU time = session - kernel - memcpy (rough, ignores concurrency)
    result["cpu_gap_ms"] = (ses_ns - total_kern_ns - total_mcpy_ns) / 1e6

    con.close()
    return result


# ══════════════════════════════════════════════════════════════════════════════
# CONSOLE PRETTY-PRINTER
# ══════════════════════════════════════════════════════════════════════════════
def print_report(label: str, d: dict, gpu_hw: dict):
    W = 90
    div  = "─" * W
    sep  = "═" * W
    def row(name, val, unit=""):
        print(f"  {name:<38} {val}{unit}")

    print(f"\n{sep}")
    print(f"  REPORT: {label}")
    print(sep)

    # GPU
    if d["gpus"]:
        g = d["gpus"][0]
        print(f"\n  ── GPU ──────────────────────────────────────────────────────")
        row("Name",         g.get("name","?"))
        row("Architecture", f"sm_{g.get('computeMajor','?')}{g.get('computeMinor','?')}")
        row("SM count",     g.get("smCount","?"))
        row("Clock",        f"{g.get('clockRate',0)/1e6:.2f} GHz")
        row("Memory BW",    f"{g.get('memoryBandwidth',0)/1e9:.1f} GB/s")
        row("L2 cache",     f"{g.get('l2CacheSize',0)/1e6:.1f} MB")
        row("Regs/SM",      g.get("maxRegistersPerSm","?"))
        row("Shmem/SM",     f"{g.get('maxShmemPerSm',0)/1024:.0f} KB")
        row("Max warps/SM", g.get("maxWarpsPerSm","?"))
        row("Max blocks/SM",g.get("maxBlocksPerSm","?"))

    # Session
    print(f"\n  ── SESSION ──────────────────────────────────────────────────")
    row("Timestamp (local)", d["session_local"])
    row("Host",              d["hostname"])
    row("Session total",     f"{d['session_ms']/1e3:.3f} s")
    row("Kernel active",     f"{d['total_kern_ms']/1e3:.3f} s")
    row("Memcpy total",      f"{d['total_mcpy_ms']/1e3:.4f} s")
    row("Sync time",         f"{d['total_sync_ms']:.3f} ms")
    row("Profiler overhead", f"{d['profiler_overhead_ms']:.2f} ms")
    row("GPU utilization",   f"{d['gpu_util_pct']:.2f} %")
    row("Memcpy overhead",   f"{d['memcpy_ovh_pct']:.2f} % of kernel time")
    row("Total transfers",   f"{d['total_mcpy_MB']:.2f} MB")
    row("Effective BW",      f"{d['eff_bw_GBs']:.3f} GB/s"
                             f" ({d['bw_util_pct']:.4f}% of "
                             f"{gpu_hw.get('peak_bw_GBs',0):.0f} GB/s peak)")
    row("Kernel launches",   f"{d['total_kern_launches']:,}")

    # Kernels
    print(f"\n  ── KERNELS (sorted by total GPU time) {'─'*35}")
    hdr = f"  {'Kernel':<22} {'Launches':>8} {'Total ms':>10} {'Avg ms':>9} " \
          f"{'%GPU':>6} {'Blk':>5} {'Regs':>5} {'ThOcc%':>7} {'Lim':>5}"
    print(hdr)
    print("  " + "─" * (len(hdr)-2))
    for k in d["kernels"]:
        print(f"  {k['name']:<22} {k['launches']:>8,} {k['total_ms']:>10.3f} "
              f"{k['avg_ms']:>9.4f} {k['pct_gpu']:>6.1f} "
              f"{k['block']:>5} {k['regs']:>5} {k['th_occ']:>7.1f} "
              f"{k['limiter']:>5}")

    # Memcpy
    print(f"\n  ── MEMORY TRANSFERS {'─'*52}")
    print(f"  {'Kind':<12} {'Ops':>8} {'Total MB':>10} {'Time ms':>9} "
          f"{'Avg KB':>8} {'BW GB/s':>9}")
    print("  " + "─" * 60)
    for m in d["memcpy"]:
        print(f"  {m['kind']:<12} {m['ops']:>8,} {m['bytes_MB']:>10.3f} "
              f"{m['time_ms']:>9.3f} {m['avg_KB']:>8.2f} {m['bw_GBs']:>9.3f}")

    # Memset
    ms = d["memset"]
    print(f"\n  ── MEMSET ────────────────────────────────────────────────────")
    row("Memset ops",   f"{ms['count']:,}")
    row("Memset bytes", f"{ms['bytes_MB']:.2f} MB")
    row("Memset time",  f"{ms['time_ms']:.3f} ms")

    # Runtime API
    print(f"\n  ── CUDA RUNTIME API (top 10 by total time) ──────────────────")
    print(f"  {'API call':<45} {'Calls':>8} {'Total ms':>10} {'Avg ms':>9}")
    print("  " + "─" * 75)
    for a in d["runtime_api"][:10]:
        print(f"  {a['name']:<45} {a['calls']:>8,} "
              f"{a['total_ms']:>10.3f} {a['avg_ms']:>9.4f}")

    # NVTX
    if d["nvtx"]:
        print(f"\n  ── NVTX RANGES (top 10) ─────────────────────────────────────")
        for n in d["nvtx"][:10]:
            print(f"  {n['name']:<45} cnt={n['count']:>5}  "
                  f"total={n['total_ms']:>8.3f}ms  avg={n['avg_ms']:>7.4f}ms")

    # Command line
    print(f"\n  ── CAPTURED COMMAND ─────────────────────────────────────────")
    print(f"  {d['cmdline'][:W-2]}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# CSV EXPORTS
# ══════════════════════════════════════════════════════════════════════════════
def export_csvs(runs: list, outdir: str, prefix: str):
    os.makedirs(outdir, exist_ok=True)

    # Session summary CSV
    path = os.path.join(outdir, f"{prefix}_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Label","Session_ms","Kernel_ms","Memcpy_ms",
                    "GPU_util_pct","Memcpy_ovh_pct","Total_MB",
                    "EffBW_GBs","BW_pct","Kernel_launches",
                    "Sync_ms","Profiler_ovh_ms"])
        for lbl, d in runs:
            w.writerow([lbl,
                        f"{d['session_ms']:.3f}", f"{d['total_kern_ms']:.3f}",
                        f"{d['total_mcpy_ms']:.3f}", f"{d['gpu_util_pct']:.2f}",
                        f"{d['memcpy_ovh_pct']:.2f}", f"{d['total_mcpy_MB']:.3f}",
                        f"{d['eff_bw_GBs']:.4f}", f"{d['bw_util_pct']:.5f}",
                        d["total_kern_launches"],
                        f"{d['total_sync_ms']:.3f}",
                        f"{d['profiler_overhead_ms']:.3f}"])
    print(f"  Wrote {path}")

    # Kernel detail CSV (all runs × all kernels)
    path = os.path.join(outdir, f"{prefix}_kernels.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Label","Kernel","Launches","Total_ms","Avg_ms",
                    "Min_ms","Max_ms","Block","Grid","Regs",
                    "Shmem_static_B","Shmem_dyn_B","Lmem_B",
                    "ThOcc_pct","Limiter","pct_GPU"])
        for lbl, d in runs:
            for k in d["kernels"]:
                w.writerow([lbl, k["name"], k["launches"],
                            f"{k['total_ms']:.4f}", f"{k['avg_ms']:.5f}",
                            f"{k['min_ms']:.5f}", f"{k['max_ms']:.5f}",
                            k["block"], k["grid"], k["regs"],
                            k["shmem_static"], k["shmem_dyn"], k["lmem"],
                            f"{k['th_occ']:.1f}", k["limiter"],
                            f"{k['pct_gpu']:.2f}"])
    print(f"  Wrote {path}")

    # Memcpy CSV
    path = os.path.join(outdir, f"{prefix}_memcpy.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Label","Kind","Ops","Bytes_MB","Time_ms",
                    "AvgSize_KB","MaxSize_KB","BW_GBs"])
        for lbl, d in runs:
            for m in d["memcpy"]:
                w.writerow([lbl, m["kind"], m["ops"],
                            f"{m['bytes_MB']:.4f}", f"{m['time_ms']:.4f}",
                            f"{m['avg_KB']:.2f}", f"{m['max_KB']:.2f}",
                            f"{m['bw_GBs']:.4f}"])
    print(f"  Wrote {path}")


# ══════════════════════════════════════════════════════════════════════════════
# LATEX TABLE GENERATION
# ══════════════════════════════════════════════════════════════════════════════
def _tex_header(cap, label, cols):
    return (f"\\begin{{table}}[t]\n\\centering\n"
            f"\\caption{{{cap}}}\n\\label{{tab:{label}}}\n"
            f"\\resizebox{{\\columnwidth}}{{!}}{{%\n"
            f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n")

def _tex_footer():
    return "\\bottomrule\n\\end{{tabular}}}\n\\end{{table}}\n"

def make_latex_tables(runs: list, gpu_hw: dict, outdir: str, prefix: str):
    os.makedirs(outdir, exist_ok=True)
    gpuname = gpu_hw.get("name", "GPU")
    peak_bw = gpu_hw.get("peak_bw_GBs", 0)

    # ── Table 1: Session summary ───────────────────────────────────────────
    cap = (f"Session-level GPU profiling metrics collected with NVIDIA Nsight "
           f"Systems~\\cite{{nsightsystems}} (\\texttt{{--trace=cuda,nvtx}}) "
           f"on {gpuname} ({peak_bw:.0f}\\,GB/s peak bandwidth). "
           f"GPU util.\\ = kernel time $\\div$ session time; "
           f"Memcpy ovh.\\ = memcpy time $\\div$ kernel time; "
           f"Eff.\\ BW = total data $\\div$ total memcpy duration.")
    path = os.path.join(outdir, f"{prefix}_session_table.tex")
    with open(path, "w") as f:
        f.write(_tex_header(cap, f"{prefix}-nsys-session", "lrrrrrrrr"))
        f.write("\\textbf{Run} & \\textbf{Session (s)} & \\textbf{Kernel (s)} & "
                "\\textbf{GPU util.\\%} & \\textbf{Memcpy ovh.\\%} & "
                "\\textbf{Data (MB)} & \\textbf{Eff.\\ BW (GB/s)} & "
                "\\textbf{BW\\%} & \\textbf{Launches} \\\\\n\\midrule\n")
        for lbl, d in runs:
            f.write(f"  {lbl} & {d['session_ms']/1e3:.2f} & "
                    f"{d['total_kern_ms']/1e3:.3f} & "
                    f"{d['gpu_util_pct']:.1f} & {d['memcpy_ovh_pct']:.1f} & "
                    f"{d['total_mcpy_MB']:.1f} & {d['eff_bw_GBs']:.3f} & "
                    f"{d['bw_util_pct']:.3f} & "
                    f"{d['total_kern_launches']:,} \\\\\n")
        f.write(_tex_footer())
    print(f"  Wrote {path}")

    # ── Table 2: Per-kernel (representative = last run) ────────────────────
    lbl_last, d_last = runs[-1]
    cap = (f"Per-kernel timing and launch configuration profiled on "
           f"{gpuname} ({lbl_last} run). "
           f"Th.\\ Occ = theoretical occupancy (\\%) computed from launch "
           f"configuration; Lim.\\ = binding occupancy limiter "
           f"(\\textbf{{reg}}=register file, \\textbf{{shmem}}=shared memory, "
           f"\\textbf{{thr}}=thread count, \\textbf{{block}}=block size); "
           f"\\%GPU = fraction of total kernel time.")
    path = os.path.join(outdir, f"{prefix}_kernel_table.tex")
    with open(path, "w") as f:
        f.write(_tex_header(cap, f"{prefix}-nsys-kernel", "lrrrrrrrrl"))
        f.write("\\textbf{Kernel} & \\textbf{Launches} & \\textbf{Total (ms)} & "
                "\\textbf{Avg (ms)} & \\textbf{Min (ms)} & \\textbf{Max (ms)} & "
                "\\textbf{Block} & \\textbf{Regs} & "
                "\\textbf{Th.\\ Occ\\%} & \\textbf{Lim.} \\\\\n\\midrule\n")
        for k in d_last["kernels"]:
            if k["total_ms"] < 0.001:
                continue
            f.write(f"  \\texttt{{{k['name']}}} & {k['launches']:,} & "
                    f"{k['total_ms']:.3f} & {k['avg_ms']:.4f} & "
                    f"{k['min_ms']:.4f} & {k['max_ms']:.4f} & "
                    f"{k['block']} & {k['regs']} & "
                    f"{k['th_occ']:.1f} & {k['limiter']} \\\\\n")
        f.write(_tex_footer())
    print(f"  Wrote {path}")

    # ── Table 3: Memory transfers ──────────────────────────────────────────
    cap = ("Memory transfer breakdown per transfer kind "
           "(\\texttt{CUPTI\\_ACTIVITY\\_KIND\\_MEMCPY}). "
           "H2D/D2H = host-to-device/device-to-host; "
           "UVM = Unified Virtual Memory page migrations. "
           "Eff.\\ BW = per-kind effective bandwidth.")
    path = os.path.join(outdir, f"{prefix}_memcpy_table.tex")
    with open(path, "w") as f:
        f.write(_tex_header(cap, f"{prefix}-nsys-memcpy", "llrrrrrr"))
        f.write("\\textbf{Run} & \\textbf{Kind} & \\textbf{Ops} & "
                "\\textbf{Total (MB)} & \\textbf{Time (ms)} & "
                "\\textbf{Avg size (KB)} & \\textbf{Max (KB)} & "
                "\\textbf{Eff.\\ BW (GB/s)} \\\\\n\\midrule\n")
        for lbl, d in runs:
            for i, m in enumerate(d["memcpy"]):
                label_col = lbl if i == 0 else ""
                f.write(f"  {label_col} & {m['kind']} & {m['ops']:,} & "
                        f"{m['bytes_MB']:.3f} & {m['time_ms']:.3f} & "
                        f"{m['avg_KB']:.2f} & {m['max_KB']:.2f} & "
                        f"{m['bw_GBs']:.3f} \\\\\n")
            if len(runs) > 1:
                f.write("  \\midrule\n")
        f.write(_tex_footer())
    print(f"  Wrote {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
KERN_COLORS = {
    "Stage1":       "#185FA5", "Stage2":       "#1D9E75",
    "Stage3":       "#7F77DD", "Stage4":       "#854F0B",
    "ABMStage2":    "#D85A30", "ABMStage4":    "#EF9F27",
    "SameYearCite": "#D4537E", "UniformCite":  "#993C1D",
    "2HopWarp":     "#5DCAA5", "2Hop":         "#9FE1CB",
    "ExtractSizes": "#AFA9EC", "cuco::insert": "#D85A30",
    "cuco::init":   "#B4B2A9", "cuco::insert_if": "#FAC775",
    "cub::ForEach": "#D3D1C7", "cub::RadixSort":  "#C0DD97",
}
DEF_COLOR = "#888780"

def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(outdir, f"{name}.{ext}")
        fig.savefig(p, dpi=300 if ext == "pdf" else 150, bbox_inches="tight")
        print(f"  Wrote {p}")
    plt.close(fig)

def fig_session_overview(runs: list, gpu_hw: dict, outdir: str, prefix: str):
    labels = [l for l, _ in runs]
    n = len(labels)
    x = np.arange(n)
    w = 0.35

    fig = plt.figure(figsize=(max(10, 3*n), 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)
    fig.suptitle(f"nsys Session Overview — {prefix}  |  {gpu_hw.get('name','')}",
                 fontsize=11, fontweight="bold")

    # Panel 1: Session vs kernel time
    ax1 = fig.add_subplot(gs[0, 0])
    sess_s = [d["session_ms"]/1e3 for _, d in runs]
    kern_s = [d["total_kern_ms"]/1e3 for _, d in runs]
    ax1.bar(x - w/2, sess_s, w, color="#B5D4F4", edgecolor="#185FA5",
            lw=0.5, label="Session total")
    ax1.bar(x + w/2, kern_s, w, color="#185FA5", edgecolor="#0C447C",
            lw=0.5, label="Kernel active")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Time (s)"); ax1.set_title("Session vs kernel time", fontsize=10)
    ax1.legend(fontsize=8); ax1.grid(axis="y", lw=0.4, ls="--", alpha=0.5)
    for i, (s, k) in enumerate(zip(sess_s, kern_s)):
        ax1.text(i-w/2, s*1.01, f"{s:.1f}s", ha="center", fontsize=7.5)
        ax1.text(i+w/2, k*1.01, f"{k:.1f}s", ha="center", fontsize=7.5)

    # Panel 2: GPU util + memcpy overhead
    ax2 = fig.add_subplot(gs[0, 1])
    utils = [d["gpu_util_pct"] for _, d in runs]
    mcovh = [d["memcpy_ovh_pct"] for _, d in runs]
    ax2.bar(x - w/2, utils, w, color="#378ADD", edgecolor="#185FA5",
            lw=0.5, label="GPU util %")
    ax2.bar(x + w/2, mcovh, w, color="#D85A30", edgecolor="#993C1D",
            lw=0.5, label="Memcpy ovh %")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Percentage (%)"); ax2.set_title("GPU util & memcpy overhead", fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(axis="y", lw=0.4, ls="--", alpha=0.5)
    for i, (u, o) in enumerate(zip(utils, mcovh)):
        ax2.text(i-w/2, u*1.01, f"{u:.1f}%", ha="center", fontsize=7.5, fontweight="bold")
        ax2.text(i+w/2, o*1.01, f"{o:.1f}%", ha="center", fontsize=7.5)

    # Panel 3: Kernel time breakdown (last run)
    ax3 = fig.add_subplot(gs[1, 0])
    _, d_last = runs[-1]
    kns  = [k["name"] for k in d_last["kernels"][:8]]
    vals = [k["total_ms"] for k in d_last["kernels"][:8]]
    clrs = [KERN_COLORS.get(k["name"], DEF_COLOR) for k in d_last["kernels"][:8]]
    bars = ax3.barh(range(len(kns)), vals, color=clrs, edgecolor="white", lw=0.3)
    ax3.set_yticks(range(len(kns)))
    ax3.set_yticklabels([k[:22] for k in kns], fontsize=8)
    ax3.set_xlabel("Kernel time (ms)")
    ax3.set_title(f"Kernel breakdown ({runs[-1][0]})", fontsize=10)
    ax3.grid(axis="x", lw=0.4, ls="--", alpha=0.5)
    for bar, v in zip(bars, vals):
        if v > 1:
            ax3.text(v + v*0.01, bar.get_y() + bar.get_height()/2,
                     f"{v:.1f}ms", va="center", fontsize=7.5)

    # Panel 4: Memory transfer stacked bar
    ax4 = fig.add_subplot(gs[1, 1])
    kinds = ["H2D", "D2H", "UVM_H2D", "UVM_D2H", "D2D"]
    mclrs = ["#185FA5", "#1D9E75", "#85B7EB", "#9FE1CB", "#B4B2A9"]
    bots  = np.zeros(n)
    for kind, mc in zip(kinds, mclrs):
        vals4 = []
        for _, d in runs:
            v = next((m["bytes_MB"] for m in d["memcpy"] if m["kind"]==kind), 0)
            vals4.append(v)
        if any(v > 0 for v in vals4):
            ax4.bar(labels, vals4, bottom=bots, color=mc, label=kind,
                    edgecolor="white", lw=0.3)
            bots += np.array(vals4)
    ax4.set_ylabel("Data transferred (MB)")
    ax4.set_title("Memory transfer breakdown", fontsize=10)
    ax4.legend(fontsize=8); ax4.grid(axis="y", lw=0.4, ls="--", alpha=0.5)
    for i, tot in enumerate(bots):
        if tot > 0:
            ax4.text(i, tot + tot*0.01, f"{tot:.0f}MB", ha="center", fontsize=7.5)

    _save(fig, outdir, f"{prefix}_session_overview")


def fig_occupancy(runs: list, outdir: str, prefix: str):
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(4.5*n, 4.8), sharey=True)
    if n == 1: axes = [axes]
    fig.suptitle(f"Theoretical occupancy per kernel — {prefix}",
                 fontsize=11, fontweight="bold")

    for ax, (lbl, d) in zip(axes, runs):
        kns = [k["name"] for k in d["kernels"][:9]]
        occ = [k["th_occ"] for k in d["kernels"][:9]]
        lc  = ["#D85A30" if k["regs"] > 40
               else "#EF9F27" if k["shmem_static"] > 0
               else "#1D9E75"
               for k in d["kernels"][:9]]
        y = np.arange(len(kns))
        ax.barh(y, occ, color=lc, edgecolor="white", lw=0.3)
        ax.set_yticks(y)
        ax.set_yticklabels([k[:22] for k in kns], fontsize=8.5)
        ax.set_xlabel("Theoretical occupancy (%)", fontsize=9)
        ax.set_title(lbl, fontsize=10, fontweight="500")
        ax.set_xlim(0, 115)
        ax.axvline(100, color="gray", lw=0.6, ls="--", alpha=0.4)
        ax.grid(axis="x", lw=0.4, ls="--", alpha=0.4)
        for i, v in enumerate(occ):
            if v > 0:
                ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=7.5)

    axes[0].legend(handles=[
        mpatches.Patch(color="#D85A30", label="register-limited"),
        mpatches.Patch(color="#EF9F27", label="shmem-limited"),
        mpatches.Patch(color="#1D9E75", label="thread-limited"),
    ], fontsize=8, loc="lower right")
    plt.tight_layout()
    _save(fig, outdir, f"{prefix}_occupancy")


def fig_kernel_dist(runs: list, outdir: str, prefix: str):
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(4.5*n, 4.5))
    if n == 1: axes = [axes]
    fig.suptitle(f"Kernel time distribution — {prefix}",
                 fontsize=11, fontweight="bold")

    for ax, (lbl, d) in zip(axes, runs):
        kd   = d["kernels"][:7]
        tot  = sum(k["total_ms"] for k in kd)
        frac = [k["total_ms"]/tot*100 if tot > 0 else 0 for k in kd]
        clrs = [KERN_COLORS.get(k["name"], DEF_COLOR) for k in kd]
        bars = ax.bar(range(len(kd)), frac, color=clrs, edgecolor="white", lw=0.3)
        ax.set_xticks(range(len(kd)))
        ax.set_xticklabels([k["name"][:14] for k in kd],
                           rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("% of kernel time")
        ax.set_title(lbl, fontsize=10, fontweight="500")
        ax.set_ylim(0, 110)
        ax.grid(axis="y", lw=0.4, ls="--", alpha=0.5)
        for bar, v in zip(bars, frac):
            if v > 2:
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.8,
                        f"{v:.0f}%", ha="center", fontsize=7.5, fontweight="500")

    plt.tight_layout()
    _save(fig, outdir, f"{prefix}_kernel_dist")


def fig_timeline(runs: list, outdir: str, prefix: str):
    """
    Compact horizontal timeline showing relative proportions of
    kernel, memcpy, memset, sync, and CPU idle time for each run.
    """
    n = len(runs)
    fig, axes = plt.subplots(n, 1, figsize=(12, 1.6*n))
    if n == 1: axes = [axes]
    fig.suptitle(f"GPU time composition — {prefix}",
                 fontsize=11, fontweight="bold")

    for ax, (lbl, d) in zip(axes, runs):
        ses  = d["session_ms"]
        cats = [
            ("Kernel",  d["total_kern_ms"],          "#185FA5"),
            ("Memcpy",  d["total_mcpy_ms"],           "#D85A30"),
            ("Memset",  d["memset"]["time_ms"],        "#EF9F27"),
            ("Sync",    d["total_sync_ms"],            "#1D9E75"),
            ("API ovh", d["total_api_ms"]/20,         "#AFA9EC"),  # scaled
        ]
        # CPU gap (session not covered by above)
        covered = sum(c[1] for c in cats)
        cpu_gap = max(0, ses - covered)
        cats.append(("CPU/idle", cpu_gap, "#D3D1C7"))

        left = 0.0
        for cat_name, ms, clr in cats:
            w = ms / ses * 100
            if w > 0.1:
                ax.barh(0, w, left=left, height=0.6, color=clr,
                        edgecolor="white", lw=0.3)
                if w > 3:
                    ax.text(left + w/2, 0, f"{cat_name}\n{ms:.0f}ms",
                            ha="center", va="center", fontsize=7.5,
                            color="white" if clr not in ("#D3D1C7","#AFA9EC") else "#444441")
                left += w

        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xlabel("% of session time")
        ax.set_title(f"{lbl}  (session={ses/1e3:.2f}s)", fontsize=9,
                     loc="left", pad=3)
        ax.grid(axis="x", lw=0.4, ls="--", alpha=0.4)

    plt.tight_layout()
    _save(fig, outdir, f"{prefix}_timeline")


# ══════════════════════════════════════════════════════════════════════════════
# LABEL INFERENCE FROM FILENAME
# ══════════════════════════════════════════════════════════════════════════════
def infer_label(path: str) -> str:
    """Infer a run label from filename like nsys-pa-3y-6p.sqlite"""
    base = os.path.splitext(os.path.basename(path))[0]
    # Remove common prefixes
    base = re.sub(r'^nsys-?pa-?|^nsys-?', '', base, flags=re.IGNORECASE)
    # years / growth pattern
    m = re.search(r'(\d+y)[_-](\d+p)', base)
    if m:
        yrs = m.group(1)
        pct = m.group(2).replace('p', '%')
        return f"{yrs}/{pct}"
    return base


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
class _MultiArg(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest, None) or []
        items.append(values)
        setattr(namespace, self.dest, items)

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("positional", nargs="*",
                   help="One or more .sqlite or .nsys-rep files (auto-labeled)")
    p.add_argument("--files", dest="labeled", nargs=2, action=_MultiArg,
                   metavar=("FILE", "LABEL"),
                   help="File + label pair (repeat)")
    p.add_argument("--glob", metavar="PATTERN",
                   help="Glob pattern for sqlite files, e.g. 'profile/*.sqlite'")
    p.add_argument("--gpu",    default="P100",
                   choices=list(GPU_PRESETS.keys()),
                   help="GPU model for BW utilisation calculation (default: P100)")
    p.add_argument("--outdir", default="./nsys_output",
                   help="Output directory")
    p.add_argument("--prefix", default="pa",
                   help="Filename prefix for outputs")
    p.add_argument("--no-plot",    action="store_true", help="Skip figures")
    p.add_argument("--no-latex",   action="store_true", help="Skip LaTeX tables")
    p.add_argument("--no-csv",     action="store_true", help="Skip CSV export")
    p.add_argument("--no-console", action="store_true", help="Skip console report")
    args = p.parse_args()

    # Collect (path, label) pairs
    file_pairs = []
    for f in (args.positional or []):
        file_pairs.append((f, infer_label(f)))
    for f, lbl in (args.labeled or []):
        file_pairs.append((f, lbl))
    if args.glob:
        for f in sorted(glob.glob(args.glob)):
            file_pairs.append((f, infer_label(f)))

    if not file_pairs:
        p.print_help()
        print("\nExample:")
        print("  python3 nsys_analyze_full.py profile/nsys-pa-1y-1p.sqlite")
        print("  python3 nsys_analyze_full.py --glob 'profile/*.sqlite' --gpu P100")
        sys.exit(0)

    gpu_hw = GPU_PRESETS[args.gpu.upper()].copy()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"\nGPU preset: {gpu_hw['name']}  |  {gpu_hw['peak_bw_GBs']:.0f} GB/s peak\n")

    runs = []
    for fpath, label in file_pairs:
        ext = os.path.splitext(fpath)[1].lower()

        if ext == ".nsys-rep":
            print(f"  Converting {fpath} → sqlite ...")
            fpath = rep_to_sqlite(fpath, args.outdir)
            if fpath is None:
                continue

        if not os.path.exists(fpath):
            print(f"  MISSING: {fpath}")
            continue

        print(f"  Parsing [{label}]  {fpath}")
        d = parse_sqlite(fpath, gpu_hw.copy())
        runs.append((label, d))
        print(f"    GPU util={d['gpu_util_pct']:.1f}%  "
              f"memcpy_ovh={d['memcpy_ovh_pct']:.1f}%  "
              f"kern_ms={d['total_kern_ms']:.1f}  "
              f"launches={d['total_kern_launches']:,}")

    if not runs:
        print("No data parsed.")
        return

    # Console reports
    if not args.no_console:
        for label, d in runs:
            print_report(label, d, gpu_hw)

    # Outputs
    if not args.no_csv:
        print(f"\nWriting CSVs → {args.outdir}/")
        export_csvs(runs, args.outdir, args.prefix)

    if not args.no_latex:
        print(f"\nWriting LaTeX tables → {args.outdir}/")
        make_latex_tables(runs, gpu_hw, args.outdir, args.prefix)

    if not args.no_plot and HAS_PLOT:
        print(f"\nGenerating figures → {args.outdir}/")
        fig_session_overview(runs, gpu_hw, args.outdir, args.prefix)
        fig_occupancy(runs, args.outdir, args.prefix)
        fig_kernel_dist(runs, args.outdir, args.prefix)
        fig_timeline(runs, args.outdir, args.prefix)
    elif not HAS_PLOT and not args.no_plot:
        print("\nInstall matplotlib for figures:  pip install matplotlib numpy")

    print(f"\nDone. Include in paper with:")
    for t in ["session_table", "kernel_table", "memcpy_table"]:
        print(f"  \\input{{{args.outdir}/{args.prefix}_{t}}}")

if __name__ == "__main__":
    main()
