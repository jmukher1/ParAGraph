#!/usr/bin/env python3
"""
nsys_export_csv.py  —  STEP 1 of 2
=====================================
Reads Nsight Systems .sqlite (or .nsys-rep) files and exports exactly
three clean CSVs consumed by nsys_plot_figures.py (Step 2).

OUTPUTS
-------
  {outdir}/{prefix}_occupancy.csv   per-kernel theoretical occupancy per run
  {outdir}/{prefix}_gpu_util.csv    GPU util% + memcpy overhead% per run
  {outdir}/{prefix}_memcpy.csv      memory transfer bytes-per-kind per run

USAGE
-----
  # Auto-discover by glob (labels inferred from filenames):
  python3 nsys_export_csv.py --glob "profile/nsys-pa-30y-*.sqlite" --gpu H100

  # Explicit file + label pairs (--files repeatable):
  python3 nsys_export_csv.py \\
      --files profile/nsys-pa-30y-1p.sqlite "30y/1%" \\
      --files profile/nsys-pa-30y-3p.sqlite "30y/3%" \\
      --files profile/nsys-pa-30y-5p.sqlite "30y/5%" \\
      --files profile/nsys-pa-30y-6p.sqlite "30y/6%" \\
      --gpu H100 --outdir ./csv_out --prefix pa

REQUIREMENTS
------------
  Python 3.8+ stdlib only (no pip packages needed)
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
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# GPU HARDWARE PRESETS
# ══════════════════════════════════════════════════════════════════════════════
GPU_PRESETS = {
    "P100": dict(name="Tesla P100-PCIE-16GB",
                 peak_bw_GBs=732.0, peak_fp32_GFLOPs=9300.0, sm_count=56,
                 warp_size=32, max_warps_sm=64, max_blocks_sm=32,
                 max_thr_sm=2048, regs_per_sm=65536, shmem_per_sm=65536),
    "A30":  dict(name="NVIDIA A30",
                 peak_bw_GBs=933.0, peak_fp32_GFLOPs=10300.0, sm_count=56,
                 warp_size=32, max_warps_sm=64, max_blocks_sm=32,
                 max_thr_sm=2048, regs_per_sm=65536, shmem_per_sm=98304),
    "A100": dict(name="NVIDIA A100 SXM4 80GB",
                 peak_bw_GBs=2039.0, peak_fp32_GFLOPs=19500.0, sm_count=108,
                 warp_size=32, max_warps_sm=64, max_blocks_sm=32,
                 max_thr_sm=2048, regs_per_sm=65536, shmem_per_sm=167936),
    "H100": dict(name="NVIDIA H100 80GB HBM3",
                 peak_bw_GBs=3350.0, peak_fp32_GFLOPs=67000.0, sm_count=132,
                 warp_size=32, max_warps_sm=64, max_blocks_sm=32,
                 max_thr_sm=2048, regs_per_sm=65536, shmem_per_sm=232448),
    "V100": dict(name="Tesla V100-SXM2-32GB",
                 peak_bw_GBs=900.0, peak_fp32_GFLOPs=14000.0, sm_count=80,
                 warp_size=32, max_warps_sm=64, max_blocks_sm=32,
                 max_thr_sm=2048, regs_per_sm=65536, shmem_per_sm=98304),
}

MEMCPY_KIND = {
    0: "Unknown", 1: "H2D",      2: "D2H",      3: "H2A",
    4: "A2H",     5: "A2A",      6: "A2D",       7: "D2A",
    8: "D2D",     9: "H2H",     10: "P2P",      11: "UVM_H2D",
    12: "UVM_D2H", 13: "UVM_D2D",
}

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


def normalize_kernel(raw: str) -> str:
    for pat, short in _KERN_RULES:
        if re.search(pat, raw):
            return short
    clean = raw.split("(")[0].split("<")[0]
    parts = [p for p in clean.split("::") if p]
    return parts[-1][:30] if parts else raw[:30]


# ══════════════════════════════════════════════════════════════════════════════
# THEORETICAL OCCUPANCY  (exact formula from nsys_analyze_full.py)
# ══════════════════════════════════════════════════════════════════════════════
def theoretical_occupancy(regs: int, block: int,
                           shmem_static: int, shmem_dynamic: int,
                           hw: dict) -> float:
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

    lim_thread = max_thr_sm // block

    if regs > 0:
        regs_per_warp  = math.ceil(regs * warp_size / 256) * 256
        regs_per_block = regs_per_warp * warps_per_block
        lim_reg = regs_per_sm // regs_per_block if regs_per_block > 0 else max_blocks_sm
    else:
        lim_reg = max_blocks_sm

    if shmem_total > 0:
        shmem_alloc = math.ceil(shmem_total / 256) * 256
        lim_shmem   = shmem_per_sm // shmem_alloc if shmem_alloc > 0 else max_blocks_sm
    else:
        lim_shmem = max_blocks_sm

    max_blocks   = min(lim_thread, lim_reg, lim_shmem, max_blocks_sm)
    active_warps = max_blocks * warps_per_block
    return round(min(active_warps / max_warps_sm, 1.0) * 100.0, 1)


def occupancy_limiter(regs: int, shmem: int, block: int) -> str:
    if regs > 40:  return "reg"
    if shmem > 0:  return "shmem"
    if block < 64: return "block"
    return "thr"


# ══════════════════════════════════════════════════════════════════════════════
# SQLITE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _table_exists(con: sqlite3.Connection, table: str) -> bool:
    r = con.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
        (table,)
    ).fetchone()
    return bool(r and r[0])


def _hw_from_db(con: sqlite3.Connection, hw: dict) -> dict:
    hw = hw.copy()
    if not _table_exists(con, "TARGET_INFO_GPU"):
        return hw
    cols = [c[1] for c in
            con.execute("PRAGMA table_info(TARGET_INFO_GPU)").fetchall()]
    row = con.execute("SELECT * FROM TARGET_INFO_GPU LIMIT 1").fetchone()
    if not row:
        return hw
    g = dict(zip(cols, row))
    if g.get("memoryBandwidth"):   hw["peak_bw_GBs"]  = g["memoryBandwidth"] / 1e9
    if g.get("maxWarpsPerSm"):     hw["max_warps_sm"]  = g["maxWarpsPerSm"]
    if g.get("maxBlocksPerSm"):    hw["max_blocks_sm"] = g["maxBlocksPerSm"]
    if g.get("maxShmemPerSm"):     hw["shmem_per_sm"]  = g["maxShmemPerSm"]
    if g.get("maxRegistersPerSm"): hw["regs_per_sm"]   = g["maxRegistersPerSm"]
    if g.get("threadsPerWarp"):    hw["warp_size"]     = g["threadsPerWarp"]
    if g.get("maxWarpsPerSm") and g.get("threadsPerWarp"):
        hw["max_thr_sm"] = g["maxWarpsPerSm"] * g["threadsPerWarp"]
    return hw


# ══════════════════════════════════════════════════════════════════════════════
# CORE PARSER
# ══════════════════════════════════════════════════════════════════════════════
def parse_sqlite(db_path: str, gpu_hw: dict) -> dict:
    con = sqlite3.connect(db_path)
    hw  = _hw_from_db(con, gpu_hw)
    sids = dict(con.execute("SELECT id, value FROM StringIds").fetchall())

    # Session duration
    ses = con.execute("SELECT duration FROM ANALYSIS_DETAILS").fetchone()
    session_ns = ses[0] if ses else 1

    # Kernels
    krows = con.execute("""
        SELECT demangledName,
               COUNT(*)                   AS launches,
               SUM(end - start)           AS total_ns,
               AVG(end - start)           AS avg_ns,
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

    total_kern_ns = sum(r[2] for r in krows if r[2]) or 1
    kernels = []
    for r in krows:
        raw   = sids.get(r[0], str(r[0]))
        name  = normalize_kernel(raw)
        regs  = int(r[4] or 0)
        block = int(r[5] or 0)
        grid  = int(r[6] or 0)
        ss    = int(r[7] or 0)
        sd    = int(r[8] or 0)
        lmem  = int(r[9] or 0)
        dur   = r[2] or 0
        occ   = theoretical_occupancy(regs, block, ss, sd, hw)
        lim   = occupancy_limiter(regs, ss + sd, block)
        kernels.append(dict(
            name=name, launches=r[1],
            total_ms=dur / 1e6, avg_ms=(r[3] or 0) / 1e6,
            regs=regs, block=block, grid=grid,
            shmem_static=ss, shmem_dyn=sd, lmem=lmem,
            th_occ=occ, limiter=lim,
            pct_gpu=round(dur / total_kern_ns * 100, 2),
        ))

    # Memcpy
    memcpy_list   = []
    total_mcpy_ns = 0
    if _table_exists(con, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        for r in con.execute("""
            SELECT copyKind, COUNT(*), SUM(bytes), SUM(end-start),
                   AVG(bytes), MAX(bytes)
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            GROUP BY copyKind ORDER BY SUM(bytes) DESC
        """).fetchall():
            ns  = r[3] or 0
            byt = r[2] or 0
            total_mcpy_ns += ns
            memcpy_list.append(dict(
                kind=MEMCPY_KIND.get(r[0], f"kind{r[0]}"),
                ops=r[1], bytes_MB=byt / 1e6, time_ms=ns / 1e6,
                avg_KB=(r[4] or 0) / 1e3, max_KB=(r[5] or 0) / 1e3,
                bw_GBs=byt / (ns * 1e-9) / 1e9 if ns > 0 else 0.0,
            ))

    gpu_util_pct   = round(total_kern_ns / session_ns * 100, 2)
    memcpy_ovh_pct = round(total_mcpy_ns / total_kern_ns * 100, 2) \
                     if total_kern_ns > 0 else 0.0

    con.close()
    return dict(
        kernels=kernels,
        gpu_util_pct=gpu_util_pct,
        memcpy_ovh_pct=memcpy_ovh_pct,
        memcpy=memcpy_list,
    )


# ══════════════════════════════════════════════════════════════════════════════
# .nsys-rep → .sqlite
# ══════════════════════════════════════════════════════════════════════════════
def rep_to_sqlite(rep_path: str, outdir: str) -> Optional[str]:
    try:
        subprocess.run(["nsys", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("  WARNING: nsys not in PATH — cannot convert .nsys-rep")
        return None
    base   = os.path.splitext(os.path.basename(rep_path))[0]
    out_db = os.path.join(outdir, f"{base}.sqlite")
    os.makedirs(outdir, exist_ok=True)
    result = subprocess.run(
        ["nsys", "export", "--type", "sqlite", "--output", out_db, rep_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  nsys export failed: {result.stderr[:300]}")
        return None
    return out_db if os.path.exists(out_db) else None


def infer_label(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    base = re.sub(r'^nsys-?pa-?|^nsys-?', '', base, flags=re.IGNORECASE)
    m = re.search(r'(\d+y)[_-](\d+p)', base)
    if m:
        return f"{m.group(1)}/{m.group(2).replace('p','%')}"
    return base


# ══════════════════════════════════════════════════════════════════════════════
# CSV WRITERS  — one function per output file
# ══════════════════════════════════════════════════════════════════════════════
def write_occupancy_csv(runs: list, outdir: str, prefix: str) -> str:
    path = os.path.join(outdir, f"{prefix}_occupancy.csv")
    os.makedirs(outdir, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "Label", "Kernel", "Launches",
            "Total_ms", "Avg_ms", "Block", "Grid", "Regs",
            "Shmem_static_B", "Shmem_dyn_B", "Lmem_B",
            "ThOcc_pct", "Limiter", "pct_GPU",
        ])
        for label, d in runs:
            for k in d["kernels"]:
                w.writerow([
                    label, k["name"], k["launches"],
                    f"{k['total_ms']:.4f}", f"{k['avg_ms']:.5f}",
                    k["block"], k["grid"], k["regs"],
                    k["shmem_static"], k["shmem_dyn"], k["lmem"],
                    f"{k['th_occ']:.1f}", k["limiter"],
                    f"{k['pct_gpu']:.2f}",
                ])
    print(f"  Wrote {path}")
    return path


def write_gpu_util_csv(runs: list, outdir: str, prefix: str) -> str:
    path = os.path.join(outdir, f"{prefix}_gpu_util.csv")
    os.makedirs(outdir, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Label", "GPU_util_pct", "Memcpy_ovh_pct"])
        for label, d in runs:
            w.writerow([
                label,
                f"{d['gpu_util_pct']:.2f}",
                f"{d['memcpy_ovh_pct']:.2f}",
            ])
    print(f"  Wrote {path}")
    return path


def write_memcpy_csv(runs: list, outdir: str, prefix: str) -> str:
    path = os.path.join(outdir, f"{prefix}_memcpy.csv")
    os.makedirs(outdir, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "Label", "Kind", "Ops", "Bytes_MB",
            "Time_ms", "AvgSize_KB", "MaxSize_KB", "BW_GBs",
        ])
        for label, d in runs:
            for m in d["memcpy"]:
                w.writerow([
                    label, m["kind"], m["ops"],
                    f"{m['bytes_MB']:.4f}", f"{m['time_ms']:.4f}",
                    f"{m['avg_KB']:.2f}", f"{m['max_KB']:.2f}",
                    f"{m['bw_GBs']:.4f}",
                ])
    print(f"  Wrote {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
class _PairAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest, None) or []
        items.append(tuple(values))
        setattr(namespace, self.dest, items)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("positional", nargs="*",
                   help=".sqlite / .nsys-rep files (label inferred from name)")
    p.add_argument("--files", dest="labeled", nargs=2, action=_PairAction,
                   metavar=("FILE", "LABEL"),
                   help="FILE LABEL pair — repeat per run")
    p.add_argument("--glob",   metavar="PATTERN",
                   help="Shell glob for input files")
    p.add_argument("--gpu",    default="P100",
                   choices=list(GPU_PRESETS.keys()),
                   help="GPU preset for occupancy math (default: P100)")
    p.add_argument("--outdir", default="./nsys_csv",
                   help="Output directory (default: ./nsys_csv)")
    p.add_argument("--prefix", default="pa",
                   help="CSV filename prefix (default: pa)")
    args = p.parse_args()

    file_pairs: list = []
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
        print("  python3 nsys_export_csv.py \\")
        print("      --files profile/nsys-pa-30y-1p.sqlite '30y/1%' \\")
        print("      --files profile/nsys-pa-30y-3p.sqlite '30y/3%' \\")
        print("      --gpu H100 --outdir ./csv --prefix pa")
        sys.exit(0)

    gpu_hw = GPU_PRESETS[args.gpu.upper()].copy()
    print(f"\nGPU : {gpu_hw['name']}  |  {gpu_hw['peak_bw_GBs']:.0f} GB/s peak")
    print(f"Out : {args.outdir}/")

    runs = []
    for fpath, label in file_pairs:
        ext = os.path.splitext(fpath)[1].lower()
        if ext == ".nsys-rep":
            print(f"  Converting {fpath} …")
            fpath = rep_to_sqlite(fpath, args.outdir)
            if fpath is None:
                continue
        if not os.path.exists(fpath):
            print(f"  MISSING: {fpath}")
            continue
        print(f"  Parsing [{label}]  {fpath}")
        d = parse_sqlite(fpath, gpu_hw)
        runs.append((label, d))
        print(f"    gpu_util={d['gpu_util_pct']:.1f}%  "
              f"memcpy_ovh={d['memcpy_ovh_pct']:.1f}%  "
              f"kernels={len(d['kernels'])}")

    if not runs:
        print("No data parsed.")
        sys.exit(1)

    print(f"\nWriting CSVs:")
    occ  = write_occupancy_csv(runs, args.outdir, args.prefix)
    util = write_gpu_util_csv(runs, args.outdir, args.prefix)
    mem  = write_memcpy_csv(runs, args.outdir, args.prefix)

    print(f"\nStep 2:")
    print(f"  python3 nsys_plot_figures.py \\")
    print(f"      --occupancy {occ} \\")
    print(f"      --gpu-util  {util} \\")
    print(f"      --memcpy    {mem} \\")
    print(f"      --outdir ./nsys_figures --prefix {args.prefix}")


if __name__ == "__main__":
    main()
