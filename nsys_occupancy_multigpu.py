#!/usr/bin/env python3
"""
nsys_occupancy_multigpu.py
==========================
Plot theoretical kernel occupancy for 30-year simulation runs across
multiple GPUs (P100, H100, A30) and growth rates (1%, 3%, 5%, 6%).

USAGE
-----
# Provide sqlite files grouped by GPU, one --gpu block per GPU:
    python3 nsys_occupancy_multigpu.py \
        --gpu P100 \
            profile/nsys-pa-30y-1p-p100.sqlite  "1%" \
            profile/nsys-pa-30y-3p-p100.sqlite  "3%" \
            profile/nsys-pa-30y-5p-p100.sqlite  "5%" \
            profile/nsys-pa-30y-6p-p100.sqlite  "6%" \
        --gpu H100 \
            profile/nsys-pa-30y-1p-h100.sqlite  "1%" \
            profile/nsys-pa-30y-3p-h100.sqlite  "3%" \
            profile/nsys-pa-30y-5p-h100.sqlite  "5%" \
            profile/nsys-pa-30y-6p-h100.sqlite  "6%" \
        --gpu A30 \
            profile/nsys-pa-30y-1p-a30.sqlite   "1%" \
            profile/nsys-pa-30y-3p-a30.sqlite   "3%" \
            profile/nsys-pa-30y-5p-a30.sqlite   "5%" \
            profile/nsys-pa-30y-6p-a30.sqlite   "6%" \
        --outdir ./nsys_output \
        --prefix pa_30y

# Glob-based (auto-labels from filenames like nsys-pa-30y-6p-h100.sqlite):
    python3 nsys_occupancy_multigpu.py \
        --gpu P100 --glob-gpu "profile/*-p100.sqlite" \
        --gpu H100 --glob-gpu "profile/*-h100.sqlite" \
        --gpu A30  --glob-gpu "profile/*-a30.sqlite"

LAYOUT
------
  Rows    = GPUs   (P100 / H100 / A30)
  Columns = growth rates (1% / 3% / 5% / 6%)

Each cell is a horizontal bar chart of per-kernel theoretical occupancy.
Bar color encodes the occupancy limiter:
  • red   = register-limited  (regs > 40)
  • amber = shared-memory-limited
  • green = thread/block-limited

REQUIREMENTS
------------
    pip install matplotlib numpy
"""

import argparse
import glob as _glob
import math
import os
import re
import sqlite3
import sys
from collections import defaultdict
from typing import Optional

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    print("ERROR: matplotlib / numpy required.  pip install matplotlib numpy")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# GPU HARDWARE PRESETS  (used to seed hw limits before sqlite overrides them)
# ══════════════════════════════════════════════════════════════════════════════
GPU_PRESETS = {
    "P100": dict(name="Tesla P100-PCIE-16GB",  peak_bw_GBs=732.0,
                 peak_fp32_GFLOPs=9300.0,  sm_count=56,
                 max_warps_sm=64, max_blocks_sm=32, max_thr_sm=2048,
                 regs_per_sm=65536, shmem_per_sm=49152, warp_size=32),
    "A30":  dict(name="NVIDIA A30",            peak_bw_GBs=933.0,
                 peak_fp32_GFLOPs=10300.0, sm_count=56,
                 max_warps_sm=64, max_blocks_sm=32, max_thr_sm=2048,
                 regs_per_sm=65536, shmem_per_sm=102400, warp_size=32),
    "A100": dict(name="NVIDIA A100 SXM4 80GB", peak_bw_GBs=2039.0,
                 peak_fp32_GFLOPs=19500.0, sm_count=108,
                 max_warps_sm=64, max_blocks_sm=32, max_thr_sm=2048,
                 regs_per_sm=65536, shmem_per_sm=167936, warp_size=32),
    "H100": dict(name="NVIDIA H100 80GB HBM3", peak_bw_GBs=3350.0,
                 peak_fp32_GFLOPs=67000.0, sm_count=132,
                 max_warps_sm=64, max_blocks_sm=32, max_thr_sm=2048,
                 regs_per_sm=65536, shmem_per_sm=233472, warp_size=32),
    "V100": dict(name="Tesla V100-SXM2-32GB",  peak_bw_GBs=900.0,
                 peak_fp32_GFLOPs=14000.0, sm_count=80,
                 max_warps_sm=64, max_blocks_sm=32, max_thr_sm=2048,
                 regs_per_sm=65536, shmem_per_sm=98304, warp_size=32),
}

# ══════════════════════════════════════════════════════════════════════════════
# KERNEL NAME NORMALIZER  (same rules as nsys_analyze_full.py)
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

def normalize_kernel(raw: str) -> str:
    for pat, short in _KERN_RULES:
        if re.search(pat, raw):
            return short
    clean = raw.split("(")[0].split("<")[0]
    parts = [p for p in clean.split("::") if p]
    return parts[-1][:28] if parts else raw[:28]


# ══════════════════════════════════════════════════════════════════════════════
# THEORETICAL OCCUPANCY CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════
def theoretical_occupancy(regs: int, block: int, shmem_static: int,
                           shmem_dynamic: int, hw: dict) -> float:
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
        lim_shmem = shmem_per_sm // shmem_alloc if shmem_alloc > 0 else max_blocks_sm
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

def _table_exists(con, table):
    row = con.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
        (table,)
    ).fetchone()
    return bool(row and row[0])


# ══════════════════════════════════════════════════════════════════════════════
# SQLITE KERNEL PARSER
# ══════════════════════════════════════════════════════════════════════════════
def parse_kernels(db_path: str, hw: dict) -> list:
    """
    Returns a list of kernel dicts with occupancy computed for `hw`.
    Fields: name, launches, total_ms, regs, block, shmem_static, shmem_dyn,
            th_occ, limiter, pct_gpu
    """
    con = sqlite3.connect(db_path)
    sids = dict(con.execute("SELECT id, value FROM StringIds").fetchall())

    # Read GPU hw limits from the sqlite (override preset where available)
    hw = hw.copy()
    if _table_exists(con, "TARGET_INFO_GPU"):
        gpu_cols = [c[1] for c in con.execute(
            "PRAGMA table_info(TARGET_INFO_GPU)").fetchall()]
        gpu_rows = con.execute("SELECT * FROM TARGET_INFO_GPU").fetchall()
        for row in gpu_rows:
            g = dict(zip(gpu_cols, row))
            if g.get("memoryBandwidth"):
                hw["peak_bw_GBs"]   = g["memoryBandwidth"] / 1e9
            if g.get("maxWarpsPerSm"):
                hw["max_warps_sm"]  = g["maxWarpsPerSm"]
            if g.get("maxBlocksPerSm"):
                hw["max_blocks_sm"] = g["maxBlocksPerSm"]
            if g.get("maxShmemPerSm"):
                hw["shmem_per_sm"]  = g["maxShmemPerSm"]
            if g.get("maxRegistersPerSm"):
                hw["regs_per_sm"]   = g["maxRegistersPerSm"]
            if g.get("threadsPerWarp"):
                hw["warp_size"]     = g["threadsPerWarp"]
            hw["max_thr_sm"] = (g.get("maxWarpsPerSm", 64) *
                                g.get("threadsPerWarp", 32))

    krows = con.execute("""
        SELECT demangledName,
               COUNT(*)           AS launches,
               SUM(end - start)   AS total_ns,
               AVG(registersPerThread)   AS regs,
               AVG(blockX)              AS block,
               AVG(staticSharedMemory)  AS shmem_static,
               AVG(dynamicSharedMemory) AS shmem_dyn
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        GROUP BY demangledName
        ORDER BY total_ns DESC
    """).fetchall()
    con.close()

    total_ns = sum(r[2] for r in krows if r[2]) or 1

    kernels = []
    for r in krows:
        raw_name = sids.get(r[0], str(r[0]))
        name     = normalize_kernel(raw_name)
        regs     = int(r[3] or 0)
        block    = int(r[4] or 0)
        ss       = int(r[5] or 0)
        sd       = int(r[6] or 0)
        dur_ns   = r[2] or 0

        occ = theoretical_occupancy(regs, block, ss, sd, hw)
        lim = occupancy_limiter(regs, ss + sd, block)

        kernels.append(dict(
            name        = name,
            launches    = r[1],
            total_ms    = dur_ns / 1e6,
            regs        = regs,
            block       = block,
            shmem_static= ss,
            shmem_dyn   = sd,
            th_occ      = occ,
            limiter     = lim,
            pct_gpu     = round(dur_ns / total_ns * 100, 2),
        ))
    return kernels


# ══════════════════════════════════════════════════════════════════════════════
# LABEL INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
def infer_label(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    base = re.sub(r'^nsys-?pa-?|^nsys-?', '', base, flags=re.IGNORECASE)
    # match e.g. "30y-6p", "30y-6p-h100"
    m = re.search(r'(\d+)y[_-](\d+)p', base)
    if m:
        return f"{m.group(2)}%"
    # match bare "6p"
    m = re.search(r'(\d+)p', base)
    if m:
        return f"{m.group(1)}%"
    return base

GROWTH_ORDER = ["1%", "3%", "5%", "6%"]

def _sort_key(label: str) -> int:
    try:
        return GROWTH_ORDER.index(label)
    except ValueError:
        return 99


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE: OCCUPANCY  (rows=GPUs, cols=growth rates)
# ══════════════════════════════════════════════════════════════════════════════
# Color scheme: limiter → bar color
LIM_COLOR = {
    "reg":   "#D85A30",   # red-orange  – register limited
    "shmem": "#EF9F27",   # amber       – shared-memory limited
    "block": "#7F77DD",   # purple      – block-size limited
    "thr":   "#1D9E75",   # teal        – thread limited
}
LIM_LABEL = {
    "reg":   "register-limited",
    "shmem": "shmem-limited",
    "block": "block-size-limited",
    "thr":   "thread-limited",
}
TOP_N_KERNELS = 9   # kernels to show per panel (by GPU time)


def fig_occupancy_multigpu(
        gpu_data: dict,          # {gpu_name: {growth_label: [kernel_dicts]}}
        outdir: str,
        prefix: str,
        top_n: int = TOP_N_KERNELS,
):
    """
    Draw a grid: rows = GPU models, cols = growth-rate labels.
    Each cell = horizontal bar chart of per-kernel theoretical occupancy.
    """
    gpu_names = list(gpu_data.keys())            # e.g. ["P100","H100","A30"]
    # Collect all growth labels across all GPUs, in canonical order
    all_labels = sorted(
        {lbl for runs in gpu_data.values() for lbl in runs},
        key=_sort_key
    )

    n_rows = len(gpu_names)
    n_cols = len(all_labels)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.2 * n_cols, 4.2 * n_rows),
        sharey=False,
        squeeze=False,
    )
    fig.suptitle(
        f"Theoretical kernel occupancy — {prefix}  (30-year runs)",
        fontsize=12, fontweight="bold", y=1.005,
    )

    for r_idx, gpu_name in enumerate(gpu_names):
        for c_idx, growth in enumerate(all_labels):
            ax = axes[r_idx][c_idx]
            kernels = gpu_data[gpu_name].get(growth, [])

            if not kernels:
                ax.text(0.5, 0.5, "no data",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=9, color="gray")
                ax.set_title(f"{gpu_name} — {growth}", fontsize=9)
                ax.axis("off")
                continue

            # Take top-N by GPU time (already sorted descending)
            kd  = kernels[:top_n]
            occ = [k["th_occ"] for k in kd]
            clr = [LIM_COLOR.get(k["limiter"], "#888780") for k in kd]
            lbl = [k["name"] for k in kd]

            y = np.arange(len(kd))
            bars = ax.barh(y, occ, color=clr, edgecolor="white", linewidth=0.4,
                           height=0.72)

            ax.set_yticks(y)
            ax.set_yticklabels(lbl, fontsize=7.5)
            ax.set_xlim(0, 115)
            ax.axvline(100, color="gray", linewidth=0.7, linestyle="--", alpha=0.5)
            ax.grid(axis="x", linewidth=0.4, linestyle="--", alpha=0.4)
            ax.set_xlabel("Theoretical occupancy (%)", fontsize=8)

            # Column header (growth rate) only on top row
            if r_idx == 0:
                ax.set_title(f"growth = {growth}", fontsize=10,
                             fontweight="600", pad=6)

            # Row header (GPU name) only on leftmost column
            if c_idx == 0:
                ax.set_ylabel(gpu_name, fontsize=11, fontweight="bold",
                              labelpad=8)

            # Value annotations
            for bar, v in zip(bars, occ):
                if v > 0:
                    ax.text(v + 1.2, bar.get_y() + bar.get_height() / 2,
                            f"{v:.0f}%", va="center", fontsize=6.5,
                            fontweight="500")

    # Shared legend (bottom of figure)
    legend_handles = [
        mpatches.Patch(color=LIM_COLOR[k], label=LIM_LABEL[k])
        for k in ("reg", "shmem", "block", "thr")
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        fontsize=9,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.03),
    )

    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        out = os.path.join(outdir, f"{prefix}_occupancy_multigpu.{ext}")
        fig.savefig(out, dpi=300 if ext == "pdf" else 150,
                    bbox_inches="tight")
        print(f"  Wrote {out}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
class _GpuAction(argparse.Action):
    """Accumulate  --gpu <NAME>  file1 label1  file2 label2 ...  blocks."""
    def __call__(self, parser, namespace, values, option_string=None):
        blocks = getattr(namespace, self.dest, None) or []
        blocks.append({"gpu": values, "pairs": []})
        setattr(namespace, self.dest, blocks)

class _PairAction(argparse.Action):
    """Append a (file, label) pair to the most-recent --gpu block."""
    def __call__(self, parser, namespace, values, option_string=None):
        blocks = getattr(namespace, "gpu_blocks", None) or []
        if not blocks:
            parser.error("--files must follow a --gpu argument")
        blocks[-1]["pairs"].append(tuple(values))
        setattr(namespace, "gpu_blocks", blocks)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--gpu", dest="gpu_blocks", action=_GpuAction,
                   metavar="GPU_NAME",
                   help="GPU label (P100|H100|A30|…). Repeat for each GPU.")
    p.add_argument("--files", dest="gpu_blocks", action=_PairAction,
                   nargs=2, metavar=("FILE", "LABEL"),
                   help="sqlite file + label pair (repeat after each --gpu)")
    p.add_argument("--glob-gpu", dest="glob_patterns", nargs=1,
                   action="append", metavar="PATTERN",
                   help="Glob pattern added to the last --gpu block")
    p.add_argument("--outdir",  default="./nsys_output")
    p.add_argument("--prefix",  default="pa_30y")
    p.add_argument("--top-n",   type=int, default=TOP_N_KERNELS,
                   help=f"Kernels per panel (default {TOP_N_KERNELS})")
    args = p.parse_args()

    gpu_blocks = args.gpu_blocks or []

    # Handle --glob-gpu (appended to the most recent --gpu block)
    # (glob_patterns is a list-of-lists from action="append")
    if args.glob_patterns:
        if not gpu_blocks:
            p.error("--glob-gpu must follow a --gpu argument")
        for (pattern,) in args.glob_patterns:
            for f in sorted(_glob.glob(pattern)):
                lbl = infer_label(f)
                gpu_blocks[-1]["pairs"].append((f, lbl))

    if not gpu_blocks:
        p.print_help()
        print("\nExample:")
        print("  python3 nsys_occupancy_multigpu.py \\")
        print("    --gpu P100 \\")
        print("      --files profile/nsys-pa-30y-1p-p100.sqlite '1%' \\")
        print("      --files profile/nsys-pa-30y-3p-p100.sqlite '3%' \\")
        print("      --files profile/nsys-pa-30y-5p-p100.sqlite '5%' \\")
        print("      --files profile/nsys-pa-30y-6p-p100.sqlite '6%' \\")
        print("    --gpu H100 \\")
        print("      --files profile/nsys-pa-30y-1p-h100.sqlite '1%' \\")
        print("      --files profile/nsys-pa-30y-3p-h100.sqlite '3%' \\")
        print("      --files profile/nsys-pa-30y-5p-h100.sqlite '5%' \\")
        print("      --files profile/nsys-pa-30y-6p-h100.sqlite '6%' \\")
        print("    --gpu A30 \\")
        print("      --files profile/nsys-pa-30y-1p-a30.sqlite  '1%' \\")
        print("      --files profile/nsys-pa-30y-3p-a30.sqlite  '3%' \\")
        print("      --files profile/nsys-pa-30y-5p-a30.sqlite  '5%' \\")
        print("      --files profile/nsys-pa-30y-6p-a30.sqlite  '6%'")
        sys.exit(0)

    # Build gpu_data: {gpu_name: {growth_label: [kernel_dicts]}}
    gpu_data: dict = {}

    for block in gpu_blocks:
        gpu_name = block["gpu"].upper()
        hw = GPU_PRESETS.get(gpu_name, GPU_PRESETS["P100"]).copy()
        gpu_data[gpu_name] = {}

        for fpath, label in block["pairs"]:
            if not os.path.exists(fpath):
                print(f"  MISSING: {fpath} — skipping")
                continue
            print(f"  [{gpu_name}] [{label}]  parsing {fpath} …")
            kernels = parse_kernels(fpath, hw)
            gpu_data[gpu_name][label] = kernels
            print(f"    → {len(kernels)} distinct kernels")

    if not any(runs for runs in gpu_data.values()):
        print("No data parsed — check file paths.")
        sys.exit(1)

    print(f"\nGenerating occupancy figure …")
    fig_occupancy_multigpu(gpu_data, args.outdir, args.prefix, args.top_n)
    print("Done.")


if __name__ == "__main__":
    main()
