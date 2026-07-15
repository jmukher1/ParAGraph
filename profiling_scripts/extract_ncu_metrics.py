#!/usr/bin/env python3
"""
extract_ncu_metrics.py
----------------------
Extracts Occupancy, Memory Bandwidth, and Warp Divergence metrics
from multiple ncu-details CSV files and produces:
  1. A per-file summary table (stdout)
  2. A wide CSV for further analysis (ncu_metrics_summary.csv)

Usage:
    python3 extract_ncu_metrics.py [glob_pattern]
    python3 extract_ncu_metrics.py "ncu-er-gnp-*.csv"   # default if omitted
"""

import sys, glob, re, os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# 1. Metric definitions
# ---------------------------------------------------------------------------
METRICS = {
    # --- Occupancy ---
    "sm__warps_active.avg.pct_of_peak_sustained_active": {
        "category": "Occupancy",
        "label":    "Achieved Occupancy (%)",
        "note":     "Active warps / max warps per SM, averaged over elapsed cycles",
    },
    "launch__occupancy_theoretical": {
        "category": "Occupancy",
        "label":    "Theoretical Occupancy (%)",
        "note":     "Limit imposed by register & shared-mem usage",
    },
    "launch__waves_per_multiprocessor": {
        "category": "Occupancy",
        "label":    "Waves per SM",
        "note":     "Grid size / (max blocks per SM * SM count); <1 means undersubscribed",
    },
    "launch__registers_per_thread": {
        "category": "Occupancy",
        "label":    "Registers / Thread",
        "note":     "Higher → fewer warps can coexist → lower occupancy",
    },
    "launch__block_size": {
        "category": "Occupancy",
        "label":    "Block Size",
        "note":     "Threads per block launched",
    },
    "launch__grid_size": {
        "category": "Occupancy",
        "label":    "Grid Size",
        "note":     "Blocks launched",
    },
    "launch__thread_count": {
        "category": "Occupancy",
        "label":    "Total Threads",
        "note":     "block_size * grid_size",
    },

    # --- Memory Bandwidth ---
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": {
        "category": "Bandwidth",
        "label":    "DRAM BW Util (% peak)",
        "note":     "Fraction of peak DRAM bandwidth used; <10% = latency-bound",
    },
    "dram__bytes.sum": {
        "category": "Bandwidth",
        "label":    "DRAM Bytes Transferred",
        "note":     "Total bytes read+written to DRAM",
    },
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": {
        "category": "Bandwidth",
        "label":    "SM Throughput (% peak)",
        "note":     "Fraction of peak SM issue throughput used",
    },
    "lts__t_sectors_srcunit_tex_op_read_hit_rate.pct": {
        "category": "Bandwidth",
        "label":    "L2 Read Hit Rate (%)",
        "note":     "High = data reuse; Low = lots of DRAM traffic",
    },
    "gpu__time_duration.sum": {
        "category": "Bandwidth",
        "label":    "Kernel Duration (ns)",
        "note":     "Total elapsed kernel time",
    },

    # --- Warp Divergence (proxy) ---
    "sm__sass_thread_inst_executed_op_fp32_pred_on.sum": {
        "category": "Divergence",
        "label":    "FP32 Pred-ON Insts",
        "note":     "FP32 insts executed under predication (some threads masked off)",
    },
    "sm__sass_thread_inst_executed_op_fp64_pred_on.sum": {
        "category": "Divergence",
        "label":    "FP64 Pred-ON Insts",
        "note":     "FP64 insts executed under predication",
    },

    # --- Launch config extras ---
    "launch__shared_mem_per_block_dynamic": {
        "category": "Occupancy",
        "label":    "Dynamic Shared Mem / Block (B)",
        "note":     "Dynamically allocated smem; limits concurrent blocks",
    },
    "launch__shared_mem_per_block_static": {
        "category": "Occupancy",
        "label":    "Static Shared Mem / Block (B)",
        "note":     "Statically allocated smem",
    },
}

LABEL_MAP = {k: v["label"] for k, v in METRICS.items()}
CAT_MAP   = {k: v["category"] for k, v in METRICS.items()}

# ---------------------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------------------
def parse_filename(path):
    """Extract (model, years, agents) from filenames like ncu-er-gnp-30y-5p-details.csv"""
    base = os.path.basename(path)
    m = re.search(r'(\d+)y-(\d+)p', base)
    if m:
        return {"years": int(m.group(1)), "agents": int(m.group(2))}
    return {"years": None, "agents": None}

def short_kernel(name):
    if 'kernelErdosRenyi' in name: return 'kernelErdosRenyiGNP'
    if 'kernelCallStage1' in name: return 'kernelCallStage1'
    if 'kernelCallStage2' in name: return 'kernelCallStage2'
    if 'kernel_extract'   in name: return 'kernel_extract_vector_sizes'
    if 'legacy::initialize' in name: return 'legacy::initialize'
    if 'legacy::insert'     in name: return 'legacy::insert'
    if 'insert_if_n'        in name: return 'open_addressing::insert_if_n'
    if 'static_kernel' in name and 'initialize_functor' in name: return 'cub::static_kernel(init)'
    if 'static_kernel' in name: return 'cub::static_kernel(thrust)'
    return name[:50]

def load_file(path):
    df = pd.read_csv(path)
    df['Metric Value'] = pd.to_numeric(
        df['Metric Value'].astype(str).str.replace(',', ''), errors='coerce'
    )
    df['ShortKernel'] = df['Kernel Name'].apply(short_kernel)
    info = parse_filename(path)
    df['years']  = info['years']
    df['agents'] = info['agents']
    df['source'] = os.path.basename(path)
    return df

# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------
def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/ncu-er-gnp-*.csv"
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files matched: {pattern}")
        sys.exit(1)

    print(f"Found {len(files)} file(s):\n  " + "\n  ".join(files) + "\n")

    all_dfs = [load_file(f) for f in files]
    df = pd.concat(all_dfs, ignore_index=True)

    # Filter to known metrics only
    df = df[df['Metric Name'].isin(METRICS)]

    # Pivot: one row per (source, kernel), one col per metric
    pivot = df.pivot_table(
        index=['source', 'years', 'agents', 'ShortKernel'],
        columns='Metric Name',
        values='Metric Value',
        aggfunc='mean'
    ).reset_index()
    pivot.columns.name = None

    # Rename metric columns to human-readable labels
    rename = {k: v for k, v in LABEL_MAP.items() if k in pivot.columns}
    pivot = pivot.rename(columns=rename)

    # Compute derived metrics
    if 'FP32 Pred-ON Insts' in pivot.columns and 'FP64 Pred-ON Insts' in pivot.columns:
        pivot['Total Pred-ON Insts (Divergence Proxy)'] = (
            pivot['FP32 Pred-ON Insts'].fillna(0) +
            pivot['FP64 Pred-ON Insts'].fillna(0)
        )

    # Sort by years, agents
    pivot = pivot.sort_values(['years', 'agents', 'ShortKernel'])

    # ---------------------------------------------------------------------------
    # 4. Print grouped summary
    # ---------------------------------------------------------------------------
    categories = {
        "OCCUPANCY": [
            'Achieved Occupancy (%)', 'Theoretical Occupancy (%)',
            'Waves per SM', 'Registers / Thread',
            'Block Size', 'Grid Size', 'Total Threads',
            'Dynamic Shared Mem / Block (B)', 'Static Shared Mem / Block (B)',
        ],
        "MEMORY BANDWIDTH": [
            'DRAM BW Util (% peak)', 'DRAM Bytes Transferred',
            'SM Throughput (% peak)', 'L2 Read Hit Rate (%)', 'Kernel Duration (ns)',
        ],
        "WARP DIVERGENCE (proxy)": [
            'FP32 Pred-ON Insts', 'FP64 Pred-ON Insts',
            'Total Pred-ON Insts (Divergence Proxy)',
        ],
    }

    for cat, cols in categories.items():
        available = [c for c in cols if c in pivot.columns]
        if not available:
            continue
        print("=" * 80)
        print(f"  {cat}")
        print("=" * 80)
        show = pivot[['source', 'years', 'agents', 'ShortKernel'] + available].copy()
        # Format numbers
        for c in available:
            if show[c].dropna().empty:
                continue
            maxval = show[c].abs().max()
            if maxval > 1e6:
                show[c] = show[c].map(lambda x: f"{x/1e6:.2f}M" if pd.notna(x) else "")
            elif maxval > 1e3:
                show[c] = show[c].map(lambda x: f"{x:.0f}" if pd.notna(x) else "")
            else:
                show[c] = show[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
        print(show.to_string(index=False))
        print()

    # ---------------------------------------------------------------------------
    # 5. Print metric notes legend
    # ---------------------------------------------------------------------------
    print("=" * 80)
    print("  METRIC NOTES")
    print("=" * 80)
    for k, v in METRICS.items():
        label = v['label']
        if label in pivot.columns or any(label in cols for cols in categories.values()):
            print(f"  [{v['category']:10s}] {label}")
            print(f"              {v['note']}")
    print()

    # ---------------------------------------------------------------------------
    # 6. Save CSV
    # ---------------------------------------------------------------------------
    out = "./ncu_metrics_summary.csv"
    os.makedirs("/mnt/user-data/outputs", exist_ok=True)
    pivot.to_csv(out, index=False)
    print(f"Saved: {out}")

    # ---------------------------------------------------------------------------
    # 7. Print quick interpretation hints
    # ---------------------------------------------------------------------------
    print()
    print("=" * 80)
    print("  INTERPRETATION GUIDE")
    print("=" * 80)
    hints = [
        ("Achieved Occupancy < 25%",   "GPU is heavily underutilized — too few active warps to hide latency"),
        ("DRAM BW Util < 10%",         "Not memory-bound; bottleneck is likely latency or compute starvation"),
        ("Waves per SM < 1",           "Grid is too small to saturate all SMs"),
        ("Registers/Thread > 64",      "High register pressure; reduces max concurrent warps per SM"),
        ("L2 Read Hit Rate < 50%",     "Poor cache reuse; irregular access pattern likely"),
        ("Large Pred-ON Insts",        "Many predicated instructions → warp divergence from branches/conditionals"),
        ("SM Throughput < 10%",        "Issue slots nearly empty; kernel is latency-bound, not ILP-bound"),
    ]
    for cond, meaning in hints:
        print(f"  ► {cond}")
        print(f"    {meaning}")
    print()

if __name__ == "__main__":
    main()
