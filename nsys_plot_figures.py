#!/usr/bin/env python3
"""
nsys_plot_figures.py  —  STEP 2 of 2
=======================================
Reads the three CSVs written by nsys_export_csv.py and produces three
publication-ready figures:

  Figure A  pa_occupancy.pdf / .png
            Theoretical occupancy per kernel — one sub-panel per run
            (replicates Image 2: four horizontal-bar panels for 30y 1/3/5/6%)

  Figure B  pa_gpu_util.pdf / .png
            GPU utilisation % and memcpy overhead % grouped bar chart
            (replicates Image 1: blue/orange grouped bars across all runs)

  Figure C  pa_memcpy.pdf / .png
            Stacked H2D / D2H / UVM_H2D / UVM_D2H transfer bar chart
            (replicates Image 4: stacked bar, total annotated on top)

USAGE
-----
  python3 nsys_plot_figures.py \\
      --occupancy ./nsys_csv/pa_occupancy.csv \\
      --gpu-util  ./nsys_csv/pa_gpu_util.csv  \\
      --memcpy    ./nsys_csv/pa_memcpy.csv    \\
      --outdir    ./nsys_figures --prefix pa

  # Only plot one figure:
  python3 nsys_plot_figures.py --occupancy pa_occupancy.csv --outdir ./figs

REQUIREMENTS
------------
  pip install matplotlib numpy pandas
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# SHARED STYLE
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8.5,
    "legend.fontsize":   8.5,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linewidth":    0.5,
})

# Limiter → bar colour (matches Image 2: green/orange/red)
LIMITER_COLOR = {
    "thr":   "#1D9E75",   # green   — thread-limited (good occupancy)
    "block": "#1D9E75",   # green   — block-size limited
    "shmem": "#EF9F27",   # amber   — shared-memory limited
    "reg":   "#D85A30",   # red     — register-limited (bad)
}
DEFAULT_OCC_COLOR = "#888780"

# Memory-kind colours (matches Image 4)
KIND_COLORS = {
    "H2D":     "#185FA5",   # dark blue
    "D2H":     "#1D9E75",   # teal
    "UVM_H2D": "#85B7EB",   # light blue
    "UVM_D2H": "#9FE1CB",   # light teal
    "D2D":     "#B4B2A9",   # grey
    "H2H":     "#D3D1C7",
    "Unknown": "#888780",
}
KIND_ORDER = ["H2D", "D2H", "UVM_H2D", "UVM_D2H", "D2D"]

# Kernel colour map for occupancy bars (matches nsys_analyze_full.py)
KERN_COLORS = {
    "Stage1":          "#185FA5",
    "Stage2":          "#1D9E75",
    "Stage3":          "#7F77DD",
    "Stage4":          "#854F0B",
    "ABMStage2":       "#D85A30",
    "ABMStage4":       "#EF9F27",
    "SameYearCite":    "#D4537E",
    "UniformCite":     "#993C1D",
    "2HopWarp":        "#5DCAA5",
    "2Hop":            "#9FE1CB",
    "ExtractSizes":    "#AFA9EC",
    "cuco::insert":    "#D85A30",
    "cuco::insert_if": "#FAC775",
    "cuco::init":      "#B4B2A9",
    "cub::ForEach":    "#D3D1C7",
    "cub::RadixSort":  "#C0DD97",
}


def _save(fig: plt.Figure, outdir: str, name: str):
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(outdir, f"{name}.{ext}")
        fig.savefig(p, bbox_inches="tight")
        print(f"  Saved {p}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE A — Theoretical Occupancy
# One horizontal-bar sub-panel per run, kernels on y-axis, occ% on x-axis.
# Bars are coloured by occupancy limiter (reg / shmem / thr / block).
# ══════════════════════════════════════════════════════════════════════════════
def fig_occupancy(csv_path: str, outdir: str, prefix: str,
                  top_n: int = 9):
    df = pd.read_csv(csv_path)
    # Drop zero-time rows (helpers / tiny kernels that never ran)
    df = df[df["Total_ms"].astype(float) > 0].copy()

    labels = df["Label"].unique().tolist()
    n      = len(labels)
    if n == 0:
        print("  occupancy CSV is empty — skipping Figure A")
        return

    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 5.2), sharey=False)
    if n == 1:
        axes = [axes]

    fig.suptitle("Theoretical occupancy per kernel",
                 fontsize=11, fontweight="bold", y=1.01)

    for ax, lbl in zip(axes, labels):
        sub = df[df["Label"] == lbl].copy()
        # Sort by pct_GPU descending so the hottest kernel is at top
        sub = sub.sort_values("pct_GPU", ascending=True).tail(top_n)

        knames  = sub["Kernel"].tolist()
        occ     = sub["ThOcc_pct"].astype(float).tolist()
        limiter = sub["Limiter"].tolist()
        colors  = [LIMITER_COLOR.get(l, DEFAULT_OCC_COLOR) for l in limiter]

        y = np.arange(len(knames))
        bars = ax.barh(y, occ, color=colors, edgecolor="white",
                       linewidth=0.4, height=0.65)

        # Annotate bar end with percentage
        for bar, v in zip(bars, occ):
            if v > 0:
                ax.text(
                    v + 1.0,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.0f}%",
                    va="center", fontsize=8.5, fontweight="500",
                )

        ax.set_xlim(0, 118)
        ax.axvline(100, color="#888780", lw=0.7, ls="--", alpha=0.45)
        ax.set_yticks(y)
        ax.set_yticklabels(knames, fontsize=8.5)
        ax.set_xlabel("Theoretical occupancy (%)", fontsize=9)
        ax.set_title(lbl, fontsize=10, fontweight="500", pad=5)
        ax.grid(axis="x", lw=0.4, ls="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Shared legend
    legend_handles = [
        mpatches.Patch(color=LIMITER_COLOR["reg"],   label="Register-limited"),
        mpatches.Patch(color=LIMITER_COLOR["shmem"], label="Shmem-limited"),
        mpatches.Patch(color=LIMITER_COLOR["thr"],   label="Thread/block-limited"),
    ]
    axes[0].legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=8.5,
        framealpha=0.9,
    )

    plt.tight_layout()
    _save(fig, outdir, f"{prefix}_occupancy")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE B — GPU Utilisation & Memcpy Overhead
# Grouped bar chart: blue = GPU util %, orange = memcpy overhead %.
# Exactly matches Image 1 layout.
# ══════════════════════════════════════════════════════════════════════════════
def fig_gpu_util(csv_path: str, outdir: str, prefix: str):
    df = pd.read_csv(csv_path)
    if df.empty:
        print("  gpu_util CSV is empty — skipping Figure B")
        return

    labels    = df["Label"].tolist()
    gpu_vals  = df["GPU_util_pct"].astype(float).tolist()
    mcpy_vals = df["Memcpy_ovh_pct"].astype(float).tolist()

    x     = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(labels)), 5))

    bars_gpu  = ax.bar(x - width / 2, gpu_vals,  width,
                       color="#378ADD", label="GPU util %",
                       edgecolor="white", linewidth=0.4)
    bars_mcpy = ax.bar(x + width / 2, mcpy_vals, width,
                       color="#D85A30", label="Memcpy ovh %",
                       edgecolor="white", linewidth=0.4)

    # Value labels on top of each bar
    for bar, v in zip(bars_gpu, gpu_vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + 0.4, f"{v:.1f}%",
                    ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold")

    for bar, v in zip(bars_mcpy, mcpy_vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + 0.4, f"{v:.1f}%",
                    ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=9)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("GPU util & memcpy overhead",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_ylim(0, max(max(gpu_vals), max(mcpy_vals)) * 1.18)
    ax.grid(axis="y", lw=0.4, ls="--", alpha=0.4)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    _save(fig, outdir, f"{prefix}_gpu_util")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE C — Memory Transfer Breakdown
# Stacked bar per run, one segment per transfer kind.
# Total annotated on top of each bar — matches Image 4 exactly.
# ══════════════════════════════════════════════════════════════════════════════
def fig_memcpy(csv_path: str, outdir: str, prefix: str):
    df = pd.read_csv(csv_path)
    if df.empty:
        print("  memcpy CSV is empty — skipping Figure C")
        return

    # Pivot: rows = Label, cols = Kind, values = Bytes_MB
    pivot = (
        df.pivot_table(index="Label", columns="Kind",
                       values="Bytes_MB", aggfunc="sum")
          .fillna(0)
    )

    # Preserve label order from the CSV
    label_order = df["Label"].unique().tolist()
    pivot = pivot.reindex(label_order)

    # Keep only known kinds (in display order), add any others at end
    present_kinds = [k for k in KIND_ORDER if k in pivot.columns]
    extra_kinds   = [k for k in pivot.columns if k not in KIND_ORDER]
    ordered_kinds = present_kinds + extra_kinds

    fig, ax = plt.subplots(figsize=(max(10, 1.1 * len(label_order)), 5.5))
    x       = np.arange(len(label_order))
    bottoms = np.zeros(len(label_order))

    for kind in ordered_kinds:
        if kind not in pivot.columns:
            continue
        vals = pivot[kind].values
        if vals.sum() == 0:
            continue
        color = KIND_COLORS.get(kind, "#888780")
        ax.bar(x, vals, bottom=bottoms, label=kind, color=color,
               edgecolor="white", linewidth=0.3, width=0.72)
        bottoms += vals

    # Total annotation on top of each bar
    for i, (lbl, tot) in enumerate(zip(label_order, bottoms)):
        if tot > 0:
            # Choose unit automatically
            if tot >= 1000:
                txt = f"{tot / 1000:.1f} GB"
            elif tot >= 1:
                txt = f"{tot:.0f} MB"
            else:
                txt = f"{tot * 1000:.0f} KB"
            ax.text(i, tot + bottoms.max() * 0.008, txt,
                    ha="center", va="bottom",
                    fontsize=8.5, fontweight="500")

    ax.set_xticks(x)
    ax.set_xticklabels(label_order, rotation=0, ha="center", fontsize=9)
    ax.set_ylabel("Data transferred (MB)")
    ax.set_title("Memory transfer breakdown",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9,
              title="Transfer kind", title_fontsize=8)
    ax.set_ylim(0, bottoms.max() * 1.14)
    ax.grid(axis="y", lw=0.4, ls="--", alpha=0.4)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    _save(fig, outdir, f"{prefix}_memcpy")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--occupancy", metavar="CSV",
                   help="Occupancy CSV from nsys_export_csv.py")
    p.add_argument("--gpu-util",  metavar="CSV",
                   help="GPU util CSV from nsys_export_csv.py")
    p.add_argument("--memcpy",    metavar="CSV",
                   help="Memcpy CSV from nsys_export_csv.py")
    p.add_argument("--outdir",  default="./nsys_figures",
                   help="Output directory (default: ./nsys_figures)")
    p.add_argument("--prefix",  default="pa",
                   help="Output filename prefix (default: pa)")
    p.add_argument("--top-n",   type=int, default=9,
                   help="Max kernels shown per occupancy panel (default: 9)")
    args = p.parse_args()

    if not any([args.occupancy, args.gpu_util, args.memcpy]):
        p.print_help()
        print("\nExample (after running Step 1):")
        print("  python3 nsys_plot_figures.py \\")
        print("      --occupancy ./nsys_csv/pa_occupancy.csv \\")
        print("      --gpu-util  ./nsys_csv/pa_gpu_util.csv  \\")
        print("      --memcpy    ./nsys_csv/pa_memcpy.csv    \\")
        print("      --outdir ./nsys_figures --prefix pa")
        sys.exit(0)

    os.makedirs(args.outdir, exist_ok=True)
    print(f"\nOutput dir : {args.outdir}/")
    print(f"Prefix     : {args.prefix}\n")

    if args.occupancy:
        if not os.path.exists(args.occupancy):
            print(f"  ERROR: {args.occupancy} not found")
        else:
            print(f"Figure A — Theoretical Occupancy")
            fig_occupancy(args.occupancy, args.outdir, args.prefix,
                          top_n=args.top_n)

    if args.gpu_util:
        if not os.path.exists(args.gpu_util):
            print(f"  ERROR: {args.gpu_util} not found")
        else:
            print(f"Figure B — GPU Util & Memcpy Overhead")
            fig_gpu_util(args.gpu_util, args.outdir, args.prefix)

    if args.memcpy:
        if not os.path.exists(args.memcpy):
            print(f"  ERROR: {args.memcpy} not found")
        else:
            print(f"Figure C — Memory Transfer Breakdown")
            fig_memcpy(args.memcpy, args.outdir, args.prefix)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
