#!/usr/bin/env python3
"""
paper_tables.py
===============
Generate publication-quality tables and figures for SC papers from
ncu CSV exports and nsys SQLite databases.

GPU: NVIDIA H100 80GB HBM3 (sm_90)
Peak FP32: 67 TFLOP/s  |  Peak BW: 3.35 TB/s  |  Ridge: 20.0 FLOP/byte

STEP 1 — export ncu-rep to CSV on the cluster:
    module purge && module load cuda/12.6
    for f in profile/ncu-pa-*.ncu-rep; do
        base=$(basename $f .ncu-rep)
        ncu --import $f --csv --page details > profile/${base}-details.csv
    done

STEP 2 — run this script:
    python3 paper_tables.py \
        --ncu  profile/ncu-pa-3y-1p-details.csv  "3y" "1" \
        --ncu  profile/ncu-pa-10y-1p-details.csv "10y" "1" \
        --ncu  profile/ncu-pa-30y-1p-details.csv "30y" "1" \
        --nsys profile/nsys-pa-3y-1p.sqlite      "3y" "1" \
        --nsys profile/nsys-pa-10y-1p.sqlite     "10y" "1" \
        --nsys profile/nsys-pa-30y-1p.sqlite     "30y" "1" \
        --outdir ./paper_output \
        --prefix pa

OUTPUT:
    paper_output/
      pa_ncu_resource_table.tex        LaTeX table: resource utilization per kernel
      pa_ncu_launch_table.tex          LaTeX table: launch configuration
      pa_nsys_session_table.tex        LaTeX table: session-level metrics
      pa_ncu_resource_table.csv        CSV versions of all tables
      pa_ncu_launch_table.csv
      pa_nsys_session_table.csv
      pa_occupancy_plot.pdf/png        Occupancy bar chart
      pa_roofline_plot.pdf/png         Roofline model
      pa_kernel_time_plot.pdf/png      Kernel time breakdown (stacked bar)
      pa_scaling_plot.pdf/png          Scaling across years
"""

import argparse, csv, math, os, re, sqlite3, sys
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("WARNING: pip install matplotlib numpy  for figures\n")

# ── H100 hardware constants ───────────────────────────────────────────────────
H100 = dict(
    name         = "NVIDIA H100 80GB HBM3",
    arch         = "sm_90",
    peak_fp32    = 67000.0,    # GFLOP/s
    peak_bw      = 3350.0,     # GB/s
    sm_count     = 132,
    max_warps_sm = 64,
    max_thr_sm   = 2048,
    max_blk_sm   = 32,
    shmem_sm_B   = 229376,     # 228 KB
    regs_sm      = 65536,
    warp_size    = 32,
)
RIDGE = H100["peak_fp32"] / H100["peak_bw"]   # 20.0 FLOP/byte

# ── ncu metric name → short alias ────────────────────────────────────────────
METRIC_MAP = {
    "sm__warps_active.avg.pct_of_peak_sustained_active":      "achieved_occ_pct",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": "dram_bw_pct",
    "sm__sass_thread_inst_executed_op_fp32_pred_on.sum":      "fp32_insts",
    "sm__sass_thread_inst_executed_op_fp64_pred_on.sum":      "fp64_insts",
    "dram__bytes.sum":                                         "dram_bytes",
    "gpu__time_duration.sum":                                  "duration_ns",
    "lts__t_sectors_srcunit_tex_op_read_hit_rate.pct":        "l2_hit_pct",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed":       "sm_throughput_pct",
    "launch__registers_per_thread":                            "regs",
    "launch__shared_mem_per_block_static":                     "shmem_B",
    "launch__shared_mem_per_block_dynamic":                    "shmem_dyn_B",
    "launch__occupancy_theoretical":                           "th_occ_pct",
    "launch__waves_per_multiprocessor":                        "waves_sm",
    "launch__block_size":                                      "block_size",
    "launch__grid_size":                                       "grid_size",
    "launch__thread_count":                                    "thread_count",
}

# ── kernel name normalizer ────────────────────────────────────────────────────
_KRULES = [
    (r"kernelCallStage1",              "Stage1"),
    (r"kernelCallStage2",              "Stage2"),
    (r"kernelCallStage3",              "Stage3"),
    (r"ABMKernelStage2",               "ABMStage2"),
    (r"ABMKernelStage4",               "ABMStage4"),
    (r"MakeSameYear",                  "SameYearCite"),
    (r"MakeUniform",                   "UniformCite"),
    (r"GetOneAndTwoHopNeighborhood_Warp","2HopWarp"),
    (r"GetOneAndTwoHop",               "2Hop"),
    (r"kernel_extract_vector",         "ExtractSizes"),
    (r"cuco.*initialize",              "cuco::init"),
    (r"cuco.*insert_if",               "cuco::insert_if"),
    (r"cuco.*insert",                  "cuco::insert"),
    (r"cub.*DeviceRadixSort",          "cub::RadixSort"),
    (r"cub.*DeviceScan",               "cub::Scan"),
    (r"cub.*for_each",                 "cub::ForEach"),
]

def norm_kernel(name: str) -> str:
    for pat, short in _KRULES:
        if re.search(pat, name):
            return short
    return name.split("(")[0].split("<")[0].split("::")[-1][:25]

def clean_val(s):
    try:
        return float(str(s).replace(",", "").strip())
    except:
        return None

# ── ncu CSV parser ────────────────────────────────────────────────────────────
def parse_ncu_csv(path: str) -> dict:
    with open(path, encoding="utf-8-sig") as f:
        raw = f.read()
    lines = raw.splitlines()
    hi = next((i for i, l in enumerate(lines)
               if "Kernel Name" in l and "Metric Name" in l), None)
    if hi is None:
        print(f"  ERROR: no CSV header in {path}")
        return {}
    kdata = defaultdict(lambda: defaultdict(list))
    for row in csv.DictReader(lines[hi:]):
        kraw   = row.get("Kernel Name", "").strip()
        metric = row.get("Metric Name",  "").strip()
        vstr   = row.get("Metric Value", "").strip()
        if not kraw or not metric:
            continue
        alias = METRIC_MAP.get(metric, None)
        if alias is None:
            continue
        val = clean_val(vstr)
        if val is not None:
            kdata[norm_kernel(kraw)][alias].append(val)
    return {k: {m: sum(v)/len(v) for m, v in md.items()}
            for k, md in kdata.items()}

# ── theoretical occupancy calculator ─────────────────────────────────────────
def th_occ(regs: int, block: int, shmem: int) -> float:
    hw = H100
    if block == 0:
        return 0.0
    wpb = math.ceil(block / hw["warp_size"])
    lim_thr  = hw["max_thr_sm"] // block
    if regs > 0:
        rpw = math.ceil(regs * hw["warp_size"] / 256) * 256
        lim_reg = hw["regs_sm"] // (rpw * wpb) if rpw * wpb > 0 else hw["max_blk_sm"]
    else:
        lim_reg = hw["max_blk_sm"]
    if shmem > 0:
        sa = math.ceil(shmem / 256) * 256
        lim_shm = hw["shmem_sm_B"] // sa
    else:
        lim_shm = hw["max_blk_sm"]
    max_blk = min(lim_thr, lim_reg, lim_shm, hw["max_blk_sm"])
    return round(min(max_blk * wpb / hw["max_warps_sm"], 1.0) * 100, 1)

def occ_limiter(m: dict) -> str:
    r = int(m.get("regs", 0))
    s = int(m.get("shmem_B", 0))
    b = int(m.get("block_size", 0))
    if r > 40:  return "reg"
    if s > 0:   return "shmem"
    if b < 64:  return "block"
    return "thr"

# ── roofline ──────────────────────────────────────────────────────────────────
def roofline(m: dict) -> dict:
    fp32 = m.get("fp32_insts", 0) or 0
    fp64 = m.get("fp64_insts", 0) or 0
    dram = m.get("dram_bytes",  0) or 0
    dur  = m.get("duration_ns", 0) or 0
    flops = (fp32 + fp64) * 2
    if dram > 0 and dur > 0 and flops > 0:
        return dict(ai   = flops / dram,
                    perf = flops / (dur * 1e-9) / 1e9,
                    bound = "compute" if flops/dram >= RIDGE else "memory")
    return {}

# ── nsys SQLite parser ────────────────────────────────────────────────────────
MEMCPY_KIND = {1:"H2D", 2:"D2H", 8:"D2D", 11:"UVM_H2D", 12:"UVM_D2H"}

def parse_nsys_sqlite(path: str) -> dict:
    con = sqlite3.connect(path)
    sids = dict(con.execute("SELECT id, value FROM StringIds").fetchall())

    # session
    ses = con.execute("SELECT duration, startTime FROM ANALYSIS_DETAILS").fetchone()
    session_ns = ses[0]

    # kernels
    krows = con.execute("""
        SELECT start, end, demangledName, registersPerThread,
               gridX, blockX, staticSharedMemory, dynamicSharedMemory,
               localMemoryPerThread
        FROM CUPTI_ACTIVITY_KIND_KERNEL""").fetchall()

    kdata = {}
    total_kern_ns = 0
    for (st, en, dn_id, regs, gx, bx, sm_s, sm_d, lmem) in krows:
        dur = en - st
        total_kern_ns += dur
        kn = norm_kernel(sids.get(dn_id, "unknown"))
        if kn not in kdata:
            kdata[kn] = dict(count=0, total_ns=0, min_ns=int(1e18), max_ns=0,
                             regs=regs, block=bx, grid_samples=[],
                             shmem_s=sm_s, lmem=lmem,
                             th_occ=th_occ(regs, bx, sm_s))
        d = kdata[kn]
        d["count"]    += 1
        d["total_ns"] += dur
        d["min_ns"]    = min(d["min_ns"], dur)
        d["max_ns"]    = max(d["max_ns"], dur)
        d["grid_samples"].append(gx)

    # memcpy
    mrows = con.execute(
        "SELECT start, end, bytes, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY"
    ).fetchall()
    mdata = {}
    total_memcpy_ns = 0
    total_bytes = 0
    for (st, en, b, ck) in mrows:
        dur  = en - st
        kind = MEMCPY_KIND.get(ck, f"kind{ck}")
        if kind not in mdata:
            mdata[kind] = dict(count=0, total_ns=0, total_bytes=0)
        mdata[kind]["count"]       += 1
        mdata[kind]["total_ns"]    += dur
        mdata[kind]["total_bytes"] += b
        total_memcpy_ns += dur
        total_bytes     += b

    con.close()

    gpu_util   = total_kern_ns / session_ns * 100
    memcpy_ovh = total_memcpy_ns / total_kern_ns * 100 if total_kern_ns > 0 else 0
    eff_bw     = total_bytes/(total_memcpy_ns*1e-9)/1e9 if total_memcpy_ns > 0 else 0

    return dict(
        session_ms      = session_ns / 1e6,
        kernel_ms       = total_kern_ns / 1e6,
        memcpy_ms       = total_memcpy_ns / 1e6,
        gpu_util_pct    = round(gpu_util, 2),
        memcpy_ovh_pct  = round(memcpy_ovh, 2),
        total_bytes_MB  = total_bytes / 1e6,
        eff_bw_GBs      = round(eff_bw, 2),
        bw_util_pct     = round(eff_bw / H100["peak_bw"] * 100, 3),
        kernels         = kdata,
        memcpy          = mdata,
        total_kern_launches = len(krows),
    )

# ── LaTeX helpers ─────────────────────────────────────────────────────────────
def latex_header(caption: str, label: str, cols: str) -> str:
    return (f"\\begin{{table}}[t]\n"
            f"\\centering\n"
            f"\\caption{{{caption}}}\n"
            f"\\label{{tab:{label}}}\n"
            f"\\resizebox{{\\columnwidth}}{{!}}{{%\n"
            f"\\begin{{tabular}}{{{cols}}}\n"
            f"\\toprule\n")

def latex_footer() -> str:
    return "\\bottomrule\n\\end{tabular}}\n\\end{table}\n"

def fmt(v, dec=1, suffix=""):
    if v is None or v == 0:
        return "—"
    return f"{v:.{dec}f}{suffix}"

# ── Table 1: ncu resource utilization ────────────────────────────────────────
def make_ncu_resource_table(runs: list, outdir: str, prefix: str):
    """
    Rows = kernels, Cols = (Ach.Occ%, DRAM BW%, SM%, L2%, Regs) × years
    One column group per simulation length.
    """
    # Collect all unique kernel names sorted by total time in last run
    all_kn = []
    if runs:
        last_kdata = runs[-1][2]
        all_kn = sorted(last_kdata,
                        key=lambda x: -last_kdata[x].get("achieved_occ_pct", 0))

    labels = [r[0] for r in runs]  # e.g. ["3y","10y","30y"]

    # CSV
    csv_path = os.path.join(outdir, f"{prefix}_ncu_resource_table.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["Kernel"]
        for lbl in labels:
            header += [f"{lbl}_AchOcc%", f"{lbl}_DRAM_BW%",
                       f"{lbl}_SM%", f"{lbl}_L2hit%", f"{lbl}_Regs"]
        header += ["Limiter", "Bound"]
        w.writerow(header)
        for kn in all_kn:
            row = [kn]
            for _, _, kdata in runs:
                m = kdata.get(kn, {})
                row += [fmt(m.get("achieved_occ_pct"),1),
                        fmt(m.get("dram_bw_pct"),1),
                        fmt(m.get("sm_throughput_pct"),1),
                        fmt(m.get("l2_hit_pct"),1),
                        str(int(m.get("regs", 0))) if m.get("regs") else "—"]
            # limiter from last run
            m_last = runs[-1][2].get(kn, {}) if runs else {}
            row += [occ_limiter(m_last) if m_last else "—",
                    roofline(m_last).get("bound","—") if m_last else "—"]
            w.writerow(row)
    print(f"  Wrote {csv_path}")

    # LaTeX
    n = len(labels)
    cols = "l" + "rrrrr" * n + "ll"
    cap  = (f"Kernel-level GPU resource utilization profiled with "
            f"NVIDIA Nsight Compute~\\cite{{nsightcompute}} on "
            f"{H100['name']} (sm\\_90). "
            f"Ach.Occ = \\texttt{{sm\\_\\_warps\\_active.avg.\\%peak}}; "
            f"DRAM BW = \\texttt{{gpu\\_\\_dram\\_throughput.avg.\\%peak}}; "
            f"SM = \\texttt{{sm\\_\\_throughput.avg.\\%peak}}; "
            f"L2 = L2 read hit rate (\\%); Regs = registers/thread. "
            f"PA model, 1\\% annual growth rate.")
    tex_path = os.path.join(outdir, f"{prefix}_ncu_resource_table.tex")
    with open(tex_path, "w") as f:
        f.write(latex_header(cap, f"{prefix}-ncu-resource", cols))
        # multi-col header
        f.write("\\multirow{2}{*}{\\textbf{Kernel}}")
        for lbl in labels:
            f.write(f" & \\multicolumn{{5}}{{c}}{{\\textbf{{{lbl}}}}}")
        f.write(" & \\textbf{Lim.} & \\textbf{Bound}\\\\\n")
        f.write("\\cmidrule(lr){" + "} \\cmidrule(lr){".join(
            str(2 + i*5) + "-" + str(6 + i*5) for i in range(n)) + "}\n")
        sub = " & ".join(
            ["\\small Ach.Occ\\% & \\small DRAM\\% & \\small SM\\% & "
             "\\small L2\\% & \\small Regs"]*n)
        f.write(f" & {sub} & & \\\\\n\\midrule\n")
        for kn in all_kn:
            row = [f"\\texttt{{{kn}}}"]
            for _, _, kdata in runs:
                m = kdata.get(kn, {})
                row += [fmt(m.get("achieved_occ_pct"),1),
                        fmt(m.get("dram_bw_pct"),1),
                        fmt(m.get("sm_throughput_pct"),1),
                        fmt(m.get("l2_hit_pct"),1),
                        str(int(m.get("regs",0))) if m.get("regs") else "—"]
            m_last = runs[-1][2].get(kn, {}) if runs else {}
            row += [occ_limiter(m_last) if m_last else "—",
                    roofline(m_last).get("bound","—") if m_last else "—"]
            f.write(" & ".join(row) + " \\\\\n")
        f.write(latex_footer())
    print(f"  Wrote {tex_path}")

# ── Table 2: ncu launch configuration ────────────────────────────────────────
def make_ncu_launch_table(runs: list, outdir: str, prefix: str):
    if not runs:
        return
    label0, _, kdata0 = runs[0]
    kns = sorted(kdata0, key=lambda x: -kdata0[x].get("achieved_occ_pct", 0))

    csv_path = os.path.join(outdir, f"{prefix}_ncu_launch_table.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Kernel", "Block", "Grid", "Regs/thr",
                    "Shmem(B)", "Th.Occ%", "Waves/SM", "Limiter"])
        for kn in kns:
            m = kdata0.get(kn, {})
            w.writerow([kn,
                        int(m.get("block_size",0)),
                        int(m.get("grid_size",0)),
                        int(m.get("regs",0)),
                        int(m.get("shmem_B",0)),
                        fmt(m.get("th_occ_pct"),1),
                        fmt(m.get("waves_sm"),2),
                        occ_limiter(m)])
    print(f"  Wrote {csv_path}")

    tex_path = os.path.join(outdir, f"{prefix}_ncu_launch_table.tex")
    cap = (f"Kernel launch configuration and theoretical occupancy "
           f"(\\texttt{{launch\\_\\_*}} metrics, no hardware counter required). "
           f"Th.Occ = theoretical occupancy (\\%); Lim = binding occupancy limiter. "
           f"Values are identical across all simulation lengths (configuration "
           f"does not vary with scale).")
    with open(tex_path, "w") as f:
        f.write(latex_header(cap, f"{prefix}-ncu-launch",
                             "lrrrrrrl"))
        f.write("\\textbf{Kernel} & \\textbf{Block} & \\textbf{Grid} & "
                "\\textbf{Regs} & \\textbf{Shmem (B)} & \\textbf{Th.Occ\\%} & "
                "\\textbf{Waves/SM} & \\textbf{Lim.} \\\\\n\\midrule\n")
        for kn in kns:
            m = kdata0.get(kn, {})
            regs = int(m.get("regs", 0))
            occ  = m.get("th_occ_pct", 0) or th_occ(
                regs, int(m.get("block_size",0)), int(m.get("shmem_B",0)))
            f.write(f"\\texttt{{{kn}}} & "
                    f"{int(m.get('block_size',0))} & "
                    f"{int(m.get('grid_size',0))} & "
                    f"{regs} & "
                    f"{int(m.get('shmem_B',0))} & "
                    f"{fmt(occ,1)} & "
                    f"{fmt(m.get('waves_sm'),2)} & "
                    f"{occ_limiter(m)} \\\\\n")
        f.write(latex_footer())
    print(f"  Wrote {tex_path}")

# ── Table 3: nsys session-level ───────────────────────────────────────────────
def make_nsys_session_table(nsys_runs: list, outdir: str, prefix: str):
    if not nsys_runs:
        return

    csv_path = os.path.join(outdir, f"{prefix}_nsys_session_table.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Years","Growth%","Session(ms)","Kernel(ms)",
                    "GPU_util%","Memcpy_ovh%","Transfers(MB)",
                    "EffBW(GB/s)","BW_util%","KernelLaunches"])
        for years, pct, ns in nsys_runs:
            w.writerow([years, pct,
                        f"{ns['session_ms']:.0f}",
                        f"{ns['kernel_ms']:.0f}",
                        f"{ns['gpu_util_pct']:.1f}",
                        f"{ns['memcpy_ovh_pct']:.1f}",
                        f"{ns['total_bytes_MB']:.1f}",
                        f"{ns['eff_bw_GBs']:.2f}",
                        f"{ns['bw_util_pct']:.3f}",
                        ns["total_kern_launches"]])
    print(f"  Wrote {csv_path}")

    tex_path = os.path.join(outdir, f"{prefix}_nsys_session_table.tex")
    cap = (f"Session-level GPU profiling metrics measured with "
           f"NVIDIA Nsight Systems~\\cite{{nsightsystems}} on "
           f"{H100['name']}. "
           f"GPU util. = kernel time / session time; "
           f"Memcpy ovh. = memory-copy time / kernel time; "
           f"BW util. = effective transfer BW as \\% of "
           f"{H100['peak_bw']:.0f}\\,GB/s peak. "
           f"PA model, 1\\% annual growth rate.")
    with open(tex_path, "w") as f:
        f.write(latex_header(cap, f"{prefix}-nsys-session",
                             "lrrrrrrrr"))
        f.write("\\textbf{Sim} & \\textbf{Session (s)} & "
                "\\textbf{Kernel (s)} & \\textbf{GPU util.\\%} & "
                "\\textbf{Memcpy ovh.\\%} & \\textbf{Data (MB)} & "
                "\\textbf{Eff.BW (GB/s)} & \\textbf{BW\\%} & "
                "\\textbf{Launches} \\\\\n\\midrule\n")
        for years, pct, ns in nsys_runs:
            f.write(f"{years}/{pct}\\% & "
                    f"{ns['session_ms']/1e3:.1f} & "
                    f"{ns['kernel_ms']/1e3:.2f} & "
                    f"{ns['gpu_util_pct']:.1f} & "
                    f"{ns['memcpy_ovh_pct']:.1f} & "
                    f"{ns['total_bytes_MB']:.1f} & "
                    f"{ns['eff_bw_GBs']:.2f} & "
                    f"{ns['bw_util_pct']:.3f} & "
                    f"{ns['total_kern_launches']:,} \\\\\n")
        f.write(latex_footer())
    print(f"  Wrote {tex_path}")

# ── FIGURES ───────────────────────────────────────────────────────────────────
COLORS = {
    "Stage1":       "#185FA5",
    "Stage2":       "#1D9E75",
    "Stage3":       "#7F77DD",
    "ABMStage2":    "#D85A30",
    "ABMStage4":    "#EF9F27",
    "SameYearCite": "#D4537E",
    "UniformCite":  "#854F0B",
    "2HopWarp":     "#5DCAA5",
    "2Hop":         "#9FE1CB",
    "ExtractSizes": "#AFA9EC",
}
DEF_CLR = "#888780"

def save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(outdir, f"{name}.{ext}")
        fig.savefig(p, dpi=300 if ext=="pdf" else 150, bbox_inches="tight")
        print(f"  Wrote {p}")
    plt.close(fig)

def fig_occupancy(runs, outdir, prefix):
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(4.5*n, 4.2), sharey=True)
    if n == 1: axes = [axes]
    for ax, (lbl, _, kdata) in zip(axes, runs):
        kns = sorted(kdata, key=lambda x: -kdata[x].get("achieved_occ_pct",0))[:8]
        th  = [kdata[k].get("th_occ_pct",0) for k in kns]
        ac  = [kdata[k].get("achieved_occ_pct",0) for k in kns]
        lim_c = ["#D85A30" if occ_limiter(kdata[k])=="reg" else "#1D9E75"
                 for k in kns]
        y = np.arange(len(kns))
        ax.barh(y-.2, th, .36, color="#B5D4F4", label="Theoretical",
                edgecolor="white", lw=.3)
        ax.barh(y+.2, ac, .36, color=lim_c,   label="Achieved",
                edgecolor="white", lw=.3)
        ax.set_yticks(y)
        ax.set_yticklabels([k[:22] for k in kns], fontsize=8)
        ax.set_xlabel("Occupancy (%)", fontsize=9)
        ax.set_title(lbl, fontsize=10, fontweight="500")
        ax.set_xlim(0, 115)
        ax.axvline(100, color="gray", lw=.6, ls="--", alpha=.4)
        ax.grid(axis="x", lw=.4, ls="--", alpha=.4)
        for i,(t,a) in enumerate(zip(th,ac)):
            if t>1: ax.text(t+1,i-.2,f"{t:.0f}%",va="center",fontsize=6.5)
            if a>1: ax.text(a+1,i+.2,f"{a:.0f}%",va="center",fontsize=6.5)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"Occupancy — {prefix} (red=register-limited, green=thread-limited)",
                 fontsize=10)
    plt.tight_layout()
    save(fig, outdir, f"{prefix}_occupancy_plot")

def fig_roofline(runs, outdir, prefix):
    fig, ax = plt.subplots(figsize=(8, 5))
    ai_r = np.logspace(-1, 2.5, 500)
    ceil = np.minimum(ai_r * H100["peak_bw"], H100["peak_fp32"])
    ax.loglog(ai_r, ceil, color="#185FA5", lw=2.0,
              label=f"{H100['name']}\n{H100['peak_fp32']/1000:.0f} TFLOP/s  "
                    f"|  {H100['peak_bw']:.0f} GB/s")
    ax.axvline(RIDGE, color="#185FA5", lw=.8, ls="--", alpha=.35)
    ax.text(RIDGE*1.05, H100["peak_fp32"]*.55,
            f"Ridge\n{RIDGE:.1f} F/B", fontsize=8, color="#185FA5")

    mkr = ["o","s","^"]
    for idx, (lbl, _, kdata) in enumerate(runs):
        for kn, m in kdata.items():
            r = roofline(m)
            if r:
                clr = COLORS.get(kn, DEF_CLR)
                ax.scatter(r["ai"], r["perf"], color=clr,
                           marker=mkr[idx%3], s=70, zorder=5,
                           edgecolors="white", lw=.5)
                ax.annotate(f"{kn}\n({lbl})", (r["ai"], r["perf"]),
                            xytext=(5,3), textcoords="offset points", fontsize=6.5)

    ax.set_xlabel("Arithmetic intensity (FLOP/byte)", fontsize=10)
    ax.set_ylabel("Performance (GFLOP/s)", fontsize=10)
    ax.set_title(f"Roofline model — {prefix}", fontsize=11)
    ax.grid(True, which="both", ls="--", lw=.4, alpha=.5)
    ax.legend(fontsize=8, loc="upper left")
    save(fig, outdir, f"{prefix}_roofline_plot")

def fig_kernel_time(nsys_runs, outdir, prefix):
    if not nsys_runs: return
    fig, axes = plt.subplots(1, len(nsys_runs),
                             figsize=(4.5*len(nsys_runs), 4.5))
    if len(nsys_runs)==1: axes=[axes]

    for ax, (years, pct, ns) in zip(axes, nsys_runs):
        kd = ns["kernels"]
        kns = sorted(kd, key=lambda x: -kd[x]["total_ns"])[:8]
        tot = sum(kd[k]["total_ns"] for k in kns)
        fracs = [kd[k]["total_ns"]/tot*100 if tot>0 else 0 for k in kns]
        clrs  = [COLORS.get(k, DEF_CLR) for k in kns]
        bars  = ax.bar(range(len(kns)), fracs, color=clrs,
                       edgecolor="white", lw=.3)
        ax.set_xticks(range(len(kns)))
        ax.set_xticklabels([k[:14] for k in kns],
                           rotation=35, ha="right", fontsize=7.5)
        ax.set_ylabel("% of GPU kernel time")
        ax.set_title(f"{years}/{pct}%", fontsize=10, fontweight="500")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", lw=.4, ls="--", alpha=.5)
        for bar, v in zip(bars, fracs):
            if v>2:
                ax.text(bar.get_x()+bar.get_width()/2, v+.8,
                        f"{v:.0f}%", ha="center", fontsize=7, fontweight="500")

    fig.suptitle(f"Kernel time breakdown — {prefix}", fontsize=11)
    plt.tight_layout()
    save(fig, outdir, f"{prefix}_kernel_time_plot")

def fig_scaling(nsys_runs, outdir, prefix):
    if len(nsys_runs) < 2: return
    labels_x  = [f"{y}y" for y,_,_ in nsys_runs]
    sess_s    = [ns["session_ms"]/1e3 for _,_,ns in nsys_runs]
    kern_s    = [ns["kernel_ms"]/1e3  for _,_,ns in nsys_runs]
    util_pct  = [ns["gpu_util_pct"]   for _,_,ns in nsys_runs]
    ovh_pct   = [ns["memcpy_ovh_pct"] for _,_,ns in nsys_runs]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(labels_x))
    w = .35
    ax = axes[0]
    ax.bar(x-w/2, sess_s, w, color="#B5D4F4", edgecolor="#185FA5", lw=.5,
           label="Session total")
    ax.bar(x+w/2, kern_s, w, color="#185FA5", edgecolor="#0C447C", lw=.5,
           label="Kernel active")
    ax.set_xticks(x); ax.set_xticklabels(labels_x)
    ax.set_ylabel("Time (s)"); ax.set_title("Session vs kernel time")
    ax.legend(fontsize=9); ax.grid(axis="y", lw=.4, ls="--", alpha=.5)
    for i,(s,k) in enumerate(zip(sess_s,kern_s)):
        ax.text(i-w/2, s+s*.01, f"{s:.0f}s", ha="center", fontsize=8)
        ax.text(i+w/2, k+k*.01, f"{k:.0f}s", ha="center", fontsize=8)

    ax2 = axes[1]
    ax2.bar(x-w/2, util_pct, w, color="#378ADD", edgecolor="#185FA5", lw=.5,
            label="GPU util %")
    ax2.bar(x+w/2, ovh_pct,  w, color="#D85A30", edgecolor="#993C1D", lw=.5,
            label="Memcpy ovh %")
    ax2.set_xticks(x); ax2.set_xticklabels(labels_x)
    ax2.set_ylabel("Percentage (%)"); ax2.set_title("GPU utilization & memcpy overhead")
    ax2.legend(fontsize=9); ax2.grid(axis="y", lw=.4, ls="--", alpha=.5)
    for i,(u,o) in enumerate(zip(util_pct,ovh_pct)):
        ax2.text(i-w/2, u+.3, f"{u:.1f}%", ha="center", fontsize=8)
        ax2.text(i+w/2, o+.3, f"{o:.1f}%", ha="center", fontsize=8)

    fig.suptitle(f"Scaling across simulation lengths — {prefix}, 1% growth",
                 fontsize=11)
    plt.tight_layout()
    save(fig, outdir, f"{prefix}_scaling_plot")

# ── CLI ───────────────────────────────────────────────────────────────────────
class _Multi(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest, None) or []
        items.append(values)
        setattr(namespace, self.dest, items)

def main():
    p = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ncu",  dest="ncu_files",  nargs=3,
                   metavar=("CSV","YEARS","GROWTH%"),
                   action=_Multi,
                   help="ncu CSV file with years and growth%")
    p.add_argument("--nsys", dest="nsys_files", nargs=3,
                   metavar=("SQLITE","YEARS","GROWTH%"),
                   action=_Multi,
                   help="nsys sqlite file with years and growth%")
    p.add_argument("--outdir", default="./paper_output")
    p.add_argument("--prefix", default="pa")
    p.add_argument("--no-plot",  action="store_true")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Parse ncu
    ncu_runs = []
    for csv_path, years, pct in (args.ncu_files or []):
        if not os.path.exists(csv_path):
            print(f"  MISSING ncu CSV: {csv_path}")
            continue
        print(f"  Parsing ncu [{years}/{pct}%]  {csv_path}")
        kdata = parse_ncu_csv(csv_path)
        if kdata:
            ncu_runs.append((f"{years}", pct, kdata))

    # Parse nsys
    nsys_runs = []
    for db_path, years, pct in (args.nsys_files or []):
        if not os.path.exists(db_path):
            print(f"  MISSING nsys sqlite: {db_path}")
            continue
        print(f"  Parsing nsys [{years}/{pct}%]  {db_path}")
        ns = parse_nsys_sqlite(db_path)
        nsys_runs.append((years, pct, ns))

    if not ncu_runs and not nsys_runs:
        print("No data. Provide --ncu or --nsys files.")
        print("On the cluster, first run:")
        print("  ncu --import profile/ncu-pa-3y-1p.ncu-rep --csv --page details \\")
        print("      > profile/ncu-pa-3y-1p-details.csv")
        return

    print(f"\nWriting tables → {args.outdir}/")
    if ncu_runs:
        make_ncu_resource_table(ncu_runs, args.outdir, args.prefix)
        make_ncu_launch_table(ncu_runs, args.outdir, args.prefix)
    if nsys_runs:
        make_nsys_session_table(nsys_runs, args.outdir, args.prefix)

    if not args.no_plot and HAS_PLOT:
        print("\nGenerating figures...")
        if ncu_runs:
            fig_occupancy(ncu_runs, args.outdir, args.prefix)
            fig_roofline(ncu_runs,  args.outdir, args.prefix)
        if nsys_runs:
            fig_kernel_time(nsys_runs, args.outdir, args.prefix)
            fig_scaling(nsys_runs,    args.outdir, args.prefix)

    print("\nDone.")

if __name__ == "__main__":
    main()
