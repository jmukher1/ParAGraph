#pragma once
// ============================================================================
//  epoch_profiler.h
//  Drop-in epoch-level timing for ParAGraph kernel.cu
//  Captures per-stage GPU kernel times + host-side costs per epoch
//  with running stats (min/mean/max/total) across all epochs.
//
//  Usage:
//    1. #include "epoch_profiler.h" at top of kernel.cu
//    2. Replace execute() with execute() from execute_profiled.cu
//    3. All __global__ kernels are UNCHANGED
// ============================================================================

#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <string>
#include <cstdio>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <limits>

// ─────────────────────────────────────────────────────────────────────────────
//  Host wall-clock timer (ms)
// ─────────────────────────────────────────────────────────────────────────────
struct HostTimer {
    using Clock = std::chrono::high_resolution_clock;
    using TP    = std::chrono::time_point<Clock>;
    TP _t0;
    void   start()   { _t0 = Clock::now(); }
    double stop_ms() {
        return std::chrono::duration<double,std::milli>(Clock::now()-_t0).count();
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  GPU event timer (ms, stream-aware)
// ─────────────────────────────────────────────────────────────────────────────
struct GpuTimer {
    cudaEvent_t _s, _e;
    GpuTimer()  { cudaEventCreate(&_s); cudaEventCreate(&_e); }
    ~GpuTimer() { cudaEventDestroy(_s); cudaEventDestroy(_e); }
    void  start(cudaStream_t st=0) { cudaEventRecord(_s,st); }
    float stop (cudaStream_t st=0) {
        cudaEventRecord(_e,st);
        cudaEventSynchronize(_e);
        float ms=0; cudaEventElapsedTime(&ms,_s,_e);
        return ms;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Per-epoch record  (all times in ms)
// ─────────────────────────────────────────────────────────────────────────────
struct EpochTiming {
    int    year          = 0;
    int    N             = 0;       // graph size at epoch start
    int    delta         = 0;       // num new nodes
    int    batch_size    = 0;       // B (first mini-batch)
    int    num_minibatch = 0;       // mini-batch count this epoch
    long long edges_out  = 0;       // edges generated this epoch

    // ── host-side pre-processing ─────────────────────────────────────────────
    double t_fill_indeg   = 0;      // FillInDegreeArr
    double t_fill_fitness = 0;      // FillFitnessArr
    double t_fill_recency = 0;      // FillRecencyArr
    double t_calc_scores  = 0;      // CalculateScores x2
    double t_node_init    = 0;      // new-node loop + mapping
    double t_gen_assign   = 0;      // getGeneratorNode loop
    double t_same_year    = 0;      // FillSameYearSourceNodes

    // ── CSR / graph upload ───────────────────────────────────────────────────
    double t_csr_build    = 0;      // prepareGraph fwd+bwd + nodeAttrMap
    float  t_upload       = 0;      // cudaMemcpyAsync scores+nodes→device

    // ── GPU kernels (accumulated across mini-batches) ────────────────────────
    float  t_slab_alloc   = 0;      // cudaMalloc+Memset for BFS slabs
    float  t_stage1_bfs   = 0;      // kernelCallStage1
    float  t_stage2_ws1   = 0;      // kernelCallStage2
    float  t_stage3_ws2   = 0;      // kernelCallStage3
    float  t_stage4_fill  = 0;      // kernelCallStage4

    // ── download + host graph update ─────────────────────────────────────────
    float  t_download     = 0;      // append_device_to_host
    double t_edge_insert  = 0;      // batch adj-map insertion
    double t_indeg_update = 0;      // updateNodeInDegreeOutDegree
    double t_fitness_asgn = 0;      // AssignPeak+Lag+Peak+PlantNodes

    // ── derived ──────────────────────────────────────────────────────────────
    double host_preproc()  const { return t_fill_indeg+t_fill_fitness+t_fill_recency+t_calc_scores; }
    double host_init()     const { return t_node_init+t_gen_assign+t_same_year; }
    float  gpu_kernels()   const { return t_stage1_bfs+t_stage2_ws1+t_stage3_ws2+t_stage4_fill; }
    float  transfers()     const { return t_upload+t_download; }
    double host_update()   const { return t_edge_insert+t_indeg_update+t_fitness_asgn; }
    double epoch_total()   const {
        return host_preproc()+host_init()+t_csr_build+transfers()+t_slab_alloc+gpu_kernels()+host_update();
    }
    double edges_per_sec() const {
        double s = gpu_kernels()/1000.0;
        return (s>0&&edges_out>0) ? edges_out/s : 0.0;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Running statistics helper
// ─────────────────────────────────────────────────────────────────────────────
struct RunStats {
    double total=0, mn=1e18, mx=-1e18, sum2=0;
    int    n=0;
    void   add(double v) {
        total+=v; mn=std::min(mn,v); mx=std::max(mx,v); sum2+=v*v; n++;
    }
    double mean()   const { return n>0?total/n:0; }
    double stddev() const {
        if(n<2) return 0;
        double m=mean(); return std::sqrt(sum2/n - m*m);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Report printer
// ─────────────────────────────────────────────────────────────────────────────
inline void print_epoch_report(const std::vector<EpochTiming>& epochs,
                                double e2e_ms, int num_threads,
                                const std::string& model, double growth_rate)
{
    auto bar  = [](int w=120){ for(int i=0;i<w;i++) putchar('='); putchar('\n'); };
    auto bar2 = [](int w=120){ for(int i=0;i<w;i++) putchar('-'); putchar('\n'); };

    bar();
    printf("  PARAGRAH GPU EPOCH BREAKDOWN  model=%s  growth=%.1f%%  threads=%d\n",
           model.c_str(), growth_rate*100.0, num_threads);
    bar();

    // ── Section 1: per-epoch table ────────────────────────────────────────────
    printf("\n[1] PER-EPOCH STAGE BREAKDOWN  (ms)\n");
    bar2();
    printf("%-4s %-8s %-6s %-5s %-4s | "
           "%-7s %-7s %-7s %-7s | "
           "%-7s %-7s %-7s | "
           "%-7s %-6s | "
           "%-5s %-7s %-7s %-7s %-7s | "
           "%-6s %-8s %-7s %-7s | "
           "%-8s %-10s\n",
        "Ep","N","Delta","B","mB",
        "FillID","FillFit","FillRec","CalcScr",
        "NodeInit","GenAsgn","SameYr",
        "CSR","Upload",
        "Slab","S1_BFS","S2_WS1","S3_WS2","S4_Fill",
        "Dwnld","EdgeIns","InDeg","FitAsgn",
        "EpTotal","Edges/s");
    bar2(300);

    RunStats rs_pre, rs_s1, rs_s2, rs_s3, rs_s4, rs_gpu, rs_xfer,
             rs_upd, rs_ep, rs_csr, rs_slab;

    for (const auto& e : epochs) {
        printf("%-4d %-8d %-6d %-5d %-4d | "
               "%-7.1f %-7.1f %-7.1f %-7.1f | "
               "%-7.1f %-7.1f %-7.1f | "
               "%-7.1f %-6.1f | "
               "%-5.1f %-7.1f %-7.1f %-7.1f %-7.1f | "
               "%-6.1f %-8.1f %-7.1f %-7.1f | "
               "%-8.0f %-10.0f\n",
            e.year, e.N, e.delta, e.batch_size, e.num_minibatch,
            e.t_fill_indeg, e.t_fill_fitness, e.t_fill_recency, e.t_calc_scores,
            e.t_node_init, e.t_gen_assign, e.t_same_year,
            e.t_csr_build, (double)e.t_upload,
            (double)e.t_slab_alloc,
            (double)e.t_stage1_bfs, (double)e.t_stage2_ws1,
            (double)e.t_stage3_ws2, (double)e.t_stage4_fill,
            (double)e.t_download,
            e.t_edge_insert, e.t_indeg_update, e.t_fitness_asgn,
            e.epoch_total(), e.edges_per_sec());

        rs_pre.add(e.host_preproc());
        rs_s1.add(e.t_stage1_bfs);
        rs_s2.add(e.t_stage2_ws1);
        rs_s3.add(e.t_stage3_ws2);
        rs_s4.add(e.t_stage4_fill);
        rs_gpu.add(e.gpu_kernels());
        rs_xfer.add(e.transfers());
        rs_upd.add(e.host_update());
        rs_ep.add(e.epoch_total());
        rs_csr.add(e.t_csr_build);
        rs_slab.add(e.t_slab_alloc);
    }
    bar2(300);

    // ── Section 2: stats per component ───────────────────────────────────────
    printf("\n[2] PER-STAGE STATISTICS ACROSS ALL EPOCHS  (ms)\n");
    bar2();
    auto pr = [](const char* lbl, const RunStats& s){
        printf("  %-28s  total=%10.0f  mean=%8.1f  std=%7.1f  min=%7.1f  max=%8.1f\n",
               lbl, s.total, s.mean(), s.stddev(), s.mn, s.mx);
    };
    pr("Host preprocessing",   rs_pre);
    pr("CSR build+upload",     rs_csr);
    pr("Slab alloc+memset",    rs_slab);
    pr("Stage1 BFS",           rs_s1);
    pr("Stage2 WS1 (1-hop)",   rs_s2);
    pr("Stage3 WS2 (2-hop)",   rs_s3);
    pr("Stage4 fill+write",    rs_s4);
    pr("GPU kernels (total)",  rs_gpu);
    pr("Transfers (up+down)",  rs_xfer);
    pr("Host graph update",    rs_upd);
    pr("Epoch total",          rs_ep);

    // ── Section 3: percentage breakdown ──────────────────────────────────────
    printf("\n[3] TOTAL TIME DISTRIBUTION\n");
    bar2();
    double tot = rs_ep.total;
    auto pct = [&](double v){ return tot>0 ? 100.0*v/tot : 0.0; };
    printf("  %-28s : %10.0f ms  (%5.1f%%)\n","Host preprocessing",rs_pre.total, pct(rs_pre.total));
    printf("  %-28s : %10.0f ms  (%5.1f%%)\n","CSR build",         rs_csr.total, pct(rs_csr.total));
    printf("  %-28s : %10.0f ms  (%5.1f%%)\n","BFS slab overhead", rs_slab.total,pct(rs_slab.total));
    printf("  %-28s : %10.0f ms  (%5.1f%%)\n","Stage1 BFS",        rs_s1.total,  pct(rs_s1.total));
    printf("  %-28s : %10.0f ms  (%5.1f%%)\n","Stage2+3 sampling", rs_s2.total+rs_s3.total,
           pct(rs_s2.total+rs_s3.total));
    printf("  %-28s : %10.0f ms  (%5.1f%%)\n","Stage4 fill+write", rs_s4.total,  pct(rs_s4.total));
    printf("  %-28s : %10.0f ms  (%5.1f%%)\n","GPU kernels total", rs_gpu.total, pct(rs_gpu.total));
    printf("  %-28s : %10.0f ms  (%5.1f%%)\n","Transfers",         rs_xfer.total,pct(rs_xfer.total));
    printf("  %-28s : %10.0f ms  (%5.1f%%)\n","Host graph update", rs_upd.total, pct(rs_upd.total));
    printf("  %-28s : %10.0f ms\n",            "Simulation total",  tot);
    printf("  %-28s : %10.0f ms\n",            "E2E total",         e2e_ms);

    // ── Section 4: BFS dominance check ───────────────────────────────────────
    printf("\n[4] BFS DOMINANCE\n");
    bar2();
    printf("  Stage1 BFS as %% of GPU kernels : %.1f%%\n",
           rs_gpu.total>0 ? 100.0*rs_s1.total/rs_gpu.total : 0.0);
    printf("  Stage1 BFS as %% of epoch total : %.1f%%\n", pct(rs_s1.total));
    printf("  GPU kernels as %% of epoch      : %.1f%%\n", pct(rs_gpu.total));

    long long tot_edges=0;
    for (const auto& e:epochs) tot_edges+=e.edges_out;
    printf("  Total edges generated          : %lld\n", tot_edges);
    printf("  Overall edges/sec (GPU time)   : %.0f\n",
           rs_gpu.total>0 ? 1000.0*tot_edges/rs_gpu.total : 0.0);
    bar();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Macros for clean call-site instrumentation
// ─────────────────────────────────────────────────────────────────────────────
#define HOST_TIME(field, code) \
    do { HostTimer _ht; _ht.start(); code; _ep.field += _ht.stop_ms(); } while(0)

#define GPU_TIME(field, stream, launch) \
    do { \
        GpuTimer _gt; _gt.start(stream); \
        launch; \
        CUDA_CHECK(cudaGetLastError()); \
        CUDA_CHECK(cudaStreamSynchronize(stream)); \
        _ep.field += _gt.stop(stream); \
    } while(0)

#define GPU_MEMOP_TIME(field, stream, op) \
    do { \
        GpuTimer _gt; _gt.start(stream); \
        op; \
        CUDA_CHECK(cudaStreamSynchronize(stream)); \
        _ep.field += _gt.stop(stream); \
    } while(0)
