#pragma once
// =============================================================================
//  abm_cpu_profiler.h
//  SC-style runtime/performance profiling for ABM::main()
//  Measures single-threaded and multi-threaded distributions at every
//  granularity described in SC performance papers:
//    - Initialization (graph load, mapping, array alloc/populate)
//    - Per-epoch: score computation, node init, generator assign,
//                 citation loop (PA: BFS+score+sample; ER: Bernoulli/sample),
//                 edge commit, fitness assign
//    - Per-node:  BFS neighborhood, MakeCitations, MakeUniformRandom,
//                 MakeER, SameYear
//    - Output write
//    - E2E
//
//  Usage:
//    1. #include "abm_cpu_profiler.h" at top of abm_main.cpp
//    2. Follow // PROF: inline instructions
//    3. Call ABMProfiler::print_sc_report() at end of ABM::main()
// =============================================================================

#include <chrono>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <omp.h>

using HRC  = std::chrono::high_resolution_clock;
using HRTp = std::chrono::time_point<HRC>;

// ─────────────────────────────────────────────────────────────────────────────
//  Wall-clock timer (ms, double precision)
// ─────────────────────────────────────────────────────────────────────────────
struct WTimer {
    HRTp _t0;
    void   start()    { _t0 = HRC::now(); }
    double stop_ms()  {
        return std::chrono::duration<double,std::milli>(HRC::now()-_t0).count();
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Per-node sample  (collected inside the OMP parallel loop, thread-local)
// ─────────────────────────────────────────────────────────────────────────────
struct NodeSample {
    int    node_id       = 0;
    int    thread_id     = 0;
    int    year          = 0;
    bool   is_pa         = true;
    double t_bfs_ms      = 0;   // GetOneAndTwoHopNeighborhood
    double t_score_ms    = 0;   // MakeCitations (scoring loop)
    double t_sample_ms   = 0;   // MakeCitations (heap sort + pick)
    double t_random_ms   = 0;   // MakeUniformRandomCitations / MakeERGNP
    double t_sameyear_ms = 0;   // MakeSameYearCitations
    double t_node_total_ms = 0; // sum of all above
    int    hop1_size     = 0;
    int    hop2_size     = 0;
    int    num_cited     = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Per-epoch breakdown
// ─────────────────────────────────────────────────────────────────────────────
struct EpochProfile {
    int    year            = 0;
    int    N               = 0;    // graph size at epoch start
    int    delta           = 0;    // cohort size
    int    num_threads     = 1;

    // host-side per-epoch work
    double t_fill_indeg    = 0;
    double t_fill_fitness  = 0;
    double t_fill_recency  = 0;
    double t_calc_scores   = 0;    // CalculateScores x2
    double t_node_init     = 0;    // new-node loop + mappings
    double t_same_year     = 0;    // FillSameYearSourceNodes
    double t_gen_assign    = 0;    // GetGeneratorNodes loop
    double t_citation_loop = 0;   // entire #pragma omp parallel for (wall)
    double t_edge_commit   = 0;   // graph->AddEdge loop
    double t_fitness_assign= 0;   // AssignPeak + Lag + Peak + PlantNodes
    double t_epoch_total   = 0;

    // per-node distribution (filled from NodeSample vector)
    std::vector<NodeSample> node_samples;

    // derived
    double cpu_preproc_ms() const {
        return t_fill_indeg + t_fill_fitness + t_fill_recency + t_calc_scores;
    }
    double epoch_total_computed() const {
        return cpu_preproc_ms() + t_node_init + t_same_year + t_gen_assign
             + t_citation_loop + t_edge_commit + t_fitness_assign;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Initialization profile
// ─────────────────────────────────────────────────────────────────────────────
struct InitProfile {
    double t_graph_load     = 0;
    double t_fitness_init   = 0;
    double t_mapping_build  = 0;
    double t_array_alloc    = 0;
    double t_weight_pop     = 0;
    double total() const {
        return t_graph_load + t_fitness_init + t_mapping_build
             + t_array_alloc + t_weight_pop;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Output profile
// ─────────────────────────────────────────────────────────────────────────────
struct OutputProfile {
    double t_write_graph    = 0;
    double t_write_attrs    = 0;
    double t_update_attrs   = 0;
    double total() const {
        return t_write_graph + t_write_attrs + t_update_attrs;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Statistics helpers
// ─────────────────────────────────────────────────────────────────────────────
struct Stats {
    double min=0, p25=0, median=0, p75=0, p95=0, p99=0, max=0, mean=0, stddev=0;
    int    count=0;
};

inline Stats compute_stats(std::vector<double> v) {
    Stats s;
    s.count = (int)v.size();
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.min    = v.front();
    s.max    = v.back();
    s.mean   = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    auto pct = [&](double p) -> double {
        double idx = p * (v.size() - 1);
        int lo = (int)idx;
        int hi = std::min(lo+1, (int)v.size()-1);
        return v[lo] + (idx - lo) * (v[hi] - v[lo]);
    };
    s.p25    = pct(0.25);
    s.median = pct(0.50);
    s.p75    = pct(0.75);
    s.p95    = pct(0.95);
    s.p99    = pct(0.99);
    double var = 0;
    for (auto x : v) var += (x - s.mean) * (x - s.mean);
    s.stddev = std::sqrt(var / v.size());
    return s;
}

inline void print_stats_row(const char* label, const Stats& s) {
    printf("  %-30s n=%-7d  min=%7.1f  p25=%7.1f  med=%7.1f  p75=%7.1f  p95=%7.1f  p99=%7.1f  max=%8.1f  mean=%7.1f  std=%7.1f ms\n",
           label, s.count,
           s.min, s.p25, s.median, s.p75, s.p95, s.p99, s.max, s.mean, s.stddev);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Global profiler singleton
// ─────────────────────────────────────────────────────────────────────────────
class ABMProfiler {
public:
    static ABMProfiler& get() {
        static ABMProfiler inst;
        return inst;
    }

    InitProfile              init;
    OutputProfile            output;
    std::vector<EpochProfile> epochs;
    double                   t_e2e_ms = 0;
    int                      num_threads = 1;
    std::string              model_name;
    double                   growth_rate = 0;

    EpochProfile& cur_epoch() { return epochs.back(); }

    void begin_epoch(int year, int N, int delta, int nthreads) {
        EpochProfile ep;
        ep.year = year; ep.N = N; ep.delta = delta; ep.num_threads = nthreads;
        epochs.push_back(ep);
    }

    // ── SC-style report ───────────────────────────────────────────────────────
    static void print_sc_report(const ABMProfiler& p) {
        auto bar  = [](int w=100){ for(int i=0;i<w;i++) putchar('='); putchar('\n'); };
        auto bar2 = [](int w=100){ for(int i=0;i<w;i++) putchar('-'); putchar('\n'); };

        bar();
        printf("  ABM CPU PERFORMANCE PROFILE  —  model=%s  growth=%.1f%%  threads=%d\n",
               p.model_name.c_str(), p.growth_rate*100.0, p.num_threads);
        bar();

        // ── Section 1: Initialization ─────────────────────────────────────────
        printf("\n[1] INITIALIZATION\n");
        bar2();
        printf("  %-30s : %8.1f ms\n", "Graph load",        p.init.t_graph_load);
        printf("  %-30s : %8.1f ms\n", "Fitness init",      p.init.t_fitness_init);
        printf("  %-30s : %8.1f ms\n", "Mapping build",     p.init.t_mapping_build);
        printf("  %-30s : %8.1f ms\n", "Array allocation",  p.init.t_array_alloc);
        printf("  %-30s : %8.1f ms\n", "Weight/alpha/deg",  p.init.t_weight_pop);
        printf("  %-30s : %8.1f ms\n", "TOTAL INIT",        p.init.total());

        // ── Section 2: Per-epoch table ────────────────────────────────────────
        printf("\n[2] PER-EPOCH BREAKDOWN  (ms)\n");
        bar2(160);
        printf("%-4s %-7s %-6s %-5s | %-7s %-7s %-7s %-7s | %-7s %-7s %-7s | %-9s %-8s %-8s | %-8s\n",
               "Ep","N","Delta","Thr",
               "FillID","FillFit","FillRec","CalcScr",
               "NodeInit","SameYr","GenAsgn",
               "CitLoop","EdgeCmt","FitAsgn",
               "EpTotal");
        bar2(160);
        double tot_preproc=0, tot_cit=0, tot_edge=0, tot_fit=0, tot_ep=0;
        for (const auto& ep : p.epochs) {
            printf("%-4d %-7d %-6d %-5d | %-7.1f %-7.1f %-7.1f %-7.1f | %-7.1f %-7.1f %-7.1f | %-9.1f %-8.1f %-8.1f | %-8.1f\n",
                   ep.year, ep.N, ep.delta, ep.num_threads,
                   ep.t_fill_indeg, ep.t_fill_fitness, ep.t_fill_recency, ep.t_calc_scores,
                   ep.t_node_init, ep.t_same_year, ep.t_gen_assign,
                   ep.t_citation_loop, ep.t_edge_commit, ep.t_fitness_assign,
                   ep.epoch_total_computed());
            tot_preproc += ep.cpu_preproc_ms();
            tot_cit     += ep.t_citation_loop;
            tot_edge    += ep.t_edge_commit;
            tot_fit     += ep.t_fitness_assign;
            tot_ep      += ep.epoch_total_computed();
        }
        bar2(160);
        double sim_total = tot_ep;
        auto pct = [&](double v){ return sim_total>0 ? 100.0*v/sim_total : 0.0; };
        printf("\n  TOTALS:\n");
        printf("  %-30s : %10.1f ms  (%5.1f%% of sim)\n","CPU preprocessing",  tot_preproc, pct(tot_preproc));
        printf("  %-30s : %10.1f ms  (%5.1f%%)\n",       "Citation loop",      tot_cit,     pct(tot_cit));
        printf("  %-30s : %10.1f ms  (%5.1f%%)\n",       "Edge commit",        tot_edge,    pct(tot_edge));
        printf("  %-30s : %10.1f ms  (%5.1f%%)\n",       "Fitness assign",     tot_fit,     pct(tot_fit));
        printf("  %-30s : %10.1f ms\n",                   "Simulation total",  sim_total);
        printf("  %-30s : %10.1f ms\n",                   "Output write",      p.output.total());
        printf("  %-30s : %10.1f ms\n",                   "E2E total",         p.t_e2e_ms);

        // ── Section 3: Per-node distributions ────────────────────────────────
        printf("\n[3] PER-NODE DISTRIBUTIONS  (ms)  — across ALL epochs\n");
        bar2(160);

        // Collect all node samples
        std::vector<double> v_bfs, v_score, v_sample, v_random, v_sameyear, v_total;
        std::vector<double> v_hop1, v_hop2, v_cited;
        // PA vs ER split
        std::vector<double> v_pa_total, v_er_total;

        for (const auto& ep : p.epochs) {
            for (const auto& ns : ep.node_samples) {
                v_bfs.push_back(ns.t_bfs_ms);
                v_score.push_back(ns.t_score_ms);
                v_sample.push_back(ns.t_sample_ms);
                v_random.push_back(ns.t_random_ms);
                v_sameyear.push_back(ns.t_sameyear_ms);
                v_total.push_back(ns.t_node_total_ms);
                v_hop1.push_back(ns.hop1_size);
                v_hop2.push_back(ns.hop2_size);
                v_cited.push_back(ns.num_cited);
                if (ns.is_pa) v_pa_total.push_back(ns.t_node_total_ms);
                else          v_er_total.push_back(ns.t_node_total_ms);
            }
        }

        printf("\n  [3a] PA kernel breakdown per node:\n");
        print_stats_row("BFS (1+2-hop neighborhood)", compute_stats(v_bfs));
        print_stats_row("Scoring (MakeCitations)",    compute_stats(v_score));
        print_stats_row("Heap sort + select",         compute_stats(v_sample));
        print_stats_row("Uniform random citations",   compute_stats(v_random));
        print_stats_row("Same-year citations",        compute_stats(v_sameyear));
        print_stats_row("Node total (PA)",            compute_stats(v_pa_total));

        if (!v_er_total.empty()) {
            printf("\n  [3b] ER kernel per node:\n");
            print_stats_row("ER citation sampling",  compute_stats(v_er_total));
        }

        printf("\n  [3c] Graph structure per node:\n");
        print_stats_row("1-hop neighborhood size",   compute_stats(v_hop1));
        print_stats_row("2-hop neighborhood size",   compute_stats(v_hop2));
        print_stats_row("Num citations made",        compute_stats(v_cited));

        // ── Section 4: Per-epoch node distributions ───────────────────────────
        printf("\n[4] PER-EPOCH NODE-LEVEL DISTRIBUTIONS  (node total ms)\n");
        bar2(160);
        printf("%-4s %-7s %-6s | %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s\n",
               "Ep","N","Delta","mean","median","p25","p75","p95","p99","min","max");
        bar2(160);
        for (const auto& ep : p.epochs) {
            std::vector<double> vt;
            for (const auto& ns : ep.node_samples)
                vt.push_back(ns.t_node_total_ms);
            if (vt.empty()) { printf("%-4d  (no samples)\n", ep.year); continue; }
            auto s = compute_stats(vt);
            printf("%-4d %-7d %-6d | %-8.2f %-8.2f %-8.2f %-8.2f %-8.2f %-8.2f %-8.2f %-8.2f\n",
                   ep.year, ep.N, ep.delta,
                   s.mean, s.median, s.p25, s.p75, s.p95, s.p99, s.min, s.max);
        }

        // ── Section 5: Thread utilization (multi-threaded runs only) ─────────
        if (p.num_threads > 1) {
            printf("\n[5] THREAD UTILIZATION  (node work distribution across threads)\n");
            bar2(160);
            // collect per-thread totals per epoch
            printf("%-4s | ", "Ep");
            for (int t = 0; t < p.num_threads; t++) printf("T%-6d ", t);
            printf("  Imbalance%%\n");
            bar2(160);
            for (const auto& ep : p.epochs) {
                std::vector<double> thread_work(p.num_threads, 0.0);
                for (const auto& ns : ep.node_samples) {
                    if (ns.thread_id < p.num_threads)
                        thread_work[ns.thread_id] += ns.t_node_total_ms;
                }
                double tw_mean = std::accumulate(thread_work.begin(), thread_work.end(), 0.0) / p.num_threads;
                double tw_max  = *std::max_element(thread_work.begin(), thread_work.end());
                double imbalance = (tw_mean > 0) ? 100.0 * (tw_max - tw_mean) / tw_max : 0.0;
                printf("%-4d | ", ep.year);
                for (int t = 0; t < p.num_threads; t++)
                    printf("%-7.0f ", thread_work[t]);
                printf("  %.1f%%\n", imbalance);
            }
        }

        // ── Section 6: Speedup summary (printed if threads > 1) ──────────────
        printf("\n[6] SUMMARY\n");
        bar2();
        printf("  Model              : %s\n",   p.model_name.c_str());
        printf("  Growth rate        : %.1f%%\n", p.growth_rate * 100.0);
        printf("  Threads            : %d\n",   p.num_threads);
        printf("  Epochs             : %d\n",   (int)p.epochs.size());
        printf("  Total nodes added  : %d\n",
               [&]{ int s=0; for(auto& e:p.epochs) s+=e.delta; return s; }());
        long long te = 0;
        for (const auto& ep : p.epochs)
            for (const auto& ns : ep.node_samples) te += ns.num_cited;
        printf("  Total edges added  : %lld\n", te);
        printf("  Simulation time    : %.1f ms  (%.1f s)\n",
               sim_total, sim_total/1000.0);
        printf("  E2E time           : %.1f ms  (%.1f s)\n",
               p.t_e2e_ms, p.t_e2e_ms/1000.0);
        printf("  Citation loop %%    : %.1f%% of sim\n", pct(tot_cit));
        printf("  Throughput         : %.0f nodes/s  %.0f edges/s\n",
               (sim_total>0) ? (1000.0*[&]{int s=0;for(auto&e:p.epochs)s+=e.delta;return s;}()/sim_total) : 0.0,
               (sim_total>0) ? (1000.0*te/sim_total) : 0.0);
        bar();
    }

private:
    ABMProfiler() = default;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Convenience macros
// ─────────────────────────────────────────────────────────────────────────────
#define ABMPROF       ABMProfiler::get()
#define ABMPROF_EP    ABMPROF.cur_epoch()

// CPU block timer:  CTIME(field, code)
#define CTIME(field, code) \
    do { WTimer _wt; _wt.start(); code; ABMPROF_EP.field += _wt.stop_ms(); } while(0)

// Init block timer:  ITIME(field, code)
#define ITIME(field, code) \
    do { WTimer _wt; _wt.start(); code; ABMPROF.init.field += _wt.stop_ms(); } while(0)

// Output block timer:  OTIME(field, code)
#define OTIME(field, code) do { WTimer _wt; _wt.start(); code; ABMPROF.output.field += _wt.stop_ms(); } while(0)