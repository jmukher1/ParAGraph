#ifndef PAPER_H
#define PAPER_H

// ─────────────────────────────────────────────────────────────────────────────
// Paper.h  –  MASS CUDA Citation-Network ABM
//
// Mirrors the graph / ABM logic from:
//   graph.cu  → DeviceGraph (CSR adjacency list), forward+backward
//   abm.cu    → GetOneAndTwoHopNeighborhood (2-level BFS)
//               MakePopulateCitations (Efraimidis-Spirakis WRS)
//               ABMKernelStage2/3/4 pipeline
// ─────────────────────────────────────────────────────────────────────────────

#include "Mass.h"
#include "Agent.h"
using mass::Agent;
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>
#include <string>

// ═════════════════════════════════════════════════════════════════════════════
// DeviceGraph  –  CSR adjacency list on device
// ═════════════════════════════════════════════════════════════════════════════
struct DeviceGraph {
    int* offsets;    // [num_nodes + 1]  CSR row-pointer array
    int* edges;      // [num_edges]      CSR column-index array
    int  num_nodes;
    int  num_edges;
};

// ═════════════════════════════════════════════════════════════════════════════
// BFS buffer constants
// ═════════════════════════════════════════════════════════════════════════════
namespace BFSConst {
    // Open-addressing hash table for visited set.
    // HASH_SIZE must be a power of 2 for the mask trick.
    // 16384 entries → 64 KB per thread. Handles ~12K unique neighbours
    // before the load factor exceeds 0.75 — needed for dense seed graphs
    // where 1-hop + 2-hop sets easily exceed 10K nodes.
    static constexpr int HASH_SIZE     = 131072;
    static constexpr int HASH_MASK     = HASH_SIZE - 1;

    // BFS frontier: must hold the full level-1 set so level-2 expansion
    // sees all 1-hop nodes, not just the first 2048.
    // CPP uses per_thread_vector_capacity = current_graph_size (uncapped).
    // 8192 covers virtually all 1-hop neighborhoods in practice.
    static constexpr int MAX_FRONTIER  = 131072;

    // Output neighbourhood arrays fed into WRS.
    // CPP is uncapped; agents need to compete with seeds in WRS,
    // so these must be large enough that high-fitness agents added in
    // year t appear in subsequent years' BFS output rather than being
    // evicted by the first 1024 (mostly seed) nodes in CSR order.
    static constexpr int MAX_1HOP      = 65536;
    static constexpr int MAX_2HOP      = 131072;
}

// ═════════════════════════════════════════════════════════════════════════════
// NeighborhoodSlab  –  per-batch device-memory allocation
//
// One struct instance is created per batch by CitationABM before
// makeCitationsWithBFSKernel is launched.  It holds flat arrays of size
// batch_size × per_thread_stride.  Each kernel thread receives pointers to
// its private slice:
//
//   visited_ht   = d_visited_hash  + tid * HASH_SIZE      (16384 ints = 64 KB)
//   curr_fr      = d_curr_frontier + tid * MAX_FRONTIER   ( 8192 ints = 32 KB)
//   next_fr      = d_next_frontier + tid * MAX_FRONTIER   ( 8192 ints = 32 KB)
//   one_hop      = d_one_hop       + tid * MAX_1HOP       ( 4096 ints = 16 KB)
//   two_hop      = d_two_hop       + tid * MAX_2HOP       ( 8192 ints = 32 KB)
//
// Total per thread: ~176 KB.  With 10 batches of ~500 threads: ~86 MB/batch.
// ═════════════════════════════════════════════════════════════════════════════
struct NeighborhoodSlab {
    int* d_visited_hash;    // [batch_size × HASH_SIZE]
    int* d_curr_frontier;   // [batch_size × MAX_FRONTIER]
    int* d_next_frontier;   // [batch_size × MAX_FRONTIER]
    int* d_one_hop;         // [batch_size × MAX_1HOP]
    int* d_two_hop;         // [batch_size × MAX_2HOP]
    int  num_threads;

    void free() {
        if (d_visited_hash)   { cudaFree(d_visited_hash);   d_visited_hash   = nullptr; }
        if (d_curr_frontier)  { cudaFree(d_curr_frontier);  d_curr_frontier  = nullptr; }
        if (d_next_frontier)  { cudaFree(d_next_frontier);  d_next_frontier  = nullptr; }
        if (d_one_hop)        { cudaFree(d_one_hop);        d_one_hop        = nullptr; }
        if (d_two_hop)        { cudaFree(d_two_hop);        d_two_hop        = nullptr; }
        num_threads = 0;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Device BFS helpers
// ─────────────────────────────────────────────────────────────────────────────

// Knuth multiplicative hash + double-hashing to avoid primary clustering.
// Returns true  → node was newly inserted.
// Returns false → already present, or table is ≥75% full (fast-fail).
__device__ __forceinline__
bool bfs_hash_insert(int* ht, int node)
{
    // Fast-fail: avoid O(HASH_SIZE) probe scan on a dense table
    // Counter stored at ht[HASH_SIZE]; load limit = 75%
    const int load_limit = (BFSConst::HASH_SIZE * 3) >> 2;
    if (ht[BFSConst::HASH_SIZE] >= load_limit) return false;

    const unsigned key = (unsigned)node;
    const unsigned h1  = (key * 2654435761u) & BFSConst::HASH_MASK;
    const unsigned h2  = ((key * 2246822519u) >> 16) | 1u;   // always odd

    for (int probe = 0; probe < BFSConst::HASH_SIZE; ++probe) {
        const int slot = (int)((h1 + (unsigned)probe * h2) & BFSConst::HASH_MASK);
        const int val  = ht[slot];
        if (val == node)  return false;
        if (val == -1) {
            ht[slot] = node;
            ht[BFSConst::HASH_SIZE]++;   // increment counter
            return true;
        }
    }
    return false;
}

// Reset: set all slots to -1, counter to 0.
// Callers must allocate HASH_SIZE + 1 ints per thread.
__device__ __forceinline__
void bfs_hash_reset(int* ht)
{
    for (int i = 0; i <= BFSConst::HASH_SIZE; ++i) ht[i] = -1;
    ht[BFSConst::HASH_SIZE] = 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// expand_neighbors
// ─────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__
void expand_neighbors(
    const DeviceGraph* __restrict__ d_fwd,
    const DeviceGraph* __restrict__ d_bwd,
    int   cur,
    int*  visited_ht,
    int*  frontier,    int& frontier_cnt, int frontier_cap,
    int*  output,      int& out_cnt,      int out_cap)
{
    if ((unsigned)cur >= (unsigned)d_fwd->num_nodes) return;

    const int fwd_nodes = d_fwd->num_nodes;
    const int bwd_nodes = d_bwd->num_nodes;

    // ── Forward edges ────────────────────────────────────────────────────────
    {
        const int s = __ldg(&d_fwd->offsets[cur]);
        const int e = __ldg(&d_fwd->offsets[cur + 1]);
        for (int i = s; i < e; ++i) {
            const int nbr = __ldg(&d_fwd->edges[i]);
            if ((unsigned)nbr >= (unsigned)fwd_nodes) continue;
            if (bfs_hash_insert(visited_ht, nbr)) {
                if (frontier_cnt < frontier_cap) frontier[frontier_cnt++] = nbr;
                if (out_cnt      < out_cap)      output[out_cnt++]        = nbr;
            }
        }
    }

    // ── Backward edges ───────────────────────────────────────────────────────
    if ((unsigned)cur < (unsigned)bwd_nodes) {
        const int s      = __ldg(&d_bwd->offsets[cur]);
        const int e      = __ldg(&d_bwd->offsets[cur + 1]);
        const int brange = e - s;

        // Cap backward fan-out: high-indegree seeds (f4/f5/f6) can have
        // 1K–102K backward neighbors; strided sampling prevents hash saturation.
        const int MAX_BWD = BFSConst::MAX_2HOP;
        const int stride  = (brange > MAX_BWD) ? (brange / MAX_BWD) : 1;

        for (int i = s; i < e; i += stride) {
            if (frontier_cnt >= frontier_cap && out_cnt >= out_cap) break;
            const int nbr = __ldg(&d_bwd->edges[i]);
            if ((unsigned)nbr >= (unsigned)fwd_nodes) continue;
            if (bfs_hash_insert(visited_ht, nbr)) {
                if (frontier_cnt < frontier_cap) frontier[frontier_cnt++] = nbr;
                if (out_cnt      < out_cap)      output[out_cnt++]        = nbr;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GetOneAndTwoHopNeighborhood
// ─────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__
void GetOneAndTwoHopNeighborhood(
    const DeviceGraph* __restrict__ d_fwd,
    const DeviceGraph* __restrict__ d_bwd,
    int   generator_node,
    int*  visited_ht,
    int*  curr_frontier,
    int*  next_frontier,
    int*  one_hop,
    int&  one_cnt,
    int*  two_hop,
    int&  two_cnt)
{
    one_cnt = two_cnt = 0;
    int curr_cnt = 0, next_cnt = 0;

    if (generator_node < 0 || generator_node >= d_fwd->num_nodes) return;

    bfs_hash_insert(visited_ht, generator_node);
    curr_frontier[curr_cnt++] = generator_node;

    // Level 0 → 1
    for (int fi = 0; fi < curr_cnt; ++fi) {
        expand_neighbors(d_fwd, d_bwd, curr_frontier[fi], visited_ht,
                         next_frontier, next_cnt, BFSConst::MAX_FRONTIER,
                         one_hop,       one_cnt,  BFSConst::MAX_1HOP);
    }

    // Pointer swap: full 1-hop frontier drives level 1→2
    { int* tmp = curr_frontier; curr_frontier = next_frontier; next_frontier = tmp; }
    curr_cnt = next_cnt;
    next_cnt = 0;

    // Level 1 → 2
    for (int fi = 0; fi < curr_cnt; ++fi) {
        expand_neighbors(d_fwd, d_bwd, curr_frontier[fi], visited_ht,
                         next_frontier, next_cnt, BFSConst::MAX_FRONTIER,
                         two_hop,       two_cnt,  BFSConst::MAX_2HOP);
    }
}


// ═════════════════════════════════════════════════════════════════════════════
// Paper agent
// ═════════════════════════════════════════════════════════════════════════════
class Paper : public Agent {
public:
    int id;
    int year;
    int citations;
    float quality;
    int age;
    curandState rand_state;

    static constexpr int   MAX_CITATIONS       = 250;
    static constexpr float FITNESS_DECAY_ALPHA = 3.0f;
    static constexpr float GAMMA               = 3.0f;

    struct State {
        int  id;
        int  year;
        bool is_seed;

        int   fitness_peak_value;
        int   fitness_lag_duration;
        int   fitness_peak_duration;
        float current_fitness;

        float pa_weight;
        float recency_weight;
        float fitness_weight;
        float alpha;

        int in_degree;
        int out_degree;
        int assigned_out_degree;

        int  generator_node;
        int  citations_made;
        bool is_new;

        int cited_papers[MAX_CITATIONS];

        __host__ __device__ State()
            : id(-1), year(0), is_seed(false),
              fitness_peak_value(1), fitness_lag_duration(0),
              fitness_peak_duration(1000), current_fitness(1.0f),
              pa_weight(0.33f), recency_weight(0.33f), fitness_weight(0.34f),
              alpha(0.7f),
              in_degree(0), out_degree(0), assigned_out_degree(0),
              generator_node(-1), citations_made(0), is_new(false)
        {
            for (int i = 0; i < MAX_CITATIONS; i++) cited_papers[i] = -1;
        }
    };

    State state;

    __host__ __device__ Paper() : Agent(0) {
        id = 0; year = 0; citations = 0; quality = 0.0f; age = 0;
    }
    __host__ __device__ Paper(int paper_id, int pub_year, float q)
        : Agent(paper_id) {
        id = paper_id; year = pub_year; citations = 0; quality = q; age = 0;
    }
    __host__ __device__ Paper(const Paper& o) : Agent(o.id) {
        id = o.id; year = o.year; citations = o.citations;
        quality = o.quality; age = o.age; rand_state = o.rand_state;
        state = o.state;
    }
    __host__ __device__ Paper& operator=(const Paper& o) {
        if (this != &o) {
            id = o.id; year = o.year; citations = o.citations;
            quality = o.quality; age = o.age; rand_state = o.rand_state;
            state = o.state;
        }
        return *this;
    }
    
    __host__ __device__  virtual ~Paper() {}

    __device__ virtual void callMethod(int functionId, void* argument) {
        switch (functionId) {
            case 0: updateAge();   break;
            case 2: addCitation(); break;
            default: break;
        }
    }
    __device__ void callMethodDevice(int functionId) {
        switch (functionId) {
            case 0: updateAge();       break;
            case 1: updateCitations(); break;
            case 2: addCitation();     break;
            default: break;
        }
    }

    __host__ std::string getType() const {
        return state.is_seed ? "seed" : "agent";
    }

    __device__ void initRandom(unsigned long sv) { curand_init(sv, id, 0, &rand_state); }
    __host__ __device__ void updateAge()    { age++; }
    __host__ __device__ void addCitation()  { citations++; }
    __device__ void updateCitations() {
        float decay = 1.0f / (1.0f + age * 0.1f);
        if (curand_uniform(&rand_state) < quality * decay) citations++;
    }

    __host__ __device__ int   getId()        const { return id; }
    __host__ __device__ int   getYear()      const { return year; }
    __host__ __device__ int   getCitations() const { return citations; }
    __host__ __device__ float getQuality()   const { return quality; }
    __host__ __device__ int   getAge()       const { return age; }
    __host__ __device__ void  setId(int i)         { id = i; }
    __host__ __device__ void  setYear(int y)       { year = y; }
    __host__ __device__ void  setCitations(int c)  { citations = c; }
    __host__ __device__ void  setQuality(float q)  { quality = q; }
    __host__ __device__ void  setAge(int a)        { age = a; }

    __device__ void updateFitness(int current_year) {
        int age_val   = current_year - state.year;
        if (age_val < state.fitness_lag_duration) { state.current_fitness = 1.0f; return; }
        int since_lag = age_val - state.fitness_lag_duration;
        if (since_lag <= state.fitness_peak_duration) {
            state.current_fitness = static_cast<float>(state.fitness_peak_value); return;
        }
        float since_pk = static_cast<float>(since_lag - state.fitness_peak_duration);
        state.current_fitness = state.fitness_peak_value
                                / powf(since_pk + 1.0f, FITNESS_DECAY_ALPHA);
    }

    // Mirrors GPU: generator is selected from the OLD population only (seeds +
    // prior-year agents).  New papers have no edges yet, so BFS from a new-paper
    // generator yields empty 1-hop/2-hop and degenerates every citation to
    // fully-random, breaking the PA/fitness-guided WRS distribution.
    __device__ void selectGenerator(int old_population, curandState* rng)
    {
        if (!state.is_new) return;
        if (old_population <= 0) { state.generator_node = -1; return; }
        int gen = (int)((unsigned)curand(rng) % (unsigned)old_population);
        state.generator_node = gen;
    }

    __device__ __forceinline__
    float computeCitationScore(float pa_norm, float fit_norm, float rec_norm) const
    {
        return state.pa_weight      * pa_norm
             + state.recency_weight * rec_norm
             + state.fitness_weight * fit_norm;
    }

    __device__ int MakePopulateCitations(
        int          num_papers,
        const int*   candidates,
        int          cand_count,
        int          k,
        const float* d_pa_arr_norm,
        const float* d_fit_arr_norm,
        const float* d_recency_arr,
        curandState* rng)
    {
        if (k <= 0 || cand_count <= 0) return 0;
        if (k > cand_count) k = cand_count;
        int room = MAX_CITATIONS - state.citations_made;
        if (k > room) k = room;
        if (k <= 0) return 0;

        const float MIN_KEY = -2.147483648e9f;

        float res_key[MAX_CITATIONS];
        int   res_id [MAX_CITATIONS];
        int   res_n   = 0;
        float min_key = -3.4e38f;
        int   min_pos = 0;

        for (int ci = 0; ci < cand_count; ++ci) {
            int cand = candidates[ci];
            if (cand < 0 || cand >= num_papers || cand == state.id) continue;

            float pa_n  = (d_pa_arr_norm  && cand < num_papers) ? d_pa_arr_norm [cand] : 1e-9f;
            float fit_n = (d_fit_arr_norm && cand < num_papers) ? d_fit_arr_norm[cand] : 1e-9f;
            float rec   = (d_recency_arr  && cand < num_papers) ? d_recency_arr [cand] : 1e-9f;

            float score = computeCitationScore(pa_n, fit_n, rec);

            float key;
            if (score > 1e-30f) {
                float u = curand_uniform(rng);
                if (u < 1e-10f) u = 1e-10f;
                key = __logf(u) / score;
            } else {
                key = MIN_KEY;
            }

            if (res_n < k) {
                res_key[res_n] = key;
                res_id [res_n] = cand;
                ++res_n;
                if (res_n == k) {
                    min_key = res_key[0]; min_pos = 0;
                    for (int r = 1; r < res_n; r++)
                        if (res_key[r] < min_key) { min_key = res_key[r]; min_pos = r; }
                }
            } else if (key > min_key) {
                res_key[min_pos] = key;
                res_id [min_pos] = cand;
                min_key = res_key[0]; min_pos = 0;
                for (int r = 1; r < res_n; r++)
                    if (res_key[r] < min_key) { min_key = res_key[r]; min_pos = r; }
            }
        }

        int added = 0;
        for (int r = 0; r < res_n && state.citations_made < MAX_CITATIONS; ++r) {
            state.cited_papers[state.citations_made++] = res_id[r];
            ++added;
        }
        return added;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // makeCitations – mirrors GPU ABMKernelStage2/3/4 pipeline exactly.
    //
    // Stage ordering matches GPU:
    //   Stage 0 : cite generator node  (always 1 edge)
    //   Stage 1 : same-year citation   (0 or 1 edge, BEFORE 1-hop WRS)
    //   Stage 2 : 1-hop WRS            (BFS 1-hop neighbourhood)
    //   Stage 3 : 2-hop WRS            (BFS 2-hop neighbourhood)
    //   Stage 4 : fully-random         (pool = [0, prev_population) only,
    //                                   mirrors GPU graphNodeSetSize = old nodes)
    //
    // Key invariant (matches GPU Stage 4 budget formula):
    //   num_rand = budget - gen - same - k_1hop - k_2hop
    //   where k_1hop/k_2hop are the PLANNED (capped) values, not actual cited.
    // ─────────────────────────────────────────────────────────────────────────
    __device__ void makeCitations(
        const Paper*       all_papers,
        int                num_papers,
        int                prev_population,
        const DeviceGraph* d_fwd,
        const DeviceGraph* d_bwd,
        int*               visited_ht,
        int*               curr_frontier,
        int*               next_frontier,
        int*               one_hop,
        int*               two_hop,
        int                current_year,
        const float*       d_pa_arr_norm,
        const float*       d_fit_arr_norm,
        const float*       d_recency_arr,
        float              fully_random_ratio,
        float              same_year_ratio,
        curandState*       rng)
    {
        if (!state.is_new) return;
        state.citations_made = 0;

        int budget  = state.assigned_out_degree;
        int num_gen = (state.generator_node >= 0) ? 1 : 0;

        // ── Stage 0: cite generator ──────────────────────────────────────────
        if (num_gen > 0 && state.citations_made < MAX_CITATIONS)
            state.cited_papers[state.citations_made++] = state.generator_node;

        // ── Determine same-year flag (Bernoulli) ──────────────────────────────
        // GPU selects a deterministic subset of floor(N * proportion) papers;
        // Bernoulli is statistically equivalent and compatible with per-thread RNG.
        int num_same = (curand_uniform(rng) < same_year_ratio) ? 1 : 0;

        // ── Stage 1: same-year citation – BEFORE 1-hop WRS ───────────────────
        // Mirrors GPU ABMKernelStage2 which calls MakeSameYearCitations first.
        // GPU picks uniformly from [0, num_new_nodes-1] with no dedup;
        // we replicate that: one direct pick, no cross-citation dedup.
        int num_new_cohort = num_papers - prev_population;
        if (num_same > 0 && num_new_cohort > 0
                && state.citations_made < MAX_CITATIONS) {
            int offset = (int)((unsigned)curand(rng) % (unsigned)num_new_cohort);
            int rid    = prev_population + offset;
            // Only skip self-citation (GPU has no explicit guard, but self-pick
            // is extremely rare; one fallback step matches GPU's intent).
            if (rid == state.id && num_new_cohort > 1)
                rid = prev_population + ((offset + 1) % num_new_cohort);
            state.cited_papers[state.citations_made++] = rid;
        } else {
            num_same = 0;   // couldn't make the same-year citation
        }

        // ── Budget: remaining slots for 1-hop + 2-hop ────────────────────────
        int num_rand_reserved = (int)floorf(fully_random_ratio * (float)budget);
        int remaining = budget - num_gen - num_same - num_rand_reserved;
        if (remaining < 0) remaining = 0;

        // ── BFS from generator ────────────────────────────────────────────────
        int one_cnt = 0, two_cnt = 0;
        if (state.generator_node >= 0) {
            bfs_hash_reset(visited_ht);
            GetOneAndTwoHopNeighborhood(
                d_fwd, d_bwd, state.generator_node, visited_ht,
                curr_frontier, next_frontier,
                one_hop, one_cnt, two_hop, two_cnt);
        }

        // ── Stage 2: 1-hop WRS ───────────────────────────────────────────────
        int k_1hop_planned = (int)ceilf(state.alpha * (float)remaining);
        int k_1hop = min(k_1hop_planned, one_cnt);
        MakePopulateCitations(num_papers,
                              one_hop, one_cnt, k_1hop,
                              d_pa_arr_norm, d_fit_arr_norm,
                              d_recency_arr, rng);

        // ── Stage 3: 2-hop WRS ───────────────────────────────────────────────
        int remaining3 = budget - num_gen - num_same - num_rand_reserved - k_1hop;
        if (remaining3 < 0) remaining3 = 0;
        int k_2hop = min(remaining3, two_cnt);
        MakePopulateCitations(num_papers,
                              two_hop, two_cnt, k_2hop,
                              d_pa_arr_norm, d_fit_arr_norm,
                              d_recency_arr, rng);

        // ── Stage 4: fully-random ─────────────────────────────────────────────
        // Budget formula mirrors GPU kernelCallStage4:
        //   num_fully_random = outdeg - gen - same - planned_inside - planned_outside
        // Pool = [0, prev_population) only, matching GPU where MakeUniformRandomCitations
        // samples from graphNodeSetSize = old nodes (before current-year nodes are
        // inserted into the node set).
        int num_rand = budget - num_gen - num_same - k_1hop - k_2hop;
        if (num_rand < 0) num_rand = 0;
        for (int i = 0; i < num_rand && state.citations_made < MAX_CITATIONS; i++) {
            for (int att = 0; att < 40; att++) {
                // Sample from OLD nodes only (seeds + prior-year agents).
                int rid = (int)((unsigned)curand(rng) % (unsigned)prev_population);
                if (rid == state.id) continue;
                bool dup = false;
                for (int ck = 0; ck < state.citations_made; ck++)
                    if (state.cited_papers[ck] == rid) { dup = true; break; }
                if (!dup) { state.cited_papers[state.citations_made++] = rid; break; }
            }
        }

        state.out_degree = state.citations_made;
    }

    __device__ void updateInDegree(const Paper* all_papers, int num_papers) {
        int cnt = 0;
        for (int i = 0; i < num_papers; i++)
            for (int j = 0; j < MAX_CITATIONS; j++)
                if (all_papers[i].state.cited_papers[j] == state.id) cnt++;
        state.in_degree = cnt;
    }

    __device__ void resetNewStatus() { state.is_new = false; }

    __host__ __device__ virtual int  size() const  { return sizeof(State); }
    __host__ __device__ virtual void pack(void* b)   { memcpy(b,      &state, sizeof(State)); }
    __host__ __device__ virtual void unpack(void* b) { memcpy(&state, b,      sizeof(State)); }
};

#endif // PAPER_H
