// ─────────────────────────────────────────────────────────────────────────────
// citation_abm_mass.cu  –  MASS CUDA Citation-Network ABM
//
// Implements the full ABM pipeline from the C++ ABM using:
//   graph.cu  →  ParseNodelist, ParseEdgelist, forward/backward CSR,
//                updateNodeInDegreeOutDegree, WriteAttributes
//   abm.cu    →  InitializeFitness, PopulateWeightArrs, selectGenerator,
//                GetOneAndTwoHopNeighborhood, MakePopulateCitations, run loop
// ─────────────────────────────────────────────────────────────────────────────

#include "Mass.h"
#include "Agent.h"
#include "Paper.h"
#include "argparse.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <algorithm>
#include <random>
#include <cmath>
#include <chrono>
#include <cassert>

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────
struct ABMConfig {
    std::string edgelist_file;
    std::string nodelist_file;
    std::string out_degree_bag_file;
    std::string recency_probabilities_file;

    float alpha;
    float fully_random_citations;
    float same_year_proportion;
    float preferential_weight;
    float recency_weight;
    float fitness_weight;

    int   initial_population;
    int   num_cycles;
    float growth_rate;

    // ── Batching ──────────────────────────────────────────────────────────────
    // Number of batches to split new papers into per simulation year.
    // Reduces peak BFS slab memory from (num_new × 176 KB) to
    // (num_new/bfs_num_batches × 176 KB).  10 is a good default.
    int bfs_num_batches;

    std::string output_file;
    std::string auxiliary_information_file;
    std::string log_file;

    int num_processors;
    int log_level;

    ABMConfig()
        : initial_population(1000), num_cycles(30), growth_rate(0.05f),
          alpha(0.7f), fully_random_citations(0.05f), same_year_proportion(0.1f),
          preferential_weight(-1.0f), recency_weight(-1.0f), fitness_weight(-1.0f),
          bfs_num_batches(10),
          num_processors(1), log_level(1) {}
};

// ─────────────────────────────────────────────────────────────────────────────
// Host-side helpers
// ─────────────────────────────────────────────────────────────────────────────
std::vector<int> loadOutDegreeBag(const std::string& f) {
    std::vector<int> bag;
    if (f.empty()) {
        for (int i = 1; i <= 50; i++) {
            int cnt = std::max(1, static_cast<int>(250 / std::pow(i, 1.5)));
            for (int j = 0; j < cnt; j++) bag.push_back(i);
        }
        return bag;
    }
    std::ifstream fs(f); std::string line;
    while (std::getline(fs, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line); std::string idx, val;
        std::getline(ss, idx, ','); std::getline(ss, val, ',');
        bag.push_back(std::stoi(val));
    }
    return bag;
}

std::map<int, float> loadRecencyProbabilities(const std::string& f) {
    std::map<int, float> probs;
    if (f.empty()) {
        for (int i = 0; i < 50; i++) probs[i] = std::exp(-0.1f * i);
        return probs;
    }
    std::ifstream fs(f); std::string line;
    while (std::getline(fs, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line); std::string yd, p;
        std::getline(ss, yd, ','); std::getline(ss, p, ',');
        probs[std::stoi(yd)] = std::stof(p);
    }
    return probs;
}

static inline char detect_delimiter(const std::string& path) {
    std::ifstream f(path); std::string line;
    std::getline(f, line);
    if (line.find(',')  != std::string::npos) return ',';
    if (line.find('\t') != std::string::npos) return '\t';
    if (line.find(' ')  != std::string::npos) return ' ';
    throw std::invalid_argument("Cannot detect delimiter for " + path);
}

// ─────────────────────────────────────────────────────────────────────────────
// Host-side CSR builder (templated: works with vector<int> and set<int>)
// ─────────────────────────────────────────────────────────────────────────────
struct HostCSR {
    std::vector<int> offsets;
    std::vector<int> edges;
    int num_nodes = 0;
    int num_edges = 0;
};

template<typename Container>
static HostCSR buildHostCSR(const std::vector<Container>& adj, int num_nodes)
{
    HostCSR csr;
    csr.num_nodes = num_nodes;
    csr.offsets.resize(num_nodes + 1, 0);

    for (int i = 0; i < num_nodes; i++)
        csr.offsets[i + 1] = (int)adj[i].size();
    for (int i = 1; i <= num_nodes; i++)
        csr.offsets[i] += csr.offsets[i - 1];

    csr.num_edges = csr.offsets[num_nodes];
    csr.edges.resize(csr.num_edges);

    std::vector<int> pos(num_nodes, 0);
    for (int i = 0; i < num_nodes; i++) {
        int start = csr.offsets[i];
        for (int nbr : adj[i])
            csr.edges[start + pos[i]++] = nbr;
    }
    return csr;
}

// ─────────────────────────────────────────────────────────────────────────────
// CUDA kernels
// ─────────────────────────────────────────────────────────────────────────────

__global__ void initCurandKernel(curandState* states, int n, unsigned long long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) curand_init(seed + i, i, 0, &states[i]);
}

__global__ void updateFitnessKernel(Paper* papers, int n, int year) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) papers[i].updateFitness(year);
}

// Mirrors GPU CPU-side getGeneratorNode() which samples from graph->getNodeSetSize()
// evaluated BEFORE new nodes are inserted into node_set.  Generator must be an
// old node (seed or prior-year agent) so BFS has edges to explore; a new-paper
// generator has no edges and yields empty 1-hop/2-hop → all citations fall to random.
__global__ void selectGeneratorKernel(Paper* papers, int n, int old_population, curandState* states) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && papers[i].state.is_new)
        papers[i].selectGenerator(old_population, &states[i]);
}

__global__ void computeRawScoresKernel(
    const Paper*       papers,
    int                n,
    float              gamma,
    const DeviceGraph* d_bwd,
    float*             raw_pa,
    float*             raw_fit)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int in_deg = 0;
    if (d_bwd && i < d_bwd->num_nodes)
        in_deg = __ldg(&d_bwd->offsets[i + 1]) - __ldg(&d_bwd->offsets[i]);

    float fit = papers[i].state.current_fitness;
    if (fit < 0.0f) fit = 0.0f;

    raw_pa [i] = powf((float)in_deg, gamma) + 1.0f;
    raw_fit[i] = powf(fit,           gamma) + 1.0f;
}

// ─────────────────────────────────────────────────────────────────────────────
// makeCitationsWithBFSKernel
// One thread per paper in the current batch.
// tid is the index within [0, batch_size), not the global new-paper index.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void makeCitationsWithBFSKernel(
    Paper*             papers,
    int*               batch_paper_indices,  // [batch_size]
    int                batch_size,
    int                num_papers,
    int                prev_population,
    DeviceGraph*       d_fwd,
    DeviceGraph*       d_bwd,
    NeighborhoodSlab   slab,
    int                current_year,
    const float*       d_pa_arr_norm,
    const float*       d_fit_arr_norm,
    const float*       d_recency_arr,
    float              rand_ratio,
    float              same_ratio,
    curandState*       rand_states)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    int paper_idx = batch_paper_indices[tid];

    // Each thread's private BFS buffers from the slab
    // Note: HASH_SIZE + 1 to hold the insertion counter
    int* visited_ht    = slab.d_visited_hash   + (long long)tid * (BFSConst::HASH_SIZE + 1);
    int* curr_frontier = slab.d_curr_frontier  + (long long)tid * BFSConst::MAX_FRONTIER;
    int* next_frontier = slab.d_next_frontier  + (long long)tid * BFSConst::MAX_FRONTIER;
    int* one_hop       = slab.d_one_hop        + (long long)tid * BFSConst::MAX_1HOP;
    int* two_hop       = slab.d_two_hop        + (long long)tid * BFSConst::MAX_2HOP;

    bfs_hash_reset(visited_ht);

    papers[paper_idx].makeCitations(
        papers, num_papers, prev_population,
        d_fwd, d_bwd,
        visited_ht, curr_frontier, next_frontier,
        one_hop, two_hop,
        current_year,
        d_pa_arr_norm, d_fit_arr_norm, d_recency_arr,
        rand_ratio, same_ratio,
        &rand_states[paper_idx]);
}

__global__ void updateInDegreeFromCSRKernel(Paper* papers, int n, DeviceGraph* d_bwd)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (i < d_bwd->num_nodes)
        papers[i].state.in_degree = d_bwd->offsets[i + 1] - d_bwd->offsets[i];
    else
        papers[i].state.in_degree = 0;
}

__global__ void updateOutDegreeFromCSRKernel(Paper* papers, int n, DeviceGraph* d_fwd)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (i < d_fwd->num_nodes)
        papers[i].state.out_degree = d_fwd->offsets[i + 1] - d_fwd->offsets[i];
    else
        papers[i].state.out_degree = 0;
}

__global__ void resetNewStatusKernel(Paper* papers, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) papers[i].resetNewStatus();
}

// ─────────────────────────────────────────────────────────────────────────────
// reassignFitnessKernel
//
// Mirrors GPU execute() which calls AssignPeakFitnessValues(graph, new_nodes_vec)
// AFTER citations are generated each year, using its own freshly-seeded RNG.
// This means the fit_peak_value stored in the aux file is a POST-citation
// re-sample — it did NOT influence which papers were cited during that year.
//
// MASS previously only assigned fitness once in addNewPapers (before citations),
// so the stored fpv was the PRE-citation value.  Adding this post-citation
// re-sample makes the aux output semantically match GPU's.
//
// The power-law sampler matches AssignPeakFitnessValues exactly:
//   p(i) ∝ scale_factor * constant * i^exponent,  i ∈ [1, 1000]
//   scale_factor=6.3742991333, constant=0.072, exponent=-1.634
// ─────────────────────────────────────────────────────────────────────────────
__global__ void reassignFitnessKernel(Paper* papers, int start_idx, int n,
                                       curandState* rand_states,
                                       const float* d_fitness_cdf,
                                       int num_fitness_vals)
{
    int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= n) return;
    int idx = start_idx + local;

    Paper& p = papers[idx];
    if (!p.state.is_new) return;   // only re-assign for this year's new agents

    // Inverse-CDF sampling from the pre-built fitness CDF
    float u = curand_uniform(&rand_states[idx]);
    // Binary search in CDF
    int lo = 0, hi = num_fitness_vals - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (d_fitness_cdf[mid] < u) lo = mid + 1;
        else                        hi = mid;
    }
    p.state.fitness_peak_value = lo + 1;   // 1-indexed (values 1..1000)

    // lag and peak duration remain at their defaults (0 and 1000) — same as
    // GPU's AssignFitnessLagDuration and AssignFitnessPeakDuration which
    // hard-code those values.
    p.state.fitness_lag_duration  = 0;
    p.state.fitness_peak_duration = 1000;
}

// ─────────────────────────────────────────────────────────────────────────────
// CitationABM  –  simulation manager
// ─────────────────────────────────────────────────────────────────────────────
class CitationABM {
private:
    ABMConfig config;

    std::vector<Paper>   h_papers;
    std::vector<int>     out_degree_bag;
    std::map<int, float> recency_probs_map;
    std::map<int, int>   nodeYearMap;
    std::vector<std::pair<int,int>> h_edges;

    Paper*       d_papers         = nullptr;
    float*       d_recency_probs  = nullptr;
    float*       d_recency_arr    = nullptr;
    float*       d_pa_arr_norm    = nullptr;
    float*       d_fit_arr_norm   = nullptr;
    curandState* d_rand_states    = nullptr;
    float*       d_fitness_cdf    = nullptr;   // pre-built CDF for post-citation fitness re-assignment

    DeviceGraph* d_fwd_graph      = nullptr;
    DeviceGraph* d_bwd_graph      = nullptr;
    int*         d_fwd_offsets    = nullptr;
    int*         d_fwd_edges      = nullptr;
    int*         d_bwd_offsets    = nullptr;
    int*         d_bwd_edges      = nullptr;

    int current_year       = 1983;
    int current_population = 0;
    int max_population     = 0;

    std::mt19937 rng;

public:
    int node_seq_id = 0;
    std::map<int, int> continuous_node_mapping;
    std::map<int, int> reverse_continuous_node_mapping;

    CitationABM(const ABMConfig& cfg) : config(cfg) {
        rng.seed(std::random_device{}());
        out_degree_bag    = loadOutDegreeBag(config.out_degree_bag_file);
        recency_probs_map = loadRecencyProbabilities(config.recency_probabilities_file);
        std::cout << "Out-degree bag: " << out_degree_bag.size()    << " entries\n";
        std::cout << "Recency probs:  " << recency_probs_map.size() << " entries\n";
        std::cout << "BFS batches:    " << config.bfs_num_batches   << " per year\n";
    }

    ~CitationABM() { cleanup(); }

    // =========================================================================
    // parseNodelist
    // =========================================================================
    void parseNodelist() {
        printf("\nInside parseNodelist...\n");
        auto t0 = std::chrono::steady_clock::now();

        char delim = detect_delimiter(config.nodelist_file);
        std::ifstream f(config.nodelist_file);
        std::string line;

        std::map<int,int> rawMap;
        while (std::getline(f, line)) {
            std::stringstream ss(line);
            std::string tok;
            std::vector<std::string> cols;
            while (std::getline(ss, tok, delim)) cols.push_back(tok);
            if (cols.empty() || cols[0][0] == '#') continue;
            rawMap[std::stoi(cols[0])] = std::stoi(cols[1]);
        }

        std::cout << "Populating continuous node mappings (" << rawMap.size() << " nodes)...\n";
        for (const auto& [orig_id, year] : rawMap) {
            continuous_node_mapping[orig_id]             = node_seq_id;
            reverse_continuous_node_mapping[node_seq_id] = orig_id;
            nodeYearMap[node_seq_id]                     = year;
            node_seq_id++;
        }
        config.initial_population = node_seq_id;

        auto t1 = std::chrono::steady_clock::now();
        std::cout << "parseNodelist done in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
                  << " ms (" << node_seq_id << " nodes)\n";
    }

    // =========================================================================
    // parseEdgelist
    // =========================================================================
    void parseEdgelist() {
        printf("\nInside parseEdgelist...\n");
        auto t0 = std::chrono::steady_clock::now();

        std::unordered_map<int,int> fast_map;
        fast_map.reserve(continuous_node_mapping.size());
        for (const auto& [k, v] : continuous_node_mapping) fast_map[k] = v;

        std::ifstream file(config.edgelist_file, std::ios::binary | std::ios::ate);
        auto file_size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::string contents(file_size, '\0');
        file.read(&contents[0], file_size);
        file.close();

        char delim = detect_delimiter(config.edgelist_file);
        long edge_count = 0;

        const char* ptr = contents.data();
        const char* end = ptr + contents.size();
        while (ptr < end) {
            if (*ptr == '#' || *ptr == '\n' || *ptr == '\r') {
                while (ptr < end && *ptr != '\n') ptr++;
                if (ptr < end) ptr++;
                continue;
            }
            const char* p1 = ptr;
            while (ptr < end && *ptr != delim && *ptr != '\n' && *ptr != '\r') ptr++;
            int citing = 0;
            for (const char* c = p1; c < ptr; c++)
                if (*c >= '0' && *c <= '9') citing = citing * 10 + (*c - '0');
            if (ptr < end && *ptr == delim) ptr++;

            const char* p2 = ptr;
            while (ptr < end && *ptr != '\n' && *ptr != '\r') ptr++;
            int cited = 0;
            for (const char* c = p2; c < ptr; c++)
                if (*c >= '0' && *c <= '9') cited = cited * 10 + (*c - '0');

            auto it_c = fast_map.find(citing);
            auto it_d = fast_map.find(cited);
            if (it_c != fast_map.end() && it_d != fast_map.end()) {
                h_edges.emplace_back(it_c->second, it_d->second);
                edge_count++;
            }
            while (ptr < end && (*ptr == '\n' || *ptr == '\r')) ptr++;
        }

        auto t1 = std::chrono::steady_clock::now();
        std::cout << "parseEdgelist done in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
                  << " ms (" << edge_count << " edges)\n";
    }

    // =========================================================================
    // buildDeviceGraphs
    //
    // Critical fix: the previous implementation used std::set<int> which stores
    // neighbors in ascending seq-id order. Since seeds have lower seq-ids
    // (0..491K) than agents (491K+), BFS always processes seed neighbors first.
    // With capped BFS arrays (MAX_1HOP=4096), agent neighbors are systematically
    // dropped, causing agents to receive fewer fitness-weighted WRS citations.
    //
    // Fix: build with dedup via unordered_set, then store in vector and shuffle
    // each row. This gives the BFS cap a uniform sample across seeds and agents,
    // matching the GPU's uncapped BFS behavior statistically.
    // =========================================================================
    void buildDeviceGraphs() {
        int N = current_population;

        // Dedup via unordered_set (order-independent), then copy to vector for shuffle
        std::vector<std::vector<int>> fwd_adj(N), bwd_adj(N);
        {
            std::vector<std::unordered_set<int>> fwd_seen(N), bwd_seen(N);
            for (const auto& [src, dst] : h_edges) {
                if (src >= 0 && src < N && dst >= 0 && dst < N) {
                    if (fwd_seen[src].insert(dst).second) fwd_adj[src].push_back(dst);
                    if (bwd_seen[dst].insert(src).second) bwd_adj[dst].push_back(src);
                }
            }
        }

        // Shuffle each adjacency list so seeds and agents are interleaved in
        // the CSR edge array.  The BFS cap then samples both proportionally.
        for (int i = 0; i < N; i++) {
            if (fwd_adj[i].size() > 1)
                std::shuffle(fwd_adj[i].begin(), fwd_adj[i].end(), rng);
            if (bwd_adj[i].size() > 1)
                std::shuffle(bwd_adj[i].begin(), bwd_adj[i].end(), rng);
        }

        HostCSR fwd_csr = buildHostCSR(fwd_adj, N);
        HostCSR bwd_csr = buildHostCSR(bwd_adj, N);

        if (d_fwd_offsets) { cudaFree(d_fwd_offsets); d_fwd_offsets = nullptr; }
        if (d_fwd_edges)   { cudaFree(d_fwd_edges);   d_fwd_edges   = nullptr; }
        if (d_bwd_offsets) { cudaFree(d_bwd_offsets); d_bwd_offsets = nullptr; }
        if (d_bwd_edges)   { cudaFree(d_bwd_edges);   d_bwd_edges   = nullptr; }
        if (d_fwd_graph)   { cudaFree(d_fwd_graph);   d_fwd_graph   = nullptr; }
        if (d_bwd_graph)   { cudaFree(d_bwd_graph);   d_bwd_graph   = nullptr; }

        cudaMalloc(&d_fwd_offsets, (N + 1)                        * sizeof(int));
        cudaMalloc(&d_fwd_edges,   std::max(1, fwd_csr.num_edges) * sizeof(int));
        cudaMalloc(&d_bwd_offsets, (N + 1)                        * sizeof(int));
        cudaMalloc(&d_bwd_edges,   std::max(1, bwd_csr.num_edges) * sizeof(int));

        cudaMemcpy(d_fwd_offsets, fwd_csr.offsets.data(), (N+1) * sizeof(int),            cudaMemcpyHostToDevice);
        cudaMemcpy(d_fwd_edges,   fwd_csr.edges.data(),   fwd_csr.num_edges * sizeof(int),cudaMemcpyHostToDevice);
        cudaMemcpy(d_bwd_offsets, bwd_csr.offsets.data(), (N+1) * sizeof(int),            cudaMemcpyHostToDevice);
        cudaMemcpy(d_bwd_edges,   bwd_csr.edges.data(),   bwd_csr.num_edges * sizeof(int),cudaMemcpyHostToDevice);

        DeviceGraph h_fwd_g, h_bwd_g;
        h_fwd_g = {d_fwd_offsets, d_fwd_edges, N, fwd_csr.num_edges};
        h_bwd_g = {d_bwd_offsets, d_bwd_edges, N, bwd_csr.num_edges};

        cudaMalloc(&d_fwd_graph, sizeof(DeviceGraph));
        cudaMalloc(&d_bwd_graph, sizeof(DeviceGraph));
        cudaMemcpy(d_fwd_graph, &h_fwd_g, sizeof(DeviceGraph), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bwd_graph, &h_bwd_g, sizeof(DeviceGraph), cudaMemcpyHostToDevice);

        std::cout << "  DeviceGraph: fwd=" << fwd_csr.num_edges
                  << " bwd=" << bwd_csr.num_edges << " edges\n";
    }

    // =========================================================================
    // collectNewEdgesFromDevice  (range variant)
    //
    // Copies papers in [abs_start, abs_end) from device, appends their
    // (deduplicated) citation edges to h_edges, and updates h_papers.
    // =========================================================================
    void collectNewEdgesFromDevice(int abs_start, int abs_end) {
        const int count = abs_end - abs_start;
        if (count <= 0) return;

        std::vector<Paper> batch_papers(count);
        cudaMemcpy(batch_papers.data(),
                   d_papers + abs_start,
                   count * sizeof(Paper),
                   cudaMemcpyDeviceToHost);

        for (int i = 0; i < count; i++) {
            const int idx = abs_start + i;
            h_papers[idx] = batch_papers[i];

            const Paper::State& s = batch_papers[i].state;
            std::set<int> seen;
            for (int j = 0; j < Paper::MAX_CITATIONS; j++) {
                int dst = s.cited_papers[j];
                if (dst >= 0 && seen.find(dst) == seen.end()) {
                    seen.insert(dst);
                    h_edges.emplace_back(s.id, dst);
                }
            }
        }
    }

    // =========================================================================
    // computeAndUploadRecencyArr
    // Mirrors CPP FillRecencyArr — computed over old_pop only; new slots zeroed.
    // =========================================================================
    void computeAndUploadRecencyArr(int old_pop, int total_pop) {
        cudaMemset(d_recency_arr, 0, total_pop * sizeof(float));
        if (old_pop <= 0) return;

        std::map<int, int> year_count;
        double unique_year_sum = 0.0;
        for (int i = 0; i < old_pop; i++) {
            int ydiff = current_year - h_papers[i].state.year;
            if (year_count.find(ydiff) == year_count.end()) {
                // GPU uses 0.0 for unmapped year-diffs (not 0.01).
                // Using a non-zero fallback inflates recency scores for years
                // outside the recency_probabilities file, diverging from GPU.
                double prob = recency_probs_map.count(ydiff) ? recency_probs_map.at(ydiff) : 0.0;
                unique_year_sum += prob;
            }
            year_count[ydiff]++;
        }
        if (unique_year_sum < 1e-15) unique_year_sum = 1.0;

        std::vector<float> h_rec(old_pop);
        for (int i = 0; i < old_pop; i++) {
            int ydiff = current_year - h_papers[i].state.year;
            double prob = recency_probs_map.count(ydiff) ? recency_probs_map.at(ydiff) : 0.0;
            int    cnt  = year_count.count(ydiff)        ? year_count.at(ydiff)         : 1;
            h_rec[i] = static_cast<float>(prob / cnt / unique_year_sum);
        }
        cudaMemcpy(d_recency_arr, h_rec.data(), old_pop * sizeof(float), cudaMemcpyHostToDevice);
    }

    // =========================================================================
    // computeAndUploadNormalizedScoreArrays
    // Mirrors CPP CalculateScores — computed over old_pop only; new slots zeroed.
    // =========================================================================
    void computeAndUploadNormalizedScoreArrays(int old_pop, int total_pop) {
        cudaMemset(d_pa_arr_norm,  0, total_pop * sizeof(float));
        cudaMemset(d_fit_arr_norm, 0, total_pop * sizeof(float));
        if (old_pop <= 0) return;

        float *d_raw_pa = nullptr, *d_raw_fit = nullptr;
        cudaMalloc(&d_raw_pa,  old_pop * sizeof(float));
        cudaMalloc(&d_raw_fit, old_pop * sizeof(float));

        constexpr float GAMMA = Paper::GAMMA;
        int bs = 256, gs = (old_pop + bs - 1) / bs;
        computeRawScoresKernel<<<gs, bs>>>(d_papers, old_pop, GAMMA, d_bwd_graph,
                                           d_raw_pa, d_raw_fit);
        cudaDeviceSynchronize();

        std::vector<float> h_raw_pa(old_pop), h_raw_fit(old_pop);
        cudaMemcpy(h_raw_pa .data(), d_raw_pa,  old_pop * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_raw_fit.data(), d_raw_fit, old_pop * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_raw_pa);
        cudaFree(d_raw_fit);

        double sum_pa = 0.0, sum_fit = 0.0;
        for (int i = 0; i < old_pop; i++) { sum_pa += h_raw_pa[i]; sum_fit += h_raw_fit[i]; }
        if (sum_pa  < 1e-30) sum_pa  = 1.0;
        if (sum_fit < 1e-30) sum_fit = 1.0;
        for (int i = 0; i < old_pop; i++) {
            h_raw_pa [i] = static_cast<float>(h_raw_pa [i] / sum_pa);
            h_raw_fit[i] = static_cast<float>(h_raw_fit[i] / sum_fit);
        }

        cudaMemcpy(d_pa_arr_norm,  h_raw_pa .data(), old_pop * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_fit_arr_norm, h_raw_fit.data(), old_pop * sizeof(float), cudaMemcpyHostToDevice);
    }

    // =========================================================================
    // allocNeighborhoodSlab
    // Allocate BFS slab for `batch_size` threads.
    // HASH_SIZE+1 per thread to hold the double-hash insertion counter.
    // =========================================================================
    NeighborhoodSlab allocNeighborhoodSlab(int batch_size) {
        NeighborhoodSlab slab;
        slab.num_threads = batch_size;

        const long long H = BFSConst::HASH_SIZE + 1;   // +1 for counter slot
        const long long F = BFSConst::MAX_FRONTIER;
        const long long O = BFSConst::MAX_1HOP;
        const long long T = BFSConst::MAX_2HOP;

        cudaMalloc(&slab.d_visited_hash,  (long long)batch_size * H * sizeof(int));
        cudaMalloc(&slab.d_curr_frontier, (long long)batch_size * F * sizeof(int));
        cudaMalloc(&slab.d_next_frontier, (long long)batch_size * F * sizeof(int));
        cudaMalloc(&slab.d_one_hop,       (long long)batch_size * O * sizeof(int));
        cudaMalloc(&slab.d_two_hop,       (long long)batch_size * T * sizeof(int));

        // Initialise hash tables: all slots = -1 (empty), counter at end = 0
        cudaMemset(slab.d_visited_hash, 0xFF, (long long)batch_size * H * sizeof(int));

        const long long total_mb = (long long)batch_size * (H + 2*F + O + T)
                                   * sizeof(int) / (1024*1024);
        std::cout << "  NeighborhoodSlab: " << batch_size << " threads × "
                  << (H + 2*F + O + T)*4/1024 << " KB = " << total_mb << " MB\n";
        return slab;
    }

    // =========================================================================
    // initialize
    // =========================================================================
    void initialize() {
        std::cout << "\n=== Initializing Citation ABM ===\n";

        parseNodelist();
        parseEdgelist();

        if (!nodeYearMap.empty()) {
            int max_yr = 0;
            for (const auto& [seq, yr] : nodeYearMap) max_yr = std::max(max_yr, yr);
            current_year = max_yr + 1;
            std::cout << "Seed max year: " << max_yr
                      << "  → simulation starts at " << current_year << "\n";
        }

        int tmp = config.initial_population;
        for (int i = 0; i < config.num_cycles; i++)
            tmp += static_cast<int>(std::ceil(tmp * config.growth_rate));
        max_population = tmp + 2000;
        std::cout << "Max population: " << max_population << "\n";

        cudaMalloc(&d_papers,        max_population * sizeof(Paper));
        cudaMalloc(&d_recency_probs, 51 * sizeof(float));
        cudaMalloc(&d_recency_arr,   max_population * sizeof(float));
        cudaMalloc(&d_pa_arr_norm,   max_population * sizeof(float));
        cudaMalloc(&d_fit_arr_norm,  max_population * sizeof(float));
        cudaMalloc(&d_rand_states,   max_population * sizeof(curandState));

        // Build and upload the fitness power-law CDF (values 1..1000).
        // Matches AssignPeakFitnessValues: p(i) ∝ 6.3742991333 * 0.072 * i^-1.634
        {
            constexpr int N = 1000;
            constexpr double scale = 6.3742991333, constant = 0.072, exponent = -1.634;
            std::vector<float> h_cdf(N);
            double sum = 0.0;
            for (int i = 1; i <= N; i++)
                sum += scale * constant * std::pow((double)i, exponent);
            double cum = 0.0;
            for (int i = 1; i <= N; i++) {
                cum += scale * constant * std::pow((double)i, exponent);
                h_cdf[i - 1] = static_cast<float>(cum / sum);
            }
            cudaMalloc(&d_fitness_cdf, N * sizeof(float));
            cudaMemcpy(d_fitness_cdf, h_cdf.data(), N * sizeof(float), cudaMemcpyHostToDevice);
        }

        float h_rec[51];
        for (int i = 0; i < 51; i++)
            h_rec[i] = recency_probs_map.count(i) ? recency_probs_map[i] : 0.01f;
        cudaMemcpy(d_recency_probs, h_rec, 51 * sizeof(float), cudaMemcpyHostToDevice);

        int bs = 256, gs = (max_population + bs - 1) / bs;
        initCurandKernel<<<gs, bs>>>(d_rand_states, max_population, (unsigned long long)time(NULL));
        cudaDeviceSynchronize();

        createInitialPopulation();
        buildDeviceGraphs();

        std::cout << "Initialization complete\n\n";
    }

    // =========================================================================
    // createInitialPopulation
    // =========================================================================
    void createInitialPopulation() {
        h_papers.reserve(max_population);

        std::vector<double> fit_probs;
        {
            constexpr double scale = 6.3742991333, constant = 0.072, exponent = -1.634;
            for (int i = 1; i <= 1000; i++)
                fit_probs.push_back(scale * constant * std::pow((double)i, exponent));
        }
        std::discrete_distribution<int> fit_dist(fit_probs.begin(), fit_probs.end());
        std::uniform_real_distribution<float> w_dist(0.0f, 1.0f);

        for (int i = 0; i < config.initial_population; i++) {
            Paper p;
            p.state.id      = i;
            p.state.year    = nodeYearMap.count(i) ? nodeYearMap[i] : current_year;
            p.state.is_seed = true;

            int fit = fit_dist(rng) + 1;
            p.state.fitness_peak_value    = fit;
            p.state.fitness_lag_duration  = 0;
            p.state.fitness_peak_duration = 1000;
            p.state.current_fitness       = static_cast<float>(fit);

            if (config.preferential_weight >= 0) {
                float tot = config.preferential_weight + config.recency_weight + config.fitness_weight;
                p.state.pa_weight      = config.preferential_weight / tot;
                p.state.recency_weight = config.recency_weight      / tot;
                p.state.fitness_weight = config.fitness_weight      / tot;
            } else {
                float a = w_dist(rng), b = w_dist(rng), c = w_dist(rng), tot = a+b+c;
                p.state.pa_weight      = a / tot;
                p.state.recency_weight = b / tot;
                p.state.fitness_weight = c / tot;
            }
            p.state.alpha               = (config.alpha >= 0) ? config.alpha : w_dist(rng);
            p.state.in_degree           = 0;
            p.state.out_degree          = 0;
            p.state.assigned_out_degree = 0;
            p.state.generator_node      = -1;
            p.state.citations_made      = 0;
            p.state.is_new              = false;
            h_papers.push_back(p);
        }

        current_population = config.initial_population;
        cudaMemcpy(d_papers, h_papers.data(),
                   current_population * sizeof(Paper), cudaMemcpyHostToDevice);
        std::cout << "Seed population: " << current_population << " papers\n";
    }

    // =========================================================================
    // addNewPapers
    // =========================================================================
    std::vector<int> addNewPapers(int num_new) {
        if (current_population + num_new > max_population) {
            std::cerr << "ERROR: would exceed max_population!\n"; return {};
        }

        std::vector<double> fit_probs;
        {
            constexpr double scale = 6.3742991333, constant = 0.072, exponent = -1.634;
            for (int i = 1; i <= 1000; i++)
                fit_probs.push_back(scale * constant * std::pow((double)i, exponent));
        }
        std::discrete_distribution<int> fit_dist(fit_probs.begin(), fit_probs.end());
        std::uniform_real_distribution<float> w_dist(0.0f, 1.0f);
        std::uniform_int_distribution<> od_idx(0, (int)out_degree_bag.size() - 1);

        std::vector<Paper> new_papers;
        std::vector<int>   new_indices;
        new_papers.reserve(num_new);
        new_indices.reserve(num_new);

        for (int i = 0; i < num_new; i++) {
            Paper p;
            p.state.id      = current_population + i;
            p.state.year    = current_year;
            p.state.is_seed = false;

            int fit = fit_dist(rng) + 1;
            p.state.fitness_peak_value    = fit;
            p.state.fitness_lag_duration  = 0;
            p.state.fitness_peak_duration = 1000;
            p.state.current_fitness       = static_cast<float>(fit);

            if (config.preferential_weight >= 0) {
                float tot = config.preferential_weight + config.recency_weight + config.fitness_weight;
                p.state.pa_weight      = config.preferential_weight / tot;
                p.state.recency_weight = config.recency_weight      / tot;
                p.state.fitness_weight = config.fitness_weight      / tot;
            } else {
                float a = w_dist(rng), b = w_dist(rng), c = w_dist(rng), tot = a+b+c;
                p.state.pa_weight      = a / tot;
                p.state.recency_weight = b / tot;
                p.state.fitness_weight = c / tot;
            }
            p.state.alpha               = (config.alpha >= 0) ? config.alpha : w_dist(rng);
            p.state.assigned_out_degree = out_degree_bag[od_idx(rng)];
            p.state.in_degree           = 0;
            p.state.out_degree          = 0;
            p.state.generator_node      = -1;
            p.state.citations_made      = 0;
            p.state.is_new              = true;

            new_papers.push_back(p);
            new_indices.push_back(current_population + i);
            h_papers.push_back(p);
        }

        cudaMemcpy(d_papers + current_population, new_papers.data(),
                   num_new * sizeof(Paper), cudaMemcpyHostToDevice);
        current_population += num_new;
        return new_indices;
    }

    // =========================================================================
    // simulateOneYear  (batched)
    //
    // Steps:
    //   1. Update fitness for all existing papers
    //   2. Add ALL new papers to host + device
    //   3. Select generators for ALL new papers (one kernel)
    //   4. Build CSR (covers full population including new papers)
    //   5. Compute score arrays over OLD population (mirrors CPP ordering)
    //   6. For each batch:
    //        a. Upload batch index slice
    //        b. Allocate BFS slab for this batch only  ← peak memory reduced
    //        c. Run citation kernel
    //        d. Free slab immediately
    //        e. Collect batch edges → h_edges
    //        f. Rebuild CSR so next batch BFS sees this batch's citations
    //   7. Update in/out degree for all papers
    //   8. Clear is_new flags
    // =========================================================================
    void simulateOneYear() {
        const int prev_population = current_population;
        int bs = 256, gs;

        std::cout << "\n--- Year " << current_year
                  << " (pop: " << current_population << ") ---\n";
        auto t0 = std::chrono::high_resolution_clock::now();

        // ── Step 1: Update fitness ────────────────────────────────────────────
        gs = (current_population + bs - 1) / bs;
        updateFitnessKernel<<<gs, bs>>>(d_papers, current_population, current_year);
        cudaDeviceSynchronize();

        // ── Step 2: Add ALL new papers at once ────────────────────────────────
        const int num_new = static_cast<int>(std::ceil(current_population * config.growth_rate));
        std::cout << "  Adding " << num_new << " new papers\n";
        const std::vector<int> new_indices = addNewPapers(num_new);
        if (new_indices.empty()) { current_year++; return; }

        // ── Step 3: Generator selection over new papers only ─────────────────
        // old_population = prev_population: generator must be an OLD node so BFS
        // starts from a connected vertex and finds a meaningful neighbourhood.
        gs = (current_population + bs - 1) / bs;
        selectGeneratorKernel<<<gs, bs>>>(d_papers, current_population, prev_population, d_rand_states);
        cudaDeviceSynchronize();

        // ── Step 4: Build CSR covering all papers (new ones have no edges yet)
        buildDeviceGraphs();

        // ── Step 5: Score arrays over OLD population only ─────────────────────
        // New papers get score=0, matching CPP where CalculateScores runs
        // before new nodes are added to the graph.
        computeAndUploadRecencyArr(prev_population, current_population);
        computeAndUploadNormalizedScoreArrays(prev_population, current_population);

        // ── Step 6: Batched BFS + citation kernel ─────────────────────────────
        const int num_batches = std::max(1, config.bfs_num_batches);
        const int batch_size  = (num_new + num_batches - 1) / num_batches;

        std::cout << "  Processing " << num_new << " papers in "
                  << num_batches << " batches of ~" << batch_size << "\n";

        for (int b = 0; b < num_batches; ++b) {

            const int batch_start  = b * batch_size;
            const int batch_end    = std::min(batch_start + batch_size, num_new);
            const int this_batch   = batch_end - batch_start;
            if (this_batch <= 0) break;

            auto tb0 = std::chrono::high_resolution_clock::now();

            // ── 6a: Upload this batch's index slice ───────────────────────────
            int* d_batch_indices = nullptr;
            cudaMalloc(&d_batch_indices, this_batch * sizeof(int));
            cudaMemcpy(d_batch_indices,
                       new_indices.data() + batch_start,
                       this_batch * sizeof(int),
                       cudaMemcpyHostToDevice);

            // ── 6b: Allocate slab for this batch only ─────────────────────────
            NeighborhoodSlab slab = allocNeighborhoodSlab(this_batch);

            // ── 6c: Run citation kernel ───────────────────────────────────────
            const int bfs_gs = (this_batch + bs - 1) / bs;
            makeCitationsWithBFSKernel<<<bfs_gs, bs>>>(
                d_papers,
                d_batch_indices,
                this_batch,
                current_population,
                prev_population,
                d_fwd_graph,
                d_bwd_graph,
                slab,
                current_year,
                d_pa_arr_norm,
                d_fit_arr_norm,
                d_recency_arr,
                config.fully_random_citations,
                config.same_year_proportion,
                d_rand_states);
            cudaDeviceSynchronize();
            {
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    std::cerr << "FATAL: batch " << b
                              << " makeCitationsWithBFSKernel: "
                              << cudaGetErrorString(err) << "\n";
                    std::exit(1);
                }
            }

            // ── 6d: Free slab immediately — don't accumulate across batches ───
            slab.free();
            cudaFree(d_batch_indices);

            // ── 6e: Collect this batch's edges to host ────────────────────────
            const int abs_start = prev_population + batch_start;
            const int abs_end   = prev_population + batch_end;
            collectNewEdgesFromDevice(abs_start, abs_end);

            // NOTE: intentionally NO buildDeviceGraphs() here.
            //
            // Previous code rebuilt CSR after every batch so later-batch BFS
            // could see earlier-batch same-year agents. This caused systematic
            // over-citation of high-fitness (f3/f4/f5) same-year agents:
            //   batch-0 f3 agent cites old node X → X's backward now includes
            //   f3 agent → batch-1 generator reaches f3 agent via 2-hop BFS →
            //   WRS score ∝ fitness^3 → f3 agent selected ~1000x more than f1.
            // Over 10 batches this roughly doubled year-t agent in-degree and
            // biased it heavily toward f3+, unlike GPU.
            //
            // GPU (gpu-opt) freezes the CSR for the entire year: all year-t
            // citations are computed from the graph state BEFORE any year-t
            // edges are added. Same-year agents are BFS-invisible; they are
            // only reachable via the uniform-random same-year Stage-1 mechanism,
            // which is fitness-group-agnostic by design.
            //
            // Fix: keep CSR frozen across all batches of the same year.
            // A single rebuild after the batch loop (Step 6g below) gives
            // updateInDegree/updateOutDegree the correct final graph.

            auto tb1 = std::chrono::high_resolution_clock::now();
            std::cout << "  Batch " << b << " [" << batch_start << "," << batch_end << ")"
                      << " done in "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(tb1 - tb0).count()
                      << " ms  h_edges=" << h_edges.size() << "\n";
        }


        // -- Step 6g: Rebuild CSR once with ALL year-t edges -----------------------
        // This single rebuild incorporates all batch edges accumulated by
        // collectNewEdgesFromDevice above. Doing it here rather than inside
        // the batch loop ensures:
        //   (a) all batches BFS used the pre-year CSR (no same-year WRS bias)
        //   (b) updateInDegree/updateOutDegree see the correct final graph
        //   (c) next year Step-4 buildDeviceGraphs sees the correct base
        buildDeviceGraphs();


        // -- Step 6h: Re-assign fitness_peak_value for all new agents (POST-citation) --
        // Mirrors GPU execute() which calls AssignPeakFitnessValues(graph, new_nodes_vec)
        // AFTER all citations are generated each year.  GPU then calls
        // AssignFitnessLagDuration (hard-coded 0) and AssignFitnessPeakDuration
        // (hard-coded 1000) -- already the defaults in Paper::State, so no action needed.
        //
        // Effect: the fit_peak_value stored in the aux file matches GPU semantics:
        // it is a fresh independent sample drawn AFTER citations, exactly as gpu-opt.
        {
            const int num_new = (int)new_indices.size();
            const int start   = prev_population;   // first new-agent sequential index
            gs = (num_new + bs - 1) / bs;
            reassignFitnessKernel<<<gs, bs>>>(
                d_papers, start, num_new,
                d_rand_states,
                d_fitness_cdf,
                1000);   // fitness values drawn from 1..1000
            cudaDeviceSynchronize();
        }

        // ── Step 7: Update in/out degree ──────────────────────────────────────
        gs = (current_population + bs - 1) / bs;
        updateInDegreeFromCSRKernel <<<gs, bs>>>(d_papers, current_population, d_bwd_graph);
        updateOutDegreeFromCSRKernel<<<gs, bs>>>(d_papers, current_population, d_fwd_graph);
        cudaDeviceSynchronize();

        // ── Step 8: Clear is_new flags ────────────────────────────────────────
        resetNewStatusKernel<<<gs, bs>>>(d_papers, current_population);
        cudaDeviceSynchronize();

        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "  Year " << current_year << " total: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
                  << " ms  (pop: " << current_population << ")\n";

        current_year++;
    }

    // =========================================================================
    // run
    // =========================================================================
    void run() {
        std::cout << "\n=== Simulation Start ===\n";
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int c = 0; c < config.num_cycles; c++) simulateOneYear();
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "\n=== Simulation Complete ===\n"
                  << "Total: " << std::chrono::duration_cast<std::chrono::seconds>(t1-t0).count()
                  << " s  Final pop: " << current_population << "\n";
    }

    // =========================================================================
    // exportResults
    // =========================================================================
    void exportResults() {
        std::cout << "\n=== Exporting Results ===\n";

        cudaMemcpy(h_papers.data(), d_papers,
                   current_population * sizeof(Paper), cudaMemcpyDeviceToHost);

        std::string ef = config.output_file.empty() ? "edges_output.csv" : config.output_file;
        std::ofstream eo(ef);
        eo << "#source,target\n";

        // ── Export from h_edges, not from cited_papers ───────────────────────
        //
        // CRITICAL BUG FIX: The previous implementation iterated cited_papers[j]
        // for each paper, but cited_papers is only populated for AGENT papers
        // (inside makeCitations).  SEED papers never call makeCitations, so their
        // cited_papers[] array is all -1.  This silently dropped all 899,050 seed
        // edges from the output, leaving only the 892,263 agent edges.
        //
        // Fix: export from h_edges, which is the single authoritative edge list
        // accumulated by:
        //   • parseEdgelist()          → seed→seed and seed→agent edges
        //   • collectNewEdgesFromDevice() → agent outgoing edges (each year/batch)
        //
        // This mirrors GPU's WriteGraph() which writes every entry in
        // forward_adj_map (both seed and agent edges), converting sequential
        // IDs back to original IDs via reverse_continuous_node_mapping.
        //
        // ID mapping:
        //   • seq 0 … initial_pop-1  → original paper ID (in reverse map)
        //   • seq initial_pop … N-1  → not in reverse map → fall back to seq ID
        //     (agents have no original-ID mapping; GPU assigns fresh integer IDs).
        // ─────────────────────────────────────────────────────────────────────
        const auto& rev = reverse_continuous_node_mapping;

        // Pre-build buffer for fast I/O (mirrors GPU WriteGraph buffered write)
        std::string buf;
        buf.reserve(h_edges.size() * 18);   // ~18 bytes per "src,dst\n" pair

        long long total_edges = 0;
        for (const auto& [src, dst] : h_edges) {
            int orig_src = rev.count(src) ? rev.at(src) : src;
            int orig_dst = rev.count(dst) ? rev.at(dst) : dst;

            // fast integer-to-string (avoids std::to_string heap allocs)
            char tmp[32];
            int len = std::snprintf(tmp, sizeof(tmp), "%d,%d\n", orig_src, orig_dst);
            buf.append(tmp, len);
            ++total_edges;

            // Flush every 8 MB to bound memory usage on large graphs
            if (buf.size() > (8u << 20)) {
                eo.write(buf.data(), buf.size());
                buf.clear();
            }
        }
        if (!buf.empty()) eo.write(buf.data(), buf.size());
        eo.close();

        std::cout << "Edges saved: " << ef << " (" << total_edges << ")\n";

        writeAttributes();
    }

    // =========================================================================
    // writeAttributes
    // =========================================================================
    void writeAttributes() {
        std::string af = config.auxiliary_information_file.empty()
                         ? "aux_info.csv" : config.auxiliary_information_file;
        std::ofstream out(af);
        if (!out) { std::cerr << "Cannot open: " << af << "\n"; return; }

        out << "node_id,type,year,"
               "pa_weight,rec_weight,fit_weight,"
               "fit_lag_duration,fit_peak_value,fit_peak_duration,"
               "alpha,in_degree,out_degree,assigned_out_degree,"
               "planted_nodes_line_number,generator_node_string\n";

        constexpr size_t FLUSH = 1u << 20;
        std::ostringstream buf;
        buf.precision(10);

        for (int i = 0; i < current_population; i++) {
            const Paper::State& s = h_papers[i].state;
            const char* type = s.is_seed ? "seed" : "agent";

            double pa_w = -1, rec_w = -1, fit_w = -1, alpha = -1;
            int aod = -1, pnln = -1;
            std::string gen_str = "no_generators";

            if (!s.is_seed) {
                pa_w    = s.pa_weight;
                rec_w   = s.recency_weight;
                fit_w   = s.fitness_weight;
                alpha   = s.alpha;
                aod     = s.assigned_out_degree;
                gen_str = (s.generator_node >= 0)
                          ? std::to_string(s.generator_node) : "no_generators";
            }

            int node_id = reverse_continuous_node_mapping.count(s.id)
                          ? reverse_continuous_node_mapping.at(s.id) : s.id;

            buf << node_id                 << ','
                << type                    << ','
                << s.year                  << ','
                << pa_w                    << ','
                << rec_w                   << ','
                << fit_w                   << ','
                << s.fitness_lag_duration  << ','
                << s.fitness_peak_value    << ','
                << s.fitness_peak_duration << ','
                << alpha                   << ','
                << s.in_degree             << ','
                << s.out_degree            << ','
                << aod                     << ','
                << pnln                    << ','
                << gen_str                 << '\n';

            if ((size_t)buf.tellp() > FLUSH) {
                out << buf.str(); buf.str(""); buf.clear();
            }
        }
        out << buf.str();
        out.close();
        std::cout << "Attributes saved: " << af
                  << " (" << current_population << " rows)\n";
    }

    // =========================================================================
    // cleanup
    // =========================================================================
    void cleanup() {
        if (d_papers)        { cudaFree(d_papers);        d_papers        = nullptr; }
        if (d_recency_probs) { cudaFree(d_recency_probs); d_recency_probs = nullptr; }
        if (d_recency_arr)   { cudaFree(d_recency_arr);   d_recency_arr   = nullptr; }
        if (d_pa_arr_norm)   { cudaFree(d_pa_arr_norm);   d_pa_arr_norm   = nullptr; }
        if (d_fit_arr_norm)  { cudaFree(d_fit_arr_norm);  d_fit_arr_norm  = nullptr; }
        if (d_rand_states)   { cudaFree(d_rand_states);   d_rand_states   = nullptr; }
        if (d_fitness_cdf)   { cudaFree(d_fitness_cdf);   d_fitness_cdf   = nullptr; }
        if (d_fwd_offsets)   { cudaFree(d_fwd_offsets);   d_fwd_offsets   = nullptr; }
        if (d_fwd_edges)     { cudaFree(d_fwd_edges);     d_fwd_edges     = nullptr; }
        if (d_bwd_offsets)   { cudaFree(d_bwd_offsets);   d_bwd_offsets   = nullptr; }
        if (d_bwd_edges)     { cudaFree(d_bwd_edges);     d_bwd_edges     = nullptr; }
        if (d_fwd_graph)     { cudaFree(d_fwd_graph);     d_fwd_graph     = nullptr; }
        if (d_bwd_graph)     { cudaFree(d_bwd_graph);     d_bwd_graph     = nullptr; }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    std::cout << "Citation Network ABM – MASS CUDA\n"
              << "=================================\n\n";

    argparse::ArgumentParser prog("citation_abm_mass");
    prog.add_argument("--edgelist").required();
    prog.add_argument("--nodelist").required();
    prog.add_argument("--out-degree-bag").required();
    prog.add_argument("--recency-probabilities").required();
    prog.add_argument("--alpha").default_value(0.5).scan<'g', double>();
    prog.add_argument("--fully-random-citations").default_value(0.0).scan<'g', double>();
    prog.add_argument("--same-year-proportion").default_value(0.0).scan<'g', double>();
    prog.add_argument("--preferential-weight").default_value(1.0).scan<'g', double>();
    prog.add_argument("--recency-weight").default_value(1.0).scan<'g', double>();
    prog.add_argument("--fitness-weight").default_value(1.0).scan<'g', double>();
    prog.add_argument("--num-cycles").default_value(100).scan<'i', int>();
    prog.add_argument("--growth-rate").default_value(0.01).scan<'g', double>();
    prog.add_argument("--bfs-batches").default_value(10).scan<'i', int>();
    prog.add_argument("--output-file").default_value(std::string("edges_output.csv"));
    prog.add_argument("--auxiliary-information-file").default_value(std::string("aux_info.csv"));
    prog.add_argument("--log-file").default_value(std::string("simulation.log"));
    prog.add_argument("--num-processors").default_value(1).scan<'i', int>();
    prog.add_argument("--log-level").default_value(1).scan<'i', int>();

    try {
        prog.parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n" << prog; return 1;
    }

    ABMConfig cfg;
    cfg.edgelist_file              = prog.get<std::string>("--edgelist");
    cfg.nodelist_file              = prog.get<std::string>("--nodelist");
    cfg.out_degree_bag_file        = prog.get<std::string>("--out-degree-bag");
    cfg.recency_probabilities_file = prog.get<std::string>("--recency-probabilities");
    cfg.alpha                      = prog.get<double>("--alpha");
    cfg.fully_random_citations     = prog.get<double>("--fully-random-citations");
    cfg.same_year_proportion       = prog.get<double>("--same-year-proportion");
    cfg.preferential_weight        = prog.get<double>("--preferential-weight");
    cfg.recency_weight             = prog.get<double>("--recency-weight");
    cfg.fitness_weight             = prog.get<double>("--fitness-weight");
    cfg.num_cycles                 = prog.get<int>("--num-cycles");
    cfg.growth_rate                = prog.get<double>("--growth-rate");
    cfg.bfs_num_batches            = prog.get<int>("--bfs-batches");
    cfg.output_file                = prog.get<std::string>("--output-file");
    cfg.auxiliary_information_file = prog.get<std::string>("--auxiliary-information-file");
    cfg.log_file                   = prog.get<std::string>("--log-file");
    cfg.num_processors             = prog.get<int>("--num-processors");
    cfg.log_level                  = prog.get<int>("--log-level") - 1;

    CitationABM sim(cfg);
    sim.initialize();
    sim.run();
    sim.exportResults();

    std::cout << "\nDone.\n";
    return 0;
}
