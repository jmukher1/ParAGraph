#include "kernel_erdos_renyi.cuh"
#include "utils.cuh"
#include <cmath>
#include <chrono>
#include <iostream>

// =============================================================================
// Profiling infrastructure -- self-contained, mirroring the PA model's
// timing-breakdown approach (see kernel.cu's execute()/EpochTiming from
// earlier work), but deliberately NOT depending on epoch_profiler.cuh:
// this file's actual build doesn't include that header (checked directly --
// kernel.cu in this same codebase snapshot doesn't reference it either), so
// depending on it here would risk an unresolved/missing-header build
// failure rather than a working profiling feature. These are named with an
// "ER" prefix specifically to avoid any collision if epoch_profiler.cuh
// *does* exist and get included elsewhere in the same translation unit in
// your actual build.
// =============================================================================

// ── Host-side wall-clock timer (std::chrono) ────────────────────────────────
struct ERHostTimer {
    std::chrono::steady_clock::time_point t_start;
    void start() { t_start = std::chrono::steady_clock::now(); }
    double stop_ms() {
        auto t_end = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;
    }
};

// ── GPU-side kernel timer (cudaEvent-based, records on a stream) ───────────
struct ERGpuTimer {
    cudaEvent_t ev_start, ev_stop;
    void start(cudaStream_t stream) {
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_stop);
        cudaEventRecord(ev_start, stream);
    }
    // Caller must have already cudaStreamSynchronize'd (or otherwise
    // ensured the stream reached ev_stop) before calling this, since
    // cudaEventElapsedTime requires both events to have completed.
    double stop(cudaStream_t stream) {
        cudaEventRecord(ev_stop, stream);
        cudaEventSynchronize(ev_stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_stop);
        return (double)ms;
    }
};

// ── Per-epoch timing breakdown for the ER model ─────────────────────────────
struct ERTiming {
    int    year = 0;
    int    N = 0;              // current_graph_size at start of epoch
    int    delta = 0;          // num_new_nodes this epoch
    double t_node_init    = 0.0;  // new-node attribute init loop (host)
    double t_upload       = 0.0;  // new_nodes_arr H2D copy
    double t_capacity     = 0.0;  // computeERCapacities (host, cheap but tracked)
    double t_bulk_alloc   = 0.0;  // create_thread_vectors_bulk (edge buffer)
    double t_kernel       = 0.0;  // actual GPU kernel execution, all batches summed
    double t_download     = 0.0;  // append_device_to_host
    double t_cleanup      = 0.0;  // cleanup_vectors_bulk + frees
    double t_graph_update = 0.0;  // commit edges into forward/backward_adj_map + degree update
    long long edges_out   = 0;

    double epoch_total() const {
        return t_node_init + t_upload + t_capacity + t_bulk_alloc +
               t_kernel + t_download + t_cleanup + t_graph_update;
    }
};

static void print_er_epoch_report(const std::vector<ERTiming>& epochs, double e2e_ms,
                                  const std::string& model_label, double growth_rate) {
    double sum_node_init = 0, sum_upload = 0, sum_capacity = 0, sum_bulk_alloc = 0;
    double sum_kernel = 0, sum_download = 0, sum_cleanup = 0, sum_graph_update = 0;
    double sum_total = 0;
    long long sum_edges = 0;

    for (const auto& ep : epochs) {
        sum_node_init    += ep.t_node_init;
        sum_upload       += ep.t_upload;
        sum_capacity     += ep.t_capacity;
        sum_bulk_alloc   += ep.t_bulk_alloc;
        sum_kernel       += ep.t_kernel;
        sum_download     += ep.t_download;
        sum_cleanup      += ep.t_cleanup;
        sum_graph_update += ep.t_graph_update;
        sum_total        += ep.epoch_total();
        sum_edges        += ep.edges_out;
    }

    auto pct = [&](double part) { return sum_total > 0.0 ? (100.0 * part / sum_total) : 0.0; };

    printf("\n========================================================================================================================\n");
    printf("  ER MODEL EPOCH BREAKDOWN  model=%s  growth=%.1f%%  epochs=%zu\n",
           model_label.c_str(), 100.0 * growth_rate, epochs.size());
    printf("========================================================================================================================\n");
    printf("  %-45s : %10.0f ms  (%5.1f%%)\n", "New-node attribute init (host)", sum_node_init, pct(sum_node_init));
    printf("  %-45s : %10.0f ms  (%5.1f%%)\n", "Upload new_nodes_arr (H2D)", sum_upload, pct(sum_upload));
    printf("  %-45s : %10.0f ms  (%5.1f%%)\n", "Capacity computation (host)", sum_capacity, pct(sum_capacity));
    printf("  %-45s : %10.0f ms  (%5.1f%%)\n", "Bulk-allocate edge buffer", sum_bulk_alloc, pct(sum_bulk_alloc));
    printf("  %-45s : %10.0f ms  (%5.1f%%)\n", "GPU kernel (edge sampling)", sum_kernel, pct(sum_kernel));
    printf("  %-45s : %10.0f ms  (%5.1f%%)\n", "Download edges (D2H, clamped read)", sum_download, pct(sum_download));
    printf("  %-45s : %10.0f ms  (%5.1f%%)\n", "Cleanup (bulk frees)", sum_cleanup, pct(sum_cleanup));
    printf("  %-45s : %10.0f ms  (%5.1f%%)\n", "Graph update (adj-map insert + degrees)", sum_graph_update, pct(sum_graph_update));
    printf("------------------------------------------------------------------------------------------------------------------------\n");
    printf("  %-45s : %10.0f ms  (%5.1f%%)\n", "SIMULATION TOTAL (sum of epochs)", sum_total, pct(sum_total));
    printf("  %-45s : %10lld\n", "Total edges created", sum_edges);
    printf("========================================================================================================================\n");

    printf("\n  Per-epoch detail:\n");
    printf("  %-6s %10s %10s | %8s %8s %8s %8s %8s %8s %8s %8s | %9s %9s\n",
           "year", "N", "delta", "init", "upload", "capac", "alloc", "kernel", "dl", "clean", "graph", "total", "edges");
    for (const auto& ep : epochs) {
        printf("  %-6d %10d %10d | %8.0f %8.0f %8.0f %8.0f %8.0f %8.0f %8.0f %8.0f | %9.0f %9lld\n",
               ep.year, ep.N, ep.delta,
               ep.t_node_init, ep.t_upload, ep.t_capacity, ep.t_bulk_alloc,
               ep.t_kernel, ep.t_download, ep.t_cleanup, ep.t_graph_update,
               ep.epoch_total(), ep.edges_out);
    }
    printf("\n  Full pipeline total (E2E, includes graph parse + output write not shown above): %.0f ms\n", e2e_ms);
}

// ============================================================================
// Erdős-Rényi G(n,p) Model
// ============================================================================
// For each new node, create edges to ALL existing nodes with probability p
__global__ void kernelErdosRenyiGNP(
    int start_idx,
    int batch_size,
    int total_N,
    ABM* abm,
    curandState* deviceStates,
    unsigned long long seed,
    int* d_new_nodes_arr,
    device_vector_generic<int2>* d_new_edges_vec_vectors,
    int current_graph_size,
    double edge_probability
) {
    int threads_per_block = blockDim.x * blockDim.y;
    int local_idx = blockIdx.x * threads_per_block +
                    threadIdx.y * blockDim.x + threadIdx.x;
    if (local_idx >= batch_size) return;
    int idx = start_idx + local_idx;
    if (idx >= total_N) return;

    // Initialize RNG for this thread
    curand_init(seed, idx, 0, &deviceStates[local_idx]);
    curandState localState = deviceStates[local_idx];

    int new_node = d_new_nodes_arr[idx];
    device_vector_generic<int2>& edges_vec = d_new_edges_vec_vectors[local_idx];
    
    // For each existing node in the graph, create edge with probability p
    for (int target = 0; target < current_graph_size; target++) {
        float rand_val = curand_uniform(&localState);
        
        if (rand_val < edge_probability) {
            // Create edge: new_node -> target
            int2 edge;
            edge.x = new_node;
            edge.y = target;
            edges_vec.push_back(edge);
        }
    }
    
    // Save RNG state
    deviceStates[local_idx] = localState;
}

// ============================================================================
// Erdős-Rényi Fixed-k Model  
// ============================================================================
// Each new node connects to exactly k random existing nodes (with replacement)
__global__ void kernelErdosRenyiFixedK(
    int start_idx,
    int batch_size,
    int total_N,
    ABM* abm,
    curandState* deviceStates,
    unsigned long long seed,
    int* d_new_nodes_arr,
    device_vector_generic<int2>* d_new_edges_vec_vectors,
    int current_graph_size,
    int edges_per_node
) {
    int threads_per_block = blockDim.x * blockDim.y;
    int local_idx = blockIdx.x * threads_per_block +
                    threadIdx.y * blockDim.x + threadIdx.x;
    if (local_idx >= batch_size) return;
    int idx = start_idx + local_idx;
    if (idx >= total_N) return;

    // Initialize RNG for this thread
    curand_init(seed, idx, 0, &deviceStates[local_idx]);
    curandState localState = deviceStates[local_idx];

    int new_node = d_new_nodes_arr[idx];
    device_vector_generic<int2>& edges_vec = d_new_edges_vec_vectors[local_idx];
    
    // Create exactly edges_per_node edges to random existing nodes
    for (int i = 0; i < edges_per_node; i++) {
        // Select random target from [0, current_graph_size)
        int target = curand(&localState) % current_graph_size;
        
        // Create edge: new_node -> target
        int2 edge;
        edge.x = new_node;
        edge.y = target;
        edges_vec.push_back(edge);
    }
    
    // Save RNG state
    deviceStates[local_idx] = localState;
}

// ============================================================================
// Host function to call appropriate ER kernel
// ============================================================================
void launchErdosRenyiKernel(
    ABM* abm,
    int start_idx,
    int batch_size,
    int total_N,
    int blocks_for_this_batch,
    dim3 threads_per_block,
    cudaStream_t stream,
    curandState* deviceStates,
    unsigned long long seed,
    int* d_new_nodes_arr,
    device_vector_generic<int2>* d_new_edges_vec_vectors,
    int current_graph_size
) {
    double er_prob = abm->get_er_edge_probability();
    int er_k = abm->get_er_edges_per_node();
    
    if (er_prob > 0.0) {
        // G(n,p) model
        kernelErdosRenyiGNP<<<blocks_for_this_batch, threads_per_block, 0, stream>>>(
            start_idx, batch_size, total_N, abm, deviceStates, seed,
            d_new_nodes_arr, d_new_edges_vec_vectors, current_graph_size, er_prob
        );
    } else if (er_k > 0) {
        // Fixed-k model
        kernelErdosRenyiFixedK<<<blocks_for_this_batch, threads_per_block, 0, stream>>>(
            start_idx, batch_size, total_N, abm, deviceStates, seed,
            d_new_nodes_arr, d_new_edges_vec_vectors, current_graph_size, er_k
        );
    }
}

// ============================================================================
// computeERCapacities
//
// Per-new-node buffer sizing for d_new_edges_vec_vectors, computed ONCE
// host-side for the whole batch (not per-thread on the device).
//
// Fixed-k: out-degree is EXACTLY er_k for every node -- no safety margin
// needed, no risk of push_back overflow at all.
//
// G(n,p): out-degree is a Binomial(current_graph_size, p) random variable
// per node. The true worst case is current_graph_size (every possible edge
// succeeds) -- allocating that per node would blow up total memory to
// O(num_new_nodes * current_graph_size), which is exactly the kind of
// waste bulk allocation is supposed to avoid. Instead, size to the
// EXPECTED value plus a generous statistical safety margin (mean + 6
// std-devs via the normal approximation to the binomial, which is
// astronomically unlikely to be exceeded for any reasonable graph size),
// clamped to current_graph_size as the hard ceiling.
//
// Overflow is NOT silent data corruption even if this margin is ever
// exceeded in practice: device_vector_generic::push_back() rejects
// over-capacity pushes with a printf warning, and append_device_to_host
// (utils.cuh) now clamps its read to `capacity` rather than trusting
// `*d_size` blindly -- see the bugfix comment there. So an underestimate
// here costs you some dropped edges (visible via the WARNING printf at
// read-back), not a crash or corrupted output.
// ============================================================================
static std::vector<int> computeERCapacities(ABM* abm, int num_new_nodes, int current_graph_size) {
    std::vector<int> capacities(num_new_nodes);
    double er_prob = abm->get_er_edge_probability();
    int er_k = abm->get_er_edges_per_node();

    if (er_prob > 0.0) {
        double mean   = current_graph_size * er_prob;
        double stddev = std::sqrt(current_graph_size * er_prob * (1.0 - er_prob));
        int cap = (int)std::ceil(mean + 6.0 * stddev) + 1;
        cap = std::min(cap, current_graph_size);
        cap = std::max(cap, 1);
        std::fill(capacities.begin(), capacities.end(), cap);
    } else {
        int cap = std::max(er_k, 1);
        std::fill(capacities.begin(), capacities.end(), cap);
    }
    return capacities;
}

// ============================================================================
// buildOneNodeConnectionsER
//
// ER-model equivalent of buildOneNodeConnections/buildOneNodeConnections_timed
// (kernel.cu), but dramatically simpler: the ER kernels don't do BFS,
// weighted sampling, generator-node selection, or same-year citations, so
// there's no CompactBFSState, no heaps, no citations_vectors<int>
// intermediate step, and no score/weight arrays to allocate here at all --
// edges are written directly into d_new_edges_vec_vectors by the kernel.
//
// Uses create_thread_vectors_bulk<int2> for the edge-output buffer from
// the start (see utils.cuh bugfix comments for why this matters: the
// UNFIXED version of that function allocates 2 cudaMalloc + 2 cudaMemcpy
// PER THREAD for metadata alone -- at num_new_nodes ~15-19K per epoch,
// that's tens of thousands of avoidable driver calls per year, the same
// class of overhead that dominated wall-clock time for the PA model until
// fixed).
// ============================================================================
void buildOneNodeConnectionsER(
    ABM* abm, Graph* graph,
    std::vector<int>& new_nodes_vec,
    std::vector<std::pair<int,int>>& new_edges_vec,
    int current_graph_size,
    int max_batch_size,
    ERTiming* _ep_ptr)
{
    ERTiming& _ep = *_ep_ptr;

    int num_new_nodes = (int)new_nodes_vec.size();
    if (num_new_nodes <= 0) return;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // ---- Upload this year's new-node id list (once, bulk) [timed] ----
    int* d_new_nodes_arr = nullptr;
    {
        ERHostTimer _ht; _ht.start();
        CUDA_CHECK(cudaMalloc(&d_new_nodes_arr, num_new_nodes * sizeof(int)));
        CUDA_CHECK(cudaMemcpyAsync(d_new_nodes_arr, new_nodes_vec.data(),
                                  num_new_nodes * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        _ep.t_upload += _ht.stop_ms();
    }

    // ---- Per-node edge-buffer capacities, computed once host-side [timed] ----
    std::vector<int> capacities;
    {
        ERHostTimer _ht; _ht.start();
        capacities = computeERCapacities(abm, num_new_nodes, current_graph_size);
        _ep.t_capacity += _ht.stop_ms();
    }

    // ---- Bulk-allocate the edge-output buffer for ALL new nodes at once
    //      [timed]. Uses the FIXED create_thread_vectors_bulk (see
    //      utils.cuh) -- O(1) CUDA API calls for metadata, not
    //      O(num_new_nodes). ----
    device_vector_generic<int2>* d_new_edges_vec_vectors = nullptr;
    {
        ERHostTimer _ht; _ht.start();
        create_thread_vectors_bulk<int2>(num_new_nodes, capacities.data(), &d_new_edges_vec_vectors);
        _ep.t_bulk_alloc += _ht.stop_ms();
    }

    int threadBlockSizeX = 16, threadBlockSizeY = 16;
    dim3 threads_per_block(threadBlockSizeX, threadBlockSizeY);
    int threadBlockSize = threadBlockSizeX * threadBlockSizeY;
    int batch_size = std::min(max_batch_size, num_new_nodes);
    unsigned long long seed = static_cast<unsigned long long>(time(NULL));

    curandState* deviceStates_pool = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceStates_pool, batch_size * sizeof(curandState)));

    // ---- GPU kernel execution, all batches [timed] ----
    for (int start = 0; start < num_new_nodes; start += batch_size) {
        int this_batch_size = std::min(batch_size, num_new_nodes - start);
        int blocks_for_this_batch = (this_batch_size + threadBlockSize - 1) / threadBlockSize;

        ERGpuTimer _gt; _gt.start(stream);
        launchErdosRenyiKernel(
            abm, start, this_batch_size, num_new_nodes,
            blocks_for_this_batch, threads_per_block, stream,
            deviceStates_pool, seed,
            d_new_nodes_arr, d_new_edges_vec_vectors,
            current_graph_size);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        _ep.t_kernel += _gt.stop(stream);
    }

    CUDA_CHECK(cudaFree(deviceStates_pool));

    // ---- Read back edges (clamped to capacity -- see utils.cuh bugfix)
    //      [timed] ----
    size_t edges_before = new_edges_vec.size();
    {
        ERHostTimer _ht; _ht.start();
        append_device_to_host<int2>(d_new_edges_vec_vectors, new_edges_vec,
                                num_new_nodes, capacities.data(), graph->getNodeSetSize());
        _ep.t_download += _ht.stop_ms();
    }
    _ep.edges_out += (long long)(new_edges_vec.size() - edges_before);

    // ---- Cleanup [timed] ----
    {
        ERHostTimer _ht; _ht.start();
        cleanup_vectors_bulk<int2>(d_new_edges_vec_vectors, num_new_nodes);
        CUDA_CHECK(cudaFree(d_new_nodes_arr));
        CUDA_CHECK(cudaStreamDestroy(stream));
        _ep.t_cleanup += _ht.stop_ms();
    }
}

// ============================================================================
// executeER
//
// Top-level epoch loop for the ER model -- structurally mirrors execute()
// (kernel.cu) but skips everything PA-specific: no fitness initialization,
// no PA/recency/fitness weight or score arrays, no out-degree bag sampling,
// no generator-node selection, no same-year citation mechanism. New nodes'
// attribute columns that don't apply to ER (pa_weight, rec_weight,
// fit_weight, alpha, fitness fields, generator_node_string) are written
// with the same "not applicable" sentinel values already used elsewhere in
// this codebase for fields that don't apply to a given node (matching how
// seed nodes report -1 for agent-only PA fields in the existing output
// schema), so WriteAttributes' fixed column set stays fully populated
// without fabricating meaningless PA-model values.
// ============================================================================
int executeER(ABM* abm) {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

    // ---- Graph parse [timed] ----
    Graph* graph;
    double t_graph_parse_ms;
    {
        ERHostTimer _ht; _ht.start();
        graph = new Graph(abm->edgelist, abm->nodelist);
        t_graph_parse_ms = _ht.stop_ms();
    }
    abm->WriteToLogFile("loaded graph (ER model)", Log::info);

    int start_year   = abm->GetMaxYear(graph) + 1;
    int next_node_id = abm->GetMaxNode(graph) + 1;

    std::vector<int> new_nodes_vec;
    std::vector<std::pair<int,int>> new_edges_vec;
    int max_batch_size = 20000;
    std::vector<ERTiming> all_epochs;

    for (int current_year = start_year; current_year < start_year + abm->num_cycles; current_year++) {
        int current_graph_size = (int)graph->GetNodeSet().size();
        int num_new_nodes = (int)std::ceil(current_graph_size * abm->growth_rate);

        ERTiming _ep;
        _ep.year = current_year;
        _ep.N = current_graph_size;
        _ep.delta = num_new_nodes;

        abm->WriteToLogFile("ER: year " + std::to_string(current_year) +
                            ", graph size " + std::to_string(current_graph_size) +
                            ", adding " + std::to_string(num_new_nodes) + " nodes", Log::info);

        // ---- New-node init [timed]: same sequential-id bookkeeping as the
        //      PA model, but with sentinel values for every PA-only
        //      attribute column (see comment above) ----
        {
            ERHostTimer _ht; _ht.start();
            for (int i = 0; i < num_new_nodes; i++) {
                int seq = current_graph_size + i;
                graph->continuous_node_mapping[next_node_id]        = seq;
                graph->reverse_continuous_node_mapping[seq]         = next_node_id;
                new_nodes_vec.push_back(seq);

                graph->SetIntAttribute("year", seq, current_year);
                graph->setType(AGENT_TYPE, seq);

                // PA-only fields: not applicable to ER, sentinel values.
                graph->SetDoubleAttribute("pa_weight", seq, -1.0);
                graph->SetDoubleAttribute("rec_weight", seq, -1.0);
                graph->SetDoubleAttribute("fit_weight", seq, -1.0);
                graph->SetDoubleAttribute("alpha", seq, -1.0);
                graph->SetIntAttribute("fit_lag_duration", seq, 0);
                graph->SetIntAttribute("fit_peak_value", seq, 0);
                graph->SetIntAttribute("fit_peak_duration", seq, 0);
                graph->SetIntAttribute("planted_nodes_line_number", seq, -1);
                // assigned_out_degree: report the Fixed-k target if
                // applicable; -1 for G(n,p), where there is no single
                // fixed target (actual out_degree, written later from
                // updateNodeInDegreeOutDegree, is the real observed value
                // either way).
                int er_k = abm->get_er_edges_per_node();
                graph->SetIntAttribute("assigned_out_degree", seq, (er_k > 0) ? er_k : -1);
                // generator_node_string: ER has no generator-node concept.
                graph->SetIntAttribute("generator_node", seq, -1);

                next_node_id++;
            }
            _ep.t_node_init += _ht.stop_ms();
        }

        // ---- Sample edges for this year's new nodes (internally timed --
        //      see buildOneNodeConnectionsER's own phase breakdown) ----
        try {
            buildOneNodeConnectionsER(abm, graph, new_nodes_vec, new_edges_vec,
                                      current_graph_size, max_batch_size, &_ep);
        } catch (const std::exception& e) {
            std::cerr << "Exception in buildOneNodeConnectionsER: " << e.what() << std::endl;
            delete graph;
            return 1;
        }

        // ---- Commit edges into the graph, update degrees [timed] (same
        //      pattern as execute()'s host graph-update block in kernel.cu) ----
        {
            ERHostTimer _ht; _ht.start();
            graph->node_set.insert(new_nodes_vec.begin(), new_nodes_vec.end());
            std::sort(new_edges_vec.begin(), new_edges_vec.end());

            std::unordered_set<int> updated_destination_nodes;
            updated_destination_nodes.reserve(new_nodes_vec.size() + new_edges_vec.size());
            updated_destination_nodes.insert(new_nodes_vec.begin(), new_nodes_vec.end());

            for (const auto& [src, dst] : new_edges_vec) {
                graph->forward_adj_map[src].insert(dst);
                graph->backward_adj_map[dst].insert(src);
                updated_destination_nodes.insert(dst);
            }

            graph->updateNodeInDegreeOutDegree(new_nodes_vec, updated_destination_nodes, current_year);
            _ep.t_graph_update += _ht.stop_ms();
        }

        if ((int)new_nodes_vec.size() > max_batch_size) {
            max_batch_size = (int)std::ceil(max_batch_size * (1 - 0.5 * abm->growth_rate));
        }

        new_nodes_vec.clear();
        new_edges_vec.clear();

        all_epochs.push_back(_ep);
        printf("\n[ER EP %d] N=%d delta=%d  upload=%.0f capac=%.0f alloc=%.0f kernel=%.0f dl=%.0f clean=%.0f graph=%.0f  total=%.0f ms  edges=%lld",
               _ep.year, _ep.N, _ep.delta,
               _ep.t_upload, _ep.t_capacity, _ep.t_bulk_alloc, _ep.t_kernel,
               _ep.t_download, _ep.t_cleanup, _ep.t_graph_update,
               _ep.epoch_total(), _ep.edges_out);
    }

    abm->WriteToLogFile("finished ER sim", Log::info);

    // ---- Output write [timed] ----
    double t_output_write_ms;
    {
        ERHostTimer _ht; _ht.start();
        graph->WriteGraph(abm->output_file);
        for (auto const& nid : graph->GetNodeSet()) {
            graph->SetIntAttribute("in_degree",  nid, graph->GetInDegree(nid));
            graph->SetIntAttribute("out_degree", nid, graph->GetOutDegree(nid));
        }
        graph->WriteAttributes(abm->auxiliary_information_file);
        t_output_write_ms = _ht.stop_ms();
    }

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    double e2e_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    std::ostringstream msg;
    msg << "\nE2E Time, model: " << abm->get_model()
        << "  num_cycles=" << abm->num_cycles
        << "  growth_rate=" << (100.0 * abm->growth_rate) << "%"
        << "  elapsed=" << (long long)(e2e_ms / 1000) << "s";
    abm->WriteToLogFile(msg.str(), Log::info);
    std::cout << msg.str() << std::endl;

    // ---- Per-epoch report (mirrors PA model's print_epoch_report) ----
    print_er_epoch_report(all_epochs, e2e_ms, abm->get_model(), abm->growth_rate);

    // ---- Full pipeline breakdown, same shape as the PA model's report
    //      from earlier in this session -- covers graph parse, simulation
    //      (epoch loop total), and output write, summing to ~100% of the
    //      measured E2E time. ----
    {
        double simulation_ms = 0.0;
        for (const auto& ep : all_epochs) simulation_ms += ep.epoch_total();

        double measured_sum_ms = t_graph_parse_ms + simulation_ms + t_output_write_ms;
        double unaccounted_ms = e2e_ms - measured_sum_ms;

        auto pct = [&](double part) { return e2e_ms > 0.0 ? (100.0 * part / e2e_ms) : 0.0; };

        printf("\n========================================================================================================================\n");
        printf("  ER MODEL FULL PIPELINE BREAKDOWN (parse -> simulation -> output write)  model=%s  growth=%.1f%%\n",
               abm->get_model().c_str(), 100.0 * abm->growth_rate);
        printf("========================================================================================================================\n");
        printf("  %-45s : %10.0f ms  (%5.1f%%)\n", "Graph parse (nodelist + edgelist)", t_graph_parse_ms, pct(t_graph_parse_ms));
        printf("  %-45s : %10.0f ms  (%5.1f%%)\n", "Simulation (epoch loop total -- see above)", simulation_ms, pct(simulation_ms));
        printf("  %-45s : %10.0f ms  (%5.1f%%)\n", "Output write (WriteGraph + WriteAttributes)", t_output_write_ms, pct(t_output_write_ms));
        printf("------------------------------------------------------------------------------------------------------------------------\n");
        printf("  %-45s : %10.0f ms  (%5.1f%%)\n", "TOTAL (measured)", measured_sum_ms, pct(measured_sum_ms));
        printf("  %-45s : %10.0f ms  (%5.1f%%)\n", "Unaccounted", unaccounted_ms, pct(unaccounted_ms));
        printf("  %-45s : %10.0f ms  (100.0%%)\n", "FULL PIPELINE TOTAL (E2E)", e2e_ms);
        printf("========================================================================================================================\n");
    }

    delete graph;
    return 0;
}
