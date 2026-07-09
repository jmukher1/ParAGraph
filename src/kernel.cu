#include <iostream>

#include "epoch_profiler.cuh"
#include "abm.cuh"
#include "device_map.cuh"
#include "int2.cuh"
#include "utils.cuh"

// Set by main.cu before execute() runs (time spent inside the ABM
// constructor: reading out-degree-bag, recency-probabilities, and
// planted-nodes files -- see ABM::ReadOutDegreeBag/ReadRecencyProbabilities/
// ReadPlantedNodes in abm.cu). Passed via a global rather than a new
// execute() parameter to avoid needing to verify/update a forward
// declaration in abm.cuh, which isn't available here.
double g_abm_construction_ms = 0.0;

__global__ void kernelCallStage1(
    int start_idx,
    int batch_size,
    int total_N,
    ABM* abm,
    Graph* graph,
    DeviceGraph* d_forward_adj_map_Graph,
    DeviceGraph* d_backward_adj_map_Graph,
    device_map<int, Node>::device_view d_nodeAttributeMap_view,
    ABMStageState* d_states,
    int* d_new_nodes_arr,
    int num_generator_node_citation,
    device_vector* one_hop_neighborhood_vectors,
    device_vector* two_hop_neighborhood_vectors,
    CompactBFSState* d_bfs_pool,
    int initial_graph_size)
{
        // NOTE: Stage 1 only touches per-node BFS/adjacency structures and
        // the generator-node lookup, never the growth arrays (pa_weight_arr,
        // out_degree_arr, etc.), so it is NOT affected by the indexing bug
        // fixed in kernelCallStage2/3/4 below. No change needed here.

        int threads_per_block = blockDim.x * blockDim.y;
        int local_idx = blockIdx.x * threads_per_block +
                        threadIdx.y * blockDim.x + threadIdx.x;
        if (local_idx >= batch_size) return;
        int idx = start_idx + local_idx;
        if (idx >= total_N) return;


        int new_node = d_new_nodes_arr[idx];
        d_states[local_idx].generator_node = abm->getGraphAttributesGeneratorNode(
            graph, 
            d_nodeAttributeMap_view, 
            new_node
        ); 
        
        abm->GetOneAndTwoHopNeighborhood(
                graph,
                idx,
                new_node,
                d_forward_adj_map_Graph,
                d_backward_adj_map_Graph,
                d_nodeAttributeMap_view,
                d_states[local_idx].generator_node,
                num_generator_node_citation,
                d_bfs_pool[local_idx],
                one_hop_neighborhood_vectors[local_idx],
                two_hop_neighborhood_vectors[local_idx]);

}

// __launch_bounds__(256, 4): hints the compiler to target 4 resident blocks/SM
// at block size 256 (rather than letting register allocation float freely).
// Measured via ncu on both H100 and RTX 4090: occupancy_limit_registers=2.0 on
// BOTH architectures with 110-118 registers/thread -- register pressure caps
// occupancy to 2 blocks/SM regardless of grid size. This hint trades some
// per-thread register budget to unlock more concurrent blocks/SM. Verify with
// ncu after rebuilding: launch__registers_per_thread should drop and
// occupancy_limit_registers should rise toward the target (4). If register
// spilling to local memory shows up (check with --metrics
// launch__local_mem_per_thread or watch for a throughput regression), back
// the target down to 3.
// -----------------------------------------------------------------------
// kernelCallStage1  -- REWRITTEN to use warp-cooperative BFS
//
// ROOT CAUSE FIXED (confirmed via ncu on both H100 and RTX 4090): the
// previous 1-thread-per-paper design produced severe warp divergence
// (measured smsp__thread_inst_executed_per_inst_executed.ratio of only
// ~2.28/32 = ~7% of a warp's lanes active per instruction, IDENTICAL on
// both GPUs) because each thread independently BFS'd a different paper
// with a different frontier size/shape. It also launched too few blocks
// to fill either GPU's SM count (avg 66 blocks into 128-132 SMs).
//
// FIX: one WARP (32 threads) now cooperates on a single paper's BFS,
// using ABM::GetOneAndTwoHopNeighborhood_Warp (abm.cu) -- a fully
// implemented, previously-unused warp-cooperative BFS that distributes
// frontier-word processing across a warp's lanes via atomicOr-guarded
// bitmap updates, rather than one thread looping serially. This directly
// reduces divergence (32 lanes cooperating on the SAME paper's frontier
// stay far more in lockstep than 32 lanes each independently BFS-ing a
// DIFFERENT paper), and as a direct side effect increases grid size ~32x
// for the same paper count (each block of 256 threads = 8 warps now
// covers only 8 papers, not 256), comfortably filling both GPUs' SM
// counts without a separate fix.
//
// Launch config changed to match: see buildOneNodeConnections /
// buildOneNodeConnections_timed below for the new warp-per-paper grid
// calculation and per-warp scratch buffer allocation (bulk-allocated,
// following the same pattern already fixed in utils.cuh for
// create_thread_vectors_int/create_thread_vectors_bulk).
// __launch_bounds__(256, 4) 
// -----------------------------------------------------------------------
__global__ void kernelCallStage1_warped(
    int start_idx,
    int batch_size,
    int total_N,
    ABM* abm,
    Graph* graph,
    DeviceGraph* d_forward_adj_map_Graph,
    DeviceGraph* d_backward_adj_map_Graph,
    device_map<int, Node>::device_view d_nodeAttributeMap_view,
    ABMStageState* d_states,
    int* d_new_nodes_arr,
    int num_generator_node_citation,
    device_vector* one_hop_neighborhood_vectors,
    device_vector* two_hop_neighborhood_vectors,
    uint32_t* d_visited_slab_warp,
    uint32_t* d_curr_slab_warp,
    uint32_t* d_next_slab_warp,
    int num_words,
    int max_vertices)
{
        const int lane_id = threadIdx.x & 31;
        const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
        const int warp_id = global_thread >> 5;   // this-batch-local warp index = paper index

        if (warp_id >= batch_size) return;
        int idx = start_idx + warp_id;
        if (idx >= total_N) return;

        int new_node = d_new_nodes_arr[idx];

        // Only lane 0 does the scalar generator-node lookup; broadcast the
        // result to every lane before the warp-cooperative BFS call.
        int generator_node = -1;
        if (lane_id == 0) {
            generator_node = abm->getGraphAttributesGeneratorNode(
                graph, d_nodeAttributeMap_view, new_node);
            d_states[warp_id].generator_node = generator_node;
        }
        generator_node = __shfl_sync(0xFFFFFFFFu, generator_node, 0);

        uint32_t* visited = d_visited_slab_warp + (long long)warp_id * num_words;
        uint32_t* curr    = d_curr_slab_warp    + (long long)warp_id * num_words;
        uint32_t* next    = d_next_slab_warp    + (long long)warp_id * num_words;

        abm->GetOneAndTwoHopNeighborhood_Warp(
                graph,
                idx,
                new_node,
                d_forward_adj_map_Graph,
                d_backward_adj_map_Graph,
                d_nodeAttributeMap_view,
                generator_node,
                num_generator_node_citation,
                visited, curr, next,
                one_hop_neighborhood_vectors[warp_id],
                two_hop_neighborhood_vectors[warp_id],
                max_vertices);

}

__global__ void kernelCallStage2(
    int start_idx,
    int batch_size,
    int total_N,
    ABM* abm,
    Graph* graph,
    set_ref_type d_same_year_source_nodes_set_ref,
    ABMStageState* d_states,
    curandState* deviceStates, 
    unsigned long long seed,
    int* d_new_nodes_arr,
    device_vector* one_hop_neighborhood_vectors,
    //device_vector_soa<float>* element_index_vec_vectors,     
    device_min_heap<float>* d_heap_array,        
    device_vector_generic<int>* citations_vectors,
    double* pa_arr, double* recency_arr, double* fit_arr,
    double* pa_weight_arr, double* rec_weight_arr,
    double* fit_weight_arr, double* alpha_arr,
    int* out_degree_arr,
    double fully_random_citations,
    int num_generator_node_citation,
    int current_year,
    int current_graph_size,
    int initial_graph_size,
    int final_graph_size) {
        
        int threads_per_block = blockDim.x * blockDim.y;
        int local_idx = blockIdx.x * threads_per_block +
                        threadIdx.y * blockDim.x + threadIdx.x;
        if (local_idx >= batch_size) return;
        int idx = start_idx + local_idx;
        if (idx >= total_N) return;

        // ====================================================================
        // BUGFIX (root cause of GPU/CPU node-count divergence):
        //
        // `idx` only ranges over THIS YEAR's new-node cohort, i.e.
        // [0, num_new_nodes_this_year). But pa_weight_arr / rec_weight_arr /
        // fit_weight_arr / alpha_arr / out_degree_arr are populated ONCE,
        // up front, for the ENTIRE simulation (size = growth_in_graph_size =
        // final_graph_size - initial_graph_size, i.e. every node that will
        // EVER be created across all years). Each node's correct position in
        // those arrays is its (continuous node id - initial_graph_size), NOT
        // its position within the current year's cohort alone. This exactly
        // mirrors how the CPU reference computes it:
        //     weight_arr_index = continuous_node_mapping[new_node] - initial_graph_size;
        //
        // Using `idx` directly here silently re-reads the FIRST batch_size
        // entries of these arrays every single year instead of this year's
        // actual entries. Concretely: year 2's nodes get year 1's weights/
        // out-degrees, year 3's nodes get year 1's again, etc. Combined with
        // growth being compounding (num_new_nodes scales with the current
        // graph size), any resulting divergence in effective out-degree
        // snowballs year over year -- which is exactly the shape of the
        // observed GPU-vs-CPU degree/node-count gap.
        //
        // Fix: offset idx by how many nodes were already created in prior
        // years (current_graph_size - initial_graph_size) before indexing
        // into the growth arrays.
        // ====================================================================
        int growth_idx = (current_graph_size - initial_graph_size) + idx;

        // initialize RNG for this node
        curand_init(seed, idx, 0, &deviceStates[local_idx]);
        
        int new_node_seq = d_new_nodes_arr[idx];

        double pa_weight  = __ldg(&pa_weight_arr[growth_idx]);
        double rec_weight = __ldg(&rec_weight_arr[growth_idx]);
        double fit_weight = __ldg(&fit_weight_arr[growth_idx]);
        double alpha      = __ldg(&alpha_arr[growth_idx]);

        int outdeg = out_degree_arr[growth_idx];

        // CPU code: int same_year_citation = same_year_source_nodes.count(i); // could be 0 or 1
        d_states[local_idx].same_year_citation = d_same_year_source_nodes_set_ref.contains(idx) ? 1 : 0;
        d_states[local_idx].num_fully_random_cited_reserved = __float2int_rd(fully_random_citations * outdeg);
        
        // Calculate number of citations from 1-hop neighborhood
        int remaining = outdeg - num_generator_node_citation
                        - d_states[local_idx].same_year_citation
                        - d_states[local_idx].num_fully_random_cited_reserved;

        d_states[local_idx].num_citations_inside = __float2int_ru(remaining * alpha);
        d_states[local_idx].num_citations_inside = min(d_states[local_idx].num_citations_inside, 
                                                (int)(one_hop_neighborhood_vectors[local_idx]).size());
        
        d_states[local_idx].num_actually_cited = 0; 

        abm->ABMKernelStage2(
                idx, new_node_seq, total_N, graph,
                &deviceStates[local_idx],
                one_hop_neighborhood_vectors[local_idx],
                d_heap_array[local_idx],
                citations_vectors[local_idx],
                pa_arr, recency_arr, fit_arr,
                pa_weight, rec_weight, fit_weight,
                current_year,
                current_graph_size,
                initial_graph_size,
                final_graph_size,
                d_states[local_idx].same_year_citation,
                d_states[local_idx].num_citations_inside,
                d_states[local_idx].num_actually_cited);
}
 
__global__ void kernelCallStage3(
        int start_idx,
        int batch_size,
        int total_N,
        ABM* abm,
        Graph* graph,
        int graphNodeSetSize,
        device_map<int, Node>::device_view d_nodeAttributeMap_view,
        ABMStageState* d_states,
        curandState* deviceStates,
        int* d_new_nodes_arr,
        device_vector* two_hop_neighborhood_vectors,
        //device_vector_soa<float>* element_index_vec_vectors,     
        device_min_heap<float>* d_heap_array,    
        device_vector_generic<int>* citations_vectors,
        set_ref_type* selected_citations_set_refs,
        double* pa_arr, double* recency_arr, double* fit_arr,
        double* pa_weight_arr, double* rec_weight_arr,
        double* fit_weight_arr, double* alpha_arr,
        int* out_degree_arr,
        int num_generator_node_citation,
        int current_year, int current_graph_size,
        int initial_graph_size, int final_graph_size) {

        int threads_per_block = blockDim.x * blockDim.y;
        int local_idx = blockIdx.x * threads_per_block +
                        threadIdx.y * blockDim.x + threadIdx.x;
        if (local_idx >= batch_size) return;
        int idx = start_idx + local_idx;
        if (idx >= total_N) return;

        // BUGFIX: see the detailed comment in kernelCallStage2 above -- same
        // issue, same fix. `idx` is this-year-local; the growth arrays are
        // indexed across the whole simulation's lifetime.
        int growth_idx = (current_graph_size - initial_graph_size) + idx;

        int new_node = d_new_nodes_arr[idx];
        int outdeg = out_degree_arr[growth_idx];

        double pa_weight  = __ldg(&pa_weight_arr[growth_idx]);
        double rec_weight = __ldg(&rec_weight_arr[growth_idx]);
        double fit_weight = __ldg(&fit_weight_arr[growth_idx]);
        double alpha      = __ldg(&alpha_arr[growth_idx]);

        int remaining = outdeg - num_generator_node_citation
                        - d_states[local_idx].same_year_citation
                        - d_states[local_idx].num_fully_random_cited_reserved
                        - d_states[local_idx].num_citations_inside;

        // Calculate number of citations from 2-hop neighborhood
        d_states[local_idx].num_citations_outside = min(remaining, (int)(two_hop_neighborhood_vectors[local_idx]).size());

        int num_actually_cited_so_far = d_states[local_idx].num_actually_cited;
        abm->ABMKernelStage3(
                idx, total_N, graph, graphNodeSetSize, 
                two_hop_neighborhood_vectors[local_idx], 
                d_heap_array[local_idx], 
                citations_vectors[local_idx],
                selected_citations_set_refs[local_idx], 
                d_states[local_idx].generator_node, new_node,
                pa_arr, recency_arr, fit_arr, 
                pa_weight, rec_weight, fit_weight,
                current_year, current_graph_size, 
                initial_graph_size, final_graph_size,
                num_actually_cited_so_far,
                d_states[local_idx].num_citations_outside,
                d_states[local_idx].num_actually_cited,
                &deviceStates[local_idx]);
}

__global__ void kernelCallStage4(
        int start_idx,
        int batch_size,
        int total_N,
        ABM* abm,
        Graph* graph,
        int graphNodeSetSize,
        device_map<int, Node>::device_view d_nodeAttributeMap_view,
        ABMStageState* d_states,
        curandState* deviceStates,
        int* d_new_nodes_arr,
        //device_vector* two_hop_neighborhood_vectors,
        //device_vector_soa<float>* element_index_vec_vectors,        
        device_vector_generic<int>* citations_vectors,
        device_vector_generic<int2>* d_new_edges_vec_vectors,
        set_ref_type* selected_citations_set_refs, 
        int* out_degree_arr,
        int num_generator_node_citation,
        int current_year, int current_graph_size,
        int initial_graph_size, int final_graph_size,
        int per_thread_selected_set_capacity) {

        int threads_per_block = blockDim.x * blockDim.y;
        int local_idx = blockIdx.x * threads_per_block +
                        threadIdx.y * blockDim.x + threadIdx.x;
        if (local_idx >= batch_size) return;
        int idx = start_idx + local_idx;
        if (idx >= total_N) return;

        // BUGFIX: see the detailed comment in kernelCallStage2 above -- same
        // issue, same fix.
        int growth_idx = (current_graph_size - initial_graph_size) + idx;

        int new_node = d_new_nodes_arr[idx];
        int outdeg = out_degree_arr[growth_idx];

        // Calculate fully random citations
        d_states[local_idx].num_fully_random_cited = outdeg - num_generator_node_citation
                                                - d_states[local_idx].same_year_citation
                                                - d_states[local_idx].num_citations_inside
                                                - d_states[local_idx].num_citations_outside;
         
        abm->ABMKernelStage4(
                idx, total_N, graph, graphNodeSetSize,
                //element_index_vec_vectors[local_idx], 
                citations_vectors[local_idx],
                d_new_edges_vec_vectors[idx], 
                selected_citations_set_refs[local_idx], 
                d_states[local_idx].generator_node, new_node,
                current_year, current_graph_size, initial_graph_size, final_graph_size,
                d_states[local_idx].num_citations_outside, 
                d_states[local_idx].num_fully_random_cited, 
                d_states[local_idx].num_actually_cited,
                &deviceStates[local_idx], 
                per_thread_selected_set_capacity);

}

// =============================================================================
// ABMKernel -- monolithic fusion of kernelCallStage1 (non-warped) + 2 + 3 + 4
//
// Called instead of the four separate kernel launches when
// abm->is_use_multistage_kernel is false. This is the "fused single-kernel"
// side of the staged-vs-fused scheduling ablation: one thread performs all
// four stages sequentially for its assigned paper, in a single launch,
// rather than four separate kernel launches each requiring their own
// grid-wide synchronization.
//
// Uses the NON-WARPED BFS (ABM::GetOneAndTwoHopNeighborhood, CompactBFSState)
// for Stage 1's portion, matching the original kernelCallStage1's per-thread
// granularity -- fusing the warp-cooperative BFS into a per-thread-granularity
// monolithic kernel isn't architecturally coherent (warp mode needs one warp
// per paper; this kernel is one thread per paper throughout), so
// is_use_multistage_kernel and use_warp_bfs are independent toggles: fusing
// is only supported with the non-warped BFS path.
//
// Every computation below (growth_idx offset, weight loads, remaining/
// citation-count arithmetic) is copied verbatim from kernelCallStage2/3/4
// above -- see those functions' comments for the detailed bugfix rationale
// (growth_idx offset) that applies identically here.
//
// NOTE: Stage 2 and Stage 3 each need their OWN heap array in the fused
// kernel (they can no longer be allocated, used, and destroyed sequentially
// between "stages" the way the separate-kernel path does, since fusion means
// both are needed live within the same single launch) -- see
// d_one_hop_heap_array / d_two_hop_heap_array below, and the corresponding
// caller-side allocation of BOTH heap arrays up front in
// buildOneNodeConnections / buildOneNodeConnections_timed's fused-kernel
// branch.
// =============================================================================
__global__ void ABMKernel(
    // ---- common ----
    int start_idx, int batch_size, int total_N,
    ABM* abm, Graph* graph, int graphNodeSetSize,
    device_map<int, Node>::device_view d_nodeAttributeMap_view,
    ABMStageState* d_states,
    int* d_new_nodes_arr,
    int num_generator_node_citation,
    int current_year, int current_graph_size,
    int initial_graph_size, int final_graph_size,
    // ---- Stage 1 (non-warped BFS) ----
    DeviceGraph* d_forward_adj_map_Graph,
    DeviceGraph* d_backward_adj_map_Graph,
    CompactBFSState* d_bfs_pool,
    device_vector* one_hop_neighborhood_vectors,
    device_vector* two_hop_neighborhood_vectors,
    // ---- Stage 2 ----
    set_ref_type d_same_year_source_nodes_set_ref,
    curandState* deviceStates,
    unsigned long long seed,
    device_min_heap<float>* d_one_hop_heap_array,
    double* pa_arr, double* recency_arr, double* fit_arr,
    double* pa_weight_arr, double* rec_weight_arr,
    double* fit_weight_arr, double* alpha_arr,
    int* out_degree_arr,
    double fully_random_citations,
    // ---- Stage 3 ----
    device_min_heap<float>* d_two_hop_heap_array,
    set_ref_type* selected_citations_set_refs,
    // ---- Stage 2/3/4 shared ----
    device_vector_generic<int>* citations_vectors,
    // ---- Stage 4 ----
    device_vector_generic<int2>* d_new_edges_vec_vectors,
    int per_thread_selected_set_capacity)
{
        int threads_per_block = blockDim.x * blockDim.y;
        int local_idx = blockIdx.x * threads_per_block +
                        threadIdx.y * blockDim.x + threadIdx.x;
        if (local_idx >= batch_size) return;
        int idx = start_idx + local_idx;
        if (idx >= total_N) return;

        // ===== STAGE 1: BFS (non-warped, identical to kernelCallStage1) =====
        int new_node = d_new_nodes_arr[idx];
        d_states[local_idx].generator_node = abm->getGraphAttributesGeneratorNode(
            graph, d_nodeAttributeMap_view, new_node);

        abm->GetOneAndTwoHopNeighborhood(
                graph, idx, new_node,
                d_forward_adj_map_Graph, d_backward_adj_map_Graph,
                d_nodeAttributeMap_view,
                d_states[local_idx].generator_node,
                num_generator_node_citation,
                d_bfs_pool[local_idx],
                one_hop_neighborhood_vectors[local_idx],
                two_hop_neighborhood_vectors[local_idx]);

        // ===== STAGE 2: same-year + 1-hop citations (identical to
        //       kernelCallStage2, including its growth_idx bugfix) =====
        int growth_idx = (current_graph_size - initial_graph_size) + idx;
        curand_init(seed, idx, 0, &deviceStates[local_idx]);
        int new_node_seq = d_new_nodes_arr[idx];

        double pa_weight  = __ldg(&pa_weight_arr[growth_idx]);
        double rec_weight = __ldg(&rec_weight_arr[growth_idx]);
        double fit_weight = __ldg(&fit_weight_arr[growth_idx]);
        double alpha      = __ldg(&alpha_arr[growth_idx]);
        int outdeg = out_degree_arr[growth_idx];

        d_states[local_idx].same_year_citation = d_same_year_source_nodes_set_ref.contains(idx) ? 1 : 0;
        d_states[local_idx].num_fully_random_cited_reserved = __float2int_rd(fully_random_citations * outdeg);

        int remaining = outdeg - num_generator_node_citation
                        - d_states[local_idx].same_year_citation
                        - d_states[local_idx].num_fully_random_cited_reserved;

        d_states[local_idx].num_citations_inside = __float2int_ru(remaining * alpha);
        d_states[local_idx].num_citations_inside = min(d_states[local_idx].num_citations_inside,
                                                (int)(one_hop_neighborhood_vectors[local_idx]).size());
        d_states[local_idx].num_actually_cited = 0;

        abm->ABMKernelStage2(
                idx, new_node_seq, total_N, graph,
                &deviceStates[local_idx],
                one_hop_neighborhood_vectors[local_idx],
                d_one_hop_heap_array[local_idx],
                citations_vectors[local_idx],
                pa_arr, recency_arr, fit_arr,
                pa_weight, rec_weight, fit_weight,
                current_year, current_graph_size, initial_graph_size, final_graph_size,
                d_states[local_idx].same_year_citation,
                d_states[local_idx].num_citations_inside,
                d_states[local_idx].num_actually_cited);

        // ===== STAGE 3: 2-hop citations (identical to kernelCallStage3) =====
        int remaining3 = outdeg - num_generator_node_citation
                        - d_states[local_idx].same_year_citation
                        - d_states[local_idx].num_fully_random_cited_reserved
                        - d_states[local_idx].num_citations_inside;
        d_states[local_idx].num_citations_outside =
            min(remaining3, (int)(two_hop_neighborhood_vectors[local_idx]).size());

        int num_actually_cited_so_far = d_states[local_idx].num_actually_cited;
        abm->ABMKernelStage3(
                idx, total_N, graph, graphNodeSetSize,
                two_hop_neighborhood_vectors[local_idx],
                d_two_hop_heap_array[local_idx],
                citations_vectors[local_idx],
                selected_citations_set_refs[local_idx],
                d_states[local_idx].generator_node, new_node,
                pa_arr, recency_arr, fit_arr,
                pa_weight, rec_weight, fit_weight,
                current_year, current_graph_size, initial_graph_size, final_graph_size,
                num_actually_cited_so_far,
                d_states[local_idx].num_citations_outside,
                d_states[local_idx].num_actually_cited,
                &deviceStates[local_idx]);

        // ===== STAGE 4: fully-random fill + edge write (identical to
        //       kernelCallStage4) =====
        d_states[local_idx].num_fully_random_cited = outdeg - num_generator_node_citation
                                                - d_states[local_idx].same_year_citation
                                                - d_states[local_idx].num_citations_inside
                                                - d_states[local_idx].num_citations_outside;

        abm->ABMKernelStage4(
                idx, total_N, graph, graphNodeSetSize,
                citations_vectors[local_idx],
                d_new_edges_vec_vectors[idx],
                selected_citations_set_refs[local_idx],
                d_states[local_idx].generator_node, new_node,
                current_year, current_graph_size, initial_graph_size, final_graph_size,
                d_states[local_idx].num_citations_outside,
                d_states[local_idx].num_fully_random_cited,
                d_states[local_idx].num_actually_cited,
                &deviceStates[local_idx],
                per_thread_selected_set_capacity);
}

// Kernel to verify initialization
__global__ void verify_bfs_pool(CompactBFSState* pool, int num_threads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_threads) {
        printf("Thread %d: max_vertices = %d, bitmap_words = %d\n", 
               idx, pool[idx].max_vertices, pool[idx].bitmap_words);
    }
}

void buildOneNodeConnections(ABM* abm, Graph* graph,
                std::vector<int>& new_nodes_vec,
                int same_year_source_nodes_capacity,
                std::set<int> same_year_source_nodes,
                std::vector<std::pair<int, int>>& new_edges_vec,
                int num_generator_node_citation,
                double* pa_arr, double* recency_arr, double* fit_arr,
                double* pa_weight_arr, double* rec_weight_arr, double* fit_weight_arr, double* alpha_arr,
                int* out_degree_arr,
                int current_year,
                int current_graph_size,
                int initial_graph_size,
                int final_graph_size,
                int max_batch_size) {
        std::cout << "\nInside optimized buildOneNodeConnections...\n";
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        std::cout << "Initial Free: " << (free_mem / (1024.0 * 1024.0))
                << " MB / " << (total_mem / (1024.0 * 1024.0)) << " MB\n";

        // ---------------------------------------------------------------------
        // CONFIGURATION
        // ---------------------------------------------------------------------
        int per_thread_citations_vector_capacity = 250;
        int per_thread_vector_capacity = current_graph_size;
        int per_thread_selected_set_capacity = per_thread_citations_vector_capacity;
        
        int num_new_nodes = new_nodes_vec.size();
        int growth_in_graph_size = final_graph_size - initial_graph_size;

        // Create CUDA stream
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        // ---------------------------------------------------------------------
        // STATIC ARRAYS (used in all 3 stages)
        // ---------------------------------------------------------------------
        int* d_new_nodes_arr = nullptr;
        CUDA_CHECK(cudaMalloc(&d_new_nodes_arr, num_new_nodes * sizeof(int)));
        CUDA_CHECK(cudaMemcpyAsync(d_new_nodes_arr, new_nodes_vec.data(),
                num_new_nodes * sizeof(int), cudaMemcpyHostToDevice, stream));

        double *d_pa_arr, *d_recency_arr, *d_fit_arr;
        double *d_pa_weight_arr, *d_rec_weight_arr, *d_fit_weight_arr, *d_alpha_arr;
        int* d_out_degree_arr;
        CUDA_CHECK(cudaMalloc(&d_pa_arr, final_graph_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_recency_arr, final_graph_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_fit_arr, final_graph_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_pa_weight_arr, growth_in_graph_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_rec_weight_arr, growth_in_graph_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_fit_weight_arr, growth_in_graph_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_alpha_arr, growth_in_graph_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_out_degree_arr, growth_in_graph_size * sizeof(int)));

        CUDA_CHECK(cudaMemcpyAsync(d_pa_arr, pa_arr, final_graph_size * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_recency_arr, recency_arr, final_graph_size * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_fit_arr, fit_arr, final_graph_size * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_pa_weight_arr, pa_weight_arr, growth_in_graph_size * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_rec_weight_arr, rec_weight_arr, growth_in_graph_size * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_fit_weight_arr, fit_weight_arr, growth_in_graph_size * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_alpha_arr, alpha_arr, growth_in_graph_size * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_out_degree_arr, out_degree_arr, growth_in_graph_size * sizeof(int), cudaMemcpyHostToDevice, stream));

        // ---------------------------------------------------------------------
        // GRAPH STRUCTURES (needed only for Stage 1)
        // ---------------------------------------------------------------------
        DeviceGraph* d_forward_adj_map_Graph;
        CUDA_CHECK(cudaMalloc(&d_forward_adj_map_Graph, sizeof(DeviceGraph)));
        prepareGraph(graph->getForwardAdjMap(), d_forward_adj_map_Graph, graph->getNodeSetSize());

        DeviceGraph* d_backward_adj_map_Graph;
        CUDA_CHECK(cudaMalloc(&d_backward_adj_map_Graph, sizeof(DeviceGraph)));
        prepareGraph(graph->getBackwardAdjMap(), d_backward_adj_map_Graph, graph->getNodeSetSize());

        int nodeAttrMapSize = graph->getNodeAttributeMapSize();
        device_map<int, Node>* d_nodeAttributeMap = new device_map<int, Node>(nodeAttrMapSize);
        convertHostMapToDeviceMap<int, Node>(graph->getNodeAttributeMap(), d_nodeAttributeMap, nodeAttrMapSize);

        // ---------------------------------------------------------------------
        // SAME-YEAR STATIC SET (used in Stage 2)
        // ---------------------------------------------------------------------
        set_type* d_same_year_source_nodes_set =
                new set_type(same_year_source_nodes_capacity,
                        cuco::empty_key<int>{empty_key_sentinel});
        convertStdSetToDeviceStaticSet(same_year_source_nodes, *d_same_year_source_nodes_set);
        auto d_same_year_source_nodes_set_ref = d_same_year_source_nodes_set->ref(
                cuco::op::insert, cuco::op::find, cuco::op::erase, cuco::op::contains);

        // ---------------------------------------------------------------------
        // PER-BATCH LAUNCH CONFIG
        // ---------------------------------------------------------------------
        int threadBlockSizeX = 16, threadBlockSizeY = 16;
        dim3 threads_per_block(threadBlockSizeX, threadBlockSizeY);
        int threadBlockSize = threadBlockSizeX * threadBlockSizeY;
        int batch_size = std::min(max_batch_size, num_new_nodes);
        unsigned long long seed = static_cast<unsigned long long>(time(NULL));

        // Pre-allocate curandState for the largest possible batch — avoids
        // a cudaMalloc/cudaFree inside the hot per-batch loop.
        curandState* deviceStates_pool = nullptr;
        CUDA_CHECK(cudaMalloc(&deviceStates_pool, batch_size * sizeof(curandState)));

        device_vector* one_hop_neighborhood_vectors;
        device_vector* two_hop_neighborhood_vectors;
        device_vector_generic<int2>* d_new_edges_vec_vectors;
        
        // BUGFIX (same root cause as kernelCallStage2/3/4): out_degree_arr is
        // sized/indexed for the WHOLE SIMULATION LIFETIME (growth_in_graph_size
        // entries), not just this year's cohort. Sizing each new node's edge-
        // output buffer from out_degree_arr[0..num_new_nodes) silently uses an
        // EARLIER YEAR's recycled out-degree values as the capacity, causing
        // "device_vector_generic full" overflows once the real (correctly
        // offset) out-degree for this year's node exceeds that stale capacity.
        int growth_offset = current_graph_size - initial_graph_size;
        create_thread_vectors_bulk<int2>(num_new_nodes, out_degree_arr + growth_offset, &d_new_edges_vec_vectors);

        // ---------------------------------------------------------------------
        // MAIN LOOP
        // ---------------------------------------------------------------------
        for (int start = 0; start < num_new_nodes; start += batch_size) {
                std::chrono::steady_clock::time_point tl0 = std::chrono::steady_clock::now(); 
                int this_batch_size = std::min(batch_size, num_new_nodes - start);
                int blocks_for_this_batch = (this_batch_size + threadBlockSize - 1) / threadBlockSize;
                int num_threads = this_batch_size;

                ABMStageState* d_states = nullptr;
                cudaMalloc(&d_states, num_threads * sizeof(ABMStageState));
                cudaMemset(d_states, 0, num_threads * sizeof(ABMStageState)); 

                create_thread_vectors_int(num_threads, per_thread_vector_capacity, &one_hop_neighborhood_vectors);
                create_thread_vectors_int(num_threads, per_thread_vector_capacity, &two_hop_neighborhood_vectors);

                // ------------------------------------------------------------------
                // 1. Precompute sizes: Compute bitmap size (2 bits per node)
                // ------------------------------------------------------------------
                int num_words = ((current_graph_size) + 31) / 32;
                size_t slab_words = (size_t) this_batch_size * num_words;
                size_t slab_bytes = slab_words * sizeof(uint32_t);

                // =============================================================
                // TASK: is_use_multistage_kernel branch
                // false -> single fused ABMKernel launch (Stage1(non-warped)+2+3+4)
                // true  -> existing 4-separate-kernel-launch path (unchanged
                //          below, other than the use_warp_bfs bugfix already
                //          applied to the Stage1 branch)
                // =============================================================
                device_vector_generic<int>* citations_vectors = nullptr;
                int* per_thread_citations_vector_capacities = new int[num_threads];
                for (int i = 0; i < num_threads; i++)
                       per_thread_citations_vector_capacities[i] = per_thread_citations_vector_capacity;
                create_thread_vectors_bulk<int>(num_threads, per_thread_citations_vector_capacities, &citations_vectors);

                ThreadSets* selected_citations_thread_sets = new ThreadSets();
                create_thread_sets(num_threads, per_thread_selected_set_capacity, selected_citations_thread_sets);

                if (!abm->is_use_multistage_kernel) {
                    // ---------------------------------------------------------------
                    // FUSED PATH: one ABMKernel launch does Stage1(non-warped)+2+3+4.
                    // Fusion only supports the non-warped BFS (see ABMKernel's
                    // top-of-definition comment for why) -- use_warp_bfs is ignored
                    // in this branch by design, regardless of its value.
                    // ---------------------------------------------------------------
                    uint32_t* d_visited_slab;
                    uint32_t* d_queue_curr_slab;
                    uint32_t* d_queue_next_slab;
                    cudaMalloc(&d_visited_slab,     slab_bytes);
                    cudaMalloc(&d_queue_curr_slab,  slab_bytes);
                    cudaMalloc(&d_queue_next_slab,  slab_bytes);
                    cudaMemset(d_visited_slab,    0, slab_bytes);
                    cudaMemset(d_queue_curr_slab, 0, slab_bytes);
                    cudaMemset(d_queue_next_slab, 0, slab_bytes);

                    CompactBFSState* h_bfs_pool = new CompactBFSState[this_batch_size];
                    for (int i = 0; i < this_batch_size; ++i) {
                            h_bfs_pool[i].max_vertices  = final_graph_size;
                            h_bfs_pool[i].bitmap_words  = num_words;
                            size_t offset = (size_t) i * num_words;
                            h_bfs_pool[i].d_visited_bitmap      = d_visited_slab    + offset;
                            h_bfs_pool[i].d_queue_bitmap_curr   = d_queue_curr_slab + offset;
                            h_bfs_pool[i].d_queue_bitmap_next   = d_queue_next_slab + offset;
                    }
                    CompactBFSState* d_bfs_pool;
                    cudaMalloc(&d_bfs_pool, this_batch_size * sizeof(CompactBFSState));
                    cudaMemcpy(d_bfs_pool, h_bfs_pool,
                            this_batch_size * sizeof(CompactBFSState), cudaMemcpyHostToDevice);
                    delete[] h_bfs_pool;

                    // Fused kernel needs BOTH heap arrays live simultaneously
                    // (Stage2's and Stage3's), unlike the multistage path which
                    // allocates/uses/destroys them sequentially between separate
                    // kernel launches.
                    DeviceHeapArray<float> one_hop_heaps_fused =
                        allocate_device_heaps_host_only<float>(num_threads, per_thread_citations_vector_capacity);
                    DeviceHeapArray<float> two_hop_heaps_fused =
                        allocate_device_heaps_host_only<float>(num_threads, per_thread_citations_vector_capacity);

                    ABMKernel<<<blocks_for_this_batch, threads_per_block, 0, stream>>>(
                            start, this_batch_size, num_new_nodes,
                            abm, graph, graph->getNodeSetSize(),
                            d_nodeAttributeMap->get_device_view(),
                            d_states, d_new_nodes_arr,
                            num_generator_node_citation,
                            current_year, current_graph_size,
                            initial_graph_size, final_graph_size,
                            d_forward_adj_map_Graph, d_backward_adj_map_Graph,
                            d_bfs_pool,
                            one_hop_neighborhood_vectors, two_hop_neighborhood_vectors,
                            d_same_year_source_nodes_set_ref,
                            deviceStates_pool, seed,
                            one_hop_heaps_fused.d_heaps,
                            d_pa_arr, d_recency_arr, d_fit_arr,
                            d_pa_weight_arr, d_rec_weight_arr, d_fit_weight_arr, d_alpha_arr,
                            d_out_degree_arr,
                            abm->get_fully_random_citations(),
                            two_hop_heaps_fused.d_heaps,
                            selected_citations_thread_sets->set_refs,
                            citations_vectors,
                            d_new_edges_vec_vectors,
                            per_thread_selected_set_capacity);

                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaStreamSynchronize(stream));

                    if (d_bfs_pool) { cudaFree(d_bfs_pool); d_bfs_pool = nullptr; }
                    CUDA_CHECK(cudaFree(d_visited_slab));
                    CUDA_CHECK(cudaFree(d_queue_curr_slab));
                    CUDA_CHECK(cudaFree(d_queue_next_slab));
                    destroy_device_heaps<float>(one_hop_heaps_fused);
                    destroy_device_heaps<float>(two_hop_heaps_fused);
                    destroy_thread_vectors_int(one_hop_neighborhood_vectors, num_threads);
                    one_hop_neighborhood_vectors = nullptr;
                    destroy_thread_vectors_int(two_hop_neighborhood_vectors, num_threads);

                } else {
                // ---------------------------------------------------------------
                // MULTISTAGE PATH (existing 4-separate-kernel-launch behavior,
                // unchanged except for the use_warp_bfs bugfix already applied
                // to the Stage1 branch above).
                // ---------------------------------------------------------------
                if (abm->use_warp_bfs) {
                    //---------------------------------------------------------------------
                    // Stage 1: Compute 1/2-hop neighborhoods -- WARP-COOPERATIVE BFS
                    //
                    // ROOT-CAUSE FIX (see kernelCallStage1_warped definition above for
                    // full rationale): one warp (32 threads) now cooperates on each
                    // paper's BFS, instead of one thread per paper. This fixes both the
                    // severe warp divergence AND the undersized grid measured via ncu on
                    // both H100 and RTX 4090 for the previous 1-thread-per-paper design.
                    //
                    // Launch config: 8 warps/block (256 threads/block, unchanged total
                    // threads/block), but each block now covers only 8 papers instead
                    // of 256 -- grid size grows ~32x for the same paper count,
                    // comfortably filling both GPUs' SM counts (128-132).
                    //
                    // Per-warp scratch: same "one big bulk allocation, sliced by
                    // pointer offset" pattern already used (and already fixed for the
                    // same driver-call-count reason) elsewhere in this file -- no
                    // CompactBFSState wrapper needed since
                    // GetOneAndTwoHopNeighborhood_Warp takes raw bitmap pointers
                    // directly, and no host-side memset needed since the warp function
                    // clears its own bitmaps warp-strided at the top of its body.
                    //---------------------------------------------------------------------
                    constexpr int WARPS_PER_BLOCK = 8;   // 256 threads/block
                    const int threads_per_block_warp = WARPS_PER_BLOCK * 32;
                    const int blocks_for_this_batch_warp =
                        (this_batch_size + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

                    const int num_words = (current_graph_size + 31) / 32;
                    const size_t slab_bytes = (size_t)this_batch_size * num_words * sizeof(uint32_t);

                    uint32_t* d_visited_slab_warp;
                    uint32_t* d_curr_slab_warp;
                    uint32_t* d_next_slab_warp;
                    cudaMalloc(&d_visited_slab_warp, slab_bytes);
                    cudaMalloc(&d_curr_slab_warp,     slab_bytes);
                    cudaMalloc(&d_next_slab_warp,     slab_bytes);

                    // BUGFIX: this branch was previously calling kernelCallStage1
                    // (the non-warped kernel, expecting a CompactBFSState* +
                    // initial_graph_size) with the WARP-mode argument list
                    // (uint32_t* scratch buffers + num_words + max_vertices) --
                    // a hard signature mismatch. Fixed to call
                    // kernelCallStage1_warped, which actually matches these
                    // arguments.
                    kernelCallStage1_warped<<<blocks_for_this_batch_warp, threads_per_block_warp, 0, stream>>>(
                            start,
                            this_batch_size,
                            num_new_nodes,
                            abm,
                            graph,
                            d_forward_adj_map_Graph,
                            d_backward_adj_map_Graph,
                            d_nodeAttributeMap->get_device_view(),
                            d_states,
                            d_new_nodes_arr,
                            num_generator_node_citation,
                            one_hop_neighborhood_vectors, // host pointer to device array
                            two_hop_neighborhood_vectors, // host pointer to device array
                            d_visited_slab_warp, d_curr_slab_warp, d_next_slab_warp,
                            num_words,
                            current_graph_size
                    );
        
            
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaStreamSynchronize(stream));

                    CUDA_CHECK(cudaFree(d_visited_slab_warp));
                    CUDA_CHECK(cudaFree(d_curr_slab_warp));
                    CUDA_CHECK(cudaFree(d_next_slab_warp));
                } else { 
                    // ----------------------------------------------
                    // 2. Allocate ONE slab for each type
                    // ----------------------------------------------
                    uint32_t* d_visited_slab;
                    uint32_t* d_queue_curr_slab;
                    uint32_t* d_queue_next_slab;

                    cudaMalloc(&d_visited_slab,     slab_bytes);
                    cudaMalloc(&d_queue_curr_slab,  slab_bytes);
                    cudaMalloc(&d_queue_next_slab,  slab_bytes);

                    // ----------------------------------------------
                    // 3. Zero all slabs (3 memsets total)
                    // ----------------------------------------------
                    cudaMemset(d_visited_slab,    0, slab_bytes);
                    cudaMemset(d_queue_curr_slab, 0, slab_bytes);
                    cudaMemset(d_queue_next_slab, 0, slab_bytes);

                    // ----------------------------------------------
                    // 4. Create host pool (no cudaMalloc inside loop)
                    // ----------------------------------------------
                    CompactBFSState* h_bfs_pool = new CompactBFSState[this_batch_size];

                    for (int i = 0; i < this_batch_size; ++i) { 
                            h_bfs_pool[i].max_vertices  = final_graph_size;
                            h_bfs_pool[i].bitmap_words  = num_words;

                            size_t offset = (size_t) i * num_words;

                            h_bfs_pool[i].d_visited_bitmap      = d_visited_slab    + offset;
                            h_bfs_pool[i].d_queue_bitmap_curr   = d_queue_curr_slab + offset;
                            h_bfs_pool[i].d_queue_bitmap_next   = d_queue_next_slab + offset;
                    }

                    CompactBFSState* d_bfs_pool; 
                    cudaMalloc(&d_bfs_pool, this_batch_size * sizeof(CompactBFSState));
                    // ----------------------------------------------
                    // 5. Copy descriptors to device
                    // ----------------------------------------------
                    cudaMemcpy(d_bfs_pool,
                            h_bfs_pool,
                            this_batch_size * sizeof(CompactBFSState),
                            cudaMemcpyHostToDevice);

                    delete[] h_bfs_pool; 

                    //---------------------------------------------------------------------
                    // Stage 1: Compute 1/2-hop neighborhoods
                    //---------------------------------------------------------------------
                    kernelCallStage1<<<blocks_for_this_batch, threads_per_block, 0, stream>>>(
                            start,
                            this_batch_size,
                            num_new_nodes,
                            abm,
                            graph,
                            d_forward_adj_map_Graph,
                            d_backward_adj_map_Graph,
                            d_nodeAttributeMap->get_device_view(),
                            d_states,
                            d_new_nodes_arr,
                            num_generator_node_citation,
                            one_hop_neighborhood_vectors, // host pointer to device array
                            two_hop_neighborhood_vectors, // host pointer to device array
                            d_bfs_pool,
                            graph->getNodeSetSize()
                    );
    
        
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaStreamSynchronize(stream));

                    // Free device memory
                    if (d_bfs_pool) {
                            cudaFree(d_bfs_pool);
                            d_bfs_pool = nullptr;
                    }

                    CUDA_CHECK(cudaFree(d_visited_slab));
                    CUDA_CHECK(cudaFree(d_queue_curr_slab));
                    CUDA_CHECK(cudaFree(d_queue_next_slab));
                }
                // Reuse pre-allocated curandState pool (hoisted above batch loop)
                curandState* deviceStates = deviceStates_pool;

                int* one_hop_sizes = extract_vector_sizes(one_hop_neighborhood_vectors, num_threads);

                //---------------------------------------------------------------------
                // Stage 2: Same-year + 1-hop citations
                //---------------------------------------------------------------------
                DeviceHeapArray<float> one_hop_heaps = allocate_device_heaps_host_only<float>(num_threads, 
                                                        per_thread_citations_vector_capacity);

                delete[] one_hop_sizes;
                
                kernelCallStage2<<<blocks_for_this_batch, threads_per_block, 0, stream>>>(
                                start, this_batch_size, num_new_nodes,
                                abm, graph,
                                d_same_year_source_nodes_set_ref, 
                                d_states,
                                deviceStates, seed,
                                d_new_nodes_arr,
                                one_hop_neighborhood_vectors,
                                one_hop_heaps.d_heaps,
                                citations_vectors,
                                d_pa_arr, d_recency_arr, d_fit_arr,
                                d_pa_weight_arr, d_rec_weight_arr, d_fit_weight_arr, d_alpha_arr,
                                d_out_degree_arr,
                                abm->get_fully_random_citations(),
                                num_generator_node_citation,
                                current_year, current_graph_size,
                                initial_graph_size, final_graph_size);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaStreamSynchronize(stream));
                
                // Free 1-hop data (no longer needed after Stage 2)
                destroy_thread_vectors_int(one_hop_neighborhood_vectors, num_threads);
                one_hop_neighborhood_vectors = nullptr;

                destroy_device_heaps<float>(one_hop_heaps); 

                int* two_hop_sizes = extract_vector_sizes(two_hop_neighborhood_vectors, num_threads);
                
                DeviceHeapArray<float> heaps = allocate_device_heaps_host_only<float>(num_threads, 
                                                        per_thread_citations_vector_capacity);
                
                delete[] two_hop_sizes;
                
                //---------------------------------------------------------------------
                // Stage 3: 2-hop + random citations + edge writes
                //---------------------------------------------------------------------
                kernelCallStage3<<<blocks_for_this_batch, threads_per_block, 0, stream>>>(
                                start, this_batch_size, num_new_nodes,
                                abm, graph, graph->getNodeSetSize(),
                                d_nodeAttributeMap->get_device_view(),
                                d_states,
                                deviceStates,
                                d_new_nodes_arr,
                                two_hop_neighborhood_vectors,
                                heaps.d_heaps,
                                citations_vectors,
                                selected_citations_thread_sets->set_refs,
                                d_pa_arr, d_recency_arr, d_fit_arr,
                                d_pa_weight_arr, d_rec_weight_arr, 
                                d_fit_weight_arr, d_alpha_arr,
                                d_out_degree_arr,
                                num_generator_node_citation,
                                current_year, current_graph_size,
                                initial_graph_size, final_graph_size);

                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaStreamSynchronize(stream));

                destroy_device_heaps<float>(heaps);

                // Free 2-hop data (no longer needed after Stage 2)
                destroy_thread_vectors_int(two_hop_neighborhood_vectors, num_threads);

                kernelCallStage4<<<blocks_for_this_batch, threads_per_block, 0, stream>>>(
                                start, this_batch_size, num_new_nodes,
                                abm, graph, graph->getNodeSetSize(),
                                d_nodeAttributeMap->get_device_view(),
                                d_states,
                                deviceStates,
                                d_new_nodes_arr,
                                citations_vectors,
                                d_new_edges_vec_vectors,
                                selected_citations_thread_sets->set_refs,
                                d_out_degree_arr,
                                num_generator_node_citation,
                                current_year, current_graph_size,
                                initial_graph_size, final_graph_size, 
                                per_thread_selected_set_capacity);

                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaStreamSynchronize(stream));
                } // end multistage-path else

                // Free all per-batch temporaries (shared by both branches)
                cleanup_vectors_bulk<int>(citations_vectors, num_threads);
                
                destroy_thread_sets(selected_citations_thread_sets);
                
                // deviceStates is pooled — freed after the batch loop
                
                CUDA_CHECK(cudaFree(d_states));

                delete[] per_thread_citations_vector_capacities;
        }

        // Free pooled curandState
        CUDA_CHECK(cudaFree(deviceStates_pool));

        // Transfer new edges to host
        // destroy_thread_vectors of d_new_edges_vec_vectors already embedded in append_device_to_host
        append_device_to_host<int2>(d_new_edges_vec_vectors, new_edges_vec, num_new_nodes, out_degree_arr + growth_offset, graph->getNodeSetSize());

        // Destroy device vectors
        cleanup_vectors_bulk<int2>(d_new_edges_vec_vectors, num_new_nodes);
                
        // ---------------------------------------------------------------------
        // FINAL CLEANUP (static structures)
        // ---------------------------------------------------------------------
        delete d_same_year_source_nodes_set;
        freeDeviceMap(d_nodeAttributeMap);
        freeDeviceGraph(d_forward_adj_map_Graph);
        freeDeviceGraph(d_backward_adj_map_Graph);

        CUDA_CHECK(cudaFree(d_new_nodes_arr));
        CUDA_CHECK(cudaFree(d_pa_arr));
        CUDA_CHECK(cudaFree(d_recency_arr));
        CUDA_CHECK(cudaFree(d_fit_arr));
        CUDA_CHECK(cudaFree(d_pa_weight_arr));
        CUDA_CHECK(cudaFree(d_rec_weight_arr));
        CUDA_CHECK(cudaFree(d_fit_weight_arr));
        CUDA_CHECK(cudaFree(d_alpha_arr));
        CUDA_CHECK(cudaFree(d_out_degree_arr));

        CUDA_CHECK(cudaStreamDestroy(stream));
}

// ─────────────────────────────────────────────────────────────────────────────
//  Internal: instrumented buildOneNodeConnections
//  Identical to original except:
//    - Accepts EpochTiming* _ep_ptr (written to by timers)
//    - GpuTimer/HostTimer wrappers around each stage
//    - BUGFIX: same growth-array indexing fix as kernelCallStage2/3/4 above
//      (this function launches those same kernels, so the fix there is
//      what actually matters -- no separate fix needed in this function's
//      own body since it doesn't index the arrays directly itself).
// ─────────────────────────────────────────────────────────────────────────────
static void buildOneNodeConnections_timed(
    ABM* abm, Graph* graph,
    std::vector<int>& new_nodes_vec,
    int same_year_source_nodes_capacity,
    std::set<int> same_year_source_nodes,
    std::vector<std::pair<int,int>>& new_edges_vec,
    int num_generator_node_citation,
    double* pa_arr, double* recency_arr, double* fit_arr,
    double* pa_weight_arr, double* rec_weight_arr,
    double* fit_weight_arr, double* alpha_arr,
    int* out_degree_arr,
    int current_year, int current_graph_size,
    int initial_graph_size, int final_graph_size,
    int max_batch_size,
    EpochTiming* _ep_ptr)
    // NOTE: previously took a `bool warpEnabled` parameter here that was
    // declared but never read anywhere in this function's body, AND the
    // call site in execute() never passed it (a missing-argument bug that
    // would fail to compile once the body actually needed it). Removed --
    // abm already carries use_warp_bfs as a member, so it's read directly
    // via abm->use_warp_bfs below instead of being threaded through as a
    // separate parameter. This also means the existing call site in
    // execute() (which doesn't pass this argument) is now correct as-is.
{
    EpochTiming& _ep = *_ep_ptr;    // alias for macros

    int per_thread_citations_vector_capacity = 250;
    int per_thread_vector_capacity           = current_graph_size;
    int per_thread_selected_set_capacity     = per_thread_citations_vector_capacity;
    int num_new_nodes        = new_nodes_vec.size();
    int growth_in_graph_size = final_graph_size - initial_graph_size;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // ── Allocate device arrays ────────────────────────────────────────────────
    int* d_new_nodes_arr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_new_nodes_arr, num_new_nodes * sizeof(int)));

    double *d_pa_arr, *d_recency_arr, *d_fit_arr;
    double *d_pa_weight_arr, *d_rec_weight_arr, *d_fit_weight_arr, *d_alpha_arr;
    int*    d_out_degree_arr;
    CUDA_CHECK(cudaMalloc(&d_pa_arr,         final_graph_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_recency_arr,    final_graph_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_fit_arr,        final_graph_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pa_weight_arr,  growth_in_graph_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rec_weight_arr, growth_in_graph_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_fit_weight_arr, growth_in_graph_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_alpha_arr,      growth_in_graph_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out_degree_arr, growth_in_graph_size * sizeof(int)));

    // ── UPLOAD: score arrays + new-nodes → device (timed) ────────────────────
    {
        GpuTimer _gt; _gt.start(stream);
        CUDA_CHECK(cudaMemcpyAsync(d_new_nodes_arr, new_nodes_vec.data(),
            num_new_nodes*sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_pa_arr,        pa_arr,        final_graph_size*sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_recency_arr,   recency_arr,   final_graph_size*sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_fit_arr,        fit_arr,       final_graph_size*sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_pa_weight_arr, pa_weight_arr, growth_in_graph_size*sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_rec_weight_arr,rec_weight_arr,growth_in_graph_size*sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_fit_weight_arr,fit_weight_arr,growth_in_graph_size*sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_alpha_arr,     alpha_arr,     growth_in_graph_size*sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_out_degree_arr,out_degree_arr,growth_in_graph_size*sizeof(int),    cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        _ep.t_upload += _gt.stop(stream);
    }

    // ── CSR BUILD (timed) ─────────────────────────────────────────────────────
    DeviceGraph* d_forward_adj_map_Graph;
    DeviceGraph* d_backward_adj_map_Graph;
    CUDA_CHECK(cudaMalloc(&d_forward_adj_map_Graph,  sizeof(DeviceGraph)));
    CUDA_CHECK(cudaMalloc(&d_backward_adj_map_Graph, sizeof(DeviceGraph)));
    {
        HostTimer _ht; _ht.start();
        prepareGraph(graph->getForwardAdjMap(),  d_forward_adj_map_Graph,  graph->getNodeSetSize());
        prepareGraph(graph->getBackwardAdjMap(), d_backward_adj_map_Graph, graph->getNodeSetSize());
        int nodeAttrMapSize = graph->getNodeAttributeMapSize();
        device_map<int,Node>* d_nodeAttributeMap = new device_map<int,Node>(nodeAttrMapSize);
        convertHostMapToDeviceMap<int,Node>(graph->getNodeAttributeMap(), d_nodeAttributeMap, nodeAttrMapSize);
        _ep.t_csr_build += _ht.stop_ms();

        // ── SAME-YEAR SET ─────────────────────────────────────────────────────
        set_type* d_same_year_source_nodes_set =
            new set_type(same_year_source_nodes_capacity, cuco::empty_key<int>{empty_key_sentinel});
        convertStdSetToDeviceStaticSet(same_year_source_nodes, *d_same_year_source_nodes_set);
        auto d_same_year_source_nodes_set_ref = d_same_year_source_nodes_set->ref(
            cuco::op::insert, cuco::op::find, cuco::op::erase, cuco::op::contains);

        // ── LAUNCH CONFIG ─────────────────────────────────────────────────────
        int threadBlockSizeX = 16, threadBlockSizeY = 16;
        dim3 threads_per_block(threadBlockSizeX, threadBlockSizeY);
        int threadBlockSize = threadBlockSizeX * threadBlockSizeY;
        int batch_size = std::min(max_batch_size, num_new_nodes);
        _ep.batch_size = batch_size;   // record first-batch B
        unsigned long long seed = static_cast<unsigned long long>(time(NULL));

        curandState* deviceStates_pool = nullptr;
        CUDA_CHECK(cudaMalloc(&deviceStates_pool, batch_size * sizeof(curandState)));

        device_vector* one_hop_neighborhood_vectors;
        device_vector* two_hop_neighborhood_vectors;
        device_vector_generic<int2>* d_new_edges_vec_vectors;
        // BUGFIX: same growth-array offset issue as in the non-timed
        // buildOneNodeConnections and the kernelCallStage2/3/4 fixes above --
        // out_degree_arr must be offset by (current_graph_size -
        // initial_graph_size) before use, or edge-buffer capacities get sized
        // from a recycled earlier year's out-degree values.
        int growth_offset = current_graph_size - initial_graph_size;
        create_thread_vectors_bulk<int2>(num_new_nodes, out_degree_arr + growth_offset, &d_new_edges_vec_vectors);

        // ── MINI-BATCH LOOP ───────────────────────────────────────────────────
        for (int start = 0; start < num_new_nodes; start += batch_size) {

            int this_batch_size  = std::min(batch_size, num_new_nodes - start);
            int blocks_for_batch = (this_batch_size + threadBlockSize - 1) / threadBlockSize;
            int num_threads      = this_batch_size;
            _ep.num_minibatch++;

            ABMStageState* d_states = nullptr;
            cudaMalloc(&d_states, num_threads * sizeof(ABMStageState));
            cudaMemset(d_states, 0, num_threads * sizeof(ABMStageState));

            create_thread_vectors_int(num_threads, per_thread_vector_capacity, &one_hop_neighborhood_vectors);
            create_thread_vectors_int(num_threads, per_thread_vector_capacity, &two_hop_neighborhood_vectors);

            int num_words = ((current_graph_size) + 31) / 32;

            curandState* deviceStates = deviceStates_pool;

            device_vector_generic<int>* citations_vectors = nullptr;
            int* per_thread_citations_vector_capacities = new int[num_threads];
            for (int i = 0; i < num_threads; i++)
                per_thread_citations_vector_capacities[i] = per_thread_citations_vector_capacity;
            create_thread_vectors_bulk<int>(num_threads, per_thread_citations_vector_capacities, &citations_vectors);

            ThreadSets* selected_citations_thread_sets = new ThreadSets();
            create_thread_sets(num_threads, per_thread_selected_set_capacity, selected_citations_thread_sets);

            // =================================================================
            // TASK: is_use_multistage_kernel branch (same as buildOneNodeConnections)
            // false -> single fused ABMKernel launch (Stage1(non-warped)+2+3+4)
            // true  -> existing 4-separate-kernel-launch path, with the
            //          abm->use_warp_bfs branch now integrated here too (this
            //          function previously lacked it entirely -- Stage1 was
            //          unconditionally non-warped, regardless of
            //          abm->use_warp_bfs's value).
            // =================================================================
            if (!abm->is_use_multistage_kernel) {
                // -------------------------------------------------------------
                // FUSED PATH: one ABMKernel launch does Stage1(non-warped)+2+3+4.
                // Fusion only supports the non-warped BFS (see ABMKernel's
                // top-of-definition comment in this file) -- abm->use_warp_bfs
                // is ignored in this branch by design.
                //
                // Timing note: the entire fused kernel's time is attributed to
                // _ep.t_stage1_bfs (there is no separate bucket for a fused
                // kernel in EpochTiming). t_stage2_ws1/t_stage3_ws2/
                // t_stage4_fill will read ~0 in fused mode -- this is an
                // intentional, documented choice rather than an arbitrary
                // split across buckets that would misrepresent where time
                // actually went.
                // -------------------------------------------------------------
                size_t slab_bytes = (size_t)this_batch_size * num_words * sizeof(uint32_t);
                uint32_t *d_visited_slab, *d_queue_curr_slab, *d_queue_next_slab;
                {
                    GpuTimer _gt; _gt.start(stream);
                    cudaMalloc(&d_visited_slab,    slab_bytes);
                    cudaMalloc(&d_queue_curr_slab, slab_bytes);
                    cudaMalloc(&d_queue_next_slab, slab_bytes);
                    cudaMemset(d_visited_slab,    0, slab_bytes);
                    cudaMemset(d_queue_curr_slab, 0, slab_bytes);
                    cudaMemset(d_queue_next_slab, 0, slab_bytes);
                    cudaStreamSynchronize(stream);
                    _ep.t_slab_alloc += _gt.stop(stream);
                }

                CompactBFSState* h_bfs_pool = new CompactBFSState[this_batch_size];
                for (int i = 0; i < this_batch_size; ++i) {
                    h_bfs_pool[i].max_vertices       = final_graph_size;
                    h_bfs_pool[i].bitmap_words       = num_words;
                    size_t offset                    = (size_t)i * num_words;
                    h_bfs_pool[i].d_visited_bitmap     = d_visited_slab    + offset;
                    h_bfs_pool[i].d_queue_bitmap_curr  = d_queue_curr_slab + offset;
                    h_bfs_pool[i].d_queue_bitmap_next  = d_queue_next_slab + offset;
                }
                CompactBFSState* d_bfs_pool;
                cudaMalloc(&d_bfs_pool, this_batch_size * sizeof(CompactBFSState));
                cudaMemcpy(d_bfs_pool, h_bfs_pool,
                           this_batch_size * sizeof(CompactBFSState), cudaMemcpyHostToDevice);
                delete[] h_bfs_pool;

                DeviceHeapArray<float> one_hop_heaps_fused =
                    allocate_device_heaps_host_only<float>(num_threads, per_thread_citations_vector_capacity);
                DeviceHeapArray<float> two_hop_heaps_fused =
                    allocate_device_heaps_host_only<float>(num_threads, per_thread_citations_vector_capacity);

                {
                    GpuTimer _gt; _gt.start(stream);
                    ABMKernel<<<blocks_for_batch, threads_per_block, 0, stream>>>(
                        start, this_batch_size, num_new_nodes,
                        abm, graph, graph->getNodeSetSize(),
                        d_nodeAttributeMap->get_device_view(),
                        d_states, d_new_nodes_arr,
                        num_generator_node_citation,
                        current_year, current_graph_size,
                        initial_graph_size, final_graph_size,
                        d_forward_adj_map_Graph, d_backward_adj_map_Graph,
                        d_bfs_pool,
                        one_hop_neighborhood_vectors, two_hop_neighborhood_vectors,
                        d_same_year_source_nodes_set_ref,
                        deviceStates_pool, seed,
                        one_hop_heaps_fused.d_heaps,
                        d_pa_arr, d_recency_arr, d_fit_arr,
                        d_pa_weight_arr, d_rec_weight_arr, d_fit_weight_arr, d_alpha_arr,
                        d_out_degree_arr,
                        abm->get_fully_random_citations(),
                        two_hop_heaps_fused.d_heaps,
                        selected_citations_thread_sets->set_refs,
                        citations_vectors,
                        d_new_edges_vec_vectors,
                        per_thread_selected_set_capacity);
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    _ep.t_stage1_bfs += _gt.stop(stream);
                }

                if (d_bfs_pool) { cudaFree(d_bfs_pool); d_bfs_pool = nullptr; }
                CUDA_CHECK(cudaFree(d_visited_slab));
                CUDA_CHECK(cudaFree(d_queue_curr_slab));
                CUDA_CHECK(cudaFree(d_queue_next_slab));
                destroy_device_heaps<float>(one_hop_heaps_fused);
                destroy_device_heaps<float>(two_hop_heaps_fused);
                destroy_thread_vectors_int(one_hop_neighborhood_vectors, num_threads);
                one_hop_neighborhood_vectors = nullptr;
                destroy_thread_vectors_int(two_hop_neighborhood_vectors, num_threads);

            } else {
                // -------------------------------------------------------------
                // MULTISTAGE PATH: existing 4-separate-kernel-launch behavior,
                // now with the abm->use_warp_bfs branch integrated for Stage1
                // (previously only present in buildOneNodeConnections, not here).
                // -------------------------------------------------------------
                if (abm->use_warp_bfs) {
                    // See kernelCallStage1_warped / buildOneNodeConnections for
                    // full rationale -- identical warp-cooperative BFS path,
                    // just wrapped in this function's GpuTimer instrumentation.
                    constexpr int WARPS_PER_BLOCK = 8;
                    const int threads_per_block_warp = WARPS_PER_BLOCK * 32;
                    const int blocks_for_batch_warp =
                        (this_batch_size + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

                    size_t slab_bytes = (size_t)this_batch_size * num_words * sizeof(uint32_t);
                    uint32_t *d_visited_slab_warp, *d_curr_slab_warp, *d_next_slab_warp;
                    {
                        GpuTimer _gt; _gt.start(stream);
                        cudaMalloc(&d_visited_slab_warp, slab_bytes);
                        cudaMalloc(&d_curr_slab_warp,     slab_bytes);
                        cudaMalloc(&d_next_slab_warp,     slab_bytes);
                        cudaStreamSynchronize(stream);
                        _ep.t_slab_alloc += _gt.stop(stream);
                    }

                    {
                        GpuTimer _gt; _gt.start(stream);
                        kernelCallStage1_warped<<<blocks_for_batch_warp, threads_per_block_warp, 0, stream>>>(
                            start, this_batch_size, num_new_nodes,
                            abm, graph,
                            d_forward_adj_map_Graph, d_backward_adj_map_Graph,
                            d_nodeAttributeMap->get_device_view(),
                            d_states, d_new_nodes_arr,
                            num_generator_node_citation,
                            one_hop_neighborhood_vectors, two_hop_neighborhood_vectors,
                            d_visited_slab_warp, d_curr_slab_warp, d_next_slab_warp,
                            num_words, current_graph_size);
                        CUDA_CHECK(cudaGetLastError());
                        CUDA_CHECK(cudaStreamSynchronize(stream));
                        _ep.t_stage1_bfs += _gt.stop(stream);
                    }

                    CUDA_CHECK(cudaFree(d_visited_slab_warp));
                    CUDA_CHECK(cudaFree(d_curr_slab_warp));
                    CUDA_CHECK(cudaFree(d_next_slab_warp));

                } else {
                    // Non-warped BFS (original behavior of this function).
                    size_t slab_bytes = (size_t)this_batch_size * num_words * sizeof(uint32_t);
                    uint32_t *d_visited_slab, *d_queue_curr_slab, *d_queue_next_slab;
                    {
                        GpuTimer _gt; _gt.start(stream);
                        cudaMalloc(&d_visited_slab,    slab_bytes);
                        cudaMalloc(&d_queue_curr_slab, slab_bytes);
                        cudaMalloc(&d_queue_next_slab, slab_bytes);
                        cudaMemset(d_visited_slab,    0, slab_bytes);
                        cudaMemset(d_queue_curr_slab, 0, slab_bytes);
                        cudaMemset(d_queue_next_slab, 0, slab_bytes);
                        cudaStreamSynchronize(stream);
                        _ep.t_slab_alloc += _gt.stop(stream);
                    }

                    CompactBFSState* h_bfs_pool = new CompactBFSState[this_batch_size];
                    for (int i = 0; i < this_batch_size; ++i) {
                        h_bfs_pool[i].max_vertices       = final_graph_size;
                        h_bfs_pool[i].bitmap_words       = num_words;
                        size_t offset                    = (size_t)i * num_words;
                        h_bfs_pool[i].d_visited_bitmap     = d_visited_slab    + offset;
                        h_bfs_pool[i].d_queue_bitmap_curr  = d_queue_curr_slab + offset;
                        h_bfs_pool[i].d_queue_bitmap_next  = d_queue_next_slab + offset;
                    }
                    CompactBFSState* d_bfs_pool;
                    cudaMalloc(&d_bfs_pool, this_batch_size * sizeof(CompactBFSState));
                    cudaMemcpy(d_bfs_pool, h_bfs_pool,
                               this_batch_size * sizeof(CompactBFSState), cudaMemcpyHostToDevice);
                    delete[] h_bfs_pool;

                    {
                        GpuTimer _gt; _gt.start(stream);
                        kernelCallStage1<<<blocks_for_batch, threads_per_block, 0, stream>>>(
                            start, this_batch_size, num_new_nodes,
                            abm, graph,
                            d_forward_adj_map_Graph, d_backward_adj_map_Graph,
                            d_nodeAttributeMap->get_device_view(),
                            d_states, d_new_nodes_arr,
                            num_generator_node_citation,
                            one_hop_neighborhood_vectors,
                            two_hop_neighborhood_vectors,
                            d_bfs_pool, graph->getNodeSetSize());
                        CUDA_CHECK(cudaGetLastError());
                        CUDA_CHECK(cudaStreamSynchronize(stream));
                        _ep.t_stage1_bfs += _gt.stop(stream);
                    }

                    if (d_bfs_pool) { cudaFree(d_bfs_pool); d_bfs_pool=nullptr; }
                    CUDA_CHECK(cudaFree(d_visited_slab));
                    CUDA_CHECK(cudaFree(d_queue_curr_slab));
                    CUDA_CHECK(cudaFree(d_queue_next_slab));
                }

                int* one_hop_sizes = extract_vector_sizes(one_hop_neighborhood_vectors, num_threads);

                DeviceHeapArray<float> one_hop_heaps =
                    allocate_device_heaps_host_only<float>(num_threads, per_thread_citations_vector_capacity);
                delete[] one_hop_sizes;

                // ── STAGE 2: WS1 1-hop sampling (timed) ──────────────────────────
                {
                    GpuTimer _gt; _gt.start(stream);
                    kernelCallStage2<<<blocks_for_batch, threads_per_block, 0, stream>>>(
                        start, this_batch_size, num_new_nodes,
                        abm, graph,
                        d_same_year_source_nodes_set_ref,
                        d_states, deviceStates, seed,
                        d_new_nodes_arr,
                        one_hop_neighborhood_vectors,
                        one_hop_heaps.d_heaps,
                        citations_vectors,
                        d_pa_arr, d_recency_arr, d_fit_arr,
                        d_pa_weight_arr, d_rec_weight_arr, d_fit_weight_arr, d_alpha_arr,
                        d_out_degree_arr,
                        abm->get_fully_random_citations(),
                        num_generator_node_citation,
                        current_year, current_graph_size,
                        initial_graph_size, final_graph_size);
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    _ep.t_stage2_ws1 += _gt.stop(stream);
                }

                destroy_thread_vectors_int(one_hop_neighborhood_vectors, num_threads);
                one_hop_neighborhood_vectors = nullptr;
                destroy_device_heaps<float>(one_hop_heaps);

                int* two_hop_sizes = extract_vector_sizes(two_hop_neighborhood_vectors, num_threads);
                DeviceHeapArray<float> heaps =
                    allocate_device_heaps_host_only<float>(num_threads, per_thread_citations_vector_capacity);
                delete[] two_hop_sizes;

                // ── STAGE 3: WS2 2-hop sampling (timed) ──────────────────────────
                {
                    GpuTimer _gt; _gt.start(stream);
                    kernelCallStage3<<<blocks_for_batch, threads_per_block, 0, stream>>>(
                        start, this_batch_size, num_new_nodes,
                        abm, graph, graph->getNodeSetSize(),
                        d_nodeAttributeMap->get_device_view(),
                        d_states, deviceStates,
                        d_new_nodes_arr,
                        two_hop_neighborhood_vectors,
                        heaps.d_heaps,
                        citations_vectors,
                        selected_citations_thread_sets->set_refs,
                        d_pa_arr, d_recency_arr, d_fit_arr,
                        d_pa_weight_arr, d_rec_weight_arr, d_fit_weight_arr, d_alpha_arr,
                        d_out_degree_arr,
                        num_generator_node_citation,
                        current_year, current_graph_size,
                        initial_graph_size, final_graph_size);
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    _ep.t_stage3_ws2 += _gt.stop(stream);
                }

                destroy_device_heaps<float>(heaps);
                destroy_thread_vectors_int(two_hop_neighborhood_vectors, num_threads);

                // ── STAGE 4: random fill + edge write (timed) ────────────────────
                {
                    GpuTimer _gt; _gt.start(stream);
                    kernelCallStage4<<<blocks_for_batch, threads_per_block, 0, stream>>>(
                        start, this_batch_size, num_new_nodes,
                        abm, graph, graph->getNodeSetSize(),
                        d_nodeAttributeMap->get_device_view(),
                        d_states, deviceStates,
                        d_new_nodes_arr,
                        citations_vectors,
                        d_new_edges_vec_vectors,
                        selected_citations_thread_sets->set_refs,
                        d_out_degree_arr,
                        num_generator_node_citation,
                        current_year, current_graph_size,
                        initial_graph_size, final_graph_size,
                        per_thread_selected_set_capacity);
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    _ep.t_stage4_fill += _gt.stop(stream);
                }
            } // end multistage-path else

            cleanup_vectors_bulk<int>(citations_vectors, num_threads);
            destroy_thread_sets(selected_citations_thread_sets);
            CUDA_CHECK(cudaFree(d_states));
            delete[] per_thread_citations_vector_capacities;
        }
        // end mini-batch loop

        CUDA_CHECK(cudaFree(deviceStates_pool));

        // ── DOWNLOAD: edge buffer → host (timed) ─────────────────────────────
        {
            GpuTimer _gt; _gt.start(stream);
            append_device_to_host<int2>(d_new_edges_vec_vectors, new_edges_vec,
                                        num_new_nodes, out_degree_arr + growth_offset,
                                        graph->getNodeSetSize());
            CUDA_CHECK(cudaStreamSynchronize(stream));
            _ep.t_download += _gt.stop(stream);
        }

        _ep.edges_out = (long long)new_edges_vec.size();

        cleanup_vectors_bulk<int2>(d_new_edges_vec_vectors, num_new_nodes);
        delete d_same_year_source_nodes_set;
        freeDeviceMap(d_nodeAttributeMap);
    } // end CSR scope

    freeDeviceGraph(d_forward_adj_map_Graph);
    freeDeviceGraph(d_backward_adj_map_Graph);
    CUDA_CHECK(cudaFree(d_new_nodes_arr));
    CUDA_CHECK(cudaFree(d_pa_arr));   CUDA_CHECK(cudaFree(d_recency_arr));
    CUDA_CHECK(cudaFree(d_fit_arr));  CUDA_CHECK(cudaFree(d_pa_weight_arr));
    CUDA_CHECK(cudaFree(d_rec_weight_arr)); CUDA_CHECK(cudaFree(d_fit_weight_arr));
    CUDA_CHECK(cudaFree(d_alpha_arr)); CUDA_CHECK(cudaFree(d_out_degree_arr));
    CUDA_CHECK(cudaStreamDestroy(stream));
}


// ─────────────────────────────────────────────────────────────────────────────
//  Instrumented execute()  — drop-in replacement
// ─────────────────────────────────────────────────────────────────────────────
int execute(ABM* abm) {

    HostTimer e2e_timer; e2e_timer.start();
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

    std::vector<EpochTiming> all_epochs;   // <-- collects per-epoch data

    // ── Graph load (timed) ────────────────────────────────────────────────────
    // Covers Graph::ParseNodelist + Graph::ParseEdgelist (both called from
    // the Graph constructor) -- this is real file I/O that was previously
    // invisible to any timing report; graph.cu prints its own ad-hoc
    // "Elapsed time: ParseNodelist/ParseEdgelist" lines to stdout, but those
    // durations were never captured or added into the pipeline breakdown.
    double t_graph_parse_ms;
    Graph* graph;
    {
        HostTimer _ht; _ht.start();
        graph = new Graph(abm->edgelist, abm->nodelist);
        t_graph_parse_ms = _ht.stop_ms();
    }
    abm->WriteToLogFile("loaded graph", Log::info);

    // ── Fitness init + array allocation + weight/alpha/outdegree population
    //    (timed) ─────────────────────────────────────────────────────────────
    double t_init_ms;
    int start_year, next_node_id, initial_next_node_id, initial_graph_size,
        final_graph_size, growth_in_graph_size;
    int*    in_degree_arr;
    int*    fitness_arr;
    double* pa_arr;
    double* fit_arr;
    double* recency_arr;
    double* random_weight_arr;
    double* current_score_arr;
    double* pa_weight_arr;
    double* rec_weight_arr;
    double* fit_weight_arr;
    double* alpha_arr;
    int*    out_degree_arr;
    {
        HostTimer _ht; _ht.start();

        abm->InitializeFitness(graph);
        abm->WriteToLogFile("initialized fitness for the seed graph", Log::debug);

        start_year           = abm->GetMaxYear(graph) + 1;
        next_node_id         = abm->GetMaxNode(graph) + 1;
        initial_next_node_id = graph->getNodeSetSize();
        initial_graph_size   = graph->GetNodeSet().size();
        final_graph_size     = abm->GetFinalGraphSize(graph);
        growth_in_graph_size = final_graph_size - initial_graph_size;

        in_degree_arr     = new int   [final_graph_size];
        fitness_arr       = new int   [final_graph_size];
        pa_arr            = new double[final_graph_size];
        fit_arr           = new double[final_graph_size];
        recency_arr       = new double[final_graph_size];
        random_weight_arr = new double[final_graph_size];
        current_score_arr = new double[final_graph_size];
        pa_weight_arr     = new double[growth_in_graph_size];
        rec_weight_arr    = new double[growth_in_graph_size];
        fit_weight_arr    = new double[growth_in_graph_size];
        alpha_arr         = new double[growth_in_graph_size];
        out_degree_arr    = new int   [growth_in_graph_size];

        abm->PopulateWeightArrs(pa_weight_arr, rec_weight_arr, fit_weight_arr, growth_in_graph_size);
        abm->PopulateAlphaArr(alpha_arr, growth_in_graph_size);
        abm->PopulateOutDegreeArr(out_degree_arr, growth_in_graph_size);

        t_init_ms = _ht.stop_ms();
    }

    std::vector<int> new_nodes_vec;
    std::set<int>    same_year_source_nodes;
    std::vector<std::pair<int,int>> new_edges_vec;
    int max_batch_size = abm->get_max_batch_size();

    // =========================================================================
    //  EPOCH LOOP
    //
    //  DEBUGGING AID (recommended, not yet added): log
    //  `current_graph_size`, `num_new_nodes`, and `new_nodes_vec.size()`
    //  right after this year's node-init loop and again immediately before
    //  `graph->node_set.insert(...)` below. If GPU output still diverges
    //  from CPU after the growth_idx fix, comparing these per-year logs
    //  between a CPU run and a GPU run on the same seed graph is the
    //  fastest way to localize a SECOND divergence source (e.g. a batch
    //  boundary or stream-sync issue in edge insertion), per the
    //  compounding-growth mechanism discussed separately.
    // =========================================================================
    for (int current_year = start_year;
         current_year < start_year + abm->num_cycles;
         current_year++)
    {
        int current_graph_size = graph->GetNodeSet().size();
        int num_new_nodes      = (int)std::ceil(current_graph_size * abm->growth_rate);

        // start new epoch record
        EpochTiming _ep;
        _ep.year = current_year;
        _ep.N    = current_graph_size;
        _ep.delta= num_new_nodes;

        printf("\n[EP %d] N=%d delta=%d", current_year, current_graph_size, num_new_nodes);

        // ── CPU pre: score arrays (timed) ─────────────────────────────────────
        HOST_TIME(t_fill_indeg,   abm->FillInDegreeArr(graph, in_degree_arr));
        HOST_TIME(t_fill_fitness, abm->FillFitnessArr(graph, current_year, fitness_arr));
        HOST_TIME(t_fill_recency, abm->FillRecencyArr(graph, current_year, recency_arr));
        {
            HostTimer _ht; _ht.start();
            abm->CalculateScores(in_degree_arr, pa_arr,  current_graph_size);
            abm->CalculateScores(fitness_arr,   fit_arr, current_graph_size);
            _ep.t_calc_scores += _ht.stop_ms();
        }

        // ── New-node init (timed) ─────────────────────────────────────────────
        {
            HostTimer _ht; _ht.start();
            for (int i = 0; i < num_new_nodes; i++) {
                int seq = current_graph_size + i;
                graph->continuous_node_mapping[next_node_id]         = seq;
                graph->reverse_continuous_node_mapping[seq]          = next_node_id;
                new_nodes_vec.push_back(seq);
                graph->SetIntAttribute("year", seq, current_year);
                graph->setType(AGENT_TYPE, seq);
                next_node_id++;
            }
            _ep.t_node_init += _ht.stop_ms();
        }

        if (!abm->is_use_batching()) {
            max_batch_size = num_new_nodes;
        } else {
            max_batch_size = std::min(max_batch_size, num_new_nodes);
        }

        // ── Same-year set (timed) ─────────────────────────────────────────────
        HOST_TIME(t_same_year,
            abm->FillSameYearSourceNodes(same_year_source_nodes, new_nodes_vec.size()));
        int same_year_source_nodes_capacity = same_year_source_nodes.size();

        // ── Generator-node assignment (timed) ────────────────────────────────
        {
            HostTimer _ht; _ht.start();
            for (size_t i = 0; i < new_nodes_vec.size(); i++) {
                int new_node        = new_nodes_vec[i];
                int generatorNodeId = abm->getGeneratorNode(graph);
                abm->updateGraphAttributesGeneratorNode(graph, new_node, generatorNodeId);
            }
            _ep.t_gen_assign += _ht.stop_ms();
        }

        int num_generator_node_citation = 1;

        // ── GPU pipeline (all stages timed inside) ───────────────────────────
        try {
            buildOneNodeConnections_timed(
                abm, graph,
                new_nodes_vec, same_year_source_nodes_capacity,
                same_year_source_nodes, new_edges_vec,
                num_generator_node_citation,
                pa_arr, recency_arr, fit_arr,
                pa_weight_arr, rec_weight_arr, fit_weight_arr, alpha_arr,
                out_degree_arr,
                current_year, current_graph_size,
                initial_graph_size, final_graph_size,
                max_batch_size,
                &_ep);
        } catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
            return 1;
        }

        // ── Host graph update: batch edge insertion (timed) ──────────────────
        {
            HostTimer _ht; _ht.start();
            graph->node_set.insert(new_nodes_vec.begin(), new_nodes_vec.end());
            std::sort(new_edges_vec.begin(), new_edges_vec.end());
            const size_t E = new_edges_vec.size();
            std::vector<std::pair<int,int>> rev_edges(E);
            for (size_t ei=0;ei<E;++ei)
                rev_edges[ei]={new_edges_vec[ei].second, new_edges_vec[ei].first};
            std::sort(rev_edges.begin(), rev_edges.end());
            std::unordered_set<int> udn;
            udn.reserve((new_nodes_vec.size()+E)*2);
            udn.insert(new_nodes_vec.begin(), new_nodes_vec.end());

            #pragma omp parallel sections num_threads(2)
            {
                #pragma omp section
                {
                    auto it = new_edges_vec.cbegin();
                    while (it != new_edges_vec.cend()) {
                        const int src = it->first;
                        const auto run_end = std::upper_bound(
                            it, new_edges_vec.cend(),
                            std::make_pair(src, std::numeric_limits<int>::max()));
                        auto& fs = graph->forward_adj_map[src];
                        for (auto e=it;e!=run_end;++e) fs.insert(fs.end(),e->second);
                        it = run_end;
                    }
                }
                #pragma omp section
                {
                    auto it = rev_edges.cbegin();
                    while (it != rev_edges.cend()) {
                        const int dst = it->first;
                        const auto run_end = std::upper_bound(
                            it, rev_edges.cend(),
                            std::make_pair(dst, std::numeric_limits<int>::max()));
                        auto& bs = graph->backward_adj_map[dst];
                        for (auto e=it;e!=run_end;++e) bs.insert(bs.end(),e->second);
                        it = run_end;
                    }
                }
            }
            for (const auto& [src,dst]:new_edges_vec) udn.insert(dst);
            _ep.t_edge_insert += _ht.stop_ms();

            // ── In-degree update (timed) ──────────────────────────────────────
            HOST_TIME(t_indeg_update,
                graph->updateNodeInDegreeOutDegree(new_nodes_vec, udn, current_year));
        }

        // ── Fitness assignment (timed) ────────────────────────────────────────
        {
            HostTimer _ht; _ht.start();
            abm->AssignPeakFitnessValues(graph, new_nodes_vec);
            abm->AssignFitnessLagDuration(graph, new_nodes_vec);
            abm->AssignFitnessPeakDuration(graph, new_nodes_vec);
            abm->PlantNodes(graph, new_nodes_vec, current_year - start_year + 1);
            _ep.t_fitness_asgn += _ht.stop_ms();
        }

        // ── Adaptive batch size ───────────────────────────────────────────────
        if ((int)new_nodes_vec.size() > max_batch_size)
            max_batch_size = (int)std::ceil(max_batch_size * (1 - 0.5 * abm->growth_rate));

        new_nodes_vec.clear();
        new_edges_vec.clear();
        same_year_source_nodes.clear();

        // store epoch record and print compact line
        all_epochs.push_back(_ep);
        printf("  pre=%.0f csr=%.0f S1=%.0f S2=%.0f S3=%.0f S4=%.0f "
               "xfer=%.0f upd=%.0f  total=%.0f ms  E/s=%.0f\n",
            _ep.host_preproc(), _ep.t_csr_build,
            (double)_ep.t_stage1_bfs, (double)_ep.t_stage2_ws1,
            (double)_ep.t_stage3_ws2, (double)_ep.t_stage4_fill,
            (double)_ep.transfers(), _ep.host_update(),
            _ep.epoch_total(), _ep.edges_per_sec());

    } // end epoch loop

    // ── Output write (timed) ──────────────────────────────────────────────────
    // Covers WriteGraph, UpdateGraphAttributesWeights/Alphas/OutDegrees, the
    // in/out-degree attribute-copy loop, and WriteAttributes. This entire
    // phase was previously unmeasured -- per the RTX4090-vs-H100 log
    // comparison, this is where the LARGEST unaccounted time gap actually
    // lived (tens of seconds difference between machines that the old
    // report couldn't explain, since nothing after the epoch loop was timed).
    double t_output_write_ms;
    {
        HostTimer _ht; _ht.start();

        abm->WriteToLogFile("finished sim", Log::info);
        graph->WriteGraph(abm->output_file);
        abm->UpdateGraphAttributesWeights(graph, initial_next_node_id,
            pa_weight_arr, rec_weight_arr, fit_weight_arr, growth_in_graph_size);
        abm->UpdateGraphAttributesAlphas(graph, initial_next_node_id,
            alpha_arr, growth_in_graph_size);
        abm->UpdateGraphAttributesOutDegrees(graph, initial_next_node_id,
            out_degree_arr, growth_in_graph_size);
        for (auto const& nid : graph->GetNodeSet()) {
            graph->SetIntAttribute("in_degree",  nid, graph->GetInDegree(nid));
            graph->SetIntAttribute("out_degree", nid, graph->GetOutDegree(nid));
        }
        graph->WriteAttributes(abm->auxiliary_information_file);

        t_output_write_ms = _ht.stop_ms();
    }

    // ── E2E timing ────────────────────────────────────────────────────────────
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    double e2e_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();

    std::ostringstream msg;
    msg << "\nE2E Time, model: PA"
        << "  num_cycles=" << abm->num_cycles
        << "  growth_rate=" << (100.0*abm->growth_rate) << "%"
        << "  threads=" << abm->num_processors
        << "  elapsed=" << (long long)(e2e_ms/1000) << "s";
    abm->WriteToLogFile(msg.str(), Log::info);
    std::cout << msg.str() << std::endl;

    // ── Print full epoch breakdown report (unchanged) ─────────────────────────
    print_epoch_report(all_epochs, e2e_ms, abm->num_processors,
                       "PA", abm->growth_rate);

    // ── Print FULL PIPELINE breakdown, covering the entire process from
    //    ABM construction through final output write, summing to ~100% of
    //    the measured E2E wall-clock time. This is a HIGHER-LEVEL report
    //    than print_epoch_report above (which only covers the epoch loop
    //    itself) -- it wraps around it, adding the three phases that were
    //    previously invisible to any timing report: ABM construction I/O,
    //    graph parsing, and post-loop output writing. ──────────────────────
    {
        // Note: this execute()-measured e2e_ms does NOT include time spent
        // in main.cu before/after execute() (CLI arg parsing, `delete abm`,
        // process startup/teardown) -- those are negligible (sub-millisecond
        // argparse calls) except for g_abm_construction_ms, which IS folded
        // in explicitly below since it's genuine file I/O, not overhead.
        double simulation_ms = 0.0;
        for (const auto& ep : all_epochs) simulation_ms += ep.epoch_total();

        // e2e_ms is measured from t0 (start of execute(), AFTER ABM
        // construction already happened in main.cu) to t1 (end of output
        // write). g_abm_construction_ms happened BEFORE t0, so the true
        // full-pipeline total is e2e_ms + g_abm_construction_ms, not e2e_ms
        // alone -- otherwise ABM construction time would be added on top of
        // 100% rather than being one of the slices that makes up 100%.
        double full_pipeline_ms = e2e_ms + g_abm_construction_ms;

        double measured_sum_ms = g_abm_construction_ms + t_graph_parse_ms +
                                 t_init_ms + simulation_ms + t_output_write_ms;
        double unaccounted_ms = full_pipeline_ms - measured_sum_ms;

        auto pct = [&](double part) {
            return full_pipeline_ms > 0.0 ? (100.0 * part / full_pipeline_ms) : 0.0;
        };

        printf("\n========================================================================================================================\n");
        printf("  FULL PIPELINE BREAKDOWN (ABM construction -> parse -> init -> simulation -> output write)  model=PA  growth=%.1f%%  threads=%d\n",
               100.0 * abm->growth_rate, abm->num_processors);
        printf("========================================================================================================================\n");
        printf("  %-70s : %10.0f ms  (%5.1f%%)\n", "ABM construction (out-degree bag, recency probs, planted nodes)", g_abm_construction_ms, pct(g_abm_construction_ms));
        printf("  %-70s : %10.0f ms  (%5.1f%%)\n", "Graph parse (nodelist + edgelist)", t_graph_parse_ms, pct(t_graph_parse_ms));
        printf("  %-70s : %10.0f ms  (%5.1f%%)\n", "Init (fitness + array alloc + weight/alpha/outdegree populate)", t_init_ms, pct(t_init_ms));
        printf("  %-70s : %10.0f ms  (%5.1f%%)\n", "Simulation (epoch loop total -- see breakdown above)", simulation_ms, pct(simulation_ms));
        printf("  %-70s : %10.0f ms  (%5.1f%%)\n", "Output write (WriteGraph + UpdateGraphAttributes* + WriteAttributes)", t_output_write_ms, pct(t_output_write_ms));
        printf("------------------------------------------------------------------------------------------------------------------------\n");
        printf("  %-70s : %10.0f ms  (%5.1f%%)\n", "TOTAL (full pipeline, measured)", measured_sum_ms, pct(measured_sum_ms));
        printf("  %-70s : %10.0f ms  (%5.1f%%)\n", "Unaccounted (measurement gaps / process overhead outside execute())", unaccounted_ms, pct(unaccounted_ms));
        printf("  %-70s : %10.0f ms  (100.0%%)\n", "FULL PIPELINE TOTAL", full_pipeline_ms);
        printf("========================================================================================================================\n");
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    delete[] in_degree_arr;  delete[] fitness_arr;    delete[] pa_arr;
    delete[] fit_arr;        delete[] recency_arr;    delete[] random_weight_arr;
    delete[] current_score_arr;
    delete[] pa_weight_arr;  delete[] rec_weight_arr; delete[] fit_weight_arr;
    delete[] alpha_arr;      delete[] out_degree_arr;
    delete graph;
    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// execute2() calls buildOneNodeConnections() (non-timed), which launches the
// SAME kernelCallStage2/3/4 fixed above -- so execute2()'s correctness is
// fixed transitively. No changes needed in this function's own body.
// ─────────────────────────────────────────────────────────────────────────────
int execute2(ABM* abm) {
        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        printf("\nStart execute."); 

        Graph* graph = new Graph(abm->edgelist, abm->nodelist);
        printf("\nGraph init done...");
        abm->WriteToLogFile("loaded graph", Log::info);

        printf("\ncalling InitializeFitness...");
        abm->InitializeFitness(graph);
        abm->WriteToLogFile("initialized fitness for the seed graph", Log::debug);

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        abm->WriteToLogFile("forward built", Log::debug);

        int start_year = abm->GetMaxYear(graph) + 1;
        int next_node_id = abm->GetMaxNode(graph) + 1;
        int initial_next_node_id = graph->getNodeSetSize();

        int initial_graph_size = graph->GetNodeSet().size();
        int final_graph_size = abm->GetFinalGraphSize(graph);
        int growth_in_graph_size = (final_graph_size - initial_graph_size);

        abm->WriteToLogFile("final graph size is " + std::to_string(final_graph_size), Log::info);
        printf("\nfinal graph size is %d from initial_graph_size of %d.", final_graph_size, initial_graph_size);
        int* in_degree_arr = new int[final_graph_size];
        int* fitness_arr = new int[final_graph_size];
        double* pa_arr = new double[final_graph_size];
        double* fit_arr = new double[final_graph_size];
        double* recency_arr = new double[final_graph_size];
        double* random_weight_arr = new double[final_graph_size];
        double* current_score_arr = new double[final_graph_size]; 
        
        std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();

        double* pa_weight_arr = new double[growth_in_graph_size];
        double* rec_weight_arr = new double[growth_in_graph_size];
        double* fit_weight_arr = new double[growth_in_graph_size];
        double* alpha_arr = new double[growth_in_graph_size];
        int* out_degree_arr = new int[growth_in_graph_size];
    
        abm->WriteToLogFile("allocated arrays", Log::debug);

        std::chrono::steady_clock::time_point t5 = std::chrono::steady_clock::now();
        
        abm->PopulateWeightArrs(pa_weight_arr, rec_weight_arr, fit_weight_arr, growth_in_graph_size);
        abm->WriteToLogFile("populated weight arrays", Log::debug);
        abm->PopulateAlphaArr(alpha_arr, growth_in_graph_size);
        abm->WriteToLogFile("populated alpha array", Log::debug);
        abm->PopulateOutDegreeArr(out_degree_arr, growth_in_graph_size);
        printf("\n populated out degree array.");

        std::vector<int> new_nodes_vec;
        std::set<int> same_year_source_nodes;
        std::vector<std::pair<int, int>> new_edges_vec;
        std::chrono::steady_clock::time_point t7 = std::chrono::steady_clock::now();
        double d71 = 0, d72 = 0, d73 = 0, d74 = 0, d75 = 0, d76 = 0, d77 = 0, d79 = 0, d781 = 0, d7081 = 0, d7980 = 0, d8081 = 0, d7981 = 0;
        int max_batch_size = 23000;
        for (int current_year = start_year; current_year < start_year + abm->num_cycles; current_year++) {
                printf("\n Entering loop for year = %d", current_year);
                std::chrono::steady_clock::time_point t70 = std::chrono::steady_clock::now();
                 
                int current_graph_size = graph->GetNodeSet().size();

                abm->WriteToLogFile("current year is: " + std::to_string(current_year) + 
                                " and the graph is " + std::to_string(current_graph_size) + " nodes large", Log::info);
                abm->FillInDegreeArr(graph, in_degree_arr);
                abm->WriteToLogFile("indegree for current year filled", Log::debug);
                abm->FillFitnessArr(graph, current_year, fitness_arr);
                abm->WriteToLogFile("fitness for current year filled", Log::debug);
        
                std::chrono::steady_clock::time_point t71 = std::chrono::steady_clock::now();
                auto duration7071 = std::chrono::duration_cast<std::chrono::milliseconds>(t71 - t70);
                d71 += duration7071.count();
                std::cout << "\nElapsed time: 70-71: total : " << d71/1000 << " secs, iter cost : " << duration7071.count()/1000 << " seconds" << std::endl;
                
                abm->FillRecencyArr(graph, current_year, recency_arr);
                abm->WriteToLogFile("recency for current year calculated", Log::debug);
                abm->CalculateScores(in_degree_arr, pa_arr, current_graph_size);

                std::chrono::steady_clock::time_point t72 = std::chrono::steady_clock::now();
                auto duration7172 = std::chrono::duration_cast<std::chrono::milliseconds>(t72 - t71);
                d72 += duration7172.count();
                std::cout << "\nElapsed time: 71-72: total : " << d72/1000 << " secs, iter cost : " << duration7172.count()/1000 << " seconds" << std::endl;
                
                abm->WriteToLogFile("indegree gammad", Log::debug);
                abm->CalculateScores(fitness_arr, fit_arr, current_graph_size);
                abm->WriteToLogFile("fitness gammad", Log::debug);

                std::chrono::steady_clock::time_point t73 = std::chrono::steady_clock::now();
                auto duration7273 = std::chrono::duration_cast<std::chrono::milliseconds>(t73 - t72);
                d73 += duration7273.count();
                std::cout << "\nElapsed time: 72-73: total : " << d73 << " millisecs, iter cost : " << duration7273.count() << " mseconds" << std::endl;

                int num_new_nodes = std::ceil(current_graph_size * abm->growth_rate);
                printf("\nnum_new_nodes = %d, current_graph_size = %d, growth rate = %lf ", num_new_nodes, current_graph_size, abm->growth_rate);
                std::chrono::steady_clock::time_point t74 = std::chrono::steady_clock::now();
                auto duration7374 = std::chrono::duration_cast<std::chrono::milliseconds>(t74 - t73);
                d74 += duration7374.count();
                std::cout << "\nElapsed time: 73-74: total : " << d74 
                        << " millisecs, iter cost : " << duration7374.count() << " mseconds" << std::endl;
                abm->WriteToLogFile("making " + std::to_string(num_new_nodes) + " nodes abm year", Log::info);
                for(int i = 0; i < num_new_nodes; i++) {
                        int next_node_seq_id = current_graph_size + i;
                        graph->continuous_node_mapping[next_node_id] = next_node_seq_id;
                        graph->reverse_continuous_node_mapping[next_node_seq_id] = next_node_id;
                        new_nodes_vec.push_back(next_node_seq_id);
                        graph->SetIntAttribute("year", next_node_seq_id, current_year);
                        graph->setType(AGENT_TYPE, next_node_seq_id);
                        next_node_id++;
                }
        
                std::chrono::steady_clock::time_point t75 = std::chrono::steady_clock::now();
                auto duration7475 = std::chrono::duration_cast<std::chrono::milliseconds>(t75 - t74);
                d75 += duration7475.count();
                std::cout << "\nElapsed time: 74-75: total : " << d75 << " millisecs, iter cost : " << duration7475.count() << " mseconds" << std::endl;

                abm->WriteToLogFile("all new nodes initialized with years and mapped", Log::debug);
                std::cout << "\nall new nodes initialized with years and mapped. FillSameYearSourceNodes with new_nodes_vec.size() = "<< new_nodes_vec.size();
                abm->FillSameYearSourceNodes(same_year_source_nodes, new_nodes_vec.size());
                std::cout << "\nchecking same_year_source_nodes size.";
                int same_year_source_nodes_capacity = same_year_source_nodes.size();

                std::chrono::steady_clock::time_point t76 = std::chrono::steady_clock::now();
                        auto duration7576 = std::chrono::duration_cast<std::chrono::milliseconds>(t76 - t75);
                d76 += duration7576.count();
                std::cout << "\nElapsed time: 75-76: total : " << d76 << " millisecs, iter cost : " << duration7576.count() << " mseconds" << std::endl;
                std::cout << "\nnew_nodes_vec.size() b4 : " << new_nodes_vec.size() << std::endl;
        
                for(size_t i = 0; i < new_nodes_vec.size(); i++) {
                        int new_node = new_nodes_vec[i];
                        int generatorNodeId = abm->getGeneratorNode(graph);
                        abm->updateGraphAttributesGeneratorNode(graph, new_node, generatorNodeId);
                }

                std::cout << "\nupdated  updateGraphAttributesGeneratorNode .... "; 
                std::chrono::steady_clock::time_point t77 = std::chrono::steady_clock::now();
                        auto duration7677 = std::chrono::duration_cast<std::chrono::milliseconds>(t77 - t76);
                        d77 += duration7677.count();
                std::cout << "\nElapsed time: 76-77: total : " << d77 << " millisecs, iter cost : " << duration7677.count() << " mseconds" << std::endl;
                int num_generator_node_citation = 1;
                std::chrono::steady_clock::time_point t78 = std::chrono::steady_clock::now();
                
                std::cout << "\ncontinuous_node_mapping.size() b5 : " << graph->continuous_node_mapping.size() 
                        << ", reverse_continuous_node_mapping size = "<< graph->reverse_continuous_node_mapping.size() << std::endl; 
                 
                try {
                        buildOneNodeConnections(abm, graph,
                                new_nodes_vec,
                                same_year_source_nodes_capacity,
                                same_year_source_nodes,
                                new_edges_vec,
                                num_generator_node_citation,
                                pa_arr, recency_arr, fit_arr,
                                pa_weight_arr,  rec_weight_arr, fit_weight_arr, alpha_arr,
                                out_degree_arr,
                                current_year, 
                                current_graph_size, 
                                initial_graph_size,
                                final_graph_size,
                                max_batch_size);
                } catch (const std::runtime_error& e) {
                        std::cerr << "Caught CUDA exception in buildOneNodeConnections: " << e.what() << std::endl;
                        return 1;
                } catch (const std::exception& e) {
                        std::cerr << "Caught generic exception in buildOneNodeConnections: " << e.what() << std::endl;
                        return 1;
                }
                
                std::chrono::steady_clock::time_point t781 = std::chrono::steady_clock::now();
                auto duration78781 = std::chrono::duration_cast<std::chrono::milliseconds>(t781 - t78);
                d781 += duration78781.count();

                std::cout << "\nYear " << current_year << ": " << num_new_nodes << " new nodes, "
                          << new_edges_vec.size() << " new edges\n";
                std::cout << "\nElapsed time: 78-781 : total : " << d781/1000.0 << " secs, " 
                        << " this iter cost : " << duration78781.count()/1000.0 << " seconds" << std::endl;

                graph->node_set.insert(new_nodes_vec.begin(), new_nodes_vec.end());

                std::sort(new_edges_vec.begin(), new_edges_vec.end());

                const size_t E = new_edges_vec.size();
                std::vector<std::pair<int,int>> rev_edges(E);
                for (size_t ei = 0; ei < E; ++ei)
                    rev_edges[ei] = {new_edges_vec[ei].second, new_edges_vec[ei].first};
                std::sort(rev_edges.begin(), rev_edges.end());

                std::unordered_set<int> updated_destination_nodes;
                updated_destination_nodes.reserve((new_nodes_vec.size() + E) * 2);
                updated_destination_nodes.insert(new_nodes_vec.begin(), new_nodes_vec.end());

                #pragma omp parallel sections num_threads(2)
                {
                    #pragma omp section
                    {
                        auto it = new_edges_vec.cbegin();
                        while (it != new_edges_vec.cend()) {
                            const int src = it->first;
                            const auto run_end = std::upper_bound(
                                it, new_edges_vec.cend(),
                                std::make_pair(src, std::numeric_limits<int>::max()));

                            auto& fwd_set = graph->forward_adj_map[src];
                            for (auto e = it; e != run_end; ++e)
                                fwd_set.insert(fwd_set.end(), e->second);

                            it = run_end;
                        }
                    }

                    #pragma omp section
                    {
                        auto it = rev_edges.cbegin();
                        while (it != rev_edges.cend()) {
                            const int dst = it->first;
                            const auto run_end = std::upper_bound(
                                it, rev_edges.cend(),
                                std::make_pair(dst, std::numeric_limits<int>::max()));

                            auto& bwd_set = graph->backward_adj_map[dst];
                            for (auto e = it; e != run_end; ++e)
                                bwd_set.insert(bwd_set.end(), e->second);

                            it = run_end;
                        }
                    }
                }

                for (const auto& [src, dst] : new_edges_vec)
                    updated_destination_nodes.insert(dst);

                std::chrono::steady_clock::time_point t79 = std::chrono::steady_clock::now();
                auto duration7879 = std::chrono::duration_cast<std::chrono::milliseconds>(t79 - t78);
                d79 += duration7879.count();
                std::cout << "\nElapsed time: 78-79 : total : " << d79/1000.0 << " secs, this iter cost : " 
                        << duration7879.count()/1000.0 << " seconds" << std::endl;
                
                graph->updateNodeInDegreeOutDegree(new_nodes_vec, updated_destination_nodes, current_year);
                std::chrono::steady_clock::time_point t80 = std::chrono::steady_clock::now();
                abm->AssignPeakFitnessValues(graph, new_nodes_vec);
                abm->WriteToLogFile("assigned peak fitness for new nodes", Log::debug);

                abm->AssignFitnessLagDuration(graph, new_nodes_vec);
                abm->WriteToLogFile("assigned fitness lag duration for new nodes", Log::debug);
                abm->AssignFitnessPeakDuration(graph, new_nodes_vec);
                abm->WriteToLogFile("assigned fitness peak duration for new nodes", Log::debug);
                abm->PlantNodes(graph, new_nodes_vec, current_year - start_year + 1);

                std::chrono::steady_clock::time_point t81 = std::chrono::steady_clock::now();
                if (new_nodes_vec.size() > max_batch_size) {
                        max_batch_size = std::ceil(max_batch_size * (1 - 0.5 * abm->growth_rate));
                }
                new_nodes_vec.clear();
                new_edges_vec.clear(); 
                
                auto duration7980 = std::chrono::duration_cast<std::chrono::milliseconds>(t80 - t79);
                auto duration8081 = std::chrono::duration_cast<std::chrono::milliseconds>(t81 - t80);
                auto duration7081 = std::chrono::duration_cast<std::chrono::milliseconds>(t81 - t70);
                d7081 += duration7081.count();
                d7980 += duration7980.count();
                d8081 += duration8081.count();
                auto duration7981 = std::chrono::duration_cast<std::chrono::milliseconds>(t81 - t79);
                d7981 += duration7981.count();
                std::cout << "\nElapsed time: 79-81 : total : " << d7981/1000 << " secs, this iter cost : " << duration7981.count()/1000 << " seconds" << std::endl; 
                std::cout << "\nElapsed time: 79-80 : total : " << d7980/1000 << " secs, this iter cost : " << duration7980.count()/1000 << " seconds" << std::endl; 
                std::cout << "\nElapsed time: 80-81 : total : " << d8081/1000 << " secs, this iter cost : " << duration8081.count()/1000 << " seconds" << std::endl; 
                std::cout << "\nElapsed time: 70-81 : total : " << d7081/1000 << " secs, this iter cost : " << duration7081.count()/1000 << " seconds" << std::endl; 
        }

        std::chrono::steady_clock::time_point t8 = std::chrono::steady_clock::now();
        auto duration78 = std::chrono::duration_cast<std::chrono::milliseconds>(t8 - t7);
        std::cout << "\nElapsed time: 7-8 :  " << duration78.count()/1000 << " seconds" << std::endl;
             
        std::cout<<"\ncompute done:: graph size = "<< graph->GetNodeSet().size();
        abm->WriteToLogFile("finished sim", Log::info);

        std::chrono::steady_clock::time_point t8p1 = std::chrono::steady_clock::now();
        auto duration88p1 = std::chrono::duration_cast<std::chrono::milliseconds>(t8p1 - t8);
        std::cout << "\nElapsed time: 8-8.1 :  " << duration88p1.count()/1000 << " seconds" << std::endl;

        std::cout << "\nWriting Graph to file: " << abm->output_file.c_str() << std::endl;
        graph->WriteGraph(abm->output_file);

        std::chrono::steady_clock::time_point t9 = std::chrono::steady_clock::now();
        auto duration8p19 = std::chrono::duration_cast<std::chrono::milliseconds>(t9 - t8p1);
        std::cout << "\nElapsed time: 8.1-9 :  " << duration8p19.count()/1000 << " seconds" << std::endl;

        abm->UpdateGraphAttributesWeights(graph, initial_next_node_id, pa_weight_arr, rec_weight_arr, fit_weight_arr, growth_in_graph_size);
        abm->WriteToLogFile("update 1", Log::debug);
        abm->UpdateGraphAttributesAlphas(graph, initial_next_node_id, alpha_arr, growth_in_graph_size);
        abm->WriteToLogFile("update 2", Log::debug);
        abm->UpdateGraphAttributesOutDegrees(graph, initial_next_node_id, out_degree_arr, growth_in_graph_size);
        abm->WriteToLogFile("update 3", Log::debug);

        std::chrono::steady_clock::time_point t10 = std::chrono::steady_clock::now();
        auto duration910 = std::chrono::duration_cast<std::chrono::milliseconds>(t10 - t9);
        std::cout << "\nElapsed time: 9-10 :  " << duration910.count()/1000 << " seconds" << std::endl;

        std::cout<<"\n graph size = "<< graph->GetNodeSet().size();
        graph->WriteAttributes(abm->auxiliary_information_file);
        abm->WriteToLogFile("wrote graph", Log::info);
        std::chrono::steady_clock::time_point t11 = std::chrono::steady_clock::now();

        delete[] in_degree_arr;
        delete[] fitness_arr;
        delete[] pa_arr;
        delete[] fit_arr;
        delete[] recency_arr;
        delete[] pa_weight_arr;
        delete[] rec_weight_arr;
        delete[] fit_weight_arr;
        delete[] alpha_arr;
        delete[] out_degree_arr;
        delete[] random_weight_arr;
        delete[] current_score_arr;
        delete graph;

        std::chrono::steady_clock::time_point t12 = std::chrono::steady_clock::now();
        auto duration01 = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
        auto duration14 = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t1);
        auto duration47 = std::chrono::duration_cast<std::chrono::milliseconds>(t7 - t4);
        auto duration07 = std::chrono::duration_cast<std::chrono::milliseconds>(t7 - t0);
        auto duration812 = std::chrono::duration_cast<std::chrono::milliseconds>(t12 - t8);
        auto duration1012 = std::chrono::duration_cast<std::chrono::milliseconds>(t12 - t10);
        auto duration1112 = std::chrono::duration_cast<std::chrono::milliseconds>(t12 - t11);

        std::cout << "\nElapsed time: 0-1 : " << duration01.count()/1000 << " seconds" << std::endl;
        std::cout << "\nElapsed time: 1-4 : " << duration14.count()/1000 << " seconds" << std::endl;
        std::cout << "\nElapsed time: 4-7 : " << duration47.count()/1000 << " seconds" << std::endl;
        std::cout << "\nElapsed time: 70-71: total : " << d71/1000 << " seconds." << std::endl;
        std::cout << "\nElapsed time: 71-72: total : " << d72/1000 << " seconds." << std::endl;
        std::cout << "\nElapsed time: 72-73: total : " << d73/1000 << " seconds." << std::endl;
        std::cout << "\nElapsed time: 73-74: total : " << d74/1000 << " seconds." << std::endl;
        std::cout << "\nElapsed time: 74-75: total : " << d75/1000 << " seconds." << std::endl;
        std::cout << "\nElapsed time: 75-76: total : " << d76/1000 << " seconds." << std::endl;
        std::cout << "\nElapsed time: 76-77: total : " << d77/1000 << " seconds." << std::endl;
        std::cout << "\nElapsed time: 77-79: total : " << d79/1000 << " seconds." << std::endl;
        std::cout << "\nElapsed time: 79-81 : total : " << d7981/1000 << " seconds." << std::endl;
        std::cout << "\nElapsed time: 79-80 : total : " << d7980/1000 << " seconds." << std::endl;
        std::cout << "\nElapsed time: 80-81 : total : " << d8081/1000 << " seconds." << std::endl;
        std::cout << "\nElapsed time: 70-81 : total : " << d7081/1000 << " seconds." << std::endl;
        std::cout << "\nElapsed time: 7-8 : " << duration78.count()/1000 << " seconds" << std::endl;
        std::cout << "\nElapsed time: 8-8.1 :  " << duration88p1.count()/1000 << " seconds" << std::endl;
        std::cout << "\nElapsed time: 8.1-9 :  " << duration8p19.count()/1000 << " seconds" << std::endl;
        std::cout << "\nElapsed time: 9-10 : " << duration910.count()/1000 << " seconds" << std::endl;
        std::cout << "\nElapsed time: 11-12 : " << duration1112.count()/1000 << " seconds" << std::endl;
        std::cout << "\nElapsed time: 10-12 : " << duration1012.count()/1000 << " seconds" << std::endl;

        std::cout << "\nElapsed time: 0-7 : " << duration07.count()/1000 << " seconds" << std::endl;
        std::cout << "\nElapsed time: 7-8 : " << duration78.count()/1000 << " seconds" << std::endl;
        std::cout << "\nElapsed time: 8-12 : " << duration812.count()/1000 << " seconds" << std::endl;
    
        return 0;
}