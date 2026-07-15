#pragma once
#ifndef INT2_H_
#define INT2_H_

#include "abm.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "epoch_profiler.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>

#include <math.h>
#include <cmath>
#include <cuda_runtime_api.h>

#include <vector>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>

#include <cuco/static_map.cuh>
#include <cuda/functional>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/logical.h>
#include <thrust/transform.h>
#include <thrust/extrema.h> // min
#include <node.cuh>
#include "device_set.cuh"
#include "device_map.cuh"
#include <device_vector.cuh>
#include "utils.cuh"

using namespace std;

#define CUDA_OK(call)                                                    \
{                                                                        \
    cudaError_t err = call;                                              \
    if (err != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                #call, __FILE__, __LINE__, cudaGetErrorString(err));     \
        exit(EXIT_FAILURE);                                              \
    }                                                                    \
}

#define CUDA_DEBUG_CHECK() { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error after %s at %s:%d — %s\n", \
                __func__, __FILE__, __LINE__, cudaGetErrorString(err)); \
        abort(); \
    } \
}

#define CUDA_SYNC_CHECK(stream) { \
    cudaError_t err = cudaStreamSynchronize(stream); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "Stream sync error in %s at %s:%d — %s\n", \
                __func__, __FILE__, __LINE__, cudaGetErrorString(err)); \
        abort(); \
    } \
}

__host__ __device__
constexpr Node make_empty_node() {
    return Node {
                -1, //int type;
                -1, // int generatorNode;
                -1, // int fitness_peak_value;
                -1, // int fitness_lag_duration;
                -1, // int fitness_peak_duration;
                -1, // int published_year;
                -1, // int year;
                -1, // int out_degree;
                -1, // int assigned_out_degree;
                -1, // int in_degree;
                -1, // int planted_nodes_line_number;

                0.0, // double preferential_attachment_weight;
                0.0, // double recency_weight;
                0.0, // double fitness_weight;
                0.0  // double alpha;
    };
}

Node const empty_node_value_sentinel = make_empty_node();

__global__ void __launch_bounds__(256, 4) kernelCallStage1(
    int start_idx,
    int batch_size,
    int total_N,
    ABM* abm,
    Graph* graph,
    device_graph* d_forward_adj_map_Graph,
    device_graph* d_backward_adj_map_Graph,
    device_map<int, Node>::device_view d_nodeAttributeMap_view,
    ABMStageState* d_states,
    int* d_new_nodes_arr,
    int num_generator_node_citation,
    device_vector* one_hop_neighborhood_vectors,
    device_vector* two_hop_neighborhood_vectors,
    CompactBFSState* d_bfs_pool,
    int initial_graph_size);

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
__global__ void __launch_bounds__(256, 4) kernelCallStage1_warped(
    int start_idx,
    int batch_size,
    int total_N,
    ABM* abm,
    Graph* graph,
    device_graph* d_forward_adj_map_Graph,
    device_graph* d_backward_adj_map_Graph,
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
    int max_vertices);

__global__ void __launch_bounds__(256, 4) kernelCallStage2(
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
    int final_graph_size);
 
__global__ void __launch_bounds__(256, 4) kernelCallStage3(
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
        int initial_graph_size, int final_graph_size);

__global__ void __launch_bounds__(256, 4) kernelCallStage4(
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
        int per_thread_selected_set_capacity);

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
__global__ void __launch_bounds__(256, 4) ABMKernel(
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
    device_graph* d_forward_adj_map_Graph,
    device_graph* d_backward_adj_map_Graph,
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
    int per_thread_selected_set_capacity);
 

// Kernel to verify initialization
__global__ void __launch_bounds__(256, 4) verify_bfs_pool(CompactBFSState* pool, int num_threads);

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
                int max_batch_size);

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
    EpochTiming* _ep_ptr);


// ─────────────────────────────────────────────────────────────────────────────
//  Instrumented execute()  — drop-in replacement
// ─────────────────────────────────────────────────────────────────────────────
int execute(ABM* abm);

// ─────────────────────────────────────────────────────────────────────────────
// execute2() calls buildOneNodeConnections() (non-timed), which launches the
// SAME kernelCallStage2/3/4 fixed above -- so execute2()'s correctness is
// fixed transitively. No changes needed in this function's own body.
// ─────────────────────────────────────────────────────────────────────────────
int execute2(ABM* abm);

#endif /* INT2_H_ */
