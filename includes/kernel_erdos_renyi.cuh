#ifndef KERNEL_ERDOS_RENYI_CUH
#define KERNEL_ERDOS_RENYI_CUH

#include "abm.cuh"
#include "graph.cuh"
#include "device_vector_generic.cuh"
#include "int2.cuh"
#include <curand_kernel.h>

// Erdős-Rényi G(n,p) model: Each edge exists with probability p
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
);

// Erdős-Rényi fixed-k model: Each new node connects to k random existing nodes
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
);

// Host function to call whichever ER kernel is configured (G(n,p) or fixed-k)
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
);

// Forward-declared here since the full definition (with profiling fields)
// lives in kernel_erdos_renyi.cu -- only a pointer to it crosses this
// header boundary, so an incomplete type is sufficient for the declaration.
struct ERTiming;

// Per-epoch batch orchestration: samples this year's new nodes' edges via
// launchErdosRenyiKernel, using bulk allocation for the edge-output buffer
// (see kernel_erdos_renyi.cu for the full rationale). Records timing for
// each phase (upload, capacity computation, bulk alloc, kernel, download,
// cleanup) into *_ep_ptr.
void buildOneNodeConnectionsER(
    ABM* abm, Graph* graph,
    std::vector<int>& new_nodes_vec,
    std::vector<std::pair<int,int>>& new_edges_vec,
    int current_graph_size,
    int max_batch_size,
    ERTiming* _ep_ptr);

// Top-level epoch loop for the ER model. Called instead of execute()
// (kernel.cu) when abm->get_model() != "pa".
int executeER(ABM* abm);

#endif // KERNEL_ERDOS_RENYI_CUH
