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

#endif // KERNEL_ERDOS_RENYI_CUH
