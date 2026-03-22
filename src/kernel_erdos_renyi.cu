#include "kernel_erdos_renyi.cuh"
#include "utils.cuh"

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
