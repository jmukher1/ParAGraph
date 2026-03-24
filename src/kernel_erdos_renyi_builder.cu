#include "kernel_dispatcher.cuh"

// Erdős-Rényi connection builder (simplified version of buildOneNodeConnections)
// This function handles the ER model without the complex PA logic

void buildErdosRenyiConnections(ABM* abm, Graph* graph,
                                std::vector<int>& new_nodes_vec,
                                std::vector<std::pair<int, int>>& new_edges_vec,
                                int current_graph_size,
                                int current_year) {
    
    int num_new_nodes = new_nodes_vec.size();
    
    if (num_new_nodes == 0) return;
    
    std::cout << "\n[ER Model] Processing " << num_new_nodes << " new nodes for year " 
              << current_year << " (graph size: " << current_graph_size << ")" << std::endl;
    
    // =====================================================================
    // SETUP
    // =====================================================================
    // Thread configuration
    int threadBlockSize = 256;
    dim3 threads_per_block(threadBlockSize, 1);
    
    // Create CUDA stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    std::cout << "\nALLOCATE DEVICE MEMORY";
    // =====================================================================
    // ALLOCATE DEVICE MEMORY
    // =====================================================================
    
    // Upload new nodes array
    int* d_new_nodes_arr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_new_nodes_arr, num_new_nodes * sizeof(int)));
    CUDA_CHECK(cudaMemcpyAsync(d_new_nodes_arr, new_nodes_vec.data(),
            num_new_nodes * sizeof(int), cudaMemcpyHostToDevice, stream));
    
    std::cout << "\n Allocate RNG states.";
    // Allocate RNG states
    curandState* deviceStates = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceStates, num_new_nodes * sizeof(curandState)));
    
    // Allocate edge output vectors (one per thread in batch)
    device_vector_generic<int2>* d_new_edges_vec_vectors = nullptr;
    
    std::cout << "\n Estimate capacity based on ER model.";
    // Estimate capacity based on ER model
    int estimated_edges_per_node;
    if (abm->get_er_edge_probability() > 0.0) {
        // G(n,p): expected edges = p * current_graph_size
        estimated_edges_per_node = (int)(abm->get_er_edge_probability() * current_graph_size * 2);
    } else {
        // Fixed-k: k edges per node
        estimated_edges_per_node = abm->get_er_edges_per_node();
    }
    estimated_edges_per_node = std::max(10, estimated_edges_per_node);
    
    int* per_thread_capacities = new int[num_new_nodes];
    for (int i = 0; i < num_new_nodes; i++) {
        per_thread_capacities[i] = estimated_edges_per_node;
    }
    
    std::cout << "\ncreate_thread_vectors_bulk<int2>(num_new_nodes, per_thread_capacities.";
    create_thread_vectors_bulk<int2>(num_new_nodes, per_thread_capacities, &d_new_edges_vec_vectors);
    
    delete[] per_thread_capacities;
    
    // =====================================================================
    // PROCESS BATCHES
    // =====================================================================
    
    unsigned long long seed = current_year * 12345ULL;
    std::cout << "\nInstead of batch - process all new nodes in parallel.";
    //for (int start = 0; start < num_new_nodes; start += batch_size) {
        int start = 0;
        int this_batch_size = num_new_nodes; // std::min(batch_size, num_new_nodes - start);
        int blocks_for_this_batch = (this_batch_size + threadBlockSize - 1) / threadBlockSize;
        
        // Launch ER kernel based on model type
        if (abm->get_er_edge_probability() > 0.0) {
            // G(n,p) model
            kernelErdosRenyiGNP<<<blocks_for_this_batch, threads_per_block, 0, stream>>>(
                start, this_batch_size, num_new_nodes, abm, deviceStates, seed,
                d_new_nodes_arr, d_new_edges_vec_vectors, current_graph_size,
                abm->get_er_edge_probability()
            );
        } else {
            // Fixed-k model
            kernelErdosRenyiFixedK<<<blocks_for_this_batch, threads_per_block, 0, stream>>>(
                start, this_batch_size, num_new_nodes, abm, deviceStates, seed,
                d_new_nodes_arr, d_new_edges_vec_vectors, current_graph_size,
                abm->get_er_edges_per_node()
            );
        }
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
    //}
    std::cout << "\nCOLLECT RESULTS";
    // =====================================================================
    // COLLECT RESULTS
    // =====================================================================
    
    // Transfer edges back to host and convert int2 to pair<int,int>
    //std::vector<int2> temp_edges_vec;
    append_device_to_host<int2>(d_new_edges_vec_vectors, new_edges_vec, num_new_nodes, 
                                nullptr, current_graph_size);
    
    /*/ Convert int2 to pair<int,int>
    for (const auto& edge : temp_edges_vec) {
        new_edges_vec.push_back(std::make_pair(edge.x, edge.y));
    }*/
    
    std::cout << "\n[ER Model] Generated " << new_edges_vec.size() << " edges" << std::endl;
    
    // =====================================================================
    // CLEANUP
    // =====================================================================
    
    cleanup_vectors_bulk<int2>(d_new_edges_vec_vectors, num_new_nodes);
    CUDA_CHECK(cudaFree(deviceStates));
    CUDA_CHECK(cudaFree(d_new_nodes_arr));
    CUDA_CHECK(cudaStreamDestroy(stream));
}
