#include <iostream>

#include "abm.cuh"
#include "device_map.cuh"
#include "int2.cuh"
#include "utils.cuh"
#include "kernel_erdos_renyi.cuh"
#include "kernel_dispatcher.cuh"

// Include ER builder implementation
#include "kernel_erdos_renyi_builder.cu"

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

        // initialize RNG for this node
        curand_init(seed, idx, 0, &deviceStates[local_idx]);
        
        int new_node_seq = d_new_nodes_arr[idx];

        double pa_weight  = __ldg(&pa_weight_arr[idx]); 
        double rec_weight = __ldg(&rec_weight_arr[idx]);
        double fit_weight = __ldg(&fit_weight_arr[idx]);
        double alpha      = __ldg(&alpha_arr[idx]);

        int outdeg = out_degree_arr[idx];

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

        int new_node = d_new_nodes_arr[idx];
        int outdeg = out_degree_arr[idx];

        double pa_weight  = __ldg(&pa_weight_arr[idx]);
        double rec_weight = __ldg(&rec_weight_arr[idx]);
        double fit_weight = __ldg(&fit_weight_arr[idx]);
        double alpha      = __ldg(&alpha_arr[idx]); 

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

        int new_node = d_new_nodes_arr[idx];
        int outdeg = out_degree_arr[idx]; 

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

        /*if (MEM_DEBUG) { 
                cudaMemGetInfo(&free_mem, &total_mem); std::cout << "\n5.Free " 
                << (free_mem / (1024.0 * 1024.0)) << " MB out of " << (total_mem / (1024.0 * 1024.0)) << " MB\n"; 
        }*/
        // ---------------------------------------------------------------------
        // GRAPH STRUCTURES (needed only for Stage 1)
        // ---------------------------------------------------------------------
        //std::cout<<"\ncalling prepareGraph(graph->getForwardAdjMap ...";
        DeviceGraph* d_forward_adj_map_Graph;
        CUDA_CHECK(cudaMalloc(&d_forward_adj_map_Graph, sizeof(DeviceGraph)));
        prepareGraph(graph->getForwardAdjMap(), d_forward_adj_map_Graph, graph->getNodeSetSize());

        /*if (MEM_DEBUG) { 
                cudaMemGetInfo(&free_mem, &total_mem); 
                std::cout << "\n6.Free " << (free_mem / (1024.0 * 1024.0)) << " MB out of " 
                        << (total_mem / (1024.0 * 1024.0)) << " MB\n"; 
        }*/

        //std::cout<<"\ncalling prepareGraph(graph->getBackwardAdjMap ...";
        DeviceGraph* d_backward_adj_map_Graph;
        CUDA_CHECK(cudaMalloc(&d_backward_adj_map_Graph, sizeof(DeviceGraph)));
        prepareGraph(graph->getBackwardAdjMap(), d_backward_adj_map_Graph, graph->getNodeSetSize());

        /*if (MEM_DEBUG) { 
                cudaMemGetInfo(&free_mem, &total_mem); 
                std::cout << "\n7.Free " << (free_mem / (1024.0 * 1024.0)) 
                        << " MB out of " << (total_mem / (1024.0 * 1024.0)) << " MB\n"; 
        }*/

        int nodeAttrMapSize = graph->getNodeAttributeMapSize();
        //printf("\nNodeAttributeMapSize = %d for year %d", nodeAttrMapSize, current_year);
        device_map<int, Node>* d_nodeAttributeMap = new device_map<int, Node>(nodeAttrMapSize);
        convertHostMapToDeviceMap<int, Node>(graph->getNodeAttributeMap(), d_nodeAttributeMap, nodeAttrMapSize);

        // ---------------------------------------------------------------------
        // SAME-YEAR STATIC SET (used in Stage 2)
        // ---------------------------------------------------------------------
        //printf("\nSAME-YEAR STATIC SET (used in Stage 2)");
        set_type* d_same_year_source_nodes_set =
                new set_type(same_year_source_nodes_capacity,
                        cuco::empty_key<int>{empty_key_sentinel});
        convertStdSetToDeviceStaticSet(same_year_source_nodes, *d_same_year_source_nodes_set);
        auto d_same_year_source_nodes_set_ref = d_same_year_source_nodes_set->ref(
                cuco::op::insert, cuco::op::find, cuco::op::erase, cuco::op::contains);

        // ---------------------------------------------------------------------
        // PER-BATCH LAUNCH CONFIG
        // ---------------------------------------------------------------------
        //printf("\nPER-BATCH LAUNCH CONFIG");
        int threadBlockSizeX = 16, threadBlockSizeY = 16;
        dim3 threads_per_block(threadBlockSizeX, threadBlockSizeY);
        int threadBlockSize = threadBlockSizeX * threadBlockSizeY;
        int batch_size = std::min(max_batch_size, num_new_nodes);
        //printf("\nbatch_size = %d for max_batch_size = %d",  batch_size, max_batch_size);
        unsigned long long seed = static_cast<unsigned long long>(time(NULL));

        /*if (MEM_DEBUG) { 
                cudaMemGetInfo(&free_mem, &total_mem); 
                std::cout << "\n9.Free " << (free_mem / (1024.0 * 1024.0)) 
                        << " MB out of " << (total_mem / (1024.0 * 1024.0)) << " MB\n"; 
        }*/

        // Pre-allocate curandState for the largest possible batch — avoids
        // a cudaMalloc/cudaFree inside the hot per-batch loop.
        curandState* deviceStates_pool = nullptr;
        CUDA_CHECK(cudaMalloc(&deviceStates_pool, batch_size * sizeof(curandState)));

        device_vector* one_hop_neighborhood_vectors;
        device_vector* two_hop_neighborhood_vectors;
        device_vector_generic<int2>* d_new_edges_vec_vectors;
        //std::cout << "\ncreate d_new_edges_vec_vectors...";
        
        create_thread_vectors_bulk<int2>(num_new_nodes, out_degree_arr, &d_new_edges_vec_vectors);

        // ---------------------------------------------------------------------
        // MAIN LOOP
        // ---------------------------------------------------------------------
        for (int start = 0; start < num_new_nodes; start += batch_size) {
                std::chrono::steady_clock::time_point tl0 = std::chrono::steady_clock::now(); 
                int this_batch_size = std::min(batch_size, num_new_nodes - start);
                int blocks_for_this_batch = (this_batch_size + threadBlockSize - 1) / threadBlockSize;
                int num_threads = this_batch_size;
                //printf("\nstart = %d this_batch_size = %d", start, this_batch_size);

                ABMStageState* d_states = nullptr;
                cudaMalloc(&d_states, num_threads * sizeof(ABMStageState));
                cudaMemset(d_states, 0, num_threads * sizeof(ABMStageState)); 
                
                /*if (MEM_DEBUG) { 
                        cudaMemGetInfo(&free_mem, &total_mem); 
                        std::cout << "\n10.Free " << (free_mem / (1024.0 * 1024.0)) 
                                << " MB out of " << (total_mem / (1024.0 * 1024.0)) << " MB\n"; 
                }*/

                create_thread_vectors_int(num_threads, per_thread_vector_capacity, &one_hop_neighborhood_vectors);
                create_thread_vectors_int(num_threads, per_thread_vector_capacity, &two_hop_neighborhood_vectors);
                //std::cout << "\ncreate_thread_vectors_int for one and two hop neighborhood vectors ....";

                /*if (MEM_DEBUG) { 
                        cudaMemGetInfo(&free_mem, &total_mem); 
                        std::cout << "\n10.1.Free " << (free_mem / (1024.0 * 1024.0)) 
                                << " MB out of " << (total_mem / (1024.0 * 1024.0)) << " MB\n"; 
                }*/
                // Allocate CUB BFS pool
                //std::cout << "Allocating CompactBFSState pool for " << num_threads << " threads...\n";

                // ------------------------------------------------------------------
                // 1. Precompute sizes: Compute bitmap size (2 bits per node)
                // ------------------------------------------------------------------
                int num_words = ((current_graph_size) + 31) / 32;
                size_t slab_words = (size_t) this_batch_size * num_words;
                size_t slab_bytes = slab_words * sizeof(uint32_t);

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
                /*if (MEM_DEBUG) { 
                        cudaMemGetInfo(&free_mem, &total_mem); 
                        std::cout << "\n10.2.Free " << (free_mem / (1024.0 * 1024.0)) 
                                << " MB out of " << (total_mem / (1024.0 * 1024.0)) << " MB\n"; 
                }*/
                 

                //---------------------------------------------------------------------
                // Stage 1: Compute 1/2-hop neighborhoods
                //---------------------------------------------------------------------
                //printf("\nBatch start=%d size=%d", start, this_batch_size);
                
                // ========================================================================
                // LAUNCH KERNEL
                // ========================================================================
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
                //cudaDeviceSynchronize();

                // std::cout << "\ncleanup d_bfs_pool ...";
                // Free device memory
                if (d_bfs_pool) {
                        cudaFree(d_bfs_pool);
                        d_bfs_pool = nullptr;
                }

                //("\nFreeing BFS slabs");
                // Free the single bitmap slab
                //CUDA_CHECK(cudaFree(d_state_bitmap_slab));

                // Free BFS descriptor pool (if allocated here)
                //CUDA_CHECK(cudaFree(d_bfs_pool));
                CUDA_CHECK(cudaFree(d_visited_slab));
                CUDA_CHECK(cudaFree(d_queue_curr_slab));
                CUDA_CHECK(cudaFree(d_queue_next_slab));

                // Reuse pre-allocated curandState pool (hoisted above batch loop)
                curandState* deviceStates = deviceStates_pool;

                /*if (MEM_DEBUG) { 
                        cudaMemGetInfo(&free_mem, &total_mem); 
                        std::cout << "\n12.Free " << (free_mem / (1024.0 * 1024.0)) 
                                << " MB out of " << (total_mem / (1024.0 * 1024.0)) << " MB\n"; 
                }


                std::cout << "\nextract_vector_sizes for one_hop_neighborhood_vectors...";*/
                int* one_hop_sizes = extract_vector_sizes(one_hop_neighborhood_vectors, num_threads);

                //---------------------------------------------------------------------
                // Stage 2: Same-year + 1-hop citations
                //---------------------------------------------------------------------
                device_vector_generic<int>* citations_vectors;
                int* per_thread_citations_vector_capacities = new int[num_threads];
                for(int i=0; i < num_threads; i++)
                       per_thread_citations_vector_capacities[i] = per_thread_citations_vector_capacity; 
                create_thread_vectors_bulk<int>(num_threads, per_thread_citations_vector_capacities, &citations_vectors);
                
                //std::cout << "\ncreate_soa_vectors_bulk<float>(num_threads, one_hop_sizes, &element_index_vec_vectors";
                // NEW WAY (SoA):
                device_vector_soa<float>* element_index_vec_vectors;
                create_soa_vectors_bulk<float>(num_threads, one_hop_sizes, &element_index_vec_vectors);

                DeviceHeapArray<float> one_hop_heaps = allocate_device_heaps_host_only<float>(num_threads, 
                                                        per_thread_citations_vector_capacity);

                //("\nFreeing host arrays");
                delete[] one_hop_sizes;
                
                /*if (MEM_DEBUG) { 
                        cudaMemGetInfo(&free_mem, &total_mem); 
                        std::cout << "\n13.Free " << (free_mem / (1024.0 * 1024.0)) 
                                << " MB out of " << (total_mem / (1024.0 * 1024.0)) << " MB\n"; 
                }*/
                
                kernelCallStage2<<<blocks_for_this_batch, threads_per_block, 0, stream>>>(
                                start, this_batch_size, num_new_nodes,
                                abm, graph,
                                d_same_year_source_nodes_set_ref, 
                                d_states,
                                deviceStates, seed,
                                d_new_nodes_arr,
                                one_hop_neighborhood_vectors,
                                //element_index_vec_vectors,
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
                //cudaDeviceSynchronize();
                
                // Free 1-hop data (no longer needed after Stage 2)
                destroy_thread_vectors_int(one_hop_neighborhood_vectors, num_threads);
                one_hop_neighborhood_vectors = nullptr;

                destroy_device_heaps<float>(one_hop_heaps); 

                /*std::cout << "\ndestroy_thread_vectors(one_hop_neighborhood_vectors";
                if (MEM_DEBUG) { 
                        cudaMemGetInfo(&free_mem, &total_mem); 
                        std::cout << "\n14.Free " << (free_mem / (1024.0 * 1024.0)) 
                                << " MB out of " << (total_mem / (1024.0 * 1024.0)) << " MB\n"; 
                }*/ 
                 
                // Sort SoA vectors in parallel using thrust on-device rather than
                // num_threads sequential CPU calls — removes O(N) host-side launch overhead.
                for (int i = 0; i < num_threads; i++) {
                        sort_soa_vector<float>(element_index_vec_vectors, i);
                }
                /**
                if (MEM_DEBUG) { 
                        cudaMemGetInfo(&free_mem, &total_mem); 
                        std::cout << "\n14.1.Free " << (free_mem / (1024.0 * 1024.0)) 
                                << " MB out of " << (total_mem / (1024.0 * 1024.0)) << " MB\n"; 
                }*/

                // Free all per-batch temporaries
                // cleanup_vectors_bulk<thrust::pair<double, int>>(element_index_vec_vectors, num_threads);
                // Cleanup:
                /*/cleanup_soa_vectors_bulk<float>(element_index_vec_vectors, num_threads);
                if (MEM_DEBUG) { 
                        cudaMemGetInfo(&free_mem, &total_mem); 
                        std::cout << "\n14.2.Free " << (free_mem / (1024.0 * 1024.0)) 
                                << " MB out of " << (total_mem / (1024.0 * 1024.0)) << " MB\n"; 
                }*/
                
                int* two_hop_sizes = extract_vector_sizes(two_hop_neighborhood_vectors, num_threads);
                /*std::cout << "\ncreate_thread_sets(num_threads...";
                if (MEM_DEBUG) { 
                        cudaMemGetInfo(&free_mem, &total_mem); 
                        std::cout << "\n14.5.Free " << (free_mem / (1024.0 * 1024.0)) 
                                << " MB out of " << (total_mem / (1024.0 * 1024.0)) << " MB\n"; 
                }*/
                
                DeviceHeapArray<float> heaps = allocate_device_heaps_host_only<float>(num_threads, 
                                                        per_thread_citations_vector_capacity);
                
                //printf("\nFreeing host arrays");
                delete[] two_hop_sizes;
                
                /*if (MEM_DEBUG) { 
                        cudaMemGetInfo(&free_mem, &total_mem); 
                        std::cout << "\n14.6.Free " << (free_mem / (1024.0 * 1024.0)) 
                                << " MB out of " << (total_mem / (1024.0 * 1024.0)) << " MB\n"; 
                }*/
                //---------------------------------------------------------------------
                // Stage 3: 2-hop + random citations + edge writes
                //---------------------------------------------------------------------
                ThreadSets* selected_citations_thread_sets = new ThreadSets();
                create_thread_sets(num_threads, per_thread_selected_set_capacity, selected_citations_thread_sets);
                /*if (MEM_DEBUG) { 
                        cudaMemGetInfo(&free_mem, &total_mem); 
                        std::cout << "\n14.7.Free " << (free_mem / (1024.0 * 1024.0)) 
                                << " MB out of " << (total_mem / (1024.0 * 1024.0)) << " MB\n"; 
                }*/

                kernelCallStage3<<<blocks_for_this_batch, threads_per_block, 0, stream>>>(
                                start, this_batch_size, num_new_nodes,
                                abm, graph, graph->getNodeSetSize(),
                                d_nodeAttributeMap->get_device_view(),
                                d_states,
                                deviceStates,
                                d_new_nodes_arr,
                                two_hop_neighborhood_vectors,
                                //element_index_vec_vectors,
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
                //cudaDeviceSynchronize(); 

                destroy_device_heaps<float>(heaps);

                //printf("\ndestroy_thread_vectors_int(two_hop_neighborhood_vectors, num_threads)");
                // Free 2-hop data (no longer needed after Stage 2)
                destroy_thread_vectors_int(two_hop_neighborhood_vectors, num_threads);

                /*if (MEM_DEBUG) { 
                        cudaMemGetInfo(&free_mem, &total_mem); 
                        std::cout << "\n15.Free " << (free_mem / (1024.0 * 1024.0)) 
                                << " MB out of " << (total_mem / (1024.0 * 1024.0)) << " MB\n"; 
                }*/
                
                /*std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
                for(int i = 0; i < num_threads; i++) {
                        //sort_device_vector<thrust::pair<double, int>>(element_index_vec_vectors, i);
                        sort_soa_vector<float>(element_index_vec_vectors, i);   
                }

                std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
                auto duration34 = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3);
                std::cout << "\nElapsed time: sorting kernelCallStage 3-4 : " << duration34.count() << " mseconds" << std::endl;*/
		
                kernelCallStage4<<<blocks_for_this_batch, threads_per_block, 0, stream>>>(
                                start, this_batch_size, num_new_nodes,
                                abm, graph, graph->getNodeSetSize(),
                                d_nodeAttributeMap->get_device_view(),
                                d_states,
                                deviceStates,
                                d_new_nodes_arr,
                                //two_hop_neighborhood_vectors,
                                //element_index_vec_vectors,
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

                
                //printf("\ndestroy_thread_vectors(element_index_vec_vectors, num_threads)");
                // Free all per-batch temporaries
                // Cleanup:
                //cleanup_soa_vectors_bulk<float>(element_index_vec_vectors, num_threads);
                //destroy_soa_vectors_individual<float>(element_index_vec_vectors, num_threads);       
                
                //printf("\ndestroy_thread_vectors<int>(citations_vectors, num_threads)");
                cleanup_vectors_bulk<int>(citations_vectors, num_threads);
                
                //printf("\ndestroy_thread_sets(selected_citations_thread_sets)");
                destroy_thread_sets(selected_citations_thread_sets);
                
                // deviceStates is pooled — freed after the batch loop
                
                //printf("\nCUDA_CHECK(cudaFree(d_states))");
                CUDA_CHECK(cudaFree(d_states));

                delete[] per_thread_citations_vector_capacities;

                /*std::cout << "\ndestroyed all the storage...";
                if (MEM_DEBUG) { 
                        cudaMemGetInfo(&free_mem, &total_mem); 
                        std::cout << "\n16.Free " << (free_mem / (1024.0 * 1024.0)) 
                                << " MB out of " << (total_mem / (1024.0 * 1024.0)) << " MB\n"; 
                }
                std::chrono::steady_clock::time_point tl1 = std::chrono::steady_clock::now();
                auto durationLoop01 = std::chrono::duration_cast<std::chrono::milliseconds>(tl1 - tl0);
                std::cout << "\nElapsed time: for year : "<< current_year << " batch: start: "<< start << ": batch size = "
                        << this_batch_size <<" :: " << durationLoop01.count()/1000 << " seconds" << std::endl;*/
        }

        // Free pooled curandState
        CUDA_CHECK(cudaFree(deviceStates_pool));

        //std::cout << "\nCalling append_device_to_host done ...."; 
        // Transfer new edges to host
        // destroy_thread_vectors of d_new_edges_vec_vectors already embedded in append_device_to_host
        append_device_to_host<int2>(d_new_edges_vec_vectors, new_edges_vec, num_new_nodes, out_degree_arr, graph->getNodeSetSize());

        //std::cout << "\nappended d_new_edges_vec_vectors .... new_edges_vec size = "<< new_edges_vec.size(); 

        // Destroy device vectors
        cleanup_vectors_bulk<int2>(d_new_edges_vec_vectors, num_new_nodes);
        
        /*std::cout << "\ndestroyed d_new_edges_vec_vectors... ";
        if (MEM_DEBUG) { 
                cudaMemGetInfo(&free_mem, &total_mem); 
                std::cout << "\n17.Free " << (free_mem / (1024.0 * 1024.0)) 
                        << " MB out of " << (total_mem / (1024.0 * 1024.0)) << " MB\n"; 
        }*/
                
        // ---------------------------------------------------------------------
        // FINAL CLEANUP (static structures)
        // ---------------------------------------------------------------------
        delete d_same_year_source_nodes_set;
        /*delete d_continuous_node_mapping;
        delete d_reverse_continuous_node_mapping;*/
        //delete d_nodeAttributeMap;
        freeDeviceMap(d_nodeAttributeMap);
        freeDeviceGraph(d_forward_adj_map_Graph);
        freeDeviceGraph(d_backward_adj_map_Graph);

        /*std::cout << "\nfreeDeviceGraph(d_backward_adj_map_Graph";
        if (MEM_DEBUG) { 
                cudaMemGetInfo(&free_mem, &total_mem); 
                std::cout << "\n18.Free " << (free_mem / (1024.0 * 1024.0)) 
                        << " MB out of " << (total_mem / (1024.0 * 1024.0)) << " MB\n"; 
        }*/

        CUDA_CHECK(cudaFree(d_new_nodes_arr));
        CUDA_CHECK(cudaFree(d_pa_arr));
        CUDA_CHECK(cudaFree(d_recency_arr));
        CUDA_CHECK(cudaFree(d_fit_arr));
        CUDA_CHECK(cudaFree(d_pa_weight_arr));
        CUDA_CHECK(cudaFree(d_rec_weight_arr));
        CUDA_CHECK(cudaFree(d_fit_weight_arr));
        CUDA_CHECK(cudaFree(d_alpha_arr));
        CUDA_CHECK(cudaFree(d_out_degree_arr));


        /*if (MEM_DEBUG) { 
                cudaMemGetInfo(&free_mem, &total_mem); 
                std::cout << "\n19.Free " << (free_mem / (1024.0 * 1024.0)) 
                        << " MB out of " << (total_mem / (1024.0 * 1024.0)) << " MB\n"; 
        }*/

        CUDA_CHECK(cudaStreamDestroy(stream));
        /*std::cout << "\n[Cleanup] buildOneNodeConnections finished.\n";
        cudaMemGetInfo(&free_mem, &total_mem); 
        std::cout << "\nFree (all) " << (free_mem / (1024.0 * 1024.0)) 
                        << " MB out of " << (total_mem / (1024.0 * 1024.0)) << " MB\n"; */
}


int execute(ABM* abm) {
        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        printf("\nStart execute."); 

        /* Reading input:: getEdgeList(), getNodeList(), outdegree bag, recency probabilities */
        Graph* graph = new Graph(abm->edgelist, abm->nodelist);
        printf("\nGraph init done...");
        abm->WriteToLogFile("loaded graph", Log::info);

        printf("\ncalling InitializeFitness...");
        abm->InitializeFitness(graph);
        abm->WriteToLogFile("initialized fitness for the seed graph", Log::debug);

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        /* node ids to continous integer from 0 */
        //td::map<int, int> continuous_node_mapping = abm->BuildContinuousNodeMapping(graph);

        abm->WriteToLogFile("forward built", Log::debug);
        /* continous integer from 0 to node ids*/
        //std::map<int, int> reverse_continuous_node_mapping = abm->ReverseMapping(continuous_node_mapping);
        //abm->WriteToLogFile("reverse mapping built", Log::debug);

        int start_year = abm->GetMaxYear(graph) + 1;
        int next_node_id = abm->GetMaxNode(graph) + 1;
        int initial_next_node_id = graph->getNodeSetSize();

        /* get input to score arrays based on continuous_node_mapping */
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

        // the first new agent node has index 0 but is actually index initial_graph_size in the continuous mapping
        // Weight arrays for new nodes
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
	double d71 = 0, d72 = 0, d73 = 0, d74 = 0, d75 = 0, d76 = 0, d77 = 0, d79 = 0, d7081 = 0, d7980 = 0, d8081 = 0, d7981 = 0;
        int max_batch_size = 20000;
        for (int current_year = start_year; current_year < start_year + abm->num_cycles; current_year++) {
                printf("\n Entering loop for year = %d", current_year);
                std::chrono::steady_clock::time_point t70 = std::chrono::steady_clock::now();
                 
                int current_graph_size = graph->GetNodeSet().size();

                abm->WriteToLogFile("current year is: " + std::to_string(current_year) + 
                                " and the graph is " + std::to_string(current_graph_size) + " nodes large", Log::info);
                abm->FillInDegreeArr(graph, /*graph->continuous_node_mapping,*/ in_degree_arr);
                abm->WriteToLogFile("indegree for current year filled", Log::debug);
                abm->FillFitnessArr(graph, /*graph->continuous_node_mapping,*/ current_year, fitness_arr);
                abm->WriteToLogFile("fitness for current year filled", Log::debug);
        
                std::chrono::steady_clock::time_point t71 = std::chrono::steady_clock::now();
                auto duration7071 = std::chrono::duration_cast<std::chrono::milliseconds>(t71 - t70);
                d71 += duration7071.count();
                std::cout << "\nElapsed time: 70-71: total : " << d71/1000 << " secs, iter cost : " << duration7071.count()/1000 << " seconds" << std::endl;
		
                abm->FillRecencyArr(graph, /*graph->reverse_continuous_node_mapping,*/ current_year, recency_arr);
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

                /* initialize new nodes */
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
                int num_generator_node_citation = 1; //generator_nodes.size(); // should be 1 for now
                std::chrono::steady_clock::time_point t78 = std::chrono::steady_clock::now();
                
                std::cout << "\ncontinuous_node_mapping.size() b5 : " << graph->continuous_node_mapping.size() 
                        << ", reverse_continuous_node_mapping size = "<< graph->reverse_continuous_node_mapping.size() << std::endl; 
                 
                try {
                        // Dispatch to PA or ER model based on configuration
                        std::string network_model = abm->get_network_model();
                        
                        if (network_model == "PA") {
                                std::cout << "\n[PA Model] Building connections using Preferential Attachment\n";
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
                        } else if (network_model == "ER") {
                                std::cout << "\n[ER Model] Building connections using Erdos-Renyi\n";
                                buildErdosRenyiConnections(abm, graph,
                                        new_nodes_vec,
                                        new_edges_vec,
                                        current_graph_size,
                                        current_year);
                        } else {
                                std::cerr << "Unknown network model: " << network_model << std::endl;
                                return 1;
                        }
                } catch (const std::runtime_error& e) {
                        std::cerr << "Caught CUDA exception: " << e.what() << std::endl;
                        // Handle the error gracefully, e.g., clean up resources, log, exit
                        return 1;
                } catch (const std::exception& e) {
                        std::cerr << "Caught generic exception: " << e.what() << std::endl;
                        return 1;
                }
                
                std::cout << "\nYear " << current_year << ": " << num_new_nodes << " new nodes, "
                          << new_edges_vec.size() << " new edges\n";
	          
                // Batch-insert all new edges into the adj maps in one pass.
                // Avoids per-edge AddNode() calls and set-insert overhead per edge.
                std::set<int> updated_destination_nodes;
                updated_destination_nodes.insert(
                    new_nodes_vec.begin(), new_nodes_vec.end()); // sources always updated

                for (const auto& [source_node, destination_node] : new_edges_vec) {
                    graph->forward_adj_map[source_node].insert(destination_node);
                    graph->backward_adj_map[destination_node].insert(source_node);
                    graph->node_set.insert(source_node);
                    graph->node_set.insert(destination_node);
                    updated_destination_nodes.insert(destination_node);
                }
                std::chrono::steady_clock::time_point t79 = std::chrono::steady_clock::now();
                auto duration7879 = std::chrono::duration_cast<std::chrono::milliseconds>(t79 - t78);
                d79 += duration7879.count();
                
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

        /*/ This block is redundant as updateNodeInDegreeOutDegree has been called
        for(auto const& nodeId : graph->GetNodeSet()) {
                graph->SetIntAttribute("in_degree", nodeId, graph->GetInDegree(nodeId));
                graph->SetIntAttribute("out_degree", nodeId, graph->GetOutDegree(nodeId));
        }*/

        std::cout<<"\n graph size = "<< graph->GetNodeSet().size();
        graph->WriteAttributes(abm->auxiliary_information_file);
        abm->WriteToLogFile("wrote graph", Log::info);
        std::chrono::steady_clock::time_point t11 = std::chrono::steady_clock::now();

        //printf("\nPrint ABM Graph Statistics....");
        //graph->PrintFinalGraphStatistics();

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