#include "utils.cuh"
#include "device_map.cuh"

using map_type = cuco::legacy::static_map<int, int>;
using device_view_type = typename map_type::device_view;

int getValueFromMap(std::map<int, int> myMap,  int key) {
    return myMap.at(key);
}

// Function to convert host std::map -> device cuco::legacy::static_map
void convertStdMapToDeviceStaticMap(
    std::map<int, int> h_map,
    cuco::legacy::static_map<int, int>& d_map) 
{
    printf("\nIn convertStdMapToDeviceStaticMap:: Flatten the map into host vectors (keys & values)...\n");

    // Step 1: Extract keys and values from std::map into host vectors
    std::vector<int> h_keys;
    std::vector<int> h_vals;
    h_keys.reserve(h_map.size());
    h_vals.reserve(h_map.size());

    for (auto const& kv : h_map) {
        h_keys.push_back(kv.first);
        h_vals.push_back(kv.second);
    }

    // Step 2: Copy to device safely
    thrust::device_vector<int> d_keys(h_keys.begin(), h_keys.end());
    thrust::device_vector<int> d_vals(h_vals.begin(), h_vals.end());

    // Allocate pair vector on device
    thrust::device_vector<thrust::pair<int,int>> d_pairs(h_map.size());

    // Zip and transform to build pairs
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(d_keys.begin(), d_vals.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(d_keys.end(), d_vals.end())),
        d_pairs.begin(),
        [] __device__ (thrust::tuple<int,int> t) {
            return thrust::pair<int,int>{thrust::get<0>(t), thrust::get<1>(t)};
        }
    );

    // Step 3: Insert into device static_map
    d_map.insert(d_pairs.begin(), d_pairs.end());

    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("CUDA Error after static_map insert: %s\n", cudaGetErrorString(err));
}

// Convert std::set<int> → cuco::static_set<int>
void convertStdSetToDeviceStaticSet(
    const std::set<int>& h_set,
    set_type& d_set) 
{
    // 1. Copy std::set into contiguous host vector
    std::vector<int> h_vec(h_set.begin(), h_set.end());

    // 2. Allocate device buffer
    thrust::device_vector<int> d_vec(h_vec.size());

    // 3. Copy from host → device
    thrust::copy(h_vec.begin(), h_vec.end(), d_vec.begin());

    // 4. Bulk insert into cuco::static_set
    // Use raw pointers from thrust::device_vector
    d_set.insert(thrust::raw_pointer_cast(d_vec.data()), 
                 thrust::raw_pointer_cast(d_vec.data() + d_vec.size()));
}

void prepareGraph(const std::map<int, std::set<int>>& hostMap,
                  device_graph* dGraph, int num_vertices)
{
    printf("\n[prepareGraph] num_vertices = %d\n", num_vertices);

    std::vector<int> h_offsets;
    std::vector<int> h_edges;

    h_offsets.resize(num_vertices + 1);

    int availableKeyIndexSize = hostMap.size();
    // -----------------------------------------------------
    // 2. Construct CSR offsets and edges
    // -----------------------------------------------------
    int offset = 0;
    for (int row = 0; row < num_vertices; ++row) {
        int key = row;
        auto it = hostMap.find(key);
        if (it != hostMap.end()) {
            const auto& adjSet = it->second;   // access the value

            h_offsets[row] = offset;
            for (int nbr : adjSet) {
                h_edges.push_back(nbr);
                offset++;
            }
            if (DATA_DEBUG && (row % 10000 == 1))
                printf("\nFor node/key %d : row %d, start %d end %d degree %d", key, row, h_offsets[row], offset, adjSet.size());
        } else {
            h_offsets[row] = offset;
            if (DATA_DEBUG && (row % 10000 == 1))
                printf("\nFor row %d, start %d end %d degree 0", row, h_offsets[row], offset);
        }
    }

    h_offsets[num_vertices] = offset;

    int num_edges = h_edges.size();

    printf("\n[prepareGraph] edges = %d, offsets = %d (should be V+1 = %d)\n",
           num_edges, (int)h_offsets.size(), num_vertices + 1);

    // -----------------------------------------------------
    // 3. Allocate device memory for CSR
    // -----------------------------------------------------
    device_graph hGraph;

    CUDA_CHECK(cudaMalloc(&hGraph.edges,  num_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&hGraph.offsets, (num_vertices + 1) * sizeof(int)));

    // -----------------------------------------------------
    // 4. Copy data to device
    // -----------------------------------------------------
    CUDA_CHECK(cudaMemcpy(hGraph.edges, h_edges.data(),
                          num_edges * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(hGraph.offsets, h_offsets.data(),
                          (num_vertices + 1) * sizeof(int), cudaMemcpyHostToDevice));

    hGraph.num_vertices = num_vertices;
    hGraph.num_edges    = num_edges;

    // -----------------------------------------------------
    // 5. Copy struct to device pointer
    // -----------------------------------------------------
    CUDA_CHECK(cudaMemcpy(dGraph, &hGraph,
                          sizeof(device_graph), cudaMemcpyHostToDevice));
}
/*void prepareGraph(const std::map<int, std::set<int>>& hostMap,
                    device_graph* dGraph) { 
    int num_vertices = hostMap.size();
    printf("\nIn prepareGraph:: num_vertices = %d", num_vertices);
    //std::vector<int> h_keys;
    std::vector<int> h_offsets;
    std::vector<int> h_edges;          
    int offset = 0;
    int index = 0;

    for (auto it = hostMap.begin(); it != hostMap.end(); ++it) {
        int key = it->first;
        const std::set<int>& adjSet = it->second;
        
        h_offsets.push_back(offset);
        //printf("\nh_offsets.push_back(offset) for key = %d", key);
        //printf("\nvalueSet size for key %d is %d", key, valueSet.size());
        for (int e : adjSet) {
            h_edges.push_back(e);
            offset++;
        }
    }
    h_offsets.emplace_back(offset); 
    int num_edges = h_edges.size();

    std::cout<<"\nFor hostMap size = "<< hostMap.size() << " : h_offsets size = " 
            << h_offsets.size() << " num_vertices+1 = "<< (num_vertices+1);

    std::cout<<"\nFor hostMap size = "<< hostMap.size() << " : h_edges size = " 
            << h_edges.size() << " num_edges = "<< num_edges;


    // ---------- 2. Copy arrays to device ----------
    device_graph hGraph;  // temporary host struct

    //cudaMalloc(&hGraph.keys,    numKeys   * sizeof(int));
    cudaMalloc(&hGraph.edges,  num_edges * sizeof(int));
    cudaMalloc(&hGraph.offsets, (num_vertices+1) * sizeof(int));

    //cudaMemcpy(hGraph.keys,    h_keys.data(),    numKeys   * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(hGraph.edges, h_edges.data(),  num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(hGraph.offsets, h_offsets.data(), (num_vertices+1) * sizeof(int), cudaMemcpyHostToDevice);

    hGraph.num_vertices = num_vertices;
    hGraph.num_edges = num_edges;

    // Copy struct itself to device
    cudaMemcpy(dGraph, &hGraph, sizeof(device_graph), cudaMemcpyHostToDevice); 
    //printf("\ncudaMemcpy(dGraph, &hGraph,....");
}
*/

// For bitmap-based vectors (int neighborhoods)
__global__ void kernel_extract_vector_sizes(device_vector* d_vectors,
                                            int* d_sizes,
                                            int num_threads)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_threads) return;

    // Defensive null check
    if (d_vectors[tid].d_size == nullptr) {
        printf("extract_vector_sizes: ERROR vec[%d].d_size is nullptr\n", tid);
        d_sizes[tid] = -1;
        return;
    }

    d_sizes[tid] = d_vectors[tid].size();
}

__host__ int* extract_vector_sizes(device_vector* d_vectors, int num_threads)
{
    // Allocate device array for sizes
    int* d_sizes = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sizes, num_threads * sizeof(int)));

    // Launch small kernel to safely read sizes on device
    int blockSize = 256;
    int gridSize  = (num_threads + blockSize - 1) / blockSize;

    kernel_extract_vector_sizes<<<gridSize, blockSize>>>(d_vectors,
                                                         d_sizes,
                                                         num_threads);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results to host
    int* h_sizes = new int[num_threads];
    CUDA_CHECK(cudaMemcpy(h_sizes, d_sizes,
                          num_threads * sizeof(int),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_sizes));
    return h_sizes; // caller owns this
}

void create_thread_sets(int num_threads, int capacity_per_thread, ThreadSets* out) {
    // Allocate array of actual set objects in managed memory
    cudaMallocManaged(&out->sets, num_threads * sizeof(set_type));
    
    // Allocate array of set references in managed memory
    cudaMallocManaged(&out->set_refs, num_threads * sizeof(set_ref_type));
    
    out->num_threads = num_threads;

    // Initialize each set using placement new
    for (int i = 0; i < num_threads; i++) {
        // Construct set object in-place
        new (&out->sets[i]) set_type(
            capacity_per_thread,
            cuco::empty_key<Key>{empty_key_sentinel}
        );
        
        // Create reference to the actual set object
        out->set_refs[i] = out->sets[i].ref(
            cuco::op::insert,
            cuco::op::find,
            cuco::op::erase,
            cuco::op::contains
        );
    } 
}

void destroy_thread_sets(ThreadSets* thread_sets) {
    if (thread_sets == nullptr) {
        return;
    }
    
    try {
        // Explicitly call destructor for each set
        if (thread_sets->sets != nullptr) {
            for (int i = 0; i < thread_sets->num_threads; i++) {
                // Call destructor - this will free internal cuco::static_set memory
                thread_sets->sets[i].~set_type();
            }
            
            // Free the array of set objects
            cudaError_t err = cudaFree(thread_sets->sets);
            if (err != cudaSuccess) {
                fprintf(stderr, "Warning: Failed to free sets array: %s\n", 
                        cudaGetErrorString(err));
            }
            thread_sets->sets = nullptr;
        }
        
        // Free the array of references
        // Note: set_refs don't own memory, they just reference the sets
        // But we still need to free the array itself
        if (thread_sets->set_refs != nullptr) {
            cudaError_t err = cudaFree(thread_sets->set_refs);
            if (err != cudaSuccess) {
                fprintf(stderr, "Warning: Failed to free set_refs array: %s\n", 
                        cudaGetErrorString(err));
            }
            thread_sets->set_refs = nullptr;
        }
        
        thread_sets->num_threads = 0;
        delete thread_sets;
        thread_sets = nullptr;
    } catch (const std::exception& e) {
        std::cerr << "Error in destroy_thread_sets: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown error in destroy_thread_sets " << std::endl;
    }
}

// Add to utils.cuh
void freedevice_graph(device_graph* d_graph) {
    // First, copy the struct back to get the device pointers
    device_graph h_graph;
    cudaMemcpy(&h_graph, d_graph, sizeof(device_graph), cudaMemcpyDeviceToHost);
    
    // Free the two internal arrays
    if (h_graph.edges != nullptr) {
        cudaFree(h_graph.edges);
    }
    if (h_graph.offsets != nullptr) {
        cudaFree(h_graph.offsets);
    }
    
    // Finally free the struct itself
    cudaFree(d_graph);
}

int2* convertToDeviceArray(std::vector<std::pair<int, int>> vec, cuco::legacy::static_map<int, int>& positionInArrayMap) {
	int2* device_array;
	int n = vec.size();

	int2* host_array = new int2[n];
	for (int i = 0; i < n; ++i) {
		host_array[i] = make_int2(vec[i].first, vec[i].second);
        //positionInArrayMap.insert(vec[i].first, i);
	}

	cudaMalloc(&device_array, n * sizeof(int2));
	cudaMemcpy(device_array, host_array, n * sizeof(int2), cudaMemcpyHostToDevice);

	delete[] host_array;
	return device_array;
}  


void print_host_map(const std::map<int, std::set<int>>& host_map) {
    for (const auto& [key, value_set] : host_map) {
        std::cout << "Key: " << key << " -> { ";
        for (int val : value_set) {
            std::cout << val << " ";
        }
        std::cout << "}\n";
    }
}

int computeFinalNumber(int initialSize, int num_cycles, double growth_rate) {
    int currentSize = initialSize;
    for(int i = 0; i < num_cycles; i ++) {
        int num_new_entrees = std::ceil(currentSize * growth_rate);
        currentSize += num_new_entrees;
    }
    return currentSize;
}

void printNode(const Node& n) {
    std::cout << "type: " << n.type << "\n"
              << "generatorNode: " << n.generatorNode << "\n"
              << "fitness_peak_value: " << n.fitness_peak_value << "\n"
              << "fitness_lag_duration: " << n.fitness_lag_duration << "\n"
              << "fitness_peak_duration: " << n.fitness_peak_duration << "\n"
              << "published_year: " << n.published_year << "\n"
              << "year: " << n.year << "\n"
              << "out_degree: " << n.out_degree << "\n"
              << "assigned_out_degree: " << n.assigned_out_degree << "\n"
              << "in_degree: " << n.in_degree << "\n"
              << "planted_nodes_line_number: " << n.planted_nodes_line_number << "\n"
              << "preferential_attachment_weight: " << n.preferential_attachment_weight << "\n"
              << "recency_weight: " << n.recency_weight << "\n"
              << "fitness_weight: " << n.fitness_weight << "\n"
              << "alpha: " << n.alpha << "\n";
}