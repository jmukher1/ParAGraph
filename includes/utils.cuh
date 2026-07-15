#pragma once
#ifndef UTILS_H_
#define UTILS_H_

#include <stdio.h>
#include <stdlib.h> 
#include <charconv>   // std::to_chars
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <vector>

#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/pair.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>

#include <cuco/static_map.cuh>
#include <cuco/static_set.cuh>
#include <cuco/static_set_ref.cuh>
#include <cuco/extent.cuh>
#include <cuda/std/functional>
#include <cuco/pair.cuh> 
#include <cuda/functional>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/logical.h>
#include <thrust/transform.h>
#include <string>
#include "device_graph.cuh"
#include "device_map.cuh"
#include "device_min_heap.cuh"
#include "device_min_heap_array.cuh"
#include "device_queue.cuh"
#include "device_set.cuh"
#include "device_string.cuh"
#include "device_vector.cuh"
#include "device_vector_generic.cuh"
#include "device_vector_soa.cuh"
#include "node.cuh"

#define MAX_STRING_LENGTH 256
#define SEED_TYPE 0
#define AGENT_TYPE 1

#define MEM_DEBUG false
#define DATA_DEBUG false

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// tuneable, must be power of 2
#define WARP_HASH_SLOTS 512    

using namespace std;

using Key   = int;
using Value = int;

using map_type = cuco::legacy::static_map<int, int>;
using device_view_type = typename map_type::device_view;

using Key = int;
//using Key2 = int2;
using extent_type = std::size_t;

Key constexpr empty_key_sentinel = -1;
//Key2 constexpr empty_key_sentinel2 = {-1,-1};
Value constexpr empty_value_sentinel = -1;

#ifdef DEBUG_MEMCHECK
constexpr size_t scale_down = 40;
#else
constexpr size_t scale_down = 1;
#endif

struct ABMStageState {
    int generator_node;

    int same_year_citation;
    int num_citations_inside;
    int num_citations_outside;
    int num_fully_random_cited;
    int num_fully_random_cited_reserved;

    int num_actually_cited;  // running counter across stages
};

// Pick the scope you want (block, device, or system). 
// For shared-memory per-block sets:
using Scope = cuda::thread_scope;
using KeyEqual = cuda::std::equal_to<int>;
//using KeyEqual2 = cuda::std::equal_to<int2>;
using ProbingScheme = cuco::double_hashing<1, cuco::detail::XXHash_32<int>>;
//using ProbingScheme2 = cuco::double_hashing<1, cuco::detail::XXHash_32<int2>>;
using Storage = cuco::storage<1>;
using Allocator = cuco::cuda_allocator<Key>;
//using Allocator2 = cuco::cuda_allocator<Key2>;

using set_type = cuco::static_set<
    Key,
    extent_type,
    cuda::thread_scope::thread_scope_block, //Scope::thread_scope_block,
    KeyEqual,
    ProbingScheme,
    Allocator,
    Storage
>; 

// Next, we can derive the non-owning reference type from the set type.
// This is the type we use in the kernel to wrap a raw shared memory array as a `static_set`.
using set_ref_type = typename set_type::ref_type<
    cuco::op::insert_tag,
    cuco::op::find_tag,
    cuco::op::erase_tag,
    cuco::op::contains_tag
>;

// Structure to hold both the set objects and their references
struct ThreadSets {
    set_type* sets;           // Array of actual set objects
    set_ref_type* set_refs;   // Array of set references
    int num_threads;
};

struct PairDesc {
    __host__ __device__
    bool operator()(const thrust::pair<float,int>& a,
                    const thrust::pair<float,int>& b) const {
        return a.first > b.first;
    }
};

int getValueFromMap(std::map<int, int> h_map, int key);

void convertStdMapToDeviceStaticMap(std::map<int, int> h_map, cuco::legacy::static_map<int, int>& d_map);

// ============================================================================
// APPEND DEVICE TO HOST - FOR GENERIC VECTORS
// ============================================================================

template <typename T>
void append_device_to_host(device_vector_generic<T>* d_vec_vectors,
                            std::vector<std::pair<int, int>>& output_vec,
                            int num_vectors,
                            int* out_degree_arr,
                            int graphNodeSetSize) {
    
    // Copy device vector array to host
    std::vector<device_vector_generic<T>> h_vecs(num_vectors);
    CUDA_CHECK(cudaMemcpy(h_vecs.data(), d_vec_vectors, 
                         num_vectors * sizeof(device_vector_generic<T>), 
                         cudaMemcpyDeviceToHost));
    int countEdges = 0;
    // Process each vector
    for (int i = 0; i < num_vectors; i++) {
        // Get size from device
        int vec_size = 0;
        if (h_vecs[i].d_size != nullptr) {
            CUDA_CHECK(cudaMemcpy(&vec_size, h_vecs[i].d_size, sizeof(int), 
                                 cudaMemcpyDeviceToHost));
        }
        
        if (vec_size == 0 || h_vecs[i].data == nullptr) {
            continue;  // Skip empty vectors
        }
        
        // Allocate host buffer for this vector's data
        std::vector<T> h_data(vec_size);
        
        // Copy data from device to host
        CUDA_CHECK(cudaMemcpy(h_data.data(), h_vecs[i].data, 
                             vec_size * sizeof(T), 
                             cudaMemcpyDeviceToHost));
        
        //int biggerCitations = 0;                     
        // Convert T (int2) to std::pair<int, int> and append
        for (int j = 0; j < vec_size; j++) {
            output_vec.push_back({h_data[j].x, h_data[j].y});

            /*if (h_data[j].y > graphNodeSetSize) {
                biggerCitations++;
                printf("\ni = %d, edge: %d -> %d, size = %d, out deg = %d, biggerCitations = %d", 
                    i, h_data[j].x, h_data[j].y, vec_size, out_degree_arr[i], biggerCitations);
            }*/
            countEdges++;
        }
    }

    //printf("\nTotal new edges = %d", countEdges);
}

void convertStdSetToDeviceStaticSet(
    const std::set<int>& h_set,
    set_type& d_set/*,
    cudaStream_t stream = 0*/);

// ============================================================================
// CREATE THREAD VECTORS - SPECIALIZED FOR INT (BITMAP)
// ============================================================================

// =============================================================================
// FIXED: create_thread_vectors_int / destroy_thread_vectors_int
//
// ROOT CAUSE (confirmed via nsys profiling): the original implementation
// called device_vector::allocate() once per thread -- 4 cudaMalloc +
// 3 cudaMemcpy + 1 cudaMemset PER THREAD. CUDA driver call overhead is
// dominated by call COUNT, not payload size -- a 4-byte allocation costs
// the driver roughly the same as a multi-MB one. At ~169,000 papers across
// a 10-epoch run, this function (called twice per batch, for one-hop and
// two-hop neighborhoods) accounted for a large share of ~20.3 seconds of
// wall-clock time that never showed up as GPU compute -- confirmed by
// nsys's cuda_api_sum matching the profiler's "Unaccounted" bucket to
// within 10ms.
//
// FIX: bulk-allocate ALL per-thread metadata (bitmap, size, capacity,
// bitmap_words) into four shared buffers, sliced by pointer offset --
// same pattern already used correctly elsewhere in this codebase for the
// BFS bitmap slabs in buildOneNodeConnections_timed. This drops CUDA API
// call count from O(num_threads) to O(1), regardless of thread count.
// device_vector's fields are public and every device-side method only
// ever dereferences these pointers -- none of them care whether a pointer
// is individually-owned or a slice of a shared buffer, so this needs NO
// changes to device_vector.cuh. Signature is UNCHANGED -- no call sites
// need updating.
// =============================================================================
inline void create_thread_vectors_int(int num_threads, int max_node_id, 
                                      device_vector** d_vectors) {
    CUDA_CHECK(cudaMallocManaged(d_vectors, num_threads * sizeof(device_vector)));

    if (num_threads <= 0) return;

    const int num_words = (max_node_id + 31) / 32;

    // ---- ONE bulk allocation per field, covering ALL threads ----
    uint32_t* d_bulk_bitmap       = nullptr;
    int*      d_bulk_size         = nullptr;
    int*      d_bulk_capacity     = nullptr;
    int*      d_bulk_bitmap_words = nullptr;

    CUDA_CHECK(cudaMalloc(&d_bulk_bitmap,       (size_t)num_threads * num_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_bulk_size,         (size_t)num_threads * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bulk_capacity,     (size_t)num_threads * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bulk_bitmap_words, (size_t)num_threads * sizeof(int)));

    // Bitmap starts all-zero (no nodes visited yet) -- one bulk memset.
    CUDA_CHECK(cudaMemset(d_bulk_bitmap, 0, (size_t)num_threads * num_words * sizeof(uint32_t)));

    // d_size starts at 0 for every thread -- memset is valid here since
    // the all-zero-bytes bit pattern for `int` is the value 0.
    CUDA_CHECK(cudaMemset(d_bulk_size, 0, (size_t)num_threads * sizeof(int)));

    // d_capacity and d_bitmap_words are the SAME repeated value
    // (max_node_id and num_words respectively) for every thread -- build
    // once host-side, upload once.
    {
        std::vector<int> h_capacity_fill(num_threads, max_node_id);
        std::vector<int> h_bitmap_words_fill(num_threads, num_words);
        CUDA_CHECK(cudaMemcpy(d_bulk_capacity, h_capacity_fill.data(),
                              num_threads * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bulk_bitmap_words, h_bitmap_words_fill.data(),
                              num_threads * sizeof(int), cudaMemcpyHostToDevice));
    }

    // ---- Assign each thread's device_vector its slice of the bulk
    //      buffers. *d_vectors is cudaMallocManaged (unified memory), so
    //      these public-field writes are directly visible to the host
    //      here AND to device code later. ----
    for (int i = 0; i < num_threads; i++) {
        new (&((*d_vectors)[i])) device_vector();   // placement-new, same as before
        device_vector& v = (*d_vectors)[i];
        v.d_bitmap       = d_bulk_bitmap       + (size_t)i * num_words;
        v.d_size         = d_bulk_size         + i;
        v.d_capacity     = d_bulk_capacity     + i;
        v.d_bitmap_words = d_bulk_bitmap_words + i;
        v.max_node_id    = max_node_id;
    }
}

// -----------------------------------------------------------------------
// destroy_thread_vectors_int
//
// BEFORE: called device_vector::destroy() per thread -- 4 cudaFree PER
// THREAD (unconditionally, since device_vector had no owns_memory guard).
//
// AFTER: 4 cudaFree TOTAL (one per bulk buffer) + the struct array free.
// Reads the bulk base pointers directly from thread 0 -- valid because
// *d_vectors is cudaMallocManaged, and because create_thread_vectors_int
// always gives every thread a slice of the SAME four bulk buffers.
//
// CRITICAL: this must NOT call v.destroy() per-thread -- that would
// attempt to cudaFree(base_ptr + offset) for offsets > 0, an invalid free
// (crash / heap corruption risk, not just a missed optimization).
// -----------------------------------------------------------------------
inline void destroy_thread_vectors_int(device_vector* d_vectors, int num_threads) {
    if (d_vectors == nullptr) return;
    if (num_threads > 0) {
        if (d_vectors[0].d_bitmap)       CUDA_CHECK(cudaFree(d_vectors[0].d_bitmap));
        if (d_vectors[0].d_size)         CUDA_CHECK(cudaFree(d_vectors[0].d_size));
        if (d_vectors[0].d_capacity)     CUDA_CHECK(cudaFree(d_vectors[0].d_capacity));
        if (d_vectors[0].d_bitmap_words) CUDA_CHECK(cudaFree(d_vectors[0].d_bitmap_words));
    }
    CUDA_CHECK(cudaFree(d_vectors));
}
 
 
__global__ void kernel_extract_vector_sizes(device_vector* d_vectors,
                                            int* d_sizes,
                                            int num_threads);
                                            
// For bitmap-based vectors (int neighborhoods)
__host__ int* extract_vector_sizes(device_vector* d_vectors, int num_threads);

template<typename T>
__host__ void create_thread_vectors_bulk(
    int num_threads,
    int* capacities_per_thread,
    device_vector_generic<T>** d_vectors)
{
    // FIXED: previously, `data` was already correctly bulk-allocated (one
    // cudaMalloc + one cudaMemset for the whole payload), but
    // d_size/d_capacity were still allocated PER THREAD -- 2 cudaMalloc +
    // 2 cudaMemcpy per thread, despite this function's original comment
    // claiming "Single cudaMalloc" (true for `data`, not for the
    // metadata). Confirmed via nsys profiling as a major contributor to
    // unaccounted wall-clock time. Now d_size/d_capacity are ALSO
    // bulk-allocated and sliced, dropping metadata CUDA API calls from
    // O(num_threads) to O(1). Signature UNCHANGED -- no call sites need
    // updating.

    // =========================================================
    // 1. Create host array of vector structs
    // =========================================================
    std::vector<device_vector_generic<T>> h_vectors(num_threads);

    // =========================================================
    // 2. Compute total device memory needed for `data`
    // =========================================================
    std::vector<size_t> offsets(num_threads);

    size_t total_elements = 0;
    for (int i = 0; i < num_threads; i++) {
        offsets[i] = total_elements;
        total_elements += capacities_per_thread[i];
    }

    // =========================================================
    // 3. Bulk allocate `data` for ALL vectors
    // =========================================================
    T* d_bulk_data = nullptr;

    if (total_elements > 0) {
        CUDA_CHECK(cudaMalloc(&d_bulk_data, total_elements * sizeof(T)));
        CUDA_CHECK(cudaMemset(d_bulk_data, 0, total_elements * sizeof(T)));
    }

    // =========================================================
    // 3b. NEW: bulk allocate d_size and d_capacity for ALL vectors,
    //     instead of the previous per-thread loop of individual 4-byte
    //     allocations.
    // =========================================================
    int* d_bulk_size     = nullptr;
    int* d_bulk_capacity = nullptr;
    if (num_threads > 0) {
        CUDA_CHECK(cudaMalloc(&d_bulk_size,     num_threads * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_bulk_capacity, num_threads * sizeof(int)));

        // All sizes start at 0 -- one bulk memset (all-zero-bytes == int 0).
        CUDA_CHECK(cudaMemset(d_bulk_size, 0, num_threads * sizeof(int)));

        // Capacities vary per thread -- upload the caller's array directly.
        CUDA_CHECK(cudaMemcpy(d_bulk_capacity, capacities_per_thread,
                              num_threads * sizeof(int), cudaMemcpyHostToDevice));
    }

    // Assign each vector its slice of the bulk allocations
    for (int i = 0; i < num_threads; i++) {
        if (capacities_per_thread[i] > 0)
            h_vectors[i].data = d_bulk_data + offsets[i];
        else
            h_vectors[i].data = nullptr;

        h_vectors[i].d_size     = d_bulk_size     + i;
        h_vectors[i].d_capacity = d_bulk_capacity + i;
        h_vectors[i].owns_memory = false;   // entire struct's memory is bulk-managed
    }

    // =========================================================
    // 4. Upload vector array to the GPU
    // =========================================================
    CUDA_CHECK(cudaMalloc(d_vectors,
                          num_threads * sizeof(device_vector_generic<T>)));

    CUDA_CHECK(cudaMemcpy(*d_vectors,
                          h_vectors.data(),
                          num_threads * sizeof(device_vector_generic<T>),
                          cudaMemcpyHostToDevice));
}


template <typename T>
void sort_device_vector(device_vector_generic<T>* d_vec_array, int index)
{
    device_vector_generic<T> h_vec;

    // Copy struct (not data) to host
    CUDA_CHECK(cudaMemcpy(&h_vec,
                          &d_vec_array[index],
                          sizeof(h_vec),
                          cudaMemcpyDeviceToHost));

    int size = 0;
    CUDA_CHECK(cudaMemcpy(&size, h_vec.d_size, sizeof(int), cudaMemcpyDeviceToHost));

    // Perform thrust sort on device
    thrust::sort(thrust::device, h_vec.data, h_vec.data + size, PairDesc());
}

// Allocation
template<typename W>
__host__ void create_soa_vectors_bulk(
    int num_vectors,
    int* capacities_per_thread,
    device_vector_soa<W>** d_vectors)
{
    //printf("\n[create_soa_vectors_bulk] Creating %d vectors\n", num_vectors);
    
    // Calculate total elements
    size_t total_elements = 0;
    std::vector<size_t> offsets(num_vectors);
    for (int i = 0; i < num_vectors; i++) {
        offsets[i] = total_elements;
        total_elements += capacities_per_thread[i];
    }
    
    /*printf("  Total elements: %zu\n", total_elements);
    printf("  Memory: weights=%zu MB, indices=%zu MB, total=%zu MB\n",
           (total_elements * sizeof(W)) / (1024*1024),
           (total_elements * sizeof(uint32_t)) / (1024*1024),
           (total_elements * (sizeof(W) + sizeof(uint32_t))) / (1024*1024));*/
    
    // Allocate bulk storage for weights and indices
    W* d_bulk_weights = nullptr;
    uint32_t* d_bulk_indices = nullptr;
    
    if (total_elements > 0) {
        CUDA_CHECK(cudaMalloc(&d_bulk_weights, total_elements * sizeof(W)));
        CUDA_CHECK(cudaMalloc(&d_bulk_indices, total_elements * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_bulk_weights, 0, total_elements * sizeof(W)));
        CUDA_CHECK(cudaMemset(d_bulk_indices, 0, total_elements * sizeof(uint32_t)));
    }
    
    // Create host array of vector descriptors
    std::vector<device_vector_soa<W>> h_vecs(num_vectors);
    
    for (int i = 0; i < num_vectors; i++) {
        // Allocate size and capacity
        CUDA_CHECK(cudaMalloc(&h_vecs[i].d_size, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&h_vecs[i].d_capacity, sizeof(int)));
        
        int zero = 0;
        int cap = capacities_per_thread[i];
        CUDA_CHECK(cudaMemcpy(h_vecs[i].d_size, &zero, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(h_vecs[i].d_capacity, &cap, sizeof(int), cudaMemcpyHostToDevice));
        
        // Point to bulk arrays
        h_vecs[i].weights = d_bulk_weights + offsets[i];
        h_vecs[i].indices = d_bulk_indices + offsets[i];
    }
    
    // Allocate device array
    CUDA_CHECK(cudaMalloc(d_vectors, num_vectors * sizeof(device_vector_soa<W>)));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(*d_vectors, h_vecs.data(),
                          num_vectors * sizeof(device_vector_soa<W>),
                          cudaMemcpyHostToDevice));
    
    //printf("[create_soa_vectors_bulk] Done\n");
}

template<typename W>
__host__ void create_soa_vectors_individual(
    int num_vectors,
    int* capacities_per_thread,
    device_vector_soa<W>** d_vectors)
{
    //printf("\n[create_soa_vectors_individual] Creating %d vectors individually\n", num_vectors);
    
    // Create host array of vector descriptors
    std::vector<device_vector_soa<W>> h_vecs(num_vectors);
    
    size_t total_memory = 0;
    
    for (int i = 0; i < num_vectors; i++) {
        int cap = capacities_per_thread[i];
        
        if (cap > 0) {
            // Allocate individual arrays for this vector
            CUDA_CHECK(cudaMalloc(&h_vecs[i].weights, cap * sizeof(W)));
            CUDA_CHECK(cudaMalloc(&h_vecs[i].indices, cap * sizeof(uint32_t)));
            CUDA_CHECK(cudaMemset(h_vecs[i].weights, 0, cap * sizeof(W)));
            CUDA_CHECK(cudaMemset(h_vecs[i].indices, 0, cap * sizeof(uint32_t)));
            
            total_memory += cap * (sizeof(W) + sizeof(uint32_t));
        } else {
            h_vecs[i].weights = nullptr;
            h_vecs[i].indices = nullptr;
        }
        
        // Allocate size and capacity
        CUDA_CHECK(cudaMalloc(&h_vecs[i].d_size, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&h_vecs[i].d_capacity, sizeof(int)));
        
        int zero = 0;
        CUDA_CHECK(cudaMemcpy(h_vecs[i].d_size, &zero, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(h_vecs[i].d_capacity, &cap, sizeof(int), cudaMemcpyHostToDevice));
    }
    
    //printf("  Total memory allocated: %.2f MB across %d vectors\n",
    //       total_memory / (1024.0 * 1024.0), num_vectors);
    
    // Allocate device array for vector descriptors
    CUDA_CHECK(cudaMalloc(d_vectors, num_vectors * sizeof(device_vector_soa<W>)));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(*d_vectors, h_vecs.data(),
                          num_vectors * sizeof(device_vector_soa<W>),
                          cudaMemcpyHostToDevice));
    
    // printf("[create_soa_vectors_individual] Done\n");
}

template<typename W>
__host__ void destroy_soa_vectors_individual(
    device_vector_soa<W>* d_vectors,
    int num_vectors)
{
    // ("\n[destroy_soa_vectors_individual] Cleaning up %d vectors\n", num_vectors);
    
    if (d_vectors == nullptr) {
        printf("  Warning: d_vectors is nullptr, nothing to clean up\n");
        return;
    }
    
    // Copy vector descriptors from device to host
    std::vector<device_vector_soa<W>> h_vecs(num_vectors);
    CUDA_CHECK(cudaMemcpy(h_vecs.data(), d_vectors,
                          num_vectors * sizeof(device_vector_soa<W>),
                          cudaMemcpyDeviceToHost));
    
    size_t total_freed = 0;
    
    // Free each vector's individual allocations
    for (int i = 0; i < num_vectors; i++) {
        // Free weights array
        if (h_vecs[i].weights != nullptr) {
            CUDA_CHECK(cudaFree(h_vecs[i].weights));
            h_vecs[i].weights = nullptr;
        }
        
        // Free indices array
        if (h_vecs[i].indices != nullptr) {
            CUDA_CHECK(cudaFree(h_vecs[i].indices));
            h_vecs[i].indices = nullptr;
        }
        
        // Free size counter
        if (h_vecs[i].d_size != nullptr) {
            // Optionally read final size for debugging
            int final_size = 0;
            cudaMemcpy(&final_size, h_vecs[i].d_size, sizeof(int), cudaMemcpyDeviceToHost);
            
            CUDA_CHECK(cudaFree(h_vecs[i].d_size));
            h_vecs[i].d_size = nullptr;
        }
        
        // Free capacity
        if (h_vecs[i].d_capacity != nullptr) {
            int cap = 0;
            cudaMemcpy(&cap, h_vecs[i].d_capacity, sizeof(int), cudaMemcpyDeviceToHost);
            total_freed += cap * (sizeof(W) + sizeof(uint32_t));
            
            CUDA_CHECK(cudaFree(h_vecs[i].d_capacity));
            h_vecs[i].d_capacity = nullptr;
        }
    }
    
    // Free the device array of vector descriptors
    CUDA_CHECK(cudaFree(d_vectors));
    
    //printf("  Total memory freed: %.2f MB\n", total_freed / (1024.0 * 1024.0));
    //printf("[destroy_soa_vectors_individual] Done\n");
}


template<typename T>
struct DeviceHeapArray {
    device_min_heap<T>* d_heaps = nullptr;
    typename device_min_heap<T>::Element* d_elements = nullptr;

    int num_heaps = 0;
    int capacity = 0;
};

template<typename T>
DeviceHeapArray<T> allocate_device_heaps_host_only(
    int num_heaps,
    int capacity_per_heap
) {
    using Heap    = device_min_heap<T>;
    using Element = typename Heap::Element;

    DeviceHeapArray<T> out;
    out.num_heaps = num_heaps;
    out.capacity  = capacity_per_heap;

    // ----------------------------------------------------
    // 1. Allocate device memory
    // ----------------------------------------------------
    cudaMalloc(&out.d_heaps,    num_heaps * sizeof(Heap));
    cudaMalloc(&out.d_elements,
               num_heaps * capacity_per_heap * sizeof(Element));

    // ----------------------------------------------------
    // 2. Create host-side heap descriptors
    // ----------------------------------------------------
    std::vector<Heap> h_heaps(num_heaps);

    for (int i = 0; i < num_heaps; ++i) {
        h_heaps[i].data =
            out.d_elements + i * capacity_per_heap;
        h_heaps[i].capacity = capacity_per_heap;
        h_heaps[i].size     = 0;
    }

    // ----------------------------------------------------
    // 3. Copy heap descriptors to device
    // ----------------------------------------------------
    cudaMemcpy(
        out.d_heaps,
        h_heaps.data(),
        num_heaps * sizeof(Heap),
        cudaMemcpyHostToDevice
    );

    return out;
}

template<typename T>
void destroy_device_heaps(DeviceHeapArray<T>& heaps) {
    if (heaps.d_elements) {
        cudaFree(heaps.d_elements);
        heaps.d_elements = nullptr;
    }

    if (heaps.d_heaps) {
        cudaFree(heaps.d_heaps);
        heaps.d_heaps = nullptr;
    }

    heaps.num_heaps = 0;
    heaps.capacity  = 0;
}

// Cleanup
template<typename W>
__host__ void cleanup_soa_vectors_bulk(
    device_vector_soa<W>* d_vectors,
    int num_vectors)
{
    std::vector<device_vector_soa<W>> h_vecs(num_vectors);
    CUDA_CHECK(cudaMemcpy(h_vecs.data(), d_vectors,
                          num_vectors * sizeof(device_vector_soa<W>),
                          cudaMemcpyDeviceToHost));
    
    // Free bulk arrays (only once!)
    if (num_vectors > 0) {
        if (h_vecs[0].weights) cudaFree(h_vecs[0].weights);
        if (h_vecs[0].indices) cudaFree(h_vecs[0].indices);
    }
    
    // Free metadata
    for (int i = 0; i < num_vectors; i++) {
        if (h_vecs[i].d_size) cudaFree(h_vecs[i].d_size);
        if (h_vecs[i].d_capacity) cudaFree(h_vecs[i].d_capacity);
    }
    
    cudaFree(d_vectors);
    //printf("[cleanup_soa_vectors_bulk] Freed %d vectors\n", num_vectors);
}

// Sorting helper
template<typename W>
__host__ void sort_soa_vector(device_vector_soa<W>* d_vectors, int vec_idx) {
    device_vector_soa<W> h_vec;
    CUDA_CHECK(cudaMemcpy(&h_vec, &d_vectors[vec_idx],
                          sizeof(device_vector_soa<W>),
                          cudaMemcpyDeviceToHost));
    
    int vec_size;
    CUDA_CHECK(cudaMemcpy(&vec_size, h_vec.d_size, sizeof(int), cudaMemcpyDeviceToHost));
    
    if (vec_size == 0) return;
    
    // Create device pointers for thrust
    thrust::device_ptr<W> d_weights_ptr(h_vec.weights);
    thrust::device_ptr<uint32_t> d_indices_ptr(h_vec.indices);
    
    // Sort by weights, carrying indices along
    thrust::sort_by_key(d_weights_ptr, d_weights_ptr + vec_size, d_indices_ptr, thrust::greater<W>());
}

// Host-side: Allocate and initialize array of min heaps
template<typename T = float>
__host__ void create_min_heap_array(
    int num_threads,                     // p threads
    int* h_capacities,                   // Array of K values [K1, K2, ..., Kp]
    device_min_heap_array<T>** d_heap_array)
{
    //printf("\n[create_min_heap_array] Creating %d min heaps\n", num_threads);
    
    // Calculate total storage needed and offsets
    std::vector<size_t> offsets(num_threads);
    size_t total_elements = 0;
    
    for (int i = 0; i < num_threads; i++) {
        offsets[i] = total_elements;
        total_elements += h_capacities[i];
        //("  Heap[%d]: capacity=%d, offset=%zu\n", 
        //       i, h_capacities[i], offsets[i]);
    }
    
    /*printf("  Total elements across all heaps: %zu\n", total_elements);
    printf("  Memory: %.2f MB\n", 
           (total_elements * sizeof(typename device_min_heap<T>::Element)) / (1024.0 * 1024.0));*/
    
    // Allocate bulk storage for all heap elements
    typename device_min_heap<T>::Element* d_bulk_storage;
    CUDA_CHECK(cudaMalloc(&d_bulk_storage, 
                          total_elements * sizeof(typename device_min_heap<T>::Element)));
    CUDA_CHECK(cudaMemset(d_bulk_storage, 0, 
                          total_elements * sizeof(typename device_min_heap<T>::Element)));
    
    // Allocate array of heap structures
    device_min_heap<T>* d_heaps;
    CUDA_CHECK(cudaMalloc(&d_heaps, num_threads * sizeof(device_min_heap<T>)));
    
    // Allocate capacities array on device
    int* d_capacities;
    CUDA_CHECK(cudaMalloc(&d_capacities, num_threads * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_capacities, h_capacities, 
                          num_threads * sizeof(int), cudaMemcpyHostToDevice));
    
    // Allocate offsets array on device
    size_t* d_offsets;
    CUDA_CHECK(cudaMalloc(&d_offsets, num_threads * sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(d_offsets, offsets.data(), 
                          num_threads * sizeof(size_t), cudaMemcpyHostToDevice));
    
    // Create heap array structure on host, then copy to device
    device_min_heap_array<T> h_heap_array;
    h_heap_array.heaps = d_heaps;
    h_heap_array.bulk_storage = d_bulk_storage;
    h_heap_array.num_heaps = num_threads;
    h_heap_array.capacities = d_capacities;
    h_heap_array.offsets = d_offsets;
    
    // Allocate and copy heap array structure to device
    CUDA_CHECK(cudaMalloc(d_heap_array, sizeof(device_min_heap_array<T>)));
    CUDA_CHECK(cudaMemcpy(*d_heap_array, &h_heap_array, 
                          sizeof(device_min_heap_array<T>), cudaMemcpyHostToDevice));
    
    //printf("[create_min_heap_array] Allocation complete\n");
}

// ============================================================================
// CREATE THREAD VECTORS - GENERIC TEMPLATE (GENERIC ARRAY)
// ============================================================================

template <typename T>
void create_thread_vectors(int num_threads, int capacity_per_thread, 
                          device_vector_generic<T>** d_vectors) {
    CUDA_CHECK(cudaMallocManaged(d_vectors, num_threads * sizeof(device_vector_generic<T>)));
    for (int i = 0; i < num_threads; i++) {
        new (&((*d_vectors)[i])) device_vector_generic<T>();
        (*d_vectors)[i].allocate(capacity_per_thread);
    }
}

/*/ For generic vectors (like int2)
template <typename T>
__host__ void create_thread_vectors_variable(
    int num_threads, 
    int* capacities_per_thread,  // Array of capacities, one per thread
    device_vector_generic<T>** d_vectors) {
    
    CUDA_CHECK(cudaMallocManaged(d_vectors, num_threads * sizeof(device_vector_generic<T>)));
    
    for (int i = 0; i < num_threads; i++) {
        if (i % 1000 == 1) {
            printf("\nFor vector [i=%d] size = %d", i, capacities_per_thread[i]);
        }
        new (&((*d_vectors)[i])) device_vector_generic<T>();
        (*d_vectors)[i].allocate(capacities_per_thread[i]);
    }
}*/

// ============================================================================
// STRATEGY 3: Bulk Allocation with Batching (Most Memory Efficient)
// ============================================================================
template<typename T>
__host__ void cleanup_vectors_bulk(
    device_vector_generic<T>* d_vectors,
    int num_threads)
{
    // FIXED: previously freed `data` once (correct), but looped
    // num_threads times freeing d_size/d_capacity individually -- 2
    // cudaFree PER THREAD. Now matches the bulk allocation in
    // create_thread_vectors_bulk above: 2 cudaFree TOTAL for the
    // metadata. 4 total frees regardless of num_threads.
    if (d_vectors == nullptr) {
        printf("[cleanup_vectors_bulk] d_vectors is nullptr, nothing to free\n");
        return;
    }

    // =========================================================
    // 1. Copy vector array from device to host
    // =========================================================
    std::vector<device_vector_generic<T>> h_vectors(num_threads);
    CUDA_CHECK(cudaMemcpy(h_vectors.data(),
                          d_vectors,
                          num_threads * sizeof(device_vector_generic<T>),
                          cudaMemcpyDeviceToHost));

    if (num_threads > 0) {
        // =========================================================
        // 2. Free bulk data buffer (only once!)
        // =========================================================
        if (h_vectors[0].data != nullptr) {
            CUDA_CHECK(cudaFree(h_vectors[0].data));
        }

        // =========================================================
        // 3. Free bulk d_size / d_capacity buffers (only once each!)
        //    -- was previously a per-thread loop; all threads share the
        //    same two underlying allocations now, so thread 0's pointers
        //    are sufficient to free both.
        // =========================================================
        if (h_vectors[0].d_size != nullptr) {
            CUDA_CHECK(cudaFree(h_vectors[0].d_size));
        }
        if (h_vectors[0].d_capacity != nullptr) {
            CUDA_CHECK(cudaFree(h_vectors[0].d_capacity));
        }
    }

    // =========================================================
    // 4. Free the vector array itself
    // =========================================================
    CUDA_CHECK(cudaFree(d_vectors));
}


// ============================================================================
// VERSION 3: BATCHED ALLOCATION (MOST MEMORY EFFICIENT)
// ============================================================================
// Pros: Fewest allocations (4 total vs 3N+1), best for huge arrays
// Cons: More complex cleanup, all vectors share buffers
// Best for: Very large number of threads (>100K)

 
template <typename T>
void destroy_thread_vectors(device_vector_generic<T>* d_vectors, int num_threads) {
    if (d_vectors == nullptr) return;
    for (int i = 0; i < num_threads; i++) {
        //d_vectors[i].destroy();
        d_vectors[i].~device_vector_generic<T>();
    }
    CUDA_CHECK(cudaFree(d_vectors));
}


template <typename T>
void create_thread_queues(int num_threads, int capacity_per_thread, device_queue<T>** d_queues) {
    //1. Allocate array of device_queue<T> structs on the device
    CUDA_CHECK(cudaMalloc((void**)d_queues, num_threads * sizeof(device_queue<T>)));

    // 2. Create a temporary host buffer of queue objects to initialize on host
    std::vector<device_queue<T>> h_queues(num_threads);

    // 3. Allocate and initialize each queue on the device
    for (int i = 0; i < num_threads; i++) {
        h_queues[i].allocate(capacity_per_thread);
    }

    // 4. Copy the initialized queue metadata (device pointers) to the device array
    CUDA_CHECK(cudaMemcpy(*d_queues,
                        h_queues.data(),
                        num_threads * sizeof(device_queue<T>),
                        cudaMemcpyHostToDevice));
}


template <typename T>
void destroy_thread_queues(device_queue<T>* d_queues, int num_queues) {
    cudaSetDevice(0);
    if (d_queues != nullptr) {
        try {
            // Copy back to host to access pointers
            std::vector<device_queue<T>> h_queues(num_queues);
            CUDA_CHECK(cudaMemcpy(h_queues.data(), d_queues,
                                num_queues * sizeof(device_queue<T>),
                                cudaMemcpyDeviceToHost));

            // Free device memory inside each queue
            for (int i = 0; i < num_queues; i++) {
                h_queues[i].destroy();
            }

            // Free top-level array
            CUDA_CHECK(cudaFree(d_queues));
        } catch (const std::exception& e) {
            std::cerr << "Error in destroy_thread_queues: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "Unknown error in destroy_thread_queues " << std::endl;
        }
    }
}

// Helper macro for CUDA checks (or use your CUDA_CHECK)
#define CUDA_CHECK_OR_RET(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        printf("\nERR: CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return false; \
    } \
} while(0)


void create_thread_sets(int num_threads, int capacity_per_thread, ThreadSets* out);

void destroy_thread_sets(ThreadSets* thread_sets);

void prepareGraph(const std::map<int, std::set<int>>& hostMap,
                    device_graph* dGraph,
                    int num_vertices);


int2* convertToDeviceArray(std::vector<std::pair<int, int>> vec, cuco::legacy::static_map<int, int>& positionInArrayMap);

inline std::string to_std_string(const device_string& dstr);

int computeFinalNumber(int initialSize, int num_cycles, double growth_rate);

void print_host_map(const std::map<int, std::set<int>>& host_map);

template <typename K, typename V>
inline void convertHostMapToDeviceMap(const std::map<K, V>& hostMap, device_map<K,V>* d_map, int finalSize) {
    // get current size
    int host_size = 0;
    int N = hostMap.size();

    // Allocate host arrays
    std::vector<K> h_keys;
    std::vector<V> h_values;
    //h_keys.reserve(N);
    //h_values.reserve(N);

    // create host pairs
    std::vector<cuco::pair<int,int>> h_pairs;
    //h_pairs.reserve(N);
    int i = 0;
    // Copy from std::map into flat arrays
    for (auto& kv : hostMap) {
        h_keys.emplace_back(kv.first);
        h_values.emplace_back(kv.second);
        h_pairs.emplace_back(kv.first, i++);
        //std::cout<<"\nkey : "<< kv.first << ", key position : " << (i-1);
    }

    // Copy to device
    cudaMemcpy(d_map->keys, h_keys.data(), N * sizeof(K), cudaMemcpyHostToDevice);
    cudaMemcpy(d_map->values, h_values.data(), N * sizeof(V), cudaMemcpyHostToDevice);

    // build device-side array of pairs (key -> index)
    cuco::pair<int,int>* d_pairs = nullptr;
    cudaMalloc(&d_pairs, sizeof(cuco::pair<int,int>) * N);

    // copy pairs to device
    cudaMemcpy(d_pairs, h_pairs.data(), sizeof(cuco::pair<int,int>) * N, cudaMemcpyHostToDevice);

    // insert as range (host API expects device iterators)
    d_map->d_index_map.insert(d_pairs, d_pairs + N);

    // cleanup
    cudaFree(d_pairs);

    // update size
    host_size = int(N);
    d_map->setSize(host_size);

    //printf("\nConverted host map with %d keys to device_map of size %d\n", N, d_map->size());
}

template <typename K, typename V>
inline void freeDeviceMap(device_map<K, V>* d_map) {
    if (d_map == nullptr) {
        return;
    }
    
    // Free device arrays
    if (d_map->keys != nullptr) {
        cudaFree(d_map->keys);
        d_map->keys = nullptr;
    }
    
    if (d_map->values != nullptr) {
        cudaFree(d_map->values);
        d_map->values = nullptr;
    }
    
    // Free device size pointer
    if (d_map->d_size != nullptr) {
        cudaFree(d_map->d_size);
        d_map->d_size = nullptr;
    }
    
    // Note: d_index_map is a cuco::static_map object
    // If it owns GPU memory internally, its destructor should handle cleanup
    // But if you need to explicitly clear it:
    // d_map->d_index_map.clear();  // If such method exists
    
    //("Freed device_map resources\n");
}

 
// Add to utils.cuh
void freedevice_graph(device_graph* d_graph);

void printNode(const Node& n);


#endif 