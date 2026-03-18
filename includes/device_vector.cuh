#pragma once

#ifndef DEVICE_VECTOR_CUH
#define DEVICE_VECTOR_CUH

#include <cuda_runtime.h>
#include <vector> 
#include "device_vector_generic.cuh"

// ============================================================================
// BITMAP-BASED DEVICE VECTOR
// Stores node IDs as bits (1 bit per node ID)
// 32x memory savings compared to storing full integers
// Perfect for sequential/dense node IDs
// ============================================================================

class device_vector {
public:
    
    // ========================================================================
    // CONSTRUCTORS / DESTRUCTORS
    // ========================================================================
    
    __host__ device_vector() 
        : d_bitmap(nullptr), d_size(nullptr), d_capacity(nullptr), 
          d_bitmap_words(nullptr), max_node_id(0) {}
    
    /*__host__ device_vector(int max_node_id_) 
        : d_bitmap(nullptr), d_size(nullptr), d_capacity(nullptr), 
          d_bitmap_words(nullptr), max_node_id(0) {
        allocate(max_node_id_);
    }*/
    
    // Disable copy
    device_vector(const device_vector&) = delete;
    device_vector& operator=(const device_vector&) = delete;
    
    __host__ ~device_vector() {
        destroy();
    }

    // ========================================================================
    // DEVICE ITERATOR (Forward Iterator over set bits)
    // ========================================================================
    class device_vector_iterator {
    private:
        const device_vector* vec;
        int current_word_idx;
        uint32_t current_word;
        int current_count;  // Number of elements we've iterated so far
        int total_size;     // Total number of set bits
        
    public:
        // Constructor for begin()
        __device__ device_vector_iterator(const device_vector* v, bool is_end = false) 
            : vec(v), current_word_idx(0), current_word(0), current_count(0) {
            
            if (is_end || vec == nullptr) {
                // End iterator
                total_size = 0;
                current_count = -1;  // Mark as end
                return;
            }
            
            total_size = (vec->d_size != nullptr) ? *(vec->d_size) : 0;
            
            if (total_size == 0) {
                current_count = -1;  // Empty, set to end
                return;
            }
            
            // Find first non-empty word
            advance_to_next_word();
        }
        
        // Advance to the next set bit
        __device__ device_vector_iterator& operator++() {
            if (current_count < 0 || current_count >= total_size) {
                return *this;  // Already at end
            }
            
            // Clear the current bit (the one we're leaving)
            int bit_pos = __ffs(current_word) - 1;
            current_word &= ~(1U << bit_pos);
            
            // Increment count - we've now processed one more element
            current_count++;
            
            // Check if we've processed all elements
            if (current_count >= total_size) {
                current_count = -1;  // Mark as end
                current_word = 0;
                return *this;
            }
            
            // If current word is empty, find next non-empty word
            if (current_word == 0) {
                current_word_idx++;
                advance_to_next_word();
                
                if (current_word == 0) {
                    // No more words with bits set
                    current_count = -1;  // Mark as end
                }
            }
            
            return *this;
        }
        
        // Post-increment
        __device__ device_vector_iterator operator++(int) {
            device_vector_iterator tmp = *this;
            ++(*this);
            return tmp;
        }
        
        // Dereference - returns current node ID
        __device__ int operator*() const {
            if (current_count < 0 || current_word == 0) {
                return -1;  // Invalid
            }
            
            int bit_pos = __ffs(current_word) - 1;
            int base_node = current_word_idx << 5;
            return base_node + bit_pos;
        }
        
        // Equality comparison
        __device__ bool operator==(const device_vector_iterator& other) const {
            if (current_count < 0 && other.current_count < 0) {
                return true;  // Both at end
            }
            return (vec == other.vec && 
                    current_word_idx == other.current_word_idx &&
                    current_count == other.current_count);
        }
        
        __device__ bool operator!=(const device_vector_iterator& other) const {
            return !(*this == other);
        }
        
        // Check if iterator is valid
        __device__ bool is_valid() const {
            return current_count >= 0 && current_count < total_size;
        }
        
    private:
        __device__ void advance_to_next_word() {
            if (vec == nullptr || vec->d_bitmap == nullptr || vec->d_bitmap_words == nullptr) {
                current_word = 0;
                return;
            }
            
            int num_words = *(vec->d_bitmap_words);
            
            while (current_word_idx < num_words) {
                current_word = vec->d_bitmap[current_word_idx];
                
                if (current_word != 0) {
                    return;  // Found non-empty word
                }
                
                current_word_idx++;
            }
            
            // No more words
            current_word = 0;
        }
        
        __device__ int get_current_bit_pos() const {
            return __ffs(current_word) - 1;
        }
    };
    
    // begin() and end() helpers
    __device__ device_vector_iterator begin() const {
        return device_vector_iterator(this, false);
    }
    
    __device__ device_vector_iterator end() const {
        return device_vector_iterator(this, true);
    }
    // ========================================================================
    // ALLOCATION / DEALLOCATION
    // ========================================================================
    
    __host__ void allocate(int max_node_id_) {
        // Cleanup existing allocations
        if (d_bitmap != nullptr) {
            CUDA_CHECK(cudaFree(d_bitmap));
            d_bitmap = nullptr;
        }
        if (d_size != nullptr) {
            CUDA_CHECK(cudaFree(d_size));
            d_size = nullptr;
        }
        if (d_capacity != nullptr) {
            CUDA_CHECK(cudaFree(d_capacity));
            d_capacity = nullptr;
        }
        if (d_bitmap_words != nullptr) {
            CUDA_CHECK(cudaFree(d_bitmap_words));
            d_bitmap_words = nullptr;
        }
        
        max_node_id = max_node_id_;
        int num_words = (max_node_id + 31) / 32;
        
        // Allocate bitmap (1 bit per potential node)
        CUDA_CHECK(cudaMalloc(&d_bitmap, num_words * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_bitmap, 0, num_words * sizeof(uint32_t)));
        
        // Allocate metadata
        int h_size = 0;
        CUDA_CHECK(cudaMalloc(&d_size, sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_size, &h_size, sizeof(int), cudaMemcpyHostToDevice));
        
        int h_capacity = max_node_id;
        CUDA_CHECK(cudaMalloc(&d_capacity, sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_capacity, &h_capacity, sizeof(int), cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaMalloc(&d_bitmap_words, sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_bitmap_words, &num_words, sizeof(int), cudaMemcpyHostToDevice));
    }
    
    __host__ void destroy() {
        try {
            if (d_bitmap) {
                cudaError_t err = cudaFree(d_bitmap);
                if (err != cudaSuccess)
                    fprintf(stderr, "cudaFree d_bitmap failed for device_vector: %s\n", cudaGetErrorString(err));
                d_bitmap = nullptr;
            }
            if (d_size) {
                cudaError_t err = cudaFree(d_size);
                if (err != cudaSuccess)
                    fprintf(stderr, "cudaFree d_size failed for device_vector: %s\n", cudaGetErrorString(err));
                d_size = nullptr;
            }
            if (d_capacity) {
                cudaError_t err = cudaFree(d_capacity);
                if (err != cudaSuccess)
                    fprintf(stderr, "cudaFree d_capacity failed for device_vector: %s\n", cudaGetErrorString(err));
                d_capacity = nullptr;
            }
            if (d_bitmap_words) {
                cudaError_t err = cudaFree(d_bitmap_words);
                if (err != cudaSuccess)
                    fprintf(stderr, "cudaFree d_bitmap_words failed for device_vector: %s\n", cudaGetErrorString(err));
                d_bitmap_words = nullptr;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error in destroy: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "Unknown error in destroy" << std::endl;
        }
    }
    
    // ========================================================================
    // DEVICE FUNCTIONS - CORE OPERATIONS
    // ========================================================================
    
    // Get current size (number of set bits)
    __device__ int size() const {
        return *d_size;
    }
    
    __device__ int get_capacity() const {
        return *d_capacity;
    }
    
    // Insert a node ID (set its bit)
    // Returns true if this is a new insertion, false if already present
    __device__ bool insert(int node_id) {
        if (node_id < 0 || node_id >= *d_capacity) {
            printf("\nERROR: insert() node_id %d out of range [0, %d)\n",
                node_id, *d_capacity);
            return false;
        }

        int word_idx = node_id >> 5;
        int bit_idx  = node_id & 31;
        uint32_t mask = 1u << bit_idx;

        uint32_t old_word;
        uint32_t new_word;

        // Atomic CAS loop
        while (true) {
            old_word = d_bitmap[word_idx];

            // Bit already set?
            if (old_word & mask)
                return false;

            new_word = old_word | mask;

            // Try to write new word atomically
            uint32_t prior = atomicCAS(&d_bitmap[word_idx], old_word, new_word);

            if (prior == old_word) {
                // Success: we set the bit for the first time
                atomicAdd(d_size, 1);
                return true;
            }

            // If CAS failed, another thread changed the word → retry
        }
    }
    
    // Check if a node ID is present
    __device__ bool contains(int node_id) const {
        if (node_id < 0 || node_id >= *d_capacity) {
            return false;
        }
        
        int word_idx = node_id >> 5;
        int bit_idx = node_id & 31;
        return (d_bitmap[word_idx] >> bit_idx) & 1;
    }
    
    // Remove a node ID (clear its bit)
    __device__ bool erase(int node_id) {
        if (node_id < 0 || node_id >= *d_capacity) {
            return false;
        }
        
        int word_idx = node_id >> 5;
        int bit_idx = node_id & 31;
        uint32_t mask = ~(1U << bit_idx);
        
        // Atomic AND to clear the bit
        uint32_t old = atomicAnd(&d_bitmap[word_idx], mask);
        
        // If bit was set before, decrement size
        if ((old >> bit_idx) & 1) {
            atomicSub(d_size, 1);
            return true;
        }
        return false;
    }
    
    // Clear all bits
    __device__ void clear() {
        int num_words = *d_bitmap_words;
        for (int i = 0; i < num_words; i++) {
            d_bitmap[i] = 0;
        }
        *d_size = 0;
    }
    
    // ========================================================================
    // DEVICE FUNCTIONS - ITERATION HELPERS
    // ========================================================================
    
    // Extract all set node IDs into an output array
    // Returns the number of nodes extracted
    __device__ int extract_to_array(int* output, int max_output) const {
        int count = 0;
        int num_words = *d_bitmap_words;
        
        for (int word_idx = 0; word_idx < num_words; word_idx++) {
            uint32_t word = d_bitmap[word_idx];
            
            if (word == 0) continue;
            
            int base_node = word_idx << 5;  // word_idx * 32
            
            // Extract each set bit
            while (word != 0 && count < max_output) {
                int bit_pos = __ffs(word) - 1;  // Find first set (1-indexed, so -1)
                int node_id = base_node + bit_pos;
                
                if (node_id < *d_capacity) {
                    output[count++] = node_id;
                }
                
                word &= ~(1U << bit_pos);  // Clear this bit
            }
            
            if (count >= max_output) break;
        }
        
        return count;
    }
    
    // Callback-based iteration over all set bits
    template<typename Func>
    __device__ void for_each(Func func) const {
        int num_words = *d_bitmap_words;
        
        for (int word_idx = 0; word_idx < num_words; word_idx++) {
            uint32_t word = d_bitmap[word_idx];
            
            if (word == 0) continue;
            
            int base_node = word_idx << 5;
            
            while (word != 0) {
                int bit_pos = __ffs(word) - 1;
                int node_id = base_node + bit_pos;
                
                if (node_id < *d_capacity) {
                    func(node_id);
                }
                
                word &= ~(1U << bit_pos);
            }
        }
    }
    
    // ========================================================================
    // DEVICE FUNCTIONS - COMPATIBILITY WITH device_vector API
    // ========================================================================
    
    // Add node (alias for insert)
    __device__ bool push_back(int node_id, const char* vectorName = "bitvector") {
        bool inserted = insert(node_id);
        if (!inserted && !contains(node_id)) {
            printf("\nERROR: %s failed to insert node %d\n", vectorName, node_id);
        }
        return inserted;
    }
    
    // Get the i-th set node ID (slower - requires iteration)
    __device__ int operator[](int idx) const {
        // Capture size at start for consistent error reporting
        int expected_size = (d_size != nullptr) ? *d_size : 0;
        
        // Early bounds check
        if (idx < 0) {
            printf("\nERROR: operator[] negative index %d (size=%d)\n", idx, expected_size);
            return -1;
        }
        
        if (idx >= expected_size) {
            printf("\nERROR: operator[] index %d >= size %d\n", idx, expected_size);
            return -1;
        }
        
        int count = 0;
        int num_words = (d_bitmap_words != nullptr) ? *d_bitmap_words : 0;
        
        for (int word_idx = 0; word_idx < num_words; word_idx++) {
            uint32_t word = d_bitmap[word_idx];
            if (word == 0) continue;
            
            int base_node = word_idx << 5;  // word_idx * 32
            
            while (word != 0) {
                int bit_pos = __ffs(word) - 1;
                int node_id = base_node + bit_pos;
                
                if (node_id < *d_capacity) {
                    if (count == idx) {
                        return node_id;
                    }
                    count++;
                }
                
                word &= ~(1U << bit_pos);
            }
        }
        
        // If we get here, the bitmap doesn't have as many bits set as d_size claims
        printf("\nERROR: operator[] index %d not found after iterating %d elements "
            "(d_size claims %d elements) - SIZE INCONSISTENCY!\n", 
            idx, count, expected_size);
        return -1;
    }
    
    // ========================================================================
    // HOST FUNCTIONS
    // ========================================================================
    
    __host__ int size_host() const {
        int host_size;
        cudaMemcpy(&host_size, d_size, sizeof(int), cudaMemcpyDeviceToHost);
        return host_size;
    }
    
    __host__ int getCapacity() const {
        int capacity;
        cudaMemcpy(&capacity, d_capacity, sizeof(int), cudaMemcpyDeviceToHost);
        return capacity;
    }
    
    // Extract all set node IDs to host vector
    __host__ std::vector<int> getData() const {
        int num_words_host;
        cudaMemcpy(&num_words_host, d_bitmap_words, sizeof(int), cudaMemcpyDeviceToHost);
        
        // Copy bitmap to host
        std::vector<uint32_t> h_bitmap(num_words_host);
        CUDA_CHECK(cudaMemcpy(h_bitmap.data(), d_bitmap, 
                             num_words_host * sizeof(uint32_t), 
                             cudaMemcpyDeviceToHost));
        
        // Extract node IDs from bitmap
        std::vector<int> result;
        result.reserve(size_host());
        
        for (int word_idx = 0; word_idx < num_words_host; word_idx++) {
            uint32_t word = h_bitmap[word_idx];
            if (word == 0) continue;
            
            int base_node = word_idx * 32;
            
            while (word != 0) {
                int bit_pos = __builtin_ffs(word) - 1;  // Host intrinsic
                int node_id = base_node + bit_pos;
                
                if (node_id < max_node_id) {
                    result.push_back(node_id);
                }
                
                word &= ~(1U << bit_pos);
            }
        }
        
        return result;
    }
    
    // Print statistics
    __host__ void print_stats() const {
        int num_words_host;
        cudaMemcpy(&num_words_host, d_bitmap_words, sizeof(int), cudaMemcpyDeviceToHost);
        /**
        printf("device_vector stats:\n");
        printf("  Max node ID: %d\n", max_node_id);
        printf("  Size: %d nodes\n", size_host());
        printf("  Capacity: %d nodes\n", getCapacity());
        printf("  Bitmap words: %d (%.2f KB)\n", 
               num_words_host, num_words_host * 4.0 / 1024.0);
        printf("  Memory savings vs int array: %.1f%%\n",
               (1.0 - (num_words_host * 4.0) / (max_node_id * 4.0)) * 100.0); 
               */
    }

public:
    uint32_t* d_bitmap;      // Bitmap storage
    int* d_size;             // Number of set bits
    int* d_capacity;         // Max node ID
    int* d_bitmap_words;     // Number of uint32_t words
    int max_node_id;         // Host-side copy of max node ID
};



#endif // device_vector_CUH