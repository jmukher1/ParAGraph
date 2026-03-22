#pragma once

#ifndef DEVICE_MIN_HEAP_CUH
#define DEVICE_MIN_HEAP_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>

// Min heap structure (compact version for per-thread heaps)
template<typename T = float>
struct device_min_heap {
    struct Element {
        T value;
        uint32_t index;
    };
    
    Element* data;      // Pointer to this heap's data section
    int capacity;       // Max capacity (K for this thread)
    int size;          // Current size
    
    __device__ void init(Element* buffer, int k) {
        data = buffer;
        capacity = k;
        size = 0;
    }
    
    __device__ int parent(int i) const { return (i - 1) / 2; }
    __device__ int left_child(int i) const { return 2 * i + 1; }
    __device__ int right_child(int i) const { return 2 * i + 2; }
    
    __device__ void swap(int i, int j) {
        Element temp = data[i];
        data[i] = data[j];
        data[j] = temp;
    }
    
    __device__ void heapify_up(int idx) {
        while (idx > 0) {
            int p = parent(idx);
            if (data[p].value <= data[idx].value) break;
            swap(idx, p);
            idx = p;
        }
    }
    
    __device__ void heapify_down(int idx) {
        while (true) {
            int smallest = idx;
            int left = left_child(idx);
            int right = right_child(idx);
            
            if (left < size && data[left].value < data[smallest].value) {
                smallest = left;
            }
            if (right < size && data[right].value < data[smallest].value) {
                smallest = right;
            }
            
            if (smallest == idx) break;
            
            swap(idx, smallest);
            idx = smallest;
        }
    }
    
    __device__ T get_min() const {
        return (size > 0) ? data[0].value : T(-1e9);
    }
    
    __device__ bool insert(T value, uint32_t idx) {
        if (size < capacity) {
            data[size].value = value;
            data[size].index = idx;
            heapify_up(size);
            size++;
            return true;
        } else if (value > data[0].value) {
            data[0].value = value;
            data[0].index = idx;
            heapify_down(0);
            return true;
        }
        return false;
    }
    
    __device__ bool is_full() const {
        return size >= capacity;
    }
    
    __device__ int get_size() const {
        return size;
    }
    
    // Extract all elements sorted (destroys heap)
    __device__ void extract_sorted(T* values, uint32_t* indices) {
        int original_size = size;
        for (int i = original_size - 1; i >= 0; i--) {
            values[i] = data[0].value;
            indices[i] = data[0].index;
            
            // Move last element to root and heapify
            data[0] = data[size - 1];
            size--;
            if (size > 0) {
                heapify_down(0);
            }
        }
        size = original_size; // Restore size
    }

    __device__ void extract_sorted_indices(uint32_t* indices) {
        int original_size = size;

        for (int i = original_size - 1; i >= 0; i--) {
            indices[i] = data[0].index;

            // Move last element to root
            data[0] = data[size - 1];
            size--;

            if (size > 0) {
                heapify_down(0);
            }
        }

        size = original_size; // restore size (heap content is scrambled, same as extract_sorted)
    }

    // Sorts the heap in-place into descending order by value
    // After this call:
    //   data[0] ... data[size-1] are sorted DESCENDING
    //   heap property is destroyed
    __device__ void sort_descending_inplace() {
        int original_size = size;

        for (int i = size - 1; i > 0; --i) {
            // Move min to the end
            swap(0, i);
            size--;
            heapify_down(0);
        }

        size = original_size; // restore logical size
    }
    
    // Non-destructive sorted extraction
    __device__ void get_sorted(T* values, uint32_t* indices) const {
        // Simple selection sort on a copy (K is small, max 250)
        Element temp[250];
        for (int i = 0; i < size; i++) {
            temp[i] = data[i];
        }
        
        // Sort temp array
        for (int i = 0; i < size - 1; i++) {
            for (int j = i + 1; j < size; j++) {
                if (temp[j].value > temp[i].value) {
                    Element t = temp[i];
                    temp[i] = temp[j];
                    temp[j] = t;
                }
            }
        }
        
        // Copy sorted results
        for (int i = 0; i < size; i++) {
            values[i] = temp[i].value;
            indices[i] = temp[i].index;
        }
    }
};

#endif // DEVICE_MIN_HEAP_CUH



