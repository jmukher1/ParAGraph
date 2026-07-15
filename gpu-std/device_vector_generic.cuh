#pragma once

#ifndef DEVICE_VECTOR_GENERIC_CUH
#define DEVICE_VECTOR_GENERIC_CUH

#include <cuda_runtime.h>
#include <iostream>
#include "device_vector.cuh"

#define CUDA_CHECK(call) do {                                  \
    cudaError_t err = (call);                                  \
    if (err != cudaSuccess) {                                  \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",           \
                __FILE__, __LINE__, cudaGetErrorString(err));  \
        throw std::runtime_error(cudaGetErrorString(err));     \
    }                                                          \
} while (0)


template <typename T>
class device_vector_generic {

public:
    __host__ device_vector_generic() 
        : data(nullptr), d_size(nullptr), d_capacity(nullptr), owns_memory(false) {}

    __host__ device_vector_generic(int capacity_) 
        : data(nullptr), d_size(nullptr), d_capacity(nullptr), owns_memory(false) {
        allocate(capacity_);
    }

    // Disable copy
    device_vector_generic(const device_vector_generic&) = delete;
    device_vector_generic& operator=(const device_vector_generic&) = delete;

    // Auto-cleanup destructor
    __host__ ~device_vector_generic() {
        //destroy();
    }

    __host__ void allocate(int capacity_) {
        // Mark that we own this memory
        owns_memory = true;
        
        // Cleanup existing allocations if reallocating
        if (data != nullptr) {
            CUDA_CHECK(cudaFree(data));
            data = nullptr;
        }
        if (d_size != nullptr) {
            CUDA_CHECK(cudaFree(d_size));
            d_size = nullptr;
        }
        if (d_capacity != nullptr) {
            CUDA_CHECK(cudaFree(d_capacity));
            d_capacity = nullptr;
        }

        // Allocate new data buffer
        CUDA_CHECK(cudaMalloc(&data, capacity_ * sizeof(T)));
        
        // Allocate and initialize size counter
        int h_size = 0;
        CUDA_CHECK(cudaMalloc(&d_size, sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_size, &h_size, sizeof(int), cudaMemcpyHostToDevice));
    
        // Allocate and initialize capacity
        int h_capacity = capacity_;
        CUDA_CHECK(cudaMalloc(&d_capacity, sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_capacity, &h_capacity, sizeof(int), cudaMemcpyHostToDevice));
    } 
    
    __host__ void destroy() {
        // Only free if we own the memory
        if (!owns_memory) {
            return;
        }

        // Don't throw exceptions in destroy/destructor - just log errors
        cudaError_t err;
        
        if (data != nullptr) {
            err = cudaFree(data);
            if (err != cudaSuccess) {
                std::cerr << "Error freeing data in destroy(): " 
                          << cudaGetErrorString(err) << std::endl;
            }
            data = nullptr;
        }

        if (d_size != nullptr) {
            err = cudaFree(d_size);
            if (err != cudaSuccess) {
                std::cerr << "Error freeing d_size in destroy(): " 
                          << cudaGetErrorString(err) << std::endl;
            }
            d_size = nullptr;
        }

        if (d_capacity != nullptr) {
            err = cudaFree(d_capacity);
            if (err != cudaSuccess) {
                std::cerr << "Error freeing d_capacity in destroy(): " 
                          << cudaGetErrorString(err) << std::endl;
            }
            d_capacity = nullptr;
        }
        
        owns_memory = false;
    }

    // ----------------------
    // Device functions
    // ----------------------
    __device__ int size() const {
        return *d_size;
    }

    __device__ int get_capacity() const {
        return *d_capacity;
    }

    // ===================== Access =====================
    __device__ T& operator[](int idx) { return data[idx]; }
    __device__ const T& operator[](int idx) const { return data[idx]; }

    __device__ T& at(int i) {
        return data[i];
    }

    __device__ bool contains(const T& t) const {
        int current_size = atomicAdd(d_size, 0);  // Read size atomically
        for(int i = 0; i < current_size; i++) {
            if (t == data[i]) {
                return true;
            }
        }
        return false;
    }

    __device__ const T& at(int i) const {
        return data[i];
    }

    __device__ T& front() {
        return data[0];
    }

    __device__ const T& front() const {
        return data[0];
    }

    __device__ T& back() {
        return data[*d_size - 1];
    }

    __device__ const T& back() const {
        return data[*d_size - 1];
    }

    // Thread-safe push_back using atomicAdd
    __device__ bool push_back(const T& value, const char* vectorName = "vector") {
        int pos = atomicAdd(d_size, 1);
        
        if (pos < *d_capacity) {
            data[pos] = value;
            return true;
        } else {
            printf("\nERR:: %s device_vector_generic full. Cannot insert:push_back: position = %d, size=%d capacity=%d\n", 
                   vectorName, pos, *d_size, *d_capacity);
            // Don't roll back - it causes race conditions
            // atomicSub(d_size, 1);  // REMOVED - causes issues
            return false;
        }
    }

    // ===================== Iterators =====================
    __device__ T* begin() { return data; }
    __device__ const T* begin() const { return data; }

    __device__ T* end() { return data + size(); }
    __device__ const T* end() const { return data + size(); }

    // ----------------------
    // Host helper functions
    // ----------------------
    __host__ int size_host() const {
        if (d_size == nullptr) return 0;
        
        int host_size;
        cudaError_t err = cudaMemcpy(&host_size, d_size, sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "Error reading size: " << cudaGetErrorString(err) << std::endl;
            return 0;
        }
        return host_size;
    }

    __host__ int getCapacity() const {
        if (d_capacity == nullptr) return 0;
        
        int capacity;
        cudaError_t err = cudaMemcpy(&capacity, d_capacity, sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "Error reading capacity: " << cudaGetErrorString(err) << std::endl;
            return 0;
        }
        return capacity;
    }

    // Copies all device data (up to current size) into a host vector and returns it
    __host__ std::vector<T> getData() const {
        int host_size = size_host();
        std::vector<T> host_data(host_size);
        if (host_size > 0 && data != nullptr) {
            CUDA_CHECK(cudaMemcpy(host_data.data(), data, host_size * sizeof(T), cudaMemcpyDeviceToHost));
        }
        return host_data;
    }
     
public:
    T* data;             // pointer to device buffer
    int* d_size;         // pointer to device counter (size)
    int* d_capacity;     // pointer to device capacity
    bool owns_memory;    // flag to track if this instance owns the memory
};


#endif