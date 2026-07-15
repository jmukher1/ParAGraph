#pragma once

#ifndef DEVICE_QUEUE_CUH
#define DEVICE_QUEUE_CUH

#include <cuda_runtime.h>

#pragma once
#include <cuda_runtime.h>
#include <stdexcept>

// Simple circular buffer queue for CUDA device use
template <typename T>
class device_queue {
public:

     __host__ device_queue() : d_data(nullptr), d_head(nullptr), d_tail(nullptr), d_size(nullptr), d_capacity(nullptr) {}

    __host__ void allocate(int capacity_) {
        // allocate storage for queue elements
        cudaMalloc(&d_data, capacity_ * sizeof(T));

        // allocate metadata scalars in unified memory
        //cudaMallocManaged(&d_head, sizeof(int));
        int h_head = 0;               // host variable (initialize to 0 or whatever value you need)
        // Allocate device memory
        cudaMalloc(&d_head, sizeof(int));
        //cudaMemset(d_head, 0, sizeof(int));
        cudaMemcpy(d_head, &h_head, sizeof(int), cudaMemcpyHostToDevice);

        //cudaMallocManaged(&d_tail, sizeof(int));
        int h_tail = 0;               // host variable (initialize to 0 or whatever value you need)
        // Allocate device memory
        cudaMalloc(&d_tail, sizeof(int));
        //cudaMemset(d_tail, 0, sizeof(int));
        cudaMemcpy(d_tail, &h_tail, sizeof(int), cudaMemcpyHostToDevice);

        //cudaMallocManaged(&d_size, sizeof(int));
        int h_size = 0;               // host variable (initialize to 0 or whatever value you need)
        // Allocate device memory
        cudaMalloc(&d_size, sizeof(int));
        //cudaMemset(d_size, 0, sizeof(int));
        cudaMemcpy(d_size, &h_size, sizeof(int), cudaMemcpyHostToDevice);

        //cudaMallocManaged(&d_capacity, sizeof(int));
        int h_capacity = capacity_;               // host variable (initialize to 0 or whatever value you need)
        // Allocate device memory
        cudaMalloc(&d_capacity, sizeof(int));
        //cudaMemset(d_capacity, capacity_, sizeof(int));
        cudaMemcpy(d_capacity, &h_capacity, sizeof(int), cudaMemcpyHostToDevice);

        //cudaDeviceSynchronize();
    } 

    __host__ ~device_queue() {
         // automatically free GPU allocations 
    }

    __host__ void destroy() {
        if (d_data != nullptr) {
            cudaFree(d_data);
            d_data = nullptr;
        }
        if (d_head != nullptr) {
            cudaFree(d_head);
            d_head = nullptr;
        }
        if (d_tail != nullptr) {
            cudaFree(d_tail);
            d_tail = nullptr;
        }
        if (d_size != nullptr) {
            cudaFree(d_size);
            d_size = nullptr;
        }
        if (d_capacity != nullptr) {
            cudaFree(d_capacity);
            d_capacity = nullptr;
        }
    }


    __device__ bool is_empty() const {
        //printf("\nchecking is_empty head=%d tail=%d", *d_head, *d_tail);
        int h = atomicAdd(d_head, 0);  // atomic read
        int t = atomicAdd(d_tail, 0);
        return (h == t);
        //return (*d_head == *d_tail);
    }

    __device__ bool push(const T& value, int nodeId) {
        int s = atomicAdd(d_size, 0);
        int capacity = atomicAdd(d_capacity, 0);

        if (s >= capacity) {
            	printf("\nERR:: queue full! push() failed for nodeId = %d: size(before)=%d capacity = %d", nodeId, s, capacity);
            	return false;
        }

        int pos = atomicAdd(d_tail, 1) % capacity;
        if (pos < 0 || pos >= capacity) return false;  // safety guard

        d_data[pos] = value;

        atomicAdd(d_size, 1);

        //printf(" -> pushed to pos=%d new_size=%d\n", pos, *d_size);
        return true;
    }

    __device__ bool pop() {
        int pos = atomicAdd(d_head, 1); // claim a head position
        int capacity = atomicAdd(d_capacity, 0);
        pos = pos % capacity;         // circular wrap

        // reduce d_size
        atomicSub(d_size, 1);
        return true;
    }

    __device__ T& front() {
        if (is_empty()) {
            asm("trap;"); // fail fast
        }
        int capacity = atomicAdd(d_capacity, 0);
        return d_data[*d_head % capacity];
    }
 

public:
    T* d_data;
    int* d_head;
    int* d_tail;
    int* d_size;
    int* d_capacity;
};


#endif  // DEVICE_QUEUE_CUH
