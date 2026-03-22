#pragma once

#ifndef DEVICE_VECTOR_SOA_CUH
#define DEVICE_VECTOR_SOA_CUH

template<typename WeightType = float>
struct device_vector_soa {
    WeightType* weights;     // All weights in one array
    uint32_t* indices;      // All indices in one array
    int* d_size;
    int* d_capacity;
    
    __device__ int size() const {
        return *d_size;
    }
    
    __device__ bool push_back(WeightType weight, uint32_t index, const char* vectorName = "vector") {
        int pos = atomicAdd(d_size, 1);
        if (pos >= *d_capacity) {
            printf("\nERR:: %s device_vector_SOA full. Cannot insert:push_back: position = %d, size=%d capacity=%d\n", 
                   vectorName, pos, *d_size, *d_capacity);
            atomicSub(d_size, 1);
            return false;
        }
        weights[pos] = weight;
        indices[pos] = index;
        return true;
    }
    
    __device__ WeightType get_weight(int idx) const {
        return weights[idx];
    }
    
    __device__ uint32_t get_index(int idx) const {
        return indices[idx];
    }
    
    __device__ void set(int idx, WeightType weight, uint32_t index) {
        weights[idx] = weight;
        indices[idx] = index;
    }
};

#endif // device_vector_soa_CUH