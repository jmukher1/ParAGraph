#pragma once

#ifndef DEVICE_MAP_CUH
#define DEVICE_MAP_CUH

#include <cuco/static_map.cuh>
#include <cuda/std/utility>
#include <cuda_runtime.h>
#include <cstdio>

// Lightweight device_map without device_view
template <typename K, typename V>
class device_map {
public:
    using index_type = int;

    using map_type = cuco::legacy::static_map<
        K, index_type,
        cuda::thread_scope_device,
        cuco::cuda_allocator<char>>;
    using view_type = typename map_type::device_view;

    // Add this to your device_map.cuh:
    struct device_view {
        K* keys;
        V* values;
        int* d_size;
        typename cuco::legacy::static_map<K, int, cuda::thread_scope_device, 
                                        cuco::cuda_allocator<char>>::device_view index_view;

        __device__ bool contains(const K& key) const {
            return index_view.contains(key);
        }

        __device__ V at(const K& key) const {
            auto it = index_view.find(key);
            int index = it->second.load(cuda::memory_order_relaxed);
            return values[index];
        }

        __device__ int size() const {
            return *d_size;
        }
    };

    // Add this method to device_map class:
    // create device_view
    device_view get_device_view() const {
        return device_view(keys, values, d_size, d_index_map.get_device_view());
    }

public:
    // ============================
    // Constructors / Destructor
    // ============================
    __host__ device_map(int cap)
        : capacity(cap),
          d_index_map(cap,
                      cuco::empty_key<K>{-1},
                      cuco::empty_value<int>{-1}),
        d_index_map_view(d_index_map.get_device_view()),
        keys(nullptr), values(nullptr), d_size(nullptr)
    {
        cudaMalloc(&keys,   cap * sizeof(K));
        cudaMalloc(&values, cap * sizeof(V));

        int h_size = 0;
        cudaMalloc(&d_size, sizeof(int));
        cudaMemcpy(d_size, &h_size, sizeof(int), cudaMemcpyHostToDevice);
    }

    __host__ ~device_map() {
        cleanup();
    }

    // Disable accidental copying
    device_map(const device_map&) = delete;
    device_map& operator=(const device_map&) = delete;

    // Allow moves - properly transfer ownership
    device_map(device_map&& other) noexcept
        : capacity(other.capacity),
          d_index_map(std::move(other.d_index_map)),
          d_index_map_view(other.d_index_map_view),
          keys(other.keys),
          values(other.values),
          d_size(other.d_size)
    {
        // Nullify source to prevent double-free
        other.keys = nullptr;
        other.values = nullptr;
        other.d_size = nullptr;
        other.capacity = 0;
    }

    device_map& operator=(device_map&& other) noexcept {
        if (this != &other) {
            // Clean up existing resources
            cleanup();
            
            // Transfer ownership
            capacity = other.capacity;
            d_index_map = std::move(other.d_index_map);
            d_index_map_view = other.d_index_map_view;
            keys = other.keys;
            values = other.values;
            d_size = other.d_size;
            
            // Nullify source
            other.keys = nullptr;
            other.values = nullptr;
            other.d_size = nullptr;
            other.capacity = 0;
        }
        return *this;
    }

    // ============================
    // Host-side operations
    // ============================

    __host__ void insert(const K& key, const V& value) {
        int h_size = 0;
        cudaMemcpy(&h_size, d_size, sizeof(int), cudaMemcpyDeviceToHost);

        if (h_size >= capacity) {
            printf("\nERR:: device_map capacity exceeded. Cannot insert key=%d (size=%d, cap=%zu)\n",
                   (int)key, h_size, capacity);
            return;
        }

        // copy key/value to GPU arrays
        cudaMemcpy(&keys[h_size], &key, sizeof(K), cudaMemcpyHostToDevice);
        cudaMemcpy(&values[h_size], &value, sizeof(V), cudaMemcpyHostToDevice);

        cuco::pair<K, int> kv{key, h_size};
        d_index_map.insert(&kv, &kv + 1);

        // increment size
        h_size++;
        cudaMemcpy(d_size, &h_size, sizeof(int), cudaMemcpyHostToDevice);
    }

    __host__ int size() const {
        int h_size = 0;
        cudaMemcpy(&h_size, d_size, sizeof(int), cudaMemcpyDeviceToHost);
        return h_size;
    }

    __host__ void setSize(int new_size) {
        cudaMemcpy(d_size, &new_size, sizeof(int), cudaMemcpyHostToDevice);
    }

    __host__ int map_capacity() const { return capacity; }

    // ============================
    // Device-side operations
    // ============================

    __device__ bool contains(const K& key) const {
        return d_index_map_view.contains(key);
    }

    __device__ V at(const K& key) const {
        auto it = d_index_map_view.find(key);
        int index = it->second.load(cuda::memory_order_relaxed);
        return values[index];
    }

    __device__ int size_device() const {
        return *d_size;
    }

private:

    void freeKeys(K* keys) {
        if (keys) {
            cudaFree(keys);
            keys = nullptr;
        }
    }

    void freeValues(V* values) {
        if (values) {
            cudaFree(values);
            values = nullptr;
        }
    }

    void freeDSize(int* d_size) {
        if (d_size) {
            cudaFree(d_size);
            d_size = nullptr;
        }
    }

    __host__ void cleanup() {
        freeKeys(keys);
        freeValues(values);
        freeDSize(d_size);
    }

public:
    // ============================
    // Members
    // ============================
    K* keys;
    V* values;
    int* d_size;
    int capacity;

    map_type d_index_map;
    view_type d_index_map_view;
};

#endif  // DEVICE_MAP_CUH