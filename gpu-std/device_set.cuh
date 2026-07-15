#pragma once

#include <cuco/static_map.cuh>
#include <cuda/std/utility> // cuda::std::pair

#define SET_CAPACITY 1024

template <typename K>
class device_set {
public:
    using index_type = int;
    using map_type   = cuco::legacy::static_map<
        K, index_type,
        cuda::thread_scope_device,
        cuco::cuda_allocator<char>>;

    // ================================
    // Device view (read-only)
    // ================================
    class device_view {
    public:
        __device__ __forceinline__
        int size() const { return *d_size; }

        __device__ __forceinline__
        bool empty() const { return *d_size == 0; }

        __device__ __forceinline__
        bool contains(const K& key) const {
            auto it = index_map_view.find(key);
            return it != index_map_view.end();
        }

        __device__ __forceinline__
        const K* find(const K& key) const {
            auto it = index_map_view.find(key);
            if (it == index_map_view.end()) return nullptr;
            return &keys[it->second];
        }

        __device__ __forceinline__
        const K at(const int idx) {
            return keys[idx];
        }

    private:
        friend class device_set<K>;
        device_view(K* keys, int* d_size, typename map_type::device_view index_map_view)
            : keys(keys), d_size(d_size), index_map_view(index_map_view) {}

        K* keys;
        int* d_size;
        typename map_type::device_view index_map_view;
    };

    // Iterator for traversing the keys array
    class iterator {
    public:
        __device__ iterator(const K* ptr) : ptr(ptr) {}

        __device__ const K& operator*() const { return *ptr; }
        __device__ const K* operator->() const { return ptr; }

        __device__ iterator& operator++() {
            ++ptr;
            return *this;
        }

        __device__ bool operator!=(const iterator& other) const {
            return ptr != other.ptr;
        }

    private:
        const K* ptr;
    };

    __host__ __device__ iterator begin() const { return iterator(keys); }
    __host__ __device__ iterator end() const { return iterator(keys + *d_size); }
    
    // ================================
    // Host-side implementation
    // ================================
    device_set()
        : capacity(SET_CAPACITY),
        d_index_map(SET_CAPACITY,
                      cuco::empty_key<K>{-1},
                      cuco::empty_value<int>{-1}) {
        allocate(SET_CAPACITY);
    }

    device_set(size_t cap)
        : capacity(cap),
          d_index_map(cap,
              cuco::empty_key<K>{-1}, cuco::empty_value<int>{-1}) {
        allocate(cap);
    }

    __host__ void allocate(int cap) {
        cudaMalloc(&keys, cap * sizeof(K));
        
        //cudaMallocManaged(&d_size, sizeof(int));
        int h_size = 0;               // host variable (initialize to 0 or whatever value you need)
        // Allocate device memory
        cudaMalloc(&d_size, sizeof(int));
        //cudaMemset(d_size, 0, sizeof(int));
        cudaMemcpy(d_size, &h_size, sizeof(int), cudaMemcpyHostToDevice); 
    } 

    ~device_set() {
        destroy();
    }

    __host__ void destroy() {
        d_index_map.~static_map();  // <---- ADD THIS
        cudaFree(keys);
        cudaFree(d_size); 
    }

    // movable automatically (because of unique_ptr)
    device_set(device_set&&) noexcept = default;
    device_set& operator=(device_set&&) noexcept = default;

    // disable copy
    device_set(const device_set&) = delete;
    device_set& operator=(const device_set&) = delete;

    // Insert key (host)
    void insert(const K& key) {
        int h_size;
        cudaMemcpy(&h_size, d_size, sizeof(int), cudaMemcpyDeviceToHost);

        if (h_size >= capacity) {
            throw std::runtime_error("device_set capacity exceeded");
        }

        cudaMemcpy(&keys[h_size], &key, sizeof(K), cudaMemcpyHostToDevice);

        cuco::pair<K, int> kv{key, h_size};
        d_index_map.insert(&kv, &kv + 1);

        h_size++;
        cudaMemcpy(d_size, &h_size, sizeof(int), cudaMemcpyHostToDevice);
    }

    __device__ __forceinline__
    const K at(const int idx) {
        return keys[idx];
    }

    // Erase key (host)
    void erase(const K& key) {
        // Lookup index from map
        auto d_view = d_index_map.get_device_view();
        // NOTE: cuco::static_map::erase only works from device kernels.
        // For host erase, you need to rebuild or manage a free list.
        // Here we leave it as TODO.
        throw std::runtime_error("erase on host not yet implemented!");
    }

    // Create device view
    device_view get_device_view() const {
        return device_view(keys, d_size, d_index_map.get_device_view());
    }

    __device__ int getSize() const {
        return *d_size;
    }

    // Host-side helpers
    int size() {
        int h_size;
        cudaMemcpy(&h_size, d_size, sizeof(int), cudaMemcpyDeviceToHost);
        return h_size;
    } 

    int set_capacity() const { return capacity; }

public:
    K* keys;
    int* d_size;
    size_t capacity;
    map_type d_index_map;
};