#pragma once
#ifndef ER_KERNELS_CUH
#define ER_KERNELS_CUH

// ─────────────────────────────────────────────────────────────────────────────
// er_kernels.cuh  –  Erdős-Rényi GPU kernels for MASS CUDA
//
// Provides:
//   - device_vector_er<T>      : per-thread dynamic output vector (GPU)
//   - create_er_vectors_bulk() : host-side bulk allocator
//   - cleanup_er_vectors_bulk(): host-side bulk deallocator
//   - collect_er_edges()       : copy device vectors → host pair<int,int>
//   - kernelErdosRenyiGNP      : G(n,p) – each edge exists w.p. p
//   - kernelErdosRenyiFixedK   : each new node connects to exactly k nodes
//   - launchERKernel()         : host dispatcher
// ─────────────────────────────────────────────────────────────────────────────

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <utility>
#include <cstdio>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
// CUDA error-check macro (local, won't conflict with MASS_base.h)
// ─────────────────────────────────────────────────────────────────────────────
#ifndef ER_CUDA_CHECK
#define ER_CUDA_CHECK(call)                                                     \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "ER CUDA error %s:%d  %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(_err));              \
            throw std::runtime_error(cudaGetErrorString(_err));                 \
        }                                                                       \
    } while (0)
#endif

// ─────────────────────────────────────────────────────────────────────────────
// int2_er  –  simple 2-int struct so we don't collide with CUDA's int2
// ─────────────────────────────────────────────────────────────────────────────
struct int2_er {
    int x;  // source (new node)
    int y;  // target (existing node)
};

// ─────────────────────────────────────────────────────────────────────────────
// device_vector_er<T>  –  minimal resizable device-side array
//
// Memory layout: data[] is a slice of a large bulk allocation managed by
// the host (create_er_vectors_bulk / cleanup_er_vectors_bulk).
// d_size and d_capacity are small per-vector device ints.
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
struct device_vector_er {
    T*   data;          // pointer into bulk device buffer
    int* d_size;        // device int: current element count
    int* d_capacity;    // device int: max capacity

    // ── device interface ─────────────────────────────────────────────────────
    __device__ bool push_back(const T& val) {
        int pos = atomicAdd(d_size, 1);
        if (pos < *d_capacity) {
            data[pos] = val;
            return true;
        }
        // overflow – nothing we can do silently on device
        return false;
    }

    __device__ int size() const { return *d_size; }
};

// ─────────────────────────────────────────────────────────────────────────────
// Host helpers: bulk allocate / free / collect
// ─────────────────────────────────────────────────────────────────────────────

// Allocate `n` per-thread device_vector_er<T> objects backed by a single
// bulk device allocation. capacities[i] is the reserved capacity for thread i.
template <typename T>
inline void create_er_vectors_bulk(int n,
                                    const int* capacities,
                                    device_vector_er<T>** d_vecs_out)
{
    std::vector<device_vector_er<T>> h(n);

    // 1. Allocate d_size and d_capacity per vector
    for (int i = 0; i < n; i++) {
        ER_CUDA_CHECK(cudaMalloc(&h[i].d_size,     sizeof(int)));
        ER_CUDA_CHECK(cudaMalloc(&h[i].d_capacity, sizeof(int)));
        int zero = 0, cap = capacities[i];
        ER_CUDA_CHECK(cudaMemcpy(h[i].d_size,     &zero, sizeof(int), cudaMemcpyHostToDevice));
        ER_CUDA_CHECK(cudaMemcpy(h[i].d_capacity, &cap,  sizeof(int), cudaMemcpyHostToDevice));
        h[i].data = nullptr;
    }

    // 2. Compute total elements and offsets
    std::vector<size_t> offsets(n);
    size_t total = 0;
    for (int i = 0; i < n; i++) { offsets[i] = total; total += capacities[i]; }

    // 3. Single bulk malloc
    T* bulk = nullptr;
    if (total > 0) {
        ER_CUDA_CHECK(cudaMalloc(&bulk, total * sizeof(T)));
        ER_CUDA_CHECK(cudaMemset(bulk, 0, total * sizeof(T)));
    }
    for (int i = 0; i < n; i++)
        h[i].data = (capacities[i] > 0) ? bulk + offsets[i] : nullptr;

    // 4. Upload array of structs to device
    ER_CUDA_CHECK(cudaMalloc(d_vecs_out, n * sizeof(device_vector_er<T>)));
    ER_CUDA_CHECK(cudaMemcpy(*d_vecs_out, h.data(),
                              n * sizeof(device_vector_er<T>),
                              cudaMemcpyHostToDevice));
}

// Free all device memory associated with a bulk-allocated vector array.
template <typename T>
inline void cleanup_er_vectors_bulk(device_vector_er<T>* d_vecs, int n)
{
    if (!d_vecs) return;

    std::vector<device_vector_er<T>> h(n);
    ER_CUDA_CHECK(cudaMemcpy(h.data(), d_vecs,
                              n * sizeof(device_vector_er<T>),
                              cudaMemcpyDeviceToHost));

    // Free bulk data buffer (pointed to by h[0].data)
    if (n > 0 && h[0].data) ER_CUDA_CHECK(cudaFree(h[0].data));

    // Free per-vector d_size / d_capacity
    for (int i = 0; i < n; i++) {
        if (h[i].d_size)     ER_CUDA_CHECK(cudaFree(h[i].d_size));
        if (h[i].d_capacity) ER_CUDA_CHECK(cudaFree(h[i].d_capacity));
    }

    ER_CUDA_CHECK(cudaFree(d_vecs));
}

// Copy edges from device_vector_er<int2_er>[] → host vector<pair<int,int>>
inline void collect_er_edges(device_vector_er<int2_er>* d_vecs,
                              int n,
                              std::vector<std::pair<int,int>>& out)
{
    std::vector<device_vector_er<int2_er>> h(n);
    ER_CUDA_CHECK(cudaMemcpy(h.data(), d_vecs,
                              n * sizeof(device_vector_er<int2_er>),
                              cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; i++) {
        int sz = 0;
        if (h[i].d_size)
            ER_CUDA_CHECK(cudaMemcpy(&sz, h[i].d_size, sizeof(int), cudaMemcpyDeviceToHost));
        if (sz == 0 || !h[i].data) continue;
        std::vector<int2_er> buf(sz);
        ER_CUDA_CHECK(cudaMemcpy(buf.data(), h[i].data,
                                  sz * sizeof(int2_er), cudaMemcpyDeviceToHost));
        for (auto& e : buf) out.push_back({e.x, e.y});
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// kernelErdosRenyiGNP  –  G(n,p): each edge to every existing node w.p. p
//
// One thread per new node (local_idx ∈ [0, batch_size)).
// new_node_ids[local_idx] is the global paper index of the new node.
// current_graph_size is the number of OLD (seed + prior-year) nodes that can
// be cited; we iterate target ∈ [0, current_graph_size).
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kernelErdosRenyiGNP(
    int                          batch_size,
    const int*                   new_node_ids,
    device_vector_er<int2_er>*   edge_vecs,
    curandState*                 rng_states,
    unsigned long long           seed,
    int                          current_graph_size,
    double                       edge_probability)
{
    int local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_idx >= batch_size) return;

    // Seed RNG per thread
    curand_init(seed, local_idx, 0, &rng_states[local_idx]);
    curandState st = rng_states[local_idx];

    int new_node = new_node_ids[local_idx];
    device_vector_er<int2_er>& ev = edge_vecs[local_idx];

    for (int target = 0; target < current_graph_size; target++) {
        if (curand_uniform(&st) < (float)edge_probability) {
            int2_er e;  e.x = new_node;  e.y = target;
            ev.push_back(e);
        }
    }

    rng_states[local_idx] = st;
}

// ─────────────────────────────────────────────────────────────────────────────
// kernelErdosRenyiFixedK  –  each new node cites exactly k random old nodes
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kernelErdosRenyiFixedK(
    int                          batch_size,
    const int*                   new_node_ids,
    device_vector_er<int2_er>*   edge_vecs,
    curandState*                 rng_states,
    unsigned long long           seed,
    int                          current_graph_size,
    int                          edges_per_node)
{
    int local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_idx >= batch_size) return;

    curand_init(seed, local_idx, 0, &rng_states[local_idx]);
    curandState st = rng_states[local_idx];

    int new_node = new_node_ids[local_idx];
    device_vector_er<int2_er>& ev = edge_vecs[local_idx];

    int k = (edges_per_node < current_graph_size) ? edges_per_node : current_graph_size;
    for (int i = 0; i < k; i++) {
        int target = (int)(curand(&st) % (unsigned)current_graph_size);
        int2_er e;  e.x = new_node;  e.y = target;
        ev.push_back(e);
    }

    rng_states[local_idx] = st;
}

// ─────────────────────────────────────────────────────────────────────────────
// launchERKernel  –  host dispatcher
//
// Parameters
//   new_node_ids_h : host array of global paper indices for the new nodes
//   num_new        : length of new_node_ids_h
//   current_graph_size : number of old (citable) nodes
//   er_prob        : > 0  → G(n,p) model; use edge_probability
//   er_k           : > 0  → fixed-k model; use edges_per_node
//   year           : used to seed the RNG uniquely per year
//   out_edges      : appended with the generated edges
// ─────────────────────────────────────────────────────────────────────────────
inline void launchERKernel(const int*                       new_node_ids_h,
                            int                              num_new,
                            int                              current_graph_size,
                            double                           er_prob,
                            int                              er_k,
                            int                              year,
                            std::vector<std::pair<int,int>>& out_edges)
{
    if (num_new <= 0 || current_graph_size <= 0) return;

    // ── 1. Upload new-node id array ──────────────────────────────────────────
    int* d_ids = nullptr;
    ER_CUDA_CHECK(cudaMalloc(&d_ids, num_new * sizeof(int)));
    ER_CUDA_CHECK(cudaMemcpy(d_ids, new_node_ids_h,
                              num_new * sizeof(int), cudaMemcpyHostToDevice));

    // ── 2. Allocate per-thread curand states ─────────────────────────────────
    curandState* d_rng = nullptr;
    ER_CUDA_CHECK(cudaMalloc(&d_rng, num_new * sizeof(curandState)));

    // ── 3. Estimate per-thread capacity for edge output vectors ─────────────
    int cap_per_thread;
    if (er_prob > 0.0) {
        // Expected edges = p * graph_size; allocate 2× for variance headroom
        cap_per_thread = (int)(er_prob * current_graph_size * 2.0) + 4;
    } else {
        cap_per_thread = er_k + 4;
    }
    cap_per_thread = (cap_per_thread < 4) ? 4 : cap_per_thread;

    std::vector<int> caps(num_new, cap_per_thread);
    device_vector_er<int2_er>* d_edge_vecs = nullptr;
    create_er_vectors_bulk<int2_er>(num_new, caps.data(), &d_edge_vecs);

    // ── 4. Launch kernel ─────────────────────────────────────────────────────
    const int BS = 256;
    const int GS = (num_new + BS - 1) / BS;
    unsigned long long seed = (unsigned long long)year * 999983ULL + 7ULL;

    if (er_prob > 0.0) {
        kernelErdosRenyiGNP<<<GS, BS>>>(
            num_new, d_ids, d_edge_vecs, d_rng, seed,
            current_graph_size, er_prob);
    } else {
        kernelErdosRenyiFixedK<<<GS, BS>>>(
            num_new, d_ids, d_edge_vecs, d_rng, seed,
            current_graph_size, er_k);
    }
    ER_CUDA_CHECK(cudaGetLastError());
    ER_CUDA_CHECK(cudaDeviceSynchronize());

    // ── 5. Collect results ───────────────────────────────────────────────────
    collect_er_edges(d_edge_vecs, num_new, out_edges);

    // ── 6. Cleanup ───────────────────────────────────────────────────────────
    cleanup_er_vectors_bulk<int2_er>(d_edge_vecs, num_new);
    ER_CUDA_CHECK(cudaFree(d_rng));
    ER_CUDA_CHECK(cudaFree(d_ids));
}

#endif // ER_KERNELS_CUH
