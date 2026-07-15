#pragma once

#ifndef DEVICE_MIN_HEAP_ARRAY_CUH
#define DEVICE_MIN_HEAP_ARRAY_CUH

#include "device_min_heap.cuh"

// Array of min heaps manager
template<typename T = float>
struct device_min_heap_array {
    device_min_heap<T>* heaps;           // Array of heap structures (one per thread)
    typename device_min_heap<T>::Element* bulk_storage;  // Bulk storage for all heap elements
    int num_heaps;                       // Number of heaps (p threads)
    int* capacities;                     // Capacity for each heap (K values)
    size_t* offsets;                     // Offset for each heap in bulk storage
};

#endif // DEVICE_MIN_HEAP_ARRAY_CUH