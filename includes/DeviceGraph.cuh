#ifndef DEVICE_GRAPH_CUH_
#define DEVICE_GRAPH_CUH_ 

#include <map>
#include <set>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>

struct DeviceGraph {
    //int* keys;       // keys of map
    int* edges;     // // CSR values (neighbors): flattened adjacency list 
    int* offsets;    // CSR offset array: offsets[i] = start index in values for keys[i]
    int num_vertices; // Number of unique keys (from the map of node to its adjacencies)
    int num_edges;
};

// must be trivially copyable
static_assert(std::is_trivially_copyable_v<DeviceGraph>, "DeviceGraph must be trivially copyable");


#endif 