#pragma once
#ifndef INT2_H_
#define INT2_H_

#include "abm.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>

#include <math.h>
#include <cmath>
#include <cuda_runtime_api.h>

#include <vector>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>

#include <cuco/static_map.cuh>
#include <cuda/functional>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/logical.h>
#include <thrust/transform.h>
#include <thrust/extrema.h> // min
#include <node.cuh>
#include "device_set.cuh"
#include "device_map.cuh"
#include <device_vector.cuh>
#include "utils.cuh"

using namespace std;

#define CUDA_OK(call)                                                    \
{                                                                        \
    cudaError_t err = call;                                              \
    if (err != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                #call, __FILE__, __LINE__, cudaGetErrorString(err));     \
        exit(EXIT_FAILURE);                                              \
    }                                                                    \
}

#define CUDA_DEBUG_CHECK() { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error after %s at %s:%d — %s\n", \
                __func__, __FILE__, __LINE__, cudaGetErrorString(err)); \
        abort(); \
    } \
}

#define CUDA_SYNC_CHECK(stream) { \
    cudaError_t err = cudaStreamSynchronize(stream); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "Stream sync error in %s at %s:%d — %s\n", \
                __func__, __FILE__, __LINE__, cudaGetErrorString(err)); \
        abort(); \
    } \
}

__host__ __device__
constexpr Node make_empty_node() {
    return Node {
                -1, //int type;
                -1, // int generatorNode;
                -1, // int fitness_peak_value;
                -1, // int fitness_lag_duration;
                -1, // int fitness_peak_duration;
                -1, // int published_year;
                -1, // int year;
                -1, // int out_degree;
                -1, // int assigned_out_degree;
                -1, // int in_degree;
                -1, // int planted_nodes_line_number;

                0.0, // double preferential_attachment_weight;
                0.0, // double recency_weight;
                0.0, // double fitness_weight;
                0.0  // double alpha;
    };
}

Node const empty_node_value_sentinel = make_empty_node();

__global__ void kernelCallStage1(
    int start_idx,
    int batch_size,
    int total_N,
    ABM* abm,
    Graph* graph,
    device_graph* d_forward_adj_map_Graph,
    device_graph* d_backward_adj_map_Graph,
    device_map<int, Node>::device_view d_nodeAttributeMap_view,
    ABMStageState* d_states,
    int* d_new_nodes_arr,
    int num_generator_node_citation,
    uint32_t* visited_pool,       // pool: batch_size * bitmap_words
    uint32_t* frontier_curr_pool,
    uint32_t* frontier_next_pool,
    device_vector* one_hop_neighborhood_vectors,
    device_vector* two_hop_neighborhood_vectors,
    int max_vertices);

void launch_stage1_kernel(
    int start_idx,
    int batch_size,
    int total_N,
    ABM* d_abm,
    Graph* d_graph,
    device_graph* d_forward_adj,
    device_graph* d_backward_adj,
    device_map<int, Node>::device_view d_nodeAttr_view,
    ABMStageState* d_states,
    int* d_new_nodes,
    int num_gen_citations,
    device_vector* d_one_hop_vecs,
    device_vector* d_two_hop_vecs,
    CompactBFSState* d_bfs_pool,
    int initial_graph_size);

__global__ void kernelCallStage2(
    int start_idx,
    int batch_size,
    int total_N,
    ABM* abm,
    Graph* graph,
    set_ref_type d_same_year_source_nodes_set_ref,
    ABMStageState* d_states,
    curandState* deviceStates,
    unsigned long long seed,
    int* d_new_nodes_arr,
    device_vector* one_hop_neighborhood_vectors,
    device_vector_soa<float>* element_index_vec_vectors,
    device_vector_generic<int>* citations_vectors,
    double* pa_arr, double* recency_arr, double* fit_arr,
    double* pa_weight_arr, double* rec_weight_arr,
    double* fit_weight_arr, double* alpha_arr,
    int* out_degree_arr,
    double fully_random_citations,
    int num_generator_node_citation,
    int current_year,
    int current_graph_size,
    int initial_graph_size,
    int final_graph_size);


__global__ void kernelCallStage3(
        int start_idx,
        int batch_size,
        int total_N,
        ABM* abm,
        Graph* graph,
        int graphNodeSetSize,
        cuco::legacy::static_map<int,int>::device_view d_continuous_node_mapping_view,
        cuco::legacy::static_map<int,int>::device_view d_reverse_continuous_node_mapping_view,
        device_map<int, Node>::device_view d_nodeAttributeMap_view,
        ABMStageState* d_states,
        curandState* deviceStates,
        int* d_new_nodes_arr,
        device_vector* two_hop_neighborhood_vectors,
        //device_vector_soa<float>* element_index_vec_vectors,
        device_min_heap<float>* d_heap_array,
        device_vector_generic<int>* citations_vectors,
        device_vector_generic<int2>* d_new_edges_vec_vectors,
        set_ref_type* selected_citations_set_refs,
        double* pa_arr, double* recency_arr, double* fit_arr,
        double* pa_weight_arr, double* rec_weight_arr,
        double* fit_weight_arr, double* alpha_arr,
        int* out_degree_arr,
        int num_generator_node_citation,
        int current_year, int current_graph_size,
        int initial_graph_size, int final_graph_size);

__global__ void kernelCallStage4(
        int start_idx,
        int batch_size,
        int total_N,
        ABM* abm,
        Graph* graph,
        int graphNodeSetSize,
        device_map<int, Node>::device_view d_nodeAttributeMap_view,
        ABMStageState* d_states,
        curandState* deviceStates,
        int* d_new_nodes_arr,
        //device_vector* two_hop_neighborhood_vectors,
        //device_vector_soa<float>* element_index_vec_vectors,
        device_vector_generic<int>* citations_vectors,
        device_vector_generic<int2>* d_new_edges_vec_vectors,
        set_ref_type* selected_citations_set_refs, 
        int* out_degree_arr,
        int num_generator_node_citation,
        int current_year, int current_graph_size,
        int initial_graph_size, int final_graph_size, 
        int per_thread_selected_set_capacity);


void buildOneNodeConnections(ABM* abm, Graph* graph,
                std::map<int, int> continuous_node_mapping,
                std::map<int, int> reverse_continuous_node_mapping,
                std::vector<int>& new_nodes_vec,
                int num_sources,
                std::set<int> same_year_source_nodes,
                std::vector<std::pair<int, int>>& new_edges_vec,
                int num_generator_node_citation,
                double* pa_arr, double* recency_arr, double* fit_arr,
                double* pa_weight_arr,  double* rec_weight_arr, double* fit_weight_arr, double* alpha_arr,
                int* out_degree_arr,
                int current_year, 
                int current_graph_size, 
                int initial_graph_size, 
                int final_graph_size,
                int max_batch_size);

int execute(ABM* abm);

#endif /* INT2_H_ */
