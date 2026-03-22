#ifndef KERNEL_DISPATCHER_CUH
#define KERNEL_DISPATCHER_CUH

#include "abm.cuh"
#include "graph.cuh"
#include <set>
#include "kernel_erdos_renyi.cuh"

// Forward declaration of existing PA function
void buildOneNodeConnections(ABM* abm, Graph* graph,
                            std::vector<int>& new_nodes_vec,
                            int same_year_source_nodes_capacity,
                            std::set<int> same_year_source_nodes,
                            std::vector<std::pair<int, int>>& new_edges_vec,
                            int num_generator_node_citation,
                            double* pa_arr, double* recency_arr, double* fit_arr,
                            double* pa_weight_arr, double* rec_weight_arr,
                            double* fit_weight_arr, double* alpha_arr,
                            int* out_degree_arr,
                            int current_year, int current_graph_size,
                            int initial_graph_size, int final_graph_size,
                            int max_batch_size);

// Forward declaration of ER function
void buildErdosRenyiConnections(ABM* abm, Graph* graph,
                                std::vector<int>& new_nodes_vec,
                                std::vector<std::pair<int, int>>& new_edges_vec,
                                int current_graph_size,
                                int current_year);

#endif // KERNEL_DISPATCHER_CUH
