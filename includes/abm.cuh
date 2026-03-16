#ifndef ABM_H
#define ABM_H

#include <cmath>
#include <chrono>
#include <condition_variable>
#include <random>
#include <thread>
#include <map>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <queue>
#include "graph.cuh"
#include "device_queue.cuh"
#include "device_set.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <vector>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>  // Optional

#include "device_string.cuh"
#include "device_set.cuh"
#include "device_vector.cuh"
#include "utils.cuh"

// ============================================================================
// CONFIGURATION CONSTANTS (from the paper)
// ============================================================================
// CTA (block) size for different expansion strategies
#define CONTRACT_EXPAND_CTA_SIZE 256
#define EXPAND_CTA_SIZE 128

// Work-stealing queue size
#define QUEUE_MULTIPLIER 2

enum Log {info, debug, error = -1}; 

// ============================================================================
// MERRILL-STYLE BFS STATE (Compact, Per-Thread)
// ============================================================================
struct CompactBFSState {
    // Visited tracking bitmap
    uint32_t* d_visited_bitmap;
    
    // Frontier bitmaps (ping-pong) - 1 bit per potential node
    uint32_t* d_queue_bitmap_curr;
    uint32_t* d_queue_bitmap_next;
    
    int bitmap_words;
    int max_vertices;
    
    // ========================================================================
    // HOST-SIDE INITIALIZATION
    // ========================================================================
    __host__ void init(int num_vertices) {
        max_vertices = num_vertices;
        bitmap_words = (num_vertices + 31) / 32;
        
        // Allocate visited bitmap
        CUDA_CHECK(cudaMalloc(&d_visited_bitmap, bitmap_words * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_visited_bitmap, 0, bitmap_words * sizeof(uint32_t)));
        
        // Allocate frontier bitmaps (ping-pong)
        CUDA_CHECK(cudaMalloc(&d_queue_bitmap_curr, bitmap_words * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_queue_bitmap_next, bitmap_words * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_queue_bitmap_curr, 0, bitmap_words * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_queue_bitmap_next, 0, bitmap_words * sizeof(uint32_t)));
    }
    
    // ========================================================================
    // HOST-SIDE CLEANUP
    // ========================================================================
    __host__ void cleanup() {
        if (d_visited_bitmap) cudaFree(d_visited_bitmap);
        if (d_queue_bitmap_curr) cudaFree(d_queue_bitmap_curr);
        if (d_queue_bitmap_next) cudaFree(d_queue_bitmap_next);
        
        d_visited_bitmap = nullptr;
        d_queue_bitmap_curr = nullptr;
        d_queue_bitmap_next = nullptr;
    }
    
    // ========================================================================
    // DEVICE-SIDE: Mark node as visited
    // Returns true if this is the first visit
    // ========================================================================
    __device__ __forceinline__ bool mark_visited(int node) {
        if (node < 0 || node >= max_vertices) return false;
        
        int word_idx = node >> 5;
        int bit_idx = node & 31;
        uint32_t mask = 1U << bit_idx;
        
        uint32_t old = atomicOr(&d_visited_bitmap[word_idx], mask);
        return (old & mask) == 0;
    }
    
    // ========================================================================
    // DEVICE-SIDE: Check if visited
    // ========================================================================
    __device__ __forceinline__ bool is_visited(int node) const {
        if (node < 0 || node >= max_vertices) return true;
        
        int word_idx = node >> 5;
        int bit_idx = node & 31;
        return (d_visited_bitmap[word_idx] >> bit_idx) & 1;
    }
    
    // ========================================================================
    // DEVICE-SIDE: Add node to next frontier
    // ========================================================================
    __device__ __forceinline__ void add_to_next_frontier(int node) {
        if (node < 0 || node >= max_vertices) return;
        
        int word_idx = node >> 5;
        int bit_idx = node & 31;
        uint32_t mask = 1U << bit_idx;
        
        atomicOr(&d_queue_bitmap_next[word_idx], mask);
    }
    
    // ========================================================================
    // DEVICE-SIDE: Check if node is in current frontier
    // ========================================================================
    __device__ __forceinline__ bool is_in_frontier(int node) const {
        if (node < 0 || node >= max_vertices) return false;
        
        int word_idx = node >> 5;
        int bit_idx = node & 31;
        return (d_queue_bitmap_curr[word_idx] >> bit_idx) & 1;
    }
    
    // ========================================================================
    // DEVICE-SIDE: Initialize frontier with root node
    // ========================================================================
    __device__ void init_frontier(int root) {
        // Clear both frontier bitmaps
        for (int i = 0; i < bitmap_words; i++) {
            d_queue_bitmap_curr[i] = 0;
            d_queue_bitmap_next[i] = 0;
        }
        
        // Add root to current frontier
        if (root >= 0 && root < max_vertices) {
            int word_idx = root >> 5;
            int bit_idx = root & 31;
            d_queue_bitmap_curr[word_idx] = 1U << bit_idx;
        }
    }
    
    // ========================================================================
    // DEVICE-SIDE: Swap frontier bitmaps
    // ========================================================================
    __device__ void swap_frontiers() {
        uint32_t* temp = d_queue_bitmap_curr;
        d_queue_bitmap_curr = d_queue_bitmap_next;
        d_queue_bitmap_next = temp;
        
        // Clear next frontier
        for (int i = 0; i < bitmap_words; i++) {
            d_queue_bitmap_next[i] = 0;
        }
    }
    
    // ========================================================================
    // DEVICE-SIDE: Reset all state
    // ========================================================================
    __device__ void reset() {
        for (int i = 0; i < bitmap_words; i++) {
            d_visited_bitmap[i] = 0;
            d_queue_bitmap_curr[i] = 0;
            d_queue_bitmap_next[i] = 0;
        }
    }
};

class ABM {
    public:
        ABM(std::string edgelist, 
            std::string nodelist, 
            std::string out_degree_bag, 
            std::string recency_probabilities, 
            std::string planted_nodes, double alpha, double fully_random_citations, 
            double preferential_weight, double recency_weight, double fitness_weight, 
            double growth_rate, int num_cycles, double same_year_proportion, 
            std::string output_file, std::string auxiliary_information_file, 
            std::string log_file, int num_processors, int log_level) : edgelist(edgelist), nodelist(nodelist), 
                        out_degree_bag(out_degree_bag), recency_probabilities(recency_probabilities), 
                        planted_nodes(planted_nodes), alpha(alpha), fully_random_citations(fully_random_citations), 
                        preferential_weight(preferential_weight), recency_weight(recency_weight), 
                        fitness_weight(fitness_weight), growth_rate(growth_rate), num_cycles(num_cycles), 
                        same_year_proportion(same_year_proportion), output_file(output_file), 
                        auxiliary_information_file(auxiliary_information_file), log_file(log_file), 
                        num_processors(num_processors), log_level(log_level) {
            if(this->log_level > -1) {
                this->start_time = std::chrono::steady_clock::now();
                this->log_file_handle.open(this->log_file);
            }
            this->num_calls_to_log_write = 0;
            this->ReadOutDegreeBag();
            this->ReadRecencyProbabilities();
            this->ReadPlantedNodes();
        };

        ~ABM() {
            if(this->log_level > -1) {
                this->log_file_handle.close();
            }
        }

    //template <typename SetRef>
	__device__ void ABMKernel(int idx, int N, int num_sources, 
                set_ref_type d_same_year_source_nodes_set_ref,
                curandState* deviceState,
                unsigned long long seed,
                Graph* graph, 
                int graphNodeSetSize,
                DeviceGraph* d_forward_adj_map_Graph,
                cuco::legacy::static_map<int, int>::device_view d_forward_adj_index_map_view,
                DeviceGraph* d_backward_adj_map_Graph,
                cuco::legacy::static_map<int, int>::device_view d_backward_adj_index_map_view,
                device_map<int, Node>::device_view d_nodeAttributeMap_view,
                int continuous_node_mapped_weight,
                cuco::legacy::static_map<int, int>::device_view d_continuous_node_mapping_view,
                cuco::legacy::static_map<int, int>::device_view d_reverse_continuous_node_mapping_view,
                int new_node,
                int per_thread_visited_set_capacity,
                int per_thread_selected_set_capacity,
                int per_thread_vector_capacity,
                int per_thread_queue_capacity,
                int num_generator_node_citation,
                device_vector_generic<int2>& d_new_edges_vec,
                set_ref_type& selected_citations,
                device_vector_generic<int>& citations,
                device_vector& one_hop_neighborhood,
                device_vector& two_hop_neighborhood,
                device_vector_soa<float>& element_index_vec,  
                set_ref_type visited_set_ref,
                device_queue<int2>& to_visit,
                double* pa_arr, double* recency_arr, double* fit_arr,
                double* pa_weight_arr, double* rec_weight_arr, 
                double* fit_weight_arr, double* alpha_arr,
                int* out_degree_arr,
                double fully_random_citations,
                int current_year, 
                int current_graph_size, 
                int initial_graph_size,
                int final_graph_size);

        __device__ void GetOneAndTwoHopNeighborhood(
                Graph* graph,
                int idx,
                int new_node,
                DeviceGraph* d_forward_adj_map,
                DeviceGraph* d_backward_adj_map,
                device_map<int, Node>::device_view d_nodeAttr_view,
                int generator_node,
                int num_generator_node_citation,
                CompactBFSState& bfs_state,
                device_vector& one_hop_neighborhood,
                device_vector& two_hop_neighborhood);

        // ============================================================================
        // WARP-COOPERATIVE BFS (Better for moderate/high degree)
        // Multiple threads in a warp cooperate on a single BFS
        // ============================================================================


        __device__ void GetOneAndTwoHopNeighborhood_Warp(
            Graph* graph,
            int idx,
            int new_node,
            DeviceGraph* d_forward_adj,
            DeviceGraph* d_backward_adj,
            device_map<int, Node>::device_view d_nodeAttr_view,
            int generator_node,
            int num_generator_node_citation,
            uint32_t* visited_bitmap,       // warp-private, size = max_vertices/32
            uint32_t* frontier_curr_bitmap, // warp-private
            uint32_t* frontier_next_bitmap, // warp-private
            device_vector& one_hop_neighborhood,
            device_vector& two_hop_neighborhood,
            int max_vertices);
         
        __device__ void ABMKernelStage2(
                    int idx, int new_node, int N,
                    Graph* graph,
                    curandState* deviceState,
                    device_vector& one_hop_neighborhood,
                    device_min_heap<float>& d_heap, 
                    // device_vector_soa<float>& element_index_vec, 
                    device_vector_generic<int>& citations,
                    double* pa_arr, double* recency_arr, double* fit_arr,
                    double pa_weight, double rec_weight, double fit_weight,
                    int current_year, 
                    int current_graph_size,
                    int initial_graph_size, 
                    int final_graph_size,
                    /*same-year flag*/ int same_year_citation,
                    /*precomputed*/ int& num_citations_inside,
                    /*in/out*/ int& num_actually_cited);

        __device__ void ABMKernelStage3(
                int idx, int N, Graph* graph, int graphNodeSetSize,
                device_vector& two_hop_neighborhood,
                device_min_heap<float>& d_heap, 
                device_vector_generic<int>& citations,
                set_ref_type& selected_citations,
                int generator_node, int new_node,
                double* pa_arr, double* recency_arr, double* fit_arr,
                double pa_weight, double rec_weight, double fit_weight,
                int current_year, int current_graph_size,
                int initial_graph_size, int final_graph_size,
                int num_actually_cited_so_far,
                int& num_citations_outside,
                int& num_actually_cited,
                curandState* deviceState);

        __device__ void ABMKernelStage4(
                int idx, int N, Graph* graph, int graphNodeSetSize,
                //device_vector_soa<float>& element_index_vec, 
                device_vector_generic<int>& citations,
                device_vector_generic<int2>& d_new_edges_vec,
                set_ref_type& selected_citations,
                int generator_node, int new_node,
                //double* pa_arr, double* recency_arr, double* fit_arr,
                //double pa_weight, double rec_weight, double fit_weight,
                int current_year, int current_graph_size,
                int initial_graph_size, int final_graph_size,
                int& num_citations_outside, int& num_fully_random_cited,
                int& num_actually_cited,
                curandState* deviceState,
                int per_thread_selected_set_capacity);

        int WriteToLogFile(std::string message, Log message_type);
        void ReadOutDegreeBag();
        void ReadRecencyProbabilities();
        void ReadPlantedNodes();
        std::map<int, int> BuildContinuousNodeMapping(Graph* graph);
        std::map<int, int> ReverseMapping(std::map<int, int> mapping);
        //__host__ __device__ thrust::device_vector GetComplement(Graph* graph, const thrust::device_vector& base_vec, const cuco::legacy::static_map<int, int>& reverse_continuous_node_mapping);
        int GetFinalGraphSize(Graph* graph);
        //std::vector<int> GetGeneratorNodes(Graph* graph, const std::map<int, int>& reverse_continuous_node_mapping);
        int getGeneratorNode(Graph* graph /*, const std::map<int, int>& reverse_continuous_node_mapping*/);
        //std::vector<int> parse_generator_nodes(const device_string& generator_node_string);
        __device__ int getGraphAttributesGeneratorNode(Graph* graph, 
            device_map<int, Node>::device_view d_nodeAttributeMap_view, int new_node);
        
        void FillInDegreeArr(Graph* graph, /*const std::map<int, int>& continuous_node_mapping,*/ int* in_degree_arr);
        void InitializeFitness(Graph* graph);
        void FillFitnessArr(Graph* graph, /*const std::map<int, int>& continuous_node_mapping,*/ int current_year, int* fitness_arr);
        void FillRecencyArr(Graph* graph, /*const std::map<int, int>& continuous_node_mapping,*/ int current_year, double* recency_arr);
        void PopulateWeightArrs(double* pa_weight_arr, double* rec_weight_arr, double* fit_weight_arr, int len);
        void PopulateAlphaArr(double* alpha_arr, int len);
        int GetMaxYear(Graph* graph);
        int GetMaxNode(Graph* graph);
        void PopulateOutDegreeArr(int* out_degree_arr, int len);
        void CalculateScores(int* src_arr, double* dst_arr, int len);
        __device__ int getRandom(int num_new_nodes, curandState* state);
        __device__ void MakeCitations(int idx, int new_node, int N, Graph* graph, 
            curandState* deviceState,
            int current_year, 
            device_vector& candidate_nodes, 
            device_vector_soa<float>& element_index_vec,
            device_vector_generic<int>& citations, 
            double* pa_arr, double* recency_arr, double* fit_arr, 
            double pa_weight, double rec_weight, double fit_weight, 
            int current_graph_size, 
            int initial_graph_size,
            int final_graph_size,
            int& num_citations);
        __device__ void PopulateCitations(int idx, int new_node, int N, Graph* graph, 
            curandState* deviceState,
            int current_year, 
            device_vector_soa<float>& element_index_vec,
            device_vector_generic<int>& citations, 
            int num_cited_so_far,
            int current_graph_size, 
            int initial_graph_size,
            int final_graph_size,
            int num_citations);
        __device__ void MakePopulateCitations(int idx, int new_node, int N, Graph* graph, 
            curandState* deviceState,
            int current_year, 
            device_vector& candidate_nodes, 
            //device_vector_soa<float>& element_index_vec,
            device_min_heap<float>& d_heap,
            device_vector_generic<int>& citations, 
            double* pa_arr, double* recency_arr, double* fit_arr, 
            double pa_weight, double rec_weight, double fit_weight, 
            int current_graph_size, 
            int initial_graph_size,
            int final_graph_size,
            int num_cited_so_far,
            int& num_citations,
            int& num_actually_cited);
        void FillSameYearSourceNodes(std::set<int>& same_year_source_nodes, int current_year_new_nodes);
        __device__ int MakeUniformRandomCitations(Graph* graph, int i,
            int graphNodeSetSize, 
            set_ref_type& selected_citations,
            int per_thread_selected_set_capacity,
            int generator_node, device_vector_generic<int>& citations, 
            int num_cited_so_far, 
            int num_citations, 
            curandState* deviceState);
        __device__ int MakeSameYearCitations(int idx, int new_node, int num_new_nodes, 
            //cuco::legacy::static_map<int, int>::device_view d_reverse_continuous_node_mapping_view, 
            device_vector_generic<int>& citations, int current_graph_size, curandState* deviceState);
        void UpdateGraphAttributesWeights(Graph* graph, int next_node_id, double* pa_weight_arr, double* rec_weight_arr, double* fit_weight_arr, int len);
        void UpdateGraphAttributesAlphas(Graph* graph, int next_node_id, double* alpha_arr, int len);
        void UpdateGraphAttributesOutDegrees(Graph* graph, int next_node_id, int* out_degree_arr, int len);
        void UpdateGraphAttributesGeneratorNodes(Graph* graph, int new_node, const std::vector<int> generator_nodes);
        void updateGraphAttributesGeneratorNode(Graph* graph, int new_node, int generatorNodeId);

        template<typename T>
        void AssignFitnessLagDuration(Graph* graph, const T& container) {
            for(auto const& node : container) {
                std::random_device rand_dev;
                std::minstd_rand generator{rand_dev()};
                /* int fitness_lag_uniform = this->fitness_lag_duration_uniform_distribution(generator); */
                int fitness_lag_uniform = 0; // MARK: hard coded to be static fitness
                graph->SetIntAttribute("fitness_lag_duration", node, fitness_lag_uniform);
            }
        } 
        
        template<typename T>
        void AssignFitnessPeakDuration(Graph* graph, const T& container) {
            for(auto const& node : container) {
                std::random_device rand_dev;
                std::minstd_rand generator{rand_dev()};
                /* int fitness_peak_uniform = this->fitness_peak_duration_uniform_distribution(generator); */
                int fitness_peak_uniform = 1000; // MARK: hard coded to be static fitness
                graph->SetIntAttribute("fitness_peak_duration", node, fitness_peak_uniform);
            }
        }

        template<typename T>
        void AssignPeakFitnessValues(Graph* graph, const T& container) {
            std::vector<double> fitness_probabilities;
            for(int i = this->fitness_value_min; i <  this->fitness_value_max + 1; i ++) {
                double scale_factor = 6.3742991333;
                double constant = 0.072;
                double exponent = -1.634;
                fitness_probabilities.push_back(scale_factor * constant * pow(i, exponent));
            }
            std::random_device rand_dev;
            std::minstd_rand generator{rand_dev()};
            std::discrete_distribution<int> int_discrete_distribution(fitness_probabilities.begin(), fitness_probabilities.end());
            for(auto const& node : container) {
                int current_fitness = int_discrete_distribution(generator) + 1;
                graph->SetIntAttribute("fitness_peak_value", node, current_fitness);
            }
        }

        void PlantNodes(Graph* graph, std::vector<int> new_nodes_vec, int current_year) {
            int planted_so_far = 0;
            if (this->planted_nodes_map.count(current_year)) {
                std::map<int, std::map<std::string, int>> current_year_map = this->planted_nodes_map.at(current_year);
                for(auto const& [line_no, line_map] : current_year_map) {
                    int current_node_type_count = line_map.at("count");
                    int current_fitness_lag_duration = line_map.at("fitness_lag_duration");
                    int current_fitness_peak_value = line_map.at("fitness_peak_value");
                    int current_fitness_peak_duration = line_map.at("fitness_peak_duration");
                    for(int i = planted_so_far; i < planted_so_far + current_node_type_count; i ++) {
                        graph->SetIntAttribute("fitness_lag_duration", new_nodes_vec.at(i), current_fitness_lag_duration);
                        graph->SetIntAttribute("fitness_peak_value", new_nodes_vec.at(i), current_fitness_peak_value);
                        graph->SetIntAttribute("fitness_peak_duration", new_nodes_vec.at(i), current_fitness_peak_duration);
                        graph->SetIntAttribute("planted_nodes_line_number", new_nodes_vec.at(i), line_no);
                    }
                    planted_so_far += current_node_type_count;
                }
            }
        }

        void test();

        __host__ __device__ double get_alpha() const { return alpha; }
        __host__ __device__ double get_fully_random_citations() const { return fully_random_citations; }
        __host__ __device__ double get_preferential_weight() const { return preferential_weight; }
        __host__ __device__ double get_recency_weight() const { return recency_weight; }
        __host__ __device__ double get_fitness_weight() const { return fitness_weight; }
        __host__ __device__ double get_same_year_proportion() const { return same_year_proportion; }



    public:
        std::string edgelist;
        std::string nodelist;
        std::string output_file;
        std::string auxiliary_information_file;
        int num_cycles;
        int num_processors;
        double growth_rate;
        
        
    protected:
        std::string out_degree_bag;
        std::string recency_probabilities;
        std::string planted_nodes;
        double alpha;
        double fully_random_citations;
        double preferential_weight;
        double recency_weight;
        double fitness_weight;
        double same_year_proportion;
        
        std::string log_file;
        int log_level;
        std::chrono::steady_clock::time_point start_time;
        std::ofstream log_file_handle;
        int num_calls_to_log_write;
        const int fitness_value_min = 1;
        const int fitness_value_max = 1000;
        const int fitness_lag_duration_min = 1;
        const int fitness_lag_duration_max = 7;
        const int fitness_peak_duration_min = 1;
        const int fitness_peak_duration_max = 7;
        const int fitness_alpha = -3;
        const int fitness_decay_alpha = 3;
        const int gamma = 3;
        const int max_author_lifetime = 30;
        const int k = 2;
        int next_author_id = 0;
        std::uniform_real_distribution<double> fitness_value_uniform_distribution{0, 1};
        std::uniform_real_distribution<double> weights_uniform_distribution{0, 1};
        std::uniform_real_distribution<double> wrs_uniform_distribution{0, 1};
        std::uniform_real_distribution<double> alpha_uniform_distribution{0, 1};
        std::uniform_int_distribution<int> fitness_lag_duration_uniform_distribution{fitness_lag_duration_min, fitness_lag_duration_max};
        std::uniform_int_distribution<int> fitness_peak_duration_uniform_distribution{fitness_peak_duration_min, fitness_peak_duration_max};
        std::vector<int> out_degree_bag_vec;
        std::map<int, double> recency_probabilities_map;
        std::map<int, std::map<int, std::map<std::string, int>>> planted_nodes_map;
};

#endif
