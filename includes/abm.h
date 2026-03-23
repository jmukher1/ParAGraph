#ifndef ABM_H
#define ABM_H

#include <cmath>
#include <chrono>
#include <condition_variable>
#include <limits>
#include <random>
#include <thread>
#include <map>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <omp.h>
#include <queue>
#include "graph.h"


enum Log {info, debug, error = -1};

class ABM {
    public:
        ABM(std::string edgelist, std::string nodelist, std::string out_degree_bag,
            std::string recency_probabilities, std::string planted_nodes, double alpha,
            double fully_random_citations, double preferential_weight, double recency_weight,
            double fitness_weight, double growth_rate, int num_cycles, double same_year_proportion,
            std::string output_file, std::string auxiliary_information_file,
            std::string log_file, int num_processors, int log_level,
            std::string model_name = "pa", double er_probability = 0.01)
            : edgelist(edgelist), nodelist(nodelist), out_degree_bag(out_degree_bag),
              recency_probabilities(recency_probabilities), planted_nodes(planted_nodes),
              alpha(alpha), fully_random_citations(fully_random_citations),
              preferential_weight(preferential_weight), recency_weight(recency_weight),
              fitness_weight(fitness_weight), growth_rate(growth_rate), num_cycles(num_cycles),
              same_year_proportion(same_year_proportion), output_file(output_file),
              auxiliary_information_file(auxiliary_information_file), log_file(log_file),
              num_processors(num_processors), log_level(log_level),
              model_name(model_name), er_probability(er_probability) {
            if(this->log_level > -1) {
                this->start_time = std::chrono::steady_clock::now();
                this->log_file_handle.open(this->log_file);
            }
            this->num_calls_to_log_write = 0;
            this->ReadOutDegreeBag();
            this->ReadRecencyProbabilities();
            this->ReadPlantedNodes();
            omp_set_num_threads(this->num_processors);
        };

        ~ABM() {
            if(this->log_level > -1) this->log_file_handle.close();
        }

        int main();
        int WriteToLogFile(std::string message, Log message_type);
        void ReadOutDegreeBag();
        void ReadRecencyProbabilities();
        void ReadPlantedNodes();
        std::map<int, int> BuildContinuousNodeMapping(Graph* graph);
        std::map<int, int> ReverseMapping(std::map<int, int> mapping);
        std::vector<int> GetComplement(Graph* graph, const std::vector<int>& base_vec, const std::map<int, int>& reverse_continuous_node_mapping);
        int GetFinalGraphSize(Graph* graph);
        std::vector<int> GetGeneratorNodes(Graph* graph, const std::map<int, int>& reverse_continuous_node_mapping);
        std::vector<int> GetGraphAttributesGeneratorNodes(Graph* graph, int new_node) const;
        std::vector<int> GetNeighborhood(Graph* graph, const std::vector<int>& generator_nodes, const std::map<int, int>& reverse_continuous_node_mapping);
        std::map<int, std::vector<int>> GetOneAndTwoHopNeighborhood(Graph* graph, const std::vector<int>& generator_nodes, const std::map<int, int>& reverse_continuous_node_mapping);
        void FillInDegreeArr(Graph* graph, const std::map<int, int>& continuous_node_mapping, int* in_degree_arr);
        void InitializeFitness(Graph* graph);
        void FillFitnessArr(Graph* graph, const std::map<int, int>& continuous_node_mapping, int current_year, int* fitness_arr);
        void FillRecencyArr(Graph* graph, const std::map<int, int>& reverse_continuous_node_mapping, int current_year, double* recency_arr);
        void PopulateWeightArrs(double* pa_weight_arr, double* rec_weight_arr, double* fit_weight_arr, int len);
        void PopulateAlphaArr(double* alpha_arr, int len);
        int GetMaxYear(Graph* graph);
        int GetMaxNode(Graph* graph);
        void PopulateOutDegreeArr(int* out_degree_arr, int len);
        void CalculateScores(int* src_arr, double* dst_arr, int len);
        int MakeCitations(Graph* graph, const std::map<int, int>& continuous_node_mapping, int current_year, const std::vector<int>& candidate_nodes, int* citations, double* pa_arr, double* recency_arr, double* fit_arr, double pa_weight, double rec_weight, double fit_weight, int current_graph_size, int num_citations);
        void FillSameYearSourceNodes(std::set<int>& same_year_source_nodes, int current_year_new_nodes);
        int MakeUniformRandomCitations(Graph* graph, const std::map<int, int>& reverse_continuous_node_mapping, std::vector<int>& generator_nodes, int* citations, int num_cited_so_far, int num_citations);
        int SampleKNodes(Graph* graph, int current_graph_size, const std::map<int, int>& reverse_continuous_node_mapping, int* citations, int k);
        // FIX: added current_graph_size parameter so only pre-existing nodes are
        // eligible targets; buffer overflow from the old 250-slot stack array is
        // also eliminated because the caller sizes the buffer to current_graph_size+1.
        int MakeERGNPCitations(Graph* graph, const std::map<int, int>& reverse_continuous_node_mapping, int* citations, double p, int current_graph_size);
        int MakeSameYearCitations(int num_new_nodes, const std::map<int, int>& reverse_continuous_node_mapping, int* citations, int current_graph_size);
        void UpdateGraphAttributesWeights(Graph* graph, int next_node_id, double* pa_weight_arr, double* rec_weight_arr, double* fit_weight_arr, int len);
        void UpdateGraphAttributesAlphas(Graph* graph, int next_node_id, double* alpha_arr, int len);
        void UpdateGraphAttributesOutDegrees(Graph* graph, int next_node_id, int* out_degree_arr, int len);
        void UpdateGraphAttributesGeneratorNodes(Graph* graph, int new_node, const std::vector<int>& generator_nodes);

        template<typename T>
        void AssignFitnessLagDuration(Graph* graph, const T& container) {
            for(auto const& node : container) {
                int fitness_lag_uniform = 0; // hard coded to be static fitness
                graph->SetIntAttribute("fitness_lag_duration", node, fitness_lag_uniform);
            }
        }

        template<typename T>
        void AssignFitnessPeakDuration(Graph* graph, const T& container) {
            for(auto const& node : container) {
                int fitness_peak_uniform = 1000; // hard coded to be static fitness
                graph->SetIntAttribute("fitness_peak_duration", node, fitness_peak_uniform);
            }
        }

        template<typename T>
        void AssignPeakFitnessValues(Graph* graph, const T& container) {
            std::vector<double> fitness_probabilities;
            for(int i = this->fitness_value_min; i < this->fitness_value_max + 1; i++) {
                fitness_probabilities.push_back(6.3742991333 * 0.072 * pow(i, -1.634));
            }
            std::random_device rand_dev;
            std::minstd_rand generator{rand_dev()};
            std::discrete_distribution<int> int_discrete_distribution(
                fitness_probabilities.begin(), fitness_probabilities.end());
            for(auto const& node : container)
                graph->SetIntAttribute("fitness_peak_value", node,
                                       int_discrete_distribution(generator) + 1);
        }

        void PlantNodes(Graph* graph, std::vector<int> new_nodes_vec, int current_year) {
            int planted_so_far = 0;
            if(this->planted_nodes_map.count(current_year)) {
                auto& cur_year_map = this->planted_nodes_map.at(current_year);
                for(auto const& [line_no, line_map] : cur_year_map) {
                    int cnt  = line_map.at("count");
                    int lag  = line_map.at("fitness_lag_duration");
                    int peak = line_map.at("fitness_peak_value");
                    int dur  = line_map.at("fitness_peak_duration");
                    for(int i = planted_so_far; i < planted_so_far + cnt; i++) {
                        graph->SetIntAttribute("fitness_lag_duration",       new_nodes_vec.at(i), lag);
                        graph->SetIntAttribute("fitness_peak_value",         new_nodes_vec.at(i), peak);
                        graph->SetIntAttribute("fitness_peak_duration",      new_nodes_vec.at(i), dur);
                        graph->SetIntAttribute("planted_nodes_line_number",  new_nodes_vec.at(i), line_no);
                    }
                    planted_so_far += cnt;
                }
            }
        }


    protected:
        std::string edgelist;
        std::string nodelist;
        std::string out_degree_bag;
        std::string recency_probabilities;
        std::string planted_nodes;
        double alpha;
        double fully_random_citations;
        double preferential_weight;
        double recency_weight;
        double fitness_weight;
        double growth_rate;
        int num_cycles;
        // Model selection
        std::string model_name     = "pa";   // "pa" | "er" | "er-gnp"
        double      er_probability = 0.01;   // used only with --model er-gnp
        double same_year_proportion;
        std::string output_file;
        std::string auxiliary_information_file;
        std::string log_file;
        int num_processors;
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
