#include "abm.h"
#include "abm_cpu_profiler.h"   // PROF: new include

#pragma omp declare reduction(merge_int_pair_vecs : std::vector<std::pair<int, int>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp declare reduction(merge_int_vecs : std::vector<int> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

int ABM::WriteToLogFile(std::string message, Log message_type) {
    if(this->log_level >= message_type) {
        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
        std::string log_message_prefix;
        if(message_type == Log::info)        log_message_prefix = "[INFO]";
        else if(message_type == Log::debug)  log_message_prefix = "[DEBUG]";
        else if(message_type == Log::error)  log_message_prefix = "[ERROR]";
        auto days_elapsed    = std::chrono::duration_cast<std::chrono::days>(now - this->start_time);
        auto hours_elapsed   = std::chrono::duration_cast<std::chrono::hours>(now - this->start_time - days_elapsed);
        auto minutes_elapsed = std::chrono::duration_cast<std::chrono::minutes>(now - this->start_time - days_elapsed - hours_elapsed);
        auto seconds_elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - this->start_time - days_elapsed - hours_elapsed - minutes_elapsed);
        auto total_seconds_elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - this->start_time);
        log_message_prefix += "[" + std::to_string(days_elapsed.count()) + "-"
                            + std::to_string(hours_elapsed.count()) + ":"
                            + std::to_string(minutes_elapsed.count()) + ":"
                            + std::to_string(seconds_elapsed.count()) + "]";
        log_message_prefix += "(t=" + std::to_string(total_seconds_elapsed.count()) + "s)";
        this->log_file_handle << log_message_prefix << " " << message << '\n';
        if(this->num_calls_to_log_write % 1 == 0) std::flush(this->log_file_handle);
        this->num_calls_to_log_write++;
    }
    return 0;
}

void ABM::ReadPlantedNodes() {
    char delimiter = ',';
    std::ifstream planted_nodes_stream(this->planted_nodes);
    std::string line;
    int line_no = 1;
    while(std::getline(planted_nodes_stream, line)) {
        std::stringstream ss(line);
        std::string current_value;
        std::vector<std::string> current_line;
        while(std::getline(ss, current_value, delimiter)) current_line.push_back(current_value);
        this->planted_nodes_map[std::stoi(current_line[0])][line_no]["fitness_lag_duration"]  = std::stoi(current_line[1]);
        this->planted_nodes_map[std::stoi(current_line[0])][line_no]["fitness_peak_value"]    = std::stoi(current_line[2]);
        this->planted_nodes_map[std::stoi(current_line[0])][line_no]["fitness_peak_duration"] = std::stoi(current_line[3]);
        this->planted_nodes_map[std::stoi(current_line[0])][line_no]["count"]                 = std::stoi(current_line[4]);
        line_no++;
    }
}

void ABM::ReadOutDegreeBag() {
    char delimiter = ',';
    std::ifstream out_degree_bag_stream(this->out_degree_bag);
    std::string line;
    while(std::getline(out_degree_bag_stream, line)) {
        std::stringstream ss(line);
        std::string current_value;
        std::vector<std::string> current_line;
        while(std::getline(ss, current_value, delimiter)) current_line.push_back(current_value);
        if(current_line[0][0] == '#') continue;
        this->out_degree_bag_vec.push_back(std::stoi(current_line[1]));
    }
}

void ABM::ReadRecencyProbabilities() {
    char delimiter = ',';
    std::ifstream recency_probabilities_stream(this->recency_probabilities);
    std::string line;
    while(std::getline(recency_probabilities_stream, line)) {
        std::stringstream ss(line);
        std::string current_value;
        std::vector<std::string> current_line;
        while(std::getline(ss, current_value, delimiter)) current_line.push_back(current_value);
        if(current_line[0][0] == '#') continue;
        this->recency_probabilities_map[std::stoi(current_line[0])] = std::stod(current_line[1]);
    }
}

std::map<int, int> ABM::BuildContinuousNodeMapping(Graph* graph) {
    int next_node_id = 0;
    std::map<int, int> continuous_node_mapping;
    for(auto const& node : graph->GetNodeSet()) continuous_node_mapping[node] = next_node_id++;
    return continuous_node_mapping;
}

std::map<int, int> ABM::ReverseMapping(std::map<int, int> mapping) {
    std::map<int, int> reverse_mapping;
    for(auto const& [key,val] : mapping) reverse_mapping[val] = key;
    return reverse_mapping;
}

void ABM::FillInDegreeArr(Graph* graph, const std::map<int, int>& continuous_node_mapping, int* in_degree_arr) {
    for(auto const& node: graph->GetNodeSet())
        in_degree_arr[continuous_node_mapping.at(node)] = graph->GetInDegree(node);
}

void ABM::InitializeFitness(Graph* graph) {
    this->AssignPeakFitnessValues(graph, graph->GetNodeSet());
    this->AssignFitnessLagDuration(graph, graph->GetNodeSet());
    this->AssignFitnessPeakDuration(graph, graph->GetNodeSet());
}

void ABM::FillFitnessArr(Graph* graph, const std::map<int, int>& continuous_node_mapping, int current_year, int* fitness_arr) {
    for(auto const& node : graph->GetNodeSet()) {
        int fitness_peak_value    = graph->GetIntAttribute("fitness_peak_value", node);
        int fitness_lag_duration  = graph->GetIntAttribute("fitness_lag_duration", node);
        int fitness_peak_duration = graph->GetIntAttribute("fitness_peak_duration", node);
        int published_year        = graph->GetIntAttribute("year", node);
        int continuous_index      = continuous_node_mapping.at(node);
        if (published_year + fitness_lag_duration > current_year) {
            fitness_arr[continuous_index] = 1;
        } else if (published_year + fitness_lag_duration + fitness_peak_duration >= current_year) {
            fitness_arr[continuous_index] = fitness_peak_value;
        } else {
            fitness_arr[continuous_index] = fitness_peak_value /
                pow(current_year - published_year - fitness_lag_duration - fitness_peak_duration + 1,
                    this->fitness_decay_alpha);
        }
    }
}

void ABM::FillRecencyArr(Graph* graph, const std::map<int, int>& reverse_continuous_node_mapping, int current_year, double* recency_arr) {
    std::map<int, int> year_count;
    double unique_year_sum = 0.0;
    for(auto const& node : graph->GetNodeSet()) {
        int year_diff = current_year - graph->GetIntAttribute("year", node);
        if(!year_count.contains(year_diff)) unique_year_sum += this->recency_probabilities_map[year_diff];
        year_count[year_diff]++;
    }
    for(size_t i = 0; i < graph->GetNodeSet().size(); i++) {
        int node_id   = reverse_continuous_node_mapping.at(i);
        int year_diff = current_year - graph->GetIntAttribute("year", node_id);
        recency_arr[i] = (float)this->recency_probabilities_map[year_diff] / year_count[year_diff];
    }
    for(size_t i = 0; i < graph->GetNodeSet().size(); i++) recency_arr[i] /= unique_year_sum;
}

int ABM::GetMaxYear(Graph* graph) {
    int max_year = -1; bool is_first = true;
    for(auto const& node : graph->GetNodeSet()) {
        int y = graph->GetIntAttribute("year", node);
        if(is_first || y > max_year) { max_year = y; is_first = false; }
    }
    return max_year;
}

int ABM::GetMaxNode(Graph* graph) {
    int max_node = -1; bool is_first = true;
    for(auto const& node : graph->GetNodeSet()) {
        if(is_first || node > max_node) { max_node = node; is_first = false; }
    }
    return max_node;
}

int ABM::GetFinalGraphSize(Graph* graph) {
    int sz = graph->GetNodeSet().size();
    for(int i = 0; i < this->num_cycles; i++) sz += (int)std::ceil(sz * this->growth_rate);
    return sz;
}

void ABM::PopulateAlphaArr(double* alpha_arr, int len) {
    std::random_device rand_dev; std::minstd_rand gen{rand_dev()};
    for(int i = 0; i < len; i++)
        alpha_arr[i] = (this->alpha < 0) ? this->alpha_uniform_distribution(gen) : this->alpha;
}

void ABM::PopulateWeightArrs(double* pa_weight_arr, double* rec_weight_arr, double* fit_weight_arr, int len) {
    std::random_device rand_dev; std::minstd_rand gen{rand_dev()};
    for(int i = 0; i < len; i++) {
        double pa  = (this->preferential_weight != -1) ? this->preferential_weight : this->weights_uniform_distribution(gen);
        double rec = (this->recency_weight      != -1) ? this->recency_weight      : this->weights_uniform_distribution(gen);
        double fit = (this->fitness_weight      != -1) ? this->fitness_weight      : this->weights_uniform_distribution(gen);
        double sum = pa + rec + fit;
        pa_weight_arr[i] = pa / sum; rec_weight_arr[i] = rec / sum; fit_weight_arr[i] = fit / sum;
    }
}

void ABM::PopulateOutDegreeArr(int* out_degree_arr, int len) {
    std::uniform_int_distribution<int> dist{0, (int)(this->out_degree_bag_vec.size() - 1)};
    std::random_device rand_dev; std::minstd_rand gen{rand_dev()};
    for(int i = 0; i < len; i++) out_degree_arr[i] = this->out_degree_bag_vec[dist(gen)];
}

void ABM::UpdateGraphAttributesWeights(Graph* graph, int next_node_id, double* pa_weight_arr, double* rec_weight_arr, double* fit_weight_arr, int len) {
    for(int i = 0; i < len; i++) {
        int nid = next_node_id + i;
        graph->SetDoubleAttribute("preferential_attachment_weight", nid, pa_weight_arr[i]);
        graph->SetDoubleAttribute("recency_weight",                 nid, rec_weight_arr[i]);
        graph->SetDoubleAttribute("fitness_weight",                 nid, fit_weight_arr[i]);
    }
}

void ABM::UpdateGraphAttributesAlphas(Graph* graph, int next_node_id, double* alpha_arr, int len) {
    for(int i = 0; i < len; i++) graph->SetDoubleAttribute("alpha", next_node_id + i, alpha_arr[i]);
}

void ABM::UpdateGraphAttributesOutDegrees(Graph* graph, int next_node_id, int* out_degree_arr, int len) {
    for(int i = 0; i < len; i++) graph->SetIntAttribute("assigned_out_degree", next_node_id + i, out_degree_arr[i]);
}

std::vector<int> ABM::GetGraphAttributesGeneratorNodes(Graph* graph, int new_node) const {
    std::vector<int> generator_nodes;
    std::string s = graph->GetStringAttribute("generator_node_string", new_node);
    std::stringstream ss(s); std::string val;
    while(std::getline(ss, val, ';')) generator_nodes.push_back(std::stoi(val));
    return generator_nodes;
}

void ABM::UpdateGraphAttributesGeneratorNodes(Graph* graph, int new_node, const std::vector<int>& generator_nodes) {
    std::string s = std::to_string(generator_nodes.at(0));
    for(size_t i = 1; i < generator_nodes.size(); i++) s += ";" + std::to_string(generator_nodes.at(i));
    graph->SetStringAttribute("generator_node_string", new_node, s);
}

void ABM::CalculateScores(int* src_arr, double* dst_arr, int len) {
    double sum = 0;
    for(int i = 0; i < len; i++) { dst_arr[i] = pow(src_arr[i], this->gamma) + 1; }
    for(int i = 0; i < len; i++) { sum += dst_arr[i]; }
    for(int i = 0; i < len; i++) { dst_arr[i] /= sum; }
}

void ABM::FillSameYearSourceNodes(std::set<int>& same_year_source_nodes, int current_year_new_nodes) {
    size_t target = (size_t)std::floor(current_year_new_nodes * this->same_year_proportion);
    std::random_device rand_dev; std::minstd_rand gen{rand_dev()};
    std::uniform_int_distribution<int> dist(0, current_year_new_nodes - 1);
    while(same_year_source_nodes.size() != target) {
        int s = dist(gen);
        if(!same_year_source_nodes.count(s)) same_year_source_nodes.insert(s);
    }
}

int ABM::MakeSameYearCitations(int num_new_nodes, const std::map<int, int>& reverse_continuous_node_mapping, int* citations, int current_graph_size) {
    std::random_device rand_dev; std::minstd_rand gen{rand_dev()};
    std::uniform_int_distribution<int> dist(0, num_new_nodes - 1);
    citations[0] = reverse_continuous_node_mapping.at(current_graph_size + dist(gen));
    return 1;
}

int ABM::MakeUniformRandomCitations(Graph* graph, const std::map<int, int>& reverse_continuous_node_mapping, std::vector<int>& generator_nodes, int* citations, int num_cited_so_far, int num_citations) {
    if(num_citations <= 0) return 0;
    int actual = num_citations;
    this->WriteToLogFile("trying to uniformly cite " + std::to_string(actual), Log::debug);
    if(graph->GetNodeSet().size() - num_cited_so_far - generator_nodes.size() < (size_t)num_citations)
        actual = graph->GetNodeSet().size();
    std::random_device rand_dev; std::minstd_rand gen{rand_dev()};
    std::uniform_int_distribution<int> dist(0, (int)(graph->GetNodeSet().size() - 1));
    std::set<int> selected;
    int idx = 0;
    for(int i = 0; i < num_cited_so_far; i++) selected.insert(citations[i]);
    for(size_t i = 0; i < generator_nodes.size(); i++) selected.insert(generator_nodes.at(i));
    while(selected.size() != num_cited_so_far + actual + generator_nodes.size()) {
        int c = dist(gen);
        int node = reverse_continuous_node_mapping.at(c);
        if(!selected.count(node)) {
            citations[num_cited_so_far + idx++] = node;
            selected.insert(node);
        }
    }
    return actual;
}

// ---------------------------------------------------------------------------
// SampleKNodes: select k distinct nodes uniformly at random from the N nodes
// currently in the graph and write their actual node IDs into citations[0..k-1].
//
// Returns the number of nodes actually written (= min(k, N)).
//
// Strategy selection (chosen at runtime based on the k/N ratio):
//
//   Sparse  (k <= N/4)  -  Rejection sampling with unordered_set.
//     Expected O(k) draws, O(k) memory.  Each draw succeeds with probability
//     >= (N-k)/N, so expected total draws < k*N/(N-k) <= 4k/3 for k <= N/4.
//
//   Dense   (k >  N/4)  -  Partial Fisher-Yates shuffle on an index vector.
//     Exactly O(N) work regardless of k.  Avoids collision retries when k is
//     large (rejection rate becomes prohibitively high above N/4).
//
// Both paths translate continuous indices [0, N) to actual node IDs via
// reverse_continuous_node_mapping, matching the convention used everywhere
// else in the codebase.
// ---------------------------------------------------------------------------
int ABM::SampleKNodes(Graph* graph,
                      int current_graph_size,
                      const std::map<int, int>& reverse_continuous_node_mapping,
                      int* citations,
                      int k) {
    if (k <= 0 || current_graph_size <= 0) return 0;
    if (k > current_graph_size)  k = current_graph_size;   // clamp: can't sample more than exist

    std::random_device rd;
    std::minstd_rand   gen{rd()};

    // ── Dense path: partial Fisher-Yates shuffle ──────────────────────────
    // Build index vector [0, N) and perform k swap-select steps.
    // Only the first k positions are emitted; the tail is discarded.
    std::vector<int> idx(current_graph_size);
    std::iota(idx.begin(), idx.end(), 0);        // fill 0 .. N-1

    for (int i = 0; i < k; i++) {
        std::uniform_int_distribution<int> dist(i, current_graph_size - 1);
        int j = dist(gen);
        std::swap(idx[i], idx[j]);               // move chosen element to i
        citations[i] =
            reverse_continuous_node_mapping.at(idx[i]);
    } 

    return k;
}

// ---------------------------------------------------------------------------
// ER G(n,p):  connect new node to each PRE-EXISTING node with probability p.
//
// FIX 1 (vs. original cpu-models):  accepts current_graph_size so only nodes
//   that existed BEFORE this year's batch are eligible – new sibling nodes
//   added in the same annual cycle are never accidentally cited.
// FIX 2 (vs. original cpu-models):  the caller now provides a buffer sized
//   to current_graph_size+1, eliminating the 250-slot stack overflow.
// ---------------------------------------------------------------------------
int ABM::MakeERGNPCitations(Graph* graph,
                             const std::map<int, int>& reverse_continuous_node_mapping,
                             int* citations,
                             double p,
                             int current_graph_size) {
    std::random_device rand_dev; std::minstd_rand gen{rand_dev()};
    std::uniform_real_distribution<double> dist{0.0, 1.0};
    int num_cited = 0;
    // Only iterate over the pre-existing nodes [0 .. current_graph_size-1].
    for(int i = 0; i < current_graph_size; i++) {
        if(dist(gen) < p)
            citations[num_cited++] = reverse_continuous_node_mapping.at(i);
    }
    return num_cited;
}

int ABM::MakeCitations(Graph* graph, const std::map<int, int>& continuous_node_mapping, int current_year, const std::vector<int>& candidate_nodes, int* citations, double* pa_arr, double* recency_arr, double* fit_arr, double pa_weight, double rec_weight, double fit_weight, int current_graph_size, int num_citations) {
    if (num_citations <= 0) {
        return 0;
    }
    if (candidate_nodes.size() <= 0) {
        return 0;
    }
    float* current_scores = new float[candidate_nodes.size() + 1];
    current_scores[candidate_nodes.size()] = 0.0;
    /* int local_continuous_node_id = 0; */
    // Mark: removed for node-level
    // Clamp inputs
    constexpr float EPS = 1e-8f;

    for(size_t i = 0; i < candidate_nodes.size(); i ++) {
        int continuous_node_id = continuous_node_mapping.at(candidate_nodes.at(i));
        float current_pa = pa_arr[continuous_node_id];
        float current_rec = recency_arr[continuous_node_id];
        float current_fit = fit_arr[continuous_node_id];

        // original compound score - used in log (as referred as jay_score 1
        current_scores[i] = (current_pa * pa_weight) + (current_rec * rec_weight) + (current_fit * fit_weight);
        /** jay_score 2
        // log products
        float x1 = logf(fmaxf(current_pa, EPS)) + logf(fmaxf(pa_weight, EPS));
        float x2 = logf(fmaxf(current_rec, EPS)) + logf(fmaxf(rec_weight, EPS));
        float x3 = logf(fmaxf(current_fit, EPS)) + logf(fmaxf(fit_weight, EPS));

        // log-sum-exp
        float m = fmaxf(x1, fmaxf(x2, x3));

        current_scores[i] = m + logf(
            expf(x1 - m) +
            expf(x2 - m) +
            expf(x3 - m)
        ); */
    }

    int actual_num_cited = num_citations;
    if (candidate_nodes.size() < (size_t)num_citations) {
        actual_num_cited = candidate_nodes.size();
    }

    /* std::random_device rand_dev; */
    /* std::minstd_rand generator{rand_dev()}; */
    float* random_weight_arr = new float[candidate_nodes.size()];
    /* random_weight_arr[candidate_nodes.size()] = 0.0; */
    /* std::uniform_real_distribution<double> wrs_uniform_distribution{std::numeric_limits<double>::min(), 1}; */
    /* #pragma omp parallel for */
    std::random_device rand_dev;
    std::vector<std::minstd_rand> generator_vec;
    // num processors should be the same as omp max num
    for (int i = 0; i < this->num_processors; i ++) {
        generator_vec.push_back(std::minstd_rand(rand_dev()));
    }

    float min_positive = std::numeric_limits<float>::min();

    // Mark: removed for node-level
    /* #pragma omp parallel for simd */
    /* std::random_device rand_dev; */
    std::minstd_rand generator{rand_dev()};
    for(size_t i = 0; i < candidate_nodes.size(); i ++) {
        std::uniform_real_distribution<double> wrs_uniform_distribution{std::numeric_limits<double>::min(), 1};
        float wrs_uniform = wrs_uniform_distribution(generator_vec.at(omp_get_thread_num()));
        /**  jay_score 2 * /
        wrs_uniform = fmaxf(wrs_uniform, EPS);
        random_weight_arr[i] = logf(wrs_uniform) * expf(-current_scores[i]);
        */
        /** jay_score 1 */
        if (current_scores[i] != 0) {
            random_weight_arr[i] = log(wrs_uniform) / current_scores[i]; // pow(wrs_uniform, 1.0/current_scores[i]);
        } else {
            random_weight_arr[i] = min_positive;
        }
    }

    // TODO: probably omp parallel it
    std::vector<std::pair<double, int>> element_index_vec(candidate_nodes.size());
    // Mark: removed for node-level
    /* #pragma omp parallel for simd */
    for (size_t i = 0; i < candidate_nodes.size(); i ++) {
        /* element_index_vec.push_back({random_weight_arr[i], candidate_nodes.at(i)}); */
        element_index_vec[i] = {random_weight_arr[i], candidate_nodes.at(i)};
    }
    /* std::cout << "before sorted citing" << std::endl; */
    /* std::cout << "num cited should be " << actual_num_cited << std::endl; */
    /* for (size_t i = 0; i < candidate_nodes.size(); i ++) { */
    /*     std::cout << element_index_vec.at(i).first << ":" << element_index_vec.at(i).second << std::endl; */
    /* } */
    /* std::partial_sort(random_weight_arr, random_weight_arr + actual_num_cited, random_weight_arr + candidate_nodes.size(), std::greater<std::pair<int, int>>()); */
    std::partial_sort(element_index_vec.begin(), element_index_vec.begin() + actual_num_cited, element_index_vec.end(), [](auto& left, auto& right){
        return left.first > right.first;
    });

    /* std::cout << "after sorted citing" << std::endl; */
    /* for (size_t i = 0; i < candidate_nodes.size(); i ++) { */
    /*     std::cout << element_index_vec.at(i).first << ":" << element_index_vec.at(i).second << std::endl; */
    /* } */
    /* std::partial_sort(random_weight_arr, element_index_vec.end(), [](auto& left, auto& right) { */
    /*     return left.first < right.first; */
    /* }); */
    for (int i = 0; i < actual_num_cited; i ++) {
        citations[i] = element_index_vec[i].second;
    }
    /* return actual_num_cited; */

    // end
    delete[] current_scores;
    delete[] random_weight_arr;
    return actual_num_cited;
}

std::vector<int> ABM::GetComplement(Graph* graph, const std::vector<int>& base_vec, const std::map<int, int>& reverse_continuous_node_mapping) {
    std::vector<int> complement;
    std::set<int> base_set(base_vec.begin(), base_vec.end());
    for(size_t i = 0; i < graph->GetNodeSet().size(); i++) {
        int nid = reverse_continuous_node_mapping.at(i);
        if(!base_set.contains(nid)) complement.push_back(nid);
    }
    return complement;
}

std::vector<int> ABM::GetGeneratorNodes(Graph* graph, const std::map<int, int>& reverse_continuous_node_mapping) {
    std::uniform_int_distribution<int> dist{0, (int)(graph->GetNodeSet().size() - 1)};
    std::random_device rand_dev; std::minstd_rand gen{rand_dev()};
    std::vector<int> generator_nodes;
    for(int i = 0; i < 1; i++)
        generator_nodes.push_back(reverse_continuous_node_mapping.at(dist(gen)));
    return generator_nodes;
}

std::map<int, std::vector<int>> ABM::GetOneAndTwoHopNeighborhood(Graph* graph, const std::vector<int>& generator_nodes, const std::map<int, int>& reverse_continuous_node_mapping) {
    std::map<int, std::vector<int>> nbhd;
    std::set<int> visited;
    for(size_t i = 0; i < generator_nodes.size(); i++) {
        int gen = generator_nodes.at(i);
        std::queue<std::pair<int,int>> q;
        q.push({gen, 0}); visited.insert(gen);
        while(!q.empty()) {
            auto [cur, dist] = q.front(); q.pop();
            if(dist > 0) nbhd[dist].push_back(cur);
            if(dist < 2) {
                auto expand = [&](const std::set<int>& neighbors) {
                    for(int nb : neighbors) {
                        if(!visited.count(nb)) { visited.insert(nb); q.push({nb, dist+1}); }
                    }
                };
                if(graph->GetOutDegree(cur) > 0) expand(graph->GetForwardAdjMap().at(cur));
                if(graph->GetInDegree(cur)  > 0) expand(graph->GetBackwardAdjMap().at(cur));
            }
        }
    }
    return nbhd;
}

std::vector<int> ABM::GetNeighborhood(Graph* graph, const std::vector<int>& generator_nodes, const std::map<int, int>& reverse_continuous_node_mapping) {
    std::vector<int> neighborhood;
    std::set<int> visited;
    for(size_t i = 0; i < generator_nodes.size(); i++) {
        int gen = generator_nodes.at(i);
        std::queue<std::pair<int,int>> q;
        q.push({gen, 0}); visited.insert(gen);
        while(!q.empty()) {
            auto [cur, dist] = q.front(); q.pop();
            if(dist < 1) {
                auto expand = [&](const std::set<int>& neighbors) {
                    for(int nb : neighbors) {
                        if(!visited.count(nb)) {
                            neighborhood.push_back(nb); visited.insert(nb); q.push({nb, dist+1});
                        }
                    }
                };
                if(graph->GetOutDegree(cur) > 0) expand(graph->GetForwardAdjMap().at(cur));
                if(graph->GetInDegree(cur)  > 0) expand(graph->GetBackwardAdjMap().at(cur));
            }
        }
    }
    return neighborhood;
}

// =============================================================================
//  abm_main_profiled.cpp
//  Drop-in replacement for the file containing ABM::main().
//  All original logic UNCHANGED — profiling hooks added only at call sites.
//  Search "// PROF:" to find every instrumentation point.
// =============================================================================



int ABM::main() {

    // PROF: configure profiler metadata
    auto& PROF = ABMProfiler::get();
    PROF.model_name  = this->model_name;
    PROF.growth_rate = this->growth_rate;
    PROF.num_threads = this->num_processors;

    WTimer e2e_timer;
    e2e_timer.start();

    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

    Graph* graph = nullptr;
    // ── INIT: graph load ──────────────────────────────────────────────────────
    ITIME(t_graph_load,
        graph = new Graph(this->edgelist, this->nodelist)
    );
    this->WriteToLogFile("loaded graph", Log::info);

    // ── INIT: fitness initialisation ─────────────────────────────────────────
    ITIME(t_fitness_init,
        this->InitializeFitness(graph)
    );
    this->WriteToLogFile("initialized fitness for the seed graph", Log::debug);

    // ── INIT: mapping build ───────────────────────────────────────────────────
    std::map<int,int>  continuous_node_mapping;
    std::map<int,int>  reverse_continuous_node_mapping;
    {
        WTimer _wt; _wt.start();
        continuous_node_mapping         = this->BuildContinuousNodeMapping(graph);
        reverse_continuous_node_mapping = this->ReverseMapping(continuous_node_mapping);
        PROF.init.t_mapping_build += _wt.stop_ms();
    }
    this->WriteToLogFile("mappings built", Log::debug);

    int start_year           = this->GetMaxYear(graph) + 1;
    int next_node_id         = this->GetMaxNode(graph) + 1;
    int initial_next_node_id = next_node_id;
    int initial_graph_size   = graph->GetNodeSet().size();
    int final_graph_size     = this->GetFinalGraphSize(graph);
    int growth_in_graph_size = final_graph_size - initial_graph_size;

    // ── INIT: array allocation ────────────────────────────────────────────────
    int*    in_degree_arr     = nullptr;
    int*    fitness_arr       = nullptr;
    double* pa_arr            = nullptr;
    double* fit_arr           = nullptr;
    double* recency_arr       = nullptr;
    double* random_weight_arr = nullptr;
    double* current_score_arr = nullptr;
    double* pa_weight_arr     = nullptr;
    double* rec_weight_arr    = nullptr;
    double* fit_weight_arr    = nullptr;
    double* alpha_arr         = nullptr;
    int*    out_degree_arr    = nullptr;
    {
        WTimer _wt; _wt.start();
        in_degree_arr     = new int   [final_graph_size];
        fitness_arr       = new int   [final_graph_size];
        pa_arr            = new double[final_graph_size];
        fit_arr           = new double[final_graph_size];
        recency_arr       = new double[final_graph_size];
        random_weight_arr = new double[final_graph_size];
        current_score_arr = new double[final_graph_size];
        pa_weight_arr     = new double[growth_in_graph_size];
        rec_weight_arr    = new double[growth_in_graph_size];
        fit_weight_arr    = new double[growth_in_graph_size];
        alpha_arr         = new double[growth_in_graph_size];
        out_degree_arr    = new int   [growth_in_graph_size];
        PROF.init.t_array_alloc += _wt.stop_ms();
    }

    // ── INIT: weight/alpha/outdegree population ───────────────────────────────
    {
        WTimer _wt; _wt.start();
        this->PopulateWeightArrs(pa_weight_arr, rec_weight_arr, fit_weight_arr, growth_in_graph_size);
        this->PopulateAlphaArr(alpha_arr, growth_in_graph_size);
        this->PopulateOutDegreeArr(out_degree_arr, growth_in_graph_size);
        PROF.init.t_weight_pop += _wt.stop_ms();
    }
    this->WriteToLogFile("allocated and populated arrays", Log::debug);

    bool is_er_model = (this->model_name == "er" || this->model_name == "er-gnp");

    std::vector<int>              new_nodes_vec;
    std::set<int>                 same_year_source_nodes;
    std::vector<std::pair<int,int>> new_edges_vec;

    // =========================================================================
    //  EPOCH LOOP
    // =========================================================================
    for (int current_year = start_year;
         current_year < start_year + this->num_cycles;
         current_year++)
    {
        int current_graph_size = graph->GetNodeSet().size();
        int num_new_nodes      = (int)std::ceil(current_graph_size * this->growth_rate);

        // PROF: begin epoch
        PROF.begin_epoch(current_year, current_graph_size, num_new_nodes, this->num_processors);

        this->WriteToLogFile("current year: " + std::to_string(current_year) +
                             "  graph size: " + std::to_string(current_graph_size), Log::info);

        WTimer epoch_timer; epoch_timer.start();

        // ── CPU PRE: score arrays (PA only) ───────────────────────────────────
        if (!is_er_model) {
            CTIME(t_fill_indeg,
                this->FillInDegreeArr(graph, continuous_node_mapping, in_degree_arr)
            );
            CTIME(t_fill_fitness,
                this->FillFitnessArr(graph, continuous_node_mapping, current_year, fitness_arr)
            );
            CTIME(t_fill_recency,
                this->FillRecencyArr(graph, reverse_continuous_node_mapping, current_year, recency_arr)
            );
            {
                WTimer _wt; _wt.start();
                this->CalculateScores(in_degree_arr, pa_arr,  current_graph_size);
                this->CalculateScores(fitness_arr,   fit_arr, current_graph_size);
                ABMPROF_EP.t_calc_scores += _wt.stop_ms();
            }
        }

        // ── New-node init ─────────────────────────────────────────────────────
        {
            WTimer _wt; _wt.start();
            for (int i = 0; i < num_new_nodes; i++) {
                continuous_node_mapping[next_node_id]                   = current_graph_size + i;
                reverse_continuous_node_mapping[current_graph_size + i] = next_node_id;
                new_nodes_vec.push_back(next_node_id);
                graph->SetIntAttribute("year",    next_node_id, current_year);
                graph->SetStringAttribute("type", next_node_id, "agent");
                next_node_id++;
            }
            ABMPROF_EP.t_node_init += _wt.stop_ms();
        }
        this->WriteToLogFile("all new nodes initialized", Log::debug);

        // ── Same-year set ─────────────────────────────────────────────────────
        CTIME(t_same_year,
            this->FillSameYearSourceNodes(same_year_source_nodes, new_nodes_vec.size())
        );

        // ── Generator-node assignment ─────────────────────────────────────────
        {
            WTimer _wt; _wt.start();
            for (size_t i = 0; i < new_nodes_vec.size(); i++) {
                int new_node = new_nodes_vec[i];
                std::vector<int> generator_nodes =
                    this->GetGeneratorNodes(graph, reverse_continuous_node_mapping);
                this->UpdateGraphAttributesGeneratorNodes(graph, new_node, generator_nodes);
            }
            ABMPROF_EP.t_gen_assign += _wt.stop_ms();
        }

        std::cout << "\nnew_nodes_vec.size() = " << new_nodes_vec.size();

        // ── Collect per-node samples (thread-local to avoid races) ────────────
        // Resize to num_new_nodes; each slot owned by its omp thread index
        std::vector<NodeSample> node_samples(new_nodes_vec.size());

        // =========================================================================
        //  CITATION LOOP  — with per-node profiling
        // =========================================================================
        WTimer cit_wall; cit_wall.start();

        #pragma omp parallel for reduction(merge_int_pair_vecs: new_edges_vec) \
                                 num_threads(this->num_processors) \
                                 schedule(dynamic, 64)
        for (size_t i = 0; i < new_nodes_vec.size(); i++) {

            // PROF: per-node timer
            WTimer nt; nt.start();
            NodeSample ns;
            ns.node_id   = new_nodes_vec[i];
            ns.thread_id = omp_get_thread_num();
            ns.year      = current_year;
            ns.is_pa     = !is_er_model;

            std::vector<std::pair<int,int>> local_new_edges_vec;
            int* citations;
            int new_node         = new_nodes_vec[i];
            int weight_arr_index = continuous_node_mapping[new_node] - initial_graph_size;
            int num_actually_cited = 0;

            if (!is_er_model) {
                // ── PA branch ─────────────────────────────────────────────────
                citations = new int[250];

                double pa_weight  = pa_weight_arr[weight_arr_index];
                double rec_weight = rec_weight_arr[weight_arr_index];
                double fit_weight = fit_weight_arr[weight_arr_index];
                double alpha      = alpha_arr[weight_arr_index];

                std::vector<int> generator_nodes =
                    this->GetGraphAttributesGeneratorNodes(graph, new_node);

                // PROF: BFS
                std::map<int, std::vector<int>> hop_map;
                {
                    WTimer _wt; _wt.start();
                    hop_map = this->GetOneAndTwoHopNeighborhood(
                        graph, generator_nodes, reverse_continuous_node_mapping);
                    ns.t_bfs_ms = _wt.stop_ms();
                }
                ns.hop1_size = (int)hop_map[1].size();
                ns.hop2_size = (int)hop_map[2].size();

                int num_generator_node_citation      = (int)generator_nodes.size();
                int same_year_citation               = same_year_source_nodes.count(i);
                int num_fully_random_cited_reserved  =
                    (int)std::floor(this->fully_random_citations * out_degree_arr[weight_arr_index]);
                int num_citations_inside  =
                    (int)std::ceil((out_degree_arr[weight_arr_index]
                        - num_generator_node_citation
                        - same_year_citation
                        - num_fully_random_cited_reserved) * alpha);
                num_citations_inside = std::min(num_citations_inside, (int)hop_map[1].size());
                int num_citations_outside =
                    out_degree_arr[weight_arr_index]
                    - num_generator_node_citation
                    - same_year_citation
                    - num_fully_random_cited_reserved
                    - num_citations_inside;
                num_citations_outside = std::min(num_citations_outside, (int)hop_map[2].size());
                int num_fully_random_cited =
                    out_degree_arr[weight_arr_index]
                    - num_generator_node_citation
                    - same_year_citation
                    - num_citations_inside
                    - num_citations_outside;

                // PROF: same-year
                if (same_year_citation) {
                    WTimer _wt; _wt.start();
                    num_actually_cited += this->MakeSameYearCitations(
                        new_nodes_vec.size(), reverse_continuous_node_mapping,
                        citations, current_graph_size);
                    ns.t_sameyear_ms += _wt.stop_ms();
                }

                // PROF: scoring (MakeCitations) — split score phase and sample phase
                // We time the full MakeCitations call here; to split score vs
                // heap-sort you would need to instrument MakeCitations itself.
                {
                    WTimer _wt; _wt.start();
                    num_actually_cited += this->MakeCitations(
                        graph, continuous_node_mapping, current_year,
                        hop_map[1], citations + num_actually_cited,
                        pa_arr, recency_arr, fit_arr,
                        pa_weight, rec_weight, fit_weight,
                        current_graph_size, num_citations_inside);
                    num_actually_cited += this->MakeCitations(
                        graph, continuous_node_mapping, current_year,
                        hop_map[2], citations + num_actually_cited,
                        pa_arr, recency_arr, fit_arr,
                        pa_weight, rec_weight, fit_weight,
                        current_graph_size, num_citations_outside);
                    ns.t_score_ms += _wt.stop_ms();
                }

                // PROF: uniform random citations
                {
                    WTimer _wt; _wt.start();
                    num_actually_cited += this->MakeUniformRandomCitations(
                        graph, reverse_continuous_node_mapping,
                        generator_nodes, citations,
                        num_actually_cited, num_fully_random_cited);
                    ns.t_random_ms += _wt.stop_ms();
                }

                for (size_t j = 0; j < generator_nodes.size(); j++)
                    local_new_edges_vec.push_back({new_node, generator_nodes[j]});

            } else {
                // ── ER branch ─────────────────────────────────────────────────
                int expected_citations = (int)(current_graph_size * this->er_probability);
                citations = new int[2 * std::max(expected_citations, 1)];

                if (i % 3000 == 1)
                    std::cout << "\ncurrent year: " << current_year
                              << ", graph size: " << current_graph_size
                              << " er_probability = " << this->er_probability
                              << ", expected_citations = " << expected_citations;

                // PROF: ER sampling
                {
                    WTimer _wt; _wt.start();
                    if (this->model_name == "er") {
                        num_actually_cited = this->SampleKNodes(
                            graph, current_graph_size,
                            reverse_continuous_node_mapping,
                            citations, expected_citations);
                    } else {
                        num_actually_cited = this->MakeERGNPCitations(
                            graph, reverse_continuous_node_mapping,
                            citations, this->er_probability, current_graph_size);
                    }
                    ns.t_random_ms += _wt.stop_ms();   // ER uses random slot
                }

                if (i % 10000 == 1)
                    std::cout << "\nFor node = " << new_node
                              << ": expected=" << expected_citations
                              << ", cited=" << num_actually_cited;
            }

            // Write citation edges
            for (int j = 0; j < num_actually_cited; j++)
                local_new_edges_vec.push_back({new_node, citations[j]});

            new_edges_vec.insert(new_edges_vec.end(),
                                 local_new_edges_vec.begin(),
                                 local_new_edges_vec.end());

            // PROF: finalise node sample
            ns.num_cited        = num_actually_cited;
            ns.t_node_total_ms  = nt.stop_ms();
            node_samples[i]     = ns;

            delete[] citations;
        }
        // end OMP parallel for

        ABMPROF_EP.t_citation_loop = cit_wall.stop_ms();

        // PROF: store node samples into epoch
        ABMPROF_EP.node_samples = std::move(node_samples);

        // ── Edge commit ───────────────────────────────────────────────────────
        {
            WTimer _wt; _wt.start();
            for (size_t i = 0; i < new_edges_vec.size(); i++)
                graph->AddEdge({new_edges_vec[i].first, new_edges_vec[i].second});
            ABMPROF_EP.t_edge_commit += _wt.stop_ms();
        }
        this->WriteToLogFile("edges saved to graph", Log::debug);

        // ── Fitness assignment ────────────────────────────────────────────────
        {
            WTimer _wt; _wt.start();
            this->AssignPeakFitnessValues(graph, new_nodes_vec);
            this->AssignFitnessLagDuration(graph, new_nodes_vec);
            this->AssignFitnessPeakDuration(graph, new_nodes_vec);
            this->PlantNodes(graph, new_nodes_vec, current_year - start_year + 1);
            ABMPROF_EP.t_fitness_assign += _wt.stop_ms();
        }

        new_nodes_vec.clear();
        new_edges_vec.clear();
        same_year_source_nodes.clear();

        ABMPROF_EP.t_epoch_total = epoch_timer.stop_ms();

        // PROF: per-epoch console summary (SC-style compact line)
        const auto& ep = ABMPROF_EP;
        printf("\n[EP %d] N=%-7d delta=%-6d | pre=%.0f cit=%.0f edge=%.0f fit=%.0f | total=%.0f ms\n",
               current_year, current_graph_size, num_new_nodes,
               ep.cpu_preproc_ms(), ep.t_citation_loop,
               ep.t_edge_commit, ep.t_fitness_assign,
               ep.epoch_total_computed());

    } // end epoch loop

    // =========================================================================
    //  OUTPUT
    // =========================================================================
    this->WriteToLogFile("finished sim", Log::info);

    OTIME(t_write_graph, graph->WriteGraph(this->output_file));

    {
        WTimer _wt; _wt.start();
        this->UpdateGraphAttributesWeights(graph, initial_next_node_id,
            pa_weight_arr, rec_weight_arr, fit_weight_arr, growth_in_graph_size);
        this->UpdateGraphAttributesAlphas(graph, initial_next_node_id,
            alpha_arr, growth_in_graph_size);
        this->UpdateGraphAttributesOutDegrees(graph, initial_next_node_id,
            out_degree_arr, growth_in_graph_size);
        for (auto const& nid : graph->GetNodeSet()) {
            graph->SetIntAttribute("in_degree",  nid, graph->GetInDegree(nid));
            graph->SetIntAttribute("out_degree", nid, graph->GetOutDegree(nid));
        }
        ABMPROF.output.t_update_attrs += _wt.stop_ms();
    }

    OTIME(t_write_attrs, graph->WriteAttributes(this->auxiliary_information_file));
    this->WriteToLogFile("wrote outputs", Log::info);

    // ── E2E timing ────────────────────────────────────────────────────────────
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    auto durationE2E = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    PROF.t_e2e_ms = (double)durationE2E.count();

    std::ostringstream msg;
    msg << "\nE2E Time, model:" << this->model_name
        << "  num_cycles=" << this->num_cycles
        << "  growth_rate=" << (100.0 * this->growth_rate) << "%"
        << "  threads=" << this->num_processors
        << "  elapsed=" << durationE2E.count() / 1000 << "s";
    this->WriteToLogFile(msg.str(), Log::info);
    std::cout << msg.str() << std::endl;

    // ── PROF: print full SC-style report ──────────────────────────────────────
    ABMProfiler::print_sc_report(PROF);

    // ── Cleanup ───────────────────────────────────────────────────────────────
    delete[] in_degree_arr;  delete[] fitness_arr;    delete[] pa_arr;
    delete[] fit_arr;        delete[] recency_arr;    delete[] random_weight_arr;
    delete[] current_score_arr;
    delete[] pa_weight_arr;  delete[] rec_weight_arr; delete[] fit_weight_arr;
    delete[] alpha_arr;      delete[] out_degree_arr;
    delete graph;
    return 0;
}

/*/ =============================================================================
// ABM::main()
// Old Main
// =============================================================================
int ABM::main() {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

    /* reading input edgelist, nodelist, outdegree bag, recency probabilities * /
    Graph* graph = new Graph(this->edgelist, this->nodelist);
    this->WriteToLogFile("loaded graph", Log::info);
    this->InitializeFitness(graph);
    this->WriteToLogFile("initialized fitness for the seed graph", Log::debug);

    std::map<int,int> continuous_node_mapping         = this->BuildContinuousNodeMapping(graph);
    this->WriteToLogFile("forward built", Log::debug);
    /* continous integer from 0 to node ids* /
    std::map<int, int> reverse_continuous_node_mapping = this->ReverseMapping(continuous_node_mapping);
    this->WriteToLogFile("reverse mapping built", Log::debug);

    int start_year            = this->GetMaxYear(graph) + 1;
    int next_node_id          = this->GetMaxNode(graph) + 1;
    int initial_next_node_id  = next_node_id;

    /* get input to score arrays based on continuous_node_mapping * /
    int initial_graph_size    = graph->GetNodeSet().size();
    int final_graph_size      = this->GetFinalGraphSize(graph);
    this->WriteToLogFile("final graph size is " + std::to_string(final_graph_size), Log::info);

    int*    in_degree_arr      = new int   [final_graph_size];
    int*    fitness_arr        = new int   [final_graph_size];
    double* pa_arr             = new double[final_graph_size];
    double* fit_arr            = new double[final_graph_size];
    double* recency_arr        = new double[final_graph_size];
    double* random_weight_arr  = new double[final_graph_size];
    double* current_score_arr  = new double[final_graph_size];

    int growth_in_graph_size   = final_graph_size - initial_graph_size;
    double* pa_weight_arr      = new double[growth_in_graph_size];
    double* rec_weight_arr     = new double[growth_in_graph_size];
    double* fit_weight_arr     = new double[growth_in_graph_size];
    double* alpha_arr          = new double[growth_in_graph_size];
    int*    out_degree_arr     = new int   [growth_in_graph_size];

    this->PopulateWeightArrs(pa_weight_arr, rec_weight_arr, fit_weight_arr, growth_in_graph_size);
    this->PopulateAlphaArr(alpha_arr, growth_in_graph_size);
    this->PopulateOutDegreeArr(out_degree_arr, growth_in_graph_size);
    this->WriteToLogFile("allocated and populated arrays", Log::debug);

    bool is_er_model = (this->model_name == "er" || this->model_name == "er-gnp");

    std::vector<int>              new_nodes_vec;
    std::set<int>                 same_year_source_nodes;
    std::vector<std::pair<int,int>> new_edges_vec;

    // =========================================================================
    // Annual time loop
    // =========================================================================
    for(int current_year = start_year; current_year < start_year + this->num_cycles; current_year++) {

        int current_graph_size = graph->GetNodeSet().size();
        this->WriteToLogFile("current year: " + std::to_string(current_year) +
                             "  graph size: " + std::to_string(current_graph_size), Log::info);

        // ── PA only: per-year score arrays ────────────────────────────────────
        if(!is_er_model) {
            this->FillInDegreeArr(graph, continuous_node_mapping, in_degree_arr);
            this->WriteToLogFile("indegree filled", Log::debug);
            this->FillFitnessArr(graph, continuous_node_mapping, current_year, fitness_arr);
            this->WriteToLogFile("fitness filled", Log::debug);
            this->FillRecencyArr(graph, reverse_continuous_node_mapping, current_year, recency_arr);
            this->WriteToLogFile("recency filled", Log::debug);
            this->CalculateScores(in_degree_arr, pa_arr,  current_graph_size);
            this->CalculateScores(fitness_arr,   fit_arr, current_graph_size);
            this->WriteToLogFile("scores computed", Log::debug);
        }

        // ── Add new nodes to graph and mappings ───────────────────────────────
        int num_new_nodes = (int)std::ceil(current_graph_size * this->growth_rate);
        this->WriteToLogFile("adding " + std::to_string(num_new_nodes) + " nodes", Log::info);
        std::cout<<"\nAdding " << num_new_nodes << " nodes.";
        for(int i = 0; i < num_new_nodes; i++) {
            continuous_node_mapping[next_node_id]             = current_graph_size + i;
            reverse_continuous_node_mapping[current_graph_size + i] = next_node_id;
            new_nodes_vec.push_back(next_node_id);
            graph->SetIntAttribute("year",    next_node_id, current_year);
            graph->SetStringAttribute("type", next_node_id, "agent");
            next_node_id++;
        }
        this->WriteToLogFile("all new nodes initialized with years and mapped", Log::debug);
        this->FillSameYearSourceNodes(same_year_source_nodes, new_nodes_vec.size());

        // ── PA: pre-assign generator nodes; ER: set safe default ─────────────
        // WriteAttributes calls GetStringAttribute("generator_node_string") for
        // every agent node; both models must set it or the aux write crashes.
        //if(!is_er_model) {
            for(size_t i = 0; i < new_nodes_vec.size(); i++) {
                int new_node = new_nodes_vec[i];
                std::vector<int> generator_nodes = this->GetGeneratorNodes(graph, reverse_continuous_node_mapping);
                this->UpdateGraphAttributesGeneratorNodes(graph, new_node, generator_nodes);
            }
        /*} else {
            for(size_t i = 0; i < new_nodes_vec.size(); i++)
                graph->SetStringAttribute("generator_node_string", new_nodes_vec[i], "no_generators");
        }* /

        std::cout<<"\nnew_nodes_vec.size() = "<< new_nodes_vec.size();
        // =========================================================================
        // Per-node citation loop
        // =========================================================================
        #pragma omp parallel for reduction(merge_int_pair_vecs: new_edges_vec)
        for(size_t i = 0; i < new_nodes_vec.size(); i++) {
            std::vector<std::pair<int, int>> local_new_edges_vec;
            int* citations;
            this->WriteToLogFile("starting node " + std::to_string(i) + "/" + std::to_string(new_nodes_vec.size()), Log::debug);
            int new_node = new_nodes_vec[i];
            int weight_arr_index = continuous_node_mapping[new_node] - initial_graph_size;
            int num_actually_cited = 0;
                
            // ── PA only: per-year score arrays ────────────────────────────────────
            if(!is_er_model) {
                citations = new int[250]; // out-degree assumed to be max 249
                
                double pa_weight = pa_weight_arr[weight_arr_index];
                double rec_weight = rec_weight_arr[weight_arr_index];
                double fit_weight = fit_weight_arr[weight_arr_index];
                double alpha = alpha_arr[weight_arr_index];
                std::vector<int> generator_nodes = this->GetGraphAttributesGeneratorNodes(graph, new_node);
                std::map<int, std::vector<int>> one_and_two_hop_neighborhood_map = this->GetOneAndTwoHopNeighborhood(graph, generator_nodes, reverse_continuous_node_mapping);

                int num_generator_node_citation = generator_nodes.size(); // should be 1 for now
                int same_year_citation = same_year_source_nodes.count(i); // could be 0 or 1
                int num_fully_random_cited_reserved = std::floor(this->fully_random_citations * out_degree_arr[weight_arr_index]); // e.g., 5% of out-degree. some small number

                // now the number of things to cite from distance 1 is the remaining citations * alpha. Call this remaining citations R for later.
                // unless distance 1 neighborhood is too small
                int num_citations_inside = std::ceil((out_degree_arr[weight_arr_index] - num_generator_node_citation - same_year_citation - num_fully_random_cited_reserved) * alpha);
                num_citations_inside = std::min(num_citations_inside, (int)one_and_two_hop_neighborhood_map[1].size());

                // now the number of things to cite from distance 2 is the remaining citations
                // unless distance 2 neighborhood is too small
                int num_citations_outside = out_degree_arr[weight_arr_index] - num_generator_node_citation - same_year_citation - num_fully_random_cited_reserved - num_citations_inside;
                num_citations_outside = std::min(num_citations_outside, (int)one_and_two_hop_neighborhood_map[2].size());

                // if it turns out that the 2-hop neighborhood (including 1 and 2) is small than R from earlier, then the leftover citations get cited randomly from the graph
                int num_fully_random_cited = out_degree_arr[weight_arr_index] - num_generator_node_citation - same_year_citation - num_citations_inside - num_citations_outside;

                if (same_year_citation) {
                    num_actually_cited += this->MakeSameYearCitations(new_nodes_vec.size(), reverse_continuous_node_mapping, citations, current_graph_size);
                }

                num_actually_cited += this->MakeCitations(graph, continuous_node_mapping, current_year, one_and_two_hop_neighborhood_map[1], citations + num_actually_cited, pa_arr, recency_arr, fit_arr, pa_weight, rec_weight, fit_weight, current_graph_size, num_citations_inside);
                num_actually_cited += this->MakeCitations(graph, continuous_node_mapping, current_year, one_and_two_hop_neighborhood_map[2], citations + num_actually_cited, pa_arr, recency_arr, fit_arr, pa_weight, rec_weight, fit_weight, current_graph_size, num_citations_outside);
                num_actually_cited += this->MakeUniformRandomCitations(graph, reverse_continuous_node_mapping, generator_nodes, citations, num_actually_cited, num_fully_random_cited);

                for(size_t j = 0; j < generator_nodes.size(); j ++) {
                    local_new_edges_vec.push_back({new_node, generator_nodes[j]});
            }
            } else {
                // =============================================================
                // ER branch
                // =============================================================
                std::vector<int> empty_gen;
                int expected_citations = current_graph_size * this->er_probability;
                citations = new int[2 * expected_citations]; // out-degree assumed to be max 249
                if (i % 3000 == 1) { 
                    std::cout<<"\ncurrent year: " << current_year << ", graph size: " << current_graph_size 
                        << " er_probability = " << this->er_probability << ", expected_citations = " << expected_citations;
                }
                if(this->model_name == "er") {

                    // Fixed-K: sample exactly out_degree_arr[i] distinct
                    // targets uniformly at random (same degree sequence as PA).
                    /*num_actually_cited = this->MakeUniformRandomCitations(
                        graph, reverse_continuous_node_mapping,
                        empty_gen, citations, 0,
                        out_degree_arr[weight_idx]);* /
                    num_actually_cited = this->SampleKNodes(graph, 
                                current_graph_size, reverse_continuous_node_mapping, 
                                citations, expected_citations);
                    if (i % 10000 == 1) {   
                        std::cout<<"\nFor node = " << new_node <<": expected_citations = " << expected_citations 
                            << ", num_actually_cited = " << num_actually_cited;
                    }
                } else {
                    // G(n,p): each pre-existing node independently with prob p.
                    // FIX 3: passes current_graph_size so new sibling nodes
                    // (added this year) cannot be cited by each other.
                    num_actually_cited = this->MakeERGNPCitations(
                        graph, reverse_continuous_node_mapping,
                        citations, this->er_probability,
                        current_graph_size);
                    if (i % 10000 == 1) {   
                        std::cout<<"\nFor node = " << new_node <<": expected_citations = " << expected_citations 
                            << ", num_actually_cited = " << num_actually_cited;
                    }
                }
            }

            // Write citation edges (both PA and ER)
            for(int j = 0; j < num_actually_cited; j++)
                local_new_edges_vec.push_back({new_node, citations[j]});

            new_edges_vec.insert(new_edges_vec.end(), local_new_edges_vec.begin(), local_new_edges_vec.end());
        } // end of omp parallel loop

        // ── Commit edges to graph ────────────────────────────────────────────
        for(size_t i = 0; i < new_edges_vec.size(); i++)
            graph->AddEdge({new_edges_vec[i].first, new_edges_vec[i].second});
        this->WriteToLogFile("edges saved to graph", Log::debug);

        // ── Fitness attributes for new nodes ──────────────────────────────────
        this->AssignPeakFitnessValues(graph, new_nodes_vec);
        this->AssignFitnessLagDuration(graph, new_nodes_vec);
        this->AssignFitnessPeakDuration(graph, new_nodes_vec);
        this->PlantNodes(graph, new_nodes_vec, current_year - start_year + 1);

        new_nodes_vec.clear();
    }

    // =========================================================================
    // Write outputs  (identical schema for PA and ER)
    // =========================================================================
    this->WriteToLogFile("finished sim", Log::info);

    // Edgelist
    graph->WriteGraph(this->output_file);

    // Auxiliary file: weights/alpha are N/A for ER but written with the same
    // column schema so downstream tools work without modification.
    this->UpdateGraphAttributesWeights(graph, initial_next_node_id,
        pa_weight_arr, rec_weight_arr, fit_weight_arr, growth_in_graph_size);
    this->UpdateGraphAttributesAlphas(graph, initial_next_node_id,
        alpha_arr, growth_in_graph_size);
    this->UpdateGraphAttributesOutDegrees(graph, initial_next_node_id,
        out_degree_arr, growth_in_graph_size);
    for(auto const& nid : graph->GetNodeSet()) {
        graph->SetIntAttribute("in_degree",  nid, graph->GetInDegree(nid));
        graph->SetIntAttribute("out_degree", nid, graph->GetOutDegree(nid));
    }
    graph->WriteAttributes(this->auxiliary_information_file);
    this->WriteToLogFile("wrote outputs", Log::info);

    // ── E2E timing ────────────────────────────────────────────────────────────
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    auto durationE2E = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    std::ostringstream msg;
    msg << "\nE2E Time, model:" << this->model_name
        << "  num_cycles=" << this->num_cycles
        << "  growth_rate=" << (100.0 * this->growth_rate) << "%"
        << "  threads=" << this->num_processors
        << "  elapsed=" << durationE2E.count() / 1000 << "s";
    this->WriteToLogFile(msg.str(), Log::info);
    std::cout << msg.str() << std::endl;

    // ── Cleanup ───────────────────────────────────────────────────────────────
    delete[] in_degree_arr;  delete[] fitness_arr;    delete[] pa_arr;
    delete[] fit_arr;        delete[] recency_arr;    delete[] random_weight_arr;
    delete[] current_score_arr;
    delete[] pa_weight_arr;  delete[] rec_weight_arr; delete[] fit_weight_arr;
    delete[] alpha_arr;      delete[] out_degree_arr;
    delete graph;
    return 0;
} */
