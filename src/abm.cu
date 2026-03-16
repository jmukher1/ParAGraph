#include "abm.cuh"
#include "device_string.cuh"


int ABM::WriteToLogFile(std::string message, Log message_type) {
    if(this->log_level >= message_type) {
        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
        std::string log_message_prefix;
        if(message_type == Log::info) {
            log_message_prefix = "[INFO]";
        } else if(message_type == Log::debug) {
            log_message_prefix = "[DEBUG]";
        } else if(message_type == Log::error) {
            log_message_prefix = "[ERROR]";
        }
        auto days_elapsed = std::chrono::duration_cast<std::chrono::days>(now - this->start_time);
        auto hours_elapsed = std::chrono::duration_cast<std::chrono::hours>(now - this->start_time - days_elapsed);
        auto minutes_elapsed = std::chrono::duration_cast<std::chrono::minutes>(now - this->start_time - days_elapsed - hours_elapsed);
        auto seconds_elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - this->start_time - days_elapsed - hours_elapsed - minutes_elapsed);
        auto total_seconds_elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - this->start_time);
        log_message_prefix += "[";
        log_message_prefix += std::to_string(days_elapsed.count());
        log_message_prefix += "-";
        log_message_prefix += std::to_string(hours_elapsed.count());
        log_message_prefix += ":";
        log_message_prefix += std::to_string(minutes_elapsed.count());
        log_message_prefix += ":";
        log_message_prefix += std::to_string(seconds_elapsed.count());
        log_message_prefix += "]";

        log_message_prefix += "(t=";
        log_message_prefix += std::to_string(total_seconds_elapsed.count());
        log_message_prefix += "s)";
        this->log_file_handle << log_message_prefix << " " << message << '\n';

        if(this->num_calls_to_log_write % 1 == 0) {
            std::flush(this->log_file_handle);
        }
        this->num_calls_to_log_write ++;
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
        while(std::getline(ss, current_value, delimiter)) {
            current_line.push_back(current_value);
        }
        std::string year = current_line[0];
        std::string fitness_lag_duration = current_line[1];
        std::string fitness_peak_value = current_line[2];
        std::string fitness_peak_duration = current_line[3];
        std::string count = current_line[4];
        this->planted_nodes_map[std::stoi(year)][line_no]["fitness_lag_duration"] = std::stoi(fitness_lag_duration);
        this->planted_nodes_map[std::stoi(year)][line_no]["fitness_peak_value"] = std::stoi(fitness_peak_value);
        this->planted_nodes_map[std::stoi(year)][line_no]["fitness_peak_duration"] = std::stoi(fitness_peak_duration);
        this->planted_nodes_map[std::stoi(year)][line_no]["count"] = std::stoi(count);
        line_no += 1;
        /* std::cout << "adding " << current_line[1] << " to  bag " << std::endl; */
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
        while(std::getline(ss, current_value, delimiter)) {
            current_line.push_back(current_value);
        }
        std::string index = current_line[0];
        if(index[0] == '#') {
            continue;
        }
        this->out_degree_bag_vec.push_back(std::stoi(current_line[1]));
        //std::cout << "adding " << current_line[1] << " to bag " << std::endl;
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
        while(std::getline(ss, current_value, delimiter)) {
            current_line.push_back(current_value);
        }
        std::string year = current_line[0];
        if(year[0] == '#') {
            continue;
        }
        int integer_year_diff = std::stoi(current_line[0]);
        double probability = std::stod(current_line[1]);
        this->recency_probabilities_map[integer_year_diff] = probability;
    }
}

void ABM::FillInDegreeArr(Graph* graph, int* in_degree_arr) {
    for(auto const& node_seq_id : graph->GetNodeSet()) {
        in_degree_arr[node_seq_id] = graph->GetIntAttribute("in_degree", node_seq_id); //graph->GetInDegree(node_seq_id);
    }
}

void ABM::InitializeFitness(Graph* graph) {
    //this->AssignPeakFitnessValues(graph, graph->GetNodeSet());
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
    
    //this->AssignFitnessLagDuration(graph, graph->GetNodeSet());
    //this->AssignFitnessPeakDuration(graph, graph->GetNodeSet());

    int fitness_lag_uniform = 0; // MARK: hard coded to be static fitness
    int fitness_peak_uniform = 1000; // MARK: hard coded to be static fitness

    const std::set<int>& nodeSet = graph->GetNodeSet();
    for(int nodeSeqId : nodeSet) { 
        int current_fitness = int_discrete_distribution(generator) + 1;
        graph->SetIntAttribute("fitness_peak_value", nodeSeqId, current_fitness);
        graph->SetIntAttribute("fitness_lag_duration", nodeSeqId, fitness_lag_uniform); // <- default value anyway
        graph->SetIntAttribute("fitness_peak_duration", nodeSeqId, fitness_peak_uniform); // <- default value anyway
    }
}

void ABM::FillFitnessArr(Graph* graph, /*const std::map<int, int>& continuous_node_mapping,*/ int current_year, int* fitness_arr) {
    for(auto const& node_seq_id : graph->GetNodeSet()) {
        //if (continuous_node_mapping.find(node) != continuous_node_mapping.end()) {
        int fitness_peak_value = graph->GetIntAttribute("fitness_peak_value", node_seq_id);
        int fitness_lag_duration = graph->GetIntAttribute("fitness_lag_duration", node_seq_id);
        int fitness_peak_duration = graph->GetIntAttribute("fitness_peak_duration", node_seq_id);
        int published_year = graph->GetIntAttribute("year", node_seq_id);
        if (published_year + fitness_lag_duration > current_year) {
            fitness_arr[node_seq_id] = 1;
        } else if (published_year + fitness_lag_duration + fitness_peak_duration >= current_year) {
            fitness_arr[node_seq_id] = fitness_peak_value;
        } else {
            double decayed_fitness_value = fitness_peak_value / pow(current_year - published_year - fitness_lag_duration - fitness_peak_duration + 1, this->fitness_decay_alpha);
            fitness_arr[node_seq_id] = decayed_fitness_value;
        }  
    }
}


void ABM::FillRecencyArr(Graph* graph, /*const std::map<int, int>& reverse_continuous_node_mapping,*/ int current_year, double* recency_arr) {
    std::map<int, int> year_count;
    double unique_year_sum = 0.0;
    for(auto const& node_seq_id : graph->GetNodeSet()) {
        int current_published_year = graph->GetIntAttribute("year", node_seq_id);
        int year_diff = current_year - current_published_year;
        if(!year_count.contains(year_diff)) {
            unique_year_sum += this->recency_probabilities_map[year_diff];
        }
        year_count[year_diff] ++;
    }
    // Mark: removed for node-level
    /* #pragma omp parallel for simd */
    //for(size_t i = 0; i < graph->GetNodeSet().size(); i ++) {
    for(auto const& node_seq_id : graph->GetNodeSet()) {
        int current_published_year = graph->GetIntAttribute("year", node_seq_id);
        int year_diff = current_year - current_published_year;
        recency_arr[node_seq_id] = (float)this->recency_probabilities_map[year_diff] / year_count[year_diff];
        if (DATA_DEBUG && node_seq_id > 491000) {
            printf("\nFillRecencyArr::For node_seq_id = %d, node_id = %d, current_year %d - current_published_year %d = year_diff %d , recency_arr[node_seq_id] = %.15lf, year_count[year_diff] = %d",
                node_seq_id, graph->reverse_continuous_node_mapping[node_seq_id], current_year, current_published_year, year_diff, recency_arr[node_seq_id], year_count[year_diff]);
        }
    }

    //printf("\nunique_year_sum = %lf",  unique_year_sum);
    // Mark: removed for node-level
    /* #pragma omp parallel for simd */
    for(auto const& node_seq_id : graph->GetNodeSet()) {
        recency_arr[node_seq_id] /= unique_year_sum;
    }
}


int ABM::GetMaxYear(Graph* graph) {
    int max_year = -1;
    //bool is_first = true;
    for(auto const& node_seq_id : graph->GetNodeSet()) {
        int current_node_year = graph->GetIntAttribute("year", node_seq_id);
        /*if (is_first) {
            max_year = current_node_year;
            is_first = false;
        }*/
        if (current_node_year > max_year) {
            max_year = current_node_year;
        }
    }
    return max_year;
}

int ABM::GetMaxNode(Graph* graph) {
    int maxNodeId = graph->getContinuousNodeMapping().rbegin()->first;
    // printf("\n**Max Node Id = %d\n", maxNodeId);
    return maxNodeId;
}

int ABM::GetFinalGraphSize(Graph* graph) {
    int current_graph_size = graph->GetNodeSet().size();
    for(int i = 0; i < this->num_cycles; i ++) {
        int num_new_nodes = std::ceil(current_graph_size * this->growth_rate);
        current_graph_size += num_new_nodes;
    }
    return current_graph_size;
}
void ABM::PopulateAlphaArr(double* alpha_arr, int len) {
    std::random_device rand_dev;
    std::minstd_rand generator{rand_dev()};
    if(this->alpha < 0) {
        for(int i = 0; i < len; i ++) {
            double alpha_uniform = this->alpha_uniform_distribution(generator);
            alpha_arr[i] = alpha_uniform;
        }
    } else {
        for(int i = 0; i < len; i ++) {
            alpha_arr[i] = this->alpha;
        }
    }
}

void ABM::PopulateWeightArrs(double* pa_weight_arr, double* rec_weight_arr, double* fit_weight_arr, int len) {
    std::random_device rand_dev;
    std::minstd_rand generator{rand_dev()};
    if(this->preferential_weight != -1 && this->recency_weight != -1 && this->fitness_weight != -1) {
        for(int i = 0; i < len; i ++) {
            double pa_uniform = this->preferential_weight;
            double rec_uniform = this->recency_weight;
            double fit_uniform = this->fitness_weight;
            double sum = pa_uniform + rec_uniform + fit_uniform;
            pa_weight_arr[i] = (double)pa_uniform / sum;
            rec_weight_arr[i] = (double)rec_uniform / sum;
            fit_weight_arr[i] = (double)fit_uniform / sum;
        }
    } else {
        for(int i = 0; i < len; i ++) {
            double pa_uniform = this->weights_uniform_distribution(generator);
            double rec_uniform = this->weights_uniform_distribution(generator);
            double fit_uniform = this->weights_uniform_distribution(generator);
            double sum = pa_uniform + rec_uniform + fit_uniform;
            pa_weight_arr[i] = (double)pa_uniform / sum;
            rec_weight_arr[i] = (double)rec_uniform / sum;
            fit_weight_arr[i] = (double)fit_uniform / sum;
        }
    }
}

void ABM::PopulateOutDegreeArr(int* out_degree_arr, int len) {
    std::uniform_int_distribution<int> outdegree_index_uniform_distribution{0, (int)(this->out_degree_bag_vec.size() - 1)};
    std::random_device rand_dev;
    std::minstd_rand generator{rand_dev()};
    for(int i = 0; i < len; i ++) {
        int index_uniform = outdegree_index_uniform_distribution(generator);
        out_degree_arr[i] = this->out_degree_bag_vec[index_uniform];
    }
}

void ABM::UpdateGraphAttributesWeights(Graph* graph, int next_node_id, double* pa_weight_arr, double* rec_weight_arr, double* fit_weight_arr, int len) {
    for(int i = 0; i < len; i ++) {
        int current_node_id = next_node_id + i;
        graph->SetDoubleAttribute("preferential_attachment_weight", current_node_id, pa_weight_arr[i]);
        graph->SetDoubleAttribute("recency_weight", current_node_id, rec_weight_arr[i]);
        graph->SetDoubleAttribute("fitness_weight", current_node_id, fit_weight_arr[i]);
    }
}

void ABM::UpdateGraphAttributesAlphas(Graph* graph, int next_node_id, double* alpha_arr, int len) {
    for(int i = 0; i < len; i ++) {
        int current_node_id = next_node_id + i;
        graph->SetDoubleAttribute("alpha", current_node_id, alpha_arr[i]);
    }
}

void ABM::UpdateGraphAttributesOutDegrees(Graph* graph, int next_node_id, int* out_degree_arr, int len) {
    for(int i = 0; i < len; i ++) {
        int current_node_id = next_node_id + i;
        graph->SetIntAttribute("assigned_out_degree", current_node_id, out_degree_arr[i]);
    }
}

__device__ int ABM::getGraphAttributesGeneratorNode(Graph* graph, 
    device_map<int, Node>::device_view d_nodeAttributeMap_view, int new_node) {
    return graph->d_getGeneratorNode(d_nodeAttributeMap_view, new_node);
}

void ABM::updateGraphAttributesGeneratorNode(Graph* graph, int new_node_seq_id, int generatorNodeSeqId) {
    graph->setGeneratorNode(new_node_seq_id, generatorNodeSeqId);
}

void ABM::CalculateScores(int* src_arr, double* dst_arr, int len) {
    double sum = 0;
    // Mark: removed for node-level
    /* #pragma omp parallel for simd */
    for(int i = 0; i < len; i ++) {
        dst_arr[i] = pow(src_arr[i], this->gamma) + 1;
    }
    // Mark: removed for node-level
    /* #pragma omp parallel for reduction(+:sum) */
    for(int i = 0; i < len; i ++) {
        sum += dst_arr[i];
    }
    // Mark: removed for node-level
    /* #pragma omp parallel for simd */
    for(int i = 0; i < len; i ++) {
        dst_arr[i] /= sum;
    }
}

void ABM::FillSameYearSourceNodes(std::set<int>& same_year_source_nodes, int current_year_new_nodes) {
    //std::cout<<"\nIn FillSameYearSourceNodes with current_year_new_nodes = " << current_year_new_nodes;
    size_t num_same_year_source_nodes = (size_t)std::floor(current_year_new_nodes * this->same_year_proportion);
    std::random_device rand_dev;
    std::minstd_rand generator{rand_dev()};
    std::uniform_int_distribution<int> int_uniform_distribution(0, current_year_new_nodes - 1);
    while(same_year_source_nodes.size() != num_same_year_source_nodes) {
        int current_source = int_uniform_distribution(generator);
        if (same_year_source_nodes.count(current_source) == 0) {
            same_year_source_nodes.insert(current_source);
            //printf("\nsame_year_source_nodes included node %d", current_source);
        }
    }
}

__device__ int ABM::getRandom(int num_new_nodes, curandState* state) {
    // uniform in [0, num_new_nodes-1]
    return curand(state) % num_new_nodes;
}

__device__ double getUniformDouble(curandState* state) {
    // curand_uniform_double returns (0.0, 1.0]
    double r = curand_uniform_double(state);

    double eps = DBL_MIN;   
    return eps + (1.0 - eps) * r; 
}

__device__ float getUniformFloat(curandState* deviceState) {
    // curand_uniform generates (0,1], so we clamp to [FLT_MIN, 1]
    double uniform_random = curand_uniform(deviceState);

    double eps = FLT_MIN;    
    return eps + (1.0 - eps) * uniform_random; 
}

__device__ int ABM::MakeSameYearCitations(int idx, int new_node, int num_new_nodes,
            //cuco::legacy::static_map<int, int>::device_view d_reverse_continuous_node_mapping_view, 
            device_vector_generic<int>& citations, int current_graph_size, curandState* deviceState) {
    int current_citation = getRandom(num_new_nodes, deviceState);
    int key = current_graph_size + current_citation;
    citations[0] = key;
    if (DATA_DEBUG && (idx % 1000 == 1)) {
        printf("\nFor idx = %d, new_node = %d, inserted key = %d", idx, new_node, key);
    }
    
    return 1;
}

__device__ int ABM::MakeUniformRandomCitations(Graph* graph, int idx,
            int graphNodeSetSize, 
            set_ref_type& selected_citations,
            int per_thread_selected_set_capacity,
            //const cuco::legacy::static_map<int, int>::device_view d_reverse_continuous_node_mapping_view, 
            int generator_node, device_vector_generic<int>& citations, 
            int num_cited_so_far, int num_citations, 
            curandState* deviceState) {
    //printf("\nEnter MakeUniformRandomCitations: num_citations = %d", num_citations);
    if (num_citations <= 0) {
        return 0;
    }
    int actual_num_cited = num_citations;
    int generator_node_size = 1;
    if (graphNodeSetSize - num_cited_so_far - generator_node_size < (size_t)num_citations) {
        // WE shoulld never be here..... ");
        actual_num_cited = graphNodeSetSize - num_cited_so_far - generator_node_size;
    }
     
    int selectedSize = 0; // as selected_citations.size() not supported in cuco static_set
    //printf("\nnum_cited_so_far = %d", num_cited_so_far);
    for(int i = 0; i < num_cited_so_far; i ++) {
        selected_citations.insert(citations[i]);
    }
    selectedSize += num_cited_so_far;
    if (!selected_citations.contains(generator_node)) {
        selected_citations.insert(generator_node);
        selectedSize++;
    } else {
        printf("\nERR::CRITICAL:: For idx %d selected_citations already contains generator_node %d", idx, generator_node);
    }
    
    int maxAllowedSelectedSize = (num_cited_so_far + actual_num_cited + generator_node_size);

    int current_citation_index = 0;
    while (selectedSize < maxAllowedSelectedSize) {
        int current_citation = getRandom((graphNodeSetSize-1), deviceState); 
        if (!selected_citations.contains(current_citation)) {
            citations[num_cited_so_far + current_citation_index] = current_citation; //current_citation_rev_mapped_node;  
            selected_citations.insert(current_citation); // WRONG: selected_citations.insert(current_citation);
            selectedSize++;  
            current_citation_index++; 
        }  
    }

    return actual_num_cited;
}

// Custom comparison functor for device code
struct compare_first_descending {
    __device__ bool operator()(const auto& left, const auto& right) const {
        return left.first > right.first;
    }
};


 
__device__ void ABM::PopulateCitations(int idx, int new_node, int N, Graph* graph, 
            curandState* deviceState,
            int current_year, 
            device_vector_soa<float>& element_index_vec,
            device_vector_generic<int>& citations, 
            int num_cited_so_far,
            int current_graph_size, 
            int initial_graph_size,
            int final_graph_size,
            int num_citations) {

    if (DATA_DEBUG && idx % 1000 == 1) {
        printf("\nInside PopulateCitations: thrust::sort done for idx = %d with num_citations = %d ", idx, num_citations);
        printf("\nInside PopulateCitations: idx = %d element_index_vec size = %d ", idx, element_index_vec.size());
    }

    if (DATA_DEBUG && element_index_vec.size() > 100000) {
        for (int i = 0; i < num_citations; i++) {
            if (element_index_vec.get_index(i) > 491532) {
                printf("\nidx = %d i = %d Agent : citations[i] = element_index_vec.get_index(i) = %d, weight = %.12lf", 
                    idx, i,  element_index_vec.get_index(i),  element_index_vec.get_weight(i));
            } else {
                printf("\nidx = %d i = %d seed : citations[i] = element_index_vec.get_index(i) = %d, weight = %.12lf", 
                    idx, i,  element_index_vec.get_index(i),  element_index_vec.get_weight(i));
            }
        }
    }
    for (int i = 0; i < num_citations; i ++) {
        //thrust::pair<double, int> tpair = element_index_vec[i];
        int citation = element_index_vec.get_index(i);
        //citations[num_cited_so_far+i] = tpair.second;
        citations[num_cited_so_far+i] = citation;
    }
}

__device__ void ABM::MakePopulateCitations(int idx, int new_node, int N, Graph* graph, 
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
            int& num_actually_cited) {
    //printf("\nInside MakeCitations");
     
    if (num_citations <= 0) {
        return;
    }

    int candidate_size = candidate_nodes.size(); 
    if (candidate_size <= 0) {
        return;
    }

    if (candidate_size < num_citations) {
        printf("\nInside MakeCitations: For idx = %d candidate_size %d < num_citations %d.. resetting num_citations", 
            idx, candidate_size, num_citations);
        num_citations = candidate_size;
    }
     
    int posWtCnt = 0; 
    float MIN_VAL = 1.0 * INT_MIN;
    for (auto it = candidate_nodes.begin(); it != candidate_nodes.end(); ++it) {
        int candidate_node = *it; //candidate_nodes[i];
        double current_score = 0; 
        // Calculate score for this candidate node
        float current_pa  = __ldg(&pa_arr[candidate_node]);
        float current_rec = __ldg(&recency_arr[candidate_node]);
        float current_fit = __ldg(&fit_arr[candidate_node]); 
        
        current_score = ((current_pa * pa_weight) +
                            (current_rec * rec_weight) +
                            (current_fit * fit_weight)); 
        
        //printf("\nFor candidate_node %d, current_score %lf = (current_pa %lf * pa_weight %lf) + (current_rec %lf * rec_weight %lf) + (current_fit %lf * fit_weight %lf)", 
        //    candidate_node, current_score, current_pa, pa_weight, current_rec, rec_weight,current_fit , fit_weight);

        /**
        // Clamp inputs
        constexpr float EPS = 1e-8f;

        // log products
        float x1 = logf(fmaxf(current_pa, EPS)) + logf(fmaxf(pa_weight, EPS));
        float x2 = logf(fmaxf(current_rec, EPS)) + logf(fmaxf(rec_weight, EPS));
        float x3 = logf(fmaxf(current_fit, EPS)) + logf(fmaxf(fit_weight, EPS));  

        // log-sum-exp
        float m = fmaxf(x1, fmaxf(x2, x3));

        float log_score = m + logf(
            expf(x1 - m) +
            expf(x2 - m) +
            expf(x3 - m)
        );
        */
        // uniform sample
        float wrs_uniform = curand_uniform(deviceState);
        //


        /**  jay_score_2
        // log-weight (this is the key you should compare)
        // Weighted Random Sampling (WRS) with Efraimidis-Spirakis algorithm
        wrs_uniform = fmaxf(wrs_uniform, EPS);
        float random_weight = logf(wrs_uniform) * expf(-log_score);
        */

        float random_weight = MIN_VAL;
        /** jay_score 1 */
        if (current_score != 0) {
            random_weight = log(wrs_uniform) / current_score; // pow(wrs_uniform, 1.0/current_score);    
        }  

        //printf("\nFor candidate_node %d, random_weight = %f, log_score %f = (current_pa %f * pa_weight %f) + (current_rec %f * rec_weight %f) + (current_fit %f * fit_weight %f)", 
        //    candidate_node, random_weight, log_score, current_pa, pa_weight, current_rec, rec_weight,current_fit , fit_weight);

        if (random_weight != 0) { 
            posWtCnt++;
        }

        //printf("\nfor idx = %d inserting random_weight = %f, candidate_node = %d", idx, random_weight, candidate_node);
        d_heap.insert(random_weight, candidate_node);
        // element_index_vec.push_back(random_weight, candidate_node, "element_index_vec");
    }
    if (DATA_DEBUG && posWtCnt > 0)
        printf("\nfor idx = %d, new_node = %d, posWtCnt = %d", idx, new_node, posWtCnt);

    if (DATA_DEBUG && (idx % 1000 == 1)) {
        printf("\nfor idx = %d, new_node = %d, d_heap.sort_descending_inplace", idx, new_node);
    }
    d_heap.sort_descending_inplace();

    for (int i = 0; i < num_citations; i ++) {
        int citation = d_heap.data[i].index; // element_index_vec.get_index(i);
        citations[num_cited_so_far+i] = citation;
        if (DATA_DEBUG && (idx % 1000 == 1)) {
            printf("\nfor idx = %d, citations[num_cited_so_far %d +i %d ] = citation %d : weight = %f", 
                idx, num_cited_so_far, i, citation, d_heap.data[i].value);
        }
    }
    num_actually_cited = num_cited_so_far + num_citations;
}

int ABM::getGeneratorNode(Graph* graph /*, const std::map<int, int>& reverse_continuous_node_mapping*/) {
    std::uniform_int_distribution<int> generator_uniform_distribution{0, (int)(graph->getNodeSetSize() - 1)};
    std::random_device rand_dev;
    std::minstd_rand generator{rand_dev()};
    
    int continuous_generator_node = generator_uniform_distribution(generator);
    return continuous_generator_node; // generator_node;
}

// DEVICE: 1- and 2-hop neighborhood using CompactBFSState + device_vector
__device__ void ABM::GetOneAndTwoHopNeighborhood(
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
    device_vector& two_hop_neighborhood)
{
    const int max_hops = 2;
    bfs_state.reset();

    // --- Seed BFS: mark generator_node visited and initialize frontier ---
    bfs_state.mark_visited(generator_node);    // mark (atomic test-and-set)
    bfs_state.init_frontier(generator_node);   // place generator_node in current frontier

    // Level-by-level BFS up to max_hops
    for (int current_distance = 0; current_distance < max_hops; ++current_distance) {
        // For each 32-bit word in current frontier bitmap
        int num_words = bfs_state.bitmap_words;
        for (int word_idx = 0; word_idx < num_words; ++word_idx) {
            // Read a local copy of the current frontier word (do not modify global bitmap here)
            uint32_t word = bfs_state.d_queue_bitmap_curr[word_idx];

            // Iterate each set bit in this word
            while (word != 0) {
                int bit_pos = __ffs(word) - 1;  // __ffs returns 1-based index of LSB set
                int current_node = (word_idx << 5) + bit_pos;

                // Clear the bit in the local copy so we progress
                word &= ~(1U << bit_pos);

                // Defensive bounds check
                if (current_node < 0 || current_node >= bfs_state.max_vertices) continue;

                // Expand forward (outgoing) neighbors
                int out_start = __ldg(&d_forward_adj_map->offsets[current_node]);
                int out_end   = __ldg(&d_forward_adj_map->offsets[current_node + 1]);
                if (DATA_DEBUG) {
                    printf("\nFor idx = %d, node = %d, out_start = %d, out_end = %d, outdeg = %d", 
                        idx, new_node, out_start, out_end, (out_end - out_start));
                }
                for (int e = out_start; e < out_end; ++e) {
                    int nbr = __ldg(&d_forward_adj_map->edges[e]);

                    // atomic test-and-set visited; if newly visited, add to next frontier
                    bool newly = bfs_state.mark_visited(nbr);
                    if (newly) {
                        bfs_state.add_to_next_frontier(nbr);

                        int discovered_distance = current_distance + 1;
                        if (discovered_distance == 1) {
                            if (one_hop_neighborhood.size() < one_hop_neighborhood.get_capacity()) {
                                one_hop_neighborhood.push_back(nbr, "one_hop_neighborhood");
                                if (DATA_DEBUG)
                                    printf("\n[1-HOP]: Outgoing idx=%d added nbr %d: new size = %d\n", idx, nbr, one_hop_neighborhood.size());

                            } else {
                                printf("WARNING: one_hop_neighborhood overflow at idx=%d, size=%d\n",
                                       idx, one_hop_neighborhood.size());
                            }
                        } else if (discovered_distance == 2) {
                            if (two_hop_neighborhood.size() < two_hop_neighborhood.get_capacity()) {
                                two_hop_neighborhood.push_back(nbr, "two_hop_neighborhood");
                                if (DATA_DEBUG)
                                    printf("\n[2-HOP]: Outgoing idx=%d added nbr %d: new size = %d\n", idx, nbr, one_hop_neighborhood.size());

                            } else {
                                printf("WARNING: two_hop_neighborhood overflow at idx=%d, size=%d\n",
                                       idx, two_hop_neighborhood.size());
                            }
                        }
                    }
                } // end forward neighbors loop

                // Expand backward (incoming) neighbors
                int in_start = __ldg(&d_backward_adj_map->offsets[current_node]);
                int in_end   = __ldg(&d_backward_adj_map->offsets[current_node + 1]);
                if (DATA_DEBUG) {
                    printf("\nFor idx = %d, node = %d, in_start = %d, in_end = %d, outdeg = %d", 
                        idx, new_node, in_start, in_end, (in_end - in_start));
                }
                for (int e = in_start; e < in_end; ++e) {
                    int nbr = __ldg(&d_backward_adj_map->edges[e]);

                    bool newly = bfs_state.mark_visited(nbr);
                    if (newly) {
                        bfs_state.add_to_next_frontier(nbr);

                        int discovered_distance = current_distance + 1;
                        if (discovered_distance == 1) {
                            if (one_hop_neighborhood.size() < one_hop_neighborhood.get_capacity()) {
                                one_hop_neighborhood.push_back(nbr, "one_hop_neighborhood");
                                if (DATA_DEBUG)
                                    printf("\n[1-HOP]: incoming idx=%d added nbr %d: new size = %d\n", idx, nbr, one_hop_neighborhood.size());
                            } else {
                                printf("WARNING: one_hop_neighborhood overflow at idx=%d, size=%d\n",
                                       idx, one_hop_neighborhood.size());
                            }
                        } else if (discovered_distance == 2) {
                            if (two_hop_neighborhood.size() < two_hop_neighborhood.get_capacity()) {
                                two_hop_neighborhood.push_back(nbr, "two_hop_neighborhood");
                                if (DATA_DEBUG)
                                    printf("\n[2-HOP]: incoming idx=%d added nbr %d: new size = %d\n", idx, nbr, one_hop_neighborhood.size());
                            } else {
                                printf("WARNING: two_hop_neighborhood overflow at idx=%d, size=%d\n",
                                       idx, two_hop_neighborhood.size());
                            }
                        }
                    }
                } // end backward neighbors loop
            } // end while(word)
        } // end for each word

        // Move next frontier bits into current frontier and clear next frontier
        bfs_state.swap_frontiers();
    } // end for distance
}

 
// ----------------------------------------------------------------------
// Hong & Kim Warp-Centric 2-hop BFS
// ----------------------------------------------------------------------
// One warp performs BFS for exactly one (idx,new_node).
// Frontier stored as warp-private bitmaps + neighbor lists.
// ----------------------------------------------------------------------

__device__ void ABM::GetOneAndTwoHopNeighborhood_Warp(
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
    int max_vertices)
{
    const unsigned FULL_MASK = 0xFFFFFFFFu;
    const int lane_id   = threadIdx.x & 31;

    // ---------------------------------------------------------------
    // Local constants
    // ---------------------------------------------------------------
    const int MAXV = max_vertices;      // = 1,000,000
    const int NUM_WORDS = (MAXV + 31) >> 5;        // bitmap size
    const int MAX_HOPS = 2;                        // Two-hop traversal

    // ---------------------------------------------------------------
    // Warp-strided clear of bitmaps
    // INITIALIZATION: Warp-strided clearing (paper code line 65-68 pattern)
    // ---------------------------------------------------------------
    for (int w = lane_id; w < NUM_WORDS; w += WARP_SIZE) {
        visited_bitmap[w]       = 0;
        frontier_curr_bitmap[w] = 0;
        frontier_next_bitmap[w] = 0;
    }
    __syncwarp();

    // ---------------------------------------------------------------
    // Mark generator visited
    // Set generator node as root (paper line 8: initialize_levels)
    // ---------------------------------------------------------------
    int gen_word = generator_node >> 5;
    int gen_bit = generator_node & 31;
    if (DATA_DEBUG && (idx % 10000 == 1)) {
        printf("\nFor idx = %d, node = %d, generator_node = %d gen_word = %d, gen_bit = %d", 
            idx, new_node, generator_node, gen_word, gen_bit);
    }

    if (lane_id == 0) {
        visited_bitmap[gen_word] |= (1u << gen_bit);
        frontier_curr_bitmap[gen_word] |= (1u << gen_bit);

        if (idx % 10000 == 1) {
            printf("\n[INIT] idx=%d node=%d: generator=%d set at word=%d bit=%d\n",
                   idx, new_node, generator_node, gen_word, gen_bit);
        }

        if (DATA_DEBUG) {
            printf("\nGenerator %d set in word %d bit %d: 0x%08x for idx=%d node=%d\n",
            generator_node, gen_word, gen_bit, 
            frontier_curr_bitmap[gen_word], idx, new_node);
        }
    }
    __syncwarp();

    // ---------------------------------------------------------------
    // 2-HOP BFS: Do exactly two levels
    // BFS MAIN LOOP: Level-synchronous (paper line 10-13)
    // ---------------------------------------------------------------
    int level = 0; 

    while (level < MAX_HOPS) { 
        // Reset next frontier
        for (int w = lane_id; w < NUM_WORDS; w += WARP_SIZE) {
            frontier_next_bitmap[w] = 0;
        }

        __syncwarp();

        if (idx % 10000 == 1 && lane_id == 0) {
            printf("\n[LEVEL %d] idx=%d node=%d: Starting frontier expansion\n",
                   level, idx, new_node);
        }

        // --------------------------------------------------------------------
        // Warp-centric frontier expansion
        // WARP-STRIDED BITMAP PROCESSING (line 57-77 adapted for bitmaps)
        // --------------------------------------------------------------------
        for (int word_idx = lane_id; word_idx < NUM_WORDS; word_idx += WARP_SIZE) {
            // Load frontier word cooperatively
            uint32_t word = frontier_curr_bitmap[word_idx];

            if (word == 0) {
                continue;
            } else {
                if (DATA_DEBUG && (idx % 10000 == 1)) {
                    printf("\n[L2] idx = %d, node = %d, word_idx=%d word=0x%08x is non-zero...generator=%d  \n",
                            idx, new_node, word_idx, word, generator_node);
                }
            }

            if (idx % 10000 == 1) {
                printf("\n[L%d] idx=%d lane=%d word_idx=%d: processing word=0x%08x\n",
                       (level+1), idx, lane_id, word_idx, word);
            }

            int base_node = word_idx << 5;
            // Process each set bit in this frontier word
            while (word != 0) {
                int bit_pos = __ffs(word) - 1;
                int vertex = base_node + bit_pos;
                if (DATA_DEBUG && (idx % 10000 == 1)) {
                    printf("\n[L2] idx = %d, node = %d, word=0x%08x is non-zero...bit_pos=%d vertex = %d \n",
                        idx, new_node, word, bit_pos, vertex);
                }
                if (vertex >= max_vertices) continue;  // Bounds check

                // Clear this bit in local copy (lane0 does this)
                word &= ~(1u << bit_pos);  // Clear this bit 

                __syncwarp();

                // ================================================================
                // FORWARD NEIGHBORS: Warp-strided expansion (paper line 22-28, 38-45)
                // ================================================================ 
                int out_degree = graph->d_GetOutDegree(d_nodeAttr_view, vertex);
                
                if (out_degree > 0) {
                    int start = d_forward_adj->offsets[vertex];
                    int end   = d_forward_adj->offsets[vertex + 1];
                    int num_nbr = end - start;

                    if (MEM_DEBUG && idx % 10000 == 1) {
                        printf("\n[FORWARD] idx=%d lane=%d vertex=%d: start=%d end=%d deg computed=%d out_degree = %d\n",
                               idx, lane_id, vertex, start, end, num_nbr, out_degree);
                    }

                    // Warp-strided neighbor iteration (SIMD pattern from paper)
                    for (int e = start; e < end; e++) {  
                        int w = d_forward_adj->edges[e];

                        if (w >= 0 && w < max_vertices) {
                            // Atomic check-and-mark (paper line 25-27, 42-44)
                            int w_word = w >> 5;
                            int w_bit  = w & 31;
                            uint32_t w_mask = (1u << w_bit);

                            uint32_t old = atomicOr(&visited_bitmap[w_word], w_mask);
                            bool is_new = ((old & w_mask) == 0);

                            if (is_new) { 
                                // Add to next frontier
                                atomicOr(&frontier_next_bitmap[w_word], w_mask);

                                // Classify by level (level=0→1-hop, level=1→2-hop)
                                if (level == 0) {
                                    one_hop_neighborhood.insert(w);
                                    if (DATA_DEBUG) {
                                        printf("\n[1-HOP] idx=%d lane=%d: added vertex %d: new size = %d\n",
                                                idx, lane_id, w, one_hop_neighborhood.size());
                                    }
                                } else if (level == 1) {
                                    two_hop_neighborhood.insert(w);
                                    if (DATA_DEBUG) {
                                        printf("\n[2-HOP] idx=%d lane=%d: added vertex %d: new size = %d\n",
                                                idx, lane_id, w, two_hop_neighborhood.size());
                                    }
                                }
                            }
                        }
                    }
                }

                // ================================================================
                // BACKWARD NEIGHBORS: Same warp-strided pattern
                // ================================================================
                int in_degree = graph->d_GetInDegree(d_nodeAttr_view, vertex);
                if (in_degree > 0) {
                    int start = d_backward_adj->offsets[vertex];
                    int end   = d_backward_adj->offsets[vertex + 1];
                    int num_nbr = end - start;

                    if (MEM_DEBUG && idx % 10000 == 1) {
                        printf("\n[BACKWARD] idx=%d lane=%d vertex=%d: start=%d end=%d in deg computed =%d in_degree = %d\n",
                               idx, lane_id, vertex, start, end, num_nbr, in_degree);
                    }

                    // Warp-strided neighbor iteration
                    for (int e = start; e < end; e++) {    
                        int w = d_backward_adj->edges[e];

                        if (w >= 0 && w < max_vertices) {
                            // Atomic check-and-mark
                            int w_word = w >> 5;
                            int w_bit  = w & 31;
                            uint32_t w_mask = (1u << w_bit);

                            uint32_t old = atomicOr(&visited_bitmap[w_word], w_mask);
                            bool is_new = ((old & w_mask) == 0);

                            if (is_new) { 
                                // Add to next frontier
                                atomicOr(&frontier_next_bitmap[w_word], w_mask);

                                // Classify by level
                                if (level == 0) {
                                    one_hop_neighborhood.insert(w);
                                    if (DATA_DEBUG) {
                                        printf("\n[1-HOP-IN] idx=%d lane=%d: added vertex %d: new size = %d\n",
                                                idx, lane_id, w, one_hop_neighborhood.size());
                                    }
                                } else if (level == 1) {
                                    two_hop_neighborhood.insert(w);
                                    if (DATA_DEBUG) {
                                        printf("\n[2-HOP-IN] idx=%d lane=%d: added vertex %d: new size = %d\n",
                                                idx, lane_id, w, two_hop_neighborhood.size());
                                    }
                                }
                            }
                        }
                    }
                }

                __syncwarp();  // Sync after processing this vertex
            }
        }

        // ========================================================================
        // Memory fence to ensure all atomic operations are visible
        // ========================================================================
        __threadfence_block();
        __syncwarp();

        if (idx % 10000 == 1 && lane_id == 0) {
            printf("\n[LEVEL %d] idx=%d node=%d: Completed, swapping frontiers\n",
                   level, idx, new_node);
        }

        // ========================================================================
        // SWAP FRONTIERS: Copy next → current (paper frontier swap pattern)
        // ========================================================================
        for (int w = lane_id; w < NUM_WORDS; w += WARP_SIZE) {
            frontier_curr_bitmap[w] = frontier_next_bitmap[w];
        }
        __syncwarp();

        level++;
    }

    // ============================================================================
    // FINAL SYNC: Ensure all operations complete
    // ============================================================================
    __syncwarp();

    if (idx % 10000 == 1 && lane_id == 0) {
        printf("\n[DONE] idx=%d node=%d: 1-hop=%d 2-hop=%d\n",
               idx, new_node, *one_hop_neighborhood.d_size, *two_hop_neighborhood.d_size);
    }
}

 
 
__device__ void ABM::ABMKernelStage2(
    int idx, int new_node, int N,
    Graph* graph,
    curandState* deviceState,
    device_vector& one_hop_neighborhood,
    device_min_heap<float>& d_heap, 
    //device_vector_soa<float>& element_index_vec, 
    device_vector_generic<int>& citations,
    double* pa_arr, double* recency_arr, double* fit_arr,
    double pa_weight, double rec_weight, double fit_weight,
    int current_year, 
    int current_graph_size,
    int initial_graph_size, 
    int final_graph_size,
    /*same-year flag*/ int same_year_citation,
    /*precomputed*/ int& num_citations_inside,
    /*in/out*/ int& num_actually_cited)
{
    if (same_year_citation) {
        num_actually_cited += this->MakeSameYearCitations(
            idx, new_node, N,
            citations,
            current_graph_size,
            deviceState);
    }

    if (DATA_DEBUG && idx %10000 == 1) {
        printf("\nCalling MakeCitations (one hop) for node %d of idx = %d one_hop_neighborhood size = %d", 
            new_node, idx, one_hop_neighborhood.size());
    }
    int num_actually_cited_so_far = num_actually_cited;

    this->MakePopulateCitations(idx, new_node, N, graph,
        deviceState,  
        current_year, one_hop_neighborhood, 
        d_heap, //element_index_vec,
        citations, 
        pa_arr, recency_arr, fit_arr,
        pa_weight, rec_weight, fit_weight,
        current_graph_size, initial_graph_size, final_graph_size,
        num_actually_cited_so_far,
        num_citations_inside,
        num_actually_cited);
        
    if (DATA_DEBUG && (idx %1000 == 1)) {
        printf("\nIn ABMKernelStage2:: idx=%d, one_hop_neighborhood.size()=%d, num_citations_inside(before min)=%d", 
            idx, one_hop_neighborhood.size(), num_citations_inside); 
    }
}


__device__ void ABM::ABMKernelStage3(
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
    curandState* deviceState)
{
    
    if (DATA_DEBUG && (idx % 1000 == 1)) {
        printf("\nIn ABMKernelStage3: idx = %d, calling MakeCitations for two_hop_neighborhood of size = %d. num_citations_outside = %d", 
            idx, two_hop_neighborhood.size(), num_citations_outside);
    }
    this->MakePopulateCitations(idx, new_node, N, graph,
        deviceState,  
        current_year, two_hop_neighborhood, 
        d_heap, //element_index_vec,
        citations, 
        pa_arr, recency_arr, fit_arr,
        pa_weight, rec_weight, fit_weight,
        current_graph_size, initial_graph_size, final_graph_size,
        num_actually_cited_so_far,
        num_citations_outside,
        num_actually_cited);
}

__device__ void ABM::ABMKernelStage4(
    int idx, int N, Graph* graph, int graphNodeSetSize,
    //device_vector_soa<float>& element_index_vec, 
    device_vector_generic<int>& citations,
    device_vector_generic<int2>& d_new_edges_vec,
    set_ref_type& selected_citations,
    int generator_node, int new_node,
    int current_year, int current_graph_size,
    int initial_graph_size, int final_graph_size,
    int& num_citations_outside, 
    int& num_fully_random_cited,
    int& num_actually_cited,
    curandState* deviceState,
    int per_thread_selected_set_capacity) 
{

    /*this->PopulateCitations(idx, 
        new_node, N, graph,
        deviceState, //d_continuous_node_mapping_view,
        current_year, element_index_vec,
        citations, num_actually_cited_so_far,
        current_graph_size, initial_graph_size, final_graph_size,
        num_citations_outside);*/

    if (DATA_DEBUG && idx % 10000 == 1) {
        printf("\nIn ABMKernelStage4: idx = %d, num_actually_cited = %d, num_citations_outside = %d, num_fully_random_cited = %d ", 
            idx, num_actually_cited, num_citations_outside, num_fully_random_cited);
    }

    num_actually_cited += this->MakeUniformRandomCitations(
        graph, idx, graphNodeSetSize, selected_citations,
        per_thread_selected_set_capacity,
        //d_reverse_continuous_node_mapping_view,
        generator_node, citations,
        num_actually_cited, num_fully_random_cited,
        deviceState); 

    d_new_edges_vec.push_back(make_int2(new_node, generator_node), "d_new_edges_vec");
    for (int j = 0; j < num_actually_cited; j++) {
        d_new_edges_vec.push_back(make_int2(new_node, citations[j]), "d_new_edges_vec");
    }

    if (DATA_DEBUG && (idx % 1000 == 1)) {
        printf("\nPushed citations: year=%d, idx=%d, d_new_edges_vec.size()=%d, num_actually_cited=%d", 
            current_year, idx, d_new_edges_vec.size(), num_actually_cited); 
    }
}
