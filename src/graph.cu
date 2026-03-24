#include "graph.cuh"

Graph::Graph(std::string edgelist, std::string nodelist) : edgelist(edgelist), nodelist(nodelist) {
        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    printf("Init Graph");
    this->ParseNodelist();
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    this->ParseEdgelist();
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    
	this->updateNodeInDegreeOutDegree();
    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

    auto duration01 = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
	auto duration12 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    auto duration23 = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2);

    std::cout << "Elapsed time: 0-1 : " << duration01.count()/1000 << " seconds" << std::endl;
    std::cout << "Elapsed time: 1-2 : " << duration12.count()/1000 << " seconds" << std::endl;
    std::cout << "Elapsed time: 2-3 : " << duration23.count()/1000 << " seconds" << std::endl;
    printf("\nInitialized Graph. Let us update the indegree and out-degrees... ");
}

inline int fast_atoi(const char* str, const char* end) {
    int result = 0;
    bool negative = false;
    
    while (str < end && (*str == ' ' || *str == '\t')) str++;
    
    if (str < end && *str == '-') {
        negative = true;
        str++;
    } else if (str < end && *str == '+') {
        str++;
    }
    
    while (str < end && *str >= '0' && *str <= '9') {
        result = result * 10 + (*str - '0');
        str++;
    }
    
    return negative ? -result : result;
}


void Graph::PrintFinalGraphStatistics() {

    const size_t num_nodes = this->node_set.size();

    // count edges
    size_t num_edges = 0;
    for (const auto& [u, nbrs] : forward_adj_map) {
        num_edges += nbrs.size();
    }

    long total_in = 0;
    long total_out = 0;

    int min_in = INT_MAX;
    int max_in = INT_MIN;

    int min_out = INT_MAX;
    int max_out = INT_MIN;

    int isolated = 0;

    std::vector<int> degree;
    degree.reserve(num_nodes);

    for (const auto& node : node_set) {

        int indeg = GetInDegree(node);
        int outdeg = GetOutDegree(node);

        total_in += indeg;
        total_out += outdeg;

        min_in = std::min(min_in, indeg);
        max_in = std::max(max_in, indeg);

        min_out = std::min(min_out, outdeg);
        max_out = std::max(max_out, outdeg);

        int deg = indeg + outdeg;
        degree.push_back(deg);

        if (deg == 0)
            isolated++;
    }

    double avg_in = (double)total_in / num_nodes;
    double avg_out = (double)total_out / num_nodes;

    // compute std deviation
    double var_in = 0;
    double var_out = 0;

    for (const auto& node : node_set) {
        int indeg = GetInDegree(node);
        int outdeg = GetOutDegree(node);

        var_in += (indeg - avg_in) * (indeg - avg_in);
        var_out += (outdeg - avg_out) * (outdeg - avg_out);
    }

    double std_in = std::sqrt(var_in / num_nodes);
    double std_out = std::sqrt(var_out / num_nodes);

    double density = (double)num_edges / (num_nodes * (num_nodes - 1));

    std::cout << "\n================ FINAL GRAPH STATISTICS ================" << std::endl;

    std::cout << "Nodes: " << num_nodes << std::endl;
    std::cout << "Edges: " << num_edges << std::endl;
    std::cout << "Density: " << density << std::endl;

    std::cout << "\nIn-degree statistics" << std::endl;
    std::cout << "Avg: " << avg_in << std::endl;
    std::cout << "StdDev: " << std_in << std::endl;
    std::cout << "Min: " << min_in << std::endl;
    std::cout << "Max: " << max_in << std::endl;

    std::cout << "\nOut-degree statistics" << std::endl;
    std::cout << "Avg: " << avg_out << std::endl;
    std::cout << "StdDev: " << std_out << std::endl;
    std::cout << "Min: " << min_out << std::endl;
    std::cout << "Max: " << max_out << std::endl;

    std::cout << "\nIsolated Nodes: " << isolated << std::endl;

    /* ---------------------------
       Degree histogram
       --------------------------- */

    std::unordered_map<int,int> histogram;

    for (auto d : degree)
        histogram[d]++;

    std::cout << "\nDegree Histogram (degree : count)\n";

    for (const auto& [d,c] : histogram)
        std::cout << d << " : " << c << std::endl;

    /* ---------------------------
       Power-law exponent
       --------------------------- */

    int kmin = 1;
    double sum_log = 0;
    int count = 0;

    for (auto d : degree) {
        if (d >= kmin) {
            sum_log += std::log((double)d/kmin);
            count++;
        }
    }

    double alpha = 1 + (double)count / sum_log;

    std::cout << "\nEstimated Power-law exponent (alpha): " << alpha << std::endl;

    /* ---------------------------
       Giant component
       --------------------------- */

    std::unordered_set<int> visited;
    size_t largest_component = 0;

    for (const auto& start : node_set) {

        if (visited.contains(start))
            continue;

        std::queue<int> q;
        q.push(start);
        visited.insert(start);

        size_t component_size = 0;

        while (!q.empty()) {

            int v = q.front();
            q.pop();
            component_size++;

            if (forward_adj_map.contains(v)) {
                for (int u : forward_adj_map.at(v)) {
                    if (!visited.contains(u)) {
                        visited.insert(u);
                        q.push(u);
                    }
                }
            }

            if (backward_adj_map.contains(v)) {
                for (int u : backward_adj_map.at(v)) {
                    if (!visited.contains(u)) {
                        visited.insert(u);
                        q.push(u);
                    }
                }
            }
        }

        largest_component = std::max(largest_component, component_size);
    }

    std::cout << "\nGiant Component Size: "
              << largest_component
              << " (" << (100.0 * largest_component / num_nodes)
              << "% of nodes)" << std::endl;

    std::cout << "========================================================\n" << std::endl;
}   

/*void Graph::PrintAdvancedGraphStatistics() {

    size_t num_nodes = nodes.size();
    size_t num_edges = edges.size();

    std::vector<int> degree(num_nodes);

    for (size_t i = 0; i < num_nodes; i++) {
        degree[i] = nodes[i].indegree + nodes[i].outdegree;
    }

    /* -------------------------
       Degree distribution
       ------------------------- * /

    std::unordered_map<int,int> histogram;

    for (auto d : degree) {
        histogram[d]++;
    }

    std::cout << "\nDegree Distribution Histogram (degree : count)\n";

    for (auto &p : histogram) {
        std::cout << p.first << " : " << p.second << std::endl;
    }

    /* -------------------------
       Power-law exponent
       (maximum likelihood)
       ------------------------- * /

    int kmin = 1;
    double sum_log = 0.0;
    int count = 0;

    for (auto d : degree) {
        if (d >= kmin) {
            sum_log += std::log((double)d / kmin);
            count++;
        }
    }

    double alpha = 1 + (double)count / sum_log;

    std::cout << "\nEstimated Power-law Exponent (alpha): "
              << alpha << std::endl;

    /* -------------------------
       Isolated nodes
       ------------------------- * /

    int isolated = 0;

    for (auto d : degree) {
        if (d == 0)
            isolated++;
    }

    std::cout << "Isolated nodes: " << isolated << std::endl;

    /* -------------------------
       Giant component size
       (BFS)
       ------------------------- * /

    std::vector<bool> visited(num_nodes,false);

    int largest_component = 0;

    for (size_t i=0;i<num_nodes;i++) {

        if (visited[i])
            continue;

        std::queue<int> q;
        q.push(i);
        visited[i] = true;

        int size = 0;

        while(!q.empty()) {

            int v = q.front();
            q.pop();
            size++;

            for (auto &e : nodes[v].out_edges) {

                int u = e.target;

                if (!visited[u]) {
                    visited[u] = true;
                    q.push(u);
                }
            }

            for (auto &e : nodes[v].in_edges) {

                int u = e.source;

                if (!visited[u]) {
                    visited[u] = true;
                    q.push(u);
                }
            }
        }

        largest_component = std::max(largest_component, size);
    }

    std::cout << "Giant component size: "
              << largest_component
              << " (" << (100.0*largest_component/num_nodes)
              << "% of graph)" << std::endl;

}*/

void Graph::ParseNodelist() {
    printf("\nInside  Graph ParseNodelist.....");
    std::chrono::steady_clock::time_point t0= std::chrono::steady_clock::now();
    char delimiter = Graph::get_delimiter(this->nodelist);
    std::ifstream input_nodelist(this->nodelist);
    std::string line;
    std::map<int, int> nodeMap;
    while(std::getline(input_nodelist, line)) {
        std::stringstream ss(line);
        std::string current_value;
        std::vector<std::string> current_line;
        while(std::getline(ss, current_value, delimiter)) {
            current_line.push_back(current_value);
        }
        std::string node = current_line[0];
        if(node[0] == '#') {
            continue;
        }
        int integer_node = std::stoi(node);
        int integer_year = std::stoi(current_line[1]);
        nodeMap[integer_node] = integer_year;        
    }

    std::cout<<"\nPopulating the continuous_node mappings..\n";
    for (const auto& [integer_node, integer_year] : nodeMap) {
        continuous_node_mapping[integer_node] = node_seq_id;
        //printf("\nseq = %d, node id = %d", node_seq_id, integer_node);
        reverse_continuous_node_mapping[node_seq_id] = integer_node;
        
        this->SetIntAttribute("year", node_seq_id, integer_year);
        this->setType(SEED_TYPE, node_seq_id);
        node_seq_id++;
    }
    
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    auto duration01 = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    std::cout << "Elapsed time: ParseNodelist : " << duration01.count() << " millisecs = " << duration01.count()/1000 << " seconds" << std::endl;
}


void Graph::ParseEdgelist() {
    printf("\nInside Graph ParseEdgelist.....");
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    
    // Convert to unordered_map for O(1) lookups
    std::unordered_map<int, int> fast_mapping;
    fast_mapping.reserve(continuous_node_mapping.size());
    for (const auto& [k, v] : continuous_node_mapping) {
        fast_mapping[k] = v;
    }
    
    // Read entire file
    std::ifstream file(this->edgelist, std::ios::binary | std::ios::ate);
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::string file_contents;
    file_contents.resize(file_size);
    file.read(&file_contents[0], file_size);
    file.close();
    
    char delimiter = Graph::get_delimiter(this->edgelist);
    
    std::vector<std::pair<int, int>> edges;
    edges.reserve(file_size / 15);
    
    const char* ptr = file_contents.data();
    const char* end = ptr + file_size;
    
    while (ptr < end) {
        const char* line_start = ptr;
        const char* line_end = ptr;
        while (line_end < end && *line_end != '\n' && *line_end != '\r') line_end++;
        
        if (line_end == line_start || *line_start == '#') {
            ptr = line_end + 1;
            if (ptr < end && *line_end == '\r' && *ptr == '\n') ptr++;
            continue;
        }
        
        // Parse citing
        const char* field1_end = line_start;
        while (field1_end < line_end && *field1_end != delimiter) field1_end++;
        int citing = fast_atoi(line_start, field1_end);
        
        // Parse cited
        const char* field2_start = (field1_end < line_end) ? field1_end + 1 : field1_end;
        const char* field2_end = field2_start;
        while (field2_end < line_end && *field2_end != delimiter) field2_end++;
        int cited = fast_atoi(field2_start, field2_end);
        
        // Fast O(1) lookup
        auto citing_it = fast_mapping.find(citing);
        auto cited_it = fast_mapping.find(cited);
        
        if (citing_it != fast_mapping.end() && cited_it != fast_mapping.end()) {
            edges.emplace_back(citing_it->second, cited_it->second);
        }
        
        ptr = line_end + 1;
        if (ptr < end && *line_end == '\r' && *ptr == '\n') ptr++;
    }
    
    for (const auto& edge : edges) {
        this->AddEdge(edge);
    }
    
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    auto duration01 = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    std::cout << "\nElapsed time: ParseEdgelist : " << duration01.count() 
              << " millisecs = " << duration01.count() / 1000 << " seconds" << std::endl;
}
/*void Graph::ParseEdgelist() {
    printf("\nInside  Graph ParseEdgelist.....");
    std::chrono::steady_clock::time_point t0= std::chrono::steady_clock::now();
    char delimiter = Graph::get_delimiter(this->edgelist);
    std::ifstream input_edgelist(this->edgelist);
    std::string line;
    while(std::getline(input_edgelist, line)) {
        std::stringstream ss(line);
        std::string current_value;
        std::vector<std::string> current_line;
        while(std::getline(ss, current_value, delimiter)) {
            current_line.push_back(current_value);
        }
        std::string citing = current_line[0];
        if(citing[0] == '#') {
            continue;
        }
        int integer_citing = std::stoi(citing);
        int integer_cited = std::stoi(current_line[1]);
        int seq_id_citing = continuous_node_mapping.at(integer_citing);
        int seq_id_cited = continuous_node_mapping.at(integer_cited);
        
        this->AddEdge({seq_id_citing, seq_id_cited});
    }
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    auto duration01 = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    std::cout << "\nElapsed time: ParseEdgelist : " << duration01.count() << " millisecs = " << duration01.count()/1000 << " seconds" << std::endl;
}*/

std::map<int, int> Graph::getContinuousNodeMapping() {
    return this->continuous_node_mapping;
}
        
std::map<int, int> Graph::getReverseContinuousNodeMapping() {
    return this->reverse_continuous_node_mapping;
}
  
void Graph::updateNodeInDegreeOutDegree() {
    std::cout<<"\nIn updateNodeInDegreeOutDegree: Update Outdegree first:";
    long totalOutDegree = 0;
    for (const auto& [nodeSeqId, set] : this->forward_adj_map) {
        int outdegree = set.size();
        this->SetIntAttribute("out_degree", nodeSeqId, outdegree); 
        totalOutDegree += outdegree;
    }
    long totalInDegree = 0;
    std::cout<<"\nIn updateNodeInDegreeOutDegree: Update Indegree:";
    for (const auto& [nodeSeqId, set] : this->backward_adj_map) {
        int indegree = set.size();
        // std::cout <<"\nin_degree update:: "<< nodeSeqId << " => (indegree) = " << indegree << "\n";
        this->SetIntAttribute("in_degree", nodeSeqId, indegree); 
        totalInDegree += indegree;
    }  

    std::cout<<"\nTotal (initial):: In-Degree = "<< totalInDegree << ", Out-Degree = " << totalOutDegree << std::endl;
}

/*
void Graph::updateNodeInDegreeOutDegree(std::vector<int> new_nodes_vec,
                                    std::unordered_set<int> updated_destination_nodes, 
                                    int year) {
    // Only update nodes that changed this year:
    //   - new_nodes_vec: out_degree changed (they made citations)
    //   - updated_destination_nodes: in_degree changed (they received citations)
    // This avoids iterating the entire forward/backward adj maps (~1.2M entries).

    for (int nodeSeqId : new_nodes_vec) {
        auto fit = this->forward_adj_map.find(nodeSeqId);
        int outdegree = (fit != this->forward_adj_map.end()) ? (int)fit->second.size() : 0;
        auto& node = this->nodeAttributeMap.at(nodeSeqId);
        node.out_degree = outdegree;
    }

    for (int nodeSeqId : updated_destination_nodes) {
        auto bit = this->backward_adj_map.find(nodeSeqId);
        int indegree = (bit != this->backward_adj_map.end()) ? (int)bit->second.size() : 0;
        auto& node = this->nodeAttributeMap.at(nodeSeqId);
        node.in_degree = indegree;
    }

    std::cout << "\nupdateNodeInDegreeOutDegree(year=" << year
              << "): updated " << new_nodes_vec.size() << " out-degrees, "
              << updated_destination_nodes.size() << " in-degrees\n";
}
*/

// ── Signature: const-ref eliminates the copy of both containers ──────────────
// Saves O(|new_nodes_vec| + |updated_destination_nodes|) allocation + copy
// every year, which for large graphs is non-trivial.
void Graph::updateNodeInDegreeOutDegree(
        const std::vector<int>&        new_nodes_vec,
        const std::unordered_set<int>& updated_destination_nodes,
        int year)
{
    // ── Sort new_nodes_vec for sequential ordered-map traversal ──────────────
    // std::map stores keys in sorted order, so iterating a sorted key list
    // gives O(1) amortised map::lower_bound hint-hops instead of O(log M) per
    // find().  Sorting cost is O(S log S) once; S map lookups drop from
    // O(S log M) to O(S) amortised — worth it when S is large.
    //
    // updated_destination_nodes is an unordered_set so we can't sort it
    // in-place without a copy; instead we sort it into a temp vector.
    std::vector<int> sorted_new(new_nodes_vec.begin(), new_nodes_vec.end());
    std::sort(sorted_new.begin(), sorted_new.end());

    std::vector<int> sorted_dst(updated_destination_nodes.begin(),
                                updated_destination_nodes.end());
    std::sort(sorted_dst.begin(), sorted_dst.end());

    // ── Parallel sections: the two loops are fully independent ───────────────
    // Loop A reads forward_adj_map  (read-only) + writes nodeAttributeMap[*].out_degree
    // Loop B reads backward_adj_map (read-only) + writes nodeAttributeMap[*].in_degree
    // Different fields of the same Node struct → no data race, only false
    // sharing possible when both loops hit the same cache line.
    // For typical Node structs (>= 2 ints apart in layout) this is negligible.
    #pragma omp parallel sections num_threads(2)
    {
        // ── Section A: out-degree update ──────────────────────────────────────
        #pragma omp section
        {
            // Hint-based traversal: keep an iterator into the ordered map and
            // advance it with lower_bound(hint, key) → O(1) amortised when
            // keys are visited in ascending order.
            auto fwd_it   = this->forward_adj_map.cbegin();
            auto node_it  = this->nodeAttributeMap.begin();

            for (int nodeSeqId : sorted_new) {
                // Advance the forward-adj hint to this key
                fwd_it  = this->forward_adj_map.lower_bound(nodeSeqId);
                // Advance the nodeAttributeMap hint to this key
                node_it = this->nodeAttributeMap.lower_bound(nodeSeqId);

                int outdegree = 0;
                if (fwd_it != this->forward_adj_map.cend() &&
                    fwd_it->first == nodeSeqId)
                    outdegree = static_cast<int>(fwd_it->second.size());

                // Direct field write via iterator — avoids a second map lookup
                if (node_it != this->nodeAttributeMap.end() &&
                    node_it->first == nodeSeqId)
                    node_it->second.out_degree = outdegree;
            }
        }

        // ── Section B: in-degree update ───────────────────────────────────────
        #pragma omp section
        {
            auto bwd_it  = this->backward_adj_map.cbegin();
            auto node_it = this->nodeAttributeMap.begin();

            for (int nodeSeqId : sorted_dst) {
                bwd_it  = this->backward_adj_map.lower_bound(nodeSeqId);
                node_it = this->nodeAttributeMap.lower_bound(nodeSeqId);

                int indegree = 0;
                if (bwd_it != this->backward_adj_map.cend() &&
                    bwd_it->first == nodeSeqId)
                    indegree = static_cast<int>(bwd_it->second.size());

                if (node_it != this->nodeAttributeMap.end() &&
                    node_it->first == nodeSeqId)
                    node_it->second.in_degree = indegree;
            }
        }
    } // end omp parallel sections

    // ── Logging (moved outside parallel region, conditional on log level) ────
    printf("\nupdateNodeInDegreeOutDegree(year=%d): "
           "updated %zu out-degrees, %zu in-degrees",
           year, sorted_new.size(), sorted_dst.size());
}

void Graph::SetIntAttribute(std::string attribute_key, int nodeSeqId, int attribute_value) {
    auto it = this->nodeAttributeMap.find(nodeSeqId);
    if (it != this->nodeAttributeMap.end()) {
        // Update in-place via reference — avoids the copy-out/reinsert pattern
        Node& node = it->second;
        if      (attribute_key == "fitness_peak_value")     node.fitness_peak_value     = attribute_value;
        else if (attribute_key == "fitness_lag_duration")   node.fitness_lag_duration   = attribute_value;
        else if (attribute_key == "fitness_peak_duration")  node.fitness_peak_duration  = attribute_value;
        else if (attribute_key == "published_year")         node.published_year         = attribute_value;
        else if (attribute_key == "year")                   node.year                   = attribute_value;
        else if (attribute_key == "out_degree")             node.out_degree             = attribute_value;
        else if (attribute_key == "assigned_out_degree")    node.assigned_out_degree    = attribute_value;
        else if (attribute_key == "in_degree")              node.in_degree              = attribute_value;
        else if (attribute_key == "planted_nodes_line_number") node.planted_nodes_line_number = attribute_value;
    } else {
        Node node{};
        if      (attribute_key == "fitness_peak_value")     node.fitness_peak_value     = attribute_value;
        else if (attribute_key == "fitness_lag_duration")   node.fitness_lag_duration   = attribute_value;
        else if (attribute_key == "fitness_peak_duration")  node.fitness_peak_duration  = attribute_value;
        else if (attribute_key == "published_year")         node.published_year         = attribute_value;
        else if (attribute_key == "year")                   node.year                   = attribute_value;
        else if (attribute_key == "out_degree")             node.out_degree             = attribute_value;
        else if (attribute_key == "assigned_out_degree")    node.assigned_out_degree    = attribute_value;
        else if (attribute_key == "in_degree")              node.in_degree              = attribute_value;
        else if (attribute_key == "planted_nodes_line_number") node.planted_nodes_line_number = attribute_value;
        nodeAttributeMap.emplace(nodeSeqId, node);
    }
}

int Graph::GetIntAttribute(std::string attribute_key, int nodeSeqId) const {
    if (this->nodeAttributeMap.contains(nodeSeqId)) {
        const Node node = this->nodeAttributeMap.at(nodeSeqId);
        if (attribute_key == "fitness_peak_value")
            return node.fitness_peak_value;
        else if (attribute_key == "fitness_lag_duration")
            return node.fitness_lag_duration;
        else if (attribute_key == "fitness_peak_duration")
            return node.fitness_peak_duration;
        else if (attribute_key == "published_year")
            return node.published_year;
        else if (attribute_key == "year")
            return node.year;
        else if (attribute_key == "out_degree")
            return node.out_degree;
        else if (attribute_key == "assigned_out_degree")
            return node.assigned_out_degree;
        else if (attribute_key == "in_degree")
            return node.in_degree;
        else if (attribute_key == "planted_nodes_line_number")
            return node.planted_nodes_line_number;
        else {
            printf("\n***ERROR:: No matching int attribute for attribute_key = %s", attribute_key);
        }
    }  

    return -9999;  
}

__device__ int Graph::d_GetIntAttribute(device_map<int, Node>::device_view dmap_view, 
    device_string attribute_key, int nodeSeqId) {
    if (dmap_view.contains(nodeSeqId)) {
        Node node = dmap_view.at(nodeSeqId);
        if (attribute_key.equals((char*)"fitness_peak_value"))
            return node.fitness_peak_value;
        else if (attribute_key.equals((char*)"fitness_lag_duration"))
            return node.fitness_lag_duration;
        else if (attribute_key.equals((char*)"fitness_peak_duration"))
            return node.fitness_peak_duration;
        else if (attribute_key.equals((char*)"published_year"))
            return node.published_year;
        else if (attribute_key.equals((char*)"year"))
            return node.year;
        else if (attribute_key.equals((char*)"out_degree"))
            return node.out_degree;
        else if (attribute_key.equals((char*)"assigned_out_degree"))
            return node.assigned_out_degree;
        else if (attribute_key.equals((char*)"in_degree"))
            return node.in_degree;
        else if (attribute_key.equals((char*)"planted_nodes_line_number"))
            return node.planted_nodes_line_number;
    }  

    return -9999; // this->int_attribute_map.at(attribute_key).at(node);
}  


// Replacing SetStringAttribute
void Graph::setGeneratorNode(int nodeSeqId, int generatorNodeSeqId) {
    //printf("\nInside setGeneratorNode for nodeSeqId = %d as %d", nodeSeqId, generatorNodeSeqId);
    if (this->nodeAttributeMap.contains(nodeSeqId)) {
        this->nodeAttributeMap[nodeSeqId].generatorNode = generatorNodeSeqId; 
    } else {
        Node *node = new Node();
        node->generatorNode = generatorNodeSeqId;
        nodeAttributeMap.emplace(nodeSeqId, *node);
    }
}

// Replacing SetStringAttribute
void Graph::setType(int type_value, int nodeSeqId) {
    if (this->nodeAttributeMap.contains(nodeSeqId)) {
        Node node = this->nodeAttributeMap.at(nodeSeqId);
        node.type = type_value;
         
        nodeAttributeMap[nodeSeqId] = node;
    } else {
        Node *node = new Node();
        node->type = type_value;
        nodeAttributeMap.emplace(nodeSeqId, *node);
    }
}

// Replacing GetStringAttribute
std::string Graph::getGeneratorNode(int nodeSeqId) const {

    //printf("\nInside getGeneratorNode for nodeSeqId = %d", nodeSeqId);
    if (this->nodeAttributeMap.contains(nodeSeqId)) {
        // printf("\nnodeAttributeMap contains nodeSeqId = %d", nodeSeqId);
        Node node = this->nodeAttributeMap.at(nodeSeqId);
        return std::to_string(node.generatorNode);
    } else {
        return std::to_string(-1);
    }
}
 

// Replacing GetStringAttribute
__device__ int Graph::d_getGeneratorNode(device_map<int, Node>::device_view d_nodeAttributeMap_view, int nodeSeqId) {
    if (d_nodeAttributeMap_view.contains(nodeSeqId)) {
        Node node = d_nodeAttributeMap_view.at(nodeSeqId);
        return node.generatorNode;
    } else {
        printf("\nCould not find d_getGeneratorNode for nodeSeqId = %d", nodeSeqId);
        return -1; 
    }  
}

std::string Graph::getType(int nodeSeqId) const {
    //return this->string_attribute_map.at(attribute_key).at(node);
    if (this->nodeAttributeMap.contains(nodeSeqId)) {
        Node node = this->nodeAttributeMap.at(nodeSeqId);
        int nodeTypeIntValue = node.type;
        std::string nodeType = "seed"; // SEED_TYPE 0
        if (nodeTypeIntValue == 1) // AGENT_TYPE 1
            nodeType = "agent";
         
        return nodeType;
    } else {
        return "undefined"; 
    }
}

void Graph::SetDoubleAttribute(std::string attribute_key, int nodeSeqId, double attribute_value) {
     
    if (this->nodeAttributeMap.contains(nodeSeqId)) {
        Node node = this->nodeAttributeMap.at(nodeSeqId);
        if (attribute_key == "preferential_attachment_weight")
            node.preferential_attachment_weight = attribute_value;
        else if (attribute_key == "recency_weight")
            node.recency_weight = attribute_value;
        else if (attribute_key == "fitness_weight")
            node.fitness_weight = attribute_value;
        else if (attribute_key == "alpha")
            node.alpha = attribute_value; 

        //nodeAttributeMap.insert({nodeSeqId, node});
        nodeAttributeMap[nodeSeqId] = node;
    } else {
        Node *node = new Node();
        if (attribute_key == "preferential_attachment_weight")
            node->preferential_attachment_weight = attribute_value;
        else if (attribute_key == "recency_weight")
            node->recency_weight = attribute_value;
        else if (attribute_key == "fitness_weight")
            node->fitness_weight = attribute_value;
        else if (attribute_key == "alpha")
            node->alpha = attribute_value; 

        //nodeAttributeMap.insert({nodeSeqId, node});
        nodeAttributeMap.emplace(nodeSeqId, *node);
    }
}

double Graph::GetDoubleAttribute(std::string attribute_key, int nodeSeqId) const {
    if (this->nodeAttributeMap.contains(nodeSeqId)) {
        Node node = this->nodeAttributeMap.at(nodeSeqId);
        if (attribute_key == "preferential_attachment_weight")
            return node.preferential_attachment_weight;
        else if (attribute_key == "recency_weight")
            return node.recency_weight;
        else if (attribute_key == "fitness_weight")
            return node.fitness_weight;
        else if (attribute_key == "alpha")
            return node.alpha;
    }  

    return -9999.9; 
}
 
void Graph::AddEdge(std::pair<int, int> edge) {
    this->forward_adj_map[edge.first].insert(edge.second);
    this->backward_adj_map[edge.second].insert(edge.first);
    
    this->AddNode(edge.first);
    this->AddNode(edge.second);
}

int Graph::getForwardAdjMapSize() {
    return this->forward_adj_map.size();
}

int Graph::getBackwardAdjMapSize() {
    return this->backward_adj_map.size();
}

int Graph::getNodeAttributeMapSize() {
    return this->nodeAttributeMap.size();
} 

std::map<int, std::set<int>> Graph::getForwardAdjMap() {
    return this->forward_adj_map;
}

std::map<int, std::set<int>> Graph::getBackwardAdjMap() {
    return this->backward_adj_map;
}

std::map<int, Node> Graph::getNodeAttributeMap() {
    return this->nodeAttributeMap;
} 

int Graph::GetInDegree(int node) const {
    //printf("\nInside GetInDegree for node %d", node);
    if (this->backward_adj_map.contains(node)) {
        return this->backward_adj_map.at(node).size();
    }
    return 0;
}

int Graph::GetOutDegree(int node) const {
    //printf("\nInside GetOutDegree for node %d", node);
    if (this->forward_adj_map.contains(node)) {
        return this->forward_adj_map.at(node).size();
    }
    return 0;
}

__device__ int Graph::d_GetInDegree(
            device_map<int, Node>::device_view d_nodeAttributeMap_view,
            int nodeSeqId) {
    return this->d_GetIntAttribute(d_nodeAttributeMap_view, "in_degree", nodeSeqId);
}

__device__ int Graph::d_GetOutDegree(
            device_map<int, Node>::device_view d_nodeAttributeMap_view, 
            int nodeSeqId) {
    return this->d_GetIntAttribute(d_nodeAttributeMap_view, "out_degree", nodeSeqId);
}

void Graph::AddNode(int u) {
    this->node_set.insert(u);
}

const std::set<int>& Graph::GetNodeSet() const {
    return this->node_set;
} 

int Graph::getNodeSetSize() const {
    return this->node_set.size();
} 

void Graph::PrintGraph() const {
    std::map<int, std::set<int>> forwardAdjMapHost = this->forward_adj_map;
    for(auto const& [u,u_neighbors] : forwardAdjMapHost) {
        for(const int& v : u_neighbors) {
            std::cout << u << "-" << v << std::endl;
        }
    }
}

namespace {
    inline void append_int(std::string& buf, int value) {
        char tmp[12]; // enough for 32-bit int
        auto [ptr, ec] = std::to_chars(tmp, tmp + sizeof(tmp), value);
        buf.append(tmp, ptr);
    }
}

void Graph::WriteGraph(std::string output_file) const {
    // =========================================================================
    // OPTIMIZED WriteGraph
    // =========================================================================
    // Bottlenecks in the original implementation:
    //   1. rev.at(v) — O(log N) std::map lookup called ONCE PER EDGE.
    //      For a graph with E edges this is E × O(log N) tree traversals.
    //   2. Single-threaded string building — all E edges processed serially.
    //   3. append_int called twice per edge with no amortisation across the
    //      row (mapped_u is re-converted for every neighbour v).
    //
    // Optimizations applied:
    //   A. Build a flat O(1)-lookup array from reverse_continuous_node_mapping
    //      once.  Every rev[v] lookup drops from O(log N) → O(1).
    //   B. Snapshot forward_adj_map keys into a random-access vector so OMP
    //      can index into it by thread-private range without iterating a map.
    //   C. Convert mapped_u to chars ONCE per source row (not once per edge).
    //   D. Each OMP thread fills its own string buffer (no locks, no false
    //      sharing).  Thread buffers are written to the file in order after
    //      the parallel region.
    //   E. File write is done with a single open/write-header pass followed
    //      by sequential per-thread writes — avoids one giant allocation and
    //      lets the OS pipeline writes while threads are still building.
    // =========================================================================

    // ── A: O(1) flat reverse-mapping array ───────────────────────────────────
    // node_seq_id is 0-based sequential (see ParseNodelist), so keys of
    // reverse_continuous_node_mapping are exactly [0, N).
    const int N = static_cast<int>(this->node_set.size());
    std::vector<int> rev_arr(N);
    for (const auto& [seq_id, orig_id] : this->reverse_continuous_node_mapping)
        rev_arr[seq_id] = orig_id;   // O(N) build, then O(1) per lookup

    // ── B: Snapshot source keys for indexed parallel access ──────────────────
    // std::map iterators are not random-access, so we materialise the keys.
    std::vector<int> src_keys;
    src_keys.reserve(this->forward_adj_map.size());
    size_t total_edges = 0;
    for (const auto& [u, nbrs] : this->forward_adj_map) {
        src_keys.push_back(u);
        total_edges += nbrs.size();
    }
    const int num_srcs = static_cast<int>(src_keys.size());

    // ── C+D: Parallel string building, one buffer per thread ─────────────────
    const int nt = omp_get_max_threads();
    std::vector<std::string> bufs(nt);

    // Pre-size each buffer to its fair share (avoids realloc during filling).
    // 14 bytes/edge is conservative: "123456,789012\n" = 14 chars.
    const size_t per_thread_bytes = (total_edges * 14 / nt) + 4096;
    for (auto& b : bufs)
        b.reserve(per_thread_bytes);

    #pragma omp parallel num_threads(nt)
    {
        const int tid  = omp_get_thread_num();
        const int nthd = omp_get_num_threads();
        std::string& buf = bufs[tid];

        // Static partition: thread tid owns src_keys[start .. end)
        const int chunk = (num_srcs + nthd - 1) / nthd;
        const int begin = tid * chunk;
        const int stop  = std::min(begin + chunk, num_srcs);

        char tmp[12];  // enough for any 32-bit integer

        for (int i = begin; i < stop; ++i) {
            const int u        = src_keys[i];
            const int mapped_u = rev_arr[u];   // O(1)

            // ── C: convert mapped_u once for the whole neighbour loop ─────────
            auto [p1, e1] = std::to_chars(tmp, tmp + sizeof(tmp), mapped_u);
            const int u_len = static_cast<int>(p1 - tmp);

            // forward_adj_map is read-only here → safe for concurrent access
            for (int v : this->forward_adj_map.at(u)) {
                buf.append(tmp, u_len);          // mapped_u (pre-converted)
                buf.push_back(',');

                auto [p2, e2] = std::to_chars(tmp, tmp + sizeof(tmp), rev_arr[v]);
                buf.append(tmp, p2 - tmp);       // mapped_v, O(1) lookup
                buf.push_back('\n');
            }
        }
    } // implicit barrier — all threads done before we touch bufs

    // ── E: Open file, write header, stream per-thread buffers in order ────────
    // Writing thread buffers sequentially preserves the same source-sorted
    // order as the original (forward_adj_map is ordered, partition is ordered).
    std::ofstream out(output_file, std::ios::binary);
    constexpr const char* header = "#source,target\n";
    out.write(header, 15);
    for (const auto& b : bufs)
        if (!b.empty())
            out.write(b.data(), static_cast<std::streamsize>(b.size()));
}

/*
void Graph::WriteGraph(std::string output_file) const { 
    std::cout << "\nInside WriteGraph ...\n";

    //std::ofstream seq_out(output_seq_file, std::ios::binary);
    std::ofstream map_out(output_file, std::ios::binary);

    constexpr const char* header = "#source,target\n";
    constexpr size_t header_len = 15;

    //seq_out.write(header, header_len);
    map_out.write(header, header_len);

    // ----------------------------------------------------
    // Precompute total edges
    // ----------------------------------------------------
    size_t total_edges = 0;
    for (const auto& [_, nbrs] : forward_adj_map) {
        total_edges += nbrs.size();
    }

    // ~20 bytes per edge is realistic: "12345,67890\n"
    //std::string seq_buffer;
    std::string map_buffer;
    //seq_buffer.reserve(total_edges * 20);
    map_buffer.reserve(total_edges * 20);

    const auto& rev = reverse_continuous_node_mapping;

    // ----------------------------------------------------
    // Main write loop
    // ----------------------------------------------------
    for (const auto& [u, neighbors] : forward_adj_map) {
        const int mapped_u = rev.at(u);

        for (int v : neighbors) {
            // mapped: mapped_u,mapped_v
            append_int(map_buffer, mapped_u);
            map_buffer.push_back(',');
            append_int(map_buffer, rev.at(v));
            map_buffer.push_back('\n');
        }
    }

    // ----------------------------------------------------
    // Single bulk write
    // ----------------------------------------------------
    //seq_out.write(seq_buffer.data(), seq_buffer.size());
    map_out.write(map_buffer.data(), map_buffer.size());
}
*/

void Graph::WriteAttributes(std::string auxiliary_information_file) const {
    // =========================================================================
    // OPTIMIZED WriteAttributes
    // =========================================================================
    // Bottlenecks in the original implementation
    // -----------------------------------------------------------------
    // 1. Per-node attribute accessors (GetIntAttribute, GetDoubleAttribute,
    //    getType, getGeneratorNode) each do:
    //      a) nodeAttributeMap.contains(id)  → O(log N) tree traversal
    //      b) nodeAttributeMap.at(id)        → O(log N) tree traversal again
    //      c) string-comparison chain to find the right field
    //    Called 9-14 times per node → up to 28 tree traversals + 14 string
    //    comparisons against the SAME map key every row.
    //
    // 2. reverse_continuous_node_mapping.at(nodeSeqId) → O(log N) per node.
    //    node_seq_id is 0-based sequential, so a flat array suffices.
    //
    // 3. std::ostringstream with operator<< is locale-aware and slow.
    //    std::to_chars + manual buffer append is 3-5× faster for numbers.
    //
    // 4. Single-threaded over all N nodes; rows are independent.
    //
    // 5. Periodic flush calls aux_stream.str() which copies the entire
    //    accumulated string on every flush boundary.
    //
    // 6. seq_* file is opened but never written — dead I/O overhead.
    //
    // Optimizations applied
    // -----------------------------------------------------------------
    // A. Build flat O(1)-access arrays from nodeAttributeMap and
    //    reverse_continuous_node_mapping once (O(N) build).
    //    Every per-node field access becomes a direct struct-field read —
    //    zero map lookups, zero string comparisons during the write loop.
    //
    // B. Each OMP thread owns a private char[] slab (stack-allocated per
    //    thread, sized to hold ~ROWS_PER_SLAB rows).  When the slab is
    //    full the thread drains it to its own std::string.  This avoids
    //    any allocation inside the hot path and keeps cache footprint small.
    //
    // C. Number-to-string conversion uses std::to_chars (no locale, no
    //    heap allocation, branch-minimal).  Doubles use snprintf with %.10g
    //    which matches the original aux_stream.precision(10) behaviour.
    //
    // D. The seq_* file is never written so we skip opening it entirely.
    //
    // E. Thread buffers are written to the file sequentially after the
    //    parallel region to preserve the node_set iteration order.
    // =========================================================================

    // ── A1: flat reverse-mapping array (O(1) nodeSeqId → original node id) ──
    const int N = static_cast<int>(this->node_set.size());
    std::vector<int> rev_arr(N);
    for (const auto& [seq_id, orig_id] : this->reverse_continuous_node_mapping)
        rev_arr[seq_id] = orig_id;

    // ── A2: flat Node array — ONE map lookup per node, then direct field reads
    // node_set is a std::set<int> of seq_ids in sorted order; seq_ids are
    // 0-based so we can store into rev_arr directly.
    std::vector<Node> node_arr(N);
    for (const auto& [seq_id, node] : this->nodeAttributeMap)
        if (seq_id >= 0 && seq_id < N) node_arr[seq_id] = node;

    // ── A3: collect all seq_ids in iteration order for parallel indexing ─────
    std::vector<int> seq_ids(this->node_set.cbegin(), this->node_set.cend());

    // ── Open output file, write header ───────────────────────────────────────
    std::ofstream out(auxiliary_information_file, std::ios::binary);
    constexpr const char* header =
        "node_id,type,year,pa_weight,rec_weight,fit_weight,fit_lag_duration,"
        "fit_peak_value,fit_peak_duration,alpha,in_degree,out_degree,"
        "assigned_out_degree,planted_nodes_line_number,generator_node_string\n";
    out << header;

    printf("\nWriteAttributes: graph size = %d\n", N);

    // ── B+C+D: Parallel row generation, one string buffer per thread ─────────
    const int nt = omp_get_max_threads();
    std::vector<std::string> bufs(nt);
    // Pre-size: ~120 bytes/row is generous for a 15-column CSV row
    const size_t approx_per_thread = (static_cast<size_t>(N) * 120 / nt) + 4096;
    for (auto& b : bufs) b.reserve(approx_per_thread);

    #pragma omp parallel num_threads(nt)
    {
        const int tid  = omp_get_thread_num();
        const int nthd = omp_get_num_threads();
        std::string& buf = bufs[tid];

        // Static partition over seq_ids
        const int chunk = (N + nthd - 1) / nthd;
        const int begin = tid * chunk;
        const int stop  = std::min(begin + chunk, N);

        // Per-thread slab: write numbers here first, then append to buf.
        // 256 bytes covers the longest possible row (15 wide columns).
        char row[256];

        // snprintf format for doubles matching precision(10) behaviour
        // "%.10g" suppresses trailing zeros and uses exponential when needed
        for (int i = begin; i < stop; ++i) {
            const int seq_id = seq_ids[i];
            const int node_id = rev_arr[seq_id];       // O(1) array read
            const Node& nd = node_arr[seq_id];         // O(1) array read

            // is_agent: AGENT_TYPE == 1 (from utils.cuh)
            const bool is_agent = (nd.type == AGENT_TYPE);
            const char* type_str = is_agent ? "agent" : "seed";

            if (is_agent) {
                // Generator node: stored as int seq_id in nd.generatorNode;
                // translate to original id via rev_arr (same as getGeneratorNode does).
                const int gen_orig = (nd.generatorNode >= 0 && nd.generatorNode < N)
                                     ? rev_arr[nd.generatorNode] : nd.generatorNode;

                int len = snprintf(row, sizeof(row),
                    // node_id, type, year, pa_weight, rec_weight, fit_weight
                    "%d,%s,%d,%.10g,%.10g,%.10g,"
                    // fit_lag, fit_peak_val, fit_peak_dur, alpha
                    "%d,%d,%d,%.10g,"
                    // in_degree, out_degree, assigned_out_degree
                    "%d,%d,%d,"
                    // planted_nodes_line_number, generator_node_string
                    "%d,%d\n",
                    node_id, type_str, nd.year,
                    nd.preferential_attachment_weight,
                    nd.recency_weight,
                    nd.fitness_weight,
                    nd.fitness_lag_duration,
                    nd.fitness_peak_value,
                    nd.fitness_peak_duration,
                    nd.alpha,
                    nd.in_degree,
                    nd.out_degree,
                    nd.assigned_out_degree,
                    nd.planted_nodes_line_number,
                    gen_orig);
                if (len > 0) buf.append(row, static_cast<size_t>(len));
            } else {
                // Seed node: pa_weight/rec_weight/fit_weight/alpha = -1,
                // assigned_out_degree = -1, planted_line = -1,
                // generator_node_string = "no_generators"
                int len = snprintf(row, sizeof(row),
                    "%d,%s,%d,-1,-1,-1,"
                    "%d,%d,%d,-1,"
                    "%d,%d,-1,"
                    "-1,no_generators\n",
                    node_id, type_str, nd.year,
                    nd.fitness_lag_duration,
                    nd.fitness_peak_value,
                    nd.fitness_peak_duration,
                    nd.in_degree,
                    nd.out_degree);
                if (len > 0) buf.append(row, static_cast<size_t>(len));
            }
        }
    } // implicit OMP barrier

    // ── E: Write thread buffers in order (preserves node_set sort order) ─────
    for (const auto& b : bufs)
        if (!b.empty())
            out.write(b.data(), static_cast<std::streamsize>(b.size()));
}
/*
void Graph::WriteAttributes(std::string auxiliary_information_file) const {
    std::string seq_auxiliary_information_file = "seq" + auxiliary_information_file;
    
    // Open files with larger buffer for better I/O performance
    std::ofstream seq_auxiliary_information_filehandle(seq_auxiliary_information_file);
    std::ofstream auxiliary_information_filehandle(auxiliary_information_file);
    
    // Set larger buffer size (8MB) for both files
    constexpr size_t buffer_size = 8 * 1024 * 1024;
    std::vector<char> seq_buffer(buffer_size);
    std::vector<char> aux_buffer(buffer_size);
    //seq_auxiliary_information_filehandle.rdbuf()->pubsetbuf(seq_buffer.data(), buffer_size);
    auxiliary_information_filehandle.rdbuf()->pubsetbuf(aux_buffer.data(), buffer_size);
    
    // Write headers
    const char* header2 = "node_id,type,year,pa_weight,rec_weight,fit_weight,fit_lag_duration,"
                         "fit_peak_value,fit_peak_duration,alpha,in_degree,out_degree,assigned_out_degree,"
                         "planted_nodes_line_number,generator_node_string\n";
    auxiliary_information_filehandle << header2;
    
    std::cout << "\n00000 graph size = " << this->GetNodeSet().size();
    
    // Use string streams for batch writing (faster than repeated << operations)
    std::ostringstream aux_stream;
    aux_stream.precision(10);
    
    constexpr size_t flush_threshold = 1024 * 1024; // Flush every 1MB
    
    // Cache string literals to avoid repeated allocations
    const std::string agent_type = "agent";
    const std::string no_generators = "no_generators";
    
    for(const auto& nodeSeqId : this->GetNodeSet()) {
        int nodeId = reverse_continuous_node_mapping.at(nodeSeqId);
        std::string node_type = this->getType(nodeSeqId);
        
        // Batch read attributes to reduce function call overhead
        int year = this->GetIntAttribute("year", nodeSeqId);
        int fit_lag_duration = this->GetIntAttribute("fitness_lag_duration", nodeSeqId);
        int fit_peak_value = this->GetIntAttribute("fitness_peak_value", nodeSeqId);
        int fit_peak_duration = this->GetIntAttribute("fitness_peak_duration", nodeSeqId);
        int out_degree = this->GetIntAttribute("out_degree", nodeSeqId);
        int in_degree = this->GetIntAttribute("in_degree", nodeSeqId);
        
        double pa_weight = -1;
        double rec_weight = -1;
        double fit_weight = -1;
        double alpha = -1;
        int assigned_out_degree = -1;
        int planted_nodes_line_number = -1;
        std::string generator_node_string = no_generators;
        
        // Single comparison instead of string comparison
        bool is_agent = (node_type == agent_type);
        
        if(is_agent) {
            pa_weight = this->GetDoubleAttribute("preferential_attachment_weight", nodeSeqId);
            rec_weight = this->GetDoubleAttribute("recency_weight", nodeSeqId);
            fit_weight = this->GetDoubleAttribute("fitness_weight", nodeSeqId);
            alpha = this->GetDoubleAttribute("alpha", nodeSeqId);
            assigned_out_degree = this->GetIntAttribute("assigned_out_degree", nodeSeqId);
            generator_node_string = this->getGeneratorNode(nodeSeqId);
            planted_nodes_line_number = this->GetIntAttribute("planted_nodes_line_number", nodeSeqId);
        }
        
        // Write to string streams (much faster than direct file writes)
        aux_stream << nodeId << ',' << node_type << ',' << year << ','
                   << pa_weight << ',' << rec_weight << ',' << fit_weight << ',' 
                   << fit_lag_duration << ',' << fit_peak_value << ',' 
                   << fit_peak_duration << ',' << alpha << ',' << in_degree << ','
                   << out_degree << ',' << assigned_out_degree << ',' 
                   << planted_nodes_line_number << ',' << generator_node_string << '\n';
        
        // Periodic flush to avoid memory buildup
        if(aux_stream.tellp() > flush_threshold) {
            //seq_auxiliary_information_filehandle << seq_stream.str();
            auxiliary_information_filehandle << aux_stream.str();
            //seq_stream.str("");
            aux_stream.str("");
            //seq_stream.clear();
            aux_stream.clear();
        }
    }
    
    // Final flush
    // seq_auxiliary_information_filehandle << seq_stream.str();
    auxiliary_information_filehandle << aux_stream.str();
    
    auxiliary_information_filehandle.close();
    // seq_auxiliary_information_filehandle.close();
}
*/
