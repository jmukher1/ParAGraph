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


void Graph::updateNodeInDegreeOutDegree(std::vector<int> new_nodes_vec,
                                    std::set<int> updated_destination_nodes, 
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
            /*/ seq: u,v
            append_int(seq_buffer, u);
            seq_buffer.push_back(',');
            append_int(seq_buffer, v);
            seq_buffer.push_back('\n');*/

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

/*void Graph::WriteGraph(std::string output_file) const {
    std::string output_seq_file = "seq_" + output_file;
    std::cout << "\nInside WriteGraph (seq)...\n";

    // Open files in binary mode
    std::ofstream seq_output_filehandle(output_seq_file, std::ios::out | std::ios::binary);
    std::ofstream output_filehandle(output_file, std::ios::out | std::ios::binary);

    // Write headers
    constexpr const char* header = "#source,target\n";
    seq_output_filehandle.write(header, 15);
    output_filehandle.write(header, 15);

    // Pre-calculate total edges for better memory allocation
    size_t total_edges = 0;
    for (const auto& [u, u_neighbors] : forward_adj_map) {
        total_edges += u_neighbors.size();
    }

    // Allocate buffers with more accurate sizing (avg ~20 bytes per edge)
    std::string seq_buffer;
    std::string map_buffer;
    seq_buffer.reserve(total_edges * 20);
    map_buffer.reserve(total_edges * 20);

    // Use faster integer to string conversion
    char int_buffer[12]; // Enough for 32-bit integers

    for (const auto& [u, u_neighbors] : forward_adj_map) {
        // Look up mapped_u once per outer loop
        int mapped_u = reverse_continuous_node_mapping.at(u);
        
        // Convert u and mapped_u to string once
        int u_len = snprintf(int_buffer, sizeof(int_buffer), "%d", u);
        std::string u_str(int_buffer, u_len);
        
        int mapped_u_len = snprintf(int_buffer, sizeof(int_buffer), "%d", mapped_u);
        std::string mapped_u_str(int_buffer, mapped_u_len);

        for (int v : u_neighbors) {
            // Sequential file: u,v
            seq_buffer.append(u_str);
            seq_buffer.push_back(',');
            
            int v_len = snprintf(int_buffer, sizeof(int_buffer), "%d", v);
            seq_buffer.append(int_buffer, v_len);
            seq_buffer.push_back('\n');

            // Mapped file: mapped_u,mapped_v
            map_buffer.append(mapped_u_str);
            map_buffer.push_back(',');
            
            int mapped_v = reverse_continuous_node_mapping.at(v);
            int mapped_v_len = snprintf(int_buffer, sizeof(int_buffer), "%d", mapped_v);
            map_buffer.append(int_buffer, mapped_v_len);
            map_buffer.push_back('\n');
        }
    }

    // Single write operation
    seq_output_filehandle.write(seq_buffer.data(), seq_buffer.size());
    output_filehandle.write(map_buffer.data(), map_buffer.size());
    
    seq_output_filehandle.close();
    output_filehandle.close();
}*/

/*void Graph::WriteGraph(std::string output_file) const {
    
    std::string output_seq_file = "seq_" + output_file;
    std::cout << "\nInside WriteGraph (seq)...\n";

    std::ofstream seq_output_filehandle(output_seq_file, std::ios::out | std::ios::binary);
    std::ofstream output_filehandle(output_file, std::ios::out | std::ios::binary);

    // Pre–write header without flushing
    seq_output_filehandle << "#source,target\n";
    output_filehandle    << "#source,target\n";

    std::string seq_buffer;
    std::string map_buffer;

    // Reserve roughly the needed size to avoid reallocation
    // (assuming avg degree 8; adjust as needed)
    size_t approx_edges = 8ull * forward_adj_map.size();
    seq_buffer.reserve(approx_edges * 12); 
    map_buffer.reserve(approx_edges * 12);

    for (const auto& [u, u_neighbors] : forward_adj_map) {

        int mapped_u = reverse_continuous_node_mapping.at(u);

        for (int v : u_neighbors) {

            // accumulate text in RAM, no flushing per-line
            seq_buffer.append(std::to_string(u));
            seq_buffer.push_back(',');
            seq_buffer.append(std::to_string(v));
            seq_buffer.push_back('\n');

            int mapped_v = reverse_continuous_node_mapping.at(v);

            map_buffer.append(std::to_string(mapped_u));
            map_buffer.push_back(',');
            map_buffer.append(std::to_string(mapped_v));
            map_buffer.push_back('\n');
        }
    }

    // One big write ― fastest possible
    seq_output_filehandle.write(seq_buffer.data(),  seq_buffer.size());
    output_filehandle.write(map_buffer.data(), map_buffer.size());
    output_filehandle.close();
    seq_output_filehandle.close();
}*/

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
    //const char* header = "node_seq_id,type,year,pa_weight,rec_weight,fit_weight,fit_lag_duration,"
                        "fit_peak_value,fit_peak_duration,alpha,in_degree,out_degree,assigned_out_degree,"
                        "planted_nodes_line_number,generator_node_string\n";
    //seq_auxiliary_information_filehandle << header;
    
    const char* header2 = "node_id,type,year,pa_weight,rec_weight,fit_weight,fit_lag_duration,"
                         "fit_peak_value,fit_peak_duration,alpha,in_degree,out_degree,assigned_out_degree,"
                         "planted_nodes_line_number,generator_node_string\n";
    auxiliary_information_filehandle << header2;
    
    std::cout << "\n00000 graph size = " << this->GetNodeSet().size();
    
    // Use string streams for batch writing (faster than repeated << operations)
    std::ostringstream /*seq_stream,*/ aux_stream;
    //seq_stream.precision(10);
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
        
        /*seq_stream << nodeSeqId << ',' << node_type << ',' << year << ','
                   << pa_weight << ',' << rec_weight << ',' << fit_weight << ',' 
                   << fit_lag_duration << ',' << fit_peak_value << ',' 
                   << fit_peak_duration << ',' << alpha << ',' << in_degree << ','
                   << out_degree << ',' << assigned_out_degree << ',' 
                   << planted_nodes_line_number << ',' << generator_node_string << '\n'; */
        
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