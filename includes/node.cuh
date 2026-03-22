#ifndef NODE_CUH_
#define NODE_CUH_ 

struct Node {
        // It may need long device_string
        // device_string type;
        int type;
        int generatorNode;
        int fitness_peak_value;
        int fitness_lag_duration = 0;
        int fitness_peak_duration = 1000;
        int published_year;
        int year;
        int out_degree;
        int assigned_out_degree;
        int in_degree;
        int planted_nodes_line_number;

        double preferential_attachment_weight;
        double recency_weight;
        double fitness_weight;
        double alpha;
};

// must be trivially copyable
static_assert(std::is_trivially_copyable_v<Node>, "Node must be trivially copyable");

namespace cuco {
template <>
struct is_bitwise_comparable<Node> : std::true_type {};
}
#endif 