#pragma once
#ifndef GRAPH_H
#define GRAPH_H

#include <set>
#include <map>
#include <vector>

#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>

#include "device_map.cuh"
#include "device_set.cuh"
#include "device_string.cuh"
#include "utils.cuh"

using namespace std; 

#include "node.cuh"

class Graph {
    public:
        //Graph() {};
        Graph(std::string edgelist, std::string nodelist);
        
        void AddEdge(std::pair<int, int> edge);
        //__device__ void d_AddEdge(thrust::pair<int, int> edge);

        static inline char get_delimiter(std::string filepath) {
            std::ifstream edgelistFromFile(filepath);
            std::string line;
            getline(edgelistFromFile, line);
            if (line.find(',') != std::string::npos) {
                return ',';
            } else if (line.find('\t') != std::string::npos) {
                return '\t';
            } else if (line.find(' ') != std::string::npos) {
                return ' ';
            }
            throw std::invalid_argument("Could not detect filetype for " + filepath);
        }


        const std::set<int>& GetNodeSet() const;
        int getNodeSetSize() const;
        const device_set<int>* d_GetNodeSet() const;
        void SetIntAttribute(std::string attribute_key, int node, int attribute_value);
        int GetIntAttribute(std::string attribute_key, int node) const;
        __device__ int d_GetIntAttribute(device_map<int, Node>::device_view d_nodeAttributeMap_view,
            device_string attribute_key, int nodeId);
        
        void SetDoubleAttribute(std::string attribute_key, int node, double attribute_value);
        double GetDoubleAttribute(std::string attribute_key, int node) const;
        __device__ double d_GetDoubleAttribute(device_map<int, Node>::device_view dmap_view,
            std::string attribute_key, int nodeId);
        //bool HasIntAttribute(std::string attribute_key, int node) const;
        void ParseNodelist();
        void ParseEdgelist();
        void updateNodeInDegreeOutDegree();
        void updateNodeInDegreeOutDegree(std::vector<int> new_nodes_vec,
                                        std::set<int> updated_destination_nodes,
                                        int year);
        //__host__ __device__ const device_map<int, Node>::device_view getd_nodeAttributeMap_view();
        std::map<int, std::set<int>> getForwardAdjMap();
        std::map<int, std::set<int>> getBackwardAdjMap();
        std::map<int, Node> getNodeAttributeMap();

        int getForwardAdjMapSize();
        int getBackwardAdjMapSize();
        int getNodeAttributeMapSize();

        int GetInDegree(int node) const;
        int GetOutDegree(int node) const;
        void AddNode(int u);
        //__device__ void dAddNode(int u);
        //__host__ __device__ void populateHostMapToDeviceMap(int num_cycles, double growth_rate);
        void PrintGraph() const;
        void WriteGraph(std::string output_file) const;
        void WriteAttributes(std::string auxiliary_information_file) const;
    
        __device__ int d_GetInDegree(
                            device_map<int, Node>::device_view d_nodeAttributeMap_view,
                            int node);
        __device__ int d_GetOutDegree(
                            device_map<int, Node>::device_view d_nodeAttributeMap_view, 
                            int node);
        
        void setGeneratorNode(int nodeId, int generatorNode);
        void setType(int type_value, int nodeId);
        std::string getGeneratorNode(int nodeId)  const;
        __device__ int d_getGeneratorNode(device_map<int, Node>::device_view d_nodeAttributeMap_view, int nodeId) ;
        std::string getType(int nodeId) const; 
        std::map<int, int> getContinuousNodeMapping();
        std::map<int, int> getReverseContinuousNodeMapping();

        
    private:
        std::string edgelist;
        std::string nodelist;
        std::set<int> node_set;
        std::map<int, std::set<int>> forward_adj_map;
        std::map<int, std::set<int>> backward_adj_map;
        int node_seq_id = 0;
    public:
        std::map<int, Node> nodeAttributeMap;
        std::map<int, int> continuous_node_mapping;
        std::map<int, int> reverse_continuous_node_mapping;

 
    //public:
        //device_map<int, Node>* d_nodeAttributeMap;
};

#endif
