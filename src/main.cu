#include <iostream>

#include "abm.cuh"
#include "int2.cuh"
#include "kernel_erdos_renyi.cuh"
#include "argparse.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

int main(int argc, char* argv[]) {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    printf("\nStart ABM....");
    argparse::ArgumentParser main_program("abm");

    main_program.add_description("Agent Based Modelling");

    main_program.add_argument("--edgelist")
        .required()
        .help("Input edgelist (source, target)");
    main_program.add_argument("--nodelist")
        .required()
        .help("Input nodelist (node, year)");
    main_program.add_argument("--out-degree-bag")
        .required()
        .help("Input out-degree bag (, out-degree)");
    main_program.add_argument("--recency-probabilities")
        .required()
        .help("Input recency bag (year, probability)");
    main_program.add_argument("--planted-nodes")
        .default_value("")
        .help("Planted nodes file (year, fitness lag duration, fitness peak value, fitess peak duration, count)");
    main_program.add_argument("--preferential-weight")
        .default_value(double(-1))
        .help("Preferential attachment weight")
        .scan<'g', double>();
    main_program.add_argument("--recency-weight")
        .default_value(double(-1))
        .help("Recency weight")
        .scan<'g', double>();
    main_program.add_argument("--fitness-weight")
        .default_value(double(-1))
        .help("Fitness weight")
        .scan<'g', double>();
    main_program.add_argument("--alpha")
        .default_value(double(-1))
        .help("Neighborhood alpha")
        .scan<'g', double>();
    main_program.add_argument("--fully-random-citations")
        .default_value(double(0.05))
        .help("Constant percentage for radom citations")
        .scan<'g', double>();
    main_program.add_argument("--growth-rate")
        .required()
        .help("Growth rate")
        .scan<'g', double>();
    main_program.add_argument("--num-cycles")
        .required()
        .help("Number of years")
        .scan<'d', int>();
    main_program.add_argument("--same-year-proportion")
        .required()
        .help("Growth rate")
        .scan<'g', double>();
    main_program.add_argument("--output-file")
        .required()
        .help("Output clustering file");
    main_program.add_argument("--auxiliary-information-file")
        .required()
        .help("Auxillary information file");
    main_program.add_argument("--log-file")
        .required()
        .help("Output log file");
    main_program.add_argument("--num-processors")
        .default_value(int(1))
        .help("Number of processors")
        .scan<'d', int>();
    main_program.add_argument("--log-level")
        .default_value(int(1))
        .help("Log level where 0 = silent, 1 = info, 2 = verbose")
        .scan<'d', int>();
    main_program.add_argument("--model")
        .default_value(std::string("pa"))
        .help("Network model: pa (Preferential Attachment) or ER (Erdos-Renyi)");
    main_program.add_argument("--er-probability")
        .default_value(double(0.0))
        .help("Erdos-Renyi: edge probability p for G(n,p) model")
        .scan<'g', double>();
    main_program.add_argument("--er-edges-per-node")
        .default_value(int(0))
        .help("Erdos-Renyi: average edges per new node (alternative to edge probability)")
        .scan<'d', int>();
    printf("\nparse_args....");
    try {
        main_program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << main_program;
        std::exit(1);
    }

    printf("\nedgelist....");
    std::string edgelist = main_program.get<std::string>("--edgelist");
    std::string nodelist = main_program.get<std::string>("--nodelist");
    std::string out_degree_bag = main_program.get<std::string>("--out-degree-bag");
    std::string recency_probabilities = main_program.get<std::string>("--recency-probabilities");
    std::string planted_nodes = main_program.get<std::string>("--planted-nodes");
    double alpha = main_program.get<double>("--alpha");
    double fully_random_citations = main_program.get<double>("--fully-random-citations");
    double preferential_weight = main_program.get<double>("--preferential-weight");
    double recency_weight = main_program.get<double>("--recency-weight");
    double fitness_weight = main_program.get<double>("--fitness-weight");
    double growth_rate = main_program.get<double>("--growth-rate");
    int num_cycles = main_program.get<int>("--num-cycles");
    printf("\nnum-cycles....");
    double same_year_proportion = main_program.get<double>("--same-year-proportion");
    std::string output_file = main_program.get<std::string>("--output-file");
    printf("\noutput-file....");
    std::string auxiliary_information_file = main_program.get<std::string>("--auxiliary-information-file");
    std::string log_file = main_program.get<std::string>("--log-file");
    printf("\nlog-file = %s....", log_file.c_str());
    int num_processors = main_program.get<int>("--num-processors");
    printf("\nnum-processors = %d....", num_processors);
    int log_level = main_program.get<int>("--log-level") - 1; // so that enum is cleaner
    printf("\nlog-level = %d", log_level);
    std::string model = main_program.get<std::string>("--model");
    double er_probability = main_program.get<double>("--er-probability");
    printf("\ner_probability = %lf", er_probability);
    int er_edges_per_node = main_program.get<int>("--er-edges-per-node");
    printf("\nmodel = %s", model.c_str());
    printf("\nInit ABM....");
    ABM* abm = new ABM(edgelist, nodelist, out_degree_bag, recency_probabilities, planted_nodes, alpha, fully_random_citations, preferential_weight, recency_weight, fitness_weight, growth_rate, num_cycles, same_year_proportion, output_file, auxiliary_information_file, log_file, num_processors, log_level, model, er_probability, er_edges_per_node);
    printf("\nExec ABM....");
    // Branch between the PA model's orchestration (execute(), kernel.cu)
    // and the ER model's orchestration (executeER(), kernel_erdos_renyi.cu)
    // based on --network-model. Previously, launchErdosRenyiKernel/
    // kernelErdosRenyiGNP/kernelErdosRenyiFixedK were fully implemented
    // but never called from anywhere -- the ER model was unreachable
    // regardless of this flag's value. executeER() (kernel_erdos_renyi.cu)
    // now provides the missing orchestration loop.
    if (abm->get_model() == "pa") {
        execute(abm);
    } else {
        executeER(abm);
    }

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    auto durationE2E = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    std::cout << "\nE2E Time for num_cycles = "<< num_cycles << " with growth_rate = "<< (100*growth_rate) << " % = " << durationE2E.count()/1000 << " seconds." << std::endl;
    delete abm;
}