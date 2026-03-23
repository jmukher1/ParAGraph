#include <iostream>

#include "argparse.h"
#include "abm.h"


int main(int argc, char* argv[]) {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    argparse::ArgumentParser main_program("citation_models");

    main_program.add_description("Citation Network Growth Models: Preferential Attachment & Erdős-Rényi");

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
    
    // Model selection
    main_program.add_argument("--model")
        .default_value(std::string("pa"))
        .help("Growth model: \n"
              "  pa      - Preferential Attachment (default)\n"
              "  er      - Erdős-Rényi fixed-k (samples k neighbors uniformly)\n"
              "  er-gnp  - Erdős-Rényi G(n,p) (each edge with probability p)");
    
    // PA model parameters
    main_program.add_argument("--preferential-weight")
        .default_value(double(-1))
        .help("Preferential attachment weight (PA model only, -1 = random)")
        .scan<'g', double>();
    main_program.add_argument("--recency-weight")
        .default_value(double(-1))
        .help("Recency weight (PA model only, -1 = random)")
        .scan<'g', double>();
    main_program.add_argument("--fitness-weight")
        .default_value(double(-1))
        .help("Fitness weight (PA model only, -1 = random)")
        .scan<'g', double>();
    main_program.add_argument("--alpha")
        .default_value(double(-1))
        .help("Neighborhood alpha (PA model only, -1 = random)")
        .scan<'g', double>();
    main_program.add_argument("--fully-random-citations")
        .default_value(double(0.05))
        .help("Constant percentage for random citations (PA model only)")
        .scan<'g', double>();
    
    // ER model parameters
    main_program.add_argument("--er-probability")
        .default_value(double(0.01))
        .help("Edge probability p for --model er-gnp (ignored for pa and er)")
        .scan<'g', double>();
    
    // Common parameters
    main_program.add_argument("--growth-rate")
        .required()
        .help("Growth rate (fraction of nodes to add each cycle)")
        .scan<'g', double>();
    main_program.add_argument("--num-cycles")
        .required()
        .help("Number of years/cycles to simulate")
        .scan<'d', int>();
    main_program.add_argument("--same-year-proportion")
        .required()
        .help("Proportion of same-year citations")
        .scan<'g', double>();
    main_program.add_argument("--output-file")
        .required()
        .help("Output edgelist file");
    main_program.add_argument("--auxiliary-information-file")
        .required()
        .help("Auxiliary information file (node attributes)");
    main_program.add_argument("--log-file")
        .required()
        .help("Output log file");
    main_program.add_argument("--num-processors")
        .default_value(int(1))
        .help("Number of OpenMP processors")
        .scan<'d', int>();
    main_program.add_argument("--log-level")
        .default_value(int(1))
        .help("Log level: 0 = silent, 1 = info, 2 = verbose")
        .scan<'d', int>();
    
    try {
        main_program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << main_program;
        std::exit(1);
    }

    // Parse arguments
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
    double same_year_proportion = main_program.get<double>("--same-year-proportion");
    std::string output_file = main_program.get<std::string>("--output-file");
    std::string auxiliary_information_file = main_program.get<std::string>("--auxiliary-information-file");
    std::string log_file = main_program.get<std::string>("--log-file");
    int num_processors = main_program.get<int>("--num-processors");
    int log_level = main_program.get<int>("--log-level") - 1; // so that enum is cleaner
    std::string model = main_program.get<std::string>("--model");
    double er_probability = main_program.get<double>("--er-probability");
    
    // Validate model selection
    if (model != "pa" && model != "er" && model != "er-gnp") {
        std::cerr << "Error: --model must be one of: pa, er, er-gnp" << std::endl;
        std::exit(1);
    }
    
    // Print model configuration
    std::cout << "========================================" << std::endl;
    std::cout << "Citation Network Growth Simulation" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Model: ";
    if (model == "pa") {
        std::cout << "Preferential Attachment (PA)" << std::endl;
        std::cout << "  - Uses degree, recency, and fitness" << std::endl;
        std::cout << "  - Alpha (neighborhood): " << (alpha < 0 ? "random" : std::to_string(alpha)) << std::endl;
    } else if (model == "er") {
        std::cout << "Erdős-Rényi Fixed-k" << std::endl;
        std::cout << "  - Each new node cites k neighbors uniformly at random" << std::endl;
    } else if (model == "er-gnp") {
        std::cout << "Erdős-Rényi G(n,p)" << std::endl;
        std::cout << "  - Each edge created with probability p = " << er_probability << std::endl;
    }
    std::cout << "Growth rate: " << growth_rate << std::endl;
    std::cout << "Num cycles: " << num_cycles << std::endl;
    std::cout << "Output: " << output_file << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Create and run ABM
    ABM* abm = new ABM(edgelist, nodelist, out_degree_bag, recency_probabilities, 
                       planted_nodes, alpha, fully_random_citations, preferential_weight, 
                       recency_weight, fitness_weight, growth_rate, num_cycles, 
                       same_year_proportion, output_file, auxiliary_information_file, 
                       log_file, num_processors, log_level, model, er_probability);
    abm->main();
    delete abm;
    
    std::cout << "========================================" << std::endl;
    std::cout << "Simulation completed successfully!" << std::endl;
    std::cout << "========================================" << std::endl;
    std::ostringstream msg;
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    auto durationE2E = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    msg << "\nE2E Time (CPU-Models), model:" << model << " for num_cycles = " << num_cycles
        << " with growth_rate = " << (100 * growth_rate)
        << " % with thread count = " << num_processors
        << " is: " << durationE2E.count() / 1000
        << " seconds.";

    std::cout << msg.str() << std::endl;
    return 0;
}
