#include <iostream>
#include <chrono>
#include <sstream>

#include "argparse.h"
#include "abm.h"


int main(int argc, char* argv[]) {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

    argparse::ArgumentParser main_program("citation_models");
    main_program.add_description(
        "Citation Network Growth Models\n"
        "  Preferential Attachment (PA) and Erdős-Rényi (ER / ER-GNP)\n"
        "  Outputs: edgelist + auxiliary node-attribute file identical to the\n"
        "  cpu (PA-only) reference implementation.");

    // ── Required inputs ───────────────────────────────────────────────────────
    main_program.add_argument("--edgelist")
        .required()
        .help("Input edgelist CSV (source, target)");
    main_program.add_argument("--nodelist")
        .required()
        .help("Input nodelist CSV (node_id, year)");
    main_program.add_argument("--out-degree-bag")
        .required()
        .help("Out-degree bag CSV (index, out_degree)");
    main_program.add_argument("--recency-probabilities")
        .required()
        .help("Recency probability CSV (year_diff, probability)");
    main_program.add_argument("--planted-nodes")
        .default_value(std::string(""))
        .help("Planted-nodes CSV (year, lag, peak_value, peak_duration, count)  [optional]");

    // ── Model selection ───────────────────────────────────────────────────────
    main_program.add_argument("--model")
        .default_value(std::string("pa"))
        .help(
            "Growth model (default: pa)\n"
            "  pa      Preferential Attachment  – degree × recency × fitness WRS\n"
            "  er      Erdős-Rényi fixed-k      – k uniform random targets, same degree seq as PA\n"
            "  er-gnp  Erdős-Rényi G(n,p)       – each pre-existing node cited with prob p");

    // ── PA-specific parameters ────────────────────────────────────────────────
    main_program.add_argument("--preferential-weight")
        .default_value(double(-1))
        .help("PA score weight  (-1 = sample uniformly per node)")
        .scan<'g', double>();
    main_program.add_argument("--recency-weight")
        .default_value(double(-1))
        .help("Recency score weight  (-1 = sample uniformly per node)")
        .scan<'g', double>();
    main_program.add_argument("--fitness-weight")
        .default_value(double(-1))
        .help("Fitness score weight  (-1 = sample uniformly per node)")
        .scan<'g', double>();
    main_program.add_argument("--alpha")
        .default_value(double(-1))
        .help("1-hop neighbourhood fraction alpha  (-1 = sample uniformly per node)")
        .scan<'g', double>();
    main_program.add_argument("--fully-random-citations")
        .default_value(double(0.05))
        .help("Fraction of citations filled randomly (PA only, default 0.05)")
        .scan<'g', double>();

    // ── ER-specific parameters ────────────────────────────────────────────────
    main_program.add_argument("--er-probability")
        .default_value(double(0.01))
        .help("Edge probability p for --model er-gnp  (ignored for pa and er)")
        .scan<'g', double>();

    // ── Common simulation parameters ──────────────────────────────────────────
    main_program.add_argument("--growth-rate")
        .required()
        .help("Annual growth rate  (fraction of current graph size added each cycle)")
        .scan<'g', double>();
    main_program.add_argument("--num-cycles")
        .required()
        .help("Number of annual cycles to simulate")
        .scan<'d', int>();
    main_program.add_argument("--same-year-proportion")
        .required()
        .help("Proportion of new nodes that cite a same-year node")
        .scan<'g', double>();

    // ── Output files ──────────────────────────────────────────────────────────
    main_program.add_argument("--output-file")
        .required()
        .help("Output edgelist file  (same format as cpu/PA reference)");
    main_program.add_argument("--auxiliary-information-file")
        .required()
        .help("Output auxiliary node-attribute file  (same schema as cpu/PA reference)");
    main_program.add_argument("--log-file")
        .required()
        .help("Output log file");

    // ── Runtime options ───────────────────────────────────────────────────────
    main_program.add_argument("--num-processors")
        .default_value(int(1))
        .help("Number of OpenMP threads  (default 1)")
        .scan<'d', int>();
    main_program.add_argument("--log-level")
        .default_value(int(1))
        .help("Log verbosity: 0 = silent, 1 = info, 2 = verbose  (default 1)")
        .scan<'d', int>();

    // ── Parse ─────────────────────────────────────────────────────────────────
    try {
        main_program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << main_program;
        std::exit(1);
    }

    // ── Extract values ────────────────────────────────────────────────────────
    std::string edgelist                  = main_program.get<std::string>("--edgelist");
    std::string nodelist                  = main_program.get<std::string>("--nodelist");
    std::string out_degree_bag            = main_program.get<std::string>("--out-degree-bag");
    std::string recency_probabilities     = main_program.get<std::string>("--recency-probabilities");
    std::string planted_nodes             = main_program.get<std::string>("--planted-nodes");
    std::string model                     = main_program.get<std::string>("--model");
    double alpha                          = main_program.get<double>("--alpha");
    double fully_random_citations         = main_program.get<double>("--fully-random-citations");
    double preferential_weight            = main_program.get<double>("--preferential-weight");
    double recency_weight                 = main_program.get<double>("--recency-weight");
    double fitness_weight                 = main_program.get<double>("--fitness-weight");
    double er_probability                 = main_program.get<double>("--er-probability");
    double growth_rate                    = main_program.get<double>("--growth-rate");
    int    num_cycles                     = main_program.get<int>("--num-cycles");
    double same_year_proportion           = main_program.get<double>("--same-year-proportion");
    std::string output_file               = main_program.get<std::string>("--output-file");
    std::string auxiliary_information_file = main_program.get<std::string>("--auxiliary-information-file");
    std::string log_file                  = main_program.get<std::string>("--log-file");
    int    num_processors                 = main_program.get<int>("--num-processors");
    int    log_level                      = main_program.get<int>("--log-level") - 1; // enum offset

    // ── Validate model ────────────────────────────────────────────────────────
    if (model != "pa" && model != "er" && model != "er-gnp") {
        std::cerr << "Error: --model must be one of: pa, er, er-gnp" << std::endl;
        std::exit(1);
    }

    // ── Print run configuration ───────────────────────────────────────────────
    std::cout << "========================================\n";
    std::cout << "Citation Network Growth Simulation\n";
    std::cout << "========================================\n";
    if (model == "pa") {
        std::cout << "Model : Preferential Attachment (PA)\n";
        std::cout << "  degree x recency x fitness weighted reservoir sampling\n";
        std::cout << "  alpha : " << (alpha < 0 ? "random per node" : std::to_string(alpha)) << "\n";
    } else if (model == "er") {
        std::cout << "Model : Erdős-Rényi fixed-k (ER)\n";
        std::cout << "  each new node cites exactly k_u distinct nodes uniformly at random\n";
        std::cout << "  (same out-degree sequence as PA for apples-to-apples comparison)\n";
    } else {
        std::cout << "Model : Erdős-Rényi G(n,p) (ER-GNP)\n";
        std::cout << "  each pre-existing node cited independently with p = " << er_probability << "\n";
    }
    std::cout << "Growth rate    : " << growth_rate << "\n";
    std::cout << "Cycles         : " << num_cycles  << "\n";
    std::cout << "Threads        : " << num_processors << "\n";
    std::cout << "Output edgelist: " << output_file << "\n";
    std::cout << "Auxiliary file : " << auxiliary_information_file << "\n";
    std::cout << "========================================\n";

    // ── Run simulation ────────────────────────────────────────────────────────
    ABM* abm = new ABM(
        edgelist, nodelist, out_degree_bag, recency_probabilities,
        planted_nodes, alpha, fully_random_citations,
        preferential_weight, recency_weight, fitness_weight,
        growth_rate, num_cycles, same_year_proportion,
        output_file, auxiliary_information_file, log_file,
        num_processors, log_level,
        model, er_probability);

    abm->main();
    delete abm;

    // ── Wall-clock summary ────────────────────────────────────────────────────
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    std::ostringstream msg;
    msg << "\n========================================"
        << "\nSimulation complete."
        << "\n  Model        : " << model
        << "\n  Cycles       : " << num_cycles
        << "\n  Growth rate  : " << (100.0 * growth_rate) << "%"
        << "\n  Threads      : " << num_processors
        << "\n  Wall time    : " << elapsed.count() / 1000.0 << " s"
        << "\n========================================";
    std::cout << msg.str() << std::endl;
    return 0;
}
