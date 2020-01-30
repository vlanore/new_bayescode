
#include "submodels/geneom_model.hpp"
#include "traceable_collection.hpp"

int main(int argc, char* argv[]) {
    // parsing command-line arguments
    ChainCmdLine cmd{argc, argv, "SingleOmega", ' ', "0.1"};
    InferenceAppArgParse args(cmd);
    cmd.parse();

    auto data = multi_gene_data::make(args.alignment.getValue(), args.treefile.getValue());

    // random generator
    auto gen = make_generator(42);

    // model
    auto model = geneom::make(data, gen);
    std::cerr << "in main, after calling geneom::make\n";
    std::cerr << "array size: " << get<gene_model_array>(model).size() << '\n';
    std::cerr << "calling gather on omega_gamma_ss\n";
    get<omega_gamma_suffstats>(model).gather();
    exit(1);

    // move success stats
    MoveStatsRegistry ms;

    // move schedule
    auto scheduler = make_move_scheduler([&gen, &model]() {

        geneom::gene_resample_sub(model, gen);

        for (int rep = 0; rep < 30; rep++) {
            geneom::gene_move_params(model, gen);
            std::cerr << "collect\n";
            geneom::gene_collect_suffstat(model);
            std::cerr << "collect ok\n";
            geneom::move_hyper(model, gen);
            geneom::gene_update_matrices(model);
        }
    });

    // trace
    auto trace = make_custom_tracer(cmd.chain_name() + ".trace",
        trace_entry("mean_om", get<omega_hypermean>(model))
    );

    // initializing components
    ChainDriver chain_driver{cmd.chain_name(), args.every.getValue(), args.until.getValue()};

    ConsoleLogger console_logger;
    ModelTracer chain(model, cmd.chain_name() + ".chain");

    // registering components to chain driver
    chain_driver.add(scheduler);
    chain_driver.add(console_logger);
    chain_driver.add(chain);
    chain_driver.add(trace);
    chain_driver.add(ms);

    // launching chain!
    chain_driver.go();
}
