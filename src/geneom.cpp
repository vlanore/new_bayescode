
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

    // move success stats
    MoveStatsRegistry ms;

    // move schedule
    auto scheduler = make_move_scheduler([&gen, &model]() {

        geneom::gene_update_matrices(model);
        geneom::gene_resample_sub(model, gen);

        for (int rep = 0; rep < 3; rep++) {
            geneom::gene_move_params(model, gen);
            geneom::gene_collect_suffstat(model);
            geneom::move_hyper(model, gen);
            geneom::gene_update_matrices(model);
        }
        geneom::gene_trace_omegas(model,std::cout);
    });


    auto trace = make_custom_tracer(cmd.chain_name() + ".trace",
        trace_entry("lnL", [& model] () {return geneom::get_log_likelihood(model);}),
        trace_entry("tl", [& model] () {return geneom::get_mean_total_length(model);}),
        trace_entry("hypermean_om", get<omega_hypermean>(model))
        // does not know how to trace gene omega's or even gene models
    );

    // initializing components
    ChainDriver chain_driver{cmd.chain_name(), args.every.getValue(), args.until.getValue()};

    ConsoleLogger console_logger{false};
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
