#include "submodels/globom_model.hpp"
#include "traceable_collection.hpp"

int main(int argc, char* argv[]) {
    // parsing command-line arguments
    ChainCmdLine cmd{argc, argv, "SingleOmega", ' ', "0.1"};
    InferenceAppArgParse args(cmd);
    cmd.parse();

    // input data
    auto data = prepare_data(args.alignment.getValue(), args.treefile.getValue());

    // random generator
    auto gen = make_generator();

    // model
    auto model = globom::make(data, gen);

    // move success stats
    MoveStatsRegistry ms;

    // move schedule
    auto scheduler = make_move_scheduler([&gen, &model]() {
        // move phyloprocess
        globom::touch_matrices(model);
        phyloprocess_(model).Move(1.0);

        // move omega
        for (int rep = 0; rep < 30; rep++) {
            // move branch lengths
            bl_suffstats_(model).gather();
            branchlengths_sm::gibbs_resample(branch_lengths_(model), bl_suffstats_(model), gen);

            // move omega
            path_suffstats_(model).gather();
            omegapath_suffstats_(model).gather();
            omega_sm::gibbs_resample(global_omega_(model), omegapath_suffstats_(model), gen);

            // move nuc rates
            nucpath_suffstats_(model).gather();
            nucrates_sm::move_nucrates(nuc_rates_(model), nucpath_suffstats_(model), gen, 1, 1.0);
            // nucrates_sm::move_nucrates(nuc_rates_(model), nucpath_suffstats_(model), gen, 1, 1.0,
            // ms);
        }
    });

    // trace
    int youpi = 2;
    auto t = make_trace(                                   //
        trace_entry("a", [&youpi]() { return youpi; }),    //
        trace_entry("b", get<global_omega, omega>(model))  //
    );

    // initializing components
    ChainDriver chain_driver{cmd.chain_name(), args.every.getValue(), args.until.getValue()};

    ConsoleLogger console_logger;
    StandardTracer chain(model, cmd.chain_name());
    StandardTracer trace(t, cmd.chain_name() + "_trace");

    // registering components to chain driver
    chain_driver.add(scheduler);
    chain_driver.add(console_logger);
    chain_driver.add(chain);
    chain_driver.add(trace);
    chain_driver.add(ms);

    // launching chain!
    chain_driver.go();
}
