#include "submodels/brownian_clock_globom_model.hpp"
#include "traceable_collection.hpp"

int main(int argc, char* argv[]) {
    // parsing command-line arguments
    ChainCmdLine cmd{argc, argv, "BrownianClockSingleOmega", ' ', "0.1"};
    InferenceAppArgParse args(cmd);
    cmd.parse();

    // input tree
    std::ifstream tis(args.treefile.getValue());
    NHXParser parser(tis);
    auto tree = make_from_parser(parser);
    auto nuc_align = FileSequenceAlignment(args.alignment.getValue());
    auto codon_align = CodonSequenceAlignment(&nuc_align);

    // random generator
    auto gen = make_generator(42);

    // model
    auto model = brownian_clock_globom::make(tree.get(), codon_align, gen);

    // move success stats
    MoveStatsRegistry ms;

    // move schedule
    auto scheduler = make_move_scheduler([&gen, &model]() {
        brownian_clock_globom::update_matrices(model);
        brownian_clock_globom::resample_sub(model, gen);

        for (int rep = 0; rep < 30; rep++) {
            brownian_clock_globom::move_params(model, gen);
            brownian_clock_globom::update_matrices(model);
        }
    });

    // trace
    auto trace = make_custom_tracer(cmd.chain_name() + ".trace",
        trace_entry("lnL", [& model] () {return get<phyloprocess>(model).GetLogLikelihood();}),
        trace_entry("tl", [& model] () {return brownian_clock_globom::get_total_ds(model);}),
        trace_entry("chrono", [& model] () {return brownian_clock_globom::get_total_time(model);}),
        trace_entry("om", get<global_omega>(model)), 
        trace_entry("tau", get<tau>(model)));

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
