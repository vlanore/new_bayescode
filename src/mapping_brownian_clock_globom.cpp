#include "submodels/mapping_brownian_clock_globom_model.hpp"
#include "traceable_collection.hpp"

int main(int argc, char* argv[]) {
    // parsing command-line arguments
    ChainCmdLine cmd{argc, argv, "BrownianClockSingleOmega", ' ', "0.1"};
    MappingInferenceAppArgParse args(cmd);
    cmd.parse();

    // input tree
    std::ifstream is(args.treefile.getValue());
    NHXParser parser(is);
    auto tree = make_from_parser(parser);

    // suff stats
    auto suffstat = pathss_factory::make_mapping_dsom_suffstats(args.suffstats.getValue());

    // random generator
    auto gen = make_generator(42);

    // model
    auto model = brownian_clock_globom::make(tree.get(), suffstat, gen);

    // move success stats
    MoveStatsRegistry ms;

    // move schedule
    auto scheduler = make_move_scheduler([&gen, &model]() {
        for (int rep = 0; rep < 300; rep++) {
            brownian_clock_globom::move_params(model, gen);
        }
        std::ofstream os("tree");
        os << brownian_clock_globom::get_annotated_tree(model);
    });

    // trace
    auto trace = make_custom_tracer(cmd.chain_name() + ".trace",
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
