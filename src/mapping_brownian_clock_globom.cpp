#include "submodels/mapping_brownian_clock_globom_model.hpp"
#include "traceable_collection.hpp"

auto make_mapping_dsom_suffstats(std::string filename)   {

    std::ifstream tis(filename.c_str());

    std::string ds_count;
    tis >> ds_count;
    auto ds_count_stream = std::stringstream(ds_count);
    NHXParser ds_count_parser(ds_count_stream);
    auto tree = make_from_parser(ds_count_parser);

    size_t nb = tree->nb_nodes()-1;
    std::vector<std::vector<double>> suffstats(4, std::vector<double>(nb, 0));

    auto branch_ds_count = node_container_from_parser<std::string>(
        ds_count_parser, [](int i, const AnnotatedTree& t) { return t.tag(i, "length"); });

    for (size_t branch=0; branch<nb; branch++)  {
        suffstats[0][branch] = std::atof(branch_ds_count[branch+1].c_str());
    }

    std::string ds_norm;
    tis >> ds_norm;
    auto ds_norm_stream = std::stringstream(ds_norm);
    NHXParser ds_norm_parser(ds_norm_stream);

    auto branch_ds_norm = node_container_from_parser<std::string>(
        ds_norm_parser, [](int i, const AnnotatedTree& t) { return t.tag(i, "length"); });

    for (size_t branch=0; branch<nb; branch++)  {
        suffstats[1][branch] = std::atof(branch_ds_norm[branch+1].c_str());
    }

    std::string dn_count;
    tis >> dn_count;
    auto dn_count_stream = std::stringstream(dn_count);
    NHXParser dn_count_parser(dn_count_stream);

    auto branch_dn_count = node_container_from_parser<std::string>(
        dn_count_parser, [](int i, const AnnotatedTree& t) { return t.tag(i, "length"); });

    for (size_t branch=0; branch<nb; branch++)  {
        suffstats[2][branch] = std::atof(branch_dn_count[branch+1].c_str());
    }

    std::string dn_norm;
    tis >> dn_norm;
    auto dn_norm_stream = std::stringstream(dn_norm);
    NHXParser dn_norm_parser(dn_norm_stream);

    auto branch_dn_norm = node_container_from_parser<std::string>(
        dn_norm_parser, [](int i, const AnnotatedTree& t) { return t.tag(i, "length"); });

    for (size_t branch=0; branch<nb; branch++)  {
        suffstats[3][branch] = std::atof(branch_dn_norm[branch+1].c_str());
    }

    return suffstats;
}

int main(int argc, char* argv[]) {
    // parsing command-line arguments
    ChainCmdLine cmd{argc, argv, "BrownianClockSingleOmega", ' ', "0.1"};
    MappingInferenceAppArgParse args(cmd);
    cmd.parse();

    // input tree
    std::ifstream is(args.alignment.getValue());
    NHXParser parser(is);
    auto tree = make_from_parser(parser);

    // suff stats
    auto suffstat = make_mapping_dsom_suffstats(args.alignment.getValue());

    // random generator
    auto gen = make_generator(42);

    // model
    auto model = brownian_clock_globom::make(tree.get(), suffstat, gen);

    // move success stats
    MoveStatsRegistry ms;

    // move schedule
    auto scheduler = make_move_scheduler([&gen, &model]() {
        for (int rep = 0; rep < 30; rep++) {
            brownian_clock_globom::move_params(model, gen);
        }
    });

    // trace
    auto trace = make_custom_tracer(cmd.chain_name() + ".trace",
        trace_entry("tl", [& model] () {return brownian_clock_globom::get_total_ds(model);}),
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
