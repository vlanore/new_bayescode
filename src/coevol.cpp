#include "submodels/coevol_model.hpp"
#include "traceable_collection.hpp"

int main(int argc, char* argv[]) {
    // parsing command-line arguments
    ChainCmdLine cmd{argc, argv, "SingleOmega", ' ', "0.1"};
    InferenceAppArgParse args(cmd);
    cmd.parse();

    // input tree
    std::ifstream tis(args.treefile.getValue());
    NHXParser parser(tis);
    auto tree = make_from_parser(parser);
    auto nuc_align = FileSequenceAlignment(args.alignment.getValue());
    auto codon_align = CodonSequenceAlignment(&nuc_align);
    auto cont_data = FileContinuousData(args.contdatafile.getValue());

    std::ifstream is(args.rootfile.getValue());
    size_t ncont;
    is >> ncont;
    auto root_mean = std::vector<double>(ncont,0);
    auto root_var = std::vector<double>(ncont,0);
    for (size_t i=0; i<ncont; i++)  {
        is >> root_mean[i] >> root_var[i];
        std::cerr << root_mean[i] << '\t' << root_var[i] << '\n';
    }

    // random generator
    auto gen = make_generator(42);

    // model
    auto model = coevol::make(tree.get(), codon_align, cont_data, root_mean, root_var, gen);

    // move success stats
    MoveStatsRegistry ms;

    // move schedule
    auto scheduler = make_move_scheduler([&gen, &model]() {
        // coevol::update_matrices(model);
        coevol::resample_sub(model, gen);
        coevol::gather_path_suffstat(model);
        for (int rep = 0; rep < 30; rep++) {
            coevol::move_params(model, gen);
        }
    });

    // trace
    auto trace = make_custom_tracer(cmd.chain_name() + ".trace",
        trace_entry("lnL", [& model] () {return get<phyloprocess>(model).GetLogLikelihood();}),
        trace_entry("tl", [& model] () {return coevol::get_total_ds(model);}),
        trace_entry("om", [& model] () {return coevol::get_mean_omega(model);}),
        trace_entry("sigma01", [& model] () {return get<sigma,value>(model)[0][1];}),
        trace_entry("sigma02", [& model] () {return get<sigma,value>(model)[0][2];}),
        trace_entry("sigma12", [& model] () {return get<sigma,value>(model)[1][2];})
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
