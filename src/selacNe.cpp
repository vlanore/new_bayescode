#include "submodels/selacNe_model.hpp"
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
    auto model = selacNe::make(tree.get(), codon_align, cont_data, root_mean, root_var, 4, gen);

    // move success stats
    MoveStatsRegistry ms;

    size_t nrep = 1;
    size_t mix_nrep = 3;

    // move schedule
    auto scheduler = make_move_scheduler([&gen, &model, &nrep, &mix_nrep]() {
        selacNe::resample_sub(model, gen);
        for (size_t rep = 0; rep < nrep; rep++) {
            selacNe::move_params(model, gen);
        }
    });

    // initializing components
    ChainDriver chain_driver{cmd.chain_name(), args.every.getValue(), args.until.getValue()};

    ConsoleLogger console_logger;
    // ChainCheckpoint chain_checkpoint(cmd.chain_name() + ".param", chain_driver, model);
    // StandardTracer trace(model, cmd.chain_name());

    // registering components to chain driver
    chain_driver.add(scheduler);
    chain_driver.add(console_logger);
    // chain_driver.add(chain_checkpoint);
    auto trace = make_custom_tracer(cmd.chain_name() + ".trace",  //
        trace_entry("lnL", [& model] () {return get<phyloprocess>(model).GetLogLikelihood();}),
        trace_entry("ds", [& model] () {return selacNe::get_total_ds(model);}),
        trace_entry("meanne", [& model] () {return selacNe::get_mean_Ne(model);}),
        trace_entry("rootne", get<rootNe>(model)),
        trace_entry("sigma01", [& model] () {return get<sigma,value>(model)[0][1];}),
        trace_entry("sigma02", [& model] () {return get<sigma,value>(model)[0][2];}),
        trace_entry("sigma12", [& model] () {return get<sigma,value>(model)[1][2];}),
        trace_entry("g_alpha", get<g_alpha>(model)),
        // trace_entry("psi", get<psi>(model)),
        trace_entry("wpol", get<wpol>(model)),
        trace_entry("wcom", get<wcom>(model))
    );
    chain_driver.add(trace);
    chain_driver.add(ms);

    // launching chain!
    chain_driver.go();
}
