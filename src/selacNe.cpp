#include "submodels/selacNe_model.hpp"
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
    auto model = selacNe::make(data, 4, gen);

    // move success stats
    MoveStatsRegistry ms;

    size_t nrep = 1;
    size_t mix_nrep = 3;

    // move schedule
    auto scheduler = make_move_scheduler([&gen, &model, &nrep, &mix_nrep]() {

        selacNe::resample_sub(model, gen);

        for (size_t rep = 0; rep < nrep; rep++) {

            selacNe::move_bl(model, gen);

            site_path_suffstat_(model).gather();

            for (size_t mixrep=0; mixrep<mix_nrep; mixrep++)    {
                selacNe::resample_g_alloc(model, gen);
                selacNe::resample_aa_alloc(model, gen);

                comp_path_suffstat_(model).gather();

                selacNe::move_selac(model, gen);
            }

            selacNe::move_Ne(model, gen);
            selacNe::move_omega(model, gen);
            selacNe::move_nuc(model, gen);
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
        trace_entry("tl", [& model] () {return selacNe::get_total_length(model);}),
        trace_entry("meanNe", [& model] () {return selacNe::get_mean_Ne(model);}),
        trace_entry("varNe", [& model] () {return selacNe::get_var_Ne(model);}),
        trace_entry("omega", get<omega>(model)),
        trace_entry("g_alpha", get<g_alpha>(model)),
        trace_entry("psi", get<psi>(model)),
        trace_entry("wpol", get<wpol>(model)),
        trace_entry("wcom", get<wcom>(model))
    );
    chain_driver.add(trace);
    chain_driver.add(ms);

    // launching chain!
    chain_driver.go();
}
