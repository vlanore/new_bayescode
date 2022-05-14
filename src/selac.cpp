#include "submodels/selac_model.hpp"
// #include "submodels/grid_selac_model.hpp"
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
    auto model = selac::make(data, 16, gen);
    // auto model = selac::make(data, 20, -5.0, 5.0, gen);

    // move success stats
    MoveStatsRegistry ms;

    size_t nrep = 10;
    size_t mix_nrep = 3;

    // move schedule
    auto scheduler = make_move_scheduler([&gen, &model, &nrep, &mix_nrep]() {

        selac::update_matrices(model);
        selac::resample_sub(model, gen);

        for (size_t rep = 0; rep < nrep; rep++) {

            selac::move_bl(model, gen);

            site_path_suffstat_(model).gather();
            for (size_t mixrep=0; mixrep<mix_nrep; mixrep++)    {
                selac::resample_g_alloc(model, gen);
                selac::resample_aa_alloc(model, gen);
                comp_path_suffstat_(model).gather();
                selac::move_selac(model, gen);
                // selac::move_hyper_selac(model, gen);
            }

            // comp_path_suffstat_(model).gather();
            gather(codon_submatrix_bidimarray_(model));
            selac::move_omega(model, gen);
            gather(codon_submatrix_bidimarray_(model));
            selac::move_nuc(model, gen);
            gather(codon_submatrix_bidimarray_(model));
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
        trace_entry("omega", get<omega>(model)),
        // trace_entry("g_var", get<g_var>(model)),
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
