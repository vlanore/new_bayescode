#include "submodels/mutseldp_model.hpp"
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
    auto model = mutseldp::make(data, 100, gen);

    // move success stats
    MoveStatsRegistry ms;

    size_t nrep = 10;
    size_t mix_nrep = 3;

    // move schedule
    auto scheduler = make_move_scheduler([&gen, &model, &nrep, &mix_nrep]() {

        mutseldp::update_matrices(model);
        mutseldp::resample_sub(model, gen);

        for (size_t rep = 0; rep < nrep; rep++) {

            mutseldp::move_bl(model, gen);

            site_path_suffstat_(model).gather();
            for (size_t mixrep=0; mixrep<mix_nrep; mixrep++)    {
                mutseldp::resample_aa_alloc(model, gen);
                comp_path_suffstat_(model).gather();
                mutseldp::move_mutseldp(model, gen);
            }

            mutseldp::move_hyper(model, gen);

            gather(codon_submatrices_(model));
            mutseldp::move_omega(model, gen);
            gather(codon_submatrices_(model));
            mutseldp::move_nuc(model, gen);
            mutseldp::update_matrices(model);
            // gather(codon_submatrices_(model));
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
        trace_entry("Keff", [& model] () {return mutseldp::get_Keff(model);}),
        trace_entry("omega", get<omega>(model)),
        trace_entry("statent", [& model] () {return mutseldp::get_statent(model);}),
        trace_entry("statalpha", [& model] () {return mutseldp::get_statalpha(model);})
    );
    chain_driver.add(trace);
    chain_driver.add(ms);

    // launching chain!
    chain_driver.go();
}
