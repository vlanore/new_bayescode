#include "submodels/siteom_model.hpp"

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
    auto model = siteom::make(data, gen);

    // move success stats
    MoveStatsRegistry ms;

    // move schedule
    auto scheduler = make_move_scheduler([&gen, &model, &ms]() {
        // move phyloprocess
        std::cerr << "touch\n";
        siteom::touch_matrices(model);
        std::cerr << "sub\n";
        phyloprocess_(model).Move(1.0);
        std::cerr << "subok\n";

        // move omega
        for (int rep = 0; rep < 30; rep++) {
            // move branch lengths
            std::cerr << "bl\n";
            bl_suffstats_(model).gather();
            branchlengths_sm::gibbs_resample(branch_lengths_(model), bl_suffstats_(model), gen);

            std::cerr << "omega\n";
            // move omega
            std::cerr << "gather sit path\n";
            site_path_suffstats_(model).gather();
            std::cerr << "gather om path\n";
            site_omegapath_suffstats_(model).gather();
            std::cerr << "gibbs resample \n";
            siteomega_sm::gibbs_resample(site_omega_(model), site_omegapath_suffstats_(model), gen);
            // omega_sm::move_hyper(site_omega_(model), site_omegapath_suffstats_(model), gen);

            std::cerr << "nuc\n";
            // move nuc rates
            nucpath_suffstats_(model).gather();
            nucrates_sm::move_nucrates(nuc_rates_(model), nucpath_suffstats_(model), gen, 1, 1.0);
            // nucrates_sm::move_nucrates(nuc_rates_(model), nucpath_suffstats_(model), gen, 1, 1.0, ms);
        }
    });

    // initializing components
    ChainDriver chain_driver{cmd.chain_name(), args.every.getValue(), args.until.getValue()};

    ConsoleLogger console_logger;
    // ChainCheckpoint chain_checkpoint(cmd.chain_name() + ".param", chain_driver, model);
    StandardTracer trace(model, cmd.chain_name());

    // registering components to chain driver
    chain_driver.add(scheduler);
    chain_driver.add(console_logger);
    // chain_driver.add(chain_checkpoint);
    chain_driver.add(trace);
    chain_driver.add(ms);

    // launching chain!
    chain_driver.go();
}
