#include "submodels/mixom_model.hpp"

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
    auto model = mixom::make(data, 3, gen);

    // move success stats
    MoveStatsRegistry ms;

    size_t nrep = 10;
    size_t mix_nrep = 3;

    // move schedule
    auto scheduler = make_move_scheduler([&gen, &model, &nrep, &mix_nrep]() {

        // move phyloprocess
        gather(get<nuc_rates, nuc_matrix>(model));
        gather(codon_submatrix_array_(model));
        phyloprocess_(model).Move(1.0);

        // move omega
        for (size_t rep = 0; rep < nrep; rep++) {

            // move branch lengths
            bl_suffstat_(model).gather();
            branchlengths_sm::gibbs_resample(branch_lengths_(model), bl_suffstat_(model), gen);

            // these won't change as long as branch lengths and nuc rates don't change
            site_path_suffstat_(model).gather();
            site_omegapath_suffstat_(model).gather();

            for (size_t mixrep=0; mixrep<mix_nrep; mixrep++)    {

                // move site allocations
                auto alloc_logprob = 
                    [&val = get<omega_array,value>(model), &ss = site_omegapath_suffstat_(model)] (int i) {
                        auto lambda = [&val, &s=ss.get(i)] (int k) {return s.GetLogProb(val[k]);};
                        return lambda;
                    };
                logprob_gibbs_resample(mixture_allocs_(model), alloc_logprob, gen);

                // needed for skipping empty components
                alloc_suffstat_(model).gather();

                // gather omega hyper suffstats (skipping empty components)
                omega_hyper_suffstat_(model).gather();

                // move omega hyper parameters
                auto hyper_logprob = 
                    [&mean = get<omega_hypermean,value>(model), 
                     &invshape = get<omega_hyperinvshape,value>(model), 
                     &ss = omega_hyper_suffstat_(model).get()] ()
                     {return ss.GetLogProb(mean, invshape);};

                sweet_scaling_move(omega_hypermean_(model), hyper_logprob, gen);
                sweet_scaling_move(omega_hyperinvshape_(model), hyper_logprob, gen);

                // gather omega path suffstats
                comp_omegapath_suffstat_(model).gather();

                // gibbs resample the omega's
                // (will also refresh empty components by effectively sampling them from prior)
                gibbs_resample(omega_array_(model), comp_omegapath_suffstat_(model), gen);

                // resample mixture weights
                // (occupancy suff stats are still valid)
                gibbs_resample(mixture_weights_(model), alloc_suffstat_(model), gen);

                // alternative version
                // first move omega hyper parameters based on marginal logprob
                comp_omegapath_suffstat_(model).gather();
                auto hyper_marginal_logprob = 
                    [&mean = get<omega_hypermean,value>(model), 
                     &invshape = get<omega_hyperinvshape,value>(model), 
                     &ss = comp_omegapath_suffstat_(model), ncomp = get<omega_array,value>(model).size()] ()  {
                        double tot = 0;
                        for (size_t i=0; i<ncomp; i++)  {
                            tot += gamma_mi::marginal_logprob(ss.get(i), mean, invshape);
                        }
                        return tot;
                    };

                sweet_scaling_move(omega_hypermean_(model), hyper_marginal_logprob, gen);
                sweet_scaling_move(omega_hyperinvshape_(model), hyper_marginal_logprob, gen);
                // then resample omega's
                gibbs_resample(omega_array_(model), comp_omegapath_suffstat_(model), gen);

                // resample mixture weights
                alloc_suffstat_(model).gather();
                gibbs_resample(mixture_weights_(model), alloc_suffstat_(model), gen);
            }

            // move nuc rates
            // first update codon matrices (important for calculating nucpath suffstat)
            gather(codon_submatrix_array_(model));
            // gather component path suffstats
            comp_path_suffstat_(model).gather();
            // gather nucpath suffstats from comp_path_suffstats
            nucpath_suffstat_(model).gather();
            // move nuc rates
            nucrates_sm::move_nucrates(nuc_rates_(model), nucpath_suffstat_(model), gen, 1, 1.0);
            // update codon matrices
            gather(codon_submatrix_array_(model));
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
    // chain_driver.add(trace);
    chain_driver.add(ms);

    // launching chain!
    chain_driver.go();
}
