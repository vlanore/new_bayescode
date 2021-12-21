#include "submodels/codonm2a_model.hpp"
#include "traceable_collection.hpp"

int main(int argc, char* argv[]) {
    // parsing command-line arguments
    ChainCmdLine cmd{argc, argv, "codonm2a", ' ', "0.1"};
    InferenceAppArgParse args(cmd);
    cmd.parse();

    // input data
    auto data = prepare_data(args.alignment.getValue(), args.treefile.getValue());

    // random generator
    auto gen = make_generator(42);

    // model
    auto model = codonm2a::make(data, gen);

    // move success stats
    MoveStatsRegistry ms;

    size_t nrep = 10;
    size_t mix_nrep = 3;

    // move schedule
    auto scheduler = make_move_scheduler([&data, &gen, &model, &nrep, &mix_nrep]() {

        // define lambdas for resampling mixture
        auto mix_logprob = 
            [&data, &w = get<mixture_weights,value>(model), &om = get<omega_array,value>(model), &ss = site_omegapath_suffstat_(model)] () {
                double tot = 0;
                for (int i=0; i<data.alignment.GetNsite(); i++)    {
                    double mean = 0;
                    for (size_t k=0; k<w.size(); k++)  {
                        double lnl = ss.get(i).GetLogProb(om[k]);
                        mean += w[k] * exp(lnl);
                    }
                    tot += log(mean);
                }
                return tot;
            };

        auto w_upd = simple_gather(mixture_weights_(model));
        auto om_upd = simple_gather(omega_array_(model));

        auto alloc_logprob = 
            [&val = get<omega_array,value>(model), &ss = site_omegapath_suffstat_(model)] (int i) {
                auto lambda = [&val, &s=ss.get(i)] (int k) {return s.GetLogProb(val[k]);};
                return lambda;
            };

        // resample mappings
        gather(get<nuc_rates, nuc_matrix>(model));
        gather(codon_submatrix_array_(model));
        phyloprocess_(model).Move(1.0);

        // move parameters
        for (size_t rep = 0; rep < nrep; rep++) {

            // move branch lengths
            bl_suffstat_(model).gather();
            branchlengths_sm::gibbs_resample(branch_lengths_(model), bl_suffstat_(model), gen);
            // gather path suffstats sitewise
            site_path_suffstat_(model).gather();
            site_omegapath_suffstat_(model).gather();

            for (size_t mixrep=0; mixrep<mix_nrep; mixrep++)    {

                // move mixture params
                slide_constrained_move(purw_(model), mix_logprob, 1, 0, 1, 10, gen, w_upd);
                slide_constrained_move(posw_(model), mix_logprob, 1, 0, 1, 10, gen, w_upd);
                slide_constrained_move(purom_(model), mix_logprob, 1, 0, 1, 10, gen, om_upd);
                scaling_move(dposom_(model), mix_logprob, 1, 10, gen, om_upd);

                // move site allocations

                logprob_gibbs_resample(mixture_allocs_(model), alloc_logprob, gen);
            }

            // move nuc rates
            // update codon matrices (important for calculating nucpath suffstat)
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
    auto trace = make_custom_tracer(cmd.chain_name() + ".trace",
        trace_entry("lnL", [& model] () {return get<phyloprocess>(model).GetLogLikelihood();}),
        trace_entry("posw", get<posw>(model)),
        trace_entry("dposom", get<dposom>(model))
        // trace_entry("posom", [& model] () {return get<dposom,value>(model) + 1.0;})
    );
    chain_driver.add(trace);
    chain_driver.add(ms);

    // launching chain!
    chain_driver.go();
}
