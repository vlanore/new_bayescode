#pragma once

#include <cmath>
#include "components/ChainCheckpoint.hpp"
#include "components/ChainDriver.hpp"
#include "components/ConsoleLogger.hpp"
#include "components/InferenceAppArgParse.hpp"
#include "components/MoveScheduler.hpp"
#include "components/StandardTracer.hpp"
#include "components/restart_check.hpp"
#include "data_preparation.hpp"
#include "lib/CodonSubMatrix.hpp"
#include "lib/CodonSuffStat.hpp"
#include "lib/PoissonSuffStat.hpp"
#include "lib/Chronogram.hpp"
#include "lib/ChronoBranchLengths.hpp"
#include "submodels/move_reporter.hpp"
#include "submodels/nucrates_sm.hpp"
#include "submodels/mgomega.hpp"
#include "submodels/submodel_external_interface.hpp"
#include "submodels/suffstat_wrappers.hpp"
#include "bayes_toolbox.hpp"

TOKEN(global_omega)
TOKEN(chronogram)
TOKEN(branch_lengths)
TOKEN(ds)
TOKEN(nuc_rates)
TOKEN(codon_submatrix)
TOKEN(phyloprocess)
TOKEN(bl_suffstats)
TOKEN(path_suffstats)
TOKEN(nucpath_suffstats)
TOKEN(omegapath_suffstats)


struct globom {
    // =============================================================================================
    template <class Gen>
    static auto make(PreparedData& data, Gen& gen) {

        auto global_omega = make_node<gamma_mi>(1.0, 1.0);
        draw(global_omega, gen);

        auto chronogram = make_chrono(data.tree.get());
        chronogram->Sample();

        auto branch_lengths = make_chrono_branch_lengths(
                data.tree.get(), 
                [&chrono = *chronogram] (int node) {return chrono.get_age(node);});
        branch_lengths->Update();

        auto ds = make_node<gamma_mi>(1.0, 1.0);
        draw(ds, gen);

        // nuc exch rates and eq freqs: uniform dirichlet
        auto nucrr_hypercenter = std::vector<double>(6, 1./6);
        auto nucrr_hyperinvconc = 1./6;
        auto nucstat_hypercenter = std::vector<double>(4, 1./4);
        auto nucstat_hyperinvconc = 1./4;

        // creates nucrr, nucstat and also the gtr matrix
        auto nuc_rates = nucrates_sm::make(
                nucrr_hypercenter, nucrr_hyperinvconc, 
                nucstat_hypercenter, nucstat_hyperinvconc, gen);

        auto codon_statespace =
            dynamic_cast<const CodonStateSpace*>(data.alignment.GetStateSpace());

        auto codon_submatrix = make_dnode_with_init<mgomega>(
                {codon_statespace, &get<nuc_matrix,value>(nuc_rates), 1.0},
                (const SubMatrix&) get<nuc_matrix,value>(nuc_rates),
                global_omega 
                );
        gather(codon_submatrix);
            
        auto phyloprocess = std::make_unique<PhyloProcess>(data.tree.get(), &data.alignment,

            // branch lengths
            [&bl = *branch_lengths, &r = get<value>(ds)] (int branch) {
                return r*bl.get_length(branch);},

            // site-specific rates
            n_to_const(1.0),

            // branch and site specific matrices (here, same matrix for everyone)
            mn_to_one(get<value>(codon_submatrix)),
            
            // site-specific matrices for root equilibrium frequencies (here same for all sites)
            n_to_one(get<value>(codon_submatrix)),

            // no polymorphism
            nullptr);

        phyloprocess->Unfold();
        std::cerr << "lnL: " << phyloprocess->GetLogLikelihood() << '\n';

        // suff stats
        auto bl_suffstats = pathss_factory::make_bl_suffstat(*phyloprocess);
        auto path_suffstats = pathss_factory::make_path_suffstat(*phyloprocess);
        auto nucpath_ss = pathss_factory::make_nucpath_suffstat(codon_statespace, get<value>(codon_submatrix), *path_suffstats);
        auto omega_ss = pathss_factory::make_omega_suffstat(get<value>(codon_submatrix), *path_suffstats);

        return make_model(
            global_omega_ = move(global_omega),
            chronogram_ = move(chronogram),
            branch_lengths_ = move(branch_lengths),
            ds_ = move(ds),
            nuc_rates_ = move(nuc_rates),
            codon_submatrix_ = move(codon_submatrix),
            phyloprocess_ = move(phyloprocess),
            bl_suffstats_ = move(bl_suffstats),
            path_suffstats_ = move(path_suffstats),
            nucpath_suffstats_ = move(nucpath_ss),
            omegapath_suffstats_ = move(omega_ss));
    }

    template<class Model>
        static auto update_matrices(Model& model) {
            gather(nuc_matrix_(nuc_rates_(model)));
            gather(codon_submatrix_(model));
        }

    template<class Model, class Gen>
        static auto resample_sub(Model& model, Gen& gen)    {
            phyloprocess_(model).Move(1.0);
        }

    template<class Model, class Gen>
        static auto move_chrono(Model& model, Gen& gen) {

            auto update = 
                [&bl = branch_lengths_(model)] 
                (int node) 
                { bl.LocalNodeUpdate(node); };

            auto branch_logprob = 
                [&ss = bl_suffstats_(model), &bl = branch_lengths_(model), &r = get<ds,value>(model)] 
                (int branch) 
                {return ss.get(branch).GetLogProb(r*bl.get_length(branch));};

            auto logprob =
                [&bl = branch_lengths_(model), branch_logprob] 
                (int node) 
                { return bl.sum_around_node(branch_logprob, node); };

            chronogram_(model).MoveTimes(update, logprob);
        }

    template<class Model, class Gen>
        static auto move_ds(Model& model, Gen& gen) {

            auto update = [] () {};

            auto logprob =
                [&ss = bl_suffstats_(model), &bl = branch_lengths_(model), &r = get<ds,value>(model)] () {
                    double total = 0;
                    for (size_t branch=0; branch<bl.nb_branches(); branch++)  {
                        total += ss.get(branch).GetLogProb(r*bl.get_length(branch));
                    }
                    return total;
                };

            scaling_move(ds_(model), logprob, 1, 10, gen, update);
        }

    template<class Model, class Gen>
        static auto move_params(Model& model, Gen& gen) {
            // move branch lengths
            bl_suffstats_(model).gather();
            move_chrono(model, gen);
            move_ds(model, gen);

            // move omega
            path_suffstats_(model).gather();
            omegapath_suffstats_(model).gather();
            gibbs_resample(global_omega_(model), omegapath_suffstats_(model), gen);

            // move nuc rates
            gather(codon_submatrix_(model));
            nucpath_suffstats_(model).gather();
            nucrates_sm::move_nucrates(nuc_rates_(model), nucpath_suffstats_(model), gen, 1, 1.0);
        }

    template<class Model>
        static auto get_total_length(Model& model)  {
            auto& bl = branch_lengths_(model);
            double tot = 0;
            for (size_t b=0; b<bl.nb_branches(); b++)   {
                tot += bl.get_length(b);
            }
            return tot;
        }
};
