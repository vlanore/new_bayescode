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
#include "chronogram.hpp"
#include "submodels/move_reporter.hpp"
#include "submodels/nucrates_sm.hpp"
#include "submodels/mgomega.hpp"
#include "submodels/submodel_external_interface.hpp"
#include "submodels/suffstat_wrappers.hpp"
#include "bayes_toolbox.hpp"
#include "tree_factory.hpp"

TOKEN(global_omega)
TOKEN(chrono)
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

    template <class Gen>
    static auto make(PreparedData& data, Gen& gen) {

        // relative dates (root has age 1, i.e. tree has depth 1)
        auto chrono = make_node_with_init<chronogram>({data.tree.get()});
        draw(chrono, gen);

        // strict clock
        // absolute synonymous substitution rate (per tree depth)
        auto ds = make_node<gamma_mi>(1.0, 1.0);
        draw(ds, gen);
        // for easier start
        get<value>(ds) = 0.1;

        // synonymous branch lengths
        auto branch_lengths = make_dnode_array<custom_dnode<double>>(data.tree->nb_branches(), 
            [&ch = get<value>(chrono), &r = get<value>(ds)] (int branch) {
                return [&older = ch[ch.get_tree().get_older_node(branch)], &younger = ch[ch.get_tree().get_younger_node(branch)], &r] (double& bl) {
                    bl = r*(older-younger);
                };
            });
        gather(branch_lengths);

        // nuc exch rates and eq freqs: uniform dirichlet
        auto nucrr_hypercenter = std::vector<double>(6, 1./6);
        auto nucrr_hyperinvconc = 1./6;
        auto nucstat_hypercenter = std::vector<double>(4, 1./4);
        auto nucstat_hyperinvconc = 1./4;

        // creates nucrr, nucstat and also the gtr matrix
        auto nuc_rates = nucrates_sm::make(
                nucrr_hypercenter, nucrr_hyperinvconc, 
                nucstat_hypercenter, nucstat_hyperinvconc, gen);

        // global dN/dS uniform across sites and branches
        auto global_omega = make_node<gamma_mi>(1.0, 1.0);
        draw(global_omega, gen);

        auto codon_statespace =
            dynamic_cast<const CodonStateSpace*>(data.alignment.GetStateSpace());

        auto codon_submatrix = make_dnode_with_init<mgomega>(
                {codon_statespace, &get<nuc_matrix,value>(nuc_rates), 1.0},
                one_to_one(get<nuc_matrix,value>(nuc_rates)),
                // why doesn't this work (seg faults):
                // one_to_one(get<nuc_matrix>(nuc_rates)),
                // this works but requires explicit cast:
                // (const SubMatrix&) get<nuc_matrix,value>(nuc_rates),
                global_omega 
                );
        gather(codon_submatrix);
            
        auto phyloprocess = std::make_unique<PhyloProcess>(data.tree.get(), &data.alignment,
            // branch lengths
            n_to_n(branch_lengths),

            // site-specific rates
            n_to_const(1.0),

            // branch and site specific matrices (here, same matrix for everyone)
            mn_to_one(get<value>(codon_submatrix)),
            
            // site-specific matrices for root equilibrium frequencies (here same for all sites)
            [&m = get<value>(codon_submatrix)] (int site) -> const std::vector<double>& {return m.eq_freqs();},

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
            chrono_ = move(chrono),
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

            auto node_update = tree_factory::do_around_node(
                    get<chrono,value>(model).get_tree(),
                    array_element_gather(branch_lengths_(model)));

            auto node_logprob = tree_factory::sum_around_node(
                    get<chrono,value>(model).get_tree(),
                    suffstat_array_element_logprob(branch_lengths_(model), bl_suffstats_(model)));

            get<chrono,value>(model).MoveTimes(node_update, node_logprob);
        }

    template<class Model, class Gen>
        static auto move_ds(Model& model, Gen& gen) {
            auto update = simple_gather(branch_lengths_(model));
            auto logprob = suffstat_logprob(branch_lengths_(model), bl_suffstats_(model));
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
            auto& bl = get<branch_lengths,value>(model);
            double tot = 0;
            for (size_t b=0; b<bl.size(); b++)   {
                tot += bl[b];
            }
            return tot;
        }
};
