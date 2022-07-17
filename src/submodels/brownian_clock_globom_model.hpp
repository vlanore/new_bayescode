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
#include "lib/UnivariateBrownianTreeProcess.hpp"
#include "univariate_branch_map.hpp"

TOKEN(global_omega)
TOKEN(chrono)
TOKEN(tau)
TOKEN(brownian_process)
TOKEN(synrate)
TOKEN(nuc_rates)
TOKEN(codon_submatrix)
TOKEN(phyloprocess)
TOKEN(bl_suffstats)
TOKEN(path_suffstats)
TOKEN(nucpath_suffstats)
TOKEN(omegapath_suffstats)
TOKEN(tau_suffstats)


static auto make_brownian_tree_process(const Tree* intree, std::function<const double& (int)> innode_age, std::function<double ()> intau, std::function<double ()> inrootmean, std::function<double ()> inrootvar)  {
    return std::make_unique<UnivariateBrownianTreeProcess>(intree, innode_age, intau, inrootmean, inrootvar);
}

struct brownian_clock_globom {

    template <class Gen>
    static auto make(const Tree* tree, const CodonSequenceAlignment& codon_data, Gen& gen)  {

        // relative dates (root has age 1, i.e. tree has depth 1)
        auto chrono = make_node_with_init<chronogram>({tree});
        draw(chrono, gen);

        // strict clock
        // absolute synonymous substitution rate (per tree depth)
        auto tau= make_node<gamma_mi>(1.0, 1.0);
        draw(tau, gen);

        auto root_mean = -3;
        auto root_var = 2;

        auto brownian_process = make_brownian_tree_process(
                tree,
                [&ch = get<value>(chrono)] (int node) -> const double& {return ch[node];},
                one_to_one(tau),
                one_to_one(root_mean),
                one_to_one(root_var));

        brownian_process->PseudoSample(0.1);

        auto synrate = branch_map::make_branch_sums(get<value>(chrono), *brownian_process);
        gather(synrate);

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
            dynamic_cast<const CodonStateSpace*>(codon_data.GetStateSpace());

        auto codon_submatrix = make_dnode_with_init<mgomega>(
                {codon_statespace, &get<nuc_matrix,value>(nuc_rates), 1.0},
                one_to_one(get<nuc_matrix,value>(nuc_rates)),
                global_omega 
                );
        gather(codon_submatrix);
            
        auto phyloprocess = std::make_unique<PhyloProcess>(tree, &codon_data,
            n_to_n(synrate),
            n_to_const(1.0),
            mn_to_one(get<value>(codon_submatrix)),
            [&m = get<value>(codon_submatrix)] (int site) -> const std::vector<double>& {return m.eq_freqs();},
            nullptr);

        phyloprocess->Unfold();
        std::cerr << "lnL: " << phyloprocess->GetLogLikelihood() << '\n';

        // suff stats
        auto bl_suffstats = pathss_factory::make_bl_suffstat(*phyloprocess);
        auto path_suffstats = pathss_factory::make_path_suffstat(*phyloprocess);
        auto nucpath_ss = pathss_factory::make_nucpath_suffstat(codon_statespace, get<value>(codon_submatrix), *path_suffstats);
        auto omega_ss = pathss_factory::make_omega_suffstat(get<value>(codon_submatrix), *path_suffstats);

        auto tau_ss = ss_factory::make_suffstat<PoissonSuffStat>(
                [&process = *brownian_process] (auto& ss) { 
                    double var = 0;
                    int n = 0;
                    process.GetSampleVariance(var, n);
                    ss.AddSuffStat(0.5*n, 0.5*var); 
                });

        return make_model(
            global_omega_ = move(global_omega),
            chrono_ = move(chrono),
            tau_ = move(tau),
            brownian_process_ = move(brownian_process),
            synrate_ = move(synrate),
            nuc_rates_ = move(nuc_rates),
            codon_submatrix_ = move(codon_submatrix),
            phyloprocess_ = move(phyloprocess),
            bl_suffstats_ = move(bl_suffstats),
            path_suffstats_ = move(path_suffstats),
            nucpath_suffstats_ = move(nucpath_ss),
            omegapath_suffstats_ = move(omega_ss),
            tau_suffstats_ = move(tau_ss));
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

            const Tree& tree = get<chrono,value>(model).get_tree();

            auto node_update = tree_factory::do_around_node(
                    tree,
                    array_element_gather(synrate_(model)));

            auto node_suffstat_logprob = tree_factory::sum_around_node(tree,
                    tree_factory::suffstat_logprob(bl_suffstats_(model),
                                    n_to_n(get<synrate,value>(model))));

            auto node_logprob =
                [&process = brownian_process_(model), node_suffstat_logprob] (int node) {
                    return process.GetNodeLogProb(node) + node_suffstat_logprob(node);
                };

            get<chrono,value>(model).MoveTimes(node_update, node_logprob);
        }

    template<class Model, class Gen>
        static auto move_ds(Model& model, Gen& gen) {

            const Tree& tree = get<chrono,value>(model).get_tree();

            // update dS on branches around changing node
            auto synrate_node_update = tree_factory::do_around_node(tree, 
                    array_element_gather(synrate_(model)));

            // compute suffstat log probs around node
            auto node_logprob = tree_factory::sum_around_node(tree,
                    tree_factory::suffstat_logprob(bl_suffstats_(model),
                                    n_to_n(get<synrate,value>(model))));

            brownian_process_(model).SingleNodeMove(1.0, synrate_node_update, node_logprob);
            brownian_process_(model).SingleNodeMove(0.3, synrate_node_update, node_logprob);
        }

    template<class Model, class Gen>
        static auto move_tau(Model& model, Gen& gen)  {
            tau_suffstats_(model).gather();
            gibbs_resample(tau_(model), tau_suffstats_(model), gen);
        }

    template<class Model, class Gen>
        static auto move_params(Model& model, Gen& gen) {
            // move branch times and rates
            bl_suffstats_(model).gather();
            move_chrono(model, gen);
            move_ds(model, gen);

            // move variance parameter of Brownian process
            move_tau(model, gen);

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
        static auto get_total_ds(Model& model)  {
            auto& bl = get<synrate,value>(model);
            double tot = 0;
            for (size_t b=0; b<bl.size(); b++)   {
                tot += bl[b];
            }
            return tot;
        }
};
