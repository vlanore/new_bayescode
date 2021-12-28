#pragma once

#include <cmath>
#include "components/ChainCheckpoint.hpp"
#include "components/ChainDriver.hpp"
#include "components/ConsoleLogger.hpp"
#include "components/InferenceAppArgParse.hpp"
#include "components/MoveScheduler.hpp"
#include "components/StandardTracer.hpp"
#include "components/restart_check.hpp"
#include "lib/CodonSubMatrix.hpp"
#include "lib/CodonSuffStat.hpp"
#include "lib/PoissonSuffStat.hpp"
#include "chronogram.hpp"
#include "lib/MultivariateBrownianTreeProcess.hpp"
#include "lib/CodonSequenceAlignment.hpp"
#include "submodels/move_reporter.hpp"
#include "submodels/nucrates_sm.hpp"
#include "submodels/mgomega.hpp"
#include "submodels/invwishart.hpp"
#include "submodels/submodel_external_interface.hpp"
#include "submodels/suffstat_wrappers.hpp"
#include "tree_factory.hpp"
#include "bayes_toolbox.hpp"

TOKEN(chrono)
TOKEN(kappa)
TOKEN(sigma)
TOKEN(brownian_process)
TOKEN(synrate)
TOKEN(omega)
TOKEN(nuc_rates)
TOKEN(codon_matrices)
TOKEN(phyloprocess)
TOKEN(path_suffstats)
TOKEN(rel_path_suffstats)
TOKEN(nucpath_suffstats)
TOKEN(dsom_suffstats)
TOKEN(covmat_suffstat)


struct coevol {

    template <class RootMean, class RootVar, class Gen>
    static auto make(const Tree* tree, const CodonSequenceAlignment& codon_data, const ContinuousData& cont_data, RootMean inroot_mean, RootVar inroot_var, Gen& gen) {

        size_t ncont = cont_data.GetNsite();
        size_t nnode = tree->nb_nodes();

        // chronogram - relative dates (root has age 1, i.e. tree has depth 1)
        auto chrono = make_node_with_init<chronogram>({tree});
        draw(chrono, gen);

        // covariance matrix
        auto kappa = std::make_unique<std::vector<double>>(ncont+2, 1.0);
        auto sigma = make_node_with_init<invwishart>(
                {ncont+2, 1},
                [&k = *kappa] () { return k; });
        draw(sigma, gen);

        // mean and variance for normal prior for brownian process at the root
        auto root_mean = make_param<std::vector<double>>(std::forward<RootMean>(inroot_mean));
        auto root_var = make_param<std::vector<double>>(std::forward<RootVar>(inroot_var));

        // brownian process
        auto brownian_process = make_brownian_tree_process(
                tree,
                [&ch = get<value>(chrono)] (int node) -> const double& {return ch[node];},
                [&s = get<value>(sigma)] () -> const CovMatrix& {return s;},
                root_mean,
                root_var);

        std::cerr << "brownian process: set and clamp\n";
        for (size_t i=0; i<ncont; i++)  {
            brownian_process->SetAndClamp(cont_data, i+2, i);
        }
        brownian_process->PseudoSample(0.1);
        
        // branch dS
        auto synrate = make_dnode_array<custom_dnode<double>>(tree->nb_branches(), 
            [&ch = get<value>(chrono), &process = *brownian_process] (int branch) {
                return 
                    [&old_age = ch[ch.get_tree().get_older_node(branch)], 
                     &young_age = ch[ch.get_tree().get_younger_node(branch)],
                     &old_val = process[ch.get_tree().get_older_node(branch)],
                     &young_val = process[ch.get_tree().get_younger_node(branch)]] ()   {
                        return 0.5 * (old_age - young_age) * (exp(old_val[0]) + exp(young_val[0]));
                    };
            });
        gather(synrate);

        // branch dN/dS
        auto omega = make_dnode_array<custom_dnode<double>>(tree->nb_branches(), 
            [&ch = get<value>(chrono), &process = *brownian_process] (int branch) {
                return 
                    [&old_val = process[ch.get_tree().get_older_node(branch)],
                     &young_val = process[ch.get_tree().get_younger_node(branch)]] ()   {
                        return 0.5 * (exp(old_val[1]) + exp(young_val[1]));
                    };
            });
        gather(omega);

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
            dynamic_cast<const CodonStateSpace*>(codon_data.GetStateSpace());

        // branch codon matrices
        auto codon_matrices = make_dnode_array_with_init<mgomega>(
                nnode,
                {codon_statespace, &get<nuc_matrix,value>(nuc_rates), 1.0},
                n_to_one(get<nuc_matrix,value>(nuc_rates)),
                [&om = get<value>(omega)] (int node) {return node ? om[node-1] : 1.0;}
                );
        gather(codon_matrices);

        auto phyloprocess = std::make_unique<PhyloProcess>(tree, &codon_data,

            // branch lengths
            n_to_n(synrate),

            // site-specific rates
            n_to_const(1.0),

            // branch and site specific matrices 
            [&m = get<value>(codon_matrices)] (int branch, int site) -> const SubMatrix& {return m[branch+1];},
            
            // root matrices
            [&m = get<value>(codon_matrices)] (int site) -> const SubMatrix& {return m[0];},

            // no polymorphism
            nullptr);

        phyloprocess->Unfold();
        std::cerr << "lnL: " << phyloprocess->GetLogLikelihood() << '\n';

        // suff stats
        auto path_suffstats = pathss_factory::make_node_path_suffstat(*phyloprocess);

        auto rel_path_suffstats = pathss_factory::make_node_relpath_suffstat(nnode, codon_statespace, *path_suffstats, [&ds=get<value>(synrate)] (int branch) {return ds[branch];});

        auto dsom_ss = pathss_factory::make_dsomega_suffstat(get<value>(codon_matrices), *rel_path_suffstats, [&om=get<value>(omega)] (int branch) {return om[branch];});

        auto nucpath_ss = pathss_factory::make_nucpath_suffstat(codon_statespace, get<value>(codon_matrices), *rel_path_suffstats, [&ds=get<value>(synrate)] (int branch) {return ds[branch];});

        auto covmat_ss = ss_factory::make_suffstat_with_init<MultivariateNormalSuffStat>({ncont+2},
                [&process = *brownian_process] (auto& ss) { ss.AddSuffStat(process); });

        return make_model(
            chrono_ = move(chrono),
            kappa_ = move(kappa),
            sigma_ = move(sigma),
            brownian_process_ = move(brownian_process),
            synrate_ = move(synrate),
            omega_ = move(omega),
            nuc_rates_ = move(nuc_rates),
            codon_matrices_ = move(codon_matrices),
            phyloprocess_ = move(phyloprocess),
            path_suffstats_ = move(path_suffstats),
            rel_path_suffstats_ = move(rel_path_suffstats),
            nucpath_suffstats_ = move(nucpath_ss),
            dsom_suffstats_ = move(dsom_ss),
            covmat_suffstat_ = move(covmat_ss));
    }

    template<class Model>
        static auto update_matrices(Model& model) {
            gather(nuc_matrix_(nuc_rates_(model)));
            gather(codon_matrices_(model));
        }

    template<class Model, class Gen>
        static auto resample_sub(Model& model, Gen& gen)    {
            phyloprocess_(model).Move(1.0);
        }

    template<class Model>
        static auto gather_path_suffstat(Model& model) {
            path_suffstats_(model).gather();
            rel_path_suffstats_(model).gather();
        }

    template<class Model, class Gen>
        static auto move_chrono(Model& model, Gen& gen) {

            auto node_update = tree_factory::do_around_node(
                    get<chrono,value>(model).get_tree(),
                    array_element_gather(synrate_(model)));

            auto branch_suffstat_logprob = 
                [&ss = dsom_suffstats_(model), &ds = get<synrate,value>(model), &om = get<omega,value>(model)]
                (int branch) {
                    return ss.get(branch).GetLogProb(ds[branch], om[branch]);
                };

            auto node_suffstat_logprob = tree_factory::sum_around_node(
                    get<chrono,value>(model).get_tree(),
                    branch_suffstat_logprob);

            auto node_logprob =
                [&process = brownian_process_(model), node_suffstat_logprob] (int node) {
                    return process.GetNodeLogProb(node) + node_suffstat_logprob(node);
                };

            get<chrono,value>(model).MoveTimes(node_update, node_logprob);
        }

    template<class Model, class Gen>
        static auto move_process(Model& model, Gen& gen) {

            auto synrate_node_update = tree_factory::do_around_node(
                    get<chrono,value>(model).get_tree(),
                    array_element_gather(synrate_(model)));

            auto omega_node_update = tree_factory::do_around_node(
                    get<chrono,value>(model).get_tree(),
                    array_element_gather(omega_(model)));

            auto branch_suffstat_logprob = 
                [&ss = dsom_suffstats_(model), &ds = get<synrate,value>(model), &om = get<omega,value>(model)]
                (int branch) {
                    return ss.get(branch).GetLogProb(ds[branch], om[branch]);
                };

            auto node_logprob = tree_factory::sum_around_node(
                    get<chrono,value>(model).get_tree(), branch_suffstat_logprob);

            auto no_update = [] (int node) {};
            auto no_logprob = [] (int node) {return 0;};

            size_t dim = get<sigma,value>(model).size();

            brownian_process_(model).SingleNodeMove(0, 1.0, synrate_node_update, node_logprob);
            brownian_process_(model).SingleNodeMove(0, 0.3, synrate_node_update, node_logprob);

            brownian_process_(model).SingleNodeMove(1, 1.0, omega_node_update, node_logprob);
            brownian_process_(model).SingleNodeMove(1, 0.3, omega_node_update, node_logprob);

            for (size_t i=2; i<dim; i++) {
                brownian_process_(model).SingleNodeMove(i, 1.0, no_update, no_logprob);
                brownian_process_(model).SingleNodeMove(i, 0.3, no_update, no_logprob);
            }
        }

    template<class Model, class Gen>
        static auto move_sigma(Model& model, Gen& gen)  {
            covmat_suffstat_(model).gather();
            gibbs_resample(sigma_(model), covmat_suffstat_(model), gen);
        }

    template<class Model, class Gen>
        static auto move_params(Model& model, Gen& gen) {

            dsom_suffstats_(model).gather();
            move_chrono(model, gen);
            move_process(model, gen);
            gather(codon_matrices_(model));

            move_sigma(model, gen);

            nucpath_suffstats_(model).gather();
            nucrates_sm::move_nucrates(nuc_rates_(model), nucpath_suffstats_(model), gen, 1, 1.0);
            gather(codon_matrices_(model));
        }

    template<class Model>
        static auto get_total_ds(Model& model)  {
            auto& ds = get<synrate,value>(model);
            double tot = 0;
            for (size_t b=0; b<ds.size(); b++)   {
                tot += ds[b];
            }
            return tot;
        }

    template<class Model>
        static auto get_mean_omega(Model& model)  {
            auto& om = get<omega,value>(model);
            double tot = 0;
            for (size_t b=0; b<om.size(); b++)   {
                tot += om[b];
            }
            tot /= om.size();
            return tot;
        }
};
