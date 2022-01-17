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
#include "lib/GammaSuffStat.hpp"
#include "submodels/branchlength_sm.hpp"
#include "submodels/mgomega.hpp"
#include "submodels/move_reporter.hpp"
#include "submodels/nucrates_sm.hpp"
#include "submodels/submodel_external_interface.hpp"
#include "submodels/suffstat_wrappers.hpp"
#include "submodels/mgomega.hpp"
#include "submodels/invwishart.hpp"
#include "tree_factory.hpp"
#include "chronogram.hpp"
#include "submodels/invwishart.hpp"
#include "lib/MultivariateBrownianTreeProcess.hpp"
#include "bayes_toolbox.hpp"

TOKEN(chrono)
TOKEN(sigma)
TOKEN(brownian_process)
TOKEN(synrate)
TOKEN(omega)
TOKEN(nucrr)
TOKEN(nucstat)
TOKEN(nuc_rates)
TOKEN(codon_matrices)
TOKEN(root_codon_matrix)
TOKEN(phyloprocess)
TOKEN(path_suffstats)
TOKEN(rel_path_suffstats)
TOKEN(nucpath_suffstats)
TOKEN(dsom_suffstats)
TOKEN(covmat_suffstat)

TOKEN(gene_model_array)

struct coevol_master {
    template <class RootMean, class RootVar, class Gen>
    static auto make(int ngene, const Tree* tree, const CodonStateSpace* codon_statespace, const ContinuousData& cont_data, RootMean inroot_mean, RootVar inroot_var, Gen& gen) {

        // number of quantitative traits
        size_t ncont = cont_data.GetNsite();

        // number of nodes and branches in tree
        size_t nnode = tree->nb_nodes();
        size_t nbranch = nnode-1;

        // chronogram - relative dates (root has age 1, i.e. tree has depth 1)
        auto chrono = make_node_with_init<chronogram>({tree});
        draw(chrono, gen);

        // covariance matrix; dim = 2 + ncont (dS, dN/dS, traits)
        auto sigma = make_node_with_init<invwishart>(
                {2 + ncont, 0},
                std::vector<double>(2 + ncont, 1.0));
        draw(sigma, gen);

        // mean and variance for normal prior for brownian process at the root
        // (dim: 2 + ncont)
        auto root_mean = make_param<std::vector<double>>(std::forward<RootMean>(inroot_mean));
        auto root_var = make_param<std::vector<double>>(std::forward<RootVar>(inroot_var));

        // brownian process
        auto brownian_process = make_brownian_tree_process(
                tree,
                [&ch = get<value>(chrono)] (int node) -> const double& {return ch[node];},
                one_to_one(get<value>(sigma)),
                root_mean,
                root_var);

        // fix brownian process to observed trait values in extant species
        std::cerr << "brownian process: condition on traits in extant species\n";
        for (size_t i=0; i<ncont; i++)  {
            // first index maps to brownian process, second index maps to continuous data
            brownian_process->SetAndClamp(cont_data, i+2, i);
        }
        // not drawing from brownian process prior (otherwise, makes unreasonable values for dS, dN/dS)
        // instead draw from some jitter around root prior
        brownian_process->PseudoSample(0.1);
        
        // branch dS = delta_time * (exp(X_up(0)) + exp(X_down(0)) / 2

        auto synrate = make_dnode_array<custom_dnode<double>>(nbranch,
            [&ch = get<value>(chrono), &process = *brownian_process] (int branch) {
                return 
                    [&old_age = ch[ch.get_tree().get_older_node(branch)], 
                     &young_age = ch[ch.get_tree().get_younger_node(branch)],
                     &old_val = process[ch.get_tree().get_older_node(branch)],
                     &young_val = process[ch.get_tree().get_younger_node(branch)]] ()   {
                        double exp_old_val = old_val[0] > 10 ? 100 : (old_val[0] < -10 ? 1e-3 : exp(old_val[0]));
                        double exp_young_val = young_val[0] > 10 ? 100 : (young_val[0] < -10 ? 1e-3 : exp(young_val[0]));
                        return 0.5 * (old_age - young_age) * (exp_old_val + exp_young_val);
                    };
            });
        gather(synrate);

        // branch dN/dS = (exp(X_up(1)) + exp(X_down(1))/2

        auto omega = make_dnode_array<custom_dnode<double>>(nbranch,
            [&ch = get<value>(chrono), &process = *brownian_process] (int branch) {
                return 
                    [&old_val = process[ch.get_tree().get_older_node(branch)],
                     &young_val = process[ch.get_tree().get_younger_node(branch)]] ()   {
                        double exp_old_val = old_val[1] > 10 ? 100 : (old_val[1] < -10 ? 1e-3 : exp(old_val[1]));
                        double exp_young_val = young_val[1] > 10 ? 100 : (young_val[1] < -10 ? 1e-3 : exp(young_val[1]));
                        return 0.5 * (exp_old_val + exp_young_val);
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

        // suffstats (branchwise independent contrasts of brownian process) for updating sigma
        auto covmat_ss = ss_factory::make_suffstat_with_init<MultivariateNormalSuffStat>({ncont+2},
                [&process = *brownian_process] (auto& ss) { ss.AddSuffStat(process); });

        // suff stats for dS and dN/dS (bi-poisson)
        // gather lambda is idle: will use mpi reduce instead
        auto dsom_ss = ss_factory::make_suffstat_array<dSOmegaPathSuffStat>(
                nbranch,
                [] (auto& omss) {});

        // nuc rates suff stats
        // gather lambda is idle: will use mpi reduce instead
        auto nucpath_ss = ss_factory::make_suffstat_with_init<NucPathSuffStat>(
                {*codon_statespace},
                [] (auto& nucss) {});

        return make_model(
            chrono_ = move(chrono),
            sigma_ = move(sigma),
            brownian_process_ = move(brownian_process),
            synrate_ = move(synrate),
            omega_ = move(omega),
            nuc_rates_ = move(nuc_rates),
            nucpath_suffstats_ = move(nucpath_ss),
            dsom_suffstats_ = move(dsom_ss),
            covmat_suffstat_ = move(covmat_ss));
    }

    template<class Model, class Gen>
        static auto move_chrono(Model& model, Gen& gen) {

            // update dS on branches around changing node
            auto node_update = tree_factory::do_around_node(
                    get<chrono,value>(model).get_tree(),
                    array_element_gather(synrate_(model)));

            // log prob of substitution mappings on a given branch,
            // given dS and dN/dS on that branch
            auto branch_suffstat_logprob = 
                [&ss = dsom_suffstats_(model), &ds = get<synrate,value>(model), &om = get<omega,value>(model)]
                (int branch) {
                    return ss.get(branch).GetLogProb(ds[branch], om[branch]);
                };

            // compute suffstat log prob for all branches around changing node
            auto node_suffstat_logprob = tree_factory::sum_around_node(
                    get<chrono,value>(model).get_tree(),
                    branch_suffstat_logprob);

            // should recompute new suffstat log prob, but also new brownian process log prob
            // locally around changing node
            auto node_logprob =
                [&process = brownian_process_(model), node_suffstat_logprob] (int node) {
                    return process.GetNodeLogProb(node) + node_suffstat_logprob(node);
                };

            // built-in method of chronogram: move node ages one by one, recursively
            // using the local node update and log prob lambdas just defined
            get<chrono,value>(model).MoveTimes(node_update, node_logprob);
        }

    template<class Model, class Gen>
        static auto move_process(Model& model, Gen& gen) {

            // update dS on given branch (for FilterMove)
            // auto synrate_branch_update = array_element_gather(synrate_(model));

            // update dN/dS on given branch (for FilterMove)
            // auto omega_branch_update = array_element_gather(omega_(model));

            // update dS on branches around changing node
            auto synrate_node_update = tree_factory::do_around_node(
                    get<chrono,value>(model).get_tree(),
                    array_element_gather(synrate_(model)));

            // update dN/dS on branches around changing node
            auto omega_node_update = tree_factory::do_around_node(
                    get<chrono,value>(model).get_tree(),
                    array_element_gather(omega_(model)));

            // see above (move chrono)
            auto branch_suffstat_logprob = 
                [&ss = dsom_suffstats_(model), &ds = get<synrate,value>(model), &om = get<omega,value>(model)]
                (int branch) {
                    return ss.get(branch).GetLogProb(ds[branch], om[branch]);
                };

            auto node_logprob = tree_factory::sum_around_node(
                    get<chrono,value>(model).get_tree(), branch_suffstat_logprob);

            auto no_update = [] (int node) {};
            auto no_logprob = [] (int node) {return 0;};

            // 2 + ncont
            size_t dim = get<sigma,value>(model).size();

            // built-in MCMC moves implemented by the brownian tree process

            // moving dS
            brownian_process_(model).SingleNodeMove(0, 1.0, synrate_node_update, node_logprob);
            brownian_process_(model).SingleNodeMove(0, 0.3, synrate_node_update, node_logprob);
            // brownian_process_(model).FilterMove(0, 10, 0.01, 0.2, synrate_branch_update, branch_suffstat_logprob);

            // moving dN/dS
            brownian_process_(model).SingleNodeMove(1, 1.0, omega_node_update, node_logprob);
            brownian_process_(model).SingleNodeMove(1, 0.3, omega_node_update, node_logprob);
            // brownian_process_(model).FilterMove(1, 10, 0.01, 0.2, omega_branch_update, branch_suffstat_logprob);

            // moving traits does not change sequence likelihood, so no update and no log prob
            // (apart from brownian process, but that's already accounted for in SingleNodeMove)
            for (size_t i=2; i<dim; i++) {
                brownian_process_(model).SingleNodeMove(i, 1.0, no_update, no_logprob);
                brownian_process_(model).SingleNodeMove(i, 0.3, no_update, no_logprob);
                // brownian_process_(model).FilterMove(i, 10, 0.1, 1.0, no_update, no_logprob);
            }
        }

    template<class Model, class Gen>
        static auto move_sigma(Model& model, Gen& gen)  {
            // gibbs resampling based on normalized branchwise independent contrasts
            covmat_suffstat_(model).gather();
            gibbs_resample(sigma_(model), covmat_suffstat_(model), gen);
        }

    // overall schedule for moving all parameters, conditional on substitution mapping

    template<class Model, class Gen>
        static auto move_nuc(Model& model, Gen& gen) {
            nucrates_sm::move_nucrates(nuc_rates_(model), nucpath_suffstats_(model), gen, 1, 1.0);
        }

    // tracing

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

struct coevol_slave {

    template <class Data, class SynRate, class Omega, class NucRR, class NucStat, class Gen>
    static auto make_gene(const Tree* tree, Data& data, SynRate& synrate, Omega& omega, NucRR& nucrr, NucStat& nucstat, Gen& gen) {

        size_t nnode = tree->nb_nodes();
        size_t nbranch = nnode-1;

        auto nuc_matrix = make_dnode_with_init<gtr>(
                {4, nucrr, nucstat, true},
                nucrr,
                nucstat);
        gather(nuc_matrix);

        auto codon_statespace =
            dynamic_cast<const CodonStateSpace*>(data.alignment.GetStateSpace());

        // branch codon matrices
        auto codon_matrices = make_dnode_array_with_init<mgomega>(
                nbranch,
                {codon_statespace, &get<value>(nuc_matrix), 1.0},
                n_to_one(get<value>(nuc_matrix)),
                [&omega] (int branch) {return omega[branch];});
        gather(codon_matrices);

        // root codon matrix with default omega = 1.0 (for root freqs)
        auto root_codon_matrix = make_dnode_with_init<mgomega>(
                {codon_statespace, &get<value>(nuc_matrix), 1.0},
                one_to_one(get<value>(nuc_matrix)),
                one_to_const(1.0));
        gather(root_codon_matrix);

        auto phyloprocess = std::make_unique<PhyloProcess>(tree, &data.alignment,

            // branch lengths
            [&synrate] (int branch) {return synrate[branch];},

            // site-specific rates
            n_to_const(1.0),

            // branch and site specific matrices 
            mn_to_m(get<value>(codon_matrices)),

            // root freqs
            // n_to_one(get<value>(root_codon_matrix).eq_freqs()),
            [&mat = get<value>(root_codon_matrix)] (int site) -> const std::vector<double>& { return mat.eq_freqs(); },

            // no polymorphism
            nullptr);

        phyloprocess->Unfold();

        // general path suff stats in absolute time
        auto path_suffstats = pathss_factory::make_node_path_suffstat(*phyloprocess);

        // relative path suff stats (events are mapped in relative time along a given branch)
        // are computed based on absolute path suff stats but we need to know the absolute branch lengths
        auto rel_path_suffstats = pathss_factory::make_node_relpath_suffstat(
                codon_statespace, 
                *path_suffstats, 
                [&synrate] (int branch) {return synrate[branch];});

        // reducing rel path suffstats into suff stats for dS and dN/dS (bi-poisson)
        auto dsom_ss = pathss_factory::make_dsomega_suffstat(
                get<value>(codon_matrices),
                *rel_path_suffstats,
                [&omega] (int branch) {return omega[branch];});

        // reducing the rel path suffstats into nuc rates suff stats
        auto nucpath_ss = pathss_factory::make_nucpath_suffstat(
                codon_statespace,
                get<value>(codon_matrices),
                get<value>(root_codon_matrix),
                *rel_path_suffstats,
                [&synrate] (int branch) {return synrate[branch];});

        return make_model(
            nuc_matrix_ = move(nuc_matrix),
            codon_matrices_ = move(codon_matrices),
            root_codon_matrix_ = move(root_codon_matrix),
            phyloprocess_ = move(phyloprocess),
            path_suffstats_ = move(path_suffstats),
            rel_path_suffstats_ = move(rel_path_suffstats),
            nucpath_suffstats_ = move(nucpath_ss),
            dsom_suffstats_ = move(dsom_ss));
    }

    template <class Data, class SynRate, class Omega, class NucRR, class NucStat, class Gen>
        static auto make_gene_array(const Tree* tree, Data& data, SynRate& synrate, Omega& omega, NucRR& nucrr, NucStat& nucstat, Gen& gen)   {

            auto lambda = [&tree, &data, &synrate, &omega, &nucrr, &nucstat, &gen] (int i) {
                return make_gene(tree, *data[i], synrate, omega, nucrr, nucstat, gen);
            };

            return make_model_array(data.size(), lambda);
        }

    template <class Data, class Gen>
    static auto make(const Tree* tree, Data& data, Gen& gen) {

        size_t nnode = tree->nb_nodes();
        size_t nbranch = nnode-1;

        auto synrate = std::make_unique<std::vector<double>>(nbranch,0.1);
        auto omega = std::make_unique<std::vector<double>>(nbranch,0.2);

        auto nucrr = std::make_unique<std::vector<double>>(6, 1./6);
        auto nucstat = std::make_unique<std::vector<double>>(4, 1./4);

        auto gene_model_array = make_gene_array(tree, data, *synrate, *omega, *nucrr, *nucstat, gen);

        // reducing dsom path suffstats across genes
        auto dsom_ss = ss_factory::make_suffstat_array<dSOmegaPathSuffStat>(
                nbranch,
                [&array = *gene_model_array, nbranch] (auto& dsomss) {
                    for (auto& gene_model : array)   {
                        auto& gene_dsomss = dsom_suffstats_(gene_model);
                        for (size_t branch=0; branch<nbranch; branch++) {
                            dsomss[branch].Add(gene_dsomss.get(branch));
                        }
                    }
                });

        // reducing nuc path suffstats across genes
        auto codon_statespace = dynamic_cast<const CodonStateSpace*>(data[0]->alignment.GetStateSpace());
        auto nuc_ss = ss_factory::make_suffstat_with_init<NucPathSuffStat>(
                {*codon_statespace},
                [&array = *gene_model_array] (auto& nucss) {
                    for (auto& gene_model : array)   {
                        nucss.Add(nucpath_suffstats_(gene_model).get());
                    }
                });

        return make_model(
                synrate_ = move(synrate),
                omega_ = move(omega),
                nucrr_ = move(nucrr),
                nucstat_ = move(nucstat),
                gene_model_array_ = move(gene_model_array),
                nucpath_suffstats_ = move(nuc_ss),
                dsom_suffstats_ = move(dsom_ss));
    }

    template <class Model, class Gen>
        static auto resample_sub(Model& model, Gen& gen)  {
            phyloprocess_(model).Move(1.0);
        }

    template <class Model, class Gen>
    static auto gene_resample_sub(Model& model, Gen& gen)    {
        for (auto& gene_model : get<gene_model_array>(model)) {
            resample_sub(gene_model, gen);
        }
    }

    template <class Model>
    static auto gene_update_nuc_matrix(Model& model) {
        for (auto& gene_model : get<gene_model_array>(model)) {
            gather(nuc_matrix_(gene_model));
        }
    }

    template <class Model>
    static auto gene_update_codon_matrices(Model& model) {
        for (auto& gene_model : get<gene_model_array>(model)) {
            gather(root_codon_matrix_(gene_model));
            gather(codon_matrices_(gene_model));
        }
    }

    template <class Model>
    static auto gene_collect_path_suffstats(Model& model) {
        for (auto& gene_model : get<gene_model_array>(model)) {
            path_suffstats_(gene_model).gather();
            rel_path_suffstats_(gene_model).gather();
        }
    }

    template <class Model>
    static auto gene_collect_dsomsuffstats(Model& model) {
        for (auto& gene_model : get<gene_model_array>(model)) {
            dsom_suffstats_(gene_model).gather();
        }
        dsom_suffstats_(model).gather();
    }

    template <class Model>
    static auto gene_collect_nucsuffstats(Model& model) {
        for (auto& gene_model : get<gene_model_array>(model)) {
            nucpath_suffstats_(gene_model).gather();
        }
        nucpath_suffstats_(model).gather();
    }
};
