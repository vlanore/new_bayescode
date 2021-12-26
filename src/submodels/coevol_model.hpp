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
#include "lib/Chronogram.hpp"
#include "lib/ChronoBranchLengths.hpp"
#include "lib/InverseWishart.hpp"
#include "lib/MultivariateBrownianTreeProcess.hpp"
#include "lib/CodonSequenceAlignment.hpp"
#include "submodels/move_reporter.hpp"
#include "submodels/nucrates_sm.hpp"
#include "submodels/mgomega.hpp"
#include "submodels/invwishart.hpp"
#include "submodels/submodel_external_interface.hpp"
#include "submodels/suffstat_wrappers.hpp"
#include "bayes_toolbox.hpp"

TOKEN(chronogram)
TOKEN(branch_lengths)
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
        size_t nbranch = nnode-1;

        auto chronogram = make_chrono(tree);
        chronogram->Sample();

        auto branch_lengths = make_chrono_branch_lengths(tree,
                [&chrono = *chronogram] (int node) {return chrono[node];});
        branch_lengths->Update();

        /*
        auto kappa = make_node_array<gamma_mi>(ncont+2, n_to_const(1.0), n_to_const(1.0));
        draw(kappa, gen);
        */
        auto kappa = std::make_unique<std::vector<double>>(ncont+2, 1.0);

        auto sigma = make_node_with_init<invwishart>(
                {ncont+2, 1},
                [&k = *kappa] () { return k; });

        draw(sigma, gen);
        auto& s = get<value>(sigma);
        s *= 0.1;

        auto root_mean = make_param<std::vector<double>>(std::forward<RootMean>(inroot_mean));
        auto root_var = make_param<std::vector<double>>(std::forward<RootVar>(inroot_var));

        std::cerr << "brownian process\n";
        auto brownian_process = make_brownian_tree_process(
                tree,
                [&chrono = *chronogram] (int node) -> const double& {return chrono[node];},
                [&s = get<value>(sigma)] () -> const CovMatrix& {return s;},
                root_mean,
                root_var);

        std::cerr << "brownian process: set and clamp\n";
        for (size_t i=0; i<ncont; i++)  {
            brownian_process->SetAndClamp(cont_data, i+2, i);
        }
        brownian_process->PseudoSample(0.1);
        
        std::cerr << "brownian process ok\n";

        auto synrate = make_branch_lengths(tree,
                [&process = *brownian_process] (int node) -> const std::vector<double>& {return process[node];},
                [&chrono = *chronogram] (int node) -> const double& {return chrono[node];},
                0);
        synrate->Update();

        auto omega = make_branch_means(tree,
                [&process = *brownian_process] (int node) -> const std::vector<double>& {return process[node];},
                1);
        omega->Update();

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

        auto codon_matrices = make_dnode_array_with_init<mgomega>(
                nnode,
                {codon_statespace, &get<nuc_matrix,value>(nuc_rates), 1.0},
                (const SubMatrix&) get<nuc_matrix,value>(nuc_rates),
                [&om = *omega] (int node) {return node ? om[node-1] : 1.0;}
                );
        gather(codon_matrices);

        auto phyloprocess = std::make_unique<PhyloProcess>(tree, &codon_data,

            // branch lengths
            [&ds = *synrate] (int branch) {return ds[branch];},

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
        auto rel_path_suffstats = ss_factory::make_suffstat_array_with_init<RelativePathSuffStat>(
                nnode,
                {codon_statespace->GetNstate()},
                [&pss = *path_suffstats, &ds = *synrate] (auto& rpss, int node) {
                    rpss[node].Add(pss.get(node), node ? ds[node-1] : 1.0); 
                },
                nnode);

        auto nucpath_ss = pathss_factory::make_nucpath_suffstat(codon_statespace, get<value>(codon_matrices), *path_suffstats);

        auto dsom_ss = ss_factory::make_suffstat_array<dSOmegaPathSuffStat>(
                nbranch,
                [&mat = get<value>(codon_matrices), &rpss = *rel_path_suffstats, &om = *omega] (auto& omss, int branch) { omss[branch].AddSuffStat(mat[branch+1], rpss.get(branch+1), om[branch]); },
                nbranch);

        /*
        auto dsom_ss = ss_factory::make_suffstat_array<dSOmegaPathSuffStat>(
                nbranch,
                [&mat = get<value>(codon_matrices), &pss = *path_suffstats, &ds = *synrate, &om = *omega] (auto& omss, int branch) { omss[branch].AddSuffStat(mat[branch+1], pss.get(branch+1), ds[branch], om[branch]); },
                nbranch);
        */

        /*
        auto dsom_ss = pathss_factory::make_dsomega_suffstat(
                get<value>(codon_matrices), 
                *path_suffstats,
                [&ds = *synrate] (int node) {return node ? ds[node-1] : 1.0;},
                [&om = *omega] (int node) {return node ? om[node-1] : 1.0;});
        */

        auto covmat_ss = ss_factory::make_suffstat_with_init<MultivariateNormalSuffStat>({ncont+2},
                [&process = *brownian_process] (auto& ss) { ss.AddSuffStat(process); });

        return make_model(
            chronogram_ = move(chronogram),
            branch_lengths_ = move(branch_lengths),
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

    template<class Model, class Gen>
        static auto move_chrono(Model& model, Gen& gen) {

            auto update = 
                [&bl = branch_lengths_(model), &ds = synrate_(model)] (int node) {
                    bl.LocalNodeUpdate(node); 
                    ds.LocalNodeUpdate(node); 
                };

            auto branch_logprob = 
                [&ss = dsom_suffstats_(model), &ds = synrate_(model), &om = omega_(model), &process = brownian_process_(model)]
                (int branch) {
                    return process.GetLocalLogProb(branch) + ss.get(branch).GetLogProb(ds[branch], om[branch]);
                };

            auto logprob =
                [&bl = branch_lengths_(model), branch_logprob] (int node) {
                    return bl.sum_around_node(branch_logprob, node); 
                };

            chronogram_(model).MoveTimes(update, logprob);
        }

    template<class Model, class Gen>
        static auto move_process(Model& model, Gen& gen) {
            auto update = 
                [&ds = synrate_(model), &om = omega_(model)] (int node) {
                    ds.LocalNodeUpdate(node); 
                    om.LocalNodeUpdate(node); 
                };

            auto branch_logprob = 
                [&ss = dsom_suffstats_(model), &ds = synrate_(model), &om = omega_(model)]
                (int branch) {
                    return ss.get(branch).GetLogProb(ds[branch], om[branch]);
                };

            auto logprob =
                [&bl = branch_lengths_(model), branch_logprob] (int node) {
                    return bl.sum_around_node(branch_logprob, node); 
                };

            brownian_process_(model).SingleNodeMove(1.0, update, logprob);
            brownian_process_(model).SingleNodeMove(0.3, update, logprob);
        }

    template<class Model, class Gen>
        static auto move_sigma(Model& model, Gen& gen)  {
            covmat_suffstat_(model).gather();
            gibbs_resample(sigma_(model), covmat_suffstat_(model), gen);
        }

    template<class Model, class Gen>
        static auto move_params(Model& model, Gen& gen) {

            // path_suffstats_(model).gather();
            dsom_suffstats_(model).gather();

            move_chrono(model, gen);
            move_process(model, gen);
            move_sigma(model, gen);
            gather(codon_matrices_(model));

            nucpath_suffstats_(model).gather();
            nucrates_sm::move_nucrates(nuc_rates_(model), nucpath_suffstats_(model), gen, 1, 1.0);
            gather(codon_matrices_(model));
        }

    template<class Model>
        static auto get_total_length(Model& model)  {
            auto& bl = branch_lengths_(model);
            double tot = 0;
            for (size_t b=0; b<bl.nb_branches(); b++)   {
                tot += bl[b];
            }
            return tot;
        }

    template<class Model>
        static auto get_total_ds(Model& model)  {
            auto& ds = synrate_(model);
            double tot = 0;
            for (size_t b=0; b<ds.nb_branches(); b++)   {
                tot += ds[b];
            }
            return tot;
        }

    template<class Model>
        static auto get_mean_omega(Model& model)  {
            auto& om = omega_(model);
            double tot = 0;
            for (size_t b=0; b<om.nb_branches(); b++)   {
                tot += om[b];
            }
            tot /= om.nb_branches();
            return tot;
        }
};
