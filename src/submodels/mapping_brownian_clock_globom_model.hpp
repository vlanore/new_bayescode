#pragma once

#include <cmath>
#include <sstream>
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
#include "tree/newick_output.hpp"

TOKEN(global_omega)
TOKEN(chrono)
TOKEN(tau)
TOKEN(brownian_process)
TOKEN(synrate)
TOKEN(dsom_suffstats)
TOKEN(omegapath_suffstats)
TOKEN(tau_suffstats)


static auto make_brownian_tree_process(const Tree* intree, std::function<const double& (int)> innode_age, std::function<double ()> intau, std::function<double ()> inrootmean, std::function<double ()> inrootvar)  {
    return std::make_unique<UnivariateBrownianTreeProcess>(intree, innode_age, intau, inrootmean, inrootvar);
}

struct brownian_clock_globom {

    template <class Gen>
    static auto make(const Tree* tree, std::vector<std::vector<double>>& suffstat, Gen& gen)  {

        size_t nbranch = tree->nb_nodes() - 1;

        // relative dates (root has age 1, i.e. tree has depth 1)
        auto chrono = make_node_with_init<chronogram>({tree});
        draw(chrono, gen);

        // strict clock
        // absolute synonymous substitution rate (per tree depth)
        auto tau= make_node<gamma_mi>(1.0, 1.0);
        draw(tau, gen);

        auto root_mean = 0;
        auto root_var = 4;

        auto brownian_process = make_brownian_tree_process(
                tree,
                [&ch = get<value>(chrono)] (int node) -> const double& {return ch[node];},
                one_to_one(tau),
                one_to_one(root_mean),
                one_to_one(root_var));

        brownian_process->PseudoSample(0.1);

        auto synrate = branch_map::make_branch_sums(get<value>(chrono), *brownian_process);
        gather(synrate);

        // global dN/dS uniform across sites and branches
        auto global_omega = make_node<gamma_mi>(1.0, 1.0);
        draw(global_omega, gen);

        // dsom_ss 
        // Ms ~ Poisson(dS_norm * ds)
        // Mn ~ Poisson(dN_norm * ds * om) ~ om^Mn * exp(-dN_norm*ds * om)
        // om_suffstat: (sum_j Mn_j, sum_j ds_j * dN_norm_j)

        auto dsom_ss = ss_factory::make_suffstat_array<dSOmegaPathSuffStat>(
                nbranch, 
                [&suffstat, &nbranch] (auto& omss) {
                    for (size_t branch=0; branch<nbranch; branch++) {
                        omss[branch].Add(suffstat[0][branch], suffstat[1][branch], suffstat[2][branch], suffstat[3][branch]);
                    }
                
                });

        (*dsom_ss).gather();

        auto omega_ss = ss_factory::make_suffstat<OmegaPathSuffStat>(
                [&ds_tree = get<value>(synrate), &dsom_ss_tree = *dsom_ss] (auto& omss, int branch) { 
                    auto& dsomss = dsom_ss_tree.get(branch);
                    double ds = ds_tree[branch];
                    omss.PoissonSuffStat::AddSuffStat(dsomss.GetNonSynCount(), ds * dsomss.GetNonSynBeta());
                },
                nbranch);

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
            dsom_suffstats_ = move(dsom_ss),
            omegapath_suffstats_ = move(omega_ss),
            tau_suffstats_ = move(tau_ss));
    }

    template<class Model, class Gen>
        static auto move_chrono(Model& model, Gen& gen) {

            const Tree& tree = get<chrono,value>(model).get_tree();

            auto node_update = tree_factory::do_around_node(
                    tree,
                    array_element_gather(synrate_(model)));

            // compute suffstat log prob for all branches around changing node
            auto node_suffstat_logprob = tree_factory::sum_around_node(tree,
                    tree_factory::suffstat_logprob(dsom_suffstats_(model),
                                    n_to_n(get<synrate,value>(model)),
                                    n_to_one(get<global_omega,value>(model))));

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
                    tree_factory::suffstat_logprob(dsom_suffstats_(model),
                                    n_to_n(get<synrate,value>(model)),
                                    n_to_one(get<global_omega,value>(model))));

            brownian_process_(model).SingleNodeMove(1.0, synrate_node_update, node_logprob);
            brownian_process_(model).SingleNodeMove(0.3, synrate_node_update, node_logprob);
        }

    template<class Model, class Gen>
        static auto move_params(Model& model, Gen& gen) {
            // move branch times and rates
            move_chrono(model, gen);
            move_ds(model, gen);

            // move variance parameter of Brownian process
            tau_suffstats_(model).gather();
            // gibbs_resample(tau_(model), tau_suffstats_(model), gen);

            // move omega
            omegapath_suffstats_(model).gather();
            gibbs_resample(global_omega_(model), omegapath_suffstats_(model), gen);
        }

    template<class Model>
        static auto get_total_time(Model& model)  {
            return get<chrono,value>(model).GetTotalTime();
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

    template<class Model>
        static std::string get_annotated_tree(Model& model) {
            const Tree& tree = get<chrono,value>(model).get_tree();
            auto branchlength = [&ch = get<chrono,value>(model), &t=tree] (int branch) {
                return ch[t.get_older_node(branch)] - ch[t.get_younger_node(branch)];
            };
            auto nodeval = [&br = get<brownian_process>(model)] (int node) {
                // return br[node];
                return exp(br[node]);
            };
            std::stringstream s;
            newick_output::print(s, &tree, nodeval, branchlength);
            return s.str();
        }
};
