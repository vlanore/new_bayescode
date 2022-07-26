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
#include "lib/PoissonSuffStat.hpp"
#include "chronogram.hpp"
#include "submodels/move_reporter.hpp"
#include "submodels/submodel_external_interface.hpp"
#include "submodels/suffstat_wrappers.hpp"
#include "bayes_toolbox.hpp"
#include "tree/tree_factory.hpp"
#include "univariate_branch_map.hpp"
#include "tree/newick_output.hpp"
#include "processes/univariate_brownian.hpp"
#include "montecarlo/tree_process.hpp"

TOKEN(global_omega)
TOKEN(chrono)
TOKEN(tau)
TOKEN(log_synrate)
TOKEN(synrate)
TOKEN(dsom_suffstats)
TOKEN(omegapath_suffstats)
TOKEN(tau_suffstats)


struct brownian_clock_globom {

    template <class Gen>
    static auto make(const Tree* tree, std::vector<std::vector<double>>& suffstat, Gen& gen)  {

        // uniform prior over divergence times
        // relative dates (root has age 1, i.e. tree has depth 1)
        auto chrono = make_node_with_init<chronogram>({tree});
        draw(chrono, gen);

        // precision (inverse variance) per time unit
        auto tau = make_node<gamma_mi>(1.0, 1.0);
        draw(tau, gen);

        // normal dist for root value
        auto root_mean = 0;
        auto root_var = 4;

        // geometric brownian subst rates
        auto log_synrate = make_node_tree_process<univariate_brownian>(
                *tree, 
                time_frame(get<value>(chrono)), 
                one_to_one(tau),
                one_to_one(root_mean),
                one_to_one(root_var));
        // draw(log_synrate, gen);
        tree_process_methods::conditional_draw(log_synrate, gen);

        // branchwise sums of exp
        auto synrate = branch_map::make_branch_sums(
                get<value>(chrono), 
                get<value>(log_synrate));
        gather(synrate);

        // global dN/dS uniform across sites and branches
        auto global_omega = make_node<gamma_mi>(1.0, 1.0);
        draw(global_omega, gen);

        // mapping suffstats for dS and dN
        auto dsom_ss = pathss_factory::make_dsom_suffstat_from_mappings(suffstat);

        // collapse dS dN mapping branch suffstats into one single suffstat for global omega
        auto omega_ss = pathss_factory::make_omega_suffstat(
                n_to_n(get<value>(synrate)),
                *dsom_ss);

        // suffstats for tau (sum of squared scaled branchwise contrasts of Brownian process)
        auto tau_ss = ss_factory::make_suffstat<PoissonSuffStat>( [] (auto& ss) {} );
        /*
        auto tau_ss = ss_factory::make_suffstat<PoissonSuffStat>(
                [&process = log_synrate] (auto& ss) { 
                tree_process_methods::add_branch_suffstat<brownian_tau>(process,ss);
                });
        std::cerr << "preliminary gather of tau ss\n";
        (*tau_ss).gather();
        */

        return make_model(
            global_omega_ = move(global_omega),
            chrono_ = move(chrono),
            tau_ = move(tau),
            log_synrate_ = move(log_synrate),
            synrate_ = move(synrate),
            dsom_suffstats_ = move(dsom_ss),
            omegapath_suffstats_ = move(omega_ss),
            tau_suffstats_ = move(tau_ss));
    }

    template<class Model, class Gen>
        static auto move_chrono(Model& model, Gen& gen) {

            auto branch_update = array_element_gather(synrate_(model));

            auto suffstat_logprob = suffstat_array_element_logprob(
                    n_to_n(get<synrate,value>(model)),
                    n_to_one(get<global_omega,value>(model)),
                    dsom_suffstats_(model));

            auto synrate_logprob = tree_process_methods::branch_logprob(get<log_synrate>(model));

            auto branch_logprob = sum_of_lambdas(suffstat_logprob, synrate_logprob);

            get<chrono,value>(model).MoveTimes(branch_update, branch_logprob);
        }

    template<class Model, class Gen>
        static auto pf_move_ds(Model& model, Gen& gen)  {
            
            auto branch_update = array_element_gather(synrate_(model));

            auto branch_logprob = suffstat_array_element_logprob(
                    n_to_n(get<synrate,value>(model)),
                    n_to_one(get<global_omega,value>(model)),
                    dsom_suffstats_(model));

            auto proposal = tree_process_methods::make_conditional_sampler(
                    get<log_synrate>(model));
                    
            auto target = tree_process_methods::make_prior_importance_sampler(
                    get<log_synrate>(model),
                    proposal,
                    branch_update, branch_logprob);

            auto pf = tree_process_methods::make_particle_filter(
                    get<log_synrate>(model), target, 1000);

            pf.run(gen);
        }

    template<class Model, class Gen>
        static auto move_ds(Model& model, Gen& gen) {

            auto branch_update = array_element_gather(synrate_(model));

            auto branch_logprob = suffstat_array_element_logprob(
                    n_to_n(get<synrate,value>(model)),
                    n_to_one(get<global_omega,value>(model)),
                    dsom_suffstats_(model));

            tree_process_methods::node_by_node_mh_move(get<log_synrate>(model), 1.0,
                    branch_update, branch_logprob, gen);

            tree_process_methods::node_by_node_mh_move(get<log_synrate>(model), 0.3,
                    branch_update, branch_logprob, gen);
        }

    template<class Model, class Gen>
        static auto move_params(Model& model, Gen& gen) {
            // move branch times and rates
            move_chrono(model, gen);
            move_ds(model, gen);
            pf_move_ds(model, gen);

            // move variance parameter of Brownian process
            // tau_suffstats_(model).gather();
            tau_suffstats_(model).get().Clear();
            tree_process_methods::add_branch_suffstat<brownian_tau>(get<log_synrate>(model),tau_suffstats_(model).get());
            gibbs_resample(tau_(model), tau_suffstats_(model), gen);

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
            auto nodeval = [&br = get<log_synrate,value>(model)] (int node) {
                // return br[node];
                return exp(br[node]);
            };
            std::stringstream s;
            newick_output::print(s, &tree, nodeval, branchlength);
            return s.str();
        }
};
