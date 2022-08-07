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
#include "processes/path2norm.hpp"

TOKEN(tree)
TOKEN(global_omega)
TOKEN(chrono)
TOKEN(tau)
TOKEN(log_synrate)
TOKEN(synrate)
TOKEN(dsom_suffstats)
TOKEN(omegapath_suffstats)
// TOKEN(logsynrate_branchsuffstats)
TOKEN(tau_suffstats)

struct brownian_clock_globom {

    template <class Gen>
    static auto make(std::string treefile, std::vector<std::vector<double>>& suffstat, Gen& gen)  {

        std::ifstream is(treefile);
        NHXParser parser(is);
        auto my_tree = make_from_parser(parser);
        auto tree = my_tree.get();

        std::ifstream tis(treefile.c_str());
        std::string bls;
        tis >> bls;
        auto bl_stream = std::stringstream(bls);
        NHXParser bl_parser(bl_stream);
        auto node_lengths = node_container_from_parser<double>(
            bl_parser, [](int i, const AnnotatedTree& t) { return atof(t.tag(i, "length").c_str()); });

        // uniform prior over divergence times
        // relative dates (root has age 1, i.e. tree has depth 1)
        auto chrono = make_node_with_init<chronogram>({tree});
        get<value>(chrono).get_ages_from_lengths(node_lengths);
        // draw(chrono, gen);

        /*
        std::stringstream s;
        auto branchlength = [&ch = get<value>(chrono), &t=tree] (int branch) {
            return ch[t->get_older_node(branch)] - ch[t->get_younger_node(branch)];
        };
        newick_output::print(s, tree, 
                [] (int) {return "";}, 
                branchlength);
                // [&bl=branch_lengths] (int branch) {return bl[branch+1];});
        std::cout << s.str() << '\n';
        std::cout.flush();
        */

        // precision (inverse variance) per time unit
        auto tau = make_node<gamma_mi>(1.0, 1.0);
        draw(tau, gen);
        get<value>(tau) = 5.0;

        // normal dist for root value
        auto root_mean = 0;
        auto root_var = 4;

        std::cerr << "log synrate (brownian)\n";
        // geometric brownian subst rates
        auto log_synrate = make_node_tree_process_with_inits<univariate_brownian, univariate_normal>(
                *tree, 
                time_frame(get<value>(chrono)), 
                0, {0},
                process_params(one_to_one(tau)),
                root_params(one_to_const(root_mean),one_to_const(root_var)));

        std::cerr << "forward draw\n";
        tree_process_methods::forward_draw(log_synrate, gen);
        /*
        std::cerr << "conditional draw\n";
        tree_process_methods::conditional_draw(log_synrate, gen);
        */

        std::cerr << "syn rate (sum of exps)\n";
        // branchwise sums of exp
        auto synrate = branch_map::make_branch_sums(
                get<value>(chrono), 
                log_synrate,
                [] (double x) {return exp(x);});
        gather(synrate);

        /*
        auto synrate = branch_map::make_branch_exp_mid_sums(log_synrate);
        gather(synrate);
        */

        std::cerr << "dn/ds\n";
        // global dN/dS uniform across sites and branches
        auto global_omega = make_node<gamma_mi>(1.0, 1.0);
        draw(global_omega, gen);

        std::cerr << "get suffstats from file\n";
        // mapping suffstats for dS and dN
        auto dsom_ss = pathss_factory::make_dsom_suffstat_from_mappings(suffstat);

        // collapse dS dN mapping branch suffstats into one single suffstat for global omega
        auto omega_ss = pathss_factory::make_omega_suffstat(
                n_to_n(get<value>(synrate)),
                // [&ds = get<value>(synrate)] (int branch) -> double {return ds[branch];},
                *dsom_ss);

        /*
        std::cerr << "log syn rate branch suff stats\n";
        auto logsynrate_branchsuffstats = path2norm::make_branch_normss(
                get<value>(chrono),
                *dsom_ss,
                5.0);
        */

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

        std::cerr << "return model\n";
        return make_model(
            tree_ = move(my_tree),
            global_omega_ = move(global_omega),
            chrono_ = move(chrono),
            tau_ = move(tau),
            log_synrate_ = move(log_synrate),
            synrate_ = move(synrate),
            dsom_suffstats_ = move(dsom_ss),
            omegapath_suffstats_ = move(omega_ss),
            // logsynrate_branchsuffstats_ = move(logsynrate_branchsuffstats),
            tau_suffstats_ = move(tau_ss));
    }

    template<class Model, class Gen>
        static auto move_chrono(Model& model, Gen& gen) {

            auto branch_update = array_element_gather(synrate_(model));

            auto suffstat_logprob = suffstat_array_element_logprob(
                    n_to_n(get<synrate,value>(model)),
                    n_to_one(get<global_omega,value>(model)),
                    dsom_suffstats_(model));

            auto synrate_logprob = tree_process_methods::around_node_logprob(get<log_synrate>(model));
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

            auto target = tree_process_methods::make_prior_importance_sampler(
                    get<log_synrate>(model),
                    branch_update, branch_logprob);

            auto pf = tree_process_methods::make_particle_filter(
                    get<log_synrate>(model), target, 1000);

            pf.run(false, 0, gen);
            pf.run(true, 0, gen);
            pf.run(true, 0.1, gen);
        }

    template<class Model, class Gen>
        static auto branch_pf_move_ds(Model& model, Gen& gen)  {
            
            gather(logsynrate_branchsuffstats_(model));

            auto branch_update = array_element_gather(synrate_(model));

            auto branch_logprob = suffstat_array_element_logprob(
                    n_to_n(get<synrate,value>(model)),
                    n_to_one(get<global_omega,value>(model)),
                    dsom_suffstats_(model));

            auto target = tree_process_methods::make_pseudo_branch_prior_importance_sampler(
                    get<log_synrate>(model),
                    branch_update, branch_logprob);

            auto pf = tree_process_methods::make_particle_filter(
                    get<log_synrate>(model), target, 1000);

            pf.run(false, 0, gen);
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

            /*
            tree_process_methods::branch_by_branch_mh_move(get<log_synrate>(model), 1.0,
                    branch_update, branch_logprob, gen);

            tree_process_methods::branch_by_branch_mh_move(get<log_synrate>(model), 0.3,
                    branch_update, branch_logprob, gen);
            */
        }

    template<class Model, class Gen>
        static auto move_params(Model& model, Gen& gen) {

            move_chrono(model, gen);

            // move branch times and rates
            move_ds(model, gen);
            // pf_move_ds(model, gen);
            // branch_pf_move_ds(model, gen);

            // move variance parameter of Brownian process
            // tau_suffstats_(model).gather();
            tau_suffstats_(model).get().Clear();
            tree_process_methods::add_branch_suffstat<brownian_tau>(get<log_synrate>(model),tau_suffstats_(model).get());
            gibbs_resample(tau_(model), tau_suffstats_(model), gen);

            // move omega
            move_omega(model, gen);
        }

    template<class Model, class Gen>
        static auto move_omega(Model& model, Gen& gen)  {
            // gibbs version
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
        static auto get_mean_log_ds(Model& model)  {
            auto& x = get<log_synrate,node_values>(model);
            double tot = 0;
            for (size_t i=0; i<x.size(); i++)   {
                tot += x[i];
            }
            return tot / x.size();
        }

    template<class Model>
        static std::string get_chronogram(Model& model) {
            auto& t = get<tree>(model);
            auto branchlength = [&ch = get<chrono,value>(model), &t] (int branch) {
                return ch[t.get_older_node(branch)] - ch[t.get_younger_node(branch)];
            };
            std::stringstream s;
            newick_output::print(s, &t, branchlength);
            return s.str();
        }

    template<class Model>
        static std::string get_annotated_tree(Model& model) {
            auto& t = get<tree>(model);
            auto branchlength = [&ch = get<chrono,value>(model), &t] (int branch) {
                return ch[t.get_older_node(branch)] - ch[t.get_younger_node(branch)];
            };
            auto nodeval = [&br = get<log_synrate,node_values>(model)] (int node) {
                // return br[node];
                return exp(br[node]);
            };
            std::stringstream s;
            newick_output::print(s, &t, nodeval, branchlength);
            return s.str();
        }
};
