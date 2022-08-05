
#include "tree/tree_factory.hpp"

struct tree_process_methods  {

    // ****************************
    // forward draw (assumes process is unclamped)

    template<class Process, class Params, class... Keys, class Gen>
    static void root_draw(Process& process, const Params& params, std::tuple<Keys...>, Gen& gen)   {
        auto& v = get<node_values>(process)[get<tree_field>(process).root()];
        node_root_distrib_t<Process>::draw(v, get<Keys>(params)()..., gen);
    }

    template<class Process, class Params, class... Keys, class Gen>
    static void path_draw(Process& process, int node, const Params& params, std::tuple<Keys...>, Gen& gen)   {
        auto& tree = get<tree_field>(process);
        auto& timeframe = get<time_frame_field>(process);
        auto node_vals = get<node_values>(process);
        auto path_vals = get<path_values>(process);

        node_distrib_t<Process>::path_draw(path_vals[tree.get_branch(node)], 
            node_vals[node], node_vals[tree.parent(node)],
            timeframe(node), timeframe(tree.parent(node)),
            get<Keys>(params)()..., gen);
    }

    template<class Tree, class Process, class Gen>
    static void recursive_forward_draw(const Tree& tree, int node, Process& process, Gen& gen)    {

        if (tree.is_root(node)) {
            root_draw(process, 
                get<struct root_params>(process), param_keys_t<node_root_distrib_t<Process>>(), gen);
        }
        else    {
            path_draw(process, node,
                get<params>(process), param_keys_t<node_distrib_t<Process>>(), gen);
        }

        for (auto c : tree.children(node))  {
            recursive_forward_draw(tree, c, process, gen);
        }
    }

    template<class Process, class Gen>
    static void forward_draw(Process& process, Gen& gen)  {

        auto& tree = get<tree_field>(process);
        recursive_forward_draw(tree, tree.root(), process, gen);
    }

    // ****************************
    // log probs

    template<class Process, class Params, class... Keys>
    static double root_logprob(Process& process, const Params& params, std::tuple<Keys...>)   {

        auto& v = get<node_values>(process)[get<tree_field>(process).root()];
        return node_root_distrib_t<Process>::logprob(v, get<Keys>(params)()...);
    }

    template<class Process, class Params, class... Keys>
    static double path_logprob(Process& process, int node, const Params& params, std::tuple<Keys...>)   {

        auto& tree = get<tree_field>(process);
        auto timeframe = get<time_frame_field>(process);
        auto node_vals = get<node_values>(process);
        auto path_vals = get<path_values>(process);

        return node_distrib_t<Process>::path_logprob(path_vals[tree.get_branch(node)], 
            node_vals[node], node_vals[tree.parent(node)],
            timeframe(node), timeframe(tree.parent(node)),
            get<Keys>(params)()...);
    }

    template<class Process>
    static double root_logprob(Process& process)    {
        return root_logprob(process,
                get<struct root_params>(process), param_keys_t<node_root_distrib_t<Process>>());
    }

    template<class Process>
    static double path_logprob(Process& process, int node)  {
        return path_logprob(process, node,
                get<params>(process), param_keys_t<node_distrib_t<Process>>());
    }

    template<class Process>
    static double around_node_logprob(Process& process, int node) {
        auto& tree = get<tree_field>(process);
        double tot = 0;
        if (tree.is_root(node)) {
            tot = root_logprob(process);
        }
        else    {
            tot = path_logprob(process, node);

        }
        for (auto c : get<tree_field>(process).children(node))  {
            tot += path_logprob(process, c);
        }
        return tot;
    }

    template<class Process>
    static auto around_node_logprob(Process& process)   {
        return [&p = process] (int node) {return around_node_logprob(p, node);};
    }

    // ****************************
    // single node moves 

    template<class Tree, class Process, class Kernel, class BranchUpdate, class BranchLogProb, class Gen>
    static void single_node_mh_move(Tree& tree, int node, Process& process, Kernel kernel, BranchUpdate update, BranchLogProb logprob, Gen& gen)  {

        auto& x = get<node_values>(process)[node];
        auto& paths = get<path_values>(process);
        auto bk = x;
        double logprobbefore = around_node_logprob(process,node) + tree_factory::sum_around_node(tree, logprob)(node);
        double logh = kernel(x, gen);
        if (! tree.is_root(node))   {
            logh += paths[node].adapt_to_younger_end(bk, x);
        }
        for (auto c : tree.children(node))  {
            logh += paths[c].adapt_to_older_end(bk, x);
        }
        tree_factory::do_around_node(tree, update)(node);
        double logprobafter = around_node_logprob(process,node) + tree_factory::sum_around_node(tree, logprob)(node);
        double delta = logprobafter - logprobbefore + logh;
        bool accept = decide(delta, gen);
        if (! accept)   {
            if (! tree.is_root(node))   {
                paths[node].adapt_to_younger_end(x, bk);
            }
            for (auto c : tree.children(node))  {
                paths[c].adapt_to_older_end(x, bk);
            }
            x = bk;
            tree_factory::do_around_node(tree, update)(node);
        }
    }

    template<class Tree, class Process, class Kernel, class BranchUpdate, class BranchLogProb, class Gen>
    static void recursive_node_mh_move(Tree& tree, int node, Process& process, Kernel kernel, BranchUpdate update, BranchLogProb logprob, Gen& gen)    {
        single_node_mh_move(tree, node, process, kernel, update, logprob, gen);
        for (auto c : tree.children(node)) {
            recursive_node_mh_move(tree, c, process, kernel, update, logprob, gen);
        }
        single_node_mh_move(tree, node, process, kernel, update, logprob, gen);
    }

    template<class Process, class Kernel, class BranchUpdate, class BranchLogProb, class Gen>
    static void node_by_node_mh_move(Process& process, Kernel kernel, BranchUpdate update, BranchLogProb logprob, Gen& gen)  {
        auto& tree = get<tree_field>(process);
        recursive_node_mh_move(tree, tree.root(), process, kernel, update, logprob, gen);
    }

    template<class Process, class BranchUpdate, class BranchLogProb, class Gen>
    static void node_by_node_mh_move(Process& process, double tuning, BranchUpdate update, BranchLogProb logprob, Gen& gen)  {
        auto& tree = get<tree_field>(process);
        auto kernel = default_kernel<typename node_distrib_t<Process>::instantT>(tuning);
        recursive_node_mh_move(tree, tree.root(), process, kernel, update, logprob, gen);
    }

    // ****************************
    // single-branch bridge moves 

    template<class Tree, class Process, class Kernel, class BranchUpdate, class BranchLogProb, class Gen>
    static void single_branch_mh_move(Tree& tree, int node, Process& process, Kernel kernel, BranchUpdate update, BranchLogProb logprob, Gen& gen)  {

        auto branch = tree.get_branch(node);

        auto& path_vals = get<path_values>(process);
        auto& path = path_vals[branch];
        auto bk = path;

        auto timeframe = get<time_frame_field>(process);
        double t_young = timeframe(node);
        double t_old = timeframe(tree.parent(node));

        double logprobbefore = path_logprob(process,node) + logprob(tree.get_branch(node));

        double logh = kernel(path, t_young, t_old, gen);
        update(tree.get_branch(node));

        double logprobafter = path_logprob(process,node) + logprob(tree.get_branch(node));

        double delta = logprobafter - logprobbefore + logh;
        bool accept = decide(delta, gen);
        if (! accept)   {
            path = bk;
            update(tree.get_branch(node));
        }
    }

    template<class Tree, class Process, class Kernel, class BranchUpdate, class BranchLogProb, class Gen>
    static void recursive_branch_mh_move(Tree& tree, int node, Process& process, Kernel kernel, BranchUpdate update, BranchLogProb logprob, Gen& gen)    {
        single_branch_mh_move(tree, node, process, kernel, update, logprob, gen);
        for (auto c : tree.children(node)) {
            recursive_branch_mh_move(tree, c, process, kernel, update, logprob, gen);
        }
        single_branch_mh_move(tree, node, process, kernel, update, logprob, gen);
    }

    template<class Process, class Kernel, class BranchUpdate, class BranchLogProb, class Gen>
    static void branch_by_branch_mh_move(Process& process, Kernel kernel, BranchUpdate update, BranchLogProb logprob, Gen& gen)  {
        auto& tree = get<tree_field>(process);
        recursive_branch_mh_move(tree, tree.root(), process, kernel, update, logprob, gen);
    }

    template<class Process, class Path, class Params, class... Keys, class Gen>
    static double unpack_bridge_kernel(Process& process, double tuning, Path& path, double t_young, double t_old, const Params& params, std::tuple<Keys...>, Gen& gen)   {
        return node_distrib_t<Process>::bridge_kernel(tuning, path, t_young, t_old,
                get<Keys>(params)()..., gen);
    }

    template<class Process, class BranchUpdate, class BranchLogProb, class Gen>
    static void branch_by_branch_mh_move(Process& process, double tuning, BranchUpdate update, BranchLogProb logprob, Gen& gen)  {
        auto& tree = get<tree_field>(process);
        auto kernel = [&process, tuning] (auto& path, double t_young, double t_old, Gen& gen) {
            return unpack_bridge_kernel(process, tuning, path, t_young, t_old,
                    get<params>(process), param_keys_t<node_distrib_t<Process>>(), gen);
        };
        recursive_branch_mh_move(tree, tree.root(), process, kernel, update, logprob, gen);
    }

    // ****************************
    // add suffstat

    /*
    template<class ParamKey, class Tree, class Process, class SS, class Params, class... Keys>
    static void local_add_branch_suffstat(ParamKey key, Tree& tree, int node, Process& process, SS& ss, const Params& params, std::tuple<Keys...>)  {

        auto& v = get<value>(process);
        auto timeframe = get<time_frame_field>(process);

        node_distrib_t<Process>::add_branch_suffstat(
                key, ss, v[node], v[tree.parent(node)],
                timeframe(tree.parent(node)) - timeframe(node),
                get<Keys>(params)()...);
    }

    template<class ParamKey, class Tree, class Process, class SS>
    static void recursive_add_branch_suffstat(ParamKey key, Tree& tree, int node, Process& process, SS& ss) {
        if (! tree.is_root(node)) {
            local_add_branch_suffstat(key, tree, node, process, ss, 
                    get<params>(process), param_keys_t<node_distrib_t<Process>>());
        }
        for (auto c : tree.children(node)) {
            recursive_add_branch_suffstat(key, tree, c, process, ss);
        }
    }

    template<class ParamKey, class Process, class SS>
    static void add_branch_suffstat(Process& process, SS& ss)    {
        auto& tree = get<tree_field>(process);
        ParamKey key;
        recursive_add_branch_suffstat(key, tree, tree.root(), process, ss);
    }
    */

    // ****************************
    // backward routines

    template<class Tree, class Process, class CondL, class Params, class... Keys>
    static void backward_branch_propagate(const Tree& tree, int node, Process& process, CondL& old_condl, CondL& young_condl, const Params& params, std::tuple<Keys...>)   {

        auto timeframe = get<time_frame_field>(process);

        node_distrib_t<Process>::backward_propagate(
                young_condl, old_condl,
                timeframe(node), timeframe(tree.parent(node)),
                get<Keys>(params)()...);

    }

    template<class Tree, class Process, class CondLArray>
    static void recursive_backward(const Tree& tree, int node, Process& process, CondLArray& old_condls, CondLArray& young_condls, const std::vector<bool> external_clamps)    {

        auto& v = get<node_values>(process);
        auto& clamp = get<constraint>(process);
        node_distrib_t<Process>::CondL::init(young_condls[node], v[node],
                clamp[node], external_clamps[node]);

        for (auto c : tree.children(node))  {
            recursive_backward(tree, c, process, old_condls, young_condls, external_clamps);
            node_distrib_t<Process>::CondL::multiply(old_condls[c], young_condls[node]);
        }

        if (! tree.is_root(node))   {
            backward_branch_propagate(tree, node, process, old_condls[node], young_condls[node],
                        get<params>(process), param_keys_t<node_distrib_t<Process>>());
        }
    }

    // ****************************
    // backward routines with branch conditional likelihoods

    template<class Tree, class Process, class CondL, class Params, class... Keys>
    static void backward_branch_propagate(const Tree& tree, int node, Process& process, CondL& old_condl, CondL& young_condl, const CondL& branch_condl, const Params& params, std::tuple<Keys...>)   {

        auto timeframe = get<time_frame_field>(process);
        node_distrib_t<Process>::backward_propagate(
                young_condl, old_condl, branch_condl, 
                timeframe(node), timeframe(tree.parent(node)),
                get<Keys>(params)()...);

    }

    template<class Tree, class Process, class CondLArray>
    static void recursive_backward(const Tree& tree, int node, Process& process, CondLArray& old_condls, CondLArray& young_condls, const CondLArray& branch_condls, const std::vector<bool>& external_clamps)    {

        auto& v = get<node_values>(process);
        auto& clamp = get<constraint>(process);
        node_distrib_t<Process>::CondL::init(young_condls[node], v[node],
                clamp[node], external_clamps[node]);

        for (auto c : tree.children(node))  {
            recursive_backward(tree, c, process, old_condls, young_condls,
                    branch_condls, external_clamps);
            node_distrib_t<Process>::CondL::multiply(old_condls[c], young_condls[node]);
        }

        if (! tree.is_root(node))   {
            backward_branch_propagate(tree, node, process, 
                    old_condls[node], young_condls[node], branch_condls[tree.get_branch(node)],
                    get<params>(process), param_keys_t<node_distrib_t<Process>>());
        }
    }

    // ****************************
    // double backward routines

    // for a given node:
    // young_condls[node][j], j=0..n-1: young condL given that node is at time t_j
    // for leaves : only young_condls[node][0] is used (at fixed time)
    // backward propagate -> 
    // old_condls[node][i]: old condl, given that immediate ancestor is at time t_i
    // but if immediate ancestor is root..

    template<class Tree, class Process, class CondLs, class Params, class... Keys>
    static void double_backward_branch_propagate(const Tree& tree, int node, int parent_node, Process& process, const std::vector<double>& times, double root_time, double leaf_time, CondLs& old_condls, CondLs& young_condls, const Params& params, std::tuple<Keys...>)   {

        if (tree.is_leaf(node)) {
            for (size_t i=0; i<times.size(); i++)   {
                node_distrib_t<Process>::backward_propagate(
                        young_condls[0], old_condls[i],
                        times[i], leaf_time,
                        get<Keys>(params)()...);
            }
        }

        else if (tree.is_root(parent_node)) {
            for (size_t j=0; j<times.size(); j++)   {
                node_distrib_t<Process>::backward_propagate(
                        young_condls[j], old_condls[0],
                        root_time, times[j],
                        get<Keys>(params)()...);
            }
        }

        else    {

            auto tmp = std::vector<typename node_distrib_t<Process>::condL::L>(
                    times.size(),
                    node_distrib_t<Process>::CondL::make_censored_init());

            for (size_t i=0; i<times.size(); i++)   {
                for (size_t j=0; j<=i; j++) {
                    node_distrib_t<Process>::backward_propagate(
                            young_condls[j], tmp[j],
                            times[i], times[j],
                            get<Keys>(params)()...);
                }
                node_distrib_t<Process>::condL::mix(old_condls[i], tmp);
            }
        }
    }

    template<class Tree, class Process, class CondLsArray>
    static void recursive_double_backward(const Tree& tree, int node, Process& process, const std::vector<double>& times, double root_time, double leaf_time, CondLsArray& old_condls, CondLsArray& young_condls)   {

        auto& v = get<node_values>(process);
        auto& clamp = get<constraint>(process);

        node_distrib_t<Process>::CondL::init(young_condls[node]);

        if (tree.is_leaf(node)) {
            node_distrib_t<Process>::CondL::init(young_condls[node][0], v[node], clamp[node]);
        }

        for (auto c : tree.children(node))  {

            recursive_double_backward(tree, c, process, 
                    times, root_time, leaf_time, old_condls, young_condls);

            double_backward_branch_propagate(tree, process, c, node,
                    times, root_time, leaf_time, old_condls, young_condls, 
                    get<params>(process), param_keys_t<node_distrib_t<Process>>());

            if (tree.is_root(node)) {
                node_distrib_t<Process>::CondL::multiply(old_condls[c][0], young_condls[node][0]);
            }
            else    {
                node_distrib_t<Process>::CondL::multiply(old_condls[c], young_condls[node]);
            }
        }
    }

    // ****************************
    // conditional draw 

    template<class Tree, class Process, class CondLArray, class Params, class... Keys, class Gen>
    static void root_conditional_draw(const Tree& tree, Process& process, CondLArray& young_condls, const Params& params, std::tuple<Keys...>, Gen& gen)   {

        auto& clamp = get<constraint>(process);
        auto node_vals = get<node_values>(process);
        node_root_distrib_t<Process>::conditional_draw(
                node_vals[tree.root()], clamp[tree.root()],
                young_condls[tree.root()],
                get<Keys>(params)()..., gen);
    }

    template<class Tree, class Process, class CondLArray, class Params, class... Keys, class Gen>
    static void node_conditional_draw(const Tree& tree, int node, Process& process, CondLArray& young_condls, const Params& params, std::tuple<Keys...>, Gen& gen)   {

        auto& clamp = get<constraint>(process);
        auto timeframe = get<time_frame_field>(process);
        auto node_vals = get<node_values>(process);

        // first draw value at node, given parent node and conditional likelihood for downstream data
        node_distrib_t<Process>::node_conditional_draw(
                node_vals[node], clamp[node], node_vals[tree.parent(node)],
                timeframe(node), timeframe(tree.parent(node)),
                young_condls[node],
                get<Keys>(params)()..., gen);
    }

    template<class Tree, class Process, class Params, class... Keys, class Gen>
    static void bridge_conditional_draw(const Tree& tree, int node, Process& process, const Params& params, std::tuple<Keys...>, Gen& gen)   {

        auto timeframe = get<time_frame_field>(process);
        auto node_vals = get<node_values>(process);
        auto path_vals = get<path_values>(process);

        // then draw bridge along branch, conditional on values at both ends
        node_distrib_t<Process>::bridge_conditional_draw(
                path_vals[tree.get_branch(node)],
                node_vals[node], node_vals[tree.parent(node)],
                timeframe(node), timeframe(tree.parent(node)),
                get<Keys>(params)()..., gen);
    }

    template<class Tree, class Process, class CondLArray, class Params, class... Keys, class Gen>
    static void path_conditional_draw(const Tree& tree, int node, Process& process, CondLArray& young_condls, const Params& params, std::tuple<Keys...>, Gen& gen)   {

        auto& clamp = get<constraint>(process);
        auto timeframe = get<time_frame_field>(process);
        auto node_vals = get<node_values>(process);
        auto path_vals = get<path_values>(process);

        // first draw value at node, given parent node and conditional likelihood for downstream data
        node_distrib_t<Process>::node_conditional_draw(
                node_vals[node], clamp[node], node_vals[tree.parent(node)],
                timeframe(node), timeframe(tree.parent(node)),
                young_condls[node],
                get<Keys>(params)()..., gen);

        // then draw bridge along branch, conditional on values at both ends
        node_distrib_t<Process>::bridge_conditional_draw(
                path_vals[tree.get_branch(node)],
                node_vals[node], node_vals[tree.parent(node)],
                timeframe(node), timeframe(tree.parent(node)),
                get<Keys>(params)()..., gen);
    }

    /*
    template<class Tree, class Process, class CondLArray, class Params, class... Keys>
    static double root_conditional_logprob(const Tree& tree, Process& process, CondLArray& young_condls, const Params& params, std::tuple<Keys...>)   {

        auto& clamp = get<constraint>(process);
        auto node_vals = get<node_values>(process);
        return node_root_distrib_t<Process>::conditional_logprob(
                node_vals[tree.root()], clamp[tree.root()],
                young_condls[tree.root()],
                get<Keys>(params)()...);
    }

    template<class Tree, class Process, class CondLArray, class Params, class... Keys>
    static double path_conditional_logprob(const Tree& tree, int node, Process& process, CondLArray& young_condls, const Params& params, std::tuple<Keys...>)   {

        auto& clamp = get<constraint>(process);
        auto timeframe = get<time_frame_field>(process);
        auto node_vals = get<node_values>(process);
        auto path_vals = get<path_values>(process);

        // first value at node, given parent node and conditional likelihood for downstream data
        double ret = node_distrib_t<Process>::node_conditional_logprob(
                node_vals[node], clamp[node], node_vals[tree.parent(node)],
                timeframe(node), timeframe(tree.parent(node)),
                young_condls[node],
                get<Keys>(params)()...);

        // then bridge along branch, conditional on values at both ends
        ret += node_distrib_t<Process>::bridge_conditional_logprob(
                path_vals[tree.get_branch(node)],
                node_vals[node], node_vals[tree.parent(node)],
                timeframe(node), timeframe(tree.parent(node)),
                get<Keys>(params)()...);

        return ret;
    }
    */

    template<class Tree, class Process, class CondLArray, class Gen>
    static void recursive_forward(const Tree& tree, int node, Process& process, CondLArray& young_condls, Gen& gen)    {

        if (tree.is_root(node)) {
            root_conditional_draw(tree, process, young_condls,
                get<struct root_params>(process), param_keys_t<node_root_distrib_t<Process>>(), gen);
        }
        else    {
            path_conditional_draw(tree, node, process, young_condls,
                get<struct params>(process), param_keys_t<node_distrib_t<Process>>(), gen);
        }

        for (auto c : tree.children(node))  {
            recursive_forward(tree, c, process, young_condls, gen);
        }
    }

    // direct backward-forward sampling
    template<class Process, class Gen>
    static void conditional_draw(Process& process, Gen& gen)  {

        using Distrib = node_distrib_t<Process>;
        using CondL = typename Distrib::CondL;
        typename CondL::L initcondl = CondL::make_init();

        std::vector<typename CondL::L> old_condls(get<node_values>(process).size(), initcondl);
        std::vector<typename CondL::L> young_condls(get<node_values>(process).size(), initcondl);

        std::vector<bool> external_clamps(get<node_values>(process).size(), false);

        auto& tree = get<tree_field>(process);

        recursive_backward(tree, tree.root(), process, old_condls, young_condls, external_clamps);
        recursive_forward(tree, tree.root(), process, young_condls, gen);
    }

    // ****************************
    // proposals, importance sampling

    // a class that implements single node/branch conditional sampling on demand
    // (longitudinal conditions, i.e. process is observed at some nodes) 
    // stores and computes conditional likelihoods upon creation
    // also returns associated log densities - can thus be used as a proposal for importance sampling

    template<class Process>
    class conditional_sampler    {

        Process& process;
        std::vector<typename node_distrib_t<Process>::CondL::L> young_condls;
        std::vector<typename node_distrib_t<Process>::CondL::L> old_condls;

        public:

        conditional_sampler(Process& in_process) :
            process(in_process), 
            young_condls(get<node_values>(process).size(), 
                    node_distrib_t<Process>::CondL::make_init()),
            old_condls(get<node_values>(process).size(), 
                    node_distrib_t<Process>::CondL::make_init())    {
        }

        void init(const std::vector<bool>& external_clamps)  {

            for (auto& condl : young_condls)    {
                node_distrib_t<Process>::CondL::init(condl);
            }
            for (auto& condl : old_condls)    {
                node_distrib_t<Process>::CondL::init(condl);
            }

            auto& tree = get<tree_field>(process);

            recursive_backward(tree, tree.root(),
                process, old_condls, young_condls, external_clamps);
        }

        template<class Gen>
        void root_draw(typename node_distrib_t<Process>::instantT& val,
                double& log_weight,
                Gen& gen)  {

            auto& tree = get<tree_field>(process);
            auto& node_vals = get<node_values>(process);

            root_conditional_draw(tree, process, young_condls,
                get<struct root_params>(process), param_keys_t<node_root_distrib_t<Process>>(), gen);

            val = node_vals[tree.root()];
        }

        template<class Gen>
        void node_draw(int node, typename node_distrib_t<Process>::instantT& val, 
                const typename node_distrib_t<Process>::instantT& parent_val, 
                double& log_weight, Gen& gen)    {

            auto& tree = get<tree_field>(process);
            auto& node_vals = get<node_values>(process);

            node_vals[tree.parent(node)] = parent_val;

            node_conditional_draw(tree, node, process, young_condls,
                get<params>(process), param_keys_t<node_distrib_t<Process>>(), gen);

            val = node_vals[node];
        }

        template<class Gen>
        void bridge_draw(int node, 
                const typename node_distrib_t<Process>::instantT& val, 
                const typename node_distrib_t<Process>::instantT& parent_val, 
                typename node_distrib_t<Process>::pathT& path,
                double& log_weight, Gen& gen)    {

            auto& tree = get<tree_field>(process);
            auto& path_vals = get<path_values>(process);

            bridge_conditional_draw(tree, node, process, 
                get<params>(process), param_keys_t<node_distrib_t<Process>>(), gen);

            path = path_vals[tree.get_branch(node)];
        }

        /*
        template<class Gen>
        void path_draw(int node, typename node_distrib_t<Process>::instantT& val, 
                const typename node_distrib_t<Process>::instantT& parent_val, 
                typename node_distrib_t<Process>::pathT& path,
                double& log_weight, Gen& gen)    {

            auto& tree = get<tree_field>(process);
            auto& node_vals = get<node_values>(process);
            auto& path_vals = get<path_values>(process);

            node_vals[tree.parent(node)] = parent_val;

            path_conditional_draw(tree, node, process, young_condls,
                get<params>(process), param_keys_t<node_distrib_t<Process>>(), gen);

            val = node_vals[node];
            path = path_vals[tree.get_branch(node)];
        }
        */

        /*
        double root_logprob(const typename node_distrib_t<Process>::instantT& val) {

            auto& tree = get<tree_field>(process);
            auto& node_vals = get<node_values>(process);

            node_vals[tree.root()] = val;

            return root_conditional_logprob(tree, process, young_condls,
                get<struct root_params>(process), param_keys_t<node_root_distrib_t<Process>>());
        }

        double path_logprob(int node, 
                const typename node_distrib_t<Process>::instantT& val,
                const typename node_distrib_t<Process>::instantT& parent_val,
                const typename node_distrib_t<Process>::pathT& path)   {

            auto& tree = get<tree_field>(process);
            auto& node_vals = get<node_values>(process);
            auto& path_vals = get<path_values>(process);

            node_vals[node] = val;
            node_vals[tree.parent(node)] = parent_val;
            path_vals[tree.get_branch(node)] = path;

            return path_conditional_logprob(tree, node, process, young_condls,
                get<params>(process), param_keys_t<node_distrib_t<Process>>());
        }
        */
    };

    template<class Process>
    static auto make_conditional_sampler(Process& process) {
        return conditional_sampler<Process>(process);
    }

    // a conditional sampler with externally given branch conditional likelihoods

    template<class Process>
    class pseudo_branch_conditional_sampler    {

        Process& process;
        std::vector<typename node_distrib_t<Process>::CondL::L> young_condls;
        std::vector<typename node_distrib_t<Process>::CondL::L> old_condls;
        const std::vector<typename node_distrib_t<Process>::CondL::L>& branch_condls;

        public:

        pseudo_branch_conditional_sampler(Process& in_process,
                const std::vector<typename node_distrib_t<Process>::CondL::L>& in_branch_condls) :
            process(in_process), 
            young_condls(get<node_values>(process).size(), 
                    node_distrib_t<Process>::CondL::make_init()),
            old_condls(get<node_values>(process).size(), 
                    node_distrib_t<Process>::CondL::make_init()),
            branch_condls(in_branch_condls) {
        }

        void init(const std::vector<bool>& external_clamps)  {

            for (auto& condl : young_condls)    {
                node_distrib_t<Process>::CondL::init(condl);
            }
            for (auto& condl : old_condls)    {
                node_distrib_t<Process>::CondL::init(condl);
            }

            auto& tree = get<tree_field>(process);

            recursive_backward(tree, tree.root(),
                process, old_condls, young_condls, branch_condls, external_clamps);
        }

        template<class Gen>
        void root_draw(typename node_distrib_t<Process>::instantT& val,
                double& log_weight, Gen& gen)  {

            auto& tree = get<tree_field>(process);
            auto& node_vals = get<node_values>(process);

            root_conditional_draw(tree, process, young_condls,
                get<struct root_params>(process), param_keys_t<node_root_distrib_t<Process>>(), gen);

            val = node_vals[tree.root()];
        }

        template<class Gen>
        void path_draw(int node, typename node_distrib_t<Process>::instantT& val, 
                const typename node_distrib_t<Process>::instantT& parent_val, 
                typename node_distrib_t<Process>::pathT& path,
                double& log_weight, Gen& gen)    {

            auto& tree = get<tree_field>(process);
            auto& node_vals = get<node_values>(process);
            auto& path_vals = get<path_values>(process);

            node_vals[tree.parent(node)] = parent_val;

            path_conditional_draw(tree, node, process, young_condls,
                get<params>(process), param_keys_t<node_distrib_t<Process>>(), gen);

            val = node_vals[node];
            path = path_vals[tree.get_branch(node)];
        }

        double pseudo_branch_logprob(int branch, 
                typename node_distrib_t<Process>::instantT& val, 
                const typename node_distrib_t<Process>::instantT& parent_val)   {

            return node_distrib_t<Process>::pseudo_branch_logprob(val, parent_val, 
                    branch_condls[branch]);
        }

        /*
        double root_logprob(const typename node_distrib_t<Process>::instantT& val) {

            auto& tree = get<tree_field>(process);
            auto& node_vals = get<node_values>(process);

            node_vals[tree.root()] = val;

            return root_conditional_logprob(tree, process, young_condls,
                get<struct root_params>(process), param_keys_t<node_root_distrib_t<Process>>());
        }

        double path_logprob(int node, 
                const typename node_distrib_t<Process>::instantT& val,
                const typename node_distrib_t<Process>::instantT& parent_val,
                const typename node_distrib_t<Process>::pathT& path)   {

            auto& tree = get<tree_field>(process);
            auto& node_vals = get<node_values>(process);
            auto& path_vals = get<path_values>(process);

            node_vals[node] = val;
            node_vals[tree.parent(node)] = parent_val;
            path_vals[tree.get_branch(node)] = path;

            return path_conditional_logprob(tree, node, process, young_condls,
                get<params>(process), param_keys_t<node_distrib_t<Process>>());
        }
        */
    };

    template<class Process>
    static auto make_pseudo_branch_conditional_sampler(Process& process,
            const std::vector<typename node_distrib_t<Process>::CondL::L>& branch_condls) {
        return pseudo_branch_conditional_sampler<Process>(process, branch_condls);
    }

    template<class Process, class BranchUpdate, class BranchLogProb>
    class prior_importance_sampler  {

        Process& process;
        conditional_sampler<Process> proposal;
        BranchUpdate update;
        BranchLogProb logprob;

        public:

        prior_importance_sampler(Process& in_process, 
            BranchUpdate in_update, BranchLogProb in_logprob) :
                process(in_process), proposal(process),
                update(in_update), logprob(in_logprob) {}

        ~prior_importance_sampler() {}

        void init(const std::vector<bool>& external_clamps)    {
            proposal.init(external_clamps);
        }

        template<class Gen>
        void root_draw(typename node_distrib_t<Process>::instantT& val, 
                double& log_weight, bool fixed_node, Gen& gen) {

            if (! fixed_node)    {
                proposal.root_draw(val, log_weight, gen);
            }
        }

        template<class Gen>
        void path_draw(int node, 
                typename node_distrib_t<Process>::instantT& val, 
                const typename node_distrib_t<Process>::instantT& parent_val, 
                typename node_distrib_t<Process>::pathT& path, 
                double& log_weight, bool fixed_node, bool fixed_branch, Gen& gen) {

            if (! fixed_node)    {
                proposal.node_draw(node, val, parent_val, log_weight, gen);
            }
            if (! fixed_branch) {
                proposal.bridge_draw(node, val, parent_val, path, log_weight, gen);
            }
            update(get<tree_field>(process).get_branch(node));
            log_weight += logprob(get<tree_field>(process).get_branch(node));
        }
    };

    template<class Process, class Update, class LogProb>
    static auto make_prior_importance_sampler(Process& process, Update update, LogProb logprob)   {
        return prior_importance_sampler<Process, Update, LogProb>(process, update, logprob);
    }

    template<class Process, class BranchUpdate, class BranchLogProb>
    class pseudo_branch_prior_importance_sampler  {

        Process& process;
        pseudo_branch_conditional_sampler<Process> proposal;
        BranchUpdate update;
        BranchLogProb logprob;

        public:

        pseudo_branch_prior_importance_sampler(Process& in_process, 
            BranchUpdate in_update, BranchLogProb in_logprob) :
                process(in_process), proposal(process),
                update(in_update), logprob(in_logprob) {}

        ~pseudo_branch_prior_importance_sampler() {}

        void init(const std::vector<bool>& external_clamps)    {
            proposal.init(external_clamps);
        }

        template<class Gen>
        void root_draw(typename node_distrib_t<Process>::instantT& val, 
                double& log_weight, bool fixed_node, Gen& gen) {

            if (! fixed_node)    {
                proposal.root_draw(val, log_weight, gen);
            }
        }

        template<class Gen>
        void path_draw(int node, 
                typename node_distrib_t<Process>::instantT& val, 
                const typename node_distrib_t<Process>::instantT& parent_val, 
                typename node_distrib_t<Process>::pathT& path, 
                double& log_weight, bool fixed_node, bool fixed_branch, Gen& gen) {

            auto& tree = get<tree_field>(process);
            int branch = tree.get_branch(node);
            if (! fixed_node)    {
                proposal.node_draw(node, val, parent_val, log_weight, gen);
            }
            if (! fixed_branch) {
                proposal.bridge_draw(node, val, parent_val, path, log_weight, gen);
            }
            update(branch);
            log_weight -= proposal.pseudo_branch_logprob(branch, val, parent_val);
            log_weight += logprob(branch);
        }
    };

    template<class Process, class Update, class LogProb>
    static auto make_pseudo_branch_prior_importance_sampler(Process& process, Update update, LogProb logprob)   {
        return pseudo_branch_prior_importance_sampler<Process, Update, LogProb>(
                process, update, logprob);
    }

    /*
    template<class Process, class Proposal, class BranchUpdate, class BranchLogProb>
    class importance_sampler  {

        Process& process;
        Proposal& proposal;
        BranchUpdate update;
        BranchLogProb logprob;

        public: 

        importance_sampler(Process& in_process, Proposal& in_proposal,
            BranchUpdate in_update, BranchLogProb in_logprob) :
                process(in_process), proposal(in_proposal),
                update(in_update), logprob(in_logprob) {}

        ~importance_sampler() {}

        template<class Gen>
        void root_draw(typename node_distrib_t<Process>::instantT& val, 
                double& log_weight, Gen& gen) {

            proposal.root_draw(val, log_weight, gen);

            log_weight -= proposal.root_logprob(val);
            log_weight += root_logprob(process,
                get<struct root_params>(process), param_keys_t<node_root_distrib_t<Process>>());
        }

        template<class Gen>
        void path_draw(int node, 
                typename node_distrib_t<Process>::instantT& val, 
                const typename node_distrib_t<Process>::instantT& parent_val, 
                typename node_distrib_t<Process>::pathT& path, 
                double& log_weight, Gen& gen) {

            proposal.path_draw(node, val, parent_val, path, log_weight, gen);

            log_weight -= proposal.path_logprob(node, val, parent_val, path);
            log_weight += path_logprob(process, node,
                get<params>(process), param_keys_t<node_distrib_t<Process>>());

            update(get<tree_field>(process).get_branch(node));
            log_weight += logprob(get<tree_field>(process).get_branch(node));
        }
    };

    template<class Process, class Proposal, class Update, class LogProb>
    static auto make_importance_sampler(Process& process, Proposal& proposal, Update update, LogProb logprob)   {
        return importance_sampler<Process, Proposal, Update, LogProb>(process, proposal, update, logprob);
    }
    */

    // ****************************
    // particle filters

    template<class Process, class WDist>
    class tree_pf   {
        
        Process& process;
        const Tree& tree;
        WDist& wdist;
        std::vector<std::vector<typename node_distrib_t<Process>::instantT>> node_swarm;
        std::vector<std::vector<typename node_distrib_t<Process>::pathT>> path_swarm;
        std::vector<double> log_weights;
        std::vector<int> node_ordering;
        std::vector<std::vector<int>> ancestors;
        size_t counter;
        std::vector<bool> external_clamps;

        public:

        tree_pf(Process& in_process, WDist& in_wdist, size_t n) :
            process(in_process),
            tree(get<tree_field>(process)),
            wdist(in_wdist),
            node_swarm(tree.nb_nodes(), 
                    std::vector<typename node_distrib_t<Process>::instantT>(n, 
                        get<node_values>(process)[0])),
            path_swarm(tree.nb_branches(), 
                    std::vector<typename node_distrib_t<Process>::pathT>(n, 
                        get<path_values>(process)[0])),
            log_weights(n, 0),
            node_ordering(tree.nb_nodes(), 0),
            ancestors(tree.nb_nodes(), std::vector<int>(n,0)),
            counter(0),
            external_clamps(tree.nb_nodes(), false) {}

        ~tree_pf() {}

        size_t size() {return log_weights.size();}

        template<class Gen>
        void init(bool conditional, double cond_frac, Gen& gen) {

            if (conditional)    {

                auto& node_vals = get<node_values>(process);
                for (size_t node=0; node<tree.nb_nodes(); node++)    {
                    node_swarm[node][0] = node_vals[node];
                }

                auto& path_vals = get<path_values>(process);
                for (size_t branch=0; branch<tree.nb_branches(); branch++)    {
                    path_swarm[branch][0] = path_vals[branch];
                }

                std::vector<double> w = {1-cond_frac, cond_frac};
                std::discrete_distribution<int> distrib(w.begin(), w.end());
                for (size_t node=0; node<tree.nb_nodes(); node++)   {
                    external_clamps[node] = distrib(gen);
                    if (external_clamps[node])  {
                        for (size_t i=1; i<size(); i++) {
                            node_swarm[node][i] = node_vals[node];
                        }
                    }
                }
            }

            wdist.init(external_clamps);

            counter = 0;
            for (auto& l : log_weights) {
                l = 0;
            }
        }

        template<class Gen>
        double run(bool conditional, double cond_frac, Gen& gen)  {

            init(conditional, cond_frac, gen);

            std::vector<int> b(size(), 0);
            for (size_t i=0; i<size(); i++)  {
                b[i] = i;
            }

            forward_pf(tree.root(), conditional, b, gen);

            // choose random particle
            int c = choose_particle(gen);

            // pull out
            for (size_t i=tree.nb_nodes()-1; i>0; i--) {
                int node = node_ordering[i];
                get<node_values>(process)[node] = node_swarm[node][c];
                get<path_values>(process)[tree.get_branch(node)] = 
                    path_swarm[tree.get_branch(node)][c];
                c = ancestors[node][c];
            }
            get<node_values>(process)[tree.root()] = node_swarm[tree.root()][c];
            return log_weights[c];
        }

        // returns anc[i]: index of the ancestor, at this node, of the particle i
        template<class Gen>
        void forward_pf(int node, bool conditional, std::vector<int>& b, Gen& gen)   {

            // when entering forward:
            // b[i]: ancestor at tree.parent(node) of current particle i
            // (current particle may currently extend downstream from current node
            // according to the order over nodes induced by the depth-first recursion)
            // if node is the first child of its parent, 
            // then current particle extends up to tree.parent(node), so b[i] = i

            auto branch = tree.get_branch(node);
            node_ordering[counter] = node;
            counter++;

            if (tree.is_root(node))   {
                // silly (not used), but just for overall consistency
                for (size_t i=0; i<size(); i++)  {
                    ancestors[node][i] = b[i];
                }
            }
            else    {
                // possibly, do bootstrap only if effective sample size is small
                std::vector<int> choose(size(), 0);
                bootstrap_pf(choose, gen);

                // keeping track of immediate ancestors
                for (size_t i=0; i<size(); i++)  {
                    ancestors[node][i] = b[choose[i]];
                }

                // updating b
                for (size_t i=0; i<size(); i++)  {
                    b[i] = ancestors[node][i];
                }
            }

            if (tree.is_root(node)) {
                for (size_t i=0; i<size(); i++)  {
                    wdist.root_draw(
                            node_swarm[node][i],
                            log_weights[i],
                            external_clamps[node] || (conditional && (!i)),  // fixed node
                            gen);
                }
            }
            else    {
                for (size_t i=0; i<size(); i++)  {
                    wdist.path_draw(node, 
                            node_swarm[node][i],
                            node_swarm[tree.parent(node)][ancestors[node][i]], 
                            path_swarm[branch][i],
                            log_weights[i],
                            external_clamps[node] || (conditional && (!i)), // fixed node
                            (conditional && (!i)),                          // fixed branch
                            gen);
                }
            }

            // send recursion
            // entering forward(c, bb) for the first child node with bb[i] = i
            // here bb[i] is the index of particle i at *this* node (i.e. the parent of c)
            // upon returning, forward will have updated bb
            // for the call to the next child node
            std::vector<int> bb(size(), 0);
            for (size_t i=0; i<size(); i++)  {
                bb[i] = i;
            }

            for (auto c : tree.children(node))  {
                forward_pf(c, conditional, bb, gen);
            }

            // when leaving forward, b should should be updated
            // to the index of particle i at tree.parent(node), based on the index at this node
            for (size_t i=0; i<size(); i++)  {
                b[i] = ancestors[node][bb[i]];
            }
        }

        template<class Gen>
        size_t choose_particle(Gen& gen)    {
            std::vector<double> w(size(), 0);
            double max = 0;
            for (size_t i=0; i<size(); i++) {
                if ((!i) || (max < log_weights[i])) {
                    max = log_weights[i];
                }
            }
            double tot = 0;
            for (size_t i=0; i<size(); i++) {
                w[i] = exp(log_weights[i] - max);
                tot += w[i];
            }
            for (size_t i=0; i<size(); i++) {
                w[i] /= tot;
            }
            std::discrete_distribution<int> distrib(w.begin(), w.end());
            return distrib(gen);
        }

        template<class Gen>
        void bootstrap_pf(std::vector<int>& choose, Gen& gen)  {
            std::vector<double> w(size(), 0);
            double max = 0;
            for (size_t i=0; i<size(); i++) {
                if ((!i) || (max < log_weights[i])) {
                    max = log_weights[i];
                }
            }
            double tot = 0;
            for (size_t i=0; i<size(); i++) {
                w[i] = exp(log_weights[i] - max);
                tot += w[i];
            }
            for (size_t i=0; i<size(); i++) {
                w[i] /= tot;
            }
            std::discrete_distribution<int> distrib(w.begin(), w.end());
            for (size_t i=0; i<size(); i++) {
                choose[i] = distrib(gen);
            }

            tot /= size();
            double log_mean_weight = log(tot) + max;
            for (size_t i=0; i<size(); i++) {
                log_weights[i] = log_mean_weight;
            }
        }
    };

    template<class Process, class WDist>
    static auto make_particle_filter(Process& process, WDist& wdist, size_t n)    {
        return tree_pf<Process, WDist>(process, wdist, n);
    }
};

