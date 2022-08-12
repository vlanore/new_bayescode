
#include "tree/tree_factory.hpp"
#include "montecarlo/split_tree.hpp"

struct tree_process_methods  {

    // ****************************
    // forward draw (assumes process is unclamped)

    template<class Process, class Params, class... Keys, class Gen>
    static void root_draw(Process& process, const Params& params, std::tuple<Keys...>, Gen& gen)   {

        assert(! node_root_distrib_t<Process>::active_constraint(
                    get<constraint>(process)[get<tree_field>(process).root()]));

        auto& v = get<node_values>(process)[get<tree_field>(process).root()];
        node_root_distrib_t<Process>::draw(v, get<Keys>(params)()..., gen);
    }

    template<class Process, class Params, class... Keys, class Gen>
    static void path_draw(Process& process, int node, const Params& params, std::tuple<Keys...>, Gen& gen)   {

        assert(! node_distrib_t<Process>::active_constraint(get<constraint>(process)[node]));

        auto& tree = get<tree_field>(process);
        auto timeframe = get<time_frame_field>(process);
        auto& node_vals = get<node_values>(process);
        auto& path_vals = get<path_values>(process);

        node_distrib_t<Process>::path_draw(path_vals[tree.get_branch(node)], 
            node_vals[node], node_vals[tree.parent(node)],
            timeframe(node), timeframe(tree.parent(node)),
            get<Keys>(params)()..., gen);
    }

    template<class Process, class Params, class... Keys, class Gen>
    static void node_draw(Process& process, int node, const Params& params, std::tuple<Keys...>, Gen& gen)   {

        assert(! node_distrib_t<Process>::active_constraint(get<constraint>(process)[node]));

        auto& tree = get<tree_field>(process);
        auto timeframe = get<time_frame_field>(process);
        auto& node_vals = get<node_values>(process);

        node_distrib_t<Process>::node_draw(node_vals[node], node_vals[tree.parent(node)],
            timeframe(node), timeframe(tree.parent(node)),
            get<Keys>(params)()..., gen);
    }

    // draw bridge along branch, conditional on values at both ends
    template<class Process, class Params, class... Keys, class Gen>
    static void bridge_draw(Process& process, int node, const Params& params, std::tuple<Keys...>, Gen& gen)   {

        auto& tree = get<tree_field>(process);
        auto timeframe = get<time_frame_field>(process);
        auto& node_vals = get<node_values>(process);
        auto& path_vals = get<path_values>(process);

        node_distrib_t<Process>::bridge_draw(
                path_vals[tree.get_branch(node)],
                node_vals[node], node_vals[tree.parent(node)],
                timeframe(node), timeframe(tree.parent(node)),
                get<Keys>(params)()..., gen);
    }

    // ****************************
    // conditional draw 

    // draw value at root node, given its prior and conditional likelihood for downstream data
    template<class Process, class CondL, class Params, class... Keys, class Gen>
    static void root_conditional_draw(Process& process, CondL& condl, const Params& params, std::tuple<Keys...>, Gen& gen)   {

        auto& tree = get<tree_field>(process);
        auto& clamps = get<constraint>(process);
        auto& node_vals = get<node_values>(process);
        node_root_distrib_t<Process>::conditional_draw(
                node_vals[tree.root()], clamps[tree.root()], condl,
                get<Keys>(params)()..., gen);
    }

    // draw value at node, given parent node and conditional likelihood for downstream data
    template<class Process, class CondL, class Params, class... Keys, class Gen>
    static void node_conditional_draw(Process& process, int node, CondL& condl, const Params& params, std::tuple<Keys...>, Gen& gen)   {

        auto& tree = get<tree_field>(process);
        auto& clamps = get<constraint>(process);
        auto timeframe = get<time_frame_field>(process);
        auto& node_vals = get<node_values>(process);

        node_distrib_t<Process>::node_conditional_draw(
                node_vals[node], clamps[node], node_vals[tree.parent(node)],
                timeframe(node), timeframe(tree.parent(node)), condl,
                get<Keys>(params)()..., gen);
    }

    // combines node and bridge conditional draws
    template<class Process, class CondL, class Params, class... Keys, class Gen>
    static void path_conditional_draw(Process& process, int node, CondL& condl, const Params& params, std::tuple<Keys...>, Gen& gen)   {

        auto& tree = get<tree_field>(process);
        auto& clamps = get<constraint>(process);
        auto timeframe = get<time_frame_field>(process);
        auto& node_vals = get<node_values>(process);
        auto& path_vals = get<path_values>(process);

        // first draw value at node, given parent node and conditional likelihood for downstream data
        node_distrib_t<Process>::node_conditional_draw(
                node_vals[node], clamps[node], node_vals[tree.parent(node)],
                timeframe(node), timeframe(tree.parent(node)), condl,
                get<Keys>(params)()..., gen);

        // then draw bridge along branch, conditional on values at both ends
        node_distrib_t<Process>::bridge_draw(
                path_vals[tree.get_branch(node)],
                node_vals[node], node_vals[tree.parent(node)],
                timeframe(node), timeframe(tree.parent(node)),
                get<Keys>(params)()..., gen);
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
        auto& node_vals = get<node_values>(process);
        auto& path_vals = get<path_values>(process);

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
        if (std::isnan(tot))    {
            std::cerr << "in tree process : around_node_logprob: nan\n";
            exit(1);
        }
        if (std::isinf(tot))    {
            std::cerr << "in tree process : around_node_logprob: inf\n";
            exit(1);
        }
        return tot;
    }

    template<class Process>
    static auto around_node_logprob(Process& process)   {
        return [&p = process] (int node) {return around_node_logprob(p, node);};
    }

    // TODO: check moves and clamps
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
            logh += paths[tree.get_branch(node)].adapt_to_younger_end(bk, x);
        }
        for (auto c : tree.children(node))  {
            logh += paths[tree.get_branch(c)].adapt_to_older_end(bk, x);
        }
        tree_factory::do_around_node(tree, update)(node);
        double logprobafter = around_node_logprob(process,node) + tree_factory::sum_around_node(tree, logprob)(node);
        double delta = logprobafter - logprobbefore + logh;
        bool accept = decide(delta, gen);
        if (! accept)   {
            if (! tree.is_root(node))   {
                paths[tree.get_branch(node)].adapt_to_younger_end(x, bk);
            }
            for (auto c : tree.children(node))  {
                paths[tree.get_branch(c)].adapt_to_older_end(x, bk);
            }
            x = bk;
            tree_factory::do_around_node(tree, update)(node);
        }
        else    {
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

        auto& path = get<path_values>(process)[branch];
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
        if (! tree.is_root(node))   {
            single_branch_mh_move(tree, node, process, kernel, update, logprob, gen);
        }
        for (auto c : tree.children(node)) {
            recursive_branch_mh_move(tree, c, process, kernel, update, logprob, gen);
        }
        if (! tree.is_root(node))   {
            single_branch_mh_move(tree, node, process, kernel, update, logprob, gen);
        }
    }

    template<class Process, class Kernel, class BranchUpdate, class BranchLogProb, class Gen>
    static void branch_by_branch_mh_move(Process& process, Kernel kernel, BranchUpdate update, BranchLogProb logprob, Gen& gen)  {
        auto& tree = get<tree_field>(process);
        recursive_branch_mh_move(tree, tree.root(), process, kernel, update, logprob, gen);
    }

    template<class Process, class Path, class Params, class... Keys, class Gen>
    static double bridge_kernel(Process& process, double tuning, Path& path, double t_young, double t_old, const Params& params, std::tuple<Keys...>, Gen& gen)   {
        return node_distrib_t<Process>::bridge_kernel(tuning, path, t_young, t_old,
                get<Keys>(params)()..., gen);
    }

    template<class Process, class BranchUpdate, class BranchLogProb, class Gen>
    static void branch_by_branch_mh_move(Process& process, double tuning, BranchUpdate update, BranchLogProb logprob, Gen& gen)  {
        auto& tree = get<tree_field>(process);
        auto kernel = [&process, tuning] (auto& path, double t_young, double t_old, Gen& gen) {
            return bridge_kernel(process, tuning, path, t_young, t_old,
                    get<params>(process), param_keys_t<node_distrib_t<Process>>(), gen);
        };
        recursive_branch_mh_move(tree, tree.root(), process, kernel, update, logprob, gen);
    }

    // ****************************
    // add suffstat

    template<class ParamKey, class Tree, class Process, class SS, class Params, class... Keys>
    static void local_add_branch_suffstat(ParamKey key, Tree& tree, int node, Process& process, SS& ss, const Params& params, std::tuple<Keys...>)  {

        auto& node_vals = get<node_values>(process);
        auto& path_vals = get<path_values>(process);
        auto timeframe = get<time_frame_field>(process);

        node_distrib_t<Process>::add_branch_suffstat(
                key, ss, path_vals[tree.get_branch(node)], 
                node_vals[node], node_vals[tree.parent(node)],
                timeframe(node), timeframe(tree.parent(node)),
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

    // ****************************
    // free forward routines

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
    // backward routines

    template<class Tree, class Process, class CondL, class Params, class... Keys>
    static void backward_branch_propagate(const Tree& tree, int node, Process& process, const CondL& young_condl, CondL& old_condl, const Params& params, std::tuple<Keys...>)   {

        auto timeframe = get<time_frame_field>(process);

        node_distrib_t<Process>::backward_propagate(
                young_condl, old_condl,
                timeframe(node), timeframe(tree.parent(node)),
                get<Keys>(params)()...);
    }

    template<class Tree, class Process, class CondLArray>
    static void recursive_backward(const Tree& tree, int node, Process& process, CondLArray& young_condls, CondLArray& old_condls, const std::vector<bool> external_clamps)    {

        auto& v = get<node_values>(process);
        auto& clamps = get<constraint>(process);
        node_distrib_t<Process>::CondL::init(young_condls[node], v[node],
                clamps[node], external_clamps[node]);
        if (normal_mean_condL::ill_defined(young_condls[node]))   {
            std::cerr << "problem in init\n";
            exit(1);
        }

        for (auto c : tree.children(node))  {
            recursive_backward(tree, c, process, young_condls, old_condls, external_clamps);
            if (normal_mean_condL::ill_defined(old_condls[c]))   {
                std::cerr << "problem in oldcondl of child\n";
                exit(1);
            }
            node_distrib_t<Process>::CondL::multiply(old_condls[c], young_condls[node]);
            if (normal_mean_condL::ill_defined(young_condls[node]))   {
                std::cerr << "problem in result of multiply\n";
                exit(1);
            }
        }

        if (! tree.is_root(node))   {
            backward_branch_propagate(tree, node, process, young_condls[node], old_condls[node],
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

    static size_t grid_index(double t, size_t n)    {
        return int((t-1e-6) * n);
    }

    static double grid_time(size_t i, size_t n) {
        return (i+0.5) / n;
    }

    static void get_time_intervals(double tmin, double tmax, size_t n, std::vector<double>& w)  {
        double delta = 1.0/n;
        size_t imin = grid_index(tmin, n);
        size_t imax = grid_index(tmax, n);
        
        for (size_t i=0; i<n; i++)  {
            w[i] = 0;
        }
        for (size_t i=imin; i<=imax; i++)   {
            double u = grid_time(i,n);
            double umin = u-0.5*delta;
            double umax = u+0.5*delta;
            if (umax > tmax)    { // last interval, incomplete from above
                umax = tmax;
            }
            if (umin < tmin)    { // first interval, incomplete from below
                umin = tmin;
            }
            w[i] = umax - umin;
        }
    }

    template<class Process, class CondLs, class T, class Constraint>
    static void init_double_condls(Process& process,
            size_t n, double tmin, double tmax,
            CondLs& condls, T& x,
            const Constraint& clamp, bool external_clamp) {

        size_t imin = grid_index(tmin, n);
        size_t imax = grid_index(tmax, n);
        for (size_t i=imin; i<=imax; i++)   {
            node_distrib_t<Process>::CondL::init(condls[i], x, clamp, external_clamp);
        }
    }

    template<class Process, class CondLs>
    static void multiply_double_condls(Process& process,
            size_t n, double tmin, double tmax,
            const CondLs& from_condls, CondLs& to_condls)   {

        size_t imin = grid_index(tmin, n);
        size_t imax = grid_index(tmax, n);
        for (size_t i=imin; i<=imax; i++)   {
            node_distrib_t<Process>::CondL::multiply(from_condls.at(i), to_condls[i]);
        }
    }

    template<class Process, class CondLs, class T, class Gen>
    static double draw_conditional_time(Process& process, int node, double tmin, double tmax, size_t n,
            const std::vector<CondLs>& condls, const T& val,
            double& logZ, size_t& index, Gen& gen)  {

        // for checks only
        const auto& tree = get<tree_field>(process);
        auto timeframe = get<time_frame_field>(process);

        std::vector<double> w(n, 0);
        std::vector<double> min_times(n, 0);
        std::vector<double> max_times(n, 0);

        size_t imin = grid_index(tmin, n);
        size_t imax = grid_index(tmax, n);

        // compute time intervals and bounds
        double tot = 0;
        for (size_t i=imin; i<=imax; i++)   {
            double u = grid_time(i,n);
            double umin = u-0.5/n;
            double umax = u+0.5/n;
            if (umax > tmax)    { // last interval, incomplete from above
                umax = tmax;
            }
            if (umin < tmin)    { // first interval, incomplete from below
                umin = tmin;
            }
            min_times[i] = umin;
            max_times[i] = umax;
            w[i] = n*(umax - umin);
            tot += w[i];
        }
        if (!tot)   {
            std::cerr << "empty interval to draw from\n";
            if (tree.is_leaf(node)) {
                std::cerr << "leaf node\n";
            }
            exit(1);
        }

        // draw time interval
        index = node_distrib_t<Process>::CondL::draw_component(condls, w,
            imin, imax, val, logZ, gen);

        // draw time within time interval
        double time = min_times[index] + draw_uniform(gen) * (max_times[index] - min_times[index]);
        if (time > 1.0) {
            std::cerr << "time > 1\n";
            std::cerr << time << '\t' << min_times[index] << '\t' << max_times[index] << '\n';
            exit(1);
        }

        if (time > timeframe(tree.parent(node))) {
            std::cerr << "draw time: inconsistent times\n";
            std::cerr << time << '\t' << timeframe(tree.parent(node)) << '\n';
            std::cerr << tmin << '\t' << tmax << '\n';
            exit(1);
        }
        if (tree.is_leaf(node))   {
            if (time)   {
                std::cerr << "error: drawing non zero time for leaf\n";
                std::cerr << time << '\t' << timeframe(tree.parent(node)) << '\n';
                std::cerr << tmin << '\t' << tmax << '\n';
                exit(1);
            }
        }
        else    {
            if ((time <=0) || (time >= 1.0))    {
                std::cerr << "error: in joint node/age draw: inconsistent time for internal node\n";
                std::cerr << time << '\t' << timeframe(tree.parent(node)) << '\n';
                std::cerr << tmin << '\t' << tmax << '\n';
                std::cerr << index << '\n';
                std::cerr << "===\n";
                for (size_t i=imin; i<=imax; i++)   {
                    std::cerr << i << '\t' << min_times[i] << '\t' << max_times[i] << '\t' << w[i] << '\t';
                    node_distrib_t<Process>::CondL::printerr(condls.at(i));
                    std::cerr << '\n';
                }
                exit(1);
            }
        }
        return time;
    }

    template<class Tree, class Process, class CondLs, class Params, class... Keys>
    static void double_backward_branch_propagate(const Tree& tree, int node, Process& process, 
            size_t n,
            double tmin, double tmax, 
            double ptmin, double ptmax,
            const CondLs& young_condls, CondLs& old_condls, std::vector<CondLs>& mid_condls,
            const Params& params, std::tuple<Keys...>)   {

        auto old_min = grid_index(ptmin,n);
        auto old_max = grid_index(ptmax,n);

        auto young_min = grid_index(tmin,n);
        auto young_max = grid_index(tmax,n);

        if (young_max > old_max)    {
            std::cerr << "in double backward propagate: young_max > old_max \n";
            exit(1);
        }

        for (size_t i=old_min; i<=old_max; i++) {
            double old_t = (ptmin == ptmax) ? ptmin : grid_time(i,n);

            for (size_t j=young_min; j<=i; j++) {
                double young_t = (tmin == tmax) ? tmin : grid_time(j,n);

                node_distrib_t<Process>::backward_propagate(
                        young_condls[j], mid_condls[i][j],
                        young_t, old_t,
                        get<Keys>(params)()...);
            }
            node_distrib_t<Process>::CondL::mix(old_condls[i], young_min, i, mid_condls[i]);
        }
    }

    template<class Tree, class Process, class CondLs>
    static void recursive_double_backward(const Tree& tree, int node, Process& process, 
            size_t n,
            std::vector<double>& node_tmin,
            std::vector<double>& node_tmax,
            std::vector<CondLs>& young_condls, 
            std::vector<CondLs>& old_condls, 
            std::vector<std::vector<CondLs>>& mid_condls, 
            const std::vector<bool>& external_clamps)   {

        auto& node_vals = get<node_values>(process);
        auto& clamps = get<constraint>(process);
        auto timeframe = get<time_frame_field>(process);

        if (external_clamps[node])  {
            node_tmax[node] = node_tmin[node] = timeframe(node);
        }
        else if (tree.is_leaf(node))    {
            node_tmax[node] = node_tmin[node] = 0;
        }
        else if (tree.is_root(node))    {
            node_tmax[node] = node_tmin[node] = 1.0;
        }
        else    {
            node_tmax[node] = node_tmax[tree.parent(node)];
            node_tmin[node] = 0;
        }

        init_double_condls(process, 
                n, node_tmin[node], node_tmax[node],
                young_condls[node], node_vals[node],
                clamps[node], external_clamps[node]);

        for (auto c : tree.children(node))  {
            recursive_double_backward(tree, c, process, n,
                    node_tmin, node_tmax,
                    young_condls, old_condls, mid_condls,
                    external_clamps);

            if (node_tmin[node] < node_tmin[c])   {
                if (external_clamps[node])  {
                    std::cerr << "error in double backward: inconsistent times\n";
                    exit(1);
                }
                node_tmin[node] = node_tmin[c];
            }

            multiply_double_condls(process,
                    n, node_tmin[node], node_tmax[node],
                    old_condls[c], young_condls[node]);
        }

        if (! tree.is_root(node))   {
            double_backward_branch_propagate(tree, node, process, n,
                    node_tmin[node], node_tmax[node],
                    node_tmin[tree.parent(node)], node_tmax[tree.parent(node)],
                    young_condls[node], old_condls[node], mid_condls[node],
                    get<params>(process), param_keys_t<node_distrib_t<Process>>());
        }
    }

    // draw value at node, given parent node and conditional likelihood for downstream data
    template<class Chrono, class Process, class CondLs, class Params, class... Keys, class Gen>
    static void node_age_val_conditional_draw(Chrono& chrono, Process& process, int node, int n,
            double tmin, double tmax,
            double ptmin, double ptmax,
            CondLs& young_condls,
            CondLs& old_condls,
            std::vector<CondLs>& mid_condls,
            double& log_weight,
            const Params& params, std::tuple<Keys...>, Gen& gen)   {

        auto& tree = get<tree_field>(process);
        auto& clamps = get<constraint>(process);
        auto& node_vals = get<node_values>(process);

        auto& parent_val = node_vals[tree.parent(node)];
        double parent_age = chrono[tree.parent(node)];

        size_t i_old = grid_index(parent_age, n);
        size_t i_young = 0;

        // if (tree.is_leaf(node)) {
        if (tmin == tmax)   {
            // should check that
            double log_weight = node_distrib_t<Process>::CondL::point_eval(old_condls[0], parent_val);
            if (log_weight) {
                std::cerr << "non zero logweight for leaf node\n";
                exit(1);
            }
            chrono[node] = tmin;
            // chrono[node] = 0;
        }
        else    {
            double logZ = 0;
            chrono[node] = draw_conditional_time(process, node, tmin, tmax, n,
                    mid_condls[i_old], parent_val, logZ, i_young, gen);


            if (chrono[node] > chrono[tree.parent(node)])   {
                std::cerr << "double forward: inconsistent times\n";
                std::cerr << chrono[node] << '\t' << chrono[tree.parent(node)] << '\n';
                std::cerr << tmin << '\t' << tmax << '\n';
                std::cerr << node << '\t' << tree.parent(node) << '\n';
                exit(1);
            }
            if (tree.is_leaf(node))   {
                if (chrono[node])   {
                    std::cerr << "error: drawing non zero time for leaf\n";
                    std::cerr << chrono[node] << '\t' << tmin << '\t' << tmax << '\n';
                    exit(1);
                }
            }
            else    {
                if ((chrono[node] <= 0) || (chrono[node] >= 1.0))    {
                    std::cerr << "error: in joint node/age draw: inconsistent time for internal node\n";
                    std::cerr << chrono[node] << '\t' << tmin << '\t' << tmax << '\n';
                    exit(1);
                }
            }

            // correcting for approximate...
            double logZ0 = node_distrib_t<Process>::CondL::point_eval(old_condls[i_old], parent_val);
            log_weight -= logZ0;

            // ...versus exact marginal likelihood (marginal over all times for this node)
            log_weight += logZ;
        }

        // draw instant value at node
        node_distrib_t<Process>::node_conditional_draw(
                node_vals[node], clamps[node], node_vals[tree.parent(node)],
                chrono[node], chrono[tree.parent(node)],
                young_condls[i_young],
                get<Keys>(params)()..., gen);

        // correct for approximate versus exact marginal likelihood (marginal over node value)
        double logY0 = node_distrib_t<Process>::CondL::point_eval(mid_condls[i_old][i_young], parent_val);
        auto tmp_condl = node_distrib_t<Process>::CondL::make_init();
        node_distrib_t<Process>::backward_propagate(
                young_condls[i_young], tmp_condl,
                chrono[node], chrono[tree.parent(node)],
                get<Keys>(params)()...);
        double logY = node_distrib_t<Process>::CondL::point_eval(tmp_condl, parent_val);
        log_weight -= logY0;
        log_weight += logY;

        /*
        // grid times
        double t_old = (ptmin == ptmax) ? ptmin : grid_time(i_old, n);
        double t_young = (tmin == tmax) ? tmin : grid_time(i_young, n);

        // correct for discrepancy between drawing instant value based on grid times (proposal)...
        double grid_log_prob = node_distrib_t<Process>::node_logprob(
                node_vals[node], node_vals[tree.parent(node)],
                t_young, t_old,
                get<Keys>(params)()...);

        // ...versus density at exact times (target)
        double exact_log_prob = node_distrib_t<Process>::node_logprob(
                node_vals[node], node_vals[tree.parent(node)],
                chrono[node], chrono[tree.parent(node)],
                get<Keys>(params)()...);

        log_weight -= grid_log_prob;
        log_weight += exact_log_prob;
        */
    }

    // ****************************
    // direct backward-forward sampling

    template<class Tree, class Process, class CondLArray, class Gen>
    static void recursive_forward(const Tree& tree, int node, Process& process, CondLArray& young_condls, Gen& gen)    {

        if (tree.is_root(node)) {
            root_conditional_draw(process, young_condls[node],
                get<struct root_params>(process), param_keys_t<node_root_distrib_t<Process>>(), gen);
        }
        else    {
            path_conditional_draw(process, node, young_condls[node],
                get<struct params>(process), param_keys_t<node_distrib_t<Process>>(), gen);
        }

        for (auto c : tree.children(node))  {
            recursive_forward(tree, c, process, young_condls, gen);
        }
    }

    template<class Process, class Gen>
    static void conditional_draw(Process& process, Gen& gen)  {

        using Distrib = node_distrib_t<Process>;
        using CondL = typename Distrib::CondL;
        typename CondL::L initcondl = CondL::make_init();

        std::vector<typename CondL::L> old_condls(get<node_values>(process).size(), initcondl);
        std::vector<typename CondL::L> young_condls(get<node_values>(process).size(), initcondl);

        std::vector<bool> external_clamps(get<node_values>(process).size(), false);

        auto& tree = get<tree_field>(process);

        recursive_backward(tree, tree.root(), process, young_condls, old_condls, external_clamps);
        recursive_forward(tree, tree.root(), process, young_condls, gen);
    }

    // ****************************
    // proposals, importance sampling

    // ****************************
    // free-forward prior importance sampling

    // a class that implements single node/branch free-forward sampling on demand
    // importance sampling assuming the process is not conditioned longitudinally 
    // (e.g. at the tips, or more generally at some nodes)

    template<class Process>
    class free_forward_sampler    {

        Process& process;

        public:

        free_forward_sampler(Process& in_process) :
            process(in_process) {}

        void init(const std::vector<bool>& external_clamps)  {
            for (auto c : external_clamps)  {
                if (c)  {
                    std::cerr << "error: free forward sampler cannot work conditionally\n";
                    exit(1);
                }
            }
        }

        template<class Gen>
        void root_draw(typename node_distrib_t<Process>::instantT& val,
                double& log_weight,
                Gen& gen)  {

            auto& tree = get<tree_field>(process);
            auto& node_vals = get<node_values>(process);

            tree_process_methods::root_draw(process,
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
            tree_process_methods::path_draw(process, node, 
                get<params>(process), param_keys_t<node_distrib_t<Process>>(), gen);

            val = node_vals[node];
            path = path_vals[tree.get_branch(node)];
        }
    };

    template<class Process>
    static auto make_free_forward_sampler(Process& process) {
        return free_forward_sampler<Process>(process);
    }

    template<class Process, class BranchUpdate, class BranchLogProb>
    class free_forward_prior_importance_sampler  {

        Process& process;
        free_forward_sampler<Process> proposal;
        BranchUpdate update;
        BranchLogProb logprob;

        public:

        free_forward_prior_importance_sampler(Process& in_process, 
            BranchUpdate in_update, BranchLogProb in_logprob) :
                process(in_process), proposal(process),
                update(in_update), logprob(in_logprob) {}

        ~free_forward_prior_importance_sampler() {}

        void init(const std::vector<bool>& external_clamps)    {
            proposal.init(external_clamps);
        }

        template<class Gen>
        void root_draw(typename node_distrib_t<Process>::instantT& val, 
                double& log_weight, bool fixed_node, Gen& gen) {

            auto& tree = get<tree_field>(process);
            auto& node_vals = get<node_values>(process);

            if (! fixed_node)    {
                proposal.root_draw(val, log_weight, gen);
            }
            else    {
                node_vals[tree.root()] = val;
            }
        }

        template<class Gen>
        void path_draw(int node, 
                typename node_distrib_t<Process>::instantT& val, 
                const typename node_distrib_t<Process>::instantT& parent_val, 
                typename node_distrib_t<Process>::pathT& path, 
                double& log_weight, bool fixed_node, bool fixed_branch, Gen& gen) {

            auto& tree = get<tree_field>(process);
            auto& node_vals = get<node_values>(process);
            auto& path_vals = get<path_values>(process);
            int branch = tree.get_branch(node);

            if (! fixed_node)   {
                proposal.path_draw(node, val, parent_val, path, log_weight, gen);
            }
            else    {
                node_vals[tree.parent(node)] = parent_val;
                node_vals[node] = val;
                path_vals[branch] = path;
            }
            update(branch);
            log_weight += logprob(branch);
        }
    };

    template<class Process, class Update, class LogProb>
    static auto make_free_forward_prior_importance_sampler(Process& process, Update update, LogProb logprob)   {
        return free_forward_prior_importance_sampler<Process, Update, LogProb>(process, update, logprob);
    }

    // ****************************
    // prior importance sampling (conditional on longitudinal constraints)
    //
    // a class that implements single node/branch conditional sampling on demand
    // (longitudinal conditions, i.e. process is observed at some nodes) 
    // stores and computes conditional likelihoods upon creation
    // used as a proposal for prior importance sampling

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

        // of note, external clamps are used for the backward (here in init)
        // but not for the conditional sampling routines (below)
        // the conditional draws would be consistent with these external constraints
        // however, when a node has an external clamp
        // these routine are not called anyway (see e.g. prior_importance_sampler)

        void init(const std::vector<bool>& external_clamps)  {

            auto& tree = get<tree_field>(process);

            recursive_backward(tree, tree.root(),
                process, young_condls, old_condls, external_clamps);
        }

        template<class Gen>
        void root_draw(typename node_distrib_t<Process>::instantT& val,
                double& log_weight,
                Gen& gen)  {

            auto& tree = get<tree_field>(process);
            auto& node_vals = get<node_values>(process);

            root_conditional_draw(process, young_condls[tree.root()],
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

            node_conditional_draw(process, node, young_condls[node],
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
            auto& node_vals = get<node_values>(process);
            auto& path_vals = get<path_values>(process);

            node_vals[node] = val;
            node_vals[tree.parent(node)] = parent_val;

            tree_process_methods::bridge_draw(process, node, 
                get<params>(process), param_keys_t<node_distrib_t<Process>>(), gen);

            path = path_vals[tree.get_branch(node)];
        }
    };

    template<class Process>
    static auto make_conditional_sampler(Process& process) {
        return conditional_sampler<Process>(process);
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

            auto& tree = get<tree_field>(process);
            auto& node_vals = get<node_values>(process);

            if (! fixed_node)    {
                proposal.root_draw(val, log_weight, gen);
            }
            else    {
                node_vals[tree.root()] = val;
            }
        }

        template<class Gen>
        void path_draw(int node, 
                typename node_distrib_t<Process>::instantT& val, 
                const typename node_distrib_t<Process>::instantT& parent_val, 
                typename node_distrib_t<Process>::pathT& path, 
                double& log_weight, bool fixed_node, bool fixed_branch, Gen& gen) {

            auto& tree = get<tree_field>(process);
            auto& node_vals = get<node_values>(process);
            auto& path_vals = get<path_values>(process);
            int branch = tree.get_branch(node);

            if (! fixed_node)    {
                proposal.node_draw(node, val, parent_val, log_weight, gen);
            }
            else    {
                node_vals[tree.parent(node)] = parent_val;
                node_vals[node] = val;
            }
            if (! fixed_branch) {
                proposal.bridge_draw(node, val, parent_val, path, log_weight, gen);
            }
            else    {
                path_vals[branch] = path;
            }
            update(branch);
            log_weight += logprob(branch);
        }
    };

    template<class Process, class Update, class LogProb>
    static auto make_prior_importance_sampler(Process& process, Update update, LogProb logprob)   {
        return prior_importance_sampler<Process, Update, LogProb>(process, update, logprob);
    }

    template<class Process, class BranchUpdate, class BranchLogProb>
    class annealed_prior_importance_sampler  {

        Process& process;
        conditional_sampler<Process> proposal;
        BranchUpdate update;
        BranchLogProb logprob;
        size_t m;
        double tuning;

        public:

        annealed_prior_importance_sampler(Process& in_process, 
            BranchUpdate in_update, BranchLogProb in_logprob, size_t in_m, double in_tuning) :
                process(in_process), proposal(process),
                update(in_update), logprob(in_logprob), m(in_m), tuning(in_tuning) {}

        ~annealed_prior_importance_sampler() {}

        void init(const std::vector<bool>& external_clamps)    {
            proposal.init(external_clamps);
        }

        template<class Gen>
        void root_draw(typename node_distrib_t<Process>::instantT& val, 
                double& log_weight, bool fixed_node, Gen& gen) {

            auto& tree = get<tree_field>(process);
            auto& node_vals = get<node_values>(process);

            if (! fixed_node)    {
                proposal.root_draw(val, log_weight, gen);
            }
            else    {
                node_vals[tree.root()] = val;
            }
        }

        template<class Gen>
        void path_draw(int node, 
                typename node_distrib_t<Process>::instantT& val, 
                const typename node_distrib_t<Process>::instantT& parent_val, 
                typename node_distrib_t<Process>::pathT& path, 
                double& log_weight, bool fixed_node, bool fixed_branch, Gen& gen) {

            auto& tree = get<tree_field>(process);
            auto& node_vals = get<node_values>(process);
            auto& path_vals = get<path_values>(process);
            int branch = tree.get_branch(node);

            if (! fixed_node)    {
                proposal.node_draw(node, val, parent_val, log_weight, gen);
            }
            else    {
                node_vals[tree.parent(node)] = parent_val;
                node_vals[node] = val;
            }

            // note: path logprob already included in acceptance ratio of single_branch_mh_move
            auto ais_logprob = [lp=logprob, mm=m] (int i)    {
                double f = double(i)/mm;
                return [f, lp] (int branch) {
                    return f*lp(branch);};
            };

            auto ais_kernel = [&proc=process, mm=m, t=tuning] (int i) {
                double f = double(i)/mm;
                double tt = t/(0.1 + sqrt(f));
                return [&proc, tt] (auto& path, double t_young, double t_old, Gen& gen) {
                    return bridge_kernel(proc, tt, path, t_young, t_old,
                            get<params>(proc), param_keys_t<node_distrib_t<Process>>(), gen);
                };
            };

            if (! fixed_branch) {

                double logw = 0;

                proposal.bridge_draw(node, val, parent_val, path, log_weight, gen);
                update(branch);
                logw += logprob(branch);

                for (size_t i=1; i<m; i++)  {
                    single_branch_mh_move(tree, node, process,
                            ais_kernel(i), update, ais_logprob(i), gen);
                    logw += logprob(branch);
                }

                log_weight += logw/m;
                path = path_vals[branch];
            }
            else    {

                double logw = 0;

                path_vals[branch] = path;
                update(branch);
                logw += logprob(branch);

                for (size_t i=m-1; i>0; i--)  {
                    single_branch_mh_move(tree, node, process,
                            ais_kernel(i), update, ais_logprob(i), gen);
                    logw += logprob(branch);
                }

                log_weight += logw/m;

                path_vals[branch] = path;
                update(branch);
            }
        }
    };

    template<class Process, class Update, class LogProb>
    static auto make_annealed_prior_importance_sampler(
            Process& process, Update update, LogProb logprob, size_t m, double tuning)   {
        return annealed_prior_importance_sampler<Process, Update, LogProb>(
                process, update, logprob, m, tuning);
    }

    // ****************************
    // double (time / process) prior importance sampling
    //

    template<class Chrono, class Process>
    class chrono_process_conditional_sampler    {

        Chrono& chrono;
        Process& process;

        size_t ntimes;
        std::vector<double> node_tmin;
        std::vector<double> node_tmax;

        std::vector<std::vector<typename node_distrib_t<Process>::CondL::L>> young_condls;
        std::vector<std::vector<typename node_distrib_t<Process>::CondL::L>> old_condls;
        std::vector<std::vector<std::vector<typename node_distrib_t<Process>::CondL::L>>> mid_condls;

        public:

        chrono_process_conditional_sampler(Chrono& in_chrono,
                Process& in_process, int in_ntimes) :
            chrono(in_chrono),
            process(in_process), 
            ntimes(in_ntimes),
            node_tmin(get<node_values>(process).size(),0),
            node_tmax(get<node_values>(process).size(),1),
            young_condls(get<node_values>(process).size(), 
                    std::vector<typename node_distrib_t<Process>::CondL::L>(ntimes,
                        node_distrib_t<Process>::CondL::make_init())),
            old_condls(get<node_values>(process).size(), 
                    std::vector<typename node_distrib_t<Process>::CondL::L>(ntimes,
                        node_distrib_t<Process>::CondL::make_init())),
            mid_condls(get<node_values>(process).size(), 
                    std::vector<std::vector<typename node_distrib_t<Process>::CondL::L>>(ntimes,
                        std::vector<typename node_distrib_t<Process>::CondL::L>(ntimes,
                            node_distrib_t<Process>::CondL::make_init())))  {
        }

        void init(const std::vector<bool>& external_clamps)  {

            auto& tree = get<tree_field>(process);

            recursive_double_backward(tree, tree.root(), process, 
                    ntimes, node_tmin, node_tmax,
                    young_condls, old_condls, mid_condls,
                    external_clamps);
        }

        template<class Gen>
        void root_draw(typename node_distrib_t<Process>::instantT& val,
                double& log_weight,
                Gen& gen)  {

            auto& tree = get<tree_field>(process);
            auto& node_vals = get<node_values>(process);

            root_conditional_draw(process, young_condls[tree.root()][0],
                get<struct root_params>(process), param_keys_t<node_root_distrib_t<Process>>(), gen);

            val = node_vals[tree.root()];
        }

        template<class Gen>
        void node_draw(int node, double& age, double parent_age,
                typename node_distrib_t<Process>::instantT& val, 
                const typename node_distrib_t<Process>::instantT& parent_val, 
                double& log_weight, Gen& gen)    {

            auto& tree = get<tree_field>(process);
            auto& node_vals = get<node_values>(process);

            chrono[tree.parent(node)] = parent_age;
            node_vals[tree.parent(node)] = parent_val;

            double tmax = (node_tmax[node] < parent_age) ? node_tmax[node] : parent_age;
            double tmin = node_tmin[node];

            node_age_val_conditional_draw(chrono, process, node,
                    ntimes, tmin, tmax,
                    node_tmin[tree.parent(node)], node_tmax[tree.parent(node)],
                    young_condls[node], old_condls[node], mid_condls[node],
                    log_weight,
                    get<params>(process), param_keys_t<node_distrib_t<Process>>(), gen);

            age = chrono[node];
            val = node_vals[node];

            if (! tree.is_leaf(node))   {
                if ((age <= 0) || (age >= 1.0)) {
                    std::cerr << "incorrect internal node age : " << age << '\n';
                    std::cerr << tmin << '\t' << tmax << '\t' << parent_age << '\n';
                    exit(1);
                }
            }
            if (parent_age <= age)  {
                std::cerr << "in double pf: inconsistent ages\n";
                std::cerr << age << '\t' << parent_age << '\t' << tmin << '\t' << tmax << '\n';
                std::cerr << "node : " << node << '\t' << tree.parent(node) << '\n';
                if (tree.is_leaf(node)) {
                    std::cerr << "node is leaf\n";
                }
                else    {
                    std::cerr << "node is internal\n";
                }
                exit(1);
            }
        }

        template<class Gen>
        void bridge_draw(int node, 
                const typename node_distrib_t<Process>::instantT& val, 
                const typename node_distrib_t<Process>::instantT& parent_val, 
                typename node_distrib_t<Process>::pathT& path,
                double& log_weight, Gen& gen)    {

            auto& tree = get<tree_field>(process);
            auto& node_vals = get<node_values>(process);
            auto& path_vals = get<path_values>(process);

            node_vals[node] = val;
            node_vals[tree.parent(node)] = parent_val;

            tree_process_methods::bridge_draw(process, node, 
                get<params>(process), param_keys_t<node_distrib_t<Process>>(), gen);

            path = path_vals[tree.get_branch(node)];
        }
    };

    template<class Chrono, class Process>
    static auto make_chrono_process_conditional_sampler(Chrono& chrono, Process& process, size_t ntimes) {
        return conditional_sampler<Process>(chrono, process, ntimes);
    }

    template<class Chrono, class Process, class BranchUpdate, class BranchLogProb>
    class chrono_process_prior_importance_sampler  {

        Chrono& chrono;
        Process& process;
        chrono_process_conditional_sampler<Chrono,Process> proposal;
        BranchUpdate update;
        BranchLogProb logprob;

        public:

        chrono_process_prior_importance_sampler(Chrono& in_chrono, Process& in_process, 
            BranchUpdate in_update, BranchLogProb in_logprob, size_t in_ntimes) :
                chrono(in_chrono), process(in_process),
                proposal(chrono, process, in_ntimes),
                update(in_update), logprob(in_logprob) {
        }

        ~chrono_process_prior_importance_sampler() {}

        void init(const std::vector<bool>& external_clamps)    {
            proposal.init(external_clamps);
        }

        template<class Gen>
        void root_draw(typename node_distrib_t<Process>::instantT& val, 
                double& log_weight, bool fixed_node, Gen& gen) {

            auto& tree = get<tree_field>(process);
            auto& node_vals = get<node_values>(process);

            if (! fixed_node)    {
                proposal.root_draw(val, log_weight, gen);
            }
            else    {
                node_vals[tree.root()] = val;
            }
        }

        template<class Gen>
        void path_draw(int node, 
                double& age, double parent_age,
                typename node_distrib_t<Process>::instantT& val, 
                const typename node_distrib_t<Process>::instantT& parent_val, 
                typename node_distrib_t<Process>::pathT& path, 
                double& log_weight, bool fixed_node, bool fixed_branch, Gen& gen) {

            auto& tree = get<tree_field>(process);
            auto& node_vals = get<node_values>(process);
            auto& path_vals = get<path_values>(process);
            int branch = tree.get_branch(node);

            if (! fixed_node)    {
                proposal.node_draw(node, age, parent_age, val, parent_val, log_weight, gen);
            }
            else    {
                chrono[tree.parent(node)] = parent_age;
                chrono[node] = age;
                node_vals[tree.parent(node)] = parent_val;
                node_vals[node] = val;
            }
            if (! fixed_branch) {
                proposal.bridge_draw(node, val, parent_val, path, log_weight, gen);
            }
            else    {
                path_vals[branch] = path;
            }
            update(branch);
            log_weight += logprob(branch);
        }
    };

    template<class Chrono, class Process, class Update, class LogProb>
    static auto make_chrono_process_prior_importance_sampler(Chrono& chrono, Process& process, Update update, LogProb logprob, size_t ntimes)   {
        return chrono_process_prior_importance_sampler<Chrono, Process, Update, LogProb>(
                chrono, process, update, logprob, ntimes);
    }


    // ****************************
    // particle filters

    template<class Gen>
    static size_t choose_particle(const std::vector<double>& log_weights, Gen& gen)    {

        std::vector<double> w(log_weights.size(), 0);
        double max = 0;
        for (size_t i=0; i<log_weights.size(); i++) {
            if ((!i) || (max < log_weights[i])) {
                max = log_weights[i];
            }
        }
        double tot = 0;
        for (size_t i=0; i<log_weights.size(); i++) {
            w[i] = exp(log_weights[i] - max);
            tot += w[i];
        }
        for (size_t i=0; i<log_weights.size(); i++) {
            w[i] /= tot;
        }
        std::discrete_distribution<int> distrib(w.begin(), w.end());
        return distrib(gen);
    }

    template<class Gen>
    static void bootstrap_pf(bool conditional, 
            std::vector<double> log_weights,
            std::vector<int>& choose, 
            double min_effsize, Gen& gen)  {

        std::vector<double> w(log_weights.size(), 0);
        double max = 0;
        for (size_t i=0; i<log_weights.size(); i++) {
            if ((!i) || (max < log_weights[i])) {
                max = log_weights[i];
            }
        }
        double tot = 0;
        for (size_t i=0; i<log_weights.size(); i++) {
            w[i] = exp(log_weights[i] - max);
            tot += w[i];
        }
        double s2 = 0;
        for (size_t i=0; i<log_weights.size(); i++) {
            w[i] /= tot;
            s2 += w[i]*w[i];
        }
        double effsize = 1.0/s2;
        if (effsize < min_effsize)   {
            std::discrete_distribution<int> distrib(w.begin(), w.end());
            if (conditional)    {
                choose[0] = 0;
            }
            else    {
                choose[0] = distrib(gen);
            }
            for (size_t i=1; i<log_weights.size(); i++) {
                choose[i] = distrib(gen);
            }

            double mean = tot / log_weights.size();
            double log_mean_weight = log(mean) + max;
            for (size_t i=0; i<log_weights.size(); i++) {
                log_weights[i] = log_mean_weight;
            }
        }
        else    {
            for (size_t i=0; i<log_weights.size(); i++) {
                choose[i] = i;
            }
        }
    }

    template<class Process, class WDist>
    class tree_pf   {
        
        Process& process;
        const Tree& tree;
        WDist& wdist;
        std::vector<std::vector<typename node_distrib_t<Process>::instantT>> node_swarm;
        std::vector<std::vector<typename node_distrib_t<Process>::pathT>> path_swarm;
        std::vector<double> log_weights;
        std::vector<int> node_ordering;
        std::vector<std::vector<int>> node_ancestors;
        std::vector<std::vector<int>> pf_ancestors;
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
            node_ancestors(tree.nb_nodes(), std::vector<int>(n,0)),
            pf_ancestors(tree.nb_nodes(), std::vector<int>(n,0)),
            counter(0),
            external_clamps(tree.nb_nodes(), false) {}

        ~tree_pf() {}

        size_t size() {return log_weights.size();}

        template<class Gen>
        void init(bool conditional, int min, int max, Gen& gen) {

            if (conditional)    {

                auto& node_vals = get<node_values>(process);
                for (size_t node=0; node<tree.nb_nodes(); node++)    {
                    node_swarm[node][0] = node_vals[node];
                }

                auto& path_vals = get<path_values>(process);
                for (size_t branch=0; branch<tree.nb_branches(); branch++)    {
                    path_swarm[branch][0] = path_vals[branch];
                }

                for (size_t node=0; node<tree.nb_nodes(); node++)   {
                    external_clamps[node] = false;
                }
                if (max < int(tree.nb_nodes()))  {

                    // split_tree::choose_components(tree, min, max, external_clamps, gen);

                    double cond_frac = ((double) max)/tree.nb_nodes();
                    std::vector<double> w = {1-cond_frac, cond_frac};
                    std::discrete_distribution<int> distrib(w.begin(), w.end());
                    for (size_t node=0; node<tree.nb_nodes(); node++)   {
                        if (tree.is_leaf(node) || tree.is_root(node))   {
                            external_clamps[node] = false;
                        }
                        else    {
                            external_clamps[node] = distrib(gen);
                        }
                    }

                    for (size_t node=0; node<tree.nb_nodes(); node++)   {
                        if (external_clamps[node])  {
                            for (size_t i=1; i<size(); i++) {
                                node_swarm[node][i] = node_vals[node];
                            }
                        }
                    }
                }
            }
            wdist.init(external_clamps);
        }

        template<class Gen>
        int run(bool conditional, int max_size, double min_effsize, Gen& gen)  {

            init(conditional, 1, max_size, gen);

            int ret = sub_run(tree.root(), conditional, min_effsize, gen);

            if (max_size < int(tree.nb_nodes())) {
                for (size_t node=0; node<tree.nb_nodes(); node++)   {
                    if (external_clamps[node])  {
                        sub_run(node, conditional, min_effsize, gen);
                    }
                }
            }
            return ret;
        }

        template<class Gen>
        int sub_run(int node, bool conditional, double min_effsize, Gen& gen)    {

            counter = 0;
            for (auto& l : log_weights) {
                l = 0;
            }

            std::vector<int> b(size(), 0);
            for (size_t i=0; i<size(); i++)  {
                b[i] = i;
            }

            // run forward particle filter
            // will stop at the nodes that are externally clamped
            forward_pf(node, conditional, min_effsize, b, gen);

            /*
            std::map<int,int> ancmap;
            for (size_t i=0; i<size(); i++)  {
                ancmap[b[i]]++;
            }
            std::cerr << ancmap[0] << '\t';
            */

            // choose random particle
            int c = choose_particle(log_weights, gen);

            // pull out
            for (int i=counter-1; i>=0; i--) {
                int node = node_ordering[i];
                get<node_values>(process)[node] = node_swarm[node][c];
                if (i)  {
                    get<path_values>(process)[tree.get_branch(node)] = 
                        path_swarm[tree.get_branch(node)][c];
                    c = pf_ancestors[i][c];
                }
            }

            return b[c];
        }

        template<class Gen>
        void forward_pf(int node, bool conditional, double min_effsize, std::vector<int>& b, Gen& gen)   {

            // when entering forward,
            // b[i] gives the ancestor at tree.parent(node) of current particle i
            // (current particle may currently extend downstream from current node
            // according to the order over nodes induced by the depth-first recursion;
            // if this node is the first child of its parent, 
            // then current particle extends up to tree.parent(node), so b[i] = i)

            node_ordering[counter] = node;
            auto& choose = pf_ancestors[counter];
            counter++;

            if (tree.is_root(node))   {
                // silly (not used), but just for overall consistency
                for (size_t i=0; i<size(); i++)  {
                    node_ancestors[node][i] = b[i];
                }
                for (size_t i=0; i<size(); i++)  {
                    choose[i] = i;
                }
            }
            else    {
                bootstrap_pf(conditional, log_weights, choose, min_effsize, gen);

                for (size_t i=0; i<size(); i++)  {
                    node_ancestors[node][i] = b[choose[i]];
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
                if (counter > 1)    {
                    auto branch = tree.get_branch(node);
                    for (size_t i=0; i<size(); i++)  {
                        wdist.path_draw(node, 
                                node_swarm[node][i],
                                node_swarm[tree.parent(node)][node_ancestors[node][i]], 
                                path_swarm[branch][i],
                                log_weights[i],
                                external_clamps[node] || (conditional && (!i)), // fixed node
                                (conditional && (!i)),                          // fixed branch
                                gen);
                    }
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

            if (! external_clamps[node])    {
                for (auto c : tree.children(node))  {
                    forward_pf(c, conditional, min_effsize, bb, gen);
                }
            }

            // when leaving forward, b should should be updated
            // to the index of particle i at tree.parent(node), 
            // based on the index of particle i at this node, 
            // such as given by bb upon returning from last call to forward

            for (size_t i=0; i<size(); i++)  {
                b[i] = node_ancestors[node][bb[i]];
            }
        }
    };

    template<class Process, class WDist>
    static auto make_particle_filter(Process& process, WDist& wdist, size_t n)    {
        return tree_pf<Process, WDist>(process, wdist, n);
    }

    template<class Chrono, class Process, class WDist>
    class joint_chrono_process_pf   {
        
        Chrono& chrono;
        Process& process;
        const Tree& tree;
        WDist& wdist;
        std::vector<std::vector<typename node_distrib_t<Process>::instantT>> node_swarm;
        std::vector<std::vector<double>> age_swarm;
        std::vector<std::vector<typename node_distrib_t<Process>::pathT>> path_swarm;
        std::vector<double> log_weights;
        std::vector<int> node_ordering;
        std::vector<std::vector<int>> node_ancestors;
        std::vector<std::vector<int>> pf_ancestors;
        size_t counter;
        std::vector<bool> external_clamps;

        public:

        joint_chrono_process_pf(Chrono& in_chrono, Process& in_process, WDist& in_wdist, size_t n) :
            chrono(in_chrono),
            process(in_process),
            tree(get<tree_field>(process)),
            wdist(in_wdist),
            node_swarm(tree.nb_nodes(), 
                    std::vector<typename node_distrib_t<Process>::instantT>(n, 
                        get<node_values>(process)[0])),
            age_swarm(tree.nb_nodes(), std::vector<double>(n,0)),
            path_swarm(tree.nb_branches(), 
                    std::vector<typename node_distrib_t<Process>::pathT>(n, 
                        get<path_values>(process)[0])),
            log_weights(n, 0),
            node_ordering(tree.nb_nodes(), 0),
            node_ancestors(tree.nb_nodes(), std::vector<int>(n,0)),
            pf_ancestors(tree.nb_nodes(), std::vector<int>(n,0)),
            counter(0),
            external_clamps(tree.nb_nodes(), false) {}

        ~joint_chrono_process_pf() {}

        size_t size() {return log_weights.size();}

        template<class Gen>
        void init(bool conditional, int min, int max, Gen& gen) {

            for (size_t node=0; node<tree.nb_nodes(); node++)    {
                if (tree.is_root(node)) {
                    for (size_t i=0; i<size(); i++) {
                        age_swarm[node][i] = 1.0;
                    }
                }
                if (tree.is_leaf(node)) {
                    for (size_t i=0; i<size(); i++) {
                        age_swarm[node][i] = 0;
                    }
                }
            }

            if (conditional)    {

                auto& node_vals = get<node_values>(process);
                for (size_t node=0; node<tree.nb_nodes(); node++)    {
                    node_swarm[node][0] = node_vals[node];
                }

                auto& path_vals = get<path_values>(process);
                for (size_t branch=0; branch<tree.nb_branches(); branch++)    {
                    path_swarm[branch][0] = path_vals[branch];
                }

                for (size_t node=0; node<tree.nb_nodes(); node++)    {
                    age_swarm[node][0] = chrono[node];
                }
            
                for (size_t node=0; node<tree.nb_nodes(); node++)   {
                    external_clamps[node] = false;
                }
                if (max < int(tree.nb_nodes()))  {

                    std::cerr << "subtrees not yet working for joint pf\n";
                    exit(1);

                    // split_tree::choose_components(tree, min, max, external_clamps, gen);

                    double cond_frac = 1.0 / max;
                    std::vector<double> w = {1-cond_frac, cond_frac};
                    std::discrete_distribution<int> distrib(w.begin(), w.end());
                    for (size_t node=0; node<tree.nb_nodes(); node++)   {
                        if (tree.is_leaf(node) || tree.is_root(node))   {
                            external_clamps[node] = false;
                        }
                        else    {
                            external_clamps[node] = distrib(gen);
                        }
                    }

                    for (size_t node=0; node<tree.nb_nodes(); node++)   {
                        if (external_clamps[node])  {
                            for (size_t i=1; i<size(); i++) {
                                node_swarm[node][i] = node_vals[node];
                                age_swarm[node][i] = chrono[node];
                            }
                        }
                    }
                }
            }
            wdist.init(external_clamps);
        }

        template<class Gen>
        int run(bool conditional, int max_size, double min_effsize, Gen& gen)  {

            init(conditional, 1, max_size, gen);

            int ret = sub_run(tree.root(), conditional, min_effsize, gen);

            if (max_size < int(tree.nb_nodes())) {
                for (size_t node=0; node<tree.nb_nodes(); node++)   {
                    if (external_clamps[node])  {
                        sub_run(node, conditional, min_effsize, gen);
                    }
                }
            }
            return ret;
        }

        template<class Gen>
        int sub_run(int node, bool conditional, double min_effsize, Gen& gen)    {

            counter = 0;
            for (auto& l : log_weights) {
                l = 0;
            }

            std::vector<int> b(size(), 0);
            for (size_t i=0; i<size(); i++)  {
                b[i] = i;
            }

            // run forward particle filter
            // will stop at the nodes that are externally clamped
            forward_pf(node, conditional, min_effsize, b, gen);

            // choose random particle
            int c = choose_particle(log_weights, gen);

            // pull out
            for (int i=counter-1; i>=0; i--) {
                int node = node_ordering[i];
                get<node_values>(process)[node] = node_swarm[node][c];
                chrono[node] = age_swarm[node][c];
                if (i)  {
                    get<path_values>(process)[tree.get_branch(node)] = 
                        path_swarm[tree.get_branch(node)][c];
                    c = pf_ancestors[i][c];
                }
            }

            return b[c];
        }

        template<class Gen>
        void forward_pf(int node, bool conditional, double min_effsize, std::vector<int>& b, Gen& gen)   {

            // when entering forward,
            // b[i] gives the ancestor at tree.parent(node) of current particle i
            // (current particle may currently extend downstream from current node
            // according to the order over nodes induced by the depth-first recursion;
            // if this node is the first child of its parent, 
            // then current particle extends up to tree.parent(node), so b[i] = i)

            node_ordering[counter] = node;
            auto& choose = pf_ancestors[counter];
            counter++;

            if (tree.is_root(node))   {
                // silly (not used), but just for overall consistency
                for (size_t i=0; i<size(); i++)  {
                    node_ancestors[node][i] = b[i];
                }
                for (size_t i=0; i<size(); i++)  {
                    choose[i] = i;
                }
            }
            else    {
                bootstrap_pf(conditional, log_weights, choose, min_effsize, gen);

                for (size_t i=0; i<size(); i++)  {
                    node_ancestors[node][i] = b[choose[i]];
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
                if (counter > 1)    {
                    auto branch = tree.get_branch(node);
                    for (size_t i=0; i<size(); i++)  {
                        wdist.path_draw(node, 
                                age_swarm[node][i],
                                age_swarm[tree.parent(node)][node_ancestors[node][i]],
                                node_swarm[node][i],
                                node_swarm[tree.parent(node)][node_ancestors[node][i]], 
                                path_swarm[branch][i],
                                log_weights[i],
                                external_clamps[node] || (conditional && (!i)), // fixed node
                                (conditional && (!i)),                          // fixed branch
                                gen);
                    }
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

            if (! external_clamps[node])    {
                for (auto c : tree.children(node))  {
                    forward_pf(c, conditional, min_effsize, bb, gen);
                }
            }

            // when leaving forward, b should should be updated
            // to the index of particle i at tree.parent(node), 
            // based on the index of particle i at this node, 
            // such as given by bb upon returning from last call to forward

            for (size_t i=0; i<size(); i++)  {
                b[i] = node_ancestors[node][bb[i]];
            }
        }
    };

    template<class Chrono, class Process, class WDist>
    static auto make_joint_chrono_process_particle_filter(Chrono& chrono, 
            Process& process, WDist& wdist, size_t n)    {
        return joint_chrono_process_pf<Chrono, Process, WDist>(chrono, process, wdist, n);
    }
};

