
#include "tree_factory.hpp"

template<class Process, class Params, class... Keys>
static double unpack_tree_process_branch_logprob(Process& process, int branch, const Params& params, std::tuple<Keys...>)  {

    auto& tree = get<tree_field>(process);
    using Distrib = node_distrib_t<Process>;
    auto timeframe = get<time_frame_field>(process);
    auto& v = get<value>(process);
    int younger = tree.get_younger_node(branch);
    int older = tree.get_older_node(branch);
    double dt = timeframe(older) - timeframe(younger);
    return Distrib::logprob(v[younger], false, v[older], dt, get<Keys>(params)()...);
}

template<class Process>
static auto tree_process_branch_logprob(Process& process) {
    return [&process] (int branch)   {
        return unpack_tree_process_branch_logprob(process, branch, 
                get<params>(process), param_keys_t<node_distrib_t<Process>>());
    };
}

template<class Process, class Params, class... Keys>
static double unpack_tree_process_node_logprob(Process& process, int node, const Params& params, std::tuple<Keys...>)   {

    using Distrib = node_distrib_t<Process>;
    auto& tree = get<tree_field>(process);
    auto timeframe = get<time_frame_field>(process);
    auto& v = get<value>(process);
    double tot = 0;
    if (tree.is_root(node)) {
        tot += Distrib::logprob(v[node], true, v[node], 0, get<Keys>(params)()...);
    }
    else    {
        double dt = timeframe(tree.parent(node)) - timeframe(node);
        tot += Distrib::logprob(v[node], false, v[tree.parent(node)], dt, get<Keys>(params)()...);
    }
    for (auto c : tree.children(node))  {
        double dt = timeframe(node) - timeframe(c);
        tot += Distrib::logprob(v[c], false, v[node], dt, get<Keys>(params)()...);
    }
    return tot;
}

template<class Process>
static auto tree_process_node_logprob(Process& process) {
    return [&process] (int node)   {
        return unpack_tree_process_node_logprob(process, node, 
                get<params>(process), param_keys_t<node_distrib_t<Process>>());
    };
}

template<class Tree, class Process, class Proposal, class BranchUpdate, class BranchLogProb, class Gen>
static void single_node_tree_process_mh_move(Tree& tree, int node, Process& process, Proposal propose, BranchUpdate update, BranchLogProb logprob, Gen& gen)  {

    auto& x = get<value>(process)[node];
    auto bk = x;
    double logprobbefore = tree_process_node_logprob(process)(node) + tree_factory::sum_around_node(tree, logprob)(node);
    double logh = propose(x, gen);
    tree_factory::do_around_node(tree, update)(node);
    double logprobafter = tree_process_node_logprob(process)(node) + tree_factory::sum_around_node(tree, logprob)(node);
    double delta = logprobafter - logprobbefore + logh;
    bool accept = decide(delta, gen);
    if (! accept)   {
        x = bk;
        tree_factory::do_around_node(tree, update)(node);
    }
}

template<class Tree, class Process, class Proposal, class BranchUpdate, class BranchLogProb, class Gen>
static void recursive_tree_process_mh_move(Tree& tree, int node, Process& process, Proposal propose, BranchUpdate update, BranchLogProb logprob, Gen& gen)    {
    single_node_tree_process_mh_move(tree, node, process, propose, update, logprob, gen);
    for (auto c : tree.children(node)) {
        recursive_tree_process_mh_move(tree, c, process, propose, update, logprob, gen);
    }
    single_node_tree_process_mh_move(tree, node, process, propose, update, logprob, gen);
}

template<class Process, class Proposal, class BranchUpdate, class BranchLogProb, class Gen>
static void tree_process_node_by_node_mh_move(Process& process, Proposal propose, BranchUpdate update, BranchLogProb logprob, Gen& gen)  {
    auto& tree = get<tree_field>(process);
    recursive_tree_process_mh_move(tree, tree.root(), process, propose, update, logprob, gen);
}

template<class ParamKey, class Tree, class Process, class SS, class Params, class... Keys>
static void tree_process_local_add_branch_suffstat(ParamKey key, Tree& tree, int node, Process& process, SS& ss, const Params& params, std::tuple<Keys...>)  {

    auto& v = get<value>(process);
    auto timeframe = get<time_frame_field>(process);

    node_distrib_t<Process>::add_branch_suffstat(
            key, ss, v[node], v[tree.parent(node)],
            timeframe(tree.parent(node)) - timeframe(node),
            get<Keys>(params)()...);
}

template<class ParamKey, class Tree, class Process, class SS>
static void recursive_tree_process_add_branch_suffstat(ParamKey key, Tree& tree, int node, Process& process, SS& ss) {
    if (! tree.is_root(node)) {
        tree_process_local_add_branch_suffstat(key, tree, node, process, ss, 
                get<params>(process), param_keys_t<node_distrib_t<Process>>());
    }
    for (auto c : tree.children(node)) {
        recursive_tree_process_add_branch_suffstat(key, tree, c, process, ss);
    }
}

template<class ParamKey, class Process, class SS>
static void tree_process_add_branch_suffstat(Process& process, SS& ss)    {
    auto& tree = get<tree_field>(process);
    ParamKey key;
    recursive_tree_process_add_branch_suffstat(key, tree, tree.root(), process, ss);
}

