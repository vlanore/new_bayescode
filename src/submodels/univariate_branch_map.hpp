#pragma once

// #include "exp_mid_sum.hpp"

struct branch_map   {

    /*
    template<class Process>
    static auto make_branch_exp_mid_sums(Process& process)  {
        auto ret = make_dnode_array<exp_mid_sum>(
                get<tree_field>(process).nb_branches(),
                [&nodes=get<node_values>(process), &tree = get<tree_field>(process)] (int branch) {
                    return nodes[tree.get_older_node(branch)];},
                [&nodes=get<node_values>(process), &tree = get<tree_field>(process)] (int branch) {
                    return nodes[tree.get_younger_node(branch)];},
                [&nodes=get<node_values>(process), &tree = get<tree_field>(process),
                 timeframe = get<time_frame_field>(process)] (int branch) {
                    return timeframe(tree.get_younger_node(branch));},
                [&nodes=get<node_values>(process), &tree = get<tree_field>(process),
                 timeframe = get<time_frame_field>(process)] (int branch) {
                    return timeframe(tree.get_older_node(branch));});
        gather(ret);
        return ret;
    };
    */

    template<class Chrono, class Process, class Lambda>
    static auto make_branch_sums(Chrono& chrono, Process& process, Lambda lambda)  {
        auto ret = make_dnode_array<custom_dnode<typename node_distrib_t<Process>::instantT>>(
                chrono.get_tree().nb_branches(),
                // [&ch=chrono, &nodes=get<node_values>(process), lambda] (int branch) { return
                [&ch=chrono, &paths=get<path_values>(process), lambda] (int branch) { return
                    [&old_age = ch[ch.get_tree().get_older_node(branch)], 
                     &young_age = ch[ch.get_tree().get_younger_node(branch)],
                     &path = paths[branch],
                     // &young_x = nodes[ch.get_tree().get_younger_node(branch)],
                     // &old_x = nodes[ch.get_tree().get_older_node(branch)],
                     lambda]
                        (typename node_distrib_t<Process>::instantT& sum)   {
                            // sum = (old_age - young_age) * (lambda(young_x) + lambda(old_x)) / 2;
                            sum = get_branch_sum(path, young_age, old_age, lambda);
                    };
                });
        gather(ret);
        return ret;
    }

    template<class Process, class Lambda>
    static auto make_branch_means(Process& process, Lambda lambda)  {
        auto ret = make_dnode_array<custom_dnode<typename node_distrib_t<Process>::instantT>>(
                get<tree_field>(process).nb_branches(),
                [&paths=get<path_values>(process), lambda] (int branch) { return
                    [&path = paths[branch], lambda] 
                        (typename node_distrib_t<Process>::instantT& mean)   {
                            mean = get_branch_mean(path, lambda);
                    };
                });
        gather(ret);
        return ret;
    }
};

