#pragma once

struct branch_map   {

    template<class Chrono, class Process, class Lambda>
    static auto make_branch_sums(Chrono& chrono, Process& process, Lambda lambda)  {
        auto ret = make_dnode_array<custom_dnode<typename node_distrib_t<Process>::instantT>>(
                chrono.get_tree().nb_branches(),
                [&ch=chrono, &process, &lambda] (int branch) { return
                    [&old_age = ch[ch.get_tree().get_older_node(branch)], 
                     &young_age = ch[ch.get_tree().get_younger_node(branch)],
                     &path = get<path_values>(process)[branch],
                     lambda] 
                        (typename node_distrib_t<Process>::instantT& sum)   {
                            sum = get_branch_sum(path, old_age, young_age, lambda);
                    };
                });
        gather(ret);
        return ret;
    }

    template<class Process, class Lambda>
    static auto make_branch_means(Process& process, Lambda lambda)  {
        auto ret = make_dnode_array<custom_dnode<typename node_distrib_t<Process>::instantT>>(
                get<tree_field>(process).nb_branches(),
                [&process, &lambda] (int branch) { return
                    [&path = get<path_values>(process)[branch], lambda] 
                        (typename node_distrib_t<Process>::instantT& mean)   {
                            mean = get_branch_mean(path, lambda);
                    };
                });
        gather(ret);
        return ret;
    }
};

