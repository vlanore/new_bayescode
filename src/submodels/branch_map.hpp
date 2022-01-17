#pragma once

struct branch_map   {

    template<class Chrono, class Process>
    static auto make_branch_sums(Chrono& chrono, Process& process, size_t idx)  {
        auto ret = make_dnode_array<custom_dnode<double>>(
                chrono.get_tree().nb_branches(),
                [&ch=chrono, &process, idx] (int branch) { return
                    [&old_age = ch[ch.get_tree().get_older_node(branch)], 
                     &young_age = ch[ch.get_tree().get_younger_node(branch)],
                     &old_val = process[ch.get_tree().get_older_node(branch)][idx],
                     &young_val = process[ch.get_tree().get_younger_node(branch)][idx]] ()   {
                        double exp_old_val = old_val > 10 ? 100 : (old_val < -10 ? 1e-3 : exp(old_val));
                        double exp_young_val = young_val > 10 ? 100 : (young_val < -10 ? 1e-3 : exp(young_val));
                        return 0.5 * (old_age - young_age) * (exp_old_val + exp_young_val);
                    };
                });
        gather(ret);
        return ret;
    }

    template<class Chrono, class Process>
    static auto make_branch_means(Chrono& chrono, Process& process, size_t idx)  {
        auto ret = make_dnode_array<custom_dnode<double>>(
                chrono.get_tree().nb_branches(),
                [&ch = chrono, &process, idx] (int branch) { return
                    [&old_val = process[ch.get_tree().get_older_node(branch)][idx],
                     &young_val = process[ch.get_tree().get_younger_node(branch)][idx]] ()   {
                        double exp_old_val = old_val > 10 ? 100 : (old_val < -10 ? 1e-3 : exp(old_val));
                        double exp_young_val = young_val > 10 ? 100 : (young_val < -10 ? 1e-3 : exp(young_val));
                        return 0.5 * (exp_old_val + exp_young_val);
                    };
                });
        gather(ret);
        return ret;
    }


};

