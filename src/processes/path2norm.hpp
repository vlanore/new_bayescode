
#include "lib/dSOmegaPathSuffStat.hpp"

struct path2norm    {

    // count ~ Poisson(beta * l)
    // mean = beta*l
    // mean rate = count / (beta*l)
    static void syn2norm(normal_mean_condL::L& condl, 
            const dSOmegaPathSuffStat& dsomss, double l, double tau0)   {

        get<0>(condl) = 0;
        get<1>(condl) = log((1.0 + dsomss.GetSynCount()) / (1.0 + dsomss.GetSynBeta() * l));
        get<2>(condl) = tau0;
    }

    template<class Chrono, class FromSS>
    static auto make_branch_normss(Chrono& chrono, FromSS& from, double tau0)    {
        auto ret = make_dnode_array<custom_dnode<normal_mean_condL::L>>(
                chrono.get_tree().nb_branches(),
                [&from, &ch=chrono, tau0] (int branch) { return
                    [&dsomss = from.get(branch), tau0,
                     &old_age = ch[ch.get_tree().get_older_node(branch)], 
                     &young_age = ch[ch.get_tree().get_younger_node(branch)] ]
                        (normal_mean_condL::L& condl)    {
                            syn2norm(condl, dsomss, old_age - young_age, tau0);
                    };
                });
        gather(ret);
        return ret;
    }
};
