#pragma once

#include "tree/interface.hpp"

template<class Lambda1, class Lambda2>
auto sum_of_lambdas(Lambda1 lambda1, Lambda2 lambda2)   {
    return [lambda1, lambda2] (auto x) {return lambda1(x) + lambda2(x);};
}

namespace tree_factory {

    template<class Lambda>
    static auto sum_around_node(const Tree& tree, Lambda lambda)    {
        return [&tree, lambda] (int node)   {
            double total = 0;
            if (! tree.is_root(node))  {
                total += lambda(tree.get_branch(node));
            }
            for (auto c : tree.children(node))  {
                total += lambda(tree.get_branch(c));
            }
            return total;
        };
    }

    template<class Lambda>
    static auto do_around_node(const Tree& tree, Lambda lambda)    {
        return [&tree, lambda] (int node)   {
            if (! tree.is_root(node))  {
                lambda(tree.get_branch(node));
            }
            for (auto c : tree.children(node))  {
                lambda(tree.get_branch(c));
            }
        };
    }

    template<class SuffStat, class Lambda>
        static auto suffstat_logprob(Proxy<SuffStat>& suffstat, Lambda lambda)  {
            return [&suffstat, lambda] () {return suffstat.get().GetLogProb(lambda());};
        }

    template<class SuffStat, class Lambda1, class Lambda2>
        static auto suffstat_logprob(Proxy<SuffStat>& suffstat, Lambda1 lambda1, Lambda2 lambda2)   {
            return [&suffstat, lambda1, lambda2] () {return suffstat.get().GetLogProb(lambda1(), lambda2());};
        }

    template<class SuffStat, class Lambda>
        static auto suffstat_logprob(Proxy<SuffStat, size_t>& suffstat, Lambda lambda)  {
            return [&suffstat, lambda] (size_t i) {return suffstat.get(i).GetLogProb(lambda(i));};
        }

    template<class SuffStat, class Lambda1, class Lambda2>
        static auto suffstat_logprob(Proxy<SuffStat, size_t>& suffstat, Lambda1 lambda1, Lambda2 lambda2)   {
            return [&suffstat, lambda1, lambda2] (size_t i) {return suffstat.get(i).GetLogProb(lambda1(i), lambda2(i));};
        }
};

