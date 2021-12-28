#pragma once

#include "tree/interface.hpp"

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
};

