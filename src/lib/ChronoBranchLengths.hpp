#pragma once
#include "tree/implem.hpp"

template<class NodeAges>
class ChronoBranchLengths {

    public:

    ChronoBranchLengths(const Tree* intree, NodeAges in_node_ages) : tree(intree), node_ages(in_node_ages), branch_lengths(intree->nb_nodes() - 1, 0)    {
    // ChronoBranchLengths(const Tree* intree, std::function<const double&(int)> in_node_ages) : tree(intree), node_ages(in_node_ages), branch_lengths(intree->nb_nodes() - 1, 0)    {
            Update();
    }

    ChronoBranchLengths(const ChronoBranchLengths& from) : tree(from.tree), node_ages(from.node_ages), branch_lengths(from.branch_lengths) {}

    ~ChronoBranchLengths() {}

    const double& get_length(size_t branch) const {
        assert((branch >=0) && (branch < branch_lengths.size()));
        return branch_lengths[branch];
    }

    const double& operator[](size_t branch) const {
        assert((branch >=0) && (branch < branch_lengths.size()));
        return branch_lengths[branch];
    }

    size_t nb_branches() const {return branch_lengths.size();}

    void Update()   {
        RecursiveUpdate(GetRoot());
    }

    void LocalUpdate(Tree::NodeIndex from)  {
        if (! tree->is_root(from))  {
            double tmp = node_ages(tree->parent(from)) - node_ages(from);
            if (tmp <= 0)   {
                std::cerr << "error: negative delta age : " << tmp << '\n';
                std::cerr << node_ages(tree->parent(from)) << '\t' << node_ages(from) << '\n';
                exit(1);
            }
            branch_lengths[branch_index(from)] = tmp;
        }
    }

    void LocalNodeUpdate(Tree::NodeIndex from)  {
        LocalUpdate(from);
        for (auto c : tree->children(from)) {
            LocalUpdate(c);
        }
    }

    template<class BranchLogProb>
    double sum_around_node(BranchLogProb branch_logprob, Tree::NodeIndex from)   {
        double total = 0;
        if (! tree->is_root(from))  {
            total += branch_logprob(branch_index(from));
        }
        for (auto c : tree->children(from))  {
            total += branch_logprob(branch_index(c));
        }
        return total;
    }

    template<class BranchUpdate>
    void do_around_node(BranchUpdate branch_update, Tree::NodeIndex from)   {
        if (! tree->is_root(from))  {
            branch_update(branch_index(from));
        }
        for (auto c : tree->children(from))  {
            branch_update(branch_index(c));
        }
    }

    private:

    const Tree* tree;
    NodeAges node_ages;
    // std::function<const double&(int)> node_ages;
    std::vector<double> branch_lengths;

    const Tree *GetTree() const { return tree; }
    Tree::NodeIndex GetRoot() const { return GetTree()->root(); }
    int GetNbranch() const { return tree->nb_nodes() - 1; }

    int branch_index(int index) const {
        if (index <= 0) {
            std::cerr << "error in Chronogram::GetBranchIndex\n";
            std::cerr << index << '\n';
            exit(1);
        }
        return index - 1;
    }

    void RecursiveUpdate(Tree::NodeIndex from)  {
        LocalUpdate(from);
        for (auto c : tree->children(from)) {
            RecursiveUpdate(c);
        }
    }

};

template<class NodeAges>
static auto make_chrono_branch_lengths(const Tree* tree, NodeAges in_node_ages)    {
    auto tmp = std::make_unique<ChronoBranchLengths<NodeAges>>(tree, in_node_ages);
    tmp->Update();
    return tmp;
}
