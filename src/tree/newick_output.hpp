
struct newick_output    {

    template<class NodeVal, class BranchLength>
    static void recursive_print(std::ostream& os, const Tree* tree, int node, NodeVal nodeval, BranchLength branchlength)  {
        if (tree->is_leaf(node))    {
            os << tree->node_name(node);
            os << "_";
            os << nodeval(node);
        }
        else    {
            os << "(";
            auto n = tree->children(node).size();
            size_t i = 0;
            for (auto c : tree->children(node)) {
                recursive_print(os, tree, c, nodeval, branchlength);
                i++;
                if (i != n) {
                    os << ",";
                }
            }
            os << ")";
            os << nodeval(node);
        }
        if (! tree->is_root(node))  {
            os << ":";
            os << branchlength(tree->get_branch(node));
        }
    }

    template<class NodeVal, class BranchLength>
    static void print(std::ostream& os, const Tree* tree, NodeVal nodeval, BranchLength branchlength)  {
        recursive_print(os, tree, tree->root(), nodeval, branchlength);
        os << ";\n";
    }

};

