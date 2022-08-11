
struct split_tree   {

    template<class Tree, class Gen>
    static void choose_components(const Tree& tree, int min, int max, 
            std::vector<bool>& clamps, Gen& gen)  {
        recursive_choose_components(tree, tree.root(), tree.nb_nodes(), min, max, clamps, gen);
    }

    template<class Tree, class Gen>
    static void recursive_choose_components(const Tree& tree, int node, int size, int min, int max, 
            std::vector<bool>& clamps, Gen& gen)    {

        if (! tree.is_root(node))   {
            clamps[node] = false;
        }
        std::pair<int,int> p = split(tree, node, min, clamps, gen);
        if (! tree.is_root(node))   {
            clamps[node] = true;
        }
        if (p.second > max)  {
            recursive_choose_components(tree, p.first, p.second, min, max, clamps, gen);
        }
        if (size - p.second > max)  {
            recursive_choose_components(tree, node, size-p.second+1, min, max, clamps, gen);
        }
    }
    
    template<class Tree, class Gen>
    static std::pair<int,int> split(const Tree& tree, size_t node, int min, 
            std::vector<bool>& clamps, Gen& gen)    {

        if (clamps[node])  {
            std::cerr << "error in split: node already clamped\n";
            exit(1);
        }
        std::vector<int> node_sizes(tree.nb_nodes(),0);
        get_subtree_sizes(tree, node, clamps, node_sizes);
        int count = 0;
        for (size_t c=0; c<tree.nb_nodes(); c++)    {
            if ((c != node) && (node_sizes[c] > min))   {
                count++;
            }
        }
        if (! count)    {
            std::cerr << "error in split: no eligible node was found\n";
            exit(1);
        }
        int choose = 1 + int(draw_uniform(gen) * count);
        size_t c=0;
        while (choose)  {
            c++;
            if (c == tree.nb_nodes())   {
                std::cerr << "in split: overflow\n";
                exit(1);
            }
            if ((c != node) && (node_sizes[c] > min))   {
                choose--;
            }
        }
        if (clamps[c]) {
            std::cerr << "error in split: already clamped\n";
            exit(1);
        }
        if (node_sizes[c] <= min)   {
            std::cerr << "error in split: size too small\n";
            exit(1);
        }
        clamps[c] = true;
        return std::make_pair(c,node_sizes[c]);
    }

    template<class Tree>
    static int get_subtree_sizes(const Tree& tree, int node, 
            std::vector<bool>& clamps, std::vector<int>& node_sizes)    {

        if (clamps[node])  {
            node_sizes[node] = 1;
        }
        else    {
            int s = 0;
            for (auto c : tree.children(node))  {
                s += get_subtree_sizes(tree, c, clamps, node_sizes);
            }
            s++;
            node_sizes[node] = s;
        }
        return node_sizes[node];
    }
};

