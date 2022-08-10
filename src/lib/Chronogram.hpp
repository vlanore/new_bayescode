#pragma once
#include "tree/implem.hpp"
#include "components/custom_tracer.hpp"

class Chronogram : public custom_tracer {

    public:

    Chronogram(const Tree *intree) : tree(intree), node_ages(intree->nb_nodes(), 0)    {
        Sample();
    }

    Chronogram(const Chronogram& from) : tree(from.tree), node_ages(from.node_ages) {}

    ~Chronogram() {}

    const double& get_age(Tree::NodeIndex node) const {
        assert((node >= 0) && (node < node_ages.size()));
        return node_ages[node];
    }

    const double& operator[](Tree::NodeIndex node) const {
        assert((node >= 0) && (node < node_ages.size()));
        return node_ages[node];
    }

    double& operator[](Tree::NodeIndex node) {
        return node_ages[node];
    }

    size_t nb_nodes() const {return node_ages.size();}
    size_t size() const {return node_ages.size();}

    //! sample all entries from prior
    void Sample() {
        double age = RecursiveSample(GetRoot());
        Rescale(1.0 / age);
    }

    // assumes branchwise update and logprob
    template<class Update, class LogProb> void MoveTimes(Update update, LogProb logprob)    {
        RecursiveMoveTimes(1.0, GetRoot(), update, logprob);
    }

    const Tree* GetTree() const { return tree; }
    const Tree& get_tree() const { return *tree; }
    Tree::NodeIndex GetRoot() const { return tree->root(); }
    int GetNnode() const { return tree->nb_nodes(); }

    double GetTotalTime() const {
        double tot = 0;
        for (size_t c=1; c<tree->nb_nodes(); c++)   {
            tot += node_ages[tree->parent(c)] - node_ages[c];
        }
        return tot;
    }

    void get_ages_from_lengths(const std::vector<double>& bl) {
        double maxage = recursive_get_ages_from_lengths(GetRoot(), bl);
        Rescale(1.0 / maxage);
    }

    void to_stream_header(std::string name, std::ostream& os) const override {
        bool cont = false;
        for (size_t node=0; node<tree->nb_nodes(); node++)  {
            if ((!tree->is_root(node)) && (!tree->is_leaf(node)))   {
                if (cont)   {
                    os << "\t";
                }
                else    {
                    cont = true;
                }
                os << "date_" << node;
            }
        }
    }
    void to_stream(std::ostream& os) const override {
        bool cont = false;
        for (size_t node=0; node<tree->nb_nodes(); node++)  {
            if ((!tree->is_root(node)) && (!tree->is_leaf(node)))   {
                if (cont)   {
                    os << '\t';
                }
                else    {
                    cont = true;
                }
                os << node_ages[node];
            }
        }
    }

    void from_stream(std::istream& is) override {
        for (size_t node=0; node<tree->nb_nodes(); node++)  {
            if (tree->is_root(node))    {
                node_ages[node] = 1.0;
            }
            else if (tree->is_leaf(node))   {
                node_ages[node] = 0;
            }
            else    {
                is >> node_ages[node];
            }
        }
    }

    private:

    double recursive_get_ages_from_lengths(Tree::NodeIndex from, const std::vector<double>& bl)   {
        double max = 0;
        std::map<int,double> submax;

        for (auto c : tree->children(from)) {
            double tmp = recursive_get_ages_from_lengths(c, bl) + bl.at(c);
            if (max < tmp)  {
                max = tmp;
            }
            submax[c] = tmp;
        }
        for (auto c : tree->children(from)) {
            recursive_rescale(c, max/submax[c]);
        }
        node_ages[from] = max;
        return max;
    }

    double RecursiveSample(Tree::NodeIndex from)  {
        double max = 0;
        for (auto c : tree->children(from)) {
            double tmp = RecursiveSample(c);
            if (max < tmp)  {
                max = tmp;
            }
        }
        double age = max;
        if (! tree->is_leaf(from))    {
           age += Random::GammaSample(1.0, 1.0);
        }
        node_ages[from] = age;
        return age;
    }

    void Rescale(double f)  {
        for (auto& a : node_ages)    {
            a *= f;
        }
    }

    void recursive_rescale(Tree::NodeIndex from, double f)    {
        for (auto c : tree->children(from)) {
            recursive_rescale(c,f);
        }
        node_ages[from] *= f;
    }

    double GetDeltaTime(Tree::NodeIndex from) const {
        if (tree->is_root(from))    {
            return 0;
        }
        return node_ages[tree->parent(from)] - node_ages[from];
    }

    double LocalProposeMove(Tree::NodeIndex from, double tuning)  {
        if (tree->is_root(from))    {
            std::cerr << "error in chronogram: move proposed on root node\n";
            exit(1);
        }
        if (tree->is_leaf(from))    {
            std::cerr << "error in chronogram: move proposed on leaf node\n";
            exit(1);
        }
        double t = node_ages[from];
        double max = node_ages[tree->parent(from)];
        double min = 0;
        for (auto c : tree->children(from)) {
            double tmp = node_ages[c];
            if (min < tmp)  {
                min = tmp;
            }
        }
        t += tuning * (max-min) * (Random::Uniform() - 0.5);
        while ((t < min) || (t > max))  {
            if (t < min)    {
                t = 2*min - t;
            }
            if (t > max)    {
                t = 2*max - t;
            }
        }
        node_ages[from] = t;
        return 0;
    }

    template<class Update, class LogProb> void RecursiveMoveTimes(double tuning, Tree::NodeIndex from, Update update, LogProb logprob)    {
        if ((! tree->is_root(from)) && (! tree->is_leaf(from))) {
            LocalMoveTime(tuning, from, update, logprob);
        }
        for (auto c : tree->children(from)) {
            RecursiveMoveTimes(tuning, c, update, logprob);
        }
        if ((! tree->is_root(from)) && (! tree->is_leaf(from))) {
            LocalMoveTime(tuning, from, update, logprob);
        }
    }

    template<class Update, class LogProb> double LocalMoveTime(double tuning, Tree::NodeIndex from, Update update, LogProb logprob) {

        double logprob1 = logprob(tree->get_branch(from));
        if (std::isnan(logprob1))  {
            std::cerr << "nan up\n";
            std::cerr << from << '\t' << tree->get_branch(from) << '\n';
            exit(1);
        }
        for (auto c : tree->children(from)) {
            logprob1 += logprob(tree->get_branch(c));
            if (std::isnan(logprob1))  {
                std::cerr << "nan down\n";
                exit(1);
            }
        }
        double bk = node_ages[from];
        double loghastings = LocalProposeMove(from, tuning);
        update(tree->get_branch(from));
        for (auto c : tree->children(from)) {
            update(tree->get_branch(c));
        }
        double logprob2 = logprob(tree->get_branch(from));
        for (auto c : tree->children(from)) {
            logprob2 += logprob(tree->get_branch(c));
        }

        double deltalogprob = logprob2 - logprob1 + loghastings;
        if (std::isnan(deltalogprob))   {
            std::cerr << "nan log prob in move time\n";
            std::cerr << loghastings << '\t' << logprob1 << '\t' << logprob2 << '\n';
            std::cerr << from << '\t' << tree->parent(from) << '\n';
            std::cerr << bk << "  ->  " << node_ages[from] << '\n';
            std::cerr << node_ages[tree->parent(from)];
            for (auto c : tree->children(from)) {
                std::cerr << '\t' << node_ages[c];
            }
            std::cerr << '\n';
            exit(1);
        }
        if (std::isinf(deltalogprob))   {
            std::cerr << "inf log prob in move time\n";
            exit(1);
        }
        int accepted = (log(Random::Uniform()) < deltalogprob);
        if (!accepted)   {
            node_ages[from] = bk;
            update(tree->get_branch(from));
            for (auto c : tree->children(from)) {
                update(tree->get_branch(c));
            }
        }
        return ((double) accepted);
    }

    const Tree* tree;
    std::vector<double> node_ages;

};

