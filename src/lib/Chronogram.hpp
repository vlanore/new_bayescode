#pragma once
#include "tree/implem.hpp"

class Chronogram {

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

    size_t nb_nodes() const {return node_ages.size();}

    //! sample all entries from prior
    void Sample() {
        double age = RecursiveSample(GetRoot());
        Rescale(1.0 / age);
    }

    template<class Update, class LogProb> void MoveTimes(Update update, LogProb logprob)    {
        RecursiveMoveTimes(1.0, GetRoot(), update, logprob);
    }

    private:

    const Tree *GetTree() const { return tree; }
    Tree::NodeIndex GetRoot() const { return GetTree()->root(); }
    int GetNnode() const { return tree->nb_nodes(); }

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
        for (auto a : node_ages)    {
            a *= f;
        }
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
        double logprob1 = logprob(from);
        double bk = node_ages[from];
        double loghastings = LocalProposeMove(from, tuning);
        update(from);
        double logprob2 = logprob(from);

        double deltalogprob = logprob2 - logprob1 + loghastings;
        int accepted = (log(Random::Uniform()) < deltalogprob);
        if (!accepted)   {
            node_ages[from] = bk;
            update(from);
        }
        return ((double) accepted);
    }

    const Tree* tree;
    std::vector<double> node_ages;

};

static auto make_chrono(const Tree* tree)   {
    return std::make_unique<Chronogram>(tree);
}

