#pragma once

class UnivariateBrownianTreeProcess   {

    public:

    UnivariateBrownianTreeProcess(const Tree* intree, std::function<const double& (int)> innode_age, std::function<double ()> intau, std::function<double ()> inrootmean, std::function<double ()> inrootvar) :
        tree(intree),
        node_age(innode_age),
        tau(intau),
        rootmean(inrootmean),
        rootvar(inrootvar),
        value(intree->nb_nodes(), 0)    {
            Sample();
    }

    const Tree *GetTree() const { return tree; }
    Tree::NodeIndex GetRoot() const { return tree->root(); }
    int GetNnode() const { return tree->nb_nodes(); }

    const double& operator[](int node) {
        return value[node];
    }

    void Shift(double delta) {
        for (int i=0; i<GetNnode(); i++)   {
            value[i] += delta;
        }
    }

    double GetContrast(Tree::NodeIndex from) const  {
        double dt = node_age(tree->parent(from)) - node_age(from);
        assert(dt > 0);
        if (dt <= 0)    {
            std::cerr << "error: negative time in chronogram\n";
            exit(1);
        }
        double scaling = sqrt(dt);
        return (value[from] - value[tree->parent(from)]) / scaling;
    }

    void PseudoSample(double scale) {
        for (size_t i=0; i<tree->nb_nodes(); i++)   {
            value[i] = scale * sqrt(rootvar()) * Random::sNormal() + rootmean();
        }
    }

    void Sample()   {
        RecursiveSample(GetRoot());
    }

    void RecursiveSample(Tree::NodeIndex from)  {
        LocalSample(from);
        for (auto c : tree->children(from)) {
            RecursiveSample(c);
        }
    }

    void LocalSample(Tree::NodeIndex from)  {
        if (tree->is_root(from))    {
            value[from] = sqrt(rootvar()) * Random::sNormal() + rootmean();
        }
        else    {
            double dt = node_age(tree->parent(from)) - node_age(from);
            assert(dt > 0);
            if (dt <= 0)    {
                std::cerr << "error: negative time in chronogram\n";
                exit(1);
            }
            double scaling = sqrt(dt);
            value[from] = value[tree->parent(from)] + Random::sNormal()*scaling/sqrt(tau());
        }
    }

    double GetLogProb() const {
        return RecursiveGetLogProb(GetRoot());
    }

    double RecursiveGetLogProb(Tree::NodeIndex from) const  {
        double total = GetLocalLogProb(from);
        for (auto c : tree->children(from)) {
            total += RecursiveGetLogProb(c);
        }
        return total;
    }

    double GetLocalLogProb(Tree::NodeIndex from) const  {

        if (tree->is_root(from))    {
            double val = value[from];
            double delta = val - rootmean();
            return - 0.5 * (log(2*Pi*rootvar()) + delta*delta/rootvar());
        }

        double dt = node_age(tree->parent(from)) - node_age(from);
        assert(dt > 0);
        if (dt <= 0)    {
            std::cerr << "error: negative time in chronogram\n";
            exit(1);
        }
        double delta = value[from] - value[tree->parent(from)];
        return -0.5*(log(2*Pi*dt/tau()) + tau()*delta*delta/dt);
    }

    double GetNodeLogProb(Tree::NodeIndex from) const   {
        double total = GetLocalLogProb(from);
        for (auto c : tree->children(from)) {
            total += GetLocalLogProb(c);
        }
        return total;
    }

    // log p(tau) = (alpha-1)*log(tau) - beta*tau
    // log p(X | tau) = 0.5*(log(tau) - tau*(X_i-X_p_i)^2/t_i)
    // log p(tau | X) = (alpha + 0.5*Nbranches - 1)*log(tau) - (beta + 0.5 * sum squared_constrasts)

    void GetSampleVariance(double& var, int& n) const    {
        RecursiveGetSampleVariance(GetRoot(), var, n);
    }

    void RecursiveGetSampleVariance(Tree::NodeIndex from, double& var, int& n) const  {

        if (! tree->is_root(from))   {
            double delta = value[from] - value[tree->parent(from)];
            double dt = node_age(tree->parent(from)) - node_age(from);
            var += delta*delta/dt;
            n++;
        }
        for (auto c : tree->children(from)) {
            RecursiveGetSampleVariance(c, var, n);
        }
    }

    template<class Update, class LogProb>
    void SingleNodeMove(double tuning, Update update, LogProb logprob) {
        RecursiveSingleNodeMove(tuning, GetRoot(), update, logprob);
    }

    template<class Update, class LogProb> 
    void RecursiveSingleNodeMove(double tuning, Tree::NodeIndex from, Update update, LogProb logprob)    {

        LocalSingleNodeMove(tuning, from, update, logprob);
        for (auto c : tree->children(from)) {
            RecursiveSingleNodeMove(tuning, c, update, logprob);
        }
        LocalSingleNodeMove(tuning, from, update, logprob);
    }

    template<class Update, class LogProb> double LocalSingleNodeMove(double tuning, Tree::NodeIndex from, Update update, LogProb logprob) {
        double logprob1 = GetNodeLogProb(from) + logprob(from);
        double delta = tuning * (Random::Uniform() - 0.5);
        value[from] += delta;
        update(from);
        double logprob2 = GetNodeLogProb(from) + logprob(from);

        double deltalogprob = logprob2 - logprob1;
        int accepted = (log(Random::Uniform()) < deltalogprob);
        if (!accepted)   {
            value[from] -= delta;
            update(from);
        }
        return ((double) accepted);
    }

    private:

    const Tree* tree;
    std::function<const double& (int)> node_age;
    std::function<double ()> tau;
    std::function<double ()> rootmean;
    std::function<double ()> rootvar;
    std::vector<double> value;
};

