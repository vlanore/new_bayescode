#pragma once

#include "CovMatrix.hpp"
#include "ContinuousData.hpp"

class MultivariateBrownianTreeProcess   {

    public:

    MultivariateBrownianTreeProcess(const Tree* intree, std::function<const double& (int)> innode_age, std::function<const CovMatrix& ()> insigma, std::function<const std::vector<double>& ()> inrootmean, std::function<const std::vector<double>& ()> inrootvar) :
        tree(intree),
        node_age(innode_age),
        sigma(insigma),
        rootmean(inrootmean),
        rootvar(inrootvar),
        value(intree->nb_nodes(), std::vector<double>(insigma().size(),0)),
        clamp(intree->nb_nodes(), std::vector<bool>(insigma().size(),false))  {
            Sample();
    }

    const Tree *GetTree() const { return tree; }
    Tree::NodeIndex GetRoot() const { return tree->root(); }
    int GetNnode() const { return tree->nb_nodes(); }

    size_t GetDim() const {
        return sigma().size();
    }

    const std::vector<double>& operator[](int node) {
        return value[node];
    }

    void SetAndClamp(const ContinuousData& data, int index, int fromindex)  {
        int k = 0;
        int n = 0;
        std::vector<int> taxon_table = data.GetTaxonSet()->get_index_table(tree);
        RecursiveSetAndClamp(GetRoot(), data, taxon_table, index, fromindex, k, n);
        std::cerr << data.GetCharacterName(fromindex) << " : " << n-k << " out of " << n << " missing\n";
    }

    void RecursiveSetAndClamp(Tree::NodeIndex from, const ContinuousData& data, const std::vector<int>& taxon_table, int index, int fromindex, int& k, int& n)   {

        if (tree->is_leaf(from))    {
			n++;
			int tax = taxon_table[from];
			if (tax != -1)	{
				double tmp = data.GetState(tax, fromindex);
				if (tmp != -1)	{
					k++;
                    value[from][index] = log(tmp);
                    clamp[from][index] = true;
				}
			}
			else	{
                std::cerr << "set and clamp : " << from << " not found\n";
			}
		}
        for (auto c : tree->children(from)) {
			RecursiveSetAndClamp(c, data, taxon_table, index, fromindex, k, n);
		}
	}

    void Shift(int index, double delta) {
        for (int i=0; i<GetNnode(); i++)   {
            if (! clamp[i][index])  {
                value[i][index] += delta;
            }
        }
    }

    void GetContrast(Tree::NodeIndex from, std::vector<double>& contrast) const {
            double dt = node_age(tree->parent(from)) - node_age(from);
            assert(dt > 0);
            if (dt <= 0)    {
                std::cerr << "error: negative time in chronogram\n";
                exit(1);
            }
            double scaling = sqrt(dt);
            const std::vector<double>& up = value[from];
            const std::vector<double>& down = value[tree->parent(from)];
            for (size_t i=0; i<GetDim(); i++)  {
                contrast[i] += (up[i] - down[i]) / scaling;
            }
    }

    void PseudoSample(double scale) {
        for (size_t i=0; i<tree->nb_nodes(); i++)   {
            for (size_t j=0; j<GetDim(); j++)   {
                if (! clamp[i][j])  {
                    value[i][j] = scale * sqrt(rootvar().at(j)) * Random::sNormal() + rootmean().at(j);
                }
            }
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
            const std::vector<bool>& cl = clamp[from];
            std::vector<double>& val = value[from];
            for (size_t i=0; i<GetDim(); i++)  {
                if (! cl[i])    {
                    val[i] = sqrt(rootvar().at(i)) * Random::sNormal() + rootmean().at(i);
                }
            }
        }
        else    {
            double dt = node_age(tree->parent(from)) - node_age(from);
            assert(dt > 0);
            if (dt <= 0)    {
                std::cerr << "error: negative time in chronogram\n";
                exit(1);
            }
            double scaling = sqrt(dt);

            const std::vector<double>& initval = value[tree->parent(from)];
            std::vector<double>& finalval = value[from];
            const std::vector<bool>& cl = clamp[from];

            // draw multivariate normal from sigma
            std::vector<double> contrast(GetDim(), 0);
            sigma().MultivariateNormalSample(contrast);

            // not conditional on clamped entries
            for (size_t i=0; i<GetDim(); i++)  {
                if (! cl[i])    {
                    finalval[i] = initval[i] + scaling*contrast[i];
                }
            }
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
            const std::vector<double>& val = value[from];
            double total = 0;
            for (size_t i=0; i<GetDim(); i++)  {
                double delta = val[i] - rootmean().at(i);
                total -= 0.5 * (log(2*Pi*rootvar().at(i)) + delta*delta/rootvar().at(i));
            }
            return total;
        }

        // X_down ~ Normal(X_up, sigma*dt)
        // X = (X_down - X_up)
        // Y = (X_down - X_up)/sqrt(dt)
        // P(Y)dY = p(X)dX
        // p(X) = p(Y) dY/dX = p(Y) / sqrt(dt)^GetDim()
        // log P(X) = log P(Y) - 0.5 * GetDim() * log(dt)

        double dt = node_age(tree->parent(from)) - node_age(from);
        assert(dt > 0);
        if (dt <= 0)    {
            std::cerr << "error: negative time in chronogram\n";
            exit(1);
        }
        double scaling = sqrt(dt);

        const std::vector<double>& up = value[from];
        const std::vector<double>& down = value[tree->parent(from)];

        std::vector<double> contrast(GetDim(), 0);
        for (size_t i=0; i<GetDim(); i++)  {
            contrast[i] = (up[i] - down[i])/scaling;
        }
        return sigma().logMultivariateNormalDensity(contrast) - 0.5*GetDim()*log(dt);
    }

    double GetNodeLogProb(Tree::NodeIndex from) const   {
        double total = GetLocalLogProb(from);
        for (auto c : tree->children(from)) {
            total += GetLocalLogProb(c);
        }
        return total;
    }

    void GetSampleCovarianceMatrix(CovMatrix& covmat, int& n) const    {
        RecursiveGetSampleCovarianceMatrix(GetRoot(), covmat, n);
    }

    void RecursiveGetSampleCovarianceMatrix(Tree::NodeIndex from, CovMatrix& covmat, int& n) const  {

        if (! tree->is_root(from))   {
            std::vector<double> contrast(GetDim(), 0);
            GetContrast(from, contrast);
            for (size_t i=0; i<GetDim(); i++)  {
                for (size_t j=0; j<GetDim(); j++)  {
                    covmat.add(i, j, contrast[i]*contrast[j]);
                }
            }
            n++;
        }
        for (auto c : tree->children(from)) {
            RecursiveGetSampleCovarianceMatrix(c, covmat, n);
        }
    }

    void GetSumOfContrasts(std::vector<double>& sum) const    {
        return RecursiveSumOfContrasts(GetRoot(), sum);
    }

    void RecursiveSumOfContrasts(Tree::NodeIndex from, std::vector<double>& sum) const {
        if (! tree->is_root(from))  {
            std::vector<double> contrast(GetDim(), 0);
            GetContrast(from, contrast);
            for (size_t i=0; i<GetDim(); i++)  {
                sum[i] += contrast[i];
            }
        }
        for (auto c : tree->children(from)) {
            RecursiveSumOfContrasts(c, sum);
        }
    }

    template<class Update, class LogProb>
    void SingleNodeMove(double tuning, Update update, LogProb logprob) {
        for (size_t i=0; i<GetDim(); i++)   {
            SingleNodeMove(i, tuning, update, logprob);
        }
    }

    template<class Update, class LogProb> 
    void SingleNodeMove(int index, double tuning, Update update, LogProb logprob)   {
        RecursiveSingleNodeMove(index, tuning, GetRoot(), update, logprob);
    }

    template<class Update, class LogProb> 
    void RecursiveSingleNodeMove(int index, double tuning, Tree::NodeIndex from, Update update, LogProb logprob)    {

        if (! clamp[from][index])    {
            LocalSingleNodeMove(index, tuning, from, update, logprob);
        }
        for (auto c : tree->children(from)) {
            RecursiveSingleNodeMove(index, tuning, c, update, logprob);
        }
        if (! clamp[from][index])    {
            LocalSingleNodeMove(index, tuning, from, update, logprob);
        }
    }

    template<class Update, class LogProb> double LocalSingleNodeMove(int index, double tuning, Tree::NodeIndex from, Update update, LogProb logprob) {
        double logprob1 = GetNodeLogProb(from) + logprob(from);
        double delta = tuning * (Random::Uniform() - 0.5);
        value[from][index] += delta;
        update(from);
        double logprob2 = GetNodeLogProb(from) + logprob(from);

        double deltalogprob = logprob2 - logprob1;
        int accepted = (log(Random::Uniform()) < deltalogprob);
        if (!accepted)   {
            value[from][index] -= delta;
            update(from);
        }
        return ((double) accepted);
    }

    private:

    const Tree* tree;
    std::function<const double& (int)> node_age;
    std::function<const CovMatrix& ()> sigma;
    std::function<const std::vector<double>& ()> rootmean;
    std::function<const std::vector<double>& ()> rootvar;
    std::vector<std::vector<double>> value;
    std::vector<std::vector<bool>> clamp;
};

static auto make_brownian_tree_process(const Tree* intree, std::function<const double& (int)> innode_age, std::function<const CovMatrix& ()> insigma, std::function<const std::vector<double>& ()> inrootmean, std::function<const std::vector<double>& ()> inrootvar)  {
    return std::make_unique<MultivariateBrownianTreeProcess>(intree, innode_age, insigma, inrootmean, inrootvar);
}

class MVBranchExpoLengthArray   {

    public:

    MVBranchExpoLengthArray(const Tree* intree, std::function<const std::vector<double>& (int)> innodetree, std::function<const double& (int)> innode_age, int inidx) :
        tree(intree),
        nodetree(innodetree),
        node_age(innode_age),
        idx(inidx),
        value(intree->nb_nodes()-1,0)  {
            Update();
    }

    const Tree *GetTree() const { return tree; }
    Tree::NodeIndex GetRoot() const { return tree->root(); }
    size_t nb_branches() const { return tree->nb_nodes() - 1; }

    int branch_index(int index) const {
        if (index <= 0) {
            std::cerr << "error in Chronogram::GetBranchIndex\n";
            std::cerr << index << '\n';
            exit(1);
        }
        return index - 1;
    }

    const double& operator[](int branch) {
        return value[branch];
    }

    void Update()   {
        RecursiveUpdate(GetRoot());
    }

    void RecursiveUpdate(Tree::NodeIndex from)  {
        LocalUpdate(from);
        for (auto c : tree->children(from)) {
            RecursiveUpdate(c);
        }
    }

    void LocalUpdate(Tree::NodeIndex from)  {
        if (! tree->is_root(from))  {
            double up = nodetree(from).at(idx);
            double down = nodetree(tree->parent(from)).at(idx);
            double mean = 0.5 * (exp(up) + exp(down));
            // double mean = (exp(up) - exp(down)) / (up - down);
            double dt = node_age(tree->parent(from)) - node_age(from);
            assert(dt > 0);
            if (dt <= 0)    {
                std::cerr << "error: negative time in chronogram\n";
                exit(1);
            }
            value[branch_index(from)] = mean * dt;
        }
    }

    void LocalNodeUpdate(Tree::NodeIndex from)  {
        LocalUpdate(from);
        for (auto c : tree->children(from)) {
            LocalUpdate(c);
        }
    }

    private:
    const Tree* tree;
    std::function<const std::vector<double>& (int)> nodetree;
    std::function<const double& (int)> node_age;
    int idx;
    std::vector<double> value;
};

static auto make_branch_lengths(const Tree* intree, std::function<const std::vector<double>& (int)> innodetree, std::function<const double& (int)> innode_age, int inidx) {
    return std::make_unique<MVBranchExpoLengthArray>(intree, innodetree, innode_age, inidx);
}

class MVBranchExpoMeanArray {

    public:

    MVBranchExpoMeanArray(const Tree* intree, std::function<const std::vector<double>& (int)> innodetree, int inidx) :
        tree(intree),
        nodetree(innodetree),
        idx(inidx),
        value(intree->nb_nodes()-1, 0)  {
            Update();
    }

    const Tree *GetTree() const { return tree; }
    Tree::NodeIndex GetRoot() const { return tree->root(); }
    size_t nb_branches() const { return tree->nb_nodes() - 1; }

    int branch_index(int index) const {
        if (index <= 0) {
            std::cerr << "error in Chronogram::GetBranchIndex\n";
            std::cerr << index << '\n';
            exit(1);
        }
        return index - 1;
    }

    const double& operator[](int branch) {
        return value[branch];
    }

    void Update()   {
        RecursiveUpdate(GetRoot());
    }

    void RecursiveUpdate(Tree::NodeIndex from)  {
        LocalUpdate(from);
        for (auto c : tree->children(from)) {
            RecursiveUpdate(c);
        }
    }

    void LocalUpdate(Tree::NodeIndex from)  {
        if (! tree->is_root(from))  {
            double up = nodetree(from).at(idx);
            double down = nodetree(tree->parent(from)).at(idx);
            double mean = 0.5 * (exp(up) + exp(down));
            // double mean = (exp(up) - exp(down)) / (up - down);
            value[branch_index(from)] = mean;
        }
    }

    void LocalNodeUpdate(Tree::NodeIndex from)  {
        LocalUpdate(from);
        for (auto c : tree->children(from)) {
            LocalUpdate(c);
        }
    }

    private:
    const Tree* tree;
    std::function<const std::vector<double>& (int)> nodetree;
    int idx;
    std::vector<double> value;
};

static auto make_branch_means(const Tree* intree, std::function<const std::vector<double>& (int)> innodetree, int inidx) {
    return std::make_unique<MVBranchExpoMeanArray>(intree, innodetree, inidx);
}

