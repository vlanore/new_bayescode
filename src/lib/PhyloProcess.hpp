#pragma once

#include <fstream>
#include <map>
#include "BranchSitePath.hpp"
// #include "PolyProcess.hpp"
#include "SequenceAlignment.hpp"
#include "SubMatrix.hpp"
#include "tree/implem.hpp"

class PolyProcess;

// PhyloProcess is a dispatcher:
// its responsibility is to create a random branch/site path
// for each branch/site pair

/**
 * \brief The core class of phylogenetic likelihood calculation and stochastic
 * mapping of substitution histories
 *
 * PhyloProcess takes as an input a tree, a sequence alignment (data), a set of
 * branch lengths, of site-specific rates and a selector of substitution
 * matrices across sites and branches. It is then responsible for organizing all
 * likelihood calculations by pruning, as stochastic mapping of substitution
 * histories. If polymorphism data is available (polyprocess is not a null pointer),
 * the likelihood of the data (number of occurrences in the population of the reference
 * and derived alleles at each site) is calculated using diffusion equations.
 */

class PhyloProcess {
  public:

    //! \brief generic constructor
    PhyloProcess(const Tree *intree, const SequenceAlignment *indata,
        std::function<double (int)> inbranchlength,
        std::function<double (int)> insiterate,
        std::function<const SubMatrix &(int, int)> insubmatrixarray,
        std::function<const std::vector<double>& (int)> inrootfreqarray,
        PolyProcess *inpolyprocess = nullptr);

    PhyloProcess(const PhyloProcess &) = delete;

    ~PhyloProcess();

    //! return log likelihood (computed using the pruning algorithm, Felsenstein
    //! 1981)
    double GetLogLikelihood() const;

    //! return log likelihood for given site
    double SiteLogLikelihood(int site) const;

    //! stochastic sampling of substitution history under current parameter
    //! configuration
    void ResampleSub();

    //! stochastic sampling of substitution history for given random fraction of
    //! sites
    double Move(double fraction);

    //! create all data structures necessary for computation
    void Unfold();

    //! delete data structures
    void Cleanup();

    //! posterior predictive resampling under current parameter configuration
    void PostPredSample(std::string name, bool rootprior = true);  // unclamped Nielsen

    //! get data from tips (after simulation) and put in into sequence alignment
    void GetLeafData(SequenceAlignment *data);

    int GetPathState(int taxon, int site) const {
        int node = reverse_taxon_table[taxon];
        auto site_leaf_path_map = pathmap[node][site];
        return site_leaf_path_map->GetFinalState();
    }

  private:
    int GetBranchIndex(int index) const {
        if (index <= 0) {
            std::cerr << "error in PhyloProcess::GetBranchIndex\n";
            std::cerr << index << '\n';
            exit(1);
        }
        return index - 1;
    }

    double GetFastLogProb() const;
    double FastSiteLogLikelihood(int site) const;

    //! return branch length for given branch, based on index of node at the tip of the branch
    double GetBranchLength(int index) const { return branchlength(GetBranchIndex(index)); }

    //! return site rate for given site (if no rates-across-sites array was given
    //! to phyloprocess, returns 1)
    double GetSiteRate(int site) const { return siterate(site); }

    //! return matrix that should be used on a given branch based on index of node at branch tip
    const SubMatrix &GetSubMatrix(int index, int site) const {
        return submatrixarray(GetBranchIndex(index), site);
    }

    const std::vector<double> &GetRootFreq(int site) const { return rootfreqarray(site); }

    const StateSpace *GetStateSpace() const { return data->GetStateSpace(); }
    const TaxonSet *GetTaxonSet() const { return data->GetTaxonSet(); }

  public:
    int GetNsite() const { return data->GetNsite(); }
    int GetNtaxa() const { return data->GetNtaxa(); }
    int GetNnode() const { return tree->nb_nodes(); }

    int GetNstate() const { return Nstate; }

  private:

    const SequenceAlignment *GetData() const { return data; }
    int GetData(int taxon, int site) const {
        if (taxon_table[taxon] == -1) {
            std::cerr << "error in taxon correspondance table\n";
            exit(1);
        }
        return data->GetState(taxon_table[taxon], site);
    }

    const Tree *GetTree() const { return tree; }
    Tree::NodeIndex GetRoot() const { return GetTree()->root(); }

    int GetMaxTrial() const { return maxtrial; }
    void SetMaxTrial(int i) { maxtrial = i; }

    void SetData(const SequenceAlignment *indata);
    void ClampData() { clampdata = true; }
    void UnclampData() { clampdata = false; }

    bool isDataCompatible(int taxon, int site, int state) const {
        return GetStateSpace()->isCompatible(GetData(taxon, site), state);
    }

    void DrawSites(double fraction);  // draw a fraction of sites which will be resampled
    void ResampleSub(int site);

  public:

    // suffstat: should be a function or lambda with signature PathSuffStat&(int, int)
    template<class SuffStatSelector>
    void AddPathSuffStat(SuffStatSelector suffstat) const {
        RecursiveAddPathSuffStat(GetRoot(), suffstat);
    }

    template<class SuffStatSelector>
    void RecursiveAddPathSuffStat(Tree::NodeIndex from, SuffStatSelector& suffstat) const    {
        LocalAddPathSuffStat(from, suffstat);
        for (auto c : tree->children(from)) { RecursiveAddPathSuffStat(c, suffstat); }
    }

    template<class SuffStatSelector>
    void LocalAddPathSuffStat(Tree::NodeIndex from, SuffStatSelector& suffstat) const {

        for (int i = 0; i < GetNsite(); i++) {
            if (missingmap[from][i] == 2) {
                suffstat(from,i).IncrementRootCount(statemap[from][i]);
            } else if (missingmap[from][i] == 1) {
                if (tree->is_root(from)) {
                    std::cerr << "error in missing map\n";
                    exit(1);
                }
                pathmap[from][i]->AddPathSuffStat(suffstat(from,i), GetBranchLength(from) * GetSiteRate(i));
            }
        }
    }

    // suffstat: should be a function or lambda with signature PoissonSuffStat&(int, int)
    template<class SuffStatSelector>
    void AddLengthSuffStat(SuffStatSelector suffstat) const {
        RecursiveAddLengthSuffStat(GetRoot(), suffstat);
    }

    template<class SuffStatSelector>
    void RecursiveAddLengthSuffStat(Tree::NodeIndex from, SuffStatSelector& suffstat) const  {
        if (!tree->is_root(from)) {
            LocalAddLengthSuffStat(from, suffstat);
        }
        for (auto c : tree->children(from)) {
            RecursiveAddLengthSuffStat(c, suffstat);
        }
    }

    template<class SuffStatSelector>
    void LocalAddLengthSuffStat(Tree::NodeIndex from, SuffStatSelector& suffstat) const  {
        int index = GetBranchIndex(from);
        for (int i = 0; i < GetNsite(); i++) {
            if (missingmap[from][i] == 1) {
                pathmap[from][i]->AddLengthSuffStat(suffstat(index,i), GetSiteRate(i), GetSubMatrix(from, i));
            }
        }
    }

    // suffstat: should be a function or lambda with signature PoissonSuffStat&(int, int)
    template<class SuffStatSelector>
    void AddRateSuffStat(SuffStatSelector& suffstat) const {
        RecursiveAddRateSuffStat(GetRoot(), suffstat);
    }

    template<class SuffStatSelector>
    void RecursiveAddRateSuffStat(Tree::NodeIndex from, SuffStatSelector& suffstat) const    {
        if (!tree->is_root(from)) {
            LocalAddRateSuffStat(from, suffstat); 
        }
        for (auto c : tree->children(from)) {
            RecursiveAddRateSuffStat(c, suffstat); 
        }
    }

    template<class SuffStatSelector>
    void LocalAddRateSuffStat(Tree::NodeIndex from, SuffStatSelector& suffstat) const    {
        double length = GetBranchLength(from);
        int index = GetBranchIndex(from);
        for (int i = 0; i < GetNsite(); i++) {
            if (missingmap[from][i] == 1) {
                pathmap[from][i]->AddLengthSuffStat(suffstat(index,i), length, GetSubMatrix(from, i));
            }
        }
    }

    /*
    void AddPolySuffStat(PolySuffStat &suffstat) const;
    void AddPolySuffStat(Array<PolySuffStat> &suffstatarray) const;
    */

  private:

    void PostPredSample(int site, bool rootprior = false);
    // rootprior == true : root state drawn from stationary probability of the
    // process
    // rootprior == false: root state drawn from posterior distribution

    // various accessors

    /*
    bool isMissing(Tree::NodeIndex node, int site) const { return false; }
    bool isMissing(const Link *link, int site) const {
        return false;
        // return (missingmap[link->GetNode()][site] ||
        // missingmap[link->Out()->GetNode()][site]);
    }
    */

    void CreateMissingMap();
    void DeleteMissingMap();
    void RecursiveCreateMissingMap(Tree::NodeIndex from);
    void FillMissingMap();
    void BackwardFillMissingMap(Tree::NodeIndex from);
    void ForwardFillMissingMap(Tree::NodeIndex from, Tree::NodeIndex up);

    void RecursiveCreate(Tree::NodeIndex from);
    void RecursiveDelete(Tree::NodeIndex from);

    void RecursiveCreateTBL(Tree::NodeIndex from);
    void RecursiveDeleteTBL(Tree::NodeIndex from);

    void Pruning(Tree::NodeIndex from, int site) const;
    void ResampleSub(Tree::NodeIndex from, int site);
    void ResampleState();
    void ResampleState(int site);
    void PruningAncestral(Tree::NodeIndex from, int site);
    void PriorSample(Tree::NodeIndex from, int site, bool rootprior);
    void PriorSample();
    void RootPosteriorDraw(int site);

    // borrowed from phylobayes
    // where should that be?
    BranchSitePath *SamplePath(
        int stateup, int statedown, double time, double rate, const SubMatrix &matrix);
    BranchSitePath *SampleRootPath(int rootstate);
    BranchSitePath *ResampleAcceptReject(int maxtrial, int stateup, int statedown, double rate,
        double totaltime, const SubMatrix &matrix);
    BranchSitePath *ResampleUniformized(
        int stateup, int statedown, double rate, double totaltime, const SubMatrix &matrix);

    const Tree *tree;
    const SequenceAlignment *data;
    std::vector<int> taxon_table;
    std::vector<int> reverse_taxon_table;

    std::function<double (int)> branchlength;
    std::function<double (int)> siterate{[](int) { return 1.0; }};
    std::function<const SubMatrix &(int, int)> submatrixarray;
    std::function<const std::vector<double> &(int)> rootfreqarray;

    PolyProcess *polyprocess;

    int *sitearray;
    mutable double *sitelnL;

    int Nstate;

    bool clampdata;

    mutable double **uppercondlmap;
    mutable double **lowercondlmap;
    mutable BranchSitePath ***pathmap;
    int **statemap;
    int **missingmap;

    int maxtrial;
    static const int unknown = -1;

    static const int DEFAULTMAXTRIAL = 100;
};
