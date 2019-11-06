#include "PhyloProcess.hpp"
#include <algorithm>
#include "PathSuffStat.hpp"
// #include "PolySuffStat.hpp"
#include "bayes_utils/src/logging.hpp"

using namespace std;

// generic
PhyloProcess::PhyloProcess(const Tree *intree, const SequenceAlignment *indata,
    std::function<const double &(int)> inbranchlength,
    std::function<const double &(int)> insiterate,
    std::function<const SubMatrix &(int, int)> insubmatrixarray,
    std::function<const SubMatrix &(int)> inrootsubmatrixarray, PolyProcess *inpolyprocess)
    :

      tree(intree),
      data(indata),
      branchlength(inbranchlength),
      siterate(insiterate),
      submatrixarray(insubmatrixarray),
      rootsubmatrixarray(inrootsubmatrixarray),
      polyprocess(inpolyprocess) {
    Nstate = data->GetNstate();
    taxon_table = data->GetTaxonSet()->get_index_table(tree);
    reverse_taxon_table = data->GetTaxonSet()->get_reverse_index_table(tree);
    maxtrial = DEFAULTMAXTRIAL;
}

PhyloProcess::~PhyloProcess() { Cleanup(); }

void PhyloProcess::SetData(const SequenceAlignment *indata) { data = indata; }

void PhyloProcess::Unfold() {
    sitearray = new int[GetNsite()];
    sitelnL = new double[GetNsite()];
    for (int i = 0; i < GetNsite(); i++) { sitearray[i] = 1; }
    statemap = new int *[GetNnode()];
    uppercondlmap = new double *[GetNnode()];
    lowercondlmap = new double *[GetNnode()];
    pathmap = new BranchSitePath **[GetNnode()];

    CreateMissingMap();
    FillMissingMap();
    INFO("Recursive create");
    RecursiveCreate(GetRoot());
    INFO("Create tbl");
    RecursiveCreateTBL(GetRoot());
    INFO("Clamp data");
    ClampData();
    INFO("Clamp data ok");
}

void PhyloProcess::Cleanup() {
    DeleteMissingMap();
    RecursiveDeleteTBL(GetRoot());
    RecursiveDelete(GetRoot());
    delete[] sitearray;
    delete[] sitelnL;
}

void PhyloProcess::CreateMissingMap() {
    missingmap = new int *[GetTree()->nb_nodes()];
    for (size_t j = 0; j < GetTree()->nb_nodes(); j++) {
        missingmap[j] = new int[GetNsite()];
        for (int i = 0; i < GetNsite(); i++) { missingmap[j][i] = -1; }
    }
}

void PhyloProcess::DeleteMissingMap() {
    for (size_t j = 0; j < GetTree()->nb_nodes(); j++) { delete[] missingmap[j]; }
    delete[] missingmap;
}

void PhyloProcess::FillMissingMap() {
    /*
        for (size_t j=0; j<GetTree()->nb_nodes(); j++)	{
                for (int i=0; i<GetNsite(); i++)	{
                        missingmap[j][i] = 0;
                }
        }
    int rootindex = GetTree()->GetRoot()->GetNode()->GetIndex();
    for (int i=0; i<GetNsite(); i++)	{
        missingmap[rootindex][i] = 2;
    }
    */

    BackwardFillMissingMap(GetRoot());
    ForwardFillMissingMap(GetRoot(), GetRoot());
}

void PhyloProcess::BackwardFillMissingMap(Tree::NodeIndex from) {
    for (int i = 0; i < GetNsite(); i++) { missingmap[from][i] = 0; }
    if (tree->is_leaf(from)) {
        for (int i = 0; i < GetNsite(); i++) {
            int state = GetData(from, i);
            if (state != -1) { missingmap[from][i] = 1; }
        }
    } else {
        for (auto c : tree->children(from)) {
            BackwardFillMissingMap(c);
            for (int i = 0; i < GetNsite(); i++) {
                if (missingmap[c][i]) { missingmap[from][i]++; }
            }
        }
    }
}

void PhyloProcess::ForwardFillMissingMap(Tree::NodeIndex from, Tree::NodeIndex up) {
    if (tree->is_root(from)) {
        for (int i = 0; i < GetNsite(); i++) {
            if (missingmap[from][i] <= 1) {
                missingmap[from][i] = 0;
            } else {
                missingmap[from][i] = 2;
            }
        }
    } else {
        for (int i = 0; i < GetNsite(); i++) {
            if (missingmap[from][i] > 0) {
                if (missingmap[up][i]) {
                    missingmap[from][i] = 1;
                } else {
                    if (tree->is_leaf(from) || (missingmap[from][i] > 1)) {
                        missingmap[from][i] = 2;
                    } else {
                        missingmap[from][i] = 0;
                    }
                }
            }
        }
    }
    for (auto c : tree->children(from)) { ForwardFillMissingMap(c, from); }
}

void PhyloProcess::RecursiveCreate(Tree::NodeIndex from) {
    auto state = new int[GetNsite()];
    statemap[from] = state;

    auto array = new BranchSitePath *[GetNsite()];
    for (int i = 0; i < GetNsite(); i++) { array[i] = 0; }
    pathmap[from] = array;

    for (auto c : tree->children(from)) { RecursiveCreate(c); }
}

void PhyloProcess::RecursiveDelete(Tree::NodeIndex from) {
    for (auto c : tree->children(from)) { RecursiveDelete(c); }

    delete[] statemap[from];

    BranchSitePath **path = pathmap[from];
    for (int i = 0; i < GetNsite(); i++) { delete path[i]; }
    delete[] path;
}

void PhyloProcess::RecursiveCreateTBL(Tree::NodeIndex from) {
    uppercondlmap[from] = new double[GetNstate() + 1];
    lowercondlmap[from] = new double[GetNstate() + 1];
    for (auto c : tree->children(from)) { RecursiveCreateTBL(c); }
}

void PhyloProcess::RecursiveDeleteTBL(Tree::NodeIndex from) {
    for (auto c : tree->children(from)) { RecursiveDeleteTBL(c); }
    delete[] uppercondlmap[from];
    delete[] lowercondlmap[from];
}

double PhyloProcess::SiteLogLikelihood(int site) const {
    Pruning(GetRoot(), site);
    double ret = 0;
    double *t = uppercondlmap[GetRoot()];
    const EVector &stat = GetRootFreq(site);

    for (int k = 0; k < GetNstate(); k++) { ret += t[k] * stat[k]; }
    if (ret == 0) {
        cerr << "pruning : 0 \n";
        for (int k = 0; k < GetNstate(); k++) { cerr << t[k] << '\t' << stat[k] << '\n'; }
        exit(1);
    }
    return log(ret) + t[GetNstate()];
}

double PhyloProcess::FastSiteLogLikelihood(int site) const {
    double ret = 0;
    double *t = uppercondlmap[GetRoot()];
    const EVector &stat = GetRootFreq(site);
    for (int k = 0; k < GetNstate(); k++) { ret += t[k] * stat[k]; }
    if (ret == 0) {
        cerr << "pruning : 0 \n";
        for (int k = 0; k < GetNstate(); k++) { cerr << t[k] << '\t' << stat[k] << '\n'; }
        exit(1);
    }
    sitelnL[site] = log(ret) + t[GetNstate()];
    return sitelnL[site];
}

double PhyloProcess::GetFastLogProb() const {
    double total = 0;
    for (int i = 0; i < GetNsite(); i++) { total += sitelnL[i]; }
    return total;
}

double PhyloProcess::GetLogLikelihood() const {
    double total = 0;
    for (int i = 0; i < GetNsite(); i++) { total += SiteLogLikelihood(i); }
    return total;
}

void PhyloProcess::Pruning(Tree::NodeIndex from, int site) const {
    double *t = uppercondlmap[from];
    if (tree->is_leaf(from)) {
        int totcomp = 0;
        for (int k = 0; k < GetNstate(); k++) {
            if (polyprocess != nullptr) {
                /*
                double prob = polyprocess->GetProb(taxon_table[from], site, k);
                if (prob > 0.0) {
                    totcomp++;
                    t[k] = prob;
                } else {
                    t[k] = 0.0;
                }
                */
            } else {
                if (isDataCompatible(from, site, k)) {
                    t[k] = 1.0;
                    totcomp++;
                } else {
                    t[k] = 0.0;
                }
            }
        }
        if (totcomp == 0) {
            cerr << "error : no compatibility\n";
            cerr << GetData(from, site) << '\n';
            exit(1);
        }

        t[GetNstate()] = 0;
    } else {
        for (int k = 0; k < GetNstate(); k++) { t[k] = 1.0; }
        t[GetNstate()] = 0;
        for (auto c : tree->children(from)) {
            Pruning(c, site);
            GetSubMatrix(c, site).BackwardPropagate(
                uppercondlmap[c], lowercondlmap[c], GetBranchLength(c) * GetSiteRate(site));
            double *tbl = lowercondlmap[c];
            for (int k = 0; k < GetNstate(); k++) { t[k] *= tbl[k]; }
            t[GetNstate()] += tbl[GetNstate()];
        }
        double max = 0;
        for (int k = 0; k < GetNstate(); k++) {
            if (t[k] < 0) {
                /*
                  cerr << "error in pruning: negative prob : " << t[k] << "\n";
                  exit(1);
                */
                t[k] = 0;
            }
            if (max < t[k]) { max = t[k]; }
        }
        if (max == 0) {
            cerr << "max = 0\n";
            cerr << "error in pruning: null likelihood\n";
            if (tree->is_root(from)) { cerr << "is root\n"; }
            cerr << '\n';
            exit(1);
            max = 1e-20;
        }
        for (int k = 0; k < GetNstate(); k++) { t[k] /= max; }
        t[GetNstate()] += log(max);
    }
}

void PhyloProcess::PruningAncestral(Tree::NodeIndex from, int site) {
    if (tree->is_root(from)) {
        double aux[GetNstate()];
        double cumulaux[GetNstate()];
        try {
            double *tbl = uppercondlmap[from];
            const EVector &stat = GetRootFreq(site);
            double tot = 0;
            for (int k = 0; k < GetNstate(); k++) {
                aux[k] = stat[k] * tbl[k];
                tot += aux[k];
                cumulaux[k] = tot;
            }
            double u = tot * Random::Uniform();
            int s = 0;
            while ((s < GetNstate()) && (cumulaux[s] < u)) { s++; }
            if (s == GetNstate()) {
                cerr << "error in pruning ancestral: overflow\n";
                exit(1);
            }
            statemap[from][site] = s;
        } catch (...) {
            cerr << "in root::PruningAncestral\n";
            for (int k = 0; k < GetNstate(); k++) { cerr << aux[k] << '\n'; }
            exit(1);
            throw;
        }
    }
    for (auto c : tree->children(from)) {
        double aux[GetNstate()];
        double cumulaux[GetNstate()];
        try {
            for (int k = 0; k < GetNstate(); k++) { aux[k] = 1; }
            GetSubMatrix(c, site).GetFiniteTimeTransitionProb(
                statemap[from][site], aux, GetBranchLength(c) * GetSiteRate(site));
            double *tbl = uppercondlmap[c];
            for (int k = 0; k < GetNstate(); k++) { aux[k] *= tbl[k]; }

            // dealing with numerical problems:
            double max = 0;
            for (int k = 0; k < GetNstate(); k++) {
                if (aux[k] < 0) { aux[k] = 0; }
                if (max < aux[k]) { max = aux[k]; }
            }
            if (max == 0) {
                auto stat = GetSubMatrix(c, site).GetStationary();
                for (int k = 0; k < GetNstate(); k++) { aux[k] = stat[k]; }
            }
            // end of dealing with dirty numerical problems

            double tot = 0;
            for (int k = 0; k < GetNstate(); k++) {
                tot += aux[k];
                cumulaux[k] = tot;
            }
            double u = tot * Random::Uniform();
            int s = 0;
            while ((s < GetNstate()) && (cumulaux[s] < u)) { s++; }
            if (s == GetNstate()) {
                cerr << "error in pruning ancestral: overflow\n";
                exit(1);
            }
            statemap[c][site] = s;
        } catch (...) {
            cerr << "in internal leave::PruningAncestral\n";
            for (int k = 0; k < GetNstate(); k++) { cerr << aux[k] << '\n'; }
            exit(1);
            throw;
        }
        PruningAncestral(c, site);
    }
}

void PhyloProcess::RootPosteriorDraw(int site) {
    double aux[GetNstate()];
    double *tbl = uppercondlmap[GetRoot()];
    const EVector &stat = GetRootFreq(site);
    for (int k = 0; k < GetNstate(); k++) { aux[k] = stat[k] * tbl[k]; }
    statemap[GetRoot()][site] = Random::DrawFromDiscreteDistribution(aux, GetNstate());
}

void PhyloProcess::PriorSample(Tree::NodeIndex from, int site, bool rootprior) {
    int &state = statemap[from][site];
    if (tree->is_root(from)) {
        if (rootprior) {
            state = Random::DrawFromDiscreteDistribution(GetRootFreq(site), GetNstate());
        } else {
            RootPosteriorDraw(site);
        }
    }
    for (auto c : tree->children(from)) {
        statemap[c][site] =
            GetSubMatrix(c, site).DrawFiniteTime(state, GetBranchLength(c) * GetSiteRate(site));
        PriorSample(c, site, rootprior);
    }
}

void PhyloProcess::ResampleState() {
    for (int i = 0; i < GetNsite(); i++) { ResampleState(i); }
}

void PhyloProcess::ResampleState(int site) {
    Pruning(GetRoot(), site);
    PruningAncestral(GetRoot(), site);
    // give information about fixed states at the tips to polyprocess
}

double PhyloProcess::Move(double fraction) {
    DrawSites(fraction);
    ResampleSub();
    // restoring full resampling mode
    DrawSites(1.0);
    return 1.0;
}

void PhyloProcess::DrawSites(double fraction) {
    for (int i = 0; i < GetNsite(); i++) { sitearray[i] = (Random::Uniform() < fraction); }
}

void PhyloProcess::ResampleSub() {

    for (int i = 0; i < GetNsite(); i++) {
        if (sitearray[i] != 0) { ResampleState(i); }
    }

    for (int i = 0; i < GetNsite(); i++) {
        if (sitearray[i] != 0) { ResampleSub(GetRoot(), i); }
    }
}

void PhyloProcess::ResampleSub(int site) {
    ResampleState(site);
    ResampleSub(GetRoot(), site);
}

void PhyloProcess::ResampleSub(Tree::NodeIndex from, int site) {
    if (tree->is_root(from)) {
        delete pathmap[from][site];
        pathmap[from][site] = SampleRootPath(statemap[from][site]);
    }
    for (auto c : tree->children(from)) {
        delete pathmap[c][site];
        pathmap[c][site] = SamplePath(statemap[from][site], statemap[c][site], GetBranchLength(c),
            GetSiteRate(site), GetSubMatrix(c, site));
        ResampleSub(c, site);
    }
}

void PhyloProcess::PostPredSample(string name, bool rootprior) {
    for (int i = 0; i < GetNsite(); i++) { PostPredSample(i, rootprior); }
    SequenceAlignment tmpdata(*GetData());
    GetLeafData(&tmpdata);
    ofstream os(name.c_str());
    tmpdata.ToStream(os);
    os.close();
}

void PhyloProcess::PostPredSample(int site, bool rootprior) {
    if (!rootprior) { Pruning(GetRoot(), site); }
    PriorSample(GetRoot(), site, rootprior);
}

void PhyloProcess::GetLeafData(SequenceAlignment *data) {
    for (size_t node = 0; node < tree->nb_nodes(); node++) {
        if (tree->is_leaf(node)) {
            int tax = taxon_table[node];
            for (int site = 0; site < GetNsite(); site++) {
                int state = statemap[tax][site];
                int obsstate = GetData(tax, site);
                if (obsstate != unknown) {
                    data->SetState(tax, site, state);
                } else {
                    data->SetState(tax, site, unknown);
                }
            }
        }
    }
}

BranchSitePath *PhyloProcess::SampleRootPath(int rootstate) {
    BranchSitePath *path = new BranchSitePath(rootstate);
    return path;
}

BranchSitePath *PhyloProcess::SamplePath(
    int stateup, int statedown, double time, double rate, const SubMatrix &matrix) {
    BranchSitePath *path = ResampleAcceptReject(1000, stateup, statedown, rate, time, matrix);
    if (!path) { path = ResampleUniformized(stateup, statedown, rate, time, matrix); }
    return path;
}

BranchSitePath *PhyloProcess::ResampleAcceptReject(int maxtrial, int stateup, int statedown,
    double rate, double totaltime, const SubMatrix &matrix) {
    int ntrial = 0;
    BranchSitePath *path = 0;

    if (rate * totaltime < 1e-10) {
        // if (rate * totaltime == 0)	{
        if (stateup != statedown) {
            cerr << "error in MatrixSubstitutionProcess::ResampleAcceptReject: "
                    "stateup != statedown, efflength == 0\n";
            exit(1);
        }
        delete path;
        path = new BranchSitePath();
        ntrial++;
        path->Reset(stateup);
    } else {
        do {
            delete path;
            path = new BranchSitePath();
            ntrial++;
            path->Reset(stateup);
            double t = 0;
            int state = stateup;

            if (state != statedown) {
                // draw waiting time conditional on at least one substitution
                double q = -rate * matrix(state, state);
                double u = -log(1 - Random::Uniform() * (1 - exp(-q * totaltime))) / q;

                t += u;
                int newstate = matrix.DrawOneStep(state);
                path->Append(newstate, u / totaltime);
                state = newstate;
            }
            while (t < totaltime) {
                // draw waiting time
                double q = -rate * matrix(state, state);
                double u = -log(1 - Random::Uniform()) / q;
                if (std::isnan(u)) {
                    cerr << "in MatrixSubstitutionProcess:: drawing exponential number: "
                            "nan\n";
                    cerr << rate << '\t' << q << '\n';
                    exit(1);
                }

                if (std::isinf(u)) {
                    cerr << "in MatrixSubstitutionProcess:: drawing exponential number: "
                            "inf\n";
                    cerr << rate << '\t' << q << '\n';
                    cerr << totaltime << '\n';
                    cerr << state << '\t' << stateup << '\n';
                    exit(1);
                }

                t += u;
                if (t < totaltime) {
                    int newstate = matrix.DrawOneStep(state);
                    path->Append(newstate, u / totaltime);
                    state = newstate;
                } else {
                    t -= u;
                    u = totaltime - t;
                    path->Last()->SetRelativeTime(u / totaltime);
                    t = totaltime;
                }
            }
        } while ((ntrial < maxtrial) && (path->Last()->GetState() != statedown));
    }

    // if endstate does not match state at the corresponding end of the branch
    // just force it to match
    // however, this is really dirty !
    // normally, in that case, one should give up with accept-reject
    // and use a uniformized method instead (but not yet adapted to the present
    // code, see below)
    if (path->Last()->GetState() != statedown) {
        // fossil
        // path->Last()->SetState(statedown);
        delete path;
        path = 0;
    }

    return path;
}

BranchSitePath *PhyloProcess::ResampleUniformized(
    int stateup, int statedown, double rate, double totaltime, const SubMatrix &matrix) {
    double length = rate * totaltime;
    int m = matrix.DrawUniformizedSubstitutionNumber(stateup, statedown, length);

    vector<double> y(m + 1);
    for (int r = 0; r < m; r++) { y[r] = Random::Uniform(); }
    y[m] = 1;
    sort(y.begin(), y.end());

    int state = stateup;

    BranchSitePath *path = new BranchSitePath();
    path->Reset(stateup);

    double t = y[0];
    for (int r = 0; r < m; r++) {
        int k = (r == m - 1) ? statedown
                             : matrix.DrawUniformizedTransition(state, statedown, m - r - 1);
        if (k != state) {
            path->Append(k, t);
            t = 0;
        }
        state = k;
        t += y[r + 1] - y[r];
    }
    path->Last()->SetRelativeTime(t);
    return path;
}

/*
void PhyloProcess::AddPolySuffStat(PolySuffStat &polysuffstat) const {
    assert(polyprocess != nullptr);
    for (int site = 0; site < GetNsite(); site++) {
        for (int taxon = 0; taxon < GetNtaxa(); taxon++) {
            int anc_state = GetPathState(taxon, site);
            polysuffstat.IncrementPolyCount(polyprocess->GetDerivedTuple(taxon, site, anc_state));
        }
    }
}
void PhyloProcess::AddPolySuffStat(Array<PolySuffStat> &suffstatarray) const {
    assert(polyprocess != nullptr);
    for (int site = 0; site < GetNsite(); site++) {
        for (int taxon = 0; taxon < GetNtaxa(); taxon++) {
            suffstatarray[site].IncrementPolyCount(
                polyprocess->GetDerivedTuple(taxon, site, GetPathState(taxon, site)));
        }
    }
}
*/

