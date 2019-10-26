
#ifndef PATHSUFFSTAT_H
#define PATHSUFFSTAT_H

#include <map>
#include "SubMatrix.hpp"
#include "SuffStat.hpp"

/**
 * \brief A general sufficient statistic for substitution histories, as a
 * function of the substitution rate matrix.
 *
 * The probability of a set of detailed substitution histories (collectively
 * denoted as S), as a function of some rate matrix Q = (Q_ab), with equilibrium
 * frequencies pi = (pi_a), can be written as:
 *
 * p(S | Q) propto (prod_a pi_a^u_a) (prod_a exp(t_a Q_aa)) (prod_ab Q_ab^v_ab),
 *
 * where u_a is the total number of times state a was seen at the root (root
 * count statistic), v_ab (pair is the total number of substitution events from
 * a to b (pair count stat), and t_a is the total waiting time in state a
 * (waiting time stat) -- all this, across all substitution histories included
 * in S.
 *
 * PathSuffStat implements this idea, by providing methods for gathering
 * sufficient statistics across substitution histories (see also
 * BranchSitePath::AddPathSuffStat), adding them across sites and/or branches,
 * and calculating the log p(S | Q) for any matrix Q
 *
 * These path suffstats can be used for any Markovian substitution process (any
 * Q). In some cases (i.e. for Muse and Gaut codon models), they can be
 * furthered simplified, as a function of the nucleotide rate parameters or the
 * omega parameter of the Q matrix, leading to even more compact suff stats (see
 * OmegaPathSuffStat and NucPathSuffStat).
 *
 * In terms of implementation, these suffstats are encoded as sparse data
 * structures (since a very small subset of all possible pairs of codons will
 * typcially be visited by the substitution history of a given site, for
 * instance). This sparse encoding is crucial for efficiency (both in terms of
 * time and in terms of RAM usage).
 */

class PathSuffStat : public SuffStat {
  public:
    PathSuffStat() {}
    ~PathSuffStat() {}

    //! set suff stats to 0
    void Clear() {
        rootcount.clear();
        paircount.clear();
        waitingtime.clear();
    }

    void IncrementRootCount(int state) { rootcount[state]++; }

    void IncrementPairCount(int state1, int state2) {
        paircount[std::pair<int, int>(state1, state2)]++;
    }

    void AddRootCount(int state, int in) { rootcount[state] += in; }

    void AddPairCount(int state1, int state2, int in) {
        paircount[std::pair<int, int>(state1, state2)] += in;
    }

    void AddWaitingTime(int state, double in) { waitingtime[state] += in; }

    //! add path sufficient statistics from PhyloProcess (site-homogeneous case)
    // void AddSuffStat(const PhyloProcess &process) { process.AddPathSuffStat(*this); }

    void Add(const PathSuffStat &suffstat) {
        for (std::map<int, int>::const_iterator i = suffstat.GetRootCountMap().begin();
             i != suffstat.GetRootCountMap().end(); i++) {
            AddRootCount(i->first, i->second);
        }
        for (std::map<std::pair<int, int>, int>::const_iterator i =
                 suffstat.GetPairCountMap().begin();
             i != suffstat.GetPairCountMap().end(); i++) {
            AddPairCount(i->first.first, i->first.second, i->second);
        }
        for (std::map<int, double>::const_iterator i = suffstat.GetWaitingTimeMap().begin();
             i != suffstat.GetWaitingTimeMap().end(); i++) {
            AddWaitingTime(i->first, i->second);
        }
    }

    PathSuffStat &operator+=(const PathSuffStat &from) {
        Add(from);
        return *this;
    }

    int GetRootCount(int state) const {
        std::map<int, int>::const_iterator i = rootcount.find(state);
        if (i == rootcount.end()) { return 0; }
        return i->second;
    }

    int GetPairCount(int state1, int state2) const {
        std::map<std::pair<int, int>, int>::const_iterator i =
            paircount.find(std::pair<int, int>(state1, state2));
        if (i == paircount.end()) { return 0; }
        return i->second;
    }

    double GetWaitingTime(int state) const {
        std::map<int, double>::const_iterator i = waitingtime.find(state);
        if (i == waitingtime.end()) { return 0; }
        return i->second;
    }

    //! return log p(S | Q) as a function of the Q matrix given as the argument
    double GetLogProb(const SubMatrix &mat) const {
        double total = 0;
        auto stat = mat.GetStationary();
        for (std::map<int, int>::const_iterator i = rootcount.begin(); i != rootcount.end(); i++) {
            total += i->second * log(stat[i->first]);
        }
        for (std::map<int, double>::const_iterator i = waitingtime.begin(); i != waitingtime.end();
             i++) {
            total += i->second * mat(i->first, i->first);
        }
        for (std::map<std::pair<int, int>, int>::const_iterator i = paircount.begin();
             i != paircount.end(); i++) {
            total += i->second * log(mat(i->first.first, i->first.second));
        }
        return total;
    }

    //! const access to the ordered map giving the root count stat (sparse data
    //! structure)
    const std::map<int, int> &GetRootCountMap() const { return rootcount; }
    //! const access to the ordered map giving the pair count stat (sparse data
    //! structure)
    const std::map<std::pair<int, int>, int> &GetPairCountMap() const { return paircount; }
    //! const access to the ordered map giving the waiting time stat (sparse data
    //! structure)
    const std::map<int, double> &GetWaitingTimeMap() const { return waitingtime; }

    bool operator==(const PathSuffStat &other) const {
        return rootcount == other.rootcount && paircount == other.paircount &&
               waitingtime == other.waitingtime;
    }

  private:
    std::map<int, int> rootcount;
    std::map<std::pair<int, int>, int> paircount;
    std::map<int, double> waitingtime;
};

#endif
