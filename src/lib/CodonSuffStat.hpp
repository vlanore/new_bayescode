#pragma once

#include <cassert>
#include <typeinfo>
#include "PathSuffStat.hpp"
#include "PoissonSuffStat.hpp"

/**
 * \brief A sufficient statistic for substitution histories, as a function of
 * the underlying nucleotide rate matrix (for Muse-Gaut codon models only)
 *
 * The generic sufficient statistics for substitution histories (S) as a
 * function of the rate matrix Q are defined in PathSuffStat. They give all
 * information needed in order to compute p(S | Q), up to a normalization
 * constant. When Q itself is a Muse and Gaut codon model parameterized by a
 * nucleotide rate matrix M, then the probability of S as a function of M, p(S |
 * M), can be expressed in terms of an even more compact (4x4) suff stat.
 *
 * NucPathSuffStat implements this idea, and provides methods for calculating
 * these suffstats based on generic PathSuffStat at the codon-level, adding them
 * over sites / branches and computing the log p(S | M) for any M.
 */

class NucPathSuffStat {
  public:
    NucPathSuffStat(const CodonStateSpace &incod = CodonStateSpace(Universal))
        : rootcount(4, 0),
          paircount(4, std::vector<int>(4, 0)),
          pairbeta(4, std::vector<double>(4, 0)),
          cod(incod) {}

    ~NucPathSuffStat() {}

    //! set suff stat to 0
    void Clear() {
        for (int i = 0; i < Nnuc; i++) {
            rootcount[i] = 0;
            for (int j = 0; j < Nnuc; j++) {
                paircount[i][j] = 0;
                pairbeta[i][j] = 0;
            }
        }
    }

    //! \brief compute the 4x4 path suff stat out of 61x61 codonpathsuffstat
    //
    //! Note that the resulting 4x4 nuc path suff stat depends on other aspects of
    //! the codon matrix (e.g. the value of omega)
    void AddSuffStat(const NucCodonSubMatrix &codonmatrix, const PathSuffStat &codonpathsuffstat) {
        const CodonStateSpace *cod = codonmatrix.GetCodonStateSpace();
        const SubMatrix *nucmatrix = codonmatrix.GetNucMatrix();

        // root part
        const std::map<int, int> &codonrootcount = codonpathsuffstat.GetRootCountMap();
        for (std::map<int, int>::const_iterator i = codonrootcount.begin();
             i != codonrootcount.end(); i++) {
            int codon = i->first;
            rootcount[cod->GetCodonPosition(0, codon)] += i->second;
            rootcount[cod->GetCodonPosition(1, codon)] += i->second;
            rootcount[cod->GetCodonPosition(2, codon)] += i->second;
        }

        const std::map<int, double> &waitingtime = codonpathsuffstat.GetWaitingTimeMap();
        for (std::map<int, double>::const_iterator i = waitingtime.begin(); i != waitingtime.end();
             i++) {
            int codon = i->first;
            for (int c2 = 0; c2 < cod->GetNstate(); c2++) {
                if (c2 != codon) {
                    int pos = cod->GetDifferingPosition(codon, c2);
                    if (pos < 3) {
                        int n1 = cod->GetCodonPosition(pos, codon);
                        int n2 = cod->GetCodonPosition(pos, c2);
                        pairbeta[n1][n2] +=
                            i->second * codonmatrix(codon, c2) / (*nucmatrix)(n1, n2);
                    }
                }
            }
        }

        const std::map<std::pair<int, int>, int> &codonpaircount =
            codonpathsuffstat.GetPairCountMap();
        for (std::map<std::pair<int, int>, int>::const_iterator i = codonpaircount.begin();
             i != codonpaircount.end(); i++) {
            int cod1 = i->first.first;
            int cod2 = i->first.second;
            int pos = cod->GetDifferingPosition(cod1, cod2);
            if (pos == 3) {
                std::cerr << "error in codon conj path suffstat\n";
                exit(1);
            }
            int n1 = cod->GetCodonPosition(pos, cod1);
            int n2 = cod->GetCodonPosition(pos, cod2);
            paircount[n1][n2] += i->second;
        }
    }

    //! \brief return the log probability as a function of a nucleotide matrix
    //!
    //! The codon state space is given as an argument (the nucleotide matrix or
    //! the suff stat themselves do not know the genetic code)
    double GetLogProb(const SubMatrix &mat, const CodonStateSpace &cod) const {
        double total = 0;
        // root part
        int nroot = 0;
        auto rootstat = mat.GetStationary();
        for (int i = 0; i < Nnuc; i++) {
            total += rootcount[i] * log(rootstat[i]);
            nroot += rootcount[i];
        }
        total -= nroot / 3 * log(cod.GetNormStat(rootstat));

        // non root part
        for (int i = 0; i < Nnuc; i++) {
            for (int j = 0; j < Nnuc; j++) {
                if (i != j) {
                    total += paircount[i][j] * log(mat(i, j));
                    total -= pairbeta[i][j] * mat(i, j);
                }
            }
        }

        return total;
    }

    double GetLogProb(const SubMatrix &mat) const {
        double total = 0;
        // root part
        int nroot = 0;
        auto rootstat = mat.GetStationary();
        for (int i = 0; i < Nnuc; i++) {
            total += rootcount[i] * log(rootstat[i]);
            nroot += rootcount[i];
        }
        total -= nroot / 3 * log(cod.GetNormStat(rootstat));

        // non root part
        for (int i = 0; i < Nnuc; i++) {
            for (int j = 0; j < Nnuc; j++) {
                if (i != j) {
                    total += paircount[i][j] * log(mat(i, j));
                    total -= pairbeta[i][j] * mat(i, j);
                }
            }
        }

        return total;
    }

    //! add another nucpath suff stat to this
    void Add(const NucPathSuffStat &from) {
        for (int i = 0; i < Nnuc; i++) { rootcount[i] += from.rootcount[i]; }
        for (int i = 0; i < Nnuc; i++) {
            for (int j = 0; j < Nnuc; j++) {
                paircount[i][j] += from.paircount[i][j];
                pairbeta[i][j] += from.pairbeta[i][j];
            }
        }
    }

    //! add another nuc path suffstat to this, operator version
    NucPathSuffStat &operator+=(const NucPathSuffStat &from) {
        Add(from);
        return *this;
    }

    template <class T>
    void serialization_interface(T &x) {
        x.add(rootcount, paircount, pairbeta);
    }

    // private:
    std::vector<int> rootcount;
    std::vector<std::vector<int>> paircount;
    std::vector<std::vector<double>> pairbeta;
    const CodonStateSpace &cod;

    bool operator==(NucPathSuffStat &other) const {
        return rootcount == other.rootcount && paircount == other.paircount &&
               pairbeta == other.pairbeta;
    }
};


/**
 * \brief A sufficient statistic for substitution histories, as a function of
 * omega=dN/dS (for codon models)
 *
 * The generic sufficient statistics for substitution histories (S) as a
 * function of the rate matrix Q are defined in PathSuffStat. They give all
 * information needed in order to compute p(S | Q), up to a normalization
 * constant. When Q itself is codon model with an omega=dN/dS acting as a
 * multiplier in front of all non-synonymous substitutions, then the probability
 * of S as a function of omega can be expressed in very compact form: p(S |
 * omega) propto omega^count exp(-beta * omega), where count (integer) and beta
 * (positive real number) are the suff stats. Note that this is in fact
 * analogous to a Poisson distribution, with mean omega, and thus,
 * OmegaPathSuffStat derives from PoissonSuffStat.
 */

class OmegaPathSuffStat : public PoissonSuffStat {
  public:
    OmegaPathSuffStat() {}
    ~OmegaPathSuffStat() {}

    //! \brief tease out syn and non-syn substitutions and sum up count and beta
    //! stats from a 61x61 codon path suffstat
    //!
    //! note that omega suff stat depends on the other aspects of the codon matrix
    //! (in particular, the nucleotide rate matrix)
    void AddSuffStat(const OmegaCodonSubMatrix &codonsubmatrix, const PathSuffStat &pathsuffstat) {
        int ncodon = codonsubmatrix.GetNstate();
        const CodonStateSpace *statespace = codonsubmatrix.GetCodonStateSpace();

        const std::map<std::pair<int, int>, int> &paircount = pathsuffstat.GetPairCountMap();
        const std::map<int, double> &waitingtime = pathsuffstat.GetWaitingTimeMap();

        double tmpbeta = 0;
        for (std::map<int, double>::const_iterator i = waitingtime.begin(); i != waitingtime.end();
             i++) {
            double totnonsynrate = 0;
            int a = i->first;
            for (int b = 0; b < ncodon; b++) {
                if (b != a) {
                    if (codonsubmatrix(a, b) != 0) {
                        if (!statespace->Synonymous(a, b)) {
                            totnonsynrate += codonsubmatrix(a, b);
                        }
                    }
                }
            }
            tmpbeta += i->second * totnonsynrate;
        }
        tmpbeta /= codonsubmatrix.GetOmega();

        int tmpcount = 0;
        for (std::map<std::pair<int, int>, int>::const_iterator i = paircount.begin();
             i != paircount.end(); i++) {
            if (!statespace->Synonymous(i->first.first, i->first.second)) { tmpcount += i->second; }
        }

        PoissonSuffStat::AddSuffStat(tmpcount, tmpbeta);
    }

    double GetLogProb(double omega) const { return count * log(omega) - beta * omega; }
};

