#pragma once

#include <cmath>
#include "SuffStat.hpp"

/**
 * \brief A Poisson-like sufficient statistic
 *
 * This sufficient statistic deals with all cases where the probability of the
 * variable(s) of interest (say X), as a function of some rate parameter r
 * (positive real number), can be written as P(X | r) \propto r^count
 * exp(-beta*r), for some suff stats count (integer) and beta (positive real
 * number). When several independent variables are entailed by the generic X
 * mentioned above, the suff stats just have to be summed over all items
 * (separately for count and beta).
 *
 * As an example, the probability of the substitution histories over all sites,
 * for a given branch of the tree, as a function of the branch length l, can be
 * written as follows: p(S | l) = K l^count exp(-beta*l), where count is the
 * total number of substitution events along the branch (across all sites), beta
 * is the average rate away from the current state (averaged over all paths) and
 * K is some normalization constant (that may depend on other parameters than
 * l). Thus, conditional on the current substitution histories, the sufficient
 * statistics count and beta can be first computed, and then MCMC moves on
 * branch lengths can be done based on the knowledge of these two numbers. Other
 * examples include the probability of substitution histories as a function of
 * omega = dN/dS (in that case, the count suff stat is the number of
 * non-synonymous substitutions only). In these two cases, the suffstats have to
 * be calculated based on the substitution histories (see
 * BranchSitePath::AddLengthSuffStat) and then summed over branches, sites, or
 * any other more subtle pattern, depending on the exact structure of the model.
 *
 * PoissonSuffStat implements this general idea:
 * collecting count and beta suff stats across all relevant items
 * and providing methods for calculating the probability, as a function of the
 * rate parameter.
 */

class PoissonSuffStat : public SuffStat {
  public:
    PoissonSuffStat() { Clear(); }
    ~PoissonSuffStat() {}

    //! set count and beta to 0
    void Clear() { count = beta = 0; }

    void IncrementCount() { count++; }

    void AddCount(int in) { count += in; }

    void AddBeta(double in) { beta += in; }

    void AddSuffStat(int incount, double inbeta) {
        count += incount;
        beta += inbeta;
    }

    void Add(const PoissonSuffStat &from) {
        count += from.GetCount();
        beta += from.GetBeta();
    }

    PoissonSuffStat &operator+=(const PoissonSuffStat &from) {
        Add(from);
        return *this;
    }

    //! write structure into generic output stream
    void ToStream(std::ostream &os) const { os << count << '\t' << beta << '\n'; }

    //! read structure from generic input stream
    void FromStream(std::istream &is) { is >> count >> beta; }

    int GetCount() const { return count; }

    double GetBeta() const { return beta; }

    //! return the log probability as a function of the rate: essentially
    //! count*log(rate) - beta*rate
    double GetLogProb(double rate) const { return count * log(rate) - beta * rate; }

    //! return the log of the marginal probability when the rate is from a gamma
    //! distribution
    double GetMarginalLogProb(double mean, double invshape) const {
        double shape = 1.0 / invshape;
        double scale = shape / mean;
        return shape * log(scale) - Random::logGamma(shape) - (shape + count) * log(scale + beta) +
               Random::logGamma(shape + count);
    }

    template <class T>
    void serialization_interface(T &x) {
        x.add(count, beta);
    }

    // protected:
    int count;
    double beta;

    bool operator==(const PoissonSuffStat& other) const {
        return count == other.count && beta == other.beta;
    }
};

