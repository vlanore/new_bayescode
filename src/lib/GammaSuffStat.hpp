#pragma once

/**
 * \brief A sufficient statistic for a collection of gamma variates, as a
 * function of the shape and scale parameters
 *
 * Suppose you have x = (x_i)_i=1..N, iid Gamma(shape,scale).
 * Then, p(X | shape,scale) can be expressed as a function of compact sufficient
 * statistics: sum x_i's, sum log(x_i)'s and N. GammaSuffStat implements this
 * idea, by providing methods for collecting these suff stats and returning the
 * log prob for a given value for the shape and scale parameters.
 */

class GammaSuffStat {
  public:
    GammaSuffStat() {Clear();}
    ~GammaSuffStat() {}

    //! set suff stats to 0
    void Clear() {
        sum = 0;
        sumlog = 0;
        n = 0;
    }

    //! add the contribution of one gamma variate (x) to this suffstat
    void AddSuffStat(double x, double logx, int c = 1) {
        sum += x;
        sumlog += logx;
        n += c;
    }

    //! (*this) += from
    void Add(const GammaSuffStat &from) {
        sum += from.sum;
        sumlog += from.sumlog;
        n += from.n;
    }

    //! (*this) += from, operator version
    GammaSuffStat &operator+=(const GammaSuffStat &from) {
        Add(from);
        return *this;
    }

    //! return log prob, as a function of the given shape and scale parameters
    double GetLogProb(double mean, double invshape) const {
        double shape = 1.0 / invshape;
        double scale = shape / mean;
        return n * (shape * log(scale) - Random::logGamma(shape)) + (shape - 1) * sumlog -
               scale * sum;
    }

    bool operator==(const GammaSuffStat& other) const {
        return sum == other.sum && sumlog == other.sumlog && n == other.n;
    }

    double sum;
    double sumlog;
    int n;
};

