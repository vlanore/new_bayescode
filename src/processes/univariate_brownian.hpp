/*Copyright or © or Copr. CNRS (2019). Contributors:
- Vincent Lanore. vincent.lanore@gmail.com

This software is a computer program whose purpose is to provide a set of C++ data structures and
functions to perform Bayesian inference with MCMC algorithms.

This software is governed by the CeCILL-C license under French law and abiding by the rules of
distribution of free software. You can use, modify and/ or redistribute the software under the terms
of the CeCILL-C license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info".

As a counterpart to the access to the source code and rights to copy, modify and redistribute
granted by the license, users are provided only with a limited warranty and the software's author,
the holder of the economic rights, and the successive licensors have only limited liability.

In this respect, the user's attention is drawn to the risks associated with loading, using,
modifying and/or developing or reproducing the software by the user in light of its specific status
of free software, that may mean that it is complicated to manipulate, and that also therefore means
that it is reserved for developers and experienced professionals having in-depth computer knowledge.
Users are therefore encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or data to be ensured and,
more generally, to use and operate it in the same conditions as regards security.

The fact that you are presently reading this means that you have had knowledge of the CeCILL-C
license and that you accept its terms.*/

#pragma once

#include "structure/distrib_utils.hpp"
#include "utils/math_utils.hpp"
#include "lib/PoissonSuffStat.hpp"

struct root_mean{};
struct root_variance{};
struct brownian_tau{};

struct univariate_brownian {

    using T = real;
    using param_decl = param_decl_t<param<brownian_tau, spos_real>, param<root_mean, real>, param<root_variance, spos_real>>;

    template <typename Gen>
    static void draw(T& x, bool isroot, const T& x0, double time, spos_real tau, real mean, spos_real variance, Gen& gen) {
        if (isroot) {
            std::normal_distribution<double> distrib(real(mean), positive_real(variance));
            x = distrib(gen);
        }
        else    {
            std::normal_distribution<double> distrib(real(x0), positive_real(time/tau));
            x = distrib(gen);
        }
    }

    static real logprob(T& x, bool isroot, const T& x0, double time, spos_real tau, real mean, spos_real variance)   {
        if (isroot) {
            double y = (x - mean) / variance;
            return -0.5 * y * y - log(variance * sqrt(2.0 * constants::pi));
        }
        double v = time/tau;
        double y = (x - x0) / v;
        return -0.5 * y * y - log(v * sqrt(2.0 * constants::pi));
    }

    static void add_branch_suffstat(brownian_tau, PoissonSuffStat& ss, const T& x, const T& x0, double time, spos_real tau, real mean, spos_real variance)   {
        double contrast = (x-x0)*(x-x0)/time;
        ss.AddSuffStat(0.5, 0.5*contrast);
    }

    // conditional likelihoods
    // normalization constant, mean and precision
    // (K, m, tau)
    // L(x) = K * exp(-0.5*tau*(x-m)^2)
    using CondL = std::tuple<double, double, double>;

    static void backward_propagate(CondL& down, const CondL& up, double time, spos_real tau, real mean, spos_real variance)  {
    }

    /*
    static void multiply_conditional_likelihoods(CondL& result, const CondL&... condl)    {
    }
    */

    template <typename Gen>
    static void posterior_root_sample(T& x, const CondL& condl, spos_real tau, real mean, spos_real variance, Gen& gen)    {
    }

    template <typename Gen>
    static void posterior_non_root_sample(T& x, const T& x0, const CondL& condl, spos_real tau, real mean, spos_real variance, Gen& gen)   {
    }
};

