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
#include "components/custom_tracer.hpp"

struct root_mean{};
struct root_variance{};
struct brownian_tau{};

struct normal_mean_condL    {

    // conditional likelihoods
    // normalization constant, mean and precision
    // (logK, m, tau)
    // L(x) = K * exp(-0.5*tau*(x-m)^2)

    using L = std::tuple<double, double, double>;
    using T = real;
    using Constraint = bool;

    static L make_init()  {
        return std::tuple<double, double, double>(0,0,0);
    }

    static void init(L& condl)  {
        get<0>(condl) = 0;
        get<1>(condl) = 0;
        get<2>(condl) = 0;
    }

    static void init(L& condl, const real& x, const bool& clamp, bool external_clamp)    {
        if (clamp || external_clamp)  {
            get<0>(condl) = 0;
            get<1>(condl) = x;
            get<2>(condl) = std::numeric_limits<double>::infinity();
        }
        else    {
            get<0>(condl) = 0;
            get<1>(condl) = 0;
            get<2>(condl) = 0;
        }
    }

    // multiply condl into res_condl
    static void multiply(const L& condl, L& res_condl)   {

        double logK1 = get<0>(condl);
        double logK2 = get<0>(res_condl);

        double m1 = get<1>(condl);
        double m2 = get<1>(res_condl);

        double tau1 = get<2>(condl);
        double tau2 = get<2>(res_condl);

        double tau = tau1 + tau2;
        double m = (tau1*m1 + tau2*m2)/tau;
        double taup = tau1*tau2/(tau1 + tau2);
        double dm = m1-m2;

        double logK = logK1 + logK2 + 0.5*log(taup/2/constants::pi) - 0.5*taup*dm*dm;

        get<0>(res_condl) = logK;
        get<1>(res_condl) = m;
        get<2>(res_condl) = tau;
    }

    // vector versions

    static void init(std::vector<L>& condls)  {
        for(auto& condl : condls)   {
            get<0>(condl) = 0;
            get<1>(condl) = 0;
            get<2>(condl) = 0;
        }
    }

    static void multiply(const std::vector<L>& from_condls, std::vector<L>& to_condls)   {
        for (size_t i=0; i<from_condls.size(); i++)  {
            multiply(from_condls.at(i), to_condls[i]);
        }
    }

    static void mix(L& condl, const std::vector<L>& condls)   {
        init(condl);
        double maxlogK = 0;
        for (const auto& l : condls)    {
            if (maxlogK < get<0>(l))   {
                maxlogK = get<0>(l);
            }
        }
        double K = 0;
        std::vector<double> w(condls.size(), 0);
        for (size_t i=0; i<condls.size(); i++)  {
            w[i] = exp(get<0>(condls.at(i))) - maxlogK;
            K += w[i];
        }
        double logK = log(K) + maxlogK;

        double mu = 0;
        double v = 0;
        for (size_t i=0; i<condls.size(); i++)  {
            mu += w[i] * get<1>(condls.at(i));
            v += w[i] / get<2>(condls.at(i));
        }
        mu /= K;
        v /= K;
        double tau = 1.0 / v;

        get<0>(condl) = logK;
        get<1>(condl) = mu;
        get<2>(condl) = tau;
    }
};

template<class T>
struct discretized_path : public std::vector<T>, public custom_tracer {

    // dimension n+2 is: number of stochastic dfs (including starting point)

    discretized_path(size_t n) : std::vector<T>(n+2,T()) {}
    discretized_path(size_t n, const T& initT) : std::vector<T>(n+2, initT) {}
    ~discretized_path() {}

    double get_width() const   {
        return 1.0 / (this->size()-1);
    }

    // valid for all points: interior + boundaries
    double get_midpoint_time(size_t i) const {
        return double(i)/(this->size()-1);
    }

    // for all points: interior + boundaries
    double get_breakpoint_time(size_t i) const {
        if (i == 0) return 0;
        if (i == this->size()) return 1.0;
        return double(i)/(this->size()-1) - 0.5*get_width();
    }

    double adapt_to_older_end(const T& x_old1, const T& x_old2) {
        for (size_t i=0; i<this->size(); i++) {
            (*this)[i] += (1-get_midpoint_time(i)) * (x_old2 - x_old1);
        }
        return 0;
    }

    double adapt_to_younger_end(const T& x_young1, const T& x_young2)   {
        for (size_t i=0; i<this->size(); i++) {
            (*this)[i] += get_midpoint_time(i) * (x_young2 - x_young1);
        }
        return 0;
    }

    /*
    template <class TI>
    void serialization_interface(TI &x) {
        for (size_t i=0; i<this->size(); i++)   {
            x.add((*this)[i]);
        }
    }
    */

    void to_stream_header(std::string name, std::ostream& os) const override {
    }

    void to_stream(std::ostream& os) const override {
    }

    void from_stream(std::istream& is) override {
    }


};

template<class T>
std::ostream& operator<<(std::ostream& os, const discretized_path<T>& p)   {
    p.to_stream(os);
    return os;
}

template<class T>
std::istream& operator>>(std::istream& is, discretized_path<T>& p) {
    p.from_stream(is);
    return is;
}

// template <class T> struct has_custom_serialization<discretized_path<T>> : std::true_type {};

template<class T, class Lambda>
static auto get_branch_mean(const discretized_path<T>& path, Lambda lambda)    {

    auto mean = lambda(path.at(0));
    mean *= (path.get_breakpoint_time(1) - path.get_breakpoint_time(0));
    for (size_t i=1; i<path.size(); i++)  {
        mean += (path.get_breakpoint_time(i+1) - path.get_breakpoint_time(i)) * lambda(path.at(i));
    }
    return mean;
}

template<class T, class Lambda>
static auto get_branch_sum(const discretized_path<T>& path, double t_young, double t_old, Lambda lambda)  {
    auto mean = lambda(path.at(0));
    mean *= (path.get_breakpoint_time(1) - path.get_breakpoint_time(0));
    for (size_t i=1; i<path.size(); i++)  {
        mean += (path.get_breakpoint_time(i+1) - path.get_breakpoint_time(i)) * lambda(path.at(i));
    }
    return mean * (t_old-t_young);
} 

struct univariate_normal    {

    using T = real;
    using param_decl = param_decl_t<param<struct mean, real>, param<struct variance, spos_real>>;

    using Constraint = bool;
    using CondL = normal_mean_condL;

    static Constraint make_init_constraint(const T& x)    {
        return false;
    }

    template <typename Gen>
    static void draw(T& x, real mean, spos_real variance, Gen& gen)   {
        std::normal_distribution<double> distrib(real(mean), positive_real(variance));
        x = distrib(gen);
    }

    static real logprob(T& x, real mean, spos_real variance)   {
        double y = (x - mean) / variance;
        return -0.5*log(2.0*constants::pi*variance) - 0.5*y*y;
    }

    template<typename Gen>
    static void conditional_draw(T& x, const Constraint& clamp, const CondL::L& condl, real mean, spos_real variance, Gen& gen) {
    }

    /*
    static double conditional_logprob(T& x, const Constraint& clamp, const CondL::L& condl, real mean, spos_real var)   {
        return 0;
    }
    */
};

struct univariate_brownian {

    using instantT = real;
    using pathT = discretized_path<instantT>;

    using param_decl = param_decl_t<param<brownian_tau, spos_real>>;

    using Constraint = bool;
    using CondL = normal_mean_condL;

    static Constraint make_init_constraint(const instantT& x)    {
        return false;
    }

    template <typename Gen>
    static void path_draw(pathT& path, instantT& x_young, const instantT& x_old, double t_young, double t_old, spos_real tau, Gen& gen)   {

        double dt = (t_old - t_young)*path.get_width();
        double vstep = dt/tau;

        path[0] = x_old;
        for (size_t i=1; i<path.size(); i++) {
            std::normal_distribution<double> distrib(real(path[i-1]), positive_real(vstep));
            path[i] = distrib(gen);
        }
        x_young = path[path.size()-1];
    }

    // why x_young and x_old ? don't seem to be necessary
    static real path_logprob(pathT& path, const instantT& x_young, const instantT& x_old, double t_young, double t_old, spos_real tau)   {

        double s2 = 0;
        for (size_t i=1; i<path.size(); i++) {
            double d = path[i] - path[i-1];
            s2 += d*d;
        }

        double dt = (t_old - t_young)*path.get_width();
        double vstep = dt/tau;
        return -0.5*(path.size()-1)*log(2.0*constants::pi*vstep) - 0.5*s2/vstep;
    }

    static void backward_propagate(const CondL::L& condl_young, CondL::L& condl_old, double t_young, double t_old, spos_real tau)  {

        get<0>(condl_old) = get<0>(condl_young);
        get<1>(condl_old) = get<1>(condl_young);
        double v = 1.0/get<2>(condl_old) + (t_old-t_young)/tau;
        get<2>(condl_old) = 1.0/v;
    }

    static void backward_propagate(const CondL::L& condl_young, const CondL::L& condl_branch, CondL::L& condl_old, double t_young, double t_old, spos_real tau)  {

        double u = 1.0/(t_old - t_young);
        double tau0 = get<2>(condl_young);
        double taub = get<2>(condl_branch);

        double m0 = get<1>(condl_young);
        double mb = get<1>(condl_branch);

        double tau1 = tau0 + u*tau + taub/4;
        double z = tau0*u*tau + tau0*taub/4 + u*tau*taub;
        double tau2 = z/tau1;
        double tau3 = tau0*u*tau*taub/z;

        double m2 = (tau0*u*tau*m0 + tau0*taub/4*(2*mb-m0) + u*tau*taub*mb)/z;

        double v2 = (tau0*u*tau*m0*m0 + tau0*taub/4*(2*mb-m0)*(2*mb-m0) + u*tau*taub*mb*mb)/z 
            - m2*m2;

        get<0>(condl_old) = get<0>(condl_young) + get<0>(condl_branch) 
            + 0.5*log(tau3/2/constants::pi) 
            - 0.5*tau2*v2;

        get<1>(condl_old) = m2;

        get<2>(condl_old) = tau2;
    }

    static double pseudo_branch_logprob(const instantT& x_young, const instantT& x_old, const CondL::L& branch_condl)  {
        instantT x = 0.5*(x_old + x_young);
        double d = x - get<1>(branch_condl);
        return get<0>(branch_condl) + 0.5*log(get<2>(branch_condl)/2/constants::pi)
            -0.5*get<2>(branch_condl)*d*d;
    }

    template<typename Gen>
    static void node_conditional_draw(instantT& x_young, const Constraint& clamp, const instantT& x_old, double t_young, double t_old, const CondL::L& condl, spos_real tau, Gen& gen) {
    }

    template <typename Gen>
    static void bridge_conditional_draw(pathT& path, const instantT& x_young, const instantT& x_old, double t_young, double t_old, spos_real tau, Gen& gen)   {
        path[0] = x_old;
        path[1] = x_young;
    }

    /*
    static double node_conditional_logprob(const instantT& x_young, const Constraint& clamp, const instantT& x_old, double t_young, double t_old, const CondL::L& condl, spos_real tau) {
        return 0;
    }

    static double bridge_conditional_logprob(const pathT& path, const instantT& x_young, const instantT& x_old, double t_young, double t_old, spos_real tau)   {
        return 0;
    }
    */

    // add a brownian path to current path
    template<typename Gen>
    static double path_kernel(double tuning, instantT& x_young, pathT& path, 
            double t_young, double t_old, spos_real tau, Gen& gen)  {

        double mean = 0;
        double var = tuning*path.get_width()/tau;

        for (size_t i=1; i<path.size(); i++)    {
            std::normal_distribution<double> distrib(real(mean), positive_real(var));
            path[i] += distrib(gen);
        }
        x_young = path[path.size()-1];
        return 0;
    }

    // add a brownian bridge to current bridge
    template<typename Gen>
    static double bridge_kernel(double tuning, pathT& path, 
            double t_young, double t_old, spos_real tau, Gen& gen)  {

        const instantT& val1 = path[path.size()-1];

        for (size_t i=1; i<path.size()-1; i++)    {

            double mean = (path[i-1] + (path.size()-i-1)*val1) / (path.size()-i);

            // check this
            double var = tuning * path.get_width()  / (1.0 + 1.0/(path.size()-i-1));

            std::normal_distribution<double> distrib(real(mean), positive_real(var));
            path[i] += distrib(gen);
        }
        return 0;
    }

    static void add_branch_suffstat(brownian_tau, PoissonSuffStat& ss, const pathT& path, const instantT& x_young, const instantT& x_old, double t_young, double t_old, spos_real tau)  {
        double contrast = (x_young-x_old)*(x_young-x_old)/(t_old-t_young);
        ss.AddSuffStat(0.5, 0.5*contrast);
    }

};

