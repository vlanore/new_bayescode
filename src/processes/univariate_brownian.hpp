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

template<class T>
struct discretized_path : public std::vector<T>, public custom_tracer {

    discretized_path(size_t n) : std::vector<T>(n+2,T()) {}
    discretized_path(size_t n, const T& initT) : std::vector<T>(n+2, initT) {}
    ~discretized_path() {}

    double get_width() const   {
        return 1.0 / (this->size()-1);
    }

    double get_midpoint_time(size_t i) const {
        return double(i)/(this->size()-1);
    }

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
static auto get_branch_sum(const discretized_path<T>& path, double t_old, double t_young, Lambda lambda)  {
    auto mean = lambda(path.at(0));
    mean *= (path.get_breakpoint_time(1) - path.get_breakpoint_time(0));
    for (size_t i=1; i<path.size(); i++)  {
        mean += (path.get_breakpoint_time(i+1) - path.get_breakpoint_time(i)) * lambda(path.at(i));
    }
    return mean * (t_old-t_young);
} 

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

    static void init(L& condl, const real& x, const bool& clamp)    {
        if (clamp)  {
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
    }

};

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
        return -0.5 * y * y - log(variance * sqrt(2.0 * constants::pi));
    }

    template<typename Gen>
    static void conditional_draw(T& x, const Constraint& clamp, const CondL::L& condl, real mean, spos_real variance, Gen& gen) {
    }

    static double conditional_logprob(T& x, const Constraint& clamp, const CondL::L& condl, real mean, spos_real var)   {
        return 0;
    }
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
        std::normal_distribution<double> distrib(real(x_old), positive_real((t_old-t_young)/tau));
        x_young = distrib(gen);
        path[0] = x_old;
        path[1] = x_young;
    }

    static real path_logprob(pathT& path, const instantT& x_young, const instantT& x_old, double t_young, double t_old, spos_real tau)   {
        double v = (t_old-t_young)/tau;
        double y = (x_young - x_old) / v;
        return -0.5 * y * y - log(v * sqrt(2.0 * constants::pi));
    }

    static void backward_propagate(const CondL::L& condl_young, CondL::L& condl_old, double t_young, double t_old, spos_real tau)  {
    }

    static void backward_propagate(const CondL::L& condl_young, const CondL::L& condl_branch, CondL::L& condl_old, double t_young, double t_old, spos_real tau)  {
    }

    template<typename Gen>
    static void node_conditional_draw(instantT& x_young, const Constraint& clamp, const instantT& x_old, double t_young, double t_old, const CondL::L& condl, spos_real tau, Gen& gen) {
    }

    template <typename Gen>
    static void bridge_conditional_draw(pathT& path, const instantT& x_young, const instantT& x_old, double t_young, double t_old, spos_real tau, Gen& gen)   {
        path[0] = x_old;
        path[1] = x_young;
    }

    static double node_conditional_logprob(const instantT& x_young, const Constraint& clamp, const instantT& x_old, double t_young, double t_old, const CondL::L& condl, spos_real tau) {
        return 0;
    }

    static double bridge_conditional_logprob(const pathT& path, const instantT& x_young, const instantT& x_old, double t_young, double t_old, spos_real tau)   {
        return 0;
    }

    // add a brownian path to current path
    template<typename Gen>
    static double path_kernel(double tuning, instantT& x_young, pathT& path, 
            double t_young, double t_old, spos_real tau, Gen& gen)  {
        for (size_t i=1; i<path.size(); i++)    {
            double mean = 0;
            double rel_prec = path.get_midpoint_time(i) - path.get_midpoint_time(i-1);
            double var = tuning/tau/rel_prec;
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

        const instantT& val0 = path[0];
        const instantT& val1 = path[path.size()-1];

        for (size_t i=1; i<path.size()-1; i++)    {
            double mean = ((path.get_midpoint_time(i) - path.get_midpoint_time(i-1))*val1 + (1.0 - path.get_midpoint_time(i))*val0) / path.get_midpoint_time(i);
            double rel_prec = 1.0/(path.get_midpoint_time(i) - path.get_midpoint_time(i-1)) + 1.0/(1.0 - path.get_midpoint_time(i));
            double var = tuning/tau/rel_prec;
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

