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

    static double logprob(T& x, const L& condl) {
        double d = x - get<1>(condl);
        return get<0>(condl) + 0.5*log(get<2>(condl)/2/constants::pi)
            - 0.5*get<2>(condl)*d*d;
    }

    // precision can be infinite, but not logK nor mean
    static bool isinf(const L& condl)   {
        return std::isinf(get<0>(condl)) || std::isinf(get<1>(condl));
    }

    static bool isnan(const L& condl)   {
        return std::isnan(get<0>(condl)) || std::isnan(get<1>(condl)) || std::isnan(get<2>(condl));
    }

    // return true if problem
    static bool ill_defined(const L& condl)    {
        bool ret = false;
        if (isinf(condl))   {
            std::cerr << "condl is inf\n";
            ret = true;
        }
        if (isnan(condl))   {
            std::cerr << "condl is nan\n";
            ret = true;
        }
        /*
        if (!get<2>(condl)) {
            if (get<1>(condl) || get<0>(condl)) {
                std::cerr << "condl is ill-defined\n";
                ret = true;
            }
        }
        */
        if (ret)    {
            printerr(condl);
            std::cerr << '\n';
        }
        return ret;
    }

    static void printerr(const L& condl)    {
        std::cerr << get<0>(condl) << '\t' << get<1>(condl) << '\t' << get<2>(condl);
    }

    // multiply condl into res_condl
    static void multiply(const L& condl, L& res_condl)   {

        if (ill_defined(condl)) {
            std::cerr << "source condl ill defined in multiply\n";
            exit(1);
        }
        if (ill_defined(res_condl)) {
            std::cerr << "target condl ill defined in multiply\n";
            exit(1);
        }

        double logK1 = get<0>(condl);
        double logK2 = get<0>(res_condl);

        double m1 = get<1>(condl);
        double m2 = get<1>(res_condl);

        double tau1 = get<2>(condl);
        double tau2 = get<2>(res_condl);

        if (std::isinf(tau2))   {
            if (std::isinf(tau1)) {
                // assert(m1 == m2);
                if (m1 != m2)   {
                    std::cerr << "error: multiplying singular condls with different means\n";
                    printerr(condl);
                    std::cerr << '\n';
                    printerr(res_condl);
                    std::cerr << '\n';
                    exit(1);
                }
            }
            double logK = logK1 + logK2;
            get<0>(res_condl) = logK;
        }
        else if (tau1 == 0) {
        }
        else if (tau2 == 0) {
            get<0>(res_condl) = get<0>(condl);
            get<1>(res_condl) = get<1>(condl);
            get<2>(res_condl) = get<2>(condl);
        }
        else    {
            double tau = tau1 + tau2;
            double m = (tau1*m1 + tau2*m2)/tau;
            double taup = tau1*tau2/(tau1 + tau2);
            double dm = m1-m2;

            double logK = logK1 + logK2 + 0.5*log(taup/2/constants::pi) - 0.5*taup*dm*dm;
            if (std::isinf(logK))   {
                std::cerr << "logK is inf\n";
                std::cerr << tau << '\t' << m << '\t' << taup << '\n';
            }

            get<0>(res_condl) = logK;
            get<1>(res_condl) = m;
            get<2>(res_condl) = tau;
        }

        if (ill_defined(res_condl))   {
            std::cerr << "error in condl multiply\n";
            printerr(condl);
            std::cerr << '\n';
            std::cerr << logK2 << '\t' << m2 << '\t' << tau2 << '\n';
            std::cerr << "---------------\n";
            printerr(res_condl);
            std::cerr << '\n';
            exit(1);
        }
    }

    static void mix(L& condl, size_t min, size_t max, const std::vector<L>& condls)   {
        init(condl);
        double maxlogK = 0;
        for (size_t i=min; i<=max; i++)  {
            if (maxlogK < get<0>(condls.at(i)))   {
                maxlogK = get<0>(condls.at(i));
            }
        }
        double K = 0;
        std::vector<double> w(condls.size(), 0);
        for (size_t i=min; i<=max; i++)  {
            w[i] = exp(get<0>(condls.at(i)) - maxlogK);
            K += w[i];
        }
        double logK = log(K) + maxlogK;

        double mu = 0;
        double v = 0;
        for (size_t i=min; i<=max; i++)  {
            mu += w[i] * get<1>(condls.at(i));
            v += w[i] / get<2>(condls.at(i));
        }
        mu /= K;
        v /= K;
        double tau = 1.0 / v;

        get<0>(condl) = logK;
        get<1>(condl) = mu;
        get<2>(condl) = tau;
        if (ill_defined(condl)) {
            std::cerr << "result of mixing condls is ill defined\n";
            for (size_t i=min; i<=max; i++)  {
                printerr(condls.at(i));
                std::cerr << '\n';
            }
            std::cerr << "===\n";
            printerr(condl);
            std::cerr << '\n';
            exit(1);
        }
    }

    static double point_eval(const L& condl, const T& val)   {
        if (! get<2>(condl))    {
            if (get<1>(condl))  {
                std::cerr << "in point eval: inconsistent condl\n";
                exit(1);
            }
            return get<0>(condl);
        }
        double diff = val - get<1>(condl);
        return get<0>(condl) - 0.5*diff*diff/get<2>(condl);
    }

    template<class Gen>
    static size_t draw_component(const std::vector<L>& condls, const std::vector<double>& weights, 
            size_t imin, size_t imax, const T& val, double& logZ, Gen& gen)   {
        double max = 0;
        std::vector<double> logw(condls.size(), 0);
        for (size_t i=imin; i<=imax; i++)   {
            logw[i] = point_eval(condls.at(i), val);
            if ((i == imin) || (max < logw[i])) {
                max = logw[i];
            }
        }
        std::vector<double> w(condls.size(), 0);
        double tot = 0;
        for (size_t i=imin; i<=imax; i++)   {
            w[i] = weights[i] * exp(logw[i] - max);
            tot += w[i];
        }
        for (size_t i=imin; i<=imax; i++)   {
            w[i] /= tot;
        }
        std::discrete_distribution<int> distrib(w.begin(), w.end());
        logZ = log(tot) + max;
        int ret = distrib(gen);
        if ((ret < int(imin)) || (ret > int(imax)))   {
            std::cerr << "error in draw component: out of bounds\n";
            std::cerr << ret << '\t' << imin << '\t' << imax << '\n';
            for (size_t i=imin; i<=imax; i++)   {
                std::cerr << i << '\t';
                printerr(condls.at(i));
                std::cerr << '\t' << weights[i] << '\t' << logw[i] << '\t' << w[i] << '\n';
            }
            exit(1);
        }
        return ret;
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

    /*
    // for all points: interior + boundaries
    double get_breakpoint_time(size_t i) const {
        if (i == 0) return 0;
        if (i == this->size()) return 1.0;
        return double(i)/(this->size()-1) - 0.5*get_width();
    }
    */

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
    auto mean = 0.5*(lambda(path.at(0)) + lambda(path.at(path.size()-1)));
    for (size_t i=1; i<path.size()-1; i++)  {
        mean += lambda(path.at(i));
    }
    mean /= path.size()-1;
    return mean;
}

template<class T, class Lambda>
static auto get_branch_sum(const discretized_path<T>& path, double t_young, double t_old, Lambda lambda)  {
    auto mean = 0.5*(lambda(path.at(0)) + lambda(path.at(path.size()-1)));
    for (size_t i=1; i<path.size()-1; i++)  {
        mean += lambda(path.at(i));
    }
    mean /= path.size()-1;
    double ret = mean * (t_old-t_young);
    if (std::isnan(ret))    {
        std::cerr << "nan branch sum : " << mean << '\t' << t_old << '\t' << t_young << '\n';
        for (size_t i=0; i<path.size(); i++)  {
            std::cerr << path.at(i) << '\t';
        }
        std::cerr << '\n';
        exit(1);
    }
    return ret;
} 

struct univariate_normal    {

    using T = real;
    using param_decl = param_decl_t<param<struct mean, real>, param<struct variance, spos_real>>;

    using Constraint = bool;
    using CondL = normal_mean_condL;

    static Constraint make_init_constraint(const T& x)    {
        return false;
    }

    static bool active_constraint(const Constraint& clamp)  {
        return clamp;
    }

    template <typename Gen>
    static void draw(T& x, real mean, spos_real variance, Gen& gen)   {
        std::normal_distribution<double> distrib(real(mean), positive_real(sqrt(variance)));
        x = distrib(gen);
    }

    static real logprob(T& x, real mean, spos_real variance)   {
        double y = (x - mean);
        return -0.5*log(2.0*constants::pi*variance) - 0.5*y*y/variance;
    }

    template<typename Gen>
    static void conditional_draw(T& x, const Constraint& clamp, const CondL::L& condl, real mean, spos_real variance, Gen& gen) {

        // check that if clamped, conditional likelihood is singular at the current value
        if (clamp)  {
            if (! std::isinf(get<2>(condl)))    {
                std::cerr << "error: clamped node but likelihood not singular\n";
                exit(1);
            }
            if (x != get<1>(condl))   {
                std::cerr << "error: clamped node value differs from likelihood mean: " << x << '\t' << get<1>(condl) << '\n';
                exit(1);
            }
        }
        else    {
            double om0 = 1.0/variance;
            double om1 = get<2>(condl);
            double om = om0 + om1;
            double m = (om0*mean + om1*get<1>(condl))/om;
            double v = 1.0/om;
            std::normal_distribution<double> distrib(real(m), positive_real(sqrt(v)));
            x = distrib(gen);
        }
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

    static bool active_constraint(const Constraint& clamp)  {
        return clamp;
    }

    template <typename Gen>
    static void node_draw(instantT& x_young, const instantT& x_old, double t_young, double t_old, spos_real tau, Gen& gen)   {

        double v = (t_old-t_young)/tau;
        std::normal_distribution<double> distrib(real(x_old), positive_real(sqrt(v)));
        x_young = distrib(gen);
    }

    template <typename Gen>
    static void path_draw(pathT& path, instantT& x_young, const instantT& x_old, double t_young, double t_old, spos_real tau, Gen& gen)   {

        size_t n = path.size();
        double dt = (t_old - t_young)/(n-1);
        double vstep = dt/tau;

        path[0] = x_old;
        for (size_t i=1; i<n; i++) {
            std::normal_distribution<double> distrib(real(path[i-1]), positive_real(sqrt(vstep)));
            path[i] = distrib(gen);
        }
        x_young = path[n-1];
    }

    // why x_young and x_old ? don't seem to be necessary
    static real path_logprob(pathT& path, const instantT& x_young, const instantT& x_old, double t_young, double t_old, spos_real tau)   {

        auto n = path.size();
        if ((fabs(path[0]-x_old) > 1e-6) || (fabs(path[n-1]-x_young) > 1e-6))   {
            std::cerr << "error in univ brownian process: path does not match end values\n";
            std::cerr << x_old << '\t' << path[0] << " --- " << path[n-1] << '\t' << x_young << '\n';
            exit(1);
        }
        double s = 0;
        for (size_t i=1; i<n; i++) {
            double d = path[i] - path[i-1];
            s += d*d;
        }
        double v = (t_old - t_young)/(n-1)/tau;
        return -0.5*(n-1)*log(2.0*constants::pi*v) - 0.5*s/v;
    }

    static real node_logprob(const instantT& x_young, const instantT& x_old, double t_young, double t_old, spos_real tau)   {
        double s = (x_young - x_old)*(x_young-x_old);
        double v = (t_old - t_young)/tau;
        return -0.5*log(2.0*constants::pi*v) - 0.5*s/v;
    }

    static void backward_propagate(const CondL::L& condl_young, CondL::L& condl_old, double t_young, double t_old, spos_real tau)  {

        get<0>(condl_old) = get<0>(condl_young);
        get<1>(condl_old) = get<1>(condl_young);
        double v = 1.0/get<2>(condl_young) + (t_old-t_young)/tau;
        get<2>(condl_old) = 1.0/v;

        if (normal_mean_condL::ill_defined(condl_old))   {
            std::cerr << "problem in result of propagate\n";
            normal_mean_condL::printerr(condl_young);
            std::cerr << '\n';
            normal_mean_condL::printerr(condl_old);
            std::cerr << '\n';
            std::cerr << t_young - t_old << '\t' << tau << '\n';
            std::cerr << v << '\t' << get<2>(condl_young) << '\t' << 1.0/get<2>(condl_young) << '\t' <<  (t_old-t_young)/tau << '\n';
            exit(1);
        }
    }

    template<typename Gen>
    static void node_conditional_draw(instantT& x_young, const Constraint& clamp, const instantT& x_old, double t_young, double t_old, const CondL::L& condl, spos_real tau, Gen& gen) {

        // check that if clamped, conditional likelihood is singular at the current value
        if (clamp)  {
            if (! std::isinf(get<2>(condl)))    {
                std::cerr << "error: clamped node but likelihood not singular\n";
                exit(1);
            }
            if (x_young != get<1>(condl))   {
                std::cerr << "error: clamped node value differs from likelihood mean: " << x_young << '\t' << get<1>(condl) << '\n';
                exit(1);
            }
        }

        // prior : mean m0 = x_old, precision om0 = tau/(t_old-t_young)
        // condl : mean m1 = get<1>(condl), precision om1 = get<2>(condl)
        // post  : om = om0 + om1; m 
        //       : m = (om0*m0 + om1*m1)/om
        
        else    {
            double om0 = tau/(t_old-t_young);
            double om1 = get<2>(condl);
            double om = om0 + om1;
            double m = (om0*x_old + om1*get<1>(condl))/om;
            double v = 1.0/om;
            std::normal_distribution<double> distrib(real(m), positive_real(sqrt(v)));
            x_young = distrib(gen);
        }
    }

    template <typename Gen>
    static void bridge_draw(pathT& path, const instantT& x_young, const instantT& x_old, double t_young, double t_old, spos_real tau, Gen& gen)   {
        if (path.size() == 2)   {
            path[0] = x_old;
            path[1] = x_young;
        }
        else    {
            size_t n = path.size();
            path[0] = x_old;
            path[n-1] = x_young;

            for (size_t i=1; i<n-1; i++)    {
                double om0 = tau/(t_old-t_young)*(n-1);
                double om1 = om0/(n-1-i);
                double om = om0 + om1;
                double m = (om0*path[i-1] + om1*x_young)/om;
                double sigma = 1.0/sqrt(om);
                std::normal_distribution<double> distrib(real(m), positive_real(sigma));
                path[i] = distrib(gen);
            }
        }
    }

    // add a brownian bridge to current bridge
    template<typename Gen>
    static double bridge_kernel(double tuning, pathT& path, 
            double t_young, double t_old, spos_real tau, Gen& gen)  {

        // brownian bridge with discrete steps delta_t = (t_old-t_young)/(path.size()-1)
        // at step i
        // variance on left is delta_t/tau
        // variance on right is (n-i-1)*delta_t/tau
        // precision on left is  tau / delta_t
        // precision on right is tau / delta_t / (n-i-1)
        // total precision: tau / delta_t * (1 + 1/(n-i-1)) = tau * tau0 / delta_t
        // total variance : delta_t / tau / tau_0 = (t_old - t_young) / (n-1) / tau / tau0

        size_t n = path.size();
        if (n==2)   {
            return 0;
        }

        pathT dpath = path;
        dpath[0] = dpath[n-1] = 0;

        for (size_t i=1; i<n-1; i++)    {

            double tau1 = 1.0;
            double tau2 = 1.0 / (n-i-1);
            double tau0 = tau1 + tau2;
            double mean = (tau1*dpath[i-1])/tau0;
            double var = tuning * (t_old-t_young) / (n-1) / tau0 / tau;

            std::normal_distribution<double> distrib(real(mean), positive_real(sqrt(var)));
            dpath[i] = distrib(gen);
        }
        for (size_t i=1; i<n-1; i++)    {
            path[i] += dpath[i];
        }
        return 0;
    }

    static void add_branch_suffstat(brownian_tau, PoissonSuffStat& ss, const pathT& path, const instantT& x_young, const instantT& x_old, double t_young, double t_old, spos_real tau)  {
        double delta = path.get_width()*(t_old-t_young);
        for (size_t i=1; i<path.size(); i++)    {
            double contrast = (path[i] - path[i-1])*(path[i] - path[i-1])/delta;
            ss.AddSuffStat(0.5, 0.5*contrast);
        }
    }
};

