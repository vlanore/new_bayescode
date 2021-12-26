#pragma once

#include "bayes_toolbox.hpp"
#include "lib/InverseWishart.hpp"

struct invwishart {

    using T = InverseWishart;
    using param_decl = param_decl_t<param<freqs , std::vector<double>>>;

    template <typename Gen>
    static void draw(T& x, const std::vector<double>& kappa, Gen& gen)    {
        x.Sample(kappa);
    }

    static real logprob(const T& x, const std::vector<double>& kappa)   {
        return x.GetLogProb(kappa);
    }

    template <class SS, typename Gen>
    static void gibbs_resample(T& x, SS& ss, const std::vector<double>& kappa, Gen& gen)    {
        return x.GibbsResample(kappa, ss);
    }
};

