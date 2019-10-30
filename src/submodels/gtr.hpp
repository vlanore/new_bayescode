#pragma once

#include "bayes_toolbox.hpp"
#include "lib/GTRSubMatrix.hpp"

struct gtr {

    using T = GTRSubMatrix;
    using param_decl = param_decl_t<param<exchrates, std::vector<double>>, param<freqs, std::vector<double>>>;

    static void gather(T& mat, const std::vector<double>& exch_rates, const std::vector<double>& eq_freqs)  {
            mat.CopyStationary(eq_freqs);
            mat.CorruptMatrix();
    }
};
