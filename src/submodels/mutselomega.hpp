#pragma once

#include "bayes_toolbox.hpp"
#include "lib/AAMutSelOmegaCodonSubMatrix.hpp"

struct mutselomega {

    using T = AAMutSelOmegaCodonSubMatrix;
    using param_decl = param_decl_t<param<nucmatrix, SubMatrix>, param<freqs, std::vector<double>>, param<real_a, spos_real>>;

    static void gather(T& mat, const SubMatrix& nucmat, const std::vector<double>& aafitness, double omega)  {
            mat.SetNucMatrix(&nucmat);
            mat.SetFitness(aafitness);
            mat.SetOmega(omega);
            mat.CorruptMatrix();
    }
};
