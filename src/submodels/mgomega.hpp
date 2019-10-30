#pragma once

#include "bayes_toolbox.hpp"
#include "lib/CodonSubMatrix.hpp"

struct mgomega {

    using T = MGOmegaCodonSubMatrix;
    using param_decl = param_decl_t<param<nucmatrix, SubMatrix>, param<rate, spos_real>>;

    static void gather(T& mat, const SubMatrix& nucmat, spos_real om)    {
            mat.SetNucMatrix(&nucmat);
            mat.SetOmega(om);
            mat.CorruptMatrix();
    }
};

