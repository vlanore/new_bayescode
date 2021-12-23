#pragma once

#include "bayes_toolbox.hpp"
#include "lib/T92SubMatrix.hpp"

struct t92 {

    using T = T92SubMatrix;
    using param_decl = param_decl_t<param<rate, spos_real>, param<prob, unit_real>>;

    static void gather(T& mat, spos_real kappa, unit_real gc) {
            mat.SetKappa(kappa);
            mat.SetGC(gc);
            mat.CorruptMatrix();
    }
};
