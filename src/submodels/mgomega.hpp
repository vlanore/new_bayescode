#pragma once

#include "lib/CodonSubMatrix.hpp"

struct gtr {

    using T = GTRSubMatrix;
    using param_decl = param_decl_t<param<exchrates, std::vector<double>>, param<freqs, std::vector<double>>>;

    static void gather(T& mat, const std::vector<double>& exch_rates, const std::vector<double>& eq_freqs)  {
            std::cerr << "in gtr gather, set nuc\n";
            mat.CopyStationary(eq_freqs);
            mat.CorruptMatrix();
            std::cerr << "gather ok\n";
    }
};

struct mgomega {

    using T = MGOmegaCodonSubMatrix;
    using param_decl = param_decl_t<param<nucmatrix, SubMatrix>, param<rate, spos_real>>;

    static void gather(T& mat, const SubMatrix& nucmat, spos_real om)    {
            std::cerr << "in gather, set nuc\n";
            mat.SetNucMatrix(&nucmat);
            std::cerr << "set om\n";
            mat.SetOmega(om);
            std::cerr << "corrupt\n";
            mat.CorruptMatrix();
            std::cerr << "gather ok\n";
    }
};

