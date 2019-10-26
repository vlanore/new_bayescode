#pragma once

#include "bayes_toolbox/src/structure/node.hpp"
#include "bayes_toolbox/src/structure/model.hpp"
#include "lib/CodonSubMatrix.hpp"

TOKEN(codon_matrix)
TOKEN(codon_matrix_proxy)

class MGOmegaCodonMatrixProxy : public Proxy<MGOmegaCodonSubMatrix&> {
    MGOmegaCodonSubMatrix& _mat;
    double& _omega;

    MGOmegaCodonSubMatrix& _get() final { return _mat; }

  public:
    MGOmegaCodonMatrixProxy(MGOmegaCodonSubMatrix& mat, double& omega)
        : _mat(mat), _omega(omega) {}

    void gather() final {
        _mat.SetOmega(_omega);
        _mat.CorruptMatrix();
    }
};

struct codonmatrix_sm {

    template<class CodonSS, class NucMatrix, class Omega>
    static auto make(CodonSS* codon_statespace, NucMatrix& nucmatrix, Omega& omega)  {

        auto codon_matrix = std::make_unique<MGOmegaCodonSubMatrix>(codon_statespace, &nucmatrix, omega, false);
        auto codon_matrix_proxy = MGOmegaCodonMatrixProxy(*codon_matrix.get(), omega);

        return make_model(                                   //
            codon_matrix_ = std::move(codon_matrix),
            codon_matrix_proxy_ = codon_matrix_proxy
        );
    }
};

