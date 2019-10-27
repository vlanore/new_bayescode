#pragma once

#include "bayes_toolbox/src/structure/node.hpp"
#include "bayes_toolbox/src/structure/model.hpp"
#include "lib/CodonSubMatrix.hpp"
// #include "Proxy.hpp"

TOKEN(codon_matrix_array)

struct sitecodonmatrix_sm {

    template<class NucMatrixSelector, class OmegaSelector, class Size>
    class MGOmegaCodonMatrixArray : public Proxy<MGOmegaCodonSubMatrix&, int> {

        std::vector<MGOmegaCodonSubMatrix> _mat;
        NucMatrixSelector _nucmatrix;
        OmegaSelector _omega;

        MGOmegaCodonSubMatrix& _get(int i) final { 
            assert(i < _mat.size());
            return _mat[i];
        }

      public:
        MGOmegaCodonMatrixArray(const CodonStateSpace* statespace, NucMatrixSelector nucmatrix, OmegaSelector omega, Size n) :
            _mat(n, MGOmegaCodonSubMatrix(statespace, &nucmatrix(0), omega(0), false)),
            _nucmatrix(nucmatrix),
            _omega(omega) {
                gather();
        }

        void gather() final {
            for (size_t i=0; i<_mat.size(); i++)  {
                _mat[i].SetOmega(_omega(i));
                _mat[i].CorruptMatrix();
            }
        }

        std::vector<MGOmegaCodonSubMatrix>& GetArray() {return _mat;}
    };

    template<class NucMatrixSelector, class OmegaSelector, class Size>
    static auto make(const CodonStateSpace* statespace, NucMatrixSelector nucmatrix, OmegaSelector omega, Size n)  {

        auto codon_matrix_array = 
            std::make_unique<MGOmegaCodonMatrixArray<NucMatrixSelector, OmegaSelector, Size>>(statespace, nucmatrix, omega, n);

        return make_model(codon_matrix_array_ = std::move(codon_matrix_array));
    }
};

