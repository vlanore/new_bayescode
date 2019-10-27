#pragma once

#include "bayes_toolbox/src/structure/node.hpp"
#include "bayes_toolbox/src/structure/model.hpp"
#include "lib/CodonSubMatrix.hpp"

TOKEN(mg_omega_proxy)

struct mg_omega {

    template<class NucMatrixSelector, class OmegaSelector>
    class MGOmegaCodonMatrixProxy : public Proxy<MGOmegaCodonSubMatrix&> {

        MGOmegaCodonSubMatrix _mat;
        NucMatrixSelector _nucmatrix;
        OmegaSelector _omega;

        MGOmegaCodonSubMatrix& _get() final { 
            return _mat;
        }

      public:
        MGOmegaCodonMatrixProxy(const CodonStateSpace* statespace, NucMatrixSelector nucmatrix, OmegaSelector omega) :
            _mat(statespace, &nucmatrix(), omega(), false),
            _nucmatrix(nucmatrix),
            _omega(omega) {
                gather();
        }

        void gather() final {
            _mat.SetOmega(_omega());
            _mat.CorruptMatrix();
        }
    };

    template<class NucMatrixSelector, class OmegaSelector, class Size>
    class MGOmegaCodonMatrixArrayProxy : public Proxy<MGOmegaCodonSubMatrix&, int> {

        std::vector<MGOmegaCodonSubMatrix> _mat;
        NucMatrixSelector _nucmatrix;
        OmegaSelector _omega;

        MGOmegaCodonSubMatrix& _get(int i) final { 
            assert(i < _mat.size());
            return _mat[i];
        }

      public:
        MGOmegaCodonMatrixArrayProxy(const CodonStateSpace* statespace, NucMatrixSelector nucmatrix, OmegaSelector omega, Size n) :
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

    // make a simple proxy
    template<class NucMatrixSelector, class OmegaSelector>
    static auto make(const CodonStateSpace* statespace, NucMatrixSelector nucmatrix, OmegaSelector omega)   {

        auto codon_matrix = 
            std::make_unique<MGOmegaCodonMatrixProxy<NucMatrixSelector, OmegaSelector>>(statespace, nucmatrix, omega);

        return make_model(mg_omega_proxy_ = std::move(codon_matrix));
    }

    // make an array proxy
    template<class NucMatrixSelector, class OmegaSelector, class Size>
    static auto make(const CodonStateSpace* statespace, NucMatrixSelector nucmatrix, OmegaSelector omega, Size n)  {

        auto codon_matrix_array = 
            std::make_unique<MGOmegaCodonMatrixArrayProxy<NucMatrixSelector, OmegaSelector, Size>>(statespace, nucmatrix, omega, n);

        return make_model(mg_omega_proxy_ = std::move(codon_matrix_array));
    }
};

