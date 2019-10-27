#pragma once

#include "bayes_toolbox/src/structure/node.hpp"
#include "bayes_toolbox/src/structure/model.hpp"
#include "lib/CodonSubMatrix.hpp"

TOKEN(codon_matrix_array)
// TOKEN(nuccodon_matrix_array_proxy)
// TOKEN(omegacodon_matrix_array_proxy)
TOKEN(mgomegacodon_matrix_array_proxy)

class MGOmegaCodonMatrixArrayProxy : public Proxy<MGOmegaCodonSubMatrix&, int> {
    std::vector<MGOmegaCodonSubMatrix>& _mat;
    std::vector<double>& _omega;

    MGOmegaCodonSubMatrix& _get(int i) final { 
        assert(i < _mat.size());
        return _mat[i];
    }

  public:
    MGOmegaCodonMatrixArrayProxy(std::vector<MGOmegaCodonSubMatrix>& mat, std::vector<double>& omega)
        : _mat(mat), _omega(omega) {}

    void gather() final {
        for (size_t i=0; i<_mat.size(); i++)  {
            _mat[i].SetOmega(_omega[i]);
            _mat[i].CorruptMatrix();
        }
    }
};

class OmegaCodonMatrixArrayProxy : public Proxy<OmegaCodonSubMatrix&, int> {
    std::vector<MGOmegaCodonSubMatrix>& _mat;
    std::vector<double>& _omega;

    OmegaCodonSubMatrix& _get(int i) final { 
        assert(i < _mat.size());
        return _mat[i];
    }

  public:
    OmegaCodonMatrixArrayProxy(std::vector<MGOmegaCodonSubMatrix>& mat, std::vector<double>& omega)
        : _mat(mat), _omega(omega) {}

    void gather() final {
        for (size_t i=0; i<_mat.size(); i++)  {
            _mat[i].SetOmega(_omega[i]);
            _mat[i].CorruptMatrix();
        }
    }
};

class NucCodonMatrixArrayProxy : public Proxy<NucCodonSubMatrix&, int> {
    std::vector<MGOmegaCodonSubMatrix>& _mat;
    // const SubMatrix& _nucmatrix;

    NucCodonSubMatrix& _get(int i) final {
        assert(i < _mat.size());
        return _mat[i];
    }

  public:
    NucCodonMatrixArrayProxy(std::vector<MGOmegaCodonSubMatrix>& mat) : _mat(mat) {}
    /*
    NucCodonMatrixArrayProxy(std::vector<MGOmegaCodonSubMatrix>& mat, const SubMatrix& nucmatrix) :
        _mat(mat), _nucmat(nucmat) {}
    */

    void gather() final {
        std::cerr << "error: in nuccodon gather\n";
        exit(1);
    }
};

struct sitecodonmatrix_sm {

    template<class CodonSS, class NucMatrix, class OmegaArray>
    static auto make(CodonSS* codon_statespace, NucMatrix& nucmatrix, OmegaArray& omega_array)  {

        auto codon_matrix_array = std::make_unique<std::vector<MGOmegaCodonSubMatrix>>(omega_array.size(), MGOmegaCodonSubMatrix(codon_statespace, &nucmatrix, 1.0, false));
        for (size_t i=0; i<omega_array.size(); i++)  {
            (*codon_matrix_array.get())[i].SetOmega(omega_array[i]);
            (*codon_matrix_array.get())[i].CorruptMatrix();
        }

        // auto nuccodon_matrix_array_proxy = NucCodonMatrixArrayProxy(*codon_matrix_array.get());
        // auto omegacodon_matrix_array_proxy = OmegaCodonMatrixArrayProxy(*codon_matrix_array.get(), omega_array);
        auto mgomegacodon_matrix_array_proxy = MGOmegaCodonMatrixArrayProxy(*codon_matrix_array.get(), omega_array);

        return make_model(
            codon_matrix_array_ = std::move(codon_matrix_array),
            // nuccodon_matrix_array_proxy_ = nuccodon_matrix_array_proxy,
            // omegacodon_matrix_array_proxy_ = omegacodon_matrix_array_proxy,
            mgomegacodon_matrix_array_proxy_ = mgomegacodon_matrix_array_proxy
        );
    }
};

