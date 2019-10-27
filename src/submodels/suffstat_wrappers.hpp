#pragma once

#include "Proxy.hpp"
#include "lib/CodonSuffStat.hpp"
#include "lib/GTRSubMatrix.hpp"
#include "lib/PhyloProcess.hpp"

// a single path suffstat attached to a phyloprocess (site- and time-homogeneous)
struct pathssw  {

    class PathSSW final : public Proxy<PathSuffStat&> {
        PathSuffStat _ss;
        PhyloProcess& _phyloprocess;

        PathSuffStat& _get() final { return _ss; }

      public:
        PathSSW(PhyloProcess& phyloprocess) : _phyloprocess(phyloprocess) {}

        void gather() final {
            auto& local_ss = _ss;
            local_ss.Clear();
            _phyloprocess.AddPathSuffStat( [&local_ss] (int branch, int site) -> PathSuffStat& { return local_ss; } );
        }
    };

    static auto make(PhyloProcess& phyloprocess)    {
        auto pathss = std::make_unique<PathSSW>(phyloprocess);
        return std::move(pathss);
    }
};

// an array of site-specific path suffstats attached to a phyloprocess (site-heterogeneous, time-homogeneous)
struct sitepathssw  {

    class SitePathSSW final : public Proxy<PathSuffStat&, int> {
        std::vector<PathSuffStat> _ss;
        PhyloProcess& _phyloprocess;

        PathSuffStat& _get(int i) final {
            assert(i < _ss.size());
            return _ss[i]; 
        }

      public:
        SitePathSSW(PhyloProcess& phyloprocess) : _ss(phyloprocess.GetNsite()), _phyloprocess(phyloprocess) {}

        void gather() final {
            for (size_t i=0; i<_ss.size(); i++) {
                _ss[i].Clear();
            }
            auto& local_ss = _ss;
            _phyloprocess.AddPathSuffStat( [&local_ss] (int branch, int site) -> PathSuffStat& { return local_ss[site]; } );
        }
    };

    static auto make(PhyloProcess& phyloprocess)    {
        auto sitepathss = std::make_unique<SitePathSSW>(phyloprocess);
        return std::move(sitepathss);
    }
};

// a single nucpathsuffstat collecting suffstats from
// either a single (codonmatrix,pathsuffstat) pair
// or an indexed series of (codonmatrix,pathsuffstat) pairs
// in both cases, through lambdas
struct nucpathssw {

    template<class CodonMatrixSelector, class PathSuffStatSelector>
    class NucPathSSW0 final : public Proxy<NucPathSuffStat&> {
        NucPathSuffStat _nucss;
        CodonMatrixSelector _codon_matrix_selector;
        PathSuffStatSelector _path_suffstat_selector;

        NucPathSuffStat& _get() final { return _nucss; }

      public:
        NucPathSSW0(const CodonStateSpace* statespace, 
                CodonMatrixSelector& codon_matrix_selector, 
                PathSuffStatSelector& path_suffstat_selector) : 

            _nucss(*statespace), 
            _codon_matrix_selector(codon_matrix_selector),
            _path_suffstat_selector(path_suffstat_selector) {}

        void gather() final {
            _nucss.Clear();
            _nucss.AddSuffStat(_codon_matrix_selector(), _path_suffstat_selector());
        }
    };

    template<class CodonMatrixSelector, class PathSuffStatSelector, class Size>
    class NucPathSSW1 final : public Proxy<NucPathSuffStat&> {
        NucPathSuffStat _nucss;
        CodonMatrixSelector _codon_matrix_selector;
        PathSuffStatSelector _path_suffstat_selector;
        Size _n;

        NucPathSuffStat& _get() final { return _nucss; }

      public:
        NucPathSSW1(const CodonStateSpace* statespace, 
                CodonMatrixSelector& codon_matrix_selector, 
                PathSuffStatSelector& path_suffstat_selector, 
                Size n) : 

            _nucss(*statespace), 
            _codon_matrix_selector(codon_matrix_selector),
            _path_suffstat_selector(path_suffstat_selector),
            _n(n) {}

        void gather() final {
            _nucss.Clear();
            for (size_t i=0; i<_n; i++)  {
                _nucss.AddSuffStat(_codon_matrix_selector(i), _path_suffstat_selector(i));
            }
        }
    };

    template<class CodonMatrixSelector, class PathSuffStatSelector>
    static auto make(const CodonStateSpace* codonstatespace, CodonMatrixSelector codonmat, PathSuffStatSelector pathss) {
            auto nucss = std::make_unique<NucPathSSW0<CodonMatrixSelector, PathSuffStatSelector>>
                (codonstatespace, codonmat, pathss);
            return std::move(nucss);
    }

    template<class CodonMatrixSelector, class PathSuffStatSelector, class Size>
    static auto make(const CodonStateSpace* codonstatespace, CodonMatrixSelector codonmat, PathSuffStatSelector pathss, Size n) {
            auto nucss = std::make_unique<NucPathSSW1<CodonMatrixSelector, PathSuffStatSelector, Size>>
                (codonstatespace, codonmat, pathss, n);
            return std::move(nucss);
    }
};

/*
class NucPathSSW final : public Proxy<NucPathSuffStat&> {
    NucPathSuffStat _nucss;
    NucCodonSubMatrix& _codon_submatrix;
    Proxy<PathSuffStat&>& _path_suffstat;

    NucPathSuffStat& _get() final { return _nucss; }

  public:
    NucPathSSW(NucCodonSubMatrix& codon_submatrix, Proxy<PathSuffStat&>& path_suffstat) : _nucss(*codon_submatrix.GetCodonStateSpace()), _codon_submatrix(codon_submatrix), _path_suffstat(path_suffstat) {}

    void gather() final {
        _nucss.Clear();
        _nucss.AddSuffStat(_codon_submatrix, _path_suffstat.get());
    }
};
*/

// =================================================================================================
class NucMatrixProxy : public Proxy<GTRSubMatrix&> {
    GTRSubMatrix& _mat;
    const std::vector<double>& _eq_freqs;

    GTRSubMatrix& _get() final { return _mat; }

  public:
    NucMatrixProxy(GTRSubMatrix& mat, const std::vector<double>& eq_freqs)
        : _mat(mat), _eq_freqs(eq_freqs) {}

    void gather() final {
        _mat.CopyStationary(_eq_freqs);
        _mat.CorruptMatrix();
    }
};

struct omega_suffstat_t {
    int count;
    double beta;
    bool operator==(const omega_suffstat_t& other) const {
        return count == other.count && beta == other.beta;
    }
};


// =================================================================================================
struct omegassw {

    template<class CodonMatrixSelector, class PathSuffStatSelector>
    class OmegaSSW0 final : public Proxy<omega_suffstat_t> {  // SSW = suff stat wrapper
        CodonMatrixSelector _codon_submatrix_selector;
        PathSuffStatSelector _path_suffstat_selector;
        OmegaPathSuffStat _ss;

        omega_suffstat_t _get() final { return {_ss.GetCount(), _ss.GetBeta()}; }

      public:
        OmegaSSW0(CodonMatrixSelector& codon_submatrix_selector, PathSuffStatSelector& path_suffstat_selector) :
            _codon_submatrix_selector(codon_submatrix_selector),
            _path_suffstat_selector(path_suffstat_selector) {}

        void gather() final {
            _ss.Clear();
            _ss.AddSuffStat(_codon_submatrix_selector(), _path_suffstat_selector());
        }
    };

    template<class CodonMatrixSelector, class PathSuffStatSelector, class Size>
    class OmegaSSW1 final : public Proxy<omega_suffstat_t> {  // SSW = suff stat wrapper
        CodonMatrixSelector _codon_submatrix_selector;
        PathSuffStatSelector _path_suffstat_selector;
        Size _n;
        OmegaPathSuffStat _ss;

        omega_suffstat_t _get() final { return {_ss.GetCount(), _ss.GetBeta()}; }

      public:
        OmegaSSW1(CodonMatrixSelector& codon_submatrix_selector, PathSuffStatSelector& path_suffstat_selector, Size n) :
            _codon_submatrix_selector(codon_submatrix_selector),
            _path_suffstat_selector(path_suffstat_selector),
            _n(n) {}

        void gather() final {
            _ss.Clear();
            for (Size i=0; i<_n; i++)   {
                _ss.AddSuffStat(_codon_submatrix_selector(i), _path_suffstat_selector(i));
            }
        }
    };

    template<class IndexSelector, class CodonMatrixSelector, class PathSuffStatSelector, class Size>
    class OmegaSSArrayW final : public Proxy<omega_suffstat_t, int> {  // SSW = suff stat wrapper
        IndexSelector _index_selector;
        CodonMatrixSelector _codon_submatrix_selector;
        PathSuffStatSelector _path_suffstat_selector;
        size_t _n;
        std::vector<OmegaPathSuffStat> _ss;

        omega_suffstat_t _get(int i) final {
            assert(i < _ss.size());
            return {_ss[i].GetCount(), _ss[i].GetBeta()}; 
        }

      public:
        OmegaSSArrayW(size_t k, IndexSelector& index_selector, CodonMatrixSelector& codon_submatrix_selector, PathSuffStatSelector& path_suffstat_selector, Size n):
            _index_selector(index_selector),
            _codon_submatrix_selector(codon_submatrix_selector),
            _path_suffstat_selector(path_suffstat_selector),
            _n(n),
            _ss(k) {}


        void gather() final {
            for (size_t i=0; i<_ss.size(); i++) {
                _ss[i].Clear();
            }
            for (size_t i=0; i<_n; i++)   {
                auto index = _index_selector(i);
                assert(index >= 0 && index < _ss.size());
                _ss[index].AddSuffStat(_codon_submatrix_selector(i), _path_suffstat_selector(i));
            }
        }
    };

    template<class CodonMatrixSelector, class PathSuffStatSelector>
    static auto make(CodonMatrixSelector codonmat, PathSuffStatSelector pathss) {
            auto omegass = std::make_unique<OmegaSSW0<CodonMatrixSelector, PathSuffStatSelector>>(codonmat, pathss);
            return std::move(omegass);
    }

    template<class CodonMatrixSelector, class PathSuffStatSelector, class Size>
    static auto make(CodonMatrixSelector codonmat, PathSuffStatSelector pathss, Size n) {
            auto omegass = std::make_unique<OmegaSSW1<CodonMatrixSelector, PathSuffStatSelector, Size>>(codonmat, pathss, n);
            return std::move(omegass);
    }

    template<class IndexSelector, class CodonMatrixSelector, class PathSuffStatSelector, class Size>
    static auto make(size_t k, IndexSelector index, CodonMatrixSelector codonmat, PathSuffStatSelector pathss, Size n) {
            auto omegass = std::make_unique<OmegaSSArrayW<IndexSelector, CodonMatrixSelector, PathSuffStatSelector, Size>>
                (k, index, codonmat, pathss, n);
            return std::move(omegass);
    }
};

// =================================================================================================
struct poisson_suffstat_t {
    int count;
    double beta;
    bool operator==(const poisson_suffstat_t& other) const {
        return count == other.count && beta == other.beta;
    }
};

class BranchArrayPoissonSSW final : public Proxy<poisson_suffstat_t, int> {
    std::vector<PoissonSuffStat> _ss;
    PhyloProcess& _phyloprocess;

    poisson_suffstat_t _get(int i) final {
        auto& local_ss = _ss[i];
        return {local_ss.GetCount(), local_ss.GetBeta()};
    }

  public:
    BranchArrayPoissonSSW(const Tree& tree, PhyloProcess& phyloprocess)
        : _ss(tree.nb_nodes() - 1), _phyloprocess(phyloprocess) {}

    void gather() final {
        for (auto i : _ss) i.Clear();
        auto& local_ss = _ss;
        _phyloprocess.AddLengthSuffStat( [&local_ss](int branch, int site) -> PoissonSuffStat& {return local_ss[branch];} );
    }
};

class OmegaSSW final : public Proxy<omega_suffstat_t> {  // SSW = suff stat wrapper
    const OmegaCodonSubMatrix& _codon_submatrix;
    Proxy<PathSuffStat&>& _path_suffstat;
    OmegaPathSuffStat _ss;

    omega_suffstat_t _get() final { return {_ss.GetCount(), _ss.GetBeta()}; }

  public:
    OmegaSSW(const OmegaCodonSubMatrix& codon_submatrix, Proxy<PathSuffStat&>& pathsuffstat)
        : _codon_submatrix(codon_submatrix), _path_suffstat(pathsuffstat) {}

    void gather() final {
        _ss.Clear();
        _ss.AddSuffStat(_codon_submatrix, _path_suffstat.get());
    }
};

class SiteOmegaSSW final : public Proxy<omega_suffstat_t, int> {  // SSW = suff stat wrapper
    const std::vector<MGOmegaCodonSubMatrix>& _codon_submatrix_array;
    // Proxy<OmegaCodonSubMatrix&, int>& _codon_submatrix_array;
    Proxy<PathSuffStat&, int>& _path_suffstat_array;
    std::vector<OmegaPathSuffStat> _ss;

    omega_suffstat_t _get(int i) final {
        assert(i < _ss.size());
        return {_ss[i].GetCount(), _ss[i].GetBeta()}; 
    }

  public:
    SiteOmegaSSW(size_t nsite, const std::vector<MGOmegaCodonSubMatrix>& codon_submatrix_array, Proxy<PathSuffStat&, int>& pathsuffstatarray)
    // SiteOmegaSSW(size_t nsite, Proxy<OmegaCodonSubMatrix&, int>& codon_submatrix_array, Proxy<PathSuffStat&, int>& pathsuffstatarray)
        : _codon_submatrix_array(codon_submatrix_array), _path_suffstat_array(pathsuffstatarray), _ss(nsite) {}

    void gather() final {
        for (size_t i=0; i<_ss.size(); i++) {
            // std::cerr << i << '\t' << _ss.size() << '\n';
            _ss[i].Clear();

            // _ss[i].AddSuffStat(_codon_submatrix_array.get(i), _path_suffstat_array.get(i));
            _ss[i].AddSuffStat(_codon_submatrix_array[i], _path_suffstat_array.get(i));
        }
    }
};
