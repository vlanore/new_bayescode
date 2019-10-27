#pragma once

#include "Proxy.hpp"
#include "lib/CodonSuffStat.hpp"
#include "lib/GTRSubMatrix.hpp"
#include "lib/PhyloProcess.hpp"

// =================================================================================================
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

// =================================================================================================
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

// =================================================================================================
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

// =================================================================================================

class ArrayCollectingNucPathSSW final : public Proxy<NucPathSuffStat&> {
    NucPathSuffStat _nucss;
    size_t _nsite;
    const std::vector<MGOmegaCodonSubMatrix>& _codon_submatrix_array;
    // Proxy<NucCodonSubMatrix&, int>& _codon_submatrix_array;
    Proxy<PathSuffStat&, int>& _path_suffstat_array;

    NucPathSuffStat& _get() final { return _nucss; }

  public:
    ArrayCollectingNucPathSSW(size_t nsite,
            const CodonStateSpace& codonstatespace,
            // Proxy<NucCodonSubMatrix&, int>& codon_submatrix_array,
            const std::vector<MGOmegaCodonSubMatrix>& codon_submatrix_array,
            Proxy<PathSuffStat&, int>& path_suffstat_array) :

        _nucss(codonstatespace),
        _nsite(nsite),
        _codon_submatrix_array(codon_submatrix_array),
        _path_suffstat_array(path_suffstat_array) {}

    void gather() final {
        _nucss.Clear();
        for (size_t i=0; i<_nsite; i++)  {
            // _nucss.AddSuffStat(_codon_submatrix_array.get(i), _path_suffstat_array.get(i));
            _nucss.AddSuffStat(_codon_submatrix_array[i], _path_suffstat_array.get(i));
        }
    }
};

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

// =================================================================================================
struct omega_suffstat_t {
    int count;
    double beta;
    bool operator==(const omega_suffstat_t& other) const {
        return count == other.count && beta == other.beta;
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
            /*
            std::cerr << "clear ok\n";
            std::cerr << _codon_submatrix_array.get(i).GetNstate() << '\n';
            std::cerr << _codon_submatrix_array.get(i).GetOmega() << '\n';
            */

            // _ss[i].AddSuffStat(_codon_submatrix_array.get(i), _path_suffstat_array.get(i));
            _ss[i].AddSuffStat(_codon_submatrix_array[i], _path_suffstat_array.get(i));
        }
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

