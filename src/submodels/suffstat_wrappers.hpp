
#pragma once

#include "Proxy.hpp"
#include "lib/CodonSuffStat.hpp"
#include "lib/GTRSubMatrix.hpp"
#include "lib/PhyloProcess.hpp"
#include "CodonSuffStat.hpp"
#include "OccupancySuffStat.hpp"
#include "structure/suffstat.hpp"

struct pathss_factory {

    // branch lengths suff stats (valid for all models)
    static auto make_bl_suffstat(PhyloProcess& phyloprocess)   {
        auto bl_suffstats = ss_factory::make_suffstat_array<PoissonSuffStat>(
                phyloprocess.GetNnode() - 1,
                [&phyloprocess] (auto& bl_ss)
                    { phyloprocess.AddLengthSuffStat( [&bl_ss] (int branch, int site) -> PoissonSuffStat& { return bl_ss[branch]; } ); });
        return bl_suffstats;
    }

    // a single path suffstat attached to a phyloprocess (site-homogeneous)
    static auto make_path_suffstat(PhyloProcess& phyloprocess)    {
        auto path_suffstat = ss_factory::make_suffstat<PathSuffStat>(
                [&phyloprocess] (auto& ss)
                    { phyloprocess.AddPathSuffStat( [&ss] (int branch, int site) -> PathSuffStat& { return ss; } ); });

        return path_suffstat;
    }

    // an array of path suffstats attached to a phyloprocess (site-heterogeneous)
    static auto make_site_path_suffstat(PhyloProcess& phyloprocess)    {
        auto site_path_suffstats = ss_factory::make_suffstat_array<PathSuffStat>(
                phyloprocess.GetNsite(),
                [&phyloprocess] (auto& site_ss)
                    { phyloprocess.AddPathSuffStat( [&site_ss] (int branch, int site) -> PathSuffStat& { return site_ss[site]; } ); });
        return site_path_suffstats;
    }

    // a nuc path suff stat, based on a single codon matrix and a single path suff stat 
    // (used in site-homogeneous models)
    static auto make_nucpath_suffstat(const CodonStateSpace* codon_statespace, NucCodonSubMatrix& mat, Proxy<PathSuffStat&>& pss)    {
        auto nucpath_ss = ss_factory::make_suffstat_with_init<NucPathSuffStat>(
                {*codon_statespace},
                [&mat, &pss] (auto& nucss) { nucss.AddSuffStat(mat, pss.get()); });
        return nucpath_ss;
    }

    // a nuc path suffstat, based on an array of codon matrices and a corresponding array of path suffstats
    // (used in site- and mixture models)
    template<class Matrix>
    static auto make_nucpath_suffstat(const CodonStateSpace* codon_statespace, std::vector<Matrix>& mat, Proxy<PathSuffStat&, int>& pss)    {
        auto nucpath_ss = ss_factory::make_suffstat_with_init<NucPathSuffStat>(
                {*codon_statespace},
                [&mat, &pss] (auto& nucss, int i) { nucss.AddSuffStat(mat[i], pss.get(i)); },
                mat.size());
        return nucpath_ss;
    }

    // an omega path suffstat based on a single matrix and path suffstat (site-homogeneous model)
    static auto make_omega_suffstat(OmegaCodonSubMatrix& mat, Proxy<PathSuffStat&>& pss)    {
        auto omega_ss = ss_factory::make_suffstat<OmegaPathSuffStat>(
                [&mat, &pss] (auto& omss) { omss.AddSuffStat(mat, pss.get()); });
        return omega_ss;
    }

    // an array of omega path suffstats, based on an array of matrices and an array path suffstats
    template<class Matrix>
    static auto make_omega_suffstat(std::vector<Matrix>& mat, Proxy<PathSuffStat&, int>& pss) {
        auto site_omega_ss = ss_factory::make_suffstat_array<OmegaPathSuffStat>(
                mat.size(),
                [&mat, &pss] (auto& omss, int i) { omss[i].AddSuffStat(mat[i], pss.get(i)); },
                mat.size());
        return site_omega_ss;
    }
};

struct mixss_factory    {

    // reducing site suff stats into component suff stats, based on allocation vector
    template<class SS>
    static auto make_reduced_suffstat(size_t ncomp, Proxy<SS&,int>& site_ss, const std::vector<size_t>& alloc)  {
        auto comp_suffstats = ss_factory::make_suffstat_array<SS>(
                ncomp,
                [&site_ss, &alloc] (auto& comp_ss, int i) 
                    { comp_ss[alloc[i]].Add(site_ss.get(i)); },
                alloc.size());
        return comp_suffstats;
    }

    static auto make_alloc_suffstat(size_t ncomp, std::vector<size_t>& alloc)    {
        auto alloc_ss = ss_factory::make_suffstat_with_init<OccupancySuffStat>(
                {ncomp},
                [&alloc] (auto& occss) { occss.AddSuffStat(alloc); });
        return alloc_ss;
    }

    template<class V, class SS>
    static auto make_alloc_logprob(std::vector<V>& val, Proxy<SS&>& ss)    {
        auto alloc_logprob = 
            [&val, &ss] () {
                auto lambda = [&val, &s=ss.get()] (int k) {return s.GetLogProb(val[k]);};
                return lambda;
            };
        return alloc_logprob;
    }

    template<class V, class SS>
    static auto make_alloc_logprob(std::vector<V>& val, Proxy<SS&, int>& ss)    {
        auto alloc_logprob = 
            [&val, &ss] (int i) {
                auto lambda = [&val, &s = ss.get(i)] (int k) {return s.GetLogProb(val[k]);};
                return lambda;
            };
        return alloc_logprob;
    }
};
