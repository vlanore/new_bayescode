
#pragma once

#include "Proxy.hpp"
#include "lib/CodonSuffStat.hpp"
#include "lib/GTRSubMatrix.hpp"
#include "lib/PhyloProcess.hpp"
#include "structure/suffstat.hpp"

struct pathss_factory {

    // a single path suffstat attached to a phyloprocess (site- and time-homogeneous)
    static auto make_path_suffstat(PhyloProcess& phyloprocess)    {
        auto path_suffstat = ss_factory::make_suffstat<PathSuffStat>(
                [&phyloprocess] (auto& ss)
                    { phyloprocess.AddPathSuffStat( [&ss] (int branch, int site) -> PathSuffStat& { return ss; } ); });

        return path_suffstat;
    }

    // an array of path suffstats attached to a phyloprocess (site-hetero and time-homogeneous)
    static auto make_site_path_suffstat(PhyloProcess& phyloprocess)    {
        auto site_path_suffstats = ss_factory::make_suffstat_array<PathSuffStat>(
                phyloprocess.GetNsite(),
                [&phyloprocess] (auto& site_ss)
                    { phyloprocess.AddPathSuffStat( [&site_ss] (int branch, int site) -> PathSuffStat& { return site_ss[site]; } ); });
        return site_path_suffstats;
    }

    // reducing site path suff stats into component path suff stats (not tried yet)
    static auto make_reduced_path_suffstat(int ncomp, Proxy<PathSuffStat,int>& site_path_suffstat, const std::vector<int>& alloc)  {
        auto comp_path_suffstats = ss_factory::make_suffstat_array<PathSuffStat>(
                ncomp,
                [&site_ss = site_path_suffstat, &z = alloc] (auto& comp_ss, int i) 
                    { comp_ss[z[i]].Add(site_ss.get(i)); },
                alloc.size());
        return comp_path_suffstats;
    }

    static auto make_bl_suffstats(PhyloProcess& phyloprocess)   {
        auto bl_suffstats = ss_factory::make_suffstat_array<PoissonSuffStat>(
                phyloprocess.GetNnode() - 1,
                [&phyloprocess] (auto& bl_ss)
                    { phyloprocess.AddLengthSuffStat( [&bl_ss] (int branch, int site) -> PoissonSuffStat& { return bl_ss[branch]; } ); });
        return bl_suffstats;
    }
};

