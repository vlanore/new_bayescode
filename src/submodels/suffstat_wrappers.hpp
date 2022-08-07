
#pragma once

#include "Proxy.hpp"
#include "lib/CodonSuffStat.hpp"
#include "lib/CodonSubMatrix.hpp"
#include "lib/GTRSubMatrix.hpp"
#include "lib/PhyloProcess.hpp"
#include "CodonSuffStat.hpp"
#include "dSOmegaPathSuffStat.hpp"
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

    static auto make_node_path_suffstat(PhyloProcess& phyloprocess) {
        auto node_path_suffstats = ss_factory::make_suffstat_array<PathSuffStat>(
                phyloprocess.GetNnode(),
                [&phyloprocess] (auto& node_ss)
                    { phyloprocess.AddPathSuffStat( [&node_ss] (int node, int site) -> PathSuffStat& { return node_ss[node]; } ); });
        return node_path_suffstats;
    }

    // an array of path suffstats attached to a phyloprocess (site-branch-heterogeneous)
    static auto make_site_node_path_suffstat(PhyloProcess& phyloprocess)    {
        auto site_node_path_suffstats = ss_factory::make_suffstat_matrix<PathSuffStat>(
                phyloprocess.GetNsite(),
                phyloprocess.GetNnode(),
                [&phyloprocess] (auto& site_node_ss)
                    { phyloprocess.AddPathSuffStat( [&site_node_ss] (int node, int site) -> PathSuffStat& { return site_node_ss[site][node]; } ); });
        return site_node_path_suffstats;
    }


    template<class BL>
    static auto make_node_relpath_suffstat(const CodonStateSpace* statespace, Proxy<PathSuffStat&, size_t>& pss, BL bl) {
        auto relpath_suffstats = ss_factory::make_suffstat_array_with_init<RelativePathSuffStat>(
                pss.size(),
                {statespace->GetNstate()},
                [&pss, bl] (auto& rpss)  {
                    for (size_t node=0; node<pss.size(); node++) {
                        rpss[node].Add(pss.get(node), node ? bl(node-1) : 1.0); 
                    }
                });
        return relpath_suffstats;
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
    static auto make_nucpath_suffstat(const CodonStateSpace* codon_statespace, std::vector<Matrix>& mat, Matrix& root_mat, Proxy<PathSuffStat&, size_t>& pss)    {
        auto nucpath_ss = ss_factory::make_suffstat_with_init<NucPathSuffStat>(
                {*codon_statespace},
                [&mat, &root_mat, &pss] (auto& nucss, int node) {
                    if (node)   {
                        nucss.AddSuffStat(mat[node-1], pss.get(node));
                    }
                    else    {
                        nucss.AddSuffStat(root_mat, pss.get(node));
                    }
                },
                mat.size() + 1);
        return nucpath_ss;
    }

    template<class Matrix, class BL>
    static auto make_nucpath_suffstat(const CodonStateSpace* codon_statespace, std::vector<Matrix>& mat, Matrix& root_mat, Proxy<RelativePathSuffStat&, size_t>& rpss, BL bl)    {
        auto nucpath_ss = ss_factory::make_suffstat_with_init<NucPathSuffStat>(
                {*codon_statespace},
                [&mat, &root_mat, &rpss, bl] (auto& nucss, int node) {
                    if (node)   {
                        nucss.AddSuffStat(mat[node-1], rpss.get(node), bl(node-1));
                    }
                    else    {
                        nucss.AddSuffStat(root_mat, rpss.get(node), 0);
                    }
                },
                mat.size() + 1);
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
    static auto make_omega_suffstat(std::vector<Matrix>& mat, Proxy<PathSuffStat&, size_t >& pss) {
        auto omega_ss = ss_factory::make_suffstat_array<OmegaPathSuffStat>(
                mat.size(),
                [&mat, &pss] (auto& omss, int i) { omss[i].AddSuffStat(mat[i], pss.get(i)); },
                mat.size());
        return omega_ss;
    }

    template<class Matrix, class OM>
    static auto make_dsomega_suffstat(std::vector<Matrix>& mat, Proxy<RelativePathSuffStat&, size_t>& rpss, OM om)  {
        auto dsom_ss = ss_factory::make_suffstat_array<dSOmegaPathSuffStat>(
                mat.size(),
                [&mat, &rpss, om, nbranch=mat.size()] (auto& omss) {
                    for (size_t branch=0; branch<nbranch; branch++) {
                        omss[branch].AddSuffStat(mat[branch], rpss.get(branch+1), om(branch)); 
                    }
                });
        return dsom_ss;
    }


    static auto make_mapping_dsom_suffstats(std::string filename)   {

        std::ifstream tis(filename.c_str());

        std::string ds_count;
        tis >> ds_count;
        auto ds_count_stream = std::stringstream(ds_count);
        NHXParser ds_count_parser(ds_count_stream);
        auto tree = make_from_parser(ds_count_parser);

        size_t nb = tree->nb_nodes()-1;
        std::vector<std::vector<double>> suffstats(4, std::vector<double>(nb, 0));

        auto branch_ds_count = node_container_from_parser<std::string>(
            ds_count_parser, [](int i, const AnnotatedTree& t) { return t.tag(i, "length"); });

        for (size_t branch=0; branch<nb; branch++)  {
            suffstats[0][branch] = std::atof(branch_ds_count[branch+1].c_str());
        }

        std::string ds_norm;
        tis >> ds_norm;
        auto ds_norm_stream = std::stringstream(ds_norm);
        NHXParser ds_norm_parser(ds_norm_stream);

        auto branch_ds_norm = node_container_from_parser<std::string>(
            ds_norm_parser, [](int i, const AnnotatedTree& t) { return t.tag(i, "length"); });

        for (size_t branch=0; branch<nb; branch++)  {
            suffstats[1][branch] = std::atof(branch_ds_norm[branch+1].c_str());
        }

        std::string dn_count;
        tis >> dn_count;
        auto dn_count_stream = std::stringstream(dn_count);
        NHXParser dn_count_parser(dn_count_stream);

        auto branch_dn_count = node_container_from_parser<std::string>(
            dn_count_parser, [](int i, const AnnotatedTree& t) { return t.tag(i, "length"); });

        for (size_t branch=0; branch<nb; branch++)  {
            suffstats[2][branch] = std::atof(branch_dn_count[branch+1].c_str());
        }

        std::string dn_norm;
        tis >> dn_norm;
        auto dn_norm_stream = std::stringstream(dn_norm);
        NHXParser dn_norm_parser(dn_norm_stream);

        auto branch_dn_norm = node_container_from_parser<std::string>(
            dn_norm_parser, [](int i, const AnnotatedTree& t) { return t.tag(i, "length"); });

        for (size_t branch=0; branch<nb; branch++)  {
            suffstats[3][branch] = std::atof(branch_dn_norm[branch+1].c_str());
        }

        return suffstats;
    }

    static auto make_dsom_suffstat_from_mappings(std::vector<std::vector<double>>& suffstat, double f)    {
        auto dsom_ss = ss_factory::make_suffstat_array<dSOmegaPathSuffStat>(
            suffstat[0].size(),
            [&suffstat, f] (auto& omss)    {
                for (size_t branch=0; branch<suffstat[0].size(); branch++) {
                    omss[branch].Add(f*suffstat[0][branch], f*suffstat[1][branch], f*suffstat[2][branch], f*suffstat[3][branch]);
                }
            });
        (*dsom_ss).gather();
        return dsom_ss;
    }

    template<class SynRate>
    static auto make_omega_suffstat(SynRate synrate, Proxy<dSOmegaPathSuffStat&, size_t>& dsomss)    {
        auto omega_ss = ss_factory::make_suffstat<OmegaPathSuffStat>(
                [synrate, &dsomss] (auto& omss, int branch) { 
                    auto& ss = dsomss.get(branch);
                    omss.PoissonSuffStat::AddSuffStat(ss.GetNonSynCount(), synrate(branch) * ss.GetNonSynBeta());
                },
                dsomss.size());
        return omega_ss;
    }
};

struct mixss_factory    {

    // reducing site suff stats into component suff stats, based on allocation vector
    template<class SS>
    static auto make_reduced_suffstat(size_t ncomp, Proxy<SS&,size_t>& site_ss, const std::vector<size_t>& alloc)  {
        auto comp_suffstats = ss_factory::make_suffstat_array<SS>(
                ncomp,
                [&site_ss, &alloc] (auto& comp_ss, int i) 
                    { comp_ss[alloc[i]].Add(site_ss.get(i)); },
                alloc.size());
        return comp_suffstats;
    }

    // reducing site suff stats into bidim component suff stats, based on 2 allocation vectors
    template<class SS>
    static auto make_reduced_suffstat(size_t ncomp1, size_t ncomp2, Proxy<SS&,size_t>& site_ss, const std::vector<size_t>& alloc1, const std::vector<size_t>& alloc2)  {
        assert(alloc1.size() == alloc2.size());
        auto comp_suffstats = ss_factory::make_suffstat_matrix<SS>(
                ncomp1, ncomp2,
                [&site_ss, &alloc1, &alloc2] (auto& comp_ss, int i) 
                    { comp_ss[alloc1[i]][alloc2[i]].Add(site_ss.get(i)); },
                alloc1.size());
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
    static auto make_alloc_logprob(std::vector<V>& val, Proxy<SS&, size_t>& ss)    {
        auto alloc_logprob = 
            [&val, &ss] (int i) {
                auto lambda = [&val, &s = ss.get(i)] (int k) {return s.GetLogProb(val[k]);};
                return lambda;
            };
        return alloc_logprob;
    }
};
