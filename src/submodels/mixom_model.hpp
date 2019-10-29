#pragma once

#include <cmath>
#include "components/ChainCheckpoint.hpp"
#include "components/ChainDriver.hpp"
#include "components/ConsoleLogger.hpp"
#include "components/InferenceAppArgParse.hpp"
#include "components/MoveScheduler.hpp"
#include "components/StandardTracer.hpp"
#include "components/restart_check.hpp"
#include "data_preparation.hpp"
#include "lib/CodonSubMatrix.hpp"
#include "lib/CodonSuffStat.hpp"
#include "lib/PoissonSuffStat.hpp"
#include "submodels/branchlength_sm.hpp"
#include "submodels/move_reporter.hpp"
#include "submodels/nucrates_sm.hpp"
#include "submodels/siteomega_sm.hpp"
#include "distributions/dirichlet.hpp"
#include "distributions/categorical.hpp"
#include "submodels/mg_omega.hpp"
#include "submodels/submodel_external_interface.hpp"
#include "submodels/suffstat_wrappers.hpp"
#include "structure/suffstat.hpp"

TOKEN(branch_lengths)
TOKEN(nuc_rates)

TOKEN(mixture_weights)
TOKEN(mixture_allocs)

TOKEN(omega_array)
TOKEN(codon_submatrix_array)

TOKEN(phyloprocess)

TOKEN(bl_suffstats)

TOKEN(site_path_suffstats)
TOKEN(comp_path_suffstats)
TOKEN(nucpath_suffstats)
TOKEN(comp_omegapath_suffstats)

template<class Alloc>
auto n_to_mix(Alloc& alloc) {
    return [&z = get<value>(alloc)] (int site) {return z[site];};
}

template<class Alloc, class C>
auto n_to_mix(Proxy<C,int>& prox, Alloc& alloc)  {
    return [&m = prox, &z = get<value>(alloc)] (int site) {return m.get(z[site]);};
}

template<class Alloc, class C>
auto mn_to_mixn(Proxy<C,int>& prox, Alloc& alloc)  {
    return [&m = prox, &z = get<value>(alloc)] (int branch, int site) {return m.get(z[site]);};
}


struct mixom {
    // =============================================================================================
    template <class Gen>
    static auto make(PreparedData& data, size_t ncomp, Gen& gen) {

        size_t nsite = data.alignment.GetNsite();

        // bl : iid gamma across sites, with constant hyperparams
        auto branch_lengths =
            branchlengths_sm::make(data.parser, *data.tree, one_to_const(0.1), one_to_const(1.0));

        // nuc exch rates and eq freqs: uniform dirichlet
        // also creates the gtr matrix
        auto nuc_rates = nucrates_sm::make(one_to_const(normalize({1, 1, 1, 1, 1, 1})),
            one_to_const(1. / 6), one_to_const(normalize({1, 1, 1, 1})), one_to_const(1. / 4), gen);

        // mixture weights
        std::vector<double> weights_hyper(ncomp, 1.0/ncomp);
        auto weights = make_node<dirichlet>(one_to_const(weights_hyper));
        set_value(weights, std::vector<double>(ncomp, 1.0/ncomp));
        auto alloc = make_node_array<categorical>(nsite, n_to_one(weights));

        // omega: iid gamma across sites, with constant hyperparameters
        auto omega = siteomega_sm::make(ncomp, one_to_const(1.0), one_to_const(1.0), gen);

        auto codon_statespace =
            dynamic_cast<const CodonStateSpace*>(data.alignment.GetStateSpace());

        // an array of MG Omega Codon matrices, with same nucrates but each with its own omega
        auto codon_submatrix_array = mg_omega::make(
                codon_statespace, 
                n_to_one(get<nuc_matrix>(nuc_rates)),
                n_to_n(get<site_omega_array, value>(omega)),
                ncomp
            );

        /*
        auto codon_submatrix_array = make_dnode_array_from<MGOmegaCodonSubMatrix>(
                ncomp,
                {codon_statespace},
                n_to_one(get<nuc_matrix>(nuc_rates)),
                n_to_n(get<site_omega_array, value>(omega))
            );
            */

        // phyloprocess
        auto phyloprocess = std::make_unique<PhyloProcess>(data.tree.get(), &data.alignment,
            n_to_n(get<bl_array, value>(branch_lengths)),
            n_to_const(1.0),
            mn_to_mixn(*codon_submatrix_array, alloc),
            n_to_mix(*codon_submatrix_array, alloc),
            nullptr);

        phyloprocess->Unfold();
        std::cerr << "lnl : " << phyloprocess->GetLogLikelihood() << '\n';

        // suff stats

        // branch lengths
        BranchArrayPoissonSSW bl_suffstats{*data.tree, *phyloprocess};

        // site path suff stats
        auto site_path_suffstats = sitepathssw::make(*phyloprocess);

        auto comp_path_suffstats = ss_factory::make_suffstat_array<PathSuffStat>(
                ncomp,
                [&site_ss = *site_path_suffstats, &z = get<value>(alloc)] (auto& comp_ss, int i) 
                    { comp_ss[z[i]].Add(site_ss.get(i)); },
                nsite);

        auto nucpath_ssw = ss_factory::make_suffstat_with_init<NucPathSuffStat>(
                {*codon_statespace},
                [&mat = *codon_submatrix_array, &pss = *comp_path_suffstats] (auto& nucss, int i) 
                    { nucss.AddSuffStat(mat.get(i), pss.get(i)); },
                ncomp);

        auto comp_omega_ssw = ss_factory::make_suffstat_array<OmegaPathSuffStat>(
                ncomp,
                [&mat = *codon_submatrix_array, &pss = *comp_path_suffstats] (auto& omss, int i) 
                    { omss[i].AddSuffStat(mat.get(i), pss.get(i)); },
                ncomp);

        return make_model(                              //
            // codon_statespace_ = codon_statespace,       //
            branch_lengths_ = move(branch_lengths),     //
            nuc_rates_ = move(nuc_rates),               //

            mixture_weights_ = move(weights),
            mixture_allocs_ = move(alloc),

            omega_array_ = move(omega),         //
            codon_submatrix_array_ = move(codon_submatrix_array),  //

            phyloprocess_ = move(phyloprocess),         //

            bl_suffstats_ = bl_suffstats,               //

            site_path_suffstats_ = move(site_path_suffstats),     //
            comp_path_suffstats_ = move(comp_path_suffstats),     //
            nucpath_suffstats_ = move(nucpath_ssw),           // 
            comp_omegapath_suffstats_ = move(comp_omega_ssw));
    }

    // =============================================================================================
    template <class Model>
    static void touch_matrices(Model& model) {
        auto& nuc_matrix_proxy = get<nuc_rates, matrix_proxy>(model);
        nuc_matrix_proxy.gather();
        auto& cod_matrix_proxy = get<codon_submatrix_array>(model);
        cod_matrix_proxy.gather();
    }
};
