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
#include "submodels/sitecodonmatrix_sm.hpp"
#include "submodels/submodel_external_interface.hpp"
#include "submodels/suffstat_wrappers.hpp"

TOKEN(site_omega)
TOKEN(branch_lengths)
TOKEN(nuc_rates)
// TOKEN(codon_statespace)
TOKEN(codon_submatrix_array)
TOKEN(phyloprocess)
TOKEN(bl_suffstats)
TOKEN(site_path_suffstats)
TOKEN(nucpath_suffstats)
TOKEN(site_omegapath_suffstats)

struct siteom {
    // =============================================================================================
    template <class Gen>
    static auto make(PreparedData& data, Gen& gen) {

        size_t nsite = data.alignment.GetNsite();

        auto site_omega = siteomega_sm::make(nsite, to_constant(1.0), to_constant(1.0), gen);

        auto branch_lengths =
            branchlengths_sm::make(data.parser, *data.tree, to_constant(0.1), to_constant(1.0));

        auto nuc_rates = nucrates_sm::make(to_constant(normalize({1, 1, 1, 1, 1, 1})),
            to_constant(1. / 6), to_constant(normalize({1, 1, 1, 1})), to_constant(1. / 4), gen);

        auto codon_statespace =
            dynamic_cast<const CodonStateSpace*>(data.alignment.GetStateSpace());

        auto codon_submatrix_array = sitecodonmatrix_sm::make(codon_statespace, get<nuc_matrix>(nuc_rates), get<site_omega_array, value>(site_omega));
        auto phyloprocess = std::make_unique<PhyloProcess>(data.tree.get(), &data.alignment,
            // branch lengths
            n_to_n(get<bl_array, value>(branch_lengths)),
            // site-specific rates: all equal to 1
            n_to_constant(1.0),
            // branch and site specific matrices (here, same matrix for everyone)
            mn_to_n(get<codon_matrix_array>(codon_submatrix_array)),
            // [&m = get<codon_matrix_array>(codon_submatrix_array)] (int branch, int site) -> const SubMatrix& { return m[site]; }, 
            // site-specific matrices for root equilibrium frequencies (here same for all sites)
            n_to_n(get<codon_matrix_array>(codon_submatrix_array)),
            // [&m = get<codon_matrix_array>(codon_submatrix_array)] (int site) -> const SubMatrix& { return m[site]; }, 
            // no polymorphism
            nullptr);

        phyloprocess->Unfold();
        std::cerr << "lnl : " << phyloprocess->GetLogLikelihood() << '\n';

        // suff stats

        BranchArrayPoissonSSW bl_suffstats{*data.tree, *phyloprocess};

        auto site_path_suffstats = std::make_unique<SitePathSSW>(*phyloprocess);

        auto nucpath_ssw = nucpathssw::make(codon_statespace,
                // n_to_n(get<mgomegacodon_matrix_array_proxy>(codon_submatrix_array)),
                n_to_n(get<codon_matrix_array>(codon_submatrix_array)),
                // [&m = get<codon_matrix_array>(codon_submatrix_array)] (int i) { return m[i]; },
                n_to_n(*site_path_suffstats),
                // [&p = *site_path_suffstats] (int i) { return p.get(i); },
                nsite);

        auto site_omega_ssw = std::make_unique<SiteOmegaSSW >(nsite, get<codon_matrix_array>(codon_submatrix_array), *site_path_suffstats);

        return make_model(                              //
            // codon_statespace_ = codon_statespace,       //
            site_omega_ = move(site_omega),         //
            branch_lengths_ = move(branch_lengths),     //
            nuc_rates_ = move(nuc_rates),               //
            codon_submatrix_array_ = move(codon_submatrix_array),  //
            phyloprocess_ = move(phyloprocess),         //
            bl_suffstats_ = bl_suffstats,               //
            site_path_suffstats_ = move(site_path_suffstats),     //
            nucpath_suffstats_ = move(nucpath_ssw),           // 
            site_omegapath_suffstats_ = move(site_omega_ssw));
    }

    // =============================================================================================
    template <class Model>
    static void touch_matrices(Model& model) {
        auto& nuc_matrix_proxy = get<nuc_rates, matrix_proxy>(model);
        nuc_matrix_proxy.gather();
        auto& cod_matrix_proxy = get<codon_submatrix_array, mgomegacodon_matrix_array_proxy>(model);
        cod_matrix_proxy.gather();
    }
};
