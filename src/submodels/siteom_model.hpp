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
#include "submodels/iidgamma_mi.hpp"
#include "submodels/mgomega.hpp"
#include "submodels/submodel_external_interface.hpp"
#include "submodels/suffstat_wrappers.hpp"
#include "bayes_toolbox.hpp"

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

        // omega: iid gamma across sites, with constant hyperparameters
        auto site_omega = iidgamma_mi::make(nsite, one_to_const(1.0), one_to_const(1.0), gen);

        // bl : iid gamma across sites, with constant hyperparams
        auto branch_lengths =
            branchlengths_sm::make(data.parser, *data.tree, one_to_const(0.1), one_to_const(1.0));

        // nuc exch rates and eq freqs: uniform dirichlet
        // also creates the gtr matrix
        auto nuc_rates = nucrates_sm::make(one_to_const(normalize({1, 1, 1, 1, 1, 1})),
            one_to_const(1. / 6), one_to_const(normalize({1, 1, 1, 1})), one_to_const(1. / 4), gen);

        auto codon_statespace =
            dynamic_cast<const CodonStateSpace*>(data.alignment.GetStateSpace());

        // an array of MG Omega Codon matrices, with same nucrates but each with its own omega
        auto codon_submatrix_array = make_dnode_array_with_init<mgomega>(
                nsite,
                // {codon_statespace},
                {codon_statespace, &get<nuc_matrix, value>(nuc_rates), 1.0},
                [&mat = get<nuc_matrix, value>(nuc_rates)] (int i) -> const SubMatrix& { return mat; },
                // n_to_one(get<nuc_matrix, value>(nuc_rates)),
                n_to_n(get<gamma_array, value>(site_omega))
            );

        // phyloprocess
        auto phyloprocess = std::make_unique<PhyloProcess>(data.tree.get(), &data.alignment,
            // branch lengths
            n_to_n(get<bl_array, value>(branch_lengths)),
            // site-specific rates: all equal to 1
            n_to_const(1.0),
            // branch and site specific matrices (here, same matrix for everyone)
            [&m = get<value>(codon_submatrix_array)] (int branch, int site) -> const SubMatrix& {return m[site];},
            // mn_to_n(*codon_submatrix_array),
            // site-specific matrices for root equilibrium frequencies (here same for all sites)
            [&m = get<value>(codon_submatrix_array)] (int site) -> const SubMatrix& {return m[site];},
            // n_to_n(*codon_submatrix_array),
            // no polymorphism
            nullptr);

        phyloprocess->Unfold();
        std::cerr << "lnl : " << phyloprocess->GetLogLikelihood() << '\n';

        // suff stats

        // branch lengths
        auto bl_suffstats = pathss_factory::make_bl_suffstats(*phyloprocess);

        // site path suff stats
        auto site_path_suffstats = pathss_factory::make_site_path_suffstat(*phyloprocess);

        // gathering nuc path suffstats across sites: sum stored in a single NucPathSuffStat
        auto nucpath_ss = ss_factory::make_suffstat_with_init<NucPathSuffStat>(
                {*codon_statespace},
                [&mat = get<value>(codon_submatrix_array), &pss = *site_path_suffstats] (auto& nucss, int i) 
                    { nucss.AddSuffStat(mat[i], pss.get(i)); },
                nsite);

        // site omega suff stats
        auto site_omega_ss = ss_factory::make_suffstat_array<OmegaPathSuffStat>(
                nsite,
                [&mat = get<value>(codon_submatrix_array), &pss = *site_path_suffstats] (auto& omss, int i) 
                    { omss[i].AddSuffStat(mat[i], pss.get(i)); },
                nsite);

        return make_model(
            // codon_statespace_ = codon_statespace,
            site_omega_ = move(site_omega),
            branch_lengths_ = move(branch_lengths),
            nuc_rates_ = move(nuc_rates),
            codon_submatrix_array_ = move(codon_submatrix_array),
            phyloprocess_ = move(phyloprocess),
            bl_suffstats_ = move(bl_suffstats),
            site_path_suffstats_ = move(site_path_suffstats),
            nucpath_suffstats_ = move(nucpath_ss),
            site_omegapath_suffstats_ = move(site_omega_ss));
    }

    // =============================================================================================
    template <class Model>
    static void touch_matrices(Model& model) {
        gather(get<nuc_rates, nuc_matrix>(model));
        gather(codon_submatrix_array_(model));
    }
};
