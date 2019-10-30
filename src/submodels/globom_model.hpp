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
#include "submodels/omega_sm.hpp"
#include "submodels/mgomega.hpp"
#include "submodels/submodel_external_interface.hpp"
#include "submodels/suffstat_wrappers.hpp"
#include "bayes_toolbox.hpp"

TOKEN(global_omega)
TOKEN(branch_lengths)
TOKEN(nuc_rates)
// TOKEN(codon_statespace)
TOKEN(codon_submatrix)
TOKEN(phyloprocess)
TOKEN(bl_suffstats)
TOKEN(path_suffstats)
TOKEN(nucpath_suffstats)
TOKEN(omegapath_suffstats)

struct globom {
    // =============================================================================================
    template <class Gen>
    static auto make(PreparedData& data, Gen& gen) {
        auto global_omega = omega_sm::make(one_to_const(1.0), one_to_const(1.0), gen);

        auto branch_lengths =
            branchlengths_sm::make(data.parser, *data.tree, one_to_const(0.1), one_to_const(1.0));

        auto nuc_rates = nucrates_sm::make(one_to_const(normalize({1, 1, 1, 1, 1, 1})),
            one_to_const(1. / 6), one_to_const(normalize({1, 1, 1, 1})), one_to_const(1. / 4), gen);

        auto codon_statespace =
            dynamic_cast<const CodonStateSpace*>(data.alignment.GetStateSpace());

        auto codon_submatrix = make_dnode_with_init<mgomega>(
                {codon_statespace, &get<nuc_matrix, value>(nuc_rates), 1.0},
                [&mat = get<nuc_matrix, value>(nuc_rates)] () -> const SubMatrix& { return mat; },
                [&om = get<omega, value>(global_omega)] () {return om; } );

        gather(codon_submatrix);
            
        auto phyloprocess = std::make_unique<PhyloProcess>(data.tree.get(), &data.alignment,
            // branch lengths
            n_to_n(get<bl_array, value>(branch_lengths)),
            // site-specific rates: all equal to 1
            n_to_const(1.0),
            // branch and site specific matrices (here, same matrix for everyone)
            [&m = get<value>(codon_submatrix)] (int branch, int site) -> const SubMatrix& {return m;},
            // this one makes a seg fault:
            // [&m = get<value>(codon_submatrix)] (int branch, int site) {return m;},
            // mn_to_one(codon_submatrix),
            // site-specific matrices for root equilibrium frequencies (here same for all sites)
            [&m = get<value>(codon_submatrix)] (int site) -> const SubMatrix& {return m;},
            // n_to_one(codon_submatrix),
            // no polymorphism
            nullptr);

        phyloprocess->Unfold();
        std::cerr << "lnL: " << phyloprocess->GetLogLikelihood() << '\n';

        // suff stats

        auto bl_suffstats = pathss_factory::make_bl_suffstats(*phyloprocess);

        auto path_suffstats = pathss_factory::make_path_suffstat(*phyloprocess);

        // gathering nuc path suffstats across sites: sum stored in a single NucPathSuffStat
        auto nucpath_ss = ss_factory::make_suffstat_with_init<NucPathSuffStat>(
                {*codon_statespace},
                [&mat = get<value>(codon_submatrix), &pss = *path_suffstats] (auto& nucss) 
                    { nucss.AddSuffStat(mat, pss.get()); });

        auto omega_ss = ss_factory::make_suffstat<OmegaPathSuffStat>(
                [&mat = get<value>(codon_submatrix), &pss = *path_suffstats] (auto& omss)
                    { omss.AddSuffStat(mat, pss.get()); });

        return make_model(
            global_omega_ = move(global_omega),
            branch_lengths_ = move(branch_lengths),
            nuc_rates_ = move(nuc_rates),
            // codon_statespace_ = codon_statespace,
            codon_submatrix_ = move(codon_submatrix),
            phyloprocess_ = move(phyloprocess),
            bl_suffstats_ = move(bl_suffstats),
            path_suffstats_ = move(path_suffstats),
            nucpath_suffstats_ = move(nucpath_ss),
            omegapath_suffstats_ = move(omega_ss));
    }

    // =============================================================================================
    template <class Model>
    static void touch_matrices(Model& model) {
        gather(get<nuc_rates, nuc_matrix>(model));
        gather(codon_submatrix_(model));
    }
};
