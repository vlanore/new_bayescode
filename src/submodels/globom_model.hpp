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
#include "submodels/mg_omega.hpp"
#include "submodels/submodel_external_interface.hpp"
#include "submodels/suffstat_wrappers.hpp"

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

        auto codon_submatrix = mg_omega::make(
                codon_statespace, 
                one_to_one(get<nuc_matrix>(nuc_rates)),
                one_to_one(get<omega , value>(global_omega))
            );

        auto phyloprocess = std::make_unique<PhyloProcess>(data.tree.get(), &data.alignment,
            // branch lengths
            n_to_n(get<bl_array, value>(branch_lengths)),
            // site-specific rates: all equal to 1
            n_to_const(1.0),
            // branch and site specific matrices (here, same matrix for everyone)
            // why should I add get??
            mn_to_one(get<mg_omega_proxy>(codon_submatrix).get()),
            // site-specific matrices for root equilibrium frequencies (here same for all sites)
            // why should I add get??
            n_to_one(get<mg_omega_proxy>(codon_submatrix).get()),
            // no polymorphism
            nullptr);

        phyloprocess->Unfold();

        // suff stats
        BranchArrayPoissonSSW bl_suffstats{*data.tree, *phyloprocess};
        auto path_suffstats = std::make_unique<PathSSW>(*phyloprocess);
        // why get???
        NucPathSSW nucpath_ssw(get<mg_omega_proxy>(codon_submatrix).get(), *path_suffstats);
        OmegaSSW omega_ssw(get<mg_omega_proxy>(codon_submatrix).get(), *path_suffstats);

        return make_model(                              //
            global_omega_ = move(global_omega),         //
            branch_lengths_ = move(branch_lengths),     //
            nuc_rates_ = move(nuc_rates),               //
            // codon_statespace_ = codon_statespace,       //
            codon_submatrix_ = move(codon_submatrix),  //
            phyloprocess_ = move(phyloprocess),         //
            bl_suffstats_ = bl_suffstats,               //
            path_suffstats_ = move(path_suffstats),     //
            nucpath_suffstats_ = nucpath_ssw,           // 
            omegapath_suffstats_ = omega_ssw);
    }

    // =============================================================================================
    template <class Model>
    static void touch_matrices(Model& model) {
        auto& nuc_matrix_proxy = get<nuc_rates, matrix_proxy>(model);
        nuc_matrix_proxy.gather();
        auto& cod_matrix_proxy = get<codon_submatrix, mg_omega_proxy>(model);
        cod_matrix_proxy.gather();
    }
};
