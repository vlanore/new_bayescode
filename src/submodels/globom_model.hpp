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
#include "submodels/submodel_external_interface.hpp"
#include "submodels/suffstat_wrappers.hpp"

TOKEN(global_omega)
TOKEN(branch_lengths)
TOKEN(nuc_rates)
TOKEN(codon_statespace)
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

        auto global_omega = omega_sm::make(to_constant(1.0), to_constant(1.0), gen);

        auto branch_lengths = branchlengths_sm::make(data.parser, *data.tree, to_constant(0.1), to_constant(1.0));

        auto nuc_rates = nucrates_sm::make(
            to_constant(normalize({1, 1, 1, 1, 1, 1})),
            to_constant(1. / 6),
            to_constant(normalize({1, 1, 1, 1})), 
            to_constant(1. / 4),
            gen);

        auto codon_statespace =
            dynamic_cast<const CodonStateSpace*>(data.alignment.GetStateSpace());

        auto codon_sub_matrix = std::make_unique<MGOmegaCodonSubMatrix>(
            codon_statespace, &get<nuc_matrix>(nuc_rates), get<omega, value>(global_omega));

        auto phyloprocess = std::make_unique<PhyloProcess>(data.tree.get(), &data.alignment, 
                // branch lengths
                n_to_n(get<bl_array, value>(branch_lengths)),
                // site-specific rates: all equal to 1
                n_to_constant(1.0),
                // branch and site specific matrices (here, same matrix for everyone)
                mn_to_one(*codon_sub_matrix.get()),
                // site-specific matrices for root equilibrium frequencies (here same for all sites)
                n_to_one(*codon_sub_matrix.get()),
                // no polymorphism
                nullptr);

        phyloprocess->Unfold();

        // suff stats
        BranchArrayPoissonSSW bl_suffstats{*data.tree, *phyloprocess};
        auto path_suffstats = std::make_unique<PathSSW>(*phyloprocess);
        NucPathSuffStat nucpath_suffstats;
        OmegaSSW omega_ssw(*codon_sub_matrix, *path_suffstats);

        return make_model(                              //
            global_omega_ = move(global_omega),         //
            branch_lengths_ = move(branch_lengths),     //
            nuc_rates_ = move(nuc_rates),               //
            codon_statespace_ = codon_statespace,       //
            codon_submatrix_ = move(codon_sub_matrix),  //
            phyloprocess_ = move(phyloprocess),         //
            bl_suffstats_ = bl_suffstats,               //
            path_suffstats_ = move(path_suffstats),     //
            nucpath_suffstats_ = nucpath_suffstats,     //
            omegapath_suffstats_ = omega_ssw);
    }

    // =============================================================================================
    template <class Model>
    static void touch_matrices(Model& model) {
        auto& nuc_matrix_proxy = get<nuc_rates, matrix_proxy>(model);
        nuc_matrix_proxy.gather();
        codon_submatrix_(model).SetOmega(get<global_omega, omega, value>(model));
        codon_submatrix_(model).CorruptMatrix();
    }

    // =============================================================================================
    template <class Model, class Gen>
    static void move_nucrates(Model& model, Gen& gen, MoveStatsRegistry& ms) {
        nucpath_suffstats_(model).Clear();
        nucpath_suffstats_(model).AddSuffStat(
            codon_submatrix_(model), path_suffstats_(model).get());

        auto nucrates_logprob = [&model]() {
            return nucpath_suffstats_(model).GetLogProb(
                get<nuc_rates, matrix_proxy>(model).get(), codon_statespace_(model));
        };

        auto touch_nucmatrix = [&model]() { get<nuc_rates, matrix_proxy>(model).gather(); };

        nucrates_sm::move_exch_rates(nuc_rates_(model), {0.1, 0.03, 0.01}, nucrates_logprob,
            touch_nucmatrix, gen, ms("exch_rates"));
        nucrates_sm::move_eq_freqs(
            nuc_rates_(model), {0.1, 0.03}, nucrates_logprob, touch_nucmatrix, gen, ms("eq_freqs"));

        globom::touch_matrices(model);
    }
};
