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
#include "submodels/mgomega.hpp"
#include "submodels/move_reporter.hpp"
#include "submodels/nucrates_sm.hpp"
#include "submodels/submodel_external_interface.hpp"
#include "submodels/suffstat_wrappers.hpp"

#include "bayes_toolbox.hpp"


TOKEN(global_omega)
TOKEN(branch_lengths)
TOKEN(nuc_rates)
TOKEN(codon_submatrix)
TOKEN(phyloprocess)
TOKEN(bl_suffstats)
TOKEN(path_suffstats)
TOKEN(nucpath_suffstats)
TOKEN(omegapath_suffstats)

struct globom_master {
    template <class Gen>
    static auto make(Gen& gen) {

        auto global_omega = make_node<gamma_mi>(one_to_const(1.0), one_to_const(1.0));
        draw(global_omega, gen);

        // will collect ss across slaves
        auto omega_ss = ss_factory::make_suffstat<OmegaPathSuffStat>([] (auto& omss) {});

        return make_model(global_omega_ = move(global_omega), omegapath_suffstats_ = move(omega_ss));
    }
};

struct globom_slave {
    template <class Gen>
    static auto make(PreparedData& data, Gen& gen) {

        auto omega = std::make_unique<double>(1.0);

        auto branch_lengths =
            branchlengths_sm::make(data.parser, *data.tree, one_to_const(0.1), one_to_const(1.0), gen);

        auto nuc_rates = nucrates_sm::make(
                std::vector<double>(6, 1./6), 1./6, std::vector<double>(4, 1./4), 1./4, gen);

        auto codon_statespace =
            dynamic_cast<const CodonStateSpace*>(data.alignment.GetStateSpace());

        auto codon_submatrix = make_dnode_with_init<mgomega>(
            {codon_statespace, &get<nuc_matrix, value>(nuc_rates), 1.0},
            [& mat = get<nuc_matrix, value>(nuc_rates)]() -> const SubMatrix& { return mat; },
            [&om = *omega] () {return om;} );

        gather(codon_submatrix);

        auto phyloprocess = std::make_unique<PhyloProcess>(data.tree.get(), &data.alignment,
            // branch lengths
            n_to_n(get<bl_array, value>(branch_lengths)),
            // site-specific rates: all equal to 1
            n_to_const(1.0),
            // branch and site specific matrices (here, same matrix for everyone)
            [& m = get<value>(codon_submatrix)](
                int branch, int site) -> const SubMatrix& { return m; },
            // site-specific matrices for root equilibrium frequencies (here same for all sites)
            [& m = get<value>(codon_submatrix)](int site) -> const SubMatrix& { return m; },
            // no polymorphism
            nullptr);

        phyloprocess->Unfold();
        std::cerr << "lnL: " << phyloprocess->GetLogLikelihood() << '\n';

        // suff stats
        auto bl_suffstats = pathss_factory::make_bl_suffstat(*phyloprocess);

        auto path_suffstats = pathss_factory::make_path_suffstat(*phyloprocess);

        // gathering nuc path suffstat from path suff stat
        auto nucpath_ss = ss_factory::make_suffstat_with_init<NucPathSuffStat>(
            {*codon_statespace}, [& mat = get<value>(codon_submatrix), &pss = *path_suffstats](
                                     auto& nucss) { nucss.AddSuffStat(mat, pss.get()); });

        auto omega_ss = ss_factory::make_suffstat<OmegaPathSuffStat>(
                [&mat = get<value>(codon_submatrix), &ss = *path_suffstats] (auto& omss) {
                    omss.AddSuffStat(mat,ss.get());
                });

        // clang-format off
        return make_model(
            global_omega_ = move(omega),
            branch_lengths_ = move(branch_lengths), 
            nuc_rates_ = move(nuc_rates),
            codon_submatrix_ = move(codon_submatrix), 
            phyloprocess_ = move(phyloprocess),
            bl_suffstats_ = move(bl_suffstats), 
            path_suffstats_ = move(path_suffstats),
            nucpath_suffstats_ = move(nucpath_ss),
            omegapath_suffstats_ = move(omega_ss) 
        );
        // clang-format on
    }
};
