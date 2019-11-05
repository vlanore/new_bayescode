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
#include "distributions/dirichlet.hpp"
#include "distributions/categorical.hpp"
#include "submodels/mgomega.hpp"
#include "submodels/submodel_external_interface.hpp"
#include "submodels/suffstat_wrappers.hpp"
#include "structure/suffstat.hpp"
#include "bayes_toolbox.hpp"

TOKEN(branch_lengths)
TOKEN(nuc_rates)

TOKEN(mixture_weights)
TOKEN(mixture_allocs)

TOKEN(omega_array)
TOKEN(codon_submatrix_array)

TOKEN(phyloprocess)

TOKEN(bl_suffstat)
TOKEN(site_path_suffstat)
TOKEN(comp_path_suffstat)
TOKEN(nucpath_suffstat)
TOKEN(site_omegapath_suffstat)
TOKEN(comp_omegapath_suffstat)
TOKEN(alloc_suffstat)

struct mixom {
    // =============================================================================================
    template <class Gen>
    static auto make(PreparedData& data, size_t ncomp, Gen& gen) {

        size_t nsite = data.alignment.GetNsite();

        // bl : iid gamma across sites, with constant hyperparams
        auto branch_lengths =
            branchlengths_sm::make(data.parser, *data.tree, 0.1, 1.0);

        // nuc exch rates and eq freqs: uniform dirichlet
        // nucrates_sm also creates the gtr matrix, accessible via nuc_matrix
        auto nuc_rates = nucrates_sm::make(
                std::vector<double>(6, 1./6), 1./6, std::vector<double>(4, 1./4), 1./4, gen);

        // first weight_hyper is the allocation initializer, second one is the parent in the DAG..
        auto weights = make_node_with_init<dirichlet>(
                std::vector<double> (ncomp, 1.0/ncomp),
                std::vector<double> (ncomp, 1.0/ncomp));

        // iid categorical site allocations
        auto alloc = make_node_array<categorical>(nsite, n_to_one(weights));
        draw(alloc, gen);

        // omega: iid gamma across mixture components, with constant hyperparameters
        auto omega_array = make_node_array<gamma_mi>(ncomp, n_to_const(1.0), n_to_const(1.0));
        draw(omega_array, gen);

        auto codon_statespace =
            dynamic_cast<const CodonStateSpace*>(data.alignment.GetStateSpace());

        // an array of MG Omega Codon matrices, one per mixture component,
        // all with same nucrates but each with its own omega
        auto codon_submatrix_array = make_dnode_array_with_init<mgomega>(
                ncomp,
                {codon_statespace, &get<nuc_matrix, value>(nuc_rates), 1.0},
                n_to_one(get<nuc_matrix, value>(nuc_rates)),
                n_to_n(get<value>(omega_array)));

        gather(codon_submatrix_array);

        // phyloprocess
        auto phyloprocess = std::make_unique<PhyloProcess>(data.tree.get(), &data.alignment,

            n_to_n(get<bl_array, value>(branch_lengths)),

            n_to_const(1.0),

            // mn_to_mixn(get<value>(codon_submatrix_array), get<value>(alloc)),
            [&m=get<value>(codon_submatrix_array), &z=get<value>(alloc)] (int branch, int site) -> const SubMatrix& {return m[z[site]];},

            // n_to_mix(get<value>(codon_submatrix_array), get<value>(alloc)),
            [&m = get<value>(codon_submatrix_array), &z = get<value>(alloc)] (int site) -> const SubMatrix& {return m[z[site]];},

            nullptr);

        phyloprocess->Unfold();
        std::cerr << "lnl : " << phyloprocess->GetLogLikelihood() << '\n';

        // suff stats

        // branch lengths
        auto bl_suffstat = pathss_factory::make_bl_suffstat(*phyloprocess);

        // site path suff stats
        auto site_path_ss = pathss_factory::make_site_path_suffstat(*phyloprocess);

        // path suffstats reduced by component
        auto comp_path_ss = mixss_factory::make_reduced_suffstat(ncomp, *site_path_ss, get<value>(alloc));

        // alloction suff stat (= current occupancies of mixture components)
        // useful for resampling mixture weigths
        auto alloc_ss = mixss_factory::make_alloc_suffstat(ncomp, get<value>(alloc));

        // omega suff stats across sites (useful for allocation moves)
        auto site_omega_ss = ss_factory::make_suffstat_array<OmegaPathSuffStat>(
                nsite,
                [&mat = get<value>(codon_submatrix_array), &pss = *site_path_ss, &z = get<value>(alloc)] (auto& omss, int i) 
                    { omss[i].AddSuffStat(mat[z[i]], pss.get(i)); },
                nsite);

        // omega path suff stats reduced by component (useful for moving omega values)
        auto comp_omega_ss = mixss_factory::make_reduced_suffstat(ncomp, *site_omega_ss, get<value>(alloc));

        // nuc path suff stats (reduced across components)
        // assumes that path suffstats have first been reduced by components
        auto nucpath_ss = pathss_factory::make_nucpath_suffstat(codon_statespace, get<value>(codon_submatrix_array), *comp_path_ss);

        return make_model(
            branch_lengths_ = move(branch_lengths),
            nuc_rates_ = move(nuc_rates),

            mixture_weights_ = move(weights),
            mixture_allocs_ = move(alloc),

            omega_array_ = move(omega_array),
            codon_submatrix_array_ = move(codon_submatrix_array),

            phyloprocess_ = move(phyloprocess),

            bl_suffstat_ = move(bl_suffstat),

            site_path_suffstat_ = move(site_path_ss),
            comp_path_suffstat_ = move(comp_path_ss),
            alloc_suffstat_ = move(alloc_ss),
            nucpath_suffstat_ = move(nucpath_ss),
            comp_omegapath_suffstat_ = move(comp_omega_ss),
            site_omegapath_suffstat_ = move(site_omega_ss));
    }
};
