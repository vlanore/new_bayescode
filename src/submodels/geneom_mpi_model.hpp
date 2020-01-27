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
#include "lib/GammaSuffStat.hpp"
#include "submodels/branchlength_sm.hpp"
#include "submodels/mgomega.hpp"
#include "submodels/move_reporter.hpp"
#include "submodels/nucrates_sm.hpp"
#include "submodels/submodel_external_interface.hpp"
#include "submodels/suffstat_wrappers.hpp"

#include "bayes_toolbox.hpp"


TOKEN(omega_hypermean)
TOKEN(omega_hyperinvshape)
TOKEN(gene_model_array)
TOKEN(omega)
TOKEN(branch_lengths)
TOKEN(nuc_rates)
TOKEN(codon_submatrix)
TOKEN(phyloprocess)
TOKEN(bl_suffstats)
TOKEN(path_suffstats)
TOKEN(nucpath_suffstats)
TOKEN(omegapath_suffstats)
TOKEN(omega_gamma_suffstats)

struct geneom_master {
    template <class Gen>
    static auto make(Gen& gen) {

        auto om_mean = make_node<exponential>(one_to_const(1.0));
        auto om_invshape = make_node<exponential>(one_to_const(1.0));
        raw_value(om_mean) = 1.0;
        raw_value(om_invshape) = 1.0;

        auto omega_gamma_ss = ss_factory::make_suffstat<GammaSuffStat>([] (auto& omss) {});

        return make_model(
                omega_hypermean_ = move(om_mean), 
                omega_hyperinvshape_ = move(om_invshape), 
                omega_gamma_suffstats_ = move(omega_gamma_ss)
            );
    }
};

struct geneom_slave {

    template <class Gen, class OmMean, class OmInvshape>
    static auto make_gene(PreparedData& data, OmMean om_mean, OmInvshape om_invshape, Gen& gen) {

        auto omega = make_node<gamma_mi>(om_mean, om_invshape);
        draw(omega, gen);

        auto branch_lengths =
            branchlengths_sm::make(data.parser, *data.tree, one_to_const(0.1), one_to_const(1.0));

        auto nuc_rates = nucrates_sm::make(
                std::vector<double>(6, 1./6), 1./6, std::vector<double>(4, 1./4), 1./4, gen);

        auto codon_statespace =
            dynamic_cast<const CodonStateSpace*>(data.alignment.GetStateSpace());

        auto codon_submatrix = make_dnode_with_init<mgomega>(
            {codon_statespace, &get<nuc_matrix, value>(nuc_rates), 1.0},
            [& mat = get<nuc_matrix, value>(nuc_rates)]() -> const SubMatrix& { return mat; },
            [&om = get<value>(omega)] () {return om;} );

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
            omega_ = move(omega),
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

    template <class Model, class Gen, class Data, class OmMean, class OmInvshape>
        static auto make_gene_array(Model& refmodel, Data& data, OmMean om_mean, OmInvshape om_invshape, Gen& gen) {
            std::vector<Model> v;
            for (auto& d : data)    {
                v.emplace_back(make_gene(*d, om_mean, om_invshape, gen));
            }
            return v;
        }

    template <class Gen, class Data>
    static auto make(Data& data, Gen& gen) {

        auto om_mean = std::make_unique<double>(1.0);
        auto om_invshape = std::make_unique<double>(1.0);

        auto refmodel = make_gene(
                *data[0],
                [&mean = *om_mean] () {return mean;},
                [&invshape = *om_invshape] () {return invshape;},
                gen);

        auto gene_model_array = make_gene_array(
                refmodel,
                data, 
                [&mean = *om_mean] () {return mean;},
                [&invshape = *om_invshape] () {return invshape;},
                gen);

        auto omega_gamma_ss = ss_factory::make_suffstat<GammaSuffStat>(
                [&array = gene_model_array] (auto& omss) {
                    for (auto& gene_model : array)   {
                        double om = get<omega,value>(gene_model);
                        omss.AddSuffStat(om, log(om), 1);
                    }
                });

        return make_model(
                // gene_model_array_ = gene_model_array,
                gene_model_array_ = move(gene_model_array),
                omega_hypermean_ = move(om_mean),
                omega_hyperinvshape_ = move(om_invshape),
                omega_gamma_suffstats_ = move(omega_gamma_ss)
            );
    }

    template <class Model, class Gen>
        static auto resample_sub(Model& model, Gen& gen)  {

            gather(get<nuc_rates, nuc_matrix>(model));
            gather(codon_submatrix_(model));
            phyloprocess_(model).Move(1.0);
        }

    template <class Model, class Gen>
        static auto move_params(Model& model, Gen& gen)  {

            // move branch lengths
            bl_suffstats_(model).gather();
            branchlengths_sm::gibbs_resample(
                branch_lengths_(model), bl_suffstats_(model), gen);
            // move nuc rates
            nucpath_suffstats_(model).gather();
            nucrates_sm::move_nucrates(
                nuc_rates_(model), nucpath_suffstats_(model), gen, 1, 1.0);

            // gather omega suffstats
            path_suffstats_(model).gather();
            omegapath_suffstats_(model).gather();
        }

    template <class Model>
    static auto update_matrices(Model& model)   {
        gather(get<codon_submatrix>(model));
    }

    template <class Model, class Gen>
    static auto gene_resample_sub(Model& model, Gen& gen)    {
        for (auto& gene_model : get<gene_model_array>(model)) {
            resample_sub(gene_model, gen);
        }
    }

    template <class Model, class Gen>
    static auto gene_move_params(Model& model, Gen& gen)    {
        for (auto& gene_model : get<gene_model_array>(model)) {
            move_params(gene_model, gen);
        }
    }

    template <class Model>
    static auto gene_collect_suffstat(Model& model)    {
        get<omega_gamma_suffstats>(model).gather();
    }


    template <class Model>
    static auto gene_update_after_receive(Model& model)    {
        for (auto& gene_model : get<gene_model_array>(model)) {
            gather(get<codon_submatrix>(gene_model));
        }
    }
};