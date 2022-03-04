
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

#include "bayes_toolbox.hpp"

#include "submodels/branchlength_sm.hpp"
#include "submodels/move_reporter.hpp"
#include "submodels/discrete_gamma.hpp"
#include "submodels/mutselomega.hpp"
#include "submodels/submodel_external_interface.hpp"
#include "submodels/suffstat_wrappers.hpp"
#include "submodels/nucrates_sm.hpp"

TOKEN(branch_lengths)
TOKEN(nuc_rates)
TOKEN(aa_dirweights)
TOKEN(aa_profiles)
TOKEN(aa_weights)
TOKEN(aa_alloc)
TOKEN(omega)
TOKEN(codon_submatrices)
TOKEN(phyloprocess)
TOKEN(bl_suffstat)
TOKEN(site_path_suffstat)
TOKEN(comp_path_suffstat)
TOKEN(omega_suffstat)
TOKEN(nucpath_suffstat)
TOKEN(aa_alloc_suffstat)

struct mutseldp {

    template <class Gen>
    static auto make(PreparedData& data, size_t aancat, Gen& gen) {

        size_t nsite = data.alignment.GetNsite();

        // bl : iid gamma across sites, with constant hyperparams
        auto branch_lengths =
            branchlengths_sm::make(data.parser, *data.tree, 0.1, 1.0, gen);

        // nuc exch rates and eq freqs: uniform dirichlet
        auto nucrr_hypercenter = std::vector<double>(6, 1./6);
        auto nucrr_hyperinvconc = 1./6;
        auto nucstat_hypercenter = std::vector<double>(4, 1./4);
        auto nucstat_hyperinvconc = 1./4;

        // creates nucrr, nucstat and also the gtr matrix
        auto nuc_rates = nucrates_sm::make(
                nucrr_hypercenter, nucrr_hyperinvconc, 
                nucstat_hypercenter, nucstat_hyperinvconc, gen);

        auto aa_dirweights = make_node_array<gamma_mi>(Naa, n_to_const(1.0), n_to_const(1.0));
        draw(aa_dirweights, gen);

        auto aa_profiles = make_node_array_with_init<dirichlet>(
                aancat,
                std::vector<double>(Naa, 1.0/Naa),
                [&w = get<value>(aa_dirweights)] (int i) -> const std::vector<double>& {return w;}
                );
        draw(aa_profiles, gen);

        // global omega modulator
        auto omega = make_node<gamma_mi>(one_to_const(1.0), one_to_const(1.0));
        draw(omega, gen);
        get<value>(omega) = 1.0;

        auto codon_statespace =
            dynamic_cast<const CodonStateSpace*>(data.alignment.GetStateSpace());
        std::vector<double> default_aa(Naa,1.0);

        auto codon_submatrices = make_dnode_array_with_init<mutselomega>(
                aancat,
                {codon_statespace, (SubMatrix*) &get<nuc_matrix,value>(nuc_rates), default_aa, 1.0, 1.0, false},
                [&mat = get<nuc_matrix,value>(nuc_rates)] (int i) -> const SubMatrix& {return mat;},
                n_to_n(aa_profiles),
                n_to_one(get<value>(omega)),
                n_to_const(1.0));

        gather(codon_submatrices);

        // mixture weights across preferred amino-acids
        auto aa_weights = make_node_with_init<dirichlet>(
                std::vector<double> (aancat, 1.0/aancat),     // initializer
                std::vector<double> (aancat, 1.0));     // hyperparam
        draw(aa_weights, gen);
        // and corresponding iid site allocations
        auto aa_alloc = make_node_array<categorical>(nsite, n_to_one(aa_weights));
        draw(aa_alloc, gen);

        // phyloprocess
        auto phyloprocess = std::make_unique<PhyloProcess>(data.tree.get(), &data.alignment,
            n_to_n(get<bl_array, value>(branch_lengths)),
            n_to_const(1.0),
            [&m=get<value>(codon_submatrices), &z=get<value>(aa_alloc)] (int branch, int site) -> const SubMatrix& {return m[z[site]];},
            [&m=get<value>(codon_submatrices), &z=get<value>(aa_alloc)] (int site) -> const std::vector<double>& {return m[z[site]].eq_freqs();},
            nullptr);

        phyloprocess->Unfold();
        std::cerr << "lnl : " << phyloprocess->GetLogLikelihood() << '\n';

        // branch lengths suff stats
        auto bl_suffstat = pathss_factory::make_bl_suffstat(*phyloprocess);

        // site path suff stats
        auto site_path_ss = pathss_factory::make_site_path_suffstat(*phyloprocess);

        // path suffstat reduced by component
        auto comp_path_ss = mixss_factory::make_reduced_suffstat(aancat, *site_path_ss, get<value>(aa_alloc));

        // alloction suff stat (= current occupancies of mixture components)
        // useful for resampling mixture weigths
        auto aa_alloc_ss = mixss_factory::make_alloc_suffstat(aancat, get<value>(aa_alloc));

        // omega suff stats (reduced across components)
        auto omega_ss = ss_factory::make_suffstat<OmegaPathSuffStat>(
                [&mat=get<value>(codon_submatrices), &pss=*comp_path_ss] (auto& omss, int i) { omss.AddSuffStat(mat[i], pss.get(i)); },
                aancat);

        // nuc path suff stats (reduced across components)
        auto nucpath_ss = ss_factory::make_suffstat_with_init<NucPathSuffStat>(
                {*codon_statespace},
                [&mat=get<value>(codon_submatrices), &pss=*comp_path_ss] (auto& nucss, int i) { nucss.AddSuffStat(mat[i], pss.get(i)); },
                aancat);

        return make_model(
            branch_lengths_ = move(branch_lengths),
            nuc_rates_ = move(nuc_rates),
            aa_dirweights_ = move(aa_dirweights),
            aa_profiles_ = move(aa_profiles),
            aa_weights_ = move(aa_weights),
            aa_alloc_ = move(aa_alloc),
            omega_ = move(omega),
            codon_submatrices_ = move(codon_submatrices),
            phyloprocess_ = move(phyloprocess),
            bl_suffstat_ = move(bl_suffstat),
            site_path_suffstat_ = move(site_path_ss),
            comp_path_suffstat_ = move(comp_path_ss),
            omega_suffstat_ = move(omega_ss),
            nucpath_suffstat_ = move(nucpath_ss),
            aa_alloc_suffstat_ = move(aa_alloc_ss));
    }

    template<class Model>
        static auto update_matrices(Model& model) {
            gather(nuc_matrix_(nuc_rates_(model)));
            gather(codon_submatrices_(model));
        }

    template<class Model, class Gen>
        static auto resample_sub(Model& model, Gen& gen)    {
            phyloprocess_(model).Move(1.0);
        }

    template<class Model, class Gen>
        static auto move_bl(Model& model, Gen& gen) {
            bl_suffstat_(model).gather();
            branchlengths_sm::gibbs_resample(branch_lengths_(model), bl_suffstat_(model), gen);
        }

    template<class Model, class Gen>
        static auto move_nuc(Model& model, Gen& gen)    {
            nucpath_suffstat_(model).gather();
            nucrates_sm::move_nucrates(nuc_rates_(model), nucpath_suffstat_(model), gen, 1, 1.0);
        }

    template<class Model, class Gen>
        static auto resample_aa_alloc(Model& model, Gen& gen)  {
            auto aa_alloc_logprob = 
                [&mat = get<codon_submatrices,value>(model), &ss = get<site_path_suffstat>(model)] (int i) {
                    return [&mat, &s=ss.get(i)] (int k) {return s.GetLogProb(mat[k]);};
                };

            logprob_gibbs_resample(aa_alloc_(model), aa_alloc_logprob, gen);

            aa_alloc_suffstat_(model).gather();
            gibbs_resample(aa_weights_(model), aa_alloc_suffstat_(model), gen);
        }

    template<class Model, class Gen>
        static auto move_omega(Model& model, Gen& gen)  {
            omega_suffstat_(model).gather();
            gibbs_resample(omega_(model), omega_suffstat_(model), gen);
        }

    template<class Model, class Gen>
        static auto move_mutseldp(Model& model, Gen& gen) {
            auto aagather = array_element_gather(codon_submatrices_(model));
            auto aalogprob = suffstat_array_element_logprob(codon_submatrices_(model), comp_path_suffstat_(model));
            auto aapropose1 = [] (std::vector<double>& aa, Gen& gen) { return profile_move(aa, 1, 1.0, gen); };
            auto aapropose2 = [] (std::vector<double>& aa, Gen& gen) { return profile_move(aa, 3, 1.0, gen); };
            auto aapropose3 = [] (std::vector<double>& aa, Gen& gen) { return profile_move(aa, 3, 0.3, gen); };
            mh_move(aa_profiles_(model), aalogprob, aapropose1, 1, gen, aagather);
            mh_move(aa_profiles_(model), aalogprob, aapropose2, 1, gen, aagather);
            mh_move(aa_profiles_(model), aalogprob, aapropose3, 1, gen, aagather);
        }

    template<class Model, class Gen>
        static auto move_hyper(Model& model, Gen& gen)  {
            auto hyper_logprob = [&a = aa_profiles_(model)] (int i) {return logprob(a);};
            // simple_logprob(aa_profiles_(model));
            auto scale_prop = [] (double& a, Gen& gen) { return scale(a, 1.0, gen); };
            auto no_update = [] (int i) {};
            mh_move(aa_dirweights_(model), hyper_logprob, scale_prop, 10, gen, no_update);
        }

    template<class Model>
        static auto get_Keff(Model& model) {
            auto& w = get<aa_weights,value>(model);
            double tot = 0;
            for (size_t i=0; i<w.size(); i++)   {
                tot -= w[i]*log(w[i]);
            }
            return exp(tot);
        }

    template<class Model>
        static auto get_statent(Model& model) {
            auto& w = get<aa_dirweights,value>(model);
            double tot = 0;
            for (size_t i=0; i<w.size(); i++)   {
                tot += w[i];
            }
            double ent = 0;
            for (size_t i=0; i<w.size(); i++)   {
                double tmp = w[i] / tot;
                ent -= tmp*log(tmp);
            }
            return ent;
        }

    template<class Model>
        static auto get_statalpha(Model& model) {
            auto& w = get<aa_dirweights,value>(model);
            double tot = 0;
            for (size_t i=0; i<w.size(); i++)   {
                tot += w[i];
            }
            return tot;
        }

    template<class Model>
        static auto get_total_length(Model& model)  {
            return branchlengths_sm::get_total_length(get<branch_lengths>(model));
        }
};
