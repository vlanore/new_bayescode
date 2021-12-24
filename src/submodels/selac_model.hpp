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
#include "submodels/t92.hpp"
#include "submodels/grantham.hpp"
#include "submodels/mutselomega.hpp"
#include "submodels/selac_profilearray.hpp"
#include "submodels/submodel_external_interface.hpp"
#include "submodels/suffstat_wrappers.hpp"

TOKEN(branch_lengths)
TOKEN(kappa)
TOKEN(gc)
TOKEN(nuc_matrix)
TOKEN(wcom)
TOKEN(wpol)
TOKEN(wvol)
TOKEN(aadist)
TOKEN(psi)
TOKEN(g_alpha)
TOKEN(g)
TOKEN(profile_array)
TOKEN(omega)
TOKEN(g_weights)
TOKEN(g_alloc)
TOKEN(aa_weights)
TOKEN(aa_alloc)
TOKEN(codon_submatrix_bidimarray)
TOKEN(phyloprocess)
TOKEN(bl_suffstat)
TOKEN(site_path_suffstat)
TOKEN(comp_path_suffstat)
TOKEN(omega_suffstat)
TOKEN(nucpath_suffstat)
TOKEN(aa_alloc_suffstat)

struct selac {

    template <class Gen>
    static auto make(PreparedData& data, size_t ncat, Gen& gen) {

        size_t nsite = data.alignment.GetNsite();

        // bl : iid gamma across sites, with constant hyperparams
        auto branch_lengths =
            branchlengths_sm::make(data.parser, *data.tree, 0.1, 1.0, gen);

        // T92 process
        auto kappa = make_node<gamma_mi>(one_to_const(1.0), one_to_const(1.0));
        draw(kappa, gen);
        auto gc = make_node<beta_ss>(one_to_const(1.0), one_to_const(1.0));
        draw(gc, gen);
        auto nuc_matrix = make_dnode_with_init<t92>(
                {get<value>(kappa), get<value>(gc), true},
                one_to_one(get<value>(kappa)),
                one_to_one(get<value>(gc)));
        gather(nuc_matrix);

        // grantham distance matrix
        auto wcom = make_node<gamma_mi>(one_to_const(1.0), one_to_const(1.0));
        auto wpol = make_node<gamma_mi>(one_to_const(1.0), one_to_const(1.0));
        auto wvol = make_node<gamma_mi>(one_to_const(1.0), one_to_const(1.0));
        raw_value(wcom) = grantham_wcom;
        raw_value(wpol) = grantham_wpol;
        raw_value(wvol) = grantham_wvol;
        auto aadist = make_node_with_init<grantham_dist>(std::vector<double>(Naa*(Naa-1)/2,0), one_to_one(wcom), one_to_one(wpol), one_to_one(wvol));
        gather(aadist);

        // expression level
        auto psi = make_node<gamma_mi>(one_to_const(10.0), one_to_const(1.0));
        draw(psi, gen);
        get<value>(psi) = 0.5;

        // discretized gamma (g) for stringency across sites, of shape g_alpha
        auto g_alpha = make_node<gamma_mi>(one_to_const(1.0), one_to_const(1.0));
        draw(g_alpha, gen);
        auto g = make_dnode_with_init<discrete_gamma>(std::vector<double>(ncat,1.0), one_to_one(g_alpha));
        gather(g);

        // amino-acid fitness profiles for each g*psi and each possible preferred amino-acid
        auto profile_array = make_dnode_array_with_init<selac_profilearray>(
                ncat,
                std::vector<std::vector<double>>(Naa, std::vector<double>(Naa, 0)),
                n_to_one(get<value>(aadist)),
                [&ppsi = get<value>(psi), &gg = get<value>(g)] (int i) {return ppsi*gg[i];});
        gather(profile_array);

        // global omega modulator
        auto omega = make_node<gamma_mi>(one_to_const(1.0), one_to_const(1.0));
        draw(omega, gen);
        get<value>(omega) = 1.0;

        // a ncat x Naa bidim array of mutsel omega codon matrices
        auto codon_statespace =
            dynamic_cast<const CodonStateSpace*>(data.alignment.GetStateSpace());
        std::vector<double> default_aa(Naa,1.0);
        auto codon_submatrix_bidimarray = make_dnode_matrix_with_init<mutselomega>(
                ncat,
                Naa,
                {codon_statespace, (SubMatrix*) &get<value>(nuc_matrix), default_aa, 1.0, 1.0, false},
                [&mat = get<value>(nuc_matrix)] (int i, int j) -> const SubMatrix& {return mat;},
                [&aa = get<value>(profile_array)] (int i, int j) {return aa[i][j];},
                mn_to_one(get<value>(omega)),
                mn_to_const(1.0));

        gather(codon_submatrix_bidimarray);

        // equal mixture weights across the discretized gamma
        // auto g_weights = make_param<std::vector<double>>(std::vector<double>(ncat, 1.0/ncat));
        auto g_weights = make_unique<std::vector<double>>(ncat, 1.0/ncat);
        // and corresponding iid categorical site allocations
        auto g_alloc = make_node_array<categorical>(nsite, [&w=*g_weights] (int i) {return w;});
        draw(g_alloc, gen);

        // mixture weights across preferred amino-acids
        auto aa_weights = make_node_with_init<dirichlet>(
                std::vector<double> (Naa, 1.0/Naa),     // initializer
                std::vector<double> (Naa, 1.0));     // hyperparam
        draw(aa_weights, gen);
        // and corresponding iid site allocations
        auto aa_alloc = make_node_array<categorical>(nsite, n_to_one(aa_weights));
        draw(aa_alloc, gen);

        // phyloprocess
        auto phyloprocess = std::make_unique<PhyloProcess>(data.tree.get(), &data.alignment,
            n_to_n(get<bl_array, value>(branch_lengths)),
            n_to_const(1.0),
            [&m=get<value>(codon_submatrix_bidimarray), &y=get<value>(g_alloc), &z=get<value>(aa_alloc)] (int branch, int site) -> const SubMatrix& {return m[y[site]][z[site]];},
            [&m=get<value>(codon_submatrix_bidimarray), &y=get<value>(g_alloc), &z=get<value>(aa_alloc)] (int site) -> const SubMatrix& {return m[y[site]][z[site]];},
            nullptr);

        phyloprocess->Unfold();
        std::cerr << "lnl : " << phyloprocess->GetLogLikelihood() << '\n';

        // branch lengths suff stats
        auto bl_suffstat = pathss_factory::make_bl_suffstat(*phyloprocess);

        // site path suff stats
        auto site_path_ss = pathss_factory::make_site_path_suffstat(*phyloprocess);

        // path suffstat reduced by component (bidim)
        auto comp_path_ss = mixss_factory::make_reduced_suffstat(ncat, Naa, *site_path_ss, get<value>(g_alloc), get<value>(aa_alloc));

        // alloction suff stat (= current occupancies of mixture components)
        // useful for resampling mixture weigths
        auto aa_alloc_ss = mixss_factory::make_alloc_suffstat(Naa, get<value>(aa_alloc));

        // omega suff stats (reduced across components)
        auto omega_ss = ss_factory::make_suffstat<OmegaPathSuffStat>(
                [&mat=get<value>(codon_submatrix_bidimarray), &pss=*comp_path_ss] (auto& omss, int i, int j) { omss.AddSuffStat(mat[i][j], pss.get(i,j)); },
                ncat, Naa);

        // nuc path suff stats (reduced across components)
        auto nucpath_ss = ss_factory::make_suffstat_with_init<NucPathSuffStat>(
                {*codon_statespace},
                [&mat=get<value>(codon_submatrix_bidimarray), &pss=*comp_path_ss] (auto& nucss, int i, int j) { nucss.AddSuffStat(mat[i][j], pss.get(i,j)); },
                ncat, Naa);

        return make_model(
            branch_lengths_ = move(branch_lengths),
            kappa_ = move(kappa),
            gc_ = move(gc),
            nuc_matrix_ = move(nuc_matrix),
            wcom_ = move(wcom),
            wpol_ = move(wpol),
            wvol_ = move(wvol),
            aadist_ = move(aadist),
            psi_ = move(psi),
            g_alpha_ = move(g_alpha),
            g_ = move(g),
            profile_array_ = move(profile_array),
            omega_ = move(omega),
            g_weights_ = move(g_weights),
            g_alloc_ = move(g_alloc),
            aa_weights_ = move(aa_weights),
            aa_alloc_ = move(aa_alloc),
            codon_submatrix_bidimarray_ = move(codon_submatrix_bidimarray),
            phyloprocess_ = move(phyloprocess),
            bl_suffstat_ = move(bl_suffstat),
            site_path_suffstat_ = move(site_path_ss),
            comp_path_suffstat_ = move(comp_path_ss),
            omega_suffstat_ = move(omega_ss),
            nucpath_suffstat_ = move(nucpath_ss),
            aa_alloc_suffstat_ = move(aa_alloc_ss));
    }

    template<class Model>
        static auto update_selac(Model& model) {
            gather(aadist_(model));
            gather(g_(model));
            gather(profile_array_(model));
        }

    template<class Model>
        static auto update_matrices(Model& model) {
            gather(nuc_matrix_(model));
            gather(codon_submatrix_bidimarray_(model));
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
            auto nuc_update = [&mat = nuc_matrix_(model)] () {gather(mat);};
            auto nuc_logprob = suffstat_logprob(nuc_matrix_(model), nucpath_suffstat_(model));
            scaling_move(kappa_(model), nuc_logprob, 1, 10, gen, nuc_update);
            slide_constrained_move(gc_(model), nuc_logprob, 1, 0, 1, 10, gen, nuc_update);
        }

    template<class Model, class Gen>
        static auto resample_g_alloc(Model& model, Gen& gen)  {
            auto g_alloc_logprob = 
                [&mat = get<codon_submatrix_bidimarray,value>(model), &zz = get<aa_alloc,value>(model), &ss = get<site_path_suffstat>(model)] (int i) {
                    return [&mat, &z=zz[i], &s=ss.get(i)] (int k) {return s.GetLogProb(mat[k][z]);};
                };
            logprob_gibbs_resample(g_alloc_(model), g_alloc_logprob, gen);

        }

    template<class Model, class Gen>
        static auto resample_aa_alloc(Model& model, Gen& gen)  {
            auto aa_alloc_logprob = 
                [&mat = get<codon_submatrix_bidimarray,value>(model), &yy = get<g_alloc,value>(model), &ss = get<site_path_suffstat>(model)] (int i) {
                    return [&mat, &y=yy[i], &s=ss.get(i)] (int k) {return s.GetLogProb(mat[y][k]);};
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
        static auto move_selac(Model& model, Gen& gen) {
            auto selac_logprob = suffstat_logprob(codon_submatrix_bidimarray_(model), comp_path_suffstat_(model));

            auto g_update = [&model] () {
                gather(g_(model)); 
                gather(profile_array_(model));
                gather(codon_submatrix_bidimarray_(model));};
            scaling_move(g_alpha_(model), selac_logprob, 1, 10, gen, g_update);
            scaling_move(g_alpha_(model), selac_logprob, 0.3, 10, gen, g_update);

            auto w_update = [&model] () {
                gather(aadist_(model));
                gather(profile_array_(model));
                gather(codon_submatrix_bidimarray_(model));};
            scaling_move(wcom_(model), selac_logprob, 1, 10, gen, w_update);
            scaling_move(wcom_(model), selac_logprob, 0.3, 10, gen, w_update);
            scaling_move(wpol_(model), selac_logprob, 1, 10, gen, w_update);
            scaling_move(wpol_(model), selac_logprob, 0.3, 10, gen, w_update);

            auto psi_update = [&model] () {
                gather(aadist_(model)); 
                gather(profile_array_(model));
                gather(codon_submatrix_bidimarray_(model));};
            scaling_move(psi_(model), selac_logprob, 1, 10, gen, psi_update);
            scaling_move(psi_(model), selac_logprob, 0.3, 10, gen, psi_update);
        }

    template<class Model>
        static auto get_total_length(Model& model)  {
            return branchlengths_sm::get_total_length(get<branch_lengths>(model));
        }
};
