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

#include "tree_factory.hpp"
#include "chronogram.hpp"
#include "lib/MultivariateBrownianTreeProcess.hpp"
#include "submodels/mgomega.hpp"
#include "submodels/invwishart.hpp"
#include "branch_map.hpp"

TOKEN(chronotree)
TOKEN(sigma)
TOKEN(brownian_process)
TOKEN(synrate)
TOKEN(branchNe)
TOKEN(rootNe)
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
TOKEN(profiles)
TOKEN(branch_Ne)
TOKEN(omega)
TOKEN(g_weights)
TOKEN(g_alloc)
TOKEN(aa_weights)
TOKEN(aa_alloc)
TOKEN(codon_matrices)
TOKEN(phyloprocess)
TOKEN(bl_suffstat)
TOKEN(site_path_suffstat)
TOKEN(comp_path_suffstat)
TOKEN(nucpath_suffstat)
TOKEN(aa_alloc_suffstat)
TOKEN(covmat_suffstat)

struct selacNe {

    template <class RootMean, class RootVar, class Gen>
    static auto make(const Tree* tree, const CodonSequenceAlignment& codon_data, const ContinuousData& cont_data, RootMean inroot_mean, RootVar inroot_var, int ncat, Gen& gen) {

        // number of quantitative traits
        size_t ncont = cont_data.GetNsite();

        // number of nodes and branches in tree
        size_t nnode = tree->nb_nodes();
        // size_t nbranch = nnode-1;

        size_t nsite = codon_data.GetNsite();

        // chronogram - relative dates (root has age 1, i.e. tree has depth 1)
        auto chronotree = make_node_with_init<chronogram>({tree});
        draw(chronotree, gen);

        // covariance matrix; dim = 2 + ncont (dS, dN/dS, traits)
        auto sigma = make_node_with_init<invwishart>(
                {2 + ncont, 0},
                std::vector<double>(2 + ncont, 1.0));
        draw(sigma, gen);

        // mean and variance for normal prior for brownian process at the root
        // (dim: 2 + ncont)
        auto root_mean = make_param<std::vector<double>>(std::forward<RootMean>(inroot_mean));
        auto root_var = make_param<std::vector<double>>(std::forward<RootVar>(inroot_var));
        // brownian process
        auto brownian_process = make_brownian_tree_process(
                tree,
                [&ch = get<value>(chronotree)] (int node) -> const double& {return ch[node];},
                one_to_one(get<value>(sigma)),
                root_mean,
                root_var);

        // fix brownian process to observed trait values in extant species
        for (size_t i=0; i<ncont; i++)  {
            // first index maps to brownian process, second index maps to continuous data
            brownian_process->SetAndClamp(cont_data, i+2, i);
        }
        // not drawing from brownian process prior
        brownian_process->PseudoSample(0.1);
        
        auto synrate = branch_map::make_branch_sums(get<value>(chronotree), *brownian_process, 0);
        gather(synrate);

        auto branchNe = branch_map::make_branch_means(get<value>(chronotree), *brownian_process, 1);
        gather(branchNe);

        auto rootNe = make_node<gamma_mi>(one_to_const(1.0), one_to_const(1.0));
        raw_value(rootNe) = 1.0;

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
        get<value>(psi) = 1.0;
        // get<value>(psi) = 0.5;

        // discretized gamma (g) for stringency across sites, of shape g_alpha
        auto g_alpha = make_node<gamma_mi>(one_to_const(1.0), one_to_const(1.0));
        draw(g_alpha, gen);
        auto g = make_dnode_with_init<discrete_gamma>(std::vector<double>(ncat,1.0), one_to_one(g_alpha));
        gather(g);

        // amino-acid fitness profiles for each g*psi and each possible preferred amino-acid
        auto profiles = make_dnode_array_with_init<selac_profilearray>(
                ncat,
                std::vector<std::vector<double>>(Naa, std::vector<double>(Naa, 0)),
                n_to_one(get<value>(aadist)),
                [&ppsi = get<value>(psi), &gg = get<value>(g)] (int i) {return ppsi*gg[i];});
        gather(profiles);

        // a ncat x Naa bidim array of mutsel omega codon matrices
        auto codon_statespace =
            dynamic_cast<const CodonStateSpace*>(codon_data.GetStateSpace());

        std::vector<double> default_aa(Naa,1.0);

        auto codon_matrices = make_dnode_cubix_with_init<mutselomega>(
                nnode,
                ncat,
                Naa,
                {codon_statespace, (SubMatrix*) &get<value>(nuc_matrix), default_aa, 1.0, 1.0, false},
                [&mat = get<value>(nuc_matrix)] (int node, int cat, int aa) -> const SubMatrix& {return mat;},
                [&pr = get<value>(profiles)] (int node, int cat, int aa) {return pr[cat][aa];},
                [] (int node, int cat, int aa) {return 1.0;},
                [&rootne=get<value>(rootNe), &Ne=get<value>(branchNe)] (int node, int cat, int aa) {return node ? Ne[node-1] : rootne;}
                );
        gather(codon_matrices);

        // equal mixture weights across the discretized gamma
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
        auto phyloprocess = std::make_unique<PhyloProcess>(tree, &codon_data,
            n_to_n(synrate),
            n_to_const(1.0),
            [&m=get<value>(codon_matrices), &y=get<value>(g_alloc), &z=get<value>(aa_alloc)] (int branch, int site) -> const SubMatrix& {return m[branch+1][y[site]][z[site]];},
            [&m=get<value>(codon_matrices), &y=get<value>(g_alloc), &z=get<value>(aa_alloc)] (int site) -> const std::vector<double>& {return m[0][y[site]][z[site]].eq_freqs();},
            nullptr);

        phyloprocess->Unfold();
        std::cerr << "lnl : " << phyloprocess->GetLogLikelihood() << '\n';

        auto bl_suffstat = pathss_factory::make_bl_suffstat(*phyloprocess);

        auto site_path_ss = pathss_factory::make_site_node_path_suffstat(*phyloprocess);

        auto comp_path_ss = ss_factory::make_suffstat_cubix<PathSuffStat>(
            nnode,
            ncat,
            Naa,
            [&site_ss=*site_path_ss, &y=get<value>(g_alloc), &z=get<value>(aa_alloc)]
            (auto& comp_ss, int site, int node) {
                comp_ss[node][y[site]][z[site]].Add(site_ss.get(site, node));
            },
            nsite,nnode);

        // nuc path suff stats (reduced across components)
        auto nucpath_ss = ss_factory::make_suffstat_with_init<NucPathSuffStat>(
            {*codon_statespace},
            [&mat=get<value>(codon_matrices), &pss=*comp_path_ss] (auto& nucss, int node, int cat, int aa) { nucss.AddSuffStat(mat[node][cat][aa], pss.get(node,cat,aa)); },
            nnode, ncat, Naa);

        // suffstats (branchwise independent contrasts of brownian process) for updating sigma
        auto covmat_ss = ss_factory::make_suffstat_with_init<MultivariateNormalSuffStat>({ncont+2},
                [&process = *brownian_process] (auto& ss) { ss.AddSuffStat(process); });

        // alloction suff stat (= current occupancies of mixture components)
        // useful for resampling mixture weigths
        auto aa_alloc_ss = mixss_factory::make_alloc_suffstat(Naa, get<value>(aa_alloc));

        return make_model(
            chronotree_ = move(chronotree),
            sigma_ = move(sigma),
            brownian_process_ = move(brownian_process),
            synrate_ = move(synrate),
            branchNe_ = move(branchNe),
            rootNe_ = move(rootNe),
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
            profiles_ = move(profiles),
            g_weights_ = move(g_weights),
            g_alloc_ = move(g_alloc),
            aa_weights_ = move(aa_weights),
            aa_alloc_ = move(aa_alloc),
            codon_matrices_ = move(codon_matrices),
            phyloprocess_ = move(phyloprocess),
            bl_suffstat_ = move(bl_suffstat),
            site_path_suffstat_ = move(site_path_ss),
            comp_path_suffstat_ = move(comp_path_ss),
            nucpath_suffstat_ = move(nucpath_ss),
            aa_alloc_suffstat_ = move(aa_alloc_ss),
            covmat_suffstat_ = move(covmat_ss));
    }

    template<class Model>
        static auto update_selac(Model& model) {
            gather(aadist_(model));
            gather(g_(model));
            gather(profiles_(model));
        }

    template<class Model>
        static auto update_matrices(Model& model) {
            gather(nuc_matrix_(model));
            gather(codon_matrices_(model));
        }

    template<class Model, class Gen>
        static auto resample_sub(Model& model, Gen& gen)    {
            phyloprocess_(model).Move(1.0);
        }

    template<class Model, class Gen>
        static auto move_nuc(Model& model, Gen& gen)    {
            nucpath_suffstat_(model).gather();
            auto nuc_update = [&mat = nuc_matrix_(model)] () {gather(mat);};
            auto nuc_logprob = suffstat_logprob(nuc_matrix_(model), nucpath_suffstat_(model));
            scaling_move(kappa_(model), nuc_logprob, 1, 10, gen, nuc_update);
            slide_constrained_move(gc_(model), nuc_logprob, 1, 0, 1, 10, gen, nuc_update);
            gather(codon_matrices_(model));
        }

    template<class Model, class Gen>
        static auto resample_g_alloc(Model& model, Gen& gen)  {
            auto g_alloc_logprob = 
                [&mat = get<codon_matrices,value>(model), &zz = get<aa_alloc,value>(model), &ss = get<site_path_suffstat>(model)] (int site) {
                    return [&mat, &z=zz[site], &ss, site] (int k) {
                        int nnode = mat.size();
                        double tot = 0;
                        for (int n=0; n<nnode; n++)    {
                            tot += ss.get(site,n).GetLogProb(mat[n][k][z]);
                        }
                        return tot;
                    };
                };
            logprob_gibbs_resample(g_alloc_(model), g_alloc_logprob, gen);

        }

    template<class Model, class Gen>
        static auto resample_aa_alloc(Model& model, Gen& gen)  {
            auto aa_alloc_logprob = 
                [&mat = get<codon_matrices,value>(model), &yy = get<g_alloc,value>(model), &ss = get<site_path_suffstat>(model)] (int site) {
                    return [&mat, &y=yy[site], &ss, site] (int k) {
                        int nnode = mat.size();
                        double tot = 0;
                        for (int n=0; n<nnode; n++)    {
                            tot += ss.get(site,n).GetLogProb(mat[n][y][k]);
                        }
                        return tot;
                    };
                };

            logprob_gibbs_resample(aa_alloc_(model), aa_alloc_logprob, gen);

            aa_alloc_suffstat_(model).gather();
            gibbs_resample(aa_weights_(model), aa_alloc_suffstat_(model), gen);
        }

    template<class Model, class Gen>
        static auto move_selac(Model& model, Gen& gen) {
            auto selac_logprob = suffstat_logprob(codon_matrices_(model), comp_path_suffstat_(model));

            auto g_update = [&model] () {
                gather(g_(model)); 
                gather(profiles_(model));
                gather(codon_matrices_(model));};
            scaling_move(g_alpha_(model), selac_logprob, 1, 10, gen, g_update);
            scaling_move(g_alpha_(model), selac_logprob, 0.3, 10, gen, g_update);

            auto w_update = [&model] () {
                gather(aadist_(model));
                gather(profiles_(model));
                gather(codon_matrices_(model));};
            scaling_move(wcom_(model), selac_logprob, 1, 10, gen, w_update);
            scaling_move(wcom_(model), selac_logprob, 0.3, 10, gen, w_update);
            scaling_move(wpol_(model), selac_logprob, 1, 10, gen, w_update);
            scaling_move(wpol_(model), selac_logprob, 0.3, 10, gen, w_update);

            /*
            auto psi_update = [&model] () {
                gather(aadist_(model)); 
                gather(profiles_(model));
                gather(codon_matrices_(model));};
            scaling_move(psi_(model), selac_logprob, 1, 10, gen, psi_update);
            scaling_move(psi_(model), selac_logprob, 0.3, 10, gen, psi_update);
            */
        }

    template<class Model, class Gen>
        static auto move_chrono(Model& model, Gen& gen) {

            const Tree& tree = get<chronotree,value>(model).get_tree();

            auto node_update = tree_factory::do_around_node(tree, array_element_gather(synrate_(model)));
            auto node_suffstat_logprob = tree_factory::sum_around_node(tree,
                    tree_factory::suffstat_logprob(bl_suffstat_(model),
                        n_to_n(get<synrate,value>(model))));

            auto node_logprob =
                [&process = brownian_process_(model), node_suffstat_logprob] (int node) {
                    return process.GetNodeLogProb(node) + node_suffstat_logprob(node);
                };

            get<chronotree,value>(model).MoveTimes(node_update, node_logprob);
        }

    template<class Model, class Gen>
        static auto move_synrate(Model& model, Gen& gen) {

            const Tree& tree = get<chronotree,value>(model).get_tree();

            auto synrate_node_update = tree_factory::do_around_node(tree, 
                    array_element_gather(synrate_(model)));

            auto synrate_node_logprob = tree_factory::sum_around_node(tree,
                    tree_factory::suffstat_logprob(bl_suffstat_(model),
                        n_to_n(get<synrate,value>(model))));

            brownian_process_(model).SingleNodeMove(0, 1.0, synrate_node_update, synrate_node_logprob);
            brownian_process_(model).SingleNodeMove(0, 0.3, synrate_node_update, synrate_node_logprob);
        }

    template<class Model, class Gen>
        static auto move_rootNe(Model& model, Gen& gen) {

            auto Ne_update = [&model] () {
                auto rootmats = subsets::slice011(codon_matrices_(model), 0);
                gather(rootmats);
            };

            auto Ne_logprob = 
                [&ss = comp_path_suffstat_(model), &mat = get<codon_matrices,value>(model)] ()  {
                size_t node = 0;
                double tot = 0;
                for (size_t cat=0; cat<mat[node].size(); cat++) {
                    for (size_t aa=0; aa<mat[node][0].size(); aa++) {
                        tot += ss.get(node,cat,aa).GetLogProb(mat[node][cat][aa]);
                    }
                }
                return tot;
            };

            scaling_move(rootNe_(model), Ne_logprob, 0.3, 10, gen, Ne_update);
        }

    template<class Model, class Gen>
        static auto move_Ne(Model& model, Gen& gen) {

            const Tree& tree = get<chronotree,value>(model).get_tree();

            auto Ne_node_update = tree_factory::do_around_node(tree,
                    [&model] (int branch)   {
                        auto branchne = subsets::element(branchNe_(model), branch);
                        gather(branchne);
                        auto branchmats = subsets::slice011(codon_matrices_(model), branch+1);
                        gather(branchmats);
                    });

            auto Ne_branch_logprob = 
                [&ss = comp_path_suffstat_(model), &mat = get<codon_matrices,value>(model)]
                (size_t branch) {
                    size_t node = branch+1;
                    double tot = 0;
                    for (size_t cat=0; cat<mat[node].size(); cat++) {
                        for (size_t aa=0; aa<mat[node][0].size(); aa++) {
                            tot += ss.get(node,cat,aa).GetLogProb(mat[node][cat][aa]);
                        }
                    }
                    return tot;
                };

            auto Ne_node_logprob = tree_factory::sum_around_node(tree,Ne_branch_logprob);
                    // tree_factory::suffstat_logprob(comp_path_suffstat_(model), 
                    //                 n_to_n(get<codon_matrices,value>(model))));

            brownian_process_(model).SingleNodeMove(1, 1.0, Ne_node_update, Ne_node_logprob);
            brownian_process_(model).SingleNodeMove(1, 0.3, Ne_node_update, Ne_node_logprob);
        }

    template<class Model, class Gen>
        static auto move_traits(Model& model, Gen& gen) {

            auto no_update = [] (int node) {};
            auto no_logprob = [] (int node) {return 0;};

            size_t dim = get<sigma,value>(model).size();

            for (size_t i=2; i<dim; i++) {
                brownian_process_(model).SingleNodeMove(i, 1.0, no_update, no_logprob);
                brownian_process_(model).SingleNodeMove(i, 0.3, no_update, no_logprob);
            }
        }

    template<class Model, class Gen>
        static auto move_sigma(Model& model, Gen& gen)  {
            covmat_suffstat_(model).gather();
            gibbs_resample(sigma_(model), covmat_suffstat_(model), gen);
        }

    template<class Model, class Gen>
        static auto move_params(Model& model, Gen& gen) {

            bl_suffstat_(model).gather();
            std::cerr << "move chrono\n";
            selacNe::move_chrono(model, gen);
            std::cerr << "move synrate\n";
            selacNe::move_synrate(model, gen);

            site_path_suffstat_(model).gather();

            std::cerr << "resample alloc\n";
            selacNe::resample_g_alloc(model, gen);
            selacNe::resample_aa_alloc(model, gen);

            comp_path_suffstat_(model).gather();

            std::cerr << "move Ne\n";
            selacNe::move_Ne(model, gen);
            std::cerr << "move traits\n";
            selacNe::move_traits(model, gen);

            std::cerr << "move sigma\n";
            selacNe::move_sigma(model, gen);

            std::cerr << "move root Ne\n";
            selacNe::move_rootNe(model, gen);

            std::cerr << "move selac\n";
            selacNe::move_selac(model, gen);
            std::cerr << "move nuc\n";
            selacNe::move_nuc(model, gen);
            std::cerr << "move ok\n";
        }

    template<class Model>
        static auto get_total_ds(Model& model)  {
            auto& ds = get<synrate,value>(model);
            double tot = 0;
            for (size_t b=0; b<ds.size(); b++)   {
                tot += ds[b];
            }
            return tot;
        }

    template<class Model>
        static auto get_mean_Ne(Model& model)  {
            auto& branchne = get<branchNe,value>(model);
            double tot = 0;
            for (size_t b=0; b<branchne.size(); b++)   {
                tot += branchne[b];
            }
            tot /= branchne.size();
            return tot;
        }
};
