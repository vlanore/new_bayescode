#pragma once

#include "bayes_toolbox/src/basic_moves.hpp"
#include "bayes_toolbox/src/distributions/exponential.hpp"
#include "bayes_toolbox/src/distributions/gamma.hpp"
#include "bayes_toolbox/src/operations/draw.hpp"
#include "bayes_toolbox/src/structure/array_utils.hpp"
#include "bayes_toolbox/src/structure/model.hpp"
#include "bayes_toolbox/src/structure/node.hpp"
#include "bayes_toolbox/utils/tagged_tuple/src/fancy_syntax.hpp"
#include "bayes_utils/src/logging.hpp"
#include "suffstat_wrappers.hpp"
#include "tree/implem.hpp"

TOKEN(bl_array)

/* Array of branch lengths, gamma iid with fixed mean and invshape.
Initialized with branch length from input tree. */
struct branchlengths_sm {
    template<class Mean, class InvShape>
    static auto make(TreeParser& parser, const Tree& tree, Mean mean, InvShape invshape)    {
        DEBUG("Getting branch lengths from tree");
        const size_t nb_branches = parser.get_tree().nb_nodes() - 1;
        auto initial_bl = branch_container_from_parser<double>(
            parser, [](int i, const auto& tree) { return stod(tree.tag(i, "length")); });
        initial_bl.erase(initial_bl.begin());
        DEBUG("Branch lengths are {}", vector_to_string(initial_bl));

        DEBUG("Creating branch length array of gamma nodes (length {})", nb_branches);
        auto bl_array = make_node_array<gamma_ss>(nb_branches,
                [invshape] (int) {return 1. / invshape();},
                [mean, invshape] (int) {return mean() * invshape();});
        set_value(bl_array, initial_bl);

        // return model
        return make_model(bl_array_ = std::move(bl_array));
    }

    template <class BLModel, class Gen>
    static auto gibbs_resample(
        BLModel& model, Proxy<PoissonSuffStat&, int>& ss, Gen& gen) {
        /* -- */
        auto& raw_vec = get<bl_array, value>(model);

        for (size_t i = 0; i < raw_vec.size(); i++) {
            auto local_ss = ss.get(i);

            auto alpha = get<bl_array, params, shape>(model)(i);
            auto beta = 1. / get<bl_array, params, struct scale>(model)(i);

            gamma_sr::draw(raw_vec[i], alpha + local_ss.count, beta + local_ss.beta, gen);
            assert(raw_vec[i] >= 0);
        }
        // DEBUG("New branch lengths are {}", vector_to_string(raw_vec));
    }
};
