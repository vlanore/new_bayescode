#pragma once

#include "bayes_toolbox/src/basic_moves.hpp"
#include "bayes_toolbox/src/distributions/exponential.hpp"
#include "bayes_toolbox/src/distributions/gamma.hpp"
#include "bayes_toolbox/src/operations/draw.hpp"
#include "bayes_toolbox/src/structure/array_utils.hpp"
#include "bayes_toolbox/src/structure/model.hpp"
#include "bayes_toolbox/src/structure/node.hpp"
#include "bayes_toolbox/utils/tagged_tuple/src/fancy_syntax.hpp"
#include "global/logging.hpp"
#include "tree/implem.hpp"

TOKEN(bl_array)
TOKEN(suffstats)

/* Array of branch lengths, gamma iid with fixed mean and invshape.
Initialized with branch length from input tree. */
struct branchlengths_submodel {
    static auto make(TreeParser& parser, const Tree& tree, double mean, double invshape) {
        DEBUG("Getting branch lengths from tree");
        const size_t nb_branches = parser.get_tree().nb_nodes();
        auto initial_bl = branch_container_from_parser<double>(
            parser, [](int i, const auto& tree) { return stod(tree.tag(i, "length")); });

        DEBUG("Creating branch length array of gamma nodes (length {})", nb_branches);
        auto bl_array = make_node_array<gamma_ss>(
            nb_branches, n_to_constant(1. / invshape), n_to_constant(mean * invshape));
        set_value(bl_array, initial_bl);

        // suffstats
        PoissonSuffStatBranchArray suffstats{tree};

        // return model
        return make_model(bl_array_ = std::move(bl_array), suffstats_ = suffstats);
    }

    template <class BLModel, class Gen>
    static auto gibbs_resample(BLModel& model, Gen& gen) {
        auto& raw_vec = get<bl_array, value>(model);
        auto& ss = get<suffstats>(model);

        for (size_t i = 0; i < raw_vec.size(); i++) {
            auto& local_ss = ss.GetVal(i);

            raw_vec[i] = gamma_ss::draw(                                             //
                get<bl_array, params, shape>(model)(i) + local_ss.GetCount(),        //
                get<bl_array, params, struct scale>(model)(i) + local_ss.GetBeta(),  //
                gen);
            assert(raw_vec[i] != 0);
        }
    }
};