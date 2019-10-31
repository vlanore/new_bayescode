#pragma once

#include "bayes_toolbox/src/basic_moves.hpp"
#include "bayes_toolbox/src/distributions/exponential.hpp"
#include "bayes_toolbox/src/distributions/gamma.hpp"
#include "bayes_toolbox/src/operations/draw.hpp"
#include "bayes_toolbox/src/operations/logprob.hpp"
#include "bayes_toolbox/src/structure/array_utils.hpp"
#include "bayes_toolbox/src/structure/model.hpp"
#include "bayes_toolbox/src/structure/node.hpp"
#include "bayes_toolbox/utils/tagged_tuple/src/fancy_syntax.hpp"
#include "bayes_utils/src/logging.hpp"
#include "suffstat_wrappers.hpp"
#include "tree/implem.hpp"

TOKEN(gamma_array)

struct iidgamma_mi {
    template <class Mean, class InvShape, class Gen>
    static auto make(size_t size, Mean mean, InvShape invshape, Gen& gen)    {
        auto gamma_array = make_node_array<gamma_mi>(size, mean, invshape);
        draw(gamma_array, gen);
        return make_model(gamma_array_ = std::move(gamma_array));
    }

    template <class IIDGammaModel, class SS, class Gen>
    static void gibbs_resample(IIDGammaModel& model, Proxy<SS&, int>& ss, Gen& gen) {
        size_t nsite = get<gamma_array, value>(model).size();
        for (size_t i=0; i<nsite; i++)  {
            double mean = get<gamma_array, params, gam_mean>(model)(i);
            double invshape = get<gamma_array, params, gam_invshape>(model)(i);
            double alpha = 1. / invshape;
            double beta = mean * invshape;
            auto ss_value = ss.get(i);
            gamma_sr::draw(get<gamma_array, value>(model)[i], alpha + ss_value.count, beta + ss_value.beta, gen);
        }
    }
};
