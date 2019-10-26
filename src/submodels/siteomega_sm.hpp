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

TOKEN(site_omega_array)

struct siteomega_sm {
    template <class Mean, class InvShape, class Gen>
    static auto make(size_t nsite, Mean mean, InvShape invshape, Gen& gen)    {
        DEBUG("Making omega with parameters mean={} and invshape={}", mean(), invshape());
        auto site_omega_array = make_node_array<gamma_ss>(nsite, [invshape] (int) {return 1. / invshape();}, [mean, invshape] (int) {return mean() * invshape();});
        draw(site_omega_array, gen);
        return make_model(site_omega_array_ = std::move(site_omega_array));
    }

    template <class OmegaModel, class Gen>
    static void gibbs_resample(OmegaModel& model, Proxy<omega_suffstat_t, int>& ss, Gen& gen) {
        /* -- */
        size_t nsite = get<site_omega_array, value>(model).size();
        double alpha = get<site_omega_array, params, shape>(model)(0);
        double beta = 1. / get<site_omega_array, params, struct scale>(model)(0);
        for (size_t i=0; i<nsite; i++)  {
            auto ss_value = ss.get(i);
            gamma_sr::draw(get<site_omega_array, value>(model)[i], alpha + ss_value.count, beta + ss_value.beta, gen);
        }
    }
};
