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

TOKEN(omega)

struct omega_sm {
    template <class Mean, class InvShape, class Gen>
    static auto make(Mean mean, InvShape invshape, Gen& gen)    {
        DEBUG("Making omega with parameters mean={} and invshape={}", mean(), invshape());
        auto omega = make_node<gamma_ss>([invshape] () {return 1. / invshape();}, [mean, invshape] () {return mean() * invshape();});
        draw(omega, gen);
        return make_model(omega_ = std::move(omega));
    }

    template <class OmegaModel, class Lambda, class Gen>
    static void move(OmegaModel& model, Lambda children_logprob, Gen& gen) {
        auto& omega = omega_(model);
        auto full_logprob = [&omega, &children_logprob]() {
            return logprob(omega) + children_logprob();
        };
        scaling_move(omega, full_logprob, gen);
        // TRACE("Omega = {}", raw_value(omega));
    }

    template <class OmegaModel, class Gen>
    static void gibbs_resample(OmegaModel& model, Proxy<omega_suffstat_t>& ss, Gen& gen) {
        /* -- */
        double alpha = get<omega, params, shape>(model)();
        double beta = 1. / get<omega, params, struct scale>(model)();
        auto ss_value = ss.get();
        gamma_sr::draw(get<omega, value>(model), alpha + ss_value.count, beta + ss_value.beta, gen);
        // TRACE("Omega = {}", get<omega, value>(model));
    }
};
