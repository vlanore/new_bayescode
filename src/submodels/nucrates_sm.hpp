#pragma once

#include "bayes_toolbox/src/basic_moves.hpp"
#include "bayes_toolbox/src/distributions/dirichlet.hpp"
#include "bayes_toolbox/src/distributions/exponential.hpp"
#include "bayes_toolbox/src/distributions/gamma.hpp"
#include "bayes_toolbox/src/operations/draw.hpp"
#include "bayes_toolbox/src/operations/logprob.hpp"
#include "bayes_toolbox/src/structure/array_utils.hpp"
#include "bayes_toolbox/src/structure/model.hpp"
#include "bayes_toolbox/src/structure/node.hpp"
#include "bayes_toolbox/utils/tagged_tuple/src/fancy_syntax.hpp"
#include "global/logging.hpp"
#include "lib/GTRSubMatrix.hpp"
#include "tree/implem.hpp"

TOKEN(eq_freq)
TOKEN(exch_rates)
TOKEN(nuc_matrix)
TOKEN(matrix_proxy)

// @todo: move elsewhere
std::vector<double> normalize(const std::vector<double>& vec) {
    double sum = 0;
    for (auto e : vec) { sum += e; }
    std::vector<double> result(vec.size());
    for (size_t i = 0; i < vec.size(); i++) { result[i] = vec[i] / sum; }
    return result;
}

struct nucrates_sm {
    template <class Gen>
    static auto make(std::function<const std::vector<double>&()> nucrelratecenter, std::function<const double&()> nucrelrateinvconc,
        std::function<const std::vector<double>&()> nucstatcenter, std::function<const double& ()> nucstatinvconc, Gen& gen) {
        /* -- */
        auto exchangeability_rates = make_node<dirichlet_cic>(nucrelratecenter, nucrelrateinvconc);
        set_value(exchangeability_rates, std::vector<double>(6, 0));
        draw(exchangeability_rates, gen);
        DEBUG("GTR model: exchangeability rates are {}.",
            vector_to_string(get<value>(exchangeability_rates)));

        auto equilibrium_frequencies =
            make_node<dirichlet_cic>(nucstatcenter, nucstatinvconc);
        set_value(equilibrium_frequencies, std::vector<double>(4, 0));
        draw(equilibrium_frequencies, gen);
        DEBUG("GTR model: equilibrium frequencies are {}.",
            vector_to_string(get<value>(equilibrium_frequencies)));

        auto nuc_matrix = std::make_unique<GTRSubMatrix>(
            4, get<value>(exchangeability_rates), get<value>(equilibrium_frequencies), true);

        auto matrix_proxy = NucMatrixProxy(*nuc_matrix, get<value>(equilibrium_frequencies));

        return make_model(                                   //
            exch_rates_ = std::move(exchangeability_rates),  //
            eq_freq_ = std::move(equilibrium_frequencies),   //
            nuc_matrix_ = std::move(nuc_matrix),             //
            matrix_proxy_ = matrix_proxy);
    }

    template <class SubModel, class LogProb, class Update, class Gen, class Reporter = NoReport>
    static void move_exch_rates(SubModel& model, std::vector<double> tunings,
        LogProb logprob_children, Update update, Gen& gen, Reporter reporter = {}) {
        /* -- */
        auto& target = exch_rates_(model);

        for (auto tuning : tunings) {
            auto bkp = backup(target);
            double logprob_before = logprob_children() + logprob(target);
            double log_hastings = profile_move(get<value>(target), tuning, gen);
            update();
            double logprob_after = logprob_children() + logprob(target);
            bool accept = decide(logprob_after - logprob_before + log_hastings, gen);
            if (!accept) {
                restore(target, bkp);
                update();
            }
            reporter.report(accept);
        }
    }

    template <class SubModel, class LogProb, class Update, class Gen, class Reporter = NoReport>
    static void move_eq_freqs(SubModel& model, std::vector<double> tunings,
        LogProb logprob_children, Update update, Gen& gen, Reporter reporter = {}) {
        /* -- */
        auto& target = eq_freq_(model);

        for (auto tuning : tunings) {
            auto bkp = backup(target);
            double logprob_before = logprob_children() + logprob(target);
            double log_hastings = profile_move(get<value>(target), tuning, gen);
            update();
            double logprob_after = logprob_children() + logprob(target);
            bool accept = decide(logprob_after - logprob_before + log_hastings, gen);
            if (!accept) {
                restore(target, bkp);
                update();
            }
            reporter.report(accept);
        }
    }
};
