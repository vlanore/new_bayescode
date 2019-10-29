#pragma once

#include "bayes_toolbox.hpp"
#include "bayes_utils/src/logging.hpp"
#include "lib/GTRSubMatrix.hpp"
#include "tree/implem.hpp"
#include "mgomega.hpp"

TOKEN(eq_freq)
TOKEN(exch_rates)
TOKEN(nuc_matrix)

// @todo: move elsewhere
std::vector<double> normalize(const std::vector<double>& vec) {
    double sum = 0;
    for (auto e : vec) { sum += e; }
    std::vector<double> result(vec.size());
    for (size_t i = 0; i < vec.size(); i++) { result[i] = vec[i] / sum; }
    return result;
}

struct nucrates_sm {
    template <class RRCenter, class RRInvConc, class StatCenter, class StatInvConc, class Gen>
    static auto make(RRCenter rrcenter, RRInvConc rrinvconc, StatCenter statcenter, StatInvConc statinvconc, Gen& gen)  {
        /* -- */
        auto exchangeability_rates = make_node<dirichlet_cic>(rrcenter, rrinvconc);
        set_value(exchangeability_rates, std::vector<double>(6, 0));
        draw(exchangeability_rates, gen);
        DEBUG("GTR model: exchangeability rates are {}.",
            vector_to_string(get<value>(exchangeability_rates)));

        auto equilibrium_frequencies =
            make_node<dirichlet_cic>(statcenter, statinvconc);
        set_value(equilibrium_frequencies, std::vector<double>(4, 0));
        draw(equilibrium_frequencies, gen);
        DEBUG("GTR model: equilibrium frequencies are {}.",
            vector_to_string(get<value>(equilibrium_frequencies)));

        auto nuc_matrix = make_dnode_with_init<gtr>(
                {4, get<value>(exchangeability_rates), get<value>(equilibrium_frequencies), true},
                get<value>(exchangeability_rates),
                get<value>(equilibrium_frequencies));

        return make_model(                                   //
            exch_rates_ = std::move(exchangeability_rates),  //
            eq_freq_ = std::move(equilibrium_frequencies),   //
            nuc_matrix_ = std::move(nuc_matrix));
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

    template <class SubModel, class LogProb, class Update, class Gen, class Reporter = NoReport>
    static void move_nucrates(SubModel& model,
        LogProb logprob_children, Update update, Gen& gen, 
        size_t nrep = 1, double tuning = 1.0,
        Reporter reporter = {}) {
        /* -- */
        std::vector<double> exch_tunings = {0.1, 0.03, 0.01};
        for (auto& t : exch_tunings)   {
            t *= tuning;
        }
        std::vector<double> eqfreqs_tunings = {0.1, 0.03};
        for (auto& t : eqfreqs_tunings)    {
            t *= tuning;
        }
        for (size_t i = 0; i<nrep; i++) {
            move_exch_rates(model, exch_tunings, logprob_children, update, gen);
            move_eq_freqs(model, eqfreqs_tunings, logprob_children, update, gen);
            /*
            move_exch_rates(model, exch_tunings, logprob_children, update, gen, reporter("exch_rates"));
            move_eq_freqs(model, eqfreqs_tunings, logprob_children, update, gen, reporter("eq_freqs"));
            */
        }
    }

    // specialised version when we give nuc path suffstats:
    // - the log prob of the children is given by the suffstats
    // - we also know that the only update to do is the nuc matrix itself
    template <class SubModel, class SS, class Gen, class Reporter = NoReport>
    static void move_nucrates(SubModel& model, SS& ss, Gen& gen, 
        size_t nrep = 1, double tuning = 1.0,
        Reporter reporter = {}) {

        auto logprob = [&model, &ss]() {
            return ss.get().GetLogProb(get<nuc_matrix, value>(model));
        };

        auto update = [&model]() {gather(nuc_matrix_(model));};

        move_nucrates(model, logprob, update, gen, nrep, tuning, reporter);
    }
};
