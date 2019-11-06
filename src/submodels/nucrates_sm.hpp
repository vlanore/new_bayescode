#pragma once

#include "bayes_toolbox.hpp"
#include "bayes_utils/src/logging.hpp"
#include "lib/GTRSubMatrix.hpp"
#include "tree/implem.hpp"

#include "gtr.hpp"

TOKEN(eq_freq)
TOKEN(exch_rates)
TOKEN(nuc_matrix)

struct nucrates_sm {
    template <class RRCenter, class RRInvConc, class StatCenter, class StatInvConc, class Gen>
    static auto make(RRCenter _rrcenter, RRInvConc _rrinvconc, StatCenter _statcenter, StatInvConc _statinvconc, Gen& gen)  {

        auto rrcenter = make_param<std::vector<double>>(std::forward<RRCenter>(_rrcenter));
        auto rrinvconc = make_param<double>(std::forward<RRInvConc>(_rrinvconc));
        auto statcenter = make_param<std::vector<double>>(std::forward<StatCenter>(_statcenter));
        auto statinvconc = make_param<double>(std::forward<StatInvConc>(_statinvconc));

        auto exchangeability_rates = 
            make_node_with_init<dirichlet_cic>(std::vector<double>(6, 1./6), rrcenter, rrinvconc);
        draw(exchangeability_rates, gen);

        auto equilibrium_frequencies =
            make_node_with_init<dirichlet_cic>(std::vector<double>(4, 1./4), statcenter, statinvconc);
        draw(equilibrium_frequencies, gen);

        auto nuc_matrix = make_dnode_with_init<gtr>(
                {4, get<value>(exchangeability_rates), get<value>(equilibrium_frequencies), true},
                get<value>(exchangeability_rates),
                get<value>(equilibrium_frequencies));

        gather(nuc_matrix);

        return make_model(                                  
            exch_rates_ = std::move(exchangeability_rates), 
            eq_freq_ = std::move(equilibrium_frequencies),  
            nuc_matrix_ = std::move(nuc_matrix));
    }

    // move with arbitrary logprob function (for children nodes only)
    template <class SubModel, class LogProb, class Gen, class Reporter = NoReport>
    static void generic_move_nucrates(SubModel& model, LogProb& nuc_logprob, Gen& gen, 
        size_t nrep = 1, double tuning_modulator = 1.0,
        Reporter reporter = {},
        std::vector<double> exch_tunings = {0.1, 0.03, 0.01},
        std::vector<double> eqfreqs_tunings = {0.1, 0.03})  {

        auto nuc_update = simple_gather(nuc_matrix_(model));

        for (size_t i = 0; i<nrep; i++) {

            for (auto& t : exch_tunings)   {
                auto proposal = [&t, &tuning_modulator] (auto& value, auto& gen) {return profile_move(value, t*tuning_modulator, gen);};
                mh_move(exch_rates_(model), nuc_logprob, proposal, 1, gen, nuc_update);
            }

            for (auto& t : eqfreqs_tunings)    {
                auto proposal = [&t, &tuning_modulator] (auto& value, auto& gen) {return profile_move(value, t*tuning_modulator, gen);};
                mh_move(eq_freq_(model), nuc_logprob, proposal, 1, gen, nuc_update);
            }
        }
    }

    // specialised version when we give nuc path suffstats:
    template <class SubModel, class SS, class Gen, class Reporter = NoReport>
    static void move_nucrates(SubModel& model, Proxy<SS&>& ss, Gen& gen, 
        size_t nrep = 1, double tuning_modulator = 1.0,
        Reporter reporter = {},
        std::vector<double> exch_tunings = {0.1, 0.03, 0.01},
        std::vector<double> eqfreqs_tunings = {0.1, 0.03})  {

        auto nuc_logprob = suffstat_logprob(nuc_matrix_(model), ss);
        auto nuc_update = simple_gather(nuc_matrix_(model));

        for (size_t i = 0; i<nrep; i++) {

            for (auto& t : exch_tunings)   {
                auto proposal = [&t, &tuning_modulator] (auto& value, auto& gen) {return profile_move(value, t*tuning_modulator, gen);};
                mh_move(exch_rates_(model), nuc_logprob, proposal, 1, gen, nuc_update);
            }

            for (auto& t : eqfreqs_tunings)    {
                auto proposal = [&t, &tuning_modulator] (auto& value, auto& gen) {return profile_move(value, t*tuning_modulator, gen);};
                mh_move(eq_freq_(model), nuc_logprob, proposal, 1, gen, nuc_update);
            }
        }
    }
};
