/*Copyright or © or Copr. CNRS (2019). Contributors:
- Vincent Lanore. vincent.lanore@gmail.com
- Bastien Boussau. boussau@gmail.com

This software is a computer program whose purpose is to provide a set of C++ data structures and
functions to perform Bayesian inference with MCMC algorithms.

This software is governed by the CeCILL-C license under French law and abiding by the rules of
distribution of free software. You can use, modify and/ or redistribute the software under the terms
of the CeCILL-C license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info".

As a counterpart to the access to the source code and rights to copy, modify and redistribute
granted by the license, users are provided only with a limited warranty and the software's author,
the holder of the economic rights, and the successive licensors have only limited liability.

In this respect, the user's attention is drawn to the risks associated with loading, using,
modifying and/or developing or reproducing the software by the user in light of its specific status
of free software, that may mean that it is complicated to manipulate, and that also therefore means
that it is reserved for developers and experienced professionals having in-depth computer knowledge.
Users are therefore encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or data to be ensured and,
more generally, to use and operate it in the same conditions as regards security.

The fact that you are presently reading this means that you have had knowledge of the CeCILL-C
license and that you accept its terms.*/


/*
In this program, we implement the model from the practical class "Introduction à JAGS : poids des enfants à la naissance" from the course Statistiques Bayésiennes et Applications.
There are boy and girl babies whose weights are given in a table, and we want to estimate the average weights per sex.
The model looks like this, in JAGS: 
model
{
for(i in 1 : N)
{
poids[i] ~ dnorm(mu[i], tau)
mu[i] <- muf * sexe[i] + mug * (1 - sexe[i])
}
mug ~ dunif(2500, 5000)
muf ~ dunif(2500, 5000)
tau <- 1/(sigma^2)
sigma ~ dunif(200, 800)
}
*/


#include <iostream>
#include "basic_moves.hpp"
#include "components/ChainCheckpoint.hpp"
#include "components/ChainDriver.hpp"
#include "components/ConsoleLogger.hpp"
#include "components/MoveScheduler.hpp"
#include "components/StandardTracer.hpp"
#include "components/restart_check.hpp"
#include "distributions/uniform.hpp"
#include "distributions/gamma.hpp"
#include "distributions/poisson.hpp"
#include "bayes_toolbox/src/bayes_toolbox.hpp"
//#include "global/logging.hpp"
#include "mcmc_utils.hpp"
#include "operations/backup.hpp"
#include "operations/logprob.hpp"
#include "operations/raw_value.hpp"
#include "operations/set_value.hpp"
#include "structure/new_view.hpp"
//#include "structure/array_utils.hpp"
#include "submodels/move_reporter.hpp"
#include "submodels/submodel_external_interface.hpp"
//#include "suffstat_utils.hpp"
#include "tagged_tuple/src/fancy_syntax.hpp"

using namespace std;

TOKEN(alpha_)
TOKEN(beta_)
TOKEN(lambda)
TOKEN(K)

auto poisson_gamma(size_t size, std::vector<int> sex_table) {
    auto sigma_ = make_node<uniform>(200.0, 800.0);
    auto muf_ =  make_node<uniform>(2500.0, 5000.0); 
    auto mug_ =  make_node<uniform>(2500.0, 5000.0);

    make_node_array<normal>(size, [& muf = get<value>(muf_), & mug = get<value>(mug_), & sex_table](int i) { return muf * sex_table[i] + mug * (1 - sex_table[i]); }, sigma_);


    auto K = make_node_matrix<poisson>(
        size, size2, [& v = get<value>(lambda)](int i, int) { return v[i]; });
    // clang-format off
    return make_model(
         alpha__ = move(alpha_),
            beta__ = move(beta_),
        lambda_ = move(lambda),
             K_ = move(K)
    );  // clang-format on
}


int main() {
    auto gen = make_generator();
    std::string chain_name = "simu_and_infer_chain";
    constexpr size_t nb_it{100'000}, len_lambda{20}, len_K{200}, every(1);
    auto m = poisson_gamma(len_lambda, len_K);


//    auto v = make_view<alpha, beta, lambda, K>(m);

    auto v = make_collection(alpha__(m), beta__(m), lambda_(m), K_(m));

    draw(v, gen);
    // display node value in stdout
    INFO("Alpha = {}", raw_value(alpha__(m)));
    INFO("Beta = {}", raw_value(beta__(m)));
    INFO("Lambda = {}", vector_to_string(get<lambda, value>(m)));
    // INFO("K = {}", get<K, value>(m));

    //    set_value(K_(m), {{1, 2, 1}, {1, 2, 2}, {1, 2, 1}, {2, 1, 2}, {2, 1, 3}});

    // move success stats
    MoveStatsRegistry ms;

    // move schedule
    auto scheduler = make_move_scheduler([&gen, &m, &ms]() {
        // // move alpha
        // scaling_move(alpha_(m), make_view<alpha, lambda>(m), gen);
        // // move beta
        // scaling_move(beta_(m), make_view<beta, lambda>(m), gen);
        scaling_move(alpha__(m), logprob_of_blanket(make_collection(alpha__(m), lambda_(m))), gen);
        scaling_move(beta__(m), logprob_of_blanket(make_collection(beta__(m), lambda_(m))), gen);


        for (size_t i = 0; i < len_lambda; i++) {
            auto lambda_mb =
                make_collection(subsets::row(K_(m), i), subsets::element(lambda_(m), i));
                mh_move(lambda_(m), logprob_of_blanket(lambda_mb),
                        [i](auto& value, auto& gen) { return scale(value[i], gen); }, gen);
        }
    });


    // initializing components
    ChainDriver chain_driver{chain_name, every, nb_it};

    ConsoleLogger console_logger;
    // ChainCheckpoint chain_checkpoint(chain_name + ".param", chain_driver, m);
    StandardTracer trace(m, chain_name);

    // registering components to chain driver
    chain_driver.add(scheduler);
    chain_driver.add(console_logger);
    // chain_driver.add(chain_checkpoint);
    chain_driver.add(trace);
    chain_driver.add(ms);

    // launching chain!
    chain_driver.go();


    // The code below works and corresponds to the code above, in a more verbose version.
    // double alpha_sum{0}, beta_sum{0};
    // vector<double> lambda_sum (len_lambda, 0.0);
    //
    // for (size_t it = 0; it < nb_it; it++) {
    //     //INFO("Alpha = {}\n", raw_value(alpha_(m)));
    //     scaling_move(alpha_(m), make_view<alpha, lambda>(m), gen);
    //     scaling_move(beta_(m), make_view<beta, lambda>(m), gen);
    //     alpha_sum += raw_value(alpha_(m));
    //     beta_sum += raw_value(beta_(m));
    //
    //     for (size_t i = 0; i < len_lambda; i++) {
    //         auto lambda_mb = make_view(make_ref<K>(m, i), make_ref<lambda>(m, i));
    //         scaling_move(lambda_(m), lambda_mb, gen, i);
    //         lambda_sum[i] += raw_value(lambda_(m), i);
    //     }
    // }
    // INFO("RESULTS OF THE MCMC: ");
    // INFO("Alpha = {}", alpha_sum / float(nb_it));
    // INFO("beta = {}", beta_sum / float(nb_it));
    //
    // for (size_t i = 0; i < len_lambda; i++) {
    //   lambda_sum[i] = lambda_sum[i]/float(nb_it) ;
    // }
    //
    // //std::cout << "Mean lambda = " << lambda_sum / (float(nb_it) * len_lambda) << std::endl;
    // INFO("Lambda = {}", vector_to_string(lambda_sum));
}
