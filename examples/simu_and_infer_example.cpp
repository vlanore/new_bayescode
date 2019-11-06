/*Copyright or © or Copr. CNRS (2019). Contributors:
- Vincent Lanore. vincent.lanore@gmail.com

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

#include <iostream>

#include "bayes_toolbox.hpp"

#include "components/ChainCheckpoint.hpp"
#include "components/ChainDriver.hpp"
#include "components/ConsoleLogger.hpp"
#include "components/MoveScheduler.hpp"
#include "components/StandardTracer.hpp"
#include "components/restart_check.hpp"
#include "submodels/move_reporter.hpp"
#include "submodels/submodel_external_interface.hpp"
#include "tagged_tuple/src/fancy_syntax.hpp"

using namespace std;

TOKEN(alpha_)
TOKEN(beta_)
TOKEN(lambda)
TOKEN(K)

auto poisson_gamma(size_t size, size_t size2) {
    auto alpha_ = make_node<exponential>(1.0);
    auto beta_ = make_node<exponential>(1.0);
    auto lambda = make_node_array<gamma_ss>(size, n_to_one(alpha_), n_to_one(beta_));
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
    constexpr size_t nb_it{1'000}, len_lambda{20}, len_K{200}, every(100);
    auto m = poisson_gamma(len_lambda, len_K);

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
        scaling_move(alpha__(m), simple_logprob(lambda_(m)), 1.0, 1, gen);
        scaling_move(beta__(m), simple_logprob(lambda_(m)), 1.0, 1, gen);
        scaling_move(lambda_(m), matrix_row_logprob(K_(m)), 1.0, 1, gen);
    });

    // initializing components
    ChainDriver chain_driver{chain_name, every, nb_it};

    ConsoleLogger console_logger;
    // ChainCheckpoint chain_checkpoint(chain_name + ".param", chain_driver, m);
    StandardTracer trace(m, chain_name);

    // registering components to chain driver
    chain_driver.add(scheduler);
    // chain_driver.add(console_logger);
    // chain_driver.add(chain_checkpoint);
    chain_driver.add(trace);
    chain_driver.add(ms);

    // launching chain!
    chain_driver.go();
}
