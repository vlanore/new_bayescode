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
#include "distributions/beta.hpp"
#include "distributions/exponential.hpp"
#include "mpi_components/broadcast.hpp"
#include "mpi_components/gather.hpp"
#include "mpi_components/unique_ptr_utils.hpp"
#include "submodels/submodel_external_interface.hpp"
using namespace std;

TOKEN(beta_weight_a)
TOKEN(beta_weight_b)
TOKEN(p)

// Model definition
auto common_bernoulli_model(size_t nb_coins, size_t nb_draws) {
    auto beta_weight_a = make_node<exponential>(1.0);
    auto beta_weight_b = make_node<exponential>(1.0);
    auto p = make_node_array<beta_ss>(nb_coins, n_to_one(beta_weight_a), n_to_one(beta_weight_b));
    // clang-format off
    return make_model(
            beta_weight_a_ = move(beta_weight_a),
            beta_weight_b_ = move(beta_weight_b),
            p_ = move(p)
    );  // clang-format on
}

int compute(char, char**) {
    auto rank = MPI::p->rank;  // getting MPI rank
    auto size = MPI::p->size;  // ... and size
    const auto master = !rank;
    auto gen = make_generator();
    const int n_obs = 10;
    const int n_coins = 5;

    // List of indices for data partition
    // indices are partitioned into size-1 bins, starting at process 1
    IndexSet indices(n_coins);
    std::generate(
        indices.begin(), indices.end(), [n = 0]() mutable { return std::to_string(n++); });
    MPI::p->message("Hi");
    Partition partition(indices, size - 1, 1);

    constexpr size_t nb_it{100'000};
    // constexpr size_t burn_in{nb_it / 10};

    // Common part of the model, shared between master and slaves
    auto m = common_bernoulli_model(master ? n_coins : partition.my_partition_size(), n_obs);
    // Master sets alpha and beta parameters and broadcasts it to slaves
    auto broadcast_a_b = broadcast(get<beta_weight_a, value>(m), get<beta_weight_b, value>(m));
    // INFO(get<beta_weight_a, value>(m));

    // Simulation
    if (master) {
        auto v = make_collection(get<beta_weight_a>(m), get<beta_weight_b>(m));
        draw(v, gen);
    }
    master_to_slave(broadcast_a_b);

    // Slaves simulate tosses, master has nullptr

    auto bern = slave_only_ptr([n_coins, n_obs, &m]() {
        return make_node_matrix<bernoulli>(
            n_coins, n_obs, [&v = get<p, value>(m)](int i, int) { return v[i]; });
    });


    if (!master) { draw(*bern, gen); }
    // Simulation done

    // Initialize node states to be different from simulation
    if (master) {
        auto v = make_collection(beta_weight_a_(m), beta_weight_b_(m));
        draw(v, gen);
    }
    master_to_slave(broadcast_a_b);
    if (!master) { draw(p_(m), gen); }
    // Initialisation done

    auto gather_p = gather(partition, get<p, value>(m));

    auto scheduler = make_move_scheduler([&]() {
        // Gather current p_ values from slaves
        slave_to_master(gather_p);
        // Master moves alpha, beta
        if (master) {
            scaling_move(beta_weight_a_(m), simple_logprob(p_(m)), 1.0, 1, gen);
            scaling_move(beta_weight_b_(m), simple_logprob(p_(m)), 1.0, 1, gen);
        }
        master_to_slave(broadcast_a_b);

        if (!master) {
            auto& k_matrix = get<value>(*bern);

            for (size_t i = 0; i < partition.my_partition_size(); ++i) {
                int k_sum = 0;
                for (auto e : k_matrix[i]) { k_sum += e; }

                // std::accumulate(k_matrix.begin(), k_matrix.end(), 0);
                beta_ss::draw(get<p, value>(m)[i], get<beta_weight_a, value>(m) + k_sum,
                    get<beta_weight_b, value>(m) + n_obs - k_sum, gen);
            }
        }
    });

    // initializing components
    ChainDriver chain_driver{"CoinTossesMPI", 1, nb_it};
    ConsoleLogger console_logger;
    StandardTracer trace(m, "CoinTossesMPI");


    // registering components to chain driver
    if (master) { chain_driver.add(trace); }
    chain_driver.add(scheduler);
    chain_driver.add(console_logger);

    // launching chain!
    chain_driver.go();

    return 0;
}


int main(int argc, char** argv) { mpi_run(argc, argv, compute); }
