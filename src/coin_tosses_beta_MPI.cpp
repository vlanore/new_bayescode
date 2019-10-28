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
#include "distributions/beta.hpp"
#include "distributions/exponential.hpp"
#include "mpi_components/broadcast.hpp"
#include "mpi_components/gather.hpp"
using namespace std;

TOKEN(beta_weight_a)
TOKEN(beta_weight_b)
TOKEN(p)
TOKEN(bern)

// Model definition
auto common_bernoulli_model(size_t nb_coins, size_t nb_draws) {
    auto beta_weight_a = make_node<exponential>(1.0);
    auto beta_weight_b = make_node<exponential>(1.0);
    auto p = make_node_array<beta_ss>(nb_coins, n_to_one(beta_weight_a), n_to_one(beta_weight_b));
    auto bern = make_node_matrix<bernoulli>(
        nb_coins, nb_draws, [& v = get<value>(p)](int i, int) { return v[i]; });
    // clang-format off
    return make_model(
            beta_weight_a_ = move(beta_weight_a),
            beta_weight_b_ = move(beta_weight_b),
            p_ = move(p),
            bern_ = move(bern)
    );  // clang-format on
}

int compute(char, char**) {
    auto rank = MPI::p->rank;  // getting MPI rank
    auto size = MPI::p->size;  // ... and size
    const auto master = !rank;
    auto gen = make_generator();
    const int n_obs = 20;
    const int n_coins = 10;

    // List of indices for data partition
    // indices are partitioned into size-1 bins, starting at process 1
    IndexSet indices(n_coins);
    std::generate(indices.begin(), indices.end(), [n = 0]() mutable { return n++; });
    Partition partition(indices, size - 1, 1);
    MPI::p->message("Partition size : {}", partition.my_partition_size());
    // if (!master) {
    //     for (auto idx : partition.my_partition()) {
    //         MPI::p->message("Partition");
    //         MPI::p->message(idx);
    //     }
    // }
    constexpr size_t nb_it{1'000'000};
    // constexpr size_t burn_in{nb_it / 10};

    // Common part of the model, shared between master and slaves
    auto m = common_bernoulli_model(master ? n_coins : partition.my_partition_size(), n_obs);
    // Master sets alpha and beta parameters and broadcasts it to slaves
    auto broadcast_a_b = broadcast(get<beta_weight_a, value>(m), get<beta_weight_b, value>(m));
    // INFO(get<beta_weight_a, value>(m));

    // Simulation
    auto v_a_b = make_collection(get<beta_weight_a>(m), get<beta_weight_b>(m));
    if (master) { draw(v_a_b, gen); }
    master_to_slave(broadcast_a_b);

    // Slaves simulate tosses, master has nullptr
    // std::unique_ptr<decltype(slave_bernoulli_model(n_coins, n_obs, m))> tosses_model{nullptr};

    if (!master) {
        // tosses_model = std::make_unique<decltype(slave_bernoulli_model(n_coins, n_obs, m))>(
        // slave_bernoulli_model(n_coins, n_obs, m));

        auto v = make_collection(p_(m), bern_(m));
        draw(v, gen);
        if (rank) {
            for (size_t i = 0; i < partition.my_partition_size(); ++i) {
                MPI::p->message("P for coin {} : {}", (i + 1) * rank, get<p, value>(m)[i]);
                // MPI::p->message("K obs :");
                // for (size_t it = 0; it < n_obs; ++it) {
                //     MPI::p->message("--- {}", get<bern, value>(m)[i][it]);
                // }
            }
        }
    }
    // Simulation done

    // Initialize node states to be different from simulation
    if (master) {
        draw(v_a_b, gen);
        MPI::p->message("Alpha sim = {}", get<beta_weight_a, value>(m));
        MPI::p->message("Beta sim = {}", get<beta_weight_b, value>(m));
    }

    master_to_slave(broadcast_a_b);

    if (!master) { draw(p_(m), gen); }
    // Initialisation done

    auto v_weight_a = make_collection(beta_weight_a_(m), p_(m));
    auto v_weight_b = make_collection(beta_weight_b_(m), p_(m));


    auto gather_p = gather(partition, get<p, value>(m));

    vector<double> p_sum(master ? n_coins : 0, 0);
    double alpha_sum = 0, beta_sum = 0;

    for (size_t it = 0; it < nb_it; ++it) {
        // Gather current p_ values from slaves
        slave_to_master(gather_p);

        // Master moves alpha, beta
        if (master) {
            // Save p
            for (size_t i = 0; i < n_coins; ++i) { p_sum[i] += get<p, value>(m)[i]; }
            // Save alpha & beta hyperparams
            alpha_sum += get<beta_weight_a, value>(m);
            beta_sum += get<beta_weight_b, value>(m);
            // Propose moves
            scaling_move(beta_weight_a_(m), logprob_of_blanket(v_weight_a), gen);
            scaling_move(beta_weight_b_(m), logprob_of_blanket(v_weight_b), gen);
        }
        master_to_slave(broadcast_a_b);


        if (!master) {
            for (size_t i = 0; i < partition.my_partition_size(); ++i) {
                // Move k_sum
                int k_sum = 0;
                auto k_vec = subsets::row(bern_(m), i);
                k_vec.across_values([&k_sum](int val) { k_sum += val; });
                beta_ss::draw(get<p, value>(m)[i], get<beta_weight_a, value>(m) + k_sum,
                    get<beta_weight_b, value>(m) + n_obs - k_sum, gen);
                // if (rank == 1 && it < 100) {
                //     MPI::p->message("K sum : {}", k_sum);
                //     MPI::p->message("P sample : {}", get<p, value>(m)[i]);
                //     // MPI::p->message("K sum : {}", k_sum);
                // }
            }
        }
        // if (it >= burn_in) { p_sum += raw_value(p_(*)); }
    }
    // float p_mean = p_sum / float(nb_it - burn_in);
    if (master) {
        vector<double> p_means(n_coins, 0);
        std::transform(
            p_sum.begin(), p_sum.end(), p_means.begin(), [](double v) { return v / nb_it; });
        for (size_t i = 0; i < n_coins; ++i) {
            MPI::p->message("P mean coin {} : {}", i + 1, p_means[i]);
        }

        MPI::p->message("Alpha = {}", alpha_sum / nb_it);
        MPI::p->message("Beta = {}", beta_sum / nb_it);
    }
    return 0;
}


int main(int argc, char** argv) { mpi_run(argc, argv, compute); }