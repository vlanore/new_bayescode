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

TOKEN(shape_a)
TOKEN(shape_b)
TOKEN(p)
TOKEN(bern)

// Model definition
auto coin_toss_model(size_t nb_coins, size_t nb_draws) {
    auto shape_a = make_node<exponential>(1.0);
    auto shape_b = make_node<exponential>(1.0);
    auto p = make_node_array<beta_ss>(nb_coins, n_to_one(shape_a), n_to_one(shape_b));
    auto bern = make_node_matrix<bernoulli>(
        nb_coins, nb_draws, [& v = get<value>(p)](int i, int) { return v[i]; });

    // clang-format off
    return make_model(
            shape_a_ = move(shape_a),
            shape_b_ = move(shape_b),
            p_ = move(p),
            bern_ = move(bern)
    );
    // clang-format on
}

template <typename Model, typename Broadcast, typename Gen>
void simulate(Model& m, Broadcast& broadcast_ab, Partition& partition, Gen& gen) {
    const auto rank = MPI::p->rank;
    const bool is_master = !rank;
    const size_t partition_size = partition.my_partition_size();

    // Master sets alpha and beta parameters and broadcasts it to slaves
    auto view_ab = make_collection(get<shape_a>(m), get<shape_b>(m));
    if (is_master) { draw(view_ab, gen); }
    master_to_slave(broadcast_ab);

    if (!is_master) {
        auto view_coin = make_collection(p_(m), bern_(m));
        draw(view_coin, gen);
        for (size_t i = 0; i < partition_size; ++i) {
            int coin_id = partition_size * (rank - 1) + (i + 1);
            MPI::p->message("Coin {} \t: P={}", coin_id, get<p, value>(m)[i]);
            // MPI::p->message("K [{}] \t: {}", coin_id, vector_to_string(get<bern, value>(m)[i]));
        }
    }
}

template <typename Model, typename Broadcast, typename Gen>
void initialize(Model& m, Broadcast& broadcast_ab, Gen& gen) {
    // Initialize node states to be different from simulation
    const bool is_master = !(MPI::p->rank);
    if (is_master) {
        draw(shape_a_(m), gen);
        draw(shape_b_(m), gen);
    }

    master_to_slave(broadcast_ab);

    if (!is_master) { draw(p_(m), gen); }
}

int compute(char, char**) {
    auto rank = MPI::p->rank;  // getting MPI rank
    auto size = MPI::p->size;  // ... and size (number of processes)
    const auto is_master = !rank;

    constexpr int n_obs{50};
    constexpr int n_coins{80};
    constexpr size_t nb_it{1'000'000};
    constexpr size_t burn_in{nb_it / 10};
    constexpr size_t sample_size{nb_it - burn_in};
    auto gen = make_generator();

    // List of indices for data partition
    // indices are partitioned into size-1 bins, starting at process 1
    IndexSet indices(n_coins);
    std::generate(indices.begin(), indices.end(), [n = 0]() mutable { return n++; });

    Partition partition(indices, size - 1, 1);
    MPI::p->message("Partition size : {}", partition.my_partition_size());

    // Common part of the model, shared between is_master and slaves
    auto m = coin_toss_model(is_master ? n_coins : partition.my_partition_size(), n_obs);
    // MPI data sharing
    auto broadcast_ab = broadcast(get<shape_a, value>(m), get<shape_b, value>(m));
    auto gather_p = gather(partition, get<p, value>(m));


    // Simulate model parameters and initialize them
    simulate(m, broadcast_ab, partition, gen);
    slave_to_master(gather_p);
    auto simulated_p = backup(get<p>(m));
    auto simulated_a = backup(get<shape_a>(m));
    auto simulated_b = backup(get<shape_b>(m));
    // Initialize model state before inference
    initialize(m, broadcast_ab, gen);


    auto view_ap = make_collection(shape_a_(m), p_(m));
    auto view_bp = make_collection(shape_b_(m), p_(m));

    // Parameters sum tracers used by master
    vector<double> p_sum(is_master ? n_coins : 0, 0);
    double alpha_sum = 0, beta_sum = 0;

    // K sum suff stat
    vector<int> k_sums(partition.my_partition_size(), 0);
    if (!is_master) {
        for (size_t i = 0; i < partition.my_partition_size(); ++i) {
            auto& k_sum = k_sums[i];
            subsets::row(bern_(m), i).across_values([&k_sum](int val) { k_sum += val; });
        }
    }

    for (size_t it = 0; it < nb_it; ++it) {
        // Gather current p_ values from slaves
        slave_to_master(gather_p);

        // Master moves alpha, beta
        if (is_master) {
            if (it >= burn_in) {
                if (it == burn_in) { MPI::p->message("Burning done !"); }
                // Save p
                for (size_t i = 0; i < n_coins; ++i) { p_sum[i] += get<p, value>(m)[i]; }
                // Save alpha & beta hyperparams
                alpha_sum += get<shape_a, value>(m);
                beta_sum += get<shape_b, value>(m);
            }
            // Propose moves
            scaling_move(shape_a_(m), logprob_of_blanket(view_ap), gen);
            scaling_move(shape_b_(m), logprob_of_blanket(view_bp), gen);
        }
        master_to_slave(broadcast_ab);

        if (!is_master) {
            for (size_t i = 0; i < partition.my_partition_size(); ++i) {
                beta_ss::draw(get<p, value>(m)[i], get<shape_a, value>(m) + k_sums[i],
                    get<shape_b, value>(m) + n_obs - k_sums[i], gen);
            }
        }
    }


    // Sampling results
    if (is_master) {
        // Compute p means
        vector<double> p_means(n_coins, 0);
        std::transform(
            p_sum.begin(), p_sum.end(), p_means.begin(), [](double v) { return v / sample_size; });
        for (size_t i = 0; i < n_coins; ++i) {
            MPI::p->message(
                "Coin {}\t: P mean = {} \t| Truth = {}", i + 1, p_means[i], simulated_p[i]);
        }
        // Compute alpha and beta means
        MPI::p->message("Alpha = {} \t| Truth = {}", alpha_sum / sample_size, simulated_a);
        MPI::p->message("Beta = {} \t| Truth = {}", beta_sum / sample_size, simulated_b);
    }
    return 0;
}


int main(int argc, char** argv) { mpi_run(argc, argv, compute); }