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
/*
#include "basic_moves.hpp"
#include "distributions/gamma.hpp"
#include "detfunctions/product.hpp"
#include "distributions/poisson.hpp"
#include "mcmc_utils.hpp"
#include "operations/backup.hpp"
#include "operations/logprob.hpp"
#include "operations/raw_value.hpp"
#include "operations/set_value.hpp"
#include "structure/View.hpp"
#include "structure/array_utils.hpp"
#include "suffstat_utils.hpp"
#include "tagged_tuple/src/fancy_syntax.hpp"
*/
using namespace std;

TOKEN(lambda1)
TOKEN(lambda2)
TOKEN(mrate)
TOKEN(K)

auto poisson_product(size_t n1, size_t n2) {

    auto lambda1 = make_node_array<gamma_ss>(n1, n_to_const(1.0), n_to_const(1.0));
    auto lambda2 = make_node_array<gamma_ss>(n2, n_to_const(1.0), n_to_const(1.0));

    auto mrate = make_dnode_matrix<product>(
            n1, n2, 
            [&l1 = get<value>(lambda1)] (int i, int j) { return l1[i]; },
            [&l2 = get<value>(lambda2)] (int i, int j) { return l2[j]; });
            // [&l1 = get<value>(lambda1), &l2 = get<value>(lambda2)] (int i, int j) { return l1[i]*l2[j]; }  

    auto K = make_node_matrix<poisson>(
            n1, n2,
            [&r = get<value>(mrate)] (int i, int j) { return r[i][j]; }
        );

    // clang-format off
    return make_model(
        lambda1_ = move(lambda1),
        lambda2_ = move(lambda2),
          mrate_ = move(mrate),
              K_ = move(K)
    );  // clang-format on
}

template <class Node, class MB, class Gen, class... IndexArgs>
void scaling_move(Node& node, MB blanket, Gen& gen, IndexArgs... args) {
    auto index = make_index(args...);
    auto bkp = backup(node, index);
    double logprob_before = logprob(blanket);
    double log_hastings = scale(raw_value(node, index), gen);
    bool accept = decide(logprob(blanket) - logprob_before + log_hastings, gen);
    if (!accept) { restore(node, bkp, index); }
}

int main() {
    auto gen = make_generator();

    constexpr size_t nb_it{100'000}, n1{5}, n2{10};
    auto m = poisson_product(n1, n2);

    // auto v = make_collection(lambda1_(m), lambda2_(m));
    // vector<vector<int>> initK(n1, vector<int>(n2,1));
    // set_value(K_(m), initK);

    for (int rep=0; rep<2; rep++)   {
        cerr << "draw model\n";
        draw(lambda1_(m), gen);
        draw(lambda2_(m), gen);
        // gather(mrate_(m));
        draw(K_(m), gen);
        cerr << "draw model ok\n";

        for (size_t i=0; i<n1; i++) {
            for (size_t j=0; j<n2; j++) {
                cerr << get<lambda1,value>(m)[i];
                cerr << '\t' << get<lambda2,value>(m)[i];
                cerr << '\t' << get<mrate,value>(m)[i][j];
                cerr << '\t' << get<K,value>(m)[i][j];
                cerr << '\n';
            }
        }
    }
    cerr << '\n' << '\n';

    /*
    for (size_t it = 0; it < nb_it; it++) {

        for (size_t i=0; i<n1; i++) {
            auto lambda1_mb =
                make_collection(subsets::row(K_(m), i), subsets::element(lambda1_(m), i));
            mh_move(lambda1_(m), logprob_of_blanket(lambda1_mb),
                    [i](auto& value, auto& gen) { return scale(value[i], gen); }, gen);
        }

        for (size_t i=0; i<n2; i++) {
            auto lambda2_mb =
                make_collection(subsets::column(K_(m), i), subsets::element(lambda2_(m), i));
            mh_move(lambda2_(m), logprob_of_blanket(lambda2_mb),
                    [i](auto& value, auto& gen) { return scale(value[i], gen); }, gen);
        }
    }
    */
}
