#include <cmath>
#include <iostream>
#include "components/ChainCheckpoint.hpp"
#include "bayes_toolbox.hpp"

TOKEN(array)

struct simple_model {

    // some complicated object
    struct C    {
        int _i;
        C(int i) : _i(i) {}
    };

    static auto make()    {
        // array of (dynamically allocated) objects of type C
        // dynamic allocation of the vector itself (but also tried without)
        auto v = std::make_unique<std::vector<std::unique_ptr<C>>>();

        // fill vector with complicated objects
        v->emplace_back(std::make_unique<C>(3));

        // move vector into tagged tuple
        return make_model(array_ = move(v));
    }
};

int main()  {
    auto model = simple_model::make();
    auto v = get<array>(model);     // error: use of deleted function
    std::cerr << "size : " << v.size() << '\n';
}
