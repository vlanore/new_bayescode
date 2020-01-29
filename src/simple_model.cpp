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
        auto v = std::vector<std::unique_ptr<C>>();
        // VL : no need to put the vector in a unique_ptr, as vectors already
        // allocate their contents on the heap

        // fill vector with complicated objects
        v.emplace_back(std::make_unique<C>(3));

        // move vector into tagged tuple
        return make_model(array_ = move(v));
    }
};

int main()  {
    auto model = simple_model::make();
    auto& v = get<array>(model);     // error: use of deleted function
    // VL: "auto v = get<array>(model)" was trying to make a copy of the whole vector,
    // which is impossible because it's a vector of unique pointers which are not copyable
    // taking a reference instead ("auto& v = ...") solves the problem
    std::cerr << "size : " << v.size() << '\n';
}
