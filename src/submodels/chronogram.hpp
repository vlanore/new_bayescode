#pragma once

#include "bayes_toolbox.hpp"
#include "tree/implem.hpp"
#include "lib/Chronogram.hpp"

struct chronogram    {

    using T = Chronogram;
    using param_decl = param_decl_t<>;

    template <typename Gen>
    static void draw(T& x, Gen& gen)    {
        x.Sample();
    }

    static real logprob(const T& x) {
        return 0;
    }
};


