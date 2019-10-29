#pragma once

#include "Process.hpp"

template <class Factory>
auto slave_only_ptr(Factory fac) {
    if (!MPI::p->rank) {
        return std::unique_ptr<decltype(fac())>(nullptr);
    } else {
        return std::make_unique<decltype(fac())>(fac());
    }
}

template <class Factory>
auto master_only_ptr(Factory fac) {
    if (MPI::p->rank) {
        return std::unique_ptr<decltype(fac())>(nullptr);
    } else {
        return std::make_unique<decltype(fac())>(fac());
    }
}