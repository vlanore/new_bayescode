#pragma once

#include "reduce.hpp"
#include "Proxy.hpp"
#include "structure/suffstat.hpp"

template <class SuffStat>
std::unique_ptr<ProxyMPI> reduce_suffstats(Proxy<SuffStat>& suffstats)   {
    if (!MPI::p->rank) {  // master
        auto reducer = new ReducerMaster();
        reducer->add(suffstats.get());
        return std::unique_ptr<ProxyMPI>(dynamic_cast<ProxyMPI*>(reducer));
    } else {  // slave
        auto reducer = new ReducerSlave();
        reducer->add(suffstats.get());
        return std::unique_ptr<ProxyMPI>(dynamic_cast<ProxyMPI*>(reducer));
    }
}

template <class SuffStat>
std::unique_ptr<ProxyMPI> reduce_suffstats(Proxy<SuffStat, size_t>& suffstats)   {
    if (!MPI::p->rank) {  // master
        auto reducer = new ReducerMaster();
        for (size_t i=0; i<suffstats.size(); i++)  {
            reducer->add(suffstats.get(i));
        }
        return std::unique_ptr<ProxyMPI>(dynamic_cast<ProxyMPI*>(reducer));
    } else {  // slave
        auto reducer = new ReducerSlave();
        for (size_t i=0; i<suffstats.size(); i++)  {
            reducer->add(suffstats.get(i));
        }
        return std::unique_ptr<ProxyMPI>(dynamic_cast<ProxyMPI*>(reducer));
    }
}

template <class SuffStat>
std::unique_ptr<ProxyMPI> reduce_suffstats(Proxy<SuffStat, size_t, size_t>& suffstats)   {
    if (!MPI::p->rank) {  // master
        auto reducer = new ReducerMaster();
        for (size_t i=0; i<suffstats.size1(); i++)  {
            for (size_t j=0; j<suffstats.size2(); j++)  {
                reducer->add(suffstats.get(i,j));
            }
        }
        return std::unique_ptr<ProxyMPI>(dynamic_cast<ProxyMPI*>(reducer));
    } else {  // slave
        auto reducer = new ReducerSlave();
        for (size_t i=0; i<suffstats.size1(); i++)  {
            for (size_t j=0; j<suffstats.size2(); j++)  {
                reducer->add(suffstats.get(i,j));
            }
        }
        return std::unique_ptr<ProxyMPI>(dynamic_cast<ProxyMPI*>(reducer));
    }
}

