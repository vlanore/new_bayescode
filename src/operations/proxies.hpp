#pragma once

#include <functional>
#include <memory>
#include <vector>
#include "interfaces.hpp"
#include "mpi_components/Process.hpp"

using ProxyMPIPtr = std::unique_ptr<ProxyMPI>;

/*
====================================================================================================
  ForAll class
  Propagates acquire and release to all proxies registered to it
==================================================================================================*/
class ForAll : public ProxyMPI {
    std::vector<ProxyMPI*> pointers;

  public:
    ForAll() = default;

    template <class... Pointers>
    ForAll(Pointers&&... pointers) : pointers({pointers...}) {}

    void add(ProxyMPI* pointer) { pointers.push_back(pointer); }

    void acquire() final {
        for (auto pointer : pointers) { pointer->acquire(); }
    }

    void release() final {
        for (auto pointer : pointers) { pointer->release(); }
    }
};

template <class... Args>
ProxyMPIPtr make_forall(Args&&... args) {
    return ProxyMPIPtr(dynamic_cast<ProxyMPI*>(new ForAll(std::forward<Args>(args)...)));
}

/*
====================================================================================================
  Group class
  Same as forall but owns its contents in the form of unique pointers
==================================================================================================*/
class Group : public ProxyMPI {
    std::vector<ProxyMPIPtr> operations;

  public:
    Group() = default;

    template <class... Operations>
    Group(ProxyMPIPtr&& operation, Operations&&... operations)
        : Group(std::forward<Operations>(operations)...) {
        /* -- */
        if (operation.get() != nullptr) {
            this->operations.insert(this->operations.begin(), std::move(operation));
        }
    }

    void add(ProxyMPIPtr ptr) {
        if (ptr.get() != nullptr) { operations.push_back(std::move(ptr)); }
    }

    void acquire() final {
        for (auto&& operation : operations) { operation->acquire(); }
    }

    void release() final {
        for (auto&& operation : operations) { operation->release(); }
    }
};

template <class... Args>
ProxyMPIPtr make_group(Args&&... args) {
    return ProxyMPIPtr(dynamic_cast<ProxyMPI*>(new Group(std::forward<Args>(args)...)));
}

/*
====================================================================================================
  Operation class
==================================================================================================*/
class Operation : public ProxyMPI {
    std::function<void()> f_acquire{[]() {}};
    std::function<void()> f_release{[]() {}};

  public:
    template <class Acquire, class Release>
    Operation(Acquire f_acquire, Release f_release) : f_acquire(f_acquire), f_release(f_release) {}

    void acquire() final { f_acquire(); }
    void release() final { f_release(); }
};

template <class Acquire, class Release>
ProxyMPIPtr make_operation(Acquire f_acquire, Release f_release) {
    return ProxyMPIPtr(dynamic_cast<ProxyMPI*>(new Operation(f_acquire, f_release)));
}

template <class Acquire>
ProxyMPIPtr make_acquire_operation(Acquire f_acquire) {
    return ProxyMPIPtr(dynamic_cast<ProxyMPI*>(new Operation(f_acquire, []() {})));
}

template <class Release>
ProxyMPIPtr make_release_operation(Release f_release) {
    return ProxyMPIPtr(dynamic_cast<ProxyMPI*>(new Operation([]() {}, f_release)));
}

/*
====================================================================================================
  ForInContainer class
==================================================================================================*/
template <class Container>
class ForInContainer : public ProxyMPI {
    using Element = typename Container::value_type;
    Container& container;
    std::function<ProxyMPI&(Element&)> get_proxy;

  public:
    template <class GetProxyMPI>
    ForInContainer(Container& container, GetProxyMPI get_proxy)
        : container(container), get_proxy(get_proxy) {}

    ForInContainer(Container& container)
        : container(container), get_proxy([](Element& e) -> ProxyMPI& { return e; }) {}

    void acquire() final {
        for (auto& element : container) { get_proxy(element).acquire(); }
    }

    void release() final {
        for (auto& element : container) { get_proxy(element).release(); }
    }
};

template <class Container, class GetProxyMPI>
ProxyMPIPtr make_for_in_container(Container& container, GetProxyMPI get_proxy) {
    return ProxyMPIPtr(dynamic_cast<ProxyMPI*>(new ForInContainer<Container>(container, get_proxy)));
}

template <class Container>
ProxyMPIPtr make_for_in_container(Container& container) {
    return ProxyMPIPtr(dynamic_cast<ProxyMPI*>(new ForInContainer<Container>(container)));
}

/*
====================================================================================================
  Conditional creation functions
==================================================================================================*/

ProxyMPIPtr slave_only(ProxyMPIPtr&& ptr) { return MPI::p->rank ? std::move(ptr) : nullptr; }

ProxyMPIPtr master_only(ProxyMPIPtr&& ptr) { return (!MPI::p->rank) ? std::move(ptr) : nullptr; }

template <class F>
ProxyMPIPtr slave_acquire(F f) {
    return slave_only(make_acquire_operation(f));
}

template <class F>
ProxyMPIPtr slave_release(F f) {
    return slave_only(make_release_operation(f));
}

template <class F>
ProxyMPIPtr master_acquire(F f) {
    return master_only(make_acquire_operation(f));
}

template <class F>
ProxyMPIPtr master_release(F f) {
    return master_only(make_release_operation(f));
}