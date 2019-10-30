#pragma once

#include "bayes_toolbox.hpp"

template <class T>
using TraceEntry = std::pair<std::string, T>;


namespace overloads {
    template <class Node>
    auto trace_entry(node_tag, const std::string& name, Node& x) {
        auto& raw = get<value>(x);
        return TraceEntry<decltype(raw)>(name, raw);
    }

    template <class F>
    auto trace_entry(unknown_tag, const std::string& name, F&& x) {
        return TraceEntry<std::decay_t<F>>(name, x);
    }
}  // namespace overloads

template <class T>
auto trace_entry(const std::string& name, T&& x) {
    return overloads::trace_entry(type_tag(x), name, x);
}

template <class... Entries>
class TraceableCollection {
    std::tuple<Entries...> data;

    template <class Info, size_t... Is>
    void declare_interface_helper(Info info, std::index_sequence<Is...>) {
        std::vector<int> ignore = {
            (model_stat(info, get<Is>(data).first, get<Is>(data).second), 0)...};
    }

  public:
    TraceableCollection(Entries... entries) : data(entries...) {}

    template <class Info>
    void declare_interface(Info info) {
        declare_interface_helper(info, std::index_sequence_for<Entries...>{});
    }
};

template <class... Entries>
auto make_trace(Entries... entries) {
    return TraceableCollection<Entries...>(entries...);
}