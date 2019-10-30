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

template <class Tag>
class TagTracer : public ChainComponent {
    Tracer model_tracer;
    std::string file_name;

  public:
    template <class M>
    TagTracer(M& m, std::string file_name)
        : model_tracer(m, processing::HasTag<Tag>()), file_name(file_name) {}

    void start() final {
        std::ofstream model_os{chain_file(file_name), std::ios_base::trunc};
        model_tracer.write_header(model_os);
    }

    void savepoint(int) final {
        std::ofstream model_os{chain_file(file_name), std::ios_base::app};
        model_tracer.write_line(model_os);
    }

    static std::string chain_file(std::string file_name) { return file_name; }
};

using ModelTracer = TagTracer<ModelNode>;
using StatTracer = TagTracer<Stat>;