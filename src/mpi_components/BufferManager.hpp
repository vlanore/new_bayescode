#pragma once

#include <numeric>
#include "ReceiveBuffer.hpp"
#include "SendBuffer.hpp"
#include "traits.hpp"

/*
====================================================================================================
  BufferManager
  A class to which ints and doubles (and vectors of ints and doubles) can be registered.
  Registered objects can then be packed in bulk into a buffer or read in bulk from a buffer.
==================================================================================================*/
class BufferManager {
    /*----------------------------------------------------------------------------------------------
      Manager state */

    // clang-format off
    struct double_array_t { double* data; size_t size; };
    struct int_array_t    { int*    data; size_t size; };
    // clang-format on

    std::vector<int_array_t> int_arrays;
    std::vector<double_array_t> double_arrays;

    std::unique_ptr<SendBuffer> _send_buffer;
    std::unique_ptr<ReceiveBuffer> _receive_buffer;

    /*----------------------------------------------------------------------------------------------
      Dispatchers */

    template <class T>
    void array_add_dispatch(T& x, std::true_type /* is partitionable */) {
        contig_add_subset_dispatch(x, 0, x.size(), is_contiguously_serializable<T>());
    }

    template <class T>
    void array_add_dispatch(T& x, std::false_type /* is not partitionable */) {
        static_assert(
            has_custom_serialization<T>::value, "BufferManager: type T lacks custom serialization");
        x.template serialization_interface<BufferManager>(*this);
    }

    template <class T>
    void contig_add_subset_dispatch(
        T& x, size_t start, size_t size, std::true_type /* is contiguously serializable */) {
        /* -- */
        add(x[start], size);
    }

    template <class T>
    void contig_add_subset_dispatch(
        T& x, size_t start, size_t size, std::false_type /* is not contiguously serializable */) {
        /* -- */
        for (size_t i = start; i < start + size; i++) { add(x[i]); }
    }

  public:
    BufferManager() = default;

    /*----------------------------------------------------------------------------------------------
      Declaration functions */

    void add(int& data, size_t size = 1) { int_arrays.push_back({&data, size}); }
    void add(double& data, size_t size = 1) { double_arrays.push_back({&data, size}); }

    template <class T>
    void add(T& x) {
        array_add_dispatch(x, is_partitionable<T>());
    }

    template <class Arg, class... Args>
    void add(Arg& arg, Args&&... args) {
        add(arg);
        add(std::forward<Args>(args)...);
    }

    template <class T>
    void add_subset(T& x, size_t start, size_t size) {
        static_assert(is_partitionable<T>::value,
            "BufferManager: cannot add subset of non-partitionable object");
        contig_add_subset_dispatch(x, start, size, is_contiguously_serializable<T>());
    }

    /*----------------------------------------------------------------------------------------------
      Size-related functions */

    size_t nb_ints() const {
        auto sum_size = [](int acc, int_array_t v) { return acc + v.size; };
        return std::accumulate(int_arrays.begin(), int_arrays.end(), 0, sum_size);
    }

    size_t nb_doubles() const {
        auto sum_size = [](int acc, double_array_t v) { return acc + v.size; };
        return std::accumulate(double_arrays.begin(), double_arrays.end(), 0, sum_size);
    }

    size_t buffer_size() const { return buffer_int_size() + buffer_double_size(); }
    size_t buffer_int_size() const { return nb_ints() * MPI::int_size(); }
    size_t buffer_double_size() const { return nb_doubles() * MPI::double_size(); }

    /*----------------------------------------------------------------------------------------------
      Send-receive interfaces */

    void* receive_buffer() {
        _receive_buffer.reset(new ReceiveBuffer(buffer_size()));
        return _receive_buffer->data();
    }
    void* receive_int_buffer() { return _receive_buffer->data(); }

    void* receive_double_buffer() {
        return static_cast<char*>(_receive_buffer->data()) + buffer_int_size();
    }

    void* send_buffer() {
        _send_buffer.reset(new SendBuffer());
        for (auto x : int_arrays) { _send_buffer->pack(x.data, x.size); }
        for (auto x : double_arrays) { _send_buffer->pack(x.data, x.size); }
        assert(buffer_size() == _send_buffer->size());
        return _send_buffer->data();
    }

    void* send_int_buffer() { return _send_buffer->data(); }

    void* send_double_buffer() {
        return static_cast<char*>(_send_buffer->data()) + buffer_int_size();
    }

    void receive() {
        assert(_receive_buffer.get() != nullptr);
        assert(_receive_buffer->size() == buffer_size());
        for (auto x : int_arrays) { _receive_buffer->unpack_array<int>(x.data, x.size); }
        for (auto x : double_arrays) { _receive_buffer->unpack_array<double>(x.data, x.size); }
    }

    /*----------------------------------------------------------------------------------------------
      Merging */
    void merge(const BufferManager& other) {
        _send_buffer.reset(nullptr);
        _receive_buffer.reset(nullptr);
        int_arrays.insert(int_arrays.end(), other.int_arrays.begin(), other.int_arrays.end());
        double_arrays.insert(
            double_arrays.end(), other.double_arrays.begin(), other.double_arrays.end());
    }
};