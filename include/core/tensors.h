// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef CORE_TENSORS_H
#define CORE_TENSORS_H

#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>  // for error formating
#include <stdexcept>
#include <vector>

#include "core/vpunn_api.h"

namespace VPUNN {

/**
 * @brief Base VPUNN Tensor class
 *
 * @tparam T tensor datatype
 */
template <typename T>
class VPUNN_API(Tensor) {
private:
    std::vector<unsigned int> _dimensions;  ///< describes the data  represented as a multidimensional tensor
    int _size;                              ///< number of elements in the _data array
    T* _data;                               ///< the data array. The instance is the owner of this heap allocated memory

public:
    /**
     * @brief Construct a new Tensor object
     * Dimensions required must be well formed, if a dimension is zero the preconditions are not met
     *
     * @param dimensions a vector of unsigned integers representing the Tensor's dimensions
     */
    explicit Tensor(const std::vector<unsigned int>& dimensions): _dimensions(dimensions) {
        _size = std::accumulate(begin(dimensions), end(dimensions), 1, std::multiplies<unsigned int>());
        _data = new T[_size];
    }

    /**
     * @brief Construct a new Tensor object from existing preallocated memory
     *
     * @param data a pointer to the tensor initialization data, the ownership is transferred to this instance
     * @param dimensions a vector of unsigned integers representing the Tensor's dimensions. Must be consistent with
     * what data holds
     */
    Tensor(T* data, const std::vector<unsigned int>& dimensions): _dimensions(dimensions), _data(data) {
        _size = std::accumulate(begin(dimensions), end(dimensions), 1, std::multiplies<unsigned int>());
    }

    /**
     * @brief Construct a new Tensor object and fills it with a value
     *
     * @param dimensions a vector of unsigned integers representing the Tensor's dimensions. Must be well formed
     * @param value a unique value to fill the entire tensor
     */
    Tensor(const std::vector<unsigned int>& dimensions, T value): Tensor(dimensions) {
        this->fill(value);
    }

    /**
     * @brief Construct a new Tensor object (copy constructor)
     *
     * @param tensor
     */
    Tensor(const Tensor& tensor) {
        _dimensions = tensor._dimensions;
        _size = tensor._size;

        // allocate new memory and then copy
        _data = new T[_size];
        assign(tensor._data, sizeof(T) * tensor._size);
    }

    /**
     * @brief move constructor
     * No check is performed on the source tensor. Precondition is to be well/consistent formed
     *
     * @param tensor to move the contents from
     */
    Tensor(Tensor&& tensor) noexcept: _dimensions{std::move(tensor._dimensions)}, _size{tensor._size} {
        _data = tensor._data;  // move the data, now we own it

        // leave the source tensor in a consistent state
        tensor._data = nullptr;  // remove the data from previous owner
        tensor._size = 0;
    }

    /**
     * @brief Assignment operator overload
     *
     * The source object must be well formed (no null data), otherwise the copy will be bad
     * @param tensor a valid tensor
     * @return Tensor& self reference
     */
    Tensor& operator=(const Tensor& tensor) {
        if (this == &tensor) {
            return *this;  // self assignment
        }

        if (_data != nullptr) {
            delete[] _data;  // clean up already allocated memory
        }

        _dimensions = tensor._dimensions;
        _size = tensor._size;

        // allocate new memory and then copy
        _data = new T[_size];
        assign(tensor._data, sizeof(T) * tensor._size);
        return *this;
    }

    /**
     * @brief Move Assignment operator overload
     *
     * The source object must be well formed (no null data), otherwise the copy will be bad
     * @param tensor a valid tensor.
     * @return Tensor& self reference
     */
    Tensor& operator=(Tensor&& tensor) {
        if (this == &tensor) {  // not really required by standard to preserve value
            return *this;       // self move,   object remains the same
        }

        std::swap(this->_data, tensor._data);  // our data will be deallocated by the tensor's destruction
        std::swap(this->_dimensions, tensor._dimensions);
        std::swap(this->_size, tensor._size);

        return *this;
    }

    /**
     * @brief Destroy the Tensor object
     *
     */
    ~Tensor() {
        delete[] _data;
    }

    /**
     * @brief Return a pointer to the internal data buffer
     * Ownership is not transferred, pointer is not guaranteed to be preserved by other Tensor operations, like
     * assignment
     *
     * @return T*
     */
    T* data() {
        return _data;
    }

    /**
     * @brief Return a std::vector<T> copy of the internal data buffer as a std::vector
     *
     * @return std::vector<T> the output buffer
     */
    std::vector<T> data_vector() {
        return std::vector<T>{_data, _data + _size};
    }

    /**
     * @brief Return a pointer to the internal data buffer
     * Ownership is not transferred, pointer is not guaranteed to be preserved by other Tensor operations, like
     * assignment
     *
     * @return T*
     */
    const T* c_ptr() const {
        return _data;
    }

    /**
     * @brief Copy data inside a tensor
     * Does not change the shape, so the input size has to match the available internal memory.
     * DOes not change the memory placement of internal data
     *
     * @param data a pointer to a source data buffer
     * @param size_in_bytes the size of the data buffer in bytes
     * @throws runtime_error in case the input size is not aligned (not a int number of Ts)
     * @throws runtime_error in case the input size is NOT equal to available data
     * @return Tensor<T>& self reference
     */
    Tensor<T>& assign(const T* data, const unsigned int size_in_bytes) {
        // Check that size is a multiple of the datatype, integer number of Ts
        if (size_in_bytes % sizeof(T)) {
            throw std::runtime_error("Trying to assign a non-aligned buffer to a tensor");
        }
        // when size changes the shape gets inconsistent,do not allow
        const unsigned int available_size_in_bytes = ((unsigned int)_size * sizeof(T));
        if (available_size_in_bytes != size_in_bytes) {
            std::stringstream buffer;
            buffer << "[ERROR]Tensor::assign(), Passed memory has a different size: " << size_in_bytes
                   << ", this is inconsistent with the current shape. Current size is: " << available_size_in_bytes
                   << " No assign performed! "
                   << " File: " << __FILE__ << " Line: " << __LINE__;
            std::string details = buffer.str();
            throw std::runtime_error(details);
        } else {
            std::memcpy(_data, (const void*)data, size_in_bytes);
        }
        return *this;
    }

    /**
     * @brief Return a reference to an internal buffer element. No boundary check is performed
     *
     * @param idx the index
     * @return T& a reference to the internal buffer element
     */
    T& operator[](const int idx) {
        return _data[idx];
    }

    /**
     * @brief Return a const reference to an internal buffer element. No boundary check is performed
     *
     * @param idx the index
     * @return T& a reference to the internal buffer element
     */
    const T& operator[](const int idx) const {
        return _data[idx];
    }

    /**
     * @brief Return the size of the tensor. How many T-elements it contains
     *
     * @return int
     */
    int size() const {
        return _size;
    }

    /**
     * @brief Return the shape of the tensor
     *
     * @return const std::vector<unsigned int>&
     */
    const std::vector<unsigned int>& shape() const {
        return _dimensions;
    }

    /**
     * @brief Fill a tensor with a value
     *
     * @param value
     * @return Tensor&
     */
    Tensor& fill(T value) {
        for (int idx = 0; idx < _size; idx++) {
            _data[idx] = value;
        }
        return *this;
    }
};

/**
 * @brief Generate a random uniform Tensor
 *
 * @tparam T tensor datatype
 * @param dimensions tensor dimensions as a vector
 * @param min minimum value
 * @param max maximum value
 * @return Tensor<T>
 */
template <typename T>
Tensor<T> random_uniform(const std::vector<unsigned int>& dimensions, float min, float max) {
    // Random numbers
    std::mt19937_64 rnd;
    std::uniform_real_distribution<T> randm_dist(min, max);

    Tensor<T> t(dimensions);

    for (int idx = 0; idx < t.size(); idx++) {
        t[idx] = randm_dist(rnd);
    }

    return t;
}

/**
 * @brief Return a Tensor of all ones
 *
 * @tparam T Tensor datatype
 * @param dimensions tensor dimensions as a vector
 * @return Tensor<T>
 */
template <typename T>
Tensor<T> ones(const std::vector<unsigned int>& dimensions) {
    Tensor<T> t(dimensions);
    t.fill(1);
    return t;
}

/**
 * @brief Return a Tensor of all zeros
 *
 * @tparam T Tensor datatype
 * @param dimensions tensor dimensions as a vector
 * @return Tensor<T>
 */
template <typename T>
Tensor<T> zeros(const std::vector<unsigned int>& dimensions) {
    Tensor<T> t(dimensions);
    t.fill(0);
    return t;
}

/**
 * @brief Return a Tensor of values from 1 to size
 *
 * @tparam T Tensor datatype
 * @param dimensions tensor dimensions as a vector
 * @return Tensor<T>
 */
template <typename T>
Tensor<T> stair(const std::vector<unsigned int>& dimensions) {
    Tensor<T> t(dimensions);
    for (T idx = 0; idx < t.size(); idx++) {
        t[idx] = idx + 1;
    }
    return t;
}

}  // namespace VPUNN

/**
 * @brief Overload of operator << (to use Tensor with cout)
 *
 * @tparam T tensor datatype
 * @param os std::ostread (example std::cout)
 * @param t reference to a Tensor object
 * @return std::ostream&
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const VPUNN::Tensor<T>& t) {
    os << "[";
    for (int idx = 0; idx < t.size(); idx++) {
        os << t[idx] << " ";
    }
    os << "]";
    return os;
}

#endif  // CORE_TENSORS_H
