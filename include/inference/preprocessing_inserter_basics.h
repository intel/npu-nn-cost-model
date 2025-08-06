// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef PREPROCESSING_INSERTER_BASICS_H
#define PREPROCESSING_INSERTER_BASICS_H

#include <vpu/types.h>
#include <sstream>  // for error formating
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

// #include <unordered_set>
// #include "inference/preprocessing.h"

namespace VPUNN {

/// Inserts different datatypes into a descriptor buffer
template <class T>
class Inserter {
protected:  // only derived instances allowed
    Inserter(std::vector<T>& output): external_output(output) {
    }

private:
    std::vector<T>&
            external_output;  ///< external descriptor container, reference should be valid during this lifetime.

    unsigned int output_size() const {
        return (unsigned int)external_output.size();
    }
    void setAValue(size_t idx, const T& value) {
        external_output[idx] = value;  // no bounds checking
    }

public:
    /** @brief fills in the positions corresponding to an enum.
     * All on zero except the position corresponding to the active enum value is put on 1
     * Enum values must start at zero and be contiguous
     *
     * @param data the enum value
     * @param offset m the index where writing starts
     * @param category_size how many values this enum has
     *
     * @tparam only_simulate, a boolean, if true, the method will do no writing, just computing the size it needs to
     * do the writing
     * @tparam E,  enum type
     *
     * @return the offset of next available position. Equivalent to how many positions are filled in the output
     * vector.
     */
    template <bool only_simulate, class E>
    size_t one_hot(E data, size_t offset, size_t category_size) {
        size_t idx = static_cast<size_t>(data);  // assuming the enums are contiguous and from zero
        if (idx < category_size || category_size == 0) {
            insert<only_simulate>(T(1.0), idx + offset);
        }
        return offset + category_size;  // this enum occupies category_size positions
    }

    template <bool only_simulate = false, class E, size_t SIZE>
    size_t insert(const std::array<E, SIZE>& vec, size_t offset) {
        for (long unsigned int idx = 0; idx < vec.size(); idx++) {
            offset = insert<only_simulate>(vec[idx], offset);
        }
        return offset;
    }

    template <bool only_simulate = false, class E>
    size_t insert(const std::vector<E>& vec, size_t offset) {
        for (auto idx = 0; idx < vec.size(); idx++) {
            offset = insert<only_simulate>(vec[idx], offset);
        }
        return offset;
    }

    ///// @brief inserter for VPUTensor
    ///// required because of array and vector inserter
    // template <bool only_simulate>
    // size_t insert(const VPUTensor& data, size_t offset) {
    //     // calls the specialization from the derived class, known at compile time
    //     return static_cast<D*>(this)->template insert<only_simulate>(data, offset);
    // }

    /// @brief insert specialization for HaloInfoHW
    template <bool only_simulate>
    size_t insert(const HaloWorkload::HaloInfoHW& data, size_t offset) {
        offset = this->insert<only_simulate>(data.top, offset);
        offset = this->insert<only_simulate>(data.bottom, offset);
        offset = this->insert<only_simulate>(data.left, offset);
        offset = this->insert<only_simulate>(data.right, offset);
        return offset;
    }

    /// @brief insert specialization for HaloInfoHWC
    template <bool only_simulate>
    size_t insert(const HaloWorkload::HaloInfoHWC& data, size_t offset) {
        offset = this->insert<only_simulate>(data.top, offset);
        offset = this->insert<only_simulate>(data.bottom, offset);
        offset = this->insert<only_simulate>(data.left, offset);
        offset = this->insert<only_simulate>(data.right, offset);
        offset = this->insert<only_simulate>(data.front, offset);
        offset = this->insert<only_simulate>(data.back, offset);
        return offset;
    }

    /// @brief insert specialization for HaloWorkload
    template <bool only_simulate>
    size_t insert(const HaloWorkload& data, size_t offset) {
        offset = this->insert<only_simulate>(data.input_0_halo, offset);
        offset = this->insert<only_simulate>(data.output_0_halo, offset);
        offset = this->insert<only_simulate>(data.output_0_halo_broadcast_cnt, offset);
        return offset;
    }

    // specialization for latest swizzling. Each version type should have its own if wants special treatment
    // instead of enum one hot style, use a boolean enabled/disabled
    template <bool only_simulate>
    size_t insert(Swizzling data, size_t offset) {
        // 1 for enabled :Key_n, zero for disabled: KEY_0
        const bool is_swizzling_enabled{(data != Swizzling::KEY_0)};
        return this->insert<only_simulate>(is_swizzling_enabled, offset);
    }

    template <bool only_simulate = false, typename E>
    size_t insert(E data, size_t offset) {
        return one_hot<only_simulate>(data, offset, static_cast<int>(E::__size));
    }

    template <bool only_simulate = false>
    size_t insert(unsigned int data, size_t offset) {
        return insert<only_simulate>(T(data), offset);
    }

    template <bool only_simulate = false>
    size_t insert(int data, size_t offset) {
        return insert<only_simulate>(T(data), offset);
    }

    template <bool only_simulate = false>
    size_t insert(bool data, size_t offset) {
        return insert<only_simulate>(T(data), offset);
    }

    template <bool only_simulate = false>
    size_t insert(const T& data, size_t offset) {
        if (!only_simulate) {
            if (offset >= this->output_size()) {
                std::stringstream buffer;
                buffer << "[ERROR] PreprocessingInserter.insert(),"
                       << " out of range write (offset) when building VPUNN descriptor, offset:" << offset
                       << ", this is beyond the allocated descriptor size:  " << this->output_size()
                       << " Cannot continue! "
                       << " File: " << __FILE__ << " Line: " << __LINE__;
                const std::string details = buffer.str();
                throw std::out_of_range(details);
            }
            this->setAValue(offset, data);  // only place where data is actually put into buffer
        }
        return offset + 1;
    }
};

}  // namespace VPUNN
#endif
