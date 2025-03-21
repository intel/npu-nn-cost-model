// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_DATATYPE_COLLECTION_SIZE_H
#define VPUNN_DATATYPE_COLLECTION_SIZE_H

#include <cassert>
#include <type_traits>

#include "dpu_types.h"
#include "dpu_types_info.h"

namespace VPUNN {

/**
 * @brief compute size in bytes for a number of elements
 * @todo: explain how packing is performed, explain the limitation (eg a 12 bit type, a 6 bit type)
 *
 * @param elements_count the number of elements
 * @param datatype elements type
 * @returns number of bytes(memory) necessary to hold the elements
 */
template <class T>
inline T compute_size_in_bytes(const T elements_count, const DataType& datatype) noexcept {
    static_assert(std::is_integral_v<T>);
    T size{0};
    const unsigned int type_dimension{dtype_to_bytes(datatype)};
    if (elements_count > 0) {
        // if type is a submultiple of 8 bits multiple elements of this type can fit into one byte
        if (type_dimension == 1) {
            const int elements_per_byte{types_per_byte(datatype)};
            /* coverity[divide_by_zero] */
            const int reminder = elements_count % elements_per_byte;
            /* coverity[divide_by_zero] */
            const T fullBytes{(elements_count / elements_per_byte)};
            size = ((reminder != 0) ? (fullBytes + 1) : fullBytes);
        } else {
            size = elements_count * type_dimension;
        }
    }
    return size;
}

/**
 * @brief compute the number of elements when we have their size in bytes and their type
 *
 * @param size_in_bytes size in bytes of elements number we want to compute
 * @param datatype elements DataType
 * @return a long that represent number of elements
 */
inline long compute_elements_count_from_bytes(const long size_in_bytes, const DataType& datatype) noexcept {
    const unsigned int bytes_per_type{dtype_to_bytes(datatype)};

    if (size_in_bytes > 0) {
        if (bytes_per_type == 1) {
            const int elements_per_byte{types_per_byte(datatype)};
            return elements_per_byte * size_in_bytes;
        }
        /* coverity[divide_by_zero] */
        return size_in_bytes / bytes_per_type;
    } else {
        return 0;
    }
}

}  // namespace VPUNN

#endif  // VPUNN_TYPES_H
