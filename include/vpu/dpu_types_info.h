// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_DPU_TYPES_INFO_H
#define VPUNN_DPU_TYPES_INFO_H

#include <array>
#include <cassert>
#include <vector>

#include "dim_enum.h"
#include "dpu_types.h"

namespace VPUNN {

// datatypes operations
/** @brief Get the size of the dtype
 *
 * @param dtype a DataType object
 * @return size in bytes.
 */
inline constexpr unsigned int dtype_to_bytes(const DataType dtype) noexcept {
    // intermediate 1+ bytes are not handled, will be when present
    switch (dtype) {
    case DataType::INT32:
    case DataType::FLOAT32:
        return 4;

    case DataType::FLOAT16:
    case DataType::BFLOAT16:
        return 2;

    // 1 byte size
    case DataType::UINT8:
    case DataType::INT8:
    case DataType::BF8:
    case DataType::HF8:
    case DataType::UINT4:
    case DataType::INT4:
    case DataType::UINT2:
    case DataType::INT2:
    case DataType::UINT1:
    case DataType::INT1:
        return 1;

    default:  // unknown types
        return 0;
    }
}

/** @brief Get the size of the dtype in bits
 *
 * @param dtype a DataType object
 * @return size in bits.
 */
inline constexpr int dtype_to_bits(const DataType dtype) noexcept {
    //@todo: handle all possible types from the enum
    switch (dtype) {
    case DataType::INT32:
    case DataType::FLOAT32:
        return 32;
    case DataType::FLOAT16:
    case DataType::BFLOAT16:
        return 16;
    case DataType::UINT8:
    case DataType::INT8:
    case DataType::BF8:
    case DataType::HF8:
        return 8;
    case DataType::UINT4:
    case DataType::INT4:
        return 4;
    case DataType::UINT2:
    case DataType::INT2:
        return 2;
    case DataType::UINT1:
    case DataType::INT1:
        return 1;

    default:  // unknown types
        return 0;
    }
}

/**
 * @brief Get how many elements that have the data type given as a parameter fit in a byte
 * types that do not fit in 1 byte are returning zero
 *
 * @example:
 * if datatype is INT4 =>2 elements per one byte
 * if datatype is INT6 =>1 elements per one byte
 * @precondition: one byte is 8 bits
 *
 * @param datatype to be analyzed
 * @return the number of elements of datatype that can be stored in a byte
 */
inline int types_per_byte(const DataType datatype) {
    const int datatype_size_in_bits{dtype_to_bits(datatype)};  // number of bits for datatype

    if (datatype_size_in_bits <= 8) {
        /* coverity[divide_by_zero] */
        return 8 / datatype_size_in_bits;  // number of elements fitting into 8 bits
    } else {
        return 0;
    }
}

// inline int bytes_per_dimension(DataType dtype, int dimensionSize) {
//     int number_of_types{types_per_byte(dtype)};
//     int reminder = dimensionSize % number_of_types;
//     int quotient = dimensionSize / number_of_types;
//     return (reminder == 0) ? (quotient) : (quotient + 1);
// }

/// @brief true if the footprint of the 2 data types are the same (at bitlevel)
inline bool is_same_datatype_footprint(const DataType d1, const DataType d2) {
    return dtype_to_bits(d1) == dtype_to_bits(d2);
}

/**
 * @brief Get the tensor serial order given a layout
 *
 * @param layout a Tensor Layout
 * @return std::array<unsigned int, 4>, order of dimensions from innermost to outermost. values represent Dim::Act
 *
 * Invalid will be mapped to the default one : ZMAJOR/ZXY
 */
constexpr std::array<unsigned int, 4> layout_to_order(Layout layout) noexcept {
    switch (layout) {
    case Layout::CMAJOR:
        return {Dim::Act::X, Dim::Act::Y, Dim::Act::Z, Dim::Act::B};  // X,Y,Z,B  from innermost to outermost dimensions
    case Layout::ZMAJOR:
        return {Dim::Act::Z, Dim::Act::X, Dim::Act::Y, Dim::Act::B};  // Z,X,Y,B  from innermost to outermost dimensions

    case Layout::XYZ:
        return {Dim::Act::X, Dim::Act::Y, Dim::Act::Z, Dim::Act::B};  // X,Y,Z, B
    case Layout::XZY:
        return {Dim::Act::X, Dim::Act::Z, Dim::Act::Y, Dim::Act::B};  // X,Z,Y, B

    case Layout::YXZ:
        return {Dim::Act::Y, Dim::Act::X, Dim::Act::Z, Dim::Act::B};  // Y,X,Z, B
    case Layout::YZX:
        return {Dim::Act::Y, Dim::Act::Z, Dim::Act::X, Dim::Act::B};  // Y,Z,X, B

    case Layout::ZXY:
        return {Dim::Act::Z, Dim::Act::X, Dim::Act::Y, Dim::Act::B};  // Z,X,Y, B
    case Layout::ZYX:
        return {Dim::Act::Z, Dim::Act::Y, Dim::Act::X, Dim::Act::B};  // Z,Y,X, B

    case Layout::INVALID:                                             // fall-through
    default:                                                          // ZMajor like, this is the first in the enum list
        return {Dim::Act::Z, Dim::Act::X, Dim::Act::Y, Dim::Act::B};  // Z,X,Y,B  from innermost to outermost dimensions
    }
}

/**
 * @brief Return grid in X, Y, Z, B format
 *
 * @param mode a DPUWorkload ExecutionMode
 * @return std::vector<unsigned int>
 */
inline std::vector<unsigned int> mpe_mode_to_grid(ExecutionMode mode) {
    switch (mode) {
    case ExecutionMode::VECTOR:
        return {16, 1, 16, 1};
    case ExecutionMode::VECTOR_FP16:
        return {4, 1, 16, 1};
    default:
        return {4, 4, 16, 1};
    }
}

/**
 * @brief Return the NTHW/NTK grid in X, Y, Z, B format
 *
 * @param mode a DPUWorkload ExecutionMode
 * @return std::vector<unsigned int>
 */
inline std::vector<unsigned int> mpe_mode_to_nthw_ntk_grid(ExecutionMode mode) {
    switch (mode) {
    case ExecutionMode::CUBOID_4x16:
        return {8, 8, 256, 1};
    case ExecutionMode::CUBOID_8x16:
        return {16, 8, 128, 1};
    case ExecutionMode::CUBOID_16x16:
        return {16, 16, 64, 1};
    default:
        return {1, 1, 1, 1};
    }
}

}  // namespace VPUNN

#endif  // VPUNN_DATATYPES_H