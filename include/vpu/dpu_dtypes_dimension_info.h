// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_DPU_DTYPES_DIMENSION_INFO_H
#define VPUNN_DPU_DTYPES_DIMENSION_INFO_H

#include "dpu_types.h"

namespace VPUNN {

class DTypeDimensionInfo {
public:
    /** @brief Get the size of the dtype
     *
     * @param dtype a DataType object
     * @return size in bytes if valid dtypes, else return -1 for invalid types
     */
    static inline constexpr int dtype_to_bytes(const DataType dtype) noexcept {
        // intermediate 1+ bytes are not handled, will be when present
        switch (dtype) {
        case DataType::INT32:
        case DataType::FLOAT32:
            return 4;

        case DataType::UINT16:
        case DataType::INT16:
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
        case DataType::FLOAT4:
        case DataType::UINT2:
        case DataType::INT2:
        case DataType::UINT1:
        case DataType::INT1:
            return 1;

        default:  // unknown types
            return -1;
        }
    }

    /** @brief Get the size of the dtype in bits
     *
     * @param dtype a DataType object
     * @return size in bits if dtype valid, else return -1 for invalid types
     */
    static inline constexpr int dtype_to_bits(const DataType dtype) noexcept {
        //@todo: handle all possible types from the enum
        switch (dtype) {
        case DataType::INT32:
        case DataType::FLOAT32:
            return 32;
        case DataType::UINT16:
        case DataType::INT16:
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
        case DataType::FLOAT4:
            return 4;
        case DataType::UINT2:
        case DataType::INT2:
            return 2;
        case DataType::UINT1:
        case DataType::INT1:
            return 1;

        default:  // unknown types
            return -1;
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
    static inline int types_per_byte(const DataType datatype) {
        const int datatype_size_in_bits{dtype_to_bits(datatype)};  // number of bits for datatype

        if (datatype_size_in_bits <= 8) {
            return 8 / datatype_size_in_bits;  // number of elements fitting into 8 bits
        } else {
            return 0;
        }
    }

    /// @brief true if the footprint of the 2 data types are the same (at bitlevel)
    static inline bool is_same_datatype_footprint(const DataType d1, const DataType d2) {
        return dtype_to_bits(d1) == dtype_to_bits(d2);
    }
};

}  // namespace VPUNN

#endif  // VPUNN_DPU_DTYPES_DIMENSION_INFO_H
