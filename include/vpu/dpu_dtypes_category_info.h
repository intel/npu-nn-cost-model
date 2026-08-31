// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_DPU_DTYPES_CATEGORY_INFO_H
#define VPUNN_DPU_DTYPES_CATEGORY_INFO_H

#include "dpu_types.h"

namespace VPUNN {

class DTypeCategoryInfo {
public:
    /// @brief True when dtype is any floating-point format used by the model.
    static inline constexpr bool is_any_float_dtype(const DataType dtype) noexcept {
        switch (dtype) {
        case DataType::FLOAT32:
        case DataType::FLOAT16:
        case DataType::BFLOAT16:
        case DataType::BF8:
        case DataType::HF8:
        case DataType::FLOAT4:
            return true;
        default:
            return false;
        }
    }

    /// @brief True when dtype belongs to FP16/BF16/FP32 native-compute family.
    static inline constexpr bool is_fp16family_dtype(const DataType dtype) noexcept {
        switch (dtype) {
        case DataType::FLOAT32:  // not supported, but kept to have all types coverage
        case DataType::FLOAT16:
        case DataType::BFLOAT16:
            return true;
        default:
            return false;
        }
    }

    /// @brief True when dtype belongs to FP8 family.
    static inline constexpr bool is_fp8family_dtype(const DataType dtype) noexcept {
        switch (dtype) {
        case DataType::BF8:
        case DataType::HF8:
            return true;
        default:
            return false;
        }
    }

    /// @brief True when dtype belongs to 16-bit integer family.
    static inline constexpr bool is_i16family_dtype(const DataType dtype) noexcept {
        switch (dtype) {
        case DataType::UINT16:
        case DataType::INT16:
            return true;
        default:
            return false;
        }
    }

    /// @brief True when dtype belongs to i8/sub-8/INT32 family used by native i8 path.
    static inline constexpr bool is_i8family_dtype(const DataType dtype) noexcept {
        switch (dtype) {
        case DataType::UINT8:
        case DataType::INT8:
        case DataType::UINT4:
        case DataType::INT4:
        case DataType::UINT2:
        case DataType::INT2:
        case DataType::UINT1:
        case DataType::INT1:

        // No support for operations in DPU with these types yet
        case DataType::INT32:

            // Note: UINT16/INT16 are excluded here - they use is_i16family_dtype() for power factor lookup
            return true;
        default:
            return false;
        }
    }

    /// @brief True when dtype is any integer format used by the model.
    static inline constexpr bool is_any_int_dtype(const DataType dtype) noexcept {
        switch (dtype) {
        case DataType::UINT8:
        case DataType::INT8:
        case DataType::UINT4:
        case DataType::INT4:
        case DataType::UINT2:
        case DataType::INT2:
        case DataType::UINT1:
        case DataType::INT1:
        case DataType::INT32:
        case DataType::UINT16:
        case DataType::INT16:
            return true;
        default:
            return false;
        }
    }
};

}  // namespace VPUNN

#endif  // VPUNN_DPU_DTYPES_CATEGORY_INFO_H
