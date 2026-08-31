// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_WORKLOAD_SEMANTICS_INFO_H
#define VPUNN_WORKLOAD_SEMANTICS_INFO_H

#include "dpu_dtypes_dimension_info.h"
#include "dpu_types.h"

namespace VPUNN {

/// @brief Shared semantic predicates for workload-like representations.
class WorkloadSemanticsInfo {
public:
    static inline bool is_elementwise_like_operation(const Operation op) {
        return ((op == Operation::ELTWISE) || (op == Operation::ELTWISE_MUL));
    }

    static inline bool is_preconditions_for_inplace_output(const Layout in_layout, const DataType in_dtype,
                                                           const Layout out_layout, const DataType out_dtype) {
        return (in_layout == out_layout) && DTypeDimensionInfo::is_same_datatype_footprint(in_dtype, out_dtype);
    }

    static inline bool is_special_no_weights_situation(const Layout in_layout, const DataType in_dtype,
                                                       const Layout out_layout, const DataType out_dtype) {
        return (in_layout != out_layout) || !DTypeDimensionInfo::is_same_datatype_footprint(in_dtype, out_dtype);
    }
};

}  // namespace VPUNN

#endif  // VPUNN_WORKLOAD_SEMANTICS_INFO_H
