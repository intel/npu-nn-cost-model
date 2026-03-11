// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#ifndef HTTP_WORKLOAD_VARIANT_H_
#define HTTP_WORKLOAD_VARIANT_H_

#include <functional>
#include <variant>
#include "vpu/dma_types.h"
#include "vpu/validation/data_dpu_operation.h"

namespace VPUNN {

/**
 * @brief Variant of all workload types supported by the HTTP cost provider.
 * @details Holds reference_wrappers to avoid copying workload objects. To support a new workload type, add it to VariantType.
 *          This header is intended to be included only from .cpp files to avoid pulling heavy type dependencies into
 *          the lightweight http_cost_provider_intf.h interface header.
 */
struct HttpWorkloadVariant {
    using VariantType = std::variant<std::reference_wrapper<const DPUOperation>,
                                    std::reference_wrapper<const DMANNWorkload_NPU27>,
                                    std::reference_wrapper<const DMANNWorkload_NPU40_50>>;

    VariantType data;

    template <typename T>
    HttpWorkloadVariant(T&& val) : data(std::forward<T>(val)) {}
};

}  // namespace VPUNN

#endif  // HTTP_WORKLOAD_VARIANT_H_
