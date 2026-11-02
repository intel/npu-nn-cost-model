// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_HTTP_WORKLOAD_JSON_H_
#define VPUNN_HTTP_WORKLOAD_JSON_H_

#include <nlohmann/json.hpp>
#include "vpu/dma_types.h"
#include "vpu/shave_workload.h"
#include "vpu/validation/data_dpu_operation.h"

namespace VPUNN {
/**
 * @brief Maps a workload type to its JSON payload key used by the HTTP profiling service.
 * @details Specialize this for each type added to HttpWorkloadVariant::VariantType.
 */
template <typename WlT>
struct WorkloadKeyTrait {
    // default as nothing , so that if we forget to specialize for a
    // workload type we get a compile error when trying to access the key
};

template <>
struct WorkloadKeyTrait<DPUOperation> {
    static constexpr const char* key = "dpu_workload";
};

template <>
struct WorkloadKeyTrait<DMANNWorkload_NPU27> {
    static constexpr const char* key = "dma_workload";
};

template <>
struct WorkloadKeyTrait<DMANNWorkload_NPU40_50> {
    static constexpr const char* key = "dma_workload";
};

template <>
struct WorkloadKeyTrait<SHAVEWorkload> {
    static constexpr const char* key = "shave_workload";
};

/**
 * @brief Converts a workload to its JSON representation for the HTTP profiling service.
 * @tparam WlT The type of the workload.
 * @param wl The workload to convert.
 * @return A JSON object representing the workload.
 * @details Each supported workload type must provide an explicit specialization of this function,
 * implemented in src/http_client/workload_json.cpp.
 * When adding a new workload type to HttpWorkloadVariant::VariantType, also add its
 * specialization declaration here and its definition in workload_json.cpp.
 */
template <typename WlT>
nlohmann::json toJson(const WlT& wl) = delete;

template <>
nlohmann::json toJson<DPUOperation>(const DPUOperation& wl);

template <>
nlohmann::json toJson<DMANNWorkload_NPU27>(const DMANNWorkload_NPU27& wl);

template <>
nlohmann::json toJson<DMANNWorkload_NPU40_50>(const DMANNWorkload_NPU40_50& wl);

template <>
nlohmann::json toJson<SHAVEWorkload>(const SHAVEWorkload& wl);
}  // namespace VPUNN
#endif  // VPUNN_HTTP_WORKLOAD_JSON_H_
