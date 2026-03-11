// Copyright © 2025 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_DPU_MAP_TYPE_SELECTOR_H
#define VPUNN_DPU_MAP_TYPE_SELECTOR_H

#include "core/map_type_selector.h"
#include "vpu/dpu_workload.h"
#include <unordered_map>

namespace VPUNN {

/**
 * @brief Custom hasher for DPUWorkload using the hash() method
 * 
 * This enables std::unordered_map to use DPUWorkload as a key with O(1) lookup
 * performance instead of O(log n) with std::map.
 */
struct DPUWorkloadHasher {
    std::size_t operator()(const DPUWorkload& wl) const noexcept {
        return static_cast<std::size_t>(wl.hash());
    }
};

/**
 * @brief Specialization of MapTypeSelector for DPUWorkload
 * 
 * Uses std::unordered_map with custom hasher for efficient O(1) average-case lookup.
 * The DPUWorkload class provides a hash() method that is used by DPUWorkloadHasher.
 * 
 * @tparam V The value type to be stored in the map
 */
template <>
struct MapTypeSelector<DPUWorkload> {
    template <typename V>
    using type = std::unordered_map<DPUWorkload, V, DPUWorkloadHasher>;
};

}  // namespace VPUNN

#endif  // VPUNN_DPU_MAP_TYPE_SELECTOR_H
