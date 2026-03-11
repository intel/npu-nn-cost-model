// Copyright © 2025 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_DMA_MAP_TYPE_SELECTOR_H
#define VPUNN_DMA_MAP_TYPE_SELECTOR_H

#include "core/map_type_selector.h"
#include "vpu/dma_workload.h"
#include <unordered_map>

namespace VPUNN {

/**
 * @brief Custom hasher for DMANNWorkload_NPU27 using the hash() method
 * 
 * This enables std::unordered_map to use DMANNWorkload_NPU27 as a key with O(1) lookup
 * performance instead of O(log n) with std::map.
 */
struct DMANNWorkload_NPU27Hasher {
    std::size_t operator()(const DMANNWorkload_NPU27& wl) const noexcept {
        return static_cast<std::size_t>(wl.hash());
    }
};

/**
 * @brief Specialization of MapTypeSelector for DMANNWorkload_NPU27
 * 
 * Uses std::unordered_map with custom hasher for efficient O(1) average-case lookup.
 * The DMANNWorkload_NPU27 class provides a hash() method that is used by DMANNWorkload_NPU27Hasher.
 * 
 * @tparam V The value type to be stored in the map
 */
template <>
struct MapTypeSelector<DMANNWorkload_NPU27> {
    template <typename V>
    using type = std::unordered_map<DMANNWorkload_NPU27, V, DMANNWorkload_NPU27Hasher>;
};

/**
 * @brief Custom hasher for DMANNWorkload_NPU40_50 using the hash() method
 * 
 * This enables std::unordered_map to use DMANNWorkload_NPU40_50 as a key with O(1) lookup
 * performance instead of O(log n) with std::map.
 */
struct DMANNWorkload_NPU40_50Hasher {
    std::size_t operator()(const DMANNWorkload_NPU40_50& wl) const noexcept {
        return static_cast<std::size_t>(wl.hash());
    }
};

/**
 * @brief Specialization of MapTypeSelector for DMANNWorkload_NPU40_50
 * 
 * Uses std::unordered_map with custom hasher for efficient O(1) average-case lookup.
 * The DMANNWorkload_NPU40_50 class provides a hash() method that is used by DMANNWorkload_NPU40_50Hasher.
 * 
 * @tparam V The value type to be stored in the map
 */
template <>
struct MapTypeSelector<DMANNWorkload_NPU40_50> {
    template <typename V>
    using type = std::unordered_map<DMANNWorkload_NPU40_50, V, DMANNWorkload_NPU40_50Hasher>;
};

}  // namespace VPUNN

#endif  // VPUNN_DMA_MAP_TYPE_SELECTOR_H
