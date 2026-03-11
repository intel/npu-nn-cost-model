// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_MAP_TYPE_SELECTOR_H
#define VPUNN_MAP_TYPE_SELECTOR_H

#include <map>
#include <unordered_map>
#include <vector>
#include "core/utils.h"

namespace VPUNN {

/**
 * @brief Type trait to select appropriate map type based on key type
 *
 * Default: use std::map for types without custom hasher (O(log n) lookup)
 * Specialize this template in the header where your key type is defined
 * to use std::unordered_map with a custom hasher for O(1) lookup
 *
 * @tparam K The key type for the map
 */
template <typename K>
struct MapTypeSelector {
    // Default implementation - can be specialized for specific types
    // template <typename V>
    // using type = std::map<K, V>;
};

// Custom hasher for std::vector<float> using FNV-1a
struct VectorFloatHasher {
    std::size_t operator()(const std::vector<float>& vec) const noexcept {
        return static_cast<std::size_t>(fnv1a_hash(vec));
    }
};

/**
 * @brief Specialization for std::vector<float>: use std::unordered_map with custom hasher
 *
 * Provides O(1) average lookup time for vector<float> keys using FNV-1a hashing
 */
template <>
struct MapTypeSelector<std::vector<float>> {
    template <typename V>
    using type = std::unordered_map<std::vector<float>, V, VectorFloatHasher>;
};

}  // namespace VPUNN

#endif  // VPUNN_MAP_TYPE_SELECTOR_H
