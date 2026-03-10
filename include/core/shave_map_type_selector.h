// Copyright © 2025 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_SHAVE_MAP_TYPE_SELECTOR_H
#define VPUNN_SHAVE_MAP_TYPE_SELECTOR_H

#include <map>
#include "core/map_type_selector.h"
#include "vpu/shave_workload.h"

namespace VPUNN {

/**
 * @brief Specialization of MapTypeSelector for SHAVEWorkload
 *
 * Uses std::map (ordered map with O(log n) lookup) for SHAVEWorkload keys.
 * This is appropriate when:
 * - A hash function is not available or not efficient for the workload
 * - Ordered iteration over keys is desired
 * - The workload comparison operator is well-defined and efficient
 *
 * @tparam V The value type to be stored in the map
 */
template <>
struct MapTypeSelector<SHAVEWorkload> {
    template <typename V>
    using type = std::map<SHAVEWorkload, V>;
};

}  // namespace VPUNN

#endif  // VPUNN_SHAVE_MAP_TYPE_SELECTOR_H
