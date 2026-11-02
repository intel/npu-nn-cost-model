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

namespace VPUNN {

/**
 * @brief Type trait to select appropriate map type based on key type
 *
 * No default `type` member is provided: K must be given an explicit specialization
 * (using std::unordered_map with a custom hasher) elsewhere, or usage fails to compile.
 *
 * @tparam K The key type for the map
 */
template <typename K>
struct MapTypeSelector {
    // intentionally empty (no `type` alias): forces an explicit specialization per K
};

}  // namespace VPUNN

#endif  // VPUNN_MAP_TYPE_SELECTOR_H
