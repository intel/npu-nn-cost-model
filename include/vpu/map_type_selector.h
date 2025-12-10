// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_MAP_TYPE_SELECTOR_H
#define VPUNN_MAP_TYPE_SELECTOR_H

#include "vpu/dma_types.h"

// Template specialization for MapTypeSelector (defined in cache.h)
// This must be in global namespace after VPUNN namespace closes
namespace VPUNN {
template <typename K>
struct MapTypeSelector;  // Forward declaration

// Specialization for DPUWorkload: use std::unordered_map with custom hasher
template <>
struct MapTypeSelector<DPUWorkload> {
    template <typename V>
    using type = std::unordered_map<DPUWorkload, V, DPUWorkloadHasher>;
};
}  // namespace VPUNN
#endif  // VPUNN_MAP_TYPE_SELECTOR_H
