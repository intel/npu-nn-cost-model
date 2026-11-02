// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_DMA_STRIDE_LUT_FALLBACK_H
#define VPUNN_DMA_STRIDE_LUT_FALLBACK_H

#include <algorithm>  // for std::max

#include "vpu/dma_types.h"                                      // for MemoryDirection
#include "vpu/hw_characteristics/itf_HW_characteristics_set.h"  // for IHWCharacteristicsSet
#include "vpu/types.h"                                          // for VPUDevice

namespace VPUNN {

/**
 * @brief Single source of truth for what happens when a (device, direction) entry is absent from the
 * DMA stride-math LUT (@ref DMAStridedMathModel::get_modeling_function_map).
 */
class DMAStrideLUTFallback {
public:
    /// Src/dst stride threshold (bytes) to use when no per-direction LUT entry exists, mirroring the LUT's
    /// own CMX (64B) vs DDR (256B) convention for the endpoint selected by (direction, is_source).
    static int threshold_bytes(MemoryDirection direction, bool is_source) {
        const bool is_ddr_side =
                is_source ? (direction == MemoryDirection::DDR2CMX || direction == MemoryDirection::DDR2DDR)
                          : (direction == MemoryDirection::CMX2DDR || direction == MemoryDirection::DDR2DDR);
        return is_ddr_side ? 256 : 64;
    }

    /// Bandwidth-based cycle estimate to use when no linear (k, b) model exists for (device, direction).
    static unsigned long long cycles_by_bandwidth(unsigned long long dma_size, const VPUDevice& device,
                                                  const IHWCharacteristicsSet& hw_info) {
        const int bw_bpc = hw_info.device(device).get_DMA_DDR_interface_bytes();  // bytes per cycle
        // Guard: non-positive bandwidth (e.g. -1 for unknown/default HW) falls back to 1 byte/cycle.
        const int safe_bw_bpc = bw_bpc > 0 ? bw_bpc : 1;
        const auto CMX_cycles = static_cast<float>(dma_size) / static_cast<float>(safe_bw_bpc);
        return std::max(1ULL, static_cast<unsigned long long>(CMX_cycles));
    }
};

}  // namespace VPUNN

#endif  // VPUNN_DMA_STRIDE_LUT_FALLBACK_H
