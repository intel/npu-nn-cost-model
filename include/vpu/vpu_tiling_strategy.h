// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_VPU_TILING_STRATEGY_H
#define VPUNN_VPU_TILING_STRATEGY_H

#include <string>
#include "vpu/dpu_types.h"

namespace VPUNN {

/// @brief VPU tiling strategy. How to split a Layer on multiple tiles
enum class VPUTilingStrategy {
    NONE,              // Clustering, replicate on each tile the input
    SOH_Overlapped,    // old SOH, this is SOH_overlapped for 2.7 inputs, Kept same enum value (1). 2.7, 4.0+
    SOK,               // 4.0+
    SOW,               // 4.0+ equivalent of SOH_Overlappeed on W dimension
    SOHW,              // 4.0+ , 4 tiles
    SOHK,              // 4.0+ , 4 tiles,   this is not HKswitch (HKS is just a H with full broadcast)
    SOH_HaloRead,      // SOH with input Halo for 2.7 (only) . (4.0 has only Overlapped)
    SOHO_K_SWITCH,     // HK switch with H = SOHO, + broadcast
    SOH_K_SWITCH,      // HK switch with H = SOH (possible in 2.7 only)
    SOK_NO_BROADCAST,  // K split , bu no broadcast. Smaller output memory
    UNKNOWN,           // not known, or not communicated (is not receiver decision to implement/apply)
    __size
};
inline static const EnumMap VPUTilingStrategy_ToText{
        link(VPUTilingStrategy::NONE, "NONE"),
        link(VPUTilingStrategy::SOH_Overlapped, "SOHO"),
        link(VPUTilingStrategy::SOK, "SOK"),
        link(VPUTilingStrategy::SOW, "SOW"),
        link(VPUTilingStrategy::SOHW, "SOHW"),
        link(VPUTilingStrategy::SOHK, "SOHK"),
        link(VPUTilingStrategy::SOH_HaloRead, "SOH_HaloRead"),
        link(VPUTilingStrategy::SOHO_K_SWITCH, "SOHO_K_SWITCH"),
        link(VPUTilingStrategy::SOH_K_SWITCH, "SOH_K_SWITCH"),
        link(VPUTilingStrategy::SOK_NO_BROADCAST, "SOK_NO_BROADCAST"),
        link(VPUTilingStrategy::UNKNOWN, "UNKNOWN"),
};

template <>
inline const EnumMap& mapToText<VPUTilingStrategy>() {
    return VPUTilingStrategy_ToText;
}

template <>
inline std::string enumName<VPUTilingStrategy>() {
    return "VPUTilingStrategy";
}

}  // namespace VPUNN

#endif  // VPUNN_VPU_TILING_STRATEGY_H
