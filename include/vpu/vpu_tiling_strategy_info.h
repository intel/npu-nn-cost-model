// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_VPU_TILING_STRATEGY_INFO_H
#define VPUNN_VPU_TILING_STRATEGY_INFO_H

#include "vpu/vpu_tiling_strategy.h"

namespace VPUNN {

class TilingStrategyInfo {
public:
    static bool isVerticalTiling(VPUTilingStrategy strategy) {
        switch (strategy) {  // SOH like, cutting the vertical dimension
        case VPUTilingStrategy::NONE:
            return false;
        case VPUTilingStrategy::SOHW:
            return true;
        case VPUTilingStrategy::SOHK:
        case VPUTilingStrategy::SOH_HaloRead:
        case VPUTilingStrategy::SOHO_K_SWITCH:
        case VPUTilingStrategy::SOH_Overlapped:
            return true;
        case VPUTilingStrategy::SOW:
            return false;
        case VPUTilingStrategy::SOK:
        case VPUTilingStrategy::SOH_K_SWITCH:
        case VPUTilingStrategy::SOK_NO_BROADCAST:
        default:
            return false;
        }
    }
    static bool isHorizontalTiling(VPUTilingStrategy strategy) {
        switch (strategy) {  // SOW like, cutting the horizontal dimension
        case VPUTilingStrategy::NONE:
            return false;
        case VPUTilingStrategy::SOHW:
            return true;
        case VPUTilingStrategy::SOHK:
        case VPUTilingStrategy::SOH_HaloRead:
        case VPUTilingStrategy::SOHO_K_SWITCH:
        case VPUTilingStrategy::SOH_Overlapped:
            return false;
        case VPUTilingStrategy::SOW:
            return true;
        case VPUTilingStrategy::SOK:
        case VPUTilingStrategy::SOH_K_SWITCH:
        case VPUTilingStrategy::SOK_NO_BROADCAST:
        default:
            return false;
        }
    }

private:
    // TilingStategyInfo() = delete;
};

}  // namespace VPUNN

#endif  // VPUNN_VPU_TILING_STRATEGY_INFO_H
