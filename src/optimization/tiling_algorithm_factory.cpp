// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include <memory>

#include "vpu/optimization/htiling.h"
#include "vpu/optimization/hwtiling.h"
#include "vpu/optimization/tiling_algorithm_factory.h"
#include "vpu/optimization/tiling_primitives.h"
#include "vpu/optimization/wtiling.h"
#include "vpu/optimization/ztiling.h"
#include "vpu/optimization/zefftiling.h"

namespace VPUNN {

/**
 * @brief Get the Tiling Algorithms objects, These are intra tile algos (splitting to DPUWorkloads)
 */
TilingAlgorithmsContainer TilingAlgorithmFactory::getTilingAlgorithms(const DPULayer& layer,
                                                                      const SplitOptions& options) {
    TilingAlgorithmsContainer algos;
    // @todo: re-factor this to be more explicit; Like pass 1 , filter out not allowed strategies, step 2 create Tiling
    // algorithms for allowed ones.

    for (auto strategy : options.availableStrategies) {
        switch (strategy) {
        case VPUSplitStrategy::Z_TILING:
            if (layer.device >= VPUDevice::NPU_RESERVED) {
                algos.push_back(std::make_unique<ZEffTiling>(layer, options.maxWorkloads));
            }
            else {
                algos.push_back(std::make_unique<ZTiling>(layer, options.maxWorkloads));
            }
            break;
        case VPUSplitStrategy::HW_TILING:
            if (TilingPrimitives::isHWTilingAllowed(layer, options))  // Z Tiling will inhibit HW on 1 dpu
                algos.push_back(std::make_unique<HWTiling>(layer, options.maxWorkloads));
            break;
        case VPUSplitStrategy::H_TILING:
            if (TilingPrimitives::isHWTilingAllowed(layer, options))
                algos.push_back(std::make_unique<HTiling>(layer, options.maxWorkloads));
            break;
        case VPUSplitStrategy::W_TILING:
            if (TilingPrimitives::isHWTilingAllowed(layer, options))
                algos.push_back(std::make_unique<WTiling>(layer, options.maxWorkloads));
            break;
        default:
            continue;
        }
    }
    return algos;
}

}  // namespace VPUNN