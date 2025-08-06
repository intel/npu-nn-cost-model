// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_WL_OPTIMIZATION_TYPES_H
#define VPUNN_WL_OPTIMIZATION_TYPES_H

#include <vector>
#include "vpu/cycles_interface_types.h"
#include "vpu/types.h"

namespace VPUNN {

/**
 * @brief Available VPU workload generation optimization targets
 *
 */
enum class VPUOptimizationTarget { LATENCY, POWER, EFFICIENCY };
/**
 * @brief Available VPU splitting strategies
 *
 */
enum class VPUSplitStrategy { HW_TILING, Z_TILING, H_TILING, W_TILING };

/**
 * @brief VPU splitting optimization configuration options
 * Used to guide the splitting of a Layer to 1 or more DPUs
 */
struct SplitOptions {
    unsigned int maxWorkloads{128U};  ///< Maximum number of workloads available. Default is 128 because of FIFO size
    unsigned int maxLatencyUs{0};     ///< Number of DPU to optimize for. maxLatencyMs = 0 means full search
    unsigned int nDPU{0};  ///< Number of DPU to optimize for. Setting nDPU = 0 VPUNN auto-detects the number of DPUs
                           ///< based on the device
    unsigned int runtimeOverhead{0};  ///<  Per workload runtime overhead in cycles

    VPUOptimizationTarget target{VPUOptimizationTarget::LATENCY};  ///< Optimization target. Default is LATENCY,USED
                                                                   ///< only for LATENCY for the moment
    std::vector<VPUSplitStrategy> availableStrategies{
            VPUSplitStrategy::HW_TILING,
            VPUSplitStrategy::Z_TILING};  ///<  Valid strategies for splitting a layer into multiple workloads. Default
                                          ///<  is all (HW tiling and Z tiling)
};

/**
 * @brief VPU Power and Performance estimates (cycles and power)
 */
struct PnPEstimates {
    CyclesInterfaceType cycles;  ///< execution cycles
    float power;                 ///< power in mW
};

}  // namespace VPUNN

#endif  // VPUNN_WL_OPTIMIZATION_API_H
