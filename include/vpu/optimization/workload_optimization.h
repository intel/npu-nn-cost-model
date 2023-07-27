// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_WL_OPTIMIZATION_API_H
#define VPUNN_WL_OPTIMIZATION_API_H

#include <utility>
#include <vector>
#include "vpu/cycles_interface_types.h"
#include "vpu/layer.h"
#include "vpu/types.h"
#include "vpu/utils.h"
#include "vpu_cost_model.h"

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

using DPUWorkloads = std::vector<DPUWorkload>;  ///< List of DPU workloads

///  describes a pair of cost and the associated DPUWorkloads
using DPUWorkloadsCost = std::pair<CyclesInterfaceType, DPUWorkloads>;

/**
 * @brief DPU Tiler interface
 */
class DPUTiler {
public:
    /**
     * @brief Generate the optimal intra-tile split for a specific DPULayer
     * @details This function takes the model, the layer to optimize and the nDPU as a parameter and returns the optimal
     * workloads split. The information about the device, sparsity are encoded in the DPULayer type. The mode is part of
     * the DPUWorkload structure
     *
     * @param layer DPULayer to optimize
     * @param options workload splits algorithm configuration options
     * @return DPUWorkloadsCost the optimal workloads split
     */
    virtual DPUWorkloadsCost intraTileSplit(const DPULayer& layer, const SplitOptions& options) = 0;

    /**
     * @brief Get the cycles and power estimate for a list of workloads.
     * @details This function does not optimize any workloads
     * but simply calculate the cost of that configuration. It is possible to pass an optional runtime overhead in
     * cycles
     *
     * @param workloads a vector of DPUWorkload
     * @param runtimeOverhead execution runtime overhead in cycles (per workload)
     * @param skip_power if true power will be zero, otherwise is calculated
     * @return PnPEstimates power and performance estimate for the workloads
     *
     * @throws exceptions from inner dependencies. like DPU invocation
     */
    virtual PnPEstimates getLayerPerformance(const DPUWorkloads& workloads, const unsigned int runtimeOverhead = 0,
                                             const bool skip_power = true) = 0;

    /**
     * @brief Destroy the DPUTiler object
     */
    virtual ~DPUTiler() = default;
};

/**
 * @brief Factory function that generates a DPUTiler instance
 *
 * @param _model a shared pointer to a VPUCostModel object
 * @return std::unique_ptr<DPUTiler>
 */
std::unique_ptr<DPUTiler> getDPUTiler(std::shared_ptr<VPUCostModel> const& _model);

}  // namespace VPUNN

#endif  // VPUNN_WL_OPTIMIZATION_API_H