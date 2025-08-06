// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_WL_OPTIMIZATION_API_H
#define VPUNN_WL_OPTIMIZATION_API_H

#include "vpu/layer.h"
#include "vpu/types.h"
#include "vpu_cost_model.h"
#include "workload_optimization_types.h"

#include "vpu/layer_split_info.h"

namespace VPUNN {

/**
 * @brief DPU Tiler interface
 */
class IDPUTiler {
public:
    /**
     * @brief Generate the optimal intra-tile split for a specific DPULayer
     * @details This function takes the model, the layer to optimize and the nDPU as a parameter and returns the optimal
     * workloads split. The information about the device, sparsity are encoded in the DPULayer type. The mode is part of
     * the DPUWorkload structure
     *
     * @param layer DPULayer to optimize
     * @param options workload splits algorithm configuration options
     * @param complete_output_splits Output parameter, will be filled with full list of splits investigated
     * @return DPUWorkloadsCost the optimal workloads split
     */
    virtual DPUWorkloadsCost intraTileSplit(
            const DPULayer& layer, const SplitOptions& options,
            std::vector<DPUWorkloadsWithCyclesSplit>* complete_output_splits = nullptr) const = 0;

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
    virtual PnPEstimates getLayerPerformance(DPUWorkloadsWithCyclesSplit& workloads_split,
                                             const unsigned int runtimeOverhead = 0,
                                             const bool skip_power = true) const = 0;

    /**
     * @brief Destroy the DPUTiler object
     */
    virtual ~IDPUTiler() = default;
};

/**
 * @brief Factory function that generates a IDPUTiler instance
 *
 * @param _model a reference to a VPUCostModel object
 * @return std::unique_ptr<IDPUTiler>
 */
std::unique_ptr<IDPUTiler> getDPUTiler(VPUCostModel& _model);

}  // namespace VPUNN

#endif  // VPUNN_WL_OPTIMIZATION_API_H
