// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_DPU_TILER_INTF_H
#define VPUNN_DPU_TILER_INTF_H

#include <vector>

#include "split_options.h"
#include "vpu/cycles_interface_types.h"
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

protected:
    /**
     * @brief Get the cycles for a list of workloads.
     * @details This function does not optimize any workloads
     * but simply calculate the cost of that configuration. It is possible to pass an optional runtime overhead in
     * cycles
     *
     * @param workloads a vector of DPUWorkload
     * @param runtimeOverhead execution runtime overhead in cycles (per workload)
     * @return CyclesInterfaceType total cycles for the workloads
     *
     * @throws exceptions from inner dependencies. like DPU invocation
     */
    virtual CyclesInterfaceType computeSplitCycles(DPUWorkloadsWithCyclesSplit& workloads_split,
                                                   const unsigned int runtimeOverhead = 0) const = 0;

public:
    /**
     * @brief Destroy the DPUTiler object
     */
    virtual ~IDPUTiler() = default;
};

}  // namespace VPUNN

#endif  // DPU_TILER_INTF_H
