// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/optimization/tiler.h"
#include "vpu/optimization/tiling_primitives.h"

#include <list>
#include <utility>

#include "vpu/cycles_interface_types.h"

namespace VPUNN {

void ITilerAlgorithm::setWorkloadsModeAndInfereInputShape(DPUWorkloadsWithCyclesSplit& workloads_split,
                                                          const ExecutionMode mode, const DPULayer& originalLayer) {
    for (auto& wl : workloads_split.workloads) {
        wl.execution_order = mode;
        TilingPrimitives::inferInputTensorShape(wl, originalLayer);
    }
}

std::list<DPUWorkloadsWithCyclesSplit> ITilerAlgorithm::split_tile_in_workloads(const ExecutionMode mode,
                                                                                const unsigned int nWorkloads) {
    std::list<DPUWorkloadsWithCyclesSplit> splitPool;
    // Optimized for 1 workloads  (todo Analyse if necessary,  what if it is not valid?)
    if (nWorkloads == 1) {
        DPUWorkloadsWithCyclesSplit workloads_split{{Cycles::NO_ERROR}, {layer_on_tile}};  // same as original
        ITilerAlgorithm::setWorkloadsModeAndInfereInputShape(workloads_split, mode,
                                                             layer_on_tile);  // computes also input tensor
        splitPool.push_back(std::move(workloads_split));
    } else {
        tileMultipleWl(splitPool, mode, nWorkloads);
    }

    return splitPool;
}

}  // namespace VPUNN