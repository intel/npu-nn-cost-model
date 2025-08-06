// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_SELECT_OPTIMAL_EXECUTION_MODE_H
#define VPUNN_SELECT_OPTIMAL_EXECUTION_MODE_H

#include <vpu/layer.h>
#include <vpu/types.h>
#include <vpu/utils.h>
//#include <vpunn.h>
#include <algorithm>
#include <string>
#include <vector>

namespace VPUNN {

static ExecutionMode aux_select_optimal_execution(VPUCostModel& model, const DPULayer& layer,
                                                  const std::vector<ExecutionMode>& available_modes) {
    // The list of cost for each mode
    std::vector<unsigned int> mode_cost;
    // Auxiliary workloads from a layer
    DPUWorkload wl = static_cast<DPUWorkload>(layer);

    for (auto nthw_ntk_mode : available_modes) {
        // Assign the mode
        wl.execution_order = nthw_ntk_mode;
        auto cost = model.DPU(wl);
        mode_cost.push_back(cost);
    }

    // Get the index with minimum cost
    auto min_index = std::min_element(mode_cost.begin(), mode_cost.end()) - mode_cost.begin();
    return available_modes[min_index];
}

ExecutionMode select_optimal_grid_VPU2x(VPUCostModel& model, const DPULayer& layer,
                                  const std::vector<ExecutionMode>& available_modes = {ExecutionMode::VECTOR,
                                                                                       ExecutionMode::MATRIX}) {
    if (layer.device != VPUDevice::VPU_2_0 && layer.device != VPUDevice::VPU_2_1) {
        Logger::error() << "Invalid VPU device type. Only 2.0 and 2.1 available";
    }

    // If the input tensor is float, then the optimal mode is VECTOR_FP16
    if (layer.inputs[0].is_fp16family()) {
        return ExecutionMode::VECTOR_FP16;
    }

    // Select the optimal model given the available ones
    return aux_select_optimal_execution(model, layer, available_modes);
}

ExecutionMode select_optimal_nthw_ntk(VPUCostModel& model, const DPULayer& layer,
                                      const std::vector<ExecutionMode>& available_modes = {
                                              ExecutionMode::CUBOID_16x16, ExecutionMode::CUBOID_4x16,
                                              ExecutionMode::CUBOID_8x16}) {
    if (layer.device != VPUDevice::VPU_2_7 && layer.device != VPUDevice::VPU_4_0 &&
        layer.device != VPUDevice::NPU_RESERVED && layer.device != VPUDevice::NPU_RESERVED_W) {
        Logger::error() << "Invalid VPU device type. Only 2.7, 4.0 and 5.0 available";
    }

    // Select the optimal model given the available ones
    return aux_select_optimal_execution(model, layer, available_modes);
}

ExecutionMode select_optimal_execution_mode(VPUCostModel& model, const DPULayer& layer) {
    switch (layer.device) {
    case VPUDevice::VPU_2_0:
    case VPUDevice::VPU_2_1:
        return select_optimal_grid_VPU2x(model, layer);
    case VPUDevice::VPU_2_7:
    case VPUDevice::VPU_4_0:
    case VPUDevice::NPU_RESERVED:
    case VPUDevice::NPU_RESERVED_W:
        return select_optimal_nthw_ntk(model, layer);
    default:
        Logger::error() << "Invalid VPU device type";
        exit(-1);
    }
}

}  // namespace VPUNN

#endif  // VPUNN_SELECT_OPTIMAL_EXECUTION_MODE_H
