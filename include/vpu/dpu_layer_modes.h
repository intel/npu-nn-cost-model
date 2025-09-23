// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_DPU_LAYER_MODES_H
#define VPUNN_DPU_LAYER_MODES_H

#include <cmath>
#include <stdexcept>
#include <vector>

#include "vpu/dpu_types.h"
#include "vpu/layer.h"
#include "vpu/device_layer_properties/device_layer_properties_holder.h"

namespace VPUNN {

/// provides differentiated information for a layer based on its content
class DPULayerModes {
private:
    /// @brief Get the valid ExecutionMode for VPU_2_0
    ///
    /// @param wl the DPULayer
    /// @return std::vector<ExecutionMode>
    static std::vector<ExecutionMode> getValidExecutionMode_2_0(const DPULayer& wl) {
        // Float input or output -> ExecutionMode::VECTOR_FP16
        if (wl.inputs[0].is_fp16family() || wl.outputs[0].is_fp16family())
            return {ExecutionMode::VECTOR_FP16};
        // Find the optimal Execution Mode given output tensor layout
        auto shape = wl.outputs[0].get_shape();
        const double W = static_cast<double>(shape[Dim::Act::X]);
        const double H = static_cast<double>(shape[Dim::Act::Y]);
        // ExecutionMode::MATRIX process tensor using a W=4 H=4 grid, calculate grid cells count for it
        const double matrixPartsCount = std::ceil(W / 4.0) * std::ceil(H / 4.0);
        // ExecutionMode::VECTOR process tensor using a W=16 H=1 grid, calculate grid cells count for it
        const double vectorPartsCount = std::ceil(W / 16.0) * H;
        // Cells count is in direct ratio with work size, so choose smaller one
        if (vectorPartsCount <= matrixPartsCount) {
            return {ExecutionMode::VECTOR};
        }
        return {ExecutionMode::MATRIX};
    }

    /// @brief Get the valid ExecutionMode for VPU_2_7
    ///
    /// @param wl the DPULayer
    /// @return std::vector<ExecutionMode>
    static std::vector<ExecutionMode> getValidExecutionMode_2_7(const DPULayer& wl) {
        // The available mode choice is based on the OP type
        switch (wl.op) {
        case Operation::CM_CONVOLUTION:  // compressconv surogate
        case Operation::DW_CONVOLUTION:
        case Operation::AVEPOOL:
        case Operation::MAXPOOL:
            return {ExecutionMode::CUBOID_16x16};
        case Operation::ELTWISE:
            return {ExecutionMode::CUBOID_8x16};
        default:
            return {ExecutionMode::CUBOID_16x16, ExecutionMode::CUBOID_8x16, ExecutionMode::CUBOID_4x16};
        }
    }

    /// @brief Get the valid ExecutionMode for NPU_5.0
    ///
    /// @param wl the DPULayer
    /// @return std::vector<ExecutionMode> to_do
    static std::vector<ExecutionMode> getValidExecutionMode_RESERVED(const DPULayer& wl) {
        // The available mode choice is based on the OP type
        switch (wl.op) {
        case Operation::DW_CONVOLUTION:
        case Operation::AVEPOOL:
        case Operation::MAXPOOL:
            return {ExecutionMode::CUBOID_16x16};
        case Operation::ELTWISE:
        case Operation::ELTWISE_MUL:
            return {ExecutionMode::CUBOID_8x16};
        case Operation::LAYER_NORM:  // is this the case?
        default:
            return {ExecutionMode::CUBOID_16x16, ExecutionMode::CUBOID_8x16, ExecutionMode::CUBOID_4x16};
        }
    }

    /// @brief Get the valid ExecutionMode for NPU_6.0
    ///
    /// @param wl the DPULayer
    /// @return std::vector<ExecutionMode>
    static std::vector<ExecutionMode> getValidExecutionMode_6_0(const DPULayer& wl) {
        switch (wl.op) {
            case Operation::CM_CONVOLUTION:
            case Operation::DW_CONVOLUTION:
            case Operation::AVEPOOL:
            case Operation::MAXPOOL:
                return {ExecutionMode::CUBOID_16x16};
            case Operation::ELTWISE:
            case Operation::ELTWISE_MUL:
                return {ExecutionMode::CUBOID_8x16};
            case Operation::LAYER_NORM:
            default:
                return {ExecutionMode::CUBOID_16x16, ExecutionMode::CUBOID_8x16, ExecutionMode::CUBOID_4x16};
        }
    }


    DPULayerModes() = default;  // no instance possible

public:
    /// @brief Get the valid ExecutionMode for the DPULayer
    ///
    /// @param wl the DPULayer
    /// @return std::vector<ExecutionMode>
    static std::vector<ExecutionMode> getValidExecutionMode(const DPULayer& wl) {
        if (wl.device >= VPUDevice::__size)
            return {};
        else
            return LayerPropertiesHolder::get_properties(wl.device).getValidTilingExecutionMode(wl);
    }
};

}  // namespace VPUNN

#endif  // VPUNN_DPU_LAYER_MODES_H
