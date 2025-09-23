// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef LAYER_PROPERTIES_BASE_H
#define LAYER_PROPERTIES_BASE_H

#include "interface_device_layer_properties.h"
#include "vpu/layer.h"
#include "vpu/types.h"

namespace VPUNN {

/// @brief Provides device-specific execution mode selection for tiling.
///
/// This template class implements the logic to return valid execution modes for a given operation
/// by using a static mapping provided by the template parameter V. It is intended to be used as a
/// building block for device-specific layer property classes.
///
/// @tparam V A struct or class that provides a static op_to_exec_mode map.
template <typename V>
class TilingExecutionMode {
private:
    inline static const std::unordered_map<Operation, std::vector<ExecutionMode>> op_to_exec_mode{
            V::op_to_exec_mode};  ///< map of operation to valid execution modes

public:
    // TilingExecutionMode() = default;

    /// @brief Returns the valid execution modes for tiling for a given DPULayer
    /// @param wl The DPULayer
    /// @return A vector of supported ExecutionMode values
    const std::vector<ExecutionMode> getValidTilingExecutionMode(const DPULayer& wl) const {
        auto it = op_to_exec_mode.find(wl.op);
        if (it != op_to_exec_mode.end()) {
            return it->second;
        }
        // Default fallback
        return {ExecutionMode::CUBOID_16x16, ExecutionMode::CUBOID_8x16, ExecutionMode::CUBOID_4x16};
    }
};

/// @brief Provides device-specific default execution mode.
///
/// This template class implements the logic to return the default execution mode for a given device,
/// using a static value provided by the template parameter V. It is intended to be used as a
/// building block for device-specific layer property classes.
///
/// @tparam V A struct or class that provides a static default_execution_mode value.
template <typename V>
class DefaultExecutionMode {
    inline static const ExecutionMode default_execution_mode{
            V::default_execution_mode};  ///< default execution mode for layers

public:
    /// @brief Returns the default execution mode for a given tensor
    /// @param tensor The VPUTensor
    /// @return The default ExecutionMode
    ExecutionMode getValidDefaultExecutionMode(const VPUTensor&) const {
        return default_execution_mode;
    }
};

/// @brief Provides device-specific tiling strategies.
///
/// This template class implements the logic to return valid tiling strategies for a given device,
/// using a static value provided by the template parameter V. It is intended to be used as a
/// building block for device-specific layer property classes.
///
/// @tparam V A struct or class that provides a static valid_tiling_strategies value.
template <typename V>
class TilingStrategies {
private:
    inline static const std::vector<VPUTilingStrategy> valid_tiling_strategies{
            V::valid_tiling_strategies};  ///< valid tiling strategies for this device

public:
    /// @brief Returns the valid tiling strategies for a certain device
    /// @return A vector of supported VPUTilingStrategy values
    const std::vector<VPUTilingStrategy> getValidTilingStrategies() const {
        return valid_tiling_strategies;
    }
};

/// @brief Aggregates all device-specific layer property behaviors into a single class.
///
/// This template class combines tiling execution mode, default execution mode, and tiling strategies
/// into a single interface for a device. It is used to define the complete set of layer properties
/// for a specific device.
///
/// @tparam TilingExecutionMode  The tiling execution mode provider.
/// @tparam DefaultExecutionMode The default execution mode provider.
/// @tparam TilingStrategies     The tiling strategies provider.
template <class TilingExecutionModeX, class DefaultExecutionModeX, class TilingStrategiesX>
class LayerProperties_All_Devices :
        public ILayerProperties,
        public TilingExecutionModeX,
        public DefaultExecutionModeX,
        public TilingStrategiesX {
public:
    // using DefaultExecutionModeX::getValidDefaultExecutionMode;
    // using TilingExecutionModeX::getValidTilingExecutionMode;
    // using TilingStrategiesX::getValidTilingStrategies;

    ExecutionMode getValidDefaultExecutionMode(const VPUTensor& tensor) const override {
        return DefaultExecutionModeX::getValidDefaultExecutionMode(tensor);
    }
    const std::vector<ExecutionMode> getValidTilingExecutionMode(const DPULayer& wl) const override {
        return TilingExecutionModeX::getValidTilingExecutionMode(wl);
    }
    const std::vector<VPUTilingStrategy> getValidTilingStrategies() const override {
        return TilingStrategiesX::getValidTilingStrategies();
    }

    // LayerProperties_All_Devices() = default;
};

/// @brief Device-specific layer property implementation for VPU 2.0/2.1.
///
/// This template class provides the layer property interface for VPU 2.0 and VPU 2.1 devices.
/// It implements custom logic for execution mode selection and default execution mode, and
/// inherits tiling strategies from the provided template parameter.
///
/// @tparam TilingStrategies The tiling strategies provider.
template <class TilingStrategiesX>
class LayerProperties_Device20 : public ILayerProperties, public TilingStrategiesX {
public:
    /// @brief Get the valid ExecutionMode for VPU_2_0
    ///
    /// @param wl the DPULayer
    /// @return std::vector<ExecutionMode>
    const std::vector<ExecutionMode> getValidTilingExecutionMode(const DPULayer& wl) const override {
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

    /// @brief Returns the default execution mode for a given tensor.
    ///
    /// For float tensors, returns VECTOR_FP16. Otherwise, returns MATRIX.
    ///
    /// @param tensor the VPUTensor
    /// @return the default ExecutionMode
    ExecutionMode getValidDefaultExecutionMode(const VPUTensor& tensor) const override {
        if (tensor.is_any_float()) {
            return ExecutionMode::VECTOR_FP16;
        } else {
            return ExecutionMode::MATRIX;
        }
    }

    /// @brief Returns the valid tiling strategies for a certain device
    /// @return A vector of supported VPUTilingStrategy values
    const std::vector<VPUTilingStrategy> getValidTilingStrategies() const override {
        return TilingStrategiesX::getValidTilingStrategies();
    }

    // LayerProperties_Device20() = default;
};

}  // namespace VPUNN

#endif
