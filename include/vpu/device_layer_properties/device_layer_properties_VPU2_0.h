// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef LAYER_PROPERTIES_VPU2_0_H
#define LAYER_PROPERTIES_VPU2_0_H

#include "device_layer_properties_base.h"
#include "interface_device_layer_properties.h"
#include "vpu/layer.h"
#include "vpu/types.h"

namespace VPUNN {

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
        // FLOAT32/INT32 not expected here: not valid in/out datatypes for VPU_2_0/2_1.
        if (DTypeCategoryInfo::is_fp16family_dtype(wl.inputs[0].get_dtype()) ||
            DTypeCategoryInfo::is_fp16family_dtype(wl.outputs[0].get_dtype()))
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
        if (DTypeCategoryInfo::is_any_float_dtype(tensor.get_dtype())) {
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

/// @brief data holder for VPU 2.0/2.1 layer properties
///
/// This class encapsulates device-specific static configuration data for VPU 2.0 and VPU 2.1.
/// Its main purpose is to provide a central definition of valid tiling strategies for these devices.
///
/// @note
/// - This class is not intended for direct use by client code.
/// - It is used as a template parameter for higher-level property classes (such as LayerProperties_Device20)
///   that implement the actual device-specific logic and interface for layer property queries.
/// - The scope of this class is internal to the layer properties implementation for VPU 2.0/2.1.
struct VPU2_0_LayerPropertiesData {
    inline static const std::vector<VPUTilingStrategy> valid_tiling_strategies{
            VPUTilingStrategy::NONE, VPUTilingStrategy::SOH_Overlapped,
            VPUTilingStrategy::SOK};  ///< list of valid tiling strategies, if here doesn't mean all of them are
                                      ///< implemented, it just means they are valid for the device
};

/// Used by LayerPropertiesHolder to get the proper layer properties for a certain device
using VPU2_0_LayerProperties = LayerProperties_Device20<TilingStrategies<VPU2_0_LayerPropertiesData>>;

}  // namespace VPUNN
#endif
