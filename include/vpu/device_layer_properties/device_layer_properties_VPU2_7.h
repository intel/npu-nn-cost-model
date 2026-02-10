// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef LAYER_PROPERTIES_VPU2_7_H
#define LAYER_PROPERTIES_VPU2_7_H

#include "device_layer_properties_base.h"
#include "interface_device_layer_properties.h"
#include "vpu/layer.h"
#include "vpu/types.h"

namespace VPUNN {

/// @brief data holder for VPU 2.7 layer properties.
///
/// This class encapsulates device-specific static configuration data for VPU 2.7.
/// Its main purpose is to provide a central definition of valid tiling strategies,
/// operation-to-execution-mode mappings, and the default execution mode for this device.
///
/// @note
/// - This class is not intended for direct use by client code.
/// - It is used as a template parameter for higher-level property classes (such as LayerProperties_All_Devices)
///   that implement the actual device-specific logic and interface for layer property queries.
/// - The scope of this class is internal to the layer properties implementation for VPU 2.7.
struct VPU2_7_LayerPropertiesData {
    inline static const std::vector<VPUTilingStrategy> valid_tiling_strategies{
            VPUTilingStrategy::NONE, VPUTilingStrategy::SOH_Overlapped,
            VPUTilingStrategy::SOK};  ///< list of valid tiling strategies, if here doesn't mean all of them are
                                      ///< implemented, it just means they are valid for the device

    inline static const std::unordered_map<Operation, std::vector<ExecutionMode>> op_to_exec_mode{
            {Operation::CM_CONVOLUTION, {ExecutionMode::CUBOID_16x16}},
            {Operation::DW_CONVOLUTION, {ExecutionMode::CUBOID_16x16}},
            {Operation::AVEPOOL, {ExecutionMode::CUBOID_16x16}},
            {Operation::MAXPOOL, {ExecutionMode::CUBOID_16x16}},
            {Operation::ELTWISE, {ExecutionMode::CUBOID_8x16}},
            {Operation::CONVOLUTION,
             {ExecutionMode::CUBOID_16x16, ExecutionMode::CUBOID_8x16, ExecutionMode::CUBOID_4x16}},
    };  ///< map of operation to valid execution modes

    inline static const ExecutionMode default_execution_mode{
            ExecutionMode::CUBOID_16x16};  ///< default execution mode for layers
};

/// used by LayerPropertiesHolder to get the proper layer properties for a certain device
using VPU2_7_LayerProperties = LayerProperties_All_Devices<TilingExecutionMode<VPU2_7_LayerPropertiesData>,
                                                           DefaultExecutionMode<VPU2_7_LayerPropertiesData>,
                                                           TilingStrategies<VPU2_7_LayerPropertiesData>>;

}  // namespace VPUNN

#endif
