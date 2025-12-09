// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef LAYER_PROPERTIES_DEFAULT_H
#define LAYER_PROPERTIES_DEFAULT_H

#include "interface_device_layer_properties.h"
#include "vpu/layer.h"
#include "vpu/types.h"

namespace VPUNN {

/// @brief Internal static data holder for default (fallback) layer properties.
/// 
/// This struct encapsulates static configuration data used as a fallback when no device-specific
/// layer properties are available. It provides a central definition of valid tiling strategies,
/// an (empty) operation-to-execution-mode mapping, and the default execution mode for unsupported or unknown devices.
/// 
/// @note
/// - This struct is not intended for direct use by client code.
/// - It is used as a template parameter for higher-level property classes (such as LayerProperties_All_Devices)
///   that implement the actual logic and interface for layer property queries.
/// - The scope of this struct is internal to the layer properties implementation for default/fallback cases.
struct Default_LayerPropertiesData {
    inline static const std::vector<VPUTilingStrategy> valid_tiling_strategies{};  ///< list of valid tiling strategies

    inline static const std::unordered_map<Operation, std::vector<ExecutionMode>> op_to_exec_mode{};  ///< map of operation to valid execution modes

    inline static const ExecutionMode default_execution_mode{
            ExecutionMode::CUBOID_16x16};  ///< default execution mode for layers
};

/// used by LayerPropertiesHolder to get the proper layer properties for a certain device
using  Default_LayerProperties = LayerProperties_All_Devices<TilingExecutionMode< Default_LayerPropertiesData>,
                                                            DefaultExecutionMode<Default_LayerPropertiesData>,
                                                            TilingStrategies<Default_LayerPropertiesData>>;
}  // namespace VPUNN
#endif
