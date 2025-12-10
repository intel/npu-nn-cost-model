// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef DEVICE_LAYER_PROPERTIES_HOLDER_H
#define DEVICE_LAYER_PROPERTIES_HOLDER_H

#include "device_layer_properties_VPU2_0.h"
#include "device_layer_properties_VPU2_7.h"
#include "device_layer_properties_VPU4_0.h"
#include "device_layer_properties_VPU5_0.h"
#include "device_layer_properties_default.h"
#include "interface_device_layer_properties.h"
#include "vpu/tuple_indexing_helper.h"

namespace VPUNN {

/// @brief Provides access to device-specific layer properties for supported VPU devices.
/// 
/// This class offers an interface to retrieve layer property information
/// (such as valid tiling strategies, execution modes, and other device-dependent characteristics)
/// for a given VPU device. It internally manages a tuple of device-specific property implementations
/// and returns the correct properties based on the device type.
/// 
/// If the requested device is not supported, a default property implementation is returned and an error is logged.
class LayerPropertiesHolder {
protected:
    /// tuple holding instances of all device-specific layer property classes
    /// the order of types in this tuple must match the device index mapping used by IndexMap
    using LayerProperties_Tuple = std::tuple<VPU2_0_LayerProperties,  //
                                             VPU2_0_LayerProperties,  //
                                             VPU2_7_LayerProperties,  //
                                             VPU4_0_LayerProperties,  //
                                             VPU5_0_LayerProperties,  //
                                             Default_LayerProperties>;

    static inline const LayerProperties_Tuple const_layer_properties{};

public:
    /// @brief Retrieves the layer properties for a given VPU device.
    /// 
    /// @param device The VPU device for which to retrieve properties.
    /// @return Reference to the corresponding ILayerProperties implementation.
    ///         If the device is not supported, returns a default implementation and logs an error
    static const ILayerProperties& get_properties(VPUDevice device) {
        return IndexMap::extract_tuple_content<const ILayerProperties&, LayerProperties_Tuple>(device,
                                                                                               const_layer_properties);
    }
};
}  // namespace VPUNN
#endif
