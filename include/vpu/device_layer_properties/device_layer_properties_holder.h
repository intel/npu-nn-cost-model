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



#include "device_layer_properties_default.h"
#include "interface_device_layer_properties.h"
#include "vpu/tuple_indexing_helper.h"

namespace VPUNN {

class LayerPropertiesHolder {
protected:
    using LayerProperties_Tuple = std::tuple<VPU2_0_LayerProperties,  //
                                             VPU2_0_LayerProperties,  //
                                             VPU2_7_LayerProperties,  //
                                             VPU4_0_LayerProperties,  //
                                             Default_LayerProperties>;

    static inline const LayerProperties_Tuple const_layer_properties{};

public:
    /**
     * @brief Retrieves the layer properties for a given VPU device.
     *
     * @param device The VPU device for which to retrieve properties.
     * @return Reference to the corresponding ILayerProperties implementation.
     *         If the device is not supported, returns a default implementation and logs an error
     */
    static const ILayerProperties& get_properties(VPUDevice device) {
        return IndexMap::extract_tuple_content<const ILayerProperties&, LayerProperties_Tuple>(device,
                                                                                               const_layer_properties);
    }
};
}  // namespace VPUNN
#endif
