// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#ifndef VPUNN_INTERFACE_HW_CHARACTERISTICS_SET_H
#define VPUNN_INTERFACE_HW_CHARACTERISTICS_SET_H

#include "vpu/dpu_types.h"
#include "vpu/hw_characteristics/itf_device_HW_characteristics.h"

namespace VPUNN {

/// interface that holds a complete set of device HW performance providers
class IHWCharacteristicsSet {
public:
    virtual const IDeviceHWCharacteristics& device(VPUDevice device) const = 0;

protected:
    virtual ~IHWCharacteristicsSet() = default;  ///< virtual destructor for proper cleanup
};

}  // namespace VPUNN

#endif  //
