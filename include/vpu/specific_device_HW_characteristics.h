// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_SPECIFIC_DEVICE_HW_CHARACTERISTICS_H
#define VPUNN_SPECIFIC_DEVICE_HW_CHARACTERISTICS_H

#include <stdexcept>
#include <tuple>
#include <variant>

#include "device_HW_characteristics.h"
#include "device_HW_characteristics_VPU2.h"
#include "device_HW_characteristics_VPU2_1.h"
#include "device_HW_characteristics_VPU2_7.h"
#include "device_HW_characteristics_VPU4.h"

namespace VPUNN {
constexpr auto characteristics = std::make_tuple(
        VPU2_0_HWCharacteristics{}, VPU2_1_HWCharacteristics{}, VPU2_7_HWCharacteristics{},
        VPU2_7_HWCharacteristics_legacy{}, VPU4_0_HWCharacteristics{}, VPU4_0_HWCharacteristics_legacy{}, Default_HWCharacteristics{});

using Characteristics =
        std::variant<VPU2_0_HWCharacteristics, VPU2_1_HWCharacteristics, VPU2_7_HWCharacteristics,
                     VPU2_7_HWCharacteristics_legacy, VPU4_0_HWCharacteristics, VPU4_0_HWCharacteristics_legacy, Default_HWCharacteristics>;

/// get specific HW characteristics
constexpr Characteristics get_HWCharacteristics(VPUDevice device) {
    switch (device) {
    case VPUDevice::VPU_2_0:
        return std::get<VPU2_0_HWCharacteristics>(characteristics);
    case VPUDevice::VPU_2_1:
        return std::get<VPU2_1_HWCharacteristics>(characteristics);
    case VPUDevice::VPU_2_7:
        return std::get<VPU2_7_HWCharacteristics>(characteristics);
    case VPUDevice::VPU_4_0:
        return std::get<VPU4_0_HWCharacteristics>(characteristics);
    default:
        return std::get<Default_HWCharacteristics>(characteristics);
    }
}

/// get specific HW characteristics for legacy mode
constexpr Characteristics get_HWCharacteristics_Legacy(VPUDevice device) {
    switch (device) {
    case VPUDevice::VPU_2_0:
        return std::get<VPU2_0_HWCharacteristics>(characteristics);
    case VPUDevice::VPU_2_1:
        return std::get<VPU2_1_HWCharacteristics>(characteristics);
    case VPUDevice::VPU_2_7:
            return std::get<VPU2_7_HWCharacteristics_legacy>(characteristics);
    case VPUDevice::VPU_4_0:
            return std::get<VPU4_0_HWCharacteristics_legacy>(characteristics);
    default:
        return std::get<Default_HWCharacteristics>(characteristics);
    }
}

}  // namespace VPUNN

#endif  // VPUNN_SPECIFIC_DEVICE_HW_CHARACTERISTICS_H
