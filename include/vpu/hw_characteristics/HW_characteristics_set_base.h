// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_BASE_HW_CHARACTERISTICS_SET_H
#define VPUNN_BASE_HW_CHARACTERISTICS_SET_H

#include <stdexcept>
#include <tuple>
#include <variant>

#include "vpu/hw_characteristics/itf_HW_characteristics_set.h"
#include "vpu/hw_characteristics/itf_device_HW_characteristics.h"

namespace VPUNN {

/// \brief The TupleType for Main_HWCharacteristics must contain one instance of each ALT_VPUXX_HWCharacteristics
///        (or compatible IHWPerformance implementation) for each supported device, in a fixed order.
///
/// The order of types in the tuple must match the device selection logic in Main_HWCharacteristics::get().
/// \tparam DeviceIndexMap , helper that knows the actual order of devices in the  TupleConfiguration
///
/// The default mapping is:
///   0: IAlt_VPU2_0_         (for VPUDevice::VPU_2_0)
///   1: IAlt_VPU2_1_         (for VPUDevice::VPU_2_1)
///   2: IAlt_VPU2_7_         (for VPUDevice::VPU_2_7)
///   3:............
///
///
///   last: IAlt_Default_HWCharacteristics        (for all other/unknown devices)
///
/// >
template <typename TupleConfiguration, typename DeviceIndexMap>
class Base_HWCharacteristicsSet : public IHWCharacteristicsSet {
public:
    Base_HWCharacteristicsSet(const TupleConfiguration& configs = TupleConfiguration{}): instances(configs) {
    }

    const IDeviceHWCharacteristics& device(VPUDevice device) const override {
        // The mapping below assumes the tuple order matches the device enum order.
        // Adjust the mapping logic as needed for your device/tuple order.

        return DeviceIndexMap::template extract_tuple_content<const IDeviceHWCharacteristics&, TupleConfiguration>(
                device, instances);
    }

private:
    const TupleConfiguration instances;  ///> instance , each device gas a position
};

}  // namespace VPUNN

#endif  //
