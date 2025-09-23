// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_DEVICE_HW_CHARACTERISTICS_CONST_REPO_H
#define VPUNN_DEVICE_HW_CHARACTERISTICS_CONST_REPO_H

#include <stdexcept>
#include <tuple>
#include <variant>

#include "vpu/hw_characteristics/device_HW_characteristics_VPU2.h"
#include "vpu/hw_characteristics/device_HW_characteristics_VPU2_1.h"
#include "vpu/hw_characteristics/device_HW_characteristics_VPU2_7.h"
#include "vpu/hw_characteristics/device_HW_characteristics_VPU4.h"
#include "vpu/hw_characteristics/device_HW_characteristics_base.h"
#include "vpu/hw_characteristics/device_HW_characteristics_default.h"

#include "vpu/hw_characteristics/device_HW_characterisics_itf_impl.h"

namespace VPUNN {

// this type is very specialized and to be used only to visit it for constexpr
using DeviceHWCharacteristicsVariant =
        std::variant<VPU2_0_HWCharacteristics, VPU2_1_HWCharacteristics, VPU2_7_HWCharacteristics,
                     VPU2_7_HWCharacteristics_legacy,  // pre VPU4
                     VPU4_0_HWCharacteristics_v0, VPU4_0_HWCharacteristics_v1, VPU4_0_HWCharacteristics_legacy,  //
                     Default_HWCharacteristics>;

class DeviceHWCHaracteristicsConstRepo {
public:
    /// get device specific HW characteristics, active by default
    static constexpr DeviceHWCharacteristicsVariant get_HWCharacteristics(VPUDevice device) {
        return get_HWCharacteristicsEvo0(device);
    }

private:
    // Tuple type for the main set of HW characteristics, with specific device configurations
    using MainEvo0SetTupleHWCharacteristics = std::tuple<VPU2_0_HWCharacteristics,     //
                                                         VPU2_1_HWCharacteristics,     //
                                                         VPU2_7_HWCharacteristics,     //
                                                         VPU4_0_HWCharacteristics_v0,  // classic
                                                         Default_HWCharacteristics>;

    using MainEvo1SetTupleHWCharacteristics = std::tuple<VPU2_0_HWCharacteristics,     //
                                                         VPU2_1_HWCharacteristics,     //
                                                         VPU2_7_HWCharacteristics,     //
                                                         VPU4_0_HWCharacteristics_v1,  // DMA update1
                                                         Default_HWCharacteristics>;

    // Legacy configuration tuple type
    using LegacySetTupleHWCharacteristic = std::tuple<VPU2_0_HWCharacteristics,         //
                                                      VPU2_1_HWCharacteristics,         //
                                                      VPU2_7_HWCharacteristics_legacy,  //
                                                      VPU4_0_HWCharacteristics_legacy,  //
                                                      Default_HWCharacteristics>;

    static_assert(std::tuple_size_v<MainEvo0SetTupleHWCharacteristics> ==
                  std::tuple_size_v<MainEvo1SetTupleHWCharacteristics>);  // Ensure both tuples have the same size
    static_assert(std::tuple_size_v<MainEvo0SetTupleHWCharacteristics> ==
                  std::tuple_size_v<LegacySetTupleHWCharacteristic>);  // Ensure both tuples have the same size

    // Helper to transform a tuple of types to a tuple of ALT_VPUXX_HWCharacteristics types
    template <typename Tuple>
    struct TransformToAlt;

    template <typename... Ts>
    struct TransformToAlt<std::tuple<Ts...>> {
        using type = std::tuple<ALT_VPUXX_HWCharacteristics<Ts>...>;
    };

public:
    using MainEvo0SetTuple = typename TransformToAlt<MainEvo0SetTupleHWCharacteristics>::type;
    using MainEvo1SetTuple = typename TransformToAlt<MainEvo1SetTupleHWCharacteristics>::type;

    // Legacy configuration tuple type
    using LegacySetTuple = typename TransformToAlt<LegacySetTupleHWCharacteristic>::type;

private:
public:
    // helper class for mapping Devices to indices in tuple and extracting content from the tuple based on device
    /// This class provides a mapping from VPUDevice enum values to indices in the tuple of HW characteristics.
    /// assumes a fixed order of the Devices  in the Tuple
    class IndexMap {
    public:
        template <VPUDevice device>
        static constexpr std::size_t get_device_index() {
            std::size_t index{0};
            if constexpr (device == VPUDevice::VPU_2_0) {
                return index;
            }
            index++;  // 1
            if constexpr (device == VPUDevice::VPU_2_1) {
                return index;
            }
            index++;  // 2
            if constexpr (device == VPUDevice::VPU_2_7) {
                return index;
            }
            index++;  // 3
            if constexpr (device == VPUDevice::VPU_4_0) {
                return index;
            }
            index++;  // 4

            return index;  // Default
        }

        template <typename ReturnType, typename TupleType>
        static constexpr ReturnType extract_tuple_content(VPUDevice device, const TupleType& theTuple) {
            switch (device) {
            case VPUDevice::VPU_2_0:
                return std::get<get_device_index<VPUDevice::VPU_2_0>()>(theTuple);
            case VPUDevice::VPU_2_1:
                return std::get<get_device_index<VPUDevice::VPU_2_1>()>(theTuple);
            case VPUDevice::VPU_2_7:
                return std::get<get_device_index<VPUDevice::VPU_2_7>()>(theTuple);
            case VPUDevice::VPU_4_0:
                return std::get<get_device_index<VPUDevice::VPU_4_0>()>(theTuple);
            default:
                return std::get<get_device_index<VPUDevice::__size>()>(theTuple);
            }

            static_assert(std::tuple_size_v<MainEvo0SetTupleHWCharacteristics> ==
                          get_device_index<VPUDevice::__size>() + 1);  // Ensure indices are aligned with the Tuple
        }
    };

    static constexpr DeviceHWCharacteristicsVariant get_HWCharacteristicsEvo0(VPUDevice device) {
        constexpr MainEvo0SetTupleHWCharacteristics const_HW_characteristics_{};
        return IndexMap::extract_tuple_content<DeviceHWCharacteristicsVariant>(device, const_HW_characteristics_);
    }
    static constexpr DeviceHWCharacteristicsVariant get_HWCharacteristicsEvo1(VPUDevice device) {
        constexpr MainEvo1SetTupleHWCharacteristics const_HW_characteristics_{};
        return IndexMap::extract_tuple_content<DeviceHWCharacteristicsVariant>(device, const_HW_characteristics_);
    }

    /// get device specific HW characteristics for legacy mode
    static constexpr DeviceHWCharacteristicsVariant get_HWCharacteristics_Legacy(VPUDevice device) {
        constexpr LegacySetTupleHWCharacteristic const_HW_characteristics_{};

        return IndexMap::extract_tuple_content<DeviceHWCharacteristicsVariant>(device, const_HW_characteristics_);
    }
};
}  // namespace VPUNN

#endif  //
