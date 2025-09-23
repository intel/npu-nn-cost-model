// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef TUPLE_INDEXING_HELPER_H
#define TUPLE_INDEXING_HELPER_H

#include <tuple>
#include "vpu/dpu_types.h"

namespace VPUNN {
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
    static ReturnType extract_tuple_content(VPUDevice device, const TupleType& theTuple) {
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
    }
};
}  // namespace VPUNN
#endif
