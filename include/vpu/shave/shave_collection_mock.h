// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef SHAVE_COLLECTION_MOCK_H
#define SHAVE_COLLECTION_MOCK_H

#include "shave_collection.h"

#include "vpu/cycles_interface_types.h"
#include "vpu/types.h"

// #include "shave_collection_VPU27.h"
#include "vpu/performance.h"

namespace VPUNN {
template <typename InstanceHolderToMock, VPUDevice deviceGen>
class ShaveInstanceHolder_Mock : public DeviceShaveContainer {
public:
    using DeviceShaveContainer::getDevice;
    const DeviceShaveContainer& getContainer() const {
        return *this;
    }

    ShaveInstanceHolder_Mock(float speed_increase)
            : DeviceShaveContainer(deviceGen), speed_increase_factor{speed_increase} {
        populate();  // fill into the base container
    }

protected:
    InstanceHolderToMock toMock;  ///< what collection we want to reproduce for another device
    const float speed_increase_factor;

    void populate() {
        // constexpr float speed_increase_factor{1.0f};
        constexpr VPUDevice wanted_device{deviceGen};
        // for all in  source, do a add
        const DeviceShaveContainer& mockable{toMock.getContainer()};
        const auto names{mockable.getShaveList()};
        for (const auto& n : names) {
            const auto& executor{mockable.getShaveExecutor(n)};
            // constexpr
            AddMock<get_dpu_fclk(wanted_device), get_cmx_fclk(wanted_device)>(executor.getName(), executor,
                                                                              speed_increase_factor);
        }

    }  ///< to be implemented automatically
};

}  // namespace VPUNN
#endif
