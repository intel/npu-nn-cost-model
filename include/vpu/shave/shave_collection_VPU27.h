// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef SHAVE_COLLECTION_VPU27_H
#define SHAVE_COLLECTION_VPU27_H

#include "shave_collection.h"

#include "vpu/cycles_interface_types.h"
#include "vpu/types.h"

namespace VPUNN {

class ShaveInstanceHolder_VPU27 : public DeviceShaveContainer {
public:
    using DeviceShaveContainer::getDevice;
    const DeviceShaveContainer& getContainer() const {
        return *this;
    }

    ShaveInstanceHolder_VPU27(): DeviceShaveContainer(VPUDevice::VPU_2_7) {
        populate();
    }

    void populate();  ///< to be implemented automatically in a .cpp file 
};

class ShaveInstanceHolder_VPU27CLassic : public DeviceShaveContainer {
public:
    using DeviceShaveContainer::getDevice;
    const DeviceShaveContainer& getContainer() const {
        return *this;
    }

    ShaveInstanceHolder_VPU27CLassic(): DeviceShaveContainer(VPUDevice::VPU_2_7) {
        populate();
    }

    void populate();  ///< to be implemented automatically
};

}  // namespace VPUNN
#endif
