// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef SHAVE_COLLECTION_NPU40_H
#define SHAVE_COLLECTION_NPU40_H

// #include "shave_collection.h"

#include "vpu/types.h"

#include "shave_collection_VPU27.h"
#include "shave_collection_mock.h"

namespace VPUNN {

using ShaveInstanceHolder_Mock_NPU40_BASE = ShaveInstanceHolder_Mock<ShaveInstanceHolder_VPU27, VPUDevice::VPU_4_0>;
using ShaveInstanceHolder_NPU40CLassic = ShaveInstanceHolder_VPU27CLassic; 
class ShaveInstanceHolder_Mock_NPU40 : public ShaveInstanceHolder_Mock_NPU40_BASE {
public:
    ShaveInstanceHolder_Mock_NPU40(): ShaveInstanceHolder_Mock_NPU40_BASE(1.0f) {
        // populate();  // done in base
    }
};

class ShaveInstanceHolder_NPU40 : public DeviceShaveContainer {
public:
    using DeviceShaveContainer::getDevice;
    const DeviceShaveContainer& getContainer() const {
        return *this;
    }

    ShaveInstanceHolder_NPU40(): DeviceShaveContainer(VPUDevice::VPU_4_0) {
        populate();
    }

    void populate();  ///< to be implemented automatically in a .cpp file
};


}  // namespace VPUNN
#endif
