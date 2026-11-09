// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_SHAVE_COLLECTION_NPU60_H
#define VPUNN_SHAVE_COLLECTION_NPU60_H

// #include "shave_collection.h"

#include "vpu/types.h"

#include "shave_collection_NPU50.h"
#include "shave_collection_mock.h"

#include "vpu/performance.h"

namespace VPUNN {

using ShaveInstanceHolder_Mock_NPU60_BASE =
        ShaveInstanceHolder_Mock<ShaveInstanceHolder_Mock_NPU50, VPUDevice::NPU_6_0>;
using ShaveInstanceHolder_NPU60CLassic = ShaveInstanceHolder_VPU27CLassic;
using ShaveInstanceHolder_HeuristicNPU60_MockBASE =
        ShaveInstanceHolder_Mock<ShaveInstanceHolder_HeuristicNPU50, VPUDevice::NPU_6_0>;

class ShaveInstanceHolder_HeuristicNPU60_Mock : public ShaveInstanceHolder_HeuristicNPU60_MockBASE {
public:
    ShaveInstanceHolder_HeuristicNPU60_Mock(): ShaveInstanceHolder_HeuristicNPU60_MockBASE(1.0f) {
        // populate();  // done in base
    }
    ~ShaveInstanceHolder_HeuristicNPU60_Mock() = default;
};

class ShaveInstanceHolder_Mock_NPU60 : public ShaveInstanceHolder_Mock_NPU60_BASE {
public:
    ShaveInstanceHolder_Mock_NPU60(): ShaveInstanceHolder_Mock_NPU60_BASE(1.0f) {
        // populate();  // done in base
    }
    ~ShaveInstanceHolder_Mock_NPU60() = default;
};

class ShaveInstanceHolder_NPU60 : public DeviceShaveContainer {
public:
    using DeviceShaveContainer::getDevice;
    const DeviceShaveContainer& getContainer() const {
        return *this;
    }

    ShaveInstanceHolder_NPU60(): DeviceShaveContainer(VPUDevice::NPU_6_0) {
        populate();
    }

    void populate();  ///< to be implemented automatically in a .cpp file
    ~ShaveInstanceHolder_NPU60() = default;
};

class ShaveInstanceHolder_HeuristicNPU60 : public DeviceShaveContainer {
public:
    using DeviceShaveContainer::getDevice;
    const DeviceShaveContainer& getContainer() const {
        return *this;
    }
    ShaveInstanceHolder_HeuristicNPU60() : DeviceShaveContainer(VPUDevice::NPU_6_0) {
        populate();
    }

    void populate();  ///< to be implemented automatically in a .cpp file
    ~ShaveInstanceHolder_HeuristicNPU60() = default;
};

}  // namespace VPUNN
#endif  // VPUNN_SHAVE_COLLECTION_NPU60_H
