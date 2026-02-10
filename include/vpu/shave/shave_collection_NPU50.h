// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef SHAVE_COLLECTION_NPU50_H
#define SHAVE_COLLECTION_NPU50_H

// #include "shave_collection.h"

#include "vpu/types.h"

#include "shave_collection_NPU40.h"
#include "shave_collection_mock.h"
#include "shave_instance_holder_factors.h"

#include "vpu/performance.h"

/**
 * This is a generated file that contains the population of the factors map for NPU5.
 * This path of the generated file and the path for input csv is given in `src/shave/CMakeLists.txt`.
 * So if you want to change the factors for NPU5, you have to change the corresponding csv file from CMakeLists and re-generate.
 */
#include "vpu/shave/generated_shave_factors_population_npu5.h"

namespace VPUNN {

// using ShaveInstanceHolder_Mock_NPU50_BASE = ShaveInstanceHolder_Mock<ShaveInstanceHolder_VPU27, VPUDevice::NPU_5_0>;
using ShaveInstanceHolder_Mock_NPU50_BASE =
        ShaveInstanceHolder_Mock<ShaveInstanceHolder_NPU40, VPUDevice::NPU_5_0>;
using ShaveInstanceHolder_NPU50CLassic = ShaveInstanceHolder_VPU27CLassic; 

class ShaveInstanceHolder_Mock_NPU50 : public ShaveInstanceHolder_Mock_NPU50_BASE {
public:
    ShaveInstanceHolder_Mock_NPU50(): ShaveInstanceHolder_Mock_NPU50_BASE(1.0f) {
        // populate();  // done in base
    }
};
class ShaveInstanceHolder_NPU50 : public DeviceShaveContainer {
public:
    using DeviceShaveContainer::getDevice;
    const DeviceShaveContainer& getContainer() const {
        return *this;
    }

    ShaveInstanceHolder_NPU50(): DeviceShaveContainer(VPUDevice::NPU_5_0) {
        populate();
    }

    void populate();  ///< to be implemented automatically in a .cpp file
};

using ShaveInstanceHolder_NPU50_WithFactors =
            ShaveInstanceHolderWithFactors<PopulatedFactorsLUT_NPU5, ShaveInstanceHolder_Mock_NPU50, VPUDevice::NPU_5_0>;
}  // namespace VPUNN
#endif
