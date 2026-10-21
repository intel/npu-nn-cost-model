// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/shave/shave_devices.h"

#include "vpu/shave/shave_collection_NPU40.h"
#include "vpu/shave/shave_collection_NPU50.h"
#include "vpu/shave/shave_collection_VPU27.h"

namespace VPUNN {

std::unique_ptr<ShaveSelector> ShaveConfiguration::createSelectorForDevice(VPUDevice device) const {
    switch (device) {
    case VPUDevice::VPU_2_0:
    case VPUDevice::VPU_2_1: {
        auto holder = std::make_unique<ShaveInstanceHolder_VPU27CLassic>();
        return std::make_unique<ShaveSelector>(std::move(holder));
    }
    case VPUDevice::VPU_2_7: {
        auto holder = std::make_unique<ShaveInstanceHolder_VPU27>();
        return std::make_unique<ShaveSelector>(std::move(holder));
    }
    case VPUDevice::VPU_4_0: {
        auto holder = std::make_unique<ShaveInstanceHolder_NPU40>();
        return std::make_unique<ShaveSelector>(std::move(holder));
    }
    case VPUDevice::NPU_5_0: {
        auto holder = std::make_unique<ShaveInstanceHolder_Mock_NPU50>();
        return std::make_unique<ShaveSelector>(std::move(holder));
    }
    case VPUDevice::NPU_5_0_W: {
        auto holder = std::make_unique<ShaveInstanceHolder_Mock_NPU50W>();
        return std::make_unique<ShaveSelector>(std::move(holder));
    }
    default: {
        auto holder = std::make_unique<DeviceShaveContainer>(VPUDevice::__size);
        return std::make_unique<ShaveSelector>(std::move(holder));
    }
    }
}

std::unique_ptr<ShaveSelector> ShaveConfiguration::createOldSelectorForDevice(VPUDevice device) const {
    switch (device) {
    case VPUDevice::VPU_2_0:
    case VPUDevice::VPU_2_1: {
        auto holder = std::make_unique<ShaveInstanceHolder_VPU27CLassic>();
        return std::make_unique<ShaveSelector>(std::move(holder));
    }
    case VPUDevice::VPU_2_7: {
        auto holder = std::make_unique<ShaveInstanceHolder_VPU27CLassic>();
        return std::make_unique<ShaveSelector>(std::move(holder));
    }
    case VPUDevice::VPU_4_0: {
        auto holder = std::make_unique<ShaveInstanceHolder_NPU40CLassic>();
        return std::make_unique<ShaveSelector>(std::move(holder));
    }
    case VPUDevice::NPU_5_0: {
        auto holder = std::make_unique<ShaveInstanceHolder_NPU50CLassic>();
        return std::make_unique<ShaveSelector>(std::move(holder));
    }
    case VPUDevice::NPU_5_0_W: {
        auto holder = std::make_unique<ShaveInstanceHolder_NPU50CLassic>();
        return std::make_unique<ShaveSelector>(std::move(holder));
    }
    default: {
        auto holder = std::make_unique<DeviceShaveContainer>(VPUDevice::__size);
        return std::make_unique<ShaveSelector>(std::move(holder));
    }
    }
}

std::unique_ptr<ShaveSelector> ShaveConfiguration::createSelectorWithFactorsForDevice(VPUDevice device) const {
    switch (device) {
    case VPUDevice::NPU_5_0: {
        auto holder = std::make_unique<ShaveInstanceHolder_NPU50_WithFactors>();
        return std::make_unique<ShaveSelector>(std::move(holder));
    }
    default:
        // if there is no factors map for the device, return the regular selector
        return createSelectorForDevice(device);
    }
}

std::unique_ptr<ShaveSelector> ShaveConfiguration::createHeuristicSelectorForDevice(VPUDevice device) const {
    switch (device) {
    case VPUDevice::NPU_5_0: {
        auto holder = std::make_unique<ShaveInstanceHolder_HeuristicNPU50>();
        return std::make_unique<ShaveSelector>(std::move(holder));
    }
    case VPUDevice::NPU_5_0_W:
        // for compatibility reasons, we need to use old selector for WCL
        return createOldSelectorForDevice(device);
    default:
        // if there is no heuristic selector for the device, return the old selector
        return createOldSelectorForDevice(device);
    }
}

std::unique_ptr<ShaveSelector> ShaveConfiguration::createHeuristicSelectorWithFactorsForDevice(VPUDevice device) const {
    switch (device) {
    case VPUDevice::NPU_5_0: {
        auto holder = std::make_unique<ShaveInstanceHolder_HeuristicNPU50_WithFactors>();
        return std::make_unique<ShaveSelector>(std::move(holder));
    }
    default:
        // if there is no heuristic selector with factors for the device, return the heuristic selector
        return createHeuristicSelectorForDevice(device);
    }
}

}  // namespace VPUNN
