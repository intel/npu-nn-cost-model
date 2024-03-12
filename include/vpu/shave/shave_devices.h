// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef SHAVE_DEVICES_H
#define SHAVE_DEVICES_H

#include <map>
#include <memory>
#include <type_traits>

#include "vpu/cycles_interface_types.h"
#include "vpu/types.h"

#include "interface_shave_op_executor.h"
#include "shave_collection.h"
#include "shave_op_executors.h"

namespace VPUNN {
/// @brief selects  in a trivial manner, from one container
class ShaveSelector {
private:
    VPUDevice device_supported;
    const DeviceShaveContainer& container;

public:
    ShaveSelector(VPUDevice device, const DeviceShaveContainer& shave_container)
            : device_supported(device), container(shave_container) {
    }
    bool isDeviceSupported(VPUDevice device) const {
        return device_supported == device;
    }
    virtual const ShaveOpExecutor& getShaveFuntion(const std::string& name) const {
        // if(container.existsShave(name)) {
        return container.getShaveExecutor(name);  // will throw if not existing
        // }
    }
    virtual std::vector<std::string> getShaveList() const {
        return container.getShaveList();
    }

protected:
    bool existsShave(const std::string& name) const {
        return container.existsShave(name);
    }
};

/// @brief selects from 2 containers , 1st with priority
class ShavePrioritySelector : public ShaveSelector {
    const DeviceShaveContainer& container2;

public:
    ShavePrioritySelector(VPUDevice device, const DeviceShaveContainer& shave_container_first,
                          const DeviceShaveContainer& shave_container_second)
            : ShaveSelector(device, shave_container_first), container2(shave_container_second) {
    }
    const ShaveOpExecutor& getShaveFuntion(const std::string& name) const override {
        if (ShaveSelector::existsShave(name)) {
            return ShaveSelector::getShaveFuntion(name);  // should not throw
        } else {
            return container2.getShaveExecutor(name);  // will throw if not existing
        }
    }
    virtual std::vector<std::string> getShaveList() const override {
        auto v1{ShaveSelector::getShaveList()};
        const auto v2{container2.getShaveList()};
        v1.insert(v1.end(), v2.begin(), v2.end());

        return v1;
    }
};

/// @brief the shave configuration. Holds instances
class ShaveConfiguration {
private:
    // instances of collections
    const ShaveInstanceHolder_VPU27CLassic shaves_20_classic{};

    const ShaveInstanceHolder_VPU27 shaves_27_new{};  ///< new list
    const ShaveInstanceHolder_VPU27CLassic shaves_27_classic{};

    const ShaveInstanceHolder_VPU27 shaves_40{};

    // selectors know on what collections to look (they are properly configured for each device)
    const ShaveSelector selector_20{VPUDevice::VPU_2_0, shaves_20_classic};
    const ShavePrioritySelector selector_27{VPUDevice::VPU_2_7, shaves_27_new,
                                            shaves_27_classic};  ///< special with 2 lists
    const ShaveSelector selector_40{VPUDevice::VPU_RESERVED, shaves_40};

    const ShaveSelector& getSelector(VPUDevice desired_device) const {
        switch (desired_device) {
        case VPUDevice::VPU_2_0:
        case VPUDevice::VPU_2_1:
            return selector_20;
            break;
        case VPUNN::VPUDevice::VPU_2_7:
            return selector_27;
            break;
        case VPUDevice::VPU_RESERVED:
            return selector_40;
            break;
        default:
            static const DeviceShaveContainer empty_shaves{VPUDevice::__size};
            static const ShaveSelector empty_selector{VPUDevice::VPU_2_0, empty_shaves};
            return empty_selector;
            break;
        }
    }

public:
    CyclesInterfaceType computeCycles(const SHAVEWorkload& swl, std::string& infoOut) const {
        // finds func inmpl, executes it, handles errors
        try {
            const auto& sel = getSelector(swl.get_device());

            const auto& shaveInstance = sel.getShaveFuntion(swl.get_name());  // may throw
            const auto cycles = shaveInstance.dpuCycles(swl);
            return cycles;
        } catch (const std::exception& e) {
            std::stringstream buffer;
            buffer << "[EXCEPTION]:could not resolve shave function: " << swl << "\n "
                   << " Original exception: " << e.what() << " File: " << __FILE__ << " Line: " << __LINE__;
            const std::string details = buffer.str();
            infoOut = details;
            return Cycles::ERROR_SHAVE;
        }

        // return Cycles::ERROR_SHAVE;
    }

    std::vector<std::string> getShaveSupportedOperations(VPUDevice device) const {
        const auto& sel = getSelector(device);
        const auto list = sel.getShaveList();
        return list;
    }
};
}  // namespace VPUNN
#endif