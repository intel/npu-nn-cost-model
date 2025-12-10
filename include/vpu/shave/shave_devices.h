// Copyright © 2024 Intel Corporation
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
#include <optional>
#include <type_traits>

#include "vpu/cycles_interface_types.h"
#include "vpu/types.h"

#include "core/cache.h"
#include "interface_shave_op_executor.h"
#include "shave_collection_NPU40.h"
#include "shave_collection_NPU50.h"
#include "shave_collection_VPU27.h"
#include "shave_op_executors.h"
#include "vpu/serialization/shave_cost_serialization_wrapper.h"
#include "vpu/validation/sanity_report.h"
#include "vpu/validation/shave_workloads_sanitizer.h"

namespace VPUNN {
/// @brief selects  in a trivial manner, from one container
class ShaveSelector {
private:
    const DeviceShaveContainer& container;

public:
    ShaveSelector(const DeviceShaveContainer& shave_container): container(shave_container) {
    }

    virtual ~ShaveSelector() = default;

    virtual const ShaveOpExecutor& getShaveFuntion(const std::string& name) const {
        return container.getShaveExecutor(name);  // will throw if not existing
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
    ShavePrioritySelector(const DeviceShaveContainer& shave_container_first,
                          const DeviceShaveContainer& shave_container_second)
            : ShaveSelector(shave_container_first), container2(shave_container_second) {
    }

    virtual ~ShavePrioritySelector() = default;

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

/// @brief the shave configuration. Holds instances of all configurations supported by this app version
class ShaveConfiguration {
private:
    // instances of collections
    const struct {
        const ShaveInstanceHolder_VPU27CLassic shaves_20_classic{};  ///< fixed frequencies in model
        const ShaveInstanceHolder_VPU27 shaves_27{};                 ///< new list

        const ShaveInstanceHolder_NPU40 shaves_40{};

        // mocks instances
        const ShaveInstanceHolder_Mock_NPU40 mock_shaves_40{};  // mock 2.7  as they are for 4.0

        const ShaveInstanceHolder_VPU27CLassic old_shave_27{};
        const ShaveInstanceHolder_NPU40CLassic old_shave_40{};

        const ShaveInstanceHolder_Mock_NPU50 mock_shaves_50{};
        const ShaveInstanceHolder_NPU50CLassic old_shave_50{};


    } collections{};

    // selectors know on what collections to look (they are properly configured for each device)
    const ShaveSelector selector_20{collections.shaves_20_classic};
    // const ShavePrioritySelector selector_27{shaves_27_new,
    //                                         shaves_27_classic};          ///< special with 2 lists
    const ShaveSelector selector_27{collections.shaves_27};  ///< only the new list
    const ShaveSelector selector_40{collections.shaves_40};

    const ShaveSelector selector_old_20{collections.shaves_20_classic};
    const ShaveSelector selector_old_27{collections.old_shave_27};
    const ShaveSelector selector_old_40{collections.old_shave_40};
    const ShaveSelector selector_50{collections.mock_shaves_50};
    const ShaveSelector selector_old_50{collections.old_shave_50};

public:
    const ShaveSelector& getSelector(VPUDevice desired_device) const {
        switch (desired_device) {
        case VPUDevice::VPU_2_0:
        case VPUDevice::VPU_2_1:
            return selector_20;
            break;
        case VPUDevice::VPU_2_7:
            return selector_27;
            break;
        case VPUDevice::VPU_4_0:
            return selector_40;
            break;
        case VPUDevice::NPU_5_0:
            return selector_50;
            break;
        default:
            static const DeviceShaveContainer empty_shaves{VPUDevice::__size};
            static const ShaveSelector empty_selector{empty_shaves};
            return empty_selector;
            break;
        }
    }

    const ShaveSelector& getOldSelector(VPUDevice desired_device) const {
        switch (desired_device) {
        case VPUDevice::VPU_2_0:
        case VPUDevice::VPU_2_1:
            return selector_old_20;
            break;
        case VPUDevice::VPU_2_7:
            return selector_old_27;
            break;
        case VPUDevice::VPU_4_0:
            return selector_old_40;
            break;
        case VPUDevice::NPU_5_0:
            return selector_old_50;
            break;
        default:
            static const DeviceShaveContainer empty_shaves{VPUDevice::__size};
            static const ShaveSelector empty_selector{empty_shaves};
            return empty_selector;
            break;
        }
    }

    /**
     * @brief Get the maximum number of parameters across all SHAVE functions
     * 
     * Iterates through all SHAVE functions to find the maximum number of expected parameters. 
     * This is useful for determining the appropriate buffer sizes or field counts needed for serialization.
     * 
     * @param sel Reference to ShaveSelector containing SHAVE function definitions
     * @return int Maximum number of parameters found across all SHAVE functions
     */
    int get_max_num_params() const {
        std::vector<int> all_num_params{};
        for (int i = 0; i < static_cast<int>(VPUDevice::__size); i++) {
            const auto& sel = getSelector(static_cast<VPUDevice>(i));
            const auto& shv_list = sel.getShaveList();
            for (const auto& name : shv_list) {
                all_num_params.push_back(sel.getShaveFuntion(name).getNumExpectedParams());
            }
        }

        return *std::max_element(all_num_params.begin(), all_num_params.end());
    }
};
}  // namespace VPUNN
#endif
