// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_SHAVE_DEVICES_H
#define VPUNN_SHAVE_DEVICES_H

#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
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
    std::unique_ptr<const DeviceShaveContainer> container;

public:
    explicit ShaveSelector(std::unique_ptr<const DeviceShaveContainer> shave_container)
            : container(std::move(shave_container)) {
    }

    virtual ~ShaveSelector() = default;

    virtual const ShaveOpExecutor& getShaveFunction(const std::string& name) const {
        return container->getShaveExecutor(name);  // will throw if not existing
    }
    virtual std::vector<std::string> getShaveList() const {
        return container->getShaveList();
    }

protected:
    bool existsShave(const std::string& name) const {
        return container->existsShave(name);
    }
};

/// @brief selects from 2 containers , 1st with priority
class ShavePrioritySelector : public ShaveSelector {
    std::unique_ptr<const DeviceShaveContainer> container2;

public:
    ShavePrioritySelector(std::unique_ptr<const DeviceShaveContainer> shave_container_first,
                          std::unique_ptr<const DeviceShaveContainer> shave_container_second)
            : ShaveSelector(std::move(shave_container_first)), container2(std::move(shave_container_second)) {
    }

    virtual ~ShavePrioritySelector() = default;

    const ShaveOpExecutor& getShaveFunction(const std::string& name) const override {
        if (ShaveSelector::existsShave(name)) {
            return ShaveSelector::getShaveFunction(name);  // should not throw
        } else {
            return container2->getShaveExecutor(name);  // will throw if not existing
        }
    }
    virtual std::vector<std::string> getShaveList() const override {
        auto v1{ShaveSelector::getShaveList()};
        const auto v2{container2->getShaveList()};
        v1.insert(v1.end(), v2.begin(), v2.end());
        return v1;
    }
};

/// @brief the shave configuration. Holds instances of selectors for different devices
/// This class is responsible for managing the SHAVE function executors for different devices and providing selectors to
/// access them. It uses lazy initialization to create selectors on demand and caches them for future use.
class ShaveConfiguration {
private:
    using SelectorCache = std::map<VPUDevice, std::unique_ptr<ShaveSelector>>;

    // Lazy-initialized selector caches (mutable for const access in getters).
    // All five caches share one mutex: construction is a one-time-per-device cost,
    // so per-cache granularity would only add complexity without measurable benefit.
    mutable std::shared_mutex cache_mutex;
    mutable SelectorCache selector_cache;
    mutable SelectorCache old_selector_cache;
    mutable SelectorCache selector_with_factors_cache;
    mutable SelectorCache heuristic_selector_cache;
    mutable SelectorCache heuristic_selector_with_factors_cache;

    // Factory methods for each selector type
    std::unique_ptr<ShaveSelector> createSelectorForDevice(VPUDevice device) const;
    std::unique_ptr<ShaveSelector> createOldSelectorForDevice(VPUDevice device) const;
    std::unique_ptr<ShaveSelector> createSelectorWithFactorsForDevice(VPUDevice device) const;
    std::unique_ptr<ShaveSelector> createHeuristicSelectorForDevice(VPUDevice device) const;
    std::unique_ptr<ShaveSelector> createHeuristicSelectorWithFactorsForDevice(VPUDevice device) const;

    /// @brief Returns the cached selector for @p device, creating it via @p make_selector on first access.
    /// Uses double-checked locking: a shared (read) lock is sufficient for the common post-warmup case,
    /// and a unique (write) lock is only acquired when construction is actually needed.
    template<typename SelectorFactory>
    const ShaveSelector& getOrCreate(SelectorCache& cache, VPUDevice device, SelectorFactory&& make_selector) const {
        {
            std::shared_lock read(cache_mutex);
            auto it = cache.find(device);
            if (it != cache.end() && it->second)
                return *it->second;
        }
        std::unique_lock write(cache_mutex);
        auto& entry = cache[device];  // re-check after acquiring write lock
        if (!entry)
            entry = make_selector(device);
        return *entry;
    }

public:
    /// @brief Returns the standard selector for @p device (model-based, no correction factors).
    const ShaveSelector& getSelector(VPUDevice device) const {
        auto make_selector = [this](VPUDevice d) { return createSelectorForDevice(d); };
        return getOrCreate(selector_cache, device, make_selector);
    }

    /// @brief Returns the legacy/old selector for @p device, used for backwards-compatible cost estimates.
    const ShaveSelector& getOldSelector(VPUDevice device) const {
        auto make_selector = [this](VPUDevice d) { return createOldSelectorForDevice(d); };
        return getOrCreate(old_selector_cache, device, make_selector);
    }

    /// @brief Returns the selector with correction factors applied for @p device.
    const ShaveSelector& getSelectorWithFactors(VPUDevice device) const {
        auto make_selector = [this](VPUDevice d) { return createSelectorWithFactorsForDevice(d); };
        return getOrCreate(selector_with_factors_cache, device, make_selector);
    }

    /// @brief Returns the heuristic-based selector for @p device.
    const ShaveSelector& getHeuristicSelector(VPUDevice device) const {
        auto make_selector = [this](VPUDevice d) { return createHeuristicSelectorForDevice(d); };
        return getOrCreate(heuristic_selector_cache, device, make_selector);
    }

    /// @brief Returns the heuristic-based selector with correction factors applied for @p device.
    const ShaveSelector& getHeuristicSelectorWithFactors(VPUDevice device) const {
        auto make_selector = [this](VPUDevice d) { return createHeuristicSelectorWithFactorsForDevice(d); };
        return getOrCreate(heuristic_selector_with_factors_cache, device, make_selector);
    }

    /**
     * @brief Get the maximum number of parameters across all SHAVE functions
     *
     * Iterates through all SHAVE functions to find the maximum number of expected parameters.
     * This is useful for determining the appropriate buffer sizes or field counts needed for serialization.
     *
     * @return int Maximum number of parameters found across all SHAVE functions
     */
    int get_max_num_params() const {
        int max_num_params = 0;
        for (int i = 0; i < static_cast<int>(VPUDevice::__size); i++) {
            const auto& sel = getSelector(static_cast<VPUDevice>(i));
            const auto& shv_list = sel.getShaveList();
            for (const auto& name : shv_list) {
                const int current_num_params = sel.getShaveFunction(name).getNumExpectedParams();
                if (current_num_params > max_num_params) {
                    max_num_params = current_num_params;
                }
            }
        }

        return max_num_params;
    }
};
}  // namespace VPUNN
#endif // VPUNN_SHAVE_DEVICES_H
