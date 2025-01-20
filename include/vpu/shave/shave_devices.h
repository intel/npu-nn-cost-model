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

#include "core/cache_descriptors.h"
#include "interface_shave_op_executor.h"
#include "shave_collection_NPU40.h"
#include "shave_collection_VPU27.h"
#include "shave_op_executors.h"
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

class ShaveCache {
private:
    SimpleLUTKeyCache<CyclesInterfaceType, SHAVEWorkload> shaveCacheRaw{1000};

protected:
    void populate();

public:
    ShaveCache() {
        populate();
    }
    // bool findCacheCycles(const SHAVEWorkload& swl, CyclesInterfaceType& result) const {
    //     const CyclesInterfaceType* ret{shaveCache.get(swl)};
    //     if (ret == nullptr) {
    //         return false;
    //     } else {
    //         result = *ret;
    //         return true;
    //     }
    // }
    std::optional<CyclesInterfaceType> findCacheCycles(const SHAVEWorkload& swl) const {
        const CyclesInterfaceType* ret{shaveCacheRaw.get(swl)};
        if (ret == nullptr) {
            return {};  // nothing , std::nullopt
        } else {
            return *ret;
        }
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
    } collections{};

    // selectors know on what collections to look (they are properly configured for each device)
    const ShaveSelector selector_20{collections.shaves_20_classic};
    // const ShavePrioritySelector selector_27{shaves_27_new,
    //                                         shaves_27_classic};          ///< special with 2 lists
    const ShaveSelector selector_27{collections.shaves_27};  ///< only the new list
    const ShaveSelector selector_40{collections.shaves_40};

    ShaveCache shaveCache;  ///< all devices cache/LUT for shave ops. Populated in ctor
    SHAVE_Workloads_Sanitizer sanitizer; ///< sanitizes the workload before processing

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
        default:
            static const DeviceShaveContainer empty_shaves{VPUDevice::__size};
            static const ShaveSelector empty_selector{empty_shaves};
            return empty_selector;
            break;
        }
    }

protected:
    /**
    @brief Sanitizes the workload before processing

    @param swl the workload to sanitize
    @param result the report of the sanitization

    @return true if the workload is sanitized, false otherwise
    */
    bool sanitize_workload(const SHAVEWorkload& swl, SanityReport& result) const {
        sanitizer.check_and_sanitize_workloads(swl, result);

        if (!result.is_usable()) {
            return false;
        }
        return true;
    }

public:
    CyclesInterfaceType computeCycles(const SHAVEWorkload& swl, std::string& infoOut, bool skipCacheSearch) const {
        // finds func inmpl, executes it, handles errors
        try {
            if (!skipCacheSearch) {  // before finding the shave imnpl check if already in cache for this request.Thisis
                                     // a one cache for all
                // @todo: specific cache per selector?
                const auto cachedData{shaveCache.findCacheCycles(swl)};
                if (cachedData) {
                    return cachedData.value();
                }
            }
            // if the swl in not found then it should be sanitized in here
            // This should be specifically only for profiled data. In the cache it can appear with different types
            SanityReport report;
            if (!sanitize_workload(swl, report)) {
				infoOut = report.info;
				return report.value();
			}

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
    }
    CyclesInterfaceType computeCycles(const SHAVEWorkload& swl, std::string& infoOut) const {
        return computeCycles(swl, infoOut, false);  // do not skip cache
    }

    bool isCached(const SHAVEWorkload& swl) const {
        const auto cachedData{shaveCache.findCacheCycles(swl)};
        return (cachedData) ? true : false;
    }

    std::vector<std::string> getShaveSupportedOperations(VPUDevice device) const {
        const auto& sel = getSelector(device);
        const auto list = sel.getShaveList();
        return list;
    }

    const ShaveOpExecutor& getShaveInstance(std::string name, VPUDevice device) const {
        const auto& sel = getSelector(device);
        const auto& shaveInstance = sel.getShaveFuntion(name);  // may throw
        return shaveInstance;
    }
};
}  // namespace VPUNN
#endif