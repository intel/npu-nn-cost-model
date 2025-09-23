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

    mutable LRUCache<SHAVEWorkload, float> cache;  ///< all devices cache/LUT for shave ops. Populated in ctor
    SHAVE_Workloads_Sanitizer sanitizer;           ///< sanitizes the workload before processing
    mutable CSVSerializer serializer;              ///< serializes workloads to a CSV file
    const bool use_shave_2_api;                    ///< Lets you select the Shave2 or Shave1 at ctor

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
        default:
            static const DeviceShaveContainer empty_shaves{VPUDevice::__size};
            static const ShaveSelector empty_selector{empty_shaves};
            return empty_selector;
            break;
        }
    }
    std::vector<std::string> get_names_for_shave_serializer() const {
        auto fields = std::vector<std::string>({"device", "operation"});
        for (int i = 0; i < 8; i++) {
            fields.emplace_back("input_" + std::to_string(i) + "_batch");
            fields.emplace_back("input_" + std::to_string(i) + "_channels");
            fields.emplace_back("input_" + std::to_string(i) + "_height");
            fields.emplace_back("input_" + std::to_string(i) + "_width");
            fields.emplace_back("input_" + std::to_string(i) + "_sparsity_enabled");
            fields.emplace_back("input_" + std::to_string(i) + "_datatype");
            fields.emplace_back("input_" + std::to_string(i) + "_layout");
        }

        fields.emplace_back("output_0_batch");
        fields.emplace_back("output_0_channels");
        fields.emplace_back("output_0_height");
        fields.emplace_back("output_0_width");
        fields.emplace_back("output_0_sparsity_enabled");
        fields.emplace_back("output_0_datatype");
        fields.emplace_back("output_0_layout");

        fields.emplace_back("shave_model_kind");
        fields.emplace_back("cycles");

        std::vector<int> all_num_params{};
        for (int i = 0; i < static_cast<int>(VPUDevice::__size); i++) {
            const auto& sel = getSelector(static_cast<VPUDevice>(i));
            const auto& shv_list = sel.getShaveList();
            for (const auto& name : shv_list) {
                all_num_params.push_back(sel.getShaveFuntion(name).getNumExpectedParams());
            }
        }

        auto max_num_params = *std::max_element(all_num_params.begin(), all_num_params.end());

        for (int i = 0; i <= max_num_params; i++) {
            fields.emplace_back("param_" + std::to_string(i));
        }

        for (int i = 0; i <= 8; i++) {
            fields.emplace_back("extra_param_" + std::to_string(i));
        }

        fields.emplace_back("loc_name");
        fields.emplace_back("info");
        fields.emplace_back("workload_uid");

        return fields;
    }

protected:
    /**
    @brief Sanitizes the workload before processing

    @param swl the workload to sanitize
    @param result the report of the sanitization

    @return true if the workload is sanitized, false otherwise
    */
    bool sanitize_workload(const SHAVEWorkload& swl, SanityReport& result) const {
        sanitizer.check_and_sanitize(swl, result);

        if (!result.is_usable()) {
            return false;
        }
        return true;
    }

private:
    /**
     * @brief Fetch cycles using the Shave2 API.
     * @param swl The workload to process.
     * @return The computed cycles.
     * @throws std::exception if the Shave2 API fails.
     */
    CyclesInterfaceType fetchCyclesFromShave2(const SHAVEWorkload& swl) const {
        const auto& selector = getSelector(swl.get_device());
        const auto& shaveInstance = selector.getShaveFuntion(swl.get_name());
        return shaveInstance.dpuCycles(swl);
    }

    /**
     * @brief Fetch cycles using the Shave1 API.
     * @param swl The workload to process.
     * @return The computed cycles.
     * @throws std::exception if the Shave1 API fails.
     */
    CyclesInterfaceType fetchCyclesFromShave1(const SHAVEWorkload& swl) const {
        const auto& selector = getOldSelector(swl.get_device());
        const auto& shaveInstance = selector.getShaveFuntion(swl.get_name());
        return shaveInstance.dpuCycles(swl);
    }

    /**
     * @brief Helper function to fetch cycles based on the selected API.
     * @param swl The workload to process.
     * @param apiUsed Output parameter to indicate which API was used ("shave_2" or "shave_1").
     * @return The computed cycles.
     * @throws std::exception if both APIs fail.
     */
    CyclesInterfaceType fetchCycles(const SHAVEWorkload& swl, std::string& apiUsed) const {
        if (use_shave_2_api) {
            try {
                apiUsed = "shave_2";
                return fetchCyclesFromShave2(swl);
            } catch (const std::exception&) {
                // Fallback to Shave1 API
            }
        }
        apiUsed = "shave_1";
        return fetchCyclesFromShave1(swl);
    }

public:
    ShaveConfiguration(const unsigned int cache_size /*= 16384*/, const std::string& cache_filename,
                       bool use_shave_2_api)
            : cache(cache_size, /*0,*/ cache_filename), use_shave_2_api(use_shave_2_api) {
        serializer.initialize("shave_workloads", FileMode::READ_WRITE, get_names_for_shave_serializer());
    }

    ShaveConfiguration(const unsigned int cache_size /* = 16384 */, const char* cache_data /* = nullptr */,
                       size_t cache_data_length /* = 0 */, bool use_shave_2_api)
            : cache(cache_size, /* 0,*/ cache_data, cache_data_length), use_shave_2_api(use_shave_2_api) {
        serializer.initialize("shave_workloads", FileMode::READ_WRITE, get_names_for_shave_serializer());
    }

    CyclesInterfaceType computeCycles(const SHAVEWorkload& swl, std::string& infoOut, bool skipCacheSearch) const {
        // finds func inmpl, executes it, handles errors
        SHAVECostSerializationWrap serialization_handler(serializer);
        try {
            CyclesInterfaceType cycles{Cycles::NO_ERROR};  // Initialize with a default error value
            std::string apiUsed{"cache"};

            if (!skipCacheSearch &&
                use_shave_2_api) {  // before finding the shave imnpl check if already in cache for this request.Thisis
                                    // a one cache for all
                // @todo: specific cache per selector?
                const auto cachedData{cache.get(swl)};
                if (cachedData) {
                    cycles = static_cast<CyclesInterfaceType>(std::floor(*cachedData));
                    serialization_handler.serializeShaveWorkloadWithCycles(swl, apiUsed, cycles);
                    return cycles;
                }
            }
            // if the swl in not found then it should be sanitized in here
            // This should be specifically only for profiled data. In the cache it can appear with different types

            // SanityReport report;
            // if (!sanitize_workload(swl, report)) {
            //     infoOut = report.info;
            //     return report.value();
            // }

            cycles = fetchCycles(swl, apiUsed);  // may throw
            serialization_handler.serializeShaveWorkloadWithCycles(swl, apiUsed, cycles);
            return cycles;

        } catch (const std::exception& e) {
            std::stringstream buffer;
            buffer << "[EXCEPTION]:could not resolve shave function: " << swl << "\n "
                   << " Original exception: " << e.what() << " File: " << __FILE__ << " Line: " << __LINE__;
            const std::string details = buffer.str();
            infoOut = std::move(details);
            return Cycles::ERROR_SHAVE;
        }
    }

    CyclesInterfaceType computeCycles(const SHAVEWorkload& swl, std::string& infoOut) const {
        return computeCycles(swl, infoOut, false);  // do not skip cache
    }

    bool isCached(const SHAVEWorkload& swl) const {
        const auto cachedData{cache.get(swl)};
        return (cachedData) ? true : false;
    }

    std::vector<std::string> getShaveSupportedOperations(VPUDevice device) const {
        const auto& sel = getSelector(device);
        const std::vector<std::string> list = sel.getShaveList();
        // clang and gcc does not support to use std::move here, so we need suppression
        /* coverity[copy_instead_of_move] */
        return list;
    }

    const ShaveOpExecutor& getShaveInstance(std::string name, VPUDevice device) const {
        const auto& sel = getSelector(device);
        const auto& shaveInstance = sel.getShaveFuntion(name);  // may throw
        return shaveInstance;
    }
    const AccessCounter& getPreloadedCacheCounter() const {
        return cache.getPreloadedCacheCounter();
    }

    bool isShave2APIused() const {
        return use_shave_2_api;
    }
};
}  // namespace VPUNN
#endif
