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


    } collections{};

    // selectors know on what collections to look (they are properly configured for each device)
    const ShaveSelector selector_20{collections.shaves_20_classic};
    // const ShavePrioritySelector selector_27{shaves_27_new,
    //                                         shaves_27_classic};          ///< special with 2 lists
    const ShaveSelector selector_27{collections.shaves_27};  ///< only the new list
    const ShaveSelector selector_40{collections.shaves_40};

    mutable LRUCache<SHAVEWorkload, float> cache;  ///< all devices cache/LUT for shave ops. Populated in ctor
    SHAVE_Workloads_Sanitizer sanitizer;           ///< sanitizes the workload before processing
    mutable CSVSerializer serializer;              ///< serializes workloads to a CSV file

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

    std::vector<std::string> get_names_for_shave_serializer() const {
        auto fields = std::vector<std::string>({"device",
                                                "operation",
                                                "input_0_batch",
                                                "input_0_channels",
                                                "input_0_height",
                                                "input_0_width",
                                                "input_0_sparsity_enabled",
                                                "input_0_datatype",
                                                "input_0_layout",
                                                "input_1_batch",
                                                "input_1_channels",
                                                "input_1_height",
                                                "input_1_width",
                                                "input_1_sparsity_enabled",
                                                "input_1_datatype",
                                                "input_1_layout",
                                                "output_0_batch",
                                                "output_0_channels",
                                                "output_0_height",
                                                "output_0_width",
                                                "output_0_datatype",
                                                "output_0_layout",
                                                "output_0_sparsity_enabled"});

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

        fields.emplace_back("loc_name");
        fields.emplace_back("info");
        fields.emplace_back("workload_uid");

        return fields;
    }

    void serialize_shave(const SHAVEWorkload& shave_wl) const {
        const auto& inputs = shave_wl.get_inputs();
        const auto& outputs = shave_wl.get_outputs();
        const auto& params = shave_wl.get_params();

        serializer.serialize(SerializableField<VPUDevice>{"device", shave_wl.get_device()});
        serializer.serialize(SerializableField<std::string>{"operation", shave_wl.get_name()});

        for (size_t i = 0; i < inputs.size(); i++) {
            serializer.serialize(
                    SerializableField<unsigned int>{"input_" + std::to_string(i) + "_batch", inputs[i].batches()});
            serializer.serialize(
                    SerializableField<unsigned int>{"input_" + std::to_string(i) + "_channels", inputs[i].channels()});
            serializer.serialize(
                    SerializableField<unsigned int>{"input_" + std::to_string(i) + "_height", inputs[i].height()});
            serializer.serialize(
                    SerializableField<unsigned int>{"input_" + std::to_string(i) + "_width", inputs[i].width()});
            serializer.serialize(
                    SerializableField<DataType>{"input_" + std::to_string(i) + "_datatype", inputs[i].get_dtype()});
            serializer.serialize(
                    SerializableField<Layout>{"input_" + std::to_string(i) + "_layout", inputs[i].get_layout()});
            serializer.serialize(SerializableField<bool>{"input_" + std::to_string(i) + "_sparsity_enabled",
                                                         inputs[i].get_sparsity()});
        }

        for (size_t i = 0; i < outputs.size(); i++) {
            serializer.serialize(
                    SerializableField<unsigned int>{"output_" + std::to_string(i) + "_batch", outputs[i].batches()});
            serializer.serialize(SerializableField<unsigned int>{"output_" + std::to_string(i) + "_channels",
                                                                 outputs[i].channels()});
            serializer.serialize(
                    SerializableField<unsigned int>{"output_" + std::to_string(i) + "_height", outputs[i].height()});
            serializer.serialize(
                    SerializableField<unsigned int>{"output_" + std::to_string(i) + "_width", outputs[i].width()});
            serializer.serialize(
                    SerializableField<DataType>{"output_" + std::to_string(i) + "_datatype", outputs[i].get_dtype()});
            serializer.serialize(
                    SerializableField<Layout>{"output_" + std::to_string(i) + "_layout", outputs[i].get_layout()});
            serializer.serialize(SerializableField<bool>{"output_" + std::to_string(i) + "_sparsity_enabled",
                                                         outputs[i].get_sparsity()});
        }

        int param_idx = 0;
        for (const auto& param : params) {
            std::visit(
                    [&](auto&& arg) {
                        using T = std::decay_t<decltype(arg)>;

                        serializer.serialize(SerializableField<T>{"param_" + std::to_string(param_idx), arg});
                    },
                    param);
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
        sanitizer.check_and_sanitize(swl, result);

        if (!result.is_usable()) {
            return false;
        }
        return true;
    }

public:
    ShaveConfiguration(const unsigned int cache_size /*= 16384*/, const std::string& cache_filename)
            : cache(cache_size, /*0,*/ cache_filename) {
        serializer.initialize("shave_workloads", FileMode::READ_WRITE, get_names_for_shave_serializer());
    }

    ShaveConfiguration(const unsigned int cache_size /* = 16384 */, const char* cache_data /* = nullptr */,
                       size_t cache_data_length /* = 0 */)
            : cache(cache_size, /* 0,*/ cache_data, cache_data_length) {
        serializer.initialize("shave_workloads", FileMode::READ_WRITE, get_names_for_shave_serializer());
    }

    CyclesInterfaceType computeCycles(const SHAVEWorkload& swl, std::string& infoOut, bool skipCacheSearch) const {
        // finds func inmpl, executes it, handles errors
        try {
            if (!skipCacheSearch) {  // before finding the shave imnpl check if already in cache for this request.Thisis
                                     // a one cache for all
                // @todo: specific cache per selector?
                const auto cachedData{cache.get(swl)};
                if (cachedData) {
                    return static_cast<CyclesInterfaceType>(std::floor(*cachedData));
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

            if (serializer.is_serialization_enabled()) {
                try {
                    serializer.serialize(SerializableField<std::string>{"loc_name", swl.get_loc_name()});
                    serializer.serialize(SerializableField<std::string>{"shave_model_kind", "shave_2"});
                    serializer.serialize(SerializableField<decltype(cycles)>{"cycles", cycles});
                    serialize_shave(swl);
                    serializer.end();
                } catch (const std::exception& e) {
                    Logger::warning() << "Encountered invalid workload while serialization: " << e.what() << "\n";
                    serializer.clean_buffers();
                }
            }

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
        /* coverity[copy_instead_of_move] */
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
