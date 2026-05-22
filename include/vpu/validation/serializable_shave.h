// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_SERIALIZABLE_SHAVE_H
#define VPUNN_SERIALIZABLE_SHAVE_H

#include <array>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "vpu/dpu_types.h"
#include "vpu/profiling_service.h"
#include "serializable_tensor.h"
#include "vpu/shave_workload.h"

namespace VPUNN {

/// @brief local type describing a SHAVE workload
/// easy to change and adapt without touching the SHAVEWorkload interface
/// Similar to DPUOperation but for SHAVE operations
/* coverity[rule_of_three_violation:FALSE] */
struct SerializableSHAVE {
    VPUDevice device{};       ///< device family, VPU2_0, 2_7, ...
    std::string operation{};  ///< operation name
    ProfilingServiceBackend profiling_service_backend_hint{
            ProfilingServiceBackend::__size};  ///< hint about what profiling service backend to use
    std::string loc_name{};   ///< location name

    // Fixed-size arrays for serialization (mutable to allow modification during deserialization)
    // Support up to 8 inputs (0-indexed: input_0 through input_7) and 4 outputs as per csv_parser
    std::array<TensorInfo, 8> input_tensors{};
    std::array<TensorInfo, 4> output_tensors{};

    // Parameters as strings for serialization
    std::array<std::string, 3> param_strings{};        ///< up to 3 parameters
    std::array<std::string, 9> extra_param_strings{};  ///< up to 9 extra parameters

    using _ref_supported_type = std::variant<std::reference_wrapper<VPUDevice>, std::reference_wrapper<std::string>,
                                             std::reference_wrapper<long long>, std::reference_wrapper<DataType>,
                                             std::reference_wrapper<Layout>, std::reference_wrapper<bool>>;

    using MemberMapType = std::unordered_map<std::string, _ref_supported_type>;

    const MemberMapType& get_member_map() const;
    MemberMapType& get_member_map();

private:
    /** @brief Get the member map for serialization
     * Cannot be generalized keys to more generic prefix_index_batch... prefix_index_channels...
     * Because _member_map should be const to avoid accidental modification of this structure
     * @return The member map for serialization
     */
    mutable MemberMapType _member_map{};
    mutable std::mutex _member_map_mutex{};

    void populate_member_map();
    void ensure_member_map() const;
    bool is_member_map_initialized() const;

public:
    /**
     * Could be an idea to reuse _member_map keys, but function _get_member_names is
     * declared static const, so it should be hard coded
     */
    static const std::vector<std::string>& _get_member_names();

    static const std::string get_wl_name() {
        return "shave_operation_";
    }

    /// constructor from a SHAVEWorkload
    explicit SerializableSHAVE(const SHAVEWorkload& w)
            : device{w.get_device()}, 
              operation{w.get_name()}, 
              profiling_service_backend_hint{w.get_profiling_service_backend()},
              loc_name{w.get_loc_name()} {
        // Convert input tensors
        const auto& inputs = w.get_inputs();
        for (size_t i = 0; i < inputs.size() && i < input_tensors.size(); i++) {
            input_tensors[i] = TensorInfo(inputs[i]);
        }

        // Convert output tensors
        const auto& outputs = w.get_outputs();
        for (size_t i = 0; i < outputs.size() && i < output_tensors.size(); i++) {
            output_tensors[i] = TensorInfo(outputs[i]);
        }

        // Convert parameters to strings
        auto paramToString = [](const SHAVEWorkload::Param& p) -> std::string {
            if (const int* pvalInt = std::get_if<int>(&p)) {
                return std::to_string(*pvalInt);
            } else if (const float* pvalFloat = std::get_if<float>(&p)) {
                return std::to_string(*pvalFloat);
            } else if (const std::string* pvalString = std::get_if<std::string>(&p)) {
                return *pvalString;
            } else if (const bool* pvalBool = std::get_if<bool>(&p)) {
                return *pvalBool ? "true" : "false";
            }
            return "";
        };

        const auto& params = w.get_params();
        for (size_t i = 0; i < params.size() && i < param_strings.size(); i++) {
            param_strings[i] = paramToString(params[i]);
        }

        // Convert extra parameters to strings with key/value format
        const auto& extra_params = w.get_extra_params();
        size_t idx = 0;
        for (const auto& [key, value] : extra_params) {
            if (idx >= extra_param_strings.size())
                break;
            extra_param_strings[idx++] = key + "/" + paramToString(value);
        }
    }

    SerializableSHAVE() = default;

    SerializableSHAVE(const SerializableSHAVE& r)
            : device{r.device},
              operation{r.operation},
              profiling_service_backend_hint{r.profiling_service_backend_hint},
              loc_name{r.loc_name},
              input_tensors{r.input_tensors},
              output_tensors{r.output_tensors},
              param_strings{r.param_strings},
              extra_param_strings{r.extra_param_strings} {
    }

    SerializableSHAVE(SerializableSHAVE&) = delete;
    SerializableSHAVE(SerializableSHAVE&&) = delete;

    SerializableSHAVE& operator=(const SerializableSHAVE&) = delete;
    SerializableSHAVE& operator=(SerializableSHAVE&) = delete;
    SerializableSHAVE& operator=(SerializableSHAVE&&) = delete;
    SerializableSHAVE& operator=(SerializableSHAVE) = delete;

    ~SerializableSHAVE() = default;

    /// @brief Convert back to SHAVEWorkload
    SHAVEWorkload clone_as_SHAVEWorkload() const;

    void clearAllFields();
};

inline std::ostream& operator<<(std::ostream& stream, const SerializableSHAVE& d) {
    stream << "SerializableSHAVE-Workload: \n"                                                                         //
           << " Device: \t" << (int)d.device << " : " << VPUDevice_ToText.at(static_cast<int>(d.device)) << " ;\n"  //
           << " Operation: \t" << d.operation << " ;\n"                                                             //
           << " Location: \t" << d.loc_name << " ;\n";                                                              //

    // inputs
    stream << " inputs: \t{\n";
    for (size_t i = 0; i < d.input_tensors.size(); i++) {
        const auto& input = d.input_tensors[i];
        // Only show non-zero tensors
        if (input.isNonZero()) {
            stream << " input[" << i << "]: \t{\n" << input << " } ;\n";
        }
    }
    stream << "\t}inputs \n";

    // outputs
    stream << " outputs: \t{\n";
    for (size_t i = 0; i < d.output_tensors.size(); i++) {
        const auto& output = d.output_tensors[i];
        // Only show non-zero tensors
        if (output.isNonZero()) {
            stream << " output[" << i << "]: \t{\n" << output << " } ;\n";
        }
    }
    stream << "\t}outputs \n";

    // parameters
    stream << " parameters: \t{\n";
    for (size_t i = 0; i < d.param_strings.size(); i++) {
        if (!d.param_strings[i].empty()) {
            stream << " param[" << i << "]: \t{ " << d.param_strings[i] << " } ;\n";
        }
    }
    stream << "\t} \n";

    // extra parameters
    stream << " extra parameters: \t{\n";
    for (size_t i = 0; i < d.extra_param_strings.size(); i++) {
        if (!d.extra_param_strings[i].empty()) {
            stream << " extra_param[" << i << "]: \t{ " << d.extra_param_strings[i] << " } ;\n";
        }
    }
    stream << "\t} \n";

    return stream;
}

}  // namespace VPUNN

#endif // VPUNN_SERIALIZABLE_SHAVE_H
