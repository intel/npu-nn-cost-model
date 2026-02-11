// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_VPU_VALIDATOR_DATA_SHAVE_OPERATION_H
#define VPUNN_VPU_VALIDATOR_DATA_SHAVE_OPERATION_H

#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "core/serializer.h"
#include "vpu/dpu_defaults.h"
#include "vpu/dpu_types.h"
#include "vpu/shave_workload.h"
#include "vpu/serializer_utils.h"
#include "vpu/validation/data_dpu_operation.h"

namespace VPUNN {

/// @brief local type describing a SHAVE workload
/// easy to change and adapt without touching the SHAVEWorkload interface
/// Similar to DPUOperation but for SHAVE operations
/* coverity[rule_of_three_violation:FALSE] */
struct SHAVEOperation {
    VPUDevice device{};      ///< device family, VPU2_0, 2_7, ...
    std::string operation{}; ///< operation name
    std::string loc_name{};  ///< location name

    // Fixed-size arrays for serialization (mutable to allow modification during deserialization)
    // Support up to 8 inputs (0-indexed: input_0 through input_7) and 1 output as per csv_parser
    std::array<TensorInfo, 8> input_tensors{};
    std::array<TensorInfo, 1> output_tensors{};

    // Parameters as strings for serialization
    std::array<std::string, 3> param_strings{};       ///< up to 3 parameters
    std::array<std::string, 9> extra_param_strings{}; ///< up to 9 extra parameters

    using _ref_supported_type = std::variant<std::reference_wrapper<VPUDevice>, std::reference_wrapper<std::string>,
                                             std::reference_wrapper<long long>, std::reference_wrapper<DataType>,
                                             std::reference_wrapper<Layout>, std::reference_wrapper<bool>>;

    /** @brief Get the member map for serialization
     * Cannot be generalized keys to more generic prefix_index_batch... prefix_index_channels...
     * Because _member_map should be const to avoid accidental modification of this structure
     * @return The member map for serialization
     */
    const std::unordered_map<std::string, _ref_supported_type> _member_map{
            {"device", std::ref(device)},
            {"operation", std::ref(operation)},
            {"loc_name", std::ref(loc_name)},

            // Input 0
            {"input_0_batch", std::ref(input_tensors[0].batch)},
            {"input_0_channels", std::ref(input_tensors[0].channels)},
            {"input_0_height", std::ref(input_tensors[0].height)},
            {"input_0_width", std::ref(input_tensors[0].width)},
            {"input_0_datatype", std::ref(input_tensors[0].datatype)},
            {"input_0_layout", std::ref(input_tensors[0].layout)},
            {"input_0_sparsity_enabled", std::ref(input_tensors[0].sparsity_enabled)},

            // Input 1
            {"input_1_batch", std::ref(input_tensors[1].batch)},
            {"input_1_channels", std::ref(input_tensors[1].channels)},
            {"input_1_height", std::ref(input_tensors[1].height)},
            {"input_1_width", std::ref(input_tensors[1].width)},
            {"input_1_datatype", std::ref(input_tensors[1].datatype)},
            {"input_1_layout", std::ref(input_tensors[1].layout)},
            {"input_1_sparsity_enabled", std::ref(input_tensors[1].sparsity_enabled)},

            // Input 2
            {"input_2_batch", std::ref(input_tensors[2].batch)},
            {"input_2_channels", std::ref(input_tensors[2].channels)},
            {"input_2_height", std::ref(input_tensors[2].height)},
            {"input_2_width", std::ref(input_tensors[2].width)},
            {"input_2_datatype", std::ref(input_tensors[2].datatype)},
            {"input_2_layout", std::ref(input_tensors[2].layout)},
            {"input_2_sparsity_enabled", std::ref(input_tensors[2].sparsity_enabled)},

            // Input 3
            {"input_3_batch", std::ref(input_tensors[3].batch)},
            {"input_3_channels", std::ref(input_tensors[3].channels)},
            {"input_3_height", std::ref(input_tensors[3].height)},
            {"input_3_width", std::ref(input_tensors[3].width)},
            {"input_3_datatype", std::ref(input_tensors[3].datatype)},
            {"input_3_layout", std::ref(input_tensors[3].layout)},
            {"input_3_sparsity_enabled", std::ref(input_tensors[3].sparsity_enabled)},

            // Input 4
            {"input_4_batch", std::ref(input_tensors[4].batch)},
            {"input_4_channels", std::ref(input_tensors[4].channels)},
            {"input_4_height", std::ref(input_tensors[4].height)},
            {"input_4_width", std::ref(input_tensors[4].width)},
            {"input_4_datatype", std::ref(input_tensors[4].datatype)},
            {"input_4_layout", std::ref(input_tensors[4].layout)},
            {"input_4_sparsity_enabled", std::ref(input_tensors[4].sparsity_enabled)},

            // Input 5
            {"input_5_batch", std::ref(input_tensors[5].batch)},
            {"input_5_channels", std::ref(input_tensors[5].channels)},
            {"input_5_height", std::ref(input_tensors[5].height)},
            {"input_5_width", std::ref(input_tensors[5].width)},
            {"input_5_datatype", std::ref(input_tensors[5].datatype)},
            {"input_5_layout", std::ref(input_tensors[5].layout)},
            {"input_5_sparsity_enabled", std::ref(input_tensors[5].sparsity_enabled)},

            // Input 6
            {"input_6_batch", std::ref(input_tensors[6].batch)},
            {"input_6_channels", std::ref(input_tensors[6].channels)},
            {"input_6_height", std::ref(input_tensors[6].height)},
            {"input_6_width", std::ref(input_tensors[6].width)},
            {"input_6_datatype", std::ref(input_tensors[6].datatype)},
            {"input_6_layout", std::ref(input_tensors[6].layout)},
            {"input_6_sparsity_enabled", std::ref(input_tensors[6].sparsity_enabled)},

            // Input 7
            {"input_7_batch", std::ref(input_tensors[7].batch)},
            {"input_7_channels", std::ref(input_tensors[7].channels)},
            {"input_7_height", std::ref(input_tensors[7].height)},
            {"input_7_width", std::ref(input_tensors[7].width)},
            {"input_7_datatype", std::ref(input_tensors[7].datatype)},
            {"input_7_layout", std::ref(input_tensors[7].layout)},
            {"input_7_sparsity_enabled", std::ref(input_tensors[7].sparsity_enabled)},

            // Output 0
            {"output_0_batch", std::ref(output_tensors[0].batch)},
            {"output_0_channels", std::ref(output_tensors[0].channels)},
            {"output_0_height", std::ref(output_tensors[0].height)},
            {"output_0_width", std::ref(output_tensors[0].width)},
            {"output_0_datatype", std::ref(output_tensors[0].datatype)},
            {"output_0_layout", std::ref(output_tensors[0].layout)},
            {"output_0_sparsity_enabled", std::ref(output_tensors[0].sparsity_enabled)},

            // Parameters
            {"param_0", std::ref(param_strings[0])},
            {"param_1", std::ref(param_strings[1])},
            {"param_2", std::ref(param_strings[2])},

            // Extra parameters
            {"extra_param_0", std::ref(extra_param_strings[0])},
            {"extra_param_1", std::ref(extra_param_strings[1])},
            {"extra_param_2", std::ref(extra_param_strings[2])},
            {"extra_param_3", std::ref(extra_param_strings[3])},
            {"extra_param_4", std::ref(extra_param_strings[4])},
            {"extra_param_5", std::ref(extra_param_strings[5])},
            {"extra_param_6", std::ref(extra_param_strings[6])},
            {"extra_param_7", std::ref(extra_param_strings[7])},
            {"extra_param_8", std::ref(extra_param_strings[8])}
        };

    /**
     * Could be an idea to reuse _member_map keys, but function _get_member_names is
     * declared static const, so it should be hard coded
     */
    static const std::vector<std::string> _get_member_names() {
        return {"device",
                "operation",
                "input_0_batch",
                "input_0_channels",
                "input_0_height",
                "input_0_width",
                "input_1_batch",
                "input_1_channels",
                "input_1_height",
                "input_1_width",
                "input_2_batch",
                "input_2_channels",
                "input_2_height",
                "input_2_width",
                "input_3_batch",
                "input_3_channels",
                "input_3_height",
                "input_3_width",
                "input_4_batch",
                "input_4_channels",
                "input_4_height",
                "input_4_width",
                "input_5_batch",
                "input_5_channels",
                "input_5_height",
                "input_5_width",
                "input_6_batch",
                "input_6_channels",
                "input_6_height",
                "input_6_width",
                "input_7_batch",
                "input_7_channels",
                "input_7_height",
                "input_7_width",
                "output_0_batch",
                "output_0_channels",
                "output_0_height",
                "output_0_width",
                "input_0_datatype",
                "input_0_layout",
                "input_0_sparsity_enabled",
                "input_1_datatype",
                "input_1_layout",
                "input_1_sparsity_enabled",
                "input_2_datatype",
                "input_2_layout",
                "input_2_sparsity_enabled",
                "input_3_datatype",
                "input_3_layout",
                "input_3_sparsity_enabled",
                "input_4_datatype",
                "input_4_layout",
                "input_4_sparsity_enabled",
                "input_5_datatype",
                "input_5_layout",
                "input_5_sparsity_enabled",
                "input_6_datatype",
                "input_6_layout",
                "input_6_sparsity_enabled",
                "input_7_datatype",
                "input_7_layout",
                "input_7_sparsity_enabled",
                "output_0_datatype",
                "output_0_layout",
                "output_0_sparsity_enabled",
                "param_0",
                "param_1",
                "param_2",
                "extra_param_0",
                "extra_param_1",
                "extra_param_2",
                "extra_param_3",
                "extra_param_4",
                "extra_param_5",
                "extra_param_6",
                "extra_param_7",
                "extra_param_8",
                "loc_name"};
    }

    static const std::string get_wl_name() {
        return "shave_operation_";
    }

    /// constructor from a SHAVEWorkload
    explicit SHAVEOperation(const SHAVEWorkload& w)
            : device{w.get_device()}, 
              operation{w.get_name()}, 
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

    SHAVEOperation() = default;

    SHAVEOperation(const SHAVEOperation& r)
            : device{r.device},
              operation{r.operation},
              loc_name{r.loc_name},
              input_tensors{r.input_tensors},
              output_tensors{r.output_tensors},
              param_strings{r.param_strings},
              extra_param_strings{r.extra_param_strings} {
    }

    SHAVEOperation(SHAVEOperation&) = delete;
    SHAVEOperation(const SHAVEOperation&&) = delete;
    SHAVEOperation(SHAVEOperation&&) = delete;

    SHAVEOperation& operator=(const SHAVEOperation&) = delete;
    SHAVEOperation& operator=(SHAVEOperation&) = delete;
    SHAVEOperation& operator=(SHAVEOperation) = delete;

    ~SHAVEOperation() = default;

    /// @brief Convert back to SHAVEWorkload
    SHAVEWorkload clone_as_SHAVEWorkload() const {
        std::vector<VPUTensor> input_vec;
        for (const auto& input_info : input_tensors) {
            // Only add non-zero inputs
            if (!(input_info.channels == 0 && input_info.height == 0 &&
                  input_info.width == 0)) {
                input_vec.push_back(input_info.toVPUTensor());
            }
        }

        std::vector<VPUTensor> output_vec;
        for (const auto& output_info : output_tensors) {
            // Only add non-zero outputs
            if (!(output_info.channels == 0 && output_info.height == 0 &&
                  output_info.width == 0)) {
                output_vec.push_back(output_info.toVPUTensor());
            }
        }

        // Convert string parameters back to variants
        SHAVEWorkload::Parameters call_params;
        auto stringToParam = [](const std::string& s) -> SHAVEWorkload::Param {
            if (s.empty()) {
                return SHAVEWorkload::Param{};
            }
            try {
                auto val = std::stoi(s);
                return SHAVEWorkload::Param{val};
            } catch (const std::invalid_argument&) {
                try {
                    // Then try float
                    auto val = std::stof(s);
                    return SHAVEWorkload::Param{val};
                } catch (const std::invalid_argument&) {
                    // Finally, treat as string
                    return SHAVEWorkload::Param{s};
                }
            }
        };

        for (const auto& param_str : param_strings) {
            if (!param_str.empty()) {
                call_params.push_back(stringToParam(param_str));
            }
        }

        // Convert extra parameters back
        SHAVEWorkload::ExtraParameters extra_param_map;
        for (const auto& extra_param_str : extra_param_strings) {
            if (extra_param_str.empty())
                continue;

            auto slash_pos = extra_param_str.find('/');
            if (slash_pos != std::string::npos) {
                std::string key = extra_param_str.substr(0, slash_pos);
                std::string value = extra_param_str.substr(slash_pos + 1);
                extra_param_map[key] = stringToParam(value);
            }
        }

        return SHAVEWorkload(operation, device, input_vec, output_vec, call_params, extra_param_map, loc_name);
    }

    void clearAllFields() {
        device = VPUDevice::VPU_2_0;
        operation.clear();
        loc_name.clear();

        for (auto& input : input_tensors) {
            input = TensorInfo{};
        }
        for (auto& output : output_tensors) {
            output = TensorInfo{};
        }
        for (auto& param : param_strings) {
            param.clear();
        }
        for (auto& extra_param : extra_param_strings) {
            extra_param.clear();
        }
    }
};

inline std::ostream& operator<<(std::ostream& stream, const SHAVEOperation& d) {
    stream << "SHAVEOperation-Workload: \n"                                                                          //
           << " device: \t" << (int)d.device << " : " << VPUDevice_ToText.at(static_cast<int>(d.device)) << " ;\n"  //
           << " Operation: \t" << d.operation << " ;\n"                                                              //
           << " Location: \t" << d.loc_name << " ;\n";                                                               //

    // inputs
    stream << " inputs: \t{\n";
    for (size_t i = 0; i < d.input_tensors.size(); i++) {
        const auto& input = d.input_tensors[i];
        // Only show non-zero tensors
        if (!(input.batch == 0 && input.channels == 0 && input.height == 0 && input.width == 0)) {
            stream << " input[" << i << "]: \t{\n" << input << " } ;\n";
        }
    }
    stream << "\t}inputs \n";

    // outputs
    stream << " outputs: \t{\n";
    for (size_t i = 0; i < d.output_tensors.size(); i++) {
        const auto& output = d.output_tensors[i];
        // Only show non-zero tensors
        if (!(output.batch == 0 && output.channels == 0 && output.height == 0 && output.width == 0)) {
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

#endif  // VPUNN_VPU_VALIDATOR_DATA_SHAVE_OPERATION_H
