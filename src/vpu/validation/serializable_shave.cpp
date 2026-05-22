// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/validation/serializable_shave.h"

#include <stdexcept>

namespace VPUNN {

void SerializableSHAVE::populate_member_map() {
    _member_map = MemberMapType{{"device", std::ref(device)},
                                {"operation", std::ref(operation)},
                                {"loc_name", std::ref(loc_name)}};

    // Input tensor slots
    for (size_t i = 0; i < input_tensors.size(); ++i) {
        const std::string pfx = "input_" + std::to_string(i) + "_";
        auto& tf = input_tensors[i];
        _member_map.emplace(pfx + "batch",            std::ref(tf.batch));
        _member_map.emplace(pfx + "channels",         std::ref(tf.channels));
        _member_map.emplace(pfx + "height",           std::ref(tf.height));
        _member_map.emplace(pfx + "width",            std::ref(tf.width));
        _member_map.emplace(pfx + "datatype",         std::ref(tf.datatype));
        _member_map.emplace(pfx + "layout",           std::ref(tf.layout));
        _member_map.emplace(pfx + "sparsity_enabled", std::ref(tf.sparsity_enabled));
    }

    // Output tensor slots
    for (size_t i = 0; i < output_tensors.size(); ++i) {
        const std::string pfx = "output_" + std::to_string(i) + "_";
        auto& tf = output_tensors[i];
        _member_map.emplace(pfx + "batch",            std::ref(tf.batch));
        _member_map.emplace(pfx + "channels",         std::ref(tf.channels));
        _member_map.emplace(pfx + "height",           std::ref(tf.height));
        _member_map.emplace(pfx + "width",            std::ref(tf.width));
        _member_map.emplace(pfx + "datatype",         std::ref(tf.datatype));
        _member_map.emplace(pfx + "layout",           std::ref(tf.layout));
        _member_map.emplace(pfx + "sparsity_enabled", std::ref(tf.sparsity_enabled));
    }

    // Parameter slots
    for (size_t i = 0; i < param_strings.size(); ++i) {
        _member_map.emplace("param_" + std::to_string(i), std::ref(param_strings[i]));
    }

    // Extra-parameter slots
    for (size_t i = 0; i < extra_param_strings.size(); ++i) {
        _member_map.emplace("extra_param_" + std::to_string(i), std::ref(extra_param_strings[i]));
    }
}

void SerializableSHAVE::ensure_member_map() const {
    std::lock_guard<std::mutex> lock(_member_map_mutex);
    if (is_member_map_initialized()) {
        return;
    }
    (const_cast<SerializableSHAVE*>(this))->populate_member_map();
}

const SerializableSHAVE::MemberMapType& SerializableSHAVE::get_member_map() const {
    ensure_member_map();
    return _member_map;
}

SerializableSHAVE::MemberMapType& SerializableSHAVE::get_member_map() {
    ensure_member_map();
    return _member_map;
}

bool SerializableSHAVE::is_member_map_initialized() const {
    return !_member_map.empty();
}


void SerializableSHAVE::clearAllFields() {
    device = VPUDevice::VPU_2_0;
    operation.clear();
    loc_name.clear();
    profiling_service_backend_hint = ProfilingServiceBackend::__size;

    for (auto& t : input_tensors) t = TensorInfo{};
    for (auto& t : output_tensors) t = TensorInfo{};
    for (auto& s : param_strings) s.clear();
    for (auto& s : extra_param_strings) s.clear();
}

SHAVEWorkload SerializableSHAVE::clone_as_SHAVEWorkload() const {
    std::vector<VPUTensor> input_vec;
    for (const auto& tf : input_tensors) {
        // Only add non-zero inputs
        if (tf.isNonZero()) {
            input_vec.push_back(tf.toVPUTensor());
        }
    }

    std::vector<VPUTensor> output_vec;
    for (const auto& tf : output_tensors) {
        // Only add non-zero outputs
        if (tf.isNonZero()) {
            output_vec.push_back(tf.toVPUTensor());
        }
    }

    // Convert string parameters back to variants
    SHAVEWorkload::Parameters call_params;
        auto stringToParam = [](const std::string& s) -> SHAVEWorkload::Param {
        if (s.empty()) {
            return SHAVEWorkload::Param{};
        }
        // Boolean spellings (extra_params serializes as "True"/"False")
        if (s == "true" || s == "True" || s == "TRUE") {
            return SHAVEWorkload::Param{true};
        }
        if (s == "false" || s == "False" || s == "FALSE") {
            return SHAVEWorkload::Param{false};
        }
        // Integer: accept only if the entire string is consumed
        try {
            size_t pos = 0;
            auto val = std::stoi(s, &pos);
            if (pos == s.size()) {
                return SHAVEWorkload::Param{val};
            }
        } catch (const std::invalid_argument&) {
        } catch (const std::out_of_range&) {
        }
        // Float: accept only if the entire string is consumed
        try {
            size_t pos = 0;
            auto val = std::stof(s, &pos);
            if (pos == s.size()) {
                return SHAVEWorkload::Param{val};
            }
        } catch (const std::invalid_argument&) {
        } catch (const std::out_of_range&) {
        }
        return SHAVEWorkload::Param{s};
    };

    for (const auto& param_str : param_strings) {
        if (!param_str.empty()) {
            call_params.push_back(stringToParam(param_str));
        }
    }

    // Convert extra parameters back
    SHAVEWorkload::ExtraParameters extra_params;
    for (const auto& extra_param_str : extra_param_strings) {
        if (extra_param_str.empty())
            continue;

        auto slash_pos = extra_param_str.find('/');
        if (slash_pos != std::string::npos) {
            std::string key = extra_param_str.substr(0, slash_pos);
            std::string value = extra_param_str.substr(slash_pos + 1);
            extra_params[key] = stringToParam(value);
        }
    }

    return SHAVEWorkload(operation, device, input_vec, output_vec, call_params, extra_params, profiling_service_backend_hint, loc_name);
}

const std::vector<std::string>& SerializableSHAVE::_get_member_names() {
    static const std::vector<std::string> names = {
                "device",
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
                "output_1_batch",
                "output_1_channels",
                "output_1_height",
                "output_1_width",
                "output_2_batch",
                "output_2_channels",
                "output_2_height",
                "output_2_width",
                "output_3_batch",
                "output_3_channels",
                "output_3_height",
                "output_3_width",
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
                "output_1_datatype",
                "output_1_layout",
                "output_1_sparsity_enabled",
                "output_2_datatype",
                "output_2_layout",
                "output_2_sparsity_enabled",
                "output_3_datatype",
                "output_3_layout",
                "output_3_sparsity_enabled",
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
                "loc_name"
    };
    return names;
}


} // namespace VPUNN