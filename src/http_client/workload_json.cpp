// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

// Defines toJson specializations for all workload types supported by the HTTP cost provider.
// This is the single place to add JSON serialization logic when introducing a new workload type:
//   1. Add the toJson<NewWorkload> definition below.
//   2. Add its declaration and WorkloadKeyTrait specialization in http_workload_variant.h.
//   3. Add the type to HttpWorkloadVariant::VariantType in http_workload_variant.h.
//   4. Add the explicit getCost instantiation in http_cost_provider_intf.cpp.

#include "http_client/workload_json.h"
#include <nlohmann/json.hpp>
#include <string>
#include <variant>
#include <vector>
#include "core/utils.h"
#include "vpu/dma_types.h"
#include "vpu/shave_workload.h"
#include "vpu/validation/data_dpu_operation.h"

namespace VPUNN {

template <>
nlohmann::json toJson<DPUOperation>(const DPUOperation& op) {
    nlohmann::json json_op;

    // TODO: extend Serialization to support JSON format instead of this
    json_op["device"] = "VPUDevice." + mapToText<VPUDevice>().at(static_cast<int>(op.device));
    json_op["operation"] = "Operation." + mapToText<Operation>().at(static_cast<int>(op.operation));

    json_op["input_0_batch"] = op.input_0.batch;
    json_op["input_0_channels"] = op.input_0.channels;
    json_op["input_0_height"] = op.input_0.height;
    json_op["input_0_width"] = op.input_0.width;

    json_op["input_1_batch"] = op.input_1.batch;
    json_op["input_1_channels"] = op.input_1.channels;
    json_op["input_1_height"] = op.input_1.height;
    json_op["input_1_width"] = op.input_1.width;

    json_op["input_sparsity_enabled"] = static_cast<int>(op.input_0.sparsity_enabled);
    json_op["weight_sparsity_enabled"] = static_cast<int>(op.input_1.sparsity_enabled);
    json_op["input_sparsity_rate"] = op.input_0.sparsity;
    json_op["weight_sparsity_rate"] = op.input_1.sparsity;

    json_op["execution_order"] = "ExecutionMode." + mapToText<ExecutionMode>().at(static_cast<int>(op.execution_order));
    json_op["activation_function"] =
            "ActivationFunction." + mapToText<ActivationFunction>().at(static_cast<int>(op.activation_function));

    json_op["kernel_height"] = op.kernel.height;
    json_op["kernel_width"] = op.kernel.width;
    json_op["kernel_pad_bottom"] = op.kernel.pad_bottom;
    json_op["kernel_pad_top"] = op.kernel.pad_top;
    json_op["kernel_pad_left"] = op.kernel.pad_left;
    json_op["kernel_pad_right"] = op.kernel.pad_right;
    json_op["kernel_stride_height"] = op.kernel.stride_height;
    json_op["kernel_stride_width"] = op.kernel.stride_width;

    json_op["output_0_batch"] = op.output_0.batch;
    json_op["output_0_channels"] = op.output_0.channels;
    json_op["output_0_height"] = op.output_0.height;
    json_op["output_0_width"] = op.output_0.width;
    json_op["output_sparsity_enabled"] = static_cast<int>(op.output_0.sparsity_enabled);

    json_op["input_0_datatype"] = "DataType." + mapToText<DataType>().at(static_cast<int>(op.input_0.datatype));
    json_op["input_0_layout"] = "Layout." + mapToText<Layout>().at(static_cast<int>(op.input_0.layout));
    json_op["input_0_swizzling"] = "Swizzling." + mapToText<Swizzling>().at(static_cast<int>(op.input_0.swizzling));
    json_op["input_1_datatype"] = "DataType." + mapToText<DataType>().at(static_cast<int>(op.input_1.datatype));
    json_op["input_1_layout"] = "Layout." + mapToText<Layout>().at(static_cast<int>(op.input_1.layout));
    json_op["input_1_swizzling"] = "Swizzling." + mapToText<Swizzling>().at(static_cast<int>(op.input_1.swizzling));
    json_op["output_0_datatype"] = "DataType." + mapToText<DataType>().at(static_cast<int>(op.output_0.datatype));
    json_op["output_0_layout"] = "Layout." + mapToText<Layout>().at(static_cast<int>(op.output_0.layout));
    json_op["output_0_swizzling"] = "Swizzling." + mapToText<Swizzling>().at(static_cast<int>(op.output_0.swizzling));

    json_op["isi_strategy"] = "ISIStrategy." + mapToText<ISIStrategy>().at(static_cast<int>(op.isi_strategy));
    json_op["output_write_tiles"] = op.output_write_tiles;

    json_op["in_place_input1"] = static_cast<int>(op.weightless_operation);
    json_op["in_place_output"] = static_cast<int>(op.in_place_output_memory);
    json_op["superdense_output"] = static_cast<int>(op.superdense);
    json_op["input_autopad"] = static_cast<int>(op.input_autopad);
    json_op["output_autopad"] = static_cast<int>(op.output_autopad);
    json_op["workload_uid"] = op.hash();

    // input_0_halo fields (TBLRFB)
    json_op["input_0_halo_top"] = op.halo.input_0_halo.top;
    json_op["input_0_halo_bottom"] = op.halo.input_0_halo.bottom;
    json_op["input_0_halo_left"] = op.halo.input_0_halo.left;
    json_op["input_0_halo_right"] = op.halo.input_0_halo.right;
    json_op["input_0_halo_front"] = op.halo.input_0_halo.front;
    json_op["input_0_halo_back"] = op.halo.input_0_halo.back;

    // output_0_halo fields (TBLRFB)
    json_op["output_0_halo_top"] = op.halo.output_0_halo.top;
    json_op["output_0_halo_bottom"] = op.halo.output_0_halo.bottom;
    json_op["output_0_halo_left"] = op.halo.output_0_halo.left;
    json_op["output_0_halo_right"] = op.halo.output_0_halo.right;
    json_op["output_0_halo_front"] = op.halo.output_0_halo.front;
    json_op["output_0_halo_back"] = op.halo.output_0_halo.back;

    // output_0_halo_broadcast_cnt fields (TBLRFB)
    json_op["output_0_halo_broadcast_top"] = op.halo.output_0_halo_broadcast_cnt.top;
    json_op["output_0_halo_broadcast_bottom"] = op.halo.output_0_halo_broadcast_cnt.bottom;
    json_op["output_0_halo_broadcast_left"] = op.halo.output_0_halo_broadcast_cnt.left;
    json_op["output_0_halo_broadcast_right"] = op.halo.output_0_halo_broadcast_cnt.right;
    json_op["output_0_halo_broadcast_front"] = op.halo.output_0_halo_broadcast_cnt.front;
    json_op["output_0_halo_broadcast_back"] = op.halo.output_0_halo_broadcast_cnt.back;

    // output_0_inbound_halo fields (TBLRFB)
    json_op["output_0_halo_inbound_top"] = op.halo.output_0_inbound_halo.top;
    json_op["output_0_halo_inbound_bottom"] = op.halo.output_0_inbound_halo.bottom;
    json_op["output_0_halo_inbound_left"] = op.halo.output_0_inbound_halo.left;
    json_op["output_0_halo_inbound_right"] = op.halo.output_0_inbound_halo.right;
    json_op["output_0_halo_inbound_front"] = op.halo.output_0_inbound_halo.front;
    json_op["output_0_halo_inbound_back"] = op.halo.output_0_inbound_halo.back;

    // aspect to do: why op.mpe_engine is not here? is it condensed by execution_order?

    // reduce_minmax_op
    json_op["minmax"] = static_cast<int>(op.reduce_minmax_op);
    json_op["wcb_mode"] = static_cast<int>(0);

    return json_op;
}

template <>
nlohmann::json toJson<DMANNWorkload_NPU27>(const DMANNWorkload_NPU27& wl) {
    nlohmann::json payload;
    payload["device"] = enumName<VPUDevice>() + "." + mapToText<VPUDevice>().at(static_cast<int>(wl.device));
    payload["num_planes"] = wl.num_planes;
    payload["length"] = wl.length;
    payload["src_width"] = wl.src_width;
    payload["dst_width"] = wl.dst_width;
    payload["src_stride"] = wl.src_stride;
    payload["dst_stride"] = wl.dst_stride;
    payload["src_plane_stride"] = wl.src_plane_stride;
    payload["dst_plane_stride"] = wl.dst_plane_stride;
    payload["transfer_direction"] = enumName<MemoryDirection>() + "." +
                                    mapToText<MemoryDirection>().at(static_cast<int>(wl.transfer_direction));
    return payload;
}

template <>
nlohmann::json toJson<DMANNWorkload_NPU40_50>(const DMANNWorkload_NPU40_50& wl) {
    nlohmann::json payload;
    payload["device"] = enumName<VPUDevice>() + "." + mapToText<VPUDevice>().at(static_cast<int>(wl.device));
    payload["src_width"] = wl.src_width;
    payload["dst_width"] = wl.dst_width;
    payload["num_dim"] = wl.num_dim;

    payload["src_stride_1"] = wl.e_dim[0].src_stride;
    payload["dst_stride_1"] = wl.e_dim[0].dst_stride;
    payload["src_dim_size_1"] = wl.e_dim[0].src_dim_size;
    payload["dst_dim_size_1"] = wl.e_dim[0].dst_dim_size;

    payload["src_stride_2"] = wl.e_dim[1].src_stride;
    payload["dst_stride_2"] = wl.e_dim[1].dst_stride;
    payload["src_dim_size_2"] = wl.e_dim[1].src_dim_size;
    payload["dst_dim_size_2"] = wl.e_dim[1].dst_dim_size;

    payload["src_stride_3"] = wl.e_dim[2].src_stride;
    payload["dst_stride_3"] = wl.e_dim[2].dst_stride;
    payload["src_dim_size_3"] = wl.e_dim[2].src_dim_size;
    payload["dst_dim_size_3"] = wl.e_dim[2].dst_dim_size;

    payload["src_stride_4"] = wl.e_dim[3].src_stride;
    payload["dst_stride_4"] = wl.e_dim[3].dst_stride;
    payload["src_dim_size_4"] = wl.e_dim[3].src_dim_size;
    payload["dst_dim_size_4"] = wl.e_dim[3].dst_dim_size;

    payload["src_stride_5"] = wl.e_dim[4].src_stride;
    payload["dst_stride_5"] = wl.e_dim[4].dst_stride;
    payload["src_dim_size_5"] = wl.e_dim[4].src_dim_size;
    payload["dst_dim_size_5"] = wl.e_dim[4].dst_dim_size;

    // Python profiling server expects num_engine as 1-indexed (1 or 2); enum is 0-indexed.
    payload["num_engine"] = static_cast<int>(wl.num_engine) + 1;
    // Python profiling server expects "Direction.<value>" (e.g. "Direction.CMX2DDR")
    payload["direction"] = "Direction." + mapToText<MemoryDirection>().at(static_cast<int>(wl.transfer_direction));

    return payload;
}

template <>
nlohmann::json toJson<SHAVEWorkload>(const SHAVEWorkload& wl) {
    nlohmann::json payload;
    payload["device"] = enumName<VPUDevice>() + "." + mapToText<VPUDevice>().at(static_cast<int>(wl.get_device()));
    payload["operation"] = wl.get_name();

    const auto& wl_inputs = wl.get_inputs();
    for (size_t i = 0; i < wl_inputs.size(); ++i) {
        std::string idx_str = std::to_string(i);
        payload["input_" + idx_str + "_batch"] = wl_inputs[i].batches();
        payload["input_" + idx_str + "_channels"] = wl_inputs[i].channels();
        payload["input_" + idx_str + "_height"] = wl_inputs[i].height();
        payload["input_" + idx_str + "_width"] = wl_inputs[i].width();
        payload["input_" + idx_str + "_datatype"] = wl_inputs[i].get_dtype();
        payload["input_" + idx_str + "_layout"] = wl_inputs[i].get_layout();
        payload["input_" + idx_str + "_sparsity_enabled"] = wl_inputs[i].get_sparsity();
    }

    const auto& wl_outputs = wl.get_outputs();
    for (size_t i = 0; i < wl_outputs.size(); ++i) {
        std::string idx_str = std::to_string(i);
        payload["output_" + idx_str + "_batch"] = wl_outputs[i].batches();
        payload["output_" + idx_str + "_channels"] = wl_outputs[i].channels();
        payload["output_" + idx_str + "_height"] = wl_outputs[i].height();
        payload["output_" + idx_str + "_width"] = wl_outputs[i].width();
        payload["output_" + idx_str + "_datatype"] = wl_outputs[i].get_dtype();
        payload["output_" + idx_str + "_layout"] = wl_outputs[i].get_layout();
        payload["output_" + idx_str + "_sparsity_enabled"] = wl_outputs[i].get_sparsity();
    }

    const auto paramToString = [](const SHAVEWorkload::Param& p) -> std::string {
        if (const int* pvalInt = std::get_if<int>(&p)) {
            return std::to_string(*pvalInt);
        } else if (const float* pvalFloat = std::get_if<float>(&p)) {
            return std::to_string(*pvalFloat);
        } else if (const std::string* pvalString = std::get_if<std::string>(&p)) {
            return *pvalString;
        } else if (const bool* pvalBool = std::get_if<bool>(&p)) {
            return *pvalBool ? std::string("true") : std::string("false");
        }
        return std::string("");
    };

    const auto get_param_strings = [&](const SHAVEWorkload::Parameters& call_params) -> std::vector<std::string> {
        std::vector<std::string> param_strs;
        for (const auto& param : call_params) {
            param_strs.push_back(paramToString(param));
        }
        return param_strs;
    };

    const auto get_extra_param_strings =
            [&](const SHAVEWorkload::ExtraParameters& extra_params) -> std::vector<std::string> {
        std::vector<std::string> extra_param_strings;
        for (const auto& [key, value] : extra_params) {
            extra_param_strings.push_back(key + "/" + paramToString(value));
        }
        return extra_param_strings;
    };

    const auto& wl_extras = get_param_strings(wl.get_params());
    for (size_t i = 0; i < wl_extras.size(); ++i) {
        payload["param_" + std::to_string(i)] = wl_extras[i];
    }
    const auto& wl_extra_params = get_extra_param_strings(wl.get_extra_params());
    for (size_t i = 0; i < wl_extra_params.size(); ++i) {
        payload["extra_param_" + std::to_string(i)] = wl_extra_params[i];
    }

    return payload;
}

}  // namespace VPUNN
