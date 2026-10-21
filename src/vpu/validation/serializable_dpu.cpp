// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/validation/serializable_dpu.h"

#include "core/primitive_hash.h"
#include "vpu/dpu_dtypes_dimension_info.h"

namespace VPUNN {

SerializableDPU::SerializableDPU(const DPUWorkload& w): dpu_operation{w} {
}

SerializableDPU::SerializableDPU(const DPUOperation& op): dpu_operation{op} {
}

SerializableDPU::SerializableDPU(const DPUWorkload& w, const IDeviceValidValues& config): dpu_operation{w, config} {
}

SerializableDPU::SerializableDPU(const SerializableDPU& r): dpu_operation{r.dpu_operation} {
}

DPUWorkload SerializableDPU::to_DPUWorkload() const {
    return dpu_operation.clone_as_DPUWorkload();
}

DPUWorkload SerializableDPU::clone_as_DPUWorkload() const {
    return to_DPUWorkload();
}

const DPUOperation& SerializableDPU::dpu_operation_data() const {
    return dpu_operation;
}

DPUOperation& SerializableDPU::dpu_operation_data() {
    return dpu_operation;
}

const SerializableDPU::MemberMapType& SerializableDPU::get_member_map() const {
    ensure_member_map();
    return _member_map;
}

SerializableDPU::MemberMapType& SerializableDPU::get_member_map() {
    ensure_member_map();
    return _member_map;
}

void SerializableDPU::ensure_member_map() const {
    std::lock_guard<std::mutex> lock(_member_map_mutex);
    if (is_member_map_initialized()) {
        return;
    }
    (const_cast<SerializableDPU*>(this))->populate_member_map();
}

bool SerializableDPU::is_member_map_initialized() const {
    return !_member_map.empty();
}

void SerializableDPU::populate_member_map() {
    _member_map = MemberMapType{
            {"device", std::ref(dpu_operation.device)},
            {"operation", std::ref(dpu_operation.operation)},
            {"input_0_batch", std::ref(dpu_operation.input_0.batch)},
            {"input_0_channels", std::ref(dpu_operation.input_0.channels)},
            {"input_0_height", std::ref(dpu_operation.input_0.height)},
            {"input_0_width", std::ref(dpu_operation.input_0.width)},
            {"input_1_batch", std::ref(dpu_operation.input_1.batch)},
            {"input_1_channels", std::ref(dpu_operation.input_1.channels)},
            {"input_1_height", std::ref(dpu_operation.input_1.height)},
            {"input_1_width", std::ref(dpu_operation.input_1.width)},
            {"input_sparsity_enabled", std::ref(dpu_operation.input_0.sparsity_enabled)},
            {"weight_sparsity_enabled", std::ref(dpu_operation.input_1.sparsity_enabled)},
            {"input_sparsity_rate", std::ref(dpu_operation.input_0.sparsity)},
            {"weight_sparsity_rate", std::ref(dpu_operation.input_1.sparsity)},
            {"execution_order", std::ref(dpu_operation.execution_order)},
            {"activation_function", std::ref(dpu_operation.activation_function)},
            {"kernel_height", std::ref(dpu_operation.kernel.height)},
            {"kernel_width", std::ref(dpu_operation.kernel.width)},
            {"kernel_pad_bottom", std::ref(dpu_operation.kernel.pad_bottom)},
            {"kernel_pad_left", std::ref(dpu_operation.kernel.pad_left)},
            {"kernel_pad_right", std::ref(dpu_operation.kernel.pad_right)},
            {"kernel_pad_top", std::ref(dpu_operation.kernel.pad_top)},
            {"kernel_stride_height", std::ref(dpu_operation.kernel.stride_height)},
            {"kernel_stride_width", std::ref(dpu_operation.kernel.stride_width)},
            {"output_0_batch", std::ref(dpu_operation.output_0.batch)},
            {"output_0_channels", std::ref(dpu_operation.output_0.channels)},
            {"output_0_height", std::ref(dpu_operation.output_0.height)},
            {"output_0_width", std::ref(dpu_operation.output_0.width)},
            {"input_0_datatype", std::ref(dpu_operation.input_0.datatype)},
            {"input_0_layout", std::ref(dpu_operation.input_0.layout)},
            {"input_0_swizzling", std::ref(dpu_operation.input_0.swizzling)},
            {"input_1_datatype", std::ref(dpu_operation.input_1.datatype)},
            {"input_1_layout", std::ref(dpu_operation.input_1.layout)},
            {"input_1_swizzling", std::ref(dpu_operation.input_1.swizzling)},
            {"output_0_datatype", std::ref(dpu_operation.output_0.datatype)},
            {"output_0_layout", std::ref(dpu_operation.output_0.layout)},
            {"output_0_swizzling", std::ref(dpu_operation.output_0.swizzling)},
            {"output_sparsity_enabled", std::ref(dpu_operation.output_0.sparsity_enabled)},
            {"isi_strategy", std::ref(dpu_operation.isi_strategy)},
            {"output_write_tiles", std::ref(dpu_operation.output_write_tiles)},

            {"input_0_halo_top", std::ref(dpu_operation.halo.input_0_halo.top)},
            {"input_0_halo_bottom", std::ref(dpu_operation.halo.input_0_halo.bottom)},
            {"input_0_halo_left", std::ref(dpu_operation.halo.input_0_halo.left)},
            {"input_0_halo_right", std::ref(dpu_operation.halo.input_0_halo.right)},
            {"input_0_halo_front", std::ref(dpu_operation.halo.input_0_halo.front)},
            {"input_0_halo_back", std::ref(dpu_operation.halo.input_0_halo.back)},

            {"output_0_halo_top", std::ref(dpu_operation.halo.output_0_halo.top)},
            {"output_0_halo_bottom", std::ref(dpu_operation.halo.output_0_halo.bottom)},
            {"output_0_halo_left", std::ref(dpu_operation.halo.output_0_halo.left)},
            {"output_0_halo_right", std::ref(dpu_operation.halo.output_0_halo.right)},
            {"output_0_halo_front", std::ref(dpu_operation.halo.output_0_halo.front)},
            {"output_0_halo_back", std::ref(dpu_operation.halo.output_0_halo.back)},

            {"output_0_halo_broadcast_top", std::ref(dpu_operation.halo.output_0_halo_broadcast_cnt.top)},
            {"output_0_halo_broadcast_bottom", std::ref(dpu_operation.halo.output_0_halo_broadcast_cnt.bottom)},
            {"output_0_halo_broadcast_left", std::ref(dpu_operation.halo.output_0_halo_broadcast_cnt.left)},
            {"output_0_halo_broadcast_right", std::ref(dpu_operation.halo.output_0_halo_broadcast_cnt.right)},
            {"output_0_halo_broadcast_front", std::ref(dpu_operation.halo.output_0_halo_broadcast_cnt.front)},
            {"output_0_halo_broadcast_back", std::ref(dpu_operation.halo.output_0_halo_broadcast_cnt.back)},

            {"output_0_halo_inbound_top", std::ref(dpu_operation.halo.output_0_inbound_halo.top)},
            {"output_0_halo_inbound_bottom", std::ref(dpu_operation.halo.output_0_inbound_halo.bottom)},
            {"output_0_halo_inbound_left", std::ref(dpu_operation.halo.output_0_inbound_halo.left)},
            {"output_0_halo_inbound_right", std::ref(dpu_operation.halo.output_0_inbound_halo.right)},
            {"output_0_halo_inbound_front", std::ref(dpu_operation.halo.output_0_inbound_halo.front)},
            {"output_0_halo_inbound_back", std::ref(dpu_operation.halo.output_0_inbound_halo.back)},

            {"sep_enabled", std::ref(dpu_operation.sep_activators.sep_activators)},
            {"sep_w",
             [this](bool set_mode, const std::string& s) -> DimType {
                 if (set_mode) {
                     DimType value;
                     if (is_unsigned_int(s, value)) {
                         dpu_operation.sep_activators.storage_elements_pointers.set_width(value);
                     }
                 }
                 return dpu_operation.sep_activators.storage_elements_pointers.width();
             }},
            {"sep_h",
             [this](bool set_mode, const std::string& s) -> DimType {
                 if (set_mode) {
                     DimType value;
                     if (is_unsigned_int(s, value)) {
                         dpu_operation.sep_activators.storage_elements_pointers.set_height(value);
                     }
                 }
                 return dpu_operation.sep_activators.storage_elements_pointers.height();
             }},
            {"sep_c",
             [this](bool set_mode, const std::string& s) -> DimType {
                 if (set_mode) {
                     DimType value;
                     if (is_unsigned_int(s, value)) {
                         dpu_operation.sep_activators.storage_elements_pointers.set_channels(value);
                     }
                 }
                 return dpu_operation.sep_activators.storage_elements_pointers.channels();
             }},
            {"sep_b",
             [this](bool set_mode, const std::string& s) -> DimType {
                 if (set_mode) {
                     DimType value;
                     if (is_unsigned_int(s, value)) {
                         dpu_operation.sep_activators.storage_elements_pointers.set_batches(value);
                     }
                 }
                 return dpu_operation.sep_activators.storage_elements_pointers.batches();
             }},
            {"sep_act_w",
             [this](bool set_mode, const std::string& s) -> DimType {
                 if (set_mode) {
                     DimType value;
                     if (is_unsigned_int(s, value)) {
                         dpu_operation.sep_activators.actual_activators_input.set_width(value);
                     }
                 }
                 return dpu_operation.sep_activators.actual_activators_input.width();
             }},
            {"sep_act_h",
             [this](bool set_mode, const std::string& s) -> DimType {
                 if (set_mode) {
                     DimType value;
                     if (is_unsigned_int(s, value)) {
                         dpu_operation.sep_activators.actual_activators_input.set_height(value);
                     }
                 }
                 return dpu_operation.sep_activators.actual_activators_input.height();
             }},
            {"sep_act_c",
             [this](bool set_mode, const std::string& s) -> DimType {
                 if (set_mode) {
                     DimType value;
                     if (is_unsigned_int(s, value)) {
                         dpu_operation.sep_activators.actual_activators_input.set_channels(value);
                     }
                 }
                 return dpu_operation.sep_activators.actual_activators_input.channels();
             }},
            {"sep_act_b",
             [this](bool set_mode, const std::string& s) -> DimType {
                 if (set_mode) {
                     DimType value;
                     if (is_unsigned_int(s, value)) {
                         dpu_operation.sep_activators.actual_activators_input.set_batches(value);
                     }
                 }
                 return dpu_operation.sep_activators.actual_activators_input.batches();
             }},
            {"sep_no_sparse_map", std::ref(dpu_operation.sep_activators.no_sparse_map)},
            {"in_place_input1",
             [this](bool set_mode, const std::string& s) -> DimType {
                 if (set_mode) {
                     setWeightlessOperation(s);
                 }
                 return dpu_operation.weightless_operation;
             }},
            {"in_place_output",
             [this](bool set_mode, const std::string& s) -> DimType {
                 if (set_mode) {
                     setInPlaceOutputMemory(s);
                 }
                 return dpu_operation.in_place_output_memory;
             }},
            {"superdense_output", std::ref(dpu_operation.superdense)},
            {"input_autopad", std::ref(dpu_operation.input_autopad)},
            {"output_autopad", std::ref(dpu_operation.output_autopad)},
            {"mpe_engine", std::ref(dpu_operation.mpe_engine)},
            {"reduce_minmax_op", std::ref(dpu_operation.reduce_minmax_op)},
    };
}

const std::vector<std::string>& SerializableDPU::_get_member_names() {
    static const std::vector<std::string> names{
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
            "input_sparsity_enabled",
            "weight_sparsity_enabled",
            "input_sparsity_rate",
            "weight_sparsity_rate",
            "execution_order",
            "activation_function",
            "kernel_height",
            "kernel_width",
            "kernel_pad_bottom",
            "kernel_pad_left",
            "kernel_pad_right",
            "kernel_pad_top",
            "kernel_stride_height",
            "kernel_stride_width",
            "output_0_batch",
            "output_0_channels",
            "output_0_height",
            "output_0_width",
            "input_0_datatype",
            "input_0_layout",
            "input_0_swizzling",
            "input_1_datatype",
            "input_1_layout",
            "input_1_swizzling",
            "output_0_datatype",
            "output_0_layout",
            "output_0_swizzling",
            "output_sparsity_enabled",
            "isi_strategy",
            "output_write_tiles",
            "input_0_halo_top",
            "input_0_halo_bottom",
            "input_0_halo_left",
            "input_0_halo_right",
            "input_0_halo_front",
            "input_0_halo_back",
            "output_0_halo_top",
            "output_0_halo_bottom",
            "output_0_halo_left",
            "output_0_halo_right",
            "output_0_halo_front",
            "output_0_halo_back",
            "output_0_halo_broadcast_top",
            "output_0_halo_broadcast_bottom",
            "output_0_halo_broadcast_left",
            "output_0_halo_broadcast_right",
            "output_0_halo_broadcast_front",
            "output_0_halo_broadcast_back",
            "output_0_halo_inbound_top",
            "output_0_halo_inbound_bottom",
            "output_0_halo_inbound_left",
            "output_0_halo_inbound_right",
            "output_0_halo_inbound_front",
            "output_0_halo_inbound_back",
            "sep_enabled",
            "sep_w",
            "sep_h",
            "sep_c",
            "sep_b",
            "sep_act_w",
            "sep_act_h",
            "sep_act_c",
            "sep_act_b",
            "sep_no_sparse_map",
            "in_place_input1",
            "in_place_output",
            "superdense_output",
            "input_autopad",
            "output_autopad",
            "mpe_engine",
            "reduce_minmax_op",
    };

    return names;
}

size_t SerializableDPU::hash() const {
    uint32_t combined_hash = fnv_offset_basis;
    const auto& member_map = get_member_map();

    // Keep this visitor in sync with SerializableDPU::_ref_supported_type and populate_member_map().
    // If a new member value type is added, it must be explicitly hashed here as well.
    for (const auto& key : _get_member_names()) {
        const auto it = member_map.find(key);
        if (it == member_map.end()) {
            continue;
        }

        // Include field name so the hash remains tied to SerializableDPU's schema and order.
        combined_hash = PrimitiveHash::hash_string(combined_hash, key);

        std::visit(
                [&combined_hash](auto&& arg) {
                    using ArgType = std::decay_t<decltype(arg)>;

                    if constexpr (std::is_same_v<ArgType, VPUNN::SetGet_MemberMapValues>) {
                        // SetGet_MemberMapValues uses (set_mode, value). For hashing we need read-only behavior,
                        // so call get-mode: set_mode=false. The second argument is ignored in this mode.
                        combined_hash =
                                PrimitiveHash::hash_uint32(combined_hash, static_cast<uint32_t>(arg(false, "")));
                    } else {
                        using ValueType = std::decay_t<decltype(arg.get())>;
                        const ValueType& value = arg.get();

                        if constexpr (std::is_enum_v<ValueType>) {
                            combined_hash = PrimitiveHash::hash_enum(combined_hash, value);
                        } else if constexpr (std::is_same_v<ValueType, bool>) {
                            combined_hash = PrimitiveHash::hash_bool(combined_hash, value);
                        } else if constexpr (std::is_same_v<ValueType, float>) {
                            combined_hash = PrimitiveHash::hash_float(combined_hash, value);
                        } else if constexpr (std::is_integral_v<ValueType> && sizeof(ValueType) <= 4U) {
                            combined_hash = PrimitiveHash::hash_uint32(combined_hash, static_cast<uint32_t>(value));
                        } else if constexpr (std::is_integral_v<ValueType> && sizeof(ValueType) > 4U) {
                            const auto uvalue = static_cast<unsigned long long>(value);
                            combined_hash = PrimitiveHash::hash_uint32(combined_hash,
                                                                       static_cast<uint32_t>(uvalue & 0xFFFFFFFFU));
                            combined_hash = PrimitiveHash::hash_uint32(
                                    combined_hash, static_cast<uint32_t>((uvalue >> 32U) & 0xFFFFFFFFU));
                        } else {
                            static_assert(std::is_same_v<ValueType, void>,
                                          "Unsupported SerializableDPU member type for hashing");
                        }
                    }
                },
                it->second);
    }

    return static_cast<size_t>(combined_hash);
}

bool SerializableDPU::is_unsigned_int(const std::string& s, VPUNN::DimType& result) {
    if (s.empty())
        return false;

    int value;

    try {
        value = std::stoi(s);

        if (value < 0)
            return false;

    } catch (const std::invalid_argument&) {
        return false;
    } catch (const std::out_of_range&) {
        return false;
    }

    result = static_cast<unsigned int>(value);
    return true;
}

void SerializableDPU::setInPlaceOutputMemory(const std::string& s) {
    dpu_operation.in_place_output_memory = false;

    VPUNN::DimType value;
    if (is_unsigned_int(s, value)) {
        if (value != 0) {
            dpu_operation.in_place_output_memory = true;
        }
    } else {
        if (dpu_operation.is_elementwise_like_operation() && dpu_operation.is_preconditions_for_inplace_output()) {
            dpu_operation.in_place_output_memory = true;
        }
    }
}

void SerializableDPU::setWeightlessOperation(const std::string& s) {
    dpu_operation.weightless_operation = false;

    VPUNN::DimType value;
    if (is_unsigned_int(s, value)) {
        if (value != 0) {
            dpu_operation.weightless_operation = true;
        }
    } else {
        if (dpu_operation.is_elementwise_like_operation() && dpu_operation.is_special_No_weights_situation()) {
            dpu_operation.weightless_operation = true;
        }
    }
}

}  // namespace VPUNN
