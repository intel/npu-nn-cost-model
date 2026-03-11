// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/dpu_workload.h"
#include <iostream>
#include "core/utils.h"

namespace VPUNN {

/// Helper function to hash a single uint32_t value using FNV-1a
static inline uint32_t hash_uint32(uint32_t h, uint32_t value) {
    h = (h ^ (value & 0xFF)) * fnv_prime;
    h = (h ^ ((value >> 8) & 0xFF)) * fnv_prime;
    h = (h ^ ((value >> 16) & 0xFF)) * fnv_prime;
    h = (h ^ (value >> 24)) * fnv_prime;
    return h;
}

/// Helper function to hash a float value with fractional rescaling
static inline uint32_t hash_float(uint32_t h, float c) {
    float scaled_value = c * 100.0f;
    uint32_t value =
            (c < 1.0f && c > -1.0f && c != 0.0f) ? static_cast<uint32_t>(scaled_value) : static_cast<uint32_t>(c);
    return hash_uint32(h, value);
}

/// Helper function to hash an enum value
template <typename T>
static inline uint32_t hash_enum(uint32_t h, T value) {
    return hash_uint32(h, static_cast<uint32_t>(value));
}

/// Helper function to hash a boolean value
static inline uint32_t hash_bool(uint32_t h, bool value) {
    return hash_uint32(h, value ? 1 : 0);
}

/// Helper function to hash a VPUTensor
static inline uint32_t hash_tensor(uint32_t h, const VPUTensor& tensor) {
    // Hash shape array
    const auto& shape = tensor.get_shape();
    for (const auto& dim : shape) {
        h = hash_uint32(h, dim);
    }
    // Hash data type, layout, and sparsity
    h = hash_enum(h, tensor.get_dtype());
    h = hash_enum(h, tensor.get_layout());
    h = hash_bool(h, tensor.get_sparsity());
    return h;
}

/// Helper function to hash a HaloWorkload
static inline uint32_t hash_halo(uint32_t h, const HaloWorkload& halo) {
    // Hash input_0_halo info
    h = hash_uint32(h, halo.input_0_halo.top);
    h = hash_uint32(h, halo.input_0_halo.bottom);
    h = hash_uint32(h, halo.input_0_halo.left);
    h = hash_uint32(h, halo.input_0_halo.right);
    h = hash_uint32(h, halo.input_0_halo.front);
    h = hash_uint32(h, halo.input_0_halo.back);

    // Hash output_0_halo info
    h = hash_uint32(h, halo.output_0_halo.top);
    h = hash_uint32(h, halo.output_0_halo.bottom);
    h = hash_uint32(h, halo.output_0_halo.left);
    h = hash_uint32(h, halo.output_0_halo.right);
    h = hash_uint32(h, halo.output_0_halo.front);
    h = hash_uint32(h, halo.output_0_halo.back);

    // Hash output_0_halo_broadcast_cnt info
    h = hash_uint32(h, halo.output_0_halo_broadcast_cnt.top);
    h = hash_uint32(h, halo.output_0_halo_broadcast_cnt.bottom);
    h = hash_uint32(h, halo.output_0_halo_broadcast_cnt.left);
    h = hash_uint32(h, halo.output_0_halo_broadcast_cnt.right);
    h = hash_uint32(h, halo.output_0_halo_broadcast_cnt.front);
    h = hash_uint32(h, halo.output_0_halo_broadcast_cnt.back);

    // Hash output_0_inbound_halo info
    h = hash_uint32(h, halo.output_0_inbound_halo.top);
    h = hash_uint32(h, halo.output_0_inbound_halo.bottom);
    h = hash_uint32(h, halo.output_0_inbound_halo.left);
    h = hash_uint32(h, halo.output_0_inbound_halo.right);
    h = hash_uint32(h, halo.output_0_inbound_halo.front);
    h = hash_uint32(h, halo.output_0_inbound_halo.back);

    return h;
}

/// Helper function to hash a SEPModeInfo
static inline uint32_t hash_sep(uint32_t h, const SEPModeInfo& sep) {
    // Hash SEP activators flag
    h = hash_bool(h, sep.sep_activators);
    // Hash SEP storage_elements_pointers shape
    h = hash_uint32(h, sep.storage_elements_pointers.width());
    h = hash_uint32(h, sep.storage_elements_pointers.height());
    h = hash_uint32(h, sep.storage_elements_pointers.channels());
    h = hash_uint32(h, sep.storage_elements_pointers.batches());
    // Hash actual_activators_input shape
    h = hash_uint32(h, sep.actual_activators_input.width());
    h = hash_uint32(h, sep.actual_activators_input.height());
    h = hash_uint32(h, sep.actual_activators_input.channels());
    h = hash_uint32(h, sep.actual_activators_input.batches());
    // Hash no_sparse_map flag
    h = hash_bool(h, sep.no_sparse_map);
    return h;
}

/// Helper function to hash an optional value
template <typename T>
static inline uint32_t hash_optional(uint32_t h, const std::optional<T>& opt) {
    if (opt.has_value()) {
        h = hash_bool(h, true);
        if constexpr (std::is_enum_v<T>) {
            h = hash_enum(h, opt.value());
        } else if constexpr (std::is_same_v<T, bool>) {
            h = hash_bool(h, opt.value());
        } else {
            h = hash_uint32(h, static_cast<uint32_t>(opt.value()));
        }
    } else {
        h = hash_bool(h, false);
    }
    return h;
}

uint32_t DPUWorkload::hash() const noexcept {
    uint32_t h = fnv_offset_basis;

    // Hash basic enums
    h = hash_enum(h, device);
    h = hash_enum(h, op);

    // Hash input and output tensors
    for (const auto& input : inputs) {
        h = hash_tensor(h, input);
    }
    for (const auto& output : outputs) {
        h = hash_tensor(h, output);
    }

    // Hash kernels, strides, and padding arrays
    for (const auto& k : kernels) {
        h = hash_uint32(h, k);
    }
    for (const auto& s : strides) {
        h = hash_uint32(h, s);
    }
    for (const auto& p : padding) {
        h = hash_uint32(h, p);
    }

    // Hash execution order and activation function
    h = hash_enum(h, execution_order);
    h = hash_enum(h, activation_function);

    // Hash sparsity values (with fractional rescaling)
    h = hash_float(h, act_sparsity);
    h = hash_float(h, weight_sparsity);

    // Hash swizzling arrays
    for (const auto& swizz : input_swizzling) {
        h = hash_enum(h, swizz);
    }
    for (const auto& swizz : output_swizzling) {
        h = hash_enum(h, swizz);
    }

    // Hash output write tiles
    h = hash_uint32(h, output_write_tiles);

    // Offsets are NOT included in hash (used only in intratile splitting, not for cache key)

    // Hash ISI strategy
    h = hash_enum(h, isi_strategy);

    // Hash weight sparsity enabled flag
    h = hash_bool(h, weight_sparsity_enabled);

    // Hash halo and SEP
    h = hash_halo(h, halo);
    h = hash_sep(h, sep_activators);

    // Hash optional weight type
    h = hash_enum(h, get_weight_type());

    // layer_info is NOT included in hash (for cache purposes)

    // Hash optional fields
    h = hash_bool(h, is_weightless_operation());
    h = hash_bool(h, is_inplace_output_memory());
    h = hash_bool(h, is_superdense());
    h = hash_bool(h, is_input_autopad());
    h = hash_bool(h, is_output_autopad());

    // Hash MPE engine
    h = hash_enum(h, mpe_engine);

    // Hash reduce_minmax_op flag
    h = hash_bool(h, reduce_minmax_op);  // makes old cache invalid. remove temporary if you need old cache and do not
                                         // use the feature

    return h;
}

bool DPUWorkload::operator==(const DPUWorkload& b) const {
    bool r{true};
    r = r && (device == b.device);
    r = r && (op == b.op);
    r = r && (inputs == b.inputs);
    r = r && (outputs == b.outputs);

    r = r && (kernels == b.kernels);
    r = r && (strides == b.strides);
    r = r && (padding == b.padding);

    r = r && (execution_order == b.execution_order);
    r = r && (activation_function == b.activation_function);

    const float EPSILON{0.00001f};
    auto is_equal = [&EPSILON](float a, float b) {
        return (std::fabs(a - b) < EPSILON);  // very simple since vals around zero
    };
    r = r && (is_equal(act_sparsity, b.act_sparsity));
    r = r && (is_equal(weight_sparsity, b.weight_sparsity));

    r = r && (input_swizzling == b.input_swizzling);
    r = r && (output_swizzling == b.output_swizzling);

    r = r && (output_write_tiles == b.output_write_tiles);
    r = r && (isi_strategy == b.isi_strategy);
    r = r && (weight_sparsity_enabled == b.weight_sparsity_enabled);

    // new halo
    r = r && (halo == b.halo);
    r = r && (sep_activators == b.sep_activators);  // sep

    r = r && (weight_type == b.weight_type);  // weight type

    r = r && (layer_info == b.layer_info);  // layer_info

    r = r && (weightless_operation == b.weightless_operation);      // weightless_operation
    r = r && (in_place_output_memory == b.in_place_output_memory);  // in_place_output_memory
    r = r && (superdense_memory == b.superdense_memory);            // superdense_memory
    r = r && (input_autopad == b.input_autopad);                    // input_autopad
    r = r && (output_autopad == b.output_autopad);                  // output_autopads
    r = r && (mpe_engine == b.mpe_engine);                          // mpe_engine
    r = r && (reduce_minmax_op == b.reduce_minmax_op);              // reduce_minmax_op
    return r;
}

bool DPUWorkload::operator<(const DPUWorkload& b) const {
    // Compare fields in order to establish a total ordering
    if (!(device == b.device))
        return device < b.device;
    if (!(op == b.op))
        return op < b.op;
    if (!(inputs == b.inputs))
        return inputs < b.inputs;
    if (!(outputs == b.outputs))
        return outputs < b.outputs;
    if (!(kernels == b.kernels))
        return kernels < b.kernels;
    if (!(strides == b.strides))
        return strides < b.strides;
    if (!(padding == b.padding))
        return padding < b.padding;
    if (!(execution_order == b.execution_order))
        return execution_order < b.execution_order;
    if (!(activation_function == b.activation_function))
        return activation_function < b.activation_function;
    if (!(act_sparsity == b.act_sparsity))
        return act_sparsity < b.act_sparsity;
    if (!(weight_sparsity == b.weight_sparsity))
        return weight_sparsity < b.weight_sparsity;
    if (!(input_swizzling == b.input_swizzling))
        return input_swizzling < b.input_swizzling;
    if (!(output_swizzling == b.output_swizzling))
        return output_swizzling < b.output_swizzling;
    if (!(output_write_tiles == b.output_write_tiles))
        return output_write_tiles < b.output_write_tiles;
    if (!(isi_strategy == b.isi_strategy))
        return isi_strategy < b.isi_strategy;
    if (!(weight_sparsity_enabled == b.weight_sparsity_enabled))
        return weight_sparsity_enabled < b.weight_sparsity_enabled;
    if (!(halo == b.halo))
        return halo < b.halo;
    if (!(sep_activators == b.sep_activators))
        return sep_activators < b.sep_activators;
    if (!(weight_type == b.weight_type))
        return weight_type < b.weight_type;
    // layer_info excluded from comparison for cache purposes
    if (!(weightless_operation == b.weightless_operation))
        return weightless_operation < b.weightless_operation;
    if (!(in_place_output_memory == b.in_place_output_memory))
        return in_place_output_memory < b.in_place_output_memory;
    if (!(superdense_memory == b.superdense_memory))
        return superdense_memory < b.superdense_memory;
    if (!(input_autopad == b.input_autopad))
        return input_autopad < b.input_autopad;
    if (!(output_autopad == b.output_autopad))
        return output_autopad < b.output_autopad;
    if (!(mpe_engine == b.mpe_engine))
        return mpe_engine < b.mpe_engine;
    if (!(reduce_minmax_op == b.reduce_minmax_op))
        return reduce_minmax_op < b.reduce_minmax_op;
    return false;  // All fields are equal
}

std::ostream& operator<<(std::ostream& stream, const VPUNN::DPUWorkload& d) {
    stream << "Workload: \n"                                                                                        //
           << " device: \t" << (int)d.device << " : " << VPUDevice_ToText.at(static_cast<int>(d.device)) << " ;\n"  //
           << " Operation: \t" << (int)d.op << " : " << Operation_ToText.at(static_cast<int>(d.op))
           << " ;\n"  //

           // inputs and outputs tensors
           << " input: \t{\n"
           << d.inputs[0] << " } ;\n"  //
           << " output: \t{\n"
           << d.outputs[0] << " } ;\n"  //

           << " kernels: [W,H]  \t{" << d.kernels[Dim::Grid::W] << "," << d.kernels[Dim::Grid::H] << "} ;\n"  //
           << " strides: [W,H]  \t{" << d.strides[Dim::Grid::W] << "," << d.strides[Dim::Grid::H] << "} ;\n"  //
           << " padding: [TBLR] \t{" << d.padding[Dim::TOP] << "," << d.padding[Dim::BOTTOM] << ","           //
           << d.padding[Dim::LEFT] << "," << d.padding[Dim::RIGHT] << "} ;\n"                                 //

           << " execution_order: \t" << (int)d.execution_order << " : "
           << ExecutionMode_ToText.at(static_cast<int>(d.execution_order)) << " ;\n"  //
           << " activation_function: \t" << (int)d.activation_function << " : "
           << ActivationFunction_ToText.at(static_cast<int>(d.activation_function)) << " ;\n"  //

           << " act_sparsity: \t" << d.act_sparsity << " ;\n"        //
           << " weight_sparsity: \t" << d.weight_sparsity << " ;\n"  //

           << " input_swizzling: \t{" << (int)d.input_swizzling[0] << "," << (int)d.input_swizzling[1] << "}"
           << " :  {" << Swizzling_ToText.at(static_cast<int>(d.input_swizzling[0])) << ", "
           << Swizzling_ToText.at(static_cast<int>(d.input_swizzling[1])) << "} ;\n"  //

           << " output_swizzling: \t{" << (int)d.output_swizzling[0] << "}"
           << " :  {" << Swizzling_ToText.at(static_cast<int>(d.output_swizzling[0])) << "} ;\n"  //

           << " output_write_tiles: \t" << d.output_write_tiles << " ;\n"    //
           << " offsets: \t{" << d.offsets[0] << "," << d.offsets[1] << ","  //
           << d.offsets[2] << "," << d.offsets[3] << "} ;\n"                 //
           << " isi_strategy: \t" << (int)d.isi_strategy << " : "
           << ISIStrategy_ToText.at(static_cast<int>(d.isi_strategy)) << " ;\n"  //
           << " weight_sparsity_enabled: \t" << (int)d.weight_sparsity_enabled << " : "
           << (d.weight_sparsity_enabled ? "true" : "false") << " ;\n"  //
           << d.halo            //<< " ;\n"                                          //
           << d.sep_activators  //<< " ;\n"                                          //
           << " weight_type: \t"
           << (d.weight_type.has_value() ? DataType_ToText.at(static_cast<int>(d.weight_type.value())) : "Same")
           << " ;\n"  //
           //<< "layer_info:" << d.layer_info << " ;\n" //  keep out since affects layer hash (until layer is more
           // decoupled)
           << " weightless_operation/in_place_input1: \t"
           << (d.weightless_operation.has_value() ? std::to_string(d.weightless_operation.value()) : "NA") << " ;\n"  //
           << " in_place_output_memory: \t"
           << (d.in_place_output_memory.has_value() ? std::to_string(d.in_place_output_memory.value()) : "NA") << " ;\n"
           << " superdense_memory: \t"
           << (d.superdense_memory.has_value() ? std::to_string(d.superdense_memory.value()) : "NA") << " ;\n"  //
           << (d.input_autopad.has_value() ? " input_autopad: \t" + std::to_string(d.input_autopad.value())
                                           : " input_autopad: NA")
           << " ;\n"  //
           << (d.output_autopad.has_value() ? " output_autopad: \t" + std::to_string(d.output_autopad.value())
                                            : " output_autopad: NA")
           << " ;\n"  //
           << " mpe engine: " << (int)d.mpe_engine << " : " << MPEEngine_ToText.at(static_cast<int>(d.mpe_engine))
           << " ;\n"                                                                        //
           << " reduce_minmax_op: \t" << (d.reduce_minmax_op ? "true" : "false") << " ;\n"  //
           << out_terminator() << "Workload "                                               // terminator
            ;
    return stream;
}

}  // namespace VPUNN
