// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/dpu_workload.h"
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

uint32_t DPUWorkload::hash() const {
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

}  // namespace VPUNN
