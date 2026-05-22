// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/workload_hash.h"

#include "core/primitive_hash.h"

namespace VPUNN {

uint32_t WorkloadHash::hash_tensor(uint32_t h, const VPUTensor& tensor) {
    // Hash shape array
    const auto& shape = tensor.get_shape();
    for (const auto& dim : shape) {
        h = PrimitiveHash::hash_uint32(h, dim);
    }
    // Hash data type, layout, and sparsity
    h = PrimitiveHash::hash_enum(h, tensor.get_dtype());
    h = PrimitiveHash::hash_enum(h, tensor.get_layout());
    h = PrimitiveHash::hash_bool(h, tensor.get_sparsity());
    return h;
}

uint32_t WorkloadHash::hash_halo(uint32_t h, const HaloWorkload& halo) {
    // Hash input_0_halo info
    h = PrimitiveHash::hash_uint32(h, halo.input_0_halo.top);
    h = PrimitiveHash::hash_uint32(h, halo.input_0_halo.bottom);
    h = PrimitiveHash::hash_uint32(h, halo.input_0_halo.left);
    h = PrimitiveHash::hash_uint32(h, halo.input_0_halo.right);
    h = PrimitiveHash::hash_uint32(h, halo.input_0_halo.front);
    h = PrimitiveHash::hash_uint32(h, halo.input_0_halo.back);

    // Hash output_0_halo info
    h = PrimitiveHash::hash_uint32(h, halo.output_0_halo.top);
    h = PrimitiveHash::hash_uint32(h, halo.output_0_halo.bottom);
    h = PrimitiveHash::hash_uint32(h, halo.output_0_halo.left);
    h = PrimitiveHash::hash_uint32(h, halo.output_0_halo.right);
    h = PrimitiveHash::hash_uint32(h, halo.output_0_halo.front);
    h = PrimitiveHash::hash_uint32(h, halo.output_0_halo.back);

    // Hash output_0_halo_broadcast_cnt info
    h = PrimitiveHash::hash_uint32(h, halo.output_0_halo_broadcast_cnt.top);
    h = PrimitiveHash::hash_uint32(h, halo.output_0_halo_broadcast_cnt.bottom);
    h = PrimitiveHash::hash_uint32(h, halo.output_0_halo_broadcast_cnt.left);
    h = PrimitiveHash::hash_uint32(h, halo.output_0_halo_broadcast_cnt.right);
    h = PrimitiveHash::hash_uint32(h, halo.output_0_halo_broadcast_cnt.front);
    h = PrimitiveHash::hash_uint32(h, halo.output_0_halo_broadcast_cnt.back);

    // Hash output_0_inbound_halo info
    h = PrimitiveHash::hash_uint32(h, halo.output_0_inbound_halo.top);
    h = PrimitiveHash::hash_uint32(h, halo.output_0_inbound_halo.bottom);
    h = PrimitiveHash::hash_uint32(h, halo.output_0_inbound_halo.left);
    h = PrimitiveHash::hash_uint32(h, halo.output_0_inbound_halo.right);
    h = PrimitiveHash::hash_uint32(h, halo.output_0_inbound_halo.front);
    h = PrimitiveHash::hash_uint32(h, halo.output_0_inbound_halo.back);

    return h;
}

uint32_t WorkloadHash::hash_sep(uint32_t h, const SEPModeInfo& sep) {
    // Hash SEP activators flag
    h = PrimitiveHash::hash_bool(h, sep.sep_activators);
    // Hash SEP storage_elements_pointers shape
    h = PrimitiveHash::hash_uint32(h, sep.storage_elements_pointers.width());
    h = PrimitiveHash::hash_uint32(h, sep.storage_elements_pointers.height());
    h = PrimitiveHash::hash_uint32(h, sep.storage_elements_pointers.channels());
    h = PrimitiveHash::hash_uint32(h, sep.storage_elements_pointers.batches());
    // Hash actual_activators_input shape
    h = PrimitiveHash::hash_uint32(h, sep.actual_activators_input.width());
    h = PrimitiveHash::hash_uint32(h, sep.actual_activators_input.height());
    h = PrimitiveHash::hash_uint32(h, sep.actual_activators_input.channels());
    h = PrimitiveHash::hash_uint32(h, sep.actual_activators_input.batches());
    // Hash no_sparse_map flag
    h = PrimitiveHash::hash_bool(h, sep.no_sparse_map);
    return h;
}

uint32_t WorkloadHash::hash_param(uint32_t h, const SHAVEWorkload::Param& p) {
    // Hash the variant index first so different types with same numeric value differ
    h = PrimitiveHash::hash_uint32(h, static_cast<uint32_t>(p.index()));
    std::visit(
            [&h](auto&& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, int>) {
                    h = PrimitiveHash::hash_uint32(h, static_cast<uint32_t>(arg));
                } else if constexpr (std::is_same_v<T, float>) {
                    h = PrimitiveHash::hash_float(h, arg);
                } else if constexpr (std::is_same_v<T, std::string>) {
                    h = PrimitiveHash::hash_string(h, arg);
                } else if constexpr (std::is_same_v<T, bool>) {
                    h = PrimitiveHash::hash_bool(h, arg);
                } else {
                    // should not compile if a new type is added to the variant and not handled here
                    static_assert(sizeof(T) == 0, "Unhandled type in SHAVEWorkload::Param variant");
                }
            },
            p);
    return h;
}

}  // namespace VPUNN