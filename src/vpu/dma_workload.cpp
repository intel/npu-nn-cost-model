// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/dma_types.h"
#include "core/utils.h"

namespace VPUNN {

/// Helper function to hash a single uint32_t value using FNV-1a
/// TODO: should this be moved to core/utils.h to be used also in vpu/dpu_workload.cpp ?
static inline uint32_t hash_uint32(uint32_t h, uint32_t value) {
    h = (h ^ (value & 0xFF)) * fnv_prime;
    h = (h ^ ((value >> 8) & 0xFF)) * fnv_prime;
    h = (h ^ ((value >> 16) & 0xFF)) * fnv_prime;
    h = (h ^ (value >> 24)) * fnv_prime;
    return h;
}

/// Helper function to hash an int value (handles negative values properly)
static inline uint32_t hash_int(uint32_t h, int value) {
    return hash_uint32(h, static_cast<uint32_t>(value));
}

/// Helper function to hash an enum value
template<typename T>
static inline uint32_t hash_enum(uint32_t h, T value) {
    return hash_uint32(h, static_cast<uint32_t>(value));
}

uint32_t DMANNWorkload_NPU27::hash() const {
    uint32_t h = fnv_offset_basis;
    
    // Hash device
    h = hash_enum(h, device);
    
    // Hash all int fields
    h = hash_int(h, num_planes);
    h = hash_int(h, length);
    h = hash_int(h, src_width);
    h = hash_int(h, dst_width);
    h = hash_int(h, src_stride);
    h = hash_int(h, dst_stride);
    h = hash_int(h, src_plane_stride);
    h = hash_int(h, dst_plane_stride);
    
    // Hash transfer direction
    h = hash_enum(h, transfer_direction);
    
    // Note: loc_name is NOT included in hash (it's for debugging/logging, not a cache key)
    
    return h;
}

uint32_t DMANNWorkload_NPU40_50::hash() const {
    uint32_t h = fnv_offset_basis;
    
    // Hash device
    h = hash_enum(h, device);
    
    // Hash widths and num_dim
    h = hash_int(h, src_width);
    h = hash_int(h, dst_width);
    h = hash_int(h, num_dim);
    
    // Hash all extra dimensions (all 5, even if not used)
    for (int i = 0; i < num_dim; i++) {
        h = hash_int(h, e_dim[i].src_stride);
        h = hash_int(h, e_dim[i].dst_stride);
        h = hash_int(h, e_dim[i].src_dim_size);
        h = hash_int(h, e_dim[i].dst_dim_size);
    }
    
    // Hash num_engine
    h = hash_enum(h, num_engine);
    
    // Hash transfer direction
    h = hash_enum(h, transfer_direction);
    
    // Note: loc_name is NOT included in hash (it's for debugging/logging, not a cache key)
    
    return h;
}

}  // namespace VPUNN
