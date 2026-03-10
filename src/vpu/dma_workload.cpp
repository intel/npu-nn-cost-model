// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#include <tuple>
#include "core/utils.h"
#include "vpu/dma_types.h"

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
template <typename T>
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

// DMANNWorkload_NPU27 operators

bool DMANNWorkload_NPU27::operator==(const DMANNWorkload_NPU27& b) const {
    return device == b.device && num_planes == b.num_planes && length == b.length && src_width == b.src_width &&
           dst_width == b.dst_width && src_stride == b.src_stride && dst_stride == b.dst_stride &&
           src_plane_stride == b.src_plane_stride && dst_plane_stride == b.dst_plane_stride &&
           transfer_direction == b.transfer_direction && loc_name == b.loc_name;
}

bool DMANNWorkload_NPU27::operator<(const DMANNWorkload_NPU27& b) const {
    return std::tie(device, num_planes, length, src_width, dst_width, src_stride, dst_stride, src_plane_stride,
                    dst_plane_stride, transfer_direction, loc_name) <
           std::tie(b.device, b.num_planes, b.length, b.src_width, b.dst_width, b.src_stride, b.dst_stride,
                    b.src_plane_stride, b.dst_plane_stride, b.transfer_direction, b.loc_name);
}

// DMANNWorkload_NPU40_50 operators

bool DMANNWorkload_NPU40_50::operator==(const DMANNWorkload_NPU40_50& b) const {
    // Compare device, widths, num_dim, and num_engine
    if (device != b.device || src_width != b.src_width || dst_width != b.dst_width || num_dim != b.num_dim ||
        num_engine != b.num_engine || transfer_direction != b.transfer_direction) {
        return false;
    }

    // Compare all extra dimensions up to num_dim
    for (int i = 0; i < num_dim; i++) {
        if (e_dim[i].src_stride != b.e_dim[i].src_stride || e_dim[i].dst_stride != b.e_dim[i].dst_stride ||
            e_dim[i].src_dim_size != b.e_dim[i].src_dim_size || e_dim[i].dst_dim_size != b.e_dim[i].dst_dim_size) {
            return false;
        }
    }

    return true;
}

bool DMANNWorkload_NPU40_50::operator<(const DMANNWorkload_NPU40_50& b) const {
    return std::tie(device, src_width, dst_width, num_dim, e_dim, num_engine, transfer_direction, loc_name) <
           std::tie(b.device, b.src_width, b.dst_width, b.num_dim, b.e_dim, b.num_engine, b.transfer_direction,
                    b.loc_name);
}

}  // namespace VPUNN
