// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/validation/serializable_dma_descriptor.h"

namespace VPUNN {

SerializableVPUDMADescriptor::SerializableVPUDMADescriptor(const VPUDMADescriptor& desc): dma_descriptor{desc} {
}

SerializableVPUDMADescriptor::SerializableVPUDMADescriptor(const SerializableVPUDMADescriptor& r)
        : dma_descriptor{r.dma_descriptor} {
}

const VPUDMADescriptor& SerializableVPUDMADescriptor::dma_descriptor_data() const {
    return dma_descriptor;
}

VPUDMADescriptor& SerializableVPUDMADescriptor::dma_descriptor_data() {
    return dma_descriptor;
}

const SerializableVPUDMADescriptor::MemberMapType& SerializableVPUDMADescriptor::get_member_map() const {
    ensure_member_map();
    return _member_map;
}

SerializableVPUDMADescriptor::MemberMapType& SerializableVPUDMADescriptor::get_member_map() {
    ensure_member_map();
    return _member_map;
}

void SerializableVPUDMADescriptor::ensure_member_map() const {
    std::lock_guard<std::mutex> lock(_member_map_mutex);
    if (is_member_map_initialized()) {
        return;
    }
    (const_cast<SerializableVPUDMADescriptor*>(this))->populate_member_map();
}

bool SerializableVPUDMADescriptor::is_member_map_initialized() const {
    return !_member_map.empty();
}

bool SerializableVPUDMADescriptor::is_non_negative_int(const std::string& s, int& result) {
    if (s.empty()) {
        return false;
    }
    try {
        const int parsed = std::stoi(s);
        if (parsed < 0) {
            return false;
        }
        result = parsed;
        return true;
    } catch (const std::invalid_argument&) {
        return false;
    } catch (const std::out_of_range&) {
        return false;
    }
}

void SerializableVPUDMADescriptor::populate_member_map() {
    // Per tensor we serialize:
    //   2 scalar fields: dtype, num_dims
    //   VPU_DMA_MAX_DIMS shape_i fields
    //   VPU_DMA_MAX_DIMS stride_i fields
    // => tensor_field_count = 2 + 2 * VPU_DMA_MAX_DIMS.
    // Full descriptor adds:
    //   1 device field
    //   2 tensor blocks (src + dst)
    //   2 location fields (src_location, dst_location)
    // => descriptor_field_count = 1 + 2 * tensor_field_count + 2.
    const std::size_t tensor_field_count = static_cast<std::size_t>(2 + (2 * VPU_DMA_MAX_DIMS));
    const std::size_t descriptor_field_count = static_cast<std::size_t>(1 + (2 * tensor_field_count) + 2);

    _member_map = MemberMapType{};
    _member_map.reserve(descriptor_field_count);
    _member_map.emplace("device", std::ref(dma_descriptor.device));

    auto add_tensor_members = [&](const std::string& prefix, VPUDMATensor& tensor,
                                  SetGet_MemberMapValues num_dims_accessor) {
        _member_map.emplace(prefix + "dtype", std::ref(tensor.dtype));
        _member_map.emplace(prefix + "num_dims", std::move(num_dims_accessor));
        for (int d = 0; d < VPU_DMA_MAX_DIMS; ++d) {
            _member_map.emplace(prefix + "shape_" + std::to_string(d), std::ref(tensor.shape[d]));
        }
        for (int d = 0; d < VPU_DMA_MAX_DIMS; ++d) {
            _member_map.emplace(prefix + "stride_" + std::to_string(d), std::ref(tensor.byte_strides[d]));
        }
    };

    add_tensor_members("src_", dma_descriptor.src, [this](bool set_mode, const std::string& s) -> DimType {
        if (set_mode) {
            int value = 0;
            if (is_non_negative_int(s, value)) {
                dma_descriptor.src.num_dims = value;
            }
        }
        return static_cast<DimType>(dma_descriptor.src.num_dims);
    });

    add_tensor_members("dst_", dma_descriptor.dst, [this](bool set_mode, const std::string& s) -> DimType {
        if (set_mode) {
            int value = 0;
            if (is_non_negative_int(s, value)) {
                dma_descriptor.dst.num_dims = value;
            }
        }
        return static_cast<DimType>(dma_descriptor.dst.num_dims);
    });

    _member_map.emplace("src_location", std::ref(dma_descriptor.src_location));
    _member_map.emplace("dst_location", std::ref(dma_descriptor.dst_location));
}

std::string SerializableVPUDMADescriptor::get_wl_name() {
    return "l1_dma_descriptor_workloads";
}

const std::vector<std::string>& SerializableVPUDMADescriptor::_get_member_names() {
    static const std::vector<std::string> names = []() {
        // Keep exactly the same field-count math as populate_member_map() so names and map stay in sync.
        // This ensures reserve() tracks schema shape as VPU_DMA_MAX_DIMS changes.
        const std::size_t tensor_field_count = static_cast<std::size_t>(2 + (2 * VPU_DMA_MAX_DIMS));
        const std::size_t descriptor_field_count = static_cast<std::size_t>(1 + (2 * tensor_field_count) + 2);

        std::vector<std::string> names_{};
        names_.reserve(descriptor_field_count);

        names_.emplace_back("device");

        auto add_tensor_member_names = [&](const std::string& prefix) {
            names_.emplace_back(prefix + "dtype");
            names_.emplace_back(prefix + "num_dims");
            for (int d = 0; d < VPU_DMA_MAX_DIMS; ++d) {
                names_.emplace_back(prefix + "shape_" + std::to_string(d));
            }
            for (int d = 0; d < VPU_DMA_MAX_DIMS; ++d) {
                names_.emplace_back(prefix + "stride_" + std::to_string(d));
            }
        };

        add_tensor_member_names("src_");
        add_tensor_member_names("dst_");

        names_.emplace_back("src_location");
        names_.emplace_back("dst_location");

        return names_;
    }();

    return names;
}

const std::vector<std::string>& SerializableVPUDMADescriptor::get_names_for_serializer() {
    static const std::vector<std::string> names = []() {
        auto names_ = _get_member_names();
        names_.emplace_back("vpunn_cycles");
        names_.emplace_back("cost_source");
        names_.emplace_back("error_info");
        return names_;
    }();

    return names;
}

}  // namespace VPUNN