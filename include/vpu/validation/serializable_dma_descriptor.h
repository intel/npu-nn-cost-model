// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_SERIALIZABLE_DMA_DESCRIPTOR_H
#define VPUNN_SERIALIZABLE_DMA_DESCRIPTOR_H

#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "vpu/dma_descriptors.h"
#include "vpu/serializer_utils.h"

namespace VPUNN {

/// Serialization adapter over VPUDMADescriptor.
/// VPUDMADescriptor remains the single source of truth for DMA descriptor semantics.
struct SerializableVPUDMADescriptor {
    using _ref_supported_type = std::variant<std::reference_wrapper<VPUDevice>, std::reference_wrapper<DataType>,
                                             std::reference_wrapper<int32_t>, std::reference_wrapper<MemoryLocation>,
                                             VPUNN::SetGet_MemberMapValues>;

    using MemberMapType = std::unordered_map<std::string, _ref_supported_type>;

    SerializableVPUDMADescriptor() = default;
    explicit SerializableVPUDMADescriptor(const VPUDMADescriptor& desc);

    SerializableVPUDMADescriptor(const SerializableVPUDMADescriptor& r);
    SerializableVPUDMADescriptor(SerializableVPUDMADescriptor&) = delete;
    SerializableVPUDMADescriptor(SerializableVPUDMADescriptor&&) = delete;

    SerializableVPUDMADescriptor& operator=(const SerializableVPUDMADescriptor&) = delete;
    SerializableVPUDMADescriptor& operator=(SerializableVPUDMADescriptor&) = delete;
    SerializableVPUDMADescriptor& operator=(SerializableVPUDMADescriptor&&) = delete;
    SerializableVPUDMADescriptor& operator=(SerializableVPUDMADescriptor) = delete;

    ~SerializableVPUDMADescriptor() = default;

    const VPUDMADescriptor& dma_descriptor_data() const;

    VPUDMADescriptor& dma_descriptor_data();

    const MemberMapType& get_member_map() const;

    MemberMapType& get_member_map();

    static std::string get_wl_name();

    static const std::vector<std::string>& _get_member_names();

    static const std::vector<std::string>& get_names_for_serializer();

private:
    VPUDMADescriptor dma_descriptor{};

    mutable MemberMapType _member_map{};
    mutable std::mutex _member_map_mutex{};

    void populate_member_map();

    void ensure_member_map() const;

    bool is_member_map_initialized() const;

    static bool is_non_negative_int(const std::string& s, int& result);
};

}  // namespace VPUNN

#endif  // VPUNN_SERIALIZABLE_DMA_DESCRIPTOR_H
