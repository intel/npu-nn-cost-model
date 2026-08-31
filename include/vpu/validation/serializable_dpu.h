// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_SERIALIZABLE_DPU_H
#define VPUNN_SERIALIZABLE_DPU_H

#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "vpu/dpu_types.h"
#include "vpu/dpu_workload.h"
#include "vpu/serializer_utils.h"
#include "vpu/validation/data_dpu_operation.h"  // KernelInfo

namespace VPUNN {

class IDeviceValidValues;

/// Serialization adapter over DPUOperation.
/// DPUOperation remains the single source of truth for workload semantics.
struct SerializableDPU {
    using _ref_supported_type =
            std::variant<std::reference_wrapper<VPUDevice>, std::reference_wrapper<Operation>,
                         std::reference_wrapper<DataType>, std::reference_wrapper<Layout>,
                         std::reference_wrapper<Swizzling>, std::reference_wrapper<ActivationFunction>,
                         std::reference_wrapper<ExecutionMode>, std::reference_wrapper<ISIStrategy>,
                         std::reference_wrapper<DimType>, std::reference_wrapper<long long>,
                         std::reference_wrapper<MPEEngine>, std::reference_wrapper<int>, std::reference_wrapper<float>,
                         std::reference_wrapper<bool>, VPUNN::SetGet_MemberMapValues>;

    using MemberMapType = std::unordered_map<std::string, _ref_supported_type>;

    SerializableDPU() = default;
    explicit SerializableDPU(const DPUWorkload& w);
    explicit SerializableDPU(const DPUOperation& op);
    SerializableDPU(const DPUWorkload& w, const IDeviceValidValues& config);

    SerializableDPU(const SerializableDPU& r);
    SerializableDPU(SerializableDPU&) = delete;
    SerializableDPU(SerializableDPU&&) = delete;

    SerializableDPU& operator=(const SerializableDPU&) = delete;
    SerializableDPU& operator=(SerializableDPU&) = delete;
    SerializableDPU& operator=(SerializableDPU&&) = delete;
    SerializableDPU& operator=(SerializableDPU) = delete;

    ~SerializableDPU() = default;

    DPUWorkload to_DPUWorkload() const;

    DPUWorkload clone_as_DPUWorkload() const;

    const DPUOperation& dpu_operation_data() const;

    DPUOperation& dpu_operation_data();

    const MemberMapType& get_member_map() const;

    MemberMapType& get_member_map();

    static const std::vector<std::string>& _get_member_names();

    size_t hash() const;

private:
    DPUOperation dpu_operation{};

    mutable MemberMapType _member_map{};
    mutable std::mutex _member_map_mutex{};

    void populate_member_map();

    void ensure_member_map() const;

    bool is_member_map_initialized() const;

    static bool is_unsigned_int(const std::string& s, VPUNN::DimType& result);

    void setInPlaceOutputMemory(const std::string& s);
    void setWeightlessOperation(const std::string& s);
};

}  // namespace VPUNN

#endif  // VPUNN_SERIALIZABLE_DPU_H
