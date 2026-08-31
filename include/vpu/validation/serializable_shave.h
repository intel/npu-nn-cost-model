// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_SERIALIZABLE_SHAVE_H
#define VPUNN_SERIALIZABLE_SHAVE_H

#include <array>
#include <functional>
#include <iosfwd>
#include <mutex>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "serializable_tensor.h"
#include "vpu/dpu_types.h"
#include "vpu/profiling_service.h"

namespace VPUNN {

class SHAVEWorkload;

/// @brief local type describing a SHAVE workload
/// easy to change and adapt without touching the SHAVEWorkload interface
/// Similar to DPUOperation but for SHAVE operations
/* coverity[rule_of_three_violation:FALSE] */
struct SerializableSHAVE {
    VPUDevice device{};       ///< device family, VPU2_0, 2_7, ...
    std::string operation{};  ///< operation name
    ProfilingServiceBackend profiling_service_backend_hint{
            ProfilingServiceBackend::__size};  ///< hint about what profiling service backend to use
    std::string loc_name{};                    ///< location name

    // Fixed-size arrays for serialization (mutable to allow modification during deserialization)
    // Support up to 8 inputs (0-indexed: input_0 through input_7) and 4 outputs as per csv_parser
    std::array<TensorInfo, 8> input_tensors{};
    std::array<TensorInfo, 4> output_tensors{};

    // Parameters as strings for serialization
    std::array<std::string, 3> param_strings{};        ///< up to 3 parameters
    std::array<std::string, 9> extra_param_strings{};  ///< up to 9 extra parameters

    using _ref_supported_type = std::variant<std::reference_wrapper<VPUDevice>, std::reference_wrapper<std::string>,
                                             std::reference_wrapper<long long>, std::reference_wrapper<DataType>,
                                             std::reference_wrapper<Layout>, std::reference_wrapper<bool>>;

    using MemberMapType = std::unordered_map<std::string, _ref_supported_type>;

    const MemberMapType& get_member_map() const;
    MemberMapType& get_member_map();

private:
    /** @brief Get the member map for serialization
     * Cannot be generalized keys to more generic prefix_index_batch... prefix_index_channels...
     * Because _member_map should be const to avoid accidental modification of this structure
     * @return The member map for serialization
     */
    mutable MemberMapType _member_map{};
    mutable std::mutex _member_map_mutex{};

    void populate_member_map();
    void ensure_member_map() const;
    bool is_member_map_initialized() const;

public:
    /**
     * Could be an idea to reuse _member_map keys, but function _get_member_names is
     * declared static const, so it should be hard coded
     */
    static const std::vector<std::string>& _get_member_names();

    static const std::string get_wl_name();

    /// constructor from a SHAVEWorkload
    explicit SerializableSHAVE(const SHAVEWorkload& w);

    SerializableSHAVE() = default;

    SerializableSHAVE(const SerializableSHAVE& r);

    SerializableSHAVE(SerializableSHAVE&) = delete;
    SerializableSHAVE(SerializableSHAVE&&) = delete;

    SerializableSHAVE& operator=(const SerializableSHAVE&) = delete;
    SerializableSHAVE& operator=(SerializableSHAVE&) = delete;
    SerializableSHAVE& operator=(SerializableSHAVE&&) = delete;
    SerializableSHAVE& operator=(SerializableSHAVE) = delete;

    ~SerializableSHAVE() = default;

    /// @brief Convert back to SHAVEWorkload
    SHAVEWorkload clone_as_SHAVEWorkload() const;

    void clearAllFields();
};

std::ostream& operator<<(std::ostream& stream, const SerializableSHAVE& d);

}  // namespace VPUNN

#endif  // VPUNN_SERIALIZABLE_SHAVE_H
