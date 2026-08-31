// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_SHAVE_WORKLOAD_H
#define VPUNN_SHAVE_WORKLOAD_H

#include <iostream>
#include <map>
#include <string>
#include <variant>
#include <vector>

#include "dpu_types.h"
#include "dpu_dtypes_dimension_info.h"
#include "vpu_tensor.h"
#include "profiling_service.h"
#include "vpu_tensor.h"

namespace VPUNN {

/**
 * @brief describes a Software layer (SHAVE) request
 */
class SHAVEWorkload {
private:
    std::string name{};  ///<  the name of the SW operation. We have a very flexible range of them.
    VPUDevice device{};  ///< The VPU device. There will be different methods/calibrations/profiling depending on device

    // input and output tensors number and content must be correlated with the operation and among themselves. Not all
    // combinations are possible
    std::vector<VPUTensor> inputs{};   ///< The input tensors. Mainly shape and datatype are used
    std::vector<VPUTensor> outputs{};  ///< The output tensors. Mainly shape and datatype are used
    std::string loc_name{};            ///< The location name

public:
    ProfilingServiceBackend profiling_service_backend_hint{
            ProfilingServiceBackend::__size};  ///< hint about what profiling service backend to use
    using Param = std::variant<int, float, std::string, bool>;
    using Parameters = std::vector<SHAVEWorkload::Param>;
    using ExtraParameters = std::map<std::string, Param>;
    bool include_code_prefetch{false};  ///< if true, code prefetch cost is added to computed cycles

private:
    Parameters call_params{};        ///<  can emulate call parameters
    ExtraParameters extra_params{};  ///< Additional parameters as a map

public:
    /// @brief ctor must exist since we have aggregate initialization possible on this type (abstract type)
    SHAVEWorkload(const std::string& operation_name, const VPUDevice& device, const std::vector<VPUTensor>& inputs,
                  const std::vector<VPUTensor>& outputs, const Parameters& params = {},
                  const ExtraParameters& extra_param = {},
                  const ProfilingServiceBackend profiling_service_backend_hint = ProfilingServiceBackend::__size,
                  const std::string& loc_name = "", const bool include_code_prefetch = false)
            : name(operation_name),
              device{device},
              inputs{inputs},
              outputs{outputs},
              loc_name(loc_name),
              profiling_service_backend_hint{profiling_service_backend_hint},
              include_code_prefetch{include_code_prefetch},
              call_params{params},
              extra_params{extra_param} {
    }

    SHAVEWorkload(const SHAVEWorkload&) = default;
    SHAVEWorkload& operator=(const SHAVEWorkload&) = default;

    /// @brief Default constructor
    SHAVEWorkload() = default;

    /// default destructor explicit stated here for gcov problems.
    ~SHAVEWorkload() = default;

    // accessors

    std::string get_name() const {
        return name;
    };
    VPUDevice get_device() const {
        return device;
    };
    const std::vector<VPUTensor>& get_inputs() const {
        return inputs;
    };
    const std::vector<VPUTensor>& get_outputs() const {
        return outputs;
    };
    const Parameters& get_params() const {
        return call_params;
    };

    const ExtraParameters& get_extra_params() const {
        return extra_params;
    }

    const std::string& get_loc_name() const {
        return loc_name;
    };

    const ProfilingServiceBackend& get_profiling_service_backend() const {
        return profiling_service_backend_hint;
    }

    bool get_include_code_prefetch() const {
        return include_code_prefetch;
    }

    /// @brief Set the location name used for debugging purposes
    // void set_loc_name(const std::string& name) {
    // loc_name = name;
    // }

    uint32_t hash() const;

    /// @brief Get the total number of elements from all input and output tensors
    long long total_number_of_elements() const {
        long long total_elements = 0;
        for (const auto& output : get_outputs()) {
            total_elements += static_cast<long long>(output.volume());
        }

        for (const auto& input : get_inputs()) {
            total_elements += static_cast<long long>(input.volume());
        }

        return total_elements;
    }

    long long total_size_in_bits() const {
        long long total_size_bits = 0;
        for (const auto& output : get_outputs()) {
            const auto out_dtype_bits = DTypeDimensionInfo::dtype_to_bits(output.get_dtype());
            if (out_dtype_bits <= 0) {
                return 0;  // unknown datatype
            } else {
                total_size_bits += static_cast<long long>(output.volume()) * static_cast<long long>(out_dtype_bits);
            }
        }

        for (const auto& input : get_inputs()) {
            const auto in_dtype_bits = DTypeDimensionInfo::dtype_to_bits(input.get_dtype());
            if (in_dtype_bits <= 0) {
                return 0;  // unknown datatype
            } else {
                total_size_bits += static_cast<long long>(input.volume()) * static_cast<long long>(in_dtype_bits);
            }
        }

        return total_size_bits;
    }

    /// @brief Convert workload to string representation
    std::string toString() const;

    /// @brief Friend declaration for operator<
    friend bool operator<(const VPUNN::SHAVEWorkload& lhs, const VPUNN::SHAVEWorkload& rhs);
};

/// @brief Stream output operator for SHAVEWorkload
std::ostream& operator<<(std::ostream& stream, const VPUNN::SHAVEWorkload& d);

/// @brief Less than comparison operator for SHAVEWorkload
bool operator<(const VPUNN::SHAVEWorkload& lhs, const VPUNN::SHAVEWorkload& rhs);

}  // namespace VPUNN

#endif  // VPUNN_SHAVE_WORKLOAD_H
