// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_INTERFACE_VALID_VALUES_H
#define VPUNN_INTERFACE_VALID_VALUES_H

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <sstream>  // for error formating
#include <stdexcept>
#include <vector>

#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>

#include "data_dpu_operation.h"
#include "interface_operations_behavior.h"
#include "vpu/types.h"

namespace VPUNN {

/// @brief infrastructure class for describing and creating  valid values
class ValidValuesInfrastructure {
public:
    /// @brief creates a list , rule: [from ... to] * multiply
    Values<int> makeList(int from, int to, int multiply = 1) const {
        Values<int> v;
        const int counts = to - from + 1;
        v.resize(counts);
        std::generate(v.begin(), v.end(), [multiply, n = from]() mutable {
            const auto crt_value = n * multiply;
            n++;
            return crt_value;
        });
        return v;
    }

    /// @brief simpler wrapper for container find method
    /// @returns true if the element was found in container
    template <class T>
    bool contains_value(const Values<T>& container, const T& element) const noexcept {
        return std::find(container.begin(), container.end(), element) != container.end();
    }

protected:
    template <typename T>
    const Values<T> concat(const Values<T>& x, const Values<T>& y) const {
        Values<T> z{x};
        z.insert(z.end(), y.cbegin(), y.cend());
        return z;
    }
};

/// @brief interface for finding out what are the valid values for a workload that has a particular device and operation
/// for stable values (independent of operation) : holds the data values that a workload can take on its fields
/// dynamic behavior is provided via methods, including pure virtual ones.
/// has also a connection to the specific behavior interface that discriminated between operations
class IDeviceValidValues : public ValidValuesInfrastructure {
public:
    /// wrapper for accessing IContainer_OperationsDynamicBehavior
    const IOperationDynamicConstraints& get_specific_behaviour(const Operation op) const {
        return operations_dynamic_behavior.get_operation_specific_behaviour(op);
    }

protected:
    const IContainer_OperationsDynamicBehavior& operations_dynamic_behavior;  ///< externally attached dynamic behavior
    /// @brief non public constructor for initializing the reference
    IDeviceValidValues(const IContainer_OperationsDynamicBehavior& op_dynamic_constraints)
            : operations_dynamic_behavior{op_dynamic_constraints} {
    }

protected:
    virtual ~IDeviceValidValues() = default;

    // virtual operations or dynamic ranges based on ops
public:
    const Values<Operation>& get_valid_operations_range() const {
        return valid_operations;
    };

    virtual const Channels& get_output_channels_range(const DPUOperation& dpu) const = 0;

    virtual const Channels& get_input_channels_range(const DPUOperation& dpu) const = 0;

    /// @brief CHanges tensor layouts to match the device convention (if possible). Useful for defaults
    virtual Layout adapt_device_comaptible_tensor_layout(Layout layout) const = 0;

    /// @brief Changes tensor swizzling to match the device special restrictions or conventions. Useful for defaults
    virtual Swizzling adapt_device_comaptible_swizzling(Swizzling swizz) const = 0;

    Values<int> get_input_height_range(const DPUOperation& dpu) const {
        const int extra_out_rows{dpu.isi_strategy == ISIStrategy::SPLIT_OVER_H
                                         ? input_heigth_start_factor_SOH - 1  // will be >1 for layers checking
                                         : 1 - 1};                            // no restriction

        const int minH_oneRow =
                dpu.kernel.height - (dpu.kernel.pad_top + dpu.kernel.pad_bottom);  // for one row of output
        const int minH{((minH_oneRow + extra_out_rows) < 1) ? 1 : (minH_oneRow + extra_out_rows)};  // at least one
        return makeList(minH, input_spatial_dim_max);
    }

    Values<int> get_input_width_range(const DPUOperation& dpu) const {
        const int minW_oneCol = dpu.kernel.width - (dpu.kernel.pad_left + dpu.kernel.pad_right);  // for one output
        const int minW{(minW_oneCol < 1) ? 1 : minW_oneCol};                                      // at least one
        return makeList(minW, input_spatial_dim_max);
    }

    Values<int> get_pad_horz_range(const DPUOperation& dpu) const {
        const int padmax = get_padMax(dpu.kernel.width);  // 7->3,  8->3
        return makeList(0, padmax);
    }
    Values<int> get_pad_vert_range(const DPUOperation& dpu) const {
        const int padmax = get_padMax(dpu.kernel.height);  // 7->3,  8->3
        return makeList(0, padmax);
    }

    /// restricts ISI options based on output write tile value.
    virtual Values<ISIStrategy> get_ISI_Strategy_Range(const DPUOperation& dpu) const noexcept {
        Values<ISIStrategy> v{isi_stategy_options};
        if (dpu.output_write_tiles <= 1) {
            // v = {ISIStrategy::CLUSTERING, ISIStrategy::SPLIT_OVER_H};
            //   Erase–remove idiom
            v.erase(std::remove_if(v.begin(), v.end(),
                                   [](const ISIStrategy& x) {
                                       // SOK not allowed for <=1
                                       return x == ISIStrategy::SPLIT_OVER_K;
                                   }),
                    v.cend());
        } else {
            // eliminate  if not SOK
            //    Erase–remove idiom
            v.erase(std::remove_if(v.begin(), v.end(),
                                   [](const ISIStrategy& x) {
                                       // SOK only  allowed for >1
                                       return x != ISIStrategy::SPLIT_OVER_K;
                                   }),
                    v.cend());
        }
        const auto& behavior = operations_dynamic_behavior.get_operation_specific_behaviour(dpu.operation);
        return behavior.filter_ISI_Strategy_Options(v);
    }

    /// restricts output_write_tile options based on operation.
    Values<int> get_output_write_tile_Range(const DPUOperation& dpu) const noexcept {
        const Values<int> v{output_write_tile_options};  // no filtering by default

        const auto& behavior = operations_dynamic_behavior.get_operation_specific_behaviour(dpu.operation);
        return behavior.filter_output_write_tile_Options(v);  // let the operation restrict it
    }

    Values<int> get_kernel_range(const DPUOperation& dpu) const {
        auto maximum_kernel_size = get_maximum_kernel_size(dpu.operation);
        return makeList(1, maximum_kernel_size);
    }

    /// @ brief strides range , depends on input zero, and operation sometimes
    /// @returns a pair of lists of values, one first for  stride Width, second for stride height
    std::pair<Values<int>, Values<int>> get_strides_range(const DPUOperation& dpu) const {
        auto max_stride_op = get_maximum_kernel_stride(dpu.operation);
        const auto max_stride_w = std::min({max_stride_op, (int)dpu.input_0.width});
        const auto max_stride_h = std::min({max_stride_op, (int)dpu.input_0.height});
        return std::make_pair(makeList(1, max_stride_w), makeList(1, max_stride_h));
    }

protected:
    /// @brief what's the maximum kernel size if operation is known?
    int get_maximum_kernel_size(const Operation& op) const noexcept {
        return (op == Operation::ELTWISE) ? 1 : kernel_max;
    }

    /// @brief what's the maximum kernel stride if operation is known?
    int get_maximum_kernel_stride(const Operation& op) const noexcept {
        return (op == Operation::ELTWISE) ? 1 : stride_max;
    }

    // restrictions described as data
public:
    Values<ExecutionMode> valid_execution_order;  ///< what executions order are permitted

    Values<Swizzling> valid_swizzlings;             ///< what swizzlings are permitted
    Swizzling default_swizzling{Swizzling::KEY_0};  ///< what is the default swizzling

    Values<Layout> valid_layouts;  ///< what layouts are permitted

    Values<VPUDevice> devices;                        ///< devices covered with this instance
    std::unordered_map<VPUDevice, int> cmx_KB_sizes;  ///< size of CMX memory for each device

    const Values<DataType> quantized_datatypes{DataType::UINT8, DataType::INT8};
    const Values<DataType> float_datatypes{DataType::FLOAT16, DataType::BFLOAT16};

    const Values<DataType> valid_datatypes{concat(quantized_datatypes, float_datatypes)};

    const Values<Operation> valid_operations{
            Operation::CONVOLUTION,     //
            Operation::DW_CONVOLUTION,  //
            Operation::CM_CONVOLUTION,  //
            Operation::ELTWISE,         //
            Operation::MAXPOOL,         //
    };

    const Values<bool> boolean_datatypes{true, false};

    static constexpr int alignement_size_bytes{16384};  // 16KB
    static constexpr int cmx_memory_aligned_overhead{
            (80 * 1024)    // runtime size
            + (16 * 1024)  // hardware profile block (hwp)
    };

    Values<int> output_write_tile_options;  ///<
    Values<ISIStrategy> isi_stategy_options;

    int input_heigth_start_factor_SOH{1};  ///<  to be set in derived implementations

protected:
    static constexpr int input_spatial_dim_max{8192 - 1};  ///< 1-8K. MAX HW dim

    static constexpr int channels_max{512 - 1};         ///< 128,512, etc: influences channels range,
    static constexpr int cm_conv_channels_max{16 - 1};  ///< CM convolution is limited to C<16 ,

    static constexpr int kernel_max{11};  ///< max nominal kernel size (hardware)
    static constexpr int stride_max{7};   ///< hardware limit to 7

    static constexpr DataType default_quantized{DataType::UINT8};
    static constexpr DataType default_float{DataType::FLOAT16};

    static constexpr float cmx_size_safety_factor{1.0};

    static constexpr int sparsity_block_size{32};  ///< here we use 32 instead of 16 to take into account SOK sparsity

public:
    /// @brief restrict the datatype normally to one per int range and one per float range
    DataType restrict_datatype(const DataType in) const {
        if (contains_value(quantized_datatypes, in)) {
            return default_quantized;
        } else if (contains_value(float_datatypes, in)) {
            return default_float;
        }
        return default_quantized;
    }

    /// @brief size of CMX in bytes
    int get_cmx_size(const VPUDevice& device) const noexcept {
        auto const it = cmx_KB_sizes.find(device);
        const int ret_val{it != cmx_KB_sizes.end() ? it->second * 1024 : 0};

        const auto scaled_ret_val{std::lround(std::ceil(ret_val * cmx_size_safety_factor))};
        return scaled_ret_val;
    }

    /// @brief maximum padding required if the kernel is known
    int get_padMax(int kernel_dim) const noexcept {
        return ((kernel_dim) / 2);  // 7->3,  8->4, 1->0, 2->1
    }

    /// @ computes output dimension based on input, kernel, padding and stride
    int compute_output_dim(int input, int pad, int pad_oppopsite, int kernel, int kernel_stride) const noexcept {
        return (((input + (pad + pad_oppopsite) - (kernel - 1) - 1) / kernel_stride) + 1);
    }

    /// @brief computes the next larger value of x that is multiple of multiple
    /// @pre only for positive values
    long long align_to(long long x, int multiple) const noexcept {
        const auto rem = x % multiple;
        return rem == 0 ? x : x + (multiple - rem);
    }

    /// @brief sparsity is applied in blocks (16 B normally) , the desired value has to be quantized taken that in
    /// consideration
    float sanitize_sparsity(long long tensor_size, float desired_sparsity_level) const {
        const auto number_of_zeros{std::round((tensor_size / sparsity_block_size) * desired_sparsity_level) *
                                   sparsity_block_size};
        return number_of_zeros / tensor_size;
    }

    /// Adapt  padding
    /// Reference :
    /// https://theano-pymc.readthedocs.io/en/latest/tutorial/conv_arithmetic.html#zero-padding-non-unit-strides
    int check_trailing_padding(int in_dim, int out_dim, int leading_pad, int kernel_radix, int stride) const noexcept {
        const int pads = stride * (out_dim - 1) + kernel_radix - leading_pad - in_dim;
        return std::max(pads, 0);
    }

    /// @brief computes size with alignment
    long long compute_size_aligned(const long long elements_count, const DataType& datatype) const noexcept {
        const auto raw_size{compute_size_raw(elements_count, datatype)};
        const long long aligned_size{align_to(raw_size, alignement_size_bytes)};
        return aligned_size;
    };

    /// @brief computes size without alignment
    long long compute_size_raw(const long long elements_count, const DataType& datatype) const noexcept {
        const int datatype_size{static_cast<int>(dtype_to_bytes(datatype))};
        return elements_count * datatype_size;
    };
};

}  // namespace VPUNN

#endif  //
