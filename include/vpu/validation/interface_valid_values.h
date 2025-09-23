// Copyright © 2024 Intel Corporation
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
#include "vpu/datatype_collection_size.h"
#include "vpu/ranges.h"
#include "vpu/types.h"

namespace VPUNN {

/// @brief infrastructure class for describing and creating  valid values
class ValidValuesInfrastructure {
public:
    /// @brief creates a list , rule: [from ... to] * multiply
    static Values<int> makeList(int from, int to, int multiply = 1) {
        const int counts = to - from + 1;
        Values<int> v(counts);
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
        return std::find(container.cbegin(), container.cend(), element) != container.cend();
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
/// /* coverity[rule_of_five_violation:FALSE] */
class IDeviceValidValues : public ValidValuesInfrastructure {
public:
    IDeviceValidValues(const IDeviceValidValues&) noexcept(false) = default;
    IDeviceValidValues& operator=(const IDeviceValidValues&) = delete;

    IDeviceValidValues(IDeviceValidValues&&) noexcept(false) = default;
    IDeviceValidValues& operator=(IDeviceValidValues&&) = delete;

    /// wrapper for accessing IContainer_OperationsDynamicBehavior
    const IOperationDynamicConstraints& get_specific_behaviour(const Operation op) const {
        return operations_dynamic_behavior.get_operation_specific_behaviour(op);
    }

    /// @brief collection of possible datatypes for different in/outs and operations
    class ValidDatatypes {
    public:
        // valid data types based on operations
        std::unordered_map<Operation, const Values<DataType>> input_datatypes;
        std::unordered_map<Operation, const Values<DataType>> output_datatypes;
        std::unordered_map<Operation, const Values<DataType>> weights_datatypes;
    };

    /// @brief collection of possible execution modes for different operations
    class ValidExecutionModes {
    public:
        // valid execution modes based on operations
        std::unordered_map<Operation, const Values<ExecutionMode>> execution_modes;
    };

protected:
    const IContainer_OperationsDynamicBehavior& operations_dynamic_behavior;  ///< externally attached dynamic behavior
    /// @brief non public constructor for initializing the reference
    IDeviceValidValues(const IContainer_OperationsDynamicBehavior& op_dynamic_constraints,     //
                       const IDeviceValidValues::ValidExecutionModes& valid_execution_order_,  //
                       const Values<Swizzling>& valid_swizzlings_,                             //
                       const Values<Layout>& valid_layouts_,                                   //
                       const Values<VPUDevice>& devices_,                                      //
                       const std::unordered_map<VPUDevice, int>& cmx_KB_sizes_,                //
                       const Values<int>& output_write_tile_options_,                          //
                       const Values<ISIStrategy>& isi_stategy_options_,                        //
                       const int& weigths_alignment_,                                          //
                       const int& input_heigth_width_start_factor_SOHW_,                       //
                       const IDeviceValidValues::ValidDatatypes& valid_datatypes,              //
                       const Values<Operation>& valid_operations_,                             //
                       const int& alignement_size_bytes_ , // page size, NPU specific
                       const int& out_innermost_dim_alignment_                                  //
                       )
            : operations_dynamic_behavior{op_dynamic_constraints},                          //
              valid_execution_order_map{valid_execution_order_},                            //
              valid_swizzlings{valid_swizzlings_},                                          //
              valid_layouts{valid_layouts_},                                                //
              devices{devices_},                                                            //
              cmx_KB_sizes{cmx_KB_sizes_},                                                  //
              output_write_tile_options{output_write_tile_options_},                        //
              isi_stategy_options{isi_stategy_options_},                                    //
              weigths_alignment{weigths_alignment_},                                        //
              input_heigth_width_start_factor_SOHW{input_heigth_width_start_factor_SOHW_},  //
              valid_datatypes_map{valid_datatypes},
              valid_operations{valid_operations_},
              alignement_size_bytes{alignement_size_bytes_},
              out_innermost_dim_alignment{out_innermost_dim_alignment_} {
    }

protected:
    /// Cannot be deleted by interface users
    virtual ~IDeviceValidValues() = default;

    // virtual operations or dynamic ranges based on ops
public:
    // return a SmartRanges that verify if a value matches
    virtual MultiSmartRanges get_input_channels_restriction(const DPUOperation& /*dpu*/) const = 0;
    virtual MultiSmartRanges get_output_channels_restriction(const DPUOperation& /*dpu*/) const = 0;

    // return a SmartRanges that verify if a value matches
    // if lower and upper bound is 1 then batch could be only equal with 1
    // but if lower bound is 1 and upper is a max value chosen by us then batch could be any value in this interval 
    // same batch restrictions for input and output tensor batch dimension
    virtual MultiSmartRanges get_batch_restrictions() const = 0; 

    /// false for places where checks regarding properties that are not connected to the abstract operation is not  to
    /// be checked (they do not really exists in that context). e.g. stencil. true: low level checks to be executed
    virtual bool mustExecuteHWLowLevelChecks(const DPUOperation& /*dpu*/) const noexcept {
        return true;  // check all by default
    };

    const Values<DataType>& get_input_valid_datatypes(const DPUOperation& dpu) const {
        const auto& ch_map{valid_datatypes_map.input_datatypes};
        return ch_map.at(dpu.operation);
    }

    const Values<DataType>& get_output_valid_datatypes(const DPUOperation& dpu) const {
        const auto& ch_map{valid_datatypes_map.output_datatypes};
        return ch_map.at(dpu.operation);
    }

    const Values<DataType>& get_weights_valid_datatypes(const DPUOperation& dpu) const {
        const auto& ch_map{valid_datatypes_map.weights_datatypes};
        return ch_map.at(dpu.operation);
    }

    /// @brief CHanges tensor layouts to match the device convention (if possible). Useful for defaults
    virtual Layout adapt_device_comaptible_tensor_layout(Layout layout) const = 0;

    /// @brief Changes tensor swizzling to match the device special restrictions or conventions. Useful for defaults
    virtual Swizzling adapt_device_comaptible_swizzling(Swizzling swizz) const = 0;

    std::pair<int, int> get_input_height_interval(const DPUOperation& dpu, bool use_extra_start = false) const {
        const int extra_out_rows{use_extra_start
                                         ? get_spatial_range_start_factor_HW() - 1  // will be >1 for layers checking
                                         : 1 - 1};                                  // no restriction

        const int minH_oneRow =
                dpu.kernel.height - (dpu.kernel.pad_top + dpu.kernel.pad_bottom);  // for one row of output
        const int minH{((minH_oneRow + extra_out_rows) < 1) ? 1 : (minH_oneRow + extra_out_rows)};  // at least one
        const auto maxH{input_spatial_dim_max};
        return std::make_pair(minH, maxH);  // force by value for constexpr
    }
    /// slow , only for generators
    Values<int> get_input_height_range(const DPUOperation& dpu) const {
        const auto interval{get_input_height_interval(dpu)};
        return makeList(interval.first, interval.second);
    }

    std::pair<int, int> get_input_width_interval(const DPUOperation& dpu, bool use_extra_start = false) const {
        const int extra_out_cols{use_extra_start
                                         ? get_spatial_range_start_factor_HW() - 1  // will be >1 for layers checking
                                         : 1 - 1};

        const int minW_oneCol = dpu.kernel.width - (dpu.kernel.pad_left + dpu.kernel.pad_right);    // for one output
        const int minW{((minW_oneCol + extra_out_cols) < 1) ? 1 : (minW_oneCol + extra_out_cols)};  // at least one
        const auto maxW{input_spatial_dim_max};
        return std::make_pair(minW, maxW);  // force by value for constexpr
    }

    /// slow , only for generators
    Values<int> get_input_width_range(const DPUOperation& dpu) const {
        const auto interval{get_input_width_interval(dpu)};
        return makeList(interval.first, interval.second);
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
    virtual Values<ISIStrategy> get_ISI_Strategy_Range(const DPUOperation& dpu) const {
        Values<ISIStrategy> v{isi_stategy_options};

        if (dpu.output_write_tiles <= 1) {  // SOK must have broadcasting, not allowed without
            //   Erase–remove idiom
            v.erase(std::remove_if(v.begin(), v.end(),
                                   [](const ISIStrategy& x) {                  // SOK not allowed, remove it
                                       return x == ISIStrategy::SPLIT_OVER_K;  // true =remove
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
        const auto max_stride_w = std::min(
                {max_stride_op, (int)dpu.input_0.width + (int)dpu.kernel.pad_left + (int)dpu.kernel.pad_right});
        const auto max_stride_h = std::min(
                {max_stride_op, (int)dpu.input_0.height + (int)dpu.kernel.pad_top + (int)dpu.kernel.pad_bottom});
        return std::make_pair(makeList(1, max_stride_w), makeList(1, max_stride_h));
    }

    /// @ brief this function is set to determine the range of strides for the split layers
    /// @returns a pair of lists of values, one first for  stride Width, second for stride height
    std::pair<Values<int>, Values<int>> get_dpu_strides_range(const DPUOperation& dpu) const {
        auto max_stride_op = get_maximum_kernel_stride(dpu.operation);
        const auto max_stride_w = max_stride_op;
        const auto max_stride_h = max_stride_op;
        return std::make_pair(makeList(1, max_stride_w), makeList(1, max_stride_h));
    }
    /// To use when generating new data
    Swizzling get_default_swizzling() const {
        return valid_swizzlings.back();  // return the last one as the one with highest swizzling
    }

    const Values<int>& get_output_write_tile_options() const {
        return output_write_tile_options;
    }

    /// provides the default/nominal weights alignment for the device. Can be used in case we need to align the weights
    /// TODO: unclear measurement unit, BYtes or samples?  Now it is interpreted as Samples
    int get_specific_weigths_alignment() const {
        return weigths_alignment;
    }

    int get_specific_out_innermost_dim_alignment() const {
        return out_innermost_dim_alignment;
    }

    const Values<Operation>& get_valid_operations() const {
        return valid_operations;
    };

    const Values<ExecutionMode>& get_valid_execution_order(const DPUOperation& dpu) const {
        const auto& ch_map{valid_execution_order_map.execution_modes};
        return ch_map.at(dpu.operation);
    }

    const Values<Swizzling>& get_valid_swizzlings() const {
        return valid_swizzlings;
    }
    const Values<Layout>& get_valid_layouts() const {
        return valid_layouts;
    }
    const Values<VPUDevice>& get_devices() const {
        return devices;
    }

    const Values<bool>& get_boolean_datatypes() const {
        return boolean_datatypes;
    }
    /// bytes alignment fro the memory chunks
    int get_page_alignment() const {
        return alignement_size_bytes;
    }

    int get_cmx_memory_aligned_overhead() const {
        return cmx_memory_80K_fixed_overhead + get_page_alignment();
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

    // restrictions described as data, set in derived classes
protected:
    const ValidExecutionModes valid_execution_order_map;  ///< what executions order are permitted for each operation
    const Values<Swizzling> valid_swizzlings;  ///< what swizzlings are permitted, ordered from least to most swizzling
    const Values<Layout> valid_layouts;        ///< what layouts are permitted
    const Values<VPUDevice> devices;           ///< devices covered with this instance

    const std::unordered_map<VPUDevice, int> cmx_KB_sizes;  ///< size of CMX memory for each device

    const Values<int> output_write_tile_options;  ///<
    const Values<ISIStrategy> isi_stategy_options;

    const int weigths_alignment{16};                    ///< default alignment for weights,
    const int input_heigth_width_start_factor_SOHW{1};  ///<  to be set in derived implementations

    int get_spatial_range_start_factor_HW() const {
        return input_heigth_width_start_factor_SOHW;
    }

    const ValidDatatypes valid_datatypes_map;  ///< valid data types for each operation

    const Values<Operation> valid_operations;  ///< valid operations for this device

    const int alignement_size_bytes;  // page size, NPU specific

    const int out_innermost_dim_alignment; ///< alignment for innermost dimension (in bytes), used when we compute output_0 tensor's memory 

    // const fixed here
private:
    inline static const Values<bool> boolean_datatypes{true, false};

public:
protected:
    static constexpr int cmx_memory_80K_fixed_overhead{80 * 1024};  ///< 80K fixed for all now

    static constexpr int input_spatial_dim_max{8192};  ///< 1-8K. MAX HW dim
    static constexpr int cm_conv_channels_max{16};     ///< CM convolution is limited to C<=16 ,

    static constexpr int kernel_max{15};  ///< max nominal kernel size (hardware)
    static constexpr int stride_max{8};   ///< hardware limit to 8

    static constexpr float cmx_size_safety_factor{1.0};

    static constexpr int sparsity_block_size{32};  ///< here we use 32 instead of 16 to take into account SOK sparsity

    /// the data type on the left will be mapped to the unique type on the right
    inline static const std::map<const DataType, const DataType> datatypes_pair_and_default{
            {DataType::INT8, DataType::UINT8},        //
            {DataType::INT4, DataType::UINT4},        //
            {DataType::INT2, DataType::UINT2},        //
            {DataType::INT1, DataType::UINT1},        //
            {DataType::BFLOAT16, DataType::FLOAT16},  //
            {DataType::BF8, DataType::HF8},           //
            {DataType::INT16, DataType::UINT16}       //
    };

public:
    /// @brief restrict the datatype if it has an alternative type. eg INT8 is same like UINT8
    DataType restrict_datatype(const DataType in) const {
        auto it = datatypes_pair_and_default.find(in);
        if (it != datatypes_pair_and_default.end()) {  // found
            return it->second;
        } else {
            return in;  // pass through if not mapped
        }
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

    /// computes output dimension based on input, kernel, padding and stride (for a spatial dimension)
    /// @returns zero if  kernel_stride is zero(avoids div by zero) else returns computed output dimension
    int compute_output_dim(int input, int pad, int pad_oppopsite, int kernel, int kernel_stride) const noexcept {
        return ((0 == kernel_stride) ? 0 : (((input + (pad + pad_oppopsite) - (kernel - 1) - 1) / kernel_stride) + 1));
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
    ///
    int check_trailing_padding(int in_dim, int out_dim, int leading_pad, int kernel_radix, int stride) const noexcept {
        const int pads = stride * (out_dim - 1) + kernel_radix - leading_pad - in_dim;
        return std::max(pads, 0);
    }

    /// @brief computes size with alignment
    long long compute_size_aligned(const long long elements_count, const DataType& datatype) const noexcept {
        const auto raw_size{compute_size_raw(elements_count, datatype)};
        const long long aligned_size{align_to(raw_size, alignement_size_bytes)};
        return aligned_size;
    }

    /// @brief computes size without alignment
    long long compute_size_raw(const long long elements_count, const DataType& datatype) const noexcept {
        const auto size{compute_size_in_bytes(elements_count, datatype)};
        return size;
    }

    bool is_valid_operation(const Operation op) const {
        return contains_value(valid_operations, op);
    }
};

}  // namespace VPUNN

#endif  //
