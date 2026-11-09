// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_DEVICE_VALID_VALUES_VPU6_H
#define VPUNN_DEVICE_VALID_VALUES_VPU6_H

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
#include "interface_valid_values.h"
#include "vpu/ranges.h"
#include "vpu/types.h"

namespace VPUNN {

/// @brief specific NPU 6.0 configuration possibilities for workload, not layer
/* coverity[rule_of_five_violation:FALSE] */
class VPU6_0_WorkloadValidValues : public IDeviceValidValues {
public:
    VPU6_0_WorkloadValidValues(const VPU6_0_WorkloadValidValues&) noexcept(false) = default;
    VPU6_0_WorkloadValidValues& operator=(const VPU6_0_WorkloadValidValues&) = delete;

    VPU6_0_WorkloadValidValues(VPU6_0_WorkloadValidValues&&) noexcept(false) = default;
    VPU6_0_WorkloadValidValues& operator=(VPU6_0_WorkloadValidValues&&) = delete;

    ~VPU6_0_WorkloadValidValues() = default;

private:
    inline static const Values<ExecutionMode> valid_execution_order_all{
            ExecutionMode::CUBOID_4x16,
            ExecutionMode::CUBOID_8x16,
            ExecutionMode::CUBOID_16x16,
    };  // 4x1, 16x1, 4x4

    inline static const Values<ExecutionMode> dcim_valid_execution_order{ExecutionMode::dCIM_32x128};

    inline static const Values<ExecutionMode> eltwise_execution_order{ExecutionMode::CUBOID_8x16};

    inline static const Values<ExecutionMode> DW_CM_CONV_and_MAXPOOL_execution_order{ExecutionMode::CUBOID_16x16};

    inline static const Values<Swizzling> valid_swizzlings_def = {Swizzling::KEY_0, Swizzling::KEY_1, Swizzling::KEY_2,
                                                                  Swizzling::KEY_3, Swizzling::KEY_4, Swizzling::KEY_5};

    inline static const Values<Layout> valid_layouts_def = {
            Layout::ZXY /*default one, ZMAJOR like*/, Layout::XYZ, Layout::XZY, Layout::YXZ, Layout::YZX, Layout::ZYX,
    };

    inline static const Values<VPUDevice> devices_def{VPUDevice::NPU_6_0};

    inline static const std::unordered_map<VPUDevice, int> cmx_KB_sizes_def{
            {devices_def[0], ((1024 + 1024) * 100) / 100},
    };

    inline static const Values<int> output_write_tile_options_def{1, 2, 3, 4, 5, 6, 7, 8};

    inline static const Values<ISIStrategy> isi_stategy_options_def = {
            ISIStrategy::CLUSTERING,
            // ISIStrategy::SPLIT_OVER_H,
            ISIStrategy::SPLIT_OVER_K,  // can this work for  Reduce operations
            // maybe extend to a KHW switch??
            ISIStrategy::SPLIT_OVER_K_nxtHW,  // can this work for  Reduce operations
    };

    inline static const int weigths_alignment_B_def{16};                   // bytes
    inline static const bool legacy_samples_alignment_weights_def{false};  // no alignment (samples)
    inline static const int out_innermost_dim_alignment_def{16};           // bytes
    inline static const int input_heigth_start_factor_SOH_def{1};

    static constexpr int alignment_size_bytes_def{32};

    inline static const std::vector<DataType> types_16bits = {DataType::FLOAT16, DataType::BFLOAT16, DataType::INT16,
                                                              DataType::UINT16};
    inline static const std::vector<DataType> types_8bits = {DataType::INT8, DataType::UINT8, DataType::HF8,
                                                             DataType::BF8};
    inline static const Values<DataType> dcim_valid_in_dtypes = {DataType::INT8, DataType::UINT8, DataType::INT16,
                                                                 DataType::UINT16};

    inline static const Values<DataType> dcim_valid_wts_dtypes = {DataType::INT8, DataType::UINT8};

    // input datatypes. smaller than output
    // scl only
    inline static const Values<DataType> valid_in_datatypes{
            DataType::INT8,      //
            DataType::UINT8,     //
            DataType::HF8,       //
            DataType::BF8,       //
            DataType::FLOAT16,   //
            DataType::BFLOAT16,  //
    };

    // out datatypes can be extended
    // valid for dCIM as well
    inline static const Values<DataType> valid_out_datatypes{
            DataType::INT8,      //
            DataType::UINT8,     //
            DataType::HF8,       //
            DataType::BF8,       //
            DataType::FLOAT16,   //
            DataType::BFLOAT16,  //
            DataType::INT32,     //
            DataType::UINT32,    // unsigned counterpart of INT32
            DataType::FLOAT32,   // ODU
            DataType::INT16,     // ODU
            DataType::UINT16,    // ODU
    };

    // scl only
    inline static const Values<DataType> valid_wt_datatypes{
            DataType::INT2,   //
            DataType::UINT2,  //
            DataType::INT4,   //
            DataType::UINT4,  //
            DataType::INT8,
            DataType::UINT8,     //
            DataType::FLOAT4,    // is this present?
            DataType::HF8,       //
            DataType::BF8,       //
            DataType::FLOAT16,   //
            DataType::BFLOAT16,  //
    };

    inline static const IDeviceValidValues::ValidDatatypes valid_datatypes_map_default{
            // valid data types based on operations
            {
                    // in
                    {Operation::CONVOLUTION, valid_in_datatypes},        //
                    {Operation::DW_CONVOLUTION, valid_in_datatypes},     //
                    {Operation::CM_CONVOLUTION, valid_in_datatypes},     //
                    {Operation::ELTWISE, valid_in_datatypes},            //
                    {Operation::MAXPOOL, valid_in_datatypes},            //
                    {Operation::AVEPOOL, valid_in_datatypes},            // add AVGPOOL
                    {Operation::REDUCE_MS, valid_in_datatypes},          //
                    {Operation::REDUCE_SUMSQUARES, valid_in_datatypes},  //
            },
            {
                    // out
                    {Operation::CONVOLUTION, valid_out_datatypes},        //
                    {Operation::DW_CONVOLUTION, valid_out_datatypes},     //
                    {Operation::CM_CONVOLUTION, valid_out_datatypes},     //
                    {Operation::ELTWISE, valid_out_datatypes},            //
                    {Operation::MAXPOOL, valid_out_datatypes},            //
                    {Operation::AVEPOOL, valid_out_datatypes},            // add AVGPOOL
                    {Operation::REDUCE_MS, valid_out_datatypes},          //
                    {Operation::REDUCE_SUMSQUARES, valid_out_datatypes},  //
            },
            {
                    // wts
                    {Operation::CONVOLUTION, valid_wt_datatypes},        //
                    {Operation::DW_CONVOLUTION, valid_wt_datatypes},     //
                    {Operation::CM_CONVOLUTION, valid_wt_datatypes},     //
                    {Operation::ELTWISE, valid_wt_datatypes},            //
                    {Operation::MAXPOOL, valid_wt_datatypes},            //
                    {Operation::AVEPOOL, valid_wt_datatypes},            // add AVGPOOL
                    {Operation::REDUCE_MS, valid_wt_datatypes},          //
                    {Operation::REDUCE_SUMSQUARES, valid_wt_datatypes},  //
            },
    };

    inline static const IDeviceValidValues::ValidDatatypes dcim_valid_datatypes_map{
            // valid data types based on operations
            {
                    // in
                    {Operation::CONVOLUTION, dcim_valid_in_dtypes},  //
            },
            {
                    // out
                    {Operation::CONVOLUTION, valid_out_datatypes},  //
            },
            {
                    // wts
                    {Operation::CONVOLUTION, dcim_valid_wts_dtypes},  //
            },
    };

    inline static const Values<Operation> valid_operations_default{
            Operation::CONVOLUTION,        //
            Operation::DW_CONVOLUTION,     //
            Operation::CM_CONVOLUTION,     //
            Operation::ELTWISE,            //
            Operation::MAXPOOL,            //
            Operation::AVEPOOL,            //
            Operation::REDUCE_MS,          //
            Operation::REDUCE_SUMSQUARES,  //
                                           // Add new ones
    };

    inline static const Values<Operation> dcim_valid_operations{Operation::CONVOLUTION};

    inline static const Values<MPEEngine> valid_mpe_engines_def{MPEEngine::SCL, MPEEngine::DCIM};

    inline static const IDeviceValidValues::ValidExecutionModes valid_execution_order_map_default{
            // valid execution modes based on operations
            {
                    {Operation::CONVOLUTION, valid_execution_order_all},                  //
                    {Operation::DW_CONVOLUTION, DW_CM_CONV_and_MAXPOOL_execution_order},  //
                    {Operation::CM_CONVOLUTION, valid_execution_order_all},               //
                    {Operation::MAXPOOL, DW_CM_CONV_and_MAXPOOL_execution_order},         //
                    {Operation::ELTWISE, eltwise_execution_order},                        //
                    {Operation::AVEPOOL, DW_CM_CONV_and_MAXPOOL_execution_order},         //
                    {Operation::REDUCE_MS, valid_execution_order_all},                    //
                    {Operation::REDUCE_SUMSQUARES, valid_execution_order_all},            //
            }};

    inline static const IDeviceValidValues::ValidExecutionModes dcim_valid_execution_order_map{
            // valid execution modes based on operations
            {{Operation::CONVOLUTION, dcim_valid_execution_order}}};

public:
    /// constructor with link to operations dynamic behavior
    VPU6_0_WorkloadValidValues(const IContainer_OperationsDynamicBehavior& op_dynamic_constraints)
            : IDeviceValidValues(op_dynamic_constraints,
                                 valid_execution_order_map_default,     //
                                 valid_swizzlings_def,                  //
                                 valid_layouts_def,                     //
                                 devices_def,                           //
                                 cmx_KB_sizes_def,                      //
                                 output_write_tile_options_def,         //
                                 isi_stategy_options_def,               //
                                 weigths_alignment_B_def,               //
                                 input_heigth_start_factor_SOH_def,     //
                                 valid_datatypes_map_default,           //
                                 valid_operations_default,              //
                                 alignment_size_bytes_def,              //
                                 out_innermost_dim_alignment_def,       //
                                 legacy_samples_alignment_weights_def,  //
                                 valid_mpe_engines_def) {};
    /// constructor with link to operations dynamic behavior and input channels rules and restrictions
    VPU6_0_WorkloadValidValues(const IContainer_OperationsDynamicBehavior& op_dynamic_constraints,
                               const std::unordered_map<Operation, SmartRanges>& input_channels_restrictions_,
                               const SmartRanges& batch_restrictions_)
            : IDeviceValidValues(op_dynamic_constraints,
                                 valid_execution_order_map_default,     //
                                 valid_swizzlings_def,                  //
                                 valid_layouts_def,                     //
                                 devices_def,                           //
                                 cmx_KB_sizes_def,                      //
                                 output_write_tile_options_def,         //
                                 isi_stategy_options_def,               //
                                 weigths_alignment_B_def,               //
                                 input_heigth_start_factor_SOH_def,     //
                                 valid_datatypes_map_default,           //
                                 valid_operations_default,              //
                                 alignment_size_bytes_def,              //
                                 out_innermost_dim_alignment_def,       //
                                 legacy_samples_alignment_weights_def,  //
                                 valid_mpe_engines_def),
              input_channels_restrictions{input_channels_restrictions_},
              batch_restrictions{{batch_restrictions_}} {};

    /// constructor with link to operations dynamic behavior and what config can be overridden,and input channels rules
    VPU6_0_WorkloadValidValues(const IContainer_OperationsDynamicBehavior& op_dynamic_constraints,  //
                               const int& input_heigth_start_factor_SOH_,
                               const std::unordered_map<Operation, SmartRanges>& input_channels_restrictions_,
                               const SmartRanges& batch_restrictions_)
            : IDeviceValidValues(op_dynamic_constraints,
                                 valid_execution_order_map_default,  //
                                 valid_swizzlings_def,               //
                                 valid_layouts_def,                  //
                                 devices_def,                        //
                                 cmx_KB_sizes_def,                   //
                                 output_write_tile_options_def,      //
                                 isi_stategy_options_def,            //
                                 weigths_alignment_B_def,            //
                                 input_heigth_start_factor_SOH_,     // special
                                 valid_datatypes_map_default,        //
                                 valid_operations_default,           //
                                 alignment_size_bytes_def,           //
                                 out_innermost_dim_alignment_def,    //
                                 legacy_samples_alignment_weights_def, valid_mpe_engines_def),
              input_channels_restrictions{input_channels_restrictions_},
              batch_restrictions{{batch_restrictions_}} {};

    /// to do: adapt this for Reduce ops where the output can have special conditions
    MultiSmartRanges get_output_channels_restriction(const DPUOperation& dpu) const override {
        if (dpu.output_autopad ||  // reduce ops have 1 channel as output
            (dpu.operation == Operation::REDUCE_MS || dpu.operation == Operation::REDUCE_SUMSQUARES)) {
            return output_channels_restrictions_any;  // any Z value (ODU autopad supports unaligned OC for NPU6)
        } else {
            return MultiSmartRanges{{output_channels_restrictions.get_range(1)}};  // 16 multiples only
        }
    }

    MultiSmartRanges get_input_channels_restriction(const DPUOperation& dpu) const override {
        auto base_restriction = input_channels_restrictions.at(dpu.operation);

        if (dpu.input_autopad) {
            if (std::any_of(types_16bits.begin(), types_16bits.end(), [&dpu](const DataType type) {
                    return type == dpu.input_0.datatype;
                })) {
                return MultiSmartRanges{{base_restriction, input_channels_restriction_extensions.get_range(
                                                                   0)}};  // index 0 for 16 bits types
            } else if (std::any_of(types_8bits.begin(), types_8bits.end(), [&dpu](const DataType type) {
                           return type == dpu.input_0.datatype;
                       })) {
                return MultiSmartRanges{{base_restriction, input_channels_restriction_extensions.get_range(
                                                                   1)}};  // index 1 for 8 bits types
            } else {
                return MultiSmartRanges{{base_restriction}};  // no change
            }
        }

        return MultiSmartRanges{{base_restriction}};
    }

    MultiSmartRanges get_batch_restrictions() const override {
        return batch_restrictions;
    }

    inline static const SmartRanges onlyOne_range{1, 1};  /// a SmartRange that contains only 1 as value => [1]

    // bool mustExecuteHWLowLevelChecks(const DPUOperation& dpu) const noexcept override {
    //     return IDeviceValidValues::mustExecuteHWLowLevelChecks(dpu);  // by default/ no skip?
    // };

    const IDeviceValidValues::ValidDatatypes& get_valid_datatypes_map(const MPEEngine& engine) const override {
        if (engine == MPEEngine::DCIM) {
            return dcim_valid_datatypes_map;
        } else {
            return valid_datatypes_map_default;
        }
    }

    Layout adapt_device_comaptible_tensor_layout(Layout layout) const override {
        if (layout == Layout::ZMAJOR) {
            layout = Layout::ZXY;
        } else if (layout == Layout::CMAJOR) {
            layout = Layout::XYZ;
        }

        return layout;
    };

    Swizzling adapt_device_comaptible_swizzling(Swizzling swizz) const override {
        return swizz;  // no change for VPU2.7 all swizzlings accepted
    };

    /// NPU6+ workloads enforce autopad channel coupling constraints
    bool has_autopad_channel_coupling() const noexcept override {
        return true;
    }

    const Values<ExecutionMode>& get_valid_execution_order(const DPUOperation& dpu) const override {
        if (dpu.mpe_engine == MPEEngine::DCIM) {
            const auto& ch_map{dcim_valid_execution_order_map.execution_modes};
            return ch_map.at(dpu.operation);
        }
        const auto& ch_map{valid_execution_order_map_default.execution_modes};
        return ch_map.at(dpu.operation);
    }

protected:  // only const attributes can be visible in derived
    // can be changed in derived constructor

    // workload restrictions
    const std::unordered_map<Operation, SmartRanges> input_channels_restrictions{
            {Operation::CONVOLUTION, SmartRanges(16, input_spatial_dim_max, 16)},
            {Operation::DW_CONVOLUTION,
             SmartRanges(16, 256, 16, 32)},  // special case {16, 32, 64, 96, 128 up to 256 in multiples of 32}
            {Operation::CM_CONVOLUTION, SmartRanges(1, cm_conv_channels_max - 1)},
            {Operation::ELTWISE, SmartRanges(16, input_spatial_dim_max, 16)},
            {Operation::MAXPOOL,
             SmartRanges(16, 256, 16, 32)},  // special case {16, 32, 64, 96, 128 up to 256 in multiples of 32}
            {Operation::AVEPOOL,
             SmartRanges(16, 256, 16,
                         32)},  // special case {16, 32, 64, 96, 128 up to 256 in multiples of 32}// add AVGPOOL
            {Operation::REDUCE_MS, SmartRanges(16, input_spatial_dim_max, 16)},  // maybe e need more flexibility
            {Operation::REDUCE_SUMSQUARES, SmartRanges(16, input_spatial_dim_max, 16)},  //  more flexibility?
    };

    /// it is used for channels restrictions when input_autopad is true
    const MultiSmartRanges input_channels_restriction_extensions{
            {SmartRanges(1, 9, 1),     // when input dtype is 16bit
             SmartRanges(1, 15, 1)}};  // extensions for input channels restrictions, 1-9 for 16bit, 1-15 for 8bit types
    const MultiSmartRanges output_channels_restrictions{
            // OR based logic
            {SmartRanges(1, 15, 1),                       // for autopad? &  <16 channels
             SmartRanges(16, input_spatial_dim_max, 16)}  // multiple of 16 if 16 or larger
    };  // with autopad also 1-15 is allowed
    const MultiSmartRanges output_channels_restrictions_any{
            {SmartRanges(1, input_spatial_dim_max, 1)}  // any Z value, ODU autopad relaxed for NPU6
    };

    const MultiSmartRanges batch_restrictions{{onlyOne_range}};  // valid batch values
};

/// @todo: further specifications needed
//////// LAYER UNSPLIT situation
/// @brief specific VPU 6.0 configuration possibilities for  layer
class VPU6_0_LayerValidValues : public VPU6_0_WorkloadValidValues {
private:
    inline static const int input_heigth_start_factor_SOH_def{2};

    inline static const std::unordered_map<Operation, SmartRanges> input_channels_restrictions_layer{
            {Operation::CONVOLUTION, SmartRanges(16, input_spatial_dim_max, 16)},
            {Operation::DW_CONVOLUTION, SmartRanges(16, input_spatial_dim_max, 16)},
            {Operation::CM_CONVOLUTION, SmartRanges(1, cm_conv_channels_max - 1)},  // why 2?
            {Operation::ELTWISE, SmartRanges(16, input_spatial_dim_max, 16)},
            {Operation::MAXPOOL, SmartRanges(16, input_spatial_dim_max, 16)},
            {Operation::AVEPOOL, SmartRanges(16, input_spatial_dim_max, 16)},  // add AVGPOOL
            {Operation::REDUCE_MS, SmartRanges(16, input_spatial_dim_max, 16)},
            {Operation::REDUCE_SUMSQUARES, SmartRanges(16, input_spatial_dim_max, 16)},

    };
    inline static const SmartRanges batch_restrictions_layer{onlyOne_range};

public:
    /// constructor with link to operations dynamic behavior
    VPU6_0_LayerValidValues(const IContainer_OperationsDynamicBehavior& op_dynamic_constraints)
            : VPU6_0_WorkloadValidValues(op_dynamic_constraints, input_heigth_start_factor_SOH_def,
                                         input_channels_restrictions_layer, batch_restrictions_layer) {
        // at layer level we are not limited like for workload level
    }

    /// to do: adapt this for Reduce ops where the output can have special conditions
    MultiSmartRanges get_output_channels_restriction(const DPUOperation& dpu) const override {
        auto restrictions = VPU6_0_WorkloadValidValues::get_output_channels_restriction(dpu);

        return (dpu.isi_strategy == ISIStrategy::SPLIT_OVER_K)
                       ? restrictions.multiply_lower(2)  // split over K => output channels must be aligned to 32 (16*2)
                       : std::move(restrictions);        // no change
    }

    bool mustExecuteHWLowLevelChecks(const DPUOperation& /*dpu*/) const noexcept override {
        return false;  // layer is above the workload, no need to check
    };

    /// At layer level, autopad channel coupling is not enforced (only at workload level after tiling)
    bool has_autopad_channel_coupling() const noexcept override {
        return false;
    }

    Values<ISIStrategy> get_ISI_Strategy_Range(const DPUOperation& dpu) const override {
        // at layer level this is the desired strategy, no limitation based on tiles/output write tiles desired
        Values<ISIStrategy> v{isi_stategy_options};
        const auto& behavior = operations_dynamic_behavior.get_operation_specific_behaviour(dpu.operation);
        return behavior.filter_ISI_Strategy_Options(v);
    }

protected:
};

//////// LAYER SPLIT on tile  situation,

/// @brief specific VPU 6.0 configuration possibilities for  layer already split on tile.
/// channels restrictions are less strict vs workload, since a further split is expected
class VPU6_0_LayerOnTileValidValues : public VPU6_0_WorkloadValidValues {
private:
    inline static const std::unordered_map<Operation, SmartRanges> input_channels_restrictions_layersplit{
            {Operation::CONVOLUTION, SmartRanges(16, input_spatial_dim_max, 16)},
            {Operation::DW_CONVOLUTION, SmartRanges(16, input_spatial_dim_max, 16)},
            {Operation::CM_CONVOLUTION, SmartRanges(1, cm_conv_channels_max - 1)},
            {Operation::ELTWISE, SmartRanges(16, input_spatial_dim_max, 16)},
            {Operation::MAXPOOL, SmartRanges(16, input_spatial_dim_max, 16)},
            {Operation::AVEPOOL, SmartRanges(16, input_spatial_dim_max, 16)},  // add AVGPOOL
            {Operation::REDUCE_MS, SmartRanges(16, input_spatial_dim_max, 16)},
            {Operation::REDUCE_SUMSQUARES, SmartRanges(16, input_spatial_dim_max, 16)},
    };
    inline static const SmartRanges batch_restrictions_layersplit{onlyOne_range};  // valid batch values

public:
    /// constructor with link to operations dynamic behavior
    VPU6_0_LayerOnTileValidValues(const IContainer_OperationsDynamicBehavior& op_dynamic_constraints)
            : VPU6_0_WorkloadValidValues(op_dynamic_constraints, input_channels_restrictions_layersplit,
                                         batch_restrictions_layersplit) {
        // at layer level we are not limited like for workload level
    }

    bool has_autopad_channel_coupling() const noexcept override {
        return false;  // layer is above the workload, no need to check
    };

    bool mustExecuteHWLowLevelChecks(const DPUOperation& /*dpu*/) const noexcept override {
        return false;  // layer is above the workload, no need to check
    };

protected:
};

}  // namespace VPUNN

#endif  //
