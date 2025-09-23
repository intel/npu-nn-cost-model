// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_DEVICE_VALID_VALUES_VPU4_H
#define VPUNN_DEVICE_VALID_VALUES_VPU4_H

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

//////////////////////// VPU 4.0  all

/// @brief specific VPU 4.0 configuration possibilities for workload, not layer
/* coverity[rule_of_five_violation:FALSE] */
class VPU4_0_WorkloadValidValues : public IDeviceValidValues {
public:
    VPU4_0_WorkloadValidValues(const VPU4_0_WorkloadValidValues&) noexcept(false) = default;
    VPU4_0_WorkloadValidValues& operator=(const VPU4_0_WorkloadValidValues&) = delete;

    VPU4_0_WorkloadValidValues(VPU4_0_WorkloadValidValues&&) noexcept(false) = default;
    VPU4_0_WorkloadValidValues& operator=(VPU4_0_WorkloadValidValues&&) = delete;

    ~VPU4_0_WorkloadValidValues() = default;

private:
    inline static const Values<ExecutionMode> valid_execution_order_all{
            ExecutionMode::CUBOID_4x16,
            ExecutionMode::CUBOID_8x16,
            ExecutionMode::CUBOID_16x16,
    };  // 4x1, 16x1, 4x4

    inline static const Values<Swizzling> valid_swizzlings_def{Swizzling::KEY_0, Swizzling::KEY_1, Swizzling::KEY_2,
                                                               Swizzling::KEY_3, Swizzling::KEY_4, Swizzling::KEY_5};
    inline static const Values<Layout> valid_layouts_def{
            Layout::ZXY /*default one, ZMAJOR like*/, Layout::XYZ, Layout::XZY, Layout::YXZ, Layout::YZX, Layout::ZYX,
    };
    inline static const Values<VPUDevice> devices_def{
            VPUDevice::VPU_4_0,
    };

    inline static const std::unordered_map<VPUDevice, int> cmx_KB_sizes_def{
            {devices_def[0], ((512 + 1024) * 100) / 100}};

    inline static const Values<int> output_write_tile_options_def{
            1, 2, 3, 4, 5, 6, 7, 8};  // maybe only 1,2,4, is real limit? keep free for now
    inline static const Values<ISIStrategy> isi_stategy_options_def{
            ISIStrategy::CLUSTERING,
            ISIStrategy::SPLIT_OVER_H,  // should be illegal, but keep it OK fro avoiding legacy side effects
            ISIStrategy::SPLIT_OVER_K,
    };

    inline static const int weigths_alignment_def{32};
    inline static const int out_innermost_dim_alignment_def{16};  ////bytes
    inline static const int input_heigth_start_factor_SOH_def{1};

    static constexpr int alignement_size_bytes_def{16384};  // 16KB or 32KB?

    // input datatypes. smaller than output
    inline static const Values<DataType> valid_in_datatypes{
            DataType::INT8,      //
            DataType::UINT8,     //
            DataType::FLOAT16,   //
            DataType::BFLOAT16,  //
    };

    // out datatypes can be extended
    inline static const Values<DataType> valid_out_datatypes{
            DataType::INT8,      //
            DataType::UINT8,     //
            DataType::FLOAT16,   //
            DataType::BFLOAT16,  //
            DataType::FLOAT32,   // ODU
    };

    inline static const Values<DataType> valid_wt_datatypes{
            DataType::INT8,
            DataType::UINT8,    //
            DataType::INT4,     //
            DataType::UINT4,    //
            DataType::FLOAT16,  //
            DataType::BFLOAT16,
    };

    inline static const IDeviceValidValues::ValidDatatypes valid_datatypes_map_default{
            // valid data types based on operations
            {
                    {Operation::CONVOLUTION, valid_in_datatypes},     //
                    {Operation::DW_CONVOLUTION, valid_in_datatypes},  //
                    {Operation::CM_CONVOLUTION, valid_in_datatypes},  //
                    {Operation::ELTWISE, valid_in_datatypes},         //
                    {Operation::MAXPOOL, valid_in_datatypes},         //
            },
            {
                    {Operation::CONVOLUTION, valid_out_datatypes},     //
                    {Operation::DW_CONVOLUTION, valid_out_datatypes},  //
                    {Operation::CM_CONVOLUTION, valid_out_datatypes},  //
                    {Operation::ELTWISE, valid_out_datatypes},         //
                    {Operation::MAXPOOL, valid_out_datatypes},         //
            },
            {
                    {Operation::CONVOLUTION, valid_wt_datatypes},     //
                    {Operation::DW_CONVOLUTION, valid_wt_datatypes},  //
                    {Operation::CM_CONVOLUTION, valid_wt_datatypes},  //
                    {Operation::ELTWISE, valid_wt_datatypes},         //
                    {Operation::MAXPOOL, valid_wt_datatypes},         //
            },
    };

    inline static const Values<Operation> valid_operations_default{
            Operation::CONVOLUTION,     //
            Operation::DW_CONVOLUTION,  //
            Operation::CM_CONVOLUTION,  //
            Operation::ELTWISE,         //
            Operation::MAXPOOL,         //
    };

    inline static const IDeviceValidValues::ValidExecutionModes valid_execution_order_map_default{
            // valid execution modes based on operations
            {
                    {Operation::CONVOLUTION, valid_execution_order_all},     //
                    {Operation::DW_CONVOLUTION, valid_execution_order_all},  //
                    {Operation::CM_CONVOLUTION, valid_execution_order_all},  //
                    {Operation::ELTWISE, valid_execution_order_all},         //
                    {Operation::MAXPOOL, valid_execution_order_all},         //
            }};

public:
    /// constructor with link to operations dynamic behavior
    VPU4_0_WorkloadValidValues(const IContainer_OperationsDynamicBehavior& op_dynamic_constraints)
            : IDeviceValidValues(op_dynamic_constraints,             //
                                 valid_execution_order_map_default,  //
                                 valid_swizzlings_def,               //
                                 valid_layouts_def,                  //
                                 devices_def,                        //
                                 cmx_KB_sizes_def,                   //
                                 output_write_tile_options_def,      //
                                 isi_stategy_options_def,            //
                                 weigths_alignment_def,              //
                                 input_heigth_start_factor_SOH_def,  //
                                 valid_datatypes_map_default,        //
                                 valid_operations_default,           //
                                 alignement_size_bytes_def,          //
                                 out_innermost_dim_alignment_def) {};

    /// constructor with link to operations dynamic behavior, input channels rules and restrictions
    VPU4_0_WorkloadValidValues(const IContainer_OperationsDynamicBehavior& op_dynamic_constraints,
                               const std::unordered_map<Operation, MultiSmartRanges>& input_channels_restrictions_)
            : IDeviceValidValues(op_dynamic_constraints,
                                 valid_execution_order_map_default,  //
                                 valid_swizzlings_def,               //
                                 valid_layouts_def,                  //
                                 devices_def,                        //
                                 cmx_KB_sizes_def,                   //
                                 output_write_tile_options_def,      //
                                 isi_stategy_options_def,            //
                                 weigths_alignment_def,              //
                                 input_heigth_start_factor_SOH_def,  //
                                 valid_datatypes_map_default,        //
                                 valid_operations_default,           //
                                 alignement_size_bytes_def,          //
                                 out_innermost_dim_alignment_def),
              input_channels_restrictions{input_channels_restrictions_} {};

    /// constructor with link to operations dynamic behavior and what config can be overridden (and input channels
    /// rules)
    VPU4_0_WorkloadValidValues(const IContainer_OperationsDynamicBehavior& op_dynamic_constraints,  //
                               const int& input_heigth_start_factor_SOH_,
                               const std::unordered_map<Operation, MultiSmartRanges>& input_channels_restrictions_)
            : IDeviceValidValues(op_dynamic_constraints,
                                 valid_execution_order_map_default,  //
                                 valid_swizzlings_def,               //
                                 valid_layouts_def,                  //
                                 devices_def,                        //
                                 cmx_KB_sizes_def,                   //
                                 output_write_tile_options_def,      //
                                 isi_stategy_options_def,            //
                                 weigths_alignment_def,              //
                                 input_heigth_start_factor_SOH_,     // special
                                 valid_datatypes_map_default,        //
                                 valid_operations_default,           //
                                 alignement_size_bytes_def,          //
                                 out_innermost_dim_alignment_def),
              input_channels_restrictions{input_channels_restrictions_} {};

    MultiSmartRanges get_output_channels_restriction(const DPUOperation&) const override {
        return output_channels_restrictions;
    }

    MultiSmartRanges get_input_channels_restriction(const DPUOperation& dpu) const override {
        return input_channels_restrictions.at(dpu.operation);
    }

    MultiSmartRanges get_batch_restrictions() const override {
        return batch_restrictions;
    }

    inline static const SmartRanges allValues_range{
            1, SmartRanges::max_limit};  /// a SmartRange that contains all possible values,
                                         /// from 1 to maxim number accepted => [1, max_limit]

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

protected:
    // can be changed in derived constructor

    /// workload  input channels restrictions
    const std::unordered_map<Operation, MultiSmartRanges> input_channels_restrictions{
            {Operation::CONVOLUTION, MultiSmartRanges{{SmartRanges(16, input_spatial_dim_max, 16)}}},
            {Operation::DW_CONVOLUTION, MultiSmartRanges{{SmartRanges(16, 64, 16, 32)}}},  // special case {16, 32, 64}
            {Operation::CM_CONVOLUTION, MultiSmartRanges{{SmartRanges(1, cm_conv_channels_max - 1)}}},
            {Operation::ELTWISE, MultiSmartRanges{{SmartRanges(16, input_spatial_dim_max, 16)}}},
            {Operation::MAXPOOL, MultiSmartRanges{{SmartRanges(16, 64, 16, 32)}}},  // special case {16, 32, 64}
    };

    const MultiSmartRanges output_channels_restrictions{{SmartRanges(16, input_spatial_dim_max, 16)}};

    const MultiSmartRanges batch_restrictions{{allValues_range}};  // valid batch values, we accept any value
};

//////// LAYER UNSPLIT situation
/// @brief specific VPU 4.0 configuration possibilities for  layer
class VPU4_0_LayerValidValues : public VPU4_0_WorkloadValidValues {
private:
    inline static const int input_heigth_start_factor_SOH_def{2};

    inline static const std::unordered_map<Operation, MultiSmartRanges> input_channels_restrictions_layer{
            {Operation::CONVOLUTION, MultiSmartRanges{{SmartRanges(16, input_spatial_dim_max, 16)}}},
            {Operation::DW_CONVOLUTION, MultiSmartRanges{{SmartRanges(16, input_spatial_dim_max, 16)}}},
            {Operation::CM_CONVOLUTION, MultiSmartRanges{{SmartRanges(1, cm_conv_channels_max - 1)}}},  // why 2
            {Operation::ELTWISE, MultiSmartRanges{{SmartRanges(16, input_spatial_dim_max, 16)}}},
            {Operation::MAXPOOL, MultiSmartRanges{{SmartRanges(16, input_spatial_dim_max, 16)}}},
    };

public:
    /// constructor with link to operations dynamic behavior
    VPU4_0_LayerValidValues(const IContainer_OperationsDynamicBehavior& op_dynamic_constraints)
            : VPU4_0_WorkloadValidValues(op_dynamic_constraints,  //
                                         input_heigth_start_factor_SOH_def, input_channels_restrictions_layer) {
        // at layer level we are not limited like for workload level
    }

    MultiSmartRanges get_output_channels_restriction(const DPUOperation& dpu) const override {
        return (dpu.isi_strategy == ISIStrategy::SPLIT_OVER_K)
                       ? output_channels_restrictions.multiply_lower(2)  // 1 bcoms 2 or 16 becomes 32
                       : output_channels_restrictions;
    }

    bool mustExecuteHWLowLevelChecks(const DPUOperation& /*dpu*/) const noexcept override {
        return false;  // layer is above the workload, no need to check
    };

    Values<ISIStrategy> get_ISI_Strategy_Range(const DPUOperation& dpu) const override {
        // at layer level this is the desired strategy, no limitation based on tiles/output write tiles desired
        Values<ISIStrategy> v{isi_stategy_options};
        const auto& behavior = operations_dynamic_behavior.get_operation_specific_behaviour(dpu.operation);
        return behavior.filter_ISI_Strategy_Options(v);
    }

protected:
};

//////// LAYER SPLIT on tile  situation,

/// @brief specific VPU 4.0 configuration possibilities for  layer already split on tile.
/// channels restrictions are less strict vs workload, since a further split is expected
class VPU4_0_LayerOnTileValidValues : public VPU4_0_WorkloadValidValues {
private:
    inline static const std::unordered_map<Operation, MultiSmartRanges> input_channels_restrictions_layersplit{
            {Operation::CONVOLUTION, MultiSmartRanges{{SmartRanges(16, input_spatial_dim_max, 16)}}},
            {Operation::DW_CONVOLUTION, MultiSmartRanges{{SmartRanges(16, input_spatial_dim_max, 16)}}},
            {Operation::CM_CONVOLUTION, MultiSmartRanges{{SmartRanges(1, cm_conv_channels_max - 1)}}},
            {Operation::ELTWISE, MultiSmartRanges{{SmartRanges(16, input_spatial_dim_max, 16)}}},
            {Operation::MAXPOOL, MultiSmartRanges{{SmartRanges(16, input_spatial_dim_max, 16)}}},
    };

public:
    /// constructor with link to operations dynamic behavior
    VPU4_0_LayerOnTileValidValues(const IContainer_OperationsDynamicBehavior& op_dynamic_constraints)
            : VPU4_0_WorkloadValidValues(op_dynamic_constraints, input_channels_restrictions_layersplit) {
        // at layer level we are not limited like for workload level
    }

    bool mustExecuteHWLowLevelChecks(const DPUOperation& /*dpu*/) const noexcept override {
        return false;  // layer is above the workload, no need to check
    };

protected:
};

}  // namespace VPUNN

#endif  //
