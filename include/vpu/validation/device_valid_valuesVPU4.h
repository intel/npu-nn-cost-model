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
class VPU4_0_WorkloadValidValues : public IDeviceValidValues {
public:
    VPU4_0_WorkloadValidValues(const VPU4_0_WorkloadValidValues&) noexcept(false) = default;
    VPU4_0_WorkloadValidValues& operator=(const VPU4_0_WorkloadValidValues&) = delete;

    VPU4_0_WorkloadValidValues(VPU4_0_WorkloadValidValues&&) noexcept(false) = default;
    VPU4_0_WorkloadValidValues& operator=(VPU4_0_WorkloadValidValues&&) = delete;

    ~VPU4_0_WorkloadValidValues() = default;

private:
    inline static const Values<ExecutionMode> valid_execution_order_def{
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
            ISIStrategy::SPLIT_OVER_H,
            ISIStrategy::SPLIT_OVER_K,
    };

    inline static const int weigths_alignment_def{32};
    inline static const int input_heigth_start_factor_SOH_def{1};

    // input and output data types should be the same
    inline static const Values<DataType> valid_in_out_datatypes{
            DataType::INT8,      //
            DataType::UINT8,     //
            DataType::FLOAT16,   //
            DataType::BFLOAT16,  //
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
                    {Operation::CONVOLUTION, valid_in_out_datatypes},     //
                    {Operation::DW_CONVOLUTION, valid_in_out_datatypes},  //
                    {Operation::CM_CONVOLUTION, valid_in_out_datatypes},  //
                    {Operation::ELTWISE, valid_in_out_datatypes},         //
                    {Operation::MAXPOOL, valid_in_out_datatypes},         //
            },
            {
                    {Operation::CONVOLUTION, valid_in_out_datatypes},     //
                    {Operation::DW_CONVOLUTION, valid_in_out_datatypes},  //
                    {Operation::CM_CONVOLUTION, valid_in_out_datatypes},  //
                    {Operation::ELTWISE, valid_in_out_datatypes},         //
                    {Operation::MAXPOOL, valid_in_out_datatypes},         //
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

public:
    /// constructor with link to operations dynamic behavior
    VPU4_0_WorkloadValidValues(const IContainer_OperationsDynamicBehavior& op_dynamic_constraints)
            : IDeviceValidValues(op_dynamic_constraints,             //
                                 valid_execution_order_def,          //
                                 valid_swizzlings_def,               //
                                 valid_layouts_def,                  //
                                 devices_def,                        //
                                 cmx_KB_sizes_def,                   //
                                 output_write_tile_options_def,      //
                                 isi_stategy_options_def,            //
                                 weigths_alignment_def,              //
                                 input_heigth_start_factor_SOH_def,  //
                                 valid_datatypes_map_default,        //
                                 valid_operations_default){};

    /// constructor with link to operations dynamic behavior, input channels rules and restrictions
    VPU4_0_WorkloadValidValues(const IContainer_OperationsDynamicBehavior& op_dynamic_constraints,
                               const std::unordered_map<Operation, SmartRanges>& input_channels_restrictions_)
            : IDeviceValidValues(op_dynamic_constraints,
                                 valid_execution_order_def,          //
                                 valid_swizzlings_def,               //
                                 valid_layouts_def,                  //
                                 devices_def,                        //
                                 cmx_KB_sizes_def,                   //
                                 output_write_tile_options_def,      //
                                 isi_stategy_options_def,            //
                                 weigths_alignment_def,              //
                                 input_heigth_start_factor_SOH_def,  //
                                 valid_datatypes_map_default,        //
                                 valid_operations_default),
              input_channels_restrictions{input_channels_restrictions_} {};

    /// constructor with link to operations dynamic behavior and what config can be overridden (and input channels
    /// rules)
    VPU4_0_WorkloadValidValues(const IContainer_OperationsDynamicBehavior& op_dynamic_constraints,  //
                               const int& input_heigth_start_factor_SOH_,
                               const std::unordered_map<Operation, SmartRanges>& input_channels_restrictions_)
            : IDeviceValidValues(op_dynamic_constraints,
                                 valid_execution_order_def,       //
                                 valid_swizzlings_def,            //
                                 valid_layouts_def,               //
                                 devices_def,                     //
                                 cmx_KB_sizes_def,                //
                                 output_write_tile_options_def,   //
                                 isi_stategy_options_def,         //
                                 weigths_alignment_def,           //
                                 input_heigth_start_factor_SOH_,  // special
                                 valid_datatypes_map_default,     //
                                 valid_operations_default),
              input_channels_restrictions{input_channels_restrictions_} {};

    SmartRanges get_output_channels_restriction(const DPUOperation&) const override {
        return output_channels_restrictions;
    }

    SmartRanges get_input_channels_restriction(const DPUOperation& dpu) const override {
        return input_channels_restrictions.at(dpu.operation);
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

protected:
    // can be changed in derived constructor

    /// workload  input channels restrictions
    const std::unordered_map<Operation, SmartRanges> input_channels_restrictions{
            {Operation::CONVOLUTION, SmartRanges(16, input_spatial_dim_max, 16)},
            {Operation::DW_CONVOLUTION, SmartRanges(16, 64, 16, 32)},  // special case {16, 32, 64}
            {Operation::CM_CONVOLUTION, SmartRanges(1, cm_conv_channels_max - 1)},
            {Operation::ELTWISE, SmartRanges(16, input_spatial_dim_max, 16)},
            {Operation::MAXPOOL, SmartRanges(16, 64, 16, 32)},  // special case {16, 32, 64}
    };
    const SmartRanges output_channels_restrictions{SmartRanges(16, input_spatial_dim_max, 16)};
};

//////// LAYER UNSPLIT situation
/// @brief specific VPU 4.0 configuration possibilities for  layer
class VPU4_0_LayerValidValues : public VPU4_0_WorkloadValidValues {
private:
    inline static const int input_heigth_start_factor_SOH_def{2};

    inline static const std::unordered_map<Operation, SmartRanges> input_channels_restrictions_layer{
            {Operation::CONVOLUTION, SmartRanges(16, input_spatial_dim_max, 16)},
            {Operation::DW_CONVOLUTION, SmartRanges(16, input_spatial_dim_max, 16)},
            {Operation::CM_CONVOLUTION, SmartRanges(1, cm_conv_channels_max - 1)},  // why 2
            {Operation::ELTWISE, SmartRanges(16, input_spatial_dim_max, 16)},
            {Operation::MAXPOOL, SmartRanges(16, input_spatial_dim_max, 16)},
    };

public:
    /// constructor with link to operations dynamic behavior
    VPU4_0_LayerValidValues(const IContainer_OperationsDynamicBehavior& op_dynamic_constraints)
            : VPU4_0_WorkloadValidValues(op_dynamic_constraints,  //
                                         input_heigth_start_factor_SOH_def, input_channels_restrictions_layer) {
        // at layer level we are not limited like for workload level
    }

    SmartRanges get_output_channels_restriction(const DPUOperation& dpu) const override {
        return (dpu.isi_strategy == ISIStrategy::SPLIT_OVER_K)
                       ? output_channels_restrictions.multiply_lower(2)  // 1 bcoms 2 or 16 becomes 32
                       : output_channels_restrictions;
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

/// @brief specific VPU 4.0 configuration possibilities for  layer already split on tile.
/// channels restrictions are less strict vs workload, since a further split is expected
class VPU4_0_LayerOnTileValidValues : public VPU4_0_WorkloadValidValues {
private:
    inline static const std::unordered_map<Operation, SmartRanges> input_channels_restrictions_layersplit{
            {Operation::CONVOLUTION, SmartRanges(16, input_spatial_dim_max, 16)},
            {Operation::DW_CONVOLUTION, SmartRanges(16, input_spatial_dim_max, 16)},
            {Operation::CM_CONVOLUTION, SmartRanges(1, cm_conv_channels_max - 1)},
            {Operation::ELTWISE, SmartRanges(16, input_spatial_dim_max, 16)},
            {Operation::MAXPOOL, SmartRanges(16, input_spatial_dim_max, 16)},
    };

public:
    /// constructor with link to operations dynamic behavior
    VPU4_0_LayerOnTileValidValues(const IContainer_OperationsDynamicBehavior& op_dynamic_constraints)
            : VPU4_0_WorkloadValidValues(op_dynamic_constraints, input_channels_restrictions_layersplit) {
        // at layer level we are not limited like for workload level
    }

protected:
};

}  // namespace VPUNN

#endif  //
