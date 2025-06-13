// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_DEVICE_VALID_VALUES_VPU2_H
#define VPUNN_DEVICE_VALID_VALUES_VPU2_H

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

/// @brief specific VPU2.0 configuration possibilities, for workload, not layer
/* coverity[rule_of_five_violation:FALSE] */
class VPU2_0_WorkloadValidValues : public IDeviceValidValues {
public:
    VPU2_0_WorkloadValidValues(const VPU2_0_WorkloadValidValues&) noexcept(false) = default;
    VPU2_0_WorkloadValidValues& operator=(const VPU2_0_WorkloadValidValues&) = delete;

    VPU2_0_WorkloadValidValues(VPU2_0_WorkloadValidValues&&) noexcept(false) = default;
    VPU2_0_WorkloadValidValues& operator=(VPU2_0_WorkloadValidValues&&) = delete;

    ~VPU2_0_WorkloadValidValues() = default;

private:
    inline static const Values<ExecutionMode> valid_execution_order_all{
            ExecutionMode::VECTOR_FP16,
            ExecutionMode::VECTOR,
            ExecutionMode::MATRIX,
    };  // 4x1, 16x1, 4x4
    inline static const Values<Swizzling> valid_swizzlings_def{Swizzling::KEY_0};
    inline static const Values<Layout> valid_layouts_def{
            Layout::ZMAJOR,
            Layout::CMAJOR,
    };  // Layout.ZXY and XYZ;
    inline static const Values<VPUDevice> devices_def{
            VPUDevice::VPU_2_0,
            VPUDevice::VPU_2_1,
    };

    inline static const std::unordered_map<VPUDevice, int> cmx_KB_sizes_def{
            {VPUDevice::VPU_2_0, 1024},
            {VPUDevice::VPU_2_1, 1024},
    };

    inline static const Values<int> output_write_tile_options_def{1};
    inline static const Values<ISIStrategy> isi_stategy_options_def{ISIStrategy::CLUSTERING};

    inline static const int weigths_alignment_def{16};
    inline static const int input_heigth_start_factor_SOH_def{1};

    static constexpr int alignement_size_bytes_def{16384};  // 16KB

    // input and output data types should be the same
    inline static const Values<DataType> valid_in_out_datatypes{
            DataType::INT8,
            DataType::UINT8,
            DataType::FLOAT16,
            DataType::BFLOAT16,
    };

    inline static const Values<DataType> valid_wt_datatypes = {
            DataType::INT8,
            DataType::UINT8,
            DataType::FLOAT16,
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
    VPU2_0_WorkloadValidValues(const IContainer_OperationsDynamicBehavior& op_dynamic_constraints)
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
                                 alignement_size_bytes_def){};

    /// constructor with link to operations dynamic behavior and what config can be overridden
    VPU2_0_WorkloadValidValues(const IContainer_OperationsDynamicBehavior& op_dynamic_constraints,  //
                               const int& input_heigth_start_factor_SOH_)
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
                                 alignement_size_bytes_def){};

protected:
    SmartRanges get_output_channels_restriction(const DPUOperation&) const override {
        return output_channels_restrictions;
    }

    SmartRanges get_input_channels_restriction(const DPUOperation& dpu) const override {
        return input_channels_restrictions.at(dpu.operation);
    }

    Layout adapt_device_comaptible_tensor_layout(Layout layout) const override {
        if (layout == Layout::ZXY)  // default
        {
            layout = Layout::ZMAJOR;
        } else if (layout == Layout::XYZ) {
            layout = Layout::CMAJOR;
        }

        return layout;
    };

    Swizzling adapt_device_comaptible_swizzling(Swizzling /*swizz*/) const override {
        return get_default_swizzling();  // this is the only supported in VPU2.0
    };

protected:  // only const attributes can be visible in derived
    /// layer input channels restrictions
    const std::unordered_map<Operation, SmartRanges> input_channels_restrictions{
            {Operation::CONVOLUTION, SmartRanges(16, input_spatial_dim_max, 16)},
            {Operation::DW_CONVOLUTION, SmartRanges(16, input_spatial_dim_max, 16)},
            {Operation::CM_CONVOLUTION, SmartRanges(1, cm_conv_channels_max)},  // why 2?
            {Operation::ELTWISE, SmartRanges(16, input_spatial_dim_max, 16)},
            {Operation::MAXPOOL, SmartRanges(16, input_spatial_dim_max, 16)},
    };
    const SmartRanges output_channels_restrictions{SmartRanges(16, input_spatial_dim_max, 16)};
};

//////// LAYER UNSPLIT situation

/// @brief specific VPU2.0 configuration possibilities, for  layer
class VPU2_0_LayerValidValues : public VPU2_0_WorkloadValidValues {
private:
    inline static const int input_heigth_start_factor_SOH_def{2};

public:
    /// constructor with link to operations dynamic behavior
    VPU2_0_LayerValidValues(const IContainer_OperationsDynamicBehavior& op_dynamic_constraints)
            : VPU2_0_WorkloadValidValues(op_dynamic_constraints, input_heigth_start_factor_SOH_def){};

    SmartRanges get_output_channels_restriction(const DPUOperation& dpu) const override {
        return (dpu.isi_strategy == ISIStrategy::SPLIT_OVER_K) ? output_channels_restrictions.multiply_lower(2)
                                                               : output_channels_restrictions;
    }

    bool mustExecuteHWLowLevelChecks(const DPUOperation& /*dpu*/) const noexcept override {
        return false;  // layer is above the workload, no need to check
    };

protected:
};

//////// LAYER SPLIT on tile  situation,

/// @brief specific VPU2.0 configuration possibilities, for  layer already split on tile.
/// channels restrictions are less strict vs workload, since a further split is expected
class VPU2_0_LayerOnTileValidValues : public VPU2_0_WorkloadValidValues {
public:
    /// constructor with link to operations dynamic behavior
    VPU2_0_LayerOnTileValidValues(const IContainer_OperationsDynamicBehavior& op_dynamic_cosntraints)
            : VPU2_0_WorkloadValidValues(op_dynamic_cosntraints){};

    bool mustExecuteHWLowLevelChecks(const DPUOperation& /*dpu*/) const noexcept override {
        return false;  // layer is above the workload, no need to check
    };

protected:
};

}  // namespace VPUNN

#endif  //
