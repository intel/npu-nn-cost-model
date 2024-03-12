// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_DEVICE_VALID_VALUES_H
#define VPUNN_DEVICE_VALID_VALUES_H

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
#include "vpu/types.h"

#include "device_valid_values_reserved.h"

namespace VPUNN {

/// @brief specific VPU2.0 configuration possibilities, for workload, not layer
class VPU2_0_WorkloadValidValues : public IDeviceValidValues {
private:
public:
    /// constructor with link to operations dynamic behavior
    VPU2_0_WorkloadValidValues(const IContainer_OperationsDynamicBehavior& op_dynamic_constraints)
            : IDeviceValidValues(op_dynamic_constraints) {
        valid_execution_order = {
                ExecutionMode::VECTOR_FP16,
                ExecutionMode::VECTOR,
                ExecutionMode::MATRIX,
        };  // 4x1, 16x1, 4x4
        default_swizzling = Swizzling::KEY_0;
        valid_swizzlings = {default_swizzling};
        valid_layouts = {
                Layout::ZMAJOR,
                Layout::CMAJOR,
        };  // Layout.ZXY and XYZ

        devices = {
                VPUDevice::VPU_2_0,
                VPUDevice::VPU_2_1,
        };
        cmx_KB_sizes = {
                {VPUDevice::VPU_2_0, 1024},
                {VPUDevice::VPU_2_1, 1024},
        };
        output_write_tile_options = {1};
        isi_stategy_options = {ISIStrategy::CLUSTERING};
    };

protected:
    const Channels& get_output_channels_range(const DPUOperation&) const override {
        return out_channels_trivial_range;
    }

    const Channels& get_input_channels_range(const DPUOperation& dpu) const override {
        return valid_input_channels.at(dpu.operation);
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
        return Swizzling::KEY_0;  // this is the only supported in VPU2.0
    };

protected:
    const std::unordered_map<Operation, Channels> valid_input_channels{
            {Operation::CONVOLUTION, makeList(1, channels_max, 16)},         //
            {Operation::DW_CONVOLUTION, makeList(1, channels_max, 16)},      //
            {Operation::CM_CONVOLUTION, makeList(2, cm_conv_channels_max)},  //
            {Operation::ELTWISE, makeList(1, channels_max, 16)},             //
            {Operation::MAXPOOL, makeList(1, channels_max, 16)},             //
    };
    const Channels out_channels_trivial_range{makeList(1, channels_max, 16)};  ///< out ch range basic
};

/// @brief specific VPU 2.7 configuration possibilities for workload, not layer
class VPU2_7_WorkloadValidValues : public IDeviceValidValues {
private:
public:
    /// constructor with link to operations dynamic behavior
    VPU2_7_WorkloadValidValues(const IContainer_OperationsDynamicBehavior& op_dynamic_cosntraints)
            : IDeviceValidValues(op_dynamic_cosntraints) {
        valid_execution_order = {
                ExecutionMode::CUBOID_4x16,
                ExecutionMode::CUBOID_8x16,
                ExecutionMode::CUBOID_16x16,
        };  //

        default_swizzling = Swizzling::KEY_0;
        valid_swizzlings = {Swizzling::KEY_0, Swizzling::KEY_1, Swizzling::KEY_2,
                            Swizzling::KEY_3, Swizzling::KEY_4, Swizzling::KEY_5};  //
        valid_layouts = {
                Layout::ZXY /*default one, ZMAJOR like*/,
                Layout::XYZ,
                Layout::XZY,
                Layout::YXZ,
                Layout::YZX,
                Layout::ZYX,
        };  //< not allowing INVALID, it is only for input_1 in particular conditions

        devices = {
                VPUDevice::VPU_2_7,
        };
        cmx_KB_sizes = {{devices[0], (2 * 1024 * 100) / 100}};  // memory increased with 0%
        output_write_tile_options = {1, 2};
        isi_stategy_options = {
                ISIStrategy::CLUSTERING,
                ISIStrategy::SPLIT_OVER_H,
                ISIStrategy::SPLIT_OVER_K,
        };  ///< full list
    };

    const Channels& get_output_channels_range(const DPUOperation& /*dpu*/) const override {
        return out_channels_trivial_range;
    }

    const Channels& get_input_channels_range(const DPUOperation& dpu) const override {
        const auto& ch_map{valid_input_channels};
        return ch_map.at(dpu.operation);
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
    const Channels out_channels_trivial_range{makeList(1, channels_max, 16)};  ///< out ch range basic

    // can be changed in derived constructor
    std::unordered_map<Operation, Channels> valid_input_channels{
            {Operation::CONVOLUTION, makeList(1, channels_max, 16)},             //
            {Operation::DW_CONVOLUTION, {16, 32, 64}},                           // {16, 32, 64}    //C=K
            {Operation::CM_CONVOLUTION, makeList(2, cm_conv_channels_max - 1)},  // CM_Conv is ConvCompressed for VPU2.7
            {Operation::ELTWISE, makeList(1, channels_max, 16)},                 // C=K
            {Operation::MAXPOOL, {16, 32, 64}},                                  // {16, 32, 64}    //C=K
    };
};

//////// LAYER UNSPLIT situation

/// @brief specific VPU2.0 configuration possibilities, for  layer
class VPU2_0_LayerValidValues : public VPU2_0_WorkloadValidValues {
public:
    /// constructor with link to operations dynamic behavior
    VPU2_0_LayerValidValues(const IContainer_OperationsDynamicBehavior& op_dynamic_cosntraints)
            : VPU2_0_WorkloadValidValues(op_dynamic_cosntraints) {
        input_heigth_start_factor_SOH = 2;
    };

    const Channels& get_output_channels_range(const DPUOperation& dpu) const override {
        return (dpu.isi_strategy == ISIStrategy::SPLIT_OVER_K) ? out_channels_trivial_range_SOK
                                                               : out_channels_trivial_range;
    }

protected:
    /// for SOK the output must be at least 32? to be split by K
    const Channels out_channels_trivial_range_SOK{makeList(2, channels_max, 16)};
};

/// @brief specific VPU 2.7 configuration possibilities for  layer
class VPU2_7_LayerValidValues : public VPU2_7_WorkloadValidValues {
private:
public:
    /// constructor with link to operations dynamic behavior
    VPU2_7_LayerValidValues(const IContainer_OperationsDynamicBehavior& op_dynamic_cosntraints)
            : VPU2_7_WorkloadValidValues(op_dynamic_cosntraints) {
        input_heigth_start_factor_SOH = 2;

        // at layer level we are not limited like for workload level
        valid_input_channels = std::unordered_map<Operation, Channels>{
                {Operation::CONVOLUTION, makeList(1, channels_max, 16)},     //
                {Operation::DW_CONVOLUTION, makeList(1, channels_max, 16)},  // {16, 32, 64}    //C=K
                {Operation::CM_CONVOLUTION,
                 makeList(2, cm_conv_channels_max - 1)},              // CM_Conv is ConvCompressed for VPU2.7
                {Operation::ELTWISE, makeList(1, channels_max, 16)},  // C=K
                {Operation::MAXPOOL, makeList(1, channels_max, 16)},  // {16, 32, 64}    //C=K
        };
    }
    const Channels& get_output_channels_range(const DPUOperation& dpu) const override {
        return (dpu.isi_strategy == ISIStrategy::SPLIT_OVER_K) ? out_channels_trivial_range_SOK
                                                               : out_channels_trivial_range;
    }

    const Channels& get_input_channels_range(const DPUOperation& dpu) const override {
        const auto& ch_map{(dpu.isi_strategy == ISIStrategy::SPLIT_OVER_K) ? valid_input_channels_SOK
                                                                           : valid_input_channels};
        return ch_map.at(dpu.operation);
    }

    Values<ISIStrategy> get_ISI_Strategy_Range(const DPUOperation& dpu) const noexcept override {
        // at layer level this is the desired strategy, no limitation based on tiles/output write tiles desired
        Values<ISIStrategy> v{isi_stategy_options};
        const auto& behavior = operations_dynamic_behavior.get_operation_specific_behaviour(dpu.operation);
        return behavior.filter_ISI_Strategy_Options(v);
    }

protected:
    /// for SOK the output must be at least 32? to be split by K
    const Channels out_channels_trivial_range_SOK{makeList(2, channels_max, 16)};

    // alternative in case ISI ==SOK, limited to be split by K
    const std::unordered_map<Operation, Channels> valid_input_channels_SOK{
            {Operation::CONVOLUTION, makeList(1, channels_max, 16)},             //
            {Operation::DW_CONVOLUTION, makeList(2, channels_max, 16)},          // min 32 channels   //C=K
            {Operation::CM_CONVOLUTION, makeList(2, cm_conv_channels_max - 1)},  // CM_Conv is ConvCompressed for VPU2.7
            {Operation::ELTWISE, makeList(2, channels_max, 16)},                 // min 32 channels    //C=K
            {Operation::MAXPOOL, makeList(1, channels_max, 16)},                 // min 32 channels    //C=K
    };
};

//////// LAYER SPLIT on tile  situation,

/// @brief specific VPU2.0 configuration possibilities, for  layer already split on tile.
/// channels restrictions are less strict vs workload, since a further split is expected
class VPU2_0_LayerOnTileValidValues : public VPU2_0_WorkloadValidValues {
public:
    /// constructor with link to operations dynamic behavior
    VPU2_0_LayerOnTileValidValues(const IContainer_OperationsDynamicBehavior& op_dynamic_cosntraints)
            : VPU2_0_WorkloadValidValues(op_dynamic_cosntraints){};

protected:
};

/// @brief specific VPU 2.7 configuration possibilities for  layer already split on tile.
/// channels restrictions are less strict vs workload, since a further split is expected
class VPU2_7_LayerOnTileValidValues : public VPU2_7_WorkloadValidValues {
private:
public:
    /// constructor with link to operations dynamic behavior
    VPU2_7_LayerOnTileValidValues(const IContainer_OperationsDynamicBehavior& op_dynamic_cosntraints)
            : VPU2_7_WorkloadValidValues(op_dynamic_cosntraints) {
        // at layer level we are not limited like for workload level
        valid_input_channels = std::unordered_map<Operation, Channels>{
                {Operation::CONVOLUTION, makeList(1, channels_max, 16)},     //
                {Operation::DW_CONVOLUTION, makeList(1, channels_max, 16)},  // {16, 32, 64}    //C=K
                {Operation::CM_CONVOLUTION,
                 makeList(2, cm_conv_channels_max - 1)},              // CM_Conv is ConvCompressed for VPU2.7
                {Operation::ELTWISE, makeList(1, channels_max, 16)},  // C=K
                {Operation::MAXPOOL, makeList(1, channels_max, 16)},  // {16, 32, 64}    //C=K
        };
    }

protected:
};

}  // namespace VPUNN

#endif  //
