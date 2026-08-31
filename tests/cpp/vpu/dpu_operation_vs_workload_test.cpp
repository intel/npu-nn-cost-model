// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
// #include "vpu_cost_model.h"
#include "vpu/validation/data_dpu_operation.h"

#include <gtest/gtest.h>
#include "common/common_helpers.h"
// #include "vpu/compatibility/types01.h"
// #include "vpu/cycles_interface_types.h"
// #include "vpu/sample_generator/random_task_generator.h"
// #include "vpu/validation/interface_valid_values.h"
// #include "vpu_dma_cost_model.h"
//
// #include "vpu/validation/dpu_operations_validator.h"
// #include "vpu/validation/memory_calculator.h"

#include <algorithm>
#include <unordered_map>

#include <optional>
#include <variant>

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {
using namespace VPUNN;

class DPUOp_vs_DPUWl_Equivalence_Functions : public ::testing::Test {
public:
    const DPUWorkload dpu_wl = {
            VPUNN::VPUDevice::NPU_5_0,
            VPUNN::Operation::ELTWISE,
            {VPUTensor(56, 56, 64, 1, DataType::INT8, Layout::ZXY)},  // input dimensions
            {VPUTensor(56, 56, 64, 1, DataType::INT8, Layout::ZXY)},  // output dimensions
            {1, 1},                                                   // kernels
            {1, 1},                                                   // strides
            {0, 0, 0, 0},                                             // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                       // execution mode
            VPUNN::ActivationFunction::NONE,                          // activation
            0.0F,                                                     // act_sparsity
            0.F,                                                      // weight_sparsity
            {swz_def, swz_def},                                       // input_swizzling
            {swz_def},                                                // output_swizzling
            1,                                                        // output_write_tiles
            {0, 0, 0, 0},                                             // offsets
            ISIStrategy::CLUSTERING,                                  // isi_strategy
            false,                                                    // weight_sparsity_enabled
    };

    const DPUOperation dpu_op{dpu_wl};
};

TEST_F(DPUOp_vs_DPUWl_Equivalence_Functions, Autopad_Flags_Preservation_Test) {
    // Verify input_autopad and output_autopad are preserved across DPUWorkload <--> DPUOperation conversions

    // Build a workload with both autopad flags set to true
    DPUWorkload wl_with_autopad = dpu_wl;
    wl_with_autopad.input_autopad = true;
    wl_with_autopad.output_autopad = true;

    // DPUWorkload -> DPUOperation: flags must be preserved 
    {
        const DPUOperation op_from_wl{wl_with_autopad};
        EXPECT_TRUE(op_from_wl.input_autopad) << "input_autopad lost during DPUWorkload -> DPUOperation";
        EXPECT_TRUE(op_from_wl.output_autopad) << "output_autopad lost during DPUWorkload -> DPUOperation";
    }
    // Also verify false -> false
    {
        const DPUOperation op_from_wl{dpu_wl};  // defaults: both false
        EXPECT_FALSE(op_from_wl.input_autopad) << "input_autopad should be false when source is false";
        EXPECT_FALSE(op_from_wl.output_autopad) << "output_autopad should be false when source is false";
    }

    // DPUOperation -> DPUWorkload (clone_as_DPUWorkload): flags must be preserved 
    {
        const DPUOperation op_with_autopad{wl_with_autopad};
        const DPUWorkload wl_cloned = op_with_autopad.clone_as_DPUWorkload();
        EXPECT_TRUE(wl_cloned.is_input_autopad()) << "input_autopad lost during DPUOperation -> DPUWorkload";
        EXPECT_TRUE(wl_cloned.is_output_autopad()) << "output_autopad lost during DPUOperation -> DPUWorkload";
    }
    // Also verify false -> false
    {
        const DPUWorkload wl_cloned = dpu_op.clone_as_DPUWorkload();  // defaults: both false
        EXPECT_FALSE(wl_cloned.is_input_autopad()) << "input_autopad should be false when source is false";
        EXPECT_FALSE(wl_cloned.is_output_autopad()) << "output_autopad should be false when source is false";
    }
}

}  // namespace VPUNN_unit_tests