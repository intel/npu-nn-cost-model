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

class Wrapper_DPUOperation : public DPUOperation {
public:
    using DPUOperation::is_preconditions_for_inplace_output;
    using DPUOperation::is_special_No_weights_situation;
    Wrapper_DPUOperation(const DPUOperation& op): DPUOperation(op) { }
};
class Wrapper_DPUWorkload : public DPUWorkload {
public:
    using DPUWorkload::is_preconditions_for_inplace_output;
    using DPUWorkload::is_special_No_weights_situation;
};

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

    const Wrapper_DPUWorkload _dpu_wl{dpu_wl};
    const Wrapper_DPUOperation _dpu_op{dpu_op};
};

TEST_F(DPUOp_vs_DPUWl_Equivalence_Functions, Function_is_elementwise_like_operation_Test) {
    EXPECT_EQ(dpu_wl.is_elementwise_like_operation(), dpu_op.is_elementwise_like_operation());
}

TEST_F(DPUOp_vs_DPUWl_Equivalence_Functions, Function_is_special_No_weights_situation_Test) {
    EXPECT_EQ(_dpu_wl.is_special_No_weights_situation(), _dpu_op.is_special_No_weights_situation());
}

TEST_F(DPUOp_vs_DPUWl_Equivalence_Functions, Function_is_preconditions_for_inplace_output_Test) {
    EXPECT_EQ(_dpu_wl.is_preconditions_for_inplace_output(), _dpu_op.is_preconditions_for_inplace_output());
}

}  // namespace VPUNN_unit_tests