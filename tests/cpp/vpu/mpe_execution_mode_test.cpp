// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
// #include "vpu_cost_model.h"
#include "vpu/validation/dpu_operations_sanitizer.h"

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

class ExecutionOrderTest : public ::testing::Test {
protected:
    static DPUWorkload mkWl(VPUDevice dev, Operation op, ExecutionMode exec, unsigned int in_ch = 16,
                            unsigned int out_ch = 16, Layout layout = Layout::ZXY) {
        VPUNN::DPUWorkload wl{
                dev,
                op,
                {VPUNN::VPUTensor(15, 50, in_ch, 1, VPUNN::DataType::UINT8, layout)},   // input dimensions
                {VPUNN::VPUTensor(15, 50, out_ch, 1, VPUNN::DataType::UINT8, layout)},  // output dimensions
                {1, 1},                                                                 // kernels
                {1, 1},                                                                 // strides
                {0, 0, 0, 0},                                                           // padding
                exec,                                                                   // execution mode
                VPUNN::ActivationFunction::NONE,                                        // activation
                0.0F,                                                                   // act_sparsity
                0.0F,                                                                   // weight_sparsity
                {Swizzling::KEY_0, Swizzling::KEY_0},                                   // input_swizzling
                {Swizzling::KEY_0},                                                     // output_swizzling
                1,                                                                      // output_write_tiles
                {0, 0, 0, 0},                                                           // offsets
                VPUNN::ISIStrategy::CLUSTERING,                                         // isi_strategy
                false,                                                                  // weight_sparsity_enabled
        };

        return wl;
    }

    VPUNN::DPU_OperationSanitizer dut;

    struct TestInput {
        DPUWorkload wl;
    };

    struct TestExpectation {
        CyclesInterfaceType err_expected;
    };

    struct TestCase {
        TestInput t_in;
        TestExpectation t_exp;
    };

    using TestsVector = std::vector<TestCase>;

    static void check_err(TestsVector& tests, DPU_OperationSanitizer& dut) {
        SanityReport sane;

        for (auto& t : tests) {
            std::cout << "Device:" << VPUDevice_ToText.at(static_cast<int>(t.t_in.wl.device))
                      << " Operation:" << Operation_ToText.at(static_cast<int>(t.t_in.wl.op))
                      << " Exec order:" << ExecutionMode_ToText.at(static_cast<int>(t.t_in.wl.execution_order)) << "\n";

            dut.check_data_consistency(t.t_in.wl, sane);

            EXPECT_EQ(sane.value(), V(t.t_exp.err_expected))
                    << sane.info << "\n error is : " << Cycles::toErrorText(sane.value()) << "\n";
        }
    }
};

TEST_F(ExecutionOrderTest, Execution_order_Test_VPU20) {
    VPUDevice device = VPUDevice::VPU_2_0;

    TestsVector tests = {
            // clang-format off
        {{mkWl(device, Operation::CONVOLUTION, ExecutionMode::CUBOID_16x16, 16, 16, Layout::ZMAJOR)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::CONVOLUTION, ExecutionMode::CUBOID_8x16,  16, 16, Layout::ZMAJOR)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::CONVOLUTION, ExecutionMode::CUBOID_4x16,  16, 16, Layout::ZMAJOR)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
               
        {{mkWl(device, Operation::DW_CONVOLUTION, ExecutionMode::CUBOID_16x16, 16, 16, Layout::ZMAJOR)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::DW_CONVOLUTION, ExecutionMode::CUBOID_8x16,  16, 16, Layout::ZMAJOR)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::DW_CONVOLUTION, ExecutionMode::CUBOID_4x16,  16, 16, Layout::ZMAJOR)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
               
        {{mkWl(device, Operation::CM_CONVOLUTION, ExecutionMode::CUBOID_16x16, 16, 16,  Layout::ZMAJOR)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::CM_CONVOLUTION, ExecutionMode::CUBOID_8x16,  16, 16,  Layout::ZMAJOR)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::CM_CONVOLUTION, ExecutionMode::CUBOID_4x16,  16, 16,  Layout::ZMAJOR)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
                                                                               
        {{mkWl(device, Operation::ELTWISE, ExecutionMode::CUBOID_16x16, 16, 16,  Layout::ZMAJOR)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::ELTWISE, ExecutionMode::CUBOID_8x16,  16, 16,  Layout::ZMAJOR)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::ELTWISE, ExecutionMode::CUBOID_4x16,  16, 16,  Layout::ZMAJOR)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
                                                                        
        {{mkWl(device, Operation::MAXPOOL, ExecutionMode::CUBOID_16x16, 16, 16,  Layout::ZMAJOR)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::MAXPOOL, ExecutionMode::CUBOID_8x16,  16, 16,  Layout::ZMAJOR)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::MAXPOOL, ExecutionMode::CUBOID_4x16,  16, 16,  Layout::ZMAJOR)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
               
        {{mkWl(device, Operation::CONVOLUTION, ExecutionMode::VECTOR, 16, 16, Layout::ZMAJOR)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::ELTWISE, ExecutionMode::VECTOR, 16, 16, Layout::ZMAJOR)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::DW_CONVOLUTION, ExecutionMode::VECTOR, 16, 16, Layout::ZMAJOR)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::CM_CONVOLUTION, ExecutionMode::VECTOR, 16, 16, Layout::ZMAJOR)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::MAXPOOL, ExecutionMode::VECTOR, 16, 16, Layout::ZMAJOR)},{Cycles::NO_ERROR}},
               
        {{mkWl(device, Operation::CONVOLUTION, ExecutionMode::MATRIX, 16, 16, Layout::ZMAJOR)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::ELTWISE, ExecutionMode::MATRIX, 16, 16, Layout::ZMAJOR)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::DW_CONVOLUTION, ExecutionMode::MATRIX, 16, 16, Layout::ZMAJOR)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::CM_CONVOLUTION, ExecutionMode::MATRIX, 16, 16, Layout::ZMAJOR)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::MAXPOOL, ExecutionMode::MATRIX, 16, 16, Layout::ZMAJOR)},{Cycles::NO_ERROR}},

            // clang-format on
    };

    check_err(tests, dut);
}

TEST_F(ExecutionOrderTest, Execution_order_Test_VPU27) {
    VPUDevice device = VPUDevice::VPU_2_7;

    TestsVector tests = {
            // clang-format off
        {{mkWl(device, Operation::CONVOLUTION, ExecutionMode::CUBOID_16x16)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::CONVOLUTION, ExecutionMode::CUBOID_8x16)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::CONVOLUTION, ExecutionMode::CUBOID_4x16)},{Cycles::NO_ERROR}},

        {{mkWl(device, Operation::DW_CONVOLUTION, ExecutionMode::CUBOID_16x16)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::DW_CONVOLUTION, ExecutionMode::CUBOID_8x16)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::DW_CONVOLUTION, ExecutionMode::CUBOID_4x16)},{Cycles::NO_ERROR}},
              
        {{mkWl(device, Operation::CM_CONVOLUTION, ExecutionMode::CUBOID_16x16)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::CM_CONVOLUTION, ExecutionMode::CUBOID_8x16)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::CM_CONVOLUTION, ExecutionMode::CUBOID_4x16)},{Cycles::NO_ERROR}},
              
        {{mkWl(device, Operation::ELTWISE, ExecutionMode::CUBOID_16x16)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::ELTWISE, ExecutionMode::CUBOID_8x16)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::ELTWISE, ExecutionMode::CUBOID_4x16)},{Cycles::NO_ERROR}},
             
        {{mkWl(device, Operation::MAXPOOL, ExecutionMode::CUBOID_16x16)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::MAXPOOL, ExecutionMode::CUBOID_8x16)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::MAXPOOL, ExecutionMode::CUBOID_4x16)},{Cycles::NO_ERROR}},
              
        {{mkWl(device, Operation::CONVOLUTION, ExecutionMode::VECTOR)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::DW_CONVOLUTION, ExecutionMode::VECTOR)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::CM_CONVOLUTION, ExecutionMode::VECTOR)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::ELTWISE, ExecutionMode::VECTOR)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::MAXPOOL, ExecutionMode::VECTOR)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
              
        {{mkWl(device, Operation::CONVOLUTION, ExecutionMode::MATRIX)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::DW_CONVOLUTION, ExecutionMode::MATRIX)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::CM_CONVOLUTION, ExecutionMode::MATRIX)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::ELTWISE, ExecutionMode::MATRIX)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::MAXPOOL, ExecutionMode::MATRIX)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},

            // clang-format on
    };

    check_err(tests, dut);
}

TEST_F(ExecutionOrderTest, Execution_order_Test_VPU40) {
    VPUDevice device = VPUDevice::VPU_4_0;

    TestsVector tests = {
            // clang-format off
        {{mkWl(device, Operation::CONVOLUTION, ExecutionMode::CUBOID_16x16)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::CONVOLUTION, ExecutionMode::CUBOID_8x16)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::CONVOLUTION, ExecutionMode::CUBOID_4x16)},{Cycles::NO_ERROR}},

        {{mkWl(device, Operation::DW_CONVOLUTION, ExecutionMode::CUBOID_16x16)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::DW_CONVOLUTION, ExecutionMode::CUBOID_8x16)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::DW_CONVOLUTION, ExecutionMode::CUBOID_4x16)},{Cycles::NO_ERROR}},
              
        {{mkWl(device, Operation::CM_CONVOLUTION, ExecutionMode::CUBOID_16x16, 15, 16)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::CM_CONVOLUTION, ExecutionMode::CUBOID_8x16, 15, 16)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::CM_CONVOLUTION, ExecutionMode::CUBOID_4x16, 15, 16)},{Cycles::NO_ERROR}},
              
        {{mkWl(device, Operation::ELTWISE, ExecutionMode::CUBOID_16x16)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::ELTWISE, ExecutionMode::CUBOID_8x16)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::ELTWISE, ExecutionMode::CUBOID_4x16)},{Cycles::NO_ERROR}},
             
        {{mkWl(device, Operation::MAXPOOL, ExecutionMode::CUBOID_16x16)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::MAXPOOL, ExecutionMode::CUBOID_8x16)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::MAXPOOL, ExecutionMode::CUBOID_4x16)},{Cycles::NO_ERROR}},
              
        {{mkWl(device, Operation::CONVOLUTION, ExecutionMode::VECTOR)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::DW_CONVOLUTION, ExecutionMode::VECTOR)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::CM_CONVOLUTION, ExecutionMode::VECTOR, 15, 16)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::ELTWISE, ExecutionMode::VECTOR)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::MAXPOOL, ExecutionMode::VECTOR)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
              
        {{mkWl(device, Operation::CONVOLUTION, ExecutionMode::MATRIX)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::DW_CONVOLUTION, ExecutionMode::MATRIX)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::CM_CONVOLUTION, ExecutionMode::MATRIX, 15, 16)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::ELTWISE, ExecutionMode::MATRIX)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::MAXPOOL, ExecutionMode::MATRIX)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},

            // clang-format on
    };

    check_err(tests, dut);
}

TEST_F(ExecutionOrderTest, Execution_order_Test_NPU50) {
    VPUDevice device = VPUDevice::NPU_5_0;

    TestsVector tests = {
            // clang-format off
        {{mkWl(device, Operation::CONVOLUTION, ExecutionMode::CUBOID_16x16)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::CONVOLUTION, ExecutionMode::CUBOID_8x16)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::CONVOLUTION, ExecutionMode::CUBOID_4x16)},{Cycles::NO_ERROR}},

        {{mkWl(device, Operation::DW_CONVOLUTION, ExecutionMode::CUBOID_16x16)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::DW_CONVOLUTION, ExecutionMode::CUBOID_8x16)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::DW_CONVOLUTION, ExecutionMode::CUBOID_4x16)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
              
        {{mkWl(device, Operation::CM_CONVOLUTION, ExecutionMode::CUBOID_16x16, 15, 16)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::CM_CONVOLUTION, ExecutionMode::CUBOID_8x16, 15, 16)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::CM_CONVOLUTION, ExecutionMode::CUBOID_4x16, 15, 16)},{Cycles::NO_ERROR}},
              
        {{mkWl(device, Operation::ELTWISE, ExecutionMode::CUBOID_16x16)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::ELTWISE, ExecutionMode::CUBOID_8x16)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::ELTWISE, ExecutionMode::CUBOID_4x16)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},

        {{mkWl(device, Operation::ELTWISE_MUL, ExecutionMode::CUBOID_16x16)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::ELTWISE_MUL, ExecutionMode::CUBOID_8x16)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::ELTWISE_MUL, ExecutionMode::CUBOID_4x16)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
             
        {{mkWl(device, Operation::MAXPOOL, ExecutionMode::CUBOID_16x16)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::MAXPOOL, ExecutionMode::CUBOID_8x16)},{Cycles::Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::MAXPOOL, ExecutionMode::CUBOID_4x16)},{Cycles::Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},

        {{mkWl(device, Operation::LAYER_NORM, ExecutionMode::CUBOID_16x16)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::LAYER_NORM, ExecutionMode::CUBOID_8x16)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::LAYER_NORM, ExecutionMode::CUBOID_4x16)},{Cycles::NO_ERROR}},
              
        {{mkWl(device, Operation::CONVOLUTION, ExecutionMode::VECTOR)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::DW_CONVOLUTION, ExecutionMode::VECTOR)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::CM_CONVOLUTION, ExecutionMode::VECTOR, 15, 16)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::ELTWISE, ExecutionMode::VECTOR)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::ELTWISE_MUL, ExecutionMode::VECTOR)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::MAXPOOL, ExecutionMode::VECTOR)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::LAYER_NORM, ExecutionMode::VECTOR)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
              
        {{mkWl(device, Operation::CONVOLUTION, ExecutionMode::MATRIX)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::DW_CONVOLUTION, ExecutionMode::MATRIX)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::CM_CONVOLUTION, ExecutionMode::MATRIX, 15, 16)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::ELTWISE, ExecutionMode::MATRIX)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::ELTWISE_MUL, ExecutionMode::MATRIX)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::MAXPOOL, ExecutionMode::MATRIX)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},
        {{mkWl(device, Operation::LAYER_NORM, ExecutionMode::MATRIX)},{Cycles::ERROR_INVALID_INPUT_CONFIGURATION}},

            // clang-format on
    };

    check_err(tests, dut);
}
}  // namespace VPUNN_unit_tests