// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/validation/dpu_operations_sanitizer.h"
#include "vpu/validation/dpu_operations_validator.h"
#include "vpu/validation/layer_sanitizer.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

#include "common/common_helpers.h"

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {
using namespace VPUNN;
class DPU_OperationSanitizerTestNPU5x : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }

private:
};

class DPU_WorkloadValidatorTestNPU5x : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }

private:
};

TEST_F(DPU_WorkloadValidatorTestNPU5x, Memory_Output_32Bits_NPU50) {
    DPU_OperationSanitizer dut;
    const VPUDevice device_req{VPUDevice::NPU_5_0};
    VPUNN::SanityReport sane;

    const DPUWorkload wl_ref{
            device_req,
            Operation::CONVOLUTION,
            {VPUTensor(64, 64, 64, 1, DataType::UINT8)},                     // input dimensions
            {VPUTensor(21, 21, 512 + 48, 1, DataType::FLOAT32)},             // output dimensions
            {3, 3},                                                          // kernels
            {3, 3},                                                          // strides
            {0, 0, 0, 0},                                                    // padding
            ExecutionMode::CUBOID_16x16,                                     // execution mode
            ActivationFunction::NONE,                                        // activation
            0.0F,                                                            // act_sparsity
            0.0F,                                                            // weight_sparsity
            {swz_def, swz_def},                                              // input_swizzling
            {swz_def},                                                       // output_swizzling
            1,                                                               // output_write_tiles
            {0, 0, 0, 0},                                                    // offsets
            ISIStrategy::CLUSTERING,                                         // isi_strategy
            false,                                                           // weight_sparsity_enabled
            {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects
            {false, {0, 0, 0, 0}, {0, 0, 0, 0}, false},                      // SEP
            DataType::UINT8,                                                 // input1 data type
    };

    {  //
        auto wl{std::move(wl_ref)};
        dut.check_data_consistency(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;

        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::ERROR_INPUT_TOO_BIG))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }
    const DPUWorkload wl_ref_less{
            device_req,
            Operation::CONVOLUTION,
            {VPUTensor(64, 64, 64, 1, DataType::UINT8)},                     // input dimensions
            {VPUTensor(21, 21, 512 + 32, 1, DataType::FLOAT32)},             // output dimensions
            {3, 3},                                                          // kernels
            {3, 3},                                                          // strides
            {0, 0, 0, 0},                                                    // padding
            ExecutionMode::CUBOID_16x16,                                     // execution mode
            ActivationFunction::NONE,                                        // activation
            0.0F,                                                            // act_sparsity
            0.0F,                                                            // weight_sparsity
            {swz_def, swz_def},                                              // input_swizzling
            {swz_def},                                                       // output_swizzling
            1,                                                               // output_write_tiles
            {0, 0, 0, 0},                                                    // offsets
            ISIStrategy::CLUSTERING,                                         // isi_strategy
            false,                                                           // weight_sparsity_enabled
            {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects
            {false, {0, 0, 0, 0}, {0, 0, 0, 0}, false},                      // SEP
            DataType::UINT8,                                                 // input1 data type
    };
    {  //
        auto wl{std::move(wl_ref_less)};
        dut.check_data_consistency(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;

        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }

    const DPUWorkload wl_ref2{
            device_req,
            Operation::ELTWISE,
            {VPUTensor(64, 64, 64, 1, DataType::UINT8)},                     // input dimensions
            {VPUTensor(64, 64, 64, 1, DataType::FLOAT32)},                   // output dimensions
            {1, 1},                                                          // kernels
            {1, 1},                                                          // strides
            {0, 0, 0, 0},                                                    // padding
            ExecutionMode::CUBOID_8x16,                                      // execution mode
            ActivationFunction::NONE,                                        // activation
            0.0F,                                                            // act_sparsity
            0.0F,                                                            // weight_sparsity
            {swz_def, swz_def},                                              // input_swizzling
            {swz_def},                                                       // output_swizzling
            1,                                                               // output_write_tiles
            {0, 0, 0, 0},                                                    // offsets
            ISIStrategy::CLUSTERING,                                         // isi_strategy
            false,                                                           // weight_sparsity_enabled
            {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects
            {false, {0, 0, 0, 0}, {0, 0, 0, 0}, false},                      // SEP
            DataType::UINT8,                                                 // input1 data type
            "",                                                              // layer_info
            false,                                                           ///< operation does not have weights
            false                                                            // in_place_output_memory{};
    };
    {  //
        auto wl{wl_ref2};
        dut.check_data_consistency(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;

        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }
    {  //
        auto wl{std::move(wl_ref2)};
        dut.check_data_consistency(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;

        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }
    const DPUWorkload wl_ref2_large{
            device_req,
            Operation::ELTWISE,
            {VPUTensor(65, 64, 64, 1, DataType::UINT8)},                     // input dimensions
            {VPUTensor(65, 64, 64, 1, DataType::FLOAT32)},                   // output dimensions
            {1, 1},                                                          // kernels
            {1, 1},                                                          // strides
            {0, 0, 0, 0},                                                    // padding
            ExecutionMode::CUBOID_8x16,                                      // execution mode
            ActivationFunction::NONE,                                        // activation
            0.0F,                                                            // act_sparsity
            0.0F,                                                            // weight_sparsity
            {swz_def, swz_def},                                              // input_swizzling
            {swz_def},                                                       // output_swizzling
            1,                                                               // output_write_tiles
            {0, 0, 0, 0},                                                    // offsets
            ISIStrategy::CLUSTERING,                                         // isi_strategy
            false,                                                           // weight_sparsity_enabled
            {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects
            {false, {0, 0, 0, 0}, {0, 0, 0, 0}, false},                      // SEP
            DataType::UINT8,                                                 // input1 data type
            "",                                                              // layer_info
            false,                                                           ///< operation does not have weights
            false                                                            // in_place_output_memory{};
    };
    {  //
        auto wl{wl_ref2_large};
        dut.check_data_consistency(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;

        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::ERROR_INPUT_TOO_BIG))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }
    {  //
        auto wl{std::move(wl_ref2_large)};
        dut.check_data_consistency(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;

        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::ERROR_INPUT_TOO_BIG))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }
}

TEST_F(DPU_WorkloadValidatorTestNPU5x, AutoPadding_support) {
    DPU_OperationSanitizer dut;
    VPUNN::SanityReport sane;

    DPUWorkload wl_ref{
            VPUDevice::NPU_5_0,
            Operation::ELTWISE,
            {VPUTensor(28, 9, 16, 1, DataType::FLOAT16)},  // input dimensions
            {VPUTensor(28, 9, 1, 1, DataType::UINT8)},     // output dimensions
            {1, 1},                                        // kernels
            {1, 1},                                        // strides
            {0, 0, 0, 0},                                  // padding
            ExecutionMode::CUBOID_8x16,                    // execution mode
            ActivationFunction::NONE,                      // activation
            0.0F,                                          // act_sparsity
            0.0F,                                          // weight_sparsity
            {swz_def, swz_def},                            // input_swizzling
            {swz_def},                                     // output_swizzling
            1,                                             // output_write_tiles
            {0, 0, 0, 0},                                  // offsets
            ISIStrategy::CLUSTERING,                       // isi_strategy
            false,                                         // weight_sparsity_enabled
    };
    wl_ref.superdense_memory = true;

    DPUWorkload wl_ref2{
            VPUDevice::NPU_5_0,
            Operation::AVEPOOL,
            {VPUTensor(32, 80, 16, 1, DataType::UINT8, Layout::ZXY)},  // input dimensions
            {VPUTensor(32, 80, 2, 1, DataType::UINT8, Layout::XYZ)},   // output dimensions
            {1, 1},                                                    // kernels
            {1, 1},                                                    // strides
            {0, 0, 0, 0},                                              // padding
            ExecutionMode::CUBOID_16x16,                               // execution mode
            ActivationFunction::NONE,                                  // activation
            0.0F,                                                      // act_sparsity
            0.0F,                                                      // weight_sparsity
            {swz_def, swz_def},                                        // input_swizzling
            {swz_def},                                                 // output_swizzling
            1,                                                         // output_write_tiles
            {0, 0, 0, 0},                                              // offsets
            ISIStrategy::CLUSTERING,                                   // isi_strategy
            false,                                                     // weight_sparsity_enabled
    };

    DPUWorkload wl_ref3{
            VPUDevice::NPU_5_0,
            Operation::ELTWISE,
            {VPUTensor(224, 43, 16, 1, DataType::UINT8, Layout::ZXY)},   // input dimensions
            {VPUTensor(224, 43, 1, 1, DataType::FLOAT16, Layout::ZXY)},  // output dimensions
            {1, 1},                                                      // kernels
            {1, 1},                                                      // strides
            {0, 0, 0, 0},                                                // padding
            ExecutionMode::CUBOID_8x16,                                  // execution mode
            ActivationFunction::NONE,                                    // activation
            0.0F,                                                        // act_sparsity
            0.0F,                                                        // weight_sparsity
            {swz_def, swz_def},                                          // input_swizzling
            {swz_def},                                                   // output_swizzling
            1,                                                           // output_write_tiles
            {0, 0, 0, 0},                                                // offsets
            ISIStrategy::CLUSTERING,                                     // isi_strategy
            false,                                                       // weight_sparsity_enabled
    };
    wl_ref3.superdense_memory = true;

    DPUWorkload wl_output_autopad{wl_ref};
    wl_output_autopad.output_autopad = true;

    dut.check_and_sanitize(wl_output_autopad, sane);
    EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
            << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
            << wl_output_autopad;

    DPUWorkload wl_input_autopad{wl_ref};
    wl_input_autopad.input_autopad = true;
    wl_input_autopad.inputs[0] = VPUTensor(
            {wl_ref.inputs[0].width(), wl_ref.inputs[0].height(), 5, wl_ref.inputs[0].batches()}, wl_ref.inputs[0]);
    wl_input_autopad.outputs[0] =
            VPUTensor({wl_ref.outputs[0].width(), wl_ref.outputs[0].height(), 16, wl_ref.outputs[0].batches()},
                      wl_ref.outputs[0]);

    dut.check_and_sanitize(wl_input_autopad, sane);
    EXPECT_EQ(sane.value(), V(VPUNN::Cycles::ERROR_INVALID_INPUT_CONFIGURATION))
            << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
            << wl_input_autopad;

    DPUWorkload wl2_output_autopad{std::move(wl_ref2)};
    wl2_output_autopad.output_autopad = true;

    dut.check_and_sanitize(wl2_output_autopad, sane);
    EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
            << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
            << wl2_output_autopad;

    DPUWorkload wl3_output_autopad{std::move(wl_ref3)};
    wl3_output_autopad.output_autopad = true;

    dut.check_and_sanitize(wl3_output_autopad, sane);
    EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
            << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
            << wl3_output_autopad;
}

}  // namespace VPUNN_unit_tests