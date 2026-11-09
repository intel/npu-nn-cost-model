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
#include "logger/logger.h"

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {
using namespace VPUNN;

class DPU_WorkloadValidatorTestNPU6x : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }
    const VPUDevice device_req{VPUDevice::NPU_6_0};

private:
    // max memeory: 2 097 152 (2MB)
};

TEST_F(DPU_WorkloadValidatorTestNPU6x, Memory_Output_32Bits) {
    DPU_OperationSanitizer dut;

    SanityReport sane;

    const DPUWorkload wl_ref{
            // memory : 2 119 296
            device_req,
            Operation::CONVOLUTION,
            {VPUTensor(64, 64, 128, 1, DataType::UINT8)},                    // input dimensions
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
            // memory: 2 072 384
            device_req,
            Operation::CONVOLUTION,
            {VPUTensor(64, 64, 128, 1, DataType::UINT8)},                    // input dimensions
            {VPUTensor(21, 21, 512 + 16, 1, DataType::FLOAT32)},             // output dimensions
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
            // memory 2088960
            device_req,
            Operation::ELTWISE,
            {VPUTensor(64, 68, 80, 1, DataType::UINT8)},                     // input dimensions
            {VPUTensor(64, 68, 80, 1, DataType::FLOAT32)},                   // output dimensions
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
            // memoery : 2119680
            device_req,
            Operation::ELTWISE,
            {VPUTensor(64, 69, 80, 1, DataType::UINT8)},                     // input dimensions
            {VPUTensor(64, 69, 80, 1, DataType::FLOAT32)},                   // output dimensions
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

TEST_F(DPU_WorkloadValidatorTestNPU6x, AutoPadding_support) {
    DPU_OperationSanitizer dut;
    VPUNN::SanityReport sane;

    DPUWorkload wl_ref{
            device_req,
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
            device_req,
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
            device_req,
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

/// @brief NPU6 ODU autopad allows any output channel value (including unaligned > 16)
TEST_F(DPU_WorkloadValidatorTestNPU6x, AutoPadding_unaligned_output_channels_gt16) {
    DPU_OperationSanitizer dut;
    VPUNN::SanityReport sane;

    // ResNet-50 inspired: AVEPOOL with unaligned output channels > 16 and autopad ON
    {
        DPUWorkload wl{
                device_req,
                Operation::AVEPOOL,
                {VPUTensor(1, 1, 128, 1, DataType::FLOAT16, Layout::ZXY)},  // input dimensions (IC=128 valid)
                {VPUTensor(1, 1, 100, 1, DataType::FLOAT16, Layout::ZXY)},  // output dimensions (OC=100 unaligned)
                {1, 1},                                                     // kernels
                {1, 1},                                                     // strides
                {0, 0, 0, 0},                                               // padding
                ExecutionMode::CUBOID_16x16,                                // execution mode
                ActivationFunction::NONE,                                   // activation
                0.0F,                                                       // act_sparsity
                0.0F,                                                       // weight_sparsity
                {swz_def, swz_def},                                         // input_swizzling
                {swz_def},                                                  // output_swizzling
                1,                                                          // output_write_tiles
                {0, 0, 0, 0},                                               // offsets
                ISIStrategy::CLUSTERING,                                    // isi_strategy
                false,                                                      // weight_sparsity_enabled
        };
        wl.output_autopad = true;

        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }

    // Convolution with unaligned output channels > 16 and autopad ON
    {
        DPUWorkload wl{
                device_req,
                Operation::CONVOLUTION,
                {VPUTensor(8, 8, 32, 1, DataType::UINT8)},  // input dimensions
                {VPUTensor(8, 8, 33, 1, DataType::UINT8)},  // output dimensions
                {1, 1},                                     // kernels
                {1, 1},                                     // strides
                {0, 0, 0, 0},                               // padding
                ExecutionMode::CUBOID_16x16,                // execution mode
                ActivationFunction::NONE,                   // activation
                0.0F,                                       // act_sparsity
                0.0F,                                       // weight_sparsity
                {swz_def, swz_def},                         // input_swizzling
                {swz_def},                                  // output_swizzling
                1,                                          // output_write_tiles
                {0, 0, 0, 0},                               // offsets
                ISIStrategy::CLUSTERING,                    // isi_strategy
                false,                                      // weight_sparsity_enabled
        };
        wl.output_autopad = true;

        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }

    // MAXPOOL with unaligned output channels > 16 and autopad ON
    {
        DPUWorkload wl{
                device_req,
                Operation::MAXPOOL,
                {VPUTensor(8, 8, 32, 1, DataType::UINT8)},  // input dimensions (IC=32 valid)
                {VPUTensor(8, 8, 25, 1, DataType::UINT8)},  // output dimensions (OC=25 unaligned)
                {1, 1},                                     // kernels
                {1, 1},                                     // strides
                {0, 0, 0, 0},                               // padding
                ExecutionMode::CUBOID_16x16,                // execution mode
                ActivationFunction::NONE,                   // activation
                0.0F,                                       // act_sparsity
                0.0F,                                       // weight_sparsity
                {swz_def, swz_def},                         // input_swizzling
                {swz_def},                                  // output_swizzling
                1,                                          // output_write_tiles
                {0, 0, 0, 0},                               // offsets
                ISIStrategy::CLUSTERING,                    // isi_strategy
                false,                                      // weight_sparsity_enabled
        };
        wl.output_autopad = true;

        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }

    // Without autopad, unaligned output channels > 16 should still fail
    {
        DPUWorkload wl{
                device_req,
                Operation::CONVOLUTION,
                {VPUTensor(8, 8, 32, 1, DataType::UINT8)},  // input dimensions
                {VPUTensor(8, 8, 33, 1, DataType::UINT8)},  // output dimensions
                {1, 1},                                     // kernels
                {1, 1},                                     // strides
                {0, 0, 0, 0},                               // padding
                ExecutionMode::CUBOID_16x16,                // execution mode
                ActivationFunction::NONE,                   // activation
                0.0F,                                       // act_sparsity
                0.0F,                                       // weight_sparsity
                {swz_def, swz_def},                         // input_swizzling
                {swz_def},                                  // output_swizzling
                1,                                          // output_write_tiles
                {0, 0, 0, 0},                               // offsets
                ISIStrategy::CLUSTERING,                    // isi_strategy
                false,                                      // weight_sparsity_enabled
        };
        wl.output_autopad = false;

        dut.check_and_sanitize(wl, sane);
        EXPECT_NE(sane.value(), V(VPUNN::Cycles::NO_ERROR)) << "Expected error for unaligned OC > 16 without autopad\n"
                                                            << wl;
    }
}

TEST_F(DPU_WorkloadValidatorTestNPU6x, Reduce_operations) {
    DPU_OperationSanitizer dut;
    VPUNN::SanityReport sane;

    const DPUWorkload wl_ref{
            device_req,
            Operation::REDUCE_MS,
            {VPUTensor(180, 172, 64, 1, DataType::UINT8)},                   // input dimensions
            {VPUTensor(180, 172, 1, 1, DataType::FLOAT32)},                  // output dimensions
            {1, 1},                                                          // kernels
            {1, 1},                                                          // strides
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
        auto wl{wl_ref};
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

TEST_F(DPU_WorkloadValidatorTestNPU6x, CMX_Capacity_Exceeded_InfoMessage) {
    // Verify that the sanitizer reports CMX overflow in a human-readable way via SanityReport::info.
    // We specifically check that:
    //   1) the workload is considered data-consistent (so the only failure is memory/CMX related), and
    //   2) check_and_sanitize() returns ERROR_INPUT_TOO_BIG, and
    //   3) the info string contains the expected message fragment ("CMX capacity exceeded").
    //
    // Why the loop:
    // CMX size / memory accounting can change across devices, configs, or validator updates. To keep the test
    // stable, we start from a known workload (that should already be too big) and grow the output tensor channels until
    // the computed CMX footprint exceeds the available CMX.

    DPUWorkload wl_too_big{
            VPUDevice::NPU_6_0,
            Operation::CONVOLUTION,
            {VPUTensor(130, 86, 256, 1, DataType::UINT8, Layout::ZXY, false)},  // input dimensions
            {VPUTensor(129, 85, 64, 1, DataType::UINT8, Layout::ZXY, false)},   // output dimensions
            {2, 2},                                                             // kernels
            {1, 1},                                                             // strides
            {0, 0, 0, 0},                                                       // padding
            ExecutionMode::CUBOID_8x16,                                         // execution mode
            ActivationFunction::NONE,                                           // activation
            0.0F,                                                               // act_sparsity
            0.0F,                                                               // weight_sparsity
            {Swizzling::KEY_0, Swizzling::KEY_0},                               // input_swizzling
            {Swizzling::KEY_0},                                                 // output_swizzling
            1,                                                                  // output_write_tiles
            {0, 0, 0, 0},                                                       // offsets
            ISIStrategy::CLUSTERING,                                            // isi_strategy
            false,                                                              // weight_sparsity_enabled
            HaloWorkload{{0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},
            SEPModeInfo{true,             // sep_enabled
                        {130, 86, 1, 1},  // sep_w/h/c/b
                        {130, 86, 256, 1},
                        false},

            // weights dtype (CSV: input_1_datatype)
            DataType::BFLOAT16,
            "",     // layer_info
            false,  // weightless_operation / in_place_input1
            false,  // in_place_output_memory
            false,  // superdense_memory
            false,  // input_autopad
            false,  // output_autopad
            CostSourceHint::AUTO,
            ProfilingServiceBackend::__size,
            MPEEngine::SCL,
            false  // reduce_minmax_op
    };

    DPU_OperationSanitizer dut;
    VPUNN::SanityReport sane;

    // This ensures we are testing the CMX overflow/reporting path, not an earlier validation failure
    ASSERT_NO_THROW(dut.check_data_consistency(wl_too_big, sane));
    ASSERT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR)) << sane.info;

    VPUNN::DPU_OperationValidator validator;
    const auto& config = validator.get_config(wl_too_big.device);
    const int available_cmx = config.get_cmx_size(wl_too_big.device);

    // Make the wl exceed CMX by increasing output channels in bounded steps.
    // Bound the loop to avoid infinite growth if memory accounting changes unexpectedly
    constexpr unsigned int max_steps = 32;
    for (unsigned int step = 0; step < max_steps; ++step) {
        const auto mem = validator.compute_wl_memory(wl_too_big);
        if (mem.cmx > available_cmx) {
            break;
        }

        const auto out = wl_too_big.outputs[0];
        wl_too_big.outputs[0] = VPUTensor(out.width(), out.height(), out.channels() + 64U, out.batches(),
                                          out.get_dtype(), out.get_layout(), out.get_sparsity());
    }

    {
        const auto mem = validator.compute_wl_memory(wl_too_big);
        ASSERT_GT(mem.cmx, available_cmx) << "Could not construct an oversized workload within the step limit.\n"
                                          << mem << "Available CMX: " << available_cmx << " B\n";
    }

    // Sanitize should detect CMX overflow and report it as INPUT_TOO_BIG
    dut.check_and_sanitize(wl_too_big, sane);
    EXPECT_EQ(sane.value(), V(VPUNN::Cycles::ERROR_INPUT_TOO_BIG))
            << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
            << wl_too_big;

    EXPECT_NE(sane.info.find("CMX capacity exceeded"), std::string::npos) << "sane.info: " << sane.info;
}

/// @brief Tests for NPU6+ output_autopad channel coupling constraints on DW-style operations.
///
/// Rules (when output_autopad is active on NPU6+):
///   Rule 1: OC must not exceed IC
///   Rule 2: IC must equal dw_aligned(OC), where:
///             OC < 16  → dw_aligned = 16
///             OC >= 16 → dw_aligned = round_up(OC, 32)
/// Applies to: DW_CONVOLUTION, MAXPOOL, AVEPOOL.
TEST_F(DPU_WorkloadValidatorTestNPU6x, DW_AutoPad_OutputChannelCouplingConstraints) {
    DPU_OperationSanitizer dut;
    VPUNN::SanityReport sane;

    // Helper lambda to build a minimal valid DW-style workload, overriding IC and OC via output_autopad.
    // Spatial dims are kept tiny (1x1) so CMX is never an issue.
    auto make_dw_wl = [&](Operation op, unsigned int ic, unsigned int oc, bool autopad) {
        DPUWorkload wl{
                device_req,
                op,
                {VPUTensor(1, 1, ic, 1, DataType::UINT8, Layout::ZXY)},  // input
                {VPUTensor(1, 1, oc, 1, DataType::UINT8, Layout::ZXY)},  // output
                {1, 1},                                                  // kernels
                {1, 1},                                                  // strides
                {0, 0, 0, 0},                                            // padding
                ExecutionMode::CUBOID_16x16,                             // execution mode (only valid for DW)
                ActivationFunction::NONE,
                0.0F,  // act_sparsity
                0.0F,  // weight_sparsity
                {swz_def, swz_def},
                {swz_def},
                1,             // output_write_tiles
                {0, 0, 0, 0},  // offsets
                ISIStrategy::CLUSTERING,
                false,  // weight_sparsity_enabled
        };
        wl.output_autopad = autopad;
        return wl;
    };

    // --- DW_CONVOLUTION valid cases ---

    // OC=25, IC=32: round_up(25,32)=32 == IC  AND  OC<=IC  →  valid
    {
        auto wl = make_dw_wl(Operation::DW_CONVOLUTION, 32, 25, true);
        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR)) << "DW_CONV IC=32 OC=25 autopad: expected valid\n"
                                                            << sane.info;
    }

    // OC=10 (<16), IC=16: dw_aligned=16 == IC  AND  OC<=IC  →  valid
    {
        auto wl = make_dw_wl(Operation::DW_CONVOLUTION, 16, 10, true);
        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR)) << "DW_CONV IC=16 OC=10 autopad: expected valid\n"
                                                            << sane.info;
    }

    // OC=96, IC=96: round_up(96,32)=96 == IC  AND  OC<=IC  →  valid (aligned OC, autopad incidentally on)
    {
        auto wl = make_dw_wl(Operation::DW_CONVOLUTION, 96, 96, true);
        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR)) << "DW_CONV IC=96 OC=96 autopad: expected valid\n"
                                                            << sane.info;
    }

    // --- DW_CONVOLUTION invalid cases ---

    // Rule 1 violation: OC=64 > IC=32
    {
        auto wl = make_dw_wl(Operation::DW_CONVOLUTION, 32, 64, true);
        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::ERROR_INVALID_INPUT_CONFIGURATION))
                << "DW_CONV IC=32 OC=64 autopad: expected error (OC > IC)\n"
                << sane.info;
    }

    // Rule 2 violation: OC=33, IC=128, but dw_aligned(33)=64 ≠ 128
    {
        auto wl = make_dw_wl(Operation::DW_CONVOLUTION, 128, 33, true);
        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::ERROR_INVALID_INPUT_CONFIGURATION))
                << "DW_CONV IC=128 OC=33 autopad: expected error (dw_aligned(33)=64 != IC=128)\n"
                << sane.info;
    }

    // --- MAXPOOL valid case ---

    // OC=25, IC=32: round_up(25,32)=32 == IC  →  valid
    {
        auto wl = make_dw_wl(Operation::MAXPOOL, 32, 25, true);
        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR)) << "MAXPOOL IC=32 OC=25 autopad: expected valid\n"
                                                            << sane.info;
    }

    // --- MAXPOOL invalid case ---

    // Rule 2 violation: OC=33, IC=128, dw_aligned(33)=64 ≠ 128
    {
        auto wl = make_dw_wl(Operation::MAXPOOL, 128, 33, true);
        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::ERROR_INVALID_INPUT_CONFIGURATION))
                << "MAXPOOL IC=128 OC=33 autopad: expected error (dw_aligned(33)=64 != IC=128)\n"
                << sane.info;
    }

    // --- AVEPOOL valid case ---

    // OC=100, IC=128: round_up(100,32)=128 == IC  →  valid
    {
        auto wl = make_dw_wl(Operation::AVEPOOL, 128, 100, true);
        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR)) << "AVEPOOL IC=128 OC=100 autopad: expected valid\n"
                                                            << sane.info;
    }

    // --- AVEPOOL invalid case ---

    // Rule 2 violation: OC=33, IC=128, dw_aligned(33)=64 ≠ 128
    {
        auto wl = make_dw_wl(Operation::AVEPOOL, 128, 33, true);
        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::ERROR_INVALID_INPUT_CONFIGURATION))
                << "AVEPOOL IC=128 OC=33 autopad: expected error (dw_aligned(33)=64 != IC=128)\n"
                << sane.info;
    }

    // --- Without output_autopad: old equality rule still applies (OC must equal IC) ---

    // DW_CONV, no autopad, IC==OC → valid
    {
        auto wl = make_dw_wl(Operation::DW_CONVOLUTION, 32, 32, false);
        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR)) << "DW_CONV IC=32 OC=32 no-autopad: expected valid\n"
                                                            << sane.info;
    }

    // DW_CONV, no autopad, IC≠OC → invalid (existing rule)
    {
        auto wl = make_dw_wl(Operation::DW_CONVOLUTION, 32, 25, false);
        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::ERROR_INVALID_INPUT_CONFIGURATION))
                << "DW_CONV IC=32 OC=25 no-autopad: expected error (OC != IC)\n"
                << sane.info;
    }
}

/// @brief Tests for NPU6+ output_autopad channel coupling constraints on ELTWISE.
///
/// Rules (when output_autopad is active on NPU6+):
///   Rule 1: OC must not exceed IC
///   Rule 2: IC must equal round_up(OC, 16)
TEST_F(DPU_WorkloadValidatorTestNPU6x, Eltwise_AutoPad_OutputChannelCouplingConstraints) {
    DPU_OperationSanitizer dut;
    VPUNN::SanityReport sane;

    auto make_eltwise_wl = [&](unsigned int ic, unsigned int oc, bool autopad) {
        DPUWorkload wl{
                device_req,
                Operation::ELTWISE,
                {VPUTensor(1, 1, ic, 1, DataType::UINT8, Layout::ZXY)},  // input
                {VPUTensor(1, 1, oc, 1, DataType::UINT8, Layout::ZXY)},  // output
                {1, 1},                                                  // kernels
                {1, 1},                                                  // strides
                {0, 0, 0, 0},                                            // padding
                ExecutionMode::CUBOID_8x16,                              // ELTWISE only supports 8x16
                ActivationFunction::NONE,
                0.0F,
                0.0F,
                {swz_def, swz_def},
                {swz_def},
                1,
                {0, 0, 0, 0},
                ISIStrategy::CLUSTERING,
                false,
        };
        wl.output_autopad = autopad;
        return wl;
    };

    // Valid: OC=16, IC=16 → round_up(16,16)=16 == IC, OC<=IC
    {
        auto wl = make_eltwise_wl(16, 16, true);
        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR)) << "ELTWISE IC=16 OC=16 autopad: expected valid\n"
                                                            << sane.info;
    }

    // Valid: OC=13, IC=16 → round_up(13,16)=16 == IC, OC<=IC
    {
        auto wl = make_eltwise_wl(16, 13, true);
        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR)) << "ELTWISE IC=16 OC=13 autopad: expected valid\n"
                                                            << sane.info;
    }

    // Valid: OC=48, IC=48 → round_up(48,16)=48 == IC, OC<=IC
    {
        auto wl = make_eltwise_wl(48, 48, true);
        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR)) << "ELTWISE IC=48 OC=48 autopad: expected valid\n"
                                                            << sane.info;
    }

    // Valid: OC=33, IC=48 → round_up(33,16)=48 == IC, OC<=IC
    {
        auto wl = make_eltwise_wl(48, 33, true);
        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR)) << "ELTWISE IC=48 OC=33 autopad: expected valid\n"
                                                            << sane.info;
    }

    // Rule 1 violation: OC=64 > IC=48
    {
        auto wl = make_eltwise_wl(48, 64, true);
        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::ERROR_INVALID_INPUT_CONFIGURATION))
                << "ELTWISE IC=48 OC=64 autopad: expected error (OC > IC)\n"
                << sane.info;
    }

    // Rule 2 violation: OC=33 → round_up=48, but IC=32 ≠ 48
    {
        auto wl = make_eltwise_wl(32, 33, true);
        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::ERROR_INVALID_INPUT_CONFIGURATION))
                << "ELTWISE IC=32 OC=33 autopad: expected error (round_up(33,16)=48 != IC=32)\n"
                << sane.info;
    }

    // Rule 2 violation: OC=13 → round_up=16, but IC=32 ≠ 16 (and OC<=IC so rule 1 passes, rule 2 fails)
    {
        auto wl = make_eltwise_wl(32, 13, true);
        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::ERROR_INVALID_INPUT_CONFIGURATION))
                << "ELTWISE IC=32 OC=13 autopad: expected error (round_up(13,16)=16 != IC=32)\n"
                << sane.info;
    }

    // No autopad: OC==IC → valid (existing rule unchanged)
    {
        auto wl = make_eltwise_wl(32, 32, false);
        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR)) << "ELTWISE IC=32 OC=32 no-autopad: expected valid\n"
                                                            << sane.info;
    }

    // No autopad: OC≠IC → invalid (existing rule unchanged)
    {
        auto wl = make_eltwise_wl(32, 13, false);
        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::ERROR_INVALID_INPUT_CONFIGURATION))
                << "ELTWISE IC=32 OC=13 no-autopad: expected error (OC != IC)\n"
                << sane.info;
    }
}

/// @brief Tests that the CMX memory footprint for a workload with unaligned output channels and
/// output_autopad ON is strictly smaller than the aligned (round-up-to-16) equivalent.
///
/// This validates that the memory calculator correctly accounts for the actual (unaligned) output
/// tensor size when output_autopad is active, rather than silently inflating it to the next
/// aligned count.  The three operations covered have different weight/input structures:
///   - CONVOLUTION: has weights (input_1), output is the dominant saving
///   - DW_CONVOLUTION: weights are kernel-sized masks, IC==OC coupling means input also shrinks
///   - ELTWISE: input_1 is the second activation tensor (same size as input_0), output saving
TEST_F(DPU_WorkloadValidatorTestNPU6x, AutoPad_UnalignedOC_MemoryIsSmallerThanAligned) {
    VPUNN::DPU_OperationValidator validator;

    // Helper: build a workload with the given op/IC/OC/autopad flag and return its CMX memory.
    // Spatial dims are kept small and identical for all variants so that only the channel count
    // drives the difference in memory.
    auto mem_for = [&](Operation op, unsigned int ic, unsigned int oc, bool autopad,
                       ExecutionMode exec = ExecutionMode::CUBOID_16x16) -> MemorySize {
        DPUWorkload wl{
                device_req,
                op,
                {VPUTensor(8, 8, ic, 1, DataType::FLOAT16, Layout::ZXY)},
                {VPUTensor(8, 8, oc, 1, DataType::FLOAT16, Layout::ZXY)},
                {1, 1},  // kernels (1x1 keeps things simple)
                {1, 1},  // strides
                {0, 0, 0, 0},
                exec,
        };
        wl.output_autopad = autopad;
        return validator.compute_wl_memory(wl);
    };

    // --- CONVOLUTION ---
    // OC=33 with autopad must cost less than OC=48 (next multiple of 16) without autopad.
    // The saving comes entirely from the smaller output tensor: 33 channels * W * H * dtype < 48 channels * ...
    // Input and weights are the same because IC does not change (CONV has independent IC/OC).
    {
        const unsigned int oc_unaligned = 33;
        const unsigned int oc_aligned = 48;  // round_up(33, 16) = 48
        const unsigned int ic = 32;

        const auto mem_unaligned = mem_for(Operation::CONVOLUTION, ic, oc_unaligned, /*autopad=*/true);
        const auto mem_aligned = mem_for(Operation::CONVOLUTION, ic, oc_aligned, /*autopad=*/false);

        EXPECT_LT(mem_unaligned.output_0, mem_aligned.output_0)
                << "CONV output_0 memory: unaligned OC=" << oc_unaligned << " should be < aligned OC=" << oc_aligned
                << "\n"
                << "unaligned: " << mem_unaligned << "\naligned:   " << mem_aligned;

        EXPECT_LT(mem_unaligned.cmx, mem_aligned.cmx)
                << "CONV total CMX: unaligned OC=" << oc_unaligned << " should be < aligned OC=" << oc_aligned << "\n"
                << "unaligned: " << mem_unaligned << "\naligned:   " << mem_aligned;

        // Input and weights are independent of OC for CONV — they must be equal.
        EXPECT_EQ(mem_unaligned.input_0, mem_aligned.input_0)
                << "CONV input_0 must be equal for same IC\n"
                << "unaligned: " << mem_unaligned << "\naligned:   " << mem_aligned;
    }

    // --- DW_CONVOLUTION ---
    // For DW-style ops IC and OC are coupled by the operation constraints. In this test we validate
    // an unaligned output channel count of OC=33 with output_autopad ON, which requires IC=64 for
    // the input/output channel relationship used by the validator. The aligned baseline is OC=IC=64.
    // The unaligned/autopad case should use less output memory, and therefore less total CMX, than
    // the fully aligned baseline.
    {
        const unsigned int oc_unaligned = 33;
        const unsigned int ic_unaligned = 64;  // Coupled IC required for the DW OC=33 autopad scenario
        const unsigned int oc_aligned = 64;    // Aligned DW baseline used for comparison (OC=IC=64)

        const auto mem_unaligned = mem_for(Operation::DW_CONVOLUTION, ic_unaligned, oc_unaligned, /*autopad=*/true);
        const auto mem_aligned = mem_for(Operation::DW_CONVOLUTION, oc_aligned, oc_aligned, /*autopad=*/false);

        EXPECT_LT(mem_unaligned.output_0, mem_aligned.output_0)
                << "DW output_0 memory: unaligned OC=" << oc_unaligned << " should be < aligned OC=" << oc_aligned
                << "\n"
                << "unaligned: " << mem_unaligned << "\naligned:   " << mem_aligned;

        EXPECT_LT(mem_unaligned.cmx, mem_aligned.cmx)
                << "DW total CMX: unaligned (OC=" << oc_unaligned << " IC=" << ic_unaligned
                << ") should be < aligned (OC=IC=" << oc_aligned << ")\n"
                << "unaligned: " << mem_unaligned << "\naligned:   " << mem_aligned;
    }

    // --- ELTWISE ---
    // ELTWISE has a second input activation (input_1), not weights.
    // The saving is from the smaller output tensor when autopad is active.
    // With autopad: OC=13, IC=round_up(13,16)=16.  Aligned baseline: OC=IC=16.
    {
        const unsigned int oc_unaligned = 7;
        const unsigned int ic_unaligned = 16;  // round_up(13, 16) = 16 (required ELTWISE coupling IC)
        const unsigned int oc_aligned = 16;

        const auto mem_unaligned =
                mem_for(Operation::ELTWISE, ic_unaligned, oc_unaligned, /*autopad=*/true, ExecutionMode::CUBOID_8x16);
        const auto mem_aligned =
                mem_for(Operation::ELTWISE, oc_aligned, oc_aligned, /*autopad=*/false, ExecutionMode::CUBOID_8x16);

        EXPECT_LT(mem_unaligned.output_0, mem_aligned.output_0)
                << "ELTWISE output_0 memory: unaligned OC=" << oc_unaligned << " should be < aligned OC=" << oc_aligned
                << "\n"
                << "unaligned: " << mem_unaligned << "\naligned:   " << mem_aligned;

        // ELTWISE input_1 is the second activation (same size as input_0), not weights
        EXPECT_EQ(mem_unaligned.input_1, mem_unaligned.input_0)
                << "ELTWISE input_1 should equal input_0 (second activation)\n"
                << "unaligned: " << mem_unaligned;
        EXPECT_EQ(mem_aligned.input_1, mem_aligned.input_0)
                << "ELTWISE input_1 should equal input_0 (second activation)\n"
                << "aligned:   " << mem_aligned;

        // Output is inplace for ELTWISE (shares memory with input), so total CMX stays the same.
        // The saving is visible only in output_0 (which is not added to total).
        EXPECT_TRUE(mem_unaligned.inplace_output) << "ELTWISE output should be inplace\n"
                                                  << "unaligned: " << mem_unaligned;
    }
}

/// NPU6 only supports dCIM_32x128 (no dCIM_64x128). check_and_sanitize no longer forces/normalizes
/// execution_order; an unsupported dCIM_64x128 selection must be left untouched and rejected by the
/// Execution_Order consistency check further down (device supports DCIM, but not this specific mode).
TEST_F(DPU_WorkloadValidatorTestNPU6x, CheckAndSanitize_RejectsUnsupportedDcim64x128) {
    DPUWorkload wl{
            device_req,
            Operation::CONVOLUTION,
            {VPUTensor(32, 32, 16, 1, DataType::UINT8)},  // input dimensions
            {VPUTensor(32, 32, 16, 1, DataType::UINT8)},  // output dimensions
            {1, 1},                                       // kernels
            {1, 1},                                       // strides
            {0, 0, 0, 0},                                 // padding
            ExecutionMode::dCIM_64x128,                   // execution mode (not valid for NPU6)
            ActivationFunction::NONE,                     // activation
            0.0F,                                         // act_sparsity
            0.0F,                                         // weight_sparsity
            {swz_def, swz_def},                           // input_swizzling
            {swz_def},                                    // output_swizzling
            1,                                            // output_write_tiles
            {0, 0, 0, 0},                                 // offsets
            ISIStrategy::CLUSTERING,                      // isi_strategy
            false,                                        // weight_sparsity_enabled
    };
    wl.mpe_engine = MPEEngine::DCIM;

    DPU_OperationSanitizer sanitizer;
    SanityReport report;
    sanitizer.check_and_sanitize(wl, report);

    EXPECT_EQ(wl.execution_order, ExecutionMode::dCIM_64x128)
            << "execution_order must be left untouched; there is no more forced normalization";
    EXPECT_EQ(report.value(), V(Cycles::ERROR_INVALID_INPUT_CONFIGURATION))
            << report.info << "\ndCIM_64x128 is not a valid execution mode for NPU6, so it must be rejected";
}

/// sanitize_preconditions no longer performs any DCIM-related defaulting/normalization; it must leave
/// execution_order exactly as the caller set it, regardless of whether that mode is valid for this device.
TEST_F(DPU_WorkloadValidatorTestNPU6x, SanitizePreconditions_DoesNotTouchExecutionOrder) {
    DPUWorkload wl{
            device_req,
            Operation::ELTWISE,
            {VPUTensor(28, 9, 16, 1, DataType::FLOAT16)},  // input dimensions
            {VPUTensor(28, 9, 16, 1, DataType::FLOAT16)},  // output dimensions
            {1, 1},                                        // kernels
            {1, 1},                                        // strides
            {0, 0, 0, 0},                                  // padding
            ExecutionMode::CUBOID_16x16,                   // execution mode (overwritten below)
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
    // DPULayer(const DPUWorkload&) overwrites execution_order via the device default, so the
    // desired mode/engine must be set after construction (see NPU_RESERVED_1 sanitizer test for details).
    DPULayer layer(wl);
    layer.mpe_engine = MPEEngine::DCIM;
    layer.execution_order = ExecutionMode::dCIM_64x128;

    LayersValidation dut;
    dut.sanitize_preconditions(layer);

    EXPECT_EQ(layer.execution_order, ExecutionMode::dCIM_64x128)
            << "sanitize_preconditions must not touch execution_order anymore";
}

/// NPU6 supports DCIM at the device level, but dcim_valid_datatypes_map/dcim_valid_execution_order_map are
/// keyed only by Operation::CONVOLUTION, so MAXPOOL+DCIM still trips a std::out_of_range .at() lookup deep
/// inside check_workload_consistency. Verifies that exception is caught and reported, never left uncaught.
TEST_F(DPU_WorkloadValidatorTestNPU6x, CheckAndSanitize_DcimWithNonConvOperation_MAXPOOL) {
    DPUWorkload wl{
            device_req,
            Operation::MAXPOOL,
            {VPUTensor(32, 32, 16, 1, DataType::UINT8)},  // input dimensions
            {VPUTensor(32, 32, 16, 1, DataType::UINT8)},  // output dimensions
            {1, 1},                                       // kernels
            {1, 1},                                       // strides
            {0, 0, 0, 0},                                 // padding
            ExecutionMode::dCIM_32x128,                   // execution mode (valid for NPU6, but for CONV only)
            ActivationFunction::NONE,                     // activation
            0.0F,                                         // act_sparsity
            0.0F,                                         // weight_sparsity
            {swz_def, swz_def},                           // input_swizzling
            {swz_def},                                    // output_swizzling
            1,                                            // output_write_tiles
            {0, 0, 0, 0},                                 // offsets
            ISIStrategy::CLUSTERING,                      // isi_strategy
            false,                                        // weight_sparsity_enabled
    };
    wl.mpe_engine = MPEEngine::DCIM;

    DPU_OperationSanitizer sanitizer;
    SanityReport report;

    // if the .at(Operation::MAXPOOL) lookup throws std::out_of_range uncaught, this call itself aborts the test
    ASSERT_NO_THROW(sanitizer.check_and_sanitize(wl, report));

    utest_log::debug() << "report.value() = " << VPUNN::Cycles::toErrorText(report.value()) << "\n";
    utest_log::debug() << "report.info    = " << report.info << "\n";

    EXPECT_EQ(report.value(), V(Cycles::ERROR_INVALID_INPUT_CONFIGURATION))
            << "DCIM + MAXPOOL should  be considered a INVALID: \n"
            << report.info;
    EXPECT_NE(report.value(), V(Cycles::NO_ERROR)) << "DCIM + MAXPOOL should not be considered a valid workload";
}

}  // namespace VPUNN_unit_tests