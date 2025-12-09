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
class DPU_OperationSanitizerTestVPU2x : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }

private:
};

class DPU_WorkloadValidatorTestVPU2x : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }

private:
};

TEST_F(DPU_OperationSanitizerTestVPU2x, basicSanitizeTest) {
    VPUNN::DPU_OperationSanitizer dut;
    VPUNN::VPUDevice device_req{VPUNN::VPUDevice::VPU_2_7};
    VPUNN::SanityReport sane;
    {
        VPUNN::DPUWorkload wl = {
                device_req,
                VPUNN::Operation::CONVOLUTION,
                {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
                {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
                {1, 1},                                                     // kernels
                {1, 1},                                                     // strides
                {0, 0, 0, 0},                                               // padding
                VPUNN::ExecutionMode::CUBOID_16x16                          // execution mode
        };

        dut.check_and_sanitize(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
        EXPECT_EQ(wl.inputs[0].get_dtype(), VPUNN::DataType::UINT8);
    }
    {
        VPUNN::DPUWorkload wl = {
                device_req,
                VPUNN::Operation::CONVOLUTION,
                {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::INT8)},  // input dimensions
                {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::INT8)},  // output dimensions
                {1, 1},                                                    // kernels
                {1, 1},                                                    // strides
                {0, 0, 0, 0},                                              // padding
                VPUNN::ExecutionMode::CUBOID_16x16                         // execution mode
        };

        dut.check_and_sanitize(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
        EXPECT_EQ(wl.inputs[0].get_dtype(), VPUNN::DataType::UINT8);
        EXPECT_EQ(wl.outputs[0].get_dtype(), VPUNN::DataType::UINT8);
    }

    {
        VPUNN::DPUWorkload wl = {
                device_req,
                VPUNN::Operation::CONVOLUTION,
                {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::BFLOAT16)},  // input dimensions
                {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::FLOAT16)},   // output dimensions
                {1, 1},                                                        // kernels
                {1, 1},                                                        // strides
                {0, 0, 0, 0},                                                  // padding
                VPUNN::ExecutionMode::CUBOID_16x16                             // execution mode
        };

        dut.check_and_sanitize(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
        EXPECT_EQ(wl.inputs[0].get_dtype(), VPUNN::DataType::FLOAT16);
        EXPECT_EQ(wl.outputs[0].get_dtype(), VPUNN::DataType::FLOAT16);
    }

    {
        VPUNN::DPUWorkload wl = {
                device_req,
                VPUNN::Operation::__size,
                {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::INT8)},  // input dimensions
                {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::INT8)},  // output dimensions
                {1, 1},                                                    // kernels
                {1, 1},                                                    // strides
                {0, 0, 0, 0},                                              // padding
                VPUNN::ExecutionMode::CUBOID_16x16                         // execution mode
        };

        dut.check_and_sanitize(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::ERROR_INVALID_INPUT_OPERATION))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }
    {
        VPUNN::DPUWorkload wl = {
                device_req,
                VPUNN::Operation::ELTWISE,
                {VPUNN::VPUTensor(1600, 1600, 64, 1, VPUNN::DataType::INT8)},  // input dimensions
                {VPUNN::VPUTensor(1600, 1600, 64, 1, VPUNN::DataType::INT8)},  // output dimensions
                {1, 1},                                                        // kernels
                {1, 1},                                                        // strides
                {0, 0, 0, 0},                                                  // padding
                VPUNN::ExecutionMode::CUBOID_16x16                             // execution mode
        };

        dut.check_and_sanitize(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::ERROR_INPUT_TOO_BIG))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }

    device_req = VPUNN::VPUDevice::VPU_2_0;
    {
        VPUNN::DPUWorkload wl = {
                device_req,
                VPUNN::Operation::CONVOLUTION,
                {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::INT8, VPUNN::Layout::ZMAJOR)},  // input dimensions
                {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::INT8, VPUNN::Layout::ZMAJOR)},  // output dimensions
                {1, 1},                                                                           // kernels
                {1, 1},                                                                           // strides
                {0, 0, 0, 0},                                                                     // padding
                VPUNN::ExecutionMode::VECTOR                                                      // execution mode
        };

        dut.check_and_sanitize(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
        EXPECT_EQ(wl.inputs[0].get_dtype(), VPUNN::DataType::UINT8);
        EXPECT_EQ(wl.outputs[0].get_dtype(), VPUNN::DataType::UINT8);
    }

    {
        VPUNN::DPUWorkload wl = {
                device_req,
                VPUNN::Operation::CONVOLUTION,
                {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::BFLOAT16,
                                  VPUNN::Layout::ZMAJOR)},  // input dimensions
                {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::FLOAT16,
                                  VPUNN::Layout::ZMAJOR)},  // output dimensions
                {1, 1},                                     // kernels
                {1, 1},                                     // strides
                {0, 0, 0, 0},                               // padding
                VPUNN::ExecutionMode::VECTOR                // execution mode
        };

        dut.check_and_sanitize(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
        EXPECT_EQ(wl.inputs[0].get_dtype(), VPUNN::DataType::FLOAT16);
        EXPECT_EQ(wl.outputs[0].get_dtype(), VPUNN::DataType::FLOAT16);
    }
}

TEST_F(DPU_WorkloadValidatorTestVPU2x, basicCheckerTest) {
    VPUNN::DPU_OperationSanitizer dut;
    VPUNN::VPUDevice device_req{VPUNN::VPUDevice::VPU_2_7};
    VPUNN::SanityReport sane;
    {
        VPUNN::DPUWorkload wl = {
                device_req,
                VPUNN::Operation::CONVOLUTION,
                {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
                {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
                {12, 1},                                                    // kernels
                {1, 1},                                                     // strides
                {1, 1, 1, 1},                                               // padding
                VPUNN::ExecutionMode::CUBOID_16x16                          // execution mode
        };

        dut.check_data_consistency(wl, sane);

        // ASSERT_EQ(sane.value(), VPUNN::Cycles::NO_ERROR)<<sane.value();
        ASSERT_EQ(sane.value(), V(VPUNN::Cycles::ERROR_INVALID_INPUT_CONFIGURATION))
                << sane.info << "\n error is : " << VPUNN::Cycles::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }
}

TEST_F(DPU_WorkloadValidatorTestVPU2x, elementwise_Test) {
    VPUNN::DPU_OperationSanitizer dut;
    const VPUNN::VPUDevice device_req{VPUNN::VPUDevice::VPU_2_7};
    VPUNN::SanityReport sane;
    const VPUNN::DPUWorkload wl_ref{
            device_req,
            VPUNN::Operation::ELTWISE,
            {VPUNN::VPUTensor(56, 56, 256, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(56, 56, 256, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {1, 1},                                                      // kernels
            {1, 1},                                                      // strides
            {0, 0, 0, 0},                                                // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                          // execution mode
            VPUNN::ActivationFunction::NONE,                             // activation
            0.0F,                                                        // act_sparsity
            0.0F,                                                        // weight_sparsity
            {swz_def, swz_def},                                          // input_swizzling
            {swz_def},                                                   // output_swizzling
            1,                                                           // output_write_tiles
            {0, 0, 0, 0},                                                // offsets
            VPUNN::ISIStrategy::CLUSTERING,                              // isi_strategy
            false,                                                       // weight_sparsity_enabled
    };

    {  // no ISI strategy
        auto wl{wl_ref};
        dut.check_data_consistency(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }

    {  // SOH->
        auto wl{wl_ref};
        wl.isi_strategy = VPUNN::ISIStrategy::SPLIT_OVER_H;
        wl.output_write_tiles = 1;  // SOH must be 1
        dut.check_data_consistency(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }
    {  // SOK-> weights are  is halved in contribution(NOT any more).  OWT=1 not allowed
        auto wl{wl_ref};
        wl.isi_strategy = VPUNN::ISIStrategy::SPLIT_OVER_K;  // not allowed for elm wise

        dut.check_data_consistency(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::ERROR_INVALID_INPUT_CONFIGURATION))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }
    {  // SOK-> weights are  is halved in contribution(NOT any more)
        auto wl{std::move(wl_ref)};
        wl.isi_strategy = VPUNN::ISIStrategy::SPLIT_OVER_K;  // OK
        wl.output_write_tiles = 2;                           // SOK requires that write tiles >1

        dut.check_data_consistency(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }
}

TEST_F(DPU_WorkloadValidatorTestVPU2x, convolution_8641_Test) {
    VPUNN::DPU_OperationSanitizer dut;
    const VPUNN::VPUDevice device_req{VPUNN::VPUDevice::VPU_2_7};
    VPUNN::SanityReport sane;
    const VPUNN::DPUWorkload wl_ref{
            device_req,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(14, 14, 512, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(7, 7, 512, 1, VPUNN::DataType::UINT8)},    // output dimensions
            {3, 3},                                                      // kernels
            {2, 2},                                                      // strides
            {1, 0, 1, 0},                                                // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                          // execution mode
            VPUNN::ActivationFunction::NONE,                             // activation
            0.0F,                                                        // act_sparsity
            0.0F,                                                        // weight_sparsity
            {swz_def, swz_def},                                          // input_swizzling
            {swz_def},                                                   // output_swizzling
            1,                                                           // output_write_tiles
            {0, 0, 0, 0},                                                // offsets
            VPUNN::ISIStrategy::CLUSTERING,                              // isi_strategy
            false,                                                       // weight_sparsity_enabled
    };

    {  // no ISI strategy
        auto wl{wl_ref};
        dut.check_data_consistency(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }

    {  // SOH->
        auto wl{wl_ref};
        wl.isi_strategy = VPUNN::ISIStrategy::SPLIT_OVER_H;
        wl.output_write_tiles = 1;

        dut.check_data_consistency(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }
    {  // SOK-> weights are  is halved in contribution(NOT any more)
        auto wl{wl_ref};
        wl.isi_strategy = VPUNN::ISIStrategy::SPLIT_OVER_K;  // output tiles must be >1

        dut.check_data_consistency(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::ERROR_INVALID_INPUT_CONFIGURATION))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }
    {  // SOK-> weights are  is halved in contribution(NOT any more)
        auto wl{std::move(wl_ref)};
        wl.isi_strategy = VPUNN::ISIStrategy::SPLIT_OVER_K;
        wl.output_write_tiles = 2;  // SOK requires that write tiles >1

        dut.check_data_consistency(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }
}

TEST_F(DPU_WorkloadValidatorTestVPU2x, avepool_172_Test) {
    VPUNN::DPU_OperationSanitizer dut;
    const VPUNN::VPUDevice device_req{VPUNN::VPUDevice::VPU_2_7};
    VPUNN::SanityReport sane;
    const VPUNN::DPUWorkload wl_ref{
            device_req,
            VPUNN::Operation::AVEPOOL,                                  /// shall not be allowed
            {VPUNN::VPUTensor(7, 7, 2048, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(1, 1, 2048, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {7, 7},                                                     // kernels
            {1, 1},                                                     // strides
            {0, 0, 0, 0},                                               // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                         // execution mode
            VPUNN::ActivationFunction::NONE,                            // activation
            0.0F,                                                       // act_sparsity
            0.0F,                                                       // weight_sparsity
            {swz_def, swz_def},                                         // input_swizzling
            {swz_def},                                                  // output_swizzling
            1,                                                          // output_write_tiles
            {0, 0, 0, 0},                                               // offsets
            VPUNN::ISIStrategy::CLUSTERING,                             // isi_strategy
            false,                                                      // weight_sparsity_enabled
    };

    {  // no ISI strategy
        auto wl{wl_ref};
        dut.check_data_consistency(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::ERROR_INVALID_INPUT_OPERATION))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }

    {  // SOH->
        auto wl{wl_ref};
        wl.isi_strategy = VPUNN::ISIStrategy::SPLIT_OVER_H;
        wl.output_write_tiles = 2;  // non cluster requires that write tiles >1

        dut.check_data_consistency(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::ERROR_INVALID_INPUT_OPERATION))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }
    {  // SOK-> weights are  is halved in contribution(NOT any more)
        auto wl{wl_ref};
        wl.isi_strategy = VPUNN::ISIStrategy::SPLIT_OVER_K;  // output tiles must be >1

        dut.check_data_consistency(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::ERROR_INVALID_INPUT_OPERATION))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }
    {  // SOK-> weights are  is halved in contribution(NOT any more)
        auto wl{std::move(wl_ref)};
        wl.isi_strategy = VPUNN::ISIStrategy::SPLIT_OVER_K;
        wl.output_write_tiles = 2;  // non cluster requires that write tiles >1

        dut.check_data_consistency(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::ERROR_INVALID_INPUT_OPERATION))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }
}

TEST_F(DPU_WorkloadValidatorTestVPU2x, dw_conv_172_Test) {
    VPUNN::DPU_OperationSanitizer dut;
    const VPUNN::VPUDevice device_req{VPUNN::VPUDevice::VPU_2_7};
    VPUNN::SanityReport sane;
    const VPUNN::DPUWorkload wl0_ref{
            device_req,
            VPUNN::Operation::DW_CONVOLUTION,                           ///
            {VPUNN::VPUTensor(7, 7, 2048, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(1, 1, 2048, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {7, 7},                                                     // kernels
            {1, 1},                                                     // strides
            {0, 0, 0, 0},                                               // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                         // execution mode
            VPUNN::ActivationFunction::NONE,                            // activation
            0.0F,                                                       // act_sparsity
            0.0F,                                                       // weight_sparsity
            {swz_def, swz_def},                                         // input_swizzling
            {swz_def},                                                  // output_swizzling
            1,                                                          // output_write_tiles
            {0, 0, 0, 0},                                               // offsets
            VPUNN::ISIStrategy::CLUSTERING,                             // isi_strategy
            false,                                                      // weight_sparsity_enabled
    };
    const VPUNN::DPUWorkload wl_ref{
            device_req,
            VPUNN::Operation::DW_CONVOLUTION,                         ///
            {VPUNN::VPUTensor(7, 7, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(1, 1, 16, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {7, 7},                                                   // kernels
            {1, 1},                                                   // strides
            {0, 0, 0, 0},                                             // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                       // execution mode
            VPUNN::ActivationFunction::NONE,                          // activation
            0.0F,                                                     // act_sparsity
            0.0F,                                                     // weight_sparsity
            {swz_def, swz_def},                                       // input_swizzling
            {swz_def},                                                // output_swizzling
            1,                                                        // output_write_tiles
            {0, 0, 0, 0},                                             // offsets
            VPUNN::ISIStrategy::CLUSTERING,                           // isi_strategy
            false,                                                    // weight_sparsity_enabled
    };

    {  // no ISI strategy
        auto wl{std::move(wl0_ref)};
        dut.check_data_consistency(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::ERROR_INVALID_INPUT_CONFIGURATION))  // input channels
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }
    {  // no ISI strategy
        auto wl{wl_ref};
        dut.check_data_consistency(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }

    {  // SOH->
        auto wl{wl_ref};
        wl.isi_strategy = VPUNN::ISIStrategy::SPLIT_OVER_H;
        wl.output_write_tiles = 1;  // SOH must be 1
        dut.check_data_consistency(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }

    {  // SOK-> weights are  is halved in contribution(NOT any more)
        auto wl{std::move(wl_ref)};
        wl.isi_strategy = VPUNN::ISIStrategy::SPLIT_OVER_K;
        wl.output_write_tiles = 2;  // SOK must be 1+

        dut.check_data_consistency(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }
}

TEST_F(DPU_WorkloadValidatorTestVPU2x, maxpool_172_Test) {
    VPUNN::DPU_OperationSanitizer dut;
    const VPUNN::VPUDevice device_req{VPUNN::VPUDevice::VPU_2_7};
    VPUNN::SanityReport sane;
    const VPUNN::DPUWorkload wl_ref{
            device_req,
            VPUNN::Operation::MAXPOOL,                                ///
            {VPUNN::VPUTensor(7, 7, 32, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(1, 1, 32, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {7, 7},                                                   // kernels
            {1, 1},                                                   // strides
            {0, 0, 0, 0},                                             // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                       // execution mode
            VPUNN::ActivationFunction::NONE,                          // activation
            0.0F,                                                     // act_sparsity
            0.0F,                                                     // weight_sparsity
            {swz_def, swz_def},                                       // input_swizzling
            {swz_def},                                                // output_swizzling
            1,                                                        // output_write_tiles
            {0, 0, 0, 0},                                             // offsets
            VPUNN::ISIStrategy::CLUSTERING,                           // isi_strategy
            false,                                                    // weight_sparsity_enabled
    };

    {  // no ISI strategy
        auto wl{wl_ref};
        dut.check_data_consistency(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }

    {  // SOH->
        auto wl{wl_ref};
        wl.isi_strategy = VPUNN::ISIStrategy::SPLIT_OVER_H;
        wl.output_write_tiles = 1;  // SOH must be 1
        dut.check_data_consistency(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }

    {  // SOK-> weights are  is halved in contribution(NOT any more)
        auto wl{std::move(wl_ref)};
        wl.isi_strategy = VPUNN::ISIStrategy::SPLIT_OVER_K;
        wl.output_write_tiles = 2;  // SOK requires that write tiles >1

        dut.check_data_consistency(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }
}

TEST_F(DPU_WorkloadValidatorTestVPU2x, maxpool_152Luca_Test) {
    VPUNN::DPU_OperationSanitizer dut;
    const VPUNN::VPUDevice device_req{VPUNN::VPUDevice::VPU_2_7};
    VPUNN::SanityReport sane;
    const VPUNN::DPUWorkload wl_ref{
            device_req,
            VPUNN::Operation::MAXPOOL,                                   ///
            {VPUNN::VPUTensor(5, 8132, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(1, 2033, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {5, 3},                                                      // kernels
            {4, 4},                                                      // strides
            {0, 0, 0, 0},                                                // padding
            VPUNN::ExecutionMode::CUBOID_8x16,                           // execution mode
            VPUNN::ActivationFunction::NONE,                             // activation
            0.0F,                                                        // act_sparsity
            0.0F,                                                        // weight_sparsity
            {swz_def, swz_def},                                          // input_swizzling
            {swz_def},                                                   // output_swizzling
            1,                                                           // output_write_tiles
            {0, 0, 0, 0},                                                // offsets
            VPUNN::ISIStrategy::CLUSTERING,                              // isi_strategy
            false,                                                       // weight_sparsity_enabled
    };

    {  // SOH->
        auto wl{std::move(wl_ref)};
        wl.isi_strategy = VPUNN::ISIStrategy::SPLIT_OVER_H;
        wl.output_write_tiles = 1;  // SOH must be 1
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

TEST_F(DPU_WorkloadValidatorTestVPU2x, equal_strides_Test) {
    VPUNN::DPU_OperationSanitizer dut;
    const VPUNN::VPUDevice device_req{VPUNN::VPUDevice::VPU_2_7};
    VPUNN::SanityReport sane;
    const VPUNN::DPUWorkload wl_ref{
            device_req,
            VPUNN::Operation::MAXPOOL,                                ///
            {VPUNN::VPUTensor(7, 7, 32, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(6, 6, 32, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {2, 2},                                                   // kernels
            {1, 1},                                                   // strides
            {0, 0, 0, 0},                                             // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                       // execution mode
            VPUNN::ActivationFunction::NONE,                          // activation
            0.0F,                                                     // act_sparsity
            0.0F,                                                     // weight_sparsity
            {swz_def, swz_def},                                       // input_swizzling
            {swz_def},                                                // output_swizzling
            1,                                                        // output_write_tiles
            {0, 0, 0, 0},                                             // offsets
            VPUNN::ISIStrategy::CLUSTERING,                           // isi_strategy
            false,                                                    // weight_sparsity_enabled
    };

    {  // same stride
        auto wl{std::move(wl_ref)};
        dut.check_data_consistency(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;

        wl.strides[0] = 2;
        wl.strides[1] = 2;
        dut.check_data_consistency(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::ERROR_INVALID_INPUT_CONFIGURATION))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }

    const VPUNN::DPUWorkload wl_ref2{
            device_req,
            VPUNN::Operation::MAXPOOL,                                ///
            {VPUNN::VPUTensor(7, 7, 32, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(6, 3, 32, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {2, 2},                                                   // kernels
            {1, 2},                                                   // strides
            {0, 0, 0, 0},                                             // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                       // execution mode
            VPUNN::ActivationFunction::NONE,                          // activation
            0.0F,                                                     // act_sparsity
            0.0F,                                                     // weight_sparsity
            {swz_def, swz_def},                                       // input_swizzling
            {swz_def},                                                // output_swizzling
            1,                                                        // output_write_tiles
            {0, 0, 0, 0},                                             // offsets
            VPUNN::ISIStrategy::CLUSTERING,                           // isi_strategy
            false,                                                    // weight_sparsity_enabled
    };
    {  // strides inequal  1-2
        auto wl{std::move(wl_ref2)};
        dut.check_data_consistency(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;

        wl.strides[0] = 2;
        wl.strides[1] = 1;
        dut.check_data_consistency(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::ERROR_INVALID_INPUT_CONFIGURATION))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;

        wl.strides[0] = 1;
        wl.strides[1] = 3;
        dut.check_data_consistency(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::ERROR_INVALID_INPUT_CONFIGURATION))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }
}
/// halo problems must be detected
TEST_F(DPU_WorkloadValidatorTestVPU2x, HALOInvalid_smokeTest) {
    DPU_OperationSanitizer dut;
    const VPUDevice device_req{VPUNN::VPUDevice::VPU_2_7};
    SanityReport sane;

    const VPUNN::DPUWorkload wl_ref{
            device_req,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(14, 14, 512, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(7, 7, 512, 1, VPUNN::DataType::UINT8)},    // output dimensions
            {3, 3},                                                      // kernels
            {2, 2},                                                      // strides
            {1, 0, 1, 0},                                                // padding TBLR
            VPUNN::ExecutionMode::CUBOID_16x16,                          // execution mode
            VPUNN::ActivationFunction::NONE,                             // activation
            0.0F,                                                        // act_sparsity
            0.0F,                                                        // weight_sparsity
            {swz_def, swz_def},                                          // input_swizzling
            {swz_def},                                                   // output_swizzling
            1,                                                           // output_write_tiles
            {0, 0, 0, 0},                                                // offsets
            VPUNN::ISIStrategy::CLUSTERING,                              // isi_strategy
            false,                                                       // weight_sparsity_enabled
    };

    const HaloWorkload h1_ref{
            {0, 1, 0, 0},  // HaloInfo2D input_0_halo, TBLR
            {0, 0, 0, 0},  // HaloInfo2D output_0_halo TBLR

            {0, 0, 0, 0},       // HaloInfo2D output_0_halo_broadcast_cnt TBLR
            {7, 0, 0, 0, 0, 0}  // HaloInfo3D output_0_inbound_halo TBLR FB
    };

    const HaloWorkload h1_ref2{
            {0, 1, 0, 0},  // HaloInfo2D input_0_halo, TBLR
            {1, 2, 3, 4},  // HaloInfo2D output_0_halo TBLR

            {10, 20, 30, 40},   // HaloInfo2D output_0_halo_broadcast_cnt TBLR
            {7, 0, 0, 0, 0, 0}  // HaloInfo3D output_0_inbound_halo TBLR FB
    };
    const HaloWorkload h1_ref3{
            {1 /*bad*/, 1, 2 /*bad*/, 0},  // HaloInfo2D input_0_halo, TBLR
            {1, 2, 3, 4},                  // HaloInfo2D output_0_halo TBLR

            {0, 0, 0, 0},       // HaloInfo2D output_0_halo_broadcast_cnt TBLR
            {7, 0, 0, 0, 0, 0}  // HaloInfo3D output_0_inbound_halo TBLR FB
    };

    VPUNN::DPUWorkload wl_ref_halo{wl_ref};
    wl_ref_halo.halo = h1_ref;

    VPUNN::DPUWorkload wl_ref_halo2{wl_ref};
    wl_ref_halo2.halo = h1_ref2;  // same as 1 but with irrelevant data for memory in

    {
        {
            auto wl{wl_ref};
            EXPECT_NO_THROW(dut.check_data_consistency(wl, sane)) << wl;

            EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
                    << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                    << wl;
        }

        {
            auto wl{std::move(wl_ref_halo)};
            EXPECT_NO_THROW(dut.check_data_consistency(wl, sane)) << wl;

            EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
                    << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                    << wl;
        }
        {
            auto wl{std::move(wl_ref_halo2)};
            EXPECT_NO_THROW(dut.check_data_consistency(wl, sane)) << wl;

            EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
                    << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                    << wl;
        }
        {
            VPUNN::DPUWorkload wl_halo{std::move(wl_ref)};
            wl_halo.halo = h1_ref3;  // bad info, padding conflict

            auto wl{std::move(wl_halo)};
            EXPECT_NO_THROW(dut.check_data_consistency(wl, sane)) << wl;

            EXPECT_EQ(sane.value(), V(Cycles::ERROR_INVALID_INPUT_CONFIGURATION))
                    << sane.info << "\n error is : " << Cycles::toErrorText(sane.value()) << "\n"
                    << wl;
        }
    }
}

}  // namespace VPUNN_unit_tests