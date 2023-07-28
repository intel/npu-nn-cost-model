// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include "vpu/validation/dpu_operations_sanitizer.h"
#include "vpu/validation/dpu_operations_validator.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>
#include "common_helpers.h"

namespace VPUNN_unit_tests {

class DPU_OperationValidator_Test : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }
    VPUNN::DPU_OperationValidator dut;  // no overhead by default

    const int cmx_overhead{0 /*80 * 1024 + 16 * 1024*/};  // cmx_memory_aligned_overhead
    const int alignment{16384};                           // alignement_size_bytes

    bool isAligned(long long mem_size) const {
        return ((mem_size % alignment) != 0) ? false : true;
    }

    long long int align(long long mem_size) const {
        const auto rem = mem_size % alignment;
        return (rem == 0) ? mem_size : mem_size + (alignment - rem);
    }
    DPU_OperationValidator_Test() {
    }

private:
};

TEST_F(DPU_OperationValidator_Test, elementwiseMemorySize_Test) {
    const VPUNN::DPUWorkload wl_ref{
            VPUNN::VPUDevice::VPU_2_7,
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
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},          // input_swizzling
            {VPUNN::Swizzling::KEY_0},                                   // output_swizzling
            1,                                                           // output_write_tiles
            {0, 0, 0, 0},                                                // offsets
            VPUNN::ISIStrategy::CLUSTERING,                              // isi_strategy
            false,                                                       // weight_sparsity_enabled
    };

    // elemntwise has in-place output, so no output contribution to total cmx size

    {  // no ISI strategy
        auto wl{wl_ref};
        EXPECT_TRUE(dut.is_supported(wl.device));

        EXPECT_TRUE(isAligned(56 * 56 * 256));

        VPUNN::MemorySize mem;
        EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

        EXPECT_EQ(mem.input_0, 56 * 56 * 256) << mem << std::endl;
        EXPECT_EQ(mem.output_0, 56 * 56 * 256) << mem << std::endl;
        EXPECT_EQ(mem.input_1, 56 * 56 * 256) << mem << std::endl;
        EXPECT_EQ(mem.cmx, cmx_overhead + mem.input_0 + mem.input_1) << mem << std::endl;
    }

    {  // SOH-> activators are halved in contribution (NOT any more)
        auto wl{wl_ref};
        wl.isi_strategy = VPUNN::ISIStrategy::SPLIT_OVER_H;
        EXPECT_TRUE(dut.is_supported(wl.device));

        VPUNN::MemorySize mem;
        EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

        EXPECT_EQ(mem.input_0, 56 * 56 * 256) << mem << std::endl;
        EXPECT_EQ(mem.output_0, 56 * 56 * 256) << mem << std::endl;
        EXPECT_EQ(mem.input_1, 56 * 56 * 256) << mem << std::endl;
        EXPECT_EQ(mem.cmx, cmx_overhead + mem.input_0 / 1 + mem.input_1) << mem << std::endl;
    }
    {  // SOK-> weights are  is halved in contribution(NOT any more)
        auto wl{wl_ref};
        wl.isi_strategy = VPUNN::ISIStrategy::SPLIT_OVER_K;
        EXPECT_TRUE(dut.is_supported(wl.device));

        VPUNN::MemorySize mem;
        EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

        EXPECT_EQ(mem.input_0, 56 * 56 * 256) << mem << std::endl;
        EXPECT_EQ(mem.output_0, 56 * 56 * 256) << mem << std::endl;
        EXPECT_EQ(mem.input_1, 56 * 56 * 256) << mem << std::endl;
        EXPECT_EQ(mem.cmx, cmx_overhead + mem.input_0 + mem.input_1 / 1) << mem << std::endl;
    }
}

TEST_F(DPU_OperationValidator_Test, convolutionMemorySize_Test) {
    const VPUNN::DPUWorkload wl_ref{
            VPUNN::VPUDevice::VPU_2_7,
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
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},          // input_swizzling
            {VPUNN::Swizzling::KEY_0},                                   // output_swizzling
            1,                                                           // output_write_tiles
            {0, 0, 0, 0},                                                // offsets
            VPUNN::ISIStrategy::CLUSTERING,                              // isi_strategy
            false,                                                       // weight_sparsity_enabled
    };

    const auto input_0_raw{14 * 14 * 512};
    const auto output_0_raw{7 * 7 * 512};
    const auto input_1_raw{3 * 3 * 512 * 512 + 512 * 16};

    const auto input_0{align(input_0_raw)};
    const auto output_0{align(output_0_raw)};
    const auto input_1{align(input_1_raw)};

    EXPECT_TRUE(isAligned(input_0));
    EXPECT_TRUE(isAligned(output_0));
    EXPECT_TRUE(isAligned(input_1));

    {  // no ISI strategy
        auto wl{wl_ref};
        EXPECT_TRUE(dut.is_supported(wl.device));

        VPUNN::MemorySize mem;
        EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

        EXPECT_EQ(mem.input_0, input_0) << mem << std::endl;
        EXPECT_EQ(mem.output_0, output_0) << mem << std::endl;
        EXPECT_EQ(mem.input_1, input_1) << mem << std::endl;
        EXPECT_EQ(mem.cmx, cmx_overhead + mem.input_0 + mem.input_1 + mem.output_0) << mem << std::endl;
    }

    {  // SOH-> activators are halved in contribution (NOT any more)
        auto wl{wl_ref};
        wl.isi_strategy = VPUNN::ISIStrategy::SPLIT_OVER_H;
        EXPECT_TRUE(dut.is_supported(wl.device));

        VPUNN::MemorySize mem;
        EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

        EXPECT_EQ(mem.input_0, input_0) << mem << std::endl;
        EXPECT_EQ(mem.output_0, output_0) << mem << std::endl;
        EXPECT_EQ(mem.input_1, input_1) << mem << std::endl;
        EXPECT_EQ(mem.cmx, cmx_overhead + mem.input_0 / 1 + mem.input_1 + mem.output_0) << mem << std::endl;
    }
    {  // SOK-> weights are  is halved in contribution (NOT any more)
        auto wl{wl_ref};
        wl.isi_strategy = VPUNN::ISIStrategy::SPLIT_OVER_K;
        EXPECT_TRUE(dut.is_supported(wl.device));

        VPUNN::MemorySize mem;
        EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

        EXPECT_EQ(mem.input_0, input_0) << mem << std::endl;
        EXPECT_EQ(mem.output_0, output_0) << mem << std::endl;
        EXPECT_EQ(mem.input_1, input_1) << mem << std::endl;
        EXPECT_EQ(mem.cmx, cmx_overhead + mem.input_0 + mem.input_1 / 1 + mem.output_0) << mem << std::endl;
    }
}

class DPU_WorkloadValidatorTest : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }

private:
};

TEST_F(DPU_WorkloadValidatorTest, basicCheckerTest) {
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

TEST_F(DPU_WorkloadValidatorTest, elementwise_Test) {
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
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},          // input_swizzling
            {VPUNN::Swizzling::KEY_0},                                   // output_swizzling
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
    {  // SOK-> weights are  is halved in contribution(NOT any more)
        auto wl{wl_ref};
        wl.isi_strategy = VPUNN::ISIStrategy::SPLIT_OVER_K;  // not allowed for elm wise

        dut.check_data_consistency(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::ERROR_INVALID_INPUT_CONFIGURATION))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }
    {  // SOK-> weights are  is halved in contribution(NOT any more)
        auto wl{wl_ref};
        wl.isi_strategy = VPUNN::ISIStrategy::SPLIT_OVER_K;  // not allowed for elm wise
        wl.output_write_tiles = 2;                           // SOK requires that write tiles >1

        dut.check_data_consistency(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::ERROR_INVALID_INPUT_CONFIGURATION))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }
}

TEST_F(DPU_WorkloadValidatorTest, convolution_8641_Test) {
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
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},          // input_swizzling
            {VPUNN::Swizzling::KEY_0},                                   // output_swizzling
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
        wl.output_write_tiles = 1;  // SOK not allows !=1

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
        auto wl{wl_ref};
        wl.isi_strategy = VPUNN::ISIStrategy::SPLIT_OVER_K;
        wl.output_write_tiles = 2;  // SOK requires that write tiles >1

        dut.check_data_consistency(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }
}

TEST_F(DPU_WorkloadValidatorTest, avepool_172_Test) {
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
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},         // input_swizzling
            {VPUNN::Swizzling::KEY_0},                                  // output_swizzling
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
        auto wl{wl_ref};
        wl.isi_strategy = VPUNN::ISIStrategy::SPLIT_OVER_K;
        wl.output_write_tiles = 2;  // non cluster requires that write tiles >1

        dut.check_data_consistency(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::ERROR_INVALID_INPUT_OPERATION))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }
}

TEST_F(DPU_WorkloadValidatorTest, dw_conv_172_Test) {
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
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},         // input_swizzling
            {VPUNN::Swizzling::KEY_0},                                  // output_swizzling
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
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},       // input_swizzling
            {VPUNN::Swizzling::KEY_0},                                // output_swizzling
            1,                                                        // output_write_tiles
            {0, 0, 0, 0},                                             // offsets
            VPUNN::ISIStrategy::CLUSTERING,                           // isi_strategy
            false,                                                    // weight_sparsity_enabled
    };

    {  // no ISI strategy
        auto wl{wl0_ref};
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
        auto wl{wl_ref};
        wl.isi_strategy = VPUNN::ISIStrategy::SPLIT_OVER_K;
        wl.output_write_tiles = 2;  // SOK must be 1+

        dut.check_data_consistency(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }
}

TEST_F(DPU_WorkloadValidatorTest, maxpool_172_Test) {
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
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},       // input_swizzling
            {VPUNN::Swizzling::KEY_0},                                // output_swizzling
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
        auto wl{wl_ref};
        wl.isi_strategy = VPUNN::ISIStrategy::SPLIT_OVER_K;
        wl.output_write_tiles = 2;  // SOK requires that write tiles >1

        dut.check_data_consistency(wl, sane);

        EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }
}

TEST_F(DPU_WorkloadValidatorTest, maxpool_152Luca_Test) {
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
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},          // input_swizzling
            {VPUNN::Swizzling::KEY_0},                                   // output_swizzling
            1,                                                           // output_write_tiles
            {0, 0, 0, 0},                                                // offsets
            VPUNN::ISIStrategy::CLUSTERING,                              // isi_strategy
            false,                                                       // weight_sparsity_enabled
    };

    {  // SOH->
        auto wl{wl_ref};
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

TEST_F(DPU_WorkloadValidatorTest, equal_strides_Test) {
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
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},       // input_swizzling
            {VPUNN::Swizzling::KEY_0},                                // output_swizzling
            1,                                                        // output_write_tiles
            {0, 0, 0, 0},                                             // offsets
            VPUNN::ISIStrategy::CLUSTERING,                           // isi_strategy
            false,                                                    // weight_sparsity_enabled
    };

    {  // same stride
        auto wl{wl_ref};
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
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},       // input_swizzling
            {VPUNN::Swizzling::KEY_0},                                // output_swizzling
            1,                                                        // output_write_tiles
            {0, 0, 0, 0},                                             // offsets
            VPUNN::ISIStrategy::CLUSTERING,                           // isi_strategy
            false,                                                    // weight_sparsity_enabled
    };
    {  // strides inequal  1-2
        auto wl{wl_ref2};
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

}  // namespace VPUNN_unit_tests
