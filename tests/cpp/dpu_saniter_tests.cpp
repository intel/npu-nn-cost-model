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

#include "common_helpers.h"

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {
using namespace VPUNN;
class DPU_OperationSanitizerTest : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }

private:
};

TEST_F(DPU_OperationSanitizerTest, basicSanitizeTest) {
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

TEST_F(DPU_WorkloadValidatorTest, basicCheckerTest_VPU40) {
    VPUNN::DPU_OperationSanitizer dut;
    VPUNN::VPUDevice device_req{VPUNN::VPUDevice::VPU_4_0};
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
TEST_F(DPU_WorkloadValidatorTest, outputWriteTilesCheckerTest_VPU40) {
    VPUNN::DPU_OperationSanitizer dut;
    VPUNN::SanityReport sane;

    const VPUNN::DPUWorkload wl_opMAXPOOL{
            VPUNN::VPUDevice::VPU_4_0,
            VPUNN::Operation::MAXPOOL,
            {VPUNN::VPUTensor(112, 112, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(56, 56, 64, 1, VPUNN::DataType::UINT8)},    // output dimensions
            {3, 3},                                                       // kernels
            {2, 2},                                                       // strides
            {1, 0, 1, 0},                                                 // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                           // execution mode
            VPUNN::ActivationFunction::NONE,                              // activation
            0.0F,                                                         // act_sparsity
            0.0F,                                                         // weight_sparsity
            {swz_def, swz_def},                                           // input_swizzling
            {swz_def},                                                    // output_swizzling
            1,                                                            // output_write_tiles
            {0, 0, 0, 0},                                                 // offsets
            VPUNN::ISIStrategy::CLUSTERING,                               // isi_strategy
            false,                                                        // weight_sparsity_enabled
    };
    const VPUNN::DPUWorkload wl_opELTWISE{
            VPUNN::VPUDevice::VPU_4_0,
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
    const VPUNN::DPUWorkload wl_opCONVOLUTION_CLUST{
            VPUNN::VPUDevice::VPU_4_0,
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
    const VPUNN::DPUWorkload wl_opCONVOLUTION_SOK = {
            VPUNN::VPUDevice::VPU_4_0,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(128, 128, 16, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(128, 128, 16, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            {5, 5},                                                         // kernels
            {1, 1},                                                         // strides
            {2, 2, 2, 2},                                                   // padding
            VPUNN::ExecutionMode::CUBOID_4x16,                              // execution mode
            VPUNN::ActivationFunction::NONE,                                // activation
            0.0F,                                                           // act_sparsity
            0.0F,                                                           // weight_sparsity
            {swz_def, swz_def},                                             // input_swizzling
            {swz_def},                                                      // output_swizzling
            2,                                                              // output_write_tiles > 1 because SOK
            {0, 0, 0, 0},                                                   // offsets
            VPUNN::ISIStrategy::SPLIT_OVER_K,                               // isi_strategy
            false,                                                          // weight_sparsity_enabled

    };

    struct TestInput {
        int number_of_tiles;
        VPUNN::DPUWorkload wl;
    };

    struct TestExpectation {
        unsigned int error_exp;
    };

    struct TestCase {
        TestInput t_in;
        TestExpectation t_exp;
        std::string test_case = "";
    };

    using TestsVector = std::vector<TestCase>;

    // this lambda function checks if number_of_tiles exists in output_write_tiles={1,2,3,4,5,6,7,8}
    auto verify_nr_of_tiles = [&sane, &dut](TestsVector& tests) {
        for (auto& t : tests) {
            std::cout << t.test_case << "\n";

            t.t_in.wl.output_write_tiles = static_cast<unsigned int>(t.t_in.number_of_tiles);
            dut.check_data_consistency(t.t_in.wl, sane);
            EXPECT_EQ(sane.value(), t.t_exp.error_exp)
                    << sane.info << "\n expected error is: " << VPUNN::Cycles::toErrorText(t.t_exp.error_exp)
                    << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                    << t.t_in.wl;
        }
    };

    {
        static constexpr unsigned int NO_ERROR_EXPECTED{VPUNN::Cycles::NO_ERROR};
        static constexpr unsigned int ERROR_EXPECTED{VPUNN::Cycles::ERROR_INVALID_INPUT_CONFIGURATION};

        // introducing elements into the vector, we have different cases of workloads and number of tiles
        TestsVector tests = {
                {{1, wl_opCONVOLUTION_CLUST}, {NO_ERROR_EXPECTED}, "Test case 1: 1 tiles, CONVOLUTION-CLUSTERING"},
                {{2, wl_opCONVOLUTION_CLUST}, {NO_ERROR_EXPECTED}, "Test case 2: 2 tiles, CONVOLUTION-CLUSTERING"},
                {{3, wl_opCONVOLUTION_CLUST}, {NO_ERROR_EXPECTED}, "Test case 3: 3 tiles, CONVOLUTION-CLUSTERING"},
                {{4, wl_opCONVOLUTION_CLUST}, {NO_ERROR_EXPECTED}, "Test case 4: 4 tiles, CONVOLUTION-CLUSTERING"},
                {{5, wl_opCONVOLUTION_CLUST}, {NO_ERROR_EXPECTED}, "Test case 5: 5 tiles, CONVOLUTION-CLUSTERING"},
                {{6, wl_opCONVOLUTION_CLUST}, {NO_ERROR_EXPECTED}, "Test case 6: 6 tiles, CONVOLUTION-CLUSTERING"},
                {{7, wl_opCONVOLUTION_CLUST}, {NO_ERROR_EXPECTED}, "Test case 7: 7 tiles, CONVOLUTION-CLUSTERING"},
                {{8, wl_opCONVOLUTION_CLUST}, {NO_ERROR_EXPECTED}, "Test case 8: 8 tiles, CONVOLUTION-CLUSTERING"},

                {{1, wl_opCONVOLUTION_SOK},
                 {ERROR_EXPECTED},
                 "Test case 9: 1 tiles, CONVOLUTION-SOK"},  // ERROR CASE SOK output_write_tiles >1
                {{2, wl_opCONVOLUTION_SOK}, {NO_ERROR_EXPECTED}, "Test case 10: 2 tiles, CONVOLUTION-SOK"},
                {{3, wl_opCONVOLUTION_SOK}, {NO_ERROR_EXPECTED}, "Test case 11: 3 tiles, CONVOLUTION-SOK"},
                {{4, wl_opCONVOLUTION_SOK}, {NO_ERROR_EXPECTED}, "Test case 12: 4 tiles, CONVOLUTION-SOK"},
                {{5, wl_opCONVOLUTION_SOK}, {NO_ERROR_EXPECTED}, "Test case 13: 5 tiles, CONVOLUTION-SOK"},
                {{6, wl_opCONVOLUTION_SOK}, {NO_ERROR_EXPECTED}, "Test case 14: 6 tiles, CONVOLUTION-SOK"},
                {{7, wl_opCONVOLUTION_SOK}, {NO_ERROR_EXPECTED}, "Test case 15: 7 tiles, CONVOLUTION-SOK"},
                {{8, wl_opCONVOLUTION_SOK}, {NO_ERROR_EXPECTED}, "Test case 16: 8 tiles, CONVOLUTION-SOK"},

                {{1, wl_opMAXPOOL}, {NO_ERROR_EXPECTED}, "Test case 17: 1 tiles, MAXPOOL"},
                {{2, wl_opMAXPOOL}, {NO_ERROR_EXPECTED}, "Test case 18: 2 tiles, MAXPOOL"},
                {{3, wl_opMAXPOOL}, {NO_ERROR_EXPECTED}, "Test case 19: 3 tiles, MAXPOOL"},
                {{4, wl_opMAXPOOL}, {NO_ERROR_EXPECTED}, "Test case 20: 4 tiles, MAXPOOL"},
                {{5, wl_opMAXPOOL}, {NO_ERROR_EXPECTED}, "Test case 21: 5 tiles, MAXPOOL"},
                {{6, wl_opMAXPOOL}, {NO_ERROR_EXPECTED}, "Test case 22: 6 tiles, MAXPOOL"},
                {{7, wl_opMAXPOOL}, {NO_ERROR_EXPECTED}, "Test case 23: 7 tiles, MAXPOOL"},
                {{8, wl_opMAXPOOL}, {NO_ERROR_EXPECTED}, "Test case 24: 8 tiles, MAXPOOL"},

                {{1, wl_opELTWISE}, {NO_ERROR_EXPECTED}, "Test case 25: 1 tiles, ELTWISE"},
                {{2, wl_opELTWISE}, {NO_ERROR_EXPECTED}, "Test case 26: 2 tiles, ELTWISE"},
                {{3, wl_opELTWISE}, {NO_ERROR_EXPECTED}, "Test case 27: 3 tiles, ELTWISE"},
                {{4, wl_opELTWISE}, {NO_ERROR_EXPECTED}, "Test case 28: 4 tiles, ELTWISE"},
                {{5, wl_opELTWISE}, {NO_ERROR_EXPECTED}, "Test case 29: 5 tiles, ELTWISE"},
                {{6, wl_opELTWISE}, {NO_ERROR_EXPECTED}, "Test case 30: 6 tiles, ELTWISE"},
                {{7, wl_opELTWISE}, {NO_ERROR_EXPECTED}, "Test case 31: 7 tiles, ELTWISE"},
                {{8, wl_opELTWISE}, {NO_ERROR_EXPECTED}, "Test case 32: 8 tiles, ELTWISE"},

                // ERROR CASES
                {{9, wl_opCONVOLUTION_CLUST}, {ERROR_EXPECTED}, "Test case 18: 9 tiles, CONVOLUTION_CLUST"},
                {{9, wl_opCONVOLUTION_SOK}, {ERROR_EXPECTED}, "Test case 19: 9 tiles, CONVOLUTION_SOK"},
                {{9, wl_opMAXPOOL}, {ERROR_EXPECTED}, "Test case 20: 9 tiles, MAXPOOL"},
                {{9, wl_opELTWISE}, {ERROR_EXPECTED}, "Test case 21: 9 tiles, ELTWISE"},

        };

        verify_nr_of_tiles(tests);
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
TEST_F(DPU_WorkloadValidatorTest, HALOInvalid_smokeTest) {
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

TEST_F(DPU_WorkloadValidatorTest, Memory_Output_32Bits_NPU40) {
    DPU_OperationSanitizer dut;
    const VPUDevice device_req{VPUDevice::VPU_4_0};
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

        EXPECT_EQ(sane.value(), V(Cycles::NO_ERROR))
                << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                << wl;

        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(Cycles::ERROR_INPUT_TOO_BIG))
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

        EXPECT_EQ(sane.value(), V(Cycles::NO_ERROR))
                << sane.info << "\n error is : " << Cycles::toErrorText(sane.value()) << "\n"
                << wl;

        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(Cycles::NO_ERROR))
                << sane.info << "\n error is : " << Cycles::toErrorText(sane.value()) << "\n"
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
            "",                                                              // layer_info
            false,                                                           ///< operation does not have weights
            false                                                            // in_place_output_memory{};
    };
    {  //
        auto wl{wl_ref2};
        dut.check_data_consistency(wl, sane);

        EXPECT_EQ(sane.value(), V(Cycles::NO_ERROR))
                << sane.info << "\n error is : " << Cycles::toErrorText(sane.value()) << "\n"
                << wl;

        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(Cycles::NO_ERROR))
                << sane.info << "\n error is : " << Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }
    {  //
        auto wl{std::move(wl_ref2)};
        dut.check_data_consistency(wl, sane);

        EXPECT_EQ(sane.value(), V(Cycles::NO_ERROR))
                << sane.info << "\n error is : " << Cycles::toErrorText(sane.value()) << "\n"
                << wl;

        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(Cycles::NO_ERROR))
                << sane.info << "\n error is : " << Cycles::toErrorText(sane.value()) << "\n"
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
            "",                                                              // layer_info
            false,                                                           ///< operation does not have weights
            false                                                            // in_place_output_memory{};
    };
    {  //
        auto wl{wl_ref2_large};
        dut.check_data_consistency(wl, sane);

        EXPECT_EQ(sane.value(), V(Cycles::NO_ERROR))
                << sane.info << "\n error is : " << Cycles::toErrorText(sane.value()) << "\n"
                << wl;

        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(Cycles::ERROR_INPUT_TOO_BIG))
                << sane.info << "\n error is : " << Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }
    {  //
        auto wl{std::move(wl_ref2_large)};
        dut.check_data_consistency(wl, sane);

        EXPECT_EQ(sane.value(), V(Cycles::NO_ERROR))
                << sane.info << "\n error is : " << Cycles::toErrorText(sane.value()) << "\n"
                << wl;

        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(Cycles::ERROR_INPUT_TOO_BIG))
                << sane.info << "\n error is : " << Cycles::toErrorText(sane.value()) << "\n"
                << wl;
    }
}

}  // namespace VPUNN_unit_tests
