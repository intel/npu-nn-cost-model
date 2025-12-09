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
class DPU_OperationSanitizerTestNPU4x : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }

private:
};

class DPU_WorkloadValidatorTestNPU4x : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }

private:
};


TEST_F(DPU_WorkloadValidatorTestNPU4x, basicCheckerTest_VPU40) {
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

TEST_F(DPU_WorkloadValidatorTestNPU4x, outputWriteTilesCheckerTest_VPU40) {
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

TEST_F(DPU_WorkloadValidatorTestNPU4x, Memory_Output_32Bits_NPU40) {
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