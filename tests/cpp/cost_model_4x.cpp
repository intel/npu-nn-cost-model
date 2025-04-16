// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include "vpu_cost_model.h"

#include <gtest/gtest.h>
#include "common_helpers.h"
#include "vpu/cycles_interface_types.h"

#include <algorithm>
#include <unordered_map>

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {
using namespace VPUNN;

class Test_CostModel_NPU4x_extension : public ::testing::Test {
public:
protected:
    const VPUDevice device{VPUDevice::VPU_4_0};
    const VPUDevice device_req{device};

    void SetUp() override {
        // TestCostModel::SetUp();
    }
    Test_CostModel_NPU4x_extension() {
    }

    void basicTest(const DPUWorkload& workload, VPUCostModel& crt_model, std::string info = "") {
        DPUWorkload wl{workload};

        std::string errInfo;
        unsigned cost_cyc{};
        ASSERT_NO_THROW(cost_cyc = crt_model.DPU(wl, errInfo)) << info << wl;

        std::stringstream buffer;
        buffer << "\nDetails : " << info << " Op:" << Operation_ToText.at(static_cast<int>(wl.op))
               << "\n  NN cyc:" << cost_cyc << " : " << Cycles::toErrorText(cost_cyc) << "\n ";
        std::string details = buffer.str();

        EXPECT_TRUE(!Cycles::isErrorCode(cost_cyc)) << info << wl << errInfo << details;

        std::cout << details
                  << "-X-------------------------------------------------------------------------------------------"
                     "-\n";
    }
    void basicNOKTest(const DPUWorkload& workload, VPUCostModel& crt_model, const CyclesInterfaceType errcode_expected,
                      std::string info = "") {
        DPUWorkload wl{workload};

        std::string errInfo;
        unsigned cost_cyc{};
        ASSERT_NO_THROW(cost_cyc = crt_model.DPU(wl, errInfo)) << info << wl;

        std::stringstream buffer;
        buffer << "\nDetails : " << info << " Op:" << Operation_ToText.at(static_cast<int>(wl.op))
               << "\n  NN cyc:" << cost_cyc << " : " << Cycles::toErrorText(cost_cyc) << "\n ";
        std::string details = buffer.str();

        EXPECT_TRUE(Cycles::isErrorCode(cost_cyc)) << info << wl << errInfo << details;
        EXPECT_EQ(errcode_expected, cost_cyc) << info << wl << errInfo << details;

        std::cout << details
                  << "-X-------------------------------------------------------------------------------------------"
                     "-\n";
    }

private:
};

TEST_F(Test_CostModel_NPU4x_extension, FP32_output_basic) {
    VPUCostModel crt_model{VPU_4_0_MODEL_PATH};

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

    //{
    //    DPUWorkload wl{wl_ref_less};
    //    basicNOKTest(wl, crt_model, Cycles::ERROR_INVALID_INPUT_CONFIGURATION);
    //}
    //{
    //    DPUWorkload wl{wl_ref2};
    //    basicNOKTest(wl, crt_model, Cycles::ERROR_INVALID_INPUT_CONFIGURATION);
    //}

    {
        DPUWorkload wl{wl_ref_less};
        basicTest(wl, crt_model, " COnv should fit in memory.");
    }
    {
        DPUWorkload wl{wl_ref2};
        basicTest(wl, crt_model, "Elemwise should fit into memory.");
    }

    auto checkSame = [&crt_model](const DPUWorkload& w1, const DPUWorkload& w2, std::string info = "") {
        DPUWorkload wl{w1};

        std::string errInfo;
        unsigned cost_cyc{};
        ASSERT_NO_THROW(cost_cyc = crt_model.DPU(wl, errInfo)) << info << wl;

        DPUWorkload wl_2{w2};

        unsigned cost_cyc2{};
        ASSERT_NO_THROW(cost_cyc2 = crt_model.DPU(wl_2, errInfo)) << info << wl_2;

        EXPECT_EQ(cost_cyc, cost_cyc2) << info << wl << wl_2;
    };

    const DPUWorkload wl_ref_less_16{
            device,
            Operation::CONVOLUTION,
            {VPUTensor(64, 64, 64, 1, DataType::UINT8)},                     // input dimensions
            {VPUTensor(21, 21, 512 + 32, 1, DataType::FLOAT16)},             // output dimensions
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

    const DPUWorkload wl_ref2_16{
            device,
            Operation::ELTWISE,
            {VPUTensor(64, 64, 64, 1, DataType::UINT8)},                     // input dimensions
            {VPUTensor(64, 64, 64, 1, DataType::FLOAT16)},                   // output dimensions
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

    {
        checkSame(wl_ref_less, wl_ref_less_16, "Test 1, convs with FP32 and FP16 output should be equal");
        checkSame(wl_ref2, wl_ref2_16, "Test 2, elemnwise with FP32 and FP16 output should be equal");
    }
}

}  // namespace VPUNN_unit_tests