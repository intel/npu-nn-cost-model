// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include "vpu/shave/activation.h"
#include "vpu/shave/data_movement.h"
#include "vpu/shave/elementwise.h"

#include <gtest/gtest.h>
#include "common_helpers.h"
#include "vpu_cost_model.h"

#include <unordered_map>

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {

/// @brief Tests that the Shave objects can be created. Not covering every functionality/shave
class TestSHAVE : public ::testing::Test {
public:
protected:
    VPUNN::VPUCostModel empty_model{VPUNN::VPUCostModel()};

    const VPUNN::VPUTensor input_0{56, 56, 16, 1, VPUNN::DataType::FLOAT16};   // input dimensions
    const VPUNN::VPUTensor output_0{56, 56, 16, 1, VPUNN::DataType::FLOAT16};  // output dimensions

    void SetUp() override {
    }

private:
};

/// @brief tests that an activation can be instantiated
TEST_F(TestSHAVE, BasicAssertionsActivationCategory) {
    constexpr unsigned int efficiencyx1K{2000};
    constexpr unsigned int latency{1000};
    auto swwl = VPUNN::SHVActivation<efficiencyx1K, latency>(VPUNN::VPUDevice::VPU_2_7,
                                                             input_0,  // input dimensions
                                                             output_0  // output dimensions
    );
    // init part
    EXPECT_EQ(swwl.device, VPUNN::VPUDevice::VPU_2_7);
    ASSERT_EQ(swwl.inputs.size(), 1);
    EXPECT_EQ(swwl.inputs[0].size(), input_0.size()) << " clearly they are not the same" << std::endl;

    ASSERT_EQ(swwl.outputs.size(), 1);
    EXPECT_EQ(swwl.outputs[0].size(), output_0.size()) << " clearly they are not the same" << std::endl;

    // behavioral part
    EXPECT_NEAR(swwl.getKernelEfficiency(), efficiencyx1K / 1000.0F, 0.001F);
    EXPECT_EQ(swwl.getLatency(), latency);

    const auto shave_cycles_sigmoid = empty_model.SHAVE(swwl);
    const auto op_count = swwl.outputs[0].size();

    EXPECT_NEAR(shave_cycles_sigmoid, round(op_count / swwl.getKernelEfficiency()) + swwl.getLatency(), 1.0F);
}

/// @brief tests that an Element wise can be instantiated
TEST_F(TestSHAVE, BasicAssertionsELMWiseCategory) {
    constexpr unsigned int efficiencyx1K{800};
    constexpr unsigned int latency{1300};
    auto swwl = VPUNN::SHVElementwise<efficiencyx1K, latency>(VPUNN::VPUDevice::VPU_2_7, {input_0},  // input dimensions
                                                              output_0  // output dimensions
    );
    // init part
    EXPECT_EQ(swwl.device, VPUNN::VPUDevice::VPU_2_7);
    ASSERT_EQ(swwl.inputs.size(), 1);
    EXPECT_EQ(swwl.inputs[0].size(), input_0.size()) << " clearly they are not the same" << std::endl;

    ASSERT_EQ(swwl.outputs.size(), 1);
    EXPECT_EQ(swwl.outputs[0].size(), output_0.size()) << " clearly they are not the same" << std::endl;

    // behavioral part
    EXPECT_NEAR(swwl.getKernelEfficiency(), efficiencyx1K / 1000.0F, 0.001F);
    EXPECT_EQ(swwl.getLatency(), latency);

    const auto shave_cycles_sigmoid = empty_model.SHAVE(swwl);
    const auto op_count = swwl.outputs[0].size();

    EXPECT_NEAR(shave_cycles_sigmoid, round(op_count / swwl.getKernelEfficiency()) + swwl.getLatency(), 1.0F);
}

/// @brief tests that an Data Movement can be instantiated
TEST_F(TestSHAVE, BasicAssertionsDataMovementCategory) {
    constexpr unsigned int efficiencyx1K{2050};
    constexpr unsigned int latency{3000};
    auto swwl = VPUNN::SHVDataMovement<efficiencyx1K, latency>(VPUNN::VPUDevice::VPU_2_7,
                                                               input_0,  // input dimensions
                                                               output_0  // output dimensions
    );
    // init part
    EXPECT_EQ(swwl.device, VPUNN::VPUDevice::VPU_2_7);
    ASSERT_EQ(swwl.inputs.size(), 1);
    EXPECT_EQ(swwl.inputs[0].size(), input_0.size()) << " clearly they are not the same" << std::endl;

    ASSERT_EQ(swwl.outputs.size(), 1);
    EXPECT_EQ(swwl.outputs[0].size(), output_0.size()) << " clearly they are not the same" << std::endl;

    // behavioral part
    EXPECT_NEAR(swwl.getKernelEfficiency(), efficiencyx1K / 1000.0F, 0.001F);
    EXPECT_EQ(swwl.getLatency(), latency);

    const auto shave_cycles_sigmoid = empty_model.SHAVE(swwl);
    const auto op_count = swwl.outputs[0].size();

    EXPECT_NEAR(shave_cycles_sigmoid, round(op_count / swwl.getKernelEfficiency()) + swwl.getLatency(), 1.0F);
}

/// @brief tests that an Sigmoid can be instantiated
TEST_F(TestSHAVE, BasicAssertionsActivationSigmoid) {
    auto swwl = VPUNN::SHVSigmoid(VPUNN::VPUDevice::VPU_2_7,
                                  input_0,  // input dimensions
                                  output_0  // output dimensions
    );
    EXPECT_EQ(swwl.device, VPUNN::VPUDevice::VPU_2_7);
    ASSERT_EQ(swwl.inputs.size(), 1);
    EXPECT_EQ(swwl.inputs[0].size(), input_0.size()) << " clearly they are not the same" << std::endl;

    ASSERT_EQ(swwl.outputs.size(), 1);
    EXPECT_EQ(swwl.outputs[0].size(), output_0.size()) << " clearly they are not the same" << std::endl;

    auto shave_cycles_sigmoid = empty_model.SHAVE(swwl);
    // Expect equality.
    EXPECT_GE(shave_cycles_sigmoid, 0u);
}

}  // namespace VPUNN_unit_tests