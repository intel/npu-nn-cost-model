// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include <gtest/gtest.h>
#include "common_helpers.h"
#include "vpu_cost_model.h"

#include <unordered_map>

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {

/// @brief tests that show backwards compatibility with NN with different versions
class TestNNModelCompatibility : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }
    // const float noninitialized_NN_output{-1.0};  //< what to expect from a uninit NN, and empty NN

    const std::string VPU20_default_{NameHelperNN::get_model_root() +
                                     "../tests/cpp/nn_model_versions/vpu_2_0-default_initial.vpunn"};
    const std::string VPU27_default_{NameHelperNN::get_model_root() +
                                     "../tests/cpp/nn_model_versions/vpu_2_7-default_initial.vpunn"};
    const std::string VPU27_10_2_{NameHelperNN::get_model_root() +
                                  "../tests/cpp/nn_model_versions/vpu_2_7_v-10-2.vpunn"};
    const std::string VPU27_11_2_{NameHelperNN::get_model_root() +
                                  "../tests/cpp/nn_model_versions/vpu_2_7_v-11-2.vpunn"};

    VPUNN::DPUWorkload wl_20{VPUNN::VPUDevice::VPU_2_0,
                             VPUNN::Operation::CONVOLUTION,
                             {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions
                             {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // output dimensions
                             {3, 3},                                                     // kernels
                             {1, 1},                                                     // strides
                             {1, 1, 1, 1},                                               // padding
                             VPUNN::ExecutionMode::MATRIX};

    VPUNN::DPUWorkload wl_27{VPUNN::VPUDevice::VPU_2_7,
                             VPUNN::Operation::CONVOLUTION,
                             {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions
                             {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // output dimensions
                             {3, 3},                                                     // kernels
                             {1, 1},                                                     // strides
                             {1, 1, 1, 1},                                               // padding
                             VPUNN::ExecutionMode::CUBOID_16x16};

    const float epsilon{0.01F};

    VPUNN::VPUCostModel ideal_model = VPUNN::VPUCostModel("");  // model without NN, output is interface 1-hw overhead

private:
};

/// test ideal model, no NN, just theoretical aspects
TEST_F(TestNNModelCompatibility, Ideal_Empty_Model) {
    {
        VPUNN::DPUWorkload wl{wl_20};
        unsigned int theoretical_dpu_cycles = ideal_model.DPU(wl);   // will be computed based o
        float theoretical_hw_util = ideal_model.hw_utilization(wl);  // a float
        EXPECT_NEAR(theoretical_hw_util, 1.0F, epsilon);

        EXPECT_EQ(theoretical_dpu_cycles, 28224);
    }
    {
        VPUNN::DPUWorkload wl{wl_27};
        unsigned int theoretical_dpu_cycles = ideal_model.DPU(wl);   // will be computed based o
        float theoretical_hw_util = ideal_model.hw_utilization(wl);  // a float
        EXPECT_NEAR(theoretical_hw_util, 1.0F, epsilon);

        EXPECT_EQ(theoretical_dpu_cycles, 3528);
    }
}

/// Test VPU2.0 exists, this version is using hw_overhead which is deprecated
TEST_F(TestNNModelCompatibility, VPU20_default_01) {
    EXPECT_THROW(VPUNN::VPUCostModel vpunn_model = VPUNN::VPUCostModel(VPU20_default_), std::runtime_error);

    // making sure that the error came from the output version:
    VPUNN::Runtime vpunn_runtime(VPU20_default_);

    EXPECT_EQ(vpunn_runtime.model_version_info().get_output_interface_version(),
              (int)(VPUNN::NNOutputVersions::OUT_HW_OVERHEAD_BOUNDED))
            << "Expected:" << (int)(VPUNN::NNOutputVersions::OUT_HW_OVERHEAD_BOUNDED);

    EXPECT_FALSE((vpunn_runtime.model_version_info().get_raw_name() == "none"))
            << "An empty/ideal model will return theoretical cycles even if output version is "
               "defaulted to 1";

    // checking error message to contain key words:
    try {
        VPUNN::VPUCostModel vpunn_model = VPUNN::VPUCostModel(VPU20_default_);
    } catch (const std::exception& e) {
        std::string err_message = e.what();
        EXPECT_TRUE((err_message.find("Cannot load/handle Models output version. The output version", 0) !=
                     std::string::npos));
    }
}

/// Test VPU2.7 1 exists, this version is using hw_overhead which is deprecated
TEST_F(TestNNModelCompatibility, VPU27_default_01) {
    EXPECT_THROW(VPUNN::VPUCostModel vpunn_model = VPUNN::VPUCostModel(VPU27_default_), std::runtime_error);

    // making sure that the error came from the output version:
    VPUNN::Runtime vpunn_runtime(VPU27_default_);

    EXPECT_EQ(vpunn_runtime.model_version_info().get_output_interface_version(),
              (int)(VPUNN::NNOutputVersions::OUT_HW_OVERHEAD_BOUNDED))
            << "Expected:" << (int)(VPUNN::NNOutputVersions::OUT_HW_OVERHEAD_BOUNDED);

    EXPECT_FALSE((vpunn_runtime.model_version_info().get_raw_name() == "none"))
            << "An empty/ideal model will return theoretical cycles even if output version is "
               "defaulted to 1";

    // checking error message to contain key words:
    try {
        VPUNN::VPUCostModel vpunn_model = VPUNN::VPUCostModel(VPU27_default_);
    } catch (const std::exception& e) {
        std::string err_message = e.what();
        EXPECT_TRUE((err_message.find("Cannot load/handle Models output version. The output version", 0) !=
                     std::string::npos));
    }
}

/// Test VPU2.7 v 10 exists
TEST_F(TestNNModelCompatibility, VPU_10_2) {
    VPUNN::DPUWorkload wl{wl_27};
    VPUNN::VPUCostModel vpunn_model = VPUNN::VPUCostModel(VPU27_10_2_);
    ASSERT_TRUE(vpunn_model.nn_initialized())
            << "Model not loaded, might be due to file location: " << VPU27_10_2_ << std::endl;

    float nn_out1 = vpunn_model.run_NN(wl);  // raw cycles
    unsigned int dpu_cycles = vpunn_model.DPU(wl);
    float hw_util = vpunn_model.hw_utilization(wl);  // a float

    // this version uses cycles at exit
    EXPECT_GE(nn_out1, 10000.0F);  // is 10023.5 swizzling 0, 10007 swizzling 5
    EXPECT_GE(dpu_cycles, 10000);
    EXPECT_NEAR(dpu_cycles, nn_out1, 1.0F);

    EXPECT_GE(hw_util, 0.35F);

    unsigned int theoretical_dpu_cycles = ideal_model.DPU(wl);  // will be computed based o

    EXPECT_NEAR(dpu_cycles / (theoretical_dpu_cycles / hw_util), 1.0F, epsilon)
            << " info: " << dpu_cycles << " " << theoretical_dpu_cycles << " " << hw_util << std::endl;

    std::cout << "VPU2.7v 10 2, default wl. "
              << "NN output/hw overhead: " << nn_out1 << ", DPU cycles: " << dpu_cycles
              << ", hw_utilization: " << hw_util << ", Theoretical DPU cycles: " << theoretical_dpu_cycles << std::endl;
}

/// Test VPU2.7 v 11 exists
TEST_F(TestNNModelCompatibility, VPU_11_2) {
    VPUNN::DPUWorkload wl{wl_27};
    VPUNN::VPUCostModel vpunn_model = VPUNN::VPUCostModel(VPU27_11_2_);
    ASSERT_TRUE(vpunn_model.nn_initialized())
            << "Model not loaded, might be due to file location: " << VPU27_11_2_ << std::endl;

    float nn_out1 = vpunn_model.run_NN(wl);  // raw cycles
    unsigned int dpu_cycles = vpunn_model.DPU(wl);
    float hw_util = vpunn_model.hw_utilization(wl);  // a float

    // this version uses cycles at exit
    EXPECT_GE(nn_out1, 9450.0F);  // was 9610.96 with padding bad, is 9498
    EXPECT_GE(dpu_cycles, 9450);
    EXPECT_NEAR(dpu_cycles, nn_out1, 1.0F);

    EXPECT_GE(hw_util, 0.37F);

    unsigned int theoretical_dpu_cycles = ideal_model.DPU(wl);  // will be computed based o

    EXPECT_NEAR(dpu_cycles / (theoretical_dpu_cycles / hw_util), 1.0F, epsilon)
            << " info: " << dpu_cycles << " " << theoretical_dpu_cycles << " " << hw_util << std::endl;

    std::cout << "VPU2.7v 11 2, default wl. "
              << "NN output/hw overhead: " << nn_out1 << ", DPU cycles: " << dpu_cycles
              << ", hw_utilization: " << hw_util << ", Theoretical DPU cycles: " << theoretical_dpu_cycles << std::endl;
}

}  // namespace VPUNN_unit_tests
