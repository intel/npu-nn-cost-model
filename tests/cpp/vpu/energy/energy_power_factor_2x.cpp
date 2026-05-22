// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include "energy_power_factor.h"

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {
using namespace VPUNN;

class TestEnergyandPF_CostModelVPU2x : public TestEnergyandPF_CostModel {
public:
protected:
    DPUWorkload wl_conv{VPUDevice::VPU_2_7,
                        Operation::CONVOLUTION,
                        {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // input dimensions
                        {VPUTensor(56, 56, 32, 1, DataType::UINT8)},  // output dimensions
                        {3, 3},                                       // kernels
                        {1, 1},                                       // strides
                        {1, 1, 1, 1},                                 // padding
                        VPUNN::ExecutionMode::CUBOID_16x16};
    DPUWorkload wl_convCM{VPUDevice::VPU_2_7,
                          Operation::CM_CONVOLUTION,
                          {VPUTensor(56, 56, 15, 1, DataType::UINT8)},  // input dimensions
                          {VPUTensor(56, 56, 32, 1, DataType::UINT8)},  // output dimensions
                          {3, 3},                                       // kernels
                          {1, 1},                                       // strides
                          {1, 1, 1, 1},                                 // padding
                          VPUNN::ExecutionMode::CUBOID_16x16};
    DPUWorkload wl_MAXP{VPUDevice::VPU_2_7,
                        Operation::MAXPOOL,
                        {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // input dimensions
                        {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // output dimensions
                        {3, 3},                                       // kernels
                        {1, 1},                                       // strides
                        {1, 1, 1, 1},                                 // padding
                        VPUNN::ExecutionMode::CUBOID_16x16};
    DPUWorkload wl_AVGP{VPUDevice::VPU_2_7,
                        Operation::AVEPOOL,
                        {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // input dimensions
                        {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // output dimensions
                        {3, 3},                                       // kernels
                        {1, 1},                                       // strides
                        {1, 1, 1, 1},                                 // padding
                        VPUNN::ExecutionMode::CUBOID_16x16};
    DPUWorkload wl_DW_conv{VPUDevice::VPU_2_7,
                           Operation::DW_CONVOLUTION,
                           {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // input dimensions
                           {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // output dimensions
                           {3, 3},                                       // kernels
                           {1, 1},                                       // strides
                           {1, 1, 1, 1},                                 // padding
                           VPUNN::ExecutionMode::CUBOID_16x16};

    DPUWorkload wl_ELT{VPUDevice::VPU_2_7,
                       Operation::ELTWISE,
                       {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // input dimensions
                       {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // output dimensions
                       {1, 1},                                       // kernels
                       {1, 1},                                       // strides
                       {0, 0, 0, 0},                                 // padding
                       VPUNN::ExecutionMode::CUBOID_16x16};
    const std::vector<DPUWorkload> wl_list{wl_conv, wl_convCM, wl_MAXP, wl_AVGP, wl_DW_conv, wl_ELT};

    DPUWorkload wl_conv_FP{VPUDevice::VPU_2_7,
                           Operation::CONVOLUTION,
                           {VPUTensor(56, 56, 16, 1, DataType::FLOAT16)},  // input dimensions
                           {VPUTensor(56, 56, 32, 1, DataType::FLOAT16)},  // output dimensions
                           {3, 3},                                         // kernels
                           {1, 1},                                         // strides
                           {1, 1, 1, 1},                                   // padding
                           VPUNN::ExecutionMode::CUBOID_16x16};
    DPUWorkload wl_convCM_FP{VPUDevice::VPU_2_7,
                             Operation::CM_CONVOLUTION,
                             {VPUTensor(56, 56, 15, 1, DataType::FLOAT16)},  // input dimensions
                             {VPUTensor(56, 56, 32, 1, DataType::FLOAT16)},  // output dimensions
                             {3, 3},                                         // kernels
                             {1, 1},                                         // strides
                             {1, 1, 1, 1},                                   // padding
                             VPUNN::ExecutionMode::CUBOID_16x16};
    DPUWorkload wl_MAXP_FP{VPUDevice::VPU_2_7,
                           Operation::MAXPOOL,
                           {VPUTensor(56, 56, 16, 1, DataType::FLOAT16)},  // input dimensions
                           {VPUTensor(56, 56, 16, 1, DataType::FLOAT16)},  // output dimensions
                           {3, 3},                                         // kernels
                           {1, 1},                                         // strides
                           {1, 1, 1, 1},                                   // padding
                           VPUNN::ExecutionMode::CUBOID_16x16};
    DPUWorkload wl_AVGP_FP{VPUDevice::VPU_2_7,
                           Operation::AVEPOOL,
                           {VPUTensor(56, 56, 16, 1, DataType::FLOAT16)},  // input dimensions
                           {VPUTensor(56, 56, 16, 1, DataType::FLOAT16)},  // output dimensions
                           {3, 3},                                         // kernels
                           {1, 1},                                         // strides
                           {1, 1, 1, 1},                                   // padding
                           VPUNN::ExecutionMode::CUBOID_16x16};
    DPUWorkload wl_DW_conv_FP{VPUDevice::VPU_2_7,
                              Operation::DW_CONVOLUTION,
                              {VPUTensor(56, 56, 16, 1, DataType::FLOAT16)},  // input dimensions
                              {VPUTensor(56, 56, 16, 1, DataType::FLOAT16)},  // output dimensions
                              {3, 3},                                         // kernels
                              {1, 1},                                         // strides
                              {1, 1, 1, 1},                                   // padding
                              VPUNN::ExecutionMode::CUBOID_16x16};

    DPUWorkload wl_ELT_FP{VPUDevice::VPU_2_7,
                          Operation::ELTWISE,
                          {VPUTensor(56, 56, 16, 1, DataType::FLOAT16)},  // input dimensions
                          {VPUTensor(56, 56, 16, 1, DataType::FLOAT16)},  // output dimensions
                          {1, 1},                                         // kernels
                          {1, 1},                                         // strides
                          {0, 0, 0, 0},                                   // padding
                          VPUNN::ExecutionMode::CUBOID_16x16};
    const std::vector<DPUWorkload> wl_list_FP{wl_conv_FP, wl_convCM_FP,  wl_MAXP_FP,
                                              wl_AVGP_FP, wl_DW_conv_FP, wl_ELT_FP};

    const float w_sparsity_level{0.69f};                                //< to be used for lists
    std::vector<DPUWorkload> wl_list_sparse{wl_conv, wl_ELT};           // only supported
    std::vector<DPUWorkload> wl_list_FP_sparse{wl_conv_FP, wl_ELT_FP};  // only supported

    TestEnergyandPF_CostModelVPU2x() {
        auto transformer = [this](DPUWorkload& c)  // modify in-place
        {
            c.weight_sparsity = w_sparsity_level;
            c.weight_sparsity_enabled = true;
        };

        std::for_each(wl_list_sparse.begin(), wl_list_sparse.end(), transformer);
        std::for_each(wl_list_FP_sparse.begin(), wl_list_FP_sparse.end(), transformer);
    }

private:
};

class TestVPUPowerFactorLUTVPU2x : public TestVPUPowerFactorLUT {
public:
protected:
    const VPUDevice defaultDevice{VPUDevice::VPU_2_0};
    const float refPowerVirusFactor{
            0.87f /*VPUPowerFactorLUT().getFP_overI8_maxPower_ratio(VPUDevice::VPU_2_0) */ /* 0.87f*/};  // to be put by
private:
};

TEST_F(TestEnergyandPF_CostModelVPU2x, BasicEnergy_INT8) {
    VPUCostModel crt_model{VPU_2_7_MODEL_PATH};
    {
        DPUWorkload wl{wl_conv};
        basicTest(wl, crt_model);
    }

    for (const auto& wl : wl_list) {
        basicTest(wl, crt_model, "All int8:");
    }
}
TEST_F(TestEnergyandPF_CostModelVPU2x, BasicEnergy_FP16) {
    VPUCostModel crt_model{VPU_2_7_MODEL_PATH};

    for (const auto& wl : wl_list_FP) {
        basicTest(wl, crt_model, "All FP16:");
    }
}

TEST_F(TestEnergyandPF_CostModelVPU2x, DPUInfoBasics) {
    VPUCostModel crt_model{VPU_2_7_MODEL_PATH};

    for (const auto& wl : wl_list) {
        basicDPUPackEquivalenceTest(wl, crt_model, "All int8:");
    }
    for (const auto& wl : wl_list_FP) {
        basicDPUPackEquivalenceTest(wl, crt_model, "All FP16:");
    }
}
TEST_F(TestEnergyandPF_CostModelVPU2x, DPUInfoBasicsSparse) {
    VPUCostModel crt_model{VPU_2_7_MODEL_PATH};

    for (const auto& wl : wl_list_sparse) {
        basicDPUPackEquivalenceTest(wl, crt_model, "All int8 sparse:");
    }
    for (const auto& wl : wl_list_FP_sparse) {
        basicDPUPackEquivalenceTest(wl, crt_model, "All FP16 sparse:");
    }
}

TEST_F(TestEnergyandPF_CostModelVPU2x, SparseEnergy_INT8) {
    VPUCostModel crt_model{VPU_2_7_MODEL_PATH};

    for (const auto& wl : wl_list_sparse) {
        basicSparseTest(wl, crt_model, "All int8 sparse:");
    }
}
TEST_F(TestEnergyandPF_CostModelVPU2x, SparseEnergy_FP16) {
    VPUCostModel crt_model{VPU_2_7_MODEL_PATH};

    for (const auto& wl : wl_list_FP_sparse) {
        basicSparseTest(wl, crt_model, "All FP16 sparse:");
    }
}

TEST_F(TestVPUPowerFactorLUTVPU2x, InsideMatchSamples) {
    const VPUPowerFactorLUT power_factor_lut;

    {
        DPUWorkload wl{defaultDevice,
                       Operation::CONVOLUTION,
                       {VPUTensor(56, 56, (unsigned int)std::pow(2, 5), 1, defaultTensorType)},
                       outputs,
                       kernels,
                       strides,
                       padding,
                       execution_order};

        float operation_pf{0.0f};
        ASSERT_NO_THROW(operation_pf =
                                power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider))
                << wl;
        EXPECT_NEAR(operation_pf, 0.92F * refPowerVirusFactor, 0.005) << wl;
    }
    {
        DPUWorkload wl{defaultDevice,
                       Operation::CONVOLUTION,
                       {VPUTensor(56, 56, (unsigned int)std::pow(2, 7), 1, defaultTensorType)},
                       outputs,
                       kernels,
                       strides,
                       padding,
                       execution_order};

        float operation_pf{0.0f};
        ASSERT_NO_THROW(operation_pf =
                                power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider))
                << wl;
        EXPECT_NEAR(operation_pf, 0.95F * refPowerVirusFactor, 0.005) << wl;
    }
    {
        DPUWorkload wl{defaultDevice,
                       Operation::CONVOLUTION,
                       {VPUTensor(56, 56, (unsigned int)std::pow(2, 9), 1, defaultTensorType)},
                       outputs,
                       kernels,
                       strides,
                       padding,
                       execution_order};

        float operation_pf{0.0f};
        ASSERT_NO_THROW(operation_pf =
                                power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider))
                << wl;
        EXPECT_NEAR(operation_pf, 0.87F * refPowerVirusFactor, 0.005) << wl;
    }

    // inside intemediary
    {
        DPUWorkload wl{defaultDevice,
                       Operation::CONVOLUTION,
                       {VPUTensor(56, 56, (unsigned int)std::pow(2, 7.5), 1, defaultTensorType)},
                       outputs,
                       kernels,
                       strides,
                       padding,
                       execution_order};

        float operation_pf{0.0f};
        ASSERT_NO_THROW(operation_pf =
                                power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider))
                << wl;
        EXPECT_NEAR(operation_pf, ((0.95F + 0.86F) / 2) * refPowerVirusFactor, 0.001) << wl;
    }

    // inside intemediary
    {
        DPUWorkload wl{defaultDevice,
                       Operation::CONVOLUTION,
                       {VPUTensor(56, 56, (unsigned int)std::pow(2, 6.333), 1, defaultTensorType)},
                       outputs,
                       kernels,
                       strides,
                       padding,
                       execution_order};

        float operation_pf{0.0f};
        ASSERT_NO_THROW(operation_pf =
                                power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider))
                << wl;
        EXPECT_NEAR(operation_pf, (1.0F + 0.333F * (0.95F - 1.0F)) * refPowerVirusFactor, 0.001) << wl;
    }
}
TEST_F(TestVPUPowerFactorLUTVPU2x, BeforeFirstSample) {
    const VPUPowerFactorLUT power_factor_lut;

    {
        DPUWorkload wl{defaultDevice,
                       Operation::CONVOLUTION,
                       {VPUTensor(56, 56, (unsigned int)std::pow(2, 0), 1, defaultTensorType)},
                       outputs,
                       kernels,
                       strides,
                       padding,
                       execution_order};

        float operation_pf{0.0f};
        ASSERT_NO_THROW(operation_pf =
                                power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider))
                << wl;
        EXPECT_NEAR(operation_pf, 0.87F * refPowerVirusFactor, 0.001) << wl;
    }
    {
        DPUWorkload wl{defaultDevice,
                       Operation::CONVOLUTION,
                       {VPUTensor(56, 56, (unsigned int)std::pow(2, 1), 1, defaultTensorType)},
                       outputs,
                       kernels,
                       strides,
                       padding,
                       execution_order};

        float operation_pf{0.0f};
        ASSERT_NO_THROW(operation_pf =
                                power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider))
                << wl;
        EXPECT_NEAR(operation_pf, 0.87F * refPowerVirusFactor, 0.001) << wl;
    }
    // just before it
    {
        DPUWorkload wl{defaultDevice,
                       Operation::CONVOLUTION,
                       {VPUTensor(56, 56, 15 /*(unsigned int)std::pow(2, 3.8F)*/, 1, defaultTensorType)},
                       outputs,
                       kernels,
                       strides,
                       padding,
                       execution_order};

        float operation_pf{0.0f};
        ASSERT_NO_THROW(operation_pf =
                                power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider))
                << wl;
        EXPECT_NEAR(operation_pf, 0.87F * refPowerVirusFactor, 0.001) << wl;
    }
    // exactly at first
    {
        DPUWorkload wl{defaultDevice,
                       Operation::CONVOLUTION,
                       {VPUTensor(56, 56, (unsigned int)std::pow(2, 4), 1, defaultTensorType)},
                       outputs,
                       kernels,
                       strides,
                       padding,
                       execution_order};

        float operation_pf{0.0f};
        ASSERT_NO_THROW(operation_pf =
                                power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider))
                << wl;
        EXPECT_NEAR(operation_pf, 0.87F * refPowerVirusFactor, 0.001) << wl;
    }
}
TEST_F(TestVPUPowerFactorLUTVPU2x, AfterLastSample) {
    const VPUPowerFactorLUT power_factor_lut;

    {  // last one
        DPUWorkload wl{defaultDevice,
                       Operation::CONVOLUTION,
                       {VPUTensor(56, 56, (unsigned int)std::pow(2, 9), 1, defaultTensorType)},
                       outputs,
                       kernels,
                       strides,
                       padding,
                       execution_order};

        float operation_pf{0.0f};
        ASSERT_NO_THROW(operation_pf =
                                power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider))
                << wl;
        EXPECT_NEAR(operation_pf, 0.87F * refPowerVirusFactor, 0.001) << wl;
    }
    {
        DPUWorkload wl{defaultDevice,
                       Operation::CONVOLUTION,
                       {VPUTensor(56, 56, (unsigned int)std::pow(2, 9.9), 1, defaultTensorType)},
                       outputs,
                       kernels,
                       strides,
                       padding,
                       execution_order};

        float operation_pf{0.0f};
        ASSERT_NO_THROW(operation_pf =
                                power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider))
                << wl;
        EXPECT_NEAR(operation_pf, 0.87F * refPowerVirusFactor, 0.001) << wl;
    }
    {
        DPUWorkload wl{defaultDevice,
                       Operation::CONVOLUTION,
                       {VPUTensor(56, 56, (unsigned int)std::pow(2, 11), 1, defaultTensorType)},
                       outputs,
                       kernels,
                       strides,
                       padding,
                       execution_order};

        float operation_pf{0.0f};
        ASSERT_NO_THROW(operation_pf =
                                power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider))
                << wl;
        EXPECT_NEAR(operation_pf, 0.87F * refPowerVirusFactor, 0.001) << wl;
    }
}

// ==================================================================================
// Fallback mechanism tests for VPU_2_0 and VPU_2_7
// SCL has INT8 + FP16. No FP8 entry -> FP8 falls back to INT8.
// DCIM engine is not in the LUT -> returns empty (0.0).
// ==================================================================================
TEST_F(TestVPUPowerFactorLUTVPU2x, VPU20_SCL_MissingFP8_FallbackToINT8) {
    const VPUPowerFactorLUT pf_lut{};
    auto wl = make_conv_wl(VPUDevice::VPU_2_0, DataType::HF8, DataType::BF8);

    float result{0.0f};
    ASSERT_NO_THROW(result = pf_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider))
            << "SCL FP8 fallback to INT8 on VPU_2_0" << wl;
    EXPECT_NEAR(result, 1.0f, 0.005) << "SCL FP8 should fallback to INT8 (1.0) on VPU_2_0" << wl;
}

TEST_F(TestVPUPowerFactorLUTVPU2x, VPU20_DCIM_INT8_ReturnsEmpty) {
    const VPUPowerFactorLUT pf_lut{};
    auto wl = make_conv_wl(VPUDevice::VPU_2_0, DataType::UINT8, DataType::UINT8, MPEEngine::DCIM);

    float result{-1.0f};
    ASSERT_NO_THROW(result = pf_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider))
            << "DCIM INT8 on VPU_2_0" << wl;
    EXPECT_NEAR(result, 0.0f, 0.005) << "DCIM should return empty (0.0) on VPU_2_0" << wl;
}

TEST_F(TestVPUPowerFactorLUTVPU2x, VPU20_DCIM_FP16_ReturnsEmpty) {
    const VPUPowerFactorLUT pf_lut{};
    auto wl = make_conv_wl(VPUDevice::VPU_2_0, DataType::FLOAT16, DataType::FLOAT16, MPEEngine::DCIM);

    float result{-1.0f};
    ASSERT_NO_THROW(result = pf_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider))
            << "DCIM FP16 on VPU_2_0" << wl;
    EXPECT_NEAR(result, 0.0f, 0.005) << "DCIM should return empty (0.0) on VPU_2_0" << wl;
}

TEST_F(TestVPUPowerFactorLUTVPU2x, VPU27_SCL_MissingFP8_FallbackToINT8) {
    const VPUPowerFactorLUT pf_lut{};
    auto wl = make_conv_wl(VPUDevice::VPU_2_7, DataType::HF8, DataType::BF8);

    float result{0.0f};
    ASSERT_NO_THROW(result = pf_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider))
            << "SCL FP8 fallback to INT8 on VPU_2_7" << wl;
    EXPECT_NEAR(result, 1.0f, 0.005) << "SCL FP8 should fallback to INT8 (1.0) on VPU_2_7" << wl;
}

TEST_F(TestVPUPowerFactorLUTVPU2x, VPU27_DCIM_INT8_ReturnsEmpty) {
    const VPUPowerFactorLUT pf_lut{};
    auto wl = make_conv_wl(VPUDevice::VPU_2_7, DataType::UINT8, DataType::UINT8, MPEEngine::DCIM);

    float result{-1.0f};
    ASSERT_NO_THROW(result = pf_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider))
            << "DCIM INT8 on VPU_2_7" << wl;
    EXPECT_NEAR(result, 0.0f, 0.005) << "DCIM should return empty (0.0) on VPU_2_7" << wl;
}

TEST_F(TestVPUPowerFactorLUTVPU2x, VPU27_DCIM_FP16_ReturnsEmpty) {
    const VPUPowerFactorLUT pf_lut{};
    auto wl = make_conv_wl(VPUDevice::VPU_2_7, DataType::FLOAT16, DataType::FLOAT16, MPEEngine::DCIM);

    float result{-1.0f};
    ASSERT_NO_THROW(result = pf_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider))
            << "DCIM FP16 on VPU_2_7" << wl;
    EXPECT_NEAR(result, 0.0f, 0.005) << "DCIM should return empty (0.0) on VPU_2_7" << wl;
}

}  // namespace VPUNN_unit_tests