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

class TestEnergyandPF_CostModelNPU5x : public TestEnergyandPF_CostModel {
public:
protected:
};

class TestVPUPowerFactorLUTNPU5x : public TestVPUPowerFactorLUT {
public:
protected:
    const VPUDevice device{VPUDevice::NPU_5_0};

private:
};

TEST_F(TestVPUPowerFactorLUTNPU5x, NPU50_AvailabilitySmoke) {
    const VPUPowerFactorLUT power_factor_lut{};

    {  // FP16
        const float adjfctor{0.88f};
        const std::array<VPUTensor, 1> outputs3200{VPUTensor(10, 10, 32, 1, DataType::FLOAT16)};
        const std::array<VPUTensor, 1> inputs1600{VPUTensor(10, 10, 16, 1, DataType::FLOAT16)};

        DPUWorkload wl{device,
                       Operation::CONVOLUTION,  // 3200X9x16 = 460800 ops
                       inputs1600,
                       outputs3200,
                       kernels,  // 3x3 =9
                       strides,
                       padding,
                       execution_order};

        float operation_pf{0.0f};
        ASSERT_NO_THROW(operation_pf =
                                power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider))
                << "FP16" << wl;
        EXPECT_NEAR(operation_pf, 1.0f * adjfctor, 0.005) << "FP16" << " adjfactor :" << adjfctor << wl;
    }
    {  // fp8
        const float adjfctor{0.88f};
        const std::array<VPUTensor, 1> outputs3200{VPUTensor(10, 10, 32, 1, DataType::BF8)};
        const std::array<VPUTensor, 1> inputs1600{VPUTensor(10, 10, 16, 1, DataType::HF8)};

        DPUWorkload wl{device,
                       Operation::CONVOLUTION,  // 3200X9x16 = 460800 ops
                       inputs1600,
                       outputs3200,
                       kernels,  // 3x3 =9
                       strides,
                       padding,
                       execution_order};

        float operation_pf{0.0f};
        ASSERT_NO_THROW(operation_pf =
                                power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider))
                << "FP8" << wl;
        EXPECT_NEAR(operation_pf, 0.72f * adjfctor, 0.005) << "FP8" << " adjfactor :" << adjfctor << wl;
    }
    {  // int8
        const float adjfctor{0.88f};
        const std::array<VPUTensor, 1> outputs3200{VPUTensor(10, 10, 32, 1, DataType::UINT8)};
        const std::array<VPUTensor, 1> inputs1600{VPUTensor(10, 10, 16, 1, DataType::UINT8)};

        DPUWorkload wl{device,
                       Operation::CONVOLUTION,  // 3200X9x16 = 460800 ops
                       inputs1600,
                       outputs3200,
                       kernels,  // 3x3 =9
                       strides,
                       padding,
                       execution_order};

        float operation_pf{0.0f};
        ASSERT_NO_THROW(operation_pf =
                                power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider))
                << "I8 " << wl;
        EXPECT_NEAR(operation_pf, 0.61f * adjfctor, 0.005) << "I8 " << " adjfactor :" << adjfctor << wl;
    }
}

// ==================================================================================
// Fallback mechanism tests for NPU_5_0
// NPU_5_0 SCL has INT8 + FP8 + FP16 (all types covered, no SCL fallback triggered).
// DCIM engine is not in the LUT -> returns empty (0.0).
// ==================================================================================
TEST_F(TestVPUPowerFactorLUTNPU5x, NPU50_DCIM_INT8_ReturnsEmpty) {
    const VPUPowerFactorLUT pf_lut{};
    auto wl = make_conv_wl(VPUDevice::NPU_5_0, DataType::UINT8, DataType::UINT8, MPEEngine::DCIM);

    float result{-1.0f};
    ASSERT_NO_THROW(result = pf_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider))
            << "DCIM INT8 on NPU_5_0" << wl;
    EXPECT_NEAR(result, 0.0f, 0.005) << "DCIM should return empty (0.0) on NPU_5_0" << wl;
}

TEST_F(TestVPUPowerFactorLUTNPU5x, NPU50_DCIM_FP8_ReturnsEmpty) {
    const VPUPowerFactorLUT pf_lut{};
    auto wl = make_conv_wl(VPUDevice::NPU_5_0, DataType::HF8, DataType::BF8, MPEEngine::DCIM);

    float result{-1.0f};
    ASSERT_NO_THROW(result = pf_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider))
            << "DCIM FP8 on NPU_5_0" << wl;
    EXPECT_NEAR(result, 0.0f, 0.005) << "DCIM should return empty (0.0) on NPU_5_0" << wl;
}

TEST_F(TestVPUPowerFactorLUTNPU5x, NPU50_DCIM_FP16_ReturnsEmpty) {
    const VPUPowerFactorLUT pf_lut{};
    auto wl = make_conv_wl(VPUDevice::NPU_5_0, DataType::FLOAT16, DataType::FLOAT16, MPEEngine::DCIM);

    float result{-1.0f};
    ASSERT_NO_THROW(result = pf_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider))
            << "DCIM FP16 on NPU_5_0" << wl;
    EXPECT_NEAR(result, 0.0f, 0.005) << "DCIM should return empty (0.0) on NPU_5_0" << wl;
}

}  // namespace VPUNN_unit_tests
