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

class TestEnergyandPF_CostModelNPU4x : public TestEnergyandPF_CostModel {
public:
protected:
};

class TestVPUPowerFactorLUTNPU4x : public TestVPUPowerFactorLUT {
public:
protected:
    const VPUDevice device{VPUDevice::VPU_4_0};

private:
};

TEST_F(TestVPUPowerFactorLUTNPU4x, NPU40_AvailabilitySmoke) {
    const VPUPowerFactorLUT power_factor_lut{};

    {  // FP16
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
        EXPECT_NEAR(operation_pf, 1.3f, 0.005) << "FP16" << wl;
    }
    {  // fp8
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
        EXPECT_NEAR(operation_pf, 0.0f, 0.005) << "FP8" << wl;
    }
    {  // int8
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
        EXPECT_NEAR(operation_pf, 1.0f, 0.005) << "I8 " << wl;
    }
}

}  // namespace VPUNN_unit_tests