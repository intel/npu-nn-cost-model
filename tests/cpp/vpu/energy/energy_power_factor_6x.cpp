// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.
#include "energy_power_factor.h"

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {
using namespace VPUNN;

class TestVPUPowerFactorLUTNPU6x : public TestVPUPowerFactorLUT {
public:
protected:
    const VPUDevice device{VPUDevice::NPU_6_0};

private:
};

// SCL Availability Smoke Tests
// NPU_6_0 SCL has INT8 + FP8 + FP16  (adjustor = 1.0)
TEST_F(TestVPUPowerFactorLUTNPU6x, NPU60_AvailabilitySmoke) {
    const VPUPowerFactorLUT power_factor_lut{};

    {  // FP16
        const float adjfctor{1.0f};
        const std::array<VPUTensor, 1> outputs3200{VPUTensor(10, 10, 32, 1, DataType::FLOAT16)};
        const std::array<VPUTensor, 1> inputs1600{VPUTensor(10, 10, 16, 1, DataType::FLOAT16)};

        DPUWorkload wl{device,  Operation::CONVOLUTION, inputs1600, outputs3200, kernels, strides,
                       padding, execution_order};

        float operation_pf{0.0f};
        ASSERT_NO_THROW(operation_pf =
                                power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider))
                << "FP16" << wl;
        EXPECT_NEAR(operation_pf, 1.18f * adjfctor, 0.005) << "FP16" << " adjfactor :" << adjfctor << wl;
    }
    {  // FP8
        const float adjfctor{1.0f};
        const std::array<VPUTensor, 1> outputs3200{VPUTensor(10, 10, 32, 1, DataType::BF8)};
        const std::array<VPUTensor, 1> inputs1600{VPUTensor(10, 10, 16, 1, DataType::HF8)};

        DPUWorkload wl{device,  Operation::CONVOLUTION, inputs1600, outputs3200, kernels, strides,
                       padding, execution_order};

        float operation_pf{0.0f};
        ASSERT_NO_THROW(operation_pf =
                                power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider))
                << "FP8" << wl;
        EXPECT_NEAR(operation_pf, 0.96f * adjfctor, 0.005) << "FP8" << " adjfactor :" << adjfctor << wl;
    }
    {  // INT8
        const float adjfctor{1.0f};
        const std::array<VPUTensor, 1> outputs3200{VPUTensor(10, 10, 32, 1, DataType::UINT8)};
        const std::array<VPUTensor, 1> inputs1600{VPUTensor(10, 10, 16, 1, DataType::UINT8)};

        DPUWorkload wl{device,  Operation::CONVOLUTION, inputs1600, outputs3200, kernels, strides,
                       padding, execution_order};

        float operation_pf{0.0f};
        ASSERT_NO_THROW(operation_pf =
                                power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider))
                << "I8 " << wl;
        EXPECT_NEAR(operation_pf, 0.82f * adjfctor, 0.005) << "I8 " << " adjfactor :" << adjfctor << wl;
    }
}

// REDUCE_MS and REDUCE_SUMSQUARES power factor tests
// These operations are new for NPU 6.0 and have dedicated LUT entries
TEST_F(TestVPUPowerFactorLUTNPU6x, NPU60_REDUCE_MS_INT8) {
    const VPUPowerFactorLUT power_factor_lut{};
    const float adjfctor{1.0f};

    const std::array<VPUTensor, 1> inputs{VPUTensor(10, 10, 64, 1, DataType::UINT8)};
    const std::array<VPUTensor, 1> outputsLocal{VPUTensor(10, 10, 1, 1, DataType::UINT8)};

    DPUWorkload wl{device, Operation::REDUCE_MS, inputs, outputsLocal, {1, 1}, strides, {0, 0, 0, 0}, execution_order};

    float operation_pf{0.0f};
    ASSERT_NO_THROW(operation_pf = power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider))
            << "REDUCE_MS I8" << wl;
    EXPECT_NEAR(operation_pf, 0.82f * adjfctor, 0.005) << "REDUCE_MS I8" << " adjfactor :" << adjfctor << wl;
}

TEST_F(TestVPUPowerFactorLUTNPU6x, NPU60_REDUCE_MS_FP8) {
    const VPUPowerFactorLUT power_factor_lut{};
    const float adjfctor{1.0f};

    const std::array<VPUTensor, 1> inputs{VPUTensor(10, 10, 64, 1, DataType::HF8)};
    const std::array<VPUTensor, 1> outputsLocal{VPUTensor(10, 10, 1, 1, DataType::BF8)};

    DPUWorkload wl{device, Operation::REDUCE_MS, inputs, outputsLocal, {1, 1}, strides, {0, 0, 0, 0}, execution_order};

    float operation_pf{0.0f};
    ASSERT_NO_THROW(operation_pf = power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider))
            << "REDUCE_MS FP8" << wl;
    EXPECT_NEAR(operation_pf, 0.96f * adjfctor, 0.005) << "REDUCE_MS FP8" << " adjfactor :" << adjfctor << wl;
}

TEST_F(TestVPUPowerFactorLUTNPU6x, NPU60_REDUCE_MS_FP16) {
    const VPUPowerFactorLUT power_factor_lut{};
    const float adjfctor{1.0f};

    const std::array<VPUTensor, 1> inputs{VPUTensor(10, 10, 64, 1, DataType::FLOAT16)};
    const std::array<VPUTensor, 1> outputsLocal{VPUTensor(10, 10, 1, 1, DataType::FLOAT16)};

    DPUWorkload wl{device, Operation::REDUCE_MS, inputs, outputsLocal, {1, 1}, strides, {0, 0, 0, 0}, execution_order};

    float operation_pf{0.0f};
    ASSERT_NO_THROW(operation_pf = power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider))
            << "REDUCE_MS FP16" << wl;
    EXPECT_NEAR(operation_pf, 1.18f * adjfctor, 0.005) << "REDUCE_MS FP16" << " adjfactor :" << adjfctor << wl;
}

TEST_F(TestVPUPowerFactorLUTNPU6x, NPU60_REDUCE_SUMSQUARES_INT8) {
    const VPUPowerFactorLUT power_factor_lut{};
    const float adjfctor{1.0f};

    const std::array<VPUTensor, 1> inputs{VPUTensor(10, 10, 64, 1, DataType::UINT8)};
    const std::array<VPUTensor, 1> outputsLocal{VPUTensor(10, 10, 1, 1, DataType::UINT8)};

    DPUWorkload wl{device,         Operation::REDUCE_SUMSQUARES, inputs, outputsLocal, {1, 1}, strides, {0, 0, 0, 0},
                   execution_order};

    float operation_pf{0.0f};
    ASSERT_NO_THROW(operation_pf = power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider))
            << "REDUCE_SUMSQUARES I8" << wl;
    EXPECT_NEAR(operation_pf, 0.0f * adjfctor, 0.005) << "REDUCE_SUMSQUARES I8" << " adjfactor :" << adjfctor << wl;
}

TEST_F(TestVPUPowerFactorLUTNPU6x, NPU60_REDUCE_SUMSQUARES_FP8) {
    const VPUPowerFactorLUT power_factor_lut{};
    const float adjfctor{1.0f};

    const std::array<VPUTensor, 1> inputs{VPUTensor(10, 10, 64, 1, DataType::HF8)};
    const std::array<VPUTensor, 1> outputsLocal{VPUTensor(10, 10, 1, 1, DataType::BF8)};

    DPUWorkload wl{device,         Operation::REDUCE_SUMSQUARES, inputs, outputsLocal, {1, 1}, strides, {0, 0, 0, 0},
                   execution_order};

    float operation_pf{0.0f};
    ASSERT_NO_THROW(operation_pf = power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider))
            << "REDUCE_SUMSQUARES FP8" << wl;
    EXPECT_NEAR(operation_pf, 0.0f * adjfctor, 0.005) << "REDUCE_SUMSQUARES FP8" << " adjfactor :" << adjfctor << wl;
}

TEST_F(TestVPUPowerFactorLUTNPU6x, NPU60_REDUCE_SUMSQUARES_FP16) {
    const VPUPowerFactorLUT power_factor_lut{};
    const float adjfctor{1.0f};

    const std::array<VPUTensor, 1> inputs{VPUTensor(10, 10, 64, 1, DataType::FLOAT16)};
    const std::array<VPUTensor, 1> outputsLocal{VPUTensor(10, 10, 1, 1, DataType::FLOAT16)};

    DPUWorkload wl{device,         Operation::REDUCE_SUMSQUARES, inputs, outputsLocal, {1, 1}, strides, {0, 0, 0, 0},
                   execution_order};

    float operation_pf{0.0f};
    ASSERT_NO_THROW(operation_pf = power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider))
            << "REDUCE_SUMSQUARES FP16" << wl;
    EXPECT_NEAR(operation_pf, 0.0f * adjfctor, 0.005) << "REDUCE_SUMSQUARES FP16" << " adjfactor :" << adjfctor << wl;
}

// DCIM fallback mechanism tests for NPU_6_0
// NPU_6_0 DCIM has INT8 + INT16 for CONVOLUTION only.
// Other types fall back to INT8 within DCIM.
TEST_F(TestVPUPowerFactorLUTNPU6x, NPU60_DCIM_INT8_Convolution) {
    const VPUPowerFactorLUT pf_lut{};
    const float adjfctor{1.0f};
    auto wl = make_conv_wl(VPUDevice::NPU_6_0, DataType::UINT8, DataType::UINT8, MPEEngine::DCIM);

    float result{-1.0f};
    ASSERT_NO_THROW(result = pf_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider))
            << "DCIM INT8 CONV on NPU_6_0" << wl;
    EXPECT_NEAR(result, 0.42f * adjfctor, 0.005)
            << "DCIM INT8 CONV on NPU_6_0" << " adjfactor :" << adjfctor << wl;
}

TEST_F(TestVPUPowerFactorLUTNPU6x, NPU60_DCIM_INT16_Convolution) {
    const VPUPowerFactorLUT pf_lut{};
    const float adjfctor{1.0f};
    const float scaling_factor{0.940f};  // INT16 scaling factor from power_lut_6x.h
    auto wl = make_conv_wl(VPUDevice::NPU_6_0, DataType::UINT16, DataType::UINT16, MPEEngine::DCIM);

    float result{-1.0f};
    ASSERT_NO_THROW(result = pf_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider))
            << "DCIM INT16 CONV on NPU_6_0" << wl;
    // INT16 DCIM LUT value at log2(channels)=6 is 0.42f * scaling_factor
    EXPECT_NEAR(result, 0.42f * scaling_factor * adjfctor, 0.005)
            << "DCIM INT16 CONV on NPU_6_0" << " adjfactor :" << adjfctor << " scaling_factor :" << scaling_factor << wl;
}

TEST_F(TestVPUPowerFactorLUTNPU6x, NPU60_DCIM_FP8_FallbackToINT8) {
    const VPUPowerFactorLUT pf_lut{};
    const float adjfctor{1.0f};
    auto wl = make_conv_wl(VPUDevice::NPU_6_0, DataType::HF8, DataType::BF8, MPEEngine::DCIM);

    float result{-1.0f};
    ASSERT_NO_THROW(result = pf_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider))
            << "DCIM FP8 fallback to INT8 on NPU_6_0" << wl;
    // FP8 not in DCIM LUT -> fallback to INT8 DCIM
    EXPECT_NEAR(result, 0.42f * adjfctor, 0.005)
            << "DCIM FP8 should fallback to INT8 on NPU_6_0" << " adjfactor :" << adjfctor << wl;
}

TEST_F(TestVPUPowerFactorLUTNPU6x, NPU60_DCIM_FP16_FallbackToINT8) {
    const VPUPowerFactorLUT pf_lut{};
    const float adjfctor{1.0f};
    auto wl = make_conv_wl(VPUDevice::NPU_6_0, DataType::FLOAT16, DataType::FLOAT16, MPEEngine::DCIM);

    float result{-1.0f};
    ASSERT_NO_THROW(result = pf_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider))
            << "DCIM FP16 fallback to INT8 on NPU_6_0" << wl;
    // FP16 not in DCIM LUT -> fallback to INT8 DCIM
    EXPECT_NEAR(result, 0.42f * adjfctor, 0.005)
            << "DCIM FP16 should fallback to INT8 on NPU_6_0" << " adjfactor :" << adjfctor << wl;
}

// SCL operations other than CONVOLUTION
// Verify power factor values for various operations on NPU_6_0
TEST_F(TestVPUPowerFactorLUTNPU6x, NPU60_SCL_Operations_INT8) {
    const VPUPowerFactorLUT power_factor_lut{};
    const float adjfctor{1.0f};

    struct OpExpected {
        Operation op;
        float expected_pf;
        std::string name;
    };

    // INT8 - LUT entries at log2(channels)=6 for most ops, log2(channels)=8 for ELTWISE/ELTWISE_MUL/LAYER_NORM
    const std::vector<OpExpected> ops_int8{
            {Operation::DW_CONVOLUTION, 12.41f, "DW_CONV"},
            {Operation::AVEPOOL, 5.12f, "AVEPOOL"},
            {Operation::MAXPOOL, 4.75f, "MAXPOOL"},
    };

    for (const auto& op_test : ops_int8) {
        const std::array<VPUTensor, 1> inputs{VPUTensor(10, 10, 64, 1, DataType::UINT8)};
        const std::array<VPUTensor, 1> outs{VPUTensor(10, 10, 64, 1, DataType::UINT8)};

        DPUWorkload wl{device, op_test.op, inputs, outs, kernels, strides, padding, execution_order};

        float operation_pf{0.0f};
        ASSERT_NO_THROW(operation_pf =
                                power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider))
                << op_test.name << " I8 " << wl;
        EXPECT_NEAR(operation_pf, op_test.expected_pf * adjfctor, 0.005)
                << op_test.name << " I8 " << " adjfactor :" << adjfctor << wl;
    }
}

TEST_F(TestVPUPowerFactorLUTNPU6x, NPU60_SCL_ELTWISE_INT8) {
    const VPUPowerFactorLUT power_factor_lut{};
    const float adjfctor{1.0f};

    // ELTWISE LUT entry is at log2(channels)=8, i.e. 256 channels
    const std::array<VPUTensor, 1> inputs{VPUTensor(10, 10, 256, 1, DataType::UINT8)};
    const std::array<VPUTensor, 1> outs{VPUTensor(10, 10, 256, 1, DataType::UINT8)};

    DPUWorkload wl{device, Operation::ELTWISE, inputs, outs, {1, 1}, strides, {0, 0, 0, 0}, execution_order};

    float operation_pf{0.0f};
    ASSERT_NO_THROW(operation_pf = power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider))
            << "ELTWISE I8 " << wl;
    EXPECT_NEAR(operation_pf, 38.69f * adjfctor, 0.005) << "ELTWISE I8 " << " adjfactor :" << adjfctor << wl;
}

TEST_F(TestVPUPowerFactorLUTNPU6x, NPU60_SCL_ELTWISE_MUL_INT8) {
    const VPUPowerFactorLUT power_factor_lut{};
    const float adjfctor{1.0f};

    // ELTWISE_MUL LUT entry is at log2(channels)=8, i.e. 256 channels
    const std::array<VPUTensor, 1> inputs{VPUTensor(10, 10, 256, 1, DataType::UINT8)};
    const std::array<VPUTensor, 1> outs{VPUTensor(10, 10, 256, 1, DataType::UINT8)};

    DPUWorkload wl{device, Operation::ELTWISE_MUL, inputs, outs, {1, 1}, strides, {0, 0, 0, 0}, execution_order};

    float operation_pf{0.0f};
    ASSERT_NO_THROW(operation_pf = power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider))
            << "ELTWISE_MUL I8 " << wl;
    EXPECT_NEAR(operation_pf, 39.52f * adjfctor, 0.005) << "ELTWISE_MUL I8 " << " adjfactor :" << adjfctor << wl;
}

}  // namespace VPUNN_unit_tests
