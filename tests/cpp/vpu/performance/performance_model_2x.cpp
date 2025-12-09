// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "performance_model.h"  // added to test get_bandwidth_cycles_per_bytesLegacy

namespace VPUNN_unit_tests {
using namespace VPUNN;

/// @brief tests for performance.h
class TestHWPerformanceModelVPU2x : public TestHWPerformanceModel_BASICS {
public:
};

TEST_F(TestHWPerformanceModelVPU2x, ArchTest2_0_BasicAssertions_Legacy) {
    const auto device = VPUDevice::VPU_2_0;
    const auto& hw{hw_info_legacy.device(device)};

    EXPECT_EQ(hw.input_channels_mac(), 1u);
    EXPECT_EQ(hw.get_nr_ppe(), 16u);
    EXPECT_EQ(hw.get_nr_macs(), 256u);
    EXPECT_EQ(hw.get_dpu_fclk(), 700u);
    EXPECT_FLOAT_EQ(dma.get_bandwidth_cycles_per_bytesLegacy(hw, VPUTensor({56, 56, 64, 1}, DataType::UINT8),
                                                             MemoryLocation::DRAM),
                    700.0f / 20000.0f);
}

TEST_F(TestHWPerformanceModelVPU2x, ArchTest2_0_BasicAssertions_Default) {
    const auto device = VPUDevice::VPU_2_0;
    const auto& hw{hw_info.device(device)};

    EXPECT_EQ(hw.input_channels_mac(), 1u);
    EXPECT_EQ(hw.get_nr_ppe(), 16u);
    EXPECT_EQ(hw.get_nr_macs(), 256u);
    EXPECT_EQ(hw.get_dpu_fclk(), 700u);
    EXPECT_FLOAT_EQ(dma.get_bandwidth_cycles_per_bytesLegacy(hw, VPUTensor({56, 56, 64, 1}, DataType::UINT8),
                                                             MemoryLocation::DRAM),
                    700.0f / 20000.0f);
}

TEST_F(TestHWPerformanceModelVPU2x, ArchTest2_1_BasicAssertions_Legacy) {
    const auto device = VPUDevice::VPU_2_1;
    const auto& hw{hw_info_legacy.device(device)};

    EXPECT_EQ(hw.input_channels_mac(), 1u);
    EXPECT_EQ(hw.get_nr_ppe(), 16u);
    EXPECT_EQ(hw.get_nr_macs(), 256u);
    EXPECT_EQ(hw.get_dpu_fclk(), 850u);
    EXPECT_FLOAT_EQ(dma.get_bandwidth_cycles_per_bytesLegacy(hw, VPUTensor({56, 56, 64, 1}, DataType::UINT8),
                                                             MemoryLocation::DRAM),
                    850.0f / 20000.0f);
}
TEST_F(TestHWPerformanceModelVPU2x, ArchTest2_1_BasicAssertions_Default) {
    const auto device = VPUDevice::VPU_2_1;
    const auto& hw{hw_info.device(device)};

    EXPECT_EQ(hw.input_channels_mac(), 1u);
    EXPECT_EQ(hw.get_nr_ppe(), 16u);
    EXPECT_EQ(hw.get_nr_macs(), 256u);
    EXPECT_EQ(hw.get_dpu_fclk(), 850u);
    EXPECT_FLOAT_EQ(dma.get_bandwidth_cycles_per_bytesLegacy(hw, VPUTensor({56, 56, 64, 1}, DataType::UINT8),
                                                             MemoryLocation::DRAM),
                    850.0f / 20000.0f);
}

TEST_F(TestHWPerformanceModelVPU2x, ArchTest2_7_BasicAssertions_Legacy) {
    const auto device = VPUDevice::VPU_2_7;
    const auto& hw{hw_info_legacy.device(device)};

    EXPECT_EQ(hw.input_channels_mac(), 8u);
    EXPECT_EQ(hw.get_nr_ppe(), 64u);
    EXPECT_EQ(hw.get_nr_macs(), 2048u);
    EXPECT_EQ(hw.get_dpu_fclk(), 1300u);
    EXPECT_FLOAT_EQ(dma.get_bandwidth_cycles_per_bytesLegacy(hw, VPUTensor({56, 56, 64, 1}, DataType::UINT8),
                                                             MemoryLocation::DRAM),
                    1300.0f / 27000.0f);
}
TEST_F(TestHWPerformanceModelVPU2x, ArchTest2_7_BasicAssertions_Default) {
    const auto device = VPUDevice::VPU_2_7;
    const auto& hw{hw_info.device(device)};

    EXPECT_EQ(hw.input_channels_mac(), 8u);
    EXPECT_EQ(hw.get_nr_ppe(), 64u);
    EXPECT_EQ(hw.get_nr_macs(), 2048u);
    EXPECT_EQ(hw.get_dpu_fclk(), 1300u);
    EXPECT_FLOAT_EQ(dma.get_bandwidth_cycles_per_bytesLegacy(hw, VPUTensor({56, 56, 64, 1}, DataType::UINT8),
                                                             MemoryLocation::DRAM),
                    1300.0f / 27000.0f);
}

TEST_F(TestHWPerformanceModel_BASICS, BITC2_7_BasicAssertions_Legacy) {
    const auto device = VPUDevice::VPU_2_7;
    const auto& hw{hw_info_legacy.device(device)};

    const auto tensor = VPUTensor({56, 56, 64, 1}, DataType::UINT8);
    // BITC enabled takes has a smaller cycles/byte bw
    EXPECT_FLOAT_EQ(dma.get_bandwidth_cycles_per_bytesLegacy(hw, tensor, MemoryLocation::CMX, false),
                    2 * dma.get_bandwidth_cycles_per_bytesLegacy(hw, tensor, MemoryLocation::CMX, true));
}

TEST_F(TestHWPerformanceModel_BASICS, BITC2_7_BasicAssertions_default) {
    const auto device = VPUDevice::VPU_2_7;
    const auto& hw{hw_info.device(device)};

    const auto tensor = VPUTensor({56, 56, 64, 1}, DataType::UINT8);
    // BITC enabled takes has a smaller cycles/byte bw
    EXPECT_FLOAT_EQ(dma.get_bandwidth_cycles_per_bytesLegacy(hw, tensor, MemoryLocation::CMX, false),
                    2 * dma.get_bandwidth_cycles_per_bytesLegacy(hw, tensor, MemoryLocation::CMX, true));
}

TEST_F(TestHWPerformanceModel_BASICS, Permute_2_7_BasicAssertions_Legacy) {
    const auto device = VPUDevice::VPU_2_7;
    const auto& hw = hw_info_legacy.device(device);

    const auto tensor_fp16 = VPUTensor({56, 56, 64, 1}, DataType::FLOAT16);
    const auto tensor_uint8 = VPUTensor({56, 56, 64, 1}, DataType::UINT8);
    // BITC enabled takes has a smaller cycles/byte bw
    EXPECT_FLOAT_EQ(dma.get_bandwidth_cycles_per_bytesLegacy(hw, tensor_fp16, MemoryLocation::CMX, false, true),
                    0.5f * 1300.0f / 975.0f);
    EXPECT_FLOAT_EQ(dma.get_bandwidth_cycles_per_bytesLegacy(hw, tensor_uint8, MemoryLocation::CMX, false, true),
                    1.0f * 1300.0f / 975.0f);
}

TEST_F(TestHWPerformanceModel_BASICS, Permute_2_7_BasicAssertions_default) {
    const auto device = VPUDevice::VPU_2_7;
    const auto& hw{hw_info.device(device)};

    const auto tensor_fp16 = VPUTensor({56, 56, 64, 1}, DataType::FLOAT16);
    const auto tensor_uint8 = VPUTensor({56, 56, 64, 1}, DataType::UINT8);
    // BITC enabled takes has a smaller cycles/byte bw
    EXPECT_FLOAT_EQ(dma.get_bandwidth_cycles_per_bytesLegacy(hw, tensor_fp16, MemoryLocation::CMX, false, true),
                    0.5f * 1300.0f / 975.0f);
    EXPECT_FLOAT_EQ(dma.get_bandwidth_cycles_per_bytesLegacy(hw, tensor_uint8, MemoryLocation::CMX, false, true),
                    1.0f * 1300.0f / 975.0f);
}

}  // namespace VPUNN_unit_tests