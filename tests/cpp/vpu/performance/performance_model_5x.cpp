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
class TestHWPerformanceModelNPU5x : public TestHWPerformanceModel_BASICS {
public:
};

TEST_F(TestHWPerformanceModelNPU5x, ArchTest5_0_BasicAssertions_Legacy) {
    const auto device = VPUDevice::NPU_5_0;
    const auto& hw{hw_info_legacy.device(device)};

    EXPECT_EQ(hw.input_channels_mac(), 8u);
    EXPECT_EQ(hw.get_nr_ppe(), 32u);
    EXPECT_EQ(hw.get_nr_macs(), 4096u);
    EXPECT_EQ(hw.get_dpu_fclk(), 1950u);
    EXPECT_FLOAT_EQ(dma.get_bandwidth_cycles_per_bytesLegacy(hw, VPUTensor({56, 56, 64, 1}, DataType::UINT8),
                                                             MemoryLocation::DRAM),
                    1950.0f / 45000.0f);
}

TEST_F(TestHWPerformanceModelNPU5x, ArchTest5_0_BasicAssertions_Default) {
    const auto device = VPUDevice::NPU_5_0;
    const auto& hw{hw_info.device(device)};

    EXPECT_EQ(hw.input_channels_mac(), 8u);
    EXPECT_EQ(hw.get_nr_ppe(), 32u);
    EXPECT_EQ(hw.get_nr_macs(), 4096u);
    EXPECT_EQ(hw.get_dpu_fclk(), 1950u);
    EXPECT_FLOAT_EQ(dma.get_bandwidth_cycles_per_bytesLegacy(hw, VPUTensor({56, 56, 64, 1}, DataType::UINT8),
                                                             MemoryLocation::DRAM),
                    1950.0f / 136000.0f);
}
TEST_F(TestHWPerformanceModelNPU5x, ArchTest5_0_BasicAssertions_evo0) {
    const auto device = VPUDevice::NPU_5_0;
    const auto& hw{hw_info_evo0.device(device)};

    EXPECT_EQ(hw.input_channels_mac(), 8u);
    EXPECT_EQ(hw.get_nr_ppe(), 32u);
    EXPECT_EQ(hw.get_nr_macs(), 4096u);
    EXPECT_EQ(hw.get_dpu_fclk(), 1950u);
    EXPECT_FLOAT_EQ(dma.get_bandwidth_cycles_per_bytesLegacy(hw, VPUTensor({56, 56, 64, 1}, DataType::UINT8),
                                                             MemoryLocation::DRAM),
                    1950.0f / 136000.0f);
}
TEST_F(TestHWPerformanceModelNPU5x, ArchTest5_0_BasicAssertions_evo1) {
    const auto device = VPUDevice::NPU_5_0;
    const auto& hw{hw_info_evo1.device(device)};

    EXPECT_EQ(hw.input_channels_mac(), 8u);
    EXPECT_EQ(hw.get_nr_ppe(), 32u);
    EXPECT_EQ(hw.get_nr_macs(), 4096u);
    EXPECT_EQ(hw.get_dpu_fclk(), 1950u);
    EXPECT_FLOAT_EQ(dma.get_bandwidth_cycles_per_bytesLegacy(hw, VPUTensor({56, 56, 64, 1}, DataType::UINT8),
                                                             MemoryLocation::DRAM),
                    1950.0f / 136000.0f);
}

}  // namespace VPUNN_unit_tests