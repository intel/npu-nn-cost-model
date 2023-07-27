// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/performance.h"

#include <gtest/gtest.h>
#include "common_helpers.h"
#include "vpu/types.h"

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {

/// @brief tests for performance.h
class TestVPUNNPerformanceModel : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }

private:
};

TEST_F(TestVPUNNPerformanceModel, ArchTest2_0_BasicAssertions) {
    const auto device = VPUNN::VPUDevice::VPU_2_0;

    EXPECT_EQ(input_channels_mac(device), 1u);
    EXPECT_EQ(get_nr_ppe(device), 16u);
    EXPECT_EQ(get_nr_macs(device), 256u);
    EXPECT_EQ(get_dpu_fclk(device), 700u);
    EXPECT_FLOAT_EQ(get_bandwidth_cycles_per_bytes(VPUNN::VPUTensor({56, 56, 64, 1}, VPUNN::DataType::UINT8), device,
                                                   VPUNN::MemoryLocation::DRAM),
                    700.0f / 20000.0f);
}

TEST_F(TestVPUNNPerformanceModel, ArchTest2_1_BasicAssertions) {
    const auto device = VPUNN::VPUDevice::VPU_2_1;

    EXPECT_EQ(input_channels_mac(device), 1u);
    EXPECT_EQ(get_nr_ppe(device), 16u);
    EXPECT_EQ(get_nr_macs(device), 256u);
    EXPECT_EQ(get_dpu_fclk(device), 850u);
    EXPECT_FLOAT_EQ(get_bandwidth_cycles_per_bytes(VPUNN::VPUTensor({56, 56, 64, 1}, VPUNN::DataType::UINT8), device,
                                                   VPUNN::MemoryLocation::DRAM),
                    850.0f / 20000.0f);
}

TEST_F(TestVPUNNPerformanceModel, ArchTest2_7_BasicAssertions) {
    const auto device = VPUNN::VPUDevice::VPU_2_7;

    EXPECT_EQ(input_channels_mac(device), 8u);
    EXPECT_EQ(get_nr_ppe(device), 64u);
    EXPECT_EQ(get_nr_macs(device), 2048u);
    EXPECT_EQ(get_dpu_fclk(device), 1300u);
    EXPECT_FLOAT_EQ(get_bandwidth_cycles_per_bytes(VPUNN::VPUTensor({56, 56, 64, 1}, VPUNN::DataType::UINT8), device,
                                                   VPUNN::MemoryLocation::DRAM),
                    1300.0f / 27000.0f);
}

TEST_F(TestVPUNNPerformanceModel, ArchTest4_0_BasicAssertions) {
    const auto device = VPUNN::VPUDevice::VPU_4_0;

    EXPECT_EQ(input_channels_mac(device), 8u);
    EXPECT_EQ(get_nr_ppe(device), 64u);
    EXPECT_EQ(get_nr_macs(device), 2048u);
    EXPECT_EQ(get_dpu_fclk(device), 1700u);
    EXPECT_FLOAT_EQ(get_bandwidth_cycles_per_bytes(VPUNN::VPUTensor({56, 56, 64, 1}, VPUNN::DataType::UINT8), device,
                                                   VPUNN::MemoryLocation::DRAM),
                    1700.0f / 45000.0f);
}

TEST_F(TestVPUNNPerformanceModel, BITC2_7_BasicAssertions) {
    const auto device = VPUNN::VPUDevice::VPU_2_7;
    const auto tensor = VPUNN::VPUTensor({56, 56, 64, 1}, VPUNN::DataType::UINT8);
    // BITC enabled takes has a smaller cycles/byte bw
    EXPECT_FLOAT_EQ(get_bandwidth_cycles_per_bytes(tensor, device, VPUNN::MemoryLocation::CMX, false),
                    2 * get_bandwidth_cycles_per_bytes(tensor, device, VPUNN::MemoryLocation::CMX, true));
}

TEST_F(TestVPUNNPerformanceModel, Permute_2_7_BasicAssertions) {
    const auto device = VPUNN::VPUDevice::VPU_2_7;
    const auto tensor_fp16 = VPUNN::VPUTensor({56, 56, 64, 1}, VPUNN::DataType::FLOAT16);
    const auto tensor_uint8 = VPUNN::VPUTensor({56, 56, 64, 1}, VPUNN::DataType::UINT8);
    // BITC enabled takes has a smaller cycles/byte bw
    EXPECT_FLOAT_EQ(get_bandwidth_cycles_per_bytes(tensor_fp16, device, VPUNN::MemoryLocation::CMX, false, true),
                    0.5f * 1300.0f / 975.0f);
    EXPECT_FLOAT_EQ(get_bandwidth_cycles_per_bytes(tensor_uint8, device, VPUNN::MemoryLocation::CMX, false, true),
                    1.0f * 1300.0f / 975.0f);
}

}  // namespace VPUNN_unit_tests