// Copyright © 2024 Intel Corporation
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
using namespace VPUNN;

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
    EXPECT_FLOAT_EQ(get_bandwidth_cycles_per_bytesLegacy(VPUNN::VPUTensor({56, 56, 64, 1}, VPUNN::DataType::UINT8),
                                                         device, VPUNN::MemoryLocation::DRAM),
                    700.0f / 20000.0f);
}

TEST_F(TestVPUNNPerformanceModel, ArchTest2_1_BasicAssertions) {
    const auto device = VPUNN::VPUDevice::VPU_2_1;

    EXPECT_EQ(input_channels_mac(device), 1u);
    EXPECT_EQ(get_nr_ppe(device), 16u);
    EXPECT_EQ(get_nr_macs(device), 256u);
    EXPECT_EQ(get_dpu_fclk(device), 850u);
    EXPECT_FLOAT_EQ(get_bandwidth_cycles_per_bytesLegacy(VPUNN::VPUTensor({56, 56, 64, 1}, VPUNN::DataType::UINT8),
                                                         device, VPUNN::MemoryLocation::DRAM),
                    850.0f / 20000.0f);
}

TEST_F(TestVPUNNPerformanceModel, ArchTest2_7_BasicAssertions) {
    const auto device = VPUNN::VPUDevice::VPU_2_7;

    EXPECT_EQ(input_channels_mac(device), 8u);
    EXPECT_EQ(get_nr_ppe(device), 64u);
    EXPECT_EQ(get_nr_macs(device), 2048u);
    EXPECT_EQ(get_dpu_fclk(device), 1300u);
    EXPECT_FLOAT_EQ(get_bandwidth_cycles_per_bytesLegacy(VPUNN::VPUTensor({56, 56, 64, 1}, VPUNN::DataType::UINT8),
                                                         device, VPUNN::MemoryLocation::DRAM),
                    1300.0f / 27000.0f);
}

TEST_F(TestVPUNNPerformanceModel, ArchTest4_0_BasicAssertions) {
    const auto device = VPUNN::VPUDevice::VPU_4_0;

    EXPECT_EQ(input_channels_mac(device), 8u);
    EXPECT_EQ(get_nr_ppe(device), 64u);
    EXPECT_EQ(get_nr_macs(device), 2048u);
    EXPECT_EQ(get_dpu_fclk(device), 1700u);
    EXPECT_FLOAT_EQ(get_bandwidth_cycles_per_bytesLegacy(VPUNN::VPUTensor({56, 56, 64, 1}, VPUNN::DataType::UINT8),
                                                         device, VPUNN::MemoryLocation::DRAM),
                    1700.0f / 45000.0f);
}

TEST_F(TestVPUNNPerformanceModel, BITC2_7_BasicAssertions) {
    const auto device = VPUNN::VPUDevice::VPU_2_7;
    const auto tensor = VPUNN::VPUTensor({56, 56, 64, 1}, VPUNN::DataType::UINT8);
    // BITC enabled takes has a smaller cycles/byte bw
    EXPECT_FLOAT_EQ(get_bandwidth_cycles_per_bytesLegacy(tensor, device, VPUNN::MemoryLocation::CMX, false),
                    2 * get_bandwidth_cycles_per_bytesLegacy(tensor, device, VPUNN::MemoryLocation::CMX, true));
}

TEST_F(TestVPUNNPerformanceModel, Permute_2_7_BasicAssertions) {
    const auto device = VPUNN::VPUDevice::VPU_2_7;
    const auto tensor_fp16 = VPUNN::VPUTensor({56, 56, 64, 1}, VPUNN::DataType::FLOAT16);
    const auto tensor_uint8 = VPUNN::VPUTensor({56, 56, 64, 1}, VPUNN::DataType::UINT8);
    // BITC enabled takes has a smaller cycles/byte bw
    EXPECT_FLOAT_EQ(get_bandwidth_cycles_per_bytesLegacy(tensor_fp16, device, VPUNN::MemoryLocation::CMX, false, true),
                    0.5f * 1300.0f / 975.0f);
    EXPECT_FLOAT_EQ(get_bandwidth_cycles_per_bytesLegacy(tensor_uint8, device, VPUNN::MemoryLocation::CMX, false, true),
                    1.0f * 1300.0f / 975.0f);
}

TEST_F(TestVPUNNPerformanceModel, LatencyTests) {
    // tests to prove compiletime calculations of constants
    static_assert(get_DMA_latency(VPUDevice::VPU_2_7, MemoryLocation::DRAM) == 1242,
                  "latency should be compile time available");
    static_assert(get_DMA_latency(VPUDevice::VPU_2_7, MemoryLocation::CMX) == 21,
                  "latency should be compile time available");

    EXPECT_EQ(get_DMA_latency(VPUDevice::VPU_2_7, MemoryLocation::DRAM), 1242);  // 956ns @1300Mhz
    EXPECT_EQ(get_DMA_latency(VPUDevice::VPU_2_7, MemoryLocation::CMX), 21);     // 16 cyc @ 975MHZ => 21.x @1300

    // 2.0 not supported
    EXPECT_EQ(get_DMA_latency(VPUDevice::VPU_2_0, MemoryLocation::DRAM), 0);
    EXPECT_EQ(get_DMA_latency(VPUDevice::VPU_2_0, MemoryLocation::CMX), 0);

    // 4.0 not yet available
    EXPECT_EQ(get_DMA_latency(VPUDevice::VPU_4_0, MemoryLocation::DRAM), 510);  // 1625 956ns @1700Mhz
    EXPECT_EQ(get_DMA_latency(VPUDevice::VPU_4_0, MemoryLocation::CMX), 56);    // 32 cyc @ 971MHZ => 56.x @1700

}
TEST_F(TestVPUNNPerformanceModel, LatencyTestsLegacy) {
    // tests to prove compiletime calculations of constants
    static_assert(get_DMA_latency_Legacy(VPUDevice::VPU_2_7, MemoryLocation::DRAM) == 1242,
                  "latency should be compile time available");
    static_assert(get_DMA_latency_Legacy(VPUDevice::VPU_2_7, MemoryLocation::CMX) == 21,
                  "latency should be compile time available");

    EXPECT_EQ(get_DMA_latency_Legacy(VPUDevice::VPU_2_7, MemoryLocation::DRAM), 1242);  // 956ns @1300Mhz
    EXPECT_EQ(get_DMA_latency_Legacy(VPUDevice::VPU_2_7, MemoryLocation::CMX), 21);     // 16 cyc @ 975MHZ => 21.x @1300

    // 2.0 not supported
    EXPECT_EQ(get_DMA_latency_Legacy(VPUDevice::VPU_2_0, MemoryLocation::DRAM), 0);
    EXPECT_EQ(get_DMA_latency_Legacy(VPUDevice::VPU_2_0, MemoryLocation::CMX), 0);

    // 4.0 not yet available
    EXPECT_EQ(get_DMA_latency_Legacy(VPUDevice::VPU_4_0, MemoryLocation::DRAM), 1625);  // 956ns @1700Mhz
    EXPECT_EQ(get_DMA_latency_Legacy(VPUDevice::VPU_4_0, MemoryLocation::CMX), 27);     // 16 cyc @ 975MHZ => 28.x @1700
}

TEST_F(TestVPUNNPerformanceModel, TestGetProfilingClkMHz) {
    EXPECT_FLOAT_EQ(get_profiling_clk_MHz(VPUDevice::VPU_2_0), 38.4f);
    EXPECT_FLOAT_EQ(get_profiling_clk_MHz(VPUDevice::VPU_2_1), 38.4f);
    EXPECT_FLOAT_EQ(get_profiling_clk_MHz(VPUDevice::VPU_2_7), 38.4f);
    EXPECT_FLOAT_EQ(get_profiling_clk_MHz(VPUDevice::VPU_4_0), 38.4f / 2);

    EXPECT_FLOAT_EQ(get_profiling_clk_MHz(VPUDevice::__size), 0);             // Test with an unknown device
    EXPECT_FLOAT_EQ(get_profiling_clk_MHz(static_cast<VPUDevice>(9999)), 0);  // Test with an unknown device
}

TEST_F(TestVPUNNPerformanceModel, TestGetProfilingClkHz) {
    EXPECT_EQ(get_profiling_clk_Hz(VPUDevice::VPU_2_0), 38400000);
    EXPECT_EQ(get_profiling_clk_Hz(VPUDevice::VPU_2_1), 38400000);
    EXPECT_EQ(get_profiling_clk_Hz(VPUDevice::VPU_2_7), 38400000);
    EXPECT_EQ(get_profiling_clk_Hz(VPUDevice::VPU_4_0), 38400000 / 2);

    EXPECT_EQ(get_profiling_clk_Hz(VPUDevice::__size), 0);             // Test with an unknown device
    EXPECT_EQ(get_profiling_clk_Hz(static_cast<VPUDevice>(9999)), 0);  // Test with an unknown device
}

}  // namespace VPUNN_unit_tests