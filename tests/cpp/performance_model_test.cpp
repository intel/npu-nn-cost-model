// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/hw_characteristics/HW_characteristics_supersets.h"
#include "vpu/performance.h"

#include <gtest/gtest.h>
#include "common_helpers.h"
#include "vpu/types.h"

#include "vpu/dma_theoretical_cost_provider.h"  // added to test get_bandwidth_cycles_per_bytesLegacy

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {
using namespace VPUNN;

/// @brief tests for performance.h
class TestHWPerformanceModel_BASICS : public ::testing::Test {
public:
protected:
    static constexpr int evoX{0};  // factor to adjust expectation between evo 0 and 1
    void SetUp() override {
    }
    // instantiate(or ref) the legacy set of configuration
    const IHWCharacteristicsSet& hw_info_legacy = HWCharacteristicsSuperSets::get_legacyConfigurationRef();

    const IHWCharacteristicsSet& hw_info{HWCharacteristicsSuperSets::mainConfiguration()};  // the default one

    const IHWCharacteristicsSet& hw_info_evo0{HWCharacteristicsSuperSets::mainEvo0Configuration()};
    const IHWCharacteristicsSet& hw_info_evo1{HWCharacteristicsSuperSets::mainEvo1Configuration()};

    DMATheoreticalCostProvider_LNL_Legacy dma;

private:
};

TEST_F(TestHWPerformanceModel_BASICS, ArchTest2_0_BasicAssertions_Legacy) {
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

TEST_F(TestHWPerformanceModel_BASICS, ArchTest2_0_BasicAssertions_Default) {
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

TEST_F(TestHWPerformanceModel_BASICS, ArchTest2_1_BasicAssertions_Legacy) {
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
TEST_F(TestHWPerformanceModel_BASICS, ArchTest2_1_BasicAssertions_Default) {
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

TEST_F(TestHWPerformanceModel_BASICS, ArchTest2_7_BasicAssertions_Legacy) {
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
TEST_F(TestHWPerformanceModel_BASICS, ArchTest2_7_BasicAssertions_Default) {
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

TEST_F(TestHWPerformanceModel_BASICS, ArchTest4_0_BasicAssertions_Legacy) {
    const auto device = VPUDevice::VPU_4_0;
    const auto& hw{hw_info_legacy.device(device)};

    EXPECT_EQ(hw.input_channels_mac(), 8u);
    EXPECT_EQ(hw.get_nr_ppe(), 64u);
    EXPECT_EQ(hw.get_nr_macs(), 2048u);
    EXPECT_EQ(hw.get_dpu_fclk(), 1700u);
    EXPECT_FLOAT_EQ(dma.get_bandwidth_cycles_per_bytesLegacy(hw, VPUTensor({56, 56, 64, 1}, DataType::UINT8),
                                                             MemoryLocation::DRAM),
                    1700.0f / 45000.0f);
}
TEST_F(TestHWPerformanceModel_BASICS, ArchTest4_0_BasicAssertions_Default) {
    const auto device = VPUDevice::VPU_4_0;
    const auto& hw{hw_info.device(device)};

    EXPECT_EQ(hw.input_channels_mac(), 8u);
    EXPECT_EQ(hw.get_nr_ppe(), 64u);
    EXPECT_EQ(hw.get_nr_macs(), 2048u);
    EXPECT_EQ(hw.get_dpu_fclk(), 1700u);
    EXPECT_FLOAT_EQ(dma.get_bandwidth_cycles_per_bytesLegacy(hw, VPUTensor({56, 56, 64, 1}, DataType::UINT8),
                                                             MemoryLocation::DRAM),
                    1700.0f / 136000.0f);
}
TEST_F(TestHWPerformanceModel_BASICS, ArchTest4_0_BasicAssertions_Evo0) {
    const auto device = VPUDevice::VPU_4_0;
    const auto& hw{hw_info_evo0.device(device)};

    EXPECT_EQ(hw.input_channels_mac(), 8u);
    EXPECT_EQ(hw.get_nr_ppe(), 64u);
    EXPECT_EQ(hw.get_nr_macs(), 2048u);
    EXPECT_EQ(hw.get_dpu_fclk(), 1700u);
    EXPECT_FLOAT_EQ(dma.get_bandwidth_cycles_per_bytesLegacy(hw, VPUTensor({56, 56, 64, 1}, DataType::UINT8),
                                                             MemoryLocation::DRAM),
                    1700.0f / 136000.0f);
}
TEST_F(TestHWPerformanceModel_BASICS, ArchTest4_0_BasicAssertions_Evo1) {
    const auto device = VPUDevice::VPU_4_0;
    const auto& hw{hw_info_evo1.device(device)};

    EXPECT_EQ(hw.input_channels_mac(), 8u);
    EXPECT_EQ(hw.get_nr_ppe(), 64u);
    EXPECT_EQ(hw.get_nr_macs(), 2048u);
    EXPECT_EQ(hw.get_dpu_fclk(), 1700u);
    EXPECT_FLOAT_EQ(dma.get_bandwidth_cycles_per_bytesLegacy(hw, VPUTensor({56, 56, 64, 1}, DataType::UINT8),
                                                             MemoryLocation::DRAM),
                    1700.0f / 136000.0f);
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

TEST_F(TestHWPerformanceModel_BASICS, LatencyTests_Default) {
    {  // tests to prove compiletime calculations of constants
        constexpr DeviceHWCharacteristicsVariant config{
                DeviceHWCHaracteristicsConstRepo::get_HWCharacteristics(VPUDevice::VPU_2_7)};

        static_assert(HWCharacteristicsVariantWrap::get_DMA_latency(config, MemoryLocation::DRAM) == 1242,
                      "latency should be compile time available");
        static_assert(HWCharacteristicsVariantWrap::get_DMA_latency(config, MemoryLocation::CMX) == 21,
                      "latency should be compile time available");
    }

    EXPECT_EQ(hw_info.device(VPUDevice::VPU_2_7).get_DMA_latency(MemoryLocation::DRAM), 1242);  // 956ns @1300Mhz
    EXPECT_EQ(hw_info.device(VPUDevice::VPU_2_7).get_DMA_latency(MemoryLocation::CMX),
              21);  // 16 cyc @ 975MHZ => 21.x @1300

    // 2.0 not supported
    EXPECT_EQ(hw_info.device(VPUDevice::VPU_2_0).get_DMA_latency(MemoryLocation::DRAM), 0);
    EXPECT_EQ(hw_info.device(VPUDevice::VPU_2_0).get_DMA_latency(MemoryLocation::CMX), 0);

    // 4.0 not yet available
    {
        EXPECT_EQ(87, (50 * (int)GlobalHarwdwareCharacteristics::get_dpu_fclk(VPUDevice::VPU_4_0)) /
                              (int)GlobalHarwdwareCharacteristics::get_cmx_fclk(VPUDevice::VPU_4_0));
        const int plus = (50 * evoX * (int)GlobalHarwdwareCharacteristics::get_dpu_fclk(VPUDevice::VPU_4_0)) /
                         (int)GlobalHarwdwareCharacteristics::get_cmx_fclk(VPUDevice::VPU_4_0);
        EXPECT_EQ(87 * evoX, plus);
        EXPECT_EQ(hw_info.device(VPUDevice::VPU_4_0).get_DMA_latency(MemoryLocation::DRAM),
                  510 + plus);  // 1625 956ns @1700Mhz
        EXPECT_EQ(hw_info.device(VPUDevice::VPU_4_0).get_DMA_latency(MemoryLocation::CMX),
                  56 + plus);  // 32 cyc @ 971MHZ => 56.x @1700
    }

}

TEST_F(TestHWPerformanceModel_BASICS, LatencyTests_Evo0) {
    {  // tests to prove compiletime calculations of constants
        constexpr DeviceHWCharacteristicsVariant config{
                DeviceHWCHaracteristicsConstRepo::get_HWCharacteristicsEvo1(VPUDevice::VPU_2_7)};

        static_assert(HWCharacteristicsVariantWrap::get_DMA_latency(config, MemoryLocation::DRAM) == 1242,
                      "latency should be compile time available");
        static_assert(HWCharacteristicsVariantWrap::get_DMA_latency(config, MemoryLocation::CMX) == 21,
                      "latency should be compile time available");
    }
    const IHWCharacteristicsSet& hw{hw_info_evo0};
    EXPECT_EQ(hw.device(VPUDevice::VPU_2_7).get_DMA_latency(MemoryLocation::DRAM), 1242);  // 956ns @1300Mhz
    EXPECT_EQ(hw.device(VPUDevice::VPU_2_7).get_DMA_latency(MemoryLocation::CMX),
              21);  // 16 cyc @ 975MHZ => 21.x @1300

    // 2.0 not supported
    EXPECT_EQ(hw.device(VPUDevice::VPU_2_0).get_DMA_latency(MemoryLocation::DRAM), 0);
    EXPECT_EQ(hw.device(VPUDevice::VPU_2_0).get_DMA_latency(MemoryLocation::CMX), 0);

    // 4.0 not yet available
    {
        EXPECT_EQ(87, (50 * (int)hw.device(VPUDevice::VPU_4_0).get_dpu_fclk()) /
                              (int)hw.device(VPUDevice::VPU_4_0).get_cmx_fclk());
        const int plus = (0 * (int)hw.device(VPUDevice::VPU_4_0).get_dpu_fclk()) /
                         (int)hw.device(VPUDevice::VPU_4_0).get_cmx_fclk();
        EXPECT_EQ(0, plus);
        EXPECT_EQ(hw.device(VPUDevice::VPU_4_0).get_DMA_latency(MemoryLocation::DRAM),
                  510 + plus);  // 1625 956ns @1700Mhz
        EXPECT_EQ(hw.device(VPUDevice::VPU_4_0).get_DMA_latency(MemoryLocation::CMX),
                  56 + plus);  // 32 cyc @ 971MHZ => 56.x @1700
    }

}

TEST_F(TestHWPerformanceModel_BASICS, LatencyTests_Evo1) {
    {  // tests to prove compiletime calculations of constants
        constexpr DeviceHWCharacteristicsVariant config{
                DeviceHWCHaracteristicsConstRepo::get_HWCharacteristicsEvo1(VPUDevice::VPU_2_7)};

        static_assert(HWCharacteristicsVariantWrap::get_DMA_latency(config, MemoryLocation::DRAM) == 1242,
                      "latency should be compile time available");
        static_assert(HWCharacteristicsVariantWrap::get_DMA_latency(config, MemoryLocation::CMX) == 21,
                      "latency should be compile time available");
    }
    const IHWCharacteristicsSet& hw{hw_info_evo1};
    EXPECT_EQ(hw.device(VPUDevice::VPU_2_7).get_DMA_latency(MemoryLocation::DRAM), 1242);  // 956ns @1300Mhz
    EXPECT_EQ(hw.device(VPUDevice::VPU_2_7).get_DMA_latency(MemoryLocation::CMX),
              21);  // 16 cyc @ 975MHZ => 21.x @1300

    // 2.0 not supported
    EXPECT_EQ(hw.device(VPUDevice::VPU_2_0).get_DMA_latency(MemoryLocation::DRAM), 0);
    EXPECT_EQ(hw.device(VPUDevice::VPU_2_0).get_DMA_latency(MemoryLocation::CMX), 0);

    // 4.0 not yet available
    {
        EXPECT_EQ(87, (50 * (int)hw.device(VPUDevice::VPU_4_0).get_dpu_fclk()) /
                              (int)hw.device(VPUDevice::VPU_4_0).get_cmx_fclk());
        const int plus = (50 * (int)hw.device(VPUDevice::VPU_4_0).get_dpu_fclk()) /
                         (int)hw.device(VPUDevice::VPU_4_0).get_cmx_fclk();
        EXPECT_EQ(87, plus);
        EXPECT_EQ(hw.device(VPUDevice::VPU_4_0).get_DMA_latency(MemoryLocation::DRAM),
                  510 + plus);  // 1625 956ns @1700Mhz
        EXPECT_EQ(hw.device(VPUDevice::VPU_4_0).get_DMA_latency(MemoryLocation::CMX),
                  56 + plus);  // 32 cyc @ 971MHZ => 56.x @1700
    }

}

TEST_F(TestHWPerformanceModel_BASICS, LatencyTestsLegacy) {
    {
        constexpr DeviceHWCharacteristicsVariant legacy_config{
                DeviceHWCHaracteristicsConstRepo::get_HWCharacteristics_Legacy(VPUDevice::VPU_2_7)};
        // tests to prove compiletime calculations of constants
        static_assert(std::visit(
                              [](auto& obj) {
                                  return obj.get_DMA_latency(MemoryLocation::DRAM);
                              },
                              legacy_config) == 1242,
                      "latency should be compile time available");

        static_assert(std::visit(
                              [](auto& obj) {
                                  return obj.get_DMA_latency(MemoryLocation::CMX);
                              },
                              legacy_config) == 21,
                      "latency should be compile time available");

        static_assert(HWCharacteristicsVariantWrap::get_DMA_latency(legacy_config, MemoryLocation::DRAM) == 1242,
                      "latency should be compile time available");
        static_assert(HWCharacteristicsVariantWrap::get_DMA_latency(legacy_config, MemoryLocation::CMX) == 21,
                      "latency should be compile time available");
    }

    EXPECT_EQ(hw_info_legacy.device(VPUDevice::VPU_2_7).get_DMA_latency(MemoryLocation::DRAM),
              1242);  // 956ns @1300Mhz
    EXPECT_EQ(hw_info_legacy.device(VPUDevice::VPU_2_7).get_DMA_latency(MemoryLocation::CMX),
              21);  // 16 cyc @ 975MHZ => 21.x @1300

    // 2.0 not supported
    EXPECT_EQ(hw_info_legacy.device(VPUDevice::VPU_2_0).get_DMA_latency(MemoryLocation::DRAM), 0);
    EXPECT_EQ(hw_info_legacy.device(VPUDevice::VPU_2_0).get_DMA_latency(MemoryLocation::CMX), 0);

    // 4.0 not yet available
    EXPECT_EQ(hw_info_legacy.device(VPUDevice::VPU_4_0).get_DMA_latency(MemoryLocation::DRAM),
              1625);  // 956ns @1700Mhz
    EXPECT_EQ(hw_info_legacy.device(VPUDevice::VPU_4_0).get_DMA_latency(MemoryLocation::CMX),
              27);  // 16 cyc @ 975MHZ => 28.x @1700

}

TEST_F(TestHWPerformanceModel_BASICS, TestGetProfilingClkMHz) {
    EXPECT_FLOAT_EQ(hw_info.device(VPUDevice::VPU_2_0).get_profiling_clk_MHz(), 38.4f);
    EXPECT_FLOAT_EQ(hw_info.device(VPUDevice::VPU_2_1).get_profiling_clk_MHz(), 38.4f);
    EXPECT_FLOAT_EQ(hw_info.device(VPUDevice::VPU_2_7).get_profiling_clk_MHz(), 38.4f);
    EXPECT_FLOAT_EQ(hw_info.device(VPUDevice::VPU_4_0).get_profiling_clk_MHz(), 38.4f / 2);

    EXPECT_FLOAT_EQ(hw_info.device(VPUDevice::__size).get_profiling_clk_MHz(),
                    0);  // Test with an unknown device
    EXPECT_FLOAT_EQ(hw_info.device(static_cast<VPUDevice>(9999)).get_profiling_clk_MHz(),
                    0);  // Test with an unknown device
}

TEST_F(TestHWPerformanceModel_BASICS, TestGetProfilingClkHz_default) {
    EXPECT_EQ(hw_info.device(VPUDevice::VPU_2_0).get_profiling_clk_Hz(), 38400000);
    EXPECT_EQ(hw_info.device(VPUDevice::VPU_2_1).get_profiling_clk_Hz(), 38400000);
    EXPECT_EQ(hw_info.device(VPUDevice::VPU_2_7).get_profiling_clk_Hz(), 38400000);
    EXPECT_EQ(hw_info.device(VPUDevice::VPU_4_0).get_profiling_clk_Hz(), 38400000 / 2);

    EXPECT_EQ(hw_info.device(VPUDevice::__size).get_profiling_clk_Hz(), 0);             // Test with an unknown device
    EXPECT_EQ(hw_info.device(static_cast<VPUDevice>(9999)).get_profiling_clk_Hz(), 0);  // Test with an unknown device
}

}  // namespace VPUNN_unit_tests
