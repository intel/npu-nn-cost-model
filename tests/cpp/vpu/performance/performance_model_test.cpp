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

#ifdef INTEL_EMBARGO_NPU5
    // 5.0 available
    {
        const int plus = (50 * evoX * (int)GlobalHarwdwareCharacteristics::get_dpu_fclk(VPUDevice::NPU_5_0)) /
                         (int)GlobalHarwdwareCharacteristics::get_cmx_fclk(VPUDevice::NPU_5_0);
        EXPECT_EQ(87 * evoX, plus);
        EXPECT_EQ(hw_info.device(VPUDevice::NPU_5_0).get_DMA_latency(MemoryLocation::DRAM),
                  585 + plus);  // 1864 956ns @1950Mhz,390 200ns @1950Mhz
        EXPECT_EQ(hw_info.device(VPUDevice::NPU_5_0).get_DMA_latency(MemoryLocation::CMX),
                  56 + plus);  // 32 cyc @ 1114MHZ  => 56.x @1950
    }
#endif  // INTEL_EMBARGO_NPU5
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

#ifdef INTEL_EMBARGO_NPU5
    // 5.0 available
    {
        const int plus = (0 * (int)hw.device(VPUDevice::NPU_5_0).get_dpu_fclk()) /
                         (int)hw.device(VPUDevice::NPU_5_0).get_cmx_fclk();
        EXPECT_EQ(0, plus);
        EXPECT_EQ(hw.device(VPUDevice::NPU_5_0).get_DMA_latency(MemoryLocation::DRAM),
                  585 + plus);  // 1864 956ns @1950Mhz,390 200ns @1950Mhz
        EXPECT_EQ(hw.device(VPUDevice::NPU_5_0).get_DMA_latency(MemoryLocation::CMX),
                  56 + plus);  // 32 cyc @ 1114MHZ  => 56.x @1950
    }
#endif  // INTEL_EMBARGO_NPU5
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

#ifdef INTEL_EMBARGO_NPU5
    // 5.0 available
    {
        const int plus = (50 * (int)hw.device(VPUDevice::NPU_5_0).get_dpu_fclk()) /
                         (int)hw.device(VPUDevice::NPU_5_0).get_cmx_fclk();
        EXPECT_EQ(87, plus);
        EXPECT_EQ(hw.device(VPUDevice::NPU_5_0).get_DMA_latency(MemoryLocation::DRAM),
                  585 + plus);  // 1864 956ns @1950Mhz,390 200ns @1950Mhz
        EXPECT_EQ(hw.device(VPUDevice::NPU_5_0).get_DMA_latency(MemoryLocation::CMX),
                  56 + plus);  // 32 cyc @ 1114MHZ  => 56.x @1950
    }
#endif  // INTEL_EMBARGO_NPU5
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

#ifdef INTEL_EMBARGO_NPU5
    // 45.0 not yet available
    EXPECT_EQ(hw_info_legacy.device(VPUDevice::NPU_5_0).get_DMA_latency(MemoryLocation::DRAM),
              1864);  // 956ns @1950Mhz
    EXPECT_EQ(hw_info_legacy.device(VPUDevice::NPU_5_0).get_DMA_latency(MemoryLocation::CMX),
              28);  // 16 cyc @ 1114MHZ  => 28.x @1950
#endif              // INTEL_EMBARGO_NPU5
}

TEST_F(TestHWPerformanceModel_BASICS, TestGetProfilingClkMHz) {
    EXPECT_FLOAT_EQ(hw_info.device(VPUDevice::VPU_2_0).get_profiling_clk_MHz(), 38.4f);
    EXPECT_FLOAT_EQ(hw_info.device(VPUDevice::VPU_2_1).get_profiling_clk_MHz(), 38.4f);
    EXPECT_FLOAT_EQ(hw_info.device(VPUDevice::VPU_2_7).get_profiling_clk_MHz(), 38.4f);
    EXPECT_FLOAT_EQ(hw_info.device(VPUDevice::VPU_4_0).get_profiling_clk_MHz(), 38.4f / 2);
#ifdef INTEL_EMBARGO_NPU5
    EXPECT_FLOAT_EQ(hw_info.device(VPUDevice::NPU_5_0).get_profiling_clk_MHz(), 38.4f / 2);
#endif  // INTEL_EMBARGO_NPU5

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
#ifdef INTEL_EMBARGO_NPU5
    EXPECT_EQ(hw_info.device(VPUDevice::NPU_5_0).get_profiling_clk_Hz(), 38400000 / 2);
#endif  // INTEL_EMBARGO_NPU5

    EXPECT_EQ(hw_info.device(VPUDevice::__size).get_profiling_clk_Hz(), 0);             // Test with an unknown device
    EXPECT_EQ(hw_info.device(static_cast<VPUDevice>(9999)).get_profiling_clk_Hz(), 0);  // Test with an unknown device
}

}  // namespace VPUNN_unit_tests
