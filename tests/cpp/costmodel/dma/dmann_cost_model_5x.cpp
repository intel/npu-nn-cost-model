// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
// #include "vpu_cost_model.h"
#include "dmann_cost_model.h"

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {
using namespace VPUNN;

class TestDMANNCostModelNPU5x : public TestDMANNCostModel {
public:
protected:
    void SetUp() override {
        TestDMANNCostModel::SetUp();
    }
    
};

class TestDMA_TH_CostModelVPU5x : public TestDMA_TH_CostModel {
public:
protected:
};


TEST_F(TestDMANNCostModelNPU5x, SmokeTestDMA_50) {
    const std::string model_path = NPU_DMA_5_0_MODEL_PATH;
    ASSERT_NO_THROW(DMACostModel<DMANNWorkload_NPU50> x{model_path});
    DMACostModel<DMANNWorkload_NPU50> dma_model(model_path);
    ASSERT_TRUE(dma_model.nn_initialized());

    {
        DMANNWorkload_NPU50 wl_50{
                VPUNN::VPUDevice::NPU_5_0,  // VPUDevice device;  ///< NPU device
                65535,                      // int src_width;
                65535,                      // int dst_width;
                0,                          // int num_dim;
                {{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}},
                Num_DMA_Engine::Num_Engine_1,
                MemoryDirection::CMX2CMX  // MemoryDirection transfer_direction;
        };
        auto dma_cycles = dma_model.computeCycles(wl_50);

        EXPECT_EQ(dma_cycles, 1129 /*@1700MHz*/) << wl_50 << Cycles::toErrorText(dma_cycles);
    }
}

TEST_F(TestDMANNCostModelNPU5x, SmokeTestDMA_50_V1) {
    const std::string model_path = NPU_DMA_5_0_V1_MODEL_PATH;
    ASSERT_NO_THROW(DMACostModel<DMANNWorkload_NPU50> x{model_path});
    DMACostModel<DMANNWorkload_NPU50> dma_model(model_path);
    ASSERT_TRUE(dma_model.nn_initialized());
    {
        DMANNWorkload_NPU50 wl_50{
                VPUNN::VPUDevice::NPU_5_0,  // VPUDevice device;  ///< NPU device
                65535,                      // int src_width;
                65535,                      // int dst_width;
                0,                          // int num_dim;
                {{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}},
                Num_DMA_Engine::Num_Engine_1,
                MemoryDirection::CMX2CMX  // MemoryDirection transfer_direction;
        };
        auto dma_cycles = dma_model.computeCycles(wl_50);

        EXPECT_EQ(dma_cycles, 1129 /*@1700MHz*/)
                << wl_50
                << Cycles::toErrorText(dma_cycles);  ///  1129 cyc is measured directly in the test, not from profiling
    }

}

// this test is for New/Updated theoretical model
TEST_F(TestDMA_TH_CostModelVPU5x, DMA_Theoretical_regresion_NPU50_Default) {
    VPUCostModel cm("empty");
    ASSERT_TRUE(!cm.nn_initialized());
    const VPUDevice device{VPUDevice::NPU_5_0};

    const int p = ((50 * evoX) * (int)GlobalHarwdwareCharacteristics::get_dpu_fclk(VPUDevice::NPU_5_0)) /
                  (int)GlobalHarwdwareCharacteristics::get_cmx_fclk(VPUDevice::NPU_5_0);
    EXPECT_EQ(87 * evoX, p);

    const std::vector<TestCase> tc{

            {mkwl(0, DRAM, CMX, device), 585 + p, "\nzero DC"},  // was 1864 (0.955 ns)
            {mkwl(0, DRAM, DRAM, device), 585 + p, "zero DD"},   // was 1864
            {mkwl(0, CMX, CMX, device), 56 + p, "zero CC"},      // was 28 (16 cyc), 56 is 32
            {mkwl(0, CMX, DRAM, device), 585 + p, "zero CD"},    // was 1864

            {mkwl(1024, MemoryLocation::DRAM, MemoryLocation::CMX, device), 614 + p + 1 * evoX, "1k DC"},  // was 1921
            {mkwl(2048, MemoryLocation::DRAM, MemoryLocation::CMX, device), 642 + p + 2 * evoX, "2k DC"},  // was 1977
            {mkwl(10000, MemoryLocation::DRAM, MemoryLocation::CMX, device), 859 + p + 12 * evoX,
             "10k DC"},  // was    2412

            {mkwl(1024, MemoryLocation::CMX, MemoryLocation::DRAM, device), 614 + p + 1 * evoX, "1k CD"},  // was 1921
            {mkwl(2048, MemoryLocation::CMX, MemoryLocation::DRAM, device), 642 + p + 2 * evoX, "2k CD"},  // was 1977
            {mkwl(10000, MemoryLocation::CMX, MemoryLocation::DRAM, device), 859 + p + 12 * evoX,
             "10k CD"},  // was 2412

            {mkwl(1024, MemoryLocation::CMX, MemoryLocation::CMX, device), 85 + p, "1k CC"},   // was 85  , 57+28(16cyc)
            {mkwl(2048, MemoryLocation::CMX, MemoryLocation::CMX, device), 113 + p, "2k CC"},  // was 141,  85+28
            {mkwl(10000, MemoryLocation::CMX, MemoryLocation::CMX, device), 302 + 28 + p, "10k CC"},  // was 576

            {mkwl(1024, MemoryLocation::DRAM, MemoryLocation::DRAM, device), 614 + p, "1k DD"},    // was 1909
            {mkwl(2048, MemoryLocation::DRAM, MemoryLocation::DRAM, device), 642 + p, "2k DD"},    // was 1953
            {mkwl(10000, MemoryLocation::DRAM, MemoryLocation::DRAM, device), 859 + p, "10k DD"},  // was 2298

            // compressed
            {mkwl_compr(1024, 2048, MemoryLocation::DRAM, MemoryLocation::CMX, device), 614 + p + 1 * evoX,
             "1kto2k DC comp"},  // was 1921
            {mkwl_compr(2048, 1024, MemoryLocation::DRAM, MemoryLocation::CMX, device), 642 + p,
             "2kto1k DC "},  // was 1953

            {mkwl_compr(1024, 2048, MemoryLocation::CMX, MemoryLocation::DRAM, device), 642 + p,
             "1kto2k CD"},  // was 1953
            {mkwl_compr(2048, 1024, MemoryLocation::CMX, MemoryLocation::DRAM, device), 642 + p + 2 * evoX,
             "2kto1k CD"},  // was 1921

            {mkwl_compr(1024, 2048, MemoryLocation::CMX, MemoryLocation::CMX, device), 85 + 28 + p,
             "1kto2k CC"},  // was 85
            {mkwl_compr(2048, 1024, MemoryLocation::CMX, MemoryLocation::CMX, device), 85 + 28 + p,
             "2kto1k CC"},  // was 85

            {mkwl_compr(1024, 2048, MemoryLocation::DRAM, MemoryLocation::DRAM, device), 642 + p,
             "1kto2k DD"},  // was 1953
            {mkwl_compr(2048, 1024, MemoryLocation::DRAM, MemoryLocation::DRAM, device), 642 + p,
             "2kto1k DD"},  // was 1953

            // permute
            {mkwl_(1024, 1024, dt, dt, Layout::ZXY, Layout::XYZ, DRAM, CMX, device), 614 + p + 1 * evoX,
             "1k DC perm"},  // was 3657
            {mkwl_(1024, 1024, dt, dt, Layout::ZXY, Layout::XYZ, CMX, DRAM, device), 614 + p + 1 * evoX,
             "1k CD perm"},  // was 3657
            {mkwl_(1024, 1024, dt, dt, Layout::ZXY, Layout::XYZ, CMX, CMX, device), 1849 + p,
             "1k CC perm"},  // was 1821
            {mkwl_(1024, 1024, dt, dt, Layout::ZXY, Layout::XYZ, DRAM, DRAM, device), 614 + p,
             "1k DD perm"},  // was 1909

    };

    auto check = [&](const TestCase& tc) {
        auto dma_cyc = cm.DMA(tc.t_in);
        if constexpr (!PerformanceMode::forceLegacy_G5) {  // auto legacy_dma_cyc =
                                                           // cm.DMATheoreticalCyclesLegacyLNL(tc.t_in);
            EXPECT_EQ(dma_cyc, tc.t_exp) << "\n ********** " << tc.t_name << "\n" << tc.t_in;

            // EXPECT_EQ(dma_cyc, legacy_dma_cyc) << tc.t_name << "\n";
        } else {
            EXPECT_GT(dma_cyc, 0) << "\n ********** " << tc.t_name << "\n" << tc.t_in;
        }
    };

    for (const auto& t : tc) {
        check(t);
    }
}

}  // namespace VPUNN_unit_tests