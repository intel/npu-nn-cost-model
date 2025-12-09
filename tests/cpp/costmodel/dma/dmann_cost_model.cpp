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

#ifdef INTEL_EMBARGO_NPU5
TEST_F(TestDMANNCostModel, Create_DMA40_and_DMA50) {
    {
        const std::string model_path = VPU_DMA_4_0_MODEL_PATH;
        ASSERT_NO_THROW(DMACostModel<DMANNWorkload_NPU40> x{model_path});
        DMACostModel<DMANNWorkload_NPU40> dma_model(model_path);
        ASSERT_TRUE(dma_model.nn_initialized());

        const DMANNWorkload_NPU40 wl = create_DMANNWorkload_NPU40();

        EXPECT_TRUE(wl.device == VPUDevice::VPU_4_0) << wl;
    }

    {
        const std::string model_path = NPU_DMA_5_0_MODEL_PATH;
        ASSERT_NO_THROW(DMACostModel<DMANNWorkload_NPU50> x{model_path});
        DMACostModel<DMANNWorkload_NPU50> dma_model(model_path);
        ASSERT_TRUE(dma_model.nn_initialized());

        const DMANNWorkload_NPU50 wl_50 = create_DMANNWorkload_NPU50();
        EXPECT_TRUE(wl_50.device == VPUDevice::NPU_5_0) << wl_50;
    }
}
#endif  // INTEL_EMBARGO_NPU5

// Demonstrate some basic assertions.
TEST_F(TestDMANNCostModel, InitAspects) {
    {  // 27
        const std::string model_path = VPU_DMA_2_7_MODEL_PATH;
        EXPECT_NO_THROW(DMACostModel<DMANNWorkload_NPU27> x(model_path));
        DMACostModel<DMANNWorkload_NPU27> vpunn_model(model_path);
        EXPECT_TRUE(vpunn_model.nn_initialized());

        const auto file_content{read_a_file(std::move(model_path))};
        ASSERT_GT(file_content.size(), 10) << "Must have some content";

        EXPECT_NO_THROW(DMACostModel<DMANNWorkload_NPU27> x(file_content.data(), file_content.size(), true));
        DMACostModel<DMANNWorkload_NPU27> vpunn_model_buf(file_content.data(), file_content.size(), true);
        EXPECT_TRUE(vpunn_model_buf.nn_initialized());

        EXPECT_NO_THROW(DMACostModel<DMANNWorkload_NPU27> x(file_content.data(), file_content.size(), false));
        DMACostModel<DMANNWorkload_NPU27> vpunn_model_buf_copy(file_content.data(), file_content.size(), false);
        EXPECT_TRUE(vpunn_model_buf_copy.nn_initialized());
    }
    {  // garbage data/no file
        const std::string model_path = "NoFileHere.vpunn";
        EXPECT_NO_THROW(DMACostModel<DMANNWorkload_NPU27> x(model_path));
        DMACostModel<DMANNWorkload_NPU27> vpunn_model(model_path);
        EXPECT_FALSE(vpunn_model.nn_initialized());

        const decltype(read_a_file("")) file_content{'M', 'u', 's', 't', 'h', 'a', 'v', 'e', ' ', '0', '1'};
        ASSERT_GT(file_content.size(), 10) << "Must have some content";

        EXPECT_NO_THROW(DMACostModel<DMANNWorkload_NPU27> x(file_content.data(), file_content.size(), true));
        DMACostModel<DMANNWorkload_NPU27> vpunn_model_buf(file_content.data(), file_content.size(), true);
        EXPECT_FALSE(vpunn_model_buf.nn_initialized());

        EXPECT_NO_THROW(DMACostModel<DMANNWorkload_NPU27> x(file_content.data(), file_content.size(), false));
        DMACostModel<DMANNWorkload_NPU27> vpunn_model_buf_copy(file_content.data(), file_content.size(), false);
        EXPECT_FALSE(vpunn_model_buf_copy.nn_initialized());

        auto cycles_27 = vpunn_model_buf.computeCycles(wl_glob_27);

        EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_27));

        EXPECT_NEAR(cycles_27, 2820, 100);  // theoretical values are now as fallback
    }
}

TEST_F(TestDMANNCostModel, Mock_40_vs_VPU27_DPU) {
    {  // 27 and 40
        DMACostModel<DMANNWorkload_NPU27> model_2_7{VPU_DMA_2_7_MODEL_PATH};
        EXPECT_TRUE(model_2_7.nn_initialized());
        DMACostModel<DMANNWorkload_NPU27> model_4_0M{VPU_DMA_2_7_MODEL_PATH};
        EXPECT_TRUE(model_4_0M.nn_initialized());

        auto cycles_27 = model_2_7.computeCycles(wl_glob_27);
        auto cycles_40 = model_4_0M.computeCycles(wl_glob_40M);

        EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_27));
        EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_40));

        auto conv_cyc27_40 = (cycles_27 * GlobalHarwdwareCharacteristics::get_dpu_fclk(wl_glob_40M.device) /
                              GlobalHarwdwareCharacteristics::get_dpu_fclk(wl_glob_27.device) /
                              2);  // 2 is the speed up factor, 64 instead of 32?
        auto delta = std::abs((int)conv_cyc27_40 - (int)cycles_40);

        EXPECT_LE(delta, 9) << wl_glob_27 << wl_glob_40M << "\n"
                            << cycles_27 << " -> " << cycles_40;  // 2 is rounding errors
    }
    {  // garbage data/no file
        const std::string model_path = "NoFileHere.vpunn";
        DMACostModel<DMANNWorkload_NPU27> model_2_7{model_path};
        EXPECT_FALSE(model_2_7.nn_initialized());
        DMACostModel<DMANNWorkload_NPU27> model_4_0M{model_path};
        EXPECT_FALSE(model_4_0M.nn_initialized());

        auto cycles_27 = model_2_7.computeCycles(wl_glob_27);
        auto cycles_40 = model_4_0M.computeCycles(wl_glob_40M);

        EXPECT_NEAR(cycles_27, 2820, 100);  // theoretical values
        EXPECT_NEAR(cycles_40, 2863, 100);  // theoretical values
    }
}

TEST_F(TestDMA_TH_CostModel, DMA_Theoretical_Debug) {
    VPUCostModel cm("empty");
    ASSERT_TRUE(!cm.nn_initialized());
    auto check = [&](const TestCase& tc) {
        auto dma_cyc = cm.DMA(tc.t_in);
        const VPUDevice d{tc.t_in.device};
        const auto exp_dpu{tc.t_exp};
        const auto f_dpu{GlobalHarwdwareCharacteristics::get_dpu_fclk(d)};
        const auto f_cmx{GlobalHarwdwareCharacteristics::get_cmx_fclk(d)};
        const float exp_CMX{(float)exp_dpu * f_cmx / f_dpu};
        const float dma_CMX_cyc{(float)dma_cyc * f_cmx / f_dpu};

        std::cout << "\n"
                  << tc.t_name << ",\t"                                            //
                  << "DPU frq: " << f_dpu << " , CMX frq: " << f_cmx << " MHz.  "  //
                  << "*** Expecting: DPUCyc " << exp_dpu << ", microsec :" << computeMicroseconds(exp_dpu, d)
                  << " VPU cyc: " << exp_CMX
                  << ", microsec: " << computeMicroseconds((unsigned)std::ceil(exp_CMX), f_cmx)
                  << " ----->  Obtained: DPUCyc : " << dma_cyc << ", microsec :" << computeMicroseconds(dma_cyc, d)
                  << " VPU cyc: " << dma_CMX_cyc
                  << ", microsec: " << computeMicroseconds((unsigned)std::ceil(dma_CMX_cyc), f_cmx);
        ;
        EXPECT_EQ(dma_cyc, tc.t_exp) << tc.t_name << "\n" << tc.t_in;
    };

#ifdef INTEL_EMBARGO_NPU5
    {
        std::cout << "\n\n NPU50 \n";
        const VPUDevice device{VPUDevice::NPU_5_0};
        const int p = (50 * evoX * (int)GlobalHarwdwareCharacteristics::get_dpu_fclk(VPUDevice::NPU_5_0)) /
                      (int)GlobalHarwdwareCharacteristics::get_cmx_fclk(VPUDevice::NPU_5_0);
        EXPECT_EQ(87 * evoX, p);

        const std::vector<TestCase> tc_50{

                {mkwl(2048, MemoryLocation::CMX, MemoryLocation::CMX, device), 113 + p,
                 "2k CC"},  // was 141  (new 85 due to 64B/cyc )

                // compressed
                {mkwl_compr(1024, 2048, MemoryLocation::CMX, MemoryLocation::CMX, device), 113 + p,
                 "1kto2k CC"},  // was 85
                {mkwl_compr(2048, 1024, MemoryLocation::CMX, MemoryLocation::CMX, device), 113 + p,
                 "2kto1k CC"},  // was 85

                //
                {mkwl(0, DRAM, CMX, device), 585 + p, "\nzero DC"},  // was 1864 (0.955 ns), 390=200ns, 585=300ns
                {mkwl(0, DRAM, DRAM, device), 585 + p, "zero DD"},   // was 1864
                {mkwl(0, CMX, CMX, device), 56 + p, "zero CC"},      // was 28 (16 cyc)
                {mkwl(0, CMX, DRAM, device), 585 + p, "zero CD"},    // was 1864

                // examples

                {mkwl(8192, DRAM, CMX, device), 810 + p + 9 * evoX,
                 "\n8k DC"},  //// 0.572micros expected at 800MHz VPU (457vpuCLKS,800DPUclks),  was 2313. TX= 128 CMX
                              /// clk
                {mkwl(8192 * 2, CMX, DRAM, device), 1034 + p + 19 * evoX,
                 "16k CD"},  //// 0.468  expected at 800MHz VPU (374 vpuclks,655 DPU clks), was 2761. TX= 256 CMX clk

                {mkwl(3 * 512 * 172 * 2, DRAM, CMX, device), 15037 + p + 625 * evoX,
                 "\n500K teor DC"},                                                   // was 30768,  TX8256 CMX clk
                {mkwl(3 * 512 * 256 * 2, DRAM, CMX, device), 22095 + p + 929 * evoX,  // was 44884, TX 12288 CMX clk
                 "\n786K real DC"},  // 10.2 (8160 VPU, 14280DPU ), and 13.8(vpu 11040, dpu 19320) expe at 800MHz VPU

        };

        for (auto& t : tc_50) {
            if constexpr (!PerformanceMode::forceLegacy_G5) {
                check(t);
            }
        }
    }
#endif  // INTEL_EMBARGO_NPU5

    {
        std::cout << "\n\n NPU40 \n";
        VPUDevice device{VPUDevice::VPU_4_0};
        const int p = (50 * evoX * (int)GlobalHarwdwareCharacteristics::get_dpu_fclk(VPUDevice::VPU_4_0)) /
                      (int)GlobalHarwdwareCharacteristics::get_cmx_fclk(VPUDevice::VPU_4_0);
        EXPECT_EQ(87 * evoX, p);

        const std::vector<TestCase> tc_40{

                {mkwl(1024, MemoryLocation::CMX, MemoryLocation::CMX, device), 85 + p, "1k CC"},   //
                {mkwl(2048, MemoryLocation::CMX, MemoryLocation::CMX, device), 113 + p, "2k CC"},  //

                // compressed
                {mkwl_compr(1024, 2048, MemoryLocation::CMX, MemoryLocation::CMX, device), 113 + p, "1kto2k CC"},  //
                {mkwl_compr(2048, 1024, MemoryLocation::CMX, MemoryLocation::CMX, device), 113 + p, "2kto1k CC"},  //

                // zero
                {mkwl(0, DRAM, CMX, device), 510 + p, "\nzero DC"},  //
                {mkwl(0, DRAM, DRAM, device), 510 + p, "zero DD"},
                {mkwl(0, CMX, CMX, device), 56 + p, "zero CC"},
                {mkwl(0, CMX, DRAM, device), 510 + p, "zero CD"},

                // examples
                {mkwl(8192, DRAM, CMX, device), 735 + p + 9 * evoX, "\n8k DC"},      // 0.572 expected at 800MHz VPU
                {mkwl(8192 * 2, CMX, DRAM, device), 959 + p + 19 * evoX, "16k CD"},  // 0.468  expected at 800MHz VPU

                {mkwl(3 * 512 * 172 * 2, DRAM, CMX, device), 14965 + p + 624 * evoX, "\n500K theory DC"},  //
                {mkwl(3 * 512 * 256 * 2, DRAM, CMX, device), 22024 + p + 929 * evoX,
                 "\n786K real DC"},  // 10.2, and 13.8 exp at 800MHz VPU

        };

        for (const auto& t : tc_40) {
            if constexpr (!PerformanceMode::forceLegacy_G4) {
                check(t);
            }
        }
    }

    {
        std::cout << "\n\n NPU27\n";
        VPUDevice device{VPUDevice::VPU_2_7};

        const std::vector<TestCase> tc_27{

                {mkwl(1024, MemoryLocation::CMX, MemoryLocation::CMX, device), 107, "1k CC"},  //
                {mkwl(2048, MemoryLocation::CMX, MemoryLocation::CMX, device), 192, "2k CC"},  //

                // compressed
                {mkwl_compr(1024, 2048, MemoryLocation::CMX, MemoryLocation::CMX, device), 107, "1kto2k CC"},  //
                {mkwl_compr(2048, 1024, MemoryLocation::CMX, MemoryLocation::CMX, device), 107, "2kto1k CC"},  //

                // zero
                {mkwl(0, DRAM, CMX, device), 1242, "\nzero DC"},  //
                {mkwl(0, DRAM, DRAM, device), 1242, "zero DD"},
                {mkwl(0, CMX, CMX, device), 21, "zero CC"},
                {mkwl(0, CMX, DRAM, device), 1242, "zero CD"},

                // examples
                {mkwl(8192, DRAM, CMX, device), 1637, "\n8k DC"},     //
                {mkwl(8192 * 2, CMX, DRAM, device), 2031, "16k CD"},  //

                {mkwl(3 * 512 * 172 * 2, DRAM, CMX, device), 26683, "\n500K theory DC"},  //
                {mkwl(3 * 512 * 256 * 2, DRAM, CMX, device), 39108, "\n786K real DC"},    //

        };

        for (auto& t : tc_27) {
            check(t);
        }
    }

    // EXPECT_TRUE(false);
}

TEST_F(TestDMA_TH_CostModel, DMA_Th_Smoke_E162767_Legacy) {
    VPUCostModel cm("empty");
    DMATheoreticalCostProvider_LNL_Legacy dma_theoretical_lnl;
    DMATheoreticalCostProvider_PTL dma_theoretical_ptl(HWCharacteristicsSuperSets::get_mainConfigurationRef());

    ASSERT_TRUE(!cm.nn_initialized());
    auto check = [&](const TestCase& tc) {
        auto dma_cyc = cm.DMA(tc.t_in);
        const VPUDevice d{tc.t_in.device};
        const auto exp_dpu{tc.t_exp};
        const auto f_dpu{GlobalHarwdwareCharacteristics::get_dpu_fclk(d)};
        const auto f_cmx{GlobalHarwdwareCharacteristics::get_cmx_fclk(d)};
        const float exp_CMX{(float)exp_dpu * f_cmx / f_dpu};
        const float dma_CMX_cyc{(float)dma_cyc * f_cmx / f_dpu};

        std::cout << "\n"
                  << tc.t_name << ",\t"                                            //
                  << "DPU frq: " << f_dpu << " , CMX frq: " << f_cmx << " MHz.  "  //
                  << "*** Expecting: DPUCyc " << exp_dpu << ", microsec :" << computeMicroseconds(exp_dpu, d)
                  << " VPU cyc: " << exp_CMX
                  << ", microsec: " << computeMicroseconds((unsigned)std::ceil(exp_CMX), f_cmx)
                  << " ----->  Obtained: DPUCyc : " << dma_cyc << ", microsec :" << computeMicroseconds(dma_cyc, d)
                  << " VPU cyc: " << dma_CMX_cyc
                  << ", microsec: " << computeMicroseconds((unsigned)std::ceil(dma_CMX_cyc), f_cmx);
        ;
        EXPECT_EQ(dma_cyc, tc.t_exp) << tc.t_name << "\n" << tc.t_in;
    };

#ifdef INTEL_EMBARGO_NPU5
    {
        std::cout << "\n\n NPU50 \n";
        const VPUDevice device{VPUDevice::NPU_5_0};
        const int p = (50 * evoX * (int)GlobalHarwdwareCharacteristics::get_dpu_fclk(VPUDevice::NPU_5_0)) /
                      (int)GlobalHarwdwareCharacteristics::get_cmx_fclk(VPUDevice::NPU_5_0);
        EXPECT_EQ(87 * evoX, p);

        TestCase case1{mkwl(4864, MemoryLocation::DRAM, MemoryLocation::CMX, device),
                       PerformanceMode::forceLegacy_G5 ? 2131 : 719 + p + 5 * evoX,
                       "2k CC"};  // original on develop branch the cost is 1891. With vpucostmodel updated, the cost
                                  // changes to 644
        check(case1);

        const DMAWorkload wl{case1.t_in};
        auto dma_now = cm.DMA(wl);
        auto dma_n = dma_theoretical_ptl.DMATheoreticalCyclesPTL_ON(wl);
        auto dma_o = dma_theoretical_lnl.DMATheoreticalCyclesLegacyLNL(wl);

        EXPECT_EQ(dma_now, PerformanceMode::forceLegacy_G5 ? 2131 : 719 + p + 5 * evoX);
        EXPECT_EQ(dma_n, 719 + p + 5 * evoX);
        EXPECT_EQ(dma_o, 2131);
    }
#endif  // INTEL_EMBARGO_NPU5

    {
        std::cout << "\n\n NPU40 \n";
        const VPUDevice device{VPUDevice::VPU_4_0};

        TestCase case1{mkwl(4864, MemoryLocation::DRAM, MemoryLocation::CMX, device),
                       PerformanceMode::forceLegacy_G4 ? 1891 : 644 + (87 + 5) * evoX,
                       "2k CC"};  // original on develop branch the cost is 1891. With vpucostmodel updated, the cost
                                  // changes to 644
        check(case1);

        const DMAWorkload wl{case1.t_in};
        auto dma_now = cm.DMA(wl);
        auto dma_n = dma_theoretical_ptl.DMATheoreticalCyclesPTL_ON(wl);
        auto dma_o = dma_theoretical_lnl.DMATheoreticalCyclesLegacyLNL(wl);

        EXPECT_EQ(dma_now, PerformanceMode::forceLegacy_G4 ? 1891 : 644 + (87 + 5) * evoX);
        EXPECT_EQ(dma_n, 644 + (87 + 5) * evoX);
        EXPECT_EQ(dma_o, 1891);
    }

    // EXPECT_TRUE(false);
}

TEST_F(TestDMA_TH_CostModel, DMATheoreticalPTL_ON_function_Test) {
    auto mkWl = [](const VPUDevice dev) {
        DMAWorkload dmaWl{
                dev,                                                       // device
                {VPUTensor(4864, 1, 1, 1, DataType::UINT8, Layout::ZXY)},  // input dimensions WHCB
                {VPUTensor(4864, 1, 1, 1, DataType::UINT8, Layout::ZXY)},  // output dimensions
                MemoryLocation::CMX,                                       // src
                MemoryLocation::DRAM,                                      // dst
                1,                                                         // owt
        };
        return dmaWl;
    };

    DMATheoreticalCostProvider_PTL dma_theoretical_ptl(HWCharacteristicsSuperSets::get_mainConfigurationRef());
    VPUCostModel cm("empty");
    ASSERT_TRUE(!cm.nn_initialized());

    {  // valid cases
        for (const auto d : valid_dev_post_LNL) {
            const DMAWorkload dmaWl{mkWl(d)};
            auto dma_n = dma_theoretical_ptl.DMATheoreticalCyclesPTL_ON(dmaWl);
            EXPECT_FALSE(Cycles::isErrorCode(dma_n)) << "Device:" << d << " Err::" << Cycles::toErrorText(dma_n);
        }
    }

    {  // invalid cases, inside DMATheoreticalCyclesPTL_ON should be some negative values because of invalid devices =>
       // we obtain error because we can not convert a negative value to cycles (=unsigned int)
        std::vector<VPUDevice> invalid_dev{(VPUDevice)20, (VPUDevice)30};
        for (const auto d : invalid_dev) {
            const DMAWorkload dmaWl{mkWl(d)};
            unsigned long int dma_n{0};
            EXPECT_NO_THROW(dma_n = dma_theoretical_ptl.DMATheoreticalCyclesPTL_ON(dmaWl));
            EXPECT_TRUE(Cycles::isErrorCode(dma_n)) << "Device:" << d << " Err::" << Cycles::toErrorText(dma_n);
            EXPECT_EQ(dma_n, Cycles::ERROR_INVALID_CONVERSION_TO_CYCLES);
        }
    }
}

}  // namespace VPUNN_unit_tests