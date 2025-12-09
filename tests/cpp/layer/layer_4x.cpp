// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include "layer.h"

namespace VPUNN_unit_tests {
using namespace VPUNN;

class VPULayerCostModelTestNPU4x : public VPULayerCostModelTest {
public:
protected:
    void SetUp() override {
        VPULayerCostModelTest::SetUp();
    }
};

TEST_F(VPULayerCostModelTestNPU4x, DISABLED_ERROR_TILE_OUTPUT_LayerInvestigation_SOHO_NPU40) {
    const VPUNN::DPUWorkload wl1{
            VPUNN::VPUDevice::VPU_4_0,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(28, 28, 512, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(28, 28, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {1, 1},                                                      // kernels
            {1, 1},                                                      // strides
            {0, 0, 0, 0},                                                // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                          // execution mode
            VPUNN::ActivationFunction::NONE,                             // activation
            0.0F,                                                        // act_sparsity
            0.0F,                                                        // weight_sparsity
            {swz_def, swz_def},                                          // input_swizzling
            {swz_def},                                                   // output_swizzling
            1,                                                           // output_write_tiles
            {0, 0, 0, 0},                                                // offsets
            VPUNN::ISIStrategy::CLUSTERING,                              // isi_strategy
            false,                                                       // weight_sparsity_enabled
    };
    show_split = true;
    const bool prefetch{true};
    {
        const VPUNN::DPULayer tst_layer(wl1);
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 5500, 5500 + 1000},
                 " SOHO + 4tiles, no memmove, "},

                {{tst_layer, {1U, 1U, 5U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {VPUNN::Cycles::ERROR_TILE_OUTPUT, true, 4000, 4000 + 1000},
                 " SOHO + 5tiles , no memmove, "},

                {{tst_layer, {1U, 1U, 5U, VPUNN::VPUTilingStrategy::SOHO_K_SWITCH, false, false, prefetch}},
                 {VPUNN::Cycles::ERROR_TILE_OUTPUT, true, 4000, 4000 + 1000},
                 " SOHO + BR + 5tiles, no memmove, "},

                {{tst_layer, {1U, 1U, 6U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {VPUNN::Cycles::ERROR_TILE_OUTPUT, true, 3000, 3000 + 800},
                 " SOHO + 6tiles, no memmove, "},

                {{tst_layer, {1U, 1U, 4U, VPUTilingStrategy::SOK, false, false, prefetch}},
                 {Cycles::NO_ERROR, true, 1000, 1000 + 600},
                 " SOK, no memmove, "},
        };
        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTestNPU4x, DISABLED_ERROR_TILE_OUTPUT_LayerInvestigation_SOK_NPU40) {
    const VPUNN::DPUWorkload wl1{
            VPUNN::VPUDevice::VPU_4_0,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(13, 65, 960, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(13, 65, 320, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {1, 1},                                                      // kernels
            {1, 1},                                                      // strides
            {0, 0, 0, 0},                                                // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                          // execution mode
            VPUNN::ActivationFunction::NONE,                             // activation
            0.0F,                                                        // act_sparsity
            0.0F,                                                        // weight_sparsity
            {swz_def, swz_def},                                          // input_swizzling
            {swz_def},                                                   // output_swizzling
            1,                                                           // output_write_tiles
            {0, 0, 0, 0},                                                // offsets
            VPUNN::ISIStrategy::CLUSTERING,                              // isi_strategy
            false,                                                       // weight_sparsity_enabled
    };
    show_split = true;
    const bool prefetch{true};
    {
        const VPUNN::DPULayer tst_layer(wl1);
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 9000, 9000 + 1000},
                 " SOHO , no memmove, "},

                {{tst_layer, {1U, 1U, 4U, VPUTilingStrategy::SOK, false, false, prefetch}},
                 {Cycles::ERROR_TILE_OUTPUT, true, 9000, 9000 + 1000},
                 " SOK + 4tiles, no memmove, "},

                {{tst_layer, {1U, 1U, 5U, VPUTilingStrategy::SOK, false, false, prefetch}},
                 {Cycles::ERROR_TILE_OUTPUT, true, 9000, 9000 + 1000},
                 " SOK + 5tiles, no memmove, "},

                {{tst_layer, {1U, 1U, 6U, VPUTilingStrategy::SOK, false, false, prefetch}},
                 {Cycles::ERROR_TILE_OUTPUT, true, 9000, 9000 + 1000},
                 " SOK + 6tiles, no memmove, "},
        };
        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTestNPU4x, CONVOLUTION_Concrete_Multiply8641_NPU40_mock) {
    const VPUNN::DPULayer tst_layer(VPUNN::VPUDevice::VPU_4_0, VPUNN::Operation::CONVOLUTION,
                                    {VPUNN::VPUTensor(14, 14, 512, 1, VPUNN::DataType::UINT8)},  // input dimensions
                                    {VPUNN::VPUTensor(7, 7, 512, 1, VPUNN::DataType::UINT8)},    // output dimensions
                                    {3, 3},                                                      // kernels
                                    {2, 2},                                                      // strides
                                    {1, 0, 1, 0}                                                 // padding
    );

    const VPUNN::VPULayerStrategy tst_strategy{1U,    1U,    2U /*tiles*/, VPUNN::VPUTilingStrategy::NONE,
                                               false, false, true};  // prefetching  true = only DPU data, no DMA.

    {  // SOH
        TestInput tin{tst_layer, tst_strategy};
        tin.strategy.tiling_strategy = VPUNN::VPUTilingStrategy::SOH_Overlapped;

        TestExpectations texp{VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 1U, 1U};

        DoRegularTest(tin, texp, "SOH with errors");
    }
    // same as for VPU 2.7

    show_split = true;
    const std::vector<TestCase> tests{
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
             {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
             "CLUSTERING , no memmove, with errors"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
             {NO_ERROR_EXPECTED, true, 28500, 37313 + 3700},  // NEED ground truth on this NPU40.  GTL 37313
             "SOK , no memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
             {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
             "SOH ,no memmove, with errors"},
    };

    executeTests(tests);
}

// this test is used to split a layer into tiles
// we use these tiles in some regression tests, in class Regression_tests_MAXPOOL_NPU40
TEST_F(VPULayerCostModelTestNPU4x, DISABLED_Maxpool_layer_split_NPU40) {
    const VPUNN::DPUWorkload wl_layer{
            VPUNN::VPUDevice::VPU_4_0,
            VPUNN::Operation::MAXPOOL,
            {VPUNN::VPUTensor(54, 54, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(53, 53, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                     // kernels
            {1, 1},                                                     // strides
            {1, 0, 1, 0},                                               // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                         // execution mode
            VPUNN::ActivationFunction::NONE,                            // activation
            0.0F,                                                       // act_sparsity
            0.0F,                                                       // weight_sparsity
            {swz_def, swz_def},                                         // input_swizzling
            {swz_def},                                                  // output_swizzling
            1,                                                          // output_write_tiles
            {0, 0, 0, 0},                                               // offsets
            VPUNN::ISIStrategy::CLUSTERING,                             // isi_strategy
            false,                                                      // weight_sparsity_enabled
    };

    bool prefetch{true};
    show_split = true;
    unsigned int no_fail = 1;
    // VPULayerCostModel& theModel = model_4_0;
    {
        const VPUNN::DPULayer tst_layer(wl_layer);
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 1U, VPUNN::VPUTilingStrategy::NONE, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 14000U, 14000U * no_fail + 1000U},  // 14492 V17:
                 "Full, no memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 5500U, 5500U * no_fail + 1000U},  // 6062 V17:
                 "SOHO /2, no memmove, "},
                {{tst_layer, {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 2500U, 2500U * no_fail + 1000U},  // 3168 v17:
                 "SOHO /4 , no memmove, "},

                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 7500U, 7500U * no_fail + 1000U},  // 8228 v17:
                 "SOK , no memmove, "},

        };
        executeTests(tests);
    }
}
TEST_F(VPULayerCostModelTestNPU4x, ELTWISE_diff_swizz_NPU40) {
    auto generate_wl = [](unsigned int ch, ExecutionMode exec, Swizzling in0, Swizzling in1, Swizzling out0) {
        DPUWorkload wl{
                VPUNN::VPUDevice::VPU_4_0,
                Operation::ELTWISE,
                {VPUTensor(112, ch, 32, 1, DataType::UINT8, Layout::ZXY)},  // input dimensions
                {VPUTensor(112, ch, 32, 1, DataType::UINT8, Layout::ZXY)},  // output dimensions
                {1, 1},                                                     // kernels
                {1, 1},                                                     // strides
                {0, 0, 0, 0},                                               // padding
                exec,                                                       // execution mode
                ActivationFunction::RELU,                                   // activation
                0.0F,                                                       // act_sparsity
                0.0F,                                                       // weight_sparsity
                {in0, in1},                                                 // input_swizzling
                {out0},                                                     // output_swizzling
                1,                                                          // output_write_tiles
                {0, 0, 0, 0},                                               // offsets
                ISIStrategy::CLUSTERING,                                    // isi_strategy
                false,                                                      // weight_sparsity_enabled
        };
        return wl;
    };
    auto test_message = [](const DPUWorkload& wl, const std::string& text) {
        // clang-format off
        std::string message = text +
                              " Operation:" + Operation_ToText.at(static_cast<int>(wl.op)) + 
                              " input_swizzling: {" +
                               Swizzling_ToText.at(static_cast<int>(wl.input_swizzling[0])) + ", "
                             + Swizzling_ToText.at(static_cast<int>(wl.input_swizzling[1])) + "} ;\n"  

                             + " output_swizzling: " 
                             + " :  {" + Swizzling_ToText.at(static_cast<int>(wl.output_swizzling[0])) + "} \n" ;

        // clang-format on

        return message;
    };

    DPUWorkload wl_swizz_000{
            generate_wl(16, ExecutionMode::CUBOID_16x16, Swizzling::KEY_0, Swizzling::KEY_0, Swizzling::KEY_0)};
    DPUWorkload wl_swizz_555{
            generate_wl(16, ExecutionMode::CUBOID_16x16, Swizzling::KEY_5, Swizzling::KEY_5, Swizzling::KEY_5)};
    DPUWorkload wl_swizz_005{
            generate_wl(16, ExecutionMode::CUBOID_16x16, Swizzling::KEY_0, Swizzling::KEY_0, Swizzling::KEY_5)};
    DPUWorkload wl_swizz_550{
            generate_wl(16, ExecutionMode::CUBOID_16x16, Swizzling::KEY_5, Swizzling::KEY_5, Swizzling::KEY_0)};
    DPUWorkload wl_swizz_055{
            generate_wl(16, ExecutionMode::CUBOID_16x16, Swizzling::KEY_0, Swizzling::KEY_5, Swizzling::KEY_5)};

    const bool prefetch{true};
    VPULayerCostModel& theModel = layer_models.getModel(VPUDevice::VPU_4_0);
    const std::string nline{
            "\n ------------------------------------------- TEST ------------------------------------------- \n"};
    const bool force_fail{false};
    show_split = false;

    auto run_layer = [=, &theModel](const DPUWorkload& wl, const VPUTilingStrategy tilStrtgy, std::string text) {
        std::cout << nline << " " << text;
        DPULayer tst_layer(wl);
        const VPULayerStrategy strategy{1U, 1U, 2U, tilStrtgy, false, false, prefetch};

        Logger::clear2ndlog();
        CyclesInterfaceType cost_cyc{};
        LayerSplitInfo detailed_split;
        ASSERT_NO_THROW(cost_cyc = theModel.Layer(tst_layer, strategy.tiling_strategy, strategy.nDPUs, strategy.nTiles,
                                                  false, false, prefetch, detailed_split))
                << tst_layer << strategy << cost_cyc;
        if (force_fail) {
            EXPECT_EQ(cost_cyc, 1) << tst_layer << strategy
                                   << (show_split ? toStringLayerSplitInfo(detailed_split) : "\n");
        }

        std::cout << " \n:" << text << " ,cyc: " << cost_cyc;
    };

    // element wise
    run_layer(wl_swizz_000, VPUTilingStrategy::SOH_Overlapped, test_message(wl_swizz_000, "SOHO "));  // 878
    run_layer(wl_swizz_000, VPUTilingStrategy::NONE, test_message(wl_swizz_000, "NONE "));            // 1914

    run_layer(wl_swizz_555, VPUTilingStrategy::SOH_Overlapped, test_message(wl_swizz_555, "SOHO "));  // 954
    run_layer(wl_swizz_555, VPUTilingStrategy::NONE, test_message(wl_swizz_555, "NONE "));            // 1782

    run_layer(wl_swizz_005, VPUTilingStrategy::SOH_Overlapped, test_message(wl_swizz_005, "SOHO "));  // 891
    run_layer(wl_swizz_005, VPUTilingStrategy::NONE, test_message(wl_swizz_005, "NONE "));            // 1775

    run_layer(wl_swizz_550, VPUTilingStrategy::SOH_Overlapped, test_message(wl_swizz_550, "SOHO "));  // 857
    run_layer(wl_swizz_550, VPUTilingStrategy::NONE, test_message(wl_swizz_550, "NONE "));            // 1767

    run_layer(wl_swizz_055, VPUTilingStrategy::SOH_Overlapped, test_message(wl_swizz_055, "SOHO "));  // 891
    run_layer(wl_swizz_055, VPUTilingStrategy::NONE, test_message(wl_swizz_055, "NONE "));            // 1775

    // tiles
    DPUWorkload tile_swizz_000{
            generate_wl(8, ExecutionMode::CUBOID_8x16, Swizzling::KEY_0, Swizzling::KEY_0, Swizzling::KEY_0)};
    DPUWorkload tile_swizz_555{
            generate_wl(8, ExecutionMode::CUBOID_8x16, Swizzling::KEY_5, Swizzling::KEY_5, Swizzling::KEY_5)};
    DPUWorkload tile_swizz_005{
            generate_wl(8, ExecutionMode::CUBOID_8x16, Swizzling::KEY_0, Swizzling::KEY_0, Swizzling::KEY_5)};
    DPUWorkload tile_swizz_550{
            generate_wl(8, ExecutionMode::CUBOID_8x16, Swizzling::KEY_5, Swizzling::KEY_5, Swizzling::KEY_0)};
    DPUWorkload tile_swizz_055{
            generate_wl(8, ExecutionMode::CUBOID_8x16, Swizzling::KEY_0, Swizzling::KEY_5, Swizzling::KEY_5)};

    auto run_dpu_wl = [=, &theModel](const DPUWorkload& wl, std::string text) {
        std::cout << std::move(nline) << " " << text;
        std::string err_info;
        DPUWorkload tst_wl{wl};
        CyclesInterfaceType dpu_cost = theModel.get_cost_model().DPU(tst_wl, err_info);

        if (force_fail) {
            EXPECT_EQ(dpu_cost, 10) << dpu_cost << "=" << Cycles::toErrorText(dpu_cost) << tst_wl << err_info;
        }

        std::cout << " \n:" << text << " ,cyc: " << dpu_cost << " = " << Cycles::toErrorText(dpu_cost);
    };

    run_dpu_wl(tile_swizz_000, test_message(tile_swizz_000, "CLU "));  // 878
    run_dpu_wl(tile_swizz_555, test_message(tile_swizz_555, "CLU "));  // 954
    run_dpu_wl(tile_swizz_005, test_message(tile_swizz_005, "CLU "));  // 891
    run_dpu_wl(tile_swizz_550, test_message(tile_swizz_550, "CLU "));  // 857
    run_dpu_wl(tile_swizz_055, test_message(tile_swizz_055, "CLU "));  // 954
}

TEST_F(VPULayerCostModelTestNPU4x, ZeroNumberOfTiles_Layer_Test_NPU40) {
    VPUNN::DPUWorkload wl{
            VPUNN::VPUDevice::VPU_4_0,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(64, 64, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(21, 21, 16, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                     // kernels
            {3, 3},                                                     // strides
            {0, 0, 0, 0},                                               // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                         // execution mode
            VPUNN::ActivationFunction::NONE,                            // activation
            0.0F,                                                       // act_sparsity
            0.0F,                                                       // weight_sparsity
            {swz_def, swz_def},                                         // input_swizzling
            {swz_def},                                                  // output_swizzling
            1,                                                          // output_write_tiles
            {0, 0, 0, 0},                                               // offsets
            VPUNN::ISIStrategy::CLUSTERING,                             // isi_strategy
            false                                                       // weight_sparsity_enabled
    };
    {
        auto tst_layer{DPULayer(wl)};

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 0U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
                 {VPUNN::Cycles::ERROR_L2_INVALID_PARAMETERS, true, 55000, 55000 + 8000},
                 "NONE , + memmove, "},
                {{tst_layer, {1U, 1U, 0U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, false}},
                 {VPUNN::Cycles::ERROR_L2_INVALID_PARAMETERS, true, 30000, 30000 + 3000},
                 "SOHO , + memmove, "},
                {{tst_layer, {1U, 1U, 0U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
                 {VPUNN::Cycles::ERROR_L2_INVALID_PARAMETERS, true, 55000, 55000 + 8000},
                 "SOK , + memmove, "},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTestNPU4x, Extreme_values_Layer_Test_NPU40) {
    const VPUNN::DPUWorkload wl = {
            VPUNN::VPUDevice::VPU_4_0,
            VPUNN::Operation::ELTWISE,
            {VPUNN::VPUTensor(0, 0, 0, 1, VPUNN::DataType::UINT8, Layout::ZXY)},  // input dimensions
            {VPUNN::VPUTensor(0, 0, 0, 1, VPUNN::DataType::UINT8, Layout::ZXY)},  // output dimensions
            {0, 0},                                                               // kernels
            {0, 0},                                                               // strides
            {0, 0, 0, 0},                                                         // padding
            VPUNN::ExecutionMode::CUBOID_8x16,                                    // execution mode
            VPUNN::ActivationFunction::NONE,                                      // activation
            0.0F,                                                                 // act_sparsity
            0.0F,                                                                 // weight_sparsity
            {Swizzling::KEY_5, Swizzling::KEY_5},                                 // input_swizzling
            {Swizzling::KEY_5},                                                   // output_swizzling
            0,                                                                    // output_write_tiles
            {0, 0, 0, 0},                                                         // offsets
            VPUNN::ISIStrategy::CLUSTERING,                                       // isi_strategy
            true,                                                                 // weight_sparsity_enabled

    };

    VPULayerCostModel& theModel = layer_models.getModel(VPUDevice::VPU_4_0);
    Logger::clear2ndlog();
    DPULayer tst_layer(wl);

    unsigned cyc{};
    LayerSplitInfo detailed_split;

    cyc = theModel.Layer(tst_layer, VPUNN::VPUTilingStrategy::SOH_Overlapped, 1U, 2U, false, false, true,
                         detailed_split);
    EXPECT_TRUE(Cycles::isErrorCode(cyc));
    // EXPECT_TRUE(false) << Cycles::toErrorText(cyc);
}

TEST_F(VPULayerCostModelTestNPU4x, Layer_PRE_split_Shave) {
    const VPUNN::SHAVEWorkload test_shave_wl{
            "sigmoid",
            VPUDevice::VPU_4_0,
            {VPUNN::VPUTensor(60, 7, 512, 1, VPUNN::DataType::FLOAT16)},
            {VPUNN::VPUTensor(60, 7, 512, 1, VPUNN::DataType::FLOAT16)},
    };
    VPULayerCostModel& theModel = layer_models.getModel(VPUDevice::VPU_4_0);

    const std::vector<SHAVEWorkload> splitLayers{test_shave_wl, test_shave_wl, test_shave_wl};
    CyclesInterfaceType pre_split_cost = theModel.LayersPreSplit(splitLayers, 2, true, true);

    EXPECT_GT(pre_split_cost, Cycles::NO_ERROR);
    EXPECT_LE(pre_split_cost, Cycles::START_ERROR_RANGE);
}

TEST_F(VPULayerCostModelTestNPU4x, Dual_Sparsity_Active_Layer_Test_NPU40) {
    // VPUNN::DPUWorkload wl_ref_2_7 = {
    //         VPUNN::VPUDevice::VPU_2_7,
    //         VPUNN::Operation::CONVOLUTION,
    //         {VPUNN::VPUTensor(5, 5, 100, 1, VPUNN::DataType::UINT8)},  // input dimensions
    //         {VPUNN::VPUTensor(3, 3, 50, 1, VPUNN::DataType::UINT8)},   // output dimensions
    //         {3, 3},                                                    // kernels
    //         {1, 1},                                                    // strides
    //         {0, 0, 0, 0},                                              // padding
    //         VPUNN::ExecutionMode::CUBOID_16x16,                        // execution mode
    //         VPUNN::ActivationFunction::NONE,                           // activation
    //         0.0F,                                                      // act_sparsity
    //         0.0F,                                                      // weight_sparsity
    //         {swz_def, swz_def},                                        // input_swizzling
    //         {swz_def},                                                 // output_swizzling
    //         1,                                                         // output_write_tiles
    //         {0, 0, 0, 0},                                              // offsets
    //         VPUNN::ISIStrategy::CLUSTERING,                            // isi_strategy
    //         false,                                                     // weight_sparsity_enabled
    // };

    VPUNN::DPUWorkload wl_ref_4_0{
            VPUNN::VPUDevice::VPU_4_0,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(64, 64, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(21, 21, 16, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                     // kernels
            {3, 3},                                                     // strides
            {0, 0, 0, 0},                                               // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                         // execution mode
            VPUNN::ActivationFunction::NONE,                            // activation
            0.0F,                                                       // act_sparsity
            0.0F,                                                       // weight_sparsity
            {swz_def, swz_def},                                         // input_swizzling
            {swz_def},                                                  // output_swizzling
            1,                                                          // output_write_tiles
            {0, 0, 0, 0},                                               // offsets
            VPUNN::ISIStrategy::CLUSTERING,                             // isi_strategy
            false                                                       // weight_sparsity_enabled
    };

    struct TestInput {
        VPUNN::DPUWorkload wl;  // the wl for which we compute memory
    };

    struct TestCase {
        TestInput t_in;
        std::string info = "";
    };

    const VPULayerStrategy strategy{1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true};
    auto verify_sparsity_influence = [&strategy](const TestCase& t, VPULayerCostModel& theModel) {
        DPULayer tst_layer(t.t_in.wl);
        Logger::clear2ndlog();
        CyclesInterfaceType cost_cyc{};
        LayerSplitInfo detailed_split;

        ASSERT_NO_THROW(cost_cyc = theModel.Layer(tst_layer, strategy.tiling_strategy, strategy.nDPUs, strategy.nTiles,
                                                  false, false, true, detailed_split))
                << tst_layer << strategy << cost_cyc;

        EXPECT_FALSE(Cycles::isErrorCode(cost_cyc)) << t.info;
    };

    ///@brief this lambda function executes a given lambda function as a parameter on each test case in a test vector
    ///@param tests a test vector
    ///@param testChecker is a lambda function
    auto run_Tests = [](const std::vector<TestCase>& tests, VPULayerCostModel& theModel, auto testCheck) {
        for (const auto& t : tests) {
            testCheck(t, theModel);
        }
    };

    {
        const std::vector<TestCase> tests4_0 = {
                // clang-format off
            /************************************************ TABLE HEADER ********************************************************/
        /*  ||                              workload                          ||         test info         ||   */

            {{wl_sparsity_initialization(wl_ref_4_0,  false, 0.0F, false, 0.0F) }, "Device 4_0: No sparsity active" },
            {{wl_sparsity_initialization(wl_ref_4_0,  true, 0.7F, false, 0.0F)}, "Device 4_0: Input sparsity active"},
            {{wl_sparsity_initialization(wl_ref_4_0,  false, 0.0F, true, 0.6F)}, "Device 4_0: Weight sparsity active" },
            {{wl_sparsity_initialization(wl_ref_4_0, true, 0.2F, true, 0.4F)}, "Device 4_0: Input + Weight sparsity"},
            {{wl_sparsity_initialization(wl_ref_4_0, true, 0.2F, true, 0.4F)}, "Device 4_0: Input + Weight sparsity"},

                // clang-format on
        };

        VPULayerCostModel& theModel = layer_models.getModel(VPUDevice::VPU_4_0);
        run_Tests(tests4_0, theModel, verify_sparsity_influence);
    }
}

TEST_F(VPULayerCostModelTestNPU4x, Layer_PRE_split_L1) {
    const VPUDevice device{VPUDevice::VPU_4_0};
    auto makewl = [&](int channels_out) {
        return DPUWorkload{
                device,
                Operation::CONVOLUTION,
                {VPUTensor(20, 20, 512, 1, DataType::UINT8)},                          // input dimensions
                {VPUTensor(20, 20, channels_out, 1, DataType::FLOAT16, Layout::XYZ)},  // output dimensions
                {1, 1},                                                                // kernels
                {1, 1},                                                                // strides
                {0, 0, 0, 0},                                                          // padding
                ExecutionMode::CUBOID_4x16,                                            // execution mode
                ActivationFunction::NONE,                                              // activation
                0.0F,                                                                  // act_sparsity
                0.0F,                                                                  // weight_sparsity
                {swz_def, swz_def},                                                    // input_swizzling
                {swz_def},                                                             // output_swizzling
                3,                                                                     // output_write_tiles
                {0, 0, 0, 0},                                                          // offsets
                ISIStrategy::SPLIT_OVER_K,                                             // isi_strategy
                false,                                                                 // weight_sparsity_enabled
                {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},        // halo aspects
                {false, {0, 0, 0, 0}, {0, 0, 0, 0}, false},                            // SEP
                DataType::UINT8,                                                       // input1 data type
                "",                                                                    // layer_info
                false,  ///< operation does not have weights
                false,  // in_place_output_memory{};
                true    // superdense
        };
    };
    const bool prefetch{true};  // prefetch was done
    VPULayerCostModel& theModel = layer_models.getModel(VPUDevice::VPU_4_0);

    // these 3 workloads should have the same result (wl kept unchanged) when calling L1 and L2 pre split with them as
    // layers
    std::vector<DPUWorkload> wls{makewl(96), makewl(80), makewl(80)};

    std::vector<CyclesInterfaceType> costsL1{};
    {  // call L1 x3
        std::string info{};
        for (const auto& w : wls) {
            const auto res = theModel.get_cost_model().DPU(w, info);
            costsL1.push_back(res);
            ASSERT_TRUE(!Cycles::isErrorCode(res)) << res << " " << info;
        }
    }

    {  // L2 pre split
        std::vector<DPULayer> splitLayersInput{};
        for (const auto& w : wls) {
            splitLayersInput.push_back(DPULayer(w));
        }

        Logger::clear2ndlog();
        CyclesInterfaceType layer_cost{};
        LayerSplitInfo detailed_split{};
        ASSERT_NO_THROW(layer_cost =
                                theModel.LayersPreSplit(splitLayersInput, 1U, false, false, prefetch, detailed_split))
                << toStringLayerSplitInfo(detailed_split);

        ASSERT_TRUE(!Cycles::isErrorCode(layer_cost)) << layer_cost << " " << toStringLayerSplitInfo(detailed_split);

        EXPECT_EQ(costsL1.size(), detailed_split.size());

        {
            int i = 0;
            const DPUWorkload& original_wl{wls[i]};
            const DPULayer& inLyr{splitLayersInput[i]};
            const OneTileLayerInfo& t{detailed_split[i]};
            const DPULayer& outLyr{t.inter_tile_split_layer};
            EXPECT_EQ(1, t.best_intra_tile_split.second.size());  // no intratile split
            CyclesInterfaceType cost_layer_best{t.best_intra_tile_split.first};
            const DPUWorkload& winner_wl{t.best_intra_tile_split.second[0]};

            EXPECT_EQ(costsL1[i], cost_layer_best)
                    << toStringLayerSplitInfo(detailed_split) << Logger::get2ndlog() << "END ERR";

            EXPECT_EQ(original_wl, winner_wl)
                    << toStringLayerSplitInfo(detailed_split) << Logger::get2ndlog() << "END ERR";

            EXPECT_EQ(inLyr, outLyr) << toStringLayerSplitInfo(detailed_split) << Logger::get2ndlog() << "END ERR";
        }
    }
}

TEST_F(VPULayerCostModelTestNPU4x, AVEPOOL_Layer_PRE_split_L2) {
    const VPUDevice device{VPUDevice::VPU_4_0};
    const DPUWorkload wl{
            device,
            Operation::AVEPOOL,
            {VPUTensor(4, 21, 5120, 1, DataType::UINT8, Layout::ZXY)},    // input dimensions
            {VPUTensor(4, 21, 5120, 1, DataType::FLOAT16, Layout::ZXY)},  // output dimensions
            {1, 1},                                                       // kernels
            {1, 1},                                                       // strides
            {0, 0, 0, 0},                                                 // padding
            ExecutionMode::CUBOID_16x16,                                  // execution mode
            ActivationFunction::NONE,                                     // activation
            0.0F,                                                         // act_sparsity
            0.0F,                                                         // weight_sparsity
            {Swizzling::KEY_0, Swizzling::KEY_0},                         // input_swizzling
            {Swizzling::KEY_0},                                           // output_swizzling
            1,                                                            // output_write_tiles
            {0, 0, 0, 0},                                                 // offsets
            ISIStrategy::CLUSTERING,                                      // isi_strategy
            false,                                                        // weight_sparsity_enabled
    };

    // Split Layer
    //     const DPUWorkload wl_split{
    //         device,
    //         Operation::DW_CONVOLUTION,
    //         {VPUTensor(4, 21, 64, 1, DataType::UINT8, Layout::ZXY)},    // input dimensions
    //         {VPUTensor(4, 21, 64, 1, DataType::FLOAT16, Layout::ZXY)},  // output dimensions
    //         {1, 1},                                                       // kernels
    //         {1, 1},                                                       // strides
    //         {0, 0, 0, 0},                                                 // padding
    //         ExecutionMode::CUBOID_16x16,                                  // execution mode
    //         ActivationFunction::NONE,                                     // activation
    //         0.0F,                                                         // act_sparsity
    //         0.0F,                                                         // weight_sparsity
    //         {Swizzling::KEY_0, Swizzling::KEY_0},                         // input_swizzling
    //         {Swizzling::KEY_0},                                           // output_swizzling
    //         1,                                                            // output_write_tiles
    //         {0, 0, 0, 0},                                                 // offsets
    //         ISIStrategy::CLUSTERING,                                      // isi_strategy
    //         false,                                                        // weight_sparsity_enabled
    // };

    const bool prefetch{true};  // prefetch was done
    VPULayerCostModel& theModel = layer_models.getModel(VPUDevice::VPU_4_0);
    const VPUTilingStrategy strategy{VPUNN::VPUTilingStrategy::NONE};

    // call L2 split
    DPULayer wl_layer(wl);
    Logger::clear2ndlog();
    CyclesInterfaceType cost_cyc_Layer{};
    LayerSplitInfo detailed_split_layer;
    ASSERT_NO_THROW(cost_cyc_Layer =
                            theModel.Layer(wl_layer, strategy, 1U, 2U, false, false, prefetch, detailed_split_layer))
            << wl_layer << (int)strategy << " : " << VPUTilingStrategy_ToText.at(static_cast<int>(strategy));

    ASSERT_TRUE(!Cycles::isErrorCode(cost_cyc_Layer)) << toStringLayerSplitInfo(detailed_split_layer);

    // L2 pre split
    Logger::clear2ndlog();
    CyclesInterfaceType pre_split_cost1{};
    LayerSplitInfo detailed_split_pre_layer{};
    std::vector<DPULayer> splitLayers1{std::move(wl_layer)};
    ASSERT_NO_THROW(pre_split_cost1 =
                            theModel.LayersPreSplit(splitLayers1, 1U, false, false, prefetch, detailed_split_pre_layer))
            << toStringLayerSplitInfo(detailed_split_pre_layer);

    ASSERT_TRUE(!Cycles::isErrorCode(pre_split_cost1))
            << pre_split_cost1 << " " << toStringLayerSplitInfo(detailed_split_pre_layer);

    EXPECT_EQ(pre_split_cost1, cost_cyc_Layer)
            << toStringLayerSplitInfo(detailed_split_layer) << toStringLayerSplitInfo(detailed_split_pre_layer)
            << Logger::get2ndlog() << "END ERR";
}

}  // namespace VPUNN_unit_tests