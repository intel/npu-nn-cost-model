// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include "vpu_layer_cost_model.h"

#include <gtest/gtest.h>
#include <sstream>  // for error formating
#include "common_helpers.h"

#include "layer_test.h"
#include "vpu/shave/layers.h"

namespace VPUNN_unit_tests {
using namespace VPUNN;

TEST_F(VPULayerCostModelTest, DISABLED_ERROR_TILE_OUTPUT_LayerInvestigation_SOHO_NPU40) {
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

TEST_F(VPULayerCostModelTest, DISABLED_ERROR_TILE_OUTPUT_LayerInvestigation_SOK_NPU40) {
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

TEST_F(VPULayerCostModelTest, LayerLoadModels) {
    EXPECT_EQ(model_2_0.get_cost_model().nn_initialized(), true);
    EXPECT_EQ(model_2_7.get_cost_model().nn_initialized(), true);
    EXPECT_EQ(model_2_7_no_dma.get_cost_model().nn_initialized(), true);
    EXPECT_EQ(model_4_0.get_cost_model().nn_initialized(), true);
}

TEST_F(VPULayerCostModelTest, LayerCostModelVPU_2_0) {
    auto layer = generate_helper_layer(16, 64);
    auto vpu20_layer_cost = model_2_0.Layer(layer, 5, 4);

    // Basic expectations
    EXPECT_GT(vpu20_layer_cost, 0u);
}

TEST_F(VPULayerCostModelTest, LayerCostModelVPU_2_7_shv) {
    auto layer = generate_helper_sw_layer(16, 64);
    auto vpu20_layer_cost = model_2_7.Layer(layer, 5, 4);

    // Basic expectations
    EXPECT_GT(vpu20_layer_cost, 0u);
}

TEST_F(VPULayerCostModelTest, LayerCostModelVPU_2_7_shv_workload) {
    auto layer = generate_helper_shave_wl_layer(VPUNN::VPUDevice::VPU_2_7, 16, 64);
    auto vpu20_layer_cost = model_2_7.Layer(layer, 5, 4);

    // Basic expectations
    EXPECT_GT(vpu20_layer_cost, 0u);
}

TEST_F(VPULayerCostModelTest, LayerCostModelVPU_2_7_shv_wl_bad_name) {
    auto layer = VPUNN::SHAVEWorkload("bad_wl",                                                     // name
                                      VPUNN::VPUDevice::VPU_2_7,                                    // VPU device
                                      {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::FLOAT16)},  // Input tensor
                                      {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::FLOAT16)}   // Output tensor
    );
    auto vpu20_layer_cost = model_2_7.Layer(layer, 5, 4);

    // Basic expectations
    EXPECT_EQ(vpu20_layer_cost, Cycles::ERROR_SHAVE);
}

TEST_F(VPULayerCostModelTest, ELTWISE_Concrete_Add14_VPU27) {
    const VPUNN::DPULayer tst_layer(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::ELTWISE,
                                    {VPUNN::VPUTensor(56, 56, 256, 1, VPUNN::DataType::UINT8)},  // input dimensions
                                    {VPUNN::VPUTensor(56, 56, 256, 1, VPUNN::DataType::UINT8)},  // output dimensions
                                    {1, 1},                                                      // kernels
                                    {1, 1},                                                      // strides
                                    {0, 0, 0, 0}                                                 // padding
    );

    // EXPECT_TRUE(false);

    {
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUTilingStrategy::NONE, false, false, true}},
                 {NO_ERROR_EXPECTED, true, 22500, 22500 + 2250},  // v16 23774, v17 22844   //v150 23213
                 "Clustering, no mem move"},
                {{tst_layer, {1U, 1U, 2U, VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {NO_ERROR_EXPECTED, true, 10000, 10000 + 2500},  // v16 11763, v17 12193   //v150 10893
                 "SOHO, no mem move"},
                {{tst_layer,
                  {1U, 1U, 2U, VPUTilingStrategy::SOK, false, false,
                   true}},                                        // SOK is now allowed for element wise (not trained)
                 {NO_ERROR_EXPECTED, true, 11000, 11000 + 2000},  // v16 17458, v17 16994  //v150 11751    GT??
                                                                  // goes back to CLU OWT=1
                 "SOK tentative on elementwise, no mem move"},
        };

        executeTests(tests);
    }
    {  // what is the SOK eqiovalent if OWT is 1
        DPUWorkload wl{std::move(tst_layer)};
        wl.inputs[0] = VPUTensor(56, 56, 256 / 2, 1, VPUNN::DataType::UINT8);
        wl.outputs[0] = VPUTensor(56, 56, 256 / 2, 1, VPUNN::DataType::UINT8);  // SOK result
        wl.isi_strategy = ISIStrategy::CLUSTERING;
        const DPULayer tst_layer2{wl};

        wl.output_write_tiles = 2;
        const DPULayer tst_layer3{wl};  // CLU +OWT2 will become SOK, but elementwise will become CLU owt=1

        const std::vector<TestCase> tests{
                {{std::move(tst_layer2), {1U, 1U, 2U, VPUTilingStrategy::NONE, false, false, true}},
                 {NO_ERROR_EXPECTED, true, 11000, 11000 + 2000},  // v16 11844 , v17 11350 //v150 ???
                 "Clustering equivalent of prev SOK, but OWT =1, no mem move"},
                {{std::move(tst_layer3), {1U, 1U, 2U, VPUTilingStrategy::NONE, false, false, true}},
                 {NO_ERROR_EXPECTED, true, 11000, 11000 + 2000},  // v16 18902 , v17 18364  //v150 ???  not trained!
                                                                  //, should = first case CLU+OWT=1
                 "Clustering equivalent of prev SOK, but OWT =2 (+ ELM), no mem move"},

        };
        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, CONVOLUTION_Concrete_Multiply8641_VPU27) {
    const VPUNN::DPULayer tst_layer(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
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

        TestExpectations texp{VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0};

        DoRegularTest(tin, texp, "SOH with errors");
    }

    const std::vector<TestCase> tests{
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
             {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
             "CLUSTERING , no memmove, with errors"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
             {NO_ERROR_EXPECTED, true, 37000, 37000 + 3500},
             "SOK , no memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
             {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
             "SOH ,no memmove, with errors"},
    };

    executeTests(tests);
}

TEST_F(VPULayerCostModelTest, CONVOLUTION_Concrete_Multiply8641_NPU40_mock) {
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

TEST_F(VPULayerCostModelTest, AVEPOOL_Concrete_GlobalAveragePool_172_VPU27) {
    const VPUNN::DPULayer tst_layer(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::AVEPOOL,       // goes to DW Conv
                                    {VPUNN::VPUTensor(7, 7, 2048, 1, VPUNN::DataType::UINT8)},  // input dimensions
                                    {VPUNN::VPUTensor(1, 1, 2048, 1, VPUNN::DataType::UINT8)},  // output dimensions
                                    {7, 7},                                                     // kernels
                                    {1, 1},                                                     // strides
                                    {0, 0, 0, 0}                                                // padding
    );
    const VPUNN::VPULayerStrategy tst_strategy{1U,    1U,    2U /*tiles*/, VPUNN::VPUTilingStrategy::NONE,
                                               false, false, true};  // prefetching  true = only DPU data, no DMA.
    const auto origMax = getModel(VPUDevice::VPU_2_7).get_maxWorkloadsPerIntraTileSplit();
    getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(50);  // original test context was 50

    {  // SOK
        TestInput tin{tst_layer, tst_strategy};
        tin.strategy.tiling_strategy = VPUNN::VPUTilingStrategy::SOK;

        TestExpectations texp{NO_ERROR_EXPECTED, false, 14000, 14000 + 2500};  // v16:16032,  GT??

        DoRegularTest(tin, texp, "SOK ok");
    }

    const std::vector<TestCase> tests{
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
             {VPUNN::Cycles::NO_ERROR, false, 50000, 50000 + 5000},
             "CLUSTERING , no memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
             {NO_ERROR_EXPECTED, false, 14000, 14000 + 2300},  // v16 16300
             "SOK , no memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
             {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 52500, 52500 + 1000},  // GT ???
             "SOHO ,no memmove"},
    };

    executeTests(tests);

    getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(origMax);
}

TEST_F(VPULayerCostModelTest, AVEPOOL_Concrete_GlobalAveragePool_172_VPU27_SOH) {
    const VPUNN::DPULayer tst_layer(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::AVEPOOL,        // goes to DW Conv
                                    {VPUNN::VPUTensor(7, 15, 2048, 1, VPUNN::DataType::UINT8)},  // input dimensions
                                    {VPUNN::VPUTensor(1, 9, 2048, 1, VPUNN::DataType::UINT8)},   // output dimensions
                                    {7, 7},                                                      // kernels
                                    {1, 1},                                                      // strides
                                    {0, 0, 0, 0}                                                 // padding
    );

    const std::vector<TestCase> tests{
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
             {VPUNN::Cycles::NO_ERROR, false, 52000, 52000 + 7000},
             "CLUSTERING , no memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
             {NO_ERROR_EXPECTED, false, 20000, 20000 + 5000},  // GT ???
             "SOK , no memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
             {VPUNN::Cycles::NO_ERROR, true, 51000, 51000 + 6000},  //
             "SOHO ,no memmove"},
    };

    const auto origMax = getModel(VPUDevice::VPU_2_7).get_maxWorkloadsPerIntraTileSplit();
    getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(50);  // original test context was 50
    executeTests(tests);
    getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(origMax);
}

TEST_F(VPULayerCostModelTest, CONVOLUTION_Concrete_Multiply8648_VPU27) {
    const VPUNN::DPULayer tst_layer(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
                                    {VPUNN::VPUTensor(7, 7, 512, 1, VPUNN::DataType::UINT8)},   // input dimensions
                                    {VPUNN::VPUTensor(7, 7, 2048, 1, VPUNN::DataType::UINT8)},  // output dimensions
                                    {1, 1},                                                     // kernels
                                    {1, 1},                                                     // strides
                                    {0, 0, 0, 0}                                                // padding
    );
    const VPUNN::VPULayerStrategy tst_strategy{1U,    1U,    2U /*tiles*/, VPUNN::VPUTilingStrategy::NONE,
                                               false, false, true};  // prefetching  true = only DPU data, no DMA.

    {  // SOK
        TestInput tin{tst_layer, tst_strategy};
        tin.strategy.tiling_strategy = VPUNN::VPUTilingStrategy::SOK;

        TestExpectations texp{NO_ERROR_EXPECTED, false, 16000, 16000 + 3000};  //

        DoRegularTest(tin, texp, "SOK convolution");
    }

    const std::vector<TestCase> tests{
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
             {NO_ERROR_EXPECTED, false, 34000, 34000 + 2000},
             "CLUSTERING , no memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
             {NO_ERROR_EXPECTED, false, 79000, 79000 + 4000},  // fetching big data
             "CLUSTERING ,with fetch required"},

            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
             {NO_ERROR_EXPECTED, false, 16000, 16000 + 3000},  //
             "SOK , no memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
             {NO_ERROR_EXPECTED, false, 16000 + 22000, 16000 + 22000 + 3000},  // fetching big data
             "SOK , with fetch required"},
    };

    executeTests(tests);
}

TEST_F(VPULayerCostModelTest, MAXPOOL_avgpoolBased_172_VPU27) {
    const VPUNN::DPULayer tst_layer(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::MAXPOOL,
                                    {VPUNN::VPUTensor(7, 7, 2048, 1, VPUNN::DataType::UINT8)},  // input dimensions
                                    {VPUNN::VPUTensor(1, 1, 2048, 1, VPUNN::DataType::UINT8)},  // output dimensions
                                    {7, 7},                                                     // kernels
                                    {1, 1},                                                     // strides
                                    {0, 0, 0, 0}                                                // padding
    );

    const std::vector<TestCase> tests{
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
             {NO_ERROR_EXPECTED, false, 52000, 52000 + 18000},  // huge change
             "CLUSTERING , no mem"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
             {NO_ERROR_EXPECTED, false, 13000, 13000 + 7000},  // huge
             "SOK , no mem"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
             {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 0, 0 + 0},  // 7 cannot be split, must be 14
             "SOH ,no mem"},
    };

    const auto origMax = getModel(VPUDevice::VPU_2_7).get_maxWorkloadsPerIntraTileSplit();
    getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(50);  // original test context was 50
    executeTests(tests);
    getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(origMax);
}

// this test is used to split a layer into tiles
// we use these tiles in some regression tests, in class Regression_tests_MAXPOOL_NPU40
TEST_F(VPULayerCostModelTest, DISABLED_Maxpool_layer_split_NPU40) {
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

/// Test batch values for devices
TEST_F(VPULayerCostModelTest, BatchValues_LayerLevel) {
    auto mkLayer = [](VPUDevice dev, unsigned int b, Layout layout = Layout::ZXY) -> DPULayer {
        VPUNN::DPUWorkload wl{
                dev,
                Operation::CONVOLUTION,
                {VPUNN::VPUTensor(15, 50, 64, b, VPUNN::DataType::UINT8, layout)},  // input dimensions
                {VPUNN::VPUTensor(15, 50, 64, b, VPUNN::DataType::UINT8, layout)},  // output dimensions
                {1, 1},                                                             // kernels
                {1, 1},                                                             // strides
                {0, 0, 0, 0},                                                       // padding
                ExecutionMode::CUBOID_16x16,                                        // execution mode
                ActivationFunction::NONE,                                           // activation
                0.0F,                                                               // act_sparsity
                0.0F,                                                               // weight_sparsity
                {Swizzling::KEY_0, Swizzling::KEY_0},                               // input_swizzling
                {Swizzling::KEY_0},                                                 // output_swizzling
        };

        DPULayer layer(wl);
        return layer;
    };

    bool prefetch{true};
    show_split = true;
    unsigned int no_fail = 1;
    {
        const std::vector<TestCase> tests{
                 {{mkLayer(VPUDevice::VPU_2_0, 0), {1U, 1U, 1U, VPUNN::VPUTilingStrategy::NONE, false, false, prefetch}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 12500, 12500 * no_fail + 1000},
                 "Device 2.0, B=0 "},
                 {{mkLayer(VPUDevice::VPU_2_0, 1, Layout::ZMAJOR), {1U, 1U, 1U, VPUNN::VPUTilingStrategy::NONE, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 12500, 12500 * no_fail + 1000},
                 "Device 2.0, B=1 "},
                 {{mkLayer(VPUDevice::VPU_2_0, 2, Layout::ZMAJOR), {1U, 1U, 1U, VPUNN::VPUTilingStrategy::NONE, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 6500, 6500 * no_fail + 1000},
                 "Device 2.0, B=2 "},

                 {{mkLayer(VPUDevice::VPU_2_7, 0), {1U, 1U, 1U, VPUNN::VPUTilingStrategy::NONE, false, false, prefetch}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 12500, 12500 * no_fail + 1000},
                 "Device 2_7, B=0 "},
                 {{mkLayer(VPUDevice::VPU_2_7, 1), {1U, 1U, 1U, VPUNN::VPUTilingStrategy::NONE, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 1500, 1500 * no_fail + 1000},
                 "Device 2.7, B=1 "},
                 {{mkLayer(VPUDevice::VPU_2_7, 2), {1U, 1U, 1U, VPUNN::VPUTilingStrategy::NONE, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 1300, 1300 * no_fail + 1000},
                 "Device 2.7, B=2 "},

                 {{mkLayer(VPUDevice::VPU_4_0, 0), {1U, 1U, 1U, VPUNN::VPUTilingStrategy::NONE, false, false, prefetch}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 12500, 12500 * no_fail + 1000},
                 "Device 4.0, B=0 "},
                 {{mkLayer(VPUDevice::VPU_4_0, 1), {1U, 1U, 1U, VPUNN::VPUTilingStrategy::NONE, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 2000, 2000 * no_fail + 1000},
                 "Device 4.0, B=1 "},
                 {{mkLayer(VPUDevice::VPU_4_0, 2), {1U, 1U, 1U, VPUNN::VPUTilingStrategy::NONE, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 1200, 1200 * no_fail + 1000},
                 "Device 4.0, B=2 "},
               

        };
        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, MAXPOOL_avgpoolBased_172_VPU27_SOH) {  // SOH Split possible at limit
    const VPUNN::DPULayer tst_layer(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::MAXPOOL,
                                    {VPUNN::VPUTensor(7, 14, 2048, 1, VPUNN::DataType::UINT8)},  // input dimensions
                                    {VPUNN::VPUTensor(1, 8, 2048, 1, VPUNN::DataType::UINT8)},   // output dimensions
                                    {7, 7},                                                      // kernels
                                    {1, 1},                                                      // strides
                                    {0, 0, 0, 0}                                                 // padding
    );
    // EXPECT_TRUE(false);

    const std::vector<TestCase> tests{
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
             {NO_ERROR_EXPECTED, false, 51000, 51000 + 24000},  // huge  GT??
             "CLUSTERING , no mem"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
             {NO_ERROR_EXPECTED, false, 21000, 21000 + 3000},
             "SOK , no mem"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
             {NO_ERROR_EXPECTED, false, 51000, 51000 + 20000},  // huge GT??
             "SOH ,output H ?"},
    };

    const auto origMax = getModel(VPUDevice::VPU_2_7).get_maxWorkloadsPerIntraTileSplit();
    getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(50);  // original test context was 50
    executeTests(tests);
    getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(origMax);
}

TEST_F(VPULayerCostModelTest, AVEPOOL_Concrete_GlobalAveragePool_172_BAD_CHANNELS_VPU27) {
    const VPUNN::DPULayer tst_layer(
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::AVEPOOL,  // replace by DW conv , only outputs ch of 16-32-64 in a workload alowwed
            {VPUNN::VPUTensor(7, 7, 2047, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(1, 1, 2047, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {7, 7},                                                     // kernels
            {1, 1},                                                     // strides
            {0, 0, 0, 0}                                                // padding
    );
    // 2047 is not MOD16, split by 2 also is not MOD 16, so no intra tile z split is possible

    const std::vector<TestCase> tests{
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
             {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 2469000, 2469000 + 1000},
             "CLUSTERING no MOD16 , no memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
             {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 142000, 142000 + 1000},
             "SOK no MOD16, no memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
             {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 1821000, 1821000 + 1000},
             "SOH no MOD16,no memmove"},
    };

    executeTests(tests);
}

TEST_F(VPULayerCostModelTest, AVEPOOL_Concrete_GlobalAveragePool_172_MOD16_NOMOD32_CH_VPU27) {
    const VPUNN::DPULayer tst_layer(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::AVEPOOL,
            {VPUNN::VPUTensor(7, 7, 2048 - 16, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(1, 1, 2048 - 16, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {7, 7},                                                          // kernels
            {1, 1},                                                          // strides
            {0, 0, 0, 0}                                                     // padding
    );
    const VPUNN::VPULayerStrategy tst_strategy{1U,    1U,    2U /*tiles*/, VPUNN::VPUTilingStrategy::NONE,
                                               false, false, true};  // prefetching  true = only DPU data, no DMA.

    const auto origMax = getModel(VPUDevice::VPU_2_7).get_maxWorkloadsPerIntraTileSplit();
    getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(50);  // original test context was 50

    {  // CLUSTERING
        TestInput tin{tst_layer, tst_strategy};
        tin.strategy.tiling_strategy = VPUNN::VPUTilingStrategy::NONE;

        // Very large because 2032 is  mod 16, but 31X64 +1X48 is not accepted, 63x32+1X16 is not reached (limit 50)
        // and we cannot reach a split like: 31x64 = 1984  , 1x32=32,  1x16=16  =>2032 split in 33 workloads
        // the 1x2032 is not accepted because ch not 16/32/64
        TestExpectations texp{VPUNN::Cycles::ERROR_TILE_OUTPUT, true, 0, 0 + 0};

        DoRegularTest(tin, texp, "CLUST not valid 1 workload");
    }

    {  // SOK
        TestInput tin{tst_layer, tst_strategy};
        tin.strategy.tiling_strategy = VPUNN::VPUTilingStrategy::SOK;

        // 2032 split to tile like 1024 and 1008, both div by 16
        // 1024 = 16x64
        // 1008 :   15x64 , 1x48 : not allowed;   31x32, 1x16  reached,    but  15x64 , 1x32, 1x16 : not possible
        // but best.   due to not possible one we get 14000(or 16k) instead of 8000
        TestExpectations texp{NO_ERROR_EXPECTED, false, 14000, 14000 + 2300};

        DoRegularTest(tin, texp, "SOK not optimum: 15x64 , 1x32, 1x16  = 1008 ");
    }

    const std::vector<TestCase> tests{
            //         {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
            //          {NO_ERROR_EXPECTED, false, 2473000, 2474000},
            //          "CLUSTERING , no memmove"},
            //         {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
            //          {NO_ERROR_EXPECTED, false, 142000, 143000},
            //          "SOK , no memmove"},
            {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
             {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 0, 0},
             "SOHO ,no memmove"},
    };

    executeTests(tests);

    getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(origMax);
}

TEST_F(VPULayerCostModelTest, AVEPOOL_Concrete_GlobalAveragePool_172_MOD16_NOMOD32_CH_VPU27_SOH) {
    const VPUNN::DPULayer tst_layer(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::AVEPOOL,
            {VPUNN::VPUTensor(7, 14, 2048 - 16, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(1, 8, 2048 - 16, 1, VPUNN::DataType::UINT8)},   // output dimensions
            {7, 7},                                                           // kernels
            {1, 1},                                                           // strides
            {0, 0, 0, 0}                                                      // padding
    );
    const VPUNN::VPULayerStrategy tst_strategy{1U,    1U,    2U /*tiles*/, VPUNN::VPUTilingStrategy::NONE,
                                               false, false, true};  // prefetching  true = only DPU data, no DMA.
    const auto origMax = getModel(VPUDevice::VPU_2_7).get_maxWorkloadsPerIntraTileSplit();
    getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(50);  // original test context was 50

    {  // CLUSTERING
        TestInput tin{tst_layer, tst_strategy};
        tin.strategy.tiling_strategy = VPUNN::VPUTilingStrategy::NONE;

        // Very large because 2032 is  mod 16, but 31X64 +1X48 is not accepted, 63x32+1X16 is not reached (limit 50)
        // and we cannot reach a split like: 31x64 = 1984  , 1x32=32,  1x16=16  =>2032 split in 33 workloads
        // the 1x2032 is not accepted because ch not 16/32/64
        TestExpectations texp{VPUNN::Cycles::ERROR_TILE_OUTPUT, true, 0, 0 + 0};

        DoRegularTest(tin, texp, "CLUST not valid 1 workload");
    }

    {  // SOK
        TestInput tin{tst_layer, tst_strategy};
        tin.strategy.tiling_strategy = VPUNN::VPUTilingStrategy::SOK;

        // 2032 split to tile like 1024 and 1008, both div by 16
        // 1024 = 16x64
        // 1008 :   15x64 , 1x48 : not allowed;   31x32, 1x16  reached,    but  15x64 , 1x32, 1x16 : not possible
        // but best.   due to not possible one we get 14000 instead of 8000
        TestExpectations texp{NO_ERROR_EXPECTED, false, 20000, 21000 + 3000};  // big range

        DoRegularTest(tin, texp, "SOK not optimum: 15x64 , 1x32, 1x16  = 1008 ");
    }

    const std::vector<TestCase> tests{
            //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
            // {NO_ERROR_EXPECTED, false, 2473000, 2474000},
            // "CLUSTERING , no memmove"},
            //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
            // {NO_ERROR_EXPECTED, false, 142000, 143000},
            // "SOK , no memmove"},
            {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
             {VPUNN::Cycles::ERROR_TILE_OUTPUT, true, 0, 0},  // but is 31000 if allowed fro split to 64 workloads
             "SOH ,no memmove"},
    };

    executeTests(tests);

    getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(origMax);
}
TEST_F(VPULayerCostModelTest, CONVOLUTION_Concrete_Multiply6326_FLOAT_FLOAT_VPU27) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(60, 7, 512, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(60, 6, 512, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            {3, 3},                                                       // kernels
            {1, 1},                                                       // strides
            {1, 0, 1, 1}                                                  // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = true;
        tst_layer.weight_sparsity = 0.723566F;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                 "CLUSTERING ,flt , sparse "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, false, 260000, 260000 + 55000},
                 "SOK , no memmove, sparse"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0 + 1000},  // was SOH no overlap  : 327500
                 "SOH , no memmove, sparse"},
        };

        executeTests(tests);
    }
}
TEST_F(VPULayerCostModelTest, CONVOLUTION_Concrete_Multiply6326_FLOAT_INT_VPU27_EISXW_76882) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(60, 7, 512, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(60, 6, 512, 1, VPUNN::DataType::UINT8)},    // output dimensions
            {3, 3},                                                       // kernels
            {1, 1},                                                       // strides
            {1, 0, 1, 1}                                                  // padding
    );
    {
        const auto tst_layer = tst_layer_ref;
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                 "CLUSTERING , flt, dense "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                 "SOK , no memmove, dense"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                 "SOH , no memmove, dense"},
        };

        executeTests(tests);
    }
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = true;
        tst_layer.weight_sparsity = 0.723566F;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                 "CLUSTERING ,flt int, sparse "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, false, 250000, 250000 + 60000},
                 "SOK , no memmove, sparse"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, false, 300000,
                  300000 + 60000},  // MAIN TEST CASE, v16 309k, pre v16 350k, GT ???
                 "SOH , no memmove, sparse"},
        };

        executeTests(tests);
    }
}
TEST_F(VPULayerCostModelTest, CONVOLUTION_Concrete_Multiply6326_INT_FLOAT_VPU27) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(60, 7, 512, 1, VPUNN::DataType::UINT8)},    // input dimensions
            {VPUNN::VPUTensor(60, 6, 512, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            {3, 3},                                                       // kernels
            {1, 1},                                                       // strides
            {1, 0, 1, 1}                                                  // padding
    );

    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = true;
        tst_layer.weight_sparsity = 0.723566F;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, false, 240000, 240000 + 33000},  // //v150 272k
                 "CLUSTERING ,flt , sparse "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, false, 130000, 130000 + 20000},  //    //v150 134k
                 "SOK , no memmove, sparse"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, false, 160000, 160000 + 20000},  //    //v150 175k
                 "SOH , no memmove, sparse"},
        };

        executeTests(tests);
    }
}
TEST_F(VPULayerCostModelTest, CONVOLUTION_Concrete_Multiply6326_INT_INT_VPU27) {
    const VPUNN::DPULayer tst_layer_ref(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
                                        {VPUNN::VPUTensor(60, 7, 512, 1, VPUNN::DataType::UINT8)},  // input dimensions
                                        {VPUNN::VPUTensor(60, 6, 512, 1, VPUNN::DataType::UINT8)},  // output dimensions
                                        {3, 3},                                                     // kernels
                                        {1, 1},                                                     // strides
                                        {1, 0, 1, 1}                                                // padding
    );
    {
        const auto tst_layer = tst_layer_ref;
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                 "CLUSTERING , int "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {NO_ERROR_EXPECTED, false, 270000, 270000 + 20000},
                 "SOK , no memmove"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                 "SOH , no memmove"},
        };

        executeTests(tests);
    }

    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = true;
        tst_layer.weight_sparsity = 0.723566F;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, false, 250000, 250000 + 30000},
                 "CLUSTERING , int "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {NO_ERROR_EXPECTED, false, 130000, 130000 + 20000},
                 "SOK , no memmove"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, false, 160000, 160000 + 20000},
                 "SOHO , no memmove"},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, CONVOLUTION_Concrete_Multiply6326_FI_VPU27) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(60, 8, 512, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(60, 6, 512, 1, VPUNN::DataType::UINT8)},    // output dimensions
            {3, 3},                                                       // kernels
            {1, 1},                                                       // strides
            {0, 0, 1, 1}                                                  // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = true;
        tst_layer.weight_sparsity = 0.723158f;

        const std::vector<TestCase> tests{
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                // {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                // "CLUSTERING ,flt , sparse "},
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                // {VPUNN::Cycles::NO_ERROR, false, 132000, 132000 + 1000},//132831
                // "SOK , no memmove, sparse"},
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 300000, 300000 + 55000},  //
                 "SOH , no memmove, sparse"},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, Fused_234_3xELEMENTWISE) {
    // note: ELEMENTWISE that have datasize change will be considered without weights
    const VPUNN::DPULayer l1F_ref(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::ELTWISE,
                                  {VPUNN::VPUTensor(480, 123, 16, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
                                  {VPUNN::VPUTensor(480, 123, 16, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
                                  {1, 1},                                                         // kernels
                                  {1, 1},                                                         // strides
                                  {0, 0, 0, 0}                                                    // padding
    );
    const VPUNN::DPULayer l1Int_ref(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::ELTWISE,
                                    {VPUNN::VPUTensor(480, 123, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions
                                    {VPUNN::VPUTensor(480, 123, 16, 1, VPUNN::DataType::UINT8)},  // output dimensions
                                    {1, 1},                                                       // kernels
                                    {1, 1},                                                       // strides
                                    {0, 0, 0, 0}                                                  // padding
    );

    const VPUNN::DPULayer l1FI_ref(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::ELTWISE,
                                   {VPUNN::VPUTensor(480, 123, 16, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
                                   {VPUNN::VPUTensor(480, 123, 16, 1, VPUNN::DataType::UINT8)},    // output dimensions
                                   {1, 1},                                                         // kernels
                                   {1, 1},                                                         // strides
                                   {0, 0, 0, 0}                                                    // padding
    );

    const VPUNN::DPULayer l1and2_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::ELTWISE,
            {VPUNN::VPUTensor(480, 123, 16, 1, VPUNN::DataType::UINT8)},    // input dimensions
            {VPUNN::VPUTensor(480, 123, 16, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            {1, 1},                                                         // kernels
            {1, 1},                                                         // strides
            {0, 0, 0, 0}                                                    // padding
    );
    const VPUNN::DPULayer l3_ref(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::ELTWISE,
                                 {VPUNN::VPUTensor(480, 122, 16, 1, VPUNN::DataType::UINT8)},    // input dimensions
                                 {VPUNN::VPUTensor(480, 122, 16, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
                                 {1, 1},                                                         // kernels
                                 {1, 1},                                                         // strides
                                 {0, 0, 0, 0}                                                    // padding
    );

    // EXPECT_TRUE(false);

    {  // float all,Layer 1 changed
        std::string t{"All floats @ Layer 1 "};
        auto tst_layer = std::move(l1F_ref);
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                 t + "CLU 0m "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 0, 0 + 0},
                 t + "SOK ,0m"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 30000, 30000 + 10000},
                 t + "SOH ,0m"},
        };
        executeTests(tests);
    }

    {  // UINT8 all, Layer 1 changed
        std::string t{"All UINT8 @ Layer 1 "};
        auto tst_layer = std::move(l1Int_ref);
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 36000, 36000 + 4000},  //
                 t + "CLU 0m "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 0, 0 + 0},
                 t + "SOK ,0m"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 21000, 21000 + 4000},  //
                 t + "SOH ,0m"},
        };
        executeTests(tests);
    }

    {  // FLOAT to INT all, Layer 1 changed, no more in place output possible , output is stand alone
        std::string t{"Reversed F toI @ Layer 1  mixed "};
        auto tst_layer = std::move(l1FI_ref);
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0 + 0},  //
                 t + "CLU 0m "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 0, 0 + 0},
                 t + "SOK ,0m"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 22000, 22000 + 6500},  // huge v16 24k v17 22323   GT??  v159NN:28370
                 t + "SOHO ,0m"},
        };
        executeTests(tests);
    }

    {  // L1&2 int -float. Has to be IMPROVED,  no more in place output possible
        std::string t{"Original Layer 1, mixed "};
        auto tst_layer = std::move(l1and2_ref);
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 40000, 40000 + 5000},  // 44608
                 t + "CLU 0m "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 0, 0 + 0},
                 t + "SOK ,0m"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 35000, 35000 + 5000},  // v16 38k , v17 37k   //v159 24k  GT??
                 t + "SOHO ,0m"},
        };
        executeTests(tests);
    }

    {  // L3  int -float. Has to be IMPROVED , no more in place output possible
        std::string t{"Original Layer 3, mixed "};
        auto tst_layer = std::move(l3_ref);
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 40000, 40000 + 5000},  // 44389
                 t + "CLU 0m "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 0, 0 + 0},
                 t + "SOK ,0m"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 35000, 35000 + 5000},  // v16 37k  v17 37k //v159 24k GT??
                 t + "SOHO ,0m"},
        };
        executeTests(tests);
    }
    // ASSERT_TRUE(false);
}

TEST_F(VPULayerCostModelTest, AVEPOOL_Concrete_GlobalAveragePool_172_MaxWorkloadSPlitAndDetails) {
    const VPUNN::DPULayer tst_layer(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::AVEPOOL,
            {VPUNN::VPUTensor(7, 7, 2048 - 16, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(1, 1, 2048 - 16, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {7, 7},                                                          // kernels
            {1, 1},                                                          // strides
            {0, 0, 0, 0}                                                     // padding
    );
    const VPUNN::VPULayerStrategy tst_strategy{1U,    1U,    2U /*tiles*/, VPUNN::VPUTilingStrategy::NONE,
                                               false, false, true};  // prefetching  true = only DPU data, no DMA.

    const auto origMax = getModel(VPUDevice::VPU_2_7).get_maxWorkloadsPerIntraTileSplit();
    getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(50);  // original test context was 50

    {  // CLUSTERING
        TestInput tin{tst_layer, tst_strategy};
        tin.strategy.tiling_strategy = VPUNN::VPUTilingStrategy::NONE;

        const auto maxWSplit = model_2_7.get_maxWorkloadsPerIntraTileSplit();

        EXPECT_EQ(50U, maxWSplit) << "max workloads split must be default";

        // Very large because 2032 is  mod 16, but 31X64 +1X48 is not accepted, 63x32+1X16 is not reached (limit 50)
        // and we cannot reach a split like: 31x64 = 1984  , 1x32=32,  1x16=16  =>2032 split in 33 workloads
        // the 1x2032 is not accepted because ch not 16/32/64
        TestExpectations texp1{VPUNN::Cycles::ERROR_TILE_OUTPUT, true, 2408000, 2408000 + 1000};

        DoRegularTest(tin, texp1, "CLUST not valid , cannot split");

        model_2_7.set_maxWorkloadsPerIntraTileSplit(64U);
        EXPECT_EQ(model_2_7.get_maxWorkloadsPerIntraTileSplit(), 64U) << "max workloads split must be set";

        // 63x32+1X16 is reached (limit 64)
        TestExpectations texp2{VPUNN::Cycles::NO_ERROR, true, 30000, 30000 + 2000};

        DoRegularTest(tin, texp2, "CLUST must be split to 64");
    }
    {
        TestInput tin{std::move(tst_layer), tst_strategy};
        tin.strategy.tiling_strategy = VPUNN::VPUTilingStrategy::NONE;
        LayerSplitInfo splitInfo{};

        auto t = model_2_7.Layer(tin.l1, tin.strategy.tiling_strategy, 1U, 2, false, false, true, splitInfo);

        EXPECT_FALSE(Cycles::isErrorCode(t));

        ASSERT_EQ(splitInfo.size(), 2U) << "Must be 2 tiles!";
        EXPECT_EQ(splitInfo[0].best_intra_tile_split.second.size(), 64U) << "Tile 1 Must be 64 workloads!";
        EXPECT_EQ(splitInfo[1].best_intra_tile_split.second.size(), 64U) << "Tile 2 Must be 64 workloads!";
        EXPECT_EQ(splitInfo[0].best_intra_tile_split.first, splitInfo[1].best_intra_tile_split.first)
                << "Tiles must be equal in cycles";
    }

    getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(origMax);
}

TEST_F(VPULayerCostModelTest, Default_MaxWorkloadSPlitAndDetails_Test) {
    std::vector<const VPULayerCostModel*> all_models{
            &model_2_0, &model_2_7, &model_2_7_no_dma, &model_4_0, 
    };

    for (const auto m : all_models) {
        const auto maxWSplit = (*m).get_maxWorkloadsPerIntraTileSplit();
        EXPECT_EQ(128U, maxWSplit) << "max workloads split must be default";
    }
}

TEST_F(VPULayerCostModelTest, 01_C01_CONVOLUTION_Multiply_6346) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(30, 23, 1024, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(30, 23, 208, 1, VPUNN::DataType::UINT8)},     // output dimensions
            {3, 3},                                                         // kernels
            {1, 1},                                                         // strides
            {1, 1, 1, 1}                                                    // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = true;
        tst_layer.weight_sparsity = 0.950016f;

        const std::vector<TestCase> tests{
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                // {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                // "CLUSTERING ,flt , sparse "},
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 270000, 270000 + 115000},
                 // v16: 322K,   v17 279k GT???k // Out chnannels %32  //v150 383k
                 "SOK , no memmove, sparse"},
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                // {VPUNN::Cycles::NO_ERROR, true, 330000, 330000 + 1000},  // due to cmx overhead
                // "SOH , no memmove, sparse"},
        };

        executeTests(tests);
    }
}
TEST_F(VPULayerCostModelTest, 01_C02_CONVOLUTION_Multiply_6356) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(60, 2, 1024, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(60, 1, 512, 1, VPUNN::DataType::UINT8)},     // output dimensions
            {3, 3},                                                        // kernels
            {1, 1},                                                        // strides
            {0, 1, 1, 1}                                                   // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = true;
        tst_layer.weight_sparsity = 0.880725f;

        const std::vector<TestCase> tests{
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                // {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                // "CLUSTERING ,flt , sparse "},
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 280000, 280000 + 28000},  // v16 285k, V17 297K,  GT??
                 "SOK , no memmove, sparse"},
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                // {VPUNN::Cycles::NO_ERROR, true, 330000, 330000 + 1000},  // due to cmx overhead
                // "SOH , no memmove, sparse"},
        };

        executeTests(tests);
    }

    const VPUNN::DPULayer tst_layer_ref2(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(60, 3, 1024, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(60, 1, 512, 1, VPUNN::DataType::UINT8)},     // output dimensions
            {3, 3},                                                        // kernels
            {1, 1},                                                        // strides
            {0, 0, 1, 1}                                                   // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref2);
        tst_layer.weight_sparsity_enabled = true;
        tst_layer.weight_sparsity = 0.880725f;

        const std::vector<TestCase> tests{
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                // {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                // "CLUSTERING ,flt , sparse "},
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 280000, 280000 + 20000},  // GT??
                 "SOK , no memmove, sparse , padding 00"},
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                // {VPUNN::Cycles::NO_ERROR, true, 330000, 330000 + 1000},  // due to cmx overhead
                // "SOH , no memmove, sparse"},
        };

        executeTests(tests);
    }
}
TEST_F(VPULayerCostModelTest, 01_C03_ConvolutionBackpropData_1055) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(240, 21, 256, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(240, 21, 128, 1, VPUNN::DataType::UINT8)},    // output dimensions
            {2, 2},                                                         // kernels
            {1, 1},                                                         // strides
            {1, 0, 1, 0}                                                    // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = true;
        tst_layer.weight_sparsity = 0.36438f;

        const std::vector<TestCase> tests{
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                // {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                // "CLUSTERING ,flt , sparse "},
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                // {VPUNN::Cycles::NO_ERROR, true, 132000, 132000 + 1000},  //
                // "SOK , no memmove, sparse"},
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 275000, 275000 + 25000},  //
                 // pre v16:29xK , after v16 28x
                 "SOHO , no memmove, sparse"},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, 01_C04_ConvolutionBackpropData_1183) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(480, 23, 128, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(480, 23, 64, 1, VPUNN::DataType::UINT8)},     // output dimensions
            {2, 2},                                                         // kernels
            {1, 1},                                                         // strides
            {1, 0, 0, 1}                                                    // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = true;
        tst_layer.weight_sparsity = 0.0664062f;

        const std::vector<TestCase> tests{
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                // {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                // "CLUSTERING ,flt , sparse "},
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                // {VPUNN::Cycles::NO_ERROR, true, 132000, 132000 + 1000},  //
                // "SOK , no memmove, sparse"},
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 170000, 170000 + 35000},  //
                 // pre v16 200k, v16: 17xk, GT ???k
                 "SOHO , no memmove, sparse"},
        };

        executeTests(tests);
    }
}
TEST_F(VPULayerCostModelTest, DW_Convolution_AsymetricKernel_1) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::DW_CONVOLUTION,
            {VPUNN::VPUTensor(168, 97, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(168, 96, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {1, 2},                                                      // kernels
            {1, 1},                                                      // strides
            {0, 0, 0, 0}                                                 // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 25000, 25000 + 5000},
                 "CLUSTERING , no memove,  "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 23000, 23000 + 3000},  //
                 "SOK , no memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 14000, 14000 + 2000},  //
                 "SOHO , no memmove, "},
                // note: SOK always wins (non overlapping regions)
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, DW_Convolution_K8Stride8_H32) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::DW_CONVOLUTION,
            {VPUNN::VPUTensor(32, 32, 64, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(4, 4, 64, 1, VPUNN::DataType::FLOAT16)},    // output dimensions
            {8, 8},                                                       // kernels
            {8, 8},                                                       // strides
            {0, 0, 0, 0}                                                  // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 4000, 4000 + 500},  // v159nn:4229  v17:4352
                 "CLUSTERING , ,  "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 3400, 3400 + 350},  // v159nn:3554  v17:3546
                 "SOK , no memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 4100, 4100 + 400},  // v159nn:4242  v17:4336
                 "SOHO , no memmove, "},

                // note: SOK always wins
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, DW_Convolution_K8Stride8_H8) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::DW_CONVOLUTION,
            {VPUNN::VPUTensor(8, 8, 128, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(1, 1, 128, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            {8, 8},                                                      // kernels
            {8, 8},                                                      // strides
            {0, 0, 0, 0}                                                 // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 5400, 5400 + 1000},
                 "CLUSTERING , ,  "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 2000, 2000 + 1000},  //
                 "SOK , no memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 7900,
                  7900 + 1000},  // H=8 with K=8 , no padding cannot be split on h
                 "SOH , no memmove, "},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, Model_C1_v2_a_0_int8_NCHW_FP16_LATENCY_API10_MLIR) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(1, 1, 16, 1, VPUNN::DataType::UINT8)},      // input dimensions
            {VPUNN::VPUTensor(1, 1, 8192, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            {1, 1},                                                       // kernels
            {1, 1},                                                       // strides
            {0, 0, 0, 0}                                                  // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 2000, 4000},
                 "SOK , no memmove, "},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, Model_G_v1_a_0_fp16_NCHW_FP16_LATENCY_API10_MLIR) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(4, 1, 8192, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(4, 1, 176, 1, VPUNN::DataType::FLOAT16)},   // output dimensions
            {1, 1},                                                       // kernels
            {1, 1},                                                       // strides
            {0, 0, 0, 0}                                                  // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 18000, 18000 + 8000},  // big
                 "SOK , no memmove, "},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, Model_L_v1_a_1_int8_NCHW_FP16_LATENCY_API10_MLIR) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(256, 129, 4, 1, VPUNN::DataType::UINT8)},     // input dimensions
            {VPUNN::VPUTensor(256, 128, 16, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            {3, 3},                                                         // kernels
            {1, 1},                                                         // strides
            {1, 0, 1, 1}                                                    // padding
    );
    // EXPECT_TRUE(false);
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 92000, 92000 + 4000},  // v16 94856 , v17 95654
                 "CLUSTERING , ,  "},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, Model_L_v1_a_1_int8_NCHW_FP16_LATENCY_API10_MLIR_Set_2) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(256, 256, 4, 1, VPUNN::DataType::UINT8)},     // input dimensions
            {VPUNN::VPUTensor(256, 256, 16, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            {3, 3},                                                         // kernels
            {1, 1},                                                         // strides
            {1, 1, 1, 1}                                                    // padding
    );
    // EXPECT_TRUE(false);
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 92000, 92000 + 4000},  // v16 94856 , v17 95654
                 "SOH , no memmove, "},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, redowa_deeplab_v3_dense_IRv11_FP16_LATENCY_MLIR_MORE_MEMORY) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::DW_CONVOLUTION,
            {VPUNN::VPUTensor(65, 16, 960, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(65, 8, 960, 1, VPUNN::DataType::FLOAT16)},   // output dimensions
            {9, 9},                                                        // kernels
            {1, 1},                                                        // strides
            {0, 0, 4, 4}                                                   // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 1, VPUNN::Cycles::START_ERROR_RANGE - 1},
                 "SOH , no memmove, "},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, redowa_deeplab_v3_dense_IRv11_FP16_INT8_LATENCY_MLIR_Set_1) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(513, 130, 4, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(257, 65, 32, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                      // kernels
            {2, 2},                                                      // strides
            {1, 0, 1, 1}                                                 // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 57700, 57700 + 5000},
                 "SOH , no memmove, "},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, redowa_deeplab_v3_dense_IRv11_FP16_INT8_LATENCY_MLIR_Set_2) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(513, 513, 4, 1, VPUNN::DataType::UINT8)},   // input dimensions
            {VPUNN::VPUTensor(257, 257, 32, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                       // kernels
            {2, 2},                                                       // strides
            {1, 1, 1, 1}                                                  // padding
    );
    // EXPECT_TRUE(false);
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 100000, 100000 + 17000},  // v16 104000 , v17 114995
                 "SOH , no memmove, "},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, ELTWISE_diff_swizz_NPU40) {
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
    VPULayerCostModel& theModel = model_4_0;
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

/// test for SOH with overlap,  and with HALO
/// overlap one does not fit into memory
TEST_F(VPULayerCostModelTest, deeplab_v3_SOH_HALO_EISXW_79152) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::DW_CONVOLUTION,
            {VPUNN::VPUTensor(65, 16, 960, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(65, 8, 960, 1, VPUNN::DataType::FLOAT16)},   // output dimensions
            {9, 9},                                                        // kernels
            {1, 1},                                                        // strides
            {0, 0, 4, 4}                                                   // padding
    );
    {
        auto tst_layer = tst_layer_ref;

        const std::vector<TestCase> tests{
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                 "SOHO , no memmove, "},
        };

        executeTests(tests);
    }
    {
        auto tst_layer = tst_layer_ref;

        const std::vector<TestCase> tests{
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_HaloRead, false, false, true}},
                 {Cycles::NO_ERROR, true, 600000, 600000 + 500000 + 350000},  // huge delta , what is the GT
                 // v17: :newmemtens 1390740
                 "SOH w in Halo , no memmove, "},
        };

        executeTests(tests);
    }
    {  // details
        DPULayer tst_layer(std::move(tst_layer_ref));
        const VPULayerStrategy strategy{1U, 1U, 2U, VPUTilingStrategy::SOH_HaloRead, false, false, true};

        VPULayerCostModel& theModel{getModel(tst_layer.device)};

        Logger::clear2ndlog();
        unsigned cost_cyc{};
        LayerSplitInfo detailed_split;
        ASSERT_NO_THROW(cost_cyc = theModel.Layer(tst_layer, strategy.tiling_strategy, strategy.nDPUs, strategy.nTiles,
                                                  strategy.input_fetching, strategy.output_spilling,
                                                  strategy.prefetching, detailed_split))
                << tst_layer << strategy << "\n code: " << cost_cyc << " : " << Cycles::toErrorText(cost_cyc);

        ASSERT_EQ(detailed_split.size(), 2);
        const auto& top{detailed_split[0].inter_tile_split_layer};
        ASSERT_EQ(top.halo.input_0_halo.top, 0);
        ASSERT_EQ(top.halo.input_0_halo.bottom, 4);
        const auto& bot{detailed_split[1].inter_tile_split_layer};
        ASSERT_EQ(bot.halo.input_0_halo.top, 4);
        ASSERT_EQ(bot.halo.input_0_halo.bottom, 0);

        /* EXPECT_EQ(cost_cyc, 3) << tst_layer << toStringLayerSplitInfo(detailed_split) << Logger::get2ndlog()
                                << "\n code: " << cost_cyc << " : " << Cycles::toErrorText(cost_cyc);*/
    }
}

/// test for SOH with overlap,  and with HALO . Same memory, only tile 2 used halo
TEST_F(VPULayerCostModelTest, SOH_HALO_EISXW_87028) {
    const DPUWorkload wl_as_layer{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(640, 61, 16, 1, DataType::FLOAT16)},  // input dimensions
            {VPUTensor(320, 30, 32, 1, DataType::FLOAT16)},  // output dimensions
            {3, 3},                                          // kernels
            {2, 2},                                          // strides
            {0, 0, 0, 1},                                    // padding
            ExecutionMode::CUBOID_16x16,                     // execution mode
            ActivationFunction::NONE,                        // activation
            0.0F,                                            // act_sparsity
            0.611111F,                                       // weight_sparsity
            {swz_def, swz_def},                              // input_swizzling
            {swz_def},                                       // output_swizzling
            1,                                               // output_write_tiles
            {0, 0, 0, 0},                                    // offsets
            ISIStrategy::CLUSTERING,                         // isi_strategy
            true,                                            // weight_sparsity_enabled
                                                             // halo aspects    default!
    };

    const VPUNN::DPULayer tst_layer_ref{wl_as_layer};
    {
        auto tst_layer = tst_layer_ref;

        const std::vector<TestCase> tests{
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {Cycles::NO_ERROR, true, 19000, 19000 + 2000},
                 "SOHO , no memmove, "},
        };

        executeTests(tests);
    }
    {
        auto tst_layer = tst_layer_ref;

        const std::vector<TestCase> tests{
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_HaloRead, false, false, true}},
                 {Cycles::NO_ERROR, true, 22000, 22000 + 1200},  // 604935 split 15
                 "SOH w in Halo , no memmove, "},
        };

        executeTests(tests);
    }
    {  // details
        DPULayer tst_layer(std::move(tst_layer_ref));
        const VPULayerStrategy strategy{1U, 1U, 2U, VPUTilingStrategy::SOH_HaloRead, false, false, true};

        VPULayerCostModel& theModel{getModel(tst_layer.device)};

        Logger::clear2ndlog();
        unsigned cost_cyc{};
        LayerSplitInfo detailed_split;
        ASSERT_NO_THROW(cost_cyc = theModel.Layer(tst_layer, strategy.tiling_strategy, strategy.nDPUs, strategy.nTiles,
                                                  strategy.input_fetching, strategy.output_spilling,
                                                  strategy.prefetching, detailed_split))
                << tst_layer << strategy << "\n code: " << cost_cyc << " : " << Cycles::toErrorText(cost_cyc);

        // EXPECT_EQ(cost_cyc, 3) << tst_layer << toStringLayerSplitInfo(detailed_split) << Logger::get2ndlog()
        //                        << "\n code: " << cost_cyc << " : " << Cycles::toErrorText(cost_cyc);

        ASSERT_EQ(detailed_split.size(), 2);
        const auto& top{detailed_split[0].inter_tile_split_layer};
        ASSERT_EQ(top.halo.input_0_halo.top, 0);
        ASSERT_EQ(top.halo.input_0_halo.bottom, 0);
        const auto& bot{detailed_split[1].inter_tile_split_layer};
        ASSERT_EQ(bot.halo.input_0_halo.top, 1);  // theonly halo is here
        ASSERT_EQ(bot.halo.input_0_halo.bottom, 0);
    }
}

TEST_F(VPULayerCostModelTest, scale_mobilenet_ssd_FP16_MLIR) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(5, 5, 128, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(3, 3, 256, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            {3, 3},                                                      // kernels
            {2, 2},                                                      // strides
            {1, 1, 1, 1}                                                 // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 7000, 7000 + 1000},
                 "SOH , no memmove, "},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, scale_Sphereface_FP16_INT8_MLIR) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(1, 1, 8192, 1, VPUNN::DataType::UINT8)},   // input dimensions
            {VPUNN::VPUTensor(1, 1, 256, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            {1, 1},                                                      // kernels
            {1, 1},                                                      // strides
            {0, 0, 0, 0}                                                 // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 10000, 10000 + 5000},  // big
                 "SOK , no memmove, "},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest,
       face_detection_adas_0001_caffe_dense_IRv11_FP16_INT8_THROUGHPUT_NCHW_NCHW_U8_FP16_API20_MLIR_Set_1) {
    const DPULayer tst_layer_ref(VPUNN::VPUDevice::VPU_2_7, Operation::DW_CONVOLUTION,
                                 {VPUNN::VPUTensor(6, 3, 128, 1, VPUNN::DataType::UINT8)},  // input dimensions
                                 {VPUNN::VPUTensor(3, 2, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
                                 {3, 3},                                                    // kernels
                                 {2, 2},                                                    // strides
                                 {1, 1, 1, 0}                                               // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 650, 650 + 200},  // v159nn:766  v17:792
                 "SOHO , no memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 250, 250 + 250},  // v159nn:335  v17:414
                 "SOK , no memmove, "},

                // note SOK always wins
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest,
       face_detection_adas_0001_caffe_dense_IRv11_FP16_INT8_THROUGHPUT_NCHW_NCHW_U8_FP16_API20_MLIR_Set_2) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(672, 96, 4, 1, VPUNN::DataType::UINT8)},   // input dimensions
            {VPUNN::VPUTensor(336, 48, 32, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                      // kernels
            {2, 2},                                                      // strides
            {1, 0, 1, 0}                                                 // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 55000, 55000 + 5000},
                 "CLU , no memmove, "},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, midas_672x384_onnx_dense_IRv10_INT8_NHWC_NHWC_U8_FP16_LATENCY_API10_MLIR_Set_1) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(672, 97, 4, 1, VPUNN::DataType::UINT8)},   // input dimensions
            {VPUNN::VPUTensor(336, 48, 32, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                      // kernels
            {2, 2},                                                      // strides
            {0, 0, 0, 1}                                                 // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 54000, 54000 + 6000},
                 "CLU , no memmove, "},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, midas_672x384_onnx_dense_IRv10_INT8_NHWC_NHWC_U8_FP16_LATENCY_API10_MLIR_Set_2) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(672, 384, 4, 1, VPUNN::DataType::UINT8)},   // input dimensions
            {VPUNN::VPUTensor(336, 192, 32, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                       // kernels
            {2, 2},                                                       // strides
            {0, 1, 0, 1}                                                  // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 99000, 99000 + 30000},  // big
                 "SOHO , no memmove, "},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, isv_SemiSupSegmentationaAtten_onnx_dense_IRv11_FP16_NCHW_NCHW_FP16_FP16_MLIR_Set_1) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::DW_CONVOLUTION,
            {VPUNN::VPUTensor(17, 17, 96, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(3, 3, 96, 1, VPUNN::DataType::FLOAT16)},    // output dimensions
            {6, 6},                                                       // kernels
            {6, 6},                                                       // strides
            {0, 1, 0, 1}                                                  // padding
    );
    // EXPECT_TRUE(false);
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 2900, 2900 + 200},  // v16 3063, v17 2973, v159nn:3045
                 "SOH O, no memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 1700, 1700 + 200},  // v17:1821 v159nn:1773
                 "SOK, no memmove, "},

                // note SOK always wins
        };

        executeTests(tests);
    }
}
TEST_F(VPULayerCostModelTest, isv_SemiSupSegmentationaAtten_onnx_dense_IRv11_FP16_NCHW_NCHW_FP16_FP16_MLIR_Set_2) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::DW_CONVOLUTION,
            {VPUNN::VPUTensor(17, 17, 240, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(3, 3, 240, 1, VPUNN::DataType::FLOAT16)},    // output dimensions
            {6, 6},                                                        // kernels
            {6, 6},                                                        // strides
            {0, 1, 0, 1}                                                   // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 7000, 7000 + 1500},  // v17:7755  v159nn:7967
                 "SOHO , no memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 3600, 3600 + 500},  // v17:3996  v159nn:3720
                 "SOK , no memmove, "},

                // note : SOK always wins
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, isv_SemiSupSegmentationaAtten_onnx_dense_IRv11_FP16_NCHW_NCHW_FP16_FP16_MLIR_Set_3) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::DW_CONVOLUTION,
            {VPUNN::VPUTensor(17, 17, 128, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(3, 3, 128, 1, VPUNN::DataType::FLOAT16)},    // output dimensions
            {6, 6},                                                        // kernels
            {6, 6},                                                        // strides
            {0, 1, 0, 1}                                                   // padding
    );
    // EXPECT_TRUE(false);
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 3800, 3800 + 400},  // v16 4084     v17 3964 v159nn:4060
                 "SOH O , no memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 3800, 3800 + 400},  // v16 4080     v17 3992 , v159nn:4080
                 "CLU , no memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 1750, 1750 + 300},  // v17:1998  v159nn:1860
                 "SOK , no memmove, "},
                // note: SOK always wins
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, isv_SemiSupSegmentationaAtten_onnx_dense_IRv11_FP16_NCHW_NCHW_FP16_FP16_MLIR_Set_4) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::DW_CONVOLUTION,
            {VPUNN::VPUTensor(17, 17, 576, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(3, 3, 576, 1, VPUNN::DataType::FLOAT16)},    // output dimensions
            {6, 6},                                                        // kernels
            {6, 6},                                                        // strides
            {0, 1, 0, 1}                                                   // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 17000, 17000 + 2000},  // v17:17838    , v159nn:18270
                 "SOH O , no memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 8000, 8000 + 1500},  // v17:8991  v159nn:8370
                 "SOK , no memmove, "},

                // note: SOK always wins
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, CONVOLUTION_Concrete_Multiply8641_VPU27_Prefetch) {
    const VPUNN::DPULayer tst_layer(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
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

        TestExpectations texp{VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0};

        DoRegularTest(tin, texp, "SOH with errors");
    }

    const std::vector<TestCase> tests{
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
             {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
             "CLUSTERING , + memmove, with errors"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
             {NO_ERROR_EXPECTED, true, 30000 + 57000, 30000 + 57000 + 3500},
             "SOK , + memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, false}},
             {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
             "SOH ,+ memmove, with errors"},
    };

    executeTests(tests);
}

TEST_F(VPULayerCostModelTest, CONVOLUTION_Concrete_Multiply6326_FLOAT_FLOAT_VPU27_Prefetch) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(60, 7, 512, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(60, 6, 512, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            {3, 3},                                                       // kernels
            {1, 1},                                                       // strides
            {1, 0, 1, 1}                                                  // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = true;
        tst_layer.weight_sparsity = 0.723566F;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                 "CLUSTERING ,flt , sparse "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, false, 260000 + 20000, 260000 + 20000 + 55000},
                 "SOK , no memmove, sparse"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, false}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0 + 1000},  // was SOH no overlap  : 327500
                 "SOH , no memmove, sparse"},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, CONVOLUTIONPrefetchTest_Multiply6326_IF) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(60, 7, 512, 1, VPUNN::DataType::UINT8)},    // input dimensions
            {VPUNN::VPUTensor(60, 6, 512, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            {3, 3},                                                       // kernels
            {1, 1},                                                       // strides
            {1, 0, 1, 1}                                                  // padding
    );

    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = true;
        tst_layer.weight_sparsity = 0.723566F;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, false, 250000 + 50000, 250000 + 50000 + 30000},
                 "CLUSTERING ,flt , sparse "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, false, 130000 + 25000, 130000 + 25000 + 20000},
                 "SOK , + memmove, sparse"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, false, 160000 + 50000, 160000 + 50000 + 20000},
                 "SOHO , + memmove, sparse"},
        };
    }
}

TEST_F(VPULayerCostModelTest, DW_Convolution_K8Stride8_H32_Prefetch) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::DW_CONVOLUTION,
            {VPUNN::VPUTensor(32, 32, 64, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(4, 4, 64, 1, VPUNN::DataType::FLOAT16)},    // output dimensions
            {8, 8},                                                       // kernels
            {8, 8},                                                       // strides
            {0, 0, 0, 0}                                                  // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 5500, 5500 + 1000},
                 "CLUSTERING , ,  "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 4501, 4501 + 1000},  // v159nn:
                 "SOK , no memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 5501, 5501 + 1000},  // v159nn:
                 "SOH , no memmove, "},
                // note SOK wins
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, DWConvolutionPrefetchTest) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::DW_CONVOLUTION,
            {VPUNN::VPUTensor(168, 97, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(168, 96, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {1, 2},                                                      // kernels
            {1, 1},                                                      // strides
            {0, 0, 0, 0}                                                 // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 27000, 28000 + 3000},
                 "CLUSTERING ,no prefetch ,  "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 24000, 24000 + 3000},
                 "SOK , no prefetch, "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 15000, 15000 + 2000},
                 "SOHO , no prefetch, "},
                // note: SOK WINS
        };

        executeTests(tests);
    }
}
TEST_F(VPULayerCostModelTest, Fused_234_3xELEMENTWISE_Prefetch) {
    // note: ELEMENTWISE that have datasize change will be considered without weights
    const VPUNN::DPULayer l1F_ref(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::ELTWISE,
                                  {VPUNN::VPUTensor(480, 123, 16, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
                                  {VPUNN::VPUTensor(480, 123, 16, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
                                  {1, 1},                                                         // kernels
                                  {1, 1},                                                         // strides
                                  {0, 0, 0, 0}                                                    // padding
    );
    const VPUNN::DPULayer l1Int_ref(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::ELTWISE,
                                    {VPUNN::VPUTensor(480, 123, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions
                                    {VPUNN::VPUTensor(480, 123, 16, 1, VPUNN::DataType::UINT8)},  // output dimensions
                                    {1, 1},                                                       // kernels
                                    {1, 1},                                                       // strides
                                    {0, 0, 0, 0}                                                  // padding
    );

    const VPUNN::DPULayer l1FI_ref(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::ELTWISE,
                                   {VPUNN::VPUTensor(480, 123, 16, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
                                   {VPUNN::VPUTensor(480, 123, 16, 1, VPUNN::DataType::UINT8)},    // output dimensions
                                   {1, 1},                                                         // kernels
                                   {1, 1},                                                         // strides
                                   {0, 0, 0, 0}                                                    // padding
    );

    const VPUNN::DPULayer l1and2_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::ELTWISE,
            {VPUNN::VPUTensor(480, 123, 16, 1, VPUNN::DataType::UINT8)},    // input dimensions
            {VPUNN::VPUTensor(480, 123, 16, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            {1, 1},                                                         // kernels
            {1, 1},                                                         // strides
            {0, 0, 0, 0}                                                    // padding
    );
    const VPUNN::DPULayer l3_ref(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::ELTWISE,
                                 {VPUNN::VPUTensor(480, 122, 16, 1, VPUNN::DataType::UINT8)},    // input dimensions
                                 {VPUNN::VPUTensor(480, 122, 16, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
                                 {1, 1},                                                         // kernels
                                 {1, 1},                                                         // strides
                                 {0, 0, 0, 0}                                                    // padding
    );

    {  // float all,Layer 1 changed
        std::string t{"All floats @ Layer 1 "};
        auto tst_layer = std::move(l1F_ref);
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                 t + "CLU +m "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 0, 0 + 0},
                 t + "SOK ,+m"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 30000 + 44000, 30000 + 44000 + 10000},  // v159 79k
                 t + "SOH ,+m"},
        };
        executeTests(tests);
    }

    {  // UINT8 all, Layer 1 changed
        std::string t{"All UINT8 @ Layer 1 "};
        auto tst_layer = std::move(l1Int_ref);
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 73000, 73000 + 5000},
                 t + "CLU +m "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 0, 0 + 0},
                 t + "SOK ,+m"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 38000, 38000 + 5000},
                 t + "SOH ,+m"},
        };
        executeTests(tests);
    }

    {  // FLOAT to INT all, Layer 1 changed , no more in place output possible
        std::string t{"Reversed F toI @ Layer 1  mixed"};
        auto tst_layer = std::move(l1FI_ref);
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0 + 0},  //
                 t + "CLU 0m "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 0, 0 + 0},
                 t + "SOK ,0m"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 58000, 58000 + 10000},
                 t + "SOHO ,0m"},
        };
        executeTests(tests);
    }

    {  // L1&2 int -float. Has to be IMPROVED, no more in place output possible
        std::string t{"Original Layer 1, mixed "};
        auto tst_layer = std::move(l1and2_ref);
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 85000, 85000 + 5000},
                 t + "CLU 0m "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 0, 0 + 0},
                 t + "SOK ,0m"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 47000, 47000 + 16000},
                 t + "SOHO ,0m"},
        };
        executeTests(tests);
    }

    {  // L3  int -float. Has to be IMPROVED, no more in place output possible
        std::string t{"Original Layer 3, mixed "};
        auto tst_layer = std::move(l3_ref);
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 85000, 85000 + 5000},
                 t + "CLU 0m "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 0, 0 + 0},
                 t + "SOK ,0m"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 46000, 46000 + 16000},
                 t + "SOH ,0m"},
        };
        executeTests(tests);
    }
    // ASSERT_TRUE(false);
}

TEST_F(VPULayerCostModelTest, ELTWISE_PrefetchTest) {
    const VPUNN::DPULayer tst_layer(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::ELTWISE,
                                    {VPUNN::VPUTensor(56, 56, 256, 1, VPUNN::DataType::UINT8)},  // input dimensions
                                    {VPUNN::VPUTensor(56, 56, 256, 1, VPUNN::DataType::UINT8)},  // output dimensions
                                    {1, 1},                                                      // kernels
                                    {1, 1},                                                      // strides
                                    {0, 0, 0, 0}                                                 // padding
    );
    const VPUNN::VPULayerStrategy tst_strategy{1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false};

    {  // clustering
        TestInput tin{tst_layer, tst_strategy};
        tin.strategy.tiling_strategy = VPUNN::VPUTilingStrategy::NONE;

        TestExpectations texp{NO_ERROR_EXPECTED, false, 55000, 55000 + 2000};  // without prefetching:  24000 , 25100

        DoRegularTest(tin, texp, "Clustering ,memove");
    }

    {  // SOHO
        TestInput tin{tst_layer, tst_strategy};
        tin.strategy.tiling_strategy = VPUNN::VPUTilingStrategy::SOH_Overlapped;

        TestExpectations texp{NO_ERROR_EXPECTED, false, 28000,
                              28000 + 3000U};  // without prefetching: 11000, 11000 + 1000U

        DoRegularTest(tin, texp, "SOHO,memove");
    }
    {  // SOK
        TestInput tin{std::move(tst_layer), tst_strategy};
        tin.strategy.tiling_strategy = VPUNN::VPUTilingStrategy::SOK;  // SOK is now allowed for element wise

        TestExpectations texp{NO_ERROR_EXPECTED, true, 26000, 26000 + 8000};

        DoRegularTest(tin, texp, "SOK tentative on elementwise, memove");
    }
}
TEST_F(VPULayerCostModelTest, MAXPOOLPrefetchTest_NPU27) {
    const VPUNN::DPULayer tst_layer(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::MAXPOOL,
                                    {VPUNN::VPUTensor(7, 7, 2048, 1, VPUNN::DataType::UINT8)},  // input dimensions
                                    {VPUNN::VPUTensor(1, 1, 2048, 1, VPUNN::DataType::UINT8)},  // output dimensions
                                    {7, 7},                                                     // kernels
                                    {1, 1},                                                     // strides
                                    {0, 0, 0, 0}                                                // padding
    );
    show_split = true;
    // MAXPOOL has no wts memory, not even wts table, . Last fix eliminated wts table so wts size =0. resulting that wts
    // prefetch gets zero vs previous version change is general, all devices. even if fro 2.7 might be a underestimation
    const int wts_adj = -2000;  // negative prefetch adjustement
    const std::vector<TestCase> tests{
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
             {NO_ERROR_EXPECTED, false, 52000 + 3000 + wts_adj, 52000 + 3000 + wts_adj + 18000},
             "CLUSTERING , no mem"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
             {NO_ERROR_EXPECTED, false, 13000 + 2000, 13000 + 2000 + 7000},  // without prefetch 13000, 13000 + 1000
             "SOK , no mem"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, false}},
             {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 0, 0 + 0},  // 7 cannot be split, must be 14
             "SOH ,no mem"},
    };

    const auto origMax = getModel(VPUDevice::VPU_2_7).get_maxWorkloadsPerIntraTileSplit();
    getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(50);  // original test context was 50
    executeTests(tests);
    getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(origMax);
}

TEST_F(VPULayerCostModelTest, AVEPOOLPrefetchTest_172_NPU27) {
    const VPUNN::DPULayer tst_layer(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::AVEPOOL,        // goes to DW Conv
                                    {VPUNN::VPUTensor(7, 15, 2048, 1, VPUNN::DataType::UINT8)},  // input dimensions
                                    {VPUNN::VPUTensor(1, 9, 2048, 1, VPUNN::DataType::UINT8)},   // output dimensions
                                    {7, 7},                                                      // kernels
                                    {1, 1},                                                      // strides
                                    {0, 0, 0, 0}                                                 // padding
    );

    const std::vector<TestCase> tests{
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
             {VPUNN::Cycles::NO_ERROR, false, 52000 + 10000, 52000 + 10000 + 7000},
             "CLUSTERING , + memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
             {NO_ERROR_EXPECTED, false, 20000 + 5000, 20000 + 5000 + 5000},
             "SOK , + memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, false}},
             {VPUNN::Cycles::NO_ERROR, true, 51000 + 9000, 51000 + 9000 + 6000},  //
             "SOHO ,+ memmove"},
    };
    const auto origMax = getModel(VPUDevice::VPU_2_7).get_maxWorkloadsPerIntraTileSplit();
    getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(50);  // original test context was 50
    executeTests(tests);
    getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(origMax);
}

TEST_F(VPULayerCostModelTest, CMCONVPrefetchTest) {
    // CONV with IC <16 to be compressed CONV
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(672, 97, 4, 1, VPUNN::DataType::UINT8)},   // input dimensions
            {VPUNN::VPUTensor(336, 48, 32, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                      // kernels
            {2, 2},                                                      // strides
            {0, 0, 0, 1}                                                 // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 55000, 55000 + 8000},
                 "NONE , + memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 30000, 30000 + 3000},
                 "SOHO , + memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 55000, 55000 + 8000},
                 "SOK , + memmove, "},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, ZeroNumberOfTiles_Layer_Test_NPU40) {
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
TEST_F(VPULayerCostModelTest, Unet_perf_SOH_SOK_after_SOHO) {
    const VPUNN::DPUWorkload wl_h5{
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(60, 5, 1024, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(60, 4, 176, 1, VPUNN::DataType::FLOAT16)},   // output dimensions
            {3, 3},                                                        // kernels
            {1, 1},                                                        // strides
            {1, 0, 1, 1},                                                  // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                            // execution mode
            VPUNN::ActivationFunction::NONE,                               // activation
            0.0F,                                                          // act_sparsity
            0.624613F,                                                     // weight_sparsity
            {swz_def, swz_def},                                            // input_swizzling
            {swz_def},                                                     // output_swizzling
            1,                                                             // output_write_tiles
            {0, 0, 0, 0},                                                  // offsets
            VPUNN::ISIStrategy::CLUSTERING,                                // isi_strategy
            true,                                                          // weight_sparsity_enabled
    };

    const VPUNN::DPUWorkload wl_h3{
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(60, 3, 1024, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(60, 2, 176, 1, VPUNN::DataType::FLOAT16)},   // output dimensions
            {3, 3},                                                        // kernels
            {1, 1},                                                        // strides
            {1, 0, 1, 1},                                                  // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                            // execution mode
            VPUNN::ActivationFunction::NONE,                               // activation
            0.0F,                                                          // act_sparsity
            0.624613F,                                                     // weight_sparsity
            {swz_def, swz_def},                                            // input_swizzling
            {swz_def},                                                     // output_swizzling
            1,                                                             // output_write_tiles
            {0, 0, 0, 0},                                                  // offsets
            VPUNN::ISIStrategy::CLUSTERING,                                // isi_strategy
            true,                                                          // weight_sparsity_enabled
    };

    // const std::string altModelName(NameHelperNN::get_model_root() + "vpu_2_7-150.vpunn");
    // VPULayerCostModel model_alternative{altModelName};
    const bool prefetch{true};
    VPULayerCostModel& theModel = model_2_7;
    // EXPECT_TRUE(false);
    //  GT??  are these corner cases?
    {
        const VPUNN::DPULayer tst_layer(wl_h5);
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 250000,
                  250000 + 30000},  // v16 251k  //v17 279k, same as CLU H3 below, v159NN:263125
                 "small H(5) SOHO , no memmove, "},

                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_HaloRead, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 300000,
                  320000 + 100000},  // huge  GT??  v16 324k    v17 373k(!!!): 396149  //v159 (SOHO) 260k  out of
                                     // reasonable range()
                 "small H(5) SOH H , no memmove, "},

                {{tst_layer, {1U, 1U, 2U, VPUTilingStrategy::NONE, false, false, prefetch}},
                 {Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0 + 0},  //
                 " small H(5) CLUSTERING , no memmove, "},

                {{tst_layer, {1U, 1U, 2U, VPUTilingStrategy::SOK, false, false, prefetch}},
                 {Cycles::NO_ERROR, true, 137000, 137000 + 11000},  // v17:146427    v159NN:138370
                 " small H(5) SOK (added for  completion), no memmove, "},

                // Note : SOHO reverses with SOHH with v17 ?,
                // note: SOK wins always ahead of SOHO (stability OK), huge deltas
        };
        executeTests(tests);
    }
    {
        const DPULayer tst_layer(wl_h3);
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {Cycles::NO_ERROR, true, 240000, 240000 + 48000},  // v16 249k,   v17 280K   GT?? v159nn:260k
                 "very small H(3) SOHO , no memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUTilingStrategy::SOH_HaloRead, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 300000,
                  300000 + 100000},  // v16 314k,   v17 360k:388k  //v159NN (SOHO) 253k  huge Gt??
                 "very small H(3) SOH H , no memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUTilingStrategy::NONE, false, false, prefetch}},
                 {Cycles::NO_ERROR, true, 250000, 250000 + 30000},  // v16 251k,   v17 276k    //v159 261k
                 "very small H(3) CLUSTERING , no memmove, "},

                {{tst_layer, {1U, 1U, 2U, VPUTilingStrategy::SOK, false, false, prefetch}},
                 {Cycles::NO_ERROR, true, 131000, 131000 + 15000},  // v17 144K  //v159NN:133245 Gt??
                 "very small H(3) SOK (added extra) , no memmove, "},

                // Note: SOK wins always ahead of SOHO (stability OK)
        };
        executeTests(tests);
    }
    //{
    //    VPUNN::DPULayer tst_layer(wl_h5);
    //    const VPULayerStrategy strategy{1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false,
    //    prefetch};

    //    Logger::clear2ndlog();
    //    unsigned cost_cyc{};
    //    LayerSplitInfo detailed_split;
    //    ASSERT_NO_THROW(cost_cyc = theModel.Layer(tst_layer, VPUNN::VPUTilingStrategy::SOH_Overlapped, 1U, 2U,
    //    false, false,
    //                                              prefetch, detailed_split))
    //            << tst_layer << strategy;

    //    EXPECT_EQ(cost_cyc, 1) << tst_layer << toStringLayerSplitInfo(detailed_split) << Logger::get2ndlog();

    //    std::string err_info;
    //    CyclesInterfaceType dpu_cost = theModel.DPU(tst_layer, err_info);
    //    EXPECT_EQ(dpu_cost, 2) << dpu_cost << "=" << Cycles::toErrorText(dpu_cost) << tst_layer << err_info;
    //}

    {
        VPUNN::DPULayer tst_layer(wl_h3);
        const VPULayerStrategy strategy{1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch};

        Logger::clear2ndlog();
        unsigned cost_cyc{};
        LayerSplitInfo detailed_split;
        ASSERT_NO_THROW(cost_cyc = theModel.Layer(tst_layer, VPUNN::VPUTilingStrategy::SOH_Overlapped, 1U, 2U, false,
                                                  false, prefetch, detailed_split))
                << tst_layer << strategy;
        // EXPECT_EQ(cost_cyc, 3) << tst_layer << toStringLayerSplitInfo(detailed_split) << Logger::get2ndlog();

        std::string err_info;
        CyclesInterfaceType dpu_cost = theModel.get_cost_model().DPU(tst_layer, err_info);
        // EXPECT_EQ(dpu_cost, 4) << dpu_cost << "=" << Cycles::toErrorText(dpu_cost) << tst_layer << err_info
        //                        << "END ERR";

        EXPECT_LE(cost_cyc, dpu_cost) << tst_layer << toStringLayerSplitInfo(detailed_split) << Logger::get2ndlog()
                                      << dpu_cost << "=" << Cycles::toErrorText(dpu_cost) << err_info << "END ERR";
    }
}

// use this as a template for investigation tests
TEST_F(VPULayerCostModelTest, DISABLED_Z_InvestigationTest) {
    const VPUNN::DPUWorkload wl_1{
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(1019, 5, 16, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(127, 1, 768, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            {11, 5},                                                       // kernels
            {8, 8},                                                        // strides
            {0, 0, 0, 0},                                                  // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                            // execution mode
            VPUNN::ActivationFunction::NONE,                               // activation
            0.0F,                                                          // act_sparsity
            0.0F,                                                          // weight_sparsity
            {swz_def, swz_def},                                            // input_swizzling
            {swz_def},                                                     // output_swizzling
            1,                                                             // output_write_tiles
            {0, 0, 0, 0},                                                  // offsets
            VPUNN::ISIStrategy::CLUSTERING,                                // isi_strategy
            false,                                                         // weight_sparsity_enabled
    };

    const VPUNN::DPULayer tst_layer_ref(
            wl_1
            // VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            //{VPUNN::VPUTensor(1019, 5, 16, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            //{VPUNN::VPUTensor(127, 1, 768, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            //{11, 5},                                                       // kernels
            //{8, 8},                                                        // strides
            //{0, 0, 0, 0}                                                   // padding
    );

    VPULayerCostModel& theModel = model_2_7;
    Logger::clear2ndlog();

    {
        auto tst_layer = std::move(tst_layer_ref);
        // tst_layer.weight_sparsity_enabled = false;
        // tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 0, 0 + 0},
                 "CLUSTERING , ,  "},
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                // {VPUNN::Cycles::NO_ERROR, true, 2000, 2000 + 1000},  //
                // "SOK , no memmove, "},
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                // {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 7900,
                //  7900 + 1000},  // H=8 with K=8 , no padding cannot be split on h
                // "SOH , no memmove, "},
        };

        executeTests(tests);
    }

    // layer aspects
    {
        DPUWorkload tst_wl{std::move(wl_1)};
        {
            DPULayer tst_layer(tst_wl);
            const VPULayerStrategy strategy{1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true};

            Logger::clear2ndlog();
            unsigned cost_cyc{};
            LayerSplitInfo detailed_split;
            ASSERT_NO_THROW(cost_cyc = theModel.Layer(tst_layer, VPUNN::VPUTilingStrategy::SOK, 1U, 2U, false, false,
                                                      true, detailed_split))
                    << tst_layer << strategy;

            EXPECT_EQ(cost_cyc, 1) << tst_layer << toStringLayerSplitInfo(detailed_split) << "\n 2ndLOG:\n"
                                   << Logger::get2ndlog();
        }
        Logger::clear2ndlog();
        std::string err_info;
        CyclesInterfaceType dpu_cost = theModel.get_cost_model().DPU(tst_wl, err_info);
        EXPECT_EQ(dpu_cost, 2) << dpu_cost << "=" << Cycles::toErrorText(dpu_cost) << tst_wl << err_info
                               << "\n 2ndLOG:\n"
                               << Logger::get2ndlog();
    }
}

TEST_F(VPULayerCostModelTest, Extreme_values_Layer_Test_NPU40) {
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

    VPULayerCostModel& theModel = model_4_0;
    Logger::clear2ndlog();
    DPULayer tst_layer(wl);

    unsigned cyc{};
    LayerSplitInfo detailed_split;

    cyc = theModel.Layer(tst_layer, VPUNN::VPUTilingStrategy::SOH_Overlapped, 1U, 2U, false, false, true,
                         detailed_split);
    EXPECT_TRUE(Cycles::isErrorCode(cyc));
    // EXPECT_TRUE(false) << Cycles::toErrorText(cyc);
}
TEST_F(VPULayerCostModelTest, Layer_PRE_split_CLUSTERING) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(60, 7, 512, 1, VPUNN::DataType::UINT8)},    // input dimensions
            {VPUNN::VPUTensor(60, 6, 512, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            {3, 3},                                                       // kernels
            {1, 1},                                                       // strides
            {1, 0, 1, 1}                                                  // padding
    );

    const bool prefetch{true};  // prefetch was done
    VPULayerCostModel& theModel = model_2_7;

    {
        VPUNN::DPULayer tst_layer(std::move(tst_layer_ref));
        tst_layer.weight_sparsity_enabled = true;
        tst_layer.weight_sparsity = 0.723566F;

        std::vector<DPULayer> splitLayers1{tst_layer};

        const VPUTilingStrategy strategy{VPUNN::VPUTilingStrategy::NONE};  // clustering 2T, no memo

        Logger::clear2ndlog();
        CyclesInterfaceType cost_cyc_Layer{};
        LayerSplitInfo detailed_split_layer;
        ASSERT_NO_THROW(cost_cyc_Layer = theModel.Layer(tst_layer, strategy, 1U, 2U, false, false, prefetch,
                                                        detailed_split_layer))
                << tst_layer << (int)strategy << " : " << VPUTilingStrategy_ToText.at(static_cast<int>(strategy));

        Logger::clear2ndlog();
        CyclesInterfaceType pre_split_cost1{};
        LayerSplitInfo detailed_split_pre_layer;
        ASSERT_NO_THROW(pre_split_cost1 = theModel.LayersPreSplit(splitLayers1, 1U, false, false, prefetch,
                                                                  detailed_split_pre_layer))
                << toStringLayerSplitInfo(detailed_split_pre_layer);

        EXPECT_EQ(cost_cyc_Layer, pre_split_cost1)
                << toStringLayerSplitInfo(detailed_split_layer) << toStringLayerSplitInfo(detailed_split_pre_layer)
                << Logger::get2ndlog() << "END ERR";

        Logger::clear2ndlog();
        CyclesInterfaceType pre_split_cost2{};
        LayerSplitInfo detailed_split_pre_layer2;
        std::vector<DPULayer> splitLayers2{};
        splitLayers2.push_back(detailed_split_layer[0].inter_tile_split_layer);
        splitLayers2.push_back(detailed_split_layer[1].inter_tile_split_layer);

        ASSERT_NO_THROW(pre_split_cost2 = theModel.LayersPreSplit(splitLayers2, 1U, false, false, prefetch,
                                                                  detailed_split_pre_layer2))
                << toStringLayerSplitInfo(detailed_split_pre_layer2);

        EXPECT_EQ(pre_split_cost2, pre_split_cost1)
                << toStringLayerSplitInfo(detailed_split_pre_layer2) << toStringLayerSplitInfo(detailed_split_pre_layer)
                << Logger::get2ndlog() << "END ERR";
    }
}

/// DMA CMX to/from DDRare equal because is limited by the maximum time (min bandwith)
TEST_F(VPULayerCostModelTest, Layer_DMA_DDRvsCMX_Smoke) {
    const VPUNN::DPULayer tst_layer_ref(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
                                        {VPUNN::VPUTensor(60, 7, 512, 1, VPUNN::DataType::INT8)},  // input dimensions
                                        {VPUNN::VPUTensor(60, 7, 512, 1, VPUNN::DataType::INT8)},  // output dimensions
                                        {3, 3},                                                    // kernels
                                        {1, 1},                                                    // strides
                                        {1, 1, 1, 1}                                               // padding
    );

    const bool prefetch{true};  // prefetch was done
    for (const auto theModel :
         std::vector<VPULayerCostModel*>{&model_2_7, &model_2_7_no_dma})  // check fallback to DMA theoretical as well
    {
        VPUNN::DPULayer tst_layer(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = true;
        tst_layer.weight_sparsity = 0.723566F;

        std::vector<DPULayer> splitLayers1{tst_layer};

        const VPUTilingStrategy strategy{VPUNN::VPUTilingStrategy::NONE};  // clustering 2T, no memo

        Logger::clear2ndlog();
        LayerSplitInfo detailed_split_layer;
        CyclesInterfaceType cost_cyc_LayerFromDDR{};
        ASSERT_NO_THROW(cost_cyc_LayerFromDDR = theModel->Layer(tst_layer, strategy, 1U, 2U, true, false, prefetch,
                                                                detailed_split_layer))
                << tst_layer << (int)strategy << " : " << VPUTilingStrategy_ToText.at(static_cast<int>(strategy))
                << toStringLayerSplitInfo(detailed_split_layer) << Logger::get2ndlog();

        Logger::clear2ndlog();
        detailed_split_layer.clear();

        CyclesInterfaceType cost_cyc_LayerFromCMX{};
        ASSERT_NO_THROW(cost_cyc_LayerFromCMX = theModel->Layer(tst_layer, strategy, 1U, 2U, false, true, prefetch,
                                                                detailed_split_layer))
                << tst_layer << (int)strategy << " : " << VPUTilingStrategy_ToText.at(static_cast<int>(strategy))
                << toStringLayerSplitInfo(detailed_split_layer) << Logger::get2ndlog();

        EXPECT_NEAR(cost_cyc_LayerFromDDR, cost_cyc_LayerFromCMX, cost_cyc_LayerFromCMX * 0.05)
                << tst_layer << Logger::get2ndlog() << "END ERR";  // 5% tolerance for ddr2cmx vs cmx2ddr

        Logger::clear2ndlog();
        detailed_split_layer.clear();

        CyclesInterfaceType cost_cyc_LayerNoMem{};
        ASSERT_NO_THROW(cost_cyc_LayerNoMem = theModel->Layer(tst_layer, strategy, 1U, 2U, false, false, prefetch,
                                                              detailed_split_layer))
                << tst_layer << (int)strategy << " : " << VPUTilingStrategy_ToText.at(static_cast<int>(strategy))
                << toStringLayerSplitInfo(detailed_split_layer) << Logger::get2ndlog();

        EXPECT_NE(cost_cyc_LayerFromDDR, cost_cyc_LayerNoMem) << tst_layer << Logger::get2ndlog() << "END ERR";

        EXPECT_GT(cost_cyc_LayerFromDDR, cost_cyc_LayerNoMem) << tst_layer << Logger::get2ndlog() << "END ERR";
        EXPECT_GT(cost_cyc_LayerFromCMX, cost_cyc_LayerNoMem) << tst_layer << Logger::get2ndlog() << "END ERR";

        EXPECT_NEAR(cost_cyc_LayerFromDDR, cost_cyc_LayerFromCMX, cost_cyc_LayerFromCMX * 0.05)
                << cost_cyc_LayerFromDDR << "\n"  // 5% tolerance for ddr2cmx vs cmx2ddr
                << cost_cyc_LayerFromCMX << "\n"
                << cost_cyc_LayerNoMem << "\n"
                << tst_layer << Logger::get2ndlog() << "END ERR";
        const auto& outT{tst_layer.outputs[0]};
        auto cmxddr_dma =
                theModel->get_cost_model().DMA(tst_layer.device, outT, outT, MemoryLocation::CMX, MemoryLocation::DRAM);
        auto ddrcmx_dma =
                theModel->get_cost_model().DMA(tst_layer.device, outT, outT, MemoryLocation::DRAM, MemoryLocation::CMX);
        EXPECT_EQ(ddrcmx_dma, cmxddr_dma) << ddrcmx_dma << "\n"
                                          << cmxddr_dma << "\n"
                                          << cost_cyc_LayerFromDDR << "\n"
                                          << cost_cyc_LayerFromCMX << "\n"
                                          << cost_cyc_LayerNoMem << "\n";
    }
}

TEST_F(VPULayerCostModelTest, Dual_Sparsity_Active_Layer_Test_NPU40) {
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

        VPULayerCostModel& theModel = model_4_0;
        run_Tests(tests4_0, theModel, verify_sparsity_influence);
    }
}

TEST_F(VPULayerCostModelTest, Layer_PRE_split_L1) {
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
    VPULayerCostModel& theModel = model_4_0;

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

TEST_F(VPULayerCostModelTest, AVEPOOL_Layer_PRE_split_L2) {
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
    VPULayerCostModel& theModel = model_4_0;
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

TEST_F(VPULayerCostModelTest, L2_instantiated_with_external_DPU) {
    {
        const std::string model_path = VPU_2_7_MODEL_PATH;
        EXPECT_NO_THROW(VPUCostModel x{model_path});
        // VPUCostModel costmodel{model_path};
        auto costmodel{std::make_shared<VPUCostModel>(model_path)};

        const VPULayerCostModel layer_costmodel{costmodel};
        EXPECT_EQ(costmodel.get(), (&layer_costmodel.get_cost_model()));
    }
    {
        const std::string model_path = VPU_4_0_MODEL_PATH;
        EXPECT_NO_THROW(VPUCostModel x{model_path});
        // VPUCostModel costmodel{model_path};
        auto costmodel{std::make_shared<VPUCostModel>(model_path)};

        const VPULayerCostModel layer_costmodel{costmodel};
        EXPECT_EQ(costmodel.get(), (&layer_costmodel.get_cost_model()));
    }
}
}  // namespace VPUNN_unit_tests
