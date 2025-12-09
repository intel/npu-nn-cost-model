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

class VPULayerCostModelTestVPU2x : public VPULayerCostModelTest {
public:
protected:
    void SetUp() override {
		VPULayerCostModelTest::SetUp();
    }
};


TEST_F(VPULayerCostModelTestVPU2x, LayerCostModelVPU_2_0) {
    auto layer = generate_helper_layer(16, 64);
    auto vpu20_layer_cost = layer_models.getModel(VPUDevice::VPU_2_0).Layer(layer, 5, 4);

    // Basic expectations
    EXPECT_GT(vpu20_layer_cost, 0u);
}

TEST_F(VPULayerCostModelTestVPU2x, LayerCostModelVPU_2_7_shv_workload) {
    auto layer = generate_helper_shave_wl_layer(VPUNN::VPUDevice::VPU_2_7, 16, 64);
    auto vpu20_layer_cost = layer_models.getModel(VPUDevice::VPU_2_7).Layer(layer, 5, 4);

    // Basic expectations
    EXPECT_GT(vpu20_layer_cost, 0u);
}

TEST_F(VPULayerCostModelTestVPU2x, LayerCostModelVPU_2_7_shv_wl_bad_name) {
    auto layer = VPUNN::SHAVEWorkload("bad_wl",                                                     // name
                                      VPUNN::VPUDevice::VPU_2_7,                                    // VPU device
                                      {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::FLOAT16)},  // Input tensor
                                      {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::FLOAT16)}   // Output tensor
    );
    auto vpu20_layer_cost = layer_models.getModel(VPUDevice::VPU_2_7).Layer(layer, 5, 4);

    // Basic expectations
    EXPECT_EQ(vpu20_layer_cost, Cycles::ERROR_SHAVE_OPERATOR_MISSING);
}

TEST_F(VPULayerCostModelTestVPU2x, ELTWISE_Concrete_Add14_VPU27) {
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

TEST_F(VPULayerCostModelTestVPU2x, CONVOLUTION_Concrete_Multiply8641_VPU27) {
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

TEST_F(VPULayerCostModelTestVPU2x, AVEPOOL_Concrete_GlobalAveragePool_172_VPU27) {
    const VPUNN::DPULayer tst_layer(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::AVEPOOL,       // goes to DW Conv
                                    {VPUNN::VPUTensor(7, 7, 2048, 1, VPUNN::DataType::UINT8)},  // input dimensions
                                    {VPUNN::VPUTensor(1, 1, 2048, 1, VPUNN::DataType::UINT8)},  // output dimensions
                                    {7, 7},                                                     // kernels
                                    {1, 1},                                                     // strides
                                    {0, 0, 0, 0}                                                // padding
    );
    const VPUNN::VPULayerStrategy tst_strategy{1U,    1U,    2U /*tiles*/, VPUNN::VPUTilingStrategy::NONE,
                                               false, false, true};  // prefetching  true = only DPU data, no DMA.
    const auto origMax = layer_models.getModel(VPUDevice::VPU_2_7).get_maxWorkloadsPerIntraTileSplit();
    layer_models.getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(50);  // original test context was 50

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

    layer_models.getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(origMax);
}

TEST_F(VPULayerCostModelTestVPU2x, AVEPOOL_Concrete_GlobalAveragePool_172_VPU27_SOH) {
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

    const auto origMax = layer_models.getModel(VPUDevice::VPU_2_7).get_maxWorkloadsPerIntraTileSplit();
    layer_models.getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(50);  // original test context was 50
    executeTests(tests);
    layer_models.getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(origMax);
}

TEST_F(VPULayerCostModelTestVPU2x, CONVOLUTION_Concrete_Multiply8648_VPU27) {
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

TEST_F(VPULayerCostModelTestVPU2x, MAXPOOL_avgpoolBased_172_VPU27) {
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

    const auto origMax = layer_models.getModel(VPUDevice::VPU_2_7).get_maxWorkloadsPerIntraTileSplit();
    layer_models.getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(50);  // original test context was 50
    executeTests(tests);
    layer_models.getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(origMax);
}

TEST_F(VPULayerCostModelTestVPU2x, MAXPOOL_avgpoolBased_172_VPU27_SOH) {  // SOH Split possible at limit
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

    const auto origMax = layer_models.getModel(VPUDevice::VPU_2_7).get_maxWorkloadsPerIntraTileSplit();
    layer_models.getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(50);  // original test context was 50
    executeTests(tests);
    layer_models.getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(origMax);
}

TEST_F(VPULayerCostModelTestVPU2x, AVEPOOL_Concrete_GlobalAveragePool_172_BAD_CHANNELS_VPU27) {
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

TEST_F(VPULayerCostModelTestVPU2x, AVEPOOL_Concrete_GlobalAveragePool_172_MOD16_NOMOD32_CH_VPU27) {
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

    const auto origMax = layer_models.getModel(VPUDevice::VPU_2_7).get_maxWorkloadsPerIntraTileSplit();
    layer_models.getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(50);  // original test context was 50

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

    layer_models.getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(origMax);
}

TEST_F(VPULayerCostModelTestVPU2x, AVEPOOL_Concrete_GlobalAveragePool_172_MOD16_NOMOD32_CH_VPU27_SOH) {
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
    const auto origMax = layer_models.getModel(VPUDevice::VPU_2_7).get_maxWorkloadsPerIntraTileSplit();
    layer_models.getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(50);  // original test context was 50

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

    layer_models.getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(origMax);
}
TEST_F(VPULayerCostModelTestVPU2x, CONVOLUTION_Concrete_Multiply6326_FLOAT_FLOAT_VPU27) {
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
TEST_F(VPULayerCostModelTestVPU2x, CONVOLUTION_Concrete_Multiply6326_FLOAT_INT_VPU27_EISXW_76882) {
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
TEST_F(VPULayerCostModelTestVPU2x, CONVOLUTION_Concrete_Multiply6326_INT_FLOAT_VPU27) {
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

TEST_F(VPULayerCostModelTestVPU2x, CONVOLUTION_Concrete_Multiply6326_INT_INT_VPU27) {
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

TEST_F(VPULayerCostModelTestVPU2x, CONVOLUTION_Concrete_Multiply6326_FI_VPU27) {
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

TEST_F(VPULayerCostModelTestVPU2x, Fused_234_3xELEMENTWISE) {
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

TEST_F(VPULayerCostModelTestVPU2x, AVEPOOL_Concrete_GlobalAveragePool_172_MaxWorkloadSPlitAndDetails) {
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

    const auto origMax = layer_models.getModel(VPUDevice::VPU_2_7).get_maxWorkloadsPerIntraTileSplit();
    layer_models.getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(50);  // original test context was 50

    {  // CLUSTERING
        TestInput tin{tst_layer, tst_strategy};
        tin.strategy.tiling_strategy = VPUNN::VPUTilingStrategy::NONE;

        const auto maxWSplit = layer_models.getModel(VPUDevice::VPU_2_7).get_maxWorkloadsPerIntraTileSplit();

        EXPECT_EQ(50U, maxWSplit) << "max workloads split must be default";

        // Very large because 2032 is  mod 16, but 31X64 +1X48 is not accepted, 63x32+1X16 is not reached (limit 50)
        // and we cannot reach a split like: 31x64 = 1984  , 1x32=32,  1x16=16  =>2032 split in 33 workloads
        // the 1x2032 is not accepted because ch not 16/32/64
        TestExpectations texp1{VPUNN::Cycles::ERROR_TILE_OUTPUT, true, 2408000, 2408000 + 1000};

        DoRegularTest(tin, texp1, "CLUST not valid , cannot split");

        layer_models.getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(64U);
        EXPECT_EQ(layer_models.getModel(VPUDevice::VPU_2_7).get_maxWorkloadsPerIntraTileSplit(), 64U)
                << "max workloads split must be set";

        // 63x32+1X16 is reached (limit 64)
        TestExpectations texp2{VPUNN::Cycles::NO_ERROR, true, 30000, 30000 + 2000};

        DoRegularTest(tin, texp2, "CLUST must be split to 64");
    }
    {
        TestInput tin{std::move(tst_layer), tst_strategy};
        tin.strategy.tiling_strategy = VPUNN::VPUTilingStrategy::NONE;
        LayerSplitInfo splitInfo{};

        auto t = layer_models.getModel(VPUDevice::VPU_2_7)
                         .Layer(tin.l1, tin.strategy.tiling_strategy, 1U, 2, false, false, true, splitInfo);

        EXPECT_FALSE(Cycles::isErrorCode(t));

        ASSERT_EQ(splitInfo.size(), 2U) << "Must be 2 tiles!";
        EXPECT_EQ(splitInfo[0].best_intra_tile_split.second.size(), 64U) << "Tile 1 Must be 64 workloads!";
        EXPECT_EQ(splitInfo[1].best_intra_tile_split.second.size(), 64U) << "Tile 2 Must be 64 workloads!";
        EXPECT_EQ(splitInfo[0].best_intra_tile_split.first, splitInfo[1].best_intra_tile_split.first)
                << "Tiles must be equal in cycles";
    }

    layer_models.getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(origMax);
}

TEST_F(VPULayerCostModelTestVPU2x, CONVOLUTION_Concrete_Multiply8641_VPU27_Prefetch) {
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

TEST_F(VPULayerCostModelTestVPU2x, CONVOLUTION_Concrete_Multiply6326_FLOAT_FLOAT_VPU27_Prefetch) {
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

TEST_F(VPULayerCostModelTestVPU2x, MAXPOOLPrefetchTest_NPU27) {
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

    const auto origMax = layer_models.getModel(VPUDevice::VPU_2_7).get_maxWorkloadsPerIntraTileSplit();
    layer_models.getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(50);  // original test context was 50
    executeTests(tests);
    layer_models.getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(origMax);
}

TEST_F(VPULayerCostModelTestVPU2x, AVEPOOLPrefetchTest_172_NPU27) {
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
    const auto origMax = layer_models.getModel(VPUDevice::VPU_2_7).get_maxWorkloadsPerIntraTileSplit();
    layer_models.getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(50);  // original test context was 50
    executeTests(tests);
    layer_models.getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(origMax);
}

TEST_F(VPULayerCostModelTestVPU2x, Unet_perf_SOH_SOK_after_SOHO) {
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
    VPULayerCostModel& theModel = layer_models.getModel(VPUDevice::VPU_2_7);
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
TEST_F(VPULayerCostModelTestVPU2x, DISABLED_Z_InvestigationTest) {
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

    VPULayerCostModel& theModel = layer_models.getModel(VPUDevice::VPU_2_7);
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

TEST_F(VPULayerCostModelTestVPU2x, Layer_PRE_split_CLUSTERING) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(60, 7, 512, 1, VPUNN::DataType::UINT8)},    // input dimensions
            {VPUNN::VPUTensor(60, 6, 512, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            {3, 3},                                                       // kernels
            {1, 1},                                                       // strides
            {1, 0, 1, 1}                                                  // padding
    );

    const bool prefetch{true};  // prefetch was done
    VPULayerCostModel& theModel = layer_models.getModel(VPUDevice::VPU_2_7);

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

}  // namespace VPUNN_unit_tests