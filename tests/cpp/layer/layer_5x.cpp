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

class VPULayerCostModelTestNPU5x : public VPULayerCostModelTest {
public:
protected:
    void SetUp() override {
        VPULayerCostModelTest::SetUp();
    }
};

TEST_F(VPULayerCostModelTestNPU5x, LayerCostModel_NPU50_SHV_too_big) {
    auto layer = VPUNN::SHAVEWorkload("swish",                                                       // name
                                      VPUNN::VPUDevice::NPU_5_0,                                     // VPU device
                                      {VPUNN::VPUTensor(4, 16, 8192, 1, VPUNN::DataType::FLOAT16)},  // Input tensor
                                      {VPUNN::VPUTensor(4, 16, 8192, 1, VPUNN::DataType::FLOAT16)}   // Output tensor
    );
    auto vpu50_layer_cost = layer_models.getModel(VPUDevice::NPU_5_0).Layer(layer, 2, 3, true, true);

    // Basic expectations
    EXPECT_LT(vpu50_layer_cost, Cycles::ERROR_SHAVE);
}

TEST_F(VPULayerCostModelTestNPU5x, TestSerializer_PreSplit) {
    const DPUWorkload wl1{
            VPUDevice::NPU_5_0,
            Operation::CONVOLUTION,
            {VPUTensor(56, 56, 128, 1, DataType::UINT8)},  // input dimensions
            {VPUTensor(56, 56, 128, 1, DataType::UINT8)},  // output dimensions
            {3, 3},                                        // kernels
            {1, 1},                                        // strides
            {1, 1, 1, 1},                                  // padding
            ExecutionMode::CUBOID_16x16,                   // execution mode
            ActivationFunction::NONE,                      // activation
            0.0F,                                          // act_sparsity
            0.0F,                                          // weight_sparsity
            {swz_def, swz_def},                            // input_swizzling
            {swz_def},                                     // output_swizzling
            1,                                             // output_write_tiles
            {0, 0, 0, 0},                                  // offsets
            ISIStrategy::CLUSTERING,                       // isi_strategy
            false,                                         // weight_sparsity_enabled
    };

    const DPUWorkload wl2{
            VPUDevice::NPU_5_0,
            Operation::CONVOLUTION,
            {VPUTensor(28, 36, 64, 1, DataType::UINT8)},  // input dimensions
            {VPUTensor(28, 36, 64, 1, DataType::UINT8)},  // output dimensions
            {3, 3},                                       // kernels
            {1, 1},                                       // strides
            {1, 1, 1, 1},                                 // padding
            ExecutionMode::CUBOID_16x16,                  // execution mode
            ActivationFunction::NONE,                     // activation
            0.0F,                                         // act_sparsity
            0.0F,                                         // weight_sparsity
            {swz_def, swz_def},                           // input_swizzling
            {swz_def},                                    // output_swizzling
            1,                                            // output_write_tiles
            {0, 0, 0, 0},                                 // offsets
            ISIStrategy::CLUSTERING,                      // isi_strategy
            false,                                        // weight_sparsity_enabled
    };

    const bool prefetch{true};  // prefetch was done
    VPULayerCostModel& theModel = layer_models.getModel(VPUDevice::NPU_5_0);
    const LayersValidation layer_validator;

    DPULayer wl_layer1(wl1);
    DPULayer wl_layer2(wl2);
    size_t serializer_layer_uid = 123;

    // L2 pre split
    Logger::clear2ndlog();
    CyclesInterfaceType pre_split_cost1{};
    LayerSplitInfo detailed_split_pre_layer{};
    std::vector<DPULayer> splitLayers1{std::move(wl_layer1), std::move(wl_layer2)};
    ASSERT_NO_THROW(pre_split_cost1 =
                            theModel.LayersPreSplit(splitLayers1, 1U, false, false, prefetch, detailed_split_pre_layer,
                                                    serializer_layer_uid, VPUTilingStrategy::SOH_Overlapped))
            << toStringLayerSplitInfo(detailed_split_pre_layer);

    ASSERT_TRUE(!Cycles::isErrorCode(pre_split_cost1))
            << pre_split_cost1 << " " << toStringLayerSplitInfo(detailed_split_pre_layer);
}

TEST_F(VPULayerCostModelTestNPU5x, TestSerializer_LayerLevel) {
    const bool force_fail{false};        // controls force failing assertion
    const int fail{force_fail ? 0 : 1};  // 1 neutral, 0 fail
    show_split = false;
    // EXPECT_TRUE(false);

    const DPUWorkload wl1{
            VPUDevice::NPU_5_0,
            Operation::CONVOLUTION,
            {VPUTensor(56, 56, 128, 1, DataType::UINT8)},  // input dimensions
            {VPUTensor(56, 56, 128, 1, DataType::UINT8)},  // output dimensions
            {3, 3},                                        // kernels
            {1, 1},                                        // strides
            {1, 1, 1, 1},                                  // padding
            ExecutionMode::CUBOID_16x16,                   // execution mode
            ActivationFunction::NONE,                      // activation
            0.0F,                                          // act_sparsity
            0.0F,                                          // weight_sparsity
            {swz_def, swz_def},                            // input_swizzling
            {swz_def},                                     // output_swizzling
            1,                                             // output_write_tiles
            {0, 0, 0, 0},                                  // offsets
            ISIStrategy::CLUSTERING,                       // isi_strategy
            false,                                         // weight_sparsity_enabled
    };

    const DPUWorkload wl2{
            VPUDevice::NPU_5_0,
            Operation::DW_CONVOLUTION,
            {VPUTensor(56, 56, 128, 1, DataType::UINT8)},  // input dimensions
            {VPUTensor(56, 56, 128, 1, DataType::UINT8)},  // output dimensions
            {3, 3},                                        // kernels
            {1, 1},                                        // strides
            {1, 1, 1, 1},                                  // padding
            ExecutionMode::CUBOID_16x16,                   // execution mode
            ActivationFunction::NONE,                      // activation
            0.0F,                                          // act_sparsity
            0.0F,                                          // weight_sparsity
            {swz_def, swz_def},                            // input_swizzling
            {swz_def},                                     // output_swizzling
            1,                                             // output_write_tiles
            {0, 0, 0, 0},                                  // offsets
            ISIStrategy::CLUSTERING,                       // isi_strategy
            false,                                         // weight_sparsity_enabled
    };
    const bool prefetch{true};

    {
        const DPULayer tst_layer1(wl1);
        const DPULayer tst_layer2(wl2);
        const std::vector<TestCase> tests{
                {{std::move(tst_layer1),
                  {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 28400, fail * 28401 + 1500},
                 "SOHO , no memmove, CONV "},
                {{std::move(tst_layer2),
                  {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 4200, fail * 4201 + 1000},
                 "SOHO , no memmove, DW_CONV "},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTestNPU5x, Layer_PRE_split_Shave_Sin) {
    VPUNN::SHAVEWorkload::Parameters params = {};
    VPUNN::SHAVEWorkload::ExtraParameters extra_param = {};
    extra_param["level"] = "VPU";

    const VPUNN::SHAVEWorkload test_shave_wl{"sin",
                                             VPUDevice::NPU_5_0,
                                             {VPUNN::VPUTensor(128, 1024, 1, 1, VPUNN::DataType::FLOAT16, Layout::YXZ)},
                                             {VPUNN::VPUTensor(128, 1024, 1, 1, VPUNN::DataType::FLOAT16, Layout::YXZ)},
                                             params,
                                             extra_param};
    VPULayerCostModel& theModel = layer_models.getModel(VPUDevice::NPU_5_0);

    const std::vector<SHAVEWorkload> splitLayers{test_shave_wl, test_shave_wl, test_shave_wl};
    CyclesInterfaceType pre_split_cost = theModel.LayersPreSplit(splitLayers, 2, true, true);

    EXPECT_GT(pre_split_cost, Cycles::NO_ERROR);
    EXPECT_LE(pre_split_cost, Cycles::START_ERROR_RANGE);
}

TEST_F(VPULayerCostModelTestNPU5x, DW_CONV_intraTileSplit_channels_options) {
    auto mk_wl = [&](int channels) {
        return DPUWorkload{
                VPUDevice::NPU_5_0,
                Operation::DW_CONVOLUTION,
                {VPUTensor(10, 6, static_cast<unsigned int>(channels), 1, DataType::UINT8)},  // input dimensions
                {VPUTensor(10, 6, static_cast<unsigned int>(channels), 1, DataType::UINT8)},  // output dimensions
                {1, 1},                                                                       // kernels
                {1, 1},                                                                       // strides
                {0, 0, 0, 0},                                                                 // padding
                ExecutionMode::CUBOID_16x16,                                                  // execution mode
                ActivationFunction::NONE,                                                     // activation
                0.0F,                                                                         // act_sparsity
                0.0F,                                                                         // weight_sparsity
                {swz_def, swz_def},                                                           // input_swizzling
                {swz_def},                                                                    // output_swizzling
                1,                                                                            // output_write_tiles
                {0, 0, 0, 0},                                                                 // offsets
                ISIStrategy::CLUSTERING,                                                      // isi_strategy
                false,                                                                        // weight_sparsity_enabled
        };
    };

    struct TestInput_ {
        DPUWorkload wl;
    };

    struct TestExpectation_ {
        std::vector<std::pair<int, bool>> dw_intraTiles_ch_split_expect;
    };

    struct TestCase_ {
        TestInput_ t_in;
        TestExpectation_ t_exp;
    };

    using TestsVector_ = std::vector<TestCase_>;
    const bool prefetch{true};  // prefetch was done
    // const LayersValidation layer_validator;
    size_t serializer_layer_uid = 123;
    VPULayerCostModel& theModel = layer_models.getModel(VPUDevice::NPU_5_0);

    auto testIntraTilesOptions = [&](TestsVector_ tests) {
        std::vector<std::pair<int, bool>> dw_intraTiles_ch_options{{16, false},  {32, false},  {64, false},
                                                                   {96, false},  {128, false}, {160, false},
                                                                   {192, false}, {224, false}, {256, false}};
        for (auto t : tests) {
            const LayersValidation layer_validator;

            DPULayer wl_layer1(t.t_in.wl);

            // L2 pre split
            Logger::clear2ndlog();
            CyclesInterfaceType pre_split_cost1{};
            LayerSplitInfo detailed_split_pre_layer{};
            std::vector<DPULayer> splitLayers1{std::move(wl_layer1)};
            ASSERT_NO_THROW(pre_split_cost1 = theModel.LayersPreSplit(splitLayers1, 1U, false, false, prefetch,
                                                                      detailed_split_pre_layer, serializer_layer_uid,
                                                                      VPUTilingStrategy::SOK))
                    << toStringLayerSplitInfo(detailed_split_pre_layer);

            ASSERT_TRUE(!Cycles::isErrorCode(pre_split_cost1))
                    << Cycles::toErrorText(pre_split_cost1) << " " << toStringLayerSplitInfo(detailed_split_pre_layer);

            int tile_index = 1;
            // for each tile we compare the channels of its intra-tile workloads
            for (const auto& split : detailed_split_pre_layer) {
                // compare intraTiles channels after splits with the different options, you can find them in
                // dw_intraTiles_ch_options
                for (int intraTile_index = 0; intraTile_index < static_cast<int>(split.all_intra_tile_splits.size());
                     intraTile_index++) {
                    const auto wls{split.all_intra_tile_splits[intraTile_index]
                                           .workloads};  // workloads after intra-tile split

                    const auto channels =
                            static_cast<int>(wls[0].outputs[0].channels());  // channels after intra-tile split

                    // mark the found option in dw_intraTiles_ch_options
                    for (auto& p : dw_intraTiles_ch_options) {
                        if (p.first == channels) {
                            p.second = true;
                            break;
                        }
                    }
                }

                // here we test that all the options defined in dw_intraTiles_ch_options have been found
                for (int i = 0; i < static_cast<int>(dw_intraTiles_ch_options.size()); i++) {
                    EXPECT_EQ(dw_intraTiles_ch_options[i].second,
                              t.t_exp.dw_intraTiles_ch_split_expect[i]
                                      .second)  // check if the split can be performed or not
                            << " Channels option " << dw_intraTiles_ch_options[i].first
                            << " found/not found mismatch with expected in intra-tile splits. Tile index " << tile_index
                            << std::endl
                            << t.t_in.wl;

                    dw_intraTiles_ch_options[i].second = false;  // reset for next tile (if any)
                }
                tile_index++;
            }
        }
    };

    // clang-format off
    TestsVector_ tests {
//      channels    |   first pair value means valid intraTiles channels options, the second value is to mark if this device accept the value as intra-tile channels value -> NPU5 accepts only 16,32,64

        {{mk_wl(32)},   {{{16, true}, {32, true}, {64, false}, {96, false}, {128, false}, {160, false}, {192, false}, {224, false}, {256, false}}}},  
        {{mk_wl(576)},  {{{16, true}, {32, true}, {64, true},  {96, false}, {128, false}, {160, false}, {192, false}, {224, false}, {256, false}}}},  
        {{mk_wl(448)},  {{{16, true}, {32, true}, {64, true},  {96, false}, {128, false}, {160, false}, {192, false}, {224, false}, {256, false}}}},  
        {{mk_wl(1024)}, {{{16, true}, {32, true}, {64, true},  {96, false}, {128, false}, {160, false}, {192, false}, {224, false}, {256, false}}}} 
    };
    // clang-format on

    testIntraTilesOptions(std::move(tests));
}

}  // namespace VPUNN_unit_tests