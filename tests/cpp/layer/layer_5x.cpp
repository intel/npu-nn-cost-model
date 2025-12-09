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

}  // namespace VPUNN_unit_tests