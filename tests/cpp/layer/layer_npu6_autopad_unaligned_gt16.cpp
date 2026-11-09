// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

// ---------------------------------------------------------------------------
// Tests for 149253: NPU6 ODU autopad should accept any output channel
// count including values > 16 that are not multiples of 16.
//
// This file covers three query levels:
//
//   Level 1 — VPUCostModel::DPU(wl)
//     Tests that the L1 DPU cost model does not return an error code when
//     output_autopad=true and output channels are unaligned and > 16.
//
//   Level 2a — VPULayerCostModel::Layer(layer, strategy)
//     Tests that the L2 layer cost model can query cost for such layers.
//
//   Level 2b — VPULayerCostModel::LayersPreSplit(tiles, ...)
//     Tests that LayersPreSplit works for pre-split tiles carrying unaligned
//     output channels > 16 with output_autopad=true.
//
// All tests in sections 1, 2a and 2b are EXPECTED TO FAIL until
// VPU6_0_WorkloadValidValues::get_output_channels_restriction() is broadened
// to accept any channel count when output_autopad=true (149253).
// ---------------------------------------------------------------------------

#include "layer.h"
#include "vpu_cost_model.h"
#include "vpu/utils.h"
#include "vpu/validation/dpu_operations_sanitizer.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "common/common_helpers.h"
#include "core/logger.h"
#include "core/serializer.h"
#include "vpu/validation/data_dpu_operation.h"
#include "vpu/validation/serializable_dpu.h"

namespace VPUNN_unit_tests {
using namespace VPUNN;

// ============================================================================
// Fixtures
// ============================================================================

// ---------------------------------------------------------------------------
// Fixture for DPU cost model (Level 1) tests.
// Uses VPUCostModel with the NPU6 neural-network model so that, once the
// validation fix is applied, the workloads reach the NN and produce real costs.
// ---------------------------------------------------------------------------
class DPUCostModel_NPU6_AutopadUnalignedGt16_Test : public ::testing::Test {
protected:
    VPUCostModel model{VPUDevice::NPU_6_0, NPU_6_0_MODEL_PATH};

    const std::vector<unsigned int> unaligned_gt16_channels{17u, 20u, 24u, 30u, 31u};

    // Helper: make a minimal CONVOLUTION workload for NPU6 with given output channels
    DPUWorkload make_conv(unsigned int out_ch, ExecutionMode exec = ExecutionMode::CUBOID_16x16) const {
        DPUWorkload wl{
                VPUDevice::NPU_6_0,
                Operation::CONVOLUTION,
                {VPUTensor(28, 28, 32, 1, DataType::UINT8)},
                {VPUTensor(28, 28, out_ch, 1, DataType::UINT8)},
                {1, 1},
                {1, 1},
                {0, 0, 0, 0},
                exec,
                ActivationFunction::NONE,
                0.0F,
                0.0F,
                {swz_def, swz_def},
                {swz_def},
                1,
                {0, 0, 0, 0},
                ISIStrategy::CLUSTERING,
                false,
        };
        wl.output_autopad = true;
        return wl;
    }

    // Helper: make an ELTWISE workload for NPU6 with given output channels
    DPUWorkload make_eltwise(unsigned int out_ch) const {
        DPUWorkload wl{
                VPUDevice::NPU_6_0,
                Operation::ELTWISE,
                {VPUTensor(28, 9, 32, 1, DataType::UINT8)},
                {VPUTensor(28, 9, out_ch, 1, DataType::UINT8)},
                {1, 1},
                {1, 1},
                {0, 0, 0, 0},
                ExecutionMode::CUBOID_8x16,
                ActivationFunction::NONE,
                0.0F,
                0.0F,
                {swz_def, swz_def},
                {swz_def},
                1,
                {0, 0, 0, 0},
                ISIStrategy::CLUSTERING,
                false,
        };
        wl.output_autopad = true;
        wl.superdense_memory = true;
        return wl;
    }

    // Helper: make a MAXPOOL workload for NPU6 with given output channels
    DPUWorkload make_maxpool(unsigned int out_ch) const {
        DPUWorkload wl{
                VPUDevice::NPU_6_0,
                Operation::MAXPOOL,
                {VPUTensor(14, 14, 32, 1, DataType::UINT8)},
                {VPUTensor(14, 14, out_ch, 1, DataType::UINT8)},
                {1, 1},
                {1, 1},
                {0, 0, 0, 0},
                ExecutionMode::CUBOID_16x16,
                ActivationFunction::NONE,
                0.0F,
                0.0F,
                {swz_def, swz_def},
                {swz_def},
                1,
                {0, 0, 0, 0},
                ISIStrategy::CLUSTERING,
                false,
        };
        wl.output_autopad = true;
        return wl;
    }
};

// ---------------------------------------------------------------------------
// Fixture for Layer / LayersPreSplit (Level 2) tests.
// Loads the NPU6 model so costs reach the neural network after the fix.
// ---------------------------------------------------------------------------
class VPULayerCM_NPU6_AutopadUnalignedGt16_Test : public VPULayerCostModelTest {
protected:
    void SetUp() override {
        VPULayerCostModelTest::SetUp();
        layer_models.addModel(VPUDevice::NPU_6_0, &dma_model_invalid, NPU_6_0_MODEL_PATH);
    }

    const bool prefetch{true};
    VPULayerCostModel& theModel = layer_models.getModel(VPUDevice::NPU_6_0);

    const std::vector<unsigned int> unaligned_gt16_channels{17u, 20u, 24u, 30u, 31u, 33u, 40u};

    // Helper: build a CONVOLUTION DPUWorkload for the given output-channel count
    DPUWorkload make_conv_wl(unsigned int out_ch,
                             ExecutionMode exec = ExecutionMode::CUBOID_16x16,
                             ISIStrategy isi = ISIStrategy::CLUSTERING) const {
        DPUWorkload wl{
                VPUDevice::NPU_6_0,
                Operation::CONVOLUTION,
                {VPUTensor(28, 28, 32, 1, DataType::UINT8)},
                {VPUTensor(28, 28, out_ch, 1, DataType::UINT8)},
                {1, 1},
                {1, 1},
                {0, 0, 0, 0},
                exec,
                ActivationFunction::NONE,
                0.0F,
                0.0F,
                {swz_def, swz_def},
                {swz_def},
                1,
                {0, 0, 0, 0},
                isi,
                false,
        };
        wl.output_autopad = true;
        return wl;
    }

    // Helper: build a MAXPOOL DPUWorkload for the given output-channel count
    DPUWorkload make_maxpool_wl(unsigned int out_ch) const {
        DPUWorkload wl{
                VPUDevice::NPU_6_0,
                Operation::MAXPOOL,
                {VPUTensor(14, 14, 32, 1, DataType::UINT8)},
                {VPUTensor(14, 14, out_ch, 1, DataType::UINT8)},
                {1, 1},
                {1, 1},
                {0, 0, 0, 0},
                ExecutionMode::CUBOID_16x16,
                ActivationFunction::NONE,
                0.0F,
                0.0F,
                {swz_def, swz_def},
                {swz_def},
                1,
                {0, 0, 0, 0},
                ISIStrategy::CLUSTERING,
                false,
        };
        wl.output_autopad = true;
        return wl;
    }
};

// ============================================================================
// SECTION 1 — DPU cost model level (VPUCostModel::DPU)
// ============================================================================

// ---------------------------------------------------------------------------
// 1.1  CONVOLUTION: DPU() must not return an error code for unaligned >16
//      output channels when output_autopad=true on NPU6.
// ---------------------------------------------------------------------------
TEST_F(DPUCostModel_NPU6_AutopadUnalignedGt16_Test,
       DPULevel_CONV_OutputAutopad_UnalignedCh_GT16_ShouldReturnValidCost) {
    // EXPECTED TO FAIL before 149253 is implemented.
    // Currently the sanitizer rejects unaligned >16 channels even with autopad,
    // so DPU() returns an error code before ever querying the neural network.

    for (const auto out_ch : unaligned_gt16_channels) {
        DPUWorkload wl = make_conv(out_ch);
        std::string errInfo;
        const auto cost = model.DPU(wl, errInfo);

        EXPECT_FALSE(Cycles::isErrorCode(cost))
                << "FAILS at DPU level: CONV out_ch=" << out_ch << " output_autopad=true on NPU6"
                << "\n  cost=" << cost << " error='" << Cycles::toErrorText(cost) << "' errInfo=" << errInfo
                << "\n  Fix needed (149253): allow any ch count when output_autopad=true in NPU6 validation";
    }
}

// ---------------------------------------------------------------------------
// 1.2  ELTWISE: DPU() must not return error for unaligned >16 output channels.
// ---------------------------------------------------------------------------
TEST_F(DPUCostModel_NPU6_AutopadUnalignedGt16_Test,
       DPULevel_ELTWISE_OutputAutopad_UnalignedCh_GT16_ShouldReturnValidCost) {
    const std::vector<unsigned int> channels{17u, 24u, 30u};

    for (const auto out_ch : channels) {
        DPUWorkload wl = make_eltwise(out_ch);
        std::string errInfo;
        const auto cost = model.DPU(wl, errInfo);

        EXPECT_FALSE(Cycles::isErrorCode(cost))
                << "FAILS at DPU level: ELTWISE out_ch=" << out_ch << " output_autopad=true on NPU6"
                << "\n  cost=" << cost << " error='" << Cycles::toErrorText(cost) << "' errInfo=" << errInfo
                << "\n  Fix needed (149253)";
    }
}

// ---------------------------------------------------------------------------
// 1.3  MAXPOOL: DPU() must not return error for unaligned >16 output channels.
// ---------------------------------------------------------------------------
TEST_F(DPUCostModel_NPU6_AutopadUnalignedGt16_Test,
       DPULevel_MAXPOOL_OutputAutopad_UnalignedCh_GT16_ShouldReturnValidCost) {
    for (const auto out_ch : unaligned_gt16_channels) {
        DPUWorkload wl = make_maxpool(out_ch);
        std::string errInfo;
        const auto cost = model.DPU(wl, errInfo);

        EXPECT_FALSE(Cycles::isErrorCode(cost))
                << "FAILS at DPU level: MAXPOOL out_ch=" << out_ch << " output_autopad=true on NPU6"
                << "\n  cost=" << cost << " error='" << Cycles::toErrorText(cost) << "' errInfo=" << errInfo
                << "\n  Fix needed (149253)";
    }
}

// ---------------------------------------------------------------------------
// 1.4  DPU-level cost must be consistent across execution modes.
//      Both CUBOID_16x16 and CUBOID_8x16 should produce valid costs.
// ---------------------------------------------------------------------------
TEST_F(DPUCostModel_NPU6_AutopadUnalignedGt16_Test,
       DPULevel_CONV_ExecModes_OutputAutopad_UnalignedCh_GT16_ShouldReturnValidCost) {
    const unsigned int out_ch = 20u;  // representative unaligned >16 value

    for (const auto exec : {ExecutionMode::CUBOID_16x16, ExecutionMode::CUBOID_8x16, ExecutionMode::CUBOID_4x16}) {
        DPUWorkload wl = make_conv(out_ch, exec);
        std::string errInfo;
        const auto cost = model.DPU(wl, errInfo);

        EXPECT_FALSE(Cycles::isErrorCode(cost))
                << "FAILS at DPU level: CONV out_ch=" << out_ch
                << " exec=" << static_cast<int>(exec) << " output_autopad=true on NPU6"
                << "\n  cost=" << cost << " error='" << Cycles::toErrorText(cost) << "' errInfo=" << errInfo
                << "\n  Fix needed (149253)";
    }
}

// ============================================================================
// SECTION 2a — Layer cost model level (VPULayerCostModel::Layer)
// ============================================================================

// ---------------------------------------------------------------------------
// 2a.1  Layer(): CONVOLUTION with unaligned >16 output channels and
//        output_autopad=true should return a valid cycle count.
// ---------------------------------------------------------------------------
// TEST_F(VPULayerCM_NPU6_AutopadUnalignedGt16_Test,
//        LayerLevel_CONV_OutputAutopad_UnalignedCh_GT16_ShouldNotReturnError) {
//     // EXPECTED TO FAIL before 149253 is implemented.

//     const VPULayerStrategy strategy{
//             1U,                          // nDPUs
//             1U,                          // nSHVs
//             1U,                          // nTiles
//             VPUTilingStrategy::NONE,     // tiling strategy (single tile, no inter split)
//             false,                       // input in DDR
//             false,                       // output in DDR
//             prefetch                     // prefetching
//     };

//     for (const auto out_ch : unaligned_gt16_channels) {
//         DPUWorkload wl = make_conv_wl(out_ch);
//         DPULayer layer(wl);

//         Logger::clear2ndlog();
//         LayerSplitInfo split_info;
//         CyclesInterfaceType cost{};
//         ASSERT_NO_THROW(cost = theModel.Layer(layer, strategy, split_info))
//                 << "Layer() threw for CONV out_ch=" << out_ch << "\n" << Logger::get2ndlog();

//         EXPECT_FALSE(Cycles::isErrorCode(cost))
//                 << "FAILS at Layer level: CONV out_ch=" << out_ch << " output_autopad=true on NPU6"
//                 << "\n  cost=" << cost << " error='" << Cycles::toErrorText(cost) << "'"
//                 << "\n" << toStringLayerSplitInfo(split_info)
//                 << "\n2ndLog:\n" << Logger::get2ndlog()
//                 << "\n  Fix needed (149253): allow any ch count when output_autopad=true in NPU6 validation";
//     }
// }

// ---------------------------------------------------------------------------
// 2a.2  Layer(): MAXPOOL with unaligned >16 output channels and
//        output_autopad=true should return a valid cycle count.
// ---------------------------------------------------------------------------
// TEST_F(VPULayerCM_NPU6_AutopadUnalignedGt16_Test,
//        LayerLevel_MAXPOOL_OutputAutopad_UnalignedCh_GT16_ShouldNotReturnError) {
//     const VPULayerStrategy strategy{1U, 1U, 1U, VPUTilingStrategy::NONE, false, false, prefetch};
//
//     for (const auto out_ch : unaligned_gt16_channels) {
//         DPUWorkload wl = make_maxpool_wl(out_ch);
//         DPULayer layer(wl);
//
//         Logger::clear2ndlog();
//         LayerSplitInfo split_info;
//         CyclesInterfaceType cost{};
//         ASSERT_NO_THROW(cost = theModel.Layer(layer, strategy, split_info))
//                 << "Layer() threw for MAXPOOL out_ch=" << out_ch << "\n" << Logger::get2ndlog();

//         EXPECT_FALSE(Cycles::isErrorCode(cost))
//                 << "FAILS at Layer level: MAXPOOL out_ch=" << out_ch << " output_autopad=true on NPU6"
//                 << "\n  cost=" << cost << " error='" << Cycles::toErrorText(cost) << "'"
//                 << "\n" << toStringLayerSplitInfo(split_info)
//                 << "\n  Fix needed (149253)";
//     }
// }

// // ---------------------------------------------------------------------------
// // 2a.3  Layer() with multi-tile SOH_Overlapped strategy: CONVOLUTION with
// //        unaligned >16 output channels should work across tiles.
// //
// // Representative scenario: the compiler splits a CONV layer over H into 2
// // tiles and asks VPUNN for the inter-tile + intra-tile split cost.
// // ---------------------------------------------------------------------------
// TEST_F(VPULayerCM_NPU6_AutopadUnalignedGt16_Test,
//        LayerLevel_CONV_SOH_OutputAutopad_UnalignedCh_GT16_ShouldNotReturnError) {
//     // 2-tile SOH_Overlapped strategy
//     const VPULayerStrategy strategy{1U, 1U, 2U, VPUTilingStrategy::SOH_Overlapped, false, false, prefetch};

//     const std::vector<unsigned int> channels_to_test{17u, 24u, 33u};

//     for (const auto out_ch : channels_to_test) {
//         // Larger spatial dims to make H-splitting meaningful
//         DPUWorkload wl{
//                 VPUDevice::NPU_6_0,
//                 Operation::CONVOLUTION,
//                 {VPUTensor(56, 56, 32, 1, DataType::UINT8)},
//                 {VPUTensor(56, 56, out_ch, 1, DataType::UINT8)},
//                 {1, 1},
//                 {1, 1},
//                 {0, 0, 0, 0},
//                 ExecutionMode::CUBOID_16x16,
//                 ActivationFunction::NONE,
//                 0.0F,
//                 0.0F,
//                 {swz_def, swz_def},
//                 {swz_def},
//                 1,
//                 {0, 0, 0, 0},
//                 ISIStrategy::CLUSTERING,
//                 false,
//         };
//         wl.output_autopad = true;
//         DPULayer layer(wl);

//         Logger::clear2ndlog();
//         LayerSplitInfo split_info;
//         CyclesInterfaceType cost{};
//         ASSERT_NO_THROW(cost = theModel.Layer(layer, strategy, split_info))
//                 << "Layer() threw for CONV SOH out_ch=" << out_ch << "\n" << Logger::get2ndlog();

//         EXPECT_FALSE(Cycles::isErrorCode(cost))
//                 << "FAILS at Layer level (SOH): CONV out_ch=" << out_ch << " output_autopad=true on NPU6"
//                 << "\n  cost=" << cost << " error='" << Cycles::toErrorText(cost) << "'"
//                 << "\n" << toStringLayerSplitInfo(split_info)
//                 << "\n  Fix needed (149253)";
//     }
// }

// ============================================================================
// SECTION 2b — LayersPreSplit level (VPULayerCostModel::LayersPreSplit)
// ============================================================================

// ---------------------------------------------------------------------------
// 2b.1  LayersPreSplit(): single tile, CONVOLUTION with unaligned >16 output
//        channels and output_autopad=true should return a valid cycle count.
//
// This is the most common compiler usage: a layer already split into N tiles
// is submitted tile-by-tile for intra-tile workload optimisation.
// ---------------------------------------------------------------------------
TEST_F(VPULayerCM_NPU6_AutopadUnalignedGt16_Test,
       LayersPreSplit_CONV_OutputAutopad_UnalignedCh_GT16_ShouldNotReturnError) {
    // EXPECTED TO FAIL before 149253 is implemented.

    for (const auto out_ch : unaligned_gt16_channels) {
        DPUWorkload wl = make_conv_wl(out_ch);
        const std::vector<DPULayer> splitLayers{DPULayer(wl)};

        Logger::clear2ndlog();
        CyclesInterfaceType cost{};
        LayerSplitInfo split_info;
        ASSERT_NO_THROW(cost = theModel.LayersPreSplit(splitLayers, 1U, false, false, prefetch, split_info))
                << "LayersPreSplit() threw for CONV out_ch=" << out_ch << "\n" << Logger::get2ndlog();

        EXPECT_FALSE(Cycles::isErrorCode(cost))
                << "FAILS at LayersPreSplit level: CONV out_ch=" << out_ch << " output_autopad=true on NPU6"
                << "\n  cost=" << cost << " error='" << Cycles::toErrorText(cost) << "'"
                << "\n" << toStringLayerSplitInfo(split_info)
                << "\n2ndLog:\n" << Logger::get2ndlog()
                << "\n  Fix needed (149253)";
    }
}

// ---------------------------------------------------------------------------
// 2b.2  LayersPreSplit(): multiple tiles (H-split), CONVOLUTION with unaligned
//        >16 output channels.
//
// Simulates a layer split into 4 H-tiles; each tile carries the same unaligned
// output channel count.  The compiler passes all tiles at once to VPUNN.
// ---------------------------------------------------------------------------
TEST_F(VPULayerCM_NPU6_AutopadUnalignedGt16_Test,
       LayersPreSplit_4Tiles_CONV_OutputAutopad_UnalignedCh_GT16_ShouldNotReturnError) {
    const std::vector<unsigned int> channels_to_test{17u, 24u, 31u, 33u};

    for (const auto out_ch : channels_to_test) {
        // 4 equal H-splits of a 56×56 CONV
        std::vector<DPULayer> splitLayers;
        for (int t = 0; t < 4; ++t) {
            DPUWorkload wl{
                    VPUDevice::NPU_6_0,
                    Operation::CONVOLUTION,
                    {VPUTensor(56, 14, 32, 1, DataType::UINT8)},   // H/4 per tile
                    {VPUTensor(56, 14, out_ch, 1, DataType::UINT8)},
                    {1, 1},
                    {1, 1},
                    {0, 0, 0, 0},
                    ExecutionMode::CUBOID_16x16,
                    ActivationFunction::NONE,
                    0.0F,
                    0.0F,
                    {swz_def, swz_def},
                    {swz_def},
                    1,
                    {0, 0, 0, 0},
                    ISIStrategy::CLUSTERING,
                    false,
            };
            wl.output_autopad = true;
            splitLayers.emplace_back(wl);
        }

        Logger::clear2ndlog();
        CyclesInterfaceType cost{};
        LayerSplitInfo split_info;
        ASSERT_NO_THROW(cost = theModel.LayersPreSplit(splitLayers, 1U, false, false, prefetch, split_info))
                << "LayersPreSplit() threw for 4-tile CONV out_ch=" << out_ch
                << "\n2ndLog:\n" << Logger::get2ndlog();

        EXPECT_FALSE(Cycles::isErrorCode(cost))
                << "FAILS at LayersPreSplit level (4 tiles): CONV out_ch=" << out_ch
                << " output_autopad=true on NPU6"
                << "\n  cost=" << cost << " error='" << Cycles::toErrorText(cost) << "'"
                << "\n" << toStringLayerSplitInfo(split_info)
                << "\n  Fix needed (149253)";

        ASSERT_EQ(split_info.size(), 4U) << "Expected 4 tile entries for out_ch=" << out_ch;
    }
}

// ---------------------------------------------------------------------------
// 2b.3  LayersPreSplit(): MAXPOOL tiles with unaligned >16 output channels.
//
// Relu is often compiled to MAXPOOL on NPU6.  The compiler may produce tiles
// with channel counts that are > 16 and unaligned, e.g. when the layer has
// 20 or 24 channels.
// ---------------------------------------------------------------------------
TEST_F(VPULayerCM_NPU6_AutopadUnalignedGt16_Test,
       LayersPreSplit_MAXPOOL_OutputAutopad_UnalignedCh_GT16_ShouldNotReturnError) {
    const std::vector<unsigned int> channels_to_test{17u, 20u, 24u, 30u};

    for (const auto out_ch : channels_to_test) {
        // 4 H-split tiles (e.g. original layer H=80 → 4 x H=20)
        std::vector<DPULayer> splitLayers;
        for (int t = 0; t < 4; ++t) {
            DPUWorkload wl{
                    VPUDevice::NPU_6_0,
                    Operation::MAXPOOL,
                    {VPUTensor(32, 20, 32, 1, DataType::FLOAT16, Layout::ZXY)},
                    {VPUTensor(32, 20, out_ch, 1, DataType::FLOAT16, Layout::ZXY)},
                    {1, 1},
                    {1, 1},
                    {0, 0, 0, 0},
                    ExecutionMode::CUBOID_16x16,
                    ActivationFunction::RELU,
                    0.0F,
                    0.0F,
                    {swz_def, swz_def},
                    {swz_def},
                    1,
                    {0, 0, 0, 0},
                    ISIStrategy::CLUSTERING,
                    false,
            };
            wl.output_autopad = true;
            splitLayers.emplace_back(wl);
        }

        Logger::clear2ndlog();
        CyclesInterfaceType cost{};
        LayerSplitInfo split_info;
        ASSERT_NO_THROW(cost = theModel.LayersPreSplit(splitLayers, 1U, false, false, prefetch, split_info))
                << "LayersPreSplit() threw for MAXPOOL out_ch=" << out_ch
                << "\n2ndLog:\n" << Logger::get2ndlog();

        EXPECT_FALSE(Cycles::isErrorCode(cost))
                << "FAILS at LayersPreSplit level: MAXPOOL out_ch=" << out_ch
                << " output_autopad=true on NPU6"
                << "\n  cost=" << cost << " error='" << Cycles::toErrorText(cost) << "'"
                << "\n" << toStringLayerSplitInfo(split_info)
                << "\n  Fix needed (149253): allow any ch count when output_autopad=true";
    }
}

// ---------------------------------------------------------------------------
// 2b.4  LayersPreSplit(): DW_CONVOLUTION tiles with unaligned >16 channels,
//        both input_autopad and output_autopad set.
//
// DW_CONVOLUTION requires input_ch == output_ch.  When channels are unaligned
// and > 16 the compiler sets BOTH autopad flags.  This test documents the
// expected behaviour after BOTH the input and output restriction fixes are
// applied.  Currently it fails because the output channel restriction alone
// blocks the workload.
//
// NOTE: this test may also require a matching relaxation of the input channel
// restriction for DW_CONVOLUTION on NPU6 in addition to 149253.
// ---------------------------------------------------------------------------
TEST_F(VPULayerCM_NPU6_AutopadUnalignedGt16_Test,
       LayersPreSplit_DWConv_BothAutopads_UnalignedCh_GT16_ShouldNotReturnError) {
    const std::vector<unsigned int> channels_to_test{17u, 20u, 24u};

    for (const auto ch : channels_to_test) {
        // Single tile, channel-preserving DW_CONV
        std::vector<DPULayer> splitLayers;
        {
            DPUWorkload wl{
                    VPUDevice::NPU_6_0,
                    Operation::DW_CONVOLUTION,
                    {VPUTensor(28, 28, 32, 1, DataType::FLOAT16, Layout::ZXY)},
                    {VPUTensor(28, 28, ch, 1, DataType::FLOAT16, Layout::ZXY)},
                    {1, 1},
                    {1, 1},
                    {0, 0, 0, 0},
                    ExecutionMode::CUBOID_16x16,
                    ActivationFunction::NONE,
                    0.0F,
                    0.0F,
                    {swz_def, swz_def},
                    {swz_def},
                    1,
                    {0, 0, 0, 0},
                    ISIStrategy::CLUSTERING,
                    false,
            };
            wl.output_autopad = true;
            splitLayers.emplace_back(wl);
        }

        Logger::clear2ndlog();
        CyclesInterfaceType cost{};
        LayerSplitInfo split_info;
        ASSERT_NO_THROW(cost = theModel.LayersPreSplit(splitLayers, 1U, false, false, prefetch, split_info))
                << "LayersPreSplit() threw for DW_CONV ch=" << ch
                << "\n2ndLog:\n" << Logger::get2ndlog();

        EXPECT_FALSE(Cycles::isErrorCode(cost))
                << "FAILS at LayersPreSplit level: DW_CONV ch=" << ch
                << " input_autopad=true, output_autopad=true on NPU6"
                << "\n  cost=" << cost << " error='" << Cycles::toErrorText(cost) << "'"
                << "\n" << toStringLayerSplitInfo(split_info)
                << "\n  Fix needed (149253) and possibly input channel restriction for DW_CONV";
    }
}

// ============================================================================
// SECTION 3 — Control cases for levels 2a/2b
//
// These tests must NOT fail after the implementation fix.
// ============================================================================

// ---------------------------------------------------------------------------
// 3.1  LayersPreSplit(): aligned channels (multiples of 16) — control case
//      that must not regress.
// ---------------------------------------------------------------------------
TEST_F(VPULayerCM_NPU6_AutopadUnalignedGt16_Test,
       Control_LayersPreSplit_CONV_AlignedCh_ShouldContinueToPass) {
    const std::vector<unsigned int> aligned_channels{16u, 32u, 64u};

    for (const auto out_ch : aligned_channels) {
        DPUWorkload wl = make_conv_wl(out_ch);
        const std::vector<DPULayer> splitLayers{DPULayer(wl)};

        Logger::clear2ndlog();
        CyclesInterfaceType cost{};
        LayerSplitInfo split_info;
        ASSERT_NO_THROW(cost = theModel.LayersPreSplit(splitLayers, 1U, false, false, prefetch, split_info))
                << "LayersPreSplit() threw for aligned CONV out_ch=" << out_ch;

        EXPECT_FALSE(Cycles::isErrorCode(cost))
                << "Regression: aligned CONV out_ch=" << out_ch << " output_autopad=true should still pass"
                << "\n  cost=" << cost << " error='" << Cycles::toErrorText(cost) << "'";
    }
}

// ---------------------------------------------------------------------------
// 3.2  LayersPreSplit(): channels < 16 with output_autopad — control case
//      that must not regress (already valid since NPU5 autopad baseline).
// ---------------------------------------------------------------------------
TEST_F(VPULayerCM_NPU6_AutopadUnalignedGt16_Test,
       Control_LayersPreSplit_CONV_ChLT16_Autopad_ShouldContinueToPass) {
    const std::vector<unsigned int> lt16_channels{1u, 8u, 15u};

    for (const auto out_ch : lt16_channels) {
        DPUWorkload wl = make_conv_wl(out_ch);
        const std::vector<DPULayer> splitLayers{DPULayer(wl)};

        Logger::clear2ndlog();
        CyclesInterfaceType cost{};
        LayerSplitInfo split_info;
        ASSERT_NO_THROW(cost = theModel.LayersPreSplit(splitLayers, 1U, false, false, prefetch, split_info))
                << "LayersPreSplit() threw for CONV out_ch=" << out_ch << " (< 16 with autopad)";

        EXPECT_FALSE(Cycles::isErrorCode(cost))
                << "Regression: CONV out_ch=" << out_ch
                << " (< 16) output_autopad=true should still pass"
                << "\n  cost=" << cost << " error='" << Cycles::toErrorText(cost) << "'";
    }
}

// ============================================================================
// SECTION 4 — Compiler-workload-derived tests (DPU / Layer / LayersPreSplit)
//
// Workload source:
//   device: NPU_6_0
//   op: AVEPOOL
//   input:  {1,1,1008,1} FLOAT16 ZXY
//   output: {1,1,1000,1} FLOAT32 ZXY
//   kernels/strides 1x1 / 1x1,  padding 0
//   exec: CUBOID_16x16
//   superdense_memory=true, output_autopad=true
//   swizzling: KEY_0 on all tensors
//
// 1000 % 16 == 8  ⟹  unaligned, > 16.  All L1/L2 tests here FAIL until fix.
// ============================================================================

class DPUCostModel_NPU6_CompilerAvepool_Test : public ::testing::Test {
protected:
    VPUCostModel model{VPUDevice::NPU_6_0, NPU_6_0_MODEL_PATH};

    // Build the representative workload, optionally overriding output channels
    // and output dtype.  out_ch is capped at ≤256 (AVEPOOL variant limit);
    // for the "exact" representative case use out_ch=248 (rem-8, same residue
    // class as the compiler-observed 1000, within the variant limit).
    DPUWorkload make_wl(unsigned int out_ch = 248u,
                        DataType out_dtype = DataType::FLOAT32) const {
        // Input channels: smallest aligned multiple of 16 >= out_ch.
        // For the representative case (248) use 256 as the input channel count.
        unsigned int in_ch = (out_ch % 16 != 0) ? VPUNN::dw_channel_align(out_ch) : out_ch;

        DPUWorkload wl{
                VPUDevice::NPU_6_0,
                Operation::AVEPOOL,
                {VPUTensor(1, 1, in_ch, 1, DataType::FLOAT16, Layout::ZXY)},
                {VPUTensor(1, 1, out_ch, 1, out_dtype, Layout::ZXY)},
                {1, 1},
                {1, 1},
                {0, 0, 0, 0},
                ExecutionMode::CUBOID_16x16,
                ActivationFunction::NONE,
                0.0F,
                0.0F,
                {Swizzling::KEY_0, Swizzling::KEY_0},
                {Swizzling::KEY_0},
                1,
                {0, 0, 0, 0},
                ISIStrategy::CLUSTERING,
                false,
        };
        wl.superdense_memory = true;
        wl.output_autopad    = true;
        return wl;
    }
};

class VPULayerCM_NPU6_CompilerAvepool_Test : public VPULayerCostModelTest {
protected:
    void SetUp() override {
        VPULayerCostModelTest::SetUp();
        layer_models.addModel(VPUDevice::NPU_6_0, &dma_model_invalid, NPU_6_0_MODEL_PATH);
    }
    const bool prefetch{true};
    VPULayerCostModel& theModel = layer_models.getModel(VPUDevice::NPU_6_0);

    DPUWorkload make_wl(unsigned int out_ch = 248u,
                        DataType out_dtype = DataType::FLOAT32) const {
        unsigned int in_ch = out_ch % 16 != 0 ? ((out_ch / 16) + 1) * 16 : out_ch;


        DPUWorkload wl{
                VPUDevice::NPU_6_0,
                Operation::AVEPOOL,
                {VPUTensor(1, 1, in_ch, 1, DataType::FLOAT16, Layout::ZXY)},
                {VPUTensor(1, 1, out_ch, 1, out_dtype, Layout::ZXY)},
                {1, 1},
                {1, 1},
                {0, 0, 0, 0},
                ExecutionMode::CUBOID_16x16,
                ActivationFunction::NONE,
                0.0F,
                0.0F,
                {Swizzling::KEY_0, Swizzling::KEY_0},
                {Swizzling::KEY_0},
                1,
                {0, 0, 0, 0},
                ISIStrategy::CLUSTERING,
                false,
        };
        wl.superdense_memory = true;
        wl.output_autopad    = true;
        return wl;
    }
};

// ---------------------------------------------------------------------------
// 4.1  DPU level — representative workload (256 in, 248 out, FP16→FP32).
//       (rem-8 residue class; within AVEPOOL variant limit of 256 channels)
// ---------------------------------------------------------------------------
TEST_F(DPUCostModel_NPU6_CompilerAvepool_Test,
       DPULevel_CompilerWL_AVEPOOL_256to248_FP16toFP32_ShouldReturnValidCost) {
    DPUWorkload wl = make_wl(248u, DataType::FLOAT32);
    std::string errInfo;
    const auto cost = model.DPU(wl, errInfo);

    EXPECT_FALSE(Cycles::isErrorCode(cost))
            << "FAILS at DPU level: representative workload AVEPOOL 256->248 FP16->FP32"
            << "\n  248 % 16 == 8 (unaligned > 16, within AVEPOOL variant limit 256)"
            << "\n  cost=" << cost << " error='" << Cycles::toErrorText(cost) << "'"
            << "\n  errInfo: " << errInfo
            << "\n  Fix needed (149253)";

}

// ---------------------------------------------------------------------------
// 4.2  DPU level — channel sweep over all remainder classes (mod 16) > 16.
//       All values ≤ 256 (AVEPOOL variant limit).
//
//  rem 1:  17, 33, 65, 97, 129
//  rem 2:  18, 34, 50
//  rem 4:  20, 36, 68
//  rem 8:  24, 40, 56, 72, 88, 104, 120, 200, 248
//  rem 12: 28, 44, 60
//  rem 15: 31, 47, 63, 79
//  near boundaries: 17, 31, 33, 63, 65, 127, 129, 255
//  (all ≤ 256 — AVEPOOL variant limit; use Layer/LayersPreSplit for larger counts)
// ---------------------------------------------------------------------------
TEST_F(DPUCostModel_NPU6_CompilerAvepool_Test,
       DPULevel_AVEPOOL_FP16toFP32_Superdense_Ch_Sweep_ShouldReturnValidCost) {
    const std::vector<unsigned int> channels{
            17u, 18u, 20u, 24u, 28u, 31u,
            33u, 34u, 36u, 40u, 44u, 47u,
            50u, 56u, 60u, 63u, 65u, 68u,
            72u, 79u, 88u, 97u, 100u, 104u,
            120u, 127u, 129u, 200u, 248u, 255u,
    };

    for (const auto out_ch : channels) {
        DPUWorkload wl = make_wl(out_ch, DataType::FLOAT32);
        std::string errInfo;
        const auto cost = model.DPU(wl, errInfo);

        EXPECT_FALSE(Cycles::isErrorCode(cost))
                << "FAILS at DPU level: AVEPOOL FP16->FP32 out_ch=" << out_ch
                << " (rem16=" << (out_ch % 16) << ")"
                << "\n  cost=" << cost << " error='" << Cycles::toErrorText(cost) << "'"
                << "\n  Fix needed (149253)";
    }
}

// ---------------------------------------------------------------------------
// 4.3  DPU level — FP16 output dtype (matches AVEPOOL FP16→FP16 variant).
// ---------------------------------------------------------------------------
TEST_F(DPUCostModel_NPU6_CompilerAvepool_Test,
       DPULevel_AVEPOOL_FP16toFP16_Superdense_Ch_Sweep_ShouldReturnValidCost) {
    const std::vector<unsigned int> channels{17u, 24u, 31u, 40u, 65u, 100u, 200u, 248u};

    for (const auto out_ch : channels) {
        DPUWorkload wl = make_wl(out_ch, DataType::FLOAT16);
        std::string errInfo;
        const auto cost = model.DPU(wl, errInfo);

        EXPECT_FALSE(Cycles::isErrorCode(cost))
                << "FAILS at DPU level: AVEPOOL FP16->FP16 out_ch=" << out_ch
                << "\n  cost=" << cost << " error='" << Cycles::toErrorText(cost) << "'"
                << "\n  Fix needed (149253)";
    }
}

// ---------------------------------------------------------------------------
// 4.4  Layer level — representative workload (256→248, NONE tiling strategy).
// ---------------------------------------------------------------------------
// TEST_F(VPULayerCM_NPU6_CompilerAvepool_Test,
//        LayerLevel_CompilerWL_AVEPOOL_256to248_FP16toFP32_ShouldNotReturnError) {
//     DPUWorkload wl = make_wl(248u, DataType::FLOAT32);
//     DPULayer layer(wl);

//     const VPULayerStrategy strategy{1U, 1U, 1U, VPUTilingStrategy::NONE, false, false, prefetch};

//     Logger::clear2ndlog();
//     LayerSplitInfo split_info;
//     CyclesInterfaceType cost{};
//     ASSERT_NO_THROW(cost = theModel.Layer(layer, strategy, split_info))
//             << "Layer() threw for representative workload\n" << Logger::get2ndlog();

//     EXPECT_FALSE(Cycles::isErrorCode(cost))
//             << "FAILS at Layer level: AVEPOOL 256->248 FP16->FP32 (superdense, output_autopad)"
//             << "\n  248 % 16 == 8 (unaligned > 16, within AVEPOOL variant limit 256)"
//             << "\n  cost=" << cost << " error='" << Cycles::toErrorText(cost) << "'"
//             << "\n" << toStringLayerSplitInfo(split_info)
//             << "\n  Fix needed (149253)";
// }

// ---------------------------------------------------------------------------
// 4.5  Layer level — channel sweep for AVEPOOL FP16→FP32 with superdense.
// ---------------------------------------------------------------------------
// TEST_F(VPULayerCM_NPU6_CompilerAvepool_Test,
//        LayerLevel_AVEPOOL_FP16toFP32_Superdense_Ch_Sweep_ShouldNotReturnError) {
//     const std::vector<unsigned int> channels{
//             17u, 20u, 24u, 31u, 33u, 40u, 56u, 65u, 100u, 200u, 248u};

//     const VPULayerStrategy strategy{1U, 1U, 1U, VPUTilingStrategy::NONE, false, false, prefetch};

//     for (const auto out_ch : channels) {
//         DPUWorkload wl = make_wl(out_ch, DataType::FLOAT32);
//         DPULayer layer(wl);

//         Logger::clear2ndlog();
//         LayerSplitInfo split_info;
//         CyclesInterfaceType cost{};
//         ASSERT_NO_THROW(cost = theModel.Layer(layer, strategy, split_info))
//                 << "Layer() threw for AVEPOOL out_ch=" << out_ch << "\n" << Logger::get2ndlog();

//         EXPECT_FALSE(Cycles::isErrorCode(cost))
//                 << "FAILS at Layer level: AVEPOOL FP16->FP32 out_ch=" << out_ch
//                 << " (rem16=" << (out_ch % 16) << ")"
//                 << "\n  cost=" << cost << " error='" << Cycles::toErrorText(cost) << "'"
//                 << "\n  Fix needed (149253)";
//     }
// }

// ---------------------------------------------------------------------------
// 4.6  LayersPreSplit level — representative workload (256→248) as a single tile.
// ---------------------------------------------------------------------------
TEST_F(VPULayerCM_NPU6_CompilerAvepool_Test,
       LayersPreSplit_CompilerWL_AVEPOOL_256to248_FP16toFP32_ShouldNotReturnError) {
    DPUWorkload wl = make_wl(248u, DataType::FLOAT32);
    const std::vector<DPULayer> splitLayers{DPULayer(wl)};

    Logger::clear2ndlog();
    LayerSplitInfo split_info;
    CyclesInterfaceType cost{};
    ASSERT_NO_THROW(cost = theModel.LayersPreSplit(splitLayers, 1U, false, false, prefetch, split_info))
            << "LayersPreSplit() threw for representative workload\n" << Logger::get2ndlog();

    EXPECT_FALSE(Cycles::isErrorCode(cost))
            << "FAILS at LayersPreSplit level: AVEPOOL 256->248 FP16->FP32"
            << "\n  248 % 16 == 8 (unaligned > 16, within AVEPOOL variant limit 256)"
            << "\n  cost=" << cost << " error='" << Cycles::toErrorText(cost) << "'"
            << "\n" << toStringLayerSplitInfo(split_info)
            << "\n  Fix needed (149253)";
}

// ---------------------------------------------------------------------------
// 4.7  LayersPreSplit level — channel sweep for AVEPOOL FP16→FP32, superdense.
// ---------------------------------------------------------------------------
TEST_F(VPULayerCM_NPU6_CompilerAvepool_Test,
       LayersPreSplit_AVEPOOL_FP16toFP32_Superdense_Ch_Sweep_ShouldNotReturnError) {
    const std::vector<unsigned int> channels{/* 17u, 20u,  24u,  28u,  31u, */ 33u, /* 40u,  44u,
                                             56u,
            63u, 65u, 72u, 88u, 97u, 100u, 104u, 129u, 200u, 210u, 226u, 248u, 260u, 500u, 1000u*/};

    for (const auto out_ch : channels) {
        DPUWorkload wl = make_wl(out_ch, DataType::FLOAT32);
        const std::vector<DPULayer> splitLayers{DPULayer(wl)};

        Logger::clear2ndlog();
        LayerSplitInfo split_info;
        CyclesInterfaceType cost{};
        ASSERT_NO_THROW(
                cost = theModel.LayersPreSplit(splitLayers, 1U, false, false, prefetch, split_info))
                << "LayersPreSplit() threw for AVEPOOL out_ch=" << out_ch
                << "\n" << Logger::get2ndlog();

        EXPECT_FALSE(Cycles::isErrorCode(cost))
                << "FAILS at LayersPreSplit level: AVEPOOL FP16->FP32 out_ch=" << out_ch
                << " (rem16=" << (out_ch % 16) << ")"
                << "\n  cost=" << cost << " error='" << Cycles::toErrorText(cost) << "'"
                << "\n  Fix needed (149253)";
    }
}
// Sweep
TEST_F(VPULayerCM_NPU6_CompilerAvepool_Test,
       LayersPreSplit_AVEPOOL_FP16toFP32_Superdense_Ch_FullSweep_ShouldNotReturnError) {
    std::vector<unsigned int> channels;
    for (unsigned int ch = 17; ch <= 512; ++ch) {
        if (ch % 16 != 0) {
			channels.push_back(ch);
		}
	}

    for (const auto out_ch : channels) {
        DPUWorkload wl = make_wl(out_ch, DataType::FLOAT32);
        const std::vector<DPULayer> splitLayers{DPULayer(wl)};

        Logger::clear2ndlog();
        LayerSplitInfo split_info;
        CyclesInterfaceType cost{};
        ASSERT_NO_THROW(cost = theModel.LayersPreSplit(splitLayers, 1U, false, false, prefetch, split_info))
                << "LayersPreSplit() threw for AVEPOOL out_ch=" << out_ch << "\n"
                << Logger::get2ndlog();

        EXPECT_FALSE(Cycles::isErrorCode(cost)) << "FAILS at LayersPreSplit level: AVEPOOL FP16->FP32 out_ch=" << out_ch 
                                                << "in_ch=" << wl.inputs[0].channels()
                                                << " (rem16=" << (out_ch % 16) << ")"
                                                << "\n  cost=" << cost << " error='" << Cycles::toErrorText(cost) << "'"
                                                << "\n  Fix needed (149253)";
    }
}

// ---------------------------------------------------------------------------
// 4.8  Control — LayersPreSplit with aligned output channels (256→256,
//       within AVEPOOL variant limit).  Must continue to pass.
// ---------------------------------------------------------------------------
TEST_F(VPULayerCM_NPU6_CompilerAvepool_Test,
       Control_LayersPreSplit_AVEPOOL_AlignedCh_256_ShouldContinueToPass) {
    DPUWorkload wl{
            VPUDevice::NPU_6_0,
            Operation::AVEPOOL,
            {VPUTensor(1, 1, 256, 1, DataType::FLOAT16, Layout::ZXY)},
            {VPUTensor(1, 1, 256, 1, DataType::FLOAT32, Layout::ZXY)},
            {1, 1},
            {1, 1},
            {0, 0, 0, 0},
            ExecutionMode::CUBOID_16x16,
            ActivationFunction::NONE,
            0.0F,
            0.0F,
            {Swizzling::KEY_0, Swizzling::KEY_0},
            {Swizzling::KEY_0},
            1,
            {0, 0, 0, 0},
            ISIStrategy::CLUSTERING,
            false,
    };
    wl.superdense_memory = true;
    wl.output_autopad    = true;
    const std::vector<DPULayer> splitLayers{DPULayer(wl)};

    Logger::clear2ndlog();
    LayerSplitInfo split_info;
    CyclesInterfaceType cost{};
    ASSERT_NO_THROW(cost = theModel.LayersPreSplit(splitLayers, 1U, false, false, prefetch, split_info));

    EXPECT_FALSE(Cycles::isErrorCode(cost))
            << "Regression: AVEPOOL 256->256 (aligned, within variant limit) should continue to pass"
            << "\n  cost=" << cost << " error='" << Cycles::toErrorText(cost) << "'";
}

// ============================================================================
// SECTION 5 — Full channel sweep (DPU level)
//
// For each operation, every output channel count in the stated range is
// exercised with output_autopad=true.  Input channel rules:
//
//   CONVOLUTION / ELTWISE  : in_ch = ceil(out_ch / 16) * 16  (always ≥ 16)
//   AVEPOOL / DW_CONVOLUTION / MAXPOOL :
//       if out_ch % 16 == 0  → in_ch = out_ch
//       elif out_ch < 16     → in_ch = 16
//       else (out_ch > 16)   → in_ch = ((out_ch / 32) + 1) * 32
//
// All tests are EXPECTED TO FAIL until 149253 is resolved.
// ============================================================================

class DPUCostModel_NPU6_FullChannelSweep_Test : public ::testing::Test {
protected:
    VPUCostModel model{VPUDevice::NPU_6_0, NPU_6_0_MODEL_PATH};

    // Round n up to the next multiple of 16.
    static unsigned int align16(unsigned int n) {
        return VPUNN::eltwise_channel_align(n);
    }

    // Input-channel rule for AVEPOOL / DW_CONVOLUTION / MAXPOOL.
    static unsigned int special_in_ch(unsigned int out_ch) {
        return VPUNN::dw_channel_align(out_ch);
    }

    // CONVOLUTION: 1x1 kernel, small spatial, in_ch aligned to 16.
    DPUWorkload make_conv(unsigned int out_ch) const {
        const unsigned int in_ch = align16(out_ch);
        DPUWorkload wl{
                VPUDevice::NPU_6_0,
                Operation::CONVOLUTION,
                {VPUTensor(1, 1, in_ch, 1, DataType::UINT8)},
                {VPUTensor(1, 1, out_ch, 1, DataType::UINT8)},
                {1, 1},
                {1, 1},
                {0, 0, 0, 0},
                ExecutionMode::CUBOID_16x16,
                ActivationFunction::NONE,
                0.0F,
                0.0F,
                {swz_def, swz_def},
                {swz_def},
                1,
                {0, 0, 0, 0},
                ISIStrategy::CLUSTERING,
                false,
        };
        wl.output_autopad = true;
        return wl;
    }

    // ELTWISE: 1x1 kernel, small spatial, in_ch aligned to 16.
    DPUWorkload make_eltwise(unsigned int out_ch) const {
        const unsigned int in_ch = align16(out_ch);
        DPUWorkload wl{
                VPUDevice::NPU_6_0,
                Operation::ELTWISE,
                {VPUTensor(1, 1, in_ch, 1, DataType::UINT8)},
                {VPUTensor(1, 1, out_ch, 1, DataType::UINT8)},
                {1, 1},
                {1, 1},
                {0, 0, 0, 0},
                ExecutionMode::CUBOID_8x16,
                ActivationFunction::NONE,
                0.0F,
                0.0F,
                {swz_def, swz_def},
                {swz_def},
                1,
                {0, 0, 0, 0},
                ISIStrategy::CLUSTERING,
                false,
        };
        wl.output_autopad    = true;
        wl.superdense_memory = true;
        return wl;
    }

    // AVEPOOL: 1x1 spatial, FP16→FP32, superdense, special in_ch rule.
    DPUWorkload make_avepool(unsigned int out_ch) const {
        const unsigned int in_ch = special_in_ch(out_ch);
        DPUWorkload wl{
                VPUDevice::NPU_6_0,
                Operation::AVEPOOL,
                {VPUTensor(1, 1, in_ch, 1, DataType::FLOAT16, Layout::ZXY)},
                {VPUTensor(1, 1, out_ch, 1, DataType::FLOAT32, Layout::ZXY)},
                {1, 1},
                {1, 1},
                {0, 0, 0, 0},
                ExecutionMode::CUBOID_16x16,
                ActivationFunction::NONE,
                0.0F,
                0.0F,
                {Swizzling::KEY_0, Swizzling::KEY_0},
                {Swizzling::KEY_0},
                1,
                {0, 0, 0, 0},
                ISIStrategy::CLUSTERING,
                false,
        };
        wl.superdense_memory = true;
        wl.output_autopad    = true;
        return wl;
    }

    // DW_CONVOLUTION: 1x1 kernel, small spatial, FP16, special in_ch rule.
    DPUWorkload make_dwconv(unsigned int out_ch) const {
        const unsigned int in_ch = special_in_ch(out_ch);
        DPUWorkload wl{
                VPUDevice::NPU_6_0,
                Operation::DW_CONVOLUTION,
                {VPUTensor(4, 4, in_ch, 1, DataType::FLOAT16, Layout::ZXY)},
                {VPUTensor(4, 4, out_ch, 1, DataType::FLOAT16, Layout::ZXY)},
                {1, 1},
                {1, 1},
                {0, 0, 0, 0},
                ExecutionMode::CUBOID_16x16,
                ActivationFunction::NONE,
                0.0F,
                0.0F,
                {swz_def, swz_def},
                {swz_def},
                1,
                {0, 0, 0, 0},
                ISIStrategy::CLUSTERING,
                false,
        };
        wl.output_autopad = true;
        return wl;
    }

    // MAXPOOL: 1x1 kernel, small spatial, special in_ch rule.
    DPUWorkload make_maxpool(unsigned int out_ch) const {
        const unsigned int in_ch = special_in_ch(out_ch);
        DPUWorkload wl{
                VPUDevice::NPU_6_0,
                Operation::MAXPOOL,
                {VPUTensor(4, 4, in_ch, 1, DataType::UINT8)},
                {VPUTensor(4, 4, out_ch, 1, DataType::UINT8)},
                {1, 1},
                {1, 1},
                {0, 0, 0, 0},
                ExecutionMode::CUBOID_16x16,
                ActivationFunction::NONE,
                0.0F,
                0.0F,
                {swz_def, swz_def},
                {swz_def},
                1,
                {0, 0, 0, 0},
                ISIStrategy::CLUSTERING,
                false,
        };
        wl.output_autopad = true;
        return wl;
    }
};

// ---------------------------------------------------------------------------
// 5.1  CONVOLUTION — every output channel from 1 to 1024.
//      in_ch is always ceil(out_ch/16)*16 (aligned to 16).
//      A CSV with all queries and their results is written alongside the test if ENABLE_VPUNN_DATA_SERIALIZATION=TRUE.
// ---------------------------------------------------------------------------
TEST_F(DPUCostModel_NPU6_FullChannelSweep_Test,
       FullSweep_CONV_OutputAutopad_Channels1to1024_ShouldReturnValidCost) {
    // Should only generate CSV when ENABLE_VPUNN_DATA_SERIALIZATION=TRUE
    Serializer<FileFormat::CSV> csv{};
    {
        auto fields = SerializableDPU::_get_member_names();
        fields.emplace_back("cycles");
        fields.emplace_back("error_info");
        csv.initialize("npu6_autopad_sweep_conv", FileMode::READ_WRITE, fields);
    }
    SerializableField<CyclesInterfaceType> cycles_field{"cycles", 0u};
    SerializableField<std::string>         err_field{"error_info", ""};

    for (unsigned int out_ch = 1u; out_ch <= 1024u; ++out_ch) {
        DPUWorkload wl = make_conv(out_ch);
        std::string errInfo;
        const auto cost = model.DPU(wl, errInfo);

        cycles_field.value = cost;
        err_field.value    = errInfo;
        csv.serialize(SerializableDPU{wl}, cycles_field, err_field);
        csv.end();

        EXPECT_FALSE(Cycles::isErrorCode(cost))
                << "CONV out_ch=" << out_ch << " in_ch=" << align16(out_ch)
                << " output_autopad=true returned error"
                << "\n  cost=" << cost << " error='" << Cycles::toErrorText(cost) << "'"
                << " errInfo=" << errInfo
                << "\n  Fix needed (149253)";
    }
}

// ---------------------------------------------------------------------------
// 5.2  ELTWISE — every output channel from 1 to 1024.
//      in_ch is always ceil(out_ch/16)*16 (aligned to 16).
//      A CSV with all queries and their results is written alongside the test if ENABLE_VPUNN_DATA_SERIALIZATION=TRUE.
// ---------------------------------------------------------------------------
TEST_F(DPUCostModel_NPU6_FullChannelSweep_Test,
       FullSweep_ELTWISE_OutputAutopad_Channels1to1024_ShouldReturnValidCost) {
    // Should only generate CSV when ENABLE_VPUNN_DATA_SERIALIZATION=TRUE
    Serializer<FileFormat::CSV> csv{};
    {
        auto fields = SerializableDPU::_get_member_names();
        fields.emplace_back("cycles");
        fields.emplace_back("error_info");
        csv.initialize("npu6_autopad_sweep_eltwise", FileMode::READ_WRITE, fields);
    }
    SerializableField<CyclesInterfaceType> cycles_field{"cycles", 0u};
    SerializableField<std::string>         err_field{"error_info", ""};

    for (unsigned int out_ch = 1u; out_ch <= 1024u; ++out_ch) {
        DPUWorkload wl = make_eltwise(out_ch);
        std::string errInfo;
        const auto cost = model.DPU(wl, errInfo);

        cycles_field.value = cost;
        err_field.value    = errInfo;
        csv.serialize(SerializableDPU{wl}, cycles_field, err_field);
        csv.end();

        EXPECT_FALSE(Cycles::isErrorCode(cost))
                << "ELTWISE out_ch=" << out_ch << " in_ch=" << align16(out_ch)
                << " output_autopad=true returned error"
                << "\n  cost=" << cost << " error='" << Cycles::toErrorText(cost) << "'"
                << " errInfo=" << errInfo
                << "\n  Fix needed (149253)";
    }
}

// ---------------------------------------------------------------------------
// 5.3  AVEPOOL — every output channel from 1 to 256.
//      in_ch follows the special rule (see fixture header).
// ---------------------------------------------------------------------------
TEST_F(DPUCostModel_NPU6_FullChannelSweep_Test,
       FullSweep_AVEPOOL_OutputAutopad_Channels1to256_ShouldReturnValidCost) {
    // Should only generate CSV when ENABLE_VPUNN_DATA_SERIALIZATION=TRUE
    Serializer<FileFormat::CSV> csv{};
    {
        auto fields = SerializableDPU::_get_member_names();
        fields.emplace_back("cycles");
        fields.emplace_back("error_info");
        csv.initialize("npu6_autopad_sweep_avepool", FileMode::READ_WRITE, fields);
    }
    SerializableField<CyclesInterfaceType> cycles_field{"cycles", 0u};
    SerializableField<std::string>         err_field{"error_info", ""};

    for (unsigned int out_ch = 1u; out_ch <= 256u; ++out_ch) {
        DPUWorkload wl = make_avepool(out_ch);
        std::string errInfo;
        const auto cost = model.DPU(wl, errInfo);

        cycles_field.value = cost;
        err_field.value    = errInfo;
        csv.serialize(SerializableDPU{wl}, cycles_field, err_field);
        csv.end();

        EXPECT_FALSE(Cycles::isErrorCode(cost))
                << "AVEPOOL out_ch=" << out_ch << " in_ch=" << special_in_ch(out_ch)
                << " output_autopad=true returned error"
                << "\n  cost=" << cost << " error='" << Cycles::toErrorText(cost) << "'"
                << " errInfo=" << errInfo
                << "\n  Fix needed (149253)";
    }
}

// ---------------------------------------------------------------------------
// 5.4  DW_CONVOLUTION — every output channel from 1 to 256.
//      in_ch follows the special rule (see fixture header).
// ---------------------------------------------------------------------------
TEST_F(DPUCostModel_NPU6_FullChannelSweep_Test,
       FullSweep_DWConv_OutputAutopad_Channels1to256_ShouldReturnValidCost) {
    // Should only generate CSV when ENABLE_VPUNN_DATA_SERIALIZATION=TRUE
    Serializer<FileFormat::CSV> csv{};
    {
        auto fields = SerializableDPU::_get_member_names();
        fields.emplace_back("cycles");
        fields.emplace_back("error_info");
        csv.initialize("npu6_autopad_sweep_dwconv", FileMode::READ_WRITE, fields);
    }
    SerializableField<CyclesInterfaceType> cycles_field{"cycles", 0u};
    SerializableField<std::string>         err_field{"error_info", ""};

    for (unsigned int out_ch = 1u; out_ch <= 256u; ++out_ch) {
        DPUWorkload wl = make_dwconv(out_ch);
        std::string errInfo;
        const auto cost = model.DPU(wl, errInfo);

        cycles_field.value = cost;
        err_field.value    = errInfo;
        csv.serialize(SerializableDPU{wl}, cycles_field, err_field);
        csv.end();

        EXPECT_FALSE(Cycles::isErrorCode(cost))
                << "DW_CONV out_ch=" << out_ch << " in_ch=" << special_in_ch(out_ch)
                << " output_autopad=true returned error"
                << "\n  cost=" << cost << " error='" << Cycles::toErrorText(cost) << "'"
                << " errInfo=" << errInfo
                << "\n  Fix needed (149253)";
    }
}

// ---------------------------------------------------------------------------
// 5.5  MAXPOOL — every output channel from 1 to 256.
//      in_ch follows the special rule (see fixture header).
// ---------------------------------------------------------------------------
TEST_F(DPUCostModel_NPU6_FullChannelSweep_Test,
       FullSweep_MAXPOOL_OutputAutopad_Channels1to256_ShouldReturnValidCost) {
    // Should only generate CSV when ENABLE_VPUNN_DATA_SERIALIZATION=TRUE
    Serializer<FileFormat::CSV> csv{};
    {
        auto fields = SerializableDPU::_get_member_names();
        fields.emplace_back("cycles");
        fields.emplace_back("error_info");
        csv.initialize("npu6_autopad_sweep_maxpool", FileMode::READ_WRITE, fields);
    }
    SerializableField<CyclesInterfaceType> cycles_field{"cycles", 0u};
    SerializableField<std::string>         err_field{"error_info", ""};

    for (unsigned int out_ch = 1u; out_ch <= 256u; ++out_ch) {
        DPUWorkload wl = make_maxpool(out_ch);
        std::string errInfo;
        const auto cost = model.DPU(wl, errInfo);

        cycles_field.value = cost;
        err_field.value    = errInfo;
        csv.serialize(SerializableDPU{wl}, cycles_field, err_field);
        csv.end();

        EXPECT_FALSE(Cycles::isErrorCode(cost))
                << "MAXPOOL out_ch=" << out_ch << " in_ch=" << special_in_ch(out_ch)
                << " output_autopad=true returned error"
                << "\n  cost=" << cost << " error='" << Cycles::toErrorText(cost) << "'"
                << " errInfo=" << errInfo
                << "\n  Fix needed (149253)";
    }
}

}  // namespace VPUNN_unit_tests
