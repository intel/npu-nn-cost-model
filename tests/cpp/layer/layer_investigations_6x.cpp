// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

// Investigation tests for NPU6x (NPU6010) layer cost model issues.
// These tests reproduce bugs encountered during compiler smoke tests and serve as regression guards.

#include "vpu_layer_cost_model.h"

#include <gtest/gtest.h>
#include <numeric>
#include <sstream>
#include "common/common_helpers.h"

#include "layer.h"
#include "vpu/shave/layers.h"

namespace VPUNN_unit_tests {
using namespace VPUNN;

/// @brief Test fixture for NPU6x layer cost model investigation tests.
///
/// Loads the NPU_6_0 NN model so cost queries
/// exercise the neural-network path, matching what the compiler does at runtime.
class VPULayerCM_InvestigationTestNPU6x : public VPULayerCostModelTest {
protected:
    void SetUp() override {
        VPULayerCostModelTest::SetUp();
        // Register the NPU_6_0 model so getModel(NPU_6_0) returns the NN-backed instance
        // instead of the empty fallback.  This matches the compiler's runtime behaviour.
        layer_models.addModel(VPUDevice::NPU_6_0, &dma_model_invalid, NPU_6_0_MODEL_PATH);
    }

    const bool prefetch{true};
    VPULayerCostModel& theModel = layer_models.getModel(VPUDevice::NPU_6_0);
};

// ---------------------------------------------------------------------------
// [E-202196] Relu activation with 8 channels (< 16) on NPU6010 produces
// "Cost is Invalid" in the compiler's split_nce_ops_onto_workloads pass.
//
// Reproduces the smoke test:
//   smoke_Act_tiling_Hw/ActivationLayerTest_HW_FP.NPU6010/
//     IS=([])_TS={(1.8.80.1280)}_TS=()_Relu_constants_value=(1)_netPRC=f16_trgDev=NPU
//
// The compiler maps standalone Relu to an NCE operation (typically DW_CONVOLUTION
// or MAXPOOL with a 1x1 kernel) and calls LayersPreSplit to determine the
// intra-tile workload split.  For 8 output channels (not a multiple of 16) the
// ZEffTiling path's candidate block sizes start at 16.  If the cost model returns
// an error for a candidate 16-channel test workload, the block list is empty and
// ZEffTiling produces no valid split, which propagates as ERROR_TILE_SPLIT_EXCEPTION
// and ultimately triggers the "Cost is Invalid" assertion in the compiler.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// DW_CONVOLUTION variant
// The original layer (1 x 8 x 80 x 1280) is pre-split along H into 4 equal
// tiles of H=20 before being passed to LayersPreSplit.
// ---------------------------------------------------------------------------
TEST_F(VPULayerCM_InvestigationTestNPU6x,
       E202196_Relu_8ch_80x1280_DWConv_PreSplit_ShouldNotReturnError) {
    // Original shape: 1 x 8 x 80 x 1280 (N x C x H x W).
    // H-split into 4 tiles: each tile H = 20.
    // In VPUNN VPUTensor layout the order is (W, H, C, N).
    const VPUDevice device{VPUDevice::NPU_6_0};

    // Build 4 H-split tiles (each H=20, 4*20=80 == original H).
    std::vector<DPULayer> splitLayers;
    for (int i = 0; i < 4; ++i) {
        DPUWorkload wl{
                device,
                Operation::DW_CONVOLUTION,
                {VPUTensor(1280, 20, 8, 1, DataType::FLOAT16, Layout::ZXY)},  // input tile  (W, H, C, N)
                {VPUTensor(1280, 20, 8, 1, DataType::FLOAT16, Layout::ZXY)},  // output tile
                {1, 1},                                                        // kernels  (KX, KY)
                {1, 1},                                                        // strides  (SX, SY)
                {0, 0, 0, 0},                                                  // padding
                ExecutionMode::CUBOID_16x16,
                ActivationFunction::RELU,
                0.0F, 0.0F,
                {Swizzling::KEY_0, Swizzling::KEY_0},
                {Swizzling::KEY_0},
                1,
                {0, 0, 0, 0},
                ISIStrategy::CLUSTERING,
                false,
        };
        // The compiler sets output_autopad=true when channels % 16 != 0,
        // and input_autopad=true when input channels < 16.
        wl.output_autopad = true;
        wl.input_autopad  = true;
        splitLayers.emplace_back(wl);
    }

    Logger::clear2ndlog();
    CyclesInterfaceType cost{};
    LayerSplitInfo detailed_split{};

    ASSERT_NO_THROW(cost = theModel.LayersPreSplit(splitLayers, 1U, false, false, prefetch, detailed_split))
            << "\n2ndLog:\n" << Logger::get2ndlog();

    EXPECT_FALSE(Cycles::isErrorCode(cost))
            << "LayersPreSplit returned error for 8-channel Relu (DW_CONV 1x1, 4 H-tiles) on NPU6010.\n"
            << "Error: " << Cycles::toErrorText(cost) << "\n"
            << "This reproduces E-202196: ZEffTiling generates no valid split for channels < 16.\n"
            << "\n2ndLog:\n" << Logger::get2ndlog();

    // Verify that channels are preserved through the split: each tile must have
    // the same input and output channel count as the original input layers.
    ASSERT_EQ(detailed_split.size(), 4U) << "Expected 4 tile entries in split info.";
    for (unsigned int j = 0; j < detailed_split.size(); ++j) {
        const auto& tile = detailed_split[j].inter_tile_split_layer;
        EXPECT_EQ(tile.inputs[0].channels(), 8U)
                << "Tile " << j << ": input channels changed from 8 to " << tile.inputs[0].channels();
        EXPECT_EQ(tile.outputs[0].channels(), 8U)
                << "Tile " << j << ": output channels changed from 8 to " << tile.outputs[0].channels();
    }
}

// ---------------------------------------------------------------------------
// MAXPOOL variant  (Relu can also be lowered to a 1x1 MaxPool on NPU)
// H-split into 4 tiles of H=20 before passing to LayersPreSplit.
// ---------------------------------------------------------------------------
TEST_F(VPULayerCM_InvestigationTestNPU6x,
       E202196_Relu_8ch_80x1280_Maxpool_PreSplit_ShouldNotReturnError) {
    const VPUDevice device{VPUDevice::NPU_6_0};

    std::vector<DPULayer> splitLayers;
    for (int i = 0; i < 4; ++i) {
        DPUWorkload wl{
                device,
                Operation::MAXPOOL,
                {VPUTensor(1280, 20, 8, 1, DataType::FLOAT16, Layout::ZXY)},  // input tile
                {VPUTensor(1280, 20, 8, 1, DataType::FLOAT16, Layout::ZXY)},  // output tile
                {1, 1},
                {1, 1},
                {0, 0, 0, 0},
                ExecutionMode::CUBOID_16x16,
                ActivationFunction::RELU,
                0.0F, 0.0F,
                {Swizzling::KEY_0, Swizzling::KEY_0},
                {Swizzling::KEY_0},
                1,
                {0, 0, 0, 0},
                ISIStrategy::CLUSTERING,
                false,
        };
        wl.output_autopad = true;
        wl.input_autopad  = true;
        splitLayers.emplace_back(wl);
    }

    Logger::clear2ndlog();
    CyclesInterfaceType cost{};
    LayerSplitInfo detailed_split{};

    ASSERT_NO_THROW(cost = theModel.LayersPreSplit(splitLayers, 1U, false, false, prefetch, detailed_split))
            << "\n2ndLog:\n" << Logger::get2ndlog();

    EXPECT_FALSE(Cycles::isErrorCode(cost))
            << "LayersPreSplit returned error for 8-channel Relu (MaxPool 1x1, 4 H-tiles) on NPU6010.\n"
            << "Error: " << Cycles::toErrorText(cost) << "\n"
            << "\n2ndLog:\n" << Logger::get2ndlog();

    // Verify that channels are preserved through the split.
    ASSERT_EQ(detailed_split.size(), 4U) << "Expected 4 tile entries in split info.";
    for (unsigned int j = 0; j < detailed_split.size(); ++j) {
        const auto& tile = detailed_split[j].inter_tile_split_layer;
        EXPECT_EQ(tile.inputs[0].channels(), 8U)
                << "Tile " << j << ": input channels changed from 8 to " << tile.inputs[0].channels();
        EXPECT_EQ(tile.outputs[0].channels(), 8U)
                << "Tile " << j << ": output channels changed from 8 to " << tile.outputs[0].channels();
    }
}

// ---------------------------------------------------------------------------
// Passing control case: same Relu shape but with 320 channels (multiple of 16).
// Original layer: 1 x 320 x 4 x 1280. H-split into 4 tiles of H=1 each.
// Mirrors the passing smoke test and verifies the test infrastructure.
// ---------------------------------------------------------------------------
TEST_F(VPULayerCM_InvestigationTestNPU6x,
       E202196_Relu_320ch_1x1280_DWConv_PreSplit_ControlCase) {
    const VPUDevice device{VPUDevice::NPU_6_0};

    // 320 % 16 == 0; no autopad needed.
    // Original layer H=4 split into 4 tiles of H=1.
    std::vector<DPULayer> splitLayers;
    for (int i = 0; i < 4; ++i) {
        DPUWorkload wl{
                device,
                Operation::DW_CONVOLUTION,
                {VPUTensor(1280, 1, 320, 1, DataType::FLOAT16, Layout::ZXY)},  // input tile
                {VPUTensor(1280, 1, 320, 1, DataType::FLOAT16, Layout::ZXY)},  // output tile
                {1, 1},
                {1, 1},
                {0, 0, 0, 0},
                ExecutionMode::CUBOID_16x16,
                ActivationFunction::RELU,
                0.0F, 0.0F,
                {Swizzling::KEY_0, Swizzling::KEY_0},
                {Swizzling::KEY_0},
                1,
                {0, 0, 0, 0},
                ISIStrategy::CLUSTERING,
                false,
        };
        splitLayers.emplace_back(wl);
    }

    Logger::clear2ndlog();
    CyclesInterfaceType cost{};
    LayerSplitInfo detailed_split{};

    ASSERT_NO_THROW(cost = theModel.LayersPreSplit(splitLayers, 1U, false, false, prefetch, detailed_split))
            << "\n2ndLog:\n" << Logger::get2ndlog();

    EXPECT_FALSE(Cycles::isErrorCode(cost))
            << "LayersPreSplit returned error for 320-channel Relu (DW_CONV 1x1, 4 H-tiles) on NPU6010.\n"
            << "Error: " << Cycles::toErrorText(cost) << "\n"
            << "\n2ndLog:\n" << Logger::get2ndlog();

    // Verify that channels are preserved through the split.
    ASSERT_EQ(detailed_split.size(), 4U) << "Expected 4 tile entries in split info.";
    for (unsigned int j = 0; j < detailed_split.size(); ++j) {
        const auto& tile = detailed_split[j].inter_tile_split_layer;
        EXPECT_EQ(tile.inputs[0].channels(), 320U)
                << "Tile " << j << ": input channels changed from 320 to " << tile.inputs[0].channels();
        EXPECT_EQ(tile.outputs[0].channels(), 320U)
                << "Tile " << j << ": output channels changed from 320 to " << tile.outputs[0].channels();
    }
}

// ---------------------------------------------------------------------------
// Autopad via LayersPreSplit() — DW_CONVOLUTION, 8 channels
// Original layer H=80 pre-split into 4 H-tiles of H=20 each.
// Both input and output have 8 channels (< 16), so both autopad flags are true.
// ---------------------------------------------------------------------------
TEST_F(VPULayerCM_InvestigationTestNPU6x,
       E202196_DWConv_8ch_AutopadVariants_PreSplit_AllNonErrorAndEqual) {
    const VPUDevice device{VPUDevice::NPU_6_0};

    std::vector<DPULayer> splitLayers;
    for (int i = 0; i < 4; ++i) {
        DPUWorkload wl{
                device,
                Operation::DW_CONVOLUTION,
                {VPUTensor(1280, 20, 8, 1, DataType::FLOAT16, Layout::ZXY)},
                {VPUTensor(1280, 20, 8, 1, DataType::FLOAT16, Layout::ZXY)},
                {1, 1}, {1, 1}, {0, 0, 0, 0},
                ExecutionMode::CUBOID_16x16, ActivationFunction::RELU,
                0.0F, 0.0F,
                {Swizzling::KEY_0, Swizzling::KEY_0}, {Swizzling::KEY_0},
                1, {0, 0, 0, 0}, ISIStrategy::CLUSTERING, false,
        };
        wl.input_autopad  = true;  // input channels 8 < 16
        wl.output_autopad = true;  // output channels 8 < 16
        splitLayers.emplace_back(wl);
    }

    Logger::clear2ndlog();
    CyclesInterfaceType cost{};
    LayerSplitInfo detailed_split{};

    ASSERT_NO_THROW(cost = theModel.LayersPreSplit(splitLayers, 1U, false, false, prefetch, detailed_split))
            << "\n2ndLog:\n" << Logger::get2ndlog();

    EXPECT_FALSE(Cycles::isErrorCode(cost))
            << "Error for DW_CONV 8ch (4 H-tiles, autopad): "
            << Cycles::toErrorText(cost) << "\n2ndLog:\n" << Logger::get2ndlog();

    // Verify that channels are preserved through the split.
    ASSERT_EQ(detailed_split.size(), 4U) << "Expected 4 tile entries in split info.";
    for (unsigned int j = 0; j < detailed_split.size(); ++j) {
        const auto& tile = detailed_split[j].inter_tile_split_layer;
        EXPECT_EQ(tile.inputs[0].channels(), 8U)
                << "Tile " << j << ": input channels changed from 8 to " << tile.inputs[0].channels();
        EXPECT_EQ(tile.outputs[0].channels(), 8U)
                << "Tile " << j << ": output channels changed from 8 to " << tile.outputs[0].channels();
    }
}

// ---------------------------------------------------------------------------
// Autopad via LayersPreSplit() — MAXPOOL, 8 channels
// Original layer H=80 pre-split into 4 H-tiles of H=20 each.
// Both input and output have 8 channels (< 16), so both autopad flags are true.
// ---------------------------------------------------------------------------
TEST_F(VPULayerCM_InvestigationTestNPU6x,
       E202196_Maxpool_8ch_AutopadVariants_PreSplit_AllNonErrorAndEqual) {
    const VPUDevice device{VPUDevice::NPU_6_0};

    std::vector<DPULayer> splitLayers;
    for (int i = 0; i < 4; ++i) {
        DPUWorkload wl{
                device,
                Operation::MAXPOOL,
                {VPUTensor(1280, 20, 8, 1, DataType::FLOAT16, Layout::ZXY)},
                {VPUTensor(1280, 20, 8, 1, DataType::FLOAT16, Layout::ZXY)},
                {1, 1}, {1, 1}, {0, 0, 0, 0},
                ExecutionMode::CUBOID_16x16, ActivationFunction::RELU,
                0.0F, 0.0F,
                {Swizzling::KEY_0, Swizzling::KEY_0}, {Swizzling::KEY_0},
                1, {0, 0, 0, 0}, ISIStrategy::CLUSTERING, false,
        };
        wl.input_autopad  = true;  // input channels 8 < 16
        wl.output_autopad = true;  // output channels 8 < 16
        splitLayers.emplace_back(wl);
    }

    Logger::clear2ndlog();
    CyclesInterfaceType cost{};
    LayerSplitInfo detailed_split{};

    ASSERT_NO_THROW(cost = theModel.LayersPreSplit(splitLayers, 1U, false, false, prefetch, detailed_split))
            << "\n2ndLog:\n" << Logger::get2ndlog();

    EXPECT_FALSE(Cycles::isErrorCode(cost))
            << "Error for MAXPOOL 8ch (4 H-tiles, autopad): "
            << Cycles::toErrorText(cost) << "\n2ndLog:\n" << Logger::get2ndlog();

    // Verify that channels are preserved through the split.
    ASSERT_EQ(detailed_split.size(), 4U) << "Expected 4 tile entries in split info.";
    for (unsigned int j = 0; j < detailed_split.size(); ++j) {
        const auto& tile = detailed_split[j].inter_tile_split_layer;
        EXPECT_EQ(tile.inputs[0].channels(), 8U)
                << "Tile " << j << ": input channels changed from 8 to " << tile.inputs[0].channels();
        EXPECT_EQ(tile.outputs[0].channels(), 8U)
                << "Tile " << j << ": output channels changed from 8 to " << tile.outputs[0].channels();
    }
}

// ---------------------------------------------------------------------------
// Autopad via LayersPreSplit() — ELTWISE, 8 channels
// Original layer H=80 pre-split into 4 H-tiles of H=20 each.
// Both input and output have 8 channels (< 16), so both autopad flags are true.
// ---------------------------------------------------------------------------
TEST_F(VPULayerCM_InvestigationTestNPU6x,
       E202196_Eltwise_8ch_80x1280_AutopadVariants_PreSplit_AllNonErrorAndEqual) {
    const VPUDevice device{VPUDevice::NPU_6_0};

    std::vector<DPULayer> splitLayers;
    for (int i = 0; i < 4; ++i) {
        DPUWorkload wl{
                device,
                Operation::ELTWISE,
                {VPUTensor(1280, 20, 8, 1, DataType::FLOAT16, Layout::ZXY)},
                {VPUTensor(1280, 20, 8, 1, DataType::FLOAT16, Layout::ZXY)},
                {1, 1}, {1, 1}, {0, 0, 0, 0},
                ExecutionMode::CUBOID_16x16, ActivationFunction::RELU,
                0.0F, 0.0F,
                {Swizzling::KEY_0, Swizzling::KEY_0}, {Swizzling::KEY_0},
                1, {0, 0, 0, 0}, ISIStrategy::CLUSTERING, false,
        };
        wl.input_autopad  = true;  // input channels 8 < 16
        wl.output_autopad = true;  // output channels 8 < 16
        splitLayers.emplace_back(wl);
    }

    Logger::clear2ndlog();
    CyclesInterfaceType cost{};
    LayerSplitInfo detailed_split{};

    ASSERT_NO_THROW(cost = theModel.LayersPreSplit(splitLayers, 1U, false, false, prefetch, detailed_split))
            << "\n2ndLog:\n" << Logger::get2ndlog();

    EXPECT_FALSE(Cycles::isErrorCode(cost))
            << "Error for ELTWISE 8ch (4 H-tiles, autopad): "
            << Cycles::toErrorText(cost) << "\n2ndLog:\n" << Logger::get2ndlog();

    // Verify that channels are preserved through the split.
    ASSERT_EQ(detailed_split.size(), 4U) << "Expected 4 tile entries in split info.";
    for (unsigned int j = 0; j < detailed_split.size(); ++j) {
        const auto& tile = detailed_split[j].inter_tile_split_layer;
        EXPECT_EQ(tile.inputs[0].channels(), 8U)
                << "Tile " << j << ": input channels changed from 8 to " << tile.inputs[0].channels();
        EXPECT_EQ(tile.outputs[0].channels(), 8U)
                << "Tile " << j << ": output channels changed from 8 to " << tile.outputs[0].channels();
    }
}

// ---------------------------------------------------------------------------
// Autopad via LayersPreSplit() — CONVOLUTION, 16 in / 8 out channels
// Original layer H=80 pre-split into 4 H-tiles of H=20 each.
// Input channels = 16 (aligned, no autopad); output channels = 8 (< 16, autopad).
// ---------------------------------------------------------------------------
TEST_F(VPULayerCM_InvestigationTestNPU6x,
       E202196_Conv_8outCh_80x1280_AutopadVariants_PreSplit_AllNonErrorAndEqual) {
    const VPUDevice device{VPUDevice::NPU_6_0};

    std::vector<DPULayer> splitLayers;
    for (int i = 0; i < 4; ++i) {
        DPUWorkload wl{
                device,
                Operation::CONVOLUTION,
                {VPUTensor(1280, 20, 16, 1, DataType::FLOAT16, Layout::ZXY)},  // 16 input ch (aligned)
                {VPUTensor(1280, 20, 8,  1, DataType::FLOAT16, Layout::ZXY)},  // 8 output ch
                {1, 1}, {1, 1}, {0, 0, 0, 0},
                ExecutionMode::CUBOID_16x16, ActivationFunction::RELU,
                0.0F, 0.0F,
                {Swizzling::KEY_0, Swizzling::KEY_0}, {Swizzling::KEY_0},
                1, {0, 0, 0, 0}, ISIStrategy::CLUSTERING, false,
        };
        wl.input_autopad  = false;  // input channels 16, not < 16
        wl.output_autopad = true;   // output channels 8 < 16
        splitLayers.emplace_back(wl);
    }

    Logger::clear2ndlog();
    CyclesInterfaceType cost{};
    LayerSplitInfo detailed_split{};

    ASSERT_NO_THROW(cost = theModel.LayersPreSplit(splitLayers, 1U, false, false, prefetch, detailed_split))
            << "\n2ndLog:\n" << Logger::get2ndlog();

    EXPECT_FALSE(Cycles::isErrorCode(cost))
            << "Error for CONV 16in/8out ch (4 H-tiles, output_autopad): "
            << Cycles::toErrorText(cost) << "\n2ndLog:\n" << Logger::get2ndlog();

    // Verify that input (16) and output (8) channels are preserved through the split.
    ASSERT_EQ(detailed_split.size(), 4U) << "Expected 4 tile entries in split info.";
    for (unsigned int j = 0; j < detailed_split.size(); ++j) {
        const auto& tile = detailed_split[j].inter_tile_split_layer;
        EXPECT_EQ(tile.inputs[0].channels(), 16U)
                << "Tile " << j << ": input channels changed from 16 to " << tile.inputs[0].channels();
        EXPECT_EQ(tile.outputs[0].channels(), 8U)
                << "Tile " << j << ": output channels changed from 8 to " << tile.outputs[0].channels();
    }
}

// ---------------------------------------------------------------------------
// Convolution 3-channel 512x512, 3x3 kernel, TANH PPE, H-OVERLAPPED split
// across 4 clusters — compiler log reference:
//   "Get invalid cost for op %170 = VPU.NCE.Convolution ... 1x3x512x512"
//
// The compiler maps the layer to 4 OVERLAPPED compute tiles (128 H rows each).
// Due to halo regions the memory tiles have slightly different H extents and
// different per-edge padding depending on their position:
//   tile 0 (top)    : input H=129, padding TBLR {1,0,1,1}
//   tile 1 (mid)    : input H=130, padding TBLR {0,0,1,1}
//   tile 2 (mid)    : input H=130, padding TBLR {0,0,1,1}
//   tile 3 (bottom) : input H=129, padding TBLR {0,1,1,1}
//
// Input and output channel count is 3 (< 16), so both input_autopad and
// output_autopad are set.  superdense_memory is set to match the compiler path.
// ---------------------------------------------------------------------------
TEST_F(VPULayerCM_InvestigationTestNPU6x,
       Conv_3ch_512x512_3x3_HOverlap4Tiles_PreSplit_ShouldNotReturnError) {
    // Original shape: 1 x 3 x 512 x 512 (N x C x H x W).
    // Pre-split along H into 4 OVERLAPPED tiles; each computes 128 output rows.
    // Padding follows the original {T=1, B=1, L=1, R=1} distributed across tiles.
    // In VPUNN VPUTensor layout the order is (W, H, C, N).
    const VPUDevice device{VPUDevice::NPU_6_0};

    // Per-tile: { input_H, pad_top, pad_bottom, pad_left, pad_right }
    struct TileSpec {
        unsigned int inputH;
        unsigned int padT, padB, padL, padR;
    };
    const std::array<TileSpec, 4> specs{{
            {129, 1, 0, 1, 1},  // tile 0: top cluster   — original top padding retained
            {130, 0, 0, 1, 1},  // tile 1: inner cluster — halo from both neighbours
            {130, 0, 0, 1, 1},  // tile 2: inner cluster — halo from both neighbours
            {129, 0, 1, 1, 1},  // tile 3: bottom cluster — original bottom padding retained
    }};

    std::vector<DPULayer> splitLayers;
    for (const auto& s : specs) {
        DPUWorkload wl{
                device,
                Operation::CONVOLUTION,
                {VPUTensor(512, s.inputH, 3, 1, DataType::FLOAT16, Layout::ZXY)},  // input tile  (W,H,C,N)
                {VPUTensor(512, 128,      3, 1, DataType::FLOAT16, Layout::ZXY)},  // output tile (W,H,C,N)
                {3, 3},                                                              // kernel  (KX, KY)
                {1, 1},                                                              // strides (SX, SY)
                {s.padT, s.padB, s.padL, s.padR},                                   // padding {T,B,L,R}
                ExecutionMode::CUBOID_16x16,
                ActivationFunction::NONE,
                0.0F, 0.0F,
                {Swizzling::KEY_0, Swizzling::KEY_0},
                {Swizzling::KEY_0},
                1,
                {0, 0, 0, 0},
                ISIStrategy::CLUSTERING,
                false,
        };
        // Channels = 3 < 16: both autopad flags are required.
        wl.input_autopad     = true;
        wl.output_autopad    = true;
        wl.superdense_memory = true;
        splitLayers.emplace_back(wl);
    }

    Logger::clear2ndlog();
    CyclesInterfaceType cost{};
    LayerSplitInfo detailed_split{};

    ASSERT_NO_THROW(cost = theModel.LayersPreSplit(splitLayers, 1U, false, false, prefetch, detailed_split))
            << "\n2ndLog:\n" << Logger::get2ndlog();

    EXPECT_FALSE(Cycles::isErrorCode(cost))
            << "LayersPreSplit returned error for 3-channel Convolution "
               "(3x3 kernel, 4 H-overlap tiles) on NPU6010.\n"
            << "Error: " << Cycles::toErrorText(cost) << "\n"
            << "Reproduces: 'Get invalid cost for op VPU.NCE.Convolution 1x3x512x512'.\n"
            << "\n2ndLog:\n" << Logger::get2ndlog();

    ASSERT_EQ(detailed_split.size(), 4U) << "Expected 4 tile entries in split info.";
    for (unsigned int j = 0; j < detailed_split.size(); ++j) {
        const auto& tile = detailed_split[j].inter_tile_split_layer;
        EXPECT_EQ(tile.inputs[0].channels(), 3U)
                << "Tile " << j << ": input channels changed from 3 to " << tile.inputs[0].channels();
        EXPECT_EQ(tile.outputs[0].channels(), 3U)
                << "Tile " << j << ": output channels changed from 3 to " << tile.outputs[0].channels();
    }
}

// ---------------------------------------------------------------------------
// AVEPOOL 1x1 + LRELU PPE, 16 input channels -> 8 output channels, OVERLAPPED
// split into 3 H-tiles (27, 27, 26), matching compiler warning repro.
// ---------------------------------------------------------------------------
TEST_F(VPULayerCM_InvestigationTestNPU6x,
       E202196_Avepool_8outCh_80x1280_Overlapped3Tiles_PreSplit_ShouldNotReturnError) {
    const VPUDevice device{VPUDevice::NPU_6_0};

    struct TileSpec {
        unsigned int h;
    };
    const std::array<TileSpec, 3> specs{{{27}, {27}, {26}}};

    std::vector<DPULayer> splitLayers;
    splitLayers.reserve(specs.size());

    for (const auto& s : specs) {
        DPUWorkload wl{
                device,
                Operation::AVEPOOL,
                {VPUTensor(1280, s.h, 16, 1, DataType::FLOAT16, Layout::ZXY)},  // input tile  (W,H,C,N)
                {VPUTensor(1280, s.h, 8,  1, DataType::FLOAT16, Layout::XYZ)},  // output tile (W,H,C,N)
                {1, 1}, {1, 1}, {0, 0, 0, 0},
                ExecutionMode::CUBOID_16x16,
                ActivationFunction::LRELU,
                0.0F, 0.0F,
                {Swizzling::KEY_0, Swizzling::KEY_0},
                {Swizzling::KEY_0},
                1,
                {0, 0, 0, 0},
                ISIStrategy::CLUSTERING,
                false,
        };

        // Matches compiler log: output channels < 16 -> output autopad enabled.
        wl.output_autopad = true;

        splitLayers.emplace_back(wl);
    }

    Logger::clear2ndlog();
    CyclesInterfaceType cost{};
    LayerSplitInfo detailed_split{};

    ASSERT_NO_THROW(cost = theModel.LayersPreSplit(splitLayers, 1U, false, false, prefetch, detailed_split))
            << "\n2ndLog:\n" << Logger::get2ndlog();

    EXPECT_FALSE(Cycles::isErrorCode(cost))
            << "LayersPreSplit returned error for AVEPOOL 1x1 (LRELU), 16->8 channels, "
               "H-tiles {27,27,26} on NPU6010.\n"
            << "Error: " << Cycles::toErrorText(cost) << "\n"
            << "\n2ndLog:\n" << Logger::get2ndlog();

    ASSERT_EQ(detailed_split.size(), 3U) << "Expected 3 tile entries in split info.";
    for (unsigned int j = 0; j < detailed_split.size(); ++j) {
        const auto& tile = detailed_split[j].inter_tile_split_layer;
        EXPECT_EQ(tile.inputs[0].channels(), 16U)
                << "Tile " << j << ": input channels changed from 16 to " << tile.inputs[0].channels();
        EXPECT_EQ(tile.outputs[0].channels(), 8U)
                << "Tile " << j << ": output channels changed from 8 to " << tile.outputs[0].channels();
    }
}

}  // namespace VPUNN_unit_tests
