// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.


#include "layer.h"

#include <gtest/gtest.h>
#include "vpu/dpu_types.h"
#include "vpu_layer_cost_model.h"
#include "vpu/shave/layers.h"
#include "vpu/sample_generator/random_task_generator.h"
#include "core/profiling.h"

#include <algorithm>
#include <numeric>
#include <iostream>

namespace VPUNN_unit_tests {
using namespace VPUNN;

/// Empty input returns empty result with isValid == false
TEST_F(VPULayerCostModelTest, LayerBatched_EmptyInput) {
    auto& model = layer_models.getModel(VPUDevice::VPU_2_7);
    std::vector<std::reference_wrapper<LayerBatchElementInfo>> empty_layers;

    BatchCostResult result = model.LayerBatched(empty_layers);

    EXPECT_FALSE(result.isValid);
    EXPECT_TRUE(result.costs.empty());
}

/// Single DPULayer gives consistent cost with Layer(), detailed_split populated through reference
TEST_F(VPULayerCostModelTest, LayerBatched_SingleDPULayer_MatchesLayer) {
    auto& model = layer_models.getModel(VPUDevice::VPU_2_7);
    VPULayerStrategy strategy = makeDefaultLayerStrategy();

    // Get reference cost via Layer()
    DPULayer layer_ref = makeBatchTestLayer(VPUDevice::VPU_2_7);
    LayerSplitInfo ref_split;
    CyclesInterfaceType ref_cost = model.Layer(layer_ref, strategy, ref_split);

    // Get cost via LayerBatched() - with detailed_split pre-populated so it gets filled in-place
    LayerBatchElementInfo lbi{makeBatchTestLayer(VPUDevice::VPU_2_7), strategy, std::make_unique<LayerSplitInfo>()};

    BatchCostResult result = model.LayerBatched({std::ref(lbi)});

    ASSERT_EQ(result.costs.size(), 1u);
    EXPECT_EQ(result.costs[0], ref_cost);
    // detailed_split was populated through the reference
    ASSERT_TRUE(lbi.detailed_split != nullptr);
}

/// Without detailed_split (nullptr), function still works correctly
TEST_F(VPULayerCostModelTest, LayerBatched_NoDetailedSplit) {
    auto& model = layer_models.getModel(VPUDevice::VPU_2_7);
    VPULayerStrategy strategy = makeDefaultLayerStrategy();

    // detailed_split defaults to nullptr - no split info requested
    LayerBatchElementInfo lbi{makeBatchTestLayer(VPUDevice::VPU_2_7), strategy};

    BatchCostResult result = model.LayerBatched({std::ref(lbi)});

    ASSERT_EQ(result.costs.size(), 1u);
    EXPECT_FALSE(Cycles::isErrorCode(result.costs[0]));
    // detailed_split was not pre-populated, stays empty
    EXPECT_EQ(lbi.detailed_split, nullptr);
}

/// Multiple distinct DPU workloads produce per-entry costs and correct isValid
TEST_F(VPULayerCostModelTest, LayerBatched_MultipleDPULayers) {
    auto& model = layer_models.getModel(VPUDevice::VPU_2_7);
    VPULayerStrategy strategy = makeDefaultLayerStrategy();

    LayerBatchElementInfo lbi1{makeBatchTestLayer(VPUDevice::VPU_2_7, 16, 16, 64), strategy, std::make_unique<LayerSplitInfo>()};
    LayerBatchElementInfo lbi2{makeBatchTestLayer(VPUDevice::VPU_2_7, 8, 8, 32), strategy, std::make_unique<LayerSplitInfo>()};

    BatchCostResult result = model.LayerBatched({std::ref(lbi1), std::ref(lbi2)});

    ASSERT_EQ(result.costs.size(), 2u);
    // Both detailed_splits populated through the reference
    ASSERT_TRUE(lbi1.detailed_split != nullptr);
    ASSERT_TRUE(lbi2.detailed_split != nullptr);

    CyclesInterfaceType c0 = result.costs[0];
    CyclesInterfaceType c1 = result.costs[1];

    if (!Cycles::isErrorCode(c0) && !Cycles::isErrorCode(c1)) {
        EXPECT_TRUE(result.isValid);
    }
}

/// Error propagation: if any config produces an error, isValid is false
TEST_F(VPULayerCostModelTest, LayerBatched_ErrorPropagation) {
    auto& model = layer_models.getModel(VPUDevice::VPU_2_7);
    VPULayerStrategy strategy = makeDefaultLayerStrategy();

    // Valid layer
    LayerBatchElementInfo lbi_valid{makeBatchTestLayer(VPUDevice::VPU_2_7, 16, 16, 64), strategy};

    // Invalid layer (0 channels -> error)
    LayerBatchElementInfo lbi_invalid{makeBatchTestLayer(VPUDevice::VPU_2_7, 16, 16, 0), strategy};

    // Valid first, invalid second
    BatchCostResult result = model.LayerBatched({std::ref(lbi_valid), std::ref(lbi_invalid)});

    ASSERT_EQ(result.costs.size(), 2u);

    CyclesInterfaceType c_invalid = result.costs[1];
    EXPECT_TRUE(Cycles::isErrorCode(c_invalid))
            << "Invalid workload should produce an error code, got: " << c_invalid;

    // isValid should be false since one config is erroneous
    EXPECT_FALSE(result.isValid)
            << "isValid should be false when any config has error";
}

/// Multiple devices can each use LayerBatched
TEST_F(VPULayerCostModelTest, LayerBatched_MultipleDevices) {
    const std::vector<VPUDevice> devices = {VPUDevice::VPU_2_0, VPUDevice::VPU_2_7, VPUDevice::VPU_4_0};
    const std::vector<Layout> layouts = {Layout::ZMAJOR, Layout::ZXY, Layout::ZXY};

    for (size_t d = 0; d < devices.size(); ++d) {
        auto& model = layer_models.getModel(devices[d]);
        VPULayerStrategy strategy = makeDefaultLayerStrategy();

        DPUWorkload wl{
                devices[d],
                Operation::CONVOLUTION,
                {VPUTensor(16, 16, 64, 1, DataType::UINT8, layouts[d])},
                {VPUTensor(16, 16, 64, 1, DataType::UINT8, layouts[d])},
                {1, 1},
                {1, 1},
                {0, 0, 0, 0},
                ExecutionMode::CUBOID_16x16,
        };

        LayerBatchElementInfo lbi{DPULayer(wl), strategy};

        BatchCostResult result;
        ASSERT_NO_THROW(result = model.LayerBatched({std::ref(lbi)}))
                << "LayerBatched should not throw for device " << static_cast<int>(devices[d]);

        ASSERT_EQ(result.costs.size(), 1u);
        ASSERT_TRUE(!Cycles::isErrorCode(result.costs[0]))
                << "Cost should be > 0 for device " << static_cast<int>(devices[d]);
    }
}

/// LayerBatched matches calling Layer() individually for each entry
TEST_F(VPULayerCostModelTest, LayerBatched_MatchesIndividualCalls) {
    auto& model = layer_models.getModel(VPUDevice::VPU_2_7);
    VPULayerStrategy strategy = makeDefaultLayerStrategy();

    // Build several distinct workloads
    struct WorkloadSpec {
        unsigned int width, height, channels;
    };
    const std::vector<WorkloadSpec> specs = {
            {16, 16, 64},
            {8, 8, 32},
            {32, 32, 16},
            {4, 4, 128},
    };

    // Prepare batch input - elements must be lvalues for reference_wrapper
    std::vector<LayerBatchElementInfo> batch_elements;
    batch_elements.reserve(specs.size());
    for (const auto& s : specs) {
        batch_elements.push_back({makeBatchTestLayer(VPUDevice::VPU_2_7, s.width, s.height, s.channels), strategy});
    }

    std::vector<std::reference_wrapper<LayerBatchElementInfo>> batch_refs;
    batch_refs.reserve(batch_elements.size());
    for (auto& elem : batch_elements) {
        batch_refs.push_back(std::ref(elem));
    }

    // Call LayerBatched
    BatchCostResult batched_result = model.LayerBatched(batch_refs);

    ASSERT_EQ(batched_result.costs.size(), specs.size());

    // Verify each per-entry cost matches the individual Layer() call
    for (size_t i = 0; i < specs.size(); ++i) {
        DPULayer layer_copy = makeBatchTestLayer(VPUDevice::VPU_2_7, specs[i].width, specs[i].height,
                                                 specs[i].channels);
        LayerSplitInfo ref_split;
        CyclesInterfaceType single_cost = model.Layer(layer_copy, strategy, ref_split);

        EXPECT_EQ(batched_result.costs[i], single_cost)
                << "Mismatch for workload index " << i;
    }
}

/// DDR input/output spilling flags affect the cost
TEST_F(VPULayerCostModelTest, LayerBatched_DDR_Spilling_Affects_Cost) {
    auto& model = layer_models.getModel(VPUDevice::VPU_2_7);

    auto make_lbi = [this](bool input_ddr, bool output_ddr) -> LayerBatchElementInfo {
        return LayerBatchElementInfo{makeBatchTestLayer(VPUDevice::VPU_2_7, 16, 16, 64),
                                     makeDefaultLayerStrategy(input_ddr, output_ddr)};
    };

    // Elements must be lvalues for reference_wrapper
    LayerBatchElementInfo lbi_no_ddr = make_lbi(false, false);
    LayerBatchElementInfo lbi_in_ddr = make_lbi(true, false);
    LayerBatchElementInfo lbi_out_ddr = make_lbi(false, true);
    LayerBatchElementInfo lbi_both_ddr = make_lbi(true, true);

    BatchCostResult result_no_ddr = model.LayerBatched({std::ref(lbi_no_ddr)});
    BatchCostResult result_in_ddr = model.LayerBatched({std::ref(lbi_in_ddr)});
    BatchCostResult result_out_ddr = model.LayerBatched({std::ref(lbi_out_ddr)});
    BatchCostResult result_both_ddr = model.LayerBatched({std::ref(lbi_both_ddr)});

    ASSERT_EQ(result_no_ddr.costs.size(), 1u);
    ASSERT_EQ(result_in_ddr.costs.size(), 1u);
    ASSERT_EQ(result_out_ddr.costs.size(), 1u);
    ASSERT_EQ(result_both_ddr.costs.size(), 1u);

    CyclesInterfaceType c_none = result_no_ddr.costs[0];
    CyclesInterfaceType c_in = result_in_ddr.costs[0];
    CyclesInterfaceType c_out = result_out_ddr.costs[0];
    CyclesInterfaceType c_both = result_both_ddr.costs[0];

    // DDR access should add DMA costs, so cost should be >= no-DDR cost
    if (!Cycles::isErrorCode(c_none) && !Cycles::isErrorCode(c_in)) {
        EXPECT_GE(c_in, c_none) << "Input DDR fetching should not reduce cost";
    }
    if (!Cycles::isErrorCode(c_none) && !Cycles::isErrorCode(c_out)) {
        EXPECT_GE(c_out, c_none) << "Output DDR spilling should not reduce cost";
    }
    if (!Cycles::isErrorCode(c_none) && !Cycles::isErrorCode(c_both)) {
        EXPECT_GE(c_both, c_none) << "Both DDR flags should not reduce cost";
    }
}

// ===== LayersPreSplitBatched tests =====

/// Empty input returns empty result with isValid == false
TEST_F(VPULayerCostModelTest, LayersPreSplitBatched_EmptyInput) {
    auto& model = layer_models.getModel(VPUDevice::VPU_2_7);
    std::vector<std::reference_wrapper<LayersPreSplitBatchElementInfo>> empty_pre_split;

    BatchCostResult result = model.LayersPreSplitBatched(empty_pre_split);

    EXPECT_FALSE(result.isValid);
    EXPECT_TRUE(result.costs.empty());
}

/// Pre-split layers with hash: cost > 0 and detailed_split populated through reference
TEST_F(VPULayerCostModelTest, LayersPreSplitBatched_WithHash) {
    auto& model = layer_models.getModel(VPUDevice::VPU_2_7);

    DPULayer full_layer = makeBatchTestLayer(VPUDevice::VPU_2_7, 16, 16, 64);
    std::vector<DPULayer> tiles = full_layer.splitAcrossTiles(VPUTilingStrategy::SOK, 2);

    VPULayersPreSplitStrategy strategy = makeDefaultPreSplitStrategy();

    LayersPreSplitBatchElementInfo psbi;
    psbi.layer_splits = tiles;
    psbi.strategy = strategy;
    psbi.fullLayerHash = full_layer.hash();
    psbi.detailed_split = std::make_unique<LayerSplitInfo>();  // request detailed split

    BatchCostResult result = model.LayersPreSplitBatched({std::ref(psbi)});

    ASSERT_EQ(result.costs.size(), 1u);

    CyclesInterfaceType cost = result.costs[0];
    ASSERT_TRUE(!Cycles::isErrorCode(cost)) << "Cost should not be an error code";
    EXPECT_GT(cost, 0u) << "Pre-split layer cost should not be zero";
    // detailed_split was populated through the reference
    ASSERT_TRUE(psbi.detailed_split != nullptr);
}

/// Pre-split layers with no hash (nullopt, defaults to 0 internally)
TEST_F(VPULayerCostModelTest, LayersPreSplitBatched_NoHash) {
    auto& model = layer_models.getModel(VPUDevice::VPU_2_7);

    DPULayer full_layer = makeBatchTestLayer(VPUDevice::VPU_2_7, 16, 16, 64);
    std::vector<DPULayer> tiles = full_layer.splitAcrossTiles(VPUTilingStrategy::SOK, 2);

    VPULayersPreSplitStrategy strategy = makeDefaultPreSplitStrategy();

    LayersPreSplitBatchElementInfo psbi;
    psbi.layer_splits = tiles;
    psbi.strategy = strategy;
    psbi.fullLayerHash = std::nullopt;  // no hash

    BatchCostResult result;
    ASSERT_NO_THROW(result = model.LayersPreSplitBatched({std::ref(psbi)}));

    ASSERT_EQ(result.costs.size(), 1u);
    EXPECT_FALSE(Cycles::isErrorCode(result.costs[0]));
    EXPECT_GT(result.costs[0], 0u);
}

/// With and without detailed_split: both code paths produce same cost
TEST_F(VPULayerCostModelTest, LayersPreSplitBatched_WithAndWithoutDetailedSplit) {
    auto& model = layer_models.getModel(VPUDevice::VPU_2_7);

    DPULayer full_layer = makeBatchTestLayer(VPUDevice::VPU_2_7, 16, 16, 64);
    std::vector<DPULayer> tiles = full_layer.splitAcrossTiles(VPUTilingStrategy::SOK, 2);

    VPULayersPreSplitStrategy strategy = makeDefaultPreSplitStrategy();

    // Without detailed_split (nullptr) - lighter overload
    LayersPreSplitBatchElementInfo psbi_no_split;
    psbi_no_split.layer_splits = tiles;
    psbi_no_split.strategy = strategy;
    psbi_no_split.fullLayerHash = full_layer.hash();
    // detailed_split defaults to nullptr

    BatchCostResult result_no_split = model.LayersPreSplitBatched({std::ref(psbi_no_split)});

    // With detailed_split pre-populated - full overload (populated through reference)
    LayersPreSplitBatchElementInfo psbi_with_split;
    psbi_with_split.layer_splits = tiles;
    psbi_with_split.strategy = strategy;
    psbi_with_split.fullLayerHash = full_layer.hash();
    psbi_with_split.detailed_split = std::make_unique<LayerSplitInfo>();

    BatchCostResult result_with_split = model.LayersPreSplitBatched({std::ref(psbi_with_split)});

    ASSERT_EQ(result_no_split.costs.size(), 1u);
    ASSERT_EQ(result_with_split.costs.size(), 1u);

    // Both code paths should produce the same cost
    EXPECT_EQ(result_no_split.costs[0], result_with_split.costs[0])
            << "Cost should be identical regardless of detailed_split presence";
}

/// Multiple pre-split entries produce per-entry costs and correct isValid
TEST_F(VPULayerCostModelTest, LayersPreSplitBatched_MultipleLayers) {
    auto& model = layer_models.getModel(VPUDevice::VPU_2_7);

    VPULayersPreSplitStrategy strategy = makeDefaultPreSplitStrategy();

    DPULayer layer1 = makeBatchTestLayer(VPUDevice::VPU_2_7, 16, 16, 64);
    std::vector<DPULayer> tiles1 = layer1.splitAcrossTiles(VPUTilingStrategy::SOK, 2);

    DPULayer layer2 = makeBatchTestLayer(VPUDevice::VPU_2_7, 8, 8, 32);
    std::vector<DPULayer> tiles2 = layer2.splitAcrossTiles(VPUTilingStrategy::SOK, 2);

    LayersPreSplitBatchElementInfo psbi1;
    psbi1.layer_splits = tiles1;
    psbi1.strategy = strategy;
    psbi1.fullLayerHash = layer1.hash();

    LayersPreSplitBatchElementInfo psbi2;
    psbi2.layer_splits = tiles2;
    psbi2.strategy = strategy;
    psbi2.fullLayerHash = layer2.hash();

    BatchCostResult result = model.LayersPreSplitBatched({std::ref(psbi1), std::ref(psbi2)});

    ASSERT_EQ(result.costs.size(), 2u);

    CyclesInterfaceType c0 = result.costs[0];
    CyclesInterfaceType c1 = result.costs[1];

    if (!Cycles::isErrorCode(c0) && !Cycles::isErrorCode(c1)) {
        EXPECT_TRUE(result.isValid);
    }
}

/// LayersPreSplitBatched matches calling LayersPreSplit individually
TEST_F(VPULayerCostModelTest, LayersPreSplitBatched_MatchesIndividualCalls) {
    auto& model = layer_models.getModel(VPUDevice::VPU_2_7);

    VPULayersPreSplitStrategy strategy = makeDefaultPreSplitStrategy();

    DPULayer full_layer = makeBatchTestLayer(VPUDevice::VPU_2_7, 16, 16, 64);
    std::vector<DPULayer> tiles = full_layer.splitAcrossTiles(VPUTilingStrategy::SOK, 2);

    // Get reference cost via LayersPreSplit()
    LayerSplitInfo ref_split;
    CyclesInterfaceType ref_cost = model.LayersPreSplit(tiles, strategy.nDPUs, strategy.input_fetching,
                                                         strategy.output_spilling, strategy.prefetching, ref_split,
                                                         full_layer.hash(), strategy.tiling_strategy);

    // Get cost via LayersPreSplitBatched()
    LayersPreSplitBatchElementInfo psbi;
    psbi.layer_splits = tiles;
    psbi.strategy = strategy;
    psbi.fullLayerHash = full_layer.hash();
    psbi.detailed_split = std::make_unique<LayerSplitInfo>();

    BatchCostResult result = model.LayersPreSplitBatched({std::ref(psbi)});

    ASSERT_EQ(result.costs.size(), 1u);
    EXPECT_EQ(result.costs[0], ref_cost)
            << "LayersPreSplitBatched cost should match individual LayersPreSplit call";
}

// ===== Non-reference_wrapper overload tests =====

/// LayerBatched(vector<LayerBatchElementInfo>&) single element matches reference_wrapper overload
TEST_F(VPULayerCostModelTest, LayerBatched_NoRefWrapper_SingleElement) {
    auto& model = layer_models.getModel(VPUDevice::VPU_2_7);
    VPULayerStrategy strategy = makeDefaultLayerStrategy();

    // Via non-ref-wrapper overload
    std::vector<LayerBatchElementInfo> layers_vec;
    layers_vec.push_back(LayerBatchElementInfo{makeBatchTestLayer(VPUDevice::VPU_2_7), strategy, std::make_unique<LayerSplitInfo>()});
    BatchCostResult result_vec = model.LayerBatched(layers_vec);

    // Via ref-wrapper overload for comparison
    LayerBatchElementInfo lbi_ref{makeBatchTestLayer(VPUDevice::VPU_2_7), strategy, std::make_unique<LayerSplitInfo>()};
    BatchCostResult result_ref = model.LayerBatched({std::ref(lbi_ref)});

    ASSERT_EQ(result_vec.costs.size(), 1u);
    ASSERT_EQ(result_ref.costs.size(), 1u);
    EXPECT_EQ(result_vec.costs[0], result_ref.costs[0]);
    EXPECT_EQ(result_vec.isValid, result_ref.isValid);
}

/// LayerBatched(vector<LayerBatchElementInfo>&) with multiple elements
TEST_F(VPULayerCostModelTest, LayerBatched_NoRefWrapper_MultipleElements) {
    auto& model = layer_models.getModel(VPUDevice::VPU_2_7);
    VPULayerStrategy strategy = makeDefaultLayerStrategy();

    std::vector<LayerBatchElementInfo> layers_vec;
    layers_vec.push_back({makeBatchTestLayer(VPUDevice::VPU_2_7, 16, 16, 64), strategy});
    layers_vec.push_back({makeBatchTestLayer(VPUDevice::VPU_2_7, 8, 8, 32), strategy});
    layers_vec.push_back({makeBatchTestLayer(VPUDevice::VPU_2_7, 32, 32, 16), strategy});

    BatchCostResult result = model.LayerBatched(layers_vec);

    ASSERT_EQ(result.costs.size(), 3u);
    for (size_t i = 0; i < result.costs.size(); ++i) {
        const auto cost = result.costs[i];
        EXPECT_FALSE(Cycles::isErrorCode(cost)) << "Cost at index " << i << " should not be an error code";
        if (!Cycles::isErrorCode(cost)) {
            EXPECT_GT(cost, 0u) << "Cost at index " << i << " should be > 0";
        }
    }
}

/// LayerBatched(vector<>&) populates detailed_split in-place
TEST_F(VPULayerCostModelTest, LayerBatched_NoRefWrapper_DetailedSplitPopulated) {
    auto& model = layer_models.getModel(VPUDevice::VPU_2_7);
    VPULayerStrategy strategy = makeDefaultLayerStrategy();

    std::vector<LayerBatchElementInfo> layers_vec;
    layers_vec.push_back(LayerBatchElementInfo{makeBatchTestLayer(VPUDevice::VPU_2_7), strategy, std::make_unique<LayerSplitInfo>()});

    BatchCostResult result = model.LayerBatched(layers_vec);

    ASSERT_EQ(result.costs.size(), 1u);
    if (!Cycles::isErrorCode(result.costs[0])) {
        ASSERT_TRUE(layers_vec[0].detailed_split != nullptr);
        EXPECT_FALSE(layers_vec[0].detailed_split->empty())
                << "detailed_split should be populated through the vector elements";
    }
}

/// LayerBatched(vector<>&) with empty input
TEST_F(VPULayerCostModelTest, LayerBatched_NoRefWrapper_EmptyInput) {
    auto& model = layer_models.getModel(VPUDevice::VPU_2_7);
    std::vector<LayerBatchElementInfo> empty_vec;

    BatchCostResult result = model.LayerBatched(empty_vec);

    EXPECT_FALSE(result.isValid);
    EXPECT_TRUE(result.costs.empty());
}

/// LayersPreSplitBatched(vector<LayersPreSplitBatchElementInfo>&) single element matches ref-wrapper overload
TEST_F(VPULayerCostModelTest, LayersPreSplitBatched_NoRefWrapper_SingleElement) {
    auto& model = layer_models.getModel(VPUDevice::VPU_2_7);
    VPULayersPreSplitStrategy strategy = makeDefaultPreSplitStrategy();

    DPULayer full_layer = makeBatchTestLayer(VPUDevice::VPU_2_7, 16, 16, 64);
    std::vector<DPULayer> tiles = full_layer.splitAcrossTiles(VPUTilingStrategy::SOK, 2);

    // Via non-ref-wrapper overload
    std::vector<LayersPreSplitBatchElementInfo> vec;
    {
        LayersPreSplitBatchElementInfo psbi;
        psbi.layer_splits = tiles;
        psbi.strategy = strategy;
        psbi.fullLayerHash = full_layer.hash();
        psbi.detailed_split = std::make_unique<LayerSplitInfo>();
        vec.push_back(std::move(psbi));
    }
    BatchCostResult result_vec = model.LayersPreSplitBatched(vec);

    // Via ref-wrapper overload for comparison
    LayersPreSplitBatchElementInfo psbi_ref;
    psbi_ref.layer_splits = tiles;
    psbi_ref.strategy = strategy;
    psbi_ref.fullLayerHash = full_layer.hash();
    psbi_ref.detailed_split = std::make_unique<LayerSplitInfo>();
    BatchCostResult result_ref = model.LayersPreSplitBatched({std::ref(psbi_ref)});

    ASSERT_EQ(result_vec.costs.size(), 1u);
    ASSERT_EQ(result_ref.costs.size(), 1u);
    EXPECT_EQ(result_vec.costs[0], result_ref.costs[0]);
    EXPECT_EQ(result_vec.isValid, result_ref.isValid);
}

/// LayersPreSplitBatched(vector<>&) with multiple elements
TEST_F(VPULayerCostModelTest, LayersPreSplitBatched_NoRefWrapper_MultipleElements) {
    auto& model = layer_models.getModel(VPUDevice::VPU_2_7);
    VPULayersPreSplitStrategy strategy = makeDefaultPreSplitStrategy();

    DPULayer layer1 = makeBatchTestLayer(VPUDevice::VPU_2_7, 16, 16, 64);
    std::vector<DPULayer> tiles1 = layer1.splitAcrossTiles(VPUTilingStrategy::SOK, 2);

    DPULayer layer2 = makeBatchTestLayer(VPUDevice::VPU_2_7, 8, 8, 32);
    std::vector<DPULayer> tiles2 = layer2.splitAcrossTiles(VPUTilingStrategy::SOK, 2);

    std::vector<LayersPreSplitBatchElementInfo> vec;
    {
        LayersPreSplitBatchElementInfo psbi1;
        psbi1.layer_splits = tiles1;
        psbi1.strategy = strategy;
        psbi1.fullLayerHash = layer1.hash();
        vec.push_back(std::move(psbi1));
    }
    {
        LayersPreSplitBatchElementInfo psbi2;
        psbi2.layer_splits = tiles2;
        psbi2.strategy = strategy;
        psbi2.fullLayerHash = layer2.hash();
        vec.push_back(std::move(psbi2));
    }

    BatchCostResult result = model.LayersPreSplitBatched(vec);

    ASSERT_EQ(result.costs.size(), 2u);
    for (size_t i = 0; i < result.costs.size(); ++i) {
        EXPECT_GT(result.costs[i], 0u) << "Cost at index " << i << " should be > 0";
    }
}

/// LayersPreSplitBatched(vector<>&) populates detailed_split in-place
TEST_F(VPULayerCostModelTest, LayersPreSplitBatched_NoRefWrapper_DetailedSplitPopulated) {
    auto& model = layer_models.getModel(VPUDevice::VPU_2_7);
    VPULayersPreSplitStrategy strategy = makeDefaultPreSplitStrategy();

    DPULayer full_layer = makeBatchTestLayer(VPUDevice::VPU_2_7, 16, 16, 64);
    std::vector<DPULayer> tiles = full_layer.splitAcrossTiles(VPUTilingStrategy::SOK, 2);

    std::vector<LayersPreSplitBatchElementInfo> vec;
    {
        LayersPreSplitBatchElementInfo psbi;
        psbi.layer_splits = tiles;
        psbi.strategy = strategy;
        psbi.fullLayerHash = full_layer.hash();
        psbi.detailed_split = std::make_unique<LayerSplitInfo>();
        vec.push_back(std::move(psbi));
    }

    BatchCostResult result = model.LayersPreSplitBatched(vec);

    ASSERT_EQ(result.costs.size(), 1u);
    if (!Cycles::isErrorCode(result.costs[0])) {
        ASSERT_TRUE(vec[0].detailed_split != nullptr);
        EXPECT_FALSE(vec[0].detailed_split->empty())
                << "detailed_split should be populated through the vector elements";
    }
}

/// LayersPreSplitBatched(vector<>&) with empty input
TEST_F(VPULayerCostModelTest, LayersPreSplitBatched_NoRefWrapper_EmptyInput) {
    auto& model = layer_models.getModel(VPUDevice::VPU_2_7);
    std::vector<LayersPreSplitBatchElementInfo> empty_vec;

    BatchCostResult result = model.LayersPreSplitBatched(empty_vec);

    EXPECT_FALSE(result.isValid);
    EXPECT_TRUE(result.costs.empty());
}

/// Caller can observe modifications through references (layer and detailed_split)
TEST_F(VPULayerCostModelTest, LayerBatched_ReferenceSemantics) {
    auto& model = layer_models.getModel(VPUDevice::VPU_2_7);
    VPULayerStrategy strategy = makeDefaultLayerStrategy();

    // Pre-populate detailed_split as empty; after call it should be populated in-place
    LayerBatchElementInfo lbi{makeBatchTestLayer(VPUDevice::VPU_2_7), strategy, std::make_unique<LayerSplitInfo>()};
    ASSERT_TRUE(lbi.detailed_split != nullptr);
    EXPECT_TRUE(lbi.detailed_split->empty());

    BatchCostResult result = model.LayerBatched({std::ref(lbi)});

    ASSERT_EQ(result.costs.size(), 1u);
    if (!Cycles::isErrorCode(result.costs[0])) {
        // The caller's lbi.detailed_split should now be populated through the reference
        ASSERT_TRUE(lbi.detailed_split != nullptr);
        EXPECT_FALSE(lbi.detailed_split->empty())
                << "detailed_split should be populated in-place through reference_wrapper";
    }
}

/// Measures execution time of LayerBatched with and without unique_ptr<LayerSplitInfo>
///
/// Avoids warmup bias, and alternates between unique_ptr and no unique_ptr
TEST_F(VPULayerCostModelTest, DISABLED_LayerSplitInfo_Unique_Ptr_Load) {
    auto device = VPUDevice::VPU_2_7;
    auto& model = layer_models.getModel(device);
    VPULayerStrategy strategy = makeDefaultLayerStrategy();

    constexpr int REPS = 10;  // repetitions per variant per batch size
    const std::vector<int> batch_sizes = {10, 100, 500}; // tested sizes for number of layers

    // Helper: build a batch vector + refs from a set of workloads, optionally with unique_ptr
    auto make_batch = [&](const std::vector<DPUWorkload>& wls, bool with_ptr) 
            -> std::vector<LayerBatchElementInfo> {
        std::vector<LayerBatchElementInfo> batch;
        batch.reserve(wls.size());
        for (const auto& wl : wls) {
            if (with_ptr)
                batch.push_back({DPULayer(wl), strategy, std::make_unique<LayerSplitInfo>()});
            else
                batch.push_back({DPULayer(wl), strategy});
        }
        return batch;
    };

    auto compute_avg = [](const std::vector<double>& accumulator) {
        double sum = std::reduce(accumulator.begin(), accumulator.end());
        return (accumulator.size() > 0) ? (sum / accumulator.size()) : 0.0;
    };

    for (int n : batch_sizes) {
        std::vector<DPUWorkload> workloads(n);
        std::generate_n(workloads.begin(), n, VPUNN::randDPUWorkload(device));

        // Warm-up: bring in model state and workload data
        {
            auto warm_layers = make_batch(workloads, /*with_ptr=*/false);
            (void)model.LayerBatched(warm_layers);
        }

        std::vector<double> no_ptr_acc, with_ptr_acc;

        BatchCostResult result_no_ptr, result_with_ptr;

        for (int rep = 0; rep < REPS; ++rep) {
            if (rep % 2 == 0) {
                // Even repetition: no-ptr first, then with-ptr
                {
                    auto elems = make_batch(workloads, false);
                    const auto t0 = VPUNN::tick();
                    result_no_ptr = model.LayerBatched(elems);
                    no_ptr_acc.push_back(VPUNN::tock(t0));
                }
                {
                    auto elems = make_batch(workloads, true);
                    const auto t0 = VPUNN::tick();
                    result_with_ptr = model.LayerBatched(elems);
                    with_ptr_acc.push_back(VPUNN::tock(t0));
                }
            } else {
                // Odd repetition: with-ptr first (reverses order to cancel warm-up bias)
                {
                    auto elems = make_batch(workloads, true);
                    const auto t0 = VPUNN::tick();
                    result_with_ptr = model.LayerBatched(elems);
                    with_ptr_acc.push_back(VPUNN::tock(t0));
                }
                {
                    auto elems = make_batch(workloads, false);
                    const auto t0 = VPUNN::tick();
                    result_no_ptr = model.LayerBatched(elems);
                    no_ptr_acc.push_back(VPUNN::tock(t0));
                }
            }
        }

        double no_ptr_avg = compute_avg(no_ptr_acc);
        double with_ptr_avg = compute_avg(with_ptr_acc);
        std::cout << "batch_size=" << n
                  << " | without unique_ptr (avg over " << REPS << " reps): " << no_ptr_avg << " ms"
                  << " | with unique_ptr (avg over " << REPS << " reps): " << with_ptr_avg << " ms"
                  << " | delta (with - without): " << (with_ptr_avg - no_ptr_avg) << " ms"
                  << std::endl;

        // both runs must return the expected number of cost entries
        ASSERT_EQ(result_no_ptr.costs.size(), static_cast<size_t>(n))
                << "batch_size=" << n << ": wrong cost count for no-ptr run";
        ASSERT_EQ(result_with_ptr.costs.size(), static_cast<size_t>(n))
                << "batch_size=" << n << ": wrong cost count for with-ptr run";
    }
}

} // namespace VPUNN_unit_tests