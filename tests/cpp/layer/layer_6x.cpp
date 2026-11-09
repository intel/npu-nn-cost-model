
// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include "layer.h"

#include <string_view>

namespace VPUNN_unit_tests {
using namespace VPUNN;

class VPULayerCostModelTestNPU6x : public VPULayerCostModelTest {
public:
protected:
    void SetUp() override {
        VPULayerCostModelTest::SetUp();
    }
    const HaloWorkload haloZero{
            {0, 0, 0, 0, 0, 0},  // input_0_halo
            {0, 0, 0, 0, 0, 0},  // output_0_halo
            {0, 0, 0, 0, 0, 0},  // output_0_halo_broadcast_cnt
            {0, 0, 0, 0, 0, 0}   // output_0_inbound_halo
    };
    // tiled workloads
    const DPUWorkload wl_tile_ref_DWconv{
            VPUDevice::NPU_6_0,
            Operation::DW_CONVOLUTION,
            {VPUTensor(56, 56, 128, 1, DataType::UINT8)},  // input dimensions
            {VPUTensor(56, 56, 128, 1, DataType::UINT8)},  // output dimensions
            {1, 1},                                        // kernels
            {1, 1},                                        // strides
            {0, 0, 0, 0},                                  // padding
            ExecutionMode::CUBOID_16x16,                   // execution mode
            ActivationFunction::NONE,                      // activation
            0.0F,                                          // act_sparsity
            0.0F,                                          // weight_sparsity
            {swz_def, swz_def},                            // input_swizzling
            {swz_def},                                     // output_swizzling
            1,                                             // output_write_tiles
            {0, 0, 0, 0},                                  // offsets
            ISIStrategy::SPLIT_OVER_K_nxtHW,               // isi_strategy
            false,                                         // weight_sparsity_enabled
    };

    const DPUWorkload wl_k3x3_CONV{
            VPUDevice::NPU_6_0,
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
            ISIStrategy::SPLIT_OVER_K_nxtHW,               // isi_strategy
            false,                                         // weight_sparsity_enabled
    };
    const DPUWorkload wl_k1x1_CONV{
            VPUDevice::NPU_6_0,
            Operation::CONVOLUTION,
            {VPUTensor(56, 56, 128, 1, DataType::UINT8)},  // input dimensions
            {VPUTensor(56, 56, 128, 1, DataType::UINT8)},  // output dimensions
            {1, 1},                                        // kernels
            {1, 1},                                        // strides
            {0, 0, 0, 0},                                  // padding
            ExecutionMode::CUBOID_16x16,                   // execution mode
            ActivationFunction::NONE,                      // activation
            0.0F,                                          // act_sparsity
            0.0F,                                          // weight_sparsity
            {swz_def, swz_def},                            // input_swizzling
            {swz_def},                                     // output_swizzling
            1,                                             // output_write_tiles
            {0, 0, 0, 0},                                  // offsets
            ISIStrategy::SPLIT_OVER_K_nxtHW,               // isi_strategy
            false,                                         // weight_sparsity_enabled
    };

    // infrastructure
    const bool prefetch{true};  // prefetch was done
    VPULayerCostModel& theModel = layer_models.getModel(VPUDevice::NPU_6_0);
    // const LayersValidation layer_validator;
    size_t serializer_layer_uid = 123;

    void check_intraTile_halo(const std::vector<DPULayer> splitLayers,
                              const std::vector<std::vector<HaloWorkload>> expected_intratile_halos,
                              const int expected_intraTiles) {
        // L2 pre split
        Logger::clear2ndlog();
        CyclesInterfaceType pre_split_cost1{};
        LayerSplitInfo detailed_split_pre_layer{};

        ASSERT_NO_THROW(pre_split_cost1 = theModel.LayersPreSplit(splitLayers, 1U, false, false, prefetch,
                                                                  detailed_split_pre_layer, serializer_layer_uid))
                << toStringLayerSplitInfo(detailed_split_pre_layer);

        EXPECT_FALSE(Cycles::isErrorCode(pre_split_cost1))
                << Cycles::toErrorText(pre_split_cost1) << " " << toStringLayerSplitInfo(detailed_split_pre_layer);

        int tile_index = 0;

        // Sanity: expected count of tiles must match
        ASSERT_EQ(detailed_split_pre_layer.size(), expected_intratile_halos.size())
                << "Tile count mismatch. Dump:\n"
                << toStringLayerSplitInfo(detailed_split_pre_layer);

        // for each tile we compare the halos of its intra-tile workloads
        for (const auto& split : detailed_split_pre_layer) {
            EXPECT_EQ(split.best_intra_tile_split.second.size(), expected_intraTiles);
            const auto& intra_tile{split.best_intra_tile_split.second};

            // compare halos
            for (int intraTile_index = 0; intraTile_index < expected_intraTiles; intraTile_index++) {
                EXPECT_EQ(intra_tile[intraTile_index].halo, (expected_intratile_halos[tile_index])[intraTile_index])
                        << "Tile: " << tile_index + 1 << " Intra Tile: " << intraTile_index << " "
                        << intra_tile[intraTile_index];
            }

            tile_index++;
        }
    }
};
// TO DO: to be used when intra-tile split for H-W is enabled
// TEST_F(VPULayerCostModelTestNPU6x, TestSerializer_PreSplit_trivial_CONV_HW) {
//    // tiled workloads
//    const DPUWorkload wl_tile_ref {
//            VPUDevice::NPU_6_0,
//            Operation::CONVOLUTION,
//            {VPUTensor(56, 128, 128, 1, DataType::UINT8)},  // input dimensions
//            {VPUTensor(56, 128, 128, 1, DataType::UINT8)},  // output dimensions
//            {1, 1},                                        // kernels
//            {1, 1},                                        // strides
//            {0, 0, 0, 0},                                  // padding
//            ExecutionMode::CUBOID_16x16,                   // execution mode
//            ActivationFunction::NONE,                      // activation
//            0.0F,                                          // act_sparsity
//            0.0F,                                          // weight_sparsity
//            {swz_def, swz_def},                            // input_swizzling
//            {swz_def},                                     // output_swizzling
//            1,                                             // output_write_tiles
//            {0, 0, 0, 0},                                  // offsets
//            ISIStrategy::SPLIT_OVER_H,                     // isi_strategy
//            false,                                         // weight_sparsity_enabled
//    };
//
//
//    ////////////////////// Halo definitions for the 4 tiles //////////////////
//    const HaloWorkload haloTile1{haloZero};
//    const HaloWorkload haloTile2{haloZero};
//    const HaloWorkload haloTile3{haloZero};
//    const HaloWorkload haloTile4{haloZero};
//
//    DPULayer wl_tile_layer(wl_tile_ref);
//
//    DPULayer wl_t1 = wl_tile_layer;
//    wl_t1.halo = haloTile1;
//
//    DPULayer wl_t2 = wl_tile_layer;
//    wl_t2.halo = haloTile2;
//
//    DPULayer wl_t3 = wl_tile_layer;
//    wl_t3.halo = haloTile3;
//
//    DPULayer wl_t4 = wl_tile_layer;
//    wl_t4.halo = haloTile4;
//    ////////////////////////////////////////////////////////////////////////
//
//    ////////// Halo for the workloads that results after presplit //////////
//    const HaloWorkload haloTile1_intraT1{
//            {0, 0, 0, 0, 0, 0},  // input_0_halo
//            {0, 0, 0, 0, 0, 0},  // output_0_halo
//            {0, 0, 0, 0, 0, 0},  // output_0_halo_broadcast_cnt
//            {0, 0, 0, 0, 0, 0}  // output_0_inbound_halo
//    };
//    const HaloWorkload haloTile1_intraT2{
//            {0, 0, 0, 0, 0, 0},  // input_0_halo
//            {0, 0, 0, 0, 0, 0},  // output_0_halo
//            {0, 0, 0, 0, 0, 0},  // output_0_halo_broadcast_cnt
//            {0, 0, 0, 0, 0, 0}  // output_0_inbound_halo
//    };
//
//    const HaloWorkload haloTile2_intraT1{haloTile1_intraT1
//
//    };
//    const HaloWorkload haloTile2_intraT2{haloTile1_intraT2};
//
//    const HaloWorkload haloTile3_intraT1{haloTile1_intraT1};
//    const HaloWorkload haloTile3_intraT2{haloTile1_intraT2};
//
//    const HaloWorkload haloTile4_intraT1{haloTile1_intraT1};
//    const HaloWorkload haloTile4_intraT2{haloTile1_intraT2};
//
//    ////////////////////////////////////////////////////////////////////////
//
//    ///////////////////////////////////////////////////////////////////////////////
//    // 4 positions in std::vector means 4 tiles, each with 2 intra-tiles
//    // Expected halo pair for one tile: first and second candidate intra-tile workloads.
//    // - first position in from std::array expected HaloWorkload for candidate #0 (first candidate) of this tile
//    // - second position from std::array expected HaloWorkload for candidate #1 (second candidate) of this tile
//    std::vector<std::vector<HaloWorkload>> expected_intratile_halos = {{haloTile1_intraT1, haloTile1_intraT2},
//                                                                       {haloTile2_intraT1, haloTile2_intraT2},
//                                                                       {haloTile3_intraT1, haloTile3_intraT2},
//                                                                       {haloTile4_intraT1, haloTile4_intraT2}};
//
//    {
//        std::vector<DPULayer> tile_wls{wl_t1, wl_t2, wl_t3, wl_t4};
//        auto tiler = getDPUTiler(theModel.get_cost_model());
//        const SplitOptions options{128U, 0, 2}; // nDPU>1 allows us to perform HWTilingStrategy
//        LayerSplitInfo detailed_split;
//         for (auto& one_tile_layer : tile_wls) {
//            std::vector<DPUWorkloadsWithCyclesSplit>
//                    all_intra_tile_splits{};  ///< all intra tile splits generated. one pair() is a split
//            const DPUWorkloadsCost cost_and_workloads =
//                    tiler->intraTileSplit(one_tile_layer, options, &all_intra_tile_splits );
//            detailed_split.emplace_back(
//                    OneTileLayerInfo{one_tile_layer, std::move(cost_and_workloads),
//                    std::move(all_intra_tile_splits)});
//        }
//
//         int tile_index = 0;
//           // for each tile we compare the halos of its intra-tile workloads
//        for (const auto& split : detailed_split) {
//            EXPECT_EQ(split.all_intra_tile_splits[2].workloads.size(), 2);
//
//            // compare halos
//            for (int i = 0; i < 2; i++) {
//                EXPECT_EQ(split.all_intra_tile_splits[2].workloads[i].halo, expected_intratile_halos[tile_index][i])
//                        << "Tile: " << tile_index + 1 << " Intra Tile: " << i << " "
//                        << split.all_intra_tile_splits[2].workloads[i];
//            }
//
//            tile_index++;
//        }
//
//    }
//}

TEST_F(VPULayerCostModelTestNPU6x, DW_CONV_intraTileSplit_channels_options) {
    // Validate that the intra-tile splitting logic for DW_CONVOLUTION only generates
    // channel-split candidates that match the allowed/expected channel granularities.
    //
    // Context:
    // LayersPreSplit() produces, for each input tile, a list of candidate intra-tile splits
    // (split.all_intra_tile_splits). Each candidate contains one or more DPU workloads:
    //   - workloads.size() == 1  -> the "no split" candidate
    //   - workloads.size()  > 1  -> an actual intra-tile split
    //
    // What we check:
    // For every split candidate that actually splits (nWorkloads > 1), we inspect the output channels
    // of all resulting workloads. Each workload's channel count must fall within
    // validIntraTileChannelOptions, which also covers uneven splits where the last workload may have
    // fewer channels than the others.

    auto mk_wl = [&](int channels) {
        return DPUWorkload{
                VPUDevice::NPU_6_0,
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

    struct TestCase {
        DPUWorkload wl;
    };

    using TestsVector_ = std::vector<TestCase>;

    auto testIntraTilesOptions = [&](const TestsVector_& tests, const SmartRanges& validIntraTileChannelOptions) {
        for (auto t : tests) {
            const LayersValidation layer_validator;

            DPULayer wl_layer1(t.wl);

            // Run the layer pre-split stage and request SOK tiling strategy
            // detailed_split_pre_layer collects all candidate intra-tile splits, which we validate below.
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

            for (const auto& split : detailed_split_pre_layer) {
                // compare intraTiles channels after splits with all tiling options that can
                // be found in the SmartRanges validIntraTileChannelOptions
                for (int intraTile_index = 0; intraTile_index < static_cast<int>(split.all_intra_tile_splits.size());
                     intraTile_index++) {
                    const auto& wls{split.all_intra_tile_splits[intraTile_index]
                                            .workloads};  // workloads after intra-tile split

                    // Skip the non split candidate (nWorkloads == 1)
                    if (wls.size() == 1U) {
                        continue;
                    }

                    const auto channels =
                            static_cast<int>(wls[0].outputs[0].channels());  // channels after intra-tile split

                    std::string range_error{};
                    const bool is_valid = validIntraTileChannelOptions.is_in(channels, range_error);
                    EXPECT_TRUE(is_valid) << "Invalid intra-tile channels option: " << channels << ". " << range_error;
                }
            }
        }
    };

    // clang-format off
    TestsVector_ tests = {
        {mk_wl(128)},                                           
        {mk_wl(576)},                                           
        {mk_wl(448)},                                           
        {mk_wl(1024)}  
    };

    SmartRanges validIntraTilesChannelsOptions{16, 256, 16, 32};
    testIntraTilesOptions(std::move(tests), std::move(validIntraTilesChannelsOptions));
    // clang-format on
}

TEST_F(VPULayerCostModelTestNPU6x, TestSerializer_PreSplit_KH_CONV_k3x3) {
    const DPUWorkload& wl_k3x3{wl_k3x3_CONV};

    /// Halo definitions for 3x3 kernel ///////////////////////////////////
    const HaloWorkload haloTile1_k3x3{
            {0, 0, 0, 0, 0, 0},         // input_0_halo
            {0, 43, 0, 0, 0, 0},        // output_0_halo
            {0, 3, 0, 0, 0, 0},         // output_0_halo_broadcast_cnt
            {0, -41, 0, 0, 0, 128 * 3}  // output_0_inbound_halo
    };
    const HaloWorkload haloTile2_k3x3{
            {0, 0, 0, 0, 0, 0},             // input_0_halo
            {15, 29, 0, 0, 0, 0},           // output_0_halo
            {1, 2, 0, 0, 0, 0},             // output_0_halo_broadcast_cnt
            {-13, -27, 0, 0, 128, 128 * 2}  // output_0_inbound_halo
    };
    const HaloWorkload haloTile3_k3x3{
            {0, 0, 0, 0, 0, 0},             // input_0_halo
            {29, 15, 0, 0, 0, 0},           // output_0_halo
            {2, 1, 0, 0, 0, 0},             // output_0_halo_broadcast_cnt
            {-27, -13, 0, 0, 128 * 2, 128}  // output_0_inbound_halo
    };
    const HaloWorkload haloTile4_k3x3{
            {0, 0, 0, 0, 0, 0},         // input_0_halo
            {43, 0, 0, 0, 0, 0},        // output_0_halo
            {3, 0, 0, 0, 0, 0},         // output_0_halo_broadcast_cnt
            {-41, 0, 0, 0, 128 * 3, 0}  // output_0_inbound_halo
    };
    ////////////////////////////////////////////////////////////////////////

    ///////////////////////////////// K3x3 HALOS /////////////////////////////////
    DPULayer wl_layer_k3x3(wl_k3x3);

    DPULayer wl_t1_k3x3 = wl_layer_k3x3;
    wl_t1_k3x3.halo = haloTile1_k3x3;

    DPULayer wl_t2_k3x3 = wl_layer_k3x3;
    wl_t2_k3x3.halo = haloTile2_k3x3;

    DPULayer wl_t3_k3x3 = wl_layer_k3x3;
    wl_t3_k3x3.halo = haloTile3_k3x3;

    DPULayer wl_t4_k3x3 = std::move(wl_layer_k3x3);
    wl_t4_k3x3.halo = haloTile4_k3x3;
    ///////////////////////////////////////////////////////////////////////////////

    {
        // 4 positions in std::vector means 4 tiles, each with 1 intra-tile
        const std::vector<std::vector<HaloWorkload>> expected_intratile_halos_k3x3 = {{haloTile1_k3x3},
                                                                                      {haloTile2_k3x3},
                                                                                      {haloTile3_k3x3},
                                                                                      {haloTile4_k3x3}};

        const std::vector<DPULayer> splitLayers_k3x3{std::move(wl_t1_k3x3), std::move(wl_t2_k3x3),
                                                     std::move(wl_t3_k3x3), std::move(wl_t4_k3x3)};
        check_intraTile_halo(std::move(splitLayers_k3x3), std::move(expected_intratile_halos_k3x3), 1);
    }

    /// SOK + output_write_tiles=1 is not supported, SOK forces us to have broadcast
    {
        const DPUWorkload wl_k3x3_SOK{
                VPUDevice::NPU_6_0,
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
                ISIStrategy::SPLIT_OVER_K,                     // isi_strategy
                false,                                         // weight_sparsity_enabled
        };

        DPULayer wl_layer_k3x3_SOK(wl_k3x3_SOK);

        DPULayer wl_t1_k3x3_SOK = wl_layer_k3x3_SOK;
        wl_t1_k3x3_SOK.halo = haloTile1_k3x3;

        DPULayer wl_t2_k3x3_SOK = wl_layer_k3x3_SOK;
        wl_t2_k3x3_SOK.halo = haloTile2_k3x3;

        DPULayer wl_t3_k3x3_SOK = wl_layer_k3x3_SOK;
        wl_t3_k3x3_SOK.halo = haloTile3_k3x3;

        DPULayer wl_t4_k3x3_SOK = std::move(wl_layer_k3x3_SOK);
        wl_t4_k3x3_SOK.halo = haloTile4_k3x3;

        Logger::clear2ndlog();
        CyclesInterfaceType pre_split_cost1{};
        LayerSplitInfo detailed_split_pre_layer{};

        std::vector<DPULayer> tile_wls_SOK{std::move(wl_t1_k3x3_SOK), std::move(wl_t2_k3x3_SOK),
                                           std::move(wl_t3_k3x3_SOK), std::move(wl_t4_k3x3_SOK)};

        ASSERT_NO_THROW(pre_split_cost1 = theModel.LayersPreSplit(tile_wls_SOK, 1U, false, false, prefetch,
                                                                  detailed_split_pre_layer, serializer_layer_uid))
                << toStringLayerSplitInfo(detailed_split_pre_layer);

        EXPECT_TRUE(Cycles::isErrorCode(pre_split_cost1))
                << Cycles::toErrorText(pre_split_cost1) << " " << toStringLayerSplitInfo(detailed_split_pre_layer);
    }
}

TEST_F(VPULayerCostModelTestNPU6x, TestSerializer_PreSplit_KH_CONV_k1x1) {
    const DPUWorkload& wl_k1x1{wl_k1x1_CONV};

    /// Halo definitions for 1x1 kernel/////////////////////////////////////
    const HaloWorkload haloTile1_k1x1{
            {0, 0, 0, 0, 0, 0},         // input_0_halo
            {0, 42, 0, 0, 0, 0},        // output_0_halo
            {0, 3, 0, 0, 0, 0},         // output_0_halo_broadcast_cnt
            {0, -42, 0, 0, 0, 128 * 3}  // output_0_inbound_halo
    };
    const HaloWorkload haloTile2_k1x1{
            {0, 0, 0, 0, 0, 0},             // input_0_halo
            {14, 28, 0, 0, 0, 0},           // output_0_halo
            {1, 2, 0, 0, 0, 0},             // output_0_halo_broadcast_cnt
            {-14, -28, 0, 0, 128, 128 * 2}  // output_0_inbound_halo
    };
    const HaloWorkload haloTile3_k1x1{
            {0, 0, 0, 0, 0, 0},             // input_0_halo
            {28, 14, 0, 0, 0, 0},           // output_0_halo
            {2, 1, 0, 0, 0, 0},             // output_0_halo_broadcast_cnt
            {-28, -14, 0, 0, 128 * 2, 128}  // output_0_inbound_halo
    };
    const HaloWorkload haloTile4_k1x1{
            {0, 0, 0, 0, 0, 0},         // input_0_halo
            {42, 0, 0, 0, 0, 0},        // output_0_halo
            {3, 0, 0, 0, 0, 0},         // output_0_halo_broadcast_cnt
            {-42, 0, 0, 0, 128 * 3, 0}  // output_0_inbound_halo
    };
    //////////////////////////////////////////////////////////////////////////

    ///////////////////////////////// K1x1 HALOS /////////////////////////////////
    DPULayer wl_layer_k1x1(wl_k1x1);

    DPULayer wl_t1_k1x1 = wl_layer_k1x1;
    wl_t1_k1x1.halo = haloTile1_k1x1;

    DPULayer wl_t2_k1x1 = wl_layer_k1x1;
    wl_t2_k1x1.halo = haloTile2_k1x1;

    DPULayer wl_t3_k1x1 = wl_layer_k1x1;
    wl_t3_k1x1.halo = haloTile3_k1x1;

    DPULayer wl_t4_k1x1 = std::move(wl_layer_k1x1);
    wl_t4_k1x1.halo = haloTile4_k1x1;
    ///////////////////////////////////////////////////////////////////////////////

    {
        // 4 positions in std::vector means 4 tiles, each with 1 intra-tile
        const std::vector<std::vector<HaloWorkload>> expected_intratile_halos_k1x1 = {{haloTile1_k1x1},
                                                                                      {haloTile2_k1x1},
                                                                                      {haloTile3_k1x1},
                                                                                      {haloTile4_k1x1}};

        const std::vector<DPULayer> splitLayers_k1x1{std::move(wl_t1_k1x1), std::move(wl_t2_k1x1),
                                                     std::move(wl_t3_k1x1), std::move(wl_t4_k1x1)};
        check_intraTile_halo(std::move(splitLayers_k1x1), std::move(expected_intratile_halos_k1x1), 1);
    }
}

TEST_F(VPULayerCostModelTestNPU6x, TestSerializer_PreSplit_trivial_CONV) {
    const DPUWorkload wl_k3x3 = [&]() {
        DPUWorkload x{wl_k3x3_CONV};
        x.isi_strategy = ISIStrategy::CLUSTERING;
        x.halo = haloZero;
        return x;
    }();

    /// Halo definitions for 3x3 kernel ///////////////////////////////////
    const HaloWorkload haloTile1_k3x3{haloZero};
    const HaloWorkload haloTile2_k3x3{haloZero};
    const HaloWorkload haloTile3_k3x3{haloZero};
    const HaloWorkload haloTile4_k3x3{haloZero};
    ////////////////////////////////////////////////////////////////////////

    ///////////////////////////////// K3x3 HALOS /////////////////////////////////
    DPULayer wl_layer_k3x3(wl_k3x3);

    DPULayer wl_t1_k3x3 = wl_layer_k3x3;
    wl_t1_k3x3.halo = haloZero;

    DPULayer wl_t2_k3x3 = wl_layer_k3x3;
    wl_t2_k3x3.halo = haloZero;

    DPULayer wl_t3_k3x3 = wl_layer_k3x3;
    wl_t3_k3x3.halo = haloZero;

    DPULayer wl_t4_k3x3 = std::move(wl_layer_k3x3);
    wl_t4_k3x3.halo = haloZero;
    ///////////////////////////////////////////////////////////////////////////////

    {
        // 4 positions in std::vector means 4 tiles, each with 1 intra-tile
        const std::vector<std::vector<HaloWorkload>> expected_intratile_halos_k3x3 = {{haloTile1_k3x3},
                                                                                      {haloTile2_k3x3},
                                                                                      {haloTile3_k3x3},
                                                                                      {haloTile4_k3x3}};

        const std::vector<DPULayer> splitLayers_k3x3{std::move(wl_t1_k3x3), std::move(wl_t2_k3x3),
                                                     std::move(wl_t3_k3x3), std::move(wl_t4_k3x3)};
        check_intraTile_halo(std::move(splitLayers_k3x3), std::move(expected_intratile_halos_k3x3), 1);
    }
}

TEST_F(VPULayerCostModelTestNPU6x, TestSerializer_PreSplit_KH_DW_CONV) {
    // tiled workloads
    const DPUWorkload& wl_tile_ref{wl_tile_ref_DWconv};

    ////////////////////// Halo definitions for the 4 tiles //////////////////
    const HaloWorkload haloTile1{
            {0, 0, 0, 0, 0, 0},         // input_0_halo
            {0, 42, 0, 0, 0, 0},        // output_0_halo
            {0, 3, 0, 0, 0, 0},         // output_0_halo_broadcast_cnt
            {0, -42, 0, 0, 0, 128 * 3}  // output_0_inbound_halo
    };
    const HaloWorkload haloTile2{
            {0, 0, 0, 0, 0, 0},             // input_0_halo
            {14, 28, 0, 0, 0, 0},           // output_0_halo
            {1, 2, 0, 0, 0, 0},             // output_0_halo_broadcast_cnt
            {-14, -28, 0, 0, 128, 128 * 2}  // output_0_inbound_halo
    };
    const HaloWorkload haloTile3{
            {0, 0, 0, 0, 0, 0},             // input_0_halo
            {28, 14, 0, 0, 0, 0},           // output_0_halo
            {2, 1, 0, 0, 0, 0},             // output_0_halo_broadcast_cnt
            {-28, -14, 0, 0, 128 * 2, 128}  // output_0_inbound_halo
    };
    const HaloWorkload haloTile4{
            {0, 0, 0, 0, 0, 0},         // input_0_halo
            {42, 0, 0, 0, 0, 0},        // output_0_halo
            {3, 0, 0, 0, 0, 0},         // output_0_halo_broadcast_cnt
            {-42, 0, 0, 0, 128 * 3, 0}  // output_0_inbound_halo
    };

    DPULayer wl_tile_layer(wl_tile_ref);

    DPULayer wl_t1 = wl_tile_layer;
    wl_t1.halo = haloTile1;

    DPULayer wl_t2 = wl_tile_layer;
    wl_t2.halo = haloTile2;

    DPULayer wl_t3 = wl_tile_layer;
    wl_t3.halo = haloTile3;

    DPULayer wl_t4 = std::move(wl_tile_layer);
    wl_t4.halo = haloTile4;
    ////////////////////////////////////////////////////////////////////////

    ////////// Halo for the workloads that results after presplit //////////
    /****************************** TILE 1 ***************************************/
    const HaloWorkload haloTile1_intraT1{
            {0, 0, 0, 0, 0, 0},              // input_0_halo
            {0, 42, 0, 0, 0, 0},             // output_0_halo
            {0, 3, 0, 0, 0, 0},              // output_0_halo_broadcast_cnt
            {0, -42, 0, 0, 0, 64 + 128 * 3}  // output_0_inbound_halo
    };
    const HaloWorkload haloTile1_intraT2{
            {0, 0, 0, 0, 0, 0},          // input_0_halo
            {0, 42, 0, 0, 0, 0},         // output_0_halo
            {0, 3, 0, 0, 0, 0},          // output_0_halo_broadcast_cnt
            {0, -42, 0, 0, 64, 128 * 3}  // output_0_inbound_halo
    };

    /****************************** TILE 2 ***************************************/
    const HaloWorkload haloTile2_intraT1{
            {0, 0, 0, 0, 0, 0},                  // input_0_halo
            {14, 28, 0, 0, 0, 0},                // output_0_halo
            {1, 2, 0, 0, 0, 0},                  // output_0_halo_broadcast_cnt
            {-14, -28, 0, 0, 128, 128 * 2 + 64}  // output_0_inbound_halo
    };
    const HaloWorkload haloTile2_intraT2{
            {0, 0, 0, 0, 0, 0},                  // input_0_halo
            {14, 28, 0, 0, 0, 0},                // output_0_halo
            {1, 2, 0, 0, 0, 0},                  // output_0_halo_broadcast_cnt
            {-14, -28, 0, 0, 128 + 64, 128 * 2}  // output_0_inbound_halo
    };

    /****************************** TILE 3 ***************************************/
    const HaloWorkload haloTile3_intraT1{
            {0, 0, 0, 0, 0, 0},                  // input_0_halo
            {28, 14, 0, 0, 0, 0},                // output_0_halo
            {2, 1, 0, 0, 0, 0},                  // output_0_halo_broadcast_cnt
            {-28, -14, 0, 0, 128 * 2, 128 + 64}  // output_0_inbound_halo
    };
    const HaloWorkload haloTile3_intraT2{
            {0, 0, 0, 0, 0, 0},                  // input_0_halo
            {28, 14, 0, 0, 0, 0},                // output_0_halo
            {2, 1, 0, 0, 0, 0},                  // output_0_halo_broadcast_cnt
            {-28, -14, 0, 0, 128 * 2 + 64, 128}  // output_0_inbound_halo
    };

    /****************************** TILE 4 ***************************************/
    const HaloWorkload haloTile4_intraT1{
            {0, 0, 0, 0, 0, 0},          // input_0_halo
            {42, 0, 0, 0, 0, 0},         // output_0_halo
            {3, 0, 0, 0, 0, 0},          // output_0_halo_broadcast_cnt
            {-42, 0, 0, 0, 128 * 3, 64}  // output_0_inbound_halo
    };
    const HaloWorkload haloTile4_intraT2{
            {0, 0, 0, 0, 0, 0},              // input_0_halo
            {42, 0, 0, 0, 0, 0},             // output_0_halo
            {3, 0, 0, 0, 0, 0},              // output_0_halo_broadcast_cnt
            {-42, 0, 0, 0, 128 * 3 + 64, 0}  // output_0_inbound_halo
    };

    ////////////////////////////////////////////////////////////////////////

    // 4 positions in std::vector means 4 tiles, each with 2 intra-tiles
    // Expected halo pair for one tile: first and second candidate intra-tile workloads.
    // - first position in from std::array expected HaloWorkload for candidate #0 (first candidate) of this tile
    // - second position from std::array expected HaloWorkload for candidate #1 (second candidate) of this tile
    std::vector<std::vector<HaloWorkload>> expected_intratile_halos = {{haloTile1_intraT1, haloTile1_intraT2},
                                                                       {haloTile2_intraT1, haloTile2_intraT2},
                                                                       {haloTile3_intraT1, haloTile3_intraT2},
                                                                       {haloTile4_intraT1, haloTile4_intraT2}};

    {
        std::vector<DPULayer> tile_wls{std::move(wl_t1), std::move(wl_t2), std::move(wl_t3), std::move(wl_t4)};
        check_intraTile_halo(std::move(tile_wls), std::move(expected_intratile_halos), 2);
    }

    /// SOK + output_write_tiles=1 is not supported, SOK forces us to have broadcast
    {
        const DPUWorkload wl_tile_ref_SOK{
                VPUDevice::NPU_6_0,
                Operation::DW_CONVOLUTION,
                {VPUTensor(56, 56, 128, 1, DataType::UINT8)},  // input dimensions
                {VPUTensor(56, 56, 128, 1, DataType::UINT8)},  // output dimensions
                {1, 1},                                        // kernels
                {1, 1},                                        // strides
                {0, 0, 0, 0},                                  // padding
                ExecutionMode::CUBOID_16x16,                   // execution mode
                ActivationFunction::NONE,                      // activation
                0.0F,                                          // act_sparsity
                0.0F,                                          // weight_sparsity
                {swz_def, swz_def},                            // input_swizzling
                {swz_def},                                     // output_swizzling
                1,                                             // output_write_tiles
                {0, 0, 0, 0},                                  // offsets
                ISIStrategy::SPLIT_OVER_K,                     // isi_strategy
                false,                                         // weight_sparsity_enabled
        };

        DPULayer wl_tile_layer_SOK(wl_tile_ref_SOK);

        DPULayer wl_t1_SOK = wl_tile_layer_SOK;
        wl_t1_SOK.halo = haloTile1;

        DPULayer wl_t2_SOK = wl_tile_layer_SOK;
        wl_t2_SOK.halo = haloTile2;

        DPULayer wl_t3_SOK = wl_tile_layer_SOK;
        wl_t3_SOK.halo = haloTile3;

        DPULayer wl_t4_SOK = std::move(wl_tile_layer_SOK);
        wl_t4_SOK.halo = haloTile4;

        Logger::clear2ndlog();
        CyclesInterfaceType pre_split_cost1{};
        LayerSplitInfo detailed_split_pre_layer{};

        std::vector<DPULayer> tile_wls_SOK{std::move(wl_t1_SOK), std::move(wl_t2_SOK), std::move(wl_t3_SOK),
                                           std::move(wl_t4_SOK)};

        ASSERT_NO_THROW(pre_split_cost1 = theModel.LayersPreSplit(tile_wls_SOK, 1U, false, false, prefetch,
                                                                  detailed_split_pre_layer, serializer_layer_uid))
                << toStringLayerSplitInfo(detailed_split_pre_layer);

        EXPECT_TRUE(Cycles::isErrorCode(pre_split_cost1))
                << Cycles::toErrorText(pre_split_cost1) << " " << toStringLayerSplitInfo(detailed_split_pre_layer);
    }
}

TEST_F(VPULayerCostModelTestNPU6x, TestSerializer_PreSplit_trivial_DW_CONV) {
    // tiled workloads
    const DPUWorkload wl_tile_ref = [&]() {
        DPUWorkload x{wl_tile_ref_DWconv};
        x.isi_strategy = ISIStrategy::CLUSTERING;
        x.halo = haloZero;
        return x;
    }();

    ////////////////////// Halo definitions for the 4 tiles //////////////////
    const HaloWorkload haloTile1{haloZero};
    const HaloWorkload haloTile2{haloZero};
    const HaloWorkload haloTile3{haloZero};
    const HaloWorkload haloTile4{haloZero};

    DPULayer wl_tile_layer(wl_tile_ref);

    DPULayer wl_t1 = wl_tile_layer;
    wl_t1.halo = haloTile1;

    DPULayer wl_t2 = wl_tile_layer;
    wl_t2.halo = haloTile2;

    DPULayer wl_t3 = wl_tile_layer;
    wl_t3.halo = haloTile3;

    DPULayer wl_t4 = std::move(wl_tile_layer);
    wl_t4.halo = haloTile4;
    ////////////////////////////////////////////////////////////////////////

    ////////// Halo for the workloads that results after presplit //////////
    const HaloWorkload haloTile1_intraT1{
            {0, 0, 0, 0, 0, 0},  // input_0_halo
            {0, 0, 0, 0, 0, 0},  // output_0_halo
            {0, 0, 0, 0, 0, 0},  // output_0_halo_broadcast_cnt
            {0, 0, 0, 0, 0, 64}  // output_0_inbound_halo
    };
    const HaloWorkload haloTile1_intraT2{
            {0, 0, 0, 0, 0, 0},  // input_0_halo
            {0, 0, 0, 0, 0, 0},  // output_0_halo
            {0, 0, 0, 0, 0, 0},  // output_0_halo_broadcast_cnt
            {0, 0, 0, 0, 64, 0}  // output_0_inbound_halo
    };

    const HaloWorkload haloTile2_intraT1{haloTile1_intraT1

    };
    const HaloWorkload haloTile2_intraT2{haloTile1_intraT2};

    const HaloWorkload haloTile3_intraT1{haloTile1_intraT1};
    const HaloWorkload haloTile3_intraT2{haloTile1_intraT2};

    const HaloWorkload haloTile4_intraT1{haloTile1_intraT1};
    const HaloWorkload haloTile4_intraT2{haloTile1_intraT2};

    ////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////////
    // 4 positions in std::vector means 4 tiles, each with 2 intra-tiles
    // Expected halo pair for one tile: first and second candidate intra-tile workloads.
    // - first position in from std::array expected HaloWorkload for candidate #0 (first candidate) of this tile
    // - second position from std::array expected HaloWorkload for candidate #1 (second candidate) of this tile
    std::vector<std::vector<HaloWorkload>> expected_intratile_halos = {{haloTile1_intraT1, haloTile1_intraT2},
                                                                       {haloTile2_intraT1, haloTile2_intraT2},
                                                                       {haloTile3_intraT1, haloTile3_intraT2},
                                                                       {haloTile4_intraT1, haloTile4_intraT2}};

    {
        std::vector<DPULayer> tile_wls{std::move(wl_t1), std::move(wl_t2), std::move(wl_t3), std::move(wl_t4)};
        check_intraTile_halo(std::move(tile_wls), std::move(expected_intratile_halos), 2);
    }
}

TEST_F(VPULayerCostModelTestNPU6x, Test_HaloCompute_DW_CONV_4tiles_4intraTiles) {
    // tiled workloads
    const DPUWorkload wl_tile_ref{
            VPUDevice::NPU_6_0,
            Operation::DW_CONVOLUTION,
            {VPUTensor(56, 56, 128, 1, DataType::UINT8)},  // input dimensions
            {VPUTensor(56, 56, 128, 1, DataType::UINT8)},  // output dimensions
            {1, 1},                                        // kernels
            {1, 1},                                        // strides
            {0, 0, 0, 0},                                  // padding
            ExecutionMode::CUBOID_16x16,                   // execution mode
            ActivationFunction::NONE,                      // activation
            0.0F,                                          // act_sparsity
            0.0F,                                          // weight_sparsity
            {swz_def, swz_def},                            // input_swizzling
            {swz_def},                                     // output_swizzling
            1,                                             // output_write_tiles
            {0, 0, 0, 0},                                  // offsets
            ISIStrategy::SPLIT_OVER_K_nxtHW,               // isi_strategy
            false,                                         // weight_sparsity_enabled
    };

    ////////////////////// Halo definitions for the 4 tiles //////////////////
    const HaloWorkload haloTile1{
            {0, 0, 0, 0, 0, 0},         // input_0_halo
            {0, 42, 0, 0, 0, 0},        // output_0_halo
            {0, 3, 0, 0, 0, 0},         // output_0_halo_broadcast_cnt
            {0, -42, 0, 0, 0, 128 * 3}  // output_0_inbound_halo
    };
    const HaloWorkload haloTile2{
            {0, 0, 0, 0, 0, 0},             // input_0_halo
            {14, 28, 0, 0, 0, 0},           // output_0_halo
            {1, 2, 0, 0, 0, 0},             // output_0_halo_broadcast_cnt
            {-14, -28, 0, 0, 128, 128 * 2}  // output_0_inbound_halo
    };
    const HaloWorkload haloTile3{
            {0, 0, 0, 0, 0, 0},             // input_0_halo
            {28, 14, 0, 0, 0, 0},           // output_0_halo
            {2, 1, 0, 0, 0, 0},             // output_0_halo_broadcast_cnt
            {-28, -14, 0, 0, 128 * 2, 128}  // output_0_inbound_halo
    };
    const HaloWorkload haloTile4{
            {0, 0, 0, 0, 0, 0},         // input_0_halo
            {42, 0, 0, 0, 0, 0},        // output_0_halo
            {3, 0, 0, 0, 0, 0},         // output_0_halo_broadcast_cnt
            {-42, 0, 0, 0, 128 * 3, 0}  // output_0_inbound_halo
    };

    DPULayer wl_tile_layer(wl_tile_ref);

    DPULayer wl_t1 = wl_tile_layer;
    wl_t1.halo = haloTile1;

    DPULayer wl_t2 = wl_tile_layer;
    wl_t2.halo = haloTile2;

    DPULayer wl_t3 = wl_tile_layer;
    wl_t3.halo = haloTile3;

    DPULayer wl_t4 = std::move(wl_tile_layer);
    wl_t4.halo = haloTile4;
    ////////////////////////////////////////////////////////////////////////

    ////////// Halo for the workloads that results after presplit //////////
    // Each tile (128 ch) is split by ZEffTiling DP into 2 x 64-channel workloads.
    // F/B values in output_inbound_halo reflect global channel offsets across all 8 workloads:
    //   workload k (0-based) has F = k*64, B = (7-k)*64
    /****************************** TILE 1 ***************************************/
    // Old 4x32 expectations (replaced by 2x64 split):
    // const HaloWorkload haloTile1_intraT1{ {0,0,0,0,0,0},{0,42,0,0,0,0},{0,3,0,0,0,0},{0,-42,0,0,0,32*3+128*3} };
    // const HaloWorkload haloTile1_intraT2{ {0,0,0,0,0,0},{0,42,0,0,0,0},{0,3,0,0,0,0},{0,-42,0,0,32,32*2+128*3} };
    // const HaloWorkload haloTile1_intraT3{ {0,0,0,0,0,0},{0,42,0,0,0,0},{0,3,0,0,0,0},{0,-42,0,0,32*2,32+128*3} };
    // const HaloWorkload haloTile1_intraT4{ {0,0,0,0,0,0},{0,42,0,0,0,0},{0,3,0,0,0,0},{0,-42,0,0,32*3,128*3} };
    const HaloWorkload haloTile1_intraT1{
            {0, 0, 0, 0, 0, 0},        // input_0_halo
            {0, 42, 0, 0, 0, 0},       // output_0_halo
            {0, 3, 0, 0, 0, 0},        // output_0_halo_broadcast_cnt
            {0, -42, 0, 0, 0, 64 * 7}  // output_0_inbound_halo  (wl0: F=0, B=7*64=448)
    };
    const HaloWorkload haloTile1_intraT2{
            {0, 0, 0, 0, 0, 0},         // input_0_halo
            {0, 42, 0, 0, 0, 0},        // output_0_halo
            {0, 3, 0, 0, 0, 0},         // output_0_halo_broadcast_cnt
            {0, -42, 0, 0, 64, 64 * 6}  // output_0_inbound_halo  (wl1: F=1*64=64, B=6*64=384)
    };

    /****************************** TILE 2 ***************************************/
    // Old 4x32 expectations (replaced by 2x64 split):
    // const HaloWorkload haloTile2_intraT1{ {0,0,0,0,0,0},{14,28,0,0,0,0},{1,2,0,0,0,0},{-14,-28,0,0,128,128*2+32*3} };
    // const HaloWorkload haloTile2_intraT2{ {0,0,0,0,0,0},{14,28,0,0,0,0},{1,2,0,0,0,0},{-14,-28,0,0,128+32,128*2+32*2}
    // }; const HaloWorkload haloTile2_intraT3{
    // {0,0,0,0,0,0},{14,28,0,0,0,0},{1,2,0,0,0,0},{-14,-28,0,0,128+32*2,128*2+32} }; const HaloWorkload
    // haloTile2_intraT4{ {0,0,0,0,0,0},{14,28,0,0,0,0},{1,2,0,0,0,0},{-14,-28,0,0,128+32*3,128*2} };
    const HaloWorkload haloTile2_intraT1{
            {0, 0, 0, 0, 0, 0},            // input_0_halo
            {14, 28, 0, 0, 0, 0},          // output_0_halo
            {1, 2, 0, 0, 0, 0},            // output_0_halo_broadcast_cnt
            {-14, -28, 0, 0, 128, 64 * 5}  // output_0_inbound_halo  (wl2: F=2*64=128, B=5*64=320)
    };
    const HaloWorkload haloTile2_intraT2{
            {0, 0, 0, 0, 0, 0},            // input_0_halo
            {14, 28, 0, 0, 0, 0},          // output_0_halo
            {1, 2, 0, 0, 0, 0},            // output_0_halo_broadcast_cnt
            {-14, -28, 0, 0, 192, 64 * 4}  // output_0_inbound_halo  (wl3: F=3*64=192, B=4*64=256)
    };

    /****************************** TILE 3 ***************************************/
    // Old 4x32 expectations (replaced by 2x64 split):
    // const HaloWorkload haloTile3_intraT1{ {0,0,0,0,0,0},{28,14,0,0,0,0},{2,1,0,0,0,0},{-28,-14,0,0,128*2,128+32*3} };
    // const HaloWorkload haloTile3_intraT2{ {0,0,0,0,0,0},{28,14,0,0,0,0},{2,1,0,0,0,0},{-28,-14,0,0,128*2+32,128+32*2}
    // }; const HaloWorkload haloTile3_intraT3{
    // {0,0,0,0,0,0},{28,14,0,0,0,0},{2,1,0,0,0,0},{-28,-14,0,0,128*2+32*2,128+32} }; const HaloWorkload
    // haloTile3_intraT4{ {0,0,0,0,0,0},{28,14,0,0,0,0},{2,1,0,0,0,0},{-28,-14,0,0,128*2+32*3,128} };
    const HaloWorkload haloTile3_intraT1{
            {0, 0, 0, 0, 0, 0},            // input_0_halo
            {28, 14, 0, 0, 0, 0},          // output_0_halo
            {2, 1, 0, 0, 0, 0},            // output_0_halo_broadcast_cnt
            {-28, -14, 0, 0, 256, 64 * 3}  // output_0_inbound_halo  (wl4: F=4*64=256, B=3*64=192)
    };
    const HaloWorkload haloTile3_intraT2{
            {0, 0, 0, 0, 0, 0},            // input_0_halo
            {28, 14, 0, 0, 0, 0},          // output_0_halo
            {2, 1, 0, 0, 0, 0},            // output_0_halo_broadcast_cnt
            {-28, -14, 0, 0, 320, 64 * 2}  // output_0_inbound_halo  (wl5: F=5*64=320, B=2*64=128)
    };

    /****************************** TILE 4 ***************************************/
    // Old 4x32 expectations (replaced by 2x64 split):
    // const HaloWorkload haloTile4_intraT1{ {0,0,0,0,0,0},{42,0,0,0,0,0},{3,0,0,0,0,0},{-42,0,0,0,128*3,32*3} };
    // const HaloWorkload haloTile4_intraT2{ {0,0,0,0,0,0},{42,0,0,0,0,0},{3,0,0,0,0,0},{-42,0,0,0,128*3+32,32*2} };
    // const HaloWorkload haloTile4_intraT3{ {0,0,0,0,0,0},{42,0,0,0,0,0},{3,0,0,0,0,0},{-42,0,0,0,128*3+32*2,32} };
    // const HaloWorkload haloTile4_intraT4{ {0,0,0,0,0,0},{42,0,0,0,0,0},{3,0,0,0,0,0},{-42,0,0,0,128*3+32*3,0} };
    const HaloWorkload haloTile4_intraT1{
            {0, 0, 0, 0, 0, 0},          // input_0_halo
            {42, 0, 0, 0, 0, 0},         // output_0_halo
            {3, 0, 0, 0, 0, 0},          // output_0_halo_broadcast_cnt
            {-42, 0, 0, 0, 384, 64 * 1}  // output_0_inbound_halo  (wl6: F=6*64=384, B=1*64=64)
    };
    const HaloWorkload haloTile4_intraT2{
            {0, 0, 0, 0, 0, 0},     // input_0_halo
            {42, 0, 0, 0, 0, 0},    // output_0_halo
            {3, 0, 0, 0, 0, 0},     // output_0_halo_broadcast_cnt
            {-42, 0, 0, 0, 448, 0}  // output_0_inbound_halo  (wl7: F=7*64=448, B=0)
    };

    ///////////////////////////////////////////////////////////////////////////////

    // 4 tiles, each split by ZEffTiling DP into 2 x 64-channel workloads.
    // Each std::array holds the 2 expected intra-tile halo values for that tile.
    std::vector<std::array<HaloWorkload, 2>> expected_intratile_halos = {{haloTile1_intraT1, haloTile1_intraT2},
                                                                         {haloTile2_intraT1, haloTile2_intraT2},
                                                                         {haloTile3_intraT1, haloTile3_intraT2},
                                                                         {haloTile4_intraT1, haloTile4_intraT2}};

    // Lambda that runs LayersPreSplit and validates halos.
    // We pass detailed_split_pre_layer to LayersPreSplit so tiler will populate all_intra_tile_splits.
    // We ASSERT sizes before indexing to avoid UB and provide clear failure messages.
    // We compare the halo of the first workload of each candidate
    auto check_intraTile_halo_ = [&](std::vector<DPULayer> splitLayers) {
        // L2 pre split
        Logger::clear2ndlog();
        CyclesInterfaceType pre_split_cost1{};
        LayerSplitInfo detailed_split_pre_layer{};

        ASSERT_NO_THROW(pre_split_cost1 = theModel.LayersPreSplit(splitLayers, 1U, false, false, prefetch,
                                                                  detailed_split_pre_layer, serializer_layer_uid))
                << toStringLayerSplitInfo(detailed_split_pre_layer);

        EXPECT_FALSE(Cycles::isErrorCode(pre_split_cost1))
                << pre_split_cost1 << " " << toStringLayerSplitInfo(detailed_split_pre_layer);

        // Sanity: expected count of tiles must match
        ASSERT_EQ(detailed_split_pre_layer.size(), expected_intratile_halos.size())
                << "Tile count mismatch. Dump:\n"
                << toStringLayerSplitInfo(detailed_split_pre_layer);

        int tile_index = 0;

        // for each tile we compare the halos of its intra-tile workloads
        for (const auto& split : detailed_split_pre_layer) {
            // DW_CONVOLUTION on NPU_6_0 has only one valid tiling execution mode
            // (CUBOID_16x16), so generateSplitsForEfficiencyBased produces exactly
            // one split candidate (index 0). Accessing [2] would be out-of-bounds.
            ASSERT_FALSE(split.all_intra_tile_splits.empty())
                    << "Expected at least 1 intra-tile split candidate but got 0";

            // The DP-optimal split for 128 channels with CUBOID_16x16 is 2x64 channels.
            ASSERT_GE(split.all_intra_tile_splits[0].workloads.size(), 2u)
                    << "Expected 2 workloads in split candidate [0] but got "
                    << split.all_intra_tile_splits[0].workloads.size();

            EXPECT_EQ(split.all_intra_tile_splits[0].workloads.size(), 2);

            // compare halos
            for (int i = 0; i < 2; i++) {
                EXPECT_EQ(split.all_intra_tile_splits[0].workloads[i].halo, expected_intratile_halos[tile_index][i])
                        << "Tile: " << tile_index + 1 << " Intra Tile: " << i << " "
                        << split.all_intra_tile_splits[0].workloads[i];
            }

            tile_index++;
        }
    };

    {
        std::vector<DPULayer> tile_wls{std::move(wl_t1), std::move(wl_t2), std::move(wl_t3), std::move(wl_t4)};
        check_intraTile_halo_(std::move(tile_wls));
    }
}

TEST_F(VPULayerCostModelTestNPU6x, Error_Invalid_Input_Config_Investigation_Test) {
    const bool force_fail{false};        // controls force failing assertion
    const int fail{force_fail ? 0 : 1};  // 1 neutral, 0 fail
    show_split = false;
    // EXPECT_TRUE(false);

    DPUWorkload wl1{
            VPUDevice::NPU_6_0,
            Operation::CONVOLUTION,
            {VPUTensor(150, 38, 12, 1, DataType::UINT8)},  // input dimensions
            {VPUTensor(150, 38, 80, 1, DataType::UINT8)},  // output dimensions
            {1, 1},                                        // kernels
            {1, 1},                                        // strides
            {0, 0, 0, 0},                                  // padding
            ExecutionMode::CUBOID_16x16,                   // execution mode
            ActivationFunction::RELU,                      // activation
            0.0F,                                          // act_sparsity
            0.0F,                                          // weight_sparsity
            {Swizzling::KEY_0, Swizzling::KEY_0},          // input_swizzling
            {Swizzling::KEY_0},                            // output_swizzling
            1,                                             // output_write_tiles
            {0, 0, 0, 0},                                  // offsets
            ISIStrategy::CLUSTERING,                       // isi_strategy
            false,                                         // weight_sparsity_enabled
    };
    wl1.input_autopad = true;
    {
        const DPULayer tst_layer1(wl1);
        const std::vector<TestCase> tests{
                {{std::move(tst_layer1),
                  VPULayerStrategy{1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 250, fail * 250 + 150},
                 "SOHO , no memmove, CONV "},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTestNPU6x, SEP_NoSparseMap_Toggles_ErrorInputTooBig) {
    // This test isolates the impact of sep_activators.no_sparse_map on validation outcome for an otherwise identical
    // workload.
    //
    // What is expected:
    // - no_sparse_map = false, the model should reject the workload as too large , with ERROR_INPUT_TOO_BIG
    // - no_sparse_map = true, the same configuration should be accepted (return a valid cycles value),
    //   therefore we only assert "not an error code" and deliberately avoid checking a cycles range.
    show_split = false;

    DPUWorkload wl_base{
            VPUDevice::NPU_6_0,
            Operation::CONVOLUTION,
            {VPUTensor(64, 64, 640, 1, DataType::FLOAT16)},  // input dimensions
            {VPUTensor(64, 64, 64, 1, DataType::FLOAT16)},   // output dimensions
            {1, 1},                                          // kernels
            {1, 1},                                          // strides
            {0, 0, 0, 0},                                    // padding
            ExecutionMode::CUBOID_16x16,                     // execution mode
            ActivationFunction::NONE,                        // activation
            0.0F,                                            // act_sparsity
            0.0F,                                            // weight_sparsity
            {Swizzling::KEY_0, Swizzling::KEY_0},            // input_swizzling
            {Swizzling::KEY_0},                              // output_swizzling
            1,                                               // output_write_tiles
            {0, 0, 0, 0},                                    // offsets
            ISIStrategy::CLUSTERING,                         // isi_strategy
            false,                                           // weight_sparsity_enabled
            haloZero,
            {
                    true,            // sep on
                    {64, 64, 1, 1},  // sep table
                    {32, 32, 640, 1},
                    false,  // no_sparse_map (toggled below)
            },
    };

    const VPULayerStrategy strategy{1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOK, false, false, prefetch};

    // toggles sep_activators.no_sparse_map and asserts either the exact error code or that the result is not an error
    // code
    auto run_case = [&](const bool no_sparse_map, const bool expect_error,
                        const CyclesInterfaceType expected_error_code, const std::string_view case_name) {
        DPUWorkload wl{wl_base};
        wl.sep_activators.no_sparse_map = no_sparse_map;

        DPULayer layer{wl};
        Logger::clear2ndlog();
        LayerSplitInfo detailed_split{};
        CyclesInterfaceType cost{};

        ASSERT_NO_THROW(cost = theModel.Layer(layer, strategy, detailed_split)) << case_name << "\n"
                                                                                << layer << strategy;

        if (expect_error) {
            EXPECT_TRUE(Cycles::isErrorCode(cost)) << case_name << "\nExpected error code, got cycles: " << cost << "\n"
                                                   << layer << strategy << "\n 2ndLOG:\n"
                                                   << Logger::get2ndlog();

            EXPECT_EQ(cost, expected_error_code)
                    << case_name << "\nUnexpected error code: " << cost << " : " << Cycles::toErrorText(cost) << "\n"
                    << layer << strategy << "\n 2ndLOG:\n"
                    << Logger::get2ndlog();
        } else {
            EXPECT_FALSE(Cycles::isErrorCode(cost)) << case_name << "\nExpected non-error cycles, got error: " << cost
                                                    << " : " << Cycles::toErrorText(cost) << "\n"
                                                    << layer << strategy << "\n 2ndLOG:\n"
                                                    << Logger::get2ndlog();
        }
    };

    // Run both scenarios on the same base workload, only sep_activators.no_sparse_map is toggled, and also the expected
    // error outcome (marked by expect_error and expected_error_code)
    run_case(false, true, VPUNN::Cycles::ERROR_INPUT_TOO_BIG, "sep.no_sparse_map=false => ERROR_INPUT_TOO_BIG");
    run_case(true, false, VPUNN::Cycles::NO_ERROR, "sep.no_sparse_map=true => non-error cycles");
}
/// investigation, cost model does not return INVALID_INPUT_CONFIGURATION error for this test, but the cost from NN is
/// very small, cost = 2
/// this test could be very fragile due to cost, because costs<0 are considered invalid, so NN could any time return an
/// error here
TEST_F(VPULayerCostModelTestNPU6x, MAXPOOL_Error_Invalid_Input_Config_Investigation_Test) {
    // const bool force_fail{false};        // controls force failing assertion
    // const int fail{force_fail ? 0 : 1};  // 1 neutral, 0 fail
    //  show_split = false;
    // EXPECT_TRUE(false);

    DPUWorkload wl1{
            VPUDevice::NPU_6_0,
            Operation::MAXPOOL,
            {VPUTensor(2, 32, 16, 1, DataType::UINT8, Layout::ZXY)},  // input dimensions
            {VPUTensor(2, 32, 16, 1, DataType::UINT8, Layout::YXZ)},  // output dimensions
            {1, 1},                                                   // kernels
            {1, 1},                                                   // strides
            {0, 0, 0, 0},                                             // padding
            ExecutionMode::CUBOID_16x16,                              // execution mode
            ActivationFunction::NONE,                                 // activation
            0.0F,                                                     // act_sparsity
            0.0F,                                                     // weight_sparsity
            {Swizzling::KEY_0, Swizzling::KEY_0},                     // input_swizzling
            {Swizzling::KEY_0},                                       // output_swizzling
            1,                                                        // output_write_tiles
            {0, 0, 0, 0},                                             // offsets
            ISIStrategy::CLUSTERING,                                  // isi_strategy
            false,                                                    // weight_sparsity_enabled
    };

    //{
    //    const DPULayer tst_layer1(wl1);
    //    const std::vector<TestCase> tests{
    //            {{std::move(tst_layer1), {1U, 1U, 4U, VPUNN::VPUTilingStrategy::NONE, false, false, prefetch}},
    //             {VPUNN::Cycles::NO_ERROR, true, 250, fail * 250 + 150},
    //             "SOHO , no memmove, CONV "},
    //    };

    //    executeTests(tests);
    //}

    // infrastructure
    const bool prefetch_{true};  // prefetch was done
    VPULayerCostModel& theModel_ = layer_models.getModel(VPUDevice::NPU_6_0);
    // const LayersValidation layer_validator;
    // size_t serializer_layer_uid = 123;

    // L2 pre split
    Logger::clear2ndlog();
    CyclesInterfaceType pre_split_cost1{};
    LayerSplitInfo detailed_split_pre_layer{};
    DPULayer wl1_layer(wl1);
    const std::vector<DPULayer> splitLayers{wl1_layer, wl1_layer, wl1_layer, wl1_layer};
    ASSERT_NO_THROW(pre_split_cost1 = theModel_.LayersPreSplit(splitLayers, 1U, false, false, prefetch_,
                                                               detailed_split_pre_layer, serializer_layer_uid))
            << toStringLayerSplitInfo(detailed_split_pre_layer);

    EXPECT_FALSE(Cycles::isErrorCode(pre_split_cost1))
            << Cycles::toErrorText(pre_split_cost1) << " Cost: " << pre_split_cost1 << " "
            << toStringLayerSplitInfo(detailed_split_pre_layer);  // NN cost 2
}

TEST_F(VPULayerCostModelTestNPU6x, TestPreSplit_AVEPOOL_NPU6_Clustering_4tiles) {
    // AVEPOOL workload - NPU_6_0, 4 identical tiles, Clustering ISI strategy, output_autopad enabled
    // Reproduces a case logged during [split-NCE-ops-onto-workloads] with the following config:
    //   input {10,2,16,1} FLOAT16 ZXY -> output {1,2,8,1} FLOAT16 XYZ
    //   kernels {10,1}, strides {1,1}, padding {0,0,0,0}, CUBOID_16x16, superdense_memory=true
    const DPUWorkload wl{
            VPUDevice::NPU_6_0,
            Operation::AVEPOOL,
            {VPUTensor(10, 2, 16, 1, DataType::FLOAT16, Layout::ZXY)},       // input dimensions
            {VPUTensor(1, 2, 8, 1, DataType::FLOAT16, Layout::XYZ)},         // output dimensions
            {10, 1},                                                         // kernels
            {1, 1},                                                          // strides
            {0, 0, 0, 0},                                                    // padding
            ExecutionMode::CUBOID_16x16,                                     // execution mode
            ActivationFunction::NONE,                                        // activation
            0.0F,                                                            // act_sparsity
            0.0F,                                                            // weight_sparsity
            {Swizzling::KEY_0, Swizzling::KEY_0},                            // input_swizzling
            {Swizzling::KEY_0},                                              // output_swizzling
            1,                                                               // output_write_tiles
            {0, 0, 0, 0},                                                    // offsets
            ISIStrategy::CLUSTERING,                                         // isi_strategy
            false,                                                           // weight_sparsity_enabled
            {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects
            {false, {0, 0, 0, 0}, {0, 0, 0, 0}, false},                      // SEP
            std::optional<DataType>{},                                       // input1 data type (opt)
            "",                                                              // layer_info
            std::optional<bool>{},                                           // weightless_operation (opt)
            std::optional<bool>{},                                           // in_place_output_memory (opt)
            std::optional<bool>{true},                                       // superdense_memory (opt)
            std::optional<bool>{},                                           // input_autopad (opt)
            std::optional<bool>{true},                                       // output_autopad (opt)
    };

    // L2 pre split - 4 identical tiles (Clustering)
    Logger::clear2ndlog();
    CyclesInterfaceType pre_split_cost{};
    LayerSplitInfo detailed_split_out{};
    const std::vector<DPULayer> splitLayers{DPULayer(wl), DPULayer(wl), DPULayer(wl), DPULayer(wl)};
    ASSERT_NO_THROW(pre_split_cost = theModel.LayersPreSplit(splitLayers, 1U, false, false, prefetch,
                                                             detailed_split_out, serializer_layer_uid))
            << toStringLayerSplitInfo(detailed_split_out);

    ASSERT_FALSE(Cycles::isErrorCode(pre_split_cost))
            << Cycles::toErrorText(pre_split_cost) << " " << toStringLayerSplitInfo(detailed_split_out)
            << Logger::get2ndlog() << "END ERR";

    EXPECT_EQ(detailed_split_out.size(), 4U) << "Expected 4 tiles in pre-split output";
}

// Tests for use_StridedMathModelForDMA flag behaviour

/// When no DMA NN provider is available the flag set via set_DMATheoreticalMathModel_enabled() must
/// change the DMA cost (switches from classic DMAWorkload-based model to VPUDMADescriptor-based
/// strided-math model), so the overall Layer cost with input_in_ddr=true must differ.
TEST_F(VPULayerCostModelTestNPU6x, StridedMathFlag_NoDMAProvider_ProducesDifferentCost) {
    // Use NPU_6_0 because this fixture targets NPU 6.x and the theoretical DMA models are device-agnostic,
    // making the flag effect easy to observe.
    const DPUWorkload wl_base{
            VPUDevice::NPU_6_0,
            Operation::CONVOLUTION,
            {VPUTensor(56, 56, 64, 1, DataType::UINT8)},
            {VPUTensor(56, 56, 64, 1, DataType::UINT8)},
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

    // Two independent models, both without a DMA NN provider (theoretical fallback).
    VPULayerCostModel model_classic{VPUDevice::NPU_6_0, NPU_6_0_MODEL_PATH};  // default: classic DMAWorkload model
    VPULayerCostModel model_strided{VPUDevice::NPU_6_0, NPU_6_0_MODEL_PATH};  // will have strided-math model enabled
    model_strided.set_DMATheoreticalMathModel_enabled();

    ASSERT_FALSE(model_classic.is_DMATheoreticalMathModel_enabled());
    ASSERT_TRUE(model_strided.is_DMATheoreticalMathModel_enabled());

    DPULayer layer_classic{wl_base};
    DPULayer layer_strided{wl_base};

    // input_in_ddr=true forces a DMA cost to be computed and added to the total.
    const CyclesInterfaceType cost_classic =
            model_classic.Layer(layer_classic, VPUTilingStrategy::NONE, 1U, 1U, /*input_in_ddr=*/true);
    const CyclesInterfaceType cost_strided =
            model_strided.Layer(layer_strided, VPUTilingStrategy::NONE, 1U, 1U, /*input_in_ddr=*/true);

    EXPECT_FALSE(Cycles::isErrorCode(cost_classic)) << Cycles::toErrorText(cost_classic);
    EXPECT_FALSE(Cycles::isErrorCode(cost_strided)) << Cycles::toErrorText(cost_strided);

    // The two theoretical models produce different DMA estimates, so the totals must differ.
    EXPECT_NE(cost_classic, cost_strided)
            << "Expected different costs when use_StridedMathModelForDMA flag differs (no DMA provider). "
            << "classic=" << cost_classic << " strided=" << cost_strided;
}

/// When a valid DMA NN provider IS available the use_StridedMathModelForDMA flag must have no
/// effect: both flag states must produce identical cycle counts because the NN provider path is
/// taken regardless of the flag.
TEST_F(VPULayerCostModelTestNPU6x, StridedMathFlag_WithDMAProvider_CostUnchanged) {
    const DPUWorkload wl_base{
            VPUDevice::NPU_6_0,
            Operation::CONVOLUTION,
            {VPUTensor(56, 56, 64, 1, DataType::UINT8)},
            {VPUTensor(56, 56, 64, 1, DataType::UINT8)},
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

    // Both models use the real NPU5.0 DMA NN provider so is_dma_model_variant() returns true.
    VPULayerCostModel model_flag_off{VPUDevice::NPU_6_0, static_cast<DMACostModelVariant>(&dma_model_5_0),
                                     NPU_6_0_MODEL_PATH};
    VPULayerCostModel model_flag_on{VPUDevice::NPU_6_0, static_cast<DMACostModelVariant>(&dma_model_5_0),
                                    NPU_6_0_MODEL_PATH};
    model_flag_on.set_DMATheoreticalMathModel_enabled();

    DPULayer layer_off{wl_base};
    DPULayer layer_on{wl_base};

    const CyclesInterfaceType cost_flag_off =
            model_flag_off.Layer(layer_off, VPUTilingStrategy::NONE, 1U, 1U, /*input_in_ddr=*/true);
    const CyclesInterfaceType cost_flag_on =
            model_flag_on.Layer(layer_on, VPUTilingStrategy::NONE, 1U, 1U, /*input_in_ddr=*/true);

    EXPECT_FALSE(Cycles::isErrorCode(cost_flag_off)) << Cycles::toErrorText(cost_flag_off);
    EXPECT_FALSE(Cycles::isErrorCode(cost_flag_on)) << Cycles::toErrorText(cost_flag_on);

    // DMA NN provider is active: the flag must not influence the result.
    EXPECT_EQ(cost_flag_off, cost_flag_on)
            << "Expected identical costs when DMA NN provider is available regardless of "
               "use_StridedMathModelForDMA flag. "
            << "flag_off=" << cost_flag_off << " flag_on=" << cost_flag_on;
}

}  // namespace VPUNN_unit_tests