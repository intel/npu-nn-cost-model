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
#include "common/common_helpers.h"

#include "layer.h"
#include "vpu/shave/layers.h"

namespace VPUNN_unit_tests {
using namespace VPUNN;

class VPULayerCM_InvestigationTestNPU5x : public VPULayerCostModelTest {
public:
protected:
    /*    void SetUp() override {
            VPUNN::Logger::clear2ndlog();
            VPUNN::Logger::activate2ndlog();
        }
        void TearDown() override {
            VPUNN::Logger::clear2ndlog();
            VPUNN::Logger::deactivate2ndlog();
       }*/
};

TEST_F(VPULayerCM_InvestigationTestNPU5x, AVEPOOL_Layer_PRE_split_L2_E159363) {
    const VPUDevice device{VPUDevice::NPU_5_0};
    const DPUWorkload wl_1{
            device,
            Operation::AVEPOOL,
            {VPUTensor(4, 22, 5120, 1, DataType::UINT8, Layout::ZXY)},       // input dimensions
            {VPUTensor(4, 22, 5120, 1, DataType::FLOAT16, Layout::ZXY)},     // output dimensions
            {1, 1},                                                          // kernels
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
            std::optional<DataType>{},  // DataType::UINT8,                 // input1 data type (opt)
            "",                         // layer_info
            std::optional<bool>{},      // false,                       // weightless_operation (opt)
            std::optional<bool>{},      // false,                      // in_place_output_memory (opt)
            std::optional<bool>{},      // false                       // superdense_memory (opt)
    };
    const DPUWorkload wl_2{
            device,
            Operation::AVEPOOL,
            {VPUTensor(4, 21, 5120, 1, DataType::UINT8, Layout::ZXY)},       // input dimensions
            {VPUTensor(4, 21, 5120, 1, DataType::FLOAT16, Layout::ZXY)},     // output dimensions
            {1, 1},                                                          // kernels
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
            std::optional<DataType>{},  // DataType::UINT8,                 // input1 data type (opt)
            "",                         // layer_info
            std::optional<bool>{},      // false,                       // weightless_operation (opt)
            std::optional<bool>{},      // false,                      // in_place_output_memory (opt)
            std::optional<bool>{},      // false                       // superdense_memory (opt)
    };
    const DPUWorkload wl_3{wl_2};

    const bool prefetch{true};  // prefetch was done
    VPULayerCostModel& theModel = layer_models.getModel(VPUDevice::NPU_5_0);

    // L2 pre split
    Logger::clear2ndlog();
    CyclesInterfaceType pre_split_cost1{};
    LayerSplitInfo detailed_split_out{};
    const std::vector<DPULayer> splitLayers1{DPULayer(wl_1), DPULayer(wl_2), DPULayer(wl_3)};
    ASSERT_NO_THROW(pre_split_cost1 = theModel.LayersPreSplit(splitLayers1, 1U, false, false, prefetch,  //
                                                              detailed_split_out))
            << toStringLayerSplitInfo(detailed_split_out);

    ASSERT_TRUE(!Cycles::isErrorCode(pre_split_cost1))
            << pre_split_cost1 << " " << toStringLayerSplitInfo(detailed_split_out) << Logger::get2ndlog() << "END ERR";

    EXPECT_EQ(detailed_split_out.size(), 3) << "split does not propagate";

    EXPECT_EQ(1, 1) << toStringLayerSplitInfo(detailed_split_out) << Logger::get2ndlog() << "END ERR";
}

TEST_F(VPULayerCM_InvestigationTestNPU5x, AVEPOOL_Layer_PRE_split_L2_OUTPUT_AUTOPAD) {
    const VPUDevice device{VPUDevice::NPU_5_0};
    const DPUWorkload wl{
            device,
            Operation::AVEPOOL,
            {VPUTensor(32, 80, 16, 1, DataType::UINT8, Layout::ZXY)},        // input dimensions
            {VPUTensor(32, 80, 2, 1, DataType::UINT8, Layout::XYZ)},         // output dimensions
            {1, 1},                                                          // kernels
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
            std::optional<DataType>{},  // DataType::UINT8,                 // input1 data type (opt)
            "",                         // layer_info
            std::optional<bool>{},      // false,                       // weightless_operation (opt)
            std::optional<bool>{},      // false,                      // in_place_output_memory (opt)
            std::optional<bool>{},      // false                       // superdense_memory (opt)
            std::optional<bool>{},      // false // input_autopad (opt)
            std::optional<bool>{true},  // true //output_autopad
    };

    VPULayerCostModel& theModel = layer_models.getModel(VPUDevice::NPU_5_0);

    // L2 pre split
    Logger::clear2ndlog();
    CyclesInterfaceType pre_split_cost1{};
    LayerSplitInfo detailed_split_out{};
    const std::vector<DPULayer> splitLayers1{DPULayer(wl), DPULayer(wl), DPULayer(wl)};
    ASSERT_NO_THROW(pre_split_cost1 = theModel.LayersPreSplit(splitLayers1, 1U, false, false, true,  //
                                                              detailed_split_out))
            << toStringLayerSplitInfo(detailed_split_out);

    ASSERT_TRUE(!Cycles::isErrorCode(pre_split_cost1))
            << pre_split_cost1 << " " << toStringLayerSplitInfo(detailed_split_out) << Logger::get2ndlog() << "END ERR";

    EXPECT_EQ(detailed_split_out.size(), 3) << "split does not propagate";

    EXPECT_EQ(1, 1) << toStringLayerSplitInfo(detailed_split_out) << Logger::get2ndlog() << "END ERR";
}

TEST_F(VPULayerCM_InvestigationTestNPU5x, L2_INPUT_OUTPUT_AUTOPAD) {
    const VPUDevice device{VPUDevice::NPU_5_0};
    const DPUWorkload wl0{
            device,
            Operation::CONVOLUTION,
            {VPUTensor(1, 1, 9, 1, DataType::UINT8, Layout::ZXY)},           // input dimensions
            {VPUTensor(1, 1, 2736, 1, DataType::FLOAT16, Layout::ZXY)},      // output dimensions
            {1, 1},                                                          // kernels
            {1, 1},                                                          // strides
            {0, 0, 0, 0},                                                    // padding
            ExecutionMode::CUBOID_16x16,                                     // execution mode
            ActivationFunction::NONE,                                        // activation
            0.0F,                                                            // act_sparsity
            0.763672F,                                                       // weight_sparsity
            {Swizzling::KEY_0, Swizzling::KEY_0},                            // input_swizzling
            {Swizzling::KEY_0},                                              // output_swizzling
            3,                                                               // output_write_tiles
            {0, 0, 0, 0},                                                    // offsets
            ISIStrategy::SPLIT_OVER_K,                                       // isi_strategy
            true,                                                            // weight_sparsity_enabled
            {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects
            {false, {0, 0, 0, 0}, {0, 0, 0, 0}, false},                      // SEP
            std::optional<DataType>{},  // DataType::UINT8,                 // input1 data type (opt)
            "",                         // layer_info
            std::optional<bool>{},      // false,                       // weightless_operation (opt)
            std::optional<bool>{},      // false,                      // in_place_output_memory (opt)
            std::optional<bool>{},      // false                       // superdense_memory (opt)
            std::optional<bool>{true},  // true // input_autopad (opt)
            std::optional<bool>{},      // false //output_autopad
    };

    const DPUWorkload wl1{
            device,
            Operation::CONVOLUTION,
            {VPUTensor(13, 22, 8, 1, DataType::FLOAT16, Layout::ZXY)},       // input dimensions
            {VPUTensor(13, 22, 16, 1, DataType::FLOAT16, Layout::ZXY)},      // output dimensions
            {1, 1},                                                          // kernels
            {1, 1},                                                          // strides
            {0, 0, 0, 0},                                                    // padding
            ExecutionMode::CUBOID_16x16,                                     // execution mode
            ActivationFunction::NONE,                                        // activation
            0.0F,                                                            // act_sparsity
            0.75F,                                                           // weight_sparsity
            {Swizzling::KEY_0, Swizzling::KEY_0},                            // input_swizzling
            {Swizzling::KEY_0},                                              // output_swizzling
            1,                                                               // output_write_tiles
            {0, 0, 0, 0},                                                    // offsets
            ISIStrategy::CLUSTERING,                                         // isi_strategy
            true,                                                            // weight_sparsity_enabled
            {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects
            {false, {0, 0, 0, 0}, {0, 0, 0, 0}, false},                      // SEP
            std::optional<DataType>{},  // DataType::UINT8,                 // input1 data type (opt)
            "",                         // layer_info
            std::optional<bool>{},      // false,                       // weightless_operation (opt)
            std::optional<bool>{},      // false,                      // in_place_output_memory (opt)
            std::optional<bool>{},      // false                       // superdense_memory (opt)
            std::optional<bool>{true},  // true // input_autopad (opt)
            std::optional<bool>{},      // false //output_autopad
    };

    const DPUWorkload wl2{
            device,
            Operation::DW_CONVOLUTION,
            {VPUTensor(798, 39, 16, 1, DataType::FLOAT16, Layout::ZXY)},     // input dimensions
            {VPUTensor(798, 39, 3, 1, DataType::FLOAT16, Layout::ZXY)},      // output dimensions
            {1, 1},                                                          // kernels
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
            std::optional<DataType>{DataType::FLOAT16},  // DataType::UINT8,                 // input1 data type (opt)
            "",                                          // layer_info
            std::optional<bool>{},                       // false,                       // weightless_operation (opt)
            std::optional<bool>{},                       // false,                      // in_place_output_memory (opt)
            std::optional<bool>{true},                   // false                       // superdense_memory (opt)
            std::optional<bool>{},                       // false // input_autopad (opt)
            std::optional<bool>{true},                   // true //output_autopad
    };

    const DPUWorkload wl3{
            device,
            Operation::MAXPOOL,
            {VPUTensor(288, 65, 16, 1, DataType::FLOAT16, Layout::ZXY)},     // input dimensions
            {VPUTensor(288, 64, 3, 1, DataType::FLOAT32, Layout::ZXY)},      // output dimensions
            {3, 3},                                                          // kernels
            {1, 1},                                                          // strides
            {1, 0, 1, 1},                                                    // padding
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
            std::optional<DataType>{},  // DataType::UINT8,                 // input1 data type (opt)
            "",                         // layer_info
            std::optional<bool>{},      // false,                       // weightless_operation (opt)
            std::optional<bool>{},      // false,                      // in_place_output_memory (opt)
            std::optional<bool>{true},  // false                       // superdense_memory (opt)
            std::optional<bool>{},      // false // input_autopad (opt)
            std::optional<bool>{true},  // true //output_autopad
    };

    const DPUWorkload wl4{
            device,
            Operation::CONVOLUTION,
            {VPUTensor(56, 76, 12, 1, DataType::UINT8, Layout::ZXY)},        // input dimensions
            {VPUTensor(56, 19, 96, 1, DataType::UINT8, Layout::XYZ)},        // output dimensions
            {1, 4},                                                          // kernels
            {1, 4},                                                          // strides
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
            std::optional<DataType>{},   // DataType::UINT8,                 // input1 data type (opt)
            "wl4",                       // layer_info
            std::optional<bool>{},       // false,                       // weightless_operation (opt)
            std::optional<bool>{},       // false,                      // in_place_output_memory (opt)
            std::optional<bool>{true},   // false                       // superdense_memory (opt)
            std::optional<bool>{true},   // true // input_autopad (opt)
            std::optional<bool>{false},  // false //output_autopad
    };

    VPULayerCostModel& theModel = layer_models.getModel(VPUDevice::NPU_5_0);

    for (const auto& wl :
         std::vector<DPUWorkload>{std::move(wl0), std::move(wl1), std::move(wl2), std::move(wl3), std::move(wl4)}) {
        // L2 pre split
        Logger::clear2ndlog();
        CyclesInterfaceType pre_split_cost1{};
        LayerSplitInfo detailed_split_out{};
        const std::vector<DPULayer> splitLayers1{DPULayer(wl), DPULayer(wl), DPULayer(wl)};
        ASSERT_NO_THROW(pre_split_cost1 = theModel.LayersPreSplit(splitLayers1, 1U, false, false, true,  //
                                                                  detailed_split_out))
                << toStringLayerSplitInfo(detailed_split_out);

        ASSERT_TRUE(!Cycles::isErrorCode(pre_split_cost1))
                << pre_split_cost1 << " " << toStringLayerSplitInfo(detailed_split_out) << Logger::get2ndlog()
                << "END ERR";

        EXPECT_EQ(detailed_split_out.size(), 3) << "split does not propagate";

        EXPECT_EQ(1, 1) << toStringLayerSplitInfo(detailed_split_out) << Logger::get2ndlog() << "END ERR";
    }
}

TEST_F(VPULayerCM_InvestigationTestNPU5x, ELMWISE_Layer_PRE_split_L2_EXX_) {
    const VPUDevice device{VPUDevice::NPU_5_0};
    const DPUWorkload wl_123{
            device,
            Operation::ELTWISE,
            {VPUTensor(43, 3, 672, 1, DataType::FLOAT16, Layout::ZXY)},      // input dimensions
            {VPUTensor(43, 3, 672, 1, DataType::FLOAT16, Layout::YZX)},      // output dimensions
            {1, 1},                                                          // kernels
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
            std::optional<DataType>{},  // DataType::UINT8,                 // input1 data type (opt)
            "",                         // layer_info
            std::optional<bool>{},      // false,                       // weightless_operation (opt)
            std::optional<bool>{},      // false,                      // in_place_output_memory (opt)
            std::optional<bool>{},      // false                       // superdense_memory (opt)
    };
    // const DPUWorkload wl_123Alt{
    //         device,
    //         Operation::ELTWISE,
    //         {VPUTensor(43, 3, 672, 1, DataType::FLOAT16, Layout::ZXY)},      // input dimensions
    //         {VPUTensor(43, 3, 672, 1, DataType::FLOAT16, Layout::YZX)},      // output dimensions
    //         {1, 1},                                                          // kernels
    //         {1, 1},                                                          // strides
    //         {0, 0, 0, 0},                                                    // padding
    //         ExecutionMode::CUBOID_16x16,                                     // execution mode
    //         ActivationFunction::NONE,                                        // activation
    //         0.0F,                                                            // act_sparsity
    //         0.0F,                                                            // weight_sparsity
    //         {Swizzling::KEY_0, Swizzling::KEY_0},                            // input_swizzling
    //         {Swizzling::KEY_0},                                              // output_swizzling
    //         1,                                                               // output_write_tiles
    //         {0, 0, 0, 0},                                                    // offsets
    //         ISIStrategy::CLUSTERING,                                         // isi_strategy
    //         false,                                                           // weight_sparsity_enabled
    //         {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects
    //         {false, {0, 0, 0, 0}, {0, 0, 0, 0}, false},                      // SEP
    //         std::optional<DataType>{},  // DataType::UINT8,                 // input1 data type (opt)
    //         "",                         // layer_info
    //         std::optional<bool>{},      // false,                       // weightless_operation (opt)
    //         std::optional<bool>{},      // false,                      // in_place_output_memory (opt)
    //         std::optional<bool>{},      // false                       // superdense_memory (opt)
    // };
    //  const DPUWorkload wl_3{wl_2};

    const bool prefetch{true};  // prefetch was done
    VPULayerCostModel& theModel = layer_models.getModel(VPUDevice::NPU_5_0);

    // L2 pre split
    Logger::clear2ndlog();
    CyclesInterfaceType pre_split_cost1{};
    LayerSplitInfo detailed_split_out{};
    const std::vector<DPULayer> splitLayers1{DPULayer(wl_123), DPULayer(wl_123), DPULayer(wl_123)};
    ASSERT_NO_THROW(pre_split_cost1 = theModel.LayersPreSplit(splitLayers1, 1U, false, false, prefetch,  //
                                                              detailed_split_out))
            << toStringLayerSplitInfo(detailed_split_out);

    ASSERT_TRUE(!Cycles::isErrorCode(pre_split_cost1))
            << pre_split_cost1 << " " << toStringLayerSplitInfo(detailed_split_out) << Logger::get2ndlog() << "END ERR";

    EXPECT_EQ(detailed_split_out.size(), 3) << "split does not propagate";

    EXPECT_EQ(1, 1) << toStringLayerSplitInfo(detailed_split_out) << Logger::get2ndlog() << "END ERR";
}

TEST_F(VPULayerCM_InvestigationTestNPU5x, ZSPLIT_Layer_PRE_split_L2_E159358) {
    constexpr bool force_LegacyZTiling{
#ifdef VPUNN_OPT_LEGACY_ZTILING
            true
#else
            false
#endif
    };
    const VPUDevice device{VPUDevice::NPU_5_0};
    const DPUWorkload wl_1{
            device,
            Operation::DW_CONVOLUTION,
            {VPUTensor(100, 2, 1200, 1, DataType::FLOAT16, Layout::ZXY)},    // input dimensions
            {VPUTensor(100, 2, 1200, 1, DataType::FLOAT16, Layout::ZXY)},    // output dimensions
            {1, 1},                                                          // kernels
            {1, 1},                                                          // strides
            {0, 0, 0, 0},                                                    // padding
            ExecutionMode::CUBOID_16x16,                                     // execution mode (not trelevant)
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
            std::optional<DataType>{},  // DataType::UINT8,                 // input1 data type (opt)
            "",                         // layer_info
            std::optional<bool>{},      // false,                       // weightless_operation (opt)
            std::optional<bool>{},      // false,                      // in_place_output_memory (opt)
            std::optional<bool>{},      // false                       // superdense_memory (opt)
    };
    const DPUWorkload wl_2{
            device,
            Operation::DW_CONVOLUTION,
            {VPUTensor(100, 1, 1200, 1, DataType::FLOAT16, Layout::ZXY)},    // input dimensions
            {VPUTensor(100, 1, 1200, 1, DataType::FLOAT16, Layout::ZXY)},    // output dimensions
            {1, 1},                                                          // kernels
            {1, 1},                                                          // strides
            {0, 0, 0, 0},                                                    // padding
            ExecutionMode::CUBOID_16x16,                                     // execution mode (not trelevant)
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
            std::optional<DataType>{},  // DataType::UINT8,                 // input1 data type (opt)
            "",                         // layer_info
            std::optional<bool>{},      // false,                       // weightless_operation (opt)
            std::optional<bool>{},      // false,                      // in_place_output_memory (opt)
            std::optional<bool>{},      // false                       // superdense_memory (opt)
    };
    const DPUWorkload wl_3{wl_2};

    const bool prefetch{true};  // prefetch was done
    VPULayerCostModel& theModel = layer_models.getModel(VPUDevice::NPU_5_0);

    // L2 pre split
    Logger::clear2ndlog();
    CyclesInterfaceType pre_split_cost1{};
    LayerSplitInfo detailed_split_out{};
    const std::vector<DPULayer> splitLayers1{DPULayer(wl_1), DPULayer(wl_2), DPULayer(wl_3)};
    ASSERT_NO_THROW(pre_split_cost1 = theModel.LayersPreSplit(splitLayers1, 1U, false, false, prefetch,  //
                                                              detailed_split_out))
            << toStringLayerSplitInfo(detailed_split_out);

    ASSERT_TRUE(!Cycles::isErrorCode(pre_split_cost1))
            << pre_split_cost1 << " " << toStringLayerSplitInfo(detailed_split_out) << Logger::get2ndlog() << "END ERR";

    EXPECT_EQ(detailed_split_out.size(), 3) << "split does not propagate";

    EXPECT_EQ(1, 1) << toStringLayerSplitInfo(detailed_split_out) << Logger::get2ndlog() << "END ERR";

    //    const auto& detailed_split_out_ALL = detailed_split_out[0].all_intra_tile_splits;
    const auto& detailed_split_out_BST = detailed_split_out[0].best_intra_tile_split.second;

    if (!force_LegacyZTiling) {  // 18x64 + 1x32 + 1x16
        EXPECT_EQ(detailed_split_out_BST.size(), 20)
                << toStringLayerSplitInfo(detailed_split_out) << Logger::get2ndlog() << "END ERR";
    } else {                                          // legacy
        EXPECT_EQ(detailed_split_out_BST.size(), 38)  // 37x32 + 1x16
                << toStringLayerSplitInfo(detailed_split_out) << Logger::get2ndlog() << "END ERR";
    }

    // EXPECT_TRUE(force_LegacyZTiling);
}

}