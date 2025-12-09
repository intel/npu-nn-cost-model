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

/// for investigating L2Api Compiler regressions
class VPULayerCM_InvestigationTest : public VPULayerCostModelTest {
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

TEST_F(VPULayerCM_InvestigationTest, DISABLED_UnimplementedStrategies_ResultsTest) {
    const bool force_fail{false};        // controls force failing assertion
    const int fail{force_fail ? 0 : 1};  // 1 neutral, 0 fail

    show_split = false;

    DPUWorkload wl_ref{
            VPUDevice::VPU_4_0,
            Operation::CONVOLUTION,
            {VPUTensor(28, 28, 256, 1, DataType::UINT8, Layout::ZXY)},  // input dimensions
            {VPUTensor(28, 28, 128, 1, DataType::UINT8, Layout::ZXY)},  // output dimensions
            {1, 1},                                                     // kernels
            {1, 1},                                                     // strides
            {0, 0, 0, 0},                                               // padding
            ExecutionMode::CUBOID_16x16,                                // execution mode
            ActivationFunction::NONE,                                   // activation
            0.0F,                                                       // act_sparsity
            0.0F,                                                       // weight_sparsity
            {swz_def, swz_def},                                         // input_swizzling
            {swz_def},                                                  // output_swizzling
            1,                                                          // output_write_tiles
            {0, 0, 0, 0},                                               // offsets
            ISIStrategy::CLUSTERING,                                    // isi_strategy
            false,                                                      // weight_sparsity_enabled
    };
    const bool prefetch{true};

    const DPULayer tst_layer(wl_ref);
    const std::vector<TestCase> tests{
            {{std::move(tst_layer), {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOW, false, false, prefetch}},
             {VPUNN::Cycles::ERROR_TILE_OUTPUT, true, 1000, fail * 1000 + 1000},
             "SOW "},

    };
    executeTests(tests);
}

TEST_F(VPULayerCM_InvestigationTest, SOW_SmokeTest) {
    const bool force_fail{false};        // controls force failing assertion
    const int fail{force_fail ? 0 : 1};  // 1 neutral, 0 fail

    show_split = true;

    DPUWorkload wl_ref{
            VPUDevice::VPU_4_0,
            Operation::CONVOLUTION,
            {VPUTensor(28, 28, 256, 1, DataType::UINT8, Layout::ZXY)},  // input dimensions
            {VPUTensor(28, 28, 128, 1, DataType::UINT8, Layout::ZXY)},  // output dimensions
            {1, 1},                                                     // kernels
            {1, 1},                                                     // strides
            {0, 0, 0, 0},                                               // padding
            ExecutionMode::CUBOID_16x16,                                // execution mode
            ActivationFunction::NONE,                                   // activation
            0.0F,                                                       // act_sparsity
            0.0F,                                                       // weight_sparsity
            {swz_def, swz_def},                                         // input_swizzling
            {swz_def},                                                  // output_swizzling
            1,                                                          // output_write_tiles
            {0, 0, 0, 0},                                               // offsets
            ISIStrategy::CLUSTERING,                                    // isi_strategy
            false,                                                      // weight_sparsity_enabled
    };
    const bool prefetch{true};

    const DPULayer tst_layer(wl_ref);
    // TODO: test should be SOHO wins,
    const std::vector<TestCase> tests{
            {{std::move(tst_layer), {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
             {VPUNN::Cycles::NO_ERROR, true, 3700, fail * 3700 + 1000},
             "SOW "},

    };
    executeTests(tests);
}

TEST_F(VPULayerCM_InvestigationTest, ModelA_CONV75_NPU40) {
    const bool force_fail{false};        // controls force failing assertion
    const int fail{force_fail ? 0 : 1};  // 1 neutral, 0 fail
    show_split = true;
    // EXPECT_TRUE(false);

    const DPUWorkload wl_{
            // orig Layer
            VPUDevice::VPU_4_0,
            Operation::CONVOLUTION,
            {VPUTensor(112, 112, 32, 1, DataType::UINT8)},  // input dimensions
            {VPUTensor(112, 112, 64, 1, DataType::UINT8)},  // output dimensions
            {1, 1},                                         // kernels
            {1, 1},                                         // strides
            {0, 0, 0, 0},                                   // padding
            ExecutionMode::CUBOID_16x16,                    // execution mode
            ActivationFunction::NONE,                       // activation
            0.0F,                                           // act_sparsity
            0.0F,                                           // weight_sparsity
            {swz_def, swz_def},                             // input_swizzling
            {swz_def},                                      // output_swizzling
            1,                                              // output_write_tiles
            {0, 0, 0, 0},                                   // offsets
            ISIStrategy::CLUSTERING,                        // isi_strategy
            false,                                          // weight_sparsity_enabled
    };

    const bool prefetch{true};

    {
        const DPULayer tst_layer(wl_);
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {Cycles::NO_ERROR, true, 2734, fail * 3734 + 1300},
                 "SOHO , no memmove, "},  // GT is 3734 for 16x16

                {{tst_layer, {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOK, false, false, prefetch}},
                 {Cycles::NO_ERROR, true, 5500, fail * 12800 + 2000},
                 "SOK , no memmove, "},  // gt is 12800+ fro all MPE
                                         // SOHO wins
        };
        executeTests(tests);
    }
}

TEST_F(VPULayerCM_InvestigationTest, EISXW_140892_err_investigation_NPU40) {
    const bool force_fail{false};        // controls force failing assertion
    const int fail{force_fail ? 0 : 1};  // 1 neutral, 0 fail
    show_split = true;
    // EXPECT_TRUE(false);

    const DPUWorkload wl_{
            // orig Layer
            VPUDevice::VPU_4_0,
            Operation::CONVOLUTION,
            {VPUTensor(11, 3, 128, 1, DataType::FLOAT16)},    // input dimensions
            {VPUTensor(11, 3, 13872, 1, DataType::FLOAT16)},  // output dimensions
            {1, 1},                                           // kernels
            {1, 1},                                           // strides
            {0, 0, 0, 0},                                     // padding
            ExecutionMode::CUBOID_16x16,                      // execution mode
            ActivationFunction::NONE,                         // activation
            0.0F,                                             // act_sparsity
            0.0F,                                             // weight_sparsity
            {Swizzling::KEY_0, Swizzling::KEY_0},             // input_swizzling
            {Swizzling::KEY_0},                               // output_swizzling
            1,                                                // output_write_tiles
            {0, 0, 0, 0},                                     // offsets
            ISIStrategy::CLUSTERING,                          // isi_strategy
            false,                                            // weight_sparsity_enabled
    };

    const bool prefetch{true};

    {
        const DPULayer tst_layer(wl_);
        const std::vector<TestCase> tests{
                {{std::move(tst_layer), {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOK, false, false, prefetch}},
                 {Cycles::ERROR_INPUT_TOO_BIG, true, 4000, fail * (4686) + 800},
                 "SOK , no memmove, "},
        };

        /*
        Memory:
        In=11*3*128*2=4224   ----> aligned 16384
        Out=11*3*13872=457776  --> aligned 458752
        Wts=1*1*13872*(128*1*1)=1775616   ----> aligned 1785856

        ====> total de 2.250.752 => MEMORY TO BIG
        */

        executeTests(tests);
    }
}
TEST_F(VPULayerCM_InvestigationTest, ModelA_GC105_NPU40) {
    const bool force_fail{false};        // controls force failing assertion
    const int fail{force_fail ? 0 : 1};  // 1 neutral, 0 fail
    show_split = true;
    // EXPECT_TRUE(false);

    const DPUWorkload wl_{
            // orig Layer
            VPUDevice::VPU_4_0,
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
        const DPULayer tst_layer(wl_);
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {Cycles::NO_ERROR, true, 5485, fail * (4 * 1410) + 500},
                 "SOHO , no memmove, "},  // GT 5640

                {{tst_layer, {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOK, false, false, prefetch}},
                 {Cycles::NO_ERROR, true, 4000, fail * (4686) + 800},
                 "SOK , no memmove, "},  // GT 4686
                                         // SOK wins
        };
        executeTests(tests);
    }
}

TEST_F(VPULayerCM_InvestigationTest, ModelA_GC124_NPU40) {
    const bool force_fail{false};        // controls force failing assertion
    const int fail{force_fail ? 0 : 1};  // 1 neutral, 0 fail
    show_split = true;
    // EXPECT_TRUE(false);

    const DPUWorkload wl_{
            // orig Layer
            VPUDevice::VPU_4_0,
            Operation::DW_CONVOLUTION,
            {VPUTensor(56, 56, 128, 1, DataType::UINT8)},  // input dimensions
            {VPUTensor(28, 28, 128, 1, DataType::UINT8)},  // output dimensions
            {3, 3},                                        // kernels
            {2, 2},                                        // strides
            {0, 1, 0, 1},                                  // padding
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
        const DPULayer tst_layer(wl_);
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {Cycles::NO_ERROR, true, 2500, fail * (1361 * 2) + 1000},
                 "SOHO , no memmove, "},  // gt = 2712

                {{tst_layer, {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOK, false, false, prefetch}},
                 {Cycles::NO_ERROR, true, 2000, fail * (2134) + 500},
                 "SOK , no memmove, "},  // gt 2134
                                         // SOK wins
        };
        executeTests(tests);
    }
}

TEST_F(VPULayerCM_InvestigationTest, ModelA_CONV251_NPU40) {
    const bool force_fail{false};        // controls force failing assertion
    const int fail{force_fail ? 0 : 1};  // 1 neutral, 0 fail
    show_split = true;
    // EXPECT_TRUE(false);

    const DPUWorkload wl_{
            // orig Layer
            VPUDevice::VPU_4_0,
            Operation::CONVOLUTION,
            {VPUTensor(7, 7, 160, 1, DataType::UINT8)},  // input dimensions
            {VPUTensor(7, 7, 64, 1, DataType::UINT8)},   // output dimensions
            {3, 3},                                      // kernels
            {1, 1},                                      // strides
            {1, 1, 1, 1},                                // padding
            ExecutionMode::CUBOID_16x16,                 // execution mode
            ActivationFunction::NONE,                    // activation
            0.0F,                                        // act_sparsity
            0.0F,                                        // weight_sparsity
            {swz_def, swz_def},                          // input_swizzling
            {swz_def},                                   // output_swizzling
            1,                                           // output_write_tiles
            {0, 0, 0, 0},                                // offsets
            ISIStrategy::CLUSTERING,                     // isi_strategy
            false,                                       // weight_sparsity_enabled
    };

    const bool prefetch{true};

    {
        const DPULayer tst_layer(wl_);
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {Cycles::NO_ERROR, true, 1200, fail * 1585 + 100},
                 "SOHO , no memmove, "},  // gt 1585

                {{tst_layer, {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOK, false, false, prefetch}},
                 {Cycles::NO_ERROR, true, 700, fail * 967 + 200},
                 "SOK , no memmove, "},  // gt 967
                                         // SOK wins
        };
        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, Layer_Detailed_SEP_split_EISXW_132541_NPU40) {
    show_split = false;
    // EXPECT_TRUE(false);
    const HaloWorkload h_zero{{0, 0, 0, 0, 0, 0},  // H in TBLRFB
                              {0, 0, 0, 0, 0, 0},
                              {0, 0, 0, 0, 0, 0},
                              {0, 0, 0, 0, 0, 0}};

    const SEPModeInfo small_sep_info{
            true,            // sep activators using Storage elements table with pointers
            {128, 1, 1, 1},  // SEP pointer table, 32 bits pointers assumed
            {30, 6, 3, 1},   // actual tensor shape for activators
            false            // no_sparse_map if true the sparse map is ignored/non existent
    };

    const SEPModeInfo big_sep_info{
            true,              // sep activators using Storage elements table with pointers
            {243, 130, 1, 1},  // SEP pointer table, 32 bits pointers assumed
            {60, 90, 220, 1},  // actual tensor shape for activators
            false              // no_sparse_map if true the sparse map is ignored/non existent
    };
    const DPUWorkload wl_small_sep{
            // orig Layer
            VPUDevice::VPU_4_0,
            Operation::CONVOLUTION,
            {VPUTensor(128, 112, 64, 1, DataType::UINT8)},  // input dimensions
            {VPUTensor(126, 110, 64, 1, DataType::UINT8)},  // output dimensions
            {3, 3},                                         // kernels
            {1, 1},                                         // strides
            {0, 0, 0, 0},                                   // padding
            ExecutionMode::CUBOID_16x16,                    // execution mode
            ActivationFunction::NONE,                       // activation
            0.0F,                                           // act_sparsity
            0.0F,                                           // weight_sparsity
            {swz_def, swz_def},                             // input_swizzling
            {swz_def},                                      // output_swizzling
            1,                                              // output_write_tiles
            {0, 0, 0, 0},                                   // offsets
            ISIStrategy::CLUSTERING,                        // isi_strategy
            false,                                          // weight_sparsity_enabled
            h_zero,                                         // halo
            small_sep_info,                                 // SEP configuration for input memory
    };

    DPUWorkload wl_big_sep{wl_small_sep};
    wl_big_sep.sep_activators = big_sep_info;

    struct TestInput {
        DPUWorkload wl;
        unsigned int n_tiles;
        VPUTilingStrategy strategy;
    };

    struct TestExpectations {
        std::vector<std::pair<WHCBTensorShape, WHCBTensorShape>>
                tensors_sep_info;  // first pair element = storage_elements_pointers
                                   // second pair element = actual_activators_input
    };

    struct SEP_TestCase {
        TestInput t_in;
        TestExpectations t_exp;
        const std::string test_case = "";
    };
    using SEP_TestsVector = std::vector<SEP_TestCase>;

    auto verify_SEP_split_equiv = [](SEP_TestCase& t) {
        const DPULayer layer(t.t_in.wl);
        std::vector<DPULayer> tiles_layer = layer.splitAcrossTiles(t.t_in.strategy, t.t_in.n_tiles);

        std::cout << "--------------------------------- " << t.test_case << " ---------------------------------\n ";

        for (unsigned int i = 0U; i < t.t_in.n_tiles; i++) {
            EXPECT_EQ(tiles_layer[i].sep_activators.storage_elements_pointers, t.t_exp.tensors_sep_info[i].first)
                    << "Storage elements pointers doesn't have the expected value \n " << tiles_layer[i];

            EXPECT_EQ(tiles_layer[i].sep_activators.actual_activators_input, t.t_exp.tensors_sep_info[i].second)
                    << "Actual activators input doesn't have the expected value \n " << tiles_layer[i];
        }
    };

    std::pair<WHCBTensorShape, WHCBTensorShape> all_tiles_small_sep = {
            {128, 1, 1, 1},
            {30, 2, 3, 1}};  // all tiles have the same sep (case when layer sep is  small_sep_info)

    std::pair<WHCBTensorShape, WHCBTensorShape> T_M_tiles_big_sep = {
            {243, 35, 1, 1},
            {60, 25, 220, 1}};  // top and middle tiles sep (case when layer sep is  big_sep_info)
    std::pair<WHCBTensorShape, WHCBTensorShape> B_tiles_big_sep = {
            {243, 33, 1, 1},
            {60, 23, 220, 1}};  // btm tile sep (case when layer sep is  big_sep_info)

    std::pair<WHCBTensorShape, WHCBTensorShape> orig_small_sep_info = {
            small_sep_info.storage_elements_pointers,
            small_sep_info.actual_activators_input};  // original small_sep_info, no split
    std::pair<WHCBTensorShape, WHCBTensorShape> orig_big_sep_info = {
            big_sep_info.storage_elements_pointers,
            big_sep_info.actual_activators_input};  // original big_sep_info, no split
    SEP_TestsVector tests{
            // clang-format off

            //strategy: SOHO  we expect SEP split
            {{wl_small_sep, 4U, VPUTilingStrategy::SOH_Overlapped},{ {all_tiles_small_sep, all_tiles_small_sep, all_tiles_small_sep, all_tiles_small_sep} }, "SOHO, Small sep test"},
            {{wl_big_sep, 4U, VPUTilingStrategy::SOH_Overlapped }, {{ T_M_tiles_big_sep,  T_M_tiles_big_sep,  T_M_tiles_big_sep, B_tiles_big_sep }}, "SOHO, Big sep test"},

            //strategy: SOK  we don't expect SEP split  => all tiles should have original SEP
            {{wl_small_sep, 4U, VPUTilingStrategy::SOK},{ {orig_small_sep_info, orig_small_sep_info, orig_small_sep_info, orig_small_sep_info} }, "SOK, Small sep test"},
            {{wl_big_sep, 4U, VPUTilingStrategy::SOK }, {{ orig_big_sep_info, orig_big_sep_info, orig_big_sep_info, orig_big_sep_info }}, "SOK, Big sep test"},
            // clang-format on
    };

    for (auto& test : tests) {
        verify_SEP_split_equiv(test);
    }
}

TEST_F(VPULayerCostModelTest, Test_layer_SEP_split_value_propagation_EISXW_132541_NPU40) {
    show_split = false;
    // EXPECT_TRUE(false);
    const HaloWorkload h_zero{{0, 0, 0, 0, 0, 0},  // H in TBLRFB
                              {0, 0, 0, 0, 0, 0},
                              {0, 0, 0, 0, 0, 0},
                              {0, 0, 0, 0, 0, 0}};

    const SEPModeInfo small_sep_info{
            true,            // sep activators using Storage elements table with pointers
            {128, 1, 1, 1},  // SEP pointer table, 32 bits pointers assumed
            {30, 6, 3, 1},   // actual tensor shape for activators
            false            // no_sparse_map if true the sparse map is ignored/non existent
    };

    const DPUWorkload wl_small_sep{
            // orig Layer
            VPUDevice::VPU_4_0,
            Operation::CONVOLUTION,
            {VPUTensor(128, 112, 64, 1, DataType::UINT8)},  // input dimensions
            {VPUTensor(126, 110, 64, 1, DataType::UINT8)},  // output dimensions
            {3, 3},                                         // kernels
            {1, 1},                                         // strides
            {0, 0, 0, 0},                                   // padding
            ExecutionMode::CUBOID_16x16,                    // execution mode
            ActivationFunction::NONE,                       // activation
            0.0F,                                           // act_sparsity
            0.0F,                                           // weight_sparsity
            {swz_def, swz_def},                             // input_swizzling
            {swz_def},                                      // output_swizzling
            1,                                              // output_write_tiles
            {0, 0, 0, 0},                                   // offsets
            ISIStrategy::CLUSTERING,                        // isi_strategy
            false,                                          // weight_sparsity_enabled
            h_zero,                                         // halo
            small_sep_info,                                 // SEP configuration for input memory
    };

    struct TestInput {
        DPUWorkload wl;
        VPULayerStrategy strategy;
    };

    struct TestExpectations {
        std::vector<std::pair<WHCBTensorShape, WHCBTensorShape>>
                tensors_sep_info;  // first pair element = storage_elements_pointers
                                   // second pair element = actual_activators_input
    };

    struct SEP_TestCase {
        TestInput t_in;
        TestExpectations t_exp;
        const std::string test_case = "";
    };
    using SEP_TestsVector = std::vector<SEP_TestCase>;

    auto verify_SEP_split_equiv = [this](SEP_TestCase& t) {
        DPULayer layer(t.t_in.wl);

        VPULayerStrategy strategy{t.t_in.strategy};
        LayerSplitInfo detailed_split;
        CyclesInterfaceType cost_cyc = layer_models.getModel(layer.device).Layer(layer, strategy, detailed_split);

        std::cout << "--------------------------------- " << t.test_case << " ---------------------------------\n ";

        EXPECT_FALSE(Cycles::isErrorCode(cost_cyc));

        for (long unsigned int i = 0; i < detailed_split.size(); i++) {
            EXPECT_EQ(detailed_split[i].inter_tile_split_layer.sep_activators.storage_elements_pointers,
                      t.t_exp.tensors_sep_info[i].first)
                    << "Storage elements pointers doesn't have the expected value \n " << cost_cyc << "\n"
                    << detailed_split[i].inter_tile_split_layer;

            EXPECT_EQ(detailed_split[i].inter_tile_split_layer.sep_activators.actual_activators_input,
                      t.t_exp.tensors_sep_info[i].second)
                    << "Actual activators input doesn't have the expected value \n " << cost_cyc << "\n"
                    << detailed_split[i].inter_tile_split_layer;
        }
    };

    std::pair<WHCBTensorShape, WHCBTensorShape> all_tiles_small_sep = {
            {128, 1, 1, 1},
            {30, 2, 3, 1}};  // all tiles have the same sep (case when layer sep is  small_sep_info)

    std::pair<WHCBTensorShape, WHCBTensorShape> orig_small_sep_info = {
            small_sep_info.storage_elements_pointers,
            small_sep_info.actual_activators_input};  // original small_sep_info, no split

    SEP_TestsVector tests{
            // clang-format off

            //strategy: SOHO  we expect SEP split
            {{wl_small_sep, {1U, 1U, 4U, VPUTilingStrategy::SOH_Overlapped, false, false, true}},{ {all_tiles_small_sep, all_tiles_small_sep, all_tiles_small_sep, all_tiles_small_sep} }, "SOHO, Small sep test"},

            //strategy: SOK  we don't expect SEP split  => all tiles should have original SEP
            {{wl_small_sep, {1U, 1U, 4U, VPUTilingStrategy::SOK, false, false, true}},{ {orig_small_sep_info, orig_small_sep_info, orig_small_sep_info, orig_small_sep_info} }, "SOK, Small sep test"},

            // clang-format on
    };

    for (auto& test : tests) {
        verify_SEP_split_equiv(test);
    }
}

}  // namespace VPUNN_unit_tests
