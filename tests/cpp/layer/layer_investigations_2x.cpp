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

class VPULayerCM_InvestigationTestVPU2x : public VPULayerCostModelTest {
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

TEST_F(VPULayerCM_InvestigationTestVPU2x, DWConv_SOK_SOH_Comparison_EISXW_92399) {
    const DPUWorkload wl_h17_orig{
            // orig Layer
            VPUDevice::VPU_2_7,
            Operation::DW_CONVOLUTION,
            {VPUTensor(17, 17, 288, 1, DataType::FLOAT16)},  // input dimensions
            {VPUTensor(17, 17, 288, 1, DataType::FLOAT16)},  // output dimensions
            {9, 9},                                          // kernels
            {1, 1},                                          // strides
            {4, 4, 4, 4},                                    // padding
            ExecutionMode::CUBOID_16x16,                     // execution mode
            ActivationFunction::NONE,                        // activation
            0.0F,                                            // act_sparsity
            0.0F,                                            // weight_sparsity
            {swz_def, swz_def},                              // input_swizzling
            {swz_def},                                       // output_swizzling
            1,                                               // output_write_tiles
            {0, 0, 0, 0},                                    // offsets
            ISIStrategy::CLUSTERING,                         // isi_strategy
            false,                                           // weight_sparsity_enabled
    };

    // const std::string altModelName(NameHelperNN::get_model_root() + "vpu_2_7-150.vpunn");
    // VPULayerCostModel model_alternative{altModelName};
    const bool prefetch{true};
    // EXPECT_TRUE(false);
    {
        const DPULayer tst_layer(wl_h17_orig);
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 95000, 95000 + 8000},  //  v16 96k        v17: 98k         //v159 99k
                 "SOHO , no memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 75000, 75000 + 15000},  // v16 77k, v17 87k,  //v159 76k  GT??
                 "SOK , no memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_HaloRead, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 114000,
                  114000 + 50000},  // v16  115k      v17: 115k:157k   //v159 124k
                 "SOH Halo , no memmove, "},

                // note:SOK wins
        };
        executeTests(tests);
    }

    VPULayerCostModel& theModel = layer_models.getModel(VPUDevice::VPU_2_7);

    // low level WL
    const DPUWorkload wl_h17_Top_K64x4{
            // tile1 top  K : 64X4 and 32 X1
            VPUDevice::VPU_2_7,
            Operation::DW_CONVOLUTION,
            {VPUTensor(17, 13, 64, 1, DataType::FLOAT16)},  // input dimensions
            {VPUTensor(17, 9, 64, 1, DataType::FLOAT16)},   // output dimensions
            {9, 9},                                         // kernels
            {1, 1},                                         // strides
            {4, 0, 4, 4},                                   // padding
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
    DPUWorkload wl_h17_Top_K32x1{wl_h17_Top_K64x4};
    wl_h17_Top_K32x1.inputs[0] = VPUTensor(17, 13, 32, 1, DataType::FLOAT16);
    wl_h17_Top_K32x1.outputs[0] = VPUTensor(17, 9, 32, 1, DataType::FLOAT16);

    const DPUWorkload wl_h17_Bot_K64x4{
            // tile 1 bot   K : 64X4 and 32 X1
            VPUDevice::VPU_2_7,
            Operation::DW_CONVOLUTION,
            {VPUTensor(17, 12, 64, 1, DataType::FLOAT16)},  // input dimensions
            {VPUTensor(17, 8, 64, 1, DataType::FLOAT16)},   // output dimensions
            {9, 9},                                         // kernels
            {1, 1},                                         // strides
            {0, 4, 4, 4},                                   // padding
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
    DPUWorkload wl_h17_Bot_K32x1{wl_h17_Bot_K64x4};
    wl_h17_Bot_K32x1.inputs[0] = VPUTensor(17, 12, 32, 1, DataType::FLOAT16);
    wl_h17_Bot_K32x1.outputs[0] = VPUTensor(17, 8, 32, 1, DataType::FLOAT16);

    // make also SOH forced version
    HaloWorkload halo_top{{0, 4, 0, 0, 0, 0},  // H in
                          {0, 0, 0, 0, 0, 0},
                          {0, 0, 0, 0, 0, 0},
                          {0, 0, 0, 0, 0, 0}};
    HaloWorkload halo_bot{{4, 0, 0, 0, 0, 0},  // H in
                          {0, 0, 0, 0, 0, 0},
                          {0, 0, 0, 0, 0, 0},
                          {0, 0, 0, 0, 0, 0}};

    DPUWorkload wl_h17_Top_SOHH_K64x4{wl_h17_Top_K64x4};
    wl_h17_Top_SOHH_K64x4.isi_strategy = ISIStrategy::SPLIT_OVER_H;
    wl_h17_Top_SOHH_K64x4.halo = halo_top;

    DPUWorkload wl_h17_Top_SOHH_K32x1{wl_h17_Top_K32x1};
    wl_h17_Top_SOHH_K32x1.isi_strategy = ISIStrategy::SPLIT_OVER_H;
    wl_h17_Top_SOHH_K32x1.halo = halo_top;

    DPUWorkload wl_h17_Bot_SOHH_K64x4{wl_h17_Bot_K64x4};
    wl_h17_Bot_SOHH_K64x4.isi_strategy = ISIStrategy::SPLIT_OVER_H;
    wl_h17_Bot_SOHH_K64x4.halo = halo_bot;

    DPUWorkload wl_h17_Bot_SOHH_K32x1{wl_h17_Bot_K32x1};
    wl_h17_Bot_SOHH_K32x1.isi_strategy = ISIStrategy::SPLIT_OVER_H;
    wl_h17_Bot_SOHH_K32x1.halo = halo_bot;

    // SOK

    const DPUWorkload wl_h17_SOKHalf_K32x4{
            // SOK  K 32 X4, k=16X1
            VPUDevice::VPU_2_7,
            Operation::DW_CONVOLUTION,
            {VPUTensor(17, 17, 32, 1, DataType::FLOAT16)},  // input dimensions
            {VPUTensor(17, 17, 32, 1, DataType::FLOAT16)},  // output dimensions
            {9, 9},                                         // kernels
            {1, 1},                                         // strides
            {4, 4, 4, 4},                                   // padding
            ExecutionMode::CUBOID_16x16,                    // execution mode
            ActivationFunction::NONE,                       // activation
            0.0F,                                           // act_sparsity
            0.0F,                                           // weight_sparsity
            {swz_def, swz_def},                             // input_swizzling
            {swz_def},                                      // output_swizzling
            2,                                              // output_write_tiles
            {0, 0, 0, 0},                                   // offsets
            ISIStrategy::SPLIT_OVER_K,                      // isi_strategy
            false,                                          // weight_sparsity_enabled
    };
    DPUWorkload wl_h17_SOKHalf_K16x1{wl_h17_SOKHalf_K32x4};
    wl_h17_SOKHalf_K16x1.inputs[0] = VPUTensor(17, 17, 16, 1, DataType::FLOAT16);
    wl_h17_SOKHalf_K16x1.outputs[0] = VPUTensor(17, 17, 16, 1, DataType::FLOAT16);

    // SOH hako
    const DPUWorkload wl_h17_Top_SOH_K64x4{
            // tile1 top  K : 64X4 and 32 X1
            VPUDevice::VPU_2_7,
            Operation::DW_CONVOLUTION,
            {VPUTensor(17, 13, 64, 1, DataType::FLOAT16)},  // input dimensions
            {VPUTensor(17, 9, 64, 1, DataType::FLOAT16)},   // output dimensions
            {9, 9},                                         // kernels
            {1, 1},                                         // strides
            {4, 0, 4, 4},                                   // padding
            ExecutionMode::CUBOID_16x16,                    // execution mode
            ActivationFunction::NONE,                       // activation
            0.0F,                                           // act_sparsity
            0.0F,                                           // weight_sparsity
            {swz_def, swz_def},                             // input_swizzling
            {swz_def},                                      // output_swizzling
            1,                                              // output_write_tiles
            {0, 0, 0, 0},                                   // offsets
            ISIStrategy::SPLIT_OVER_H,                      // isi_strategy
            false,                                          // weight_sparsity_enabled
            {{0, 4, 0, 0, 0, 0},                            // H in
             {0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0}},
    };
    DPUWorkload wl_h17_Top_SOH_K32x1{wl_h17_Top_SOH_K64x4};
    wl_h17_Top_SOH_K32x1.inputs[0] = VPUTensor(17, 13, 32, 1, DataType::FLOAT16);
    wl_h17_Top_SOH_K32x1.outputs[0] = VPUTensor(17, 9, 32, 1, DataType::FLOAT16);

    const DPUWorkload wl_h17_Bot_SOH_K16x18{
            // tile 1 bot   K : 16x18
            VPUDevice::VPU_2_7,
            Operation::DW_CONVOLUTION,
            {VPUTensor(17, 12, 16, 1, DataType::FLOAT16)},  // input dimensions
            {VPUTensor(17, 8, 16, 1, DataType::FLOAT16)},   // output dimensions
            {9, 9},                                         // kernels
            {1, 1},                                         // strides
            {0, 4, 4, 4},                                   // padding
            ExecutionMode::CUBOID_16x16,                    // execution mode
            ActivationFunction::NONE,                       // activation
            0.0F,                                           // act_sparsity
            0.0F,                                           // weight_sparsity
            {swz_def, swz_def},                             // input_swizzling
            {swz_def},                                      // output_swizzling
            1,                                              // output_write_tiles
            {0, 0, 0, 0},                                   // offsets
            ISIStrategy::SPLIT_OVER_H,                      // isi_strategy
            false,                                          // weight_sparsity_enabled
            {{4, 0, 0, 0, 0, 0},                            // H in
             {0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0}},
    };

    // direct WL @ DPU level tests on
    const bool on{false};  // controls force failing assertion

    auto case_run = [=, &theModel](const DPUWorkload& workload, const std::string& whatTest) {
        DPUWorkload tst_layer(workload);
        std::string err_info;

        CyclesInterfaceType dpu_cost = theModel.get_cost_model().DPU(tst_layer, err_info);
        if (on) {
            EXPECT_EQ(dpu_cost, 10) << whatTest << dpu_cost << "=" << Cycles::toErrorText(dpu_cost) << "\n"
                                    << tst_layer << err_info << "END ERR";
        }
        std::cout << "  " << whatTest << dpu_cost << "=" << Cycles::toErrorText(dpu_cost) << "\n";
    };

    //     EXPECT_TRUE(on);  // something has to fail to  see couts

    {
        Logger::clear2ndlog();
        std::string err_info;

        {  // SOHO SPLITS
            std::cout << "\n SOHO SPLITS: ";

            case_run(wl_h17_Top_K64x4, "\nTEST of: SOHO TOP K=64  X4\n");  // v16 21284  v17: 21867     //v150 23k
            case_run(wl_h17_Top_K32x1, "\nTEST of: SOHO TOP K=32  X1\n");  // v16 11077  v17: 11101     //v150 11k
            case_run(wl_h17_Bot_K64x4, "\nTEST of:SOHO BOT K=64 X4\n");    // v16 15294  v17: 15885     //v150 17k
            case_run(wl_h17_Bot_K32x1, "\nTEST of:SOHO BOT K=32 x1 \n");   // v16 7744   v17: 7709     //v150 8595
        }

        {  /// SOK splits
            std::cout << "\n SOK SPLITS: ";

            case_run(wl_h17_SOKHalf_K32x4, "\nTEST of:SOK K=32  x4 =144\n");  // v16 17190  v17:19196  //v150 16k
            case_run(wl_h17_SOKHalf_K16x1, "\nTEST of:SOK K=16 x1 \n");       // v16 8976   v17:10286  //v150 9k
        }

        {  // SOH Halo splits
            std::cout << "\n SOH H SPLITS, not from Layer:  ";

            case_run(wl_h17_Top_SOH_K64x4,
                     "\nTEST of: SOH H TOP K=64  X4\n");  // v16 25719  v17: 25484:37151  //v150 28k
            case_run(wl_h17_Top_SOH_K32x1,
                     "\nTEST of: SOHH TOP K=32  X1\n");  // v16 13072  v17: 13138:17528  //v150 13k
            case_run(wl_h17_Bot_SOH_K16x18, "\nTEST of:SOHH BOT K=16 X18\n");  // v16 4693   v17: 4627:7195  //v150 5281
        }

        {  // SOH H from SOHO
            std::cout << "\n SOH H like SPLITS equivalent of SOHO from Layer  (same compute , but isi SOH)";

            case_run(wl_h17_Top_SOHH_K64x4,
                     "\nTEST of: SOH H from SOHO TOP K=64  X4\n");  // v16 25719  v17:25484:37151   //v150 28k
            case_run(wl_h17_Top_SOHH_K32x1,
                     "\nTEST of: SOH H from SOHO  TOP K=32  X1\n");  // v16 13072  v17:13138:17528   //v150 13k
            case_run(wl_h17_Bot_SOHH_K64x4,
                     "\nTEST of:SOH H from SOHO  BOT K=64 X4\n");  // v16 19045  v17:18012:31000   //v150 21k
            case_run(wl_h17_Bot_SOHH_K32x1,
                     "\nTEST of:SOH H from SOHO  BOT K=32 x1 \n");  // v16 9605   v17:9042:15035   //v150 10k
        }
    }
}

TEST_F(VPULayerCM_InvestigationTestVPU2x, CONV_TILE_Mineeva_EISXW_9xxxxx) {
    // Investigation on October 2023, older VPU version used. refresh provided
    const DPUWorkload wl_T1_3{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(512, 172, 4, 1, DataType::UINT8)},     // input dimensions
            {VPUTensor(512, 171, 16, 1, DataType::FLOAT16)},  // output dimensions
            {3, 3},                                           // kernels
            {1, 1},                                           // strides
            {1, 0, 1, 1},                                     // padding Top, Bottom, Left,  Right
            VPUNN::ExecutionMode::CUBOID_16x16,               // execution mode
            VPUNN::ActivationFunction::NONE,                  // activation
            0.0F,                                             // act_sparsity
            0.0F,                                             // weight_sparsity
            {swz_def, swz_def},                               // input_swizzling
            {swz_def},                                        // output_swizzling
            1,                                                // output_write_tiles
            {0, 0, 0, 0},                                     // offsets
            VPUNN::ISIStrategy::CLUSTERING,                   // isi_strategy
            false,                                            // weight_sparsity_enabled
    };
    const DPUWorkload wl_T2_3{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(512, 173, 4, 1, DataType::UINT8)},     // input dimensions
            {VPUTensor(512, 171, 16, 1, DataType::FLOAT16)},  // output dimensions
            {3, 3},                                           // kernels
            {1, 1},                                           // strides
            {0, 0, 1, 1},                                     // padding Top, Bottom, Left,  Right
            VPUNN::ExecutionMode::CUBOID_16x16,               // execution mode
            VPUNN::ActivationFunction::NONE,                  // activation
            0.0F,                                             // act_sparsity
            0.0F,                                             // weight_sparsity
            {swz_def, swz_def},                               // input_swizzling
            {swz_def},                                        // output_swizzling
            1,                                                // output_write_tiles
            {0, 0, 0, 0},                                     // offsets
            VPUNN::ISIStrategy::CLUSTERING,                   // isi_strategy
            false,                                            // weight_sparsity_enabled
    };
    const DPUWorkload wl_T3_3{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(512, 171, 4, 1, DataType::UINT8)},     // input dimensions
            {VPUTensor(512, 170, 16, 1, DataType::FLOAT16)},  // output dimensions
            {3, 3},                                           // kernels
            {1, 1},                                           // strides
            {0, 1, 1, 1},                                     // padding
            VPUNN::ExecutionMode::CUBOID_16x16,               // execution mode
            VPUNN::ActivationFunction::NONE,                  // activation
            0.0F,                                             // act_sparsity
            0.0F,                                             // weight_sparsity
            {swz_def, swz_def},                               // input_swizzling
            {swz_def},                                        // output_swizzling
            1,                                                // output_write_tiles
            {0, 0, 0, 0},                                     // offsets
            VPUNN::ISIStrategy::CLUSTERING,                   // isi_strategy
            false,                                            // weight_sparsity_enabled
    };

    ///////////////////
    const DPUWorkload wl_T1_4{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(512, 129, 4, 1, DataType::UINT8)},     // input dimensions
            {VPUTensor(512, 128, 16, 1, DataType::FLOAT16)},  // output dimensions
            {3, 3},                                           // kernels
            {1, 1},                                           // strides
            {1, 0, 1, 1},                                     // padding Top, Bottom, Left,  Right
            VPUNN::ExecutionMode::CUBOID_16x16,               // execution mode
            VPUNN::ActivationFunction::NONE,                  // activation
            0.0F,                                             // act_sparsity
            0.0F,                                             // weight_sparsity
            {swz_def, swz_def},                               // input_swizzling
            {swz_def},                                        // output_swizzling
            1,                                                // output_write_tiles
            {0, 0, 0, 0},                                     // offsets
            VPUNN::ISIStrategy::CLUSTERING,                   // isi_strategy
            false,                                            // weight_sparsity_enabled
    };
    const DPUWorkload wl_T2_4{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(512, 130, 4, 1, DataType::UINT8)},     // input dimensions
            {VPUTensor(512, 128, 16, 1, DataType::FLOAT16)},  // output dimensions
            {3, 3},                                           // kernels
            {1, 1},                                           // strides
            {0, 0, 1, 1},                                     // padding Top, Bottom, Left,  Right
            VPUNN::ExecutionMode::CUBOID_16x16,               // execution mode
            VPUNN::ActivationFunction::NONE,                  // activation
            0.0F,                                             // act_sparsity
            0.0F,                                             // weight_sparsity
            {swz_def, swz_def},                               // input_swizzling
            {swz_def},                                        // output_swizzling
            1,                                                // output_write_tiles
            {0, 0, 0, 0},                                     // offsets
            VPUNN::ISIStrategy::CLUSTERING,                   // isi_strategy
            false,                                            // weight_sparsity_enabled
    };
    const DPUWorkload wl_T3_4{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(512, 130, 4, 1, DataType::UINT8)},     // input dimensions
            {VPUTensor(512, 128, 16, 1, DataType::FLOAT16)},  // output dimensions
            {3, 3},                                           // kernels
            {1, 1},                                           // strides
            {0, 0, 1, 1},                                     // padding Top, Bottom, Left,  Right
            VPUNN::ExecutionMode::CUBOID_16x16,               // execution mode
            VPUNN::ActivationFunction::NONE,                  // activation
            0.0F,                                             // act_sparsity
            0.0F,                                             // weight_sparsity
            {swz_def, swz_def},                               // input_swizzling
            {swz_def},                                        // output_swizzling
            1,                                                // output_write_tiles
            {0, 0, 0, 0},                                     // offsets
            VPUNN::ISIStrategy::CLUSTERING,                   // isi_strategy
            false,                                            // weight_sparsity_enabled
    };
    const DPUWorkload wl_T4_4{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(512, 129, 4, 1, DataType::UINT8)},     // input dimensions
            {VPUTensor(512, 128, 16, 1, DataType::FLOAT16)},  // output dimensions
            {3, 3},                                           // kernels
            {1, 1},                                           // strides
            {1, 0, 1, 1},                                     // padding Top, Bottom, Left,  Right
            VPUNN::ExecutionMode::CUBOID_16x16,               // execution mode
            VPUNN::ActivationFunction::NONE,                  // activation
            0.0F,                                             // act_sparsity
            0.0F,                                             // weight_sparsity
            {swz_def, swz_def},                               // input_swizzling
            {swz_def},                                        // output_swizzling
            1,                                                // output_write_tiles
            {0, 1, 0, 0},                                     // offsets
            VPUNN::ISIStrategy::CLUSTERING,                   // isi_strategy
            false,                                            // weight_sparsity_enabled
    };

    // const std::string altModelName(NameHelperNN::get_model_root() + "vpu_2_7-150.vpunn");
    // VPULayerCostModel model_alternative{altModelName};
    // const bool prefetch{true};
    VPULayerCostModel& theModel = layer_models.getModel(VPUDevice::VPU_2_7);

    using LayerResult = std::pair<std::string, CyclesInterfaceType>;
    using Results = std::vector<LayerResult>;

    auto executeLayer = [&theModel](const DPUWorkload& wl, const std::string whatT, const VPULayerStrategy& stateg,
                                    Results& res) {
        VPUNN::DPULayer tst_layer(wl);
        std::string whatTest{"\n" + whatT + "\n"};

        Logger::clear2ndlog();
        CyclesInterfaceType cost_cyc{};
        LayerSplitInfo detailed_split;
        ASSERT_NO_THROW(cost_cyc = theModel.Layer(tst_layer, stateg.tiling_strategy, stateg.nDPUs, stateg.nTiles,
                                                  stateg.input_fetching, stateg.output_spilling, stateg.prefetching,
                                                  detailed_split))
                << whatTest << tst_layer << stateg << cost_cyc;
        res.emplace_back(std::make_pair(whatT, cost_cyc));
        // EXPECT_EQ(cost_cyc, 1) << whatTest << tst_layer << toStringLayerSplitInfo(detailed_split);  //<<
        //   Logger::get2ndlog();//99639
        //  return cost_cyc;
    };

    {
        const VPULayerStrategy strategy{1U,    1U,    2U,  VPUNN::VPUTilingStrategy::SOH_Overlapped,
                                        false, false, true /*prefetch*/};
        std::cout << "\n xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx Run 1   "
                     "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n";

        Results r3T;
        executeLayer(wl_T1_3, "TEST of:wl_T1_3", strategy, r3T);
        executeLayer(wl_T2_3, "TEST of:wl_T2_3", strategy, r3T);
        executeLayer(wl_T3_3, "TEST of:wl_T3_3", strategy, r3T);
        //------
        Results r4T;
        executeLayer(wl_T1_4, "TEST of:wl_T1_4", strategy, r4T);
        executeLayer(wl_T2_4, "TEST of:wl_T2_4", strategy, r4T);
        executeLayer(wl_T3_4, "TEST of:wl_T3_4", strategy, r4T);
        executeLayer(wl_T4_4, "TEST of:wl_T4_4", strategy, r4T);

        //----------------------------
        std::cout << "\n ------------- Prefetch : " << strategy.prefetching << "-------------------------";

        {
            CyclesInterfaceType total = 0;
            std::cout << "\n ------------- 3 Tile Results ------------------";
            for (const auto& i : r3T) {
                std::cout << "\n " << i.first << " cycles: " << i.second;
                total = total + i.second;
            }
            std::cout << "\n TOTAL " << total;
        }
        {
            CyclesInterfaceType total = 0;
            std::cout << "\n ------------- 4 Tile Results ------------------";
            for (const auto& i : r4T) {
                std::cout << "\n " << i.first << " cycles: " << i.second;
                total = total + i.second;
            }
            std::cout << "\n TOTAL " << total;
        }
    }
    {
        const VPULayerStrategy strategy{1U,    1U,    2U,   VPUNN::VPUTilingStrategy::SOH_Overlapped,
                                        false, false, false /*prefetch*/};
        std::cout << "\n xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx Run 1   "
                     "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n";

        Results r3T;
        executeLayer(wl_T1_3, "TEST of:wl_T1_3", strategy, r3T);
        executeLayer(wl_T2_3, "TEST of:wl_T2_3", strategy, r3T);
        executeLayer(wl_T3_3, "TEST of:wl_T3_3", strategy, r3T);
        //------
        Results r4T;
        executeLayer(wl_T1_4, "TEST of:wl_T1_4", strategy, r4T);
        executeLayer(wl_T2_4, "TEST of:wl_T2_4", strategy, r4T);
        executeLayer(wl_T3_4, "TEST of:wl_T3_4", strategy, r4T);
        executeLayer(wl_T4_4, "TEST of:wl_T4_4", strategy, r4T);

        //----------------------------
        std::cout << "\n ------------- Prefetch : " << strategy.prefetching << "-------------------------";
        {
            CyclesInterfaceType total = 0;
            std::cout << "\n ------------- 3 Tile Results ------------------";
            for (const auto& i : r3T) {
                std::cout << "\n " << i.first << " cycles: " << i.second;
                total = total + i.second;
            }
            std::cout << "\n TOTAL " << total;
        }
        {
            CyclesInterfaceType total = 0;
            std::cout << "\n ------------- 4 Tile Results ------------------";
            for (const auto& i : r4T) {
                std::cout << "\n " << i.first << " cycles: " << i.second;
                total = total + i.second;
            }
            std::cout << "\n TOTAL " << total;
        }
    }

    //{
    //    VPUNN::DPULayer tst_layer(wl_T1_3);
    //    std::string whatTest{"\nTEST of:wl_T1_3 \n"};

    //    Logger::clear2ndlog();
    //    CyclesInterfaceType cost_cyc{};
    //    LayerSplitInfo detailed_split;
    //    ASSERT_NO_THROW(cost_cyc = theModel.Layer(tst_layer, strategy.tiling_strategy, strategy.nDPUs,
    //    strategy.nTiles,
    //                                              false, false, prefetch, detailed_split))
    //            << whatTest << tst_layer << strategy << cost_cyc;
    //    EXPECT_EQ(cost_cyc, 1) << whatTest << tst_layer << toStringLayerSplitInfo(detailed_split);  //<<
    //    //  Logger::get2ndlog();//99639
    //}
}

TEST_F(VPULayerCM_InvestigationTestVPU2x, RuntimeELT_CONV_SOH_SOK_EISXW_98656) {
    auto gen_eltwise = [](const int wi, const int hi, const int ci, const ISIStrategy isi, const int owt,
                          const ExecutionMode em) {
        const DPUWorkload wl_elm_layer{
                // real swizz is probably 0,0,5.
                VPUDevice::VPU_2_7,
                Operation::ELTWISE,
                {VPUTensor(wi, hi, ci, 1, DataType::UINT8)},  // input dimensions
                {VPUTensor(wi, hi, ci, 1, DataType::UINT8)},  // output dimensions
                {1, 1},                                       // kernels
                {1, 1},                                       // strides
                {0, 0, 0, 0},                                 // padding
                em,                                           // ExecutionMode::CUBOID_16x16,   // execution mode
                ActivationFunction::NONE,                     // activation
                0.0F,                                         // act_sparsity
                0.0F,                                         // weight_sparsity
                {Swizzling::KEY_0, Swizzling::KEY_0},         // input_swizzling
                {Swizzling::KEY_0},                           // output_swizzling
                (unsigned int)owt,                            // output_write_tiles
                {0, 0, 0, 0},                                 // offsets
                isi,                                          // isi_strategy
                false,                                        // weight_sparsity_enabled
        };
        // clang and gcc does not support to use std::move here, so we need suppression
        /* coverity[copy_instead_of_move] */
        return wl_elm_layer;
    };

    auto gen_conv = [](const int wi, const int hi, const int ci, const int co, const ISIStrategy isi, const int owt,
                       const ExecutionMode em) {
        const DPUWorkload wl_elm_layer{
                VPUDevice::VPU_2_7,
                Operation::CONVOLUTION,
                {VPUTensor(wi, hi, ci, 1, DataType::UINT8)},  // input dimensions
                {VPUTensor(wi, hi, co, 1, DataType::UINT8)},  // output dimensions
                {1, 1},                                       // kernels
                {1, 1},                                       // strides
                {0, 0, 0, 0},                                 // padding
                em,                                           // ExecutionMode::CUBOID_16x16,     // execution mode
                ActivationFunction::NONE,                     // activation
                0.0F,                                         // act_sparsity
                0.259766F,                                    // weight_sparsity
                {swz_def, swz_def},                           // input_swizzling
                {swz_def},                                    // output_swizzling
                (unsigned int)owt,                            // output_write_tiles
                {0, 0, 0, 0},                                 // offsets
                isi,                                          // ISIStrategy::CLUSTERING,      // isi_strategy
                true,                                         // weight_sparsity_enabled
        };
        // clang and gcc does not support to use std::move here, so we need suppression
        /* coverity[copy_instead_of_move] */
        return wl_elm_layer;
    };

    // element wise SWIZZ in 0 at input and 5 at output in the real world.Cannot simulate mix swizzlings, not
    // trained
    const DPUWorkload wl_elm_layer{gen_eltwise(14, 14, 1024, ISIStrategy::CLUSTERING, 1, ExecutionMode::CUBOID_16x16)};
    const DPUWorkload wl_conv_layer{
            gen_conv(14, 14, 1024, 256, ISIStrategy::CLUSTERING, 1, ExecutionMode::CUBOID_16x16)};

    const bool prefetch{true};
    VPULayerCostModel& theModel = layer_models.getModel(VPUDevice::VPU_2_7);
    const std::string nline{"\n ------------- NEW TEST------------------------------------ ------------------\n"};
    const bool force_fail{false};

    // EXPECT_TRUE(false);

    auto run_layer = [=, &theModel](const DPUWorkload& layer, const VPUTilingStrategy tilStrtgy, std::string text) {
        std::cout << nline << " " << text;
        DPULayer tst_layer(layer);
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
    run_layer(wl_elm_layer, VPUTilingStrategy::SOH_Overlapped,
              "emlwise_SOHo");  // v16:2998 v17:3232  GTvpux 3169   GT SOH?:7315 ????
    run_layer(wl_elm_layer, VPUTilingStrategy::NONE,
              "emlwise_CLUSTERING");  // v16:6138 v17:7198 (5875 if swizzON)  GTvpux 13000 GT : swiz: 000: 16820 , swiz:
                                      // 005 : 12903cc, swizz505: 16463; swizz: 555: 6329

    // CONV
    run_layer(wl_conv_layer, VPUTilingStrategy::SOH_Overlapped,
              "conv_SOHo");                                                // v16: 13613 (14x7) v17: 13623  GTvpux 14764
    run_layer(wl_conv_layer, VPUTilingStrategy::SOK, "conv_SOK");          // v16: 14045        v17: 14067  GTvpux 14695
    run_layer(wl_conv_layer, VPUTilingStrategy::NONE, "conv_clustering");  // v16: 26370        v17: 26864
    run_layer(wl_conv_layer, VPUTilingStrategy::SOH_HaloRead, "conv_SOHH nonsense k=1");  // v16: 16433  v17: 17656  GT:

    // this layers are reproducing variants of split layers
    const ExecutionMode em{ExecutionMode::CUBOID_8x16};

    const DPUWorkload wl_elm_full{
            gen_eltwise(14, 14, 1024, ISIStrategy::CLUSTERING, 1, em)};  //      v16: 6138   v17: 7198
    const DPUWorkload wl_elm_SOH_clu_top{
            gen_eltwise(14, 8, 1024, ISIStrategy::CLUSTERING, 1, em)};  //      v16: 3056   v17: 3435
    const DPUWorkload wl_elm_SOH_clu_bottom{
            gen_eltwise(14, 6, 1024, ISIStrategy::CLUSTERING, 1, em)};  //   v16: 2894  v17: 3191

    const DPUWorkload wl_conv_SOK_1and2{
            gen_conv(14, 14, 1024, 128, ISIStrategy::SPLIT_OVER_K, 2, em)};  //  v16: 16252  v17: 15304

    const DPUWorkload wl_conv_SOH_clu_TOP{
            gen_conv(14, 8, 1024, 256, ISIStrategy::CLUSTERING, 1, em)};  //   v16: 14866   v17:14776
    const DPUWorkload wl_conv_SOH_clu_BTM{
            gen_conv(14, 6, 1024, 256, ISIStrategy::CLUSTERING, 1, em)};  // v16: 14746   v17: 14468

    const DPUWorkload wl_conv_SOH_soh_TOP{
            gen_conv(14, 8, 1024, 256, ISIStrategy::SPLIT_OVER_H, 1, em)};  // v16: 16415   v17:18919
    const DPUWorkload wl_conv_SOH_soh_BTM{
            gen_conv(14, 6, 1024, 256, ISIStrategy::SPLIT_OVER_H, 1, em)};  // v16: 16503   v17:17383

    std::cout << "\n-------------- WORKLOADS TESTS------------";

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

    run_dpu_wl(wl_elm_full, "elmnt full");
    run_dpu_wl(wl_elm_SOH_clu_top, "elmws TOP clu");
    run_dpu_wl(wl_elm_SOH_clu_bottom, "elmws BOT clu");

    run_dpu_wl(wl_conv_SOK_1and2, "CONV SOK both");
    run_dpu_wl(wl_conv_SOH_clu_TOP, "CONV SOH clu TOP");
    run_dpu_wl(wl_conv_SOH_clu_BTM, "CONV SOH clu BTM");

    run_dpu_wl(wl_conv_SOH_soh_TOP, "CONV SOH soh TOP");
    run_dpu_wl(wl_conv_SOH_soh_BTM, "CONV SOH soh BTM");
}

// this is a test fro the situation of K=4096 ans when doing intratile splits the N=50 limit of max splits does not
// allow the algo to reach a workload with K=64. Reason 4096/64>50
TEST_F(VPULayerCM_InvestigationTestVPU2x, Layer_MAXP_EISXW_na_MINGQI_NPU27) {
    const bool force_fail{};             // controls force failing assertion
    const int fail{force_fail ? 0 : 1};  // 1 neutral, 0 fail
    show_split = true;
    // EXPECT_TRUE(false);

    const DPUWorkload wl_{
            VPUDevice::VPU_2_7,
            Operation::MAXPOOL,
            {VPUTensor(40, 8, 4096, 1, DataType::UINT8, Layout::ZXY)},  // input dimensions
            {VPUTensor(40, 8, 4096, 1, DataType::UINT8, Layout::ZXY)},  // output dimensions
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
            // zeroHalo,                                                  // halo
            // sepInfo,                                                   // SEP
    };

    const bool prefetch{true};

    {  // note:
        const DPULayer tst_layer(wl_);
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_HaloRead, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 22000,
                  fail * 22000 + 2200},  // 2// V17: x   v159NN: ? GTVPUX:   GTL: ??
                 "SOHh , no memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 22000,
                  fail * 22000 + 2200},  // 2// V17: x   v159NN: ? GTVPUX:   GTL: ??
                 "SOHO , no memmove, "},

                // note:?? wins
        };

        const auto origMax = layer_models.getModel(VPUDevice::VPU_2_7).get_maxWorkloadsPerIntraTileSplit();
        layer_models.getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(64);
        executeTests(tests);
        layer_models.getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(origMax);
    }

    VPULayerCostModel& theModel = layer_models.getModel(VPUDevice::VPU_2_7);

    auto case_run = [=, &theModel](const DPUWorkload& workload, const std::string& whatTest) {
        DPUWorkload tst_layer(workload);
        std::string err_info;

        CyclesInterfaceType dpu_cost = theModel.get_cost_model().DPU(tst_layer, err_info);
        if (force_fail) {
            EXPECT_EQ(dpu_cost, 10) << whatTest << dpu_cost << "=" << Cycles::toErrorText(dpu_cost) << "\n"
                                    << tst_layer << err_info << "END ERR";
        }
        std::cout << "  " << whatTest << dpu_cost << "=" << Cycles::toErrorText(dpu_cost) << "\n";
    };

    //     EXPECT_TRUE(on);  // something has to fail to  see couts

    // low level WL
    {
        const DPUWorkload wl_SOHH_invalid_{
                VPUDevice::VPU_2_7,
                Operation::MAXPOOL,
                {VPUTensor(40, 4, 4096, 1, DataType::INT8, Layout::ZXY)},  // input dimensions
                {VPUTensor(40, 4, 4096, 1, DataType::INT8, Layout::ZXY)},  // output dimensions
                {1, 1},                                                    // kernels
                {1, 1},                                                    // strides
                {0, 0, 0, 0},                                              // padding
                ExecutionMode::CUBOID_16x16,                               // execution mode
                ActivationFunction::NONE,                                  // activation
                0.0F,                                                      // act_sparsity
                0.0F,                                                      // weight_sparsity
                {swz_def, swz_def},                                        // input_swizzling
                {swz_def},                                                 // output_swizzling
                1,                                                         // output_write_tiles
                {0, 0, 0, 0},                                              // offsets
                ISIStrategy::SPLIT_OVER_H,                                 // isi_strategy
                false,                                                     // weight_sparsity_enabled
                // zeroHalo,                                                  // halo
                // sepInfo,                                                   // SEP
        };
        const DPUWorkload wl_SOHH_K64_{

                VPUDevice::VPU_2_7,
                Operation::MAXPOOL,
                {VPUTensor(40, 4, 64, 1, DataType::INT8, Layout::ZXY)},  // input dimensions
                {VPUTensor(40, 4, 64, 1, DataType::INT8, Layout::ZXY)},  // output dimensions
                {1, 1},                                                  // kernels
                {1, 1},                                                  // strides
                {0, 0, 0, 0},                                            // padding
                ExecutionMode::CUBOID_16x16,                             // execution mode
                ActivationFunction::NONE,                                // activation
                0.0F,                                                    // act_sparsity
                0.0F,                                                    // weight_sparsity
                {swz_def, swz_def},                                      // input_swizzling
                {swz_def},                                               // output_swizzling
                1,                                                       // output_write_tiles
                {0, 0, 0, 0},                                            // offsets
                ISIStrategy::SPLIT_OVER_H,                               // isi_strategy
                false,                                                   // weight_sparsity_enabled
        };

        {
            Logger::clear2ndlog();
            std::string err_info;

            {  // SOHO SPLITS
                std::cout << "\n ------- SOHO SPLITS: ------- \n";

                case_run(wl_SOHH_invalid_, "TEST of: SOHh  K=4096");  // V17: x   v159NN: ? GTVPUX:   GTL: ??
                case_run(wl_SOHH_K64_, "TEST of: SOHh K=64 ");
            }
        }
    }
}

class VPULayerInvstgt_EISXW_119193_Deeplab_v3 : public VPULayerCM_InvestigationTestVPU2x {
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

    bool force_fail{false};  // controls force failing assertion
    bool simple_fail_all{false};

    auto case_run(const DPUWorkload& workload, const std::string& whatTest) {
        DPUWorkload tst_layer(workload);
        std::string err_info;

        CyclesInterfaceType dpu_cost =
                layer_models.getModel(VPUDevice::VPU_2_7).get_cost_model().DPU(tst_layer, err_info);
        if (force_fail) {
            EXPECT_EQ(dpu_cost, 10) << whatTest << dpu_cost << "=" << Cycles::toErrorText(dpu_cost) << "\n"
                                    << tst_layer << err_info << "END ERR";
        }
        std::cout << "  " << whatTest << dpu_cost << "=" << Cycles::toErrorText(dpu_cost) << "\n";
    };

    // HALO variants
    const HaloWorkload h_zero{{0, 0, 0, 0, 0, 0},  // H in TBLRFB
                              {0, 0, 0, 0, 0, 0},
                              {0, 0, 0, 0, 0, 0},
                              {0, 0, 0, 0, 0, 0}};

    const HaloWorkload::HaloInfoHWC h_in_topTile_4{0, 4, 0, 0, 0, 0};  // H in TBLRFB
    const HaloWorkload::HaloInfoHWC h_in_botTile_4{4, 0, 0, 0, 0, 0};  // H in TBLRFB

    const DPUWorkload base_wl_CLU{
            // orig Layer
            VPUDevice::VPU_2_7,
            Operation::DW_CONVOLUTION,
            {VPUTensor(65, 65, 960, 1, DataType::FLOAT16)},  // input dimensions
            {VPUTensor(65, 65, 960, 1, DataType::FLOAT16)},  // output dimensions
            {9, 9},                                          // kernels
            {1, 1},                                          // strides
            {4, 4, 4, 4},                                    // padding
            ExecutionMode::CUBOID_16x16,                     // execution mode
            ActivationFunction::NONE,                        // activation
            0.0F,                                            // act_sparsity
            0.0F,                                            // weight_sparsity
            {swz_def, swz_def},                              // input_swizzling
            {swz_def},                                       // output_swizzling
            1,                                               // output_write_tiles
            {0, 0, 0, 0},                                    // offsets
            ISIStrategy::CLUSTERING,                         // isi_strategy
            false,                                           // weight_sparsity_enabled
    };

    const DPUWorkload base_wl_SOK{
            VPUDevice::VPU_2_7,
            Operation::DW_CONVOLUTION,
            {VPUTensor(65, 11, 64, 1, DataType::FLOAT16)},  // input dimensions
            {VPUTensor(65, 3, 64, 1, DataType::FLOAT16)},   // output dimensions
            {9, 9},                                         // kernels
            {1, 1},                                         // strides
            {0, 0, 4, 4},                                   // padding
            ExecutionMode::CUBOID_16x16,                    // execution mode
            ActivationFunction::NONE,                       // activation
            0.0F,                                           // act_sparsity
            0.0F,                                           // weight_sparsity
            {swz_def, swz_def},                             // input_swizzling
            {swz_def},                                      // output_swizzling
            2,                                              // output_write_tiles
            {0, 0, 0, 0},                                   // offsets
            ISIStrategy::SPLIT_OVER_K,                      // isi_strategy
            false,                                          // weight_sparsity_enabled
                                                            // no incoming halo
    };

    const DPUWorkload base_wl_SOHH{
            // temporal tile 1-6, Top tile  after SOHH split,  intratule K64x15 = 960
            VPUDevice::VPU_2_7,
            Operation::DW_CONVOLUTION,
            {VPUTensor(65, 12, 64, 1, DataType::FLOAT16)},  // input dimensions
            {VPUTensor(65, 4, 64, 1, DataType::FLOAT16)},   // output dimensions
            {9, 9},                                         // kernels
            {1, 1},                                         // strides
            {0, 0, 4, 4},                                   // padding
            ExecutionMode::CUBOID_16x16,                    // execution mode
            ActivationFunction::NONE,                       // activation
            0.0F,                                           // act_sparsity
            0.0F,                                           // weight_sparsity
            {swz_def, swz_def},                             // input_swizzling
            {swz_def},                                      // output_swizzling
            1,                                              // output_write_tiles
            {0, 0, 0, 0},                                   // offsets
            ISIStrategy::SPLIT_OVER_H,                      // isi_strategy
            false,                                          // weight_sparsity_enabled
    };

    /// with this function, starting from a base wl, you can create a new workload with the values you provide as
    /// parameters
    /// @param base_wl: the base wl you want to modify
    /// @param in_h: the input tensor height
    /// @param in_c: the input tensor channels
    /// @param out_h: output tensor height
    /// @param out_c: output tensor channels
    /// @param top_padd: top padding for workload
    /// @param btm_padd: bottom padding for workload
    /// @param input0_halo: input_0_halo information for workload
    /// @return a new DPUWorkload
    DPUWorkload makeWL(const DPUWorkload& base_wl, unsigned int in_h, unsigned int in_c, unsigned int out_h,
                       unsigned int out_c, unsigned int top_padd, unsigned int btm_padd,
                       HaloWorkload::HaloInfoHWC input0_halo = {0, 0, 0, 0, 0, 0}) const {
        DPUWorkload ret_wl{base_wl};

        // input tensor dimensions WHCB
        ret_wl.inputs[0].set_shape({base_wl.inputs[0].get_shape()[0], in_h, in_c, base_wl.inputs[0].get_shape()[3]});
        // output tensor dimensions WHCB
        ret_wl.outputs[0].set_shape(
                {base_wl.outputs[0].get_shape()[0], out_h, out_c, base_wl.outputs[0].get_shape()[3]});

        // padding
        ret_wl.padding[Dim::TOP] = top_padd;
        ret_wl.padding[Dim::BOTTOM] = btm_padd;

        // input halo
        ret_wl.halo.input_0_halo = input0_halo;

        return ret_wl;
    }

    // orig Layer
    const DPUWorkload wl_full{makeWL(base_wl_CLU, 65, 960, 65, 960, 4, 4)};

    //  note Temporal Tiles are always overlapped (SOHO)

    /// SOK wing Layers
    // temporal tile 0, layer level, SOK  wing
    const DPUWorkload wl_Ktt0{makeWL(base_wl_CLU, 7, 960 /*64 or 32*/, 3, 960 /*64 or 32*/, 4, 0)};

    // temporal tile 1, layer level, SOK  wing
    const DPUWorkload wl_Ktt1{makeWL(base_wl_CLU, 10, 960 /*64 or 32*/, 3, 960 /*64 or 32*/, 1, 0)};

    // temporal tile 2, 3-19, layer level, SOK  wing
    const DPUWorkload wl_Ktt2{makeWL(base_wl_CLU, 11, 960 /*64 or 32*/, 3, 960 /*64 or 32*/, 0, 0)};

    // temporal tile 22, layer level, SOK  wing
    const DPUWorkload wl_Ktt22{makeWL(base_wl_CLU, 9, 960 /*64 or 32*/, 3, 960 /*64 or 32*/, 0, 2)};

    // temporal tile 23, layer level, SOK  wing
    const DPUWorkload wl_Ktt23{makeWL(base_wl_CLU, 6, 960 /*64 or 32*/, 2, 960 /*64 or 32*/, 0, 4)};

    /// SOH wing layers
    // temporal tile 0, layer level, SOH H  wing
    const DPUWorkload wl_Htt0{makeWL(base_wl_CLU, 13, 960 /*64x15*/, 4 + 4 + 1, 960 /*64x15*/, 4, 0)};

    // temporal tile 1-6, layer level, SOH H  wing
    const DPUWorkload wl_Htt1{makeWL(base_wl_CLU, 16, 960 /*64x15*/, 4 + 4, 960 /*64x15*/, 0, 0)};

    // temporal tile 7, layer level, SOH H  wing
    const DPUWorkload wl_Htt7{makeWL(base_wl_CLU, 12, 960 /*64x15*/, 4 + 4, 960 /*64x15*/, 0, 4)};

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // workloads most used
    // SOK WING
    // SOK split , top=bottom K64 x7
    const DPUWorkload wl_Ktt2_TB_SOK64x7{makeWL(base_wl_SOK, 11, 64 /*64 or 32*/, 3, 64 /*64 or 32*/, 0, 0)};

    // SOK split , top=bottom K32 x1
    const DPUWorkload wl_Ktt2_TB_SOK32x1{makeWL(base_wl_SOK, 11, 32 /*64 or 32*/, 3, 32 /*64 or 32*/, 0, 0)};

    /// SOH  wing
    // temporal tile 1-6, Top tile  after SOHH split,  intratule K64x15 = 960
    const DPUWorkload wl_Htt1_Top_SOHH_K64x15{
            makeWL(base_wl_SOHH, 12, 64 /*64x15*/, 4, 64 /*64x15*/, 0, 0, h_in_topTile_4)};

    // temporal tile 1-6, Top tile  after SOHH split,  intratile K64x15 = 960
    const DPUWorkload wl_Htt1_Bot_SOHH_K64x15{
            makeWL(base_wl_SOHH, 12, 64 /*64x15*/, 4, 64 /*64x15*/, 0, 0, h_in_botTile_4)};

    // k32
    // temporal tile 1-6, Top tile  after SOHH split,  intratule K64x15 = 960
    const DPUWorkload wl_Htt1_Top_SOHH_K32x30{makeWL(base_wl_SOHH, 12, 32, 4, 32, 0, 0, h_in_topTile_4)};

    // temporal tile 1-6, Top tile  after SOHH split,  intratile K64x15 = 960
    const DPUWorkload wl_Htt1_Bot_SOHH_K32x30{makeWL(base_wl_SOHH, 12, 32, 4, 32, 0, 0, h_in_botTile_4)};

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // first workloads
    // SOK WING
    // SOK split , top=bottom K64 x7
    const DPUWorkload wl_Ktt0_TB_SOK64x7{makeWL(base_wl_SOK, 7, 64 /*64 or 32*/, 3, 64 /*64 or 32*/, 4, 0)};

    // SOK split , top=bottom K32 x1
    const DPUWorkload wl_Ktt0_TB_SOK32x1{makeWL(base_wl_SOK, 7, 32 /*64 or 32*/, 3, 32 /*64 or 32*/, 4, 0)};

    // SOK split , top=bottom K64 x7
    const DPUWorkload wl_Ktt1_TB_SOK64x7{makeWL(base_wl_SOK, 10, 64 /*64 or 32*/, 3, 64 /*64 or 32*/, 1, 0)};

    // SOK split , top=bottom K32 x1
    const DPUWorkload wl_Ktt1_TB_SOK32x1{makeWL(base_wl_SOK, 10, 32 /*64 or 32*/, 3, 32 /*64 or 32*/, 1, 0)};

    /// SOH  wing
    // temporal tile 0, Top tile  after SOHH split,  intratule K64x15 = 960
    const DPUWorkload wl_Htt0_Top_SOHH_K64x15{
            makeWL(base_wl_SOHH, 9, 64 /*64x15*/, 5, 64 /*64x15*/, 4, 0, h_in_topTile_4)};

    // temporal tile 0, Btm tile  after SOHH split,  intratile K64x15 = 960
    const DPUWorkload wl_Htt0_Bot_SOHH_K64x15{
            makeWL(base_wl_SOHH, 12, 64 /*64x15*/, 4, 64 /*64x15*/, 0, 0, h_in_botTile_4)};
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // last workloads
    // SOK WING

    // SOK split , top=bottom K64 x7
    const DPUWorkload wl_Ktt22_TB_SOK64x7{makeWL(base_wl_SOK, 9, 64 /*64 or 32*/, 3, 64 /*64 or 32*/, 0, 2)};
    // SOK split , top=bottom K32 x1
    const DPUWorkload wl_Ktt22_TB_SOK32x1{makeWL(base_wl_SOK, 9, 32 /*64 or 32*/, 3, 32 /*64 or 32*/, 0, 2)};

    // SOK split , top=bottom K64 x7
    const DPUWorkload wl_Ktt23_TB_SOK64x7{makeWL(base_wl_SOK, 6, 64 /*64 or 32*/, 2, 64 /*64 or 32*/, 0, 4)};

    // SOK split , top=bottom K32 x1
    const DPUWorkload wl_Ktt23_TB_SOK32x1{makeWL(base_wl_SOK, 6, 32 /*64 or 32*/, 2, 32 /*64 or 32*/, 0, 4)};

    /// SOH  wing
    // temporal tile 0, Top tile  after SOHH split,  intratule K64x15 = 960
    const DPUWorkload wl_Htt7_Top_SOHH_K64x15{
            makeWL(base_wl_SOHH, 12, 64 /*64x15*/, 4, 64 /*64x15*/, 0, 0, h_in_topTile_4)};

    // temporal tile 0, Btm tile  after SOHH split,  intratile K64x15 = 960
    const DPUWorkload wl_Htt7_Bot_SOHH_K64x15{
            makeWL(base_wl_SOHH, 8, 64 /*64x15*/, 4, 64 /*64x15*/, 0, 4, h_in_botTile_4)};
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
};

// Test made by JB ans AS, on VPU2.7 showing soh/SOK flip
// SOHO is only first layer, rest is SOHH
TEST_F(VPULayerInvstgt_EISXW_119193_Deeplab_v3, Most_used_temporalTiles_SOKtt2_19_SOHtt1_6) {
    const unsigned int fail{force_fail ? 0u : 1u};  // 1 neutral, 0 fail
    // show_split = true;
    //  EXPECT_TRUE(false);

    const bool prefetch{true};

    {
        // SOK wing
        std::cout << "\n ------- SOK  WING:   tt2-19 ------- \n";
        const DPULayer tst_layer(wl_Ktt2);  // has 3 output rows, what h split!
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 480000u,
                  /*fail * */ (480000u + 20000u)},  // v17:485910  GT:xx  (intratile:)
                                                    // VPUXVPUNN(old v):xx. v159NN: xx
                 "SOHO , no memmove, "},

                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, prefetch}},  // RELEVANT
                 {VPUNN::Cycles::NO_ERROR, true, 240000,
                  fail * (240000u + 20000)},  // v17:248143  GT:xx  (intratile K64x7 K32x1)
                                              // VPUXVPUNN(old v): xx, v159NN:xx   VPUXnn:264960(160)/242082(159)
                 "SOK , no memmove, "},

                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_HaloRead, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 800000,
                  /* fail * */ (800000u + 500000)},  // v17:822045:1264680    GT:? v159NN: xx
                 "SOH Halo , no memmove, "},

                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 480000, (480000u + 20000u)},  // v17:CLU:492045  GT:?  (intratile:)
                 "FUll , no memmove, "},

                // note:?? wins
        };
        executeTests(tests);
    }

    {
        // SOH  wing
        std::cout << "\n ------- SOH  WING:   tt1-6 ------- \n";
        const DPULayer tst_layer(wl_Htt1);  // h =8
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 1,
                  /*fail * */ (1u + 1u)},  // v17:ERROR_INPUT_TOO_BIG  GT:xx  (intratile: )   VPUXVPUNN(old
                                           // v):xx. v159NN: xx
                 "SOHO , no memmove, "},

                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, prefetch}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 1,
                  /*fail * */ (1u + 1)},  // v17: ERROR_INPUT_TOO_BIG  GT:xx  (intratile x1)
                                          // VPUXVPUNN(old v): xx, v159NN:xx
                 "SOK , no memmove, "},

                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_HaloRead, false, false, prefetch}},  // RELEVANT
                 {VPUNN::Cycles::NO_ERROR, true, 750000,
                  fail * (750000u + 750000)},  // v17:763575: 1390740 , (intratile:  K64x15:K32x30) GT:? v159NN: xx
                                               // . VPUXNN: 783870
                 "SOH Halo , no memmove, "},

                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, prefetch}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 1, (1 + 1)},  // v17: ERROR_INPUT_TOO_BIG  GT:?
                 "FUll , no memmove, "},

                // note:x wins
        };
        executeTests(tests);
    }

    // EXPECT_TRUE(false);  // something has to fail to  see couts
    EXPECT_FALSE(simple_fail_all);  // centrally

    // low level WL
    {
        // SOK wing
        // SOHO: NA
        //  SOK
        // wl_Ktt2_TB_SOK64x7;
        // wl_Ktt2_TB_SOK32x1
        DPUWorkload wl_Ktt2_TB_SOK64x7_CLU{wl_Ktt2_TB_SOK64x7};
        wl_Ktt2_TB_SOK64x7_CLU.isi_strategy = ISIStrategy::CLUSTERING;
        wl_Ktt2_TB_SOK64x7_CLU.output_write_tiles = 1;

        DPUWorkload wl_Ktt2_TB_SOK32x1_CLU{wl_Ktt2_TB_SOK32x1};
        wl_Ktt2_TB_SOK32x1_CLU.isi_strategy = ISIStrategy::CLUSTERING;
        wl_Ktt2_TB_SOK32x1_CLU.output_write_tiles = 1;
        // SOHH: NA

        // SOHH wing
        // wl_Htt1_Top_SOHH_K64x15;
        // wl_Htt1_Bot_SOHH_K64x15;

        {
            Logger::clear2ndlog();
            std::string err_info;

            {  // SOK SPLITS
                std::cout << "\n ------- SOK SPLITS on SOK wing: ------- ";

                case_run(wl_Ktt2_TB_SOK64x7, "\nTEST of: SOK  K=64 X7\n");  //   v17: 33021     GT xx   v159nn:xx
                case_run(wl_Ktt2_TB_SOK32x1, "\nTEST of: SOK  K=32 X1\n");  //   v17: 16996     GT xx   v159nn:xx
                // Top bot are in parallel (x7+x1) =  ()||() = ()|()  = xx
                // GT parallelism (2x)||(2x)||( 2x) = ()|( )  =xx

                case_run(wl_Ktt2_TB_SOK64x7_CLU,
                         "\nTEST of: CLUowt1  K=64 X7\n");  //   v17: 32803     GT xx v159nn:xx
                case_run(wl_Ktt2_TB_SOK32x1_CLU,
                         "\nTEST of: CLUowt1  K=32 X1\n");  //   v17: 16947     GT xx v159nn:xx
            }

            {  /// SOH H splits
                std::cout << "\n ------- SOH H SPLITS on SOH wing: ------- ";

                case_run(wl_Htt1_Top_SOHH_K64x15,
                         "\nTEST of: SOH H  K=64 X15\n");  //   v17: 50905:97771     GT xx v159nn:xx
                case_run(wl_Htt1_Bot_SOHH_K64x15,
                         "\nTEST of: SOH H  K=64 X15\n");  //   v17: 50905:97771     GT xx v159nn:xx

                case_run(wl_Htt1_Top_SOHH_K32x30,
                         "\nTEST of: SOH H  K=32 X30\n");  //   v17: x:46358     GT xx v159nn:xx
                case_run(wl_Htt1_Bot_SOHH_K32x30,
                         "\nTEST of: SOH H  K=32 X30\n");  //   v17: x:46358     GT xx v159nn:xx

                // SOHH : x
                // GT: xx
            }

            // who wins?
            // GT:
        }
    }
    {
        // SOHO forcing
        DPUWorkload wl_Top_SOHO_K64x15{wl_Htt1_Top_SOHH_K64x15};
        wl_Top_SOHO_K64x15.isi_strategy = ISIStrategy::CLUSTERING;
        wl_Top_SOHO_K64x15.halo = h_zero;

        DPUWorkload wl_Bot_SOHO_K64x15{wl_Htt1_Bot_SOHH_K64x15};
        wl_Bot_SOHO_K64x15.isi_strategy = ISIStrategy::CLUSTERING;
        wl_Bot_SOHO_K64x15.halo = h_zero;

        DPUWorkload wl_Top_SOHO_K32x30{wl_Htt1_Top_SOHH_K32x30};
        wl_Top_SOHO_K32x30.isi_strategy = ISIStrategy::CLUSTERING;
        wl_Top_SOHO_K32x30.halo = h_zero;

        DPUWorkload wl_Bot_SOHO_K32x30{wl_Htt1_Bot_SOHH_K32x30};
        wl_Bot_SOHO_K32x30.isi_strategy = ISIStrategy::CLUSTERING;
        wl_Bot_SOHO_K32x30.halo = h_zero;

        // SOHH fake : Compute tensor susbstituted by memory tensor
        DPUWorkload wl_Top_SOHH_K64x15_fake{wl_Htt1_Top_SOHH_K64x15};
        wl_Top_SOHH_K64x15_fake.inputs[0] = {VPUTensor(65, 12 - 3, 64, 1, DataType::FLOAT16)};  //-3 trick
        //  wl_Top_SOHH_K64x15_fake.halo = h_zero;//irrelevant for NN
        // wl_Top_SOHH_K64x15_fake.padding[Dim::Padding::TOP] = 1;  // help a little so input+pad >=k
        // wl_Top_SOHH_K64x15_fake.output_write_tiles = 2;//just makes the time marginally (1%) smaller
        // wl_Top_SOHH_K64x15_fake.isi_strategy = ISIStrategy::CLUSTERING;//32428

        DPUWorkload wl_Bot_SOHH_K64x15_fake{wl_Htt1_Bot_SOHH_K64x15};
        wl_Bot_SOHH_K64x15_fake.inputs[0] = {VPUTensor(65, 12 - 4, 64, 1, DataType::FLOAT16)};
        // wl_Bot_SOHH_K64x15_fake.halo = h_zero;

        {  // SOH H variants

            std::cout << "\n ------------------------       SOH wing , Variants        "
                         "----------------------------------";
            std::cout << "\n SOHO like SPLITS  from Layer  (same compute tensor , but CLU + halo=0)";

            case_run(wl_Top_SOHO_K64x15,
                     "\nTEST of: SOHO like TOP K=64 X15\n");  //   v17: 33238     GTM 36524 v159nn:xx
            case_run(wl_Bot_SOHO_K64x15,
                     "\nTEST of: SOHo like Bit K=64 X15\n");  //   v17: 33238     GT ??xx v159nn:xx

            case_run(wl_Top_SOHO_K32x30,
                     "\nTEST of: SOHO like TOP K32 x30\n");  //   v17: 17144      GTM xx v159nn:xx
            case_run(wl_Bot_SOHO_K32x30,
                     "\nTEST of: SOHo like Bit K=32 x30\n");  //   v17: 17144     GT ??xx v159nn:xx

            // Top mid bot are in parallel (X15 intra tile) =  (15x )||(15x ) = ()|()  =  xx
            // GT parallelism (15x )||(15x )= ()|()|()  =

            std::cout << "\n FAKE SOH H like SPLITS  from Layer  (memo tensor as compute , but isi SOH + halo)";
            // this tries also alternative , asking the NN using the memo tensor instead

            force_fail = false;
            case_run(wl_Top_SOHH_K64x15_fake,
                     "\nTEST of:fake SOHH TOP K=64 X15\n");  //   v17:    78746(pad 1 trick),63750(pad2) ,44342(pad4
                                                             //   trick)
            // 80926(-3 trick)
            //   GT xx v159nn:xx
            case_run(wl_Bot_SOHH_K64x15_fake,
                     "\nTEST of:fake SOHH BOT K=64 X15\n");  //   v17: 97771 (H permissive(<k))     GT x v159nn:xx
            // Top mid bot are in parallel (X15 intra tile) =  (15x )||(15x ) = ()|()  =  xx
            // GT parallelism (15x )||(15x )= ()|()|()  = NO SPECIAL GT
        }
        // SOH H is
    }
}
TEST_F(VPULayerInvstgt_EISXW_119193_Deeplab_v3, SOHH_1x_Most_used_temporalTiles_SOHtt1_6) {
    // const unsigned int fail{force_fail ? 0u : 1u};  // 1 neutral, 0 fail
    // show_split = true;
    // EXPECT_TRUE(false);
    EXPECT_FALSE(simple_fail_all);  // centrally

    // workloads level

    DPUWorkload wl_larger2x_Top_SOHH{wl_Htt1_Top_SOHH_K64x15};
    wl_larger2x_Top_SOHH.inputs[0] = {VPUTensor(65, 12, 64, 1, DataType::FLOAT16)};
    wl_larger2x_Top_SOHH.outputs[0] = {VPUTensor(65, 4, 64, 1, DataType::FLOAT16)};

    DPUWorkload wl_Top_SOHO2x_K64x15{wl_larger2x_Top_SOHH};
    wl_Top_SOHO2x_K64x15.isi_strategy = ISIStrategy::CLUSTERING;
    wl_Top_SOHO2x_K64x15.halo = h_zero;

    // SOHH fake : Compute tensor susbstituted by memory tensor
    DPUWorkload wl_Top_SOHH2x_K64x15_fake{wl_larger2x_Top_SOHH};
    wl_Top_SOHH2x_K64x15_fake.inputs[0] = {VPUTensor(65, 12 - 3, 64, 1, DataType::FLOAT16)};  // reduce with halo
    //  wl_Top_SOHH2x_K64x15_fake.halo = h_zero;//irrelevant for NN
    // wl_Top_SOHH2x_K64x15_fake.padding[Dim::Padding::TOP] = 1;  // help a little so input+pad >=k
    // wl_Top_SOHH_K64x15_fake.output_write_tiles = 2;//just makes the time marginally (1%) smaller
    // wl_Top_SOHH_K64x15_fake.isi_strategy = ISIStrategy::CLUSTERING;//32428

    {  // SOH H variants

        std::cout << "\n ------------------------       SOH wing , Variants with nominal1x split "
                     "-------------------------------";

        case_run(wl_larger2x_Top_SOHH, "\nTEST of: SOH H 1x  K=64 X15\n");  //   v17: (1x=50905)     GT xx v159nn:xx

        std::cout << "\n SOHO like SPLITS  from Layer  (same compute tensor , but CLU + halo=0)";
        case_run(wl_Top_SOHO2x_K64x15,
                 "\nTEST of: SOHO 1x like TOP K=64 X15\n");  //   v17: (1x=33238)     GTM xx v159nn:xx

        std::cout << "\n FAKE SOH H like SPLITS  from Layer  (memo tensor as compute , but isi SOH + halo)";

        // this tries also alternative , asking the NN using the memo tensor instead
        force_fail = false;
        case_run(wl_Top_SOHH2x_K64x15_fake,
                 "\nTEST of:fake SOHH x1 TOP K=64 X15\n");  // v17: 78746(pad 1 trick),63750(pad2) ,44342(pad4
                                                            // trick), 80926(-3 trick)
    }
}
TEST_F(VPULayerInvstgt_EISXW_119193_Deeplab_v3, SOHH_2x_upvariants_Most_used_temporalTiles_SOHtt1_6) {
    // const unsigned int fail{force_fail ? 0u : 1u};  // 1 neutral, 0 fail
    // show_split = true;
    // EXPECT_TRUE(false);
    EXPECT_FALSE(simple_fail_all);  // centrally

    // workloads level

    DPUWorkload wl_larger2x_Top_SOHH{wl_Htt1_Top_SOHH_K64x15};
    wl_larger2x_Top_SOHH.inputs[0] = {VPUTensor(65, 12 + 4, 64, 1, DataType::FLOAT16)};
    wl_larger2x_Top_SOHH.outputs[0] = {VPUTensor(65, 4 + 4, 64, 1, DataType::FLOAT16)};

    DPUWorkload wl_Top_SOHO2x_K64x15{wl_larger2x_Top_SOHH};
    wl_Top_SOHO2x_K64x15.isi_strategy = ISIStrategy::CLUSTERING;
    wl_Top_SOHO2x_K64x15.halo = h_zero;

    // SOHH fake : Compute tensor susbstituted by memory tensor
    DPUWorkload wl_Top_SOHH2x_K64x15_fake{wl_larger2x_Top_SOHH};
    wl_Top_SOHH2x_K64x15_fake.inputs[0] = {VPUTensor(65, 12 + 4 - 4, 64, 1, DataType::FLOAT16)};  // reduce with halo
    //  wl_Top_SOHH2x_K64x15_fake.halo = h_zero;//irrelevant for NN
    // wl_Top_SOHH2x_K64x15_fake.padding[Dim::Padding::TOP] = 1;  // help a little so input+pad >=k
    // wl_Top_SOHH_K64x15_fake.output_write_tiles = 2;//just makes the time marginally (1%) smaller
    // wl_Top_SOHH_K64x15_fake.isi_strategy = ISIStrategy::CLUSTERING;//32428

    {  // SOH H variants

        std::cout << "\n ------------------------       SOH wing , Variants with larger split "
                     "-------------------------------";

        case_run(wl_larger2x_Top_SOHH,
                 "\nTEST of: SOH H 2x  K=64 X15\n");  //   v17:78395: (1x=50905)     GT 119721!!  v159nn:xx

        std::cout << "\n SOHO like SPLITS  from Layer  (same compute tensor , but CLU + halo=0)";
        case_run(wl_Top_SOHO2x_K64x15,
                 "\nTEST of: SOHO 2x like TOP K=64 X15\n");  //   v17:60464 (1x=33238)     GTM 65801??? v159nn:xx

        std::cout << "\n FAKE SOH H like SPLITS  from Layer  (memo tensor as compute , but isi SOH + halo)";

        // this tries also alternative , asking the NN using the memo tensor instead
        force_fail = false;
        case_run(wl_Top_SOHH2x_K64x15_fake,
                 "\nTEST of:fake SOHH x2 TOP K=64 X15\n");  // v17:129543  GT:
        // 1x was  v17: 78746(pad 1 trick),63750(pad2) ,44342(pad4 trick)
    }
}
TEST_F(VPULayerInvstgt_EISXW_119193_Deeplab_v3, SOHH_5x_upvariants_Most_used_temporalTiles_SOHtt1_6) {
    // const unsigned int fail{force_fail ? 0u : 1u};  // 1 neutral, 0 fail
    // show_split = true;
    // EXPECT_TRUE(false);
    EXPECT_FALSE(simple_fail_all);  // centrally

    // workloads level

    DPUWorkload wl_larger2x_Top_SOHH{wl_Htt1_Top_SOHH_K64x15};  // 12 in 4 out
    wl_larger2x_Top_SOHH.inputs[0] = {VPUTensor(65, 12 + 4 * 4, 64, 1, DataType::FLOAT16)};
    wl_larger2x_Top_SOHH.outputs[0] = {VPUTensor(65, 4 * 5, 64, 1, DataType::FLOAT16)};

    DPUWorkload wl_Top_SOHO2x_K64x15{wl_larger2x_Top_SOHH};
    wl_Top_SOHO2x_K64x15.isi_strategy = ISIStrategy::CLUSTERING;
    wl_Top_SOHO2x_K64x15.halo = h_zero;

    // SOHH fake : Compute tensor susbstituted by memory tensor
    DPUWorkload wl_Top_SOHH2x_K64x15_fake{wl_larger2x_Top_SOHH};
    wl_Top_SOHH2x_K64x15_fake.inputs[0] = {
            VPUTensor(65, 12 + 4 * 4 - 4, 64, 1, DataType::FLOAT16)};  // reduce with halo
    //  wl_Top_SOHH2x_K64x15_fake.halo = h_zero;//irrelevant for NN
    // wl_Top_SOHH2x_K64x15_fake.padding[Dim::Padding::TOP] = 1;  // help a little so input+pad >=k
    // wl_Top_SOHH_K64x15_fake.output_write_tiles = 2;//just makes the time marginally (1%) smaller
    // wl_Top_SOHH_K64x15_fake.isi_strategy = ISIStrategy::CLUSTERING;//32428

    {  // SOH H variants

        std::cout << "\n ------------------------       SOH wing , Variants with larger split 5x: input:12+16,   20 "
                     "output "
                     "-------------------------------";

        case_run(wl_larger2x_Top_SOHH,
                 "\nTEST of: SOH H x  K=64 X15\n");  //   v17:195908 (1x=50905)     GT xx v159nn:xx

        std::cout << "\n SOHO like SPLITS  from Layer  (same compute tensor , but CLU + halo=0)";
        case_run(wl_Top_SOHO2x_K64x15,
                 "\nTEST of: SOHO x like TOP K=64 X15\n");  //   v17:157279 (1x=33238)     GTM xx v159nn:xx

        std::cout << "\n FAKE SOH H like SPLITS  from Layer  (memo tensor as compute , but isi SOH + halo)";

        // this tries also alternative , asking the NN using the memo tensor instead
        force_fail = false;
        case_run(wl_Top_SOHH2x_K64x15_fake,
                 "\nTEST of:fake SOHH x TOP K=64 X15\n");  // v17:222645
        // 1x was  v17: 78746(pad 1 trick),63750(pad2) ,44342(pad4 trick)
    }
}

TEST_F(VPULayerInvstgt_EISXW_119193_Deeplab_v3, TemporalTiles_SOKtt0_1_SOHtt0) {
    const unsigned int fail{force_fail ? 0u : 1u};  // 1 neutral, 0 fail
    // show_split = true;
    //  EXPECT_TRUE(false);

    const bool prefetch{true};
    // SOK wing
    {
        std::cout << "\n ------- SOK  WING:   tt0 ------- \n";
        const DPULayer tst_layer(wl_Ktt0);  // has 3 output rows, what h split!
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 460000u,
                  /*fail * */ (460000u + 20000u)},  // v17:464670  GT:xx  (intratile:)
                                                    // VPUXVPUNN(old v):xx. v159NN: xx
                 "SOHO , no memmove, "},

                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, prefetch}},  // RELEVANT
                 {VPUNN::Cycles::NO_ERROR, true, 230000,
                  fail * (230000u + 20000)},  // v17:241568  GT:xx  (intratile K64x7 K32x1)
                                              // VPUXVPUNN(old v): xx, v159NN:xx
                 "SOK , no memmove, "},

                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_HaloRead, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 710000,
                  /*fail * */ (710000u + 500000)},  // v17:725460: 1169790    GT:? v159NN: xx
                 "SOH Halo , no memmove, "},

                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 460000, (460000u + 30000u)},  // v17:CLU:479220  GT:?  (intratile:)
                 "FUll , no memmove, "},

                // note:?? wins
        };
        executeTests(tests);
    }

    {
        std::cout << "\n ------- SOK  WING:   tt1 ------- \n";
        const DPULayer tst_layer(wl_Ktt1);  // has 3 output rows, what h split!
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 470000u,
                  /*fail * */ (470000u + 30000u)},  // v17:483450  GT:xx  (intratile:)
                                                    // VPUXVPUNN(old v):xx. v159NN: xx
                 "SOHO , no memmove, "},

                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, prefetch}},  // RELEVANT
                 {VPUNN::Cycles::NO_ERROR, true, 230000,
                  fail * (230000u + 30000)},  // v17:246757  GT:xx  (intratile K64x7 K32x1)
                                              // VPUXVPUNN(old v): xx, v159NN:xx
                 "SOK , no memmove, "},

                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_HaloRead, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 810000,
                  /*fail * */ (810000u + 550000)},  // v17:822045:1226910    GT:? v159NN: xx
                 "SOH Halo , no memmove, "},

                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 470000, (470000u + 30000u)},  // v17:CLU:489450  GT:?  (intratile:)
                 "FUll , no memmove, "},

                // note:?? wins
        };
        executeTests(tests);
    }

    {
        // SOH  wing
        std::cout << "\n ------- SOH  WING:   tt0 ------- \n";
        const DPULayer tst_layer(wl_Htt0);  // h =9, padding T:4
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 1,
                  /*fail * */ (1u + 1u)},  // v17:ERROR_INPUT_TOO_BIG  GT:xx  (intratile: )   VPUXVPUNN(old
                                           // v):xx. v159NN: xx
                 "SOHO , no memmove, "},

                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 596000,
                  /*fail * */ (596000u + 20000)},  // v17:607920  GT:xx  (intratile x1)
                                                   // VPUXVPUNN(old v): xx, v159NN:xx
                 "SOK , no memmove, "},

                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_HaloRead, false, false, prefetch}},  // RELEVANT
                 {VPUNN::Cycles::NO_ERROR, true, 890000,
                  fail * (890000u + 900000)},  // v17:905415:1712340 , (intratile:  K64x15) GT:? v159NN: xx
                 "SOH Halo , no memmove, "},

                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, prefetch}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 1, (1 + 1)},  // v17: ERROR_INPUT_TOO_BIG  GT:?
                 "FUll , no memmove, "},

                // note:x wins
        };
        executeTests(tests);
    }

    // EXPECT_TRUE(false);  // something has to fail to  see couts
    EXPECT_FALSE(simple_fail_all);  // centrally

    // low level WL
    {
        // ktt0
        DPUWorkload wl_Ktt0_TB_SOK64x7_CLU{wl_Ktt0_TB_SOK64x7};
        wl_Ktt0_TB_SOK64x7_CLU.isi_strategy = ISIStrategy::CLUSTERING;
        wl_Ktt0_TB_SOK64x7_CLU.output_write_tiles = 1;

        DPUWorkload wl_Ktt0_TB_SOK32x1_CLU{wl_Ktt0_TB_SOK32x1};
        wl_Ktt0_TB_SOK32x1_CLU.isi_strategy = ISIStrategy::CLUSTERING;
        wl_Ktt0_TB_SOK32x1_CLU.output_write_tiles = 1;

        // ktt1
        DPUWorkload wl_Ktt1_TB_SOK64x7_CLU{wl_Ktt1_TB_SOK64x7};
        wl_Ktt1_TB_SOK64x7_CLU.isi_strategy = ISIStrategy::CLUSTERING;
        wl_Ktt1_TB_SOK64x7_CLU.output_write_tiles = 1;

        DPUWorkload wl_Ktt1_TB_SOK32x1_CLU{wl_Ktt1_TB_SOK32x1};
        wl_Ktt1_TB_SOK32x1_CLU.isi_strategy = ISIStrategy::CLUSTERING;
        wl_Ktt1_TB_SOK32x1_CLU.output_write_tiles = 1;
        {
            Logger::clear2ndlog();
            std::string err_info;

            {  // SOK SPLITS ktt0
                std::cout << "\n ------- SOK SPLITS on SOK wing ktt0: ------- ";

                case_run(wl_Ktt0_TB_SOK64x7, "\nTEST of: SOK  K=64 X7\n");  //   v17: 32159     GT xx   v159nn:xx
                case_run(wl_Ktt0_TB_SOK32x1, "\nTEST of: SOK  K=32 X1\n");  //   v17: 16455     GT xx   v159nn:xx
                // Top bot are in parallel (x7+x1) =  ()||() = ()|()  = xx
                // GT parallelism (2x)||(2x)||( 2x) = ()|( )  =xx

                case_run(wl_Ktt0_TB_SOK64x7_CLU,
                         "\nTEST of: CLUowt1  K=64 X7\n");  //   v17: 31948     GT xx v159nn:xx
                case_run(wl_Ktt0_TB_SOK32x1_CLU,
                         "\nTEST of: CLUowt1  K=32 X1\n");  //   v17: 16401     GT xx v159nn:xx
            }

            {  // SOK SPLITS ktt1
                std::cout << "\n ------- SOK SPLITS on SOK wing ktt1: ------- ";

                case_run(wl_Ktt1_TB_SOK64x7, "\nTEST of: SOK  K=64 X7\n");  //   v17: 32839     GT xx   v159nn:xx
                case_run(wl_Ktt1_TB_SOK32x1, "\nTEST of: SOK  K=32 X1\n");  //   v17: 16884     GT xx   v159nn:xx
                // Top bot are in parallel (x7+x1) =  ()||() = ()|()  = xx
                // GT parallelism (2x)||(2x)||( 2x) = ()|( )  =xx

                case_run(wl_Ktt1_TB_SOK64x7_CLU,
                         "\nTEST of: CLUowt1  K=64 X7\n");  //   v17: 32630     GT xx v159nn:xx
                case_run(wl_Ktt1_TB_SOK32x1_CLU,
                         "\nTEST of: CLUowt1  K=32 X1\n");  //   v17: 16809    GT xx v159nn:xx
            }

            {  /// SOH H splits
                std::cout << "\n ------- SOH H SPLITS on SOH wing: ------- ";

                case_run(wl_Htt0_Top_SOHH_K64x15,
                         "\nTEST of: TOP SOH H  K=64 X15\n");  //   v17: 60361 : 117396   GT xx v159nn:xx
                case_run(wl_Htt0_Bot_SOHH_K64x15,
                         "\nTEST of: BOT SOH H  K=64 X15\n");  //   v17: 50905:97771     GT xx v159nn:xx

                // SOHH : x
                // GT: xx
            }

            // who wins?
            // GT:
        }
    }

    {
        // SOHO forcing
        DPUWorkload wl_Top_SOHO_K64x15{wl_Htt0_Top_SOHH_K64x15};
        wl_Top_SOHO_K64x15.isi_strategy = ISIStrategy::CLUSTERING;
        wl_Top_SOHO_K64x15.halo = h_zero;

        DPUWorkload wl_Bot_SOHO_K64x15{wl_Htt0_Bot_SOHH_K64x15};
        wl_Bot_SOHO_K64x15.isi_strategy = ISIStrategy::CLUSTERING;
        wl_Bot_SOHO_K64x15.halo = h_zero;

        // SOHH fake : Compute tensor susbstituted by memory tensor
        DPUWorkload wl_Top_SOHH_K64x15_fake{wl_Htt0_Top_SOHH_K64x15};
        wl_Top_SOHH_K64x15_fake.inputs[0] = {VPUTensor(65, 9 - 4, 64, 1, DataType::FLOAT16)};  //-3 trick does not work
        wl_Top_SOHH_K64x15_fake.halo = h_zero;                                                 // irrelevant for NN
        wl_Top_SOHH_K64x15_fake.padding[Dim::Padding::BOTTOM] = 4;  // help a little so input+pad >=k
        // wl_Top_SOHH_K64x15_fake.output_write_tiles = 2;//just makes the time marginally (1%) smaller
        // wl_Top_SOHH_K64x15_fake.isi_strategy = ISIStrategy::CLUSTERING;//32428

        DPUWorkload wl_Bot_SOHH_K64x15_fake{wl_Htt0_Bot_SOHH_K64x15};
        wl_Bot_SOHH_K64x15_fake.inputs[0] = {VPUTensor(65, 12 - 4, 64, 1, DataType::FLOAT16)};
        // wl_Bot_SOHH_K64x15_fake.halo = h_zero;

        {  // SOH H variants

            std::cout << "\n ------------------------       SOH wing , Variants        "
                         "----------------------------------";
            std::cout << "\n SOHO like SPLITS  from Layer  (same compute tensor , but CLU + halo=0)";

            case_run(wl_Top_SOHO_K64x15,
                     "\nTEST of: SOHO like TOP K=64 X15\n");  //   v17: 52354     GTM xx v159nn:xx
            case_run(wl_Bot_SOHO_K64x15,
                     "\nTEST of: SOHo like Bot K=64 X15\n");  //   v17: 33238     GT ??xx v159nn:xx
            // Top mid bot are in parallel (X15 intra tile) =  (15x )||(15x ) = ()|()  =  xx
            // GT parallelism (15x )||(15x )= ()|()|()  =

            std::cout << "\n FAKE SOH H like SPLITS  from Layer  (memo tensor as compute , but isi SOH + halo)";
            // this tries also alternative , asking the NN using the memo tensor instead

            force_fail = false;
            case_run(wl_Top_SOHH_K64x15_fake,
                     "\nTEST of:fake SOHH TOP K=64 X15\n");  //   v17:53787 (with padding 4)
            case_run(wl_Bot_SOHH_K64x15_fake,
                     "\nTEST of:fake SOHH BOT K=64 X15\n");  //   v17: xx   GT x v159nn:xx
            // Top mid bot are in parallel (X15 intra tile) =  (15x )||(15x ) = ()|()  =  xx
            // GT parallelism (15x )||(15x )= ()|()|()  = NO SPECIAL GT
        }
        // SOH H is
    }
}

TEST_F(VPULayerInvstgt_EISXW_119193_Deeplab_v3, TemporalTiles_SOKtt22_23_SOHtt7) {
    const unsigned int fail{force_fail ? 0u : 1u};  // 1 neutral, 0 fail
    // show_split = true;
    //  EXPECT_TRUE(false);

    const bool prefetch{true};
    // SOK wing
    {
        std::cout << "\n ------- SOK  WING:   tt22 ------- \n";
        const DPULayer tst_layer(wl_Ktt22);  // has 3 output rows, what h split!
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 470000u,
                  /*fail * */ (470000u + 30000u)},  // v17:486345  GT:xx  (intratile:)
                                                    // VPUXVPUNN(old v):xx. v159NN: xx
                 "SOHO , no memmove, "},

                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, prefetch}},  // RELEVANT
                 {VPUNN::Cycles::NO_ERROR, true, 230000,
                  fail * (230000u + 20000)},  // v17:247870  GT:xx  (intratile K64x7 K32x1)
                                              // VPUXVPUNN(old v): xx, v159NN:xx
                 "SOK , no memmove, "},

                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_HaloRead, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 700000,
                  /*fail * */ (700000u + 1200000)},  // v17:779340:1809450    GT:? v159NN: xx
                 "SOH Halo , no memmove, "},

                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 480000, (480000u + 20000u)},  // v17:CLU:491565  GT:?  (intratile:)
                 "FUll , no memmove, "},

                // note:?? wins
        };
        executeTests(tests);
    }

    {
        std::cout << "\n ------- SOK  WING:   tt23 ------- \n";
        const DPULayer tst_layer(wl_Ktt23);  // has 3 output rows, what h split!
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 430000u,
                  /*fail * */ (430000u + 30000u)},  // v17:438270  GT:xx  (intratile:)
                                                    // VPUXVPUNN(old v):xx. v159NN: xx
                 "SOHO , no memmove, "},

                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, prefetch}},  // RELEVANT
                 {VPUNN::Cycles::NO_ERROR, true, 230000,
                  fail * (230000u + 30000)},  // v17:235680  GT:xx  (intratile K64x7 K32x1)
                                              // VPUXVPUNN(old v): xx, v159NN:xx
                 "SOK , no memmove, "},

                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_HaloRead, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 650000,
                  /*fail * */ (650000u + 750000)},  // v17: 662175:1272630   GT:? v159NN: xx
                 "SOH Halo , no memmove, "},

                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 450000, (450000u + 30000u)},  // v17:CLU:469140  GT:?  (intratile:)
                 "FUll , no memmove, "},

                // note:?? wins
        };
        executeTests(tests);
    }

    {
        // SOH  wing
        std::cout << "\n ------- SOH  WING:   tt7 ------- \n";
        const DPULayer tst_layer(wl_Htt7);  // h =8
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 1,
                  /*fail * */ (1u + 1u)},  // v17:ERROR_INPUT_TOO_BIG  GT:xx  (intratile: )   VPUXVPUNN(old
                                           // v):xx. v159NN: xx
                 "SOHO , no memmove, "},

                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 410000,
                  /*fail * */ (410000u + 40000)},  // v17:434268,  GT:xx  (intratile x1)
                                                   // VPUXVPUNN(old v): xx, v159NN:xx
                 "SOK , no memmove, "},

                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_HaloRead, false, false, prefetch}},  // RELEVANT
                 {VPUNN::Cycles::NO_ERROR, true, 750000,
                  fail * (750000u + 750000)},  // v17:763575:1390740 , (intratile:  K64x15) GT:? v159NN: xx
                 "SOH Halo , no memmove, "},

                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, prefetch}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 1, (1 + 1)},  // v17: ERROR_INPUT_TOO_BIG  GT:?
                 "FUll , no memmove, "},

                // note:x wins
        };
        executeTests(tests);
    }

    // EXPECT_TRUE(false);  // something has to fail to  see couts
    EXPECT_FALSE(simple_fail_all);  // centrally

    // low level WL
    {
        // ktt22
        DPUWorkload wl_Ktt22_TB_SOK64x7_CLU{wl_Ktt22_TB_SOK64x7};
        wl_Ktt22_TB_SOK64x7_CLU.isi_strategy = ISIStrategy::CLUSTERING;
        wl_Ktt22_TB_SOK64x7_CLU.output_write_tiles = 1;

        DPUWorkload wl_Ktt22_TB_SOK32x1_CLU{wl_Ktt22_TB_SOK32x1};
        wl_Ktt22_TB_SOK32x1_CLU.isi_strategy = ISIStrategy::CLUSTERING;
        wl_Ktt22_TB_SOK32x1_CLU.output_write_tiles = 1;

        // ktt23
        DPUWorkload wl_Ktt23_TB_SOK64x7_CLU{wl_Ktt23_TB_SOK64x7};
        wl_Ktt23_TB_SOK64x7_CLU.isi_strategy = ISIStrategy::CLUSTERING;
        wl_Ktt23_TB_SOK64x7_CLU.output_write_tiles = 1;

        DPUWorkload wl_Ktt23_TB_SOK32x1_CLU{wl_Ktt23_TB_SOK32x1};
        wl_Ktt23_TB_SOK32x1_CLU.isi_strategy = ISIStrategy::CLUSTERING;
        wl_Ktt23_TB_SOK32x1_CLU.output_write_tiles = 1;
        {
            Logger::clear2ndlog();
            std::string err_info;

            {  // SOK SPLITS ktt22
                std::cout << "\n ------- SOK SPLITS on SOK wing ktt22: ------- ";

                case_run(wl_Ktt22_TB_SOK64x7, "\nTEST of: SOK  K=64 X7\n");  //   v17: 32984     GT xx   v159nn:xx
                case_run(wl_Ktt22_TB_SOK32x1, "\nTEST of: SOK  K=32 X1\n");  //   v17: 16982     GT xx   v159nn:xx
                // Top bot are in parallel (x7+x1) =  ()||() = ()|()  = xx
                // GT parallelism (2x)||(2x)||( 2x) = ()|( )  =xx

                case_run(wl_Ktt22_TB_SOK64x7_CLU,
                         "\nTEST of: CLUowt1  K=64 X7\n");  //   v17: 32771     GT xx v159nn:xx
                case_run(wl_Ktt22_TB_SOK32x1_CLU,
                         "\nTEST of: CLUowt1  K=32 X1\n");  //   v17: 16920     GT xx v159nn:xx
            }

            {  // SOK SPLITS ktt23
                std::cout << "\n ------- SOK SPLITS on SOK wing ktt23: ------- ";

                case_run(wl_Ktt23_TB_SOK64x7, "\nTEST of: SOK  K=64 X7\n");  //   v17: 31484     GT xx   v159nn:xx
                case_run(wl_Ktt23_TB_SOK32x1, "\nTEST of: SOK  K=32 X1\n");  //   v17: 15712     GT xx   v159nn:xx
                // Top bot are in parallel (x7+x1) =  ()||() = ()|()  = xx
                // GT parallelism (2x)||(2x)||( 2x) = ()|( )  =xx

                case_run(wl_Ktt23_TB_SOK64x7_CLU,
                         "\nTEST of: CLUowt1  K=64 X7\n");  //   v17: 31289     GT xx v159nn:xx
                case_run(wl_Ktt23_TB_SOK32x1_CLU,
                         "\nTEST of: CLUowt1  K=32 X1\n");  //   v17: 15638    GT xx v159nn:xx
            }

            {  /// SOH H splits
                std::cout << "\n ------- SOH H SPLITS on SOH wing: ------- ";

                case_run(wl_Htt7_Top_SOHH_K64x15,
                         "\nTEST of: Top SOH H  K=64 X15\n");  //   v17: 50905:97771     GT xx v159nn:xx
                case_run(wl_Htt7_Bot_SOHH_K64x15,
                         "\nTEST of: Bot SOH H  K=64 X15\n");  //   v17: 43940:92090     GT xx v159nn:xx

                // SOHH : x
                // GT: xx
            }

            // who wins?
            // GT:
        }
    }

    {
        // SOHO forcing
        DPUWorkload wl_Top_SOHO_K64x15{wl_Htt7_Top_SOHH_K64x15};
        wl_Top_SOHO_K64x15.isi_strategy = ISIStrategy::CLUSTERING;
        wl_Top_SOHO_K64x15.halo = h_zero;

        DPUWorkload wl_Bot_SOHO_K64x15{wl_Htt7_Bot_SOHH_K64x15};
        wl_Bot_SOHO_K64x15.isi_strategy = ISIStrategy::CLUSTERING;
        wl_Bot_SOHO_K64x15.halo = h_zero;

        // SOHH fake : Compute tensor susbstituted by memory tensor
        DPUWorkload wl_Top_SOHH_K64x15_fake{wl_Htt7_Top_SOHH_K64x15};
        wl_Top_SOHH_K64x15_fake.inputs[0] = {
                VPUTensor(65, 12 - 3, 64, 1, DataType::FLOAT16)};  //-3 trick   (should be -4)
        //  wl_Top_SOHH_K64x15_fake.halo = h_zero;//irrelevant for NN
        // wl_Top_SOHH_K64x15_fake.padding[Dim::Padding::BOTTOM] = 4;  // help a little so input+pad >=k
        // wl_Top_SOHH_K64x15_fake.output_write_tiles = 2;//just makes the time marginally (1%) smaller
        // wl_Top_SOHH_K64x15_fake.isi_strategy = ISIStrategy::CLUSTERING;//32428

        DPUWorkload wl_Bot_SOHH_K64x15_fake{wl_Htt7_Bot_SOHH_K64x15};
        wl_Bot_SOHH_K64x15_fake.inputs[0] = {VPUTensor(65, 8 - 4, 64, 1, DataType::FLOAT16)};
        // wl_Bot_SOHH_K64x15_fake.halo = h_zero;

        {  // SOH H variants

            std::cout << "\n ------------------------       SOH wing , Variants        "
                         "----------------------------------";
            std::cout << "\n SOHO like SPLITS  from Layer  (same compute tensor , but CLU + halo=0)";

            case_run(wl_Top_SOHO_K64x15,
                     "\nTEST of: SOHO like TOP K=64 X15\n");  //   v17: 33238     GTM xx v159nn:xx
            case_run(wl_Bot_SOHO_K64x15,
                     "\nTEST of: SOHo like Bot K=64 X15\n");  //   v17: 32899     GT ??xx v159nn:xx
            // Top mid bot are in parallel (X15 intra tile) =  (15x )||(15x ) = ()|()  =  xx
            // GT parallelism (15x )||(15x )= ()|()|()  =

            std::cout << "\n FAKE SOH H like SPLITS  from Layer  (memo tensor as compute , but isi SOH + halo)";
            // this tries also alternative , asking the NN using the memo tensor instead

            force_fail = false;
            case_run(wl_Top_SOHH_K64x15_fake,
                     "\nTEST of:fake SOHH TOP K=64 X15\n");  //   v17:80926 (-3 trick)
            case_run(wl_Bot_SOHH_K64x15_fake,
                     "\nTEST of:fake SOHH BOT K=64 X15\n");  //   v17: xx   GT x v159nn:xx
            // Top mid bot are in parallel (X15 intra tile) =  (15x )||(15x ) = ()|()  =  xx
            // GT parallelism (15x )||(15x )= ()|()|()  = NO SPECIAL GT
        }
        // SOH H is
    }
}

TEST_F(VPULayerInvstgt_EISXW_119193_Deeplab_v3, Fake_SOHH) {
    // FOrking memeory tensor in place of compute tensor for the SOHH
    // show_split = true;
    // EXPECT_TRUE(false);
    EXPECT_FALSE(simple_fail_all);  // centrally

    // workloads level
    std::cout << "\n ------------------------       SOH wing , Variants  FAKE SOHH     -----\n";
    {  // TT0
        // SOHH fake : Compute tensor susbstituted by memory tensor
        DPUWorkload wl_Top_SOHH_K64x15_fake{wl_Htt0_Top_SOHH_K64x15};
        wl_Top_SOHH_K64x15_fake.inputs[0] = {VPUTensor(65, 9 - 4, 64, 1, DataType::FLOAT16)};  //-3 trick does not work
        wl_Top_SOHH_K64x15_fake.halo = h_zero;

        DPUWorkload wl_Bot_SOHH_K64x15_fake{wl_Htt0_Bot_SOHH_K64x15};
        wl_Bot_SOHH_K64x15_fake.inputs[0] = {VPUTensor(65, 12 - 4, 64, 1, DataType::FLOAT16)};
        wl_Bot_SOHH_K64x15_fake.halo = h_zero;

        {  // SOH H variants
            std::cout << "\n TT0  FAKE SOH H like SPLITS  from Layer  (memo tensor as compute , but isi SOH + halo)";
            case_run(wl_Top_SOHH_K64x15_fake,
                     "\nTEST of:fake SOHH TOP K=64 X15\n");  //   v17:117396 (with code hack)//117396
            case_run(wl_Bot_SOHH_K64x15_fake,
                     "\nTEST of:fake SOHH BOT K=64 X15\n");  //   v17: 97771   GT x v159nn:xx
        }
    }
    {  // TT1..6
        // SOHH fake : Compute tensor susbstituted by memory tensor
        DPUWorkload wl_Top_SOHH_K64x15_fake{wl_Htt1_Top_SOHH_K64x15};
        wl_Top_SOHH_K64x15_fake.inputs[0] = {VPUTensor(65, 12 - 4, 64, 1, DataType::FLOAT16)};  //-3 trick does not work
        wl_Top_SOHH_K64x15_fake.halo = h_zero;

        DPUWorkload wl_Bot_SOHH_K64x15_fake{wl_Htt1_Bot_SOHH_K64x15};
        wl_Bot_SOHH_K64x15_fake.inputs[0] = {VPUTensor(65, 12 - 4, 64, 1, DataType::FLOAT16)};
        wl_Bot_SOHH_K64x15_fake.halo = h_zero;

        {  // SOH H variants
            std::cout << "\n TT0  FAKE SOH H like SPLITS  from Layer  (memo tensor as compute , but isi SOH + halo)";
            case_run(wl_Top_SOHH_K64x15_fake,
                     "\nTEST of:fake SOHH TOP K=64 X15\n");  //   v17:97771 (with code hack)
            case_run(wl_Bot_SOHH_K64x15_fake,
                     "\nTEST of:fake SOHH BOT K=64 X15\n");  //   v17: 97771   GT x v159nn:xx
        }
    }
    {  // TT7
        // SOHH fake : Compute tensor susbstituted by memory tensor
        DPUWorkload wl_Top_SOHH_K64x15_fake{wl_Htt7_Top_SOHH_K64x15};
        wl_Top_SOHH_K64x15_fake.inputs[0] = {VPUTensor(65, 12 - 4, 64, 1, DataType::FLOAT16)};  //-3 trick does not work
        wl_Top_SOHH_K64x15_fake.halo = h_zero;

        DPUWorkload wl_Bot_SOHH_K64x15_fake{wl_Htt7_Bot_SOHH_K64x15};
        wl_Bot_SOHH_K64x15_fake.inputs[0] = {VPUTensor(65, 8 - 4, 64, 1, DataType::FLOAT16)};
        wl_Bot_SOHH_K64x15_fake.halo = h_zero;

        {  // SOH H variants
            std::cout << "\n TT0  FAKE SOH H like SPLITS  from Layer  (memo tensor as compute , but isi SOH + halo)";
            case_run(wl_Top_SOHH_K64x15_fake,
                     "\nTEST of:fake SOHH TOP K=64 X15\n");  //   v17:97771 (with code hack)
            case_run(wl_Bot_SOHH_K64x15_fake,
                     "\nTEST of:fake SOHH BOT K=64 X15\n");  //   v17: 92090   GT x v159nn:xx
        }
    }
}

TEST_F(VPULayerInvstgt_EISXW_119193_Deeplab_v3, TemporalTiles_SOHO_possible_internal) {
    const unsigned int fail{force_fail ? 0u : 1u};  // 1 neutral, 0 fail
    // show_split = true;
    //  EXPECT_TRUE(false);

    const bool prefetch{true};

    // SOH  wing
    std::cout << "\n ------- SOH  WING:  out H=8 (from 64) ------- \n";

    DPUWorkload wl_Htt1_H7{wl_Htt1};
    wl_Htt1_H7.inputs[0] = {VPUTensor(65, 16 - 1, 960 /*64x15*/, 1, DataType::FLOAT16)};
    wl_Htt1_H7.outputs[0] = {VPUTensor(65, 4 + 4 - 1, 960 /*64x15*/, 1, DataType::FLOAT16)};
    DPUWorkload wl_Htt1_H6{wl_Htt1};
    wl_Htt1_H6.inputs[0] = {VPUTensor(65, 16 - 2, 960 /*64x15*/, 1, DataType::FLOAT16)};
    wl_Htt1_H6.outputs[0] = {VPUTensor(65, 4 + 4 - 2, 960 /*64x15*/, 1, DataType::FLOAT16)};
    DPUWorkload wl_Htt1_H5{wl_Htt1};
    wl_Htt1_H5.inputs[0] = {VPUTensor(65, 16 - 3, 960 /*64x15*/, 1, DataType::FLOAT16)};
    wl_Htt1_H5.outputs[0] = {VPUTensor(65, 4 + 4 - 3, 960 /*64x15*/, 1, DataType::FLOAT16)};

    const std::vector<TestCase> tests{
            {{(DPULayer)wl_Htt1, {1U, 1U, 2U, VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
             {Cycles::ERROR_INPUT_TOO_BIG, true, 1, fail * (1u + 1u)},  // v17:ERROR_INPUT_TOO_BIG  GT:xx  (intratile: )
                                                                        // VPUXVPUNN(old v):xx. v159NN: xx
             "SOHO H8 , no memmove, "},

            {{(DPULayer)wl_Htt1, {1U, 1U, 2U, VPUTilingStrategy::SOH_HaloRead, false, false, prefetch}},  // RELEVANT
             {Cycles::NO_ERROR, true, 750000,
              fail * (750000u +
                      750000)},  // v17:763575:1390740 , (intratile:  K64x15) GT:? v159NN: xx  . VPUXNN: 783870
             "SOH Halo H8 , no memmove, "},

            //--------------------h7
            {{(DPULayer)wl_Htt1_H7, {1U, 1U, 2U, VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
             {Cycles::ERROR_INPUT_TOO_BIG, true, 480000, fail * (480000u + 20000u)},  // v17:  GT:xx  (intratile: )
                                                                                      // VPUXVPUNN(old v):xx. v159NN: xx
             "SOHO H7 , no memmove, "},

            {{(DPULayer)wl_Htt1_H7, {1U, 1U, 2U, VPUTilingStrategy::SOH_HaloRead, false, false, prefetch}},  // RELEVANT
             {Cycles::NO_ERROR, true, 770000,
              fail * (770000 + 750000)},  // v17: 783165:1390740 , (intratile:  K64x15) GT:? v159NN: xx  . VPUXNN:
             "SOH Halo H7 , no memmove, "},
            //--------------------h6
            {{(DPULayer)wl_Htt1_H6, {1U, 1U, 2U, VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
             {Cycles::NO_ERROR, true, 480000u, fail * (480000u + 200000u)},  // v17:492045  GT:xx  (intratile: )
                                                                             // VPUXVPUNN(old
                                                                             // v):xx. v159NN: xx
             "SOHO H6 , no memmove, "},

            {{(DPULayer)wl_Htt1_H6, {1U, 1U, 2U, VPUTilingStrategy::SOH_HaloRead, false, false, prefetch}},  // RELEVANT
             {Cycles::NO_ERROR, true, 770000,
              fail * (770000 + 700000)},  // v17:  :1317750 , (intratile:  K64x15) GT:? v159NN: xx  . VPUXNN:
             "SOH Halo H6 , no memmove, "},

            //--------------------h5
            {{(DPULayer)wl_Htt1_H5, {1U, 1U, 2U, VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
             {Cycles::NO_ERROR, true, 480000u, fail * (480000u + 20000u)},  // v17:492045  GT:xx  (intratile: )
                                                                            // VPUXVPUNN(old
                                                                            // v):xx. v159NN: xx
             "SOHO H5 , no memmove, "},

            {{(DPULayer)wl_Htt1_H5, {1U, 1U, 2U, VPUTilingStrategy::SOH_HaloRead, false, false, prefetch}},  // RELEVANT
             {Cycles::NO_ERROR, true, 750000,
              fail * (750000u + 7500000)},  // v17:804900:1317750 , (intratile:  K64x15) GT:? v159NN: xx  . VPUXNN:
             "SOH Halo H5 , no memmove, "},
    };
    executeTests(tests);

    // EXPECT_TRUE(false);  // something has to fail to  see couts
    EXPECT_FALSE(simple_fail_all);  // centrally
}


}