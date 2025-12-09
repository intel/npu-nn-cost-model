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

class VPULayerCM_InvestigationTestNPU4x : public VPULayerCostModelTest {
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

TEST_F(VPULayerCM_InvestigationTestNPU4x, MEXP_C2_ELTWISE_1662_EISXW_126389_NPU40) {
    const bool force_fail{false};        // controls force failing assertion
    const int fail{force_fail ? 0 : 1};  // 1 neutral, 0 fail

    // EXPECT_TRUE(false);                  // something has to fail to  see couts
    show_split = false;

    DPUWorkload wl_ref{
            // orig Layer
            VPUDevice::VPU_4_0,
            Operation::ELTWISE,
            {VPUTensor(28, 28, 256, 1, DataType::UINT8, Layout::ZXY)},  // input dimensions
            {VPUTensor(28, 28, 256, 1, DataType::UINT8, Layout::ZXY)},  // output dimensions
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

    {
        const DPULayer tst_layer(wl_ref);
        // TODO: test should be SOHO wins,
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 1000, fail * 1000 + 1000},  // 1082  v17:1593  GTvpux:2232.95cyc
                 "SOHO /4 + no broadcast, no memmove, "},

                {{tst_layer, {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOHO_K_SWITCH, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 2100 - 300, fail * 2100 + 300},  // 1082  v17:1593  GTvpux:4600cyc
                 "SOHO B /4 +HK broadcast, no memmove, "},

                {{tst_layer, {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOK, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 1600, fail * 1600 + 1000},  // 1082  v17:1618  GTvpux:2232.95cyc
                 "SOK /4 , no memmove, "},

        };
        executeTests(tests);
    }

    VPUCostModel& theModel = layer_models.getModel(VPUDevice::VPU_4_0).get_cost_model();

    auto case_run = [=, &theModel](const DPUWorkload& workload, const std::string& whatTest) {
        DPUWorkload tst_layer(workload);
        std::string err_info;

        CyclesInterfaceType dpu_cost = theModel.DPU(tst_layer, err_info);
        if (force_fail) {
            EXPECT_EQ(dpu_cost, 10) << whatTest << dpu_cost << "=" << Cycles::toErrorText(dpu_cost) << "\n"
                                    << tst_layer << err_info << "END ERR";
        }
        std::cout << "  " << whatTest << dpu_cost << "=" << Cycles::toErrorText(dpu_cost) << "\n";
    };

    //  EXPECT_TRUE(false);  // something has to fail to  see couts

    // low level WL
    {
        const DPUWorkload wl_SOHO_8x16{
                VPUDevice::VPU_4_0,
                Operation::ELTWISE,
                {VPUTensor(28, 7, 256, 1, DataType::UINT8)},  // input dimensions
                {VPUTensor(28, 7, 256, 1, DataType::UINT8)},  // output dimensions
                {1, 1},                                       // kernels
                {1, 1},                                       // strides
                {0, 0, 0, 0},                                 // padding
                ExecutionMode::CUBOID_8x16,                   // execution mode
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

        //   DPUWorkload wl_SOHO_16x16{wl_SOHO_8x16};
        //  wl_SOHO_16x16.execution_order = ExecutionMode::CUBOID_16x16;

        //   DPUWorkload wl_SOHO_4x16{wl_SOHO_8x16};
        //  wl_SOHO_4x16.execution_order = ExecutionMode::CUBOID_4x16;

        DPUWorkload wl_SOHO_8x16_HK{wl_SOHO_8x16};
        wl_SOHO_8x16_HK.output_write_tiles = 4;
        {
            Logger::clear2ndlog();
            std::string err_info;

            {  // SOHO SPLITS
                std::cout << "\n ------- SOHO SPLITS: ------- ";
                // case_run(wl_SOHO_16x16, "TEST of: SOHO exec 16x16 ");  // 1089
                case_run(wl_SOHO_8x16, "\nTEST of: SOHO exec 8x16 ");  // 1082
                //  case_run(wl_SOHO_4x16, "TEST of: SOHO exec 4x16 ");    // 1476
                case_run(wl_SOHO_8x16_HK, "TEST of: SOHO exec 8x16 + broadcast ");  // 1082 v17:1593
            }
        }
    }
}

TEST_F(VPULayerCM_InvestigationTestNPU4x, MEXP_C2_CONV_4634_4662_EISXW_126389_NPU40) {
    const bool force_fail{false};        // controls force failing assertion
    const int fail{force_fail ? 0 : 1};  // 1 neutral, 0 fail

    // EXPECT_TRUE(false);                  // something has to fail to  see couts
    show_split = true;

    DPUWorkload wl_4634{
            // orig Layer
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

    //  DPUWorkload wl_4662{wl_4634};
    // wl_4634 is equivalent with wl_4662
    std::cout << "\n ------- wl_4634 ------- \n";
    {
        const DPULayer tst_layer(wl_4634);

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 3850 - 385 * 2,
                  fail * 3850 + 509},  // 4416 v17:4358  GTvpux:4168.05cyc GTL 3850  (intra_tile: 1)
                 "SOHO /4 , no memmove, "},

                {{tst_layer, {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOHO_K_SWITCH, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 3933 - 2 * 350,
                  fail * 3933 + 620},  //  v17:  GTvpux:4168.05cyc GTL3933  (intra_tile: 1)
                 "SOHO HK /4 , no memmove, "},

                {{tst_layer, {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOK, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 2200,
                  fail * 3435 + 800},  // 3138 v17:4233  GTvpux:4214.3cyc  (intra_tile: 1)
                 "SOK /4 , no memmove, "},

                {{tst_layer, {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOK_NO_BROADCAST, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 2200,
                  fail * 3435 + 800},  // 3138 v17:4233  GTvpux:4214.3cyc  (intra_tile: 1)
                 "SOK noB /4 , no memmove, "},

                // vpuxGT: SOH wins
                // v17: SOK wins with a small diff 2.95%
                // LNL NN SOK wins but with under-prediction 2300 vs gt =3400
        };
        executeTests(tests);
    }

    VPULayerCostModel& theModel = layer_models.getModel(VPUDevice::VPU_4_0);
    {  // test decision
        DPULayer tst_layer(wl_4634);
        const VPULayerStrategy strategy_SOHO{1U,    1U,    4U,      VPUNN::VPUTilingStrategy::SOH_Overlapped,
                                             false, false, prefetch};
        const VPULayerStrategy strategy_HK{1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOHO_K_SWITCH, false, false, prefetch};
        const VPULayerStrategy strategy_SOK{1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOK, false, false, prefetch};
        const VPULayerStrategy strategy_SOK_noB{1U,    1U,    4U,      VPUNN::VPUTilingStrategy::SOK_NO_BROADCAST,
                                                false, false, prefetch};

        auto cost_SOHO = theModel.Layer(tst_layer, strategy_SOHO);
        auto cost_HK = theModel.Layer(tst_layer, strategy_HK);
        auto cost_SOK = theModel.Layer(tst_layer, strategy_SOK);
        auto cost_SOKnoB = theModel.Layer(tst_layer, strategy_SOK_noB);

        EXPECT_FALSE(Cycles::isErrorCode(cost_SOHO));
        EXPECT_FALSE(Cycles::isErrorCode(cost_HK));
        EXPECT_FALSE(Cycles::isErrorCode(cost_SOK));
        EXPECT_FALSE(Cycles::isErrorCode(cost_SOKnoB));

        EXPECT_GT(cost_HK, cost_SOK) << "SOK should be better vs HK";
        EXPECT_GT(cost_SOHO, cost_SOK) << "SOK should be better vs SOHO";
        EXPECT_GT(cost_SOK, cost_SOKnoB) << "SOKnoB should be better vs SOK";
    }

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

    auto mod_execution = [](const DPUWorkload& wl, ExecutionMode em) -> DPUWorkload {
        DPUWorkload wl_{wl};
        wl_.execution_order = em;
        return wl_;
    };

    //  EXPECT_TRUE(false);  // something has to fail to  see couts

    // low level WL
    {
        // all tiles are the same
        const DPUWorkload SOHO_8x16{
                VPUDevice::VPU_4_0,
                Operation::CONVOLUTION,
                {VPUTensor(28, 7, 256, 1, DataType::UINT8)},  // input dimensions
                {VPUTensor(28, 7, 128, 1, DataType::UINT8)},  // output dimensions
                {1, 1},                                       // kernels
                {1, 1},                                       // strides
                {0, 0, 0, 0},                                 // padding
                ExecutionMode::CUBOID_8x16,                   // execution mode
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
        DPUWorkload HK_8x16{SOHO_8x16};
        HK_8x16.output_write_tiles = 4;

        const DPUWorkload SOHO_16x16{mod_execution(SOHO_8x16, ExecutionMode::CUBOID_16x16)};
        DPUWorkload HK_16x16{SOHO_16x16};
        HK_16x16.output_write_tiles = 4;

        const DPUWorkload SOHO_4x16{mod_execution(SOHO_8x16, ExecutionMode::CUBOID_4x16)};
        DPUWorkload HK_4x16{SOHO_4x16};
        HK_4x16.output_write_tiles = 4;

        // K64
        DPUWorkload SOHO_K64_8x16{SOHO_8x16};
        SOHO_K64_8x16.outputs[0] = VPUTensor(28, 7, 64, 1, DataType::UINT8);
        DPUWorkload SOHO_K64_16x16{SOHO_16x16};
        SOHO_K64_16x16.outputs[0] = VPUTensor(28, 7, 64, 1, DataType::UINT8);
        DPUWorkload SOHO_K64_4x16{SOHO_4x16};
        SOHO_K64_4x16.outputs[0] = VPUTensor(28, 7, 64, 1, DataType::UINT8);

        // K32
        DPUWorkload SOHO_K32_8x16{SOHO_8x16};
        SOHO_K32_8x16.outputs[0] = VPUTensor(28, 7, 32, 1, DataType::UINT8);
        DPUWorkload SOHO_K32_16x16{SOHO_16x16};
        SOHO_K32_16x16.outputs[0] = VPUTensor(28, 7, 32, 1, DataType::UINT8);
        DPUWorkload SOHO_K32_4x16{SOHO_4x16};
        SOHO_K32_4x16.outputs[0] = VPUTensor(28, 7, 32, 1, DataType::UINT8);

        // K16
        DPUWorkload SOHO_K16_8x16{SOHO_8x16};
        SOHO_K16_8x16.outputs[0] = VPUTensor(28, 7, 16, 1, DataType::UINT8);
        DPUWorkload SOHO_K16_16x16{SOHO_16x16};
        SOHO_K16_16x16.outputs[0] = VPUTensor(28, 7, 16, 1, DataType::UINT8);
        DPUWorkload SOHO_K16_4x16{SOHO_4x16};
        SOHO_K16_4x16.outputs[0] = VPUTensor(28, 7, 16, 1, DataType::UINT8);

        // all tiles are the same
        const DPUWorkload SOK_16x16{
                VPUDevice::VPU_4_0,
                Operation::CONVOLUTION,
                {VPUTensor(28, 28, 256, 1, DataType::UINT8)},  // input dimensions
                {VPUTensor(28, 28, 32, 1, DataType::UINT8)},   // output dimensions
                {1, 1},                                        // kernels
                {1, 1},                                        // strides
                {0, 0, 0, 0},                                  // padding
                ExecutionMode::CUBOID_16x16,                   // execution mode
                ActivationFunction::NONE,                      // activation
                0.0F,                                          // act_sparsity
                0.0F,                                          // weight_sparsity
                {swz_def, swz_def},                            // input_swizzling
                {swz_def},                                     // output_swizzling
                4,                                             // output_write_tiles
                {0, 0, 0, 0},                                  // offsets
                ISIStrategy::SPLIT_OVER_K,                     // isi_strategy
                false,                                         // weight_sparsity_enabled
        };
        const DPUWorkload SOK_8x16{mod_execution(SOK_16x16, ExecutionMode::CUBOID_8x16)};
        const DPUWorkload SOK_4x16{mod_execution(SOK_16x16, ExecutionMode::CUBOID_4x16)};

        DPUWorkload SOK_16x16_noB{SOK_16x16};
        SOK_16x16_noB.output_write_tiles = 1;
        SOK_16x16_noB.isi_strategy = ISIStrategy::CLUSTERING;
        const DPUWorkload SOK_8x16_noB{mod_execution(SOK_16x16_noB, ExecutionMode::CUBOID_8x16)};
        const DPUWorkload SOK_4x16_noB{mod_execution(SOK_16x16_noB, ExecutionMode::CUBOID_4x16)};

        {
            Logger::clear2ndlog();
            std::string err_info;

            {
                std::cout << "\n ------- SPLITS: ------- ";
                case_run(SOHO_8x16, "\nTEST of: SOHO 8x16 ");  // v17:4358  4921
                case_run(SOK_8x16, "TEST of: SOK 8x16 ");      // v17:4914  3610
                case_run(SOK_8x16_noB, "TEST of: SOK_noB 8x16 ");
                case_run(HK_8x16, "TEST of: HK 8x16 ");  // v17:4717  4935

                case_run(SOHO_16x16, "\nTEST of: SOHO 16x16 ");  // v17:4429  4416
                case_run(SOK_16x16, "TEST of: SOK 16x16 ");      // v17:4233  3138
                case_run(SOK_16x16_noB, "TEST of: SOK_noB 16x16 ");
                case_run(HK_16x16, "TEST of: HK 16x16 ");  // v17:4584  4421

                case_run(SOHO_4x16, "\nTEST of: SOHO 4x16 ");  // v17: 4472  4726
                case_run(SOK_4x16, "TEST of: SOK 4x16 ");      // v17: 6259  4515
                case_run(SOK_4x16_noB, "TEST of: SOK_noB 4x16 ");
                case_run(HK_4x16, "TEST of: HK 4x16 ");  // v17: 4542  4751
            }

            {
                std::cout << "\n ------- SPLITS: (intra tiles) ------- ";
                case_run(SOHO_K64_8x16, "\nTEST of: SOHO k64 8x16 ");  // v17:2172  2239  New NN:2116
                case_run(SOHO_K64_16x16, "TEST of: SOHO k64 16x16 ");  // v17:2261  2126  New NN:2153
                case_run(SOHO_K64_4x16, "TEST of: SOHO k64 4x16 ");    //                 New NN:2531

                case_run(SOHO_K32_8x16, "\nTEST of: SOHO k32 8x16 ");  // v17:1318  1300  New NN:867
                case_run(SOHO_K32_16x16, "TEST of: SOHO k32 16x16 ");  // v17:1231  1030  New NN:867
                case_run(SOHO_K32_4x16, "TEST of: SOHO k32 4x16 ");    //                 New NN:807

                case_run(SOHO_K16_8x16, "\nTEST of: SOHO k16 8x16 ");  // v17:1303  798  New NN:704
                case_run(SOHO_K16_16x16, "TEST of: SOHO k16 16x16 ");  // v17:1001  629  New NN:651
                case_run(SOHO_K16_4x16, "TEST of: SOHO k16 4x16 ");    //                New NN:611
            }
        }
    }
}

TEST_F(VPULayerCM_InvestigationTestNPU4x, MEXP_C2_CONV_4648_4676_EISXW_126389_NPU40) {
    const bool force_fail{false};        // controls force failing assertion
    const int fail{force_fail ? 0 : 1};  // 1 neutral, 0 fail

    // EXPECT_TRUE(false);                  // something has to fail to  see couts
    show_split = false;

    DPUWorkload wl_4648{
            // orig Layer
            VPUDevice::VPU_4_0,
            Operation::CONVOLUTION,
            {VPUTensor(28, 28, 128, 1, DataType::UINT8, Layout::ZXY)},  // input dimensions
            {VPUTensor(28, 28, 256, 1, DataType::UINT8, Layout::ZXY)},  // output dimensions
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

    DPUWorkload wl_4676{
            // orig Layer
            VPUDevice::VPU_4_0,
            Operation::CONVOLUTION,
            {VPUTensor(28, 28, 128, 1, DataType::UINT8, Layout::ZXY)},  // input dimensions
            {VPUTensor(14, 14, 512, 1, DataType::UINT8, Layout::ZXY)},  // output dimensions
            {3, 3},                                                     // kernels
            {2, 2},                                                     // strides
            {0, 1, 0, 1},                                               // padding
            ExecutionMode::CUBOID_8x16,                                 // execution mode
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

    std::cout << "\n ------- wl_4648 ------- \n";
    {
        const DPULayer tst_layer(wl_4648);
        // TODO: test should be like SOK wins
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 3000, fail * 3900 + 1000},  // v17:4495
                 "SOHO /4 , no memmove, "},

                {{tst_layer, {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOK, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 3500 - 800, fail * 3900 + 1000},  // v17:4384
                 "SOK /4 , no memmove, "},
        };
        executeTests(tests);
    }

    std::cout << "\n ------- wl_4676 ------- \n";
    {
        const DPULayer tst_layer(wl_4676);
        // TODO: test should be like SOHO wins
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 18000 - 3000, fail * 18500 + 1000},  // v17:18556
                 "SOHO /4 , no memmove, "},

                {{tst_layer, {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOK, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 18000 - 2500, fail * 18000 + 1000},  // v17:18685
                 "SOK /4 , no memmove, "},
        };
        executeTests(tests);
    }

    VPULayerCostModel& theModel = layer_models.getModel(VPUDevice::VPU_4_0);

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

    auto mod_execution = [](const DPUWorkload& wl, ExecutionMode em) -> DPUWorkload {
        DPUWorkload wl_{wl};
        wl_.execution_order = em;
        return wl_;
    };

    //  EXPECT_TRUE(false);  // something has to fail to  see couts

    // low level WL
    {
        const DPUWorkload SOHO_4648_8x16{
                VPUDevice::VPU_4_0,
                Operation::CONVOLUTION,
                {VPUTensor(28, 7, 128, 1, DataType::UINT8)},  // input dimensions
                {VPUTensor(28, 7, 256, 1, DataType::UINT8)},  // output dimensions
                {1, 1},                                       // kernels
                {1, 1},                                       // strides
                {0, 0, 0, 0},                                 // padding
                ExecutionMode::CUBOID_8x16,                   // execution mode
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

        DPUWorkload HK_4648_8x16{SOHO_4648_8x16};
        HK_4648_8x16.output_write_tiles = 4;

        const DPUWorkload SOHO_4648_16x16{mod_execution(SOHO_4648_8x16, ExecutionMode::CUBOID_16x16)};
        DPUWorkload HK_4648_16x16{SOHO_4648_16x16};
        HK_4648_16x16.output_write_tiles = 4;

        const DPUWorkload SOHO_4648_4x16{mod_execution(SOHO_4648_8x16, ExecutionMode::CUBOID_4x16)};
        DPUWorkload HK_4648_4x16{SOHO_4648_4x16};
        HK_4648_4x16.output_write_tiles = 4;

        const DPUWorkload SOK_4648_4x16{
                VPUDevice::VPU_4_0,
                Operation::CONVOLUTION,
                {VPUTensor(28, 28, 128, 1, DataType::UINT8)},  // input dimensions
                {VPUTensor(28, 28, 64, 1, DataType::UINT8)},   // output dimensions
                {1, 1},                                        // kernels
                {1, 1},                                        // strides
                {0, 0, 0, 0},                                  // padding
                ExecutionMode::CUBOID_4x16,                    // execution mode
                ActivationFunction::NONE,                      // activation
                0.0F,                                          // act_sparsity
                0.0F,                                          // weight_sparsity
                {swz_def, swz_def},                            // input_swizzling
                {swz_def},                                     // output_swizzling
                4,                                             // output_write_tiles
                {0, 0, 0, 0},                                  // offsets
                ISIStrategy::SPLIT_OVER_K,                     // isi_strategy
                false,                                         // weight_sparsity_enabled
        };

        const DPUWorkload SOK_4648_16x16{mod_execution(SOK_4648_4x16, ExecutionMode::CUBOID_16x16)};
        const DPUWorkload SOK_4648_8x16{mod_execution(SOK_4648_4x16, ExecutionMode::CUBOID_8x16)};

        const DPUWorkload SOHO_4676_T_M_8x16{
                // top and middle
                VPUDevice::VPU_4_0,
                Operation::CONVOLUTION,
                {VPUTensor(28, 9, 128, 1, DataType::UINT8)},  // input dimensions
                {VPUTensor(14, 4, 512, 1, DataType::UINT8)},  // output dimensions
                {3, 3},                                       // kernels
                {2, 2},                                       // strides
                {0, 0, 0, 1},                                 // padding
                ExecutionMode::CUBOID_8x16,                   // execution mode
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

        DPUWorkload HK_4676_T_M_8x16{SOHO_4676_T_M_8x16};
        HK_4676_T_M_8x16.output_write_tiles = 4;

        const DPUWorkload SOHO_4676_T_M_16x16{mod_execution(SOHO_4676_T_M_8x16, ExecutionMode::CUBOID_16x16)};
        DPUWorkload HK_4676_T_M_16x16{SOHO_4676_T_M_16x16};
        HK_4676_T_M_16x16.output_write_tiles = 4;

        const DPUWorkload SOHO_4676_T_M_4x16{mod_execution(SOHO_4676_T_M_8x16, ExecutionMode::CUBOID_4x16)};
        DPUWorkload HK_4676_T_M_4x16{SOHO_4676_T_M_4x16};
        HK_4676_T_M_4x16.output_write_tiles = 4;

        //////////////////////////////////////////////////////////////

        DPUWorkload SOHO_4676_B_8x16{SOHO_4676_T_M_8x16};
        SOHO_4676_B_8x16.inputs[0].set_shape({28, 4, 128, 1});
        SOHO_4676_B_8x16.outputs[0].set_shape({14, 2, 512, 1});
        SOHO_4676_B_8x16.padding = {0, 1, 0, 1};

        DPUWorkload HK_4676_B_8x16{SOHO_4676_B_8x16};
        HK_4676_B_8x16.output_write_tiles = 4;

        const DPUWorkload SOHO_4676_B_16x16{mod_execution(SOHO_4676_B_8x16, ExecutionMode::CUBOID_16x16)};
        DPUWorkload HK_4676_B_16x16{SOHO_4676_B_16x16};
        HK_4676_B_16x16.output_write_tiles = 4;

        const DPUWorkload SOHO_4676_B_4x16{mod_execution(SOHO_4676_B_8x16, ExecutionMode::CUBOID_4x16)};
        DPUWorkload HK_4676_B_4x16{SOHO_4676_B_4x16};
        HK_4676_B_4x16.output_write_tiles = 4;

        //////////////////////////////////////////////////////////////

        const DPUWorkload SOK_4676_16x16{
                VPUDevice::VPU_4_0,
                Operation::CONVOLUTION,
                {VPUTensor(28, 28, 128, 1, DataType::UINT8)},  // input dimensions
                {VPUTensor(14, 14, 128, 1, DataType::UINT8)},  // output dimensions
                {3, 3},                                        // kernels
                {2, 2},                                        // strides
                {0, 1, 0, 1},                                  // padding
                ExecutionMode::CUBOID_16x16,                   // execution mode
                ActivationFunction::NONE,                      // activation
                0.0F,                                          // act_sparsity
                0.0F,                                          // weight_sparsity
                {swz_def, swz_def},                            // input_swizzling
                {swz_def},                                     // output_swizzling
                4,                                             // output_write_tiles
                {0, 0, 0, 0},                                  // offsets
                ISIStrategy::SPLIT_OVER_K,                     // isi_strategy
                false,                                         // weight_sparsity_enabled
        };
        const DPUWorkload SOK_4676_8x16{mod_execution(SOK_4676_16x16, ExecutionMode::CUBOID_8x16)};
        const DPUWorkload SOK_4676_4x16{mod_execution(SOK_4676_16x16, ExecutionMode::CUBOID_4x16)};

        {
            Logger::clear2ndlog();
            std::string err_info;

            {
                std::cout << "\n ------- SPLITS: wl_4648 ------- \n";
                /////////////////////////////// SOHO /////////////////////////////////
                case_run(SOHO_4648_16x16, "TEST of: SOHO 16x16 ");  // v17:4645  4779
                case_run(SOHO_4648_8x16, "TEST of: SOHO 8x16  ");   // v17:4495  5175
                case_run(SOHO_4648_4x16, "TEST of: SOHO 4x16  ");   // v17:4682  5581

                /////////////////////////////// HK /////////////////////////////////
                case_run(HK_4648_16x16, "TEST of: HK 16x16 ");  // v17:4762  4802
                case_run(HK_4648_8x16, "TEST of: HK 8x16  ");   // v17:4775  5187
                case_run(HK_4648_4x16, "TEST of: HK 4x16  ");   // v17:4861  5615

                /////////////////////////////// SOK /////////////////////////////////
                case_run(SOK_4648_16x16, "TEST of: SOK 16x16 ");  // v17:4994  5099
                case_run(SOK_4648_8x16, "TEST of: SOK 8x16 ");    // v17:4606  4156
                case_run(SOK_4648_4x16, "TEST of: SOK 4x16 ");    // v17:4384  4639
            }

            {
                std::cout << "\n ------- SPLITS: wl_4676 ------- \n";
                /////////////////////////////// SOHO /////////////////////////////////
                case_run(SOHO_4676_T_M_16x16, "TEST of: SOHO top and middle 16x16 ");  // v17:20036  17596
                case_run(SOHO_4676_B_16x16, "TEST of: SOHO bottom 16x16 ");            // v17:19195  17372

                case_run(SOHO_4676_T_M_8x16, "TEST of: SOHO top and middle 8x16 ");  // v17:18556  17085
                case_run(SOHO_4676_B_8x16, "TEST of: SOHO bottom 8x16 ");            // v17:18439  16788

                case_run(SOHO_4676_T_M_4x16, "TEST of: SOHO top and middle 4x16 ");  // v17:20822  18689
                case_run(SOHO_4676_B_4x16, "TEST of: SOHO bottom 4x16 ");            // v17:20452  18425

                /////////////////////////////// HK /////////////////////////////////
                case_run(HK_4676_T_M_16x16, "TEST of: HK top and middle 16x16  ");  // v17:20049  17649
                case_run(HK_4676_B_16x16, "TEST of: HK bottom 16x16 ");             // v17:19199  17419

                case_run(HK_4676_T_M_8x16, "TEST of: HK top and middle 8x16 ");  // v17:18647  17157
                case_run(HK_4676_B_8x16, "TEST of: HK bottom 8x16 ");            // v17:18517  16849

                case_run(HK_4676_T_M_4x16, "TEST of: HK top and middle 4x16 ");  // v17:20986  18758
                case_run(HK_4676_B_4x16, "TEST of: HK bottom 4x16 ");            // v17:20586  18485

                /////////////////////////////// SOK /////////////////////////////////
                case_run(SOK_4676_16x16, "TEST of: SOK 16x16 ");  // v17:18685  15078
                case_run(SOK_4676_8x16, "TEST of: SOK 8x16 ");    // v17:19461  16056
                case_run(SOK_4676_4x16, "TEST of: SOK 4x16 ");    // v17:19284  16226
            }
        }
    }
}

TEST_F(VPULayerCM_InvestigationTestNPU4x, MoreTiles_MAXP_EISXW_99246_NPU40) {
    const VPUNN::DPUWorkload wl_MXP_layer{
            VPUNN::VPUDevice::VPU_4_0,
            VPUNN::Operation::MAXPOOL,
            {VPUNN::VPUTensor(112, 112, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(56, 56, 64, 1, VPUNN::DataType::UINT8)},    // output dimensions
            {3, 3},                                                       // kernels
            {2, 2},                                                       // strides
            {1, 0, 1, 0},                                                 // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                           // execution mode
            VPUNN::ActivationFunction::NONE,                              // activation
            0.0F,                                                         // act_sparsity
            0.0F,                                                         // weight_sparsity
            {swz_def, swz_def},                                           // input_swizzling
            {swz_def},                                                    // output_swizzling
            1,                                                            // output_write_tiles
            {0, 0, 0, 0},                                                 // offsets
            VPUNN::ISIStrategy::CLUSTERING,                               // isi_strategy
            false,                                                        // weight_sparsity_enabled
    };

    // const std::string altModelName(NameHelperNN::get_model_root() + "vpu_2_7-150.vpunn");
    // VPULayerCostModel model_alternative{altModelName};
    bool prefetch{true};
    VPULayerCostModel& theModel = layer_models.getModel(VPUDevice::VPU_4_0);

    {
        const VPUNN::DPULayer tst_layer(wl_MXP_layer);

        const std::vector<TestCase> tests{
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, prefetch}},
                // {VPUNN::Cycles::NO_ERROR, true, 21000, 21000 + 1100},  // v17 21341   v159NN:22003
                // "CLU /2, no memmove, "},
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                // {VPUNN::Cycles::NO_ERROR, true, 11000, 11000 + 1000},  // v17 11164    v159NN:11612
                // "SOHO /2, no memmove, "},

                {{std::move(tst_layer), {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 4000 /*6000*/, 4500 + 500},  // v17 6132    v159NN:6234/ GTL:4500
                 "SOHO /4 , no memmove, "},

                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_HaloRead, false, false, prefetch}},
                // {VPUNN::Cycles::NO_ERROR, true, 11000, 11000 + 2000},  // v17  11540   v159NN:12680
                // "SOH H /2 , no memmove, "},
                //{{tst_layer, {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOH_HaloRead, false, false, prefetch}},
                // {VPUNN::Cycles::NO_ERROR, true, 6500, 6500 + 750},  // v17 6674    v159NN:7201
                // "SOH H /4 , no memmove, "},

        };
        executeTests(tests);
    }
    const std::string nline{"\n ------------- NEW TEST------------------------------------ ------------------"};
    auto verify_cost_cyc = [&nline, &wl_MXP_layer, prefetch, &theModel](unsigned int nTiles,
                                                                        CyclesInterfaceType& cost_cyc) {
        std::cout << nline;
        std::cout << "\n TILES: " << nTiles << "\n";

        VPUNN::DPULayer tst_layer(wl_MXP_layer);
        const VPULayerStrategy strategy{1U,    1U,    nTiles,  VPUNN::VPUTilingStrategy::SOH_Overlapped,
                                        false, false, prefetch};

        Logger::clear2ndlog();
        // CyclesInterfaceType cost_cyc{};
        LayerSplitInfo detailed_split;
        ASSERT_NO_THROW(cost_cyc = theModel.Layer(tst_layer, strategy.tiling_strategy, strategy.nDPUs, strategy.nTiles,
                                                  false, false, prefetch, detailed_split))
                << tst_layer << strategy << cost_cyc;
        // EXPECT_EQ(cost_cyc, 2) << tst_layer << toStringLayerSplitInfo(detailed_split) << Logger::get2ndlog();

        EXPECT_EQ(detailed_split.size(), strategy.nTiles) << detailed_split.size();
        for (int i = 0; i < static_cast<int>(detailed_split.size()); i++) {
            // make sure that the cost of workloads that were inferred to be the best after
            ///< performing the intra-tile split algorithm is not an error code
            EXPECT_FALSE(Cycles::isErrorCode(detailed_split[i].best_intra_tile_split.first));

            // Zero value is not an error and can communicate something like it cannot solve the request.
            EXPECT_NE(detailed_split[i].best_intra_tile_split.first,
                      0);  // best_intra_tile_split is a pair, cost is the first element of that pair
        }

        // make sure that cost_cyc is not an error code
        EXPECT_FALSE(Cycles::isErrorCode(cost_cyc)) << Cycles::toErrorText(cost_cyc);
    };
    // element wise
    {
        CyclesInterfaceType cost_cyc_1Tiles;
        CyclesInterfaceType cost_cyc_2Tiles;
        CyclesInterfaceType cost_cyc_4Tiles;

        // calculate cost_cyc for a layer, using different values for tiles
        verify_cost_cyc(1U, cost_cyc_1Tiles);
        verify_cost_cyc(2U, cost_cyc_2Tiles);
        verify_cost_cyc(4U, cost_cyc_4Tiles);

        // check if the cost_cyc for a larger number of tiles is smaller than the cost_cyc for a smaller number of
        // tiles
        ASSERT_GT(cost_cyc_1Tiles, cost_cyc_2Tiles);
        ASSERT_GT(cost_cyc_2Tiles, cost_cyc_4Tiles);
        ASSERT_GT(cost_cyc_1Tiles, cost_cyc_4Tiles);
    }
}

TEST_F(VPULayerCM_InvestigationTestNPU4x, CONV_Act_sparsity_EISXW_117195_INT_NPU40) {
    const bool force_fail{false};        // controls force failing assertion
    const int fail{force_fail ? 0 : 1};  // 1 neutral, 0 fail
                                         // EXPECT_TRUE(false);                  // something has to fail to  see couts
    show_split = true;

    DPUWorkload wl_ref{
            // orig Layer
            VPUDevice::VPU_4_0,
            Operation::CONVOLUTION,
            {VPUTensor(28, 28, 256, 1, DataType::UINT8, Layout::ZXY)},  // input dimensions
            {VPUTensor(14, 14, 256, 1, DataType::UINT8, Layout::ZXY)},  // output dimensions
            {3, 3},                                                     // kernels
            {2, 2},                                                     // strides
            {1, 0, 1, 0},                                               // padding
            ExecutionMode::CUBOID_16x16,                                // execution mode
            ActivationFunction::NONE,                                   // activation
            0.0F,                                                       // act_sparsity     // in ticket is: 0.6
            0.0F,                                                       // weight_sparsity  // in ticket is: 0.400662
            {swz_def, swz_def},                                         // input_swizzling
            {swz_def},                                                  // output_swizzling
            1,                                                          // output_write_tiles
            {0, 0, 0, 0},                                               // offsets
            ISIStrategy::CLUSTERING,                                    // isi_strategy
            false,                                                      // weight_sparsity_enabled
    };

    const float act_sprsty{0.6F};
    const float wt_sprsty{0.400662F};

    const bool prefetch{true};
    {
        const DPULayer tst_layer_no_spars(wl_ref);
        const DPULayer tst_layer_input_spars(wl_sparsity_initialization(wl_ref, true, act_sprsty, false, 0.0F));
        const DPULayer tst_layer_weight_spars(wl_sparsity_initialization(wl_ref, false, 0.0F, true, wt_sprsty));
        const DPULayer tst_layer_dualspars(wl_sparsity_initialization(wl_ref, true, act_sprsty, true, wt_sprsty));

        const std::vector<TestCase> tests{
                // no sparsity
                {{tst_layer_no_spars, {1U, 1U, 4U, VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {Cycles::NO_ERROR, true, 16500,
                  fail * (20000)},  // v17:18409   v159nn:18301   // GTM:18754  //GTL: 19300 (16x16)
                 "SOHO, No sparsity "},

                {{tst_layer_no_spars, {1U, 1U, 4U, VPUTilingStrategy::SOK, false, false, prefetch}},
                 {Cycles::NO_ERROR, true, 15500,
                  fail * (18000 + 2000)},  // v17:31319 (19134 owt)   v159nn:17898  GTM:18915 GTL:19081
                 "SOK , No sparsity "},

                ////////////////////////////////////////////////////////////////

                // input sparsity
                {{tst_layer_input_spars, {1U, 1U, 4U, VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {Cycles::NO_ERROR, true, 8000,
                  fail * (9500 + 13000)},    // v17:13151   v159nn:18301 (NA)  //GTM: 13145 GTL 9500
                 "SOHO , Input sparsity "},  // HUGE error

                {{tst_layer_input_spars, {1U, 1U, 4U, VPUTilingStrategy::SOK, false, false, prefetch}},
                 {Cycles::NO_ERROR, true, 5200,
                  fail * (9500 + 4000)},   // v17:15820 (owt 8685)  v159nn:17898 (NA) GTM:8717  /GTL 9500
                 "SOK , Input sparsity"},  // BIG error

                ////////////////////////////////////////////////////////////////

                // weight sparsity
                {{tst_layer_weight_spars, {1U, 1U, 4U, VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {Cycles::NO_ERROR, true, 12000, fail * (12000 + 4000)},  // v17:12190   v159nn:14519  // GTML::14189
                 "SOHO , Weight sparsity "},

                {{tst_layer_weight_spars, {1U, 1U, 4U, VPUTilingStrategy::SOK, false, false, prefetch}},
                 {Cycles::NO_ERROR, true, 10500,
                  fail * (12000 + 3000)},  // v17:21485 (owt 12979)   v159nn:13590 GTML:14332
                 "SOK , Weight sparsity "},

                ////////////////////////////////////////////////////////////////

                // dualsparsity
                {{tst_layer_dualspars, {1U, 1U, 4U, VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {Cycles::NO_ERROR, true, 10500,
                  fail * (8100 + 5000)},  // v17:12190 VPUNNticket: 15249  VPUXGT: 4734ns
                                          // cyc:8047 @1700    v159nn:14519  GTM: 10481 GTL: 8000?
                 "SOHO , Dualsparsity"},

                {{tst_layer_dualspars, {1U, 1U, 4U, VPUTilingStrategy::SOK, false, false, prefetch}},
                 {Cycles::NO_ERROR, true, 5000,
                  fail * (6100 + 4900)},  // v17:15820 ( 8685owt) VPUNNticket: 13722  VPUXGT: 6559ns cyc:11150 @1700
                                          // v159nn:13590  GTM: 8470 GTL:6000?
                 "SOK , Dualsparsity "},

                // SOK wins!

                // vpuxGT: SOH wins (8k-11k). WHY? what config?
                // cp17: SOH wins (15k - 12k), but SOK with owt=4 is big.  see at end
                // cp17  owt lim to 2: SOK wins  8685 to 12190
                // v159nn, no act sparsity available: SOK wins (owt less sensitive)
                // GT MTL: SOHO:10481    SOK:8470  =>  SOK wins
        };
        executeTests(tests);
    }

    // low level WL

    // SOH
    DPUWorkload wl_SOHO_Top{
            VPUDevice::VPU_4_0,
            Operation::CONVOLUTION,
            {VPUTensor(28, 8, 256, 1, DataType::UINT8, Layout::ZXY)},  // input dimensions
            {VPUTensor(14, 4, 256, 1, DataType::UINT8, Layout::ZXY)},  // output dimensions
            {3, 3},                                                    // kernels
            {2, 2},                                                    // strides
            {1, 0, 1, 0},                                              // padding
            ExecutionMode::CUBOID_8x16,                                // execution mode
            ActivationFunction::NONE,                                  // activation
            0.0F,                                                      // act_sparsity
            0.0F,                                                      // weight_sparsity
            {swz_def, swz_def},                                        // input_swizzling
            {swz_def},                                                 // output_swizzling
            1,                                                         // output_write_tiles
            {0, 0, 0, 0},                                              // offsets
            ISIStrategy::CLUSTERING,                                   // isi_strategy
            false,                                                     // weight_sparsity_enabled
    };

    DPUWorkload wl_SOHO_Mid{wl_SOHO_Top};
    wl_SOHO_Mid.inputs[0] = VPUTensor(28, 9, 256, 1, DataType::UINT8, Layout::ZXY);
    wl_SOHO_Mid.padding = {0, 0, 1, 0};

    DPUWorkload wl_SOHO_Bot{wl_SOHO_Top};
    wl_SOHO_Bot.inputs[0] = VPUTensor(28, 5, 256, 1, DataType::UINT8, Layout::ZXY);
    wl_SOHO_Bot.outputs[0] = VPUTensor(14, 2, 256, 1, DataType::UINT8, Layout::ZXY);
    wl_SOHO_Bot.padding = {0, 0, 1, 0};

    // SOK
    DPUWorkload wl_SOK_All{
            VPUDevice::VPU_4_0,
            Operation::CONVOLUTION,
            {VPUTensor(28, 28, 256, 1, DataType::UINT8, Layout::ZXY)},  // input dimensions
            {VPUTensor(14, 14, 64, 1, DataType::UINT8, Layout::ZXY)},   // output dimensions
            {3, 3},                                                     // kernels
            {2, 2},                                                     // strides
            {1, 0, 1, 0},                                               // padding
            ExecutionMode::CUBOID_16x16,                                // execution mode //IMPORTANT
            ActivationFunction::NONE,                                   // activation
            0.0F,                                                       // act_sparsity
            0.0F,                                                       // weight_sparsity
            {swz_def, swz_def},                                         // input_swizzling
            {swz_def},                                                  // output_swizzling
            4,                                                          // output_write_tiles
            {0, 0, 0, 0},                                               // offsets
            ISIStrategy::SPLIT_OVER_K,                                  // isi_strategy
            false,                                                      // weight_sparsity_enabled
    };

    VPULayerCostModel& theModel = layer_models.getModel(VPUDevice::VPU_4_0);

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

    {
        Logger::clear2ndlog();
        std::string err_info;
        // SOHO
        const DPUWorkload wl_SOHO_Top_No_spars{wl_sparsity_initialization(wl_SOHO_Top, false, 0.0f, false, 0.0f)};
        const DPUWorkload wl_SOHO_Mid_No_spars{wl_sparsity_initialization(wl_SOHO_Mid, false, 0.0f, false, 0.0f)};
        const DPUWorkload wl_SOHO_Bot_No_spars{wl_sparsity_initialization(wl_SOHO_Bot, false, 0.0f, false, 0.0f)};

        const DPUWorkload wl_SOHO_Top_Input_spars{
                wl_sparsity_initialization(wl_SOHO_Top, true, act_sprsty, false, 0.0f)};
        const DPUWorkload wl_SOHO_Mid_Input_spars{
                wl_sparsity_initialization(wl_SOHO_Mid, true, act_sprsty, false, 0.0f)};
        const DPUWorkload wl_SOHO_Bot_Input_spars{
                wl_sparsity_initialization(wl_SOHO_Bot, true, act_sprsty, false, 0.0f)};

        const DPUWorkload wl_SOHO_Top_Weight_spars{
                wl_sparsity_initialization(wl_SOHO_Top, false, 0.0f, true, wt_sprsty)};
        const DPUWorkload wl_SOHO_Mid_Weight_spars{
                wl_sparsity_initialization(wl_SOHO_Mid, false, 0.0f, true, wt_sprsty)};
        const DPUWorkload wl_SOHO_Bot_Weight_spars{
                wl_sparsity_initialization(wl_SOHO_Bot, false, 0.0f, true, wt_sprsty)};

        const DPUWorkload wl_SOHO_Top_Dualspars{
                wl_sparsity_initialization(wl_SOHO_Top, true, act_sprsty, true, wt_sprsty)};
        const DPUWorkload wl_SOHO_Mid_Dualspars{
                wl_sparsity_initialization(wl_SOHO_Mid, true, act_sprsty, true, wt_sprsty)};
        const DPUWorkload wl_SOHO_Bot_Dualspars{
                wl_sparsity_initialization(wl_SOHO_Bot, true, act_sprsty, true, wt_sprsty)};

        // SOK owt=4
        const DPUWorkload wl_SOK_No_spars{wl_sparsity_initialization(wl_SOK_All, false, 0.0f, false, 0.0f)};
        const DPUWorkload wl_SOK_Input_spars{wl_sparsity_initialization(wl_SOK_All, true, act_sprsty, false, 0.0f)};
        const DPUWorkload wl_SOK_Weight_spars{wl_sparsity_initialization(wl_SOK_All, false, 0.0f, true, wt_sprsty)};
        const DPUWorkload wl_SOK_Dualspars{wl_sparsity_initialization(wl_SOK_All, true, act_sprsty, true, wt_sprsty)};

        // SOK owt=3
        DPUWorkload wl_SOK_No_spars_owt3{wl_SOK_No_spars};
        wl_SOK_No_spars_owt3.output_write_tiles = 3;

        DPUWorkload wl_SOK_Input_spars_owt3{wl_SOK_Input_spars};
        wl_SOK_Input_spars_owt3.output_write_tiles = 3;

        DPUWorkload wl_SOK_Weight_spars_owt3{wl_SOK_Weight_spars};
        wl_SOK_Weight_spars_owt3.output_write_tiles = 3;

        DPUWorkload wl_SOK_Dualspars_owt3{wl_SOK_Dualspars};
        wl_SOK_Dualspars_owt3.output_write_tiles = 3;

        // SOK owt=2
        DPUWorkload wl_SOK_No_spars_owt2{wl_SOK_No_spars};
        wl_SOK_No_spars_owt2.output_write_tiles = 2;

        DPUWorkload wl_SOK_Input_spars_owt2{wl_SOK_Input_spars};
        wl_SOK_Input_spars_owt2.output_write_tiles = 2;

        DPUWorkload wl_SOK_Weight_spars_owt2{wl_SOK_Weight_spars};
        wl_SOK_Weight_spars_owt2.output_write_tiles = 2;

        DPUWorkload wl_SOK_Dualspars_owt2{wl_SOK_Dualspars};
        wl_SOK_Dualspars_owt2.output_write_tiles = 2;

        // SOK owt=1 + CLU
        DPUWorkload wl_SOK_No_spars_owt1{wl_SOK_No_spars};
        wl_SOK_No_spars_owt1.output_write_tiles = 1;
        wl_SOK_No_spars_owt1.isi_strategy = ISIStrategy::CLUSTERING;

        DPUWorkload wl_SOK_Input_spars_owt1{wl_SOK_Input_spars};
        wl_SOK_Input_spars_owt1.output_write_tiles = 1;
        wl_SOK_Input_spars_owt1.isi_strategy = ISIStrategy::CLUSTERING;

        DPUWorkload wl_SOK_Weight_spars_owt1{wl_SOK_Weight_spars};
        wl_SOK_Weight_spars_owt1.output_write_tiles = 1;
        wl_SOK_Weight_spars_owt1.isi_strategy = ISIStrategy::CLUSTERING;

        DPUWorkload wl_SOK_Dualspars_owt1{wl_SOK_Dualspars};
        wl_SOK_Dualspars_owt1.output_write_tiles = 1;
        wl_SOK_Dualspars_owt1.isi_strategy = ISIStrategy::CLUSTERING;

        {  // SOHO SPLITS
            std::cout << "\n ------- SOHO SPLITS: ------- ";

            case_run(wl_SOHO_Top_No_spars,
                     "\nTEST of: SOHO TOP, no sparsity\n");  //   v17: 18154  GTM:18754  v159nn:18237    GTL:19440
                                                             //   (19301:16x16)
            case_run(wl_SOHO_Mid_No_spars,
                     "\nTEST of: SOHO MID, no sparsity \n");  //   v17: 18409  GTM:18754  v159nn:18301   GTL:19400
                                                              //   (19300)
            case_run(wl_SOHO_Bot_No_spars,
                     "\nTEST of :SOHO BOT, no sparsity \n");  //   v17: 18206  GTM:18752  v159nn:18240   GTL:19350
                                                              //   (19204)
            // Top mid bot are in parallel=  (18154)||(18409)||(18206) ==> 18409
            // GTM:18754
            std::cout << "\n ------- : ------- ";

            // input sparsity
            case_run(wl_SOHO_Top_Input_spars,
                     "\nTEST of: SOHO TOP, input sparsity\n");  //   v17: 13107  GTM:13114 (intrp)  v159nn:18237
            case_run(wl_SOHO_Mid_Input_spars,
                     "\nTEST of: SOHO MID, input sparsity \n");  //   v17: 13151  GTM:13145 (intrp) v159nn:18301
            case_run(wl_SOHO_Bot_Input_spars,
                     "\nTEST of :SOHO BOT, input sparsity \n");  //   v17: 13025  GTM:12904 (intrp) v159nn:18240
            // Top mid bot are in parallel=  (13107)||(13151)||(13025) ==> 13151
            // GTM: 13145
            std::cout << "\n ------- : ------- ";

            // weight sparsity
            case_run(wl_SOHO_Top_Weight_spars,
                     "\nTEST of: SOHO TOP, weight sparsity\n");  //   v17: 11878  GTM:14188 v159nn:14427
            case_run(wl_SOHO_Mid_Weight_spars,
                     "\nTEST of: SOHO MID, weight sparsity \n");  //   v17: 12012  GTM:14189 v159nn:14519
            case_run(wl_SOHO_Bot_Weight_spars,
                     "\nTEST of :SOHO BOT, weight sparsity \n");  //   v17: 12190  GTM:14186 v159nn:14449
            // Top mid bot are in parallel=  (11878)||(12012)||(12190) ==> 12190
            // GTM::14189
            std::cout << "\n ------- : ------- ";

            //  dualsparsity
            case_run(wl_SOHO_Top_Dualspars,
                     "\nTEST of: SOHO TOP, dual sparsity\n");  //   v17: 11878  GTM:10457 (intrp)   v159nn:14427
            case_run(wl_SOHO_Mid_Dualspars,
                     "\nTEST of: SOHO MID, dual sparsity \n");  //   v17: 12012  GTM:10481 (intrp)   v159nn:14519
            case_run(wl_SOHO_Bot_Dualspars,
                     "\nTEST of :SOHO BOT, dual sparsity \n");  //   v17: 12190  GTM:10354 (intrp)  v159nn:14449
            // Top mid bot are in parallel=  (11878)||(12012)||(12190) ==> 12190
            // v159NNversion: ==>14519
            // GTM: 10481
        }

        // SOK GTM to be reanalyzed/redone!
        {  /// SOK splits owt=4
            std::cout << "\n ------- SOK SPLITS: owt 4 ------- ";
            case_run(wl_SOK_No_spars,
                     "\nTEST of:SOK, no sparsity, owt4\n");  // v17: 31631(19318)  GTM:19472 v159nn:17898
            case_run(wl_SOK_Input_spars,
                     "\nTEST of:SOK, input sparsity, owt4\n");  // v17: 15820  GTM:   v159nn:17898 (na)
            case_run(wl_SOK_Weight_spars,
                     "\nTEST of:SOK, weight sparsity, owt4\n");  // v17: 21485  GTM:   v159nn:13590
            case_run(wl_SOK_Dualspars,
                     "\nTEST of:SOK, input and weight sparsity, owt4\n");  // v17: 15820  GTM: v159nn:13590
        }

        {  /// SOK splits owt=3
            std::cout << "\n ------- SOK SPLITS: owt 3 ------- ";
            case_run(wl_SOK_No_spars_owt3, "\nTEST of:SOK, no sparsity, owt3\n");  // v17: 19537(!)  GTM:na v159nn:17953
            case_run(wl_SOK_Input_spars_owt3,
                     "\nTEST of:SOK, input sparsity, owt3\n");  // v17: 9071   GTM: v159nn:17953
            case_run(wl_SOK_Weight_spars_owt3,
                     "\nTEST of:SOK, weight sparsity, owt3\n");  // v17: 13293  GTM:    v159nn:13558
            case_run(wl_SOK_Dualspars_owt3,
                     "\nTEST of:SOK, dual sparsity, owt3\n");  // v17: 9071   GTM:    v159nn:13558
        }
        //(!) before owr lim to 2!

        {  /// SOK splits owt=2
            std::cout << "\n ------- SOK SPLITS: owt 2 ------- ";
            case_run(wl_SOK_No_spars_owt2, "\nTEST of:SOK, no sparsity, owt2\n");  // v17: 19134  GTM:18915 v159nn:18019
            case_run(wl_SOK_Input_spars_owt2,
                     "\nTEST of:SOK, input sparsity, owt2\n");  // v17: 8685   GTM:8717  v159nn:  18019
            case_run(wl_SOK_Weight_spars_owt2,
                     "\nTEST of:SOK, weight sparsity, owt2\n");  // v17: 12979  GTM:14332    v159nn:13503
            case_run(wl_SOK_Dualspars_owt2,
                     "\nTEST of:SOK, input and weight sparsity, owt2\n");  // v17: 8685   GTM:8470    v159nn:13503
        }

        {  /// SOK splits owt=1 + CLU
            std::cout << "\n ------- SOK SPLITS: owt 1 + CLU ------- ";
            case_run(wl_SOK_No_spars_owt1,
                     "\nTEST of:SOK, no sparsity, owt1 + CLU\n");  // v17: 19134  GTM:18738 v159nn:18020
            case_run(wl_SOK_Input_spars_owt1,
                     "\nTEST of:SOK, input sparsity, owt1 + CLU\n");  // v17: 8598   GTM:8540    v159nn:18020
            case_run(wl_SOK_Weight_spars_owt1,
                     "\nTEST of:SOK, weight sparsity, owt1 + CLU\n");  // v17: 12861  GTM:14153    v159nn:13697
            case_run(wl_SOK_Dualspars_owt1,
                     "\nTEST of:SOK, input and weight sparsity, owt1 + CLU\n");  // v17: 8598 GTM:8200     v159nn:13697
        }

        // vpouxGT: SOH: 4734ns cycles: 8047@1700   , SOK:6559ns cyc:11150@1700      =>SOH wins

        // cp17: SOH:12190, SOK(owt4): 15820, ==> SOH Wins,      but (owt=1,2): SOK =8811 (act spars) ==> SOK wins due
        // to act sparsity
        // v159nn: no act spars: SOH:14519, SOK(owt4): 13590, SOK Wins,   but (owt=1,2): SOK =13503 (wt
        // spars) ==> SOK still wins
        // based on this  the implementation will limit OWT to 2 fro VPU2.7 trained data.
        // GTM:   SOHO: 10481   SOK?: 8100
    }
}

TEST_F(VPULayerCM_InvestigationTestNPU4x, DWConv_SOK_SOH_decision_EISXW_117314_2d_3_NPU40) {
    // mobilenet_v1/MobilenetV1/Conv2d_3_depthwise/depthwise
    // direct WL @ DPU level tests on
    const bool force_fail{false};        // controls force failing assertion
    const int fail{force_fail ? 0 : 1};  // 1 neutral, 0 fail
    show_split = false;
    // EXPECT_TRUE(false);

    const DPUWorkload wl_2d_3{
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

    {  // note: Real LNL NN has big delta vs GTL
        const DPULayer tst_layer(wl_2d_3);
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 5000,  // optimized for LNL 3x3 K32x4
                  fail * 5001 + 1500},  // 2T:22396  4T: 12120  GTM:12596  (intratile: K64x2)   VPUXVPUNN(old
                                        // v):11630. v159NN: 12220  GTL: 9518 (K64x2), 5648 (K32x4)
                 "SOHO , no memmove, "},

                {{tst_layer, {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOK, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 4500,  // optimized for LNL 3x3 K32x4
                  fail * 4500 + 1000},  // 2T:21805   4T:11721 (11026 owt)  GTM:10752 for OWT=2 (intratile x1)
                                        // VPUXVPUNN(old v): 9951, v159NN:9688  GTL: ~4900 (K32)
                 "SOK , no memmove, "},

                //// note: SOHH is not possible on NPU40!
                //{{tst_layer, {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOH_HaloRead, false, false, prefetch}},
                // {VPUNN::Cycles::NO_ERROR, true, 6000,  // optimized for LNL 3x3 K32x4
                //  fail * 6500 + 2000},                  // 2T: 23556    4T:14410:15816  GTM:? v159NN: 13760
                // "SOH Halo , no memmove, "},

                {{tst_layer, {1U, 1U, 1U, VPUNN::VPUTilingStrategy::NONE, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 16500,  // optimized for LNL 3x3 K32x4  GTL: 4x4900 = 19600 max
                  19500 + 2000},                         // CLU:43042  GTM:?
                 "FUll , no memmove, "},

                // note:SOK wins  (with or without LNL optimization)
        };
        executeTests(tests);
    }

    VPULayerCostModel& theModel = layer_models.getModel(VPUDevice::VPU_4_0);

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
        const DPUWorkload wl_SOHO_Top_K64x2{
                VPUDevice::VPU_4_0,
                Operation::DW_CONVOLUTION,
                {VPUTensor(56, 15, 64, 1, DataType::UINT8)},  // input dimensions
                {VPUTensor(56, 14, 64, 1, DataType::UINT8)},  // output dimensions
                {3, 3},                                       // kernels
                {1, 1},                                       // strides
                {1, 0, 1, 1},                                 // padding
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
        DPUWorkload wl_SOHO_Mid_K64x2{wl_SOHO_Top_K64x2};
        wl_SOHO_Mid_K64x2.inputs[0] = VPUTensor(56, 16, 64, 1, DataType::UINT8);
        wl_SOHO_Mid_K64x2.padding = {0, 0, 1, 1};

        DPUWorkload wl_SOHO_Bot_K64x2{wl_SOHO_Top_K64x2};
        wl_SOHO_Bot_K64x2.padding = {0, 1, 1, 1};

        // SOHO K32
        DPUWorkload wl_SOHO_Top_K32x4{wl_SOHO_Top_K64x2};
        wl_SOHO_Top_K32x4.inputs[0] = VPUTensor(56, 15, 32, 1, DataType::UINT8);
        wl_SOHO_Top_K32x4.outputs[0] = VPUTensor(56, 14, 32, 1, DataType::UINT8);

        DPUWorkload wl_SOHO_Mid_K32x4{wl_SOHO_Mid_K64x2};
        wl_SOHO_Mid_K32x4.inputs[0] = VPUTensor(56, 16, 32, 1, DataType::UINT8);
        wl_SOHO_Mid_K32x4.outputs[0] = VPUTensor(56, 14, 32, 1, DataType::UINT8);

        DPUWorkload wl_SOHO_Bot_K32x4{wl_SOHO_Bot_K64x2};
        wl_SOHO_Bot_K32x4.inputs[0] = VPUTensor(56, 15, 32, 1, DataType::UINT8);
        wl_SOHO_Bot_K32x4.outputs[0] = VPUTensor(56, 14, 32, 1, DataType::UINT8);

        // SOK

        const DPUWorkload wl_SOK_All_K32x1{
                VPUDevice::VPU_4_0,
                Operation::DW_CONVOLUTION,
                {VPUTensor(56, 56, 32, 1, DataType::UINT8)},  // input dimensions
                {VPUTensor(56, 56, 32, 1, DataType::UINT8)},  // output dimensions
                {3, 3},                                       // kernels
                {1, 1},                                       // strides
                {1, 1, 1, 1},                                 // padding
                ExecutionMode::CUBOID_16x16,                  // execution mode
                ActivationFunction::NONE,                     // activation
                0.0F,                                         // act_sparsity
                0.0F,                                         // weight_sparsity
                {swz_def, swz_def},                           // input_swizzling
                {swz_def},                                    // output_swizzling
                4,                                            // output_write_tiles
                {0, 0, 0, 0},                                 // offsets
                ISIStrategy::SPLIT_OVER_K,                    // isi_strategy
                false,                                        // weight_sparsity_enabled
        };
        DPUWorkload wl_SOK_All_K32x1_OWT2{wl_SOK_All_K32x1};
        // wl_SOK_All_K32x1_OWT2.isi_strategy=ISIStrategy::CLUSTERING
        wl_SOK_All_K32x1_OWT2.output_write_tiles = 2;

        DPUWorkload wl_SOK_All_K32x1_ForceCLU{wl_SOK_All_K32x1};
        wl_SOK_All_K32x1_ForceCLU.isi_strategy = ISIStrategy::CLUSTERING;
        wl_SOK_All_K32x1_ForceCLU.output_write_tiles = 1;

        {
            Logger::clear2ndlog();
            std::string err_info;

            {  // SOHO SPLITS
                std::cout << "\n ------- SOHO SPLITS: ------- ";

                case_run(wl_SOHO_Top_K64x2,
                         "\nTEST of: SOHO TOP K=64 X2\n");  //   v17: 5927 (lM:4386)    GTM 6044   v159nn:6179  GTL:4795
                case_run(wl_SOHO_Mid_K64x2, "\nTEST of: SOHO MID K=64 X2\n");  //   v17: 6060     GTM 6265   v159nn:6324
                case_run(wl_SOHO_Bot_K64x2, "\nTEST of :SOHO BOT K=64 X2\n");  //   v17: 5972     GTM 6298   v159nn:6266
                // Top mid bot are in parallel (X2 intra tile) =  (2x5927)||(2x6060)||(2x5972) = ()|(12120)|()  = 12120
                // GT parallelism (2x6044)||(2x6265)||( 2x6298) = ()|()|(12596 )  =12596

                case_run(wl_SOHO_Top_K32x4, "\nTEST of: SOHO TOP K=32 X4\n");  //   v17:     GTM    GTL:1412
                case_run(wl_SOHO_Mid_K32x4, "\nTEST of: SOHO MID K=32 X4\n");  //   v17:     GTM    GTL:1410
                case_run(wl_SOHO_Bot_K32x4, "\nTEST of :SOHO BOT K=32 X4\n");  //   v17:     GTM    GTL:1400
            }

            {  /// SOK splits
                std::cout << "\n ------- SOK SPLITS: ------- ";

                case_run(wl_SOK_All_K32x1,
                         "\nTEST of:SOK K=32 owt=4\n");  // v17:11721 (11026 owt2) (lM:5513)  GT ?10752 (owt=2)
                                                         // v159nn:9688 GTL:5150
                case_run(wl_SOK_All_K32x1_OWT2, "\nTEST of:SOK K=32 owt=2\n");  // v17:11026  GTM 10752    v159nn:9387
                case_run(wl_SOK_All_K32x1_ForceCLU, "\nTEST of:SOK K=32 CLU!\n");  // v17:10873  GTM 10620 v159nn:10641
                // SOK : 11721,   SOK owt lim to 2: 11026
                // GT: 10700
            }

            // SOK wins in all cases
            // GT: sok wins vs SOHO
        }
    }
    {
        // SOH H section.
        // HOW the base runtimes look for SOHH splits
        const HaloWorkload halo_top{{0, 1, 0, 0, 0, 0},  // H in TBLRFB
                                    {0, 0, 0, 0, 0, 0},
                                    {0, 0, 0, 0, 0, 0},
                                    {0, 0, 0, 0, 0, 0}};

        const HaloWorkload halo_mid{{0, 1, 0, 0, 0, 0},  // H in TBLRFB
                                    {0, 0, 0, 0, 0, 0},
                                    {0, 0, 0, 0, 0, 0},
                                    {0, 0, 0, 0, 0, 0}};
        const HaloWorkload halo_bot{{1, 0, 0, 0, 0, 0},  // H in TBLRFB
                                    {0, 0, 0, 0, 0, 0},
                                    {0, 0, 0, 0, 0, 0},
                                    {0, 0, 0, 0, 0, 0}};

        const DPUWorkload fake_SOHH_Top_K64x2{
                VPUDevice::VPU_4_0,
                Operation::DW_CONVOLUTION,
                {VPUTensor(56, 14, 64, 1, DataType::UINT8)},  // input dimensions
                {VPUTensor(56, 14, 64, 1, DataType::UINT8)},  // output dimensions
                {3, 3},                                       // kernels
                {1, 1},                                       // strides
                {1, 0, 1, 1},                                 // padding
                ExecutionMode::CUBOID_16x16,                  // execution mode
                ActivationFunction::NONE,                     // activation
                0.0F,                                         // act_sparsity
                0.0F,                                         // weight_sparsity
                {swz_def, swz_def},                           // input_swizzling
                {swz_def},                                    // output_swizzling
                1,                                            // output_write_tiles
                {0, 0, 0, 0},                                 // offsets
                ISIStrategy::SPLIT_OVER_H,                    // isi_strategy
                false,                                        // weight_sparsity_enabled
                                                              // halo_top,                                     // halo
                                                              //  sep
        };

        DPUWorkload fake_SOHH_Mid_K64x2{fake_SOHH_Top_K64x2};
        fake_SOHH_Mid_K64x2.padding = {0, 0, 1, 1};
        // fake_SOHH_Mid_K64x2.halo = halo_mid;

        DPUWorkload fake_SOHH_Bot_K64x2{fake_SOHH_Top_K64x2};
        fake_SOHH_Bot_K64x2.padding = {0, 1, 1, 1};
        // fake_SOHH_Bot_K64x2.halo = halo_bot;

        const DPUWorkload real_SOHH_Top_K64x2{
                VPUDevice::VPU_4_0,
                Operation::DW_CONVOLUTION,
                {VPUTensor(56, 15, 64, 1, DataType::UINT8)},  // input dimensions
                {VPUTensor(56, 14, 64, 1, DataType::UINT8)},  // output dimensions
                {3, 3},                                       // kernels
                {1, 1},                                       // strides
                {1, 0, 1, 1},                                 // padding
                ExecutionMode::CUBOID_16x16,                  // execution mode
                ActivationFunction::NONE,                     // activation
                0.0F,                                         // act_sparsity
                0.0F,                                         // weight_sparsity
                {swz_def, swz_def},                           // input_swizzling
                {swz_def},                                    // output_swizzling
                1,                                            // output_write_tiles
                {0, 0, 0, 0},                                 // offsets
                ISIStrategy::SPLIT_OVER_H,                    // isi_strategy
                false,                                        // weight_sparsity_enabled
                halo_top,                                     // halo
                                                              // sep
        };
        DPUWorkload real_SOHH_Mid_K64x2{real_SOHH_Top_K64x2};
        real_SOHH_Mid_K64x2.inputs[0] = VPUTensor(56, 16, 64, 1, DataType::UINT8);
        real_SOHH_Mid_K64x2.padding = {0, 0, 1, 1};
        real_SOHH_Mid_K64x2.halo = halo_mid;

        DPUWorkload real_SOHH_Bot_K64x2{real_SOHH_Top_K64x2};
        real_SOHH_Bot_K64x2.padding = {0, 1, 1, 1};
        real_SOHH_Bot_K64x2.halo = halo_bot;

        {  // SOH H variants

            std::cout << "\n ------------------------       SOH H         ----------------------------------";
            std::cout << "\n SOH H like SPLITS  from Layer  (same compute tensor , but isi SOH + halo)";

            case_run(real_SOHH_Top_K64x2,
                     "\nTEST of: SOHH TOP K=64 X2\n");  //   v17: 6767:7018     GTM 7398 v159nn:6914
            case_run(real_SOHH_Mid_K64x2,
                     "\nTEST of: SOHH MID K=64 X2\n");  //   v17: 7205:7483     GTM ??7398++(NA)   v159nn:7116
            case_run(real_SOHH_Bot_K64x2,
                     "\nTEST of :SOHH BOT K=64 X2\n");  //   v17: 6795:7045     GTM ?7398(NA) v159nn:6998
            // Top mid bot are in parallel (X2 intra tile) =  (2x )||(2x7205:7398 )||(2x ) = ()|(14410: 14800 )|()  =
            // 14410 GT parallelism (2x )||(2x7400 )||( 2x ) = ()|()|()  = 14800

            std::cout << "\n FAKE SOH H like SPLITS  from Layer  (memo tensor as compute , but isi SOH + halo)";
            // this tries also alternative , asking the NN using the memo tensor instead

            case_run(fake_SOHH_Top_K64x2, "\nTEST of:fake SOHH TOP K=64 X2\n");  //   v17: 7018     GTM 7398 v159nn:7651
            case_run(fake_SOHH_Mid_K64x2, "\nTEST of:fake SOHH MID K=64 X2\n");  //   v17: 7957     GTM x v159nn:9256
            case_run(fake_SOHH_Bot_K64x2, "\nTEST of:fake SOHH BOT K=64 X2\n");  //   v17: 7045     GTM x v159nn:7781
            // Top mid bot are in parallel (X2 intra tile) =  (2x )||(2x7957 )||(2x ) = ()|(15914 )|()  =  15914
            // GT parallelism (2x )||(2x )||( 2x ) = ()|()|(  )  =14800 no special GT
        }

        // SOH H is inefficient. having a halo row degrades a lot vs that row in overlapped memory
    }
}

TEST_F(VPULayerCM_InvestigationTestNPU4x, SOHO_ELTWISE_EISXW_127594_MIneeva10June_NPU40) {
    const bool force_fail{false};        // controls force failing assertion
    const int fail{force_fail ? 0 : 1};  // 1 neutral, 0 fail
    show_split = true;
    // EXPECT_TRUE(false);

    //  orig layer is 250x250x64
    const DPUWorkload wl_27{
            // orig Layer
            VPUDevice::VPU_4_0,
            Operation::ELTWISE,
            {VPUTensor(250, 27, 64, 1, DataType::UINT8)},  // input dimensions
            {VPUTensor(250, 27, 64, 1, DataType::UINT8)},  // output dimensions
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
            ISIStrategy::CLUSTERING,                       // isi_strategy
            false,                                         // weight_sparsity_enabled
    };

    const DPUWorkload wl_28{
            // orig Layer
            VPUDevice::VPU_4_0,
            Operation::ELTWISE,
            {VPUTensor(250, 28, 64, 1, DataType::UINT8)},  // input dimensions
            {VPUTensor(250, 28, 64, 1, DataType::UINT8)},  // output dimensions
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
            ISIStrategy::CLUSTERING,                       // isi_strategy
            false,                                         // weight_sparsity_enabled
    };

    const bool prefetch{true};

    {  // note: Real LNL NN has big delta vs GTL
        const DPULayer tst_layer_28(wl_28);
        const DPULayer tst_layer_27(wl_27);
        const std::vector<TestCase> tests{
                {{std::move(tst_layer_28),
                  {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 2655 - 260, fail * (3200 + 384)},  //  4T , 4x8: 3484 GTLNL:
                 "SOHO 28 , no memmove, "},
                {{std::move(tst_layer_27),
                  {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 2655 - 260, fail * (3498 + 349)},  //  4T , 3x7+1x6:3498  GTLNL:
                 "SOHO 27 , no memmove, "},

                // note:?? wins
        };
        executeTests(tests);
    }

    VPULayerCostModel& theModel = layer_models.getModel(VPUDevice::VPU_4_0);

    auto case_run = [=, &theModel](const DPUWorkload& workload, const std::string& whatTest) {
        DPUWorkload tst_layer{workload};
        std::string err_info;

        CyclesInterfaceType dpu_cost = theModel.get_cost_model().DPU(tst_layer, err_info);
        if (force_fail) {
            EXPECT_EQ(dpu_cost, 10) << whatTest << dpu_cost << "=" << Cycles::toErrorText(dpu_cost) << "\n"
                                    << tst_layer << err_info << "END ERR";
        }
        std::cout << "*" << whatTest << dpu_cost << " =" << Cycles::toErrorText(dpu_cost) << "\n";
    };

    //     EXPECT_TRUE(on);  // something has to fail to  see couts

    // low level WL
    {
        const DPUWorkload wl_H7_16x16{
                VPUDevice::VPU_4_0,
                Operation::ELTWISE,
                {VPUTensor(250, 7, 64, 1, DataType::UINT8)},  // input dimensions
                {VPUTensor(250, 7, 64, 1, DataType::UINT8)},  // output dimensions
                {1, 1},                                       // kernels
                {1, 1},                                       // strides
                {0, 0, 0, 0},                                 // padding
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
        DPUWorkload wl_H6_16x16{wl_H7_16x16};
        wl_H6_16x16.inputs[0] = VPUTensor(250, 6, 64, 1, DataType::UINT8);
        wl_H6_16x16.outputs[0] = VPUTensor(250, 6, 64, 1, DataType::UINT8);

        DPUWorkload wl_H7_8x16{wl_H7_16x16};
        wl_H7_8x16.execution_order = ExecutionMode::CUBOID_8x16;
        DPUWorkload wl_H6_8x16{wl_H6_16x16};
        wl_H6_8x16.execution_order = ExecutionMode::CUBOID_8x16;

        DPUWorkload wl_H7_4x16{wl_H7_16x16};
        wl_H7_4x16.execution_order = ExecutionMode::CUBOID_4x16;
        DPUWorkload wl_H6_4x16{wl_H6_16x16};
        wl_H6_4x16.execution_order = ExecutionMode::CUBOID_4x16;

        // CUBOID_8x16

        {
            Logger::clear2ndlog();
            std::string err_info;

            {  // SOHO SPLITS
                std::cout << "\n ------- SOHO SPLITS: ------- \n";

                case_run(wl_H7_16x16, "TEST of: SOHO H = 7  16x16: ");  //   v17: 3402  GTL:?
                case_run(wl_H6_16x16, "TEST of: SOHO H = 6  16x16: ");  //   v17: 3395  GTL:?

                std::cout << std::endl;
                case_run(wl_H7_8x16, "TEST of: SOHO H = 7  8x16: ");  //   v17:3484  GTL:?
                case_run(wl_H6_8x16, "TEST of: SOHO H = 6  8x16: ");  //   v17:3498  GTL:?

                std::cout << std::endl;
                case_run(wl_H7_4x16, "TEST of: SOHO H = 7  4x16: ");  //   v17:4045  GTL:?
                case_run(wl_H6_4x16, "TEST of: SOHO H = 6  4x16: ");  //   v17:4075  GTL:?

                // std::cout << wl_H6_4x16;
            }
        }
    }
}

TEST_F(VPULayerCM_InvestigationTestNPU4x, Model_N_v1_CONV_EISXW_127644_NPU40) {
    const bool force_fail{false};  // controls force failing assertion
    const bool force_fail_case_run{false};
    const int fail{force_fail ? 0 : 1};  // 1 neutral, 0 fail
    show_split = false;
    // EXPECT_TRUE(false);

    const DPUWorkload wl_{
            // orig Layer
            VPUDevice::VPU_4_0,
            Operation::CONVOLUTION,
            {VPUTensor(16, 16, 160, 1, DataType::UINT8)},  // input dimensions
            {VPUTensor(16, 16, 160, 1, DataType::UINT8)},  // output dimensions
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
    // SOHO , 4 tiles: 9877 cyc
    // SOK , 4 tiles: 9742

    const DPUWorkload wl_halfK{
            // orig Layer
            VPUDevice::VPU_4_0,
            Operation::CONVOLUTION,
            {VPUTensor(16, 16, 160, 1, DataType::UINT8)},  // input dimensions
            {VPUTensor(16, 16, 80, 1, DataType::UINT8)},   // output dimensions
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

    {  //
        std::cout << "\n ------- CONV_34 full K =160 ------- \n";
        const DPULayer tst_layer(wl_);
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 7000,
                  fail * 7001 +
                          2000},  // V17: 7509   v159NN:8606/8572 ? GTVPUX:8400ns (2x 4200ns)   GTL: 7385   split: x1?
                 "SOHO full K"},

                {{tst_layer, {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOK, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 8000,
                  fail * 8001 +
                          2100},  // V17: 8456   v159NN: 9813 GTVPUX:9736ns {short:6500ns}   GTL: 9407 split:K48x3+K16x1
                 "SOK "},

        };
        executeTests(tests);
    }
    {  //
        std::cout << "\n ------- CONV_34 half  K =80 outputs ------- \n";
        const DPULayer tst_layer(wl_halfK);
        const std::vector<TestCase> tests{
                {{std::move(tst_layer), {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 3500,
                  fail * 3501 + 1000},  // V17: 3889 *2 = 7778    v159NN: 4234 *2=8468 GTVPUX:8400ns (2x 4200ns)
                                        // GTL:3811 *2 = 7622
                 "SOHO half K"},

        };
        executeTests(tests);
    }
    // note:v17: SOHO wins,  (not split in half)
    // v159 SOHO wins also in the split way K80x2

    VPULayerCostModel& theModel = layer_models.getModel(VPUDevice::VPU_4_0);

    auto case_run = [=, &theModel](const DPUWorkload& workload, const std::string& whatTest) {
        DPUWorkload tst_layer(workload);
        std::string err_info;

        CyclesInterfaceType dpu_cost = theModel.get_cost_model().DPU(tst_layer, err_info);
        if (force_fail_case_run) {
            EXPECT_EQ(dpu_cost, 10) << whatTest << dpu_cost << "=" << Cycles::toErrorText(dpu_cost) << "\n"
                                    << tst_layer << err_info << "END ERR";
        }
        std::cout << whatTest << ": " << dpu_cost << "=" << Cycles::toErrorText(dpu_cost) << "\n";
    };

    //     EXPECT_TRUE(on);  // something has to fail to  see couts

    auto mod_execution = [](const DPUWorkload& wl, ExecutionMode em) -> DPUWorkload {
        DPUWorkload wl_{wl};
        wl_.execution_order = em;
        return wl_;
    };

    // low level WL
    {
        const DPUWorkload wl_SOHO_K160_Top_x1{
                VPUDevice::VPU_4_0,
                Operation::CONVOLUTION,
                {VPUTensor(16, 5, 160, 1, DataType::UINT8)},  // input dimensions
                {VPUTensor(16, 4, 160, 1, DataType::UINT8)},  // output dimensions
                {3, 3},                                       // kernels
                {1, 1},                                       // strides
                {1, 0, 1, 1},                                 // padding
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
        DPUWorkload wl_SOHO_K160_Mid_x1{wl_SOHO_K160_Top_x1};
        wl_SOHO_K160_Mid_x1.inputs[0] = VPUTensor(16, 6, 160, 1, DataType::UINT8);
        wl_SOHO_K160_Mid_x1.padding = {0, 0, 1, 1};

        DPUWorkload wl_SOHO_K160_Bot_x1{wl_SOHO_K160_Top_x1};
        wl_SOHO_K160_Bot_x1.padding = {0, 1, 1, 1};

        // em 8x16
        const DPUWorkload wl_SOHO_K160_Top_em8x16{mod_execution(wl_SOHO_K160_Top_x1, ExecutionMode::CUBOID_8x16)};
        const DPUWorkload wl_SOHO_K160_Mid_em8x16{mod_execution(wl_SOHO_K160_Mid_x1, ExecutionMode::CUBOID_8x16)};
        const DPUWorkload wl_SOHO_K160_Bot_em8x16{mod_execution(wl_SOHO_K160_Bot_x1, ExecutionMode::CUBOID_8x16)};

        // em 4x16
        const DPUWorkload wl_SOHO_K160_Top_em4x16{mod_execution(wl_SOHO_K160_Top_x1, ExecutionMode::CUBOID_4x16)};
        const DPUWorkload wl_SOHO_K160_Mid_em4x16{mod_execution(wl_SOHO_K160_Mid_x1, ExecutionMode::CUBOID_4x16)};
        const DPUWorkload wl_SOHO_K160_Bot_em4x16{mod_execution(wl_SOHO_K160_Bot_x1, ExecutionMode::CUBOID_4x16)};

        const DPUWorkload wl_SOHO_K80_Top_x2{
                VPUDevice::VPU_4_0,
                Operation::CONVOLUTION,
                {VPUTensor(16, 5, 160, 1, DataType::UINT8)},  // input dimensions
                {VPUTensor(16, 4, 80, 1, DataType::UINT8)},   // output dimensions
                {3, 3},                                       // kernels
                {1, 1},                                       // strides
                {1, 0, 1, 1},                                 // padding
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
        DPUWorkload wl_SOHO_K80_Mid_x2{wl_SOHO_K80_Top_x2};
        wl_SOHO_K80_Mid_x2.inputs[0] = VPUTensor(16, 6, 160, 1, DataType::UINT8);
        wl_SOHO_K80_Mid_x2.padding = {0, 0, 1, 1};

        DPUWorkload wl_SOHO_K80_Bot_x2{wl_SOHO_K80_Top_x2};
        wl_SOHO_K80_Bot_x2.padding = {0, 1, 1, 1};

        // em 8x16
        const DPUWorkload wl_SOHO_K80_Top_em8x16{mod_execution(wl_SOHO_K80_Top_x2, ExecutionMode::CUBOID_8x16)};
        const DPUWorkload wl_SOHO_K80_Mid_em8x16{mod_execution(wl_SOHO_K80_Mid_x2, ExecutionMode::CUBOID_8x16)};
        const DPUWorkload wl_SOHO_K80_Bot_em8x16{mod_execution(wl_SOHO_K80_Bot_x2, ExecutionMode::CUBOID_8x16)};

        // em 4x16
        const DPUWorkload wl_SOHO_K80_Top_em4x16{mod_execution(wl_SOHO_K80_Top_x2, ExecutionMode::CUBOID_4x16)};
        const DPUWorkload wl_SOHO_K80_Mid_em4x16{mod_execution(wl_SOHO_K80_Mid_x2, ExecutionMode::CUBOID_4x16)};
        const DPUWorkload wl_SOHO_K80_Bot_em4x16{mod_execution(wl_SOHO_K80_Bot_x2, ExecutionMode::CUBOID_4x16)};

        // SOK part
        const DPUWorkload wl_SOK_K48{
                VPUDevice::VPU_4_0,
                Operation::CONVOLUTION,
                {VPUTensor(16, 16, 160, 1, DataType::UINT8)},  // input dimensions
                {VPUTensor(16, 16, 48, 1, DataType::UINT8)},   // output dimensions
                {3, 3},                                        // kernels
                {1, 1},                                        // strides
                {1, 1, 1, 1},                                  // padding
                ExecutionMode::CUBOID_16x16,                   // execution mode
                ActivationFunction::NONE,                      // activation
                0.0F,                                          // act_sparsity
                0.0F,                                          // weight_sparsity
                {swz_def, swz_def},                            // input_swizzling
                {swz_def},                                     // output_swizzling
                4,                                             // output_write_tiles
                {0, 0, 0, 0},                                  // offsets
                ISIStrategy::SPLIT_OVER_K,                     // isi_strategy
                false,                                         // weight_sparsity_enabled
        };
        DPUWorkload wl_SOK_K32{wl_SOK_K48};
        wl_SOK_K32.outputs[0] = VPUTensor(16, 16, 32, 1, DataType::UINT8);
        DPUWorkload wl_SOK_K16{wl_SOK_K48};
        wl_SOK_K16.outputs[0] = VPUTensor(16, 16, 16, 1, DataType::UINT8);
        DPUWorkload wl_SOK_K64{wl_SOK_K48};
        wl_SOK_K64.outputs[0] = VPUTensor(16, 16, 64, 1, DataType::UINT8);

        // em 8x16
        const DPUWorkload wl_SOK_K48_em8x16{mod_execution(wl_SOK_K48, ExecutionMode::CUBOID_8x16)};
        const DPUWorkload wl_SOK_K32_em8x16{mod_execution(wl_SOK_K32, ExecutionMode::CUBOID_8x16)};
        const DPUWorkload wl_SOK_K16_em8x16{mod_execution(wl_SOK_K16, ExecutionMode::CUBOID_8x16)};
        const DPUWorkload wl_SOK_K64_em8x16{mod_execution(wl_SOK_K64, ExecutionMode::CUBOID_8x16)};

        // em 4x16
        const DPUWorkload wl_SOK_K48_em4x16{mod_execution(wl_SOK_K48, ExecutionMode::CUBOID_4x16)};
        const DPUWorkload wl_SOK_K32_em4x16{mod_execution(wl_SOK_K32, ExecutionMode::CUBOID_4x16)};
        const DPUWorkload wl_SOK_K16_em4x16{mod_execution(wl_SOK_K16, ExecutionMode::CUBOID_4x16)};
        const DPUWorkload wl_SOK_K64_em4x16{mod_execution(wl_SOK_K64, ExecutionMode::CUBOID_4x16)};

        {
            Logger::clear2ndlog();
            std::string err_info;

            {  // SOHO SPLITS
                std::cout << "\n ------- SOHO SPLITS: ------- \n";
                // clang-format off
                case_run(wl_SOHO_K160_Top_x1, "SOHO T_K160X1");  // V17:8141 x   v159NN: ? GTVPUX:   GTL: ??
                case_run(wl_SOHO_K160_Mid_x1, "SOHO M_K160X1");  // V17:8346 x   v159NN: ? GTVPUX:   GTL: ??
                case_run(wl_SOHO_K160_Bot_x1, "SOHO B_K160X1");  // V17:8259 x   v159NN: ? GTVPUX:   GTL: ??
                std::cout << "\n";
                case_run(wl_SOHO_K160_Top_em8x16, "SOHO T_K160_em8x16");  // V17:7433   v159NN: 8562 GTVPUX:   GTL: 7382
                case_run(wl_SOHO_K160_Mid_em8x16, "SOHO M_K160_em8x16");  // V17:7509 ! v159NN: 8606 GTVPUX:   GTL: 7384,80
                case_run(wl_SOHO_K160_Bot_em8x16, "SOHO B_K160_em8x16");  // V17:7483   v159NN: 8572 GTVPUX:   GTL: 7385
                std::cout << "\n";
                case_run(wl_SOHO_K160_Top_em4x16, "SOHO T_K160_em4x16");  // V17:8301 x   v159NN: ? GTVPUX:   GTL: ??
                case_run(wl_SOHO_K160_Mid_em4x16, "SOHO M_K160_em4x16");  // V17:8377 x   v159NN: ? GTVPUX:   GTL: ??
                case_run(wl_SOHO_K160_Bot_em4x16, "SOHO B_K160_em4x16");  // V17:8333 x   v159NN: ? GTVPUX:   GTL: ??

                std::cout << "\nK80 output\n";

                case_run(wl_SOHO_K80_Top_x2, "SOHO T_K80 X2");  // V17:4655 x   v159NN: ? GTVPUX:   GTL: ??
                case_run(wl_SOHO_K80_Mid_x2, "SOHO M_K80 X2");  // V17:4766 x   v159NN: ? GTVPUX:   GTL: ??
                case_run(wl_SOHO_K80_Bot_x2, "SOHO B_K80 X2");  // V17:4718 x   v159NN: ? GTVPUX:   GTL: ??
                std::cout << "\n";
                case_run(wl_SOHO_K80_Top_em8x16, "SOHO T_K80_em8x16");  // V17:3832 x   v159NN: 4211 GTVPUX:   GTL: 3811
                case_run(wl_SOHO_K80_Mid_em8x16, "SOHO M_K80_em8x16");  // V17:3889 W   v159NN: 4234 GTVPUX:   GTL: 3810,
                case_run(wl_SOHO_K80_Bot_em8x16, "SOHO B_K80_em8x16");  // V17:3841 x   v159NN: 4222 GTVPUX:   GTL: 3810
                std::cout << "\n";
                case_run(wl_SOHO_K80_Top_em4x16, "SOHO T_K80_em4x16");  // V17:4797 x   v159NN: ? GTVPUX:   GTL: ??
                case_run(wl_SOHO_K80_Mid_em4x16, "SOHO M_K80_em4x16");  // V17:4898 x   v159NN: ? GTVPUX:   GTL: ??
                case_run(wl_SOHO_K80_Bot_em4x16, "SOHO B_K80_em4x16");  // V17:4858 x   v159NN: ? GTVPUX:   GTL: ??
                std::cout << "\n";


                std::cout << "\n ------- SOK SPLITS: ------- \n";
                case_run(wl_SOK_K48, "SOK  K=48 em 16x16");  // V17:8456 Wx3 v159NN: 10158 GTVPUX:   GTL: 9406,9350,9407
                case_run(wl_SOK_K32, "SOK  K=32 em 16x16");  // V17:5619     v159NN: 6777  GTVPUX:   GTL: ??
                case_run(wl_SOK_K16, "SOK  K=16 em 16x16");  // V17:4399 W   v159NN: 4931  GTVPUX:   GTL: 3435
                case_run(wl_SOK_K64, "SOK  K=64 em 16x16");  // V17:11194    v159NN: 13492 GTVPUX:   GTL: 12291
                std::cout << "\n";

                case_run(wl_SOK_K48_em8x16, "SOK  K=48 em 8x16");  // V17:9562 x   v159NN: ? GTVPUX:   GTL: ??
                case_run(wl_SOK_K32_em8x16, "SOK  K=32 em 8x16");  // V17:7834 x   v159NN: ? GTVPUX:   GTL: ??
                case_run(wl_SOK_K16_em8x16, "SOK  K=16 em 8x16");  // V17:7580 x   v159NN: ? GTVPUX:   GTL: ??
                case_run(wl_SOK_K64_em8x16, "SOK  K=64 em 8x16");  // V17:12591    v159NN: ? GTVPUX:   GTL: ??
                std::cout << "\n";

                case_run(wl_SOK_K48_em4x16, "SOK  K=48 em 4x16");  // V17:11117   v159NN: ? GTVPUX:   GTL: ??
                case_run(wl_SOK_K32_em4x16, "SOK  K=32 em 4x16");  // V17:9761    v159NN: ? GTVPUX:   GTL: ??
                case_run(wl_SOK_K16_em4x16, "SOK  K=16 em 4x16");  // V17:8488    v159NN: ? GTVPUX:   GTL: ??
                case_run(wl_SOK_K64_em4x16, "SOK  K=64 em 4x16");  // V17:12879   v159NN: ? GTVPUX:   GTL: ??

                // clang-format on
            }
        }
    }
}

/// tests covering conv4 and conv8 from the model
///
/// Profiling results with details are  to be found on one note
TEST_F(VPULayerCM_InvestigationTestNPU4x, Model_E_v9_CONV_EISXW_127649_NPU40) {
    const bool force_fail{false};  // controls force failing assertion
    const bool force_fail_case_run{false};
    const int fail{force_fail ? 0 : 1};  // 1 neutral, 0 fail
                                         // show_split = true;
    //  EXPECT_TRUE(false);

    const DPUWorkload conv4{
            VPUDevice::VPU_4_0,
            Operation::CONVOLUTION,
            {VPUTensor(56, 32, 64, 1, DataType::UINT8)},  // input dimensions
            {VPUTensor(56, 32, 64, 1, DataType::UINT8)},  // output dimensions
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
    DPUWorkload conv4s{conv4};
    conv4s.weight_sparsity = 0.128038f;  // very small sparsity , why?
    conv4s.weight_sparsity_enabled = true;

    const DPUWorkload conv8{
            VPUDevice::VPU_4_0,
            Operation::CONVOLUTION,
            {VPUTensor(28, 16, 96, 1, DataType::UINT8)},  // input dimensions
            {VPUTensor(28, 16, 96, 1, DataType::UINT8)},  // output dimensions
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

    const bool prefetch{true};
    std::cout << "\n ------- CONV4 not sparse ------- \n";
    {  //
        const DPULayer tst_layer(conv4);
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 6U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 8297 - 1500,
                  fail * (8297 +
                          829)},  // V17:8493 (HK 8588)     v159NN:9153 (HK 9118)  ? GTVPUX:4527ns (HK 4579us) GTL:8297
                 "CONV4 nosparse SOHO"},

                {{tst_layer, {1U, 1U, 6U, VPUNN::VPUTilingStrategy::SOK, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 9100 /*old 13000*/,
                  fail * 13001 + 2000},  // V17: 13710   v159NN:14617  GTVPUX:   GTL:   split: ONLY 4 not 6
                 "CONV4 nosparse SOK (not really a relevant case)"},

        };
        executeTests(tests);
    }
    std::cout << "\n ------- CONV4  sparse ------- \n";
    {  //
        const DPULayer tst_layer(conv4s);
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 6U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 6500,
                  fail * 7001 + 2000},  // V17:8162 (HK 8261)    v159NN:8999 (HK 8845)  ?VPU:8356 GTVPUX:4527ns GTL:8332
                 "CONV4s SOHO"},

                {{tst_layer, {1U, 1U, 6U, VPUNN::VPUTilingStrategy::SOK, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 9001 /*old 13000*/,
                  fail * 13001 + 2000},  // V17: 13506   v159NN:14106  GTVPUX:   GTL: NA  split:  ONLY 4 not 6
                 "CONV4s SOK (not really a relevant case) "},

        };
        executeTests(tests);
    }
    // CONV 4
    //  note:v17: SOHO always wins, HK(slower) also vs SOK
    // v159  wins

    std::cout << "\n ------- CONV8 ------- \n";
    {  //
        const DPULayer tst_layer(conv8);
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 6U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 4748 - 500,
                  fail * (4748 + 500)},  // V17: 5058 (HK 5144)   v159NN:5538 (HK 5564) ? VPU:5358    GTVPUX: 2620ns
                                         // GTL: 4748  split in 6 (5*3+1)
                 "CONV8 SOHO"},

                {{tst_layer, {1U, 1U, 6U, VPUNN::VPUTilingStrategy::SOK, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 3814 - 381,
                  fail * (3814 + 381 * 2 + 50)},  // V17: 4593   v159NN:5700  VPU:5222  GTVPUX: 2700ns  GTL:3814(16x16),
                                                  // or 37xx   split:in 6, !! WHAT is the MTL GT?!!
                 "CONV8 SOK "},

                {{tst_layer, {1U, 1U, 6U, VPUNN::VPUTilingStrategy::SOHO_K_SWITCH, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 4838 - 600,
                  fail * (4838 + 600)},  // V17: (HK 5144)   v159NN: (HK 5564) ? VPU:    GTVPUX: 2620ns
                                         // GTL: 4838  split in 6 (5*3+1)
                 "CONV8 SOHOBroadcast.HK"},

        };
        executeTests(tests);
    }
    // CONV 8
    // note:v17: SOK wins by small margin,
    // v159 SOH wins by smaller margin wins

    VPULayerCostModel& theModel = layer_models.getModel(VPUDevice::VPU_4_0);

    auto case_run = [=, &theModel](const DPUWorkload& workload, const std::string& whatTest) {
        DPUWorkload tst_layer(workload);
        std::string err_info;

        CyclesInterfaceType dpu_cost = theModel.get_cost_model().DPU(tst_layer, err_info);
        if (force_fail_case_run) {
            EXPECT_EQ(dpu_cost, 10) << whatTest << dpu_cost << "=" << Cycles::toErrorText(dpu_cost) << "\n"
                                    << tst_layer << err_info << "END ERR";
        }
        std::cout << whatTest << ": " << dpu_cost << "=" << Cycles::toErrorText(dpu_cost) << "\n";
    };

    //     EXPECT_TRUE(on);  // something has to fail to  see couts

    auto mod_execution = [](const DPUWorkload& wl, ExecutionMode em) -> DPUWorkload {
        DPUWorkload wl_{wl};
        wl_.execution_order = em;
        return wl_;
    };

    // low level WL
    {
        // CONV4 soho
        DPUWorkload conv4_SOHO_T{std::move(conv4s)};
        conv4_SOHO_T.inputs[0] = VPUTensor(56, 7, 64, 1, DataType::UINT8);
        conv4_SOHO_T.outputs[0] = VPUTensor(56, 6, 64, 1, DataType::UINT8);
        conv4_SOHO_T.padding = {1, 0, 1, 1};

        DPUWorkload conv4_SOHO_M{conv4_SOHO_T};
        conv4_SOHO_M.inputs[0] = VPUTensor(56, 8, 64, 1, DataType::UINT8);
        conv4_SOHO_M.padding = {0, 0, 1, 1};

        DPUWorkload conv4_SOHO_B{conv4_SOHO_T};
        conv4_SOHO_B.inputs[0] = VPUTensor(56, 3, 64, 1, DataType::UINT8);
        conv4_SOHO_B.outputs[0] = VPUTensor(56, 2, 64, 1, DataType::UINT8);
        conv4_SOHO_B.padding = {0, 1, 1, 1};

        // em 8x16
        const DPUWorkload conv4_SOHO_T_em8x16{mod_execution(conv4_SOHO_T, ExecutionMode::CUBOID_8x16)};
        const DPUWorkload conv4_SOHO_M_em8x16{mod_execution(conv4_SOHO_M, ExecutionMode::CUBOID_8x16)};
        const DPUWorkload conv4_SOHO_B_em8x16{mod_execution(conv4_SOHO_B, ExecutionMode::CUBOID_8x16)};

        // em 4x16
        const DPUWorkload conv4_SOHO_T_em4x16{mod_execution(conv4_SOHO_T, ExecutionMode::CUBOID_4x16)};
        const DPUWorkload conv4_SOHO_M_em4x16{mod_execution(conv4_SOHO_M, ExecutionMode::CUBOID_4x16)};
        const DPUWorkload conv4_SOHO_B_em4x16{mod_execution(conv4_SOHO_B, ExecutionMode::CUBOID_4x16)};

        // CONV8  SOHO
        DPUWorkload conv8_SOHO_T{conv8};
        conv8_SOHO_T.inputs[0] = VPUTensor(28, 4, 96, 1, DataType::UINT8);
        conv8_SOHO_T.outputs[0] = VPUTensor(28, 3, 96, 1, DataType::UINT8);
        conv8_SOHO_T.padding = {1, 0, 1, 1};

        DPUWorkload conv8_SOHO_M{conv8_SOHO_T};
        conv8_SOHO_M.inputs[0] = VPUTensor(28, 5, 96, 1, DataType::UINT8);
        conv8_SOHO_M.padding = {0, 0, 1, 1};

        DPUWorkload conv8_SOHO_B{conv8_SOHO_T};
        conv8_SOHO_B.inputs[0] = VPUTensor(28, 2, 96, 1, DataType::UINT8);
        conv8_SOHO_B.outputs[0] = VPUTensor(28, 1, 96, 1, DataType::UINT8);
        conv8_SOHO_B.padding = {0, 1, 1, 1};

        // em 8x16
        const DPUWorkload conv8_SOHO_T_em8x16{mod_execution(conv8_SOHO_T, ExecutionMode::CUBOID_8x16)};
        const DPUWorkload conv8_SOHO_M_em8x16{mod_execution(conv8_SOHO_M, ExecutionMode::CUBOID_8x16)};
        const DPUWorkload conv8_SOHO_B_em8x16{mod_execution(conv8_SOHO_B, ExecutionMode::CUBOID_8x16)};

        // em 4x16
        const DPUWorkload conv8_SOHO_T_em4x16{mod_execution(conv8_SOHO_T, ExecutionMode::CUBOID_4x16)};
        const DPUWorkload conv8_SOHO_M_em4x16{mod_execution(conv8_SOHO_M, ExecutionMode::CUBOID_4x16)};
        const DPUWorkload conv8_SOHO_B_em4x16{mod_execution(conv8_SOHO_B, ExecutionMode::CUBOID_4x16)};

        // SOK part  only for conv8
        DPUWorkload conv8_SOK_K16{std::move(conv8)};
        conv8_SOK_K16.outputs[0] = VPUTensor(28, 16, 16, 1, DataType::UINT8);
        conv8_SOK_K16.output_write_tiles = 6;

        // em 8x16
        const DPUWorkload conv8_SOK_K16_em8x16{mod_execution(conv8_SOK_K16, ExecutionMode::CUBOID_8x16)};
        // em 4x16
        const DPUWorkload conv8_SOK_K16_em4x16{mod_execution(conv8_SOK_K16, ExecutionMode::CUBOID_4x16)};

        {
            Logger::clear2ndlog();
            std::string err_info;

            {  // SOHO SPLITS
                std::cout << "\n ------- SOHO SPLITS CONV4(s?): ------- \n";
                // clang-format off
                case_run(conv4_SOHO_T, "conv4_SOHO_T");  // V17:8439   v159NN: ? GTVPUX:   GTL: ??
                case_run(conv4_SOHO_M, "conv4_SOHO_M");  // V17:8652   v159NN: ? GTVPUX:   GTL: ??
                case_run(conv4_SOHO_B, "conv4_SOHO_B");  // V17:5166   v159NN: ? GTVPUX:   GTL: ??
                std::cout << "\n";
                case_run(conv4_SOHO_T_em8x16, "Sconv4_SOHO_T_em8x16");  // V17: 7958  v159NN:  GTVPUX:   GTL: 8329
                case_run(conv4_SOHO_M_em8x16, "Sconv4_SOHO_M_em8x16");  // V17: 8162! v159NN:  GTVPUX:   GTL: 8330,
                case_run(conv4_SOHO_B_em8x16, "Sconv4_SOHO_B_em8x16");  // V17:4178   v159NN:  GTVPUX:   GTL: 4335
                std::cout << "\n";
                case_run(conv4_SOHO_T_em4x16, "conv4_SOHO_T_em4x16");  // V17: 8753   v159NN: ? GTVPUX:   GTL: ??
                case_run(conv4_SOHO_M_em4x16, "conv4_SOHO_M_em4x16");  // V17: 9139   v159NN: ? GTVPUX:   GTL: ??
                case_run(conv4_SOHO_B_em4x16, "conv4_SOHO_B_em4x16");  // V17: 6218   v159NN: ? GTVPUX:   GTL: ??

                std::cout << "\n ------- SOHO SPLITS conv8: ------- \n";
                case_run(conv8_SOHO_T, "conv8_SOHO_T");  // V17: x   v159NN: ? GTVPUX:   GTL: ??
                case_run(conv8_SOHO_M, "conv8_SOHO_M");  // V17: x   v159NN: ? GTVPUX:   GTL: ??
                case_run(conv8_SOHO_B, "conv8_SOHO_B");  // V17: x   v159NN: ? GTVPUX:   GTL: ??
                std::cout << "\n";
                case_run(conv8_SOHO_T_em8x16, "Sconv8_SOHO_T_em8x16");  // V17:   v159NN:  GTVPUX:   GTL: 
                case_run(conv8_SOHO_M_em8x16, "Sconv8_SOHO_M_em8x16");  // V17: ! v159NN:  GTVPUX:   GTL: ,
                case_run(conv8_SOHO_B_em8x16, "Sconv8_SOHO_B_em8x16");  // V17:   v159NN:  GTVPUX:   GTL: 
                std::cout << "\n";
                case_run(conv8_SOHO_T_em4x16, "conv8_SOHO_T_em4x16");  // V17: x   v159NN: ? GTVPUX:   GTL: ??
                case_run(conv8_SOHO_M_em4x16, "conv8_SOHO_M_em4x16");  // V17: x   v159NN: ? GTVPUX:   GTL: ??
                case_run(conv8_SOHO_B_em4x16, "conv8_SOHO_B_em4x16");  // V17: x   v159NN: ? GTVPUX:   GTL: ??



                std::cout << "\n ------- SOK SPLITS conv8: ------- \n";
                case_run(conv8_SOK_K16, "SOK conv8  K=16 em 16x16");  // V17: Wx3 v159NN:  GTVPUX:   GTL: 
                case_run(conv8_SOK_K16_em8x16, "SOK conv8  K=16 em 8x16");  // V17:x   v159NN: ? GTVPUX:   GTL: ??
                case_run(conv8_SOK_K16_em4x16, "SOK conv8 K=16 em 4x16");  // V17:   v159NN: ? GTVPUX:   GTL: ??

                // clang-format on
            }
        }
    }
}

TEST_F(VPULayerCM_InvestigationTestNPU4x, WhisperFP16_BIG_CONV_EISXW_131119_NPU40) {
    const bool force_fail{false};        // controls force failing assertion
    const int fail{force_fail ? 0 : 1};  // 1 neutral, 0 fail
    show_split = true;
    // EXPECT_TRUE(false);

    const DPUWorkload wl_{
            // orig Layer
            VPUDevice::VPU_4_0,
            Operation::CONVOLUTION,
            {VPUTensor(3000, 1, 80, 1, DataType::FLOAT16)},   // input dimensions
            {VPUTensor(3000, 1, 512, 1, DataType::FLOAT16)},  // output dimensions
            {3, 1},                                           // kernels
            {1, 1},                                           // strides
            {0, 0, 1, 1},                                     // padding
            ExecutionMode::CUBOID_16x16,                      // execution mode
            ActivationFunction::NONE,                         // activation
            0.0F,                                             // act_sparsity
            0.0F,                                             // weight_sparsity
            {swz_def, swz_def},                               // input_swizzling
            {swz_def},                                        // output_swizzling
            1,                                                // output_write_tiles
            {0, 0, 0, 0},                                     // offsets
            ISIStrategy::CLUSTERING,                          // isi_strategy
            false,                                            // weight_sparsity_enabled
    };

    const bool prefetch{true};

    {  // note: Real LNL NN has big delta vs GTL
        const DPULayer tst_layer(wl_);
        const std::vector<TestCase> tests{
                {{std::move(tst_layer),
                  {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOK_NO_BROADCAST, false, false, prefetch}},
                 {Cycles::NO_ERROR, true, 335000, fail * 370000 + 30000},  // 2// V17:384288   v159NN: ?   GTL: ?? todo
                 "SOK no broadcast , no memmove, "},

                // note:?? wins
        };
        executeTests(tests);
    }

    VPULayerCostModel& theModel = layer_models.getModel(VPUDevice::VPU_4_0);

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
        const DPUWorkload wl_SOK_TOP_16x{
                VPUDevice::VPU_4_0,
                Operation::CONVOLUTION,
                {VPUTensor(3000, 1, 80, 1, DataType::FLOAT16)},       // input dimensions
                {VPUTensor(3000, 1, 512 / 4, 1, DataType::FLOAT16)},  // output dimensions
                {3, 1},                                               // kernels
                {1, 1},                                               // strides
                {0, 0, 1, 1},                                         // padding
                ExecutionMode::CUBOID_16x16,                          // execution mode
                ActivationFunction::NONE,                             // activation
                0.0F,                                                 // act_sparsity
                0.0F,                                                 // weight_sparsity
                {swz_def, swz_def},                                   // input_swizzling
                {swz_def},                                            // output_swizzling
                4,                                                    // output_write_tiles
                {0, 0, 0, 0},                                         // offsets
                ISIStrategy::SPLIT_OVER_K,                            // isi_strategy
                false,                                                // weight_sparsity_enabled
        };
        DPUWorkload wl_SOK_TOP_CLU_16x{wl_SOK_TOP_16x};
        wl_SOK_TOP_CLU_16x.isi_strategy = ISIStrategy::CLUSTERING;
        wl_SOK_TOP_CLU_16x.output_write_tiles = 1;

        DPUWorkload wl_SOK_TOP_halo_16x{wl_SOK_TOP_16x};
        wl_SOK_TOP_halo_16x.halo.output_0_inbound_halo.back = 512 - 128;  // inbound halo= rest of channels

        // same with other execution mode
        DPUWorkload wl_SOK_TOP_8x{wl_SOK_TOP_16x};
        wl_SOK_TOP_8x.execution_order = ExecutionMode::CUBOID_8x16;
        DPUWorkload wl_SOK_TOP_CLU_8x{wl_SOK_TOP_CLU_16x};
        wl_SOK_TOP_CLU_8x.execution_order = ExecutionMode::CUBOID_8x16;
        DPUWorkload wl_SOK_TOP_halo_8x{wl_SOK_TOP_halo_16x};
        wl_SOK_TOP_halo_8x.execution_order = ExecutionMode::CUBOID_8x16;

        DPUWorkload wl_SOK_TOP_CLU_4x{wl_SOK_TOP_CLU_16x};
        wl_SOK_TOP_CLU_4x.execution_order = ExecutionMode::CUBOID_4x16;

        {
            Logger::clear2ndlog();
            std::string err_info;

            {  // SOK  SPLITS
                std::cout << "\n ------- SOK SPLITS: ------- \n";

                case_run(wl_SOK_TOP_CLU_16x,
                         "TEST of: _16x SOK CLU no broadcast TOP K=128 ");  // V17: x   v159NN: ? GTVPUX:   GTL: ??
                case_run(wl_SOK_TOP_16x, "TEST of: _16x SOK broadcast TOP K=128 ");
                case_run(wl_SOK_TOP_halo_16x, "TEST of: _16x SOK TOP broadcast HALO OK,  K=128 ");

                //  Memory is too big

                case_run(wl_SOK_TOP_CLU_8x,
                         "\nTEST of: _8x SOK CLU no broadcast TOP K=128 ");  // V17: x   v159NN: ? GTVPUX:   GTL: ??
                case_run(wl_SOK_TOP_8x, "TEST of: _8x SOK broadcast TOP K=128 ");
                case_run(wl_SOK_TOP_halo_8x, "TEST of: _8x SOK TOP broadcast HALO OK,  K=128 ");

                // 4x
                case_run(wl_SOK_TOP_CLU_4x,
                         "\nTEST of: _4x SOK CLU no broadcast TOP K=128 ");  // V17: x   v159NN: ? GTVPUX:   GTL: ??
            }
        }

        {
            std::cout << "\n ------- SOK SPLITS: MEMORY ASPECTS ------- ";
            DPU_OperationValidator dut;
            // DPU_SplitLayersValidator dut2;

            MemorySize mem;
            {
                std::cout << "\n   ---- SOK , CLU, no halo  ------- ";
                DPUWorkload wl{std::move(wl_SOK_TOP_CLU_16x)};
                EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;
                EXPECT_LE(mem.cmx, (1024 * 1024 + 1024 * 512)) << "\nMemory size: " << mem << wl << std::endl;
                std::cout << mem << std::endl;
            }
            {
                std::cout << "\n   ---- SOK broadcast, no halo  ------- ";
                DPUWorkload wl = wl_SOK_TOP_16x;
                EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;
                EXPECT_LE(mem.cmx, (1024 * 1024 + 1024 * 512)) << "\nMemory size: " << mem << wl << std::endl;
                std::cout << mem << std::endl;
            }
            {
                std::cout << "\n   ---- SOK broadcast, WITH halo  ------- ";
                DPUWorkload wl{std::move(wl_SOK_TOP_halo_16x)};
                EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;
                EXPECT_GE(mem.cmx, (1024 * 1024 + 1024 * 512)) << "\nMemory size: " << mem << wl << std::endl;
                std::cout << mem << std::endl;
            }
            {
                std::cout << "\n   ---- FULL LAYER  no halo  ------- ";
                DPUWorkload wl = std::move(wl_);
                EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;
                EXPECT_GE(mem.cmx, (1024 * 1024 + 1024 * 512)) << "\nMemory size: " << mem << wl << std::endl;
                std::cout << mem << std::endl;
            }
        }
        {
            std::cout << "\n   ------- LAYER Again, but on already split clusters.   ------- ";
            {                                              // note: Real LNL NN has big delta vs GTL
                const DPULayer tst_layer(wl_SOK_TOP_16x);  // has owt so it will force a broadcast
                const std::vector<TestCase> tests{
                        {{std::move(tst_layer), {1U, 1U, 4U, VPUNN::VPUTilingStrategy::NONE, false, false, prefetch}},
                         {Cycles::NO_ERROR, true, 340000, fail * 370000 + 30000},
                         "Tile split + CLU+ broadcast , no memmove, "},

                };
                executeTests(tests);
            }
        }
    }
}

TEST_F(VPULayerCM_InvestigationTestNPU4x, Layer_EISXW_132141_SEP_split_Qualitative_NPU40) {
    const bool force_fail{};             // controls force failing assertion
    const int fail{force_fail ? 0 : 1};  // 1 neutral, 0 fail
    show_split = false;
    // EXPECT_TRUE(false);
    const HaloWorkload h_zero{{0, 0, 0, 0, 0, 0},  // H in TBLRFB
                              {0, 0, 0, 0, 0, 0},
                              {0, 0, 0, 0, 0, 0},
                              {0, 0, 0, 0, 0, 0}};
    const SEPModeInfo sepInfoInitial{
            true,               // sep activators using Storage elements table with pointers
            {243, 139, 1, 1},   // SEP pointer table, 32 bits pointers assumed
            {120, 68, 128, 1},  // actual tensor shape for activators
            false               // no_sparse_map if true the sparse map is ignored/non existent
    };
    const SEPModeInfo sepInfoInitialFixed{
            true,               // sep activators using Storage elements table with pointers
            {243, 71, 1, 1},    // SEP pointer table, 32 bits pointers assumed
            {120, 35, 128, 1},  // actual tensor shape for activators
            false               // no_sparse_map if true the sparse map is ignored/non existent
    };

    const DPUWorkload wl_layer_initial{
            // orig Layer
            VPUDevice::VPU_4_0,
            Operation::CONVOLUTION,
            {VPUTensor(243, 71, 128, 1, DataType::UINT8)},   // input dimensions
            {VPUTensor(240, 68, 64, 1, DataType::FLOAT16)},  // output dimensions
            {4, 4},                                          // kernels
            {1, 1},                                          // strides
            {0, 0, 0, 0},                                    // padding
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
            h_zero,                                          // halo
            sepInfoInitial,                                  // SEP configuration for input memory
    };

    // create a workload with fixed SEP configuration
    DPUWorkload wl_layer_fixed_sep = wl_layer_initial;
    wl_layer_fixed_sep.sep_activators = sepInfoInitialFixed;

    const DPUWorkload wl_SOHO_Initial{
            // orig Layer
            VPUDevice::VPU_4_0,
            Operation::CONVOLUTION,
            {VPUTensor(243, 20, 128, 1, DataType::UINT8)},   // input dimensions
            {VPUTensor(240, 17, 64, 1, DataType::FLOAT16)},  // output dimensions
            {4, 4},                                          // kernels
            {1, 1},                                          // strides
            {0, 0, 0, 0},                                    // padding
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
            h_zero,                                          // halo
            sepInfoInitial,                                  // SEP configuration for input memory
    };
    const DPUWorkload wl_SOHO_InitialFixed = [&] {
        DPUWorkload wl{wl_SOHO_Initial};
        wl.sep_activators = sepInfoInitialFixed;
        return wl;
    }();

    const SEPModeInfo sepInfoInitialSPLIT{
            true,                                         // sep activators using Storage elements table with pointers
            {243, ceil_division(139U * 20, 71U), 1, 1},   // SEP pointer table, 32 bits pointers assumed
            {120, ceil_division(68U * 20, 71U), 128, 1},  // actual tensor shape for activators
            false  // no_sparse_map if true the sparse map is ignored/non existent
    };

    const SEPModeInfo sepInfoInitialSOLITFixed{
            true,                                         // sep activators using Storage elements table with pointers
            {243, ceil_division(71U * 20, 71U), 1, 1},    // SEP pointer table, 32 bits pointers assumed
            {120, ceil_division(35U * 20, 71U), 128, 1},  // actual tensor shape for activators
            false  // no_sparse_map if true the sparse map is ignored/non existent
    };

    const DPUWorkload wl_SOHO_SEP_Split_initial = [&] {
        DPUWorkload wl{wl_SOHO_Initial};
        wl.sep_activators = sepInfoInitialSPLIT;
        return wl;
    }();

    const DPUWorkload wl_SOHO_SEP_Split_initial_Fix = [&] {
        DPUWorkload wl{wl_SOHO_Initial};
        wl.sep_activators = sepInfoInitialSOLITFixed;
        return wl;
    }();

    const bool prefetch{true};

    {  //
        // const DPULayer tst_layer(wl_layer_initial);
        //  same runtime for all SOHO and CLU
        // todo: check also in a test directly the split output
        // todo: add HK switch as SOHO + broadcast
        // here we do not care about the runtime , but we care about NO ERROR at sanitization (fits to memory)
        const std::vector<TestCase> tests{
                {{DPULayer(wl_layer_initial), {1U, 1U, 4U, VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {Cycles::NO_ERROR, true, 260000, fail * 300000 + 30000},  // 2// V17: x   v159NN: ? GTVPUX:   GTL: ??
                 "SOHO full I1, no memmove, "},

                {{DPULayer(wl_layer_fixed_sep),
                  {1U, 1U, 4U, VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {Cycles::NO_ERROR, true, 260000, fail * 300000 + 30000},  // 2// V17: x   v159NN: ? GTVPUX:   GTL: ??
                 "SOHO full I1 fixed, no memmove, "},

                {{DPULayer(wl_SOHO_Initial), {1U, 1U, 4U, VPUTilingStrategy::NONE, false, false, prefetch}},
                 {Cycles::ERROR_INPUT_TOO_BIG, true, 260000,
                  fail * 300000 + 30000},  // 2// V17: x   v159NN: ? GTVPUX:   GTL: ??
                 "CLU SPLIT sep initial , no memmove, "},

                {{DPULayer(wl_SOHO_InitialFixed), {1U, 1U, 4U, VPUTilingStrategy::NONE, false, false, prefetch}},
                 {Cycles::NO_ERROR, true, 260000, fail * 300000 + 30000},  // 2// V17: x   v159NN: ? GTVPUX:   GTL: ??
                 "CLU SPLIT sep initial fixed, no memmove, "},

                // with good split of sep
                {{DPULayer(wl_SOHO_SEP_Split_initial), {1U, 1U, 4U, VPUTilingStrategy::NONE, false, false, prefetch}},
                 {Cycles::NO_ERROR, true, 260000, fail * 300000 + 30000},  // 2// V17: x   v159NN: ? GTVPUX:   GTL: ??
                 "CLU SPLIT sep initial SPLIT , no memmove, "},

                {{DPULayer(wl_SOHO_SEP_Split_initial_Fix),
                  {1U, 1U, 4U, VPUTilingStrategy::NONE, false, false, prefetch}},
                 {Cycles::NO_ERROR, true, 260000, fail * 300000 + 30000},  // 2// V17: x   v159NN: ? GTVPUX:   GTL: ??
                 "CLU SPLIT sep initial SPLIT fixed, no memmove, "},

        };

        executeTests(tests);
    }

    VPULayerCostModel& theModel = layer_models.getModel(VPUDevice::VPU_4_0);

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
        {
            Logger::clear2ndlog();
            std::string err_info;

            {  // SOHO SPLITS
                std::cout << "\n ------- SOHO SPLITS: ------- \n";

                case_run(wl_SOHO_Initial, "TEST of: SOHO initial SEP : ");  // V17: x   v159NN: ? GTVPUX:   GTL: ??
                case_run(wl_SOHO_InitialFixed, "TEST of: SOHO initial Fixed SEP : ");

                case_run(wl_SOHO_SEP_Split_initial, "TEST of: SOHO SPLIT  initial sep : ");
                case_run(wl_SOHO_SEP_Split_initial_Fix, "TEST of: SOHO SPLIT  initial Fix sep : ");
                // case_run(wl_SOHO_Mid_K64x2, "TEST of: SOHO MID K=64 X2");
                // case_run(wl_SOHO_Bot_K64x2, "TEST of :SOHO BOT K=64 X2");
                //  Top mid bot are in parallel (X2 intra tile) =  (2x5927)||(2x6060)||(2x5972) = ()|(12120)|()  =
                //  12120 GT parallelism (2x6044)||(2x6265)||( 2x6298) = ()|()|(12596 )  =12596
            }
        }
    }
}

/// a simple template for investigation of Layer splits ops
TEST_F(VPULayerCM_InvestigationTestNPU4x, zTemplate_Layer_EISXW_xxxxxxx_NPUXX) {
    const bool force_fail{};             // controls force failing assertion
    const int fail{force_fail ? 0 : 1};  // 1 neutral, 0 fail
    show_split = false;
    // EXPECT_TRUE(false);

    const DPUWorkload wl_{
            // orig Layer
            VPUDevice::VPU_4_0,
            Operation::ELTWISE,
            {VPUTensor(250, 27, 64, 1, DataType::UINT8)},  // input dimensions
            {VPUTensor(250, 27, 64, 1, DataType::UINT8)},  // output dimensions
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
            ISIStrategy::CLUSTERING,                       // isi_strategy
            false,                                         // weight_sparsity_enabled
    };

    const bool prefetch{true};

    {  // note: Real LNL NN has big delta vs GTL
        const DPULayer tst_layer(wl_);
        const std::vector<TestCase> tests{
                {{std::move(tst_layer), {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 1000, fail * 5001 + 1500},  // 2// V17: x   v159NN: ? GTVPUX:   GTL: ??
                 "SOHO , no memmove, "},

                // note:?? wins
        };
        executeTests(tests);
    }

    VPULayerCostModel& theModel = layer_models.getModel(VPUDevice::VPU_4_0);

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
        const DPUWorkload wl_SOHO_Top_K64x2{
                VPUDevice::VPU_4_0,
                Operation::DW_CONVOLUTION,
                {VPUTensor(56, 15, 64, 1, DataType::UINT8)},  // input dimensions
                {VPUTensor(56, 14, 64, 1, DataType::UINT8)},  // output dimensions
                {3, 3},                                       // kernels
                {1, 1},                                       // strides
                {1, 0, 1, 1},                                 // padding
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
        DPUWorkload wl_SOHO_Mid_K64x2{wl_SOHO_Top_K64x2};
        wl_SOHO_Mid_K64x2.inputs[0] = VPUTensor(56, 16, 64, 1, DataType::UINT8);
        wl_SOHO_Mid_K64x2.padding = {0, 0, 1, 1};

        DPUWorkload wl_SOHO_Bot_K64x2{wl_SOHO_Top_K64x2};
        wl_SOHO_Bot_K64x2.padding = {0, 1, 1, 1};

        {
            Logger::clear2ndlog();
            std::string err_info;

            {  // SOHO SPLITS
                std::cout << "\n ------- SOHO SPLITS: ------- \n";

                case_run(wl_SOHO_Top_K64x2, "TEST of: SOHO TOP K=64 X2");  // V17: x   v159NN: ? GTVPUX:   GTL: ??
                case_run(wl_SOHO_Mid_K64x2, "TEST of: SOHO MID K=64 X2");
                case_run(wl_SOHO_Bot_K64x2, "TEST of :SOHO BOT K=64 X2");
                // Top mid bot are in parallel (X2 intra tile) =  (2x5927)||(2x6060)||(2x5972) = ()|(12120)|()  =
                // 12120 GT parallelism (2x6044)||(2x6265)||( 2x6298) = ()|()|(12596 )  =12596
            }
        }
    }
}

}