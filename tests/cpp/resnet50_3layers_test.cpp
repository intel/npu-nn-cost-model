// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include <gtest/gtest.h>
#include <sstream>  // for error formating
#include "common_helpers.h"

#include "cost_model_test.h"
#include "vpu_dma_cost_model.h"

#include <optional>
#include <variant>

namespace VPUNN_unit_tests {
using namespace VPUNN;

class TestResnet50_3Layers : public TestCostModel {
public:
    struct TestData {
        DPUWorkload wl;
        std::string info;
        std::array<unsigned int, 3> tolerance;

        TestData(const DPUWorkload& wl_, const std::string& info_, const std::array<unsigned int, 3> tolerance_ = {0,0,0})
                : wl(wl_), info(info_), tolerance(tolerance_) {
        }
    };

protected:
    static const VPUDevice dev{VPUDevice::VPU_2_7};
    // Layer 1 elm Float to int with Layout change!
    inline static const DPUWorkload s1_elmws_c0{
            dev,
            Operation::ELTWISE,
            {VPUTensor(114, 3, 224, 1, DataType::FLOAT16, Layout::ZXY)},  // input dimensions
            {VPUTensor(114, 3, 224, 1, DataType::UINT8, Layout::YZX)},    // output dimensions
            {1, 1},                                                       // kernels
            {1, 1},                                                       // strides
            {0, 0, 0, 0},                                                 // padding
            ExecutionMode::CUBOID_16x16,                                  // execution mode
            ActivationFunction::RELU,                                     // activation
            0.0F,                                                         // act_sparsity
            0.0F,                                                         // weight_sparsity
            {Swizzling::KEY_0, Swizzling::KEY_0},                         // input_swizzling
            {Swizzling::KEY_0},                                           // output_swizzling
            1,                                                            // output_write_tiles
            {0, 0, 0, 0},                                                 // offsets
            ISIStrategy::CLUSTERING,                                      // isi_strategy
            false,                                                        // weight_sparsity_enabled
    };

    inline static const DPUWorkload makeL1_Elmwise() {
        DPUWorkload clone = s1_elmws_c0;
        {
            clone.inputs = {VPUTensor(115, 3, 224, 1, DataType::FLOAT16, Layout::ZXY)};
            clone.outputs = {VPUTensor(115, 3, 224, 1, DataType::UINT8, Layout::YZX)};
        }
        return clone;
    }

    inline static const DPUWorkload s1_elmws_c1{makeL1_Elmwise()};
    const std::string s1_elmws_name{"Elmwise ZXY>YZX F16toUI8 	"};

    DMAWorkload dma_s1_elmws_c0{
            dev,                                                          // device
            {VPUTensor(114, 3, 224, 1, DataType::FLOAT16, Layout::ZXY)},  // input dimensions
            {VPUTensor(114, 3, 224, 1, DataType::FLOAT16, Layout::ZXY)},  // output dimensions
            MemoryLocation::DRAM,                                         // src
            MemoryLocation::CMX,                                          // dst
            1,                                                            // owt
    };
    DMAWorkload dma_s1_elmws_c1{
            dev,                                                          // device
            {VPUTensor(115, 3, 224, 1, DataType::FLOAT16, Layout::ZXY)},  // input dimensions
            {VPUTensor(115, 3, 224, 1, DataType::FLOAT16, Layout::ZXY)},  // output dimensions
            MemoryLocation::DRAM,                                         // src
            MemoryLocation::CMX,                                          // dst
            1,                                                            // owt
    };

    // Layer 2 conv
    inline static const DPUWorkload s2_mult_c0{
            dev,
            Operation::CONVOLUTION,
            {VPUTensor(224, 114, 3, 1, DataType::UINT8)},  // input dimensions
            {VPUTensor(112, 56, 64, 1, DataType::UINT8)},  // output dimensions
            {7, 7},                                        // kernels
            {2, 2},                                        // strides
            {3, 0, 3, 2},                                  // padding
            ExecutionMode::CUBOID_16x16,                   // execution mode
            ActivationFunction::RELU,                      // activation
            0.0F,                                          // act_sparsity
            0.0F,                                          // weight_sparsity
            {swz_def, swz_def},                            // input_swizzling
            {swz_def},                                     // output_swizzling
            1,                                             // output_write_tiles
            {0, 0, 0, 0},                                  // offsets
            ISIStrategy::CLUSTERING,                       // isi_strategy
            false,                                         // weight_sparsity_enabled
    };

    inline static const DPUWorkload makeL2_Conv7x7() {
        DPUWorkload clone = s2_mult_c0;
        {
            clone.inputs = {VPUTensor(224, 115, 3, 1, DataType::UINT8)};
            // same output
            clone.padding = {0, 2, 3, 2};
        }
        return clone;
    }
    inline static const DPUWorkload s2_mult_c1{makeL2_Conv7x7()};
    const std::string s2_mult_name{"Conv7x7>K64 			 	"};

    DMAWorkload dma_s2_INT32_WTable{
            // WTABle 1024
            dev,                                                              // device
            {VPUTensor(4, 1, 64, 4 /*INT32*/, DataType::INT8, Layout::ZXY)},  // input dimensions
            {VPUTensor(4, 1, 64, 4 /*INT32*/, DataType::INT8, Layout::ZXY)},  // output dimensions
            MemoryLocation::DRAM,                                             // src
            MemoryLocation::CMX,                                              // dst
            2,                                                                // owt
    };
    DMAWorkload dma_s2_UINT8_W{
            // should be 7x7=49 X3 X64
            dev,                                                       // device
            {VPUTensor(160, 1, 64, 1, DataType::UINT8, Layout::ZXY)},  // input dimensions
            {VPUTensor(160, 1, 64, 1, DataType::UINT8, Layout::ZXY)},  // output dimensions
            MemoryLocation::DRAM,                                      // src
            MemoryLocation::CMX,                                       // dst
            2,                                                         // owt
    };

    // Layer 3 maxpool
    const DPUWorkload s3_maxp_c0{
            dev,
            Operation::MAXPOOL,
            {VPUTensor(112, 56, 64, 1, DataType::UINT8)},  // input dimensions
            {VPUTensor(56, 28, 64, 1, DataType::UINT8)},   // output dimensions
            {3, 3},                                        // kernels
            {2, 2},                                        // strides
            {1, 0, 1, 0},                                  // padding
            ExecutionMode::CUBOID_16x16,                   // execution mode
            ActivationFunction::RELU,                      // activation
            0.0F,                                          // act_sparsity
            0.0F,                                          // weight_sparsity
            {swz_def, swz_def},                            // input_swizzling
            {swz_def},                                     // output_swizzling
            1,                                             // output_write_tiles
            {0, 0, 0, 0},                                  // offsets
            ISIStrategy::SPLIT_OVER_H,                     // isi_strategy
            false,                                         // weight_sparsity_enabled
    };
    const DPUWorkload makeL3_MaxPooling3x3() {
        DPUWorkload clone = s3_maxp_c0;
        {
            clone.inputs = {VPUTensor(112, 56, 64, 1, DataType::UINT8)};  // why 57?
            // same output
            clone.padding = {1, 0, 1, 0};
        }
        return clone;
    }
    const DPUWorkload s3_maxp_c1{makeL3_MaxPooling3x3()};
    const std::string s3_maxp_name{"MaxP 3x3>K64 			 	"};

    // Layer 4 Conv
    inline static const DPUWorkload s4_conv_c0{
            dev,
            Operation::CONVOLUTION,
            {VPUTensor(56, 28, 64, 1, DataType::UINT8)},  // input dimensions
            {VPUTensor(56, 28, 64, 1, DataType::UINT8)},  // output dimensions
            {1, 1},                                       // kernels
            {1, 1},                                       // strides
            {0, 0, 0, 0},                                 // padding
            ExecutionMode::CUBOID_16x16,                  // execution mode
            ActivationFunction::RELU,                     // activation
            0.0F,                                         // act_sparsity
            0.0F,                                         // weight_sparsity
            {swz_def, swz_def},                           // input_swizzling
            {swz_def},                                    // output_swizzling
            1,                                            // output_write_tiles
            {0, 0, 0, 0},                                 // offsets
            ISIStrategy::SPLIT_OVER_H,                    // why not CLU? // isi_strategy
            false,                                        // weight_sparsity_enabled
    };
    inline static const DPUWorkload s4_conv_c1{s4_conv_c0};
    const std::string s4_conv_name{"Conv 1x1>K64 			 	"};

    DMAWorkload dma_s4_fused_W_WT_UINT8{
            // w(64x64) + WT 1024
            dev,                                                       // device
            {VPUTensor(5120, 1, 1, 1, DataType::UINT8, Layout::ZXY)},  // input dimensions
            {VPUTensor(5120, 1, 1, 1, DataType::UINT8, Layout::ZXY)},  // output dimensions
            MemoryLocation::DRAM,                                      // src
            MemoryLocation::CMX,                                       // dst
            2,                                                         // owt
    };

    // Layer 5 Elm to float
    inline static const DPUWorkload s5_elmws_c0{
            dev,
            Operation::ELTWISE,
            {VPUTensor(56, 28, 64, 1, DataType::UINT8, Layout::ZXY)},    // input dimensions
            {VPUTensor(56, 28, 64, 1, DataType::FLOAT16, Layout::XYZ)},  // output dimensions
            {1, 1},                                                      // kernels
            {1, 1},                                                      // strides
            {0, 0, 0, 0},                                                // padding
            ExecutionMode::CUBOID_8x16,                                  // execution mode
            ActivationFunction::RELU,                                    // activation
            0.0F,                                                        // act_sparsity
            0.0F,                                                        // weight_sparsity
            {Swizzling::KEY_0, Swizzling::KEY_0},                        // input_swizzling
            {Swizzling::KEY_0},                                          // output_swizzling
            1,                                                           // output_write_tiles
            {0, 0, 0, 0},                                                // offsets
            ISIStrategy::SPLIT_OVER_H,                                   // isi_strategy
            false,                                                       // weight_sparsity_enabled
    };
    inline static const DPUWorkload s5_elmws_c1{s5_elmws_c0};
    const std::string s5_elmws_name{"Elm ZXY>XYZ 1x1>K64 UI8toF16"};

    DMAWorkload dma_Out_F16{
            dev,                                                         // device
            {VPUTensor(56, 28, 64, 1, DataType::FLOAT16, Layout::XYZ)},  // input dimensions
            {VPUTensor(56, 28, 64, 1, DataType::FLOAT16, Layout::XYZ)},  // output dimensions
            MemoryLocation::CMX,                                         // src
            MemoryLocation::DRAM,                                        // dst
            1,                                                           // owt
    };

    const std::vector<DPUWorkload> cluster_0{s1_elmws_c0, s2_mult_c0, s3_maxp_c0, s4_conv_c0, s5_elmws_c0};
    const std::vector<DPUWorkload> cluster_1{s1_elmws_c1, s2_mult_c1, s3_maxp_c1, s4_conv_c1, s5_elmws_c1};

    const std::vector<std::string> cluster_named{s1_elmws_name, s2_mult_name, s3_maxp_name, s4_conv_name,
                                                 s5_elmws_name};

    void SetUp() override {
        TestCostModel::SetUp();
    }

    TestResnet50_3Layers() {
    }

private:
};
TEST_F(TestResnet50_3Layers, DPUInfo_DPU_ResNet50F3_EISXW_91782) {
    std::vector<std::pair<std::string, DPUWorkload>> named_cluster_0;
    for (size_t i = 0; i < cluster_0.size(); ++i) {
        named_cluster_0.emplace_back(std::make_pair(cluster_named[i], cluster_0[i]));
    }
    std::vector<std::pair<std::string, DPUWorkload>> named_cluster_1;
    for (size_t i = 0; i < cluster_1.size(); ++i) {
        named_cluster_1.emplace_back(std::make_pair(cluster_named[i], cluster_1[i]));
    }

    // 27
    const std::string modelFile{VPU_2_7_MODEL_PATH};
    VPUCostModel test_model{modelFile};
    EXPECT_TRUE(test_model.nn_initialized());

    // EXPECT_EQ(1, 0);  // force fail, uncomment to have the log in tests

    {
        std::cout << "\n----------------------CLUSTER "
                     "0---------------------------------------------------------------------------  ";
        for (const auto& wl : named_cluster_0) {
            auto pInfo = test_model.DPUInfo(wl.second);
            EXPECT_FALSE(Cycles::isErrorCode(pInfo.DPUCycles)) << pInfo;
            std::cout << "\n---------------------------------------------------------  ";
            std::cout << "\n " << wl.first;
            std::cout << "\n " << wl.second;
            std::cout << "\n ***** ALT FORMAT Tile LAYER: ******** \n"
                      << WLHelp::toDictString(wl.second);  // dictionary style output for wl
            std::cout << "\n " << pInfo;
        }

        std::cout << "\n----------------------CLUSTER "
                     "1-----------------------------------------------------------------------------  ";
        for (const auto& wl : named_cluster_1) {
            auto pInfo = test_model.DPUInfo(wl.second);
            EXPECT_FALSE(Cycles::isErrorCode(pInfo.DPUCycles)) << pInfo;
            std::cout << "\n---------------------------------------------------------  ";
            std::cout << "\n " << wl.first;
            std::cout << "\n " << wl.second;
            std::cout << "\n ***** ALT FORMAT Tile LAYER: ******** \n"
                      << WLHelp::toDictString(wl.second);  // dictionary style output for wl
            std::cout << "\n " << pInfo;
        }
    }
    {
        std::cout << "\nName, \t Cycles,\t Energy,   ";
        std::cout << "\n----------------------CLUSTER 0-----------------------------------  ";
        for (const auto& wl : named_cluster_0) {
            auto pInfo = test_model.DPUInfo(wl.second);
            EXPECT_FALSE(Cycles::isErrorCode(pInfo.DPUCycles)) << pInfo;
            std::cout << "\n " << wl.first << ": \t\t\t\t\t " << pInfo.DPUCycles << " \t " << pInfo.energy;
        }

        std::cout << "\n----------------------CLUSTER 1-----------------------------------  ";
        for (const auto& wl : named_cluster_1) {
            auto pInfo = test_model.DPUInfo(wl.second);
            EXPECT_FALSE(Cycles::isErrorCode(pInfo.DPUCycles)) << pInfo;
            std::cout << "\n " << wl.first << ": \t\t\t\t\t " << pInfo.DPUCycles << " \t " << pInfo.energy;
        }
    }
}
TEST_F(TestResnet50_3Layers, DMA_ResNet50F3_EISXW_91782) {
    std::vector<std::pair<std::string, DMAWorkload>> named_DMA;

    named_DMA.emplace_back(std::make_pair("DMA_input_1_0", dma_s1_elmws_c0));  // GT? ns 5104  cyc 6635
    named_DMA.emplace_back(std::make_pair("DMA_input_1_1", dma_s1_elmws_c1));  // GT? ns 5651 cyc 7346

    named_DMA.emplace_back(std::make_pair("DMA_conv7x7_WTAble", dma_s2_INT32_WTable));  // GT? ns 468  cyc 608
    named_DMA.emplace_back(std::make_pair("DMA_conv7x7_W", dma_s2_UINT8_W));            // GT? ns 1093  cyc 1420

    named_DMA.emplace_back(std::make_pair("DMA_conv1x1: fusedWWT", dma_s4_fused_W_WT_UINT8));  // GT? ns 598  cyc 777

    named_DMA.emplace_back(std::make_pair("DMA_outputX:each", dma_Out_F16));  // GT? ns 9557 or 9244  cyc 12424 or 12017

    // 27
    const std::string modelFile{VPU_2_7_MODEL_PATH};
    VPUCostModel test_model{modelFile};
    EXPECT_TRUE(test_model.nn_initialized());

    DMACostModel<DMANNWorkload_NPU27> dma_model_2_7{VPU_DMA_2_7_MODEL_PATH};
    EXPECT_TRUE(dma_model_2_7.nn_initialized());

    // EXPECT_EQ(1, 0);  // force fail

    {
        std::cout << "\n----------------------DMA list "
                     "0---------------------------------------------------------------------------  ";
        for (const auto& wl : named_DMA) {
            CyclesInterfaceType cycles = test_model.DMA(wl.second);
            EXPECT_FALSE(Cycles::isErrorCode(cycles)) << cycles;

            const auto dmawl{DMAWorkloadTransformer::create_workload(wl.second)};
            CyclesInterfaceType cycles_dmann = dma_model_2_7.computeCycles(dmawl);
            EXPECT_FALSE(Cycles::isErrorCode(cycles_dmann)) << cycles_dmann;

            std::cout << "\n---------------------------------------------------------  ";
            std::cout << "\n Cycles OLD: " << cycles << ", Cycles dmann: " << cycles_dmann;  // cycles
            std::cout << "\n " << wl.first;                                                  // name
            std::cout << "\n " << wl.second;                                                 // DMA
            std::cout << "\n " << dmawl;                                                     // DMA wl
            // std::cout << "\n ***** ALT FORMAT Tile LAYER: ******** \n"
            //           << WLHelp::toDictString(wl.second);  // dictionary style output for wl
            // std::cout << "\n " << pInfo;
        }
    }
    {
        std::cout << "\n----------------------DMA list SHORT "
                     "---------------------------------------------------------------------------  ";
        std::cout << "\n name \t\t cycles OLD  \t cycles DMANN 1 plane ";
        for (const auto& wl : named_DMA) {
            CyclesInterfaceType cycles = test_model.DMA(wl.second);
            EXPECT_FALSE(Cycles::isErrorCode(cycles)) << cycles;

            const auto dmawl{DMAWorkloadTransformer::create_workload(wl.second)};
            CyclesInterfaceType cycles_dmann = dma_model_2_7.computeCycles(dmawl);
            EXPECT_FALSE(Cycles::isErrorCode(cycles_dmann)) << cycles_dmann;

            std::cout << "\n " << wl.first << "\t" << cycles << "\t" << cycles_dmann;  // name plus values
        }
    }
}
TEST_F(TestResnet50_3Layers, Energy_ResNet50F3_EISXW_101805) {
    // wl vector
    std::vector<std::pair<std::string, DPUWorkload>> named_cluster_0;
    for (size_t i = 0; i < cluster_0.size(); ++i) {
        named_cluster_0.emplace_back(std::make_pair(cluster_named[i], cluster_0[i]));
    }
    std::vector<std::pair<std::string, DPUWorkload>> named_cluster_1;
    for (size_t i = 0; i < cluster_1.size(); ++i) {
        named_cluster_1.emplace_back(std::make_pair(cluster_named[i], cluster_1[i]));
    }

    const std::string modelFile{VPU_2_7_MODEL_PATH};
    VPUCostModel test_model{modelFile};
    EXPECT_TRUE(test_model.nn_initialized());

    struct TestInput {
        VPUNN::DPUWorkload wl;  // a vector of wl
        std::string name_wl;
    };

    struct TestExpectation {
        CyclesInterfaceType cyc;
        float energy;
    };

    struct TestCase {
        TestInput t_in;
        TestExpectation t_exp;
        float c_tol_percent{5.0f};
    };

    using TestsVector = std::vector<TestCase>;
    auto test_energy_and_cyc = [&test_model](TestsVector tests_cluster, CyclesInterfaceType cyc_min_necessary_abs_diff,
                                             float energy_min_necessary_abs_diff, float energy_tol_percent) {
        int i = 1;
        for (auto& test : tests_cluster) {
            std::cout << "\n------ Test case " << i << " " << test.t_in.name_wl << "-------------\n";
            auto pInfo = test_model.DPUInfo(test.t_in.wl);
            EXPECT_TRUE(CompareValues::isEqual(pInfo.DPUCycles, test.t_exp.cyc, cyc_min_necessary_abs_diff,
                                               test.c_tol_percent))
                    << "Inferred: " << pInfo.DPUCycles << ", Expected : " << test.t_exp.cyc;
            EXPECT_TRUE(CompareValues::isEqual(pInfo.energy, test.t_exp.energy, energy_min_necessary_abs_diff,
                                               energy_tol_percent))
                    << "Inferred: " << pInfo.energy << ", Expected : " << test.t_exp.energy;

            std::cout << "\n T: " << test.t_in.name_wl << ", NN Cycles: " << pInfo.DPUCycles
                      << ", Energy: " << pInfo.energy << ", NN expected: " << test.t_exp.cyc;
            i++;
        }
    };

    // we still have big delta (30%+) for layer 3:  Conv 1x1>K64: (but small runtimes), and a big one for maxpooling
    // Layer 3:  MaxP 3x3>K64:

    // EXPECT_TRUE(false);// uncomment to fail and get a log

    const TestsVector tests_cluster_0 = {
            {{s1_elmws_c0, s1_elmws_name}, {9071, 487.5F}, 10.0f},     // old 5076    v16 8933    v17 9748 gt 9071.4
            {{s2_mult_c0, s2_mult_name}, {110667, 153663.9F}, 15.0f},  // old 118733  v16 124061  v17 118658 gt 110667.7
            {{s3_maxp_c0, s3_maxp_name}, {17941, 4851.0F}, 60.0f},     // old 12381   v16 12108   v17 11432 gt 17941.3
            {{s4_conv_c0, s4_conv_name}, {3456, 3136.0F}, 35.0F},      // old 4745    v16 4369    v17 4324 gt 3456.7 big
            {{s5_elmws_c0, s5_elmws_name}, {23706, 245.0F}, 20.0f}};   // old 3909    v16 26334   v17 28362 gt 23706.8

    const TestsVector tests_cluster_1 = {
            {{s1_elmws_c1, s1_elmws_name}, {9113, 494.0F}, 10.0f},     // old 5114,  v16 9006    v17 9815   gt 9113
            {{s2_mult_c1, s2_mult_name}, {110667, 153663.9F}, 10.0f},  // old 118153 v16 121676  v17 117425 gt 110667.7
            {{s3_maxp_c1, s3_maxp_name},
             {23517, 4851.0F},
             110.0f},  // old 12680, v16 12108 v17 11432  gt 23517 big del,gt??
            {{s4_conv_c1, s4_conv_name}, {3452, 3136.0F}, 30.0f},     // old 4745,  v16 4369    v17 4324   gt 3452.8 big
            {{s5_elmws_c1, s5_elmws_name}, {23705, 245.0F}, 20.0f}};  // old 3909,  v16 26334   v17 28362  gt 23705.5

    test_energy_and_cyc(tests_cluster_0, 50U, 50.0F, 5.0F);
    test_energy_and_cyc(tests_cluster_1, 50U, 50.0F, 5.0F);

    /*
        std::cout << "\n ------------------------------------------------------------------------\n ";
        std::cout << "\n\n" << s1_elmws_name << "\n" << s1_elmws_c0;
        std::cout << "\n\n" << s2_mult_name << "\n" << s2_mult_c0;
        std::cout << "\n\n" << s3_maxp_name << "\n" << s3_maxp_c0;
        std::cout << "\n\n" << s4_conv_name << "\n" << s4_conv_c0;
        std::cout << "\n\n" << s5_elmws_name << "\n" << s5_elmws_c0;

        std::cout << "\n ------------------------------------------------------------------------\n ";

        std::cout << "\n\n" << s1_elmws_name << "\n" << s1_elmws_c1;
        std::cout << "\n\n" << s2_mult_name << "\n" << s2_mult_c1;
        std::cout << "\n\n" << s3_maxp_name << "\n" << s3_maxp_c1;
        std::cout << "\n\n" << s4_conv_name << "\n" << s4_conv_c1;
        std::cout << "\n\n" << s5_elmws_name << "\n" << s5_elmws_c1;

        std::cout << "\n ------------------------------------------------------------------------\n ";
    */
}

// wls taken as examples are from ticket: EISXW-91782
class DifferentSwizz : public TestResnet50_3Layers, public testing::WithParamInterface<TestResnet50_3Layers::TestData> {
protected:
    // change input0 and output0 swizzling for a workload
    DPUWorkload changeSwizz(const DPUWorkload& wl_ref, Swizzling swizz_in0, Swizzling swizz_out0) {
        DPUWorkload wl{wl_ref};
        wl.input_swizzling[0] = swizz_in0;
        wl.output_swizzling[0] = swizz_out0;
        return wl;
    }

    // change device for a forkload
    DPUWorkload ch_dev(const DPUWorkload& wl_ref, VPUDevice device) {
        DPUWorkload wl{wl_ref};
        wl.device = device;
        return wl;
    }

    std::string wl_info(DPUWorkload wl) {
        return " Op: " + Operation_ToText.at(static_cast<int>(wl.op)) +
               " in0_swizz: " + Swizzling_ToText.at(static_cast<int>(wl.input_swizzling[0])) +
               " out0_swizz: " + Swizzling_ToText.at(static_cast<int>(wl.output_swizzling[0])) + "\n ";
    }

    std::array<std::string, 4> name_wl(std::array<DPUWorkload, 4> wls) {
        std::array<std::string, 4> named_wls{};
        for (int i = 0; i < 4; i++) {
            named_wls[i] = wl_info(wls[i]);
        }
        return named_wls;
    }

public:
    using TestResnet50_3Layers::s1_elmws_c0;
    using TestResnet50_3Layers::s2_mult_c0;
    using TestResnet50_3Layers::s4_conv_c0;
    using TestResnet50_3Layers::s5_elmws_c0;

    using TestResnet50_3Layers::s1_elmws_c1;
    using TestResnet50_3Layers::s2_mult_c1;
    using TestResnet50_3Layers::s4_conv_c1;
    using TestResnet50_3Layers::s5_elmws_c1;

    // using TestResnet50_3Layers::TestData;
};

TEST_P(DifferentSwizz, Diff_In_Out_Swizz_Cycles_Comparing_NPU40) {
    TestData data = GetParam();
    DPUWorkload wl_ref{data.wl};
    std::string tst_info{data.info};
    std::array<unsigned int, 3> tolerance{
            data.tolerance};  // cycles tolerance for DPU cost cycle for different swizzling

    // change device
    VPUDevice device{VPUDevice::VPU_4_0};
    DPUWorkload wl = ch_dev(wl_ref, device);

    // different input0 and output0 swizzling for the same workload
    DPUWorkload wl_K00 = changeSwizz(wl, Swizzling::KEY_0, Swizzling::KEY_0);
    DPUWorkload wl_K05 = changeSwizz(wl, Swizzling::KEY_0, Swizzling::KEY_5);
    DPUWorkload wl_K50 = changeSwizz(wl, Swizzling::KEY_5, Swizzling::KEY_0);
    DPUWorkload wl_K55 = changeSwizz(wl, Swizzling::KEY_5, Swizzling::KEY_5);

    std::array<DPUWorkload, 4> wls{wl_K00, wl_K05, wl_K50, wl_K55};
    std::array<std::string, 4> named_wls{name_wl(wls)};

    auto lambda = [this, &tst_info, &tolerance](std::array<DPUWorkload, 4> wls, std::array<std::string, 4> named_wls) {
        unsigned cost_cyc{};
        std::string info;
        std::array<unsigned int, 4> wls_cost_cyc;

        std::cout << "-----------------------------------------------------" << tst_info
                  << "-----------------------------------------------------\n";

        for (int i = 0; i < 4; i++) {
            ASSERT_NO_THROW(cost_cyc = getModel(wls[i].device).DPU(wls[i], info)) << named_wls[i] << "\n" << wls[i];
            wls_cost_cyc[i] = cost_cyc;
        }

        // array positions named after swizzling
        int swizz00 = 0;
        int swizz05 = 1;
        int swizz50 = 2;
        int swizz55 = 3;
        //tolerante in functie de test
        EXPECT_GT(wls_cost_cyc[swizz00] + tolerance[0], wls_cost_cyc[swizz55])
                << "Cost cycle for workload " << named_wls[swizz00]
                << " should be greater than cost cycle for workload " << named_wls[swizz55] << "\n";

        EXPECT_GT(wls_cost_cyc[swizz00] + tolerance[1], wls_cost_cyc[swizz50])
                << "Cost cycle for workload " << named_wls[swizz00]
                << " should be greater than cost cycle for workload " << named_wls[swizz50] << "\n";

        EXPECT_GT(wls_cost_cyc[swizz00] + tolerance[2], wls_cost_cyc[swizz05])
                << "Cost cycle for workload " << named_wls[swizz00]
                << " should be greater than cost cycle for workload " << named_wls[swizz05] << "\n";
    };

    lambda(wls, named_wls);
}

// these parameters are indexes we will use to access an element from this array -> test_wls
INSTANTIATE_TEST_SUITE_P(DifferentSwizzCollectionTests, DifferentSwizz,
                         ::testing::Values(DifferentSwizz::TestData{DifferentSwizz::s1_elmws_c0, "CLUSTER0", {0,0,0}},
                                           // TestData{DifferentSwizz::s2_mult_c0, "CLUSTER0"},
                                           // TestData{DifferentSwizz::s4_conv_c0, "CLUSTER0"},
                                           DifferentSwizz::TestData{DifferentSwizz::s5_elmws_c0, "CLUSTER0", {500,500,0}},

                                           DifferentSwizz::TestData{DifferentSwizz::s1_elmws_c1, "CLUSTER1", {0,0,0}},
                                           // TestData{DifferentSwizz::s2_mult_c1, "CLUSTER1"},
                                           // TestData{DifferentSwizz::s4_conv_c1, "CLUSTER1"},
                                           DifferentSwizz::TestData{DifferentSwizz::s5_elmws_c1, "CLUSTER1", {500,500,0}}
                                           ));

}  // namespace VPUNN_unit_tests