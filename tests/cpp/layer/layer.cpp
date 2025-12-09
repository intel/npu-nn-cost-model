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
#include "layer.h"
#include "vpu/shave/layers.h"

namespace VPUNN_unit_tests {
using namespace VPUNN;

TEST_F(VPULayerCostModelTest, LayerLoadModels) {
    EXPECT_EQ(layer_models.getModel(VPUDevice::VPU_2_0).get_cost_model().nn_initialized(), true);
    EXPECT_EQ(layer_models.getModel(VPUDevice::VPU_2_7).get_cost_model().nn_initialized(), true);
    EXPECT_EQ(model_2_7_no_dma.get_cost_model().nn_initialized(), true);
    EXPECT_EQ(layer_models.getModel(VPUDevice::VPU_4_0).get_cost_model().nn_initialized(), true);
#ifdef INTEL_EMBARGO_NPU5
    EXPECT_EQ(layer_models.getModel(VPUDevice::NPU_5_0).get_cost_model().nn_initialized(), true);
#endif  // INTEL_EMBARGO_NPU5
}

/// Test batch values for devices
TEST_F(VPULayerCostModelTest, BatchValues_LayerLevel) {
    auto mkLayer = [](VPUDevice dev, unsigned int b, Layout layout = Layout::ZXY) -> DPULayer {
        VPUNN::DPUWorkload wl{
                dev,
                Operation::CONVOLUTION,
                {VPUNN::VPUTensor(15, 50, 64, b, VPUNN::DataType::UINT8, layout)},  // input dimensions
                {VPUNN::VPUTensor(15, 50, 64, b, VPUNN::DataType::UINT8, layout)},  // output dimensions
                {1, 1},                                                             // kernels
                {1, 1},                                                             // strides
                {0, 0, 0, 0},                                                       // padding
                ExecutionMode::CUBOID_16x16,                                        // execution mode
                ActivationFunction::NONE,                                           // activation
                0.0F,                                                               // act_sparsity
                0.0F,                                                               // weight_sparsity
                {Swizzling::KEY_0, Swizzling::KEY_0},                               // input_swizzling
                {Swizzling::KEY_0},                                                 // output_swizzling
        };

        DPULayer layer(wl);
        return layer;
    };

    bool prefetch{true};
    show_split = true;
    unsigned int no_fail = 1;
    {
        const std::vector<TestCase> tests{
                {{mkLayer(VPUDevice::VPU_2_0, 0), {1U, 1U, 1U, VPUNN::VPUTilingStrategy::NONE, false, false, prefetch}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 12500, 12500 * no_fail + 1000},
                 "Device 2.0, B=0 "},
                {{mkLayer(VPUDevice::VPU_2_0, 1, Layout::ZMAJOR),
                  {1U, 1U, 1U, VPUNN::VPUTilingStrategy::NONE, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 12500, 12500 * no_fail + 1000},
                 "Device 2.0, B=1 "},
                {{mkLayer(VPUDevice::VPU_2_0, 2, Layout::ZMAJOR),
                  {1U, 1U, 1U, VPUNN::VPUTilingStrategy::NONE, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 6500, 6500 * no_fail + 1000},
                 "Device 2.0, B=2 "},

                {{mkLayer(VPUDevice::VPU_2_7, 0), {1U, 1U, 1U, VPUNN::VPUTilingStrategy::NONE, false, false, prefetch}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 12500, 12500 * no_fail + 1000},
                 "Device 2_7, B=0 "},
                {{mkLayer(VPUDevice::VPU_2_7, 1), {1U, 1U, 1U, VPUNN::VPUTilingStrategy::NONE, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 1500, 1500 * no_fail + 1000},
                 "Device 2.7, B=1 "},
                {{mkLayer(VPUDevice::VPU_2_7, 2), {1U, 1U, 1U, VPUNN::VPUTilingStrategy::NONE, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 1300, 1300 * no_fail + 1000},
                 "Device 2.7, B=2 "},

                {{mkLayer(VPUDevice::VPU_4_0, 0), {1U, 1U, 1U, VPUNN::VPUTilingStrategy::NONE, false, false, prefetch}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 12500, 12500 * no_fail + 1000},
                 "Device 4.0, B=0 "},
                {{mkLayer(VPUDevice::VPU_4_0, 1), {1U, 1U, 1U, VPUNN::VPUTilingStrategy::NONE, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 2000, 2000 * no_fail + 1000},
                 "Device 4.0, B=1 "},
                {{mkLayer(VPUDevice::VPU_4_0, 2), {1U, 1U, 1U, VPUNN::VPUTilingStrategy::NONE, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 1200, 1200 * no_fail + 1000},
                 "Device 4.0, B=2 "},
#ifdef INTEL_EMBARGO_NPU5
                {{mkLayer(VPUDevice::NPU_5_0, 0), {1U, 1U, 1U, VPUNN::VPUTilingStrategy::NONE, false, false, prefetch}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 12500, 12500 * no_fail + 1000},
                 "Device 5.0, B=0 "},
                {{mkLayer(VPUDevice::NPU_5_0, 1), {1U, 1U, 1U, VPUNN::VPUTilingStrategy::NONE, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 1000, 1000 * no_fail + 1000},
                 "Device 5.0, B=1 "},
                {{mkLayer(VPUDevice::NPU_5_0, 2), {1U, 1U, 1U, VPUNN::VPUTilingStrategy::NONE, false, false, prefetch}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 12500, 12500 * no_fail + 1000},
                 "Device 5.0, B=2 "},
#endif  // INTEL_EMBARGO_NPU5
        };
        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, Default_MaxWorkloadSPlitAndDetails_Test) {
    std::vector<const VPULayerCostModel*> all_models{
            &layer_models.getModel(VPUDevice::VPU_2_0), &layer_models.getModel(VPUDevice::VPU_2_7), &model_2_7_no_dma,
            &layer_models.getModel(VPUDevice::VPU_4_0),
#ifdef INTEL_EMBARGO_NPU5
            &layer_models.getModel(VPUDevice::NPU_5_0),
#endif  // INTEL_EMBARGO_NPU5
    };

    for (const auto m : all_models) {
        const auto maxWSplit = (*m).get_maxWorkloadsPerIntraTileSplit();
        EXPECT_EQ(128U, maxWSplit) << "max workloads split must be default";
    }
}

TEST_F(VPULayerCostModelTest, 01_C01_CONVOLUTION_Multiply_6346) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(30, 23, 1024, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(30, 23, 208, 1, VPUNN::DataType::UINT8)},     // output dimensions
            {3, 3},                                                         // kernels
            {1, 1},                                                         // strides
            {1, 1, 1, 1}                                                    // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = true;
        tst_layer.weight_sparsity = 0.950016f;

        const std::vector<TestCase> tests{
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                // {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                // "CLUSTERING ,flt , sparse "},
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 270000, 270000 + 115000},
                 // v16: 322K,   v17 279k GT???k // Out chnannels %32  //v150 383k
                 "SOK , no memmove, sparse"},
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                // {VPUNN::Cycles::NO_ERROR, true, 330000, 330000 + 1000},  // due to cmx overhead
                // "SOH , no memmove, sparse"},
        };

        executeTests(tests);
    }
}
TEST_F(VPULayerCostModelTest, 01_C02_CONVOLUTION_Multiply_6356) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(60, 2, 1024, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(60, 1, 512, 1, VPUNN::DataType::UINT8)},     // output dimensions
            {3, 3},                                                        // kernels
            {1, 1},                                                        // strides
            {0, 1, 1, 1}                                                   // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = true;
        tst_layer.weight_sparsity = 0.880725f;

        const std::vector<TestCase> tests{
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                // {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                // "CLUSTERING ,flt , sparse "},
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 280000, 280000 + 28000},  // v16 285k, V17 297K,  GT??
                 "SOK , no memmove, sparse"},
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                // {VPUNN::Cycles::NO_ERROR, true, 330000, 330000 + 1000},  // due to cmx overhead
                // "SOH , no memmove, sparse"},
        };

        executeTests(tests);
    }

    const VPUNN::DPULayer tst_layer_ref2(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(60, 3, 1024, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(60, 1, 512, 1, VPUNN::DataType::UINT8)},     // output dimensions
            {3, 3},                                                        // kernels
            {1, 1},                                                        // strides
            {0, 0, 1, 1}                                                   // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref2);
        tst_layer.weight_sparsity_enabled = true;
        tst_layer.weight_sparsity = 0.880725f;

        const std::vector<TestCase> tests{
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                // {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                // "CLUSTERING ,flt , sparse "},
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 280000, 280000 + 20000},  // GT??
                 "SOK , no memmove, sparse , padding 00"},
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                // {VPUNN::Cycles::NO_ERROR, true, 330000, 330000 + 1000},  // due to cmx overhead
                // "SOH , no memmove, sparse"},
        };

        executeTests(tests);
    }
}
TEST_F(VPULayerCostModelTest, 01_C03_ConvolutionBackpropData_1055) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(240, 21, 256, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(240, 21, 128, 1, VPUNN::DataType::UINT8)},    // output dimensions
            {2, 2},                                                         // kernels
            {1, 1},                                                         // strides
            {1, 0, 1, 0}                                                    // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = true;
        tst_layer.weight_sparsity = 0.36438f;

        const std::vector<TestCase> tests{
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                // {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                // "CLUSTERING ,flt , sparse "},
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                // {VPUNN::Cycles::NO_ERROR, true, 132000, 132000 + 1000},  //
                // "SOK , no memmove, sparse"},
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 275000, 275000 + 25000},  //
                 // pre v16:29xK , after v16 28x
                 "SOHO , no memmove, sparse"},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, 01_C04_ConvolutionBackpropData_1183) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(480, 23, 128, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(480, 23, 64, 1, VPUNN::DataType::UINT8)},     // output dimensions
            {2, 2},                                                         // kernels
            {1, 1},                                                         // strides
            {1, 0, 0, 1}                                                    // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = true;
        tst_layer.weight_sparsity = 0.0664062f;

        const std::vector<TestCase> tests{
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                // {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                // "CLUSTERING ,flt , sparse "},
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                // {VPUNN::Cycles::NO_ERROR, true, 132000, 132000 + 1000},  //
                // "SOK , no memmove, sparse"},
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 170000, 170000 + 35000},  //
                 // pre v16 200k, v16: 17xk, GT ???k
                 "SOHO , no memmove, sparse"},
        };

        executeTests(tests);
    }
}
TEST_F(VPULayerCostModelTest, DW_Convolution_AsymetricKernel_1) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::DW_CONVOLUTION,
            {VPUNN::VPUTensor(168, 97, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(168, 96, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {1, 2},                                                      // kernels
            {1, 1},                                                      // strides
            {0, 0, 0, 0}                                                 // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 25000, 25000 + 5000},
                 "CLUSTERING , no memove,  "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 23000, 23000 + 3000},  //
                 "SOK , no memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 14000, 14000 + 2000},  //
                 "SOHO , no memmove, "},
                // note: SOK always wins (non overlapping regions)
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, DW_Convolution_K8Stride8_H32) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::DW_CONVOLUTION,
            {VPUNN::VPUTensor(32, 32, 64, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(4, 4, 64, 1, VPUNN::DataType::FLOAT16)},    // output dimensions
            {8, 8},                                                       // kernels
            {8, 8},                                                       // strides
            {0, 0, 0, 0}                                                  // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 4000, 4000 + 500},  // v159nn:4229  v17:4352
                 "CLUSTERING , ,  "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 3400, 3400 + 350},  // v159nn:3554  v17:3546
                 "SOK , no memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 4100, 4100 + 400},  // v159nn:4242  v17:4336
                 "SOHO , no memmove, "},

                // note: SOK always wins
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, DW_Convolution_K8Stride8_H8) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::DW_CONVOLUTION,
            {VPUNN::VPUTensor(8, 8, 128, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(1, 1, 128, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            {8, 8},                                                      // kernels
            {8, 8},                                                      // strides
            {0, 0, 0, 0}                                                 // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 5400, 5400 + 1000},
                 "CLUSTERING , ,  "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 2000, 2000 + 1000},  //
                 "SOK , no memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 7900,
                  7900 + 1000},  // H=8 with K=8 , no padding cannot be split on h
                 "SOH , no memmove, "},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, Model_C1_v2_a_0_int8_NCHW_FP16_LATENCY_API10_MLIR) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(1, 1, 16, 1, VPUNN::DataType::UINT8)},      // input dimensions
            {VPUNN::VPUTensor(1, 1, 8192, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            {1, 1},                                                       // kernels
            {1, 1},                                                       // strides
            {0, 0, 0, 0}                                                  // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 2000, 4000},
                 "SOK , no memmove, "},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, Model_G_v1_a_0_fp16_NCHW_FP16_LATENCY_API10_MLIR) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(4, 1, 8192, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(4, 1, 176, 1, VPUNN::DataType::FLOAT16)},   // output dimensions
            {1, 1},                                                       // kernels
            {1, 1},                                                       // strides
            {0, 0, 0, 0}                                                  // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 18000, 18000 + 8000},  // big
                 "SOK , no memmove, "},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, Model_L_v1_a_1_int8_NCHW_FP16_LATENCY_API10_MLIR) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(256, 129, 4, 1, VPUNN::DataType::UINT8)},     // input dimensions
            {VPUNN::VPUTensor(256, 128, 16, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            {3, 3},                                                         // kernels
            {1, 1},                                                         // strides
            {1, 0, 1, 1}                                                    // padding
    );
    // EXPECT_TRUE(false);
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 92000, 92000 + 4000},  // v16 94856 , v17 95654
                 "CLUSTERING , ,  "},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, Model_L_v1_a_1_int8_NCHW_FP16_LATENCY_API10_MLIR_Set_2) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(256, 256, 4, 1, VPUNN::DataType::UINT8)},     // input dimensions
            {VPUNN::VPUTensor(256, 256, 16, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            {3, 3},                                                         // kernels
            {1, 1},                                                         // strides
            {1, 1, 1, 1}                                                    // padding
    );
    // EXPECT_TRUE(false);
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 92000, 92000 + 4000},  // v16 94856 , v17 95654
                 "SOH , no memmove, "},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, redowa_deeplab_v3_dense_IRv11_FP16_LATENCY_MLIR_MORE_MEMORY) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::DW_CONVOLUTION,
            {VPUNN::VPUTensor(65, 16, 960, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(65, 8, 960, 1, VPUNN::DataType::FLOAT16)},   // output dimensions
            {9, 9},                                                        // kernels
            {1, 1},                                                        // strides
            {0, 0, 4, 4}                                                   // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 1, VPUNN::Cycles::START_ERROR_RANGE - 1},
                 "SOH , no memmove, "},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, redowa_deeplab_v3_dense_IRv11_FP16_INT8_LATENCY_MLIR_Set_1) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(513, 130, 4, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(257, 65, 32, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                      // kernels
            {2, 2},                                                      // strides
            {1, 0, 1, 1}                                                 // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 57700, 57700 + 5000},
                 "SOH , no memmove, "},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, redowa_deeplab_v3_dense_IRv11_FP16_INT8_LATENCY_MLIR_Set_2) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(513, 513, 4, 1, VPUNN::DataType::UINT8)},   // input dimensions
            {VPUNN::VPUTensor(257, 257, 32, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                       // kernels
            {2, 2},                                                       // strides
            {1, 1, 1, 1}                                                  // padding
    );
    // EXPECT_TRUE(false);
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 100000, 100000 + 17000},  // v16 104000 , v17 114995
                 "SOH , no memmove, "},
        };

        executeTests(tests);
    }
}

/// test for SOH with overlap,  and with HALO
/// overlap one does not fit into memory
TEST_F(VPULayerCostModelTest, deeplab_v3_SOH_HALO_EISXW_79152) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::DW_CONVOLUTION,
            {VPUNN::VPUTensor(65, 16, 960, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(65, 8, 960, 1, VPUNN::DataType::FLOAT16)},   // output dimensions
            {9, 9},                                                        // kernels
            {1, 1},                                                        // strides
            {0, 0, 4, 4}                                                   // padding
    );
    {
        auto tst_layer = tst_layer_ref;

        const std::vector<TestCase> tests{
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                 "SOHO , no memmove, "},
        };

        executeTests(tests);
    }
    {
        auto tst_layer = tst_layer_ref;

        const std::vector<TestCase> tests{
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_HaloRead, false, false, true}},
                 {Cycles::NO_ERROR, true, 600000, 600000 + 500000 + 350000},  // huge delta , what is the GT
                 // v17: :newmemtens 1390740
                 "SOH w in Halo , no memmove, "},
        };

        executeTests(tests);
    }
    {  // details
        DPULayer tst_layer(std::move(tst_layer_ref));
        const VPULayerStrategy strategy{1U, 1U, 2U, VPUTilingStrategy::SOH_HaloRead, false, false, true};

        VPULayerCostModel& theModel{layer_models.getModel(tst_layer.device)};

        Logger::clear2ndlog();
        unsigned cost_cyc{};
        LayerSplitInfo detailed_split;
        ASSERT_NO_THROW(cost_cyc = theModel.Layer(tst_layer, strategy.tiling_strategy, strategy.nDPUs, strategy.nTiles,
                                                  strategy.input_fetching, strategy.output_spilling,
                                                  strategy.prefetching, detailed_split))
                << tst_layer << strategy << "\n code: " << cost_cyc << " : " << Cycles::toErrorText(cost_cyc);

        ASSERT_EQ(detailed_split.size(), 2);
        const auto& top{detailed_split[0].inter_tile_split_layer};
        ASSERT_EQ(top.halo.input_0_halo.top, 0);
        ASSERT_EQ(top.halo.input_0_halo.bottom, 4);
        const auto& bot{detailed_split[1].inter_tile_split_layer};
        ASSERT_EQ(bot.halo.input_0_halo.top, 4);
        ASSERT_EQ(bot.halo.input_0_halo.bottom, 0);

        /* EXPECT_EQ(cost_cyc, 3) << tst_layer << toStringLayerSplitInfo(detailed_split) << Logger::get2ndlog()
                                << "\n code: " << cost_cyc << " : " << Cycles::toErrorText(cost_cyc);*/
    }
}

/// test for SOH with overlap,  and with HALO . Same memory, only tile 2 used halo
TEST_F(VPULayerCostModelTest, SOH_HALO_EISXW_87028) {
    const DPUWorkload wl_as_layer{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(640, 61, 16, 1, DataType::FLOAT16)},  // input dimensions
            {VPUTensor(320, 30, 32, 1, DataType::FLOAT16)},  // output dimensions
            {3, 3},                                          // kernels
            {2, 2},                                          // strides
            {0, 0, 0, 1},                                    // padding
            ExecutionMode::CUBOID_16x16,                     // execution mode
            ActivationFunction::NONE,                        // activation
            0.0F,                                            // act_sparsity
            0.611111F,                                       // weight_sparsity
            {swz_def, swz_def},                              // input_swizzling
            {swz_def},                                       // output_swizzling
            1,                                               // output_write_tiles
            {0, 0, 0, 0},                                    // offsets
            ISIStrategy::CLUSTERING,                         // isi_strategy
            true,                                            // weight_sparsity_enabled
                                                             // halo aspects    default!
    };

    const VPUNN::DPULayer tst_layer_ref{wl_as_layer};
    {
        auto tst_layer = tst_layer_ref;

        const std::vector<TestCase> tests{
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {Cycles::NO_ERROR, true, 19000, 19000 + 2000},
                 "SOHO , no memmove, "},
        };

        executeTests(tests);
    }
    {
        auto tst_layer = tst_layer_ref;

        const std::vector<TestCase> tests{
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_HaloRead, false, false, true}},
                 {Cycles::NO_ERROR, true, 22000, 22000 + 1200},  // 604935 split 15
                 "SOH w in Halo , no memmove, "},
        };

        executeTests(tests);
    }
    {  // details
        DPULayer tst_layer(std::move(tst_layer_ref));
        const VPULayerStrategy strategy{1U, 1U, 2U, VPUTilingStrategy::SOH_HaloRead, false, false, true};

        VPULayerCostModel& theModel{layer_models.getModel(tst_layer.device)};

        Logger::clear2ndlog();
        unsigned cost_cyc{};
        LayerSplitInfo detailed_split;
        ASSERT_NO_THROW(cost_cyc = theModel.Layer(tst_layer, strategy.tiling_strategy, strategy.nDPUs, strategy.nTiles,
                                                  strategy.input_fetching, strategy.output_spilling,
                                                  strategy.prefetching, detailed_split))
                << tst_layer << strategy << "\n code: " << cost_cyc << " : " << Cycles::toErrorText(cost_cyc);

        // EXPECT_EQ(cost_cyc, 3) << tst_layer << toStringLayerSplitInfo(detailed_split) << Logger::get2ndlog()
        //                        << "\n code: " << cost_cyc << " : " << Cycles::toErrorText(cost_cyc);

        ASSERT_EQ(detailed_split.size(), 2);
        const auto& top{detailed_split[0].inter_tile_split_layer};
        ASSERT_EQ(top.halo.input_0_halo.top, 0);
        ASSERT_EQ(top.halo.input_0_halo.bottom, 0);
        const auto& bot{detailed_split[1].inter_tile_split_layer};
        ASSERT_EQ(bot.halo.input_0_halo.top, 1);  // theonly halo is here
        ASSERT_EQ(bot.halo.input_0_halo.bottom, 0);
    }
}

TEST_F(VPULayerCostModelTest, scale_mobilenet_ssd_FP16_MLIR) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(5, 5, 128, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(3, 3, 256, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            {3, 3},                                                      // kernels
            {2, 2},                                                      // strides
            {1, 1, 1, 1}                                                 // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 7000, 7000 + 1000},
                 "SOH , no memmove, "},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, scale_Sphereface_FP16_INT8_MLIR) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(1, 1, 8192, 1, VPUNN::DataType::UINT8)},   // input dimensions
            {VPUNN::VPUTensor(1, 1, 256, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            {1, 1},                                                      // kernels
            {1, 1},                                                      // strides
            {0, 0, 0, 0}                                                 // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 10000, 10000 + 5000},  // big
                 "SOK , no memmove, "},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest,
       face_detection_adas_0001_caffe_dense_IRv11_FP16_INT8_THROUGHPUT_NCHW_NCHW_U8_FP16_API20_MLIR_Set_1) {
    const DPULayer tst_layer_ref(VPUNN::VPUDevice::VPU_2_7, Operation::DW_CONVOLUTION,
                                 {VPUNN::VPUTensor(6, 3, 128, 1, VPUNN::DataType::UINT8)},  // input dimensions
                                 {VPUNN::VPUTensor(3, 2, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
                                 {3, 3},                                                    // kernels
                                 {2, 2},                                                    // strides
                                 {1, 1, 1, 0}                                               // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 650, 650 + 200},  // v159nn:766  v17:792
                 "SOHO , no memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 250, 250 + 250},  // v159nn:335  v17:414
                 "SOK , no memmove, "},

                // note SOK always wins
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest,
       face_detection_adas_0001_caffe_dense_IRv11_FP16_INT8_THROUGHPUT_NCHW_NCHW_U8_FP16_API20_MLIR_Set_2) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(672, 96, 4, 1, VPUNN::DataType::UINT8)},   // input dimensions
            {VPUNN::VPUTensor(336, 48, 32, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                      // kernels
            {2, 2},                                                      // strides
            {1, 0, 1, 0}                                                 // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 55000, 55000 + 5000},
                 "CLU , no memmove, "},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, midas_672x384_onnx_dense_IRv10_INT8_NHWC_NHWC_U8_FP16_LATENCY_API10_MLIR_Set_1) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(672, 97, 4, 1, VPUNN::DataType::UINT8)},   // input dimensions
            {VPUNN::VPUTensor(336, 48, 32, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                      // kernels
            {2, 2},                                                      // strides
            {0, 0, 0, 1}                                                 // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 54000, 54000 + 6000},
                 "CLU , no memmove, "},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, midas_672x384_onnx_dense_IRv10_INT8_NHWC_NHWC_U8_FP16_LATENCY_API10_MLIR_Set_2) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(672, 384, 4, 1, VPUNN::DataType::UINT8)},   // input dimensions
            {VPUNN::VPUTensor(336, 192, 32, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                       // kernels
            {2, 2},                                                       // strides
            {0, 1, 0, 1}                                                  // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{std::move(tst_layer), {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 99000, 99000 + 30000},  // big
                 "SOHO , no memmove, "},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, isv_SemiSupSegmentationaAtten_onnx_dense_IRv11_FP16_NCHW_NCHW_FP16_FP16_MLIR_Set_1) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::DW_CONVOLUTION,
            {VPUNN::VPUTensor(17, 17, 96, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(3, 3, 96, 1, VPUNN::DataType::FLOAT16)},    // output dimensions
            {6, 6},                                                       // kernels
            {6, 6},                                                       // strides
            {0, 1, 0, 1}                                                  // padding
    );
    // EXPECT_TRUE(false);
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 2900, 2900 + 200},  // v16 3063, v17 2973, v159nn:3045
                 "SOH O, no memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 1700, 1700 + 200},  // v17:1821 v159nn:1773
                 "SOK, no memmove, "},

                // note SOK always wins
        };

        executeTests(tests);
    }
}
TEST_F(VPULayerCostModelTest, isv_SemiSupSegmentationaAtten_onnx_dense_IRv11_FP16_NCHW_NCHW_FP16_FP16_MLIR_Set_2) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::DW_CONVOLUTION,
            {VPUNN::VPUTensor(17, 17, 240, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(3, 3, 240, 1, VPUNN::DataType::FLOAT16)},    // output dimensions
            {6, 6},                                                        // kernels
            {6, 6},                                                        // strides
            {0, 1, 0, 1}                                                   // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 7000, 7000 + 1500},  // v17:7755  v159nn:7967
                 "SOHO , no memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 3600, 3600 + 500},  // v17:3996  v159nn:3720
                 "SOK , no memmove, "},

                // note : SOK always wins
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, isv_SemiSupSegmentationaAtten_onnx_dense_IRv11_FP16_NCHW_NCHW_FP16_FP16_MLIR_Set_3) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::DW_CONVOLUTION,
            {VPUNN::VPUTensor(17, 17, 128, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(3, 3, 128, 1, VPUNN::DataType::FLOAT16)},    // output dimensions
            {6, 6},                                                        // kernels
            {6, 6},                                                        // strides
            {0, 1, 0, 1}                                                   // padding
    );
    // EXPECT_TRUE(false);
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 3800, 3800 + 400},  // v16 4084     v17 3964 v159nn:4060
                 "SOH O , no memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 3800, 3800 + 400},  // v16 4080     v17 3992 , v159nn:4080
                 "CLU , no memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 1750, 1750 + 300},  // v17:1998  v159nn:1860
                 "SOK , no memmove, "},
                // note: SOK always wins
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, isv_SemiSupSegmentationaAtten_onnx_dense_IRv11_FP16_NCHW_NCHW_FP16_FP16_MLIR_Set_4) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::DW_CONVOLUTION,
            {VPUNN::VPUTensor(17, 17, 576, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(3, 3, 576, 1, VPUNN::DataType::FLOAT16)},    // output dimensions
            {6, 6},                                                        // kernels
            {6, 6},                                                        // strides
            {0, 1, 0, 1}                                                   // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 17000, 17000 + 2000},  // v17:17838    , v159nn:18270
                 "SOH O , no memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 8000, 8000 + 1500},  // v17:8991  v159nn:8370
                 "SOK , no memmove, "},

                // note: SOK always wins
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, CONVOLUTIONPrefetchTest_Multiply6326_IF) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(60, 7, 512, 1, VPUNN::DataType::UINT8)},    // input dimensions
            {VPUNN::VPUTensor(60, 6, 512, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            {3, 3},                                                       // kernels
            {1, 1},                                                       // strides
            {1, 0, 1, 1}                                                  // padding
    );

    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = true;
        tst_layer.weight_sparsity = 0.723566F;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, false, 250000 + 50000, 250000 + 50000 + 30000},
                 "CLUSTERING ,flt , sparse "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, false, 130000 + 25000, 130000 + 25000 + 20000},
                 "SOK , + memmove, sparse"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, false, 160000 + 50000, 160000 + 50000 + 20000},
                 "SOHO , + memmove, sparse"},
        };
    }
}

TEST_F(VPULayerCostModelTest, DW_Convolution_K8Stride8_H32_Prefetch) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::DW_CONVOLUTION,
            {VPUNN::VPUTensor(32, 32, 64, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(4, 4, 64, 1, VPUNN::DataType::FLOAT16)},    // output dimensions
            {8, 8},                                                       // kernels
            {8, 8},                                                       // strides
            {0, 0, 0, 0}                                                  // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 5500, 5500 + 1000},
                 "CLUSTERING , ,  "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 4501, 4501 + 1000},  // v159nn:
                 "SOK , no memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 5501, 5501 + 1000},  // v159nn:
                 "SOH , no memmove, "},
                // note SOK wins
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, DWConvolutionPrefetchTest) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::DW_CONVOLUTION,
            {VPUNN::VPUTensor(168, 97, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(168, 96, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {1, 2},                                                      // kernels
            {1, 1},                                                      // strides
            {0, 0, 0, 0}                                                 // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 27000, 28000 + 3000},
                 "CLUSTERING ,no prefetch ,  "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 24000, 24000 + 3000},
                 "SOK , no prefetch, "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 15000, 15000 + 2000},
                 "SOHO , no prefetch, "},
                // note: SOK WINS
        };

        executeTests(tests);
    }
}
TEST_F(VPULayerCostModelTest, Fused_234_3xELEMENTWISE_Prefetch) {
    // note: ELEMENTWISE that have datasize change will be considered without weights
    const VPUNN::DPULayer l1F_ref(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::ELTWISE,
                                  {VPUNN::VPUTensor(480, 123, 16, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
                                  {VPUNN::VPUTensor(480, 123, 16, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
                                  {1, 1},                                                         // kernels
                                  {1, 1},                                                         // strides
                                  {0, 0, 0, 0}                                                    // padding
    );
    const VPUNN::DPULayer l1Int_ref(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::ELTWISE,
                                    {VPUNN::VPUTensor(480, 123, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions
                                    {VPUNN::VPUTensor(480, 123, 16, 1, VPUNN::DataType::UINT8)},  // output dimensions
                                    {1, 1},                                                       // kernels
                                    {1, 1},                                                       // strides
                                    {0, 0, 0, 0}                                                  // padding
    );

    const VPUNN::DPULayer l1FI_ref(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::ELTWISE,
                                   {VPUNN::VPUTensor(480, 123, 16, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
                                   {VPUNN::VPUTensor(480, 123, 16, 1, VPUNN::DataType::UINT8)},    // output dimensions
                                   {1, 1},                                                         // kernels
                                   {1, 1},                                                         // strides
                                   {0, 0, 0, 0}                                                    // padding
    );

    const VPUNN::DPULayer l1and2_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::ELTWISE,
            {VPUNN::VPUTensor(480, 123, 16, 1, VPUNN::DataType::UINT8)},    // input dimensions
            {VPUNN::VPUTensor(480, 123, 16, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            {1, 1},                                                         // kernels
            {1, 1},                                                         // strides
            {0, 0, 0, 0}                                                    // padding
    );
    const VPUNN::DPULayer l3_ref(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::ELTWISE,
                                 {VPUNN::VPUTensor(480, 122, 16, 1, VPUNN::DataType::UINT8)},    // input dimensions
                                 {VPUNN::VPUTensor(480, 122, 16, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
                                 {1, 1},                                                         // kernels
                                 {1, 1},                                                         // strides
                                 {0, 0, 0, 0}                                                    // padding
    );

    {  // float all,Layer 1 changed
        std::string t{"All floats @ Layer 1 "};
        auto tst_layer = std::move(l1F_ref);
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                 t + "CLU +m "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 0, 0 + 0},
                 t + "SOK ,+m"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 30000 + 44000, 30000 + 44000 + 10000},  // v159 79k
                 t + "SOH ,+m"},
        };
        executeTests(tests);
    }

    {  // UINT8 all, Layer 1 changed
        std::string t{"All UINT8 @ Layer 1 "};
        auto tst_layer = std::move(l1Int_ref);
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 73000, 73000 + 5000},
                 t + "CLU +m "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 0, 0 + 0},
                 t + "SOK ,+m"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 38000, 38000 + 5000},
                 t + "SOH ,+m"},
        };
        executeTests(tests);
    }

    {  // FLOAT to INT all, Layer 1 changed , no more in place output possible
        std::string t{"Reversed F toI @ Layer 1  mixed"};
        auto tst_layer = std::move(l1FI_ref);
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0 + 0},  //
                 t + "CLU 0m "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 0, 0 + 0},
                 t + "SOK ,0m"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 58000, 58000 + 10000},
                 t + "SOHO ,0m"},
        };
        executeTests(tests);
    }

    {  // L1&2 int -float. Has to be IMPROVED, no more in place output possible
        std::string t{"Original Layer 1, mixed "};
        auto tst_layer = std::move(l1and2_ref);
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 85000, 85000 + 5000},
                 t + "CLU 0m "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 0, 0 + 0},
                 t + "SOK ,0m"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 47000, 47000 + 16000},
                 t + "SOHO ,0m"},
        };
        executeTests(tests);
    }

    {  // L3  int -float. Has to be IMPROVED, no more in place output possible
        std::string t{"Original Layer 3, mixed "};
        auto tst_layer = std::move(l3_ref);
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 85000, 85000 + 5000},
                 t + "CLU 0m "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 0, 0 + 0},
                 t + "SOK ,0m"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 46000, 46000 + 16000},
                 t + "SOH ,0m"},
        };
        executeTests(tests);
    }
    // ASSERT_TRUE(false);
}

TEST_F(VPULayerCostModelTest, ELTWISE_PrefetchTest) {
    const VPUNN::DPULayer tst_layer(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::ELTWISE,
                                    {VPUNN::VPUTensor(56, 56, 256, 1, VPUNN::DataType::UINT8)},  // input dimensions
                                    {VPUNN::VPUTensor(56, 56, 256, 1, VPUNN::DataType::UINT8)},  // output dimensions
                                    {1, 1},                                                      // kernels
                                    {1, 1},                                                      // strides
                                    {0, 0, 0, 0}                                                 // padding
    );
    const VPUNN::VPULayerStrategy tst_strategy{1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false};

    {  // clustering
        TestInput tin{tst_layer, tst_strategy};
        tin.strategy.tiling_strategy = VPUNN::VPUTilingStrategy::NONE;

        TestExpectations texp{NO_ERROR_EXPECTED, false, 55000, 55000 + 2000};  // without prefetching:  24000 , 25100

        DoRegularTest(tin, texp, "Clustering ,memove");
    }

    {  // SOHO
        TestInput tin{tst_layer, tst_strategy};
        tin.strategy.tiling_strategy = VPUNN::VPUTilingStrategy::SOH_Overlapped;

        TestExpectations texp{NO_ERROR_EXPECTED, false, 28000,
                              28000 + 3000U};  // without prefetching: 11000, 11000 + 1000U

        DoRegularTest(tin, texp, "SOHO,memove");
    }
    {  // SOK
        TestInput tin{std::move(tst_layer), tst_strategy};
        tin.strategy.tiling_strategy = VPUNN::VPUTilingStrategy::SOK;  // SOK is now allowed for element wise

        TestExpectations texp{NO_ERROR_EXPECTED, true, 26000, 26000 + 8000};

        DoRegularTest(tin, texp, "SOK tentative on elementwise, memove");
    }
}

TEST_F(VPULayerCostModelTest, CMCONVPrefetchTest) {
    // CONV with IC <16 to be compressed CONV
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(672, 97, 4, 1, VPUNN::DataType::UINT8)},   // input dimensions
            {VPUNN::VPUTensor(336, 48, 32, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                      // kernels
            {2, 2},                                                      // strides
            {0, 0, 0, 1}                                                 // padding
    );
    {
        auto tst_layer = std::move(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 55000, 55000 + 8000},
                 "NONE , + memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 30000, 30000 + 3000},
                 "SOHO , + memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 55000, 55000 + 8000},
                 "SOK , + memmove, "},
        };

        executeTests(tests);
    }
}

/// DMA CMX to/from DDRare equal because is limited by the maximum time (min bandwith)
TEST_F(VPULayerCostModelTest, Layer_DMA_DDRvsCMX_Smoke) {
    const VPUNN::DPULayer tst_layer_ref(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
                                        {VPUNN::VPUTensor(60, 7, 512, 1, VPUNN::DataType::INT8)},  // input dimensions
                                        {VPUNN::VPUTensor(60, 7, 512, 1, VPUNN::DataType::INT8)},  // output dimensions
                                        {3, 3},                                                    // kernels
                                        {1, 1},                                                    // strides
                                        {1, 1, 1, 1}                                               // padding
    );

    const bool prefetch{true};  // prefetch was done
    for (const auto theModel :
         std::vector<VPULayerCostModel*>{&layer_models.getModel(VPUDevice::VPU_2_7),
                                         &model_2_7_no_dma})  // check fallback to DMA theoretical as well
    {
        VPUNN::DPULayer tst_layer(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = true;
        tst_layer.weight_sparsity = 0.723566F;

        std::vector<DPULayer> splitLayers1{tst_layer};

        const VPUTilingStrategy strategy{VPUNN::VPUTilingStrategy::NONE};  // clustering 2T, no memo

        Logger::clear2ndlog();
        LayerSplitInfo detailed_split_layer;
        CyclesInterfaceType cost_cyc_LayerFromDDR{};
        ASSERT_NO_THROW(cost_cyc_LayerFromDDR = theModel->Layer(tst_layer, strategy, 1U, 2U, true, false, prefetch,
                                                                detailed_split_layer))
                << tst_layer << (int)strategy << " : " << VPUTilingStrategy_ToText.at(static_cast<int>(strategy))
                << toStringLayerSplitInfo(detailed_split_layer) << Logger::get2ndlog();

        Logger::clear2ndlog();
        detailed_split_layer.clear();

        CyclesInterfaceType cost_cyc_LayerFromCMX{};
        ASSERT_NO_THROW(cost_cyc_LayerFromCMX = theModel->Layer(tst_layer, strategy, 1U, 2U, false, true, prefetch,
                                                                detailed_split_layer))
                << tst_layer << (int)strategy << " : " << VPUTilingStrategy_ToText.at(static_cast<int>(strategy))
                << toStringLayerSplitInfo(detailed_split_layer) << Logger::get2ndlog();

        EXPECT_NEAR(cost_cyc_LayerFromDDR, cost_cyc_LayerFromCMX, cost_cyc_LayerFromCMX * 0.05)
                << tst_layer << Logger::get2ndlog() << "END ERR";  // 5% tolerance for ddr2cmx vs cmx2ddr

        Logger::clear2ndlog();
        detailed_split_layer.clear();

        CyclesInterfaceType cost_cyc_LayerNoMem{};
        ASSERT_NO_THROW(cost_cyc_LayerNoMem = theModel->Layer(tst_layer, strategy, 1U, 2U, false, false, prefetch,
                                                              detailed_split_layer))
                << tst_layer << (int)strategy << " : " << VPUTilingStrategy_ToText.at(static_cast<int>(strategy))
                << toStringLayerSplitInfo(detailed_split_layer) << Logger::get2ndlog();

        EXPECT_NE(cost_cyc_LayerFromDDR, cost_cyc_LayerNoMem) << tst_layer << Logger::get2ndlog() << "END ERR";

        EXPECT_GT(cost_cyc_LayerFromDDR, cost_cyc_LayerNoMem) << tst_layer << Logger::get2ndlog() << "END ERR";
        EXPECT_GT(cost_cyc_LayerFromCMX, cost_cyc_LayerNoMem) << tst_layer << Logger::get2ndlog() << "END ERR";

        EXPECT_NEAR(cost_cyc_LayerFromDDR, cost_cyc_LayerFromCMX, cost_cyc_LayerFromCMX * 0.05)
                << cost_cyc_LayerFromDDR << "\n"  // 5% tolerance for ddr2cmx vs cmx2ddr
                << cost_cyc_LayerFromCMX << "\n"
                << cost_cyc_LayerNoMem << "\n"
                << tst_layer << Logger::get2ndlog() << "END ERR";
        const auto& outT{tst_layer.outputs[0]};
        auto cmxddr_dma =
                theModel->get_cost_model().DMA(tst_layer.device, outT, outT, MemoryLocation::CMX, MemoryLocation::DRAM);
        auto ddrcmx_dma =
                theModel->get_cost_model().DMA(tst_layer.device, outT, outT, MemoryLocation::DRAM, MemoryLocation::CMX);
        EXPECT_EQ(ddrcmx_dma, cmxddr_dma) << ddrcmx_dma << "\n"
                                          << cmxddr_dma << "\n"
                                          << cost_cyc_LayerFromDDR << "\n"
                                          << cost_cyc_LayerFromCMX << "\n"
                                          << cost_cyc_LayerNoMem << "\n";
    }
}

TEST_F(VPULayerCostModelTest, L2_instantiated_with_external_DPU) {
    {
        const std::string model_path = VPU_2_7_MODEL_PATH;
        EXPECT_NO_THROW(VPUCostModel x{model_path});
        // VPUCostModel costmodel{model_path};
        auto costmodel{std::make_shared<VPUCostModel>(model_path)};
        const VPULayerCostModel layer_costmodel{costmodel};
        EXPECT_EQ(costmodel.get(), (&layer_costmodel.get_cost_model()));
    }
    {
        const std::string model_path = VPU_4_0_MODEL_PATH;
        EXPECT_NO_THROW(VPUCostModel x{model_path});
        // VPUCostModel costmodel{model_path};
        auto costmodel{std::make_shared<VPUCostModel>(model_path)};
        const VPULayerCostModel layer_costmodel{costmodel};
        EXPECT_EQ(costmodel.get(), (&layer_costmodel.get_cost_model()));
    }
}

TEST_F(VPULayerCostModelTest, Regression_sprint125_investigation_Test) {
    const VPUNN::DPUWorkload wl_0{
            VPUNN::VPUDevice::VPU_4_0,
            VPUNN::Operation::ELTWISE,
            {VPUNN::VPUTensor(512, 5, 512, 1, VPUNN::DataType::FLOAT16, Layout::ZXY)},  // input dimensions
            {VPUNN::VPUTensor(512, 5, 512, 1, VPUNN::DataType::FLOAT16, Layout::YZX)},  // output dimensions
            {1, 1},                                                                     // kernels
            {1, 1},                                                                     // strides
            {0, 0, 0, 0},                                                               // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                                         // execution mode
            VPUNN::ActivationFunction::NONE,                                            // activation
            0.0F,                                                                       // act_sparsity
            0.0F,                                                                       // weight_sparsity
            {swz_def, swz_def},                                                         // input_swizzling
            {swz_def},                                                                  // output_swizzling
            1,                                                                          // output_write_tiles
            {0, 0, 0, 0},                                                               // offsets
            VPUNN::ISIStrategy::CLUSTERING,                                             // isi_strategy
            false,                                                                      // weight_sparsity_enabled
    };
    const VPUNN::DPUWorkload wl_1{
            VPUNN::VPUDevice::VPU_4_0,
            VPUNN::Operation::ELTWISE,
            {VPUNN::VPUTensor(1024, 1, 1024, 1, VPUNN::DataType::FLOAT16, Layout::ZXY)},  // input dimensions
            {VPUNN::VPUTensor(1024, 1, 1024, 1, VPUNN::DataType::FLOAT16, Layout::YZX)},  // output dimensions
            {1, 1},                                                                       // kernels
            {1, 1},                                                                       // strides
            {0, 0, 0, 0},                                                                 // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                                           // execution mode
            VPUNN::ActivationFunction::NONE,                                              // activation
            0.0F,                                                                         // act_sparsity
            0.0F,                                                                         // weight_sparsity
            {swz_def, swz_def},                                                           // input_swizzling
            {swz_def},                                                                    // output_swizzling
            1,                                                                            // output_write_tiles
            {0, 0, 0, 0},                                                                 // offsets
            VPUNN::ISIStrategy::CLUSTERING,                                               // isi_strategy
            false,                                                                        // weight_sparsity_enabled
    };

    bool prefetch{true};
    show_split = true;
    unsigned int no_fail = 1;

    {
        const VPUNN::DPULayer tst_layer0(wl_0);
        const VPUNN::DPULayer tst_layer1(wl_1);
        const std::vector<TestCase> tests{
                {{tst_layer0, {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOW, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 36500U, 36500U * no_fail + 1000U},
                 "SOW, ch 512 "},
                {{tst_layer1, {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOW, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 73800U, 73800U * no_fail + 1000U},
                 "SOW, ch 1024 "},

        };
        executeTests(tests);
    }
}

}  // namespace VPUNN_unit_tests
