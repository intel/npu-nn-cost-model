// Copyright © 2023 Intel Corporation
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
#include "vpu_layer_cost_model.h"

namespace VPUNN_unit_tests {

class DPULayerTest : public ::testing::Test {
public:
protected:
    VPUNN::VPULayerCostModel model_2_7{VPU_2_7_MODEL_PATH};
    VPUNN::VPULayerCostModel model_2_0{VPU_2_0_MODEL_PATH};

    void SetUp() override {
        VPUNN::Logger::clear2ndlog();
        VPUNN::Logger::activate2ndlog();
    }
    void TearDown() override {
        VPUNN::Logger::clear2ndlog();
        VPUNN::Logger::deactivate2ndlog();
    }

    using ModelDescriptor =
            VPUNNModelsFiles::ModelDescriptor;  ///< make this type directly visible inside of this class
    const VPUNNModelsFiles& the_NN_models{VPUNNModelsFiles::getModels()};  ///< the paths to available NN models

    VPUNN::DPULayer generate_helper_layer(const unsigned int dim, const unsigned int channels) {
        return VPUNN::DPULayer(
                VPUNN::VPUDevice::VPU_2_0,                                            // VPU device
                VPUNN::Operation::CONVOLUTION,                                        // Operation
                {VPUNN::VPUTensor(dim, dim, channels, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
                {VPUNN::VPUTensor(dim, dim, channels, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
                {3, 3},                                                               // kernels
                {1, 1},                                                               // strides
                {1, 1, 1, 1}                                                          // padding
        );
    }

    VPUNN::SHVSigmoid generate_helper_sw_layer(const unsigned int dim, const unsigned int channels) {
        return VPUNN::SHVSigmoid(VPUNN::VPUDevice::VPU_2_0,                                          // VPU device
                                 VPUNN::VPUTensor(dim, dim, channels, 1, VPUNN::DataType::FLOAT16),  // Input tensor
                                 VPUNN::VPUTensor(dim, dim, channels, 1, VPUNN::DataType::FLOAT16)   // Output tensor
        );
    }
};

TEST_F(DPULayerTest, LayerLoadModels) {
    EXPECT_EQ(model_2_7.nn_initialized(), true);
    EXPECT_EQ(model_2_0.nn_initialized(), true);
}

TEST_F(DPULayerTest, SplitAcrossTileSOH) {
    auto wl = generate_helper_layer(16, 64);

    auto SOH_single_tile = wl.splitAcrossTiles(VPUNN::VPUTilingStrategy::SOH, 1);
    auto SOH_two_tile = wl.splitAcrossTiles(VPUNN::VPUTilingStrategy::SOH, 2);
    auto SOH_four_tile = wl.splitAcrossTiles(VPUNN::VPUTilingStrategy::SOH, 4);

    // Basic expectations
    EXPECT_EQ(SOH_single_tile.size(), 1);
    EXPECT_EQ(SOH_two_tile.size(), 2);
    EXPECT_EQ(SOH_four_tile.size(), 4);

    // The shape of the single split must be equal to the origial layer
    EXPECT_EQ(SOH_single_tile[0].outputs[0].get_shape(), wl.outputs[0].get_shape());

    EXPECT_EQ(SOH_two_tile[0].outputs[0].size(), wl.outputs[0].size() / 2);
    EXPECT_EQ(SOH_two_tile[0].outputs[0].get_shape()[1] * 2, wl.outputs[0].get_shape()[1]);

    EXPECT_EQ(SOH_four_tile[0].outputs[0].size(), wl.outputs[0].size() / 4);
    EXPECT_EQ(SOH_four_tile[0].outputs[0].get_shape()[1] * 4, wl.outputs[0].get_shape()[1]);
}

TEST_F(DPULayerTest, SplitAcrossTileSOK) {
    auto wl = generate_helper_layer(16, 64);

    auto SOK_single_tile = wl.splitAcrossTiles(VPUNN::VPUTilingStrategy::SOK, 1);
    auto SOK_two_tile = wl.splitAcrossTiles(VPUNN::VPUTilingStrategy::SOK, 2);
    auto SOK_four_tile = wl.splitAcrossTiles(VPUNN::VPUTilingStrategy::SOK, 4);

    // Basic expectations
    EXPECT_EQ(SOK_single_tile.size(), 1);
    EXPECT_EQ(SOK_two_tile.size(), 2);
    EXPECT_EQ(SOK_four_tile.size(), 4);

    // The shape of the single split must be equal to the origial layer
    EXPECT_EQ(SOK_single_tile[0].outputs[0].get_shape(), wl.outputs[0].get_shape());

    EXPECT_EQ(SOK_two_tile[0].outputs[0].size(), wl.outputs[0].size() / 2);
    EXPECT_EQ(SOK_two_tile[0].outputs[0].get_shape()[2] * 2, wl.outputs[0].get_shape()[2]);

    EXPECT_EQ(SOK_four_tile[0].outputs[0].size(), wl.outputs[0].size() / 4);
    EXPECT_EQ(SOK_four_tile[0].outputs[0].get_shape()[2] * 4, wl.outputs[0].get_shape()[2]);
}

TEST_F(DPULayerTest, SplitAcrossTileSOKAsymmetric) {
    auto wl = generate_helper_layer(16, 64);

    auto SOK_asymmetric = wl.splitAcrossTiles(VPUNN::VPUTilingStrategy::SOK, 5);

    // Basic expectations
    EXPECT_EQ(SOK_asymmetric.size(), 4);

    for (unsigned int idx = 0; idx < SOK_asymmetric.size(); idx++) {
        EXPECT_EQ(SOK_asymmetric[idx].outputs[0].get_shape()[2], 16u);
    }

    auto wl_2 = generate_helper_layer(16, 48);

    auto SOK_asymmetric_2 = wl_2.splitAcrossTiles(VPUNN::VPUTilingStrategy::SOK, 5);

    // Basic expectations
    EXPECT_EQ(SOK_asymmetric_2.size(), 3);

    for (unsigned int idx = 0; idx < SOK_asymmetric_2.size(); idx++) {
        EXPECT_EQ(SOK_asymmetric_2[idx].outputs[0].get_shape()[2], 16u);
    }
}

using namespace VPUNN;

class VPULayerCostModelTest : /* public ::testing::Test,*/ public DPULayerTest {
public:
protected:
    void SetUp() override {
        DPULayerTest::SetUp();
    }
    const unsigned int MAX_COST{10000000};  // ten million
    static constexpr unsigned int NO_ERROR_EXPECTED{VPUNN::Cycles::NO_ERROR};

    struct TestInput {
        DPULayer l1;
        VPULayerStrategy strategy;
    };

    struct TestExpectations {
        CyclesInterfaceType err_expected{NO_ERROR_EXPECTED};
        bool strict_err_check{false};
        CyclesInterfaceType min_cyc{0};
        CyclesInterfaceType max_cyc{0};
    };

    void DoRegularTest(const TestInput& t_in, const TestExpectations& t_exp, const std::string& test_case = "") {
        DPULayer l1{t_in.l1};
        VPULayerStrategy strategy{t_in.strategy};

        std::string t_header{"** Test Case: " + test_case + "\n"};
        std::cout << ">> " << t_header << std::endl;

        Logger::clear2ndlog();
        unsigned cost_cyc{};
        ASSERT_NO_THROW(cost_cyc = model_2_7.Layer(l1, strategy)) << t_header << l1 << strategy;

        const auto err_expected{t_exp.err_expected};
        if (Cycles::isErrorCode(err_expected)) {  // error code is expected
            EXPECT_TRUE(Cycles::isErrorCode(cost_cyc)) << t_header << "Expected ERROR, but value received: " << cost_cyc
                                                       << " : " << Cycles::toErrorText(cost_cyc) << std::endl
                                                       << "Expected ERROR code: " << err_expected << " : "
                                                       << Cycles::toErrorText(err_expected) << std::endl
                                                       << l1 << strategy << Logger::get2ndlog();

            if (Cycles::isErrorCode(cost_cyc) &&
                t_exp.strict_err_check) {  // in case is error code AND we want to have exact value
                EXPECT_EQ(cost_cyc, err_expected) << t_header << "ERROR code received: " << cost_cyc << " : "
                                                  << Cycles::toErrorText(cost_cyc) << std::endl
                                                  << "Expected ERROR code: " << err_expected << " : "
                                                  << Cycles::toErrorText(err_expected) << std::endl
                                                  << l1 << strategy << Logger::get2ndlog();
            }

        } else {  // regular cycle value expected
            EXPECT_FALSE(Cycles::isErrorCode(cost_cyc))
                    << t_header << "Unexpected ERROR code: " << cost_cyc << " : " << Cycles::toErrorText(cost_cyc)
                    << "\n  expected in [ " << t_exp.min_cyc << " , " << t_exp.max_cyc << " ] \n"
                    << l1 << strategy << Logger::get2ndlog();

            if (!Cycles::isErrorCode(cost_cyc)) {  // if cycle value check against other ranges
                EXPECT_GT(cost_cyc, 0u) << t_header << l1 << strategy;
                EXPECT_LT(cost_cyc, MAX_COST) << t_header << l1 << strategy;  // 1 million

                // EXPECT_THAT(cost_cyc, AllOf(Gt(24000u), Lt(24100u))) << l1 << strategy;
                EXPECT_TRUE(cost_cyc >= t_exp.min_cyc && cost_cyc <= t_exp.max_cyc)
                        << t_header << " Cost not in interval. cost: " << cost_cyc << ",  expected in [ "
                        << t_exp.min_cyc << " , " << t_exp.max_cyc << " ] \n"
                        << l1 << strategy;
            }
        }

        std::cout << t_header << " *** ERROR/Cycles code: " << cost_cyc << " : " << Cycles::toErrorText(cost_cyc)
                  << std::endl
                  << "------------------------------------------------------------------------" << std::endl;
    }

    struct TestCase {
        TestInput t_in;
        TestExpectations t_exp;
        const std::string test_case = "";
    };
    using TestsVector = std::vector<TestCase>;

    void executeTests(const TestsVector& tests) {
        int test_index = 0;
        for (const auto& t : tests) {
            std::stringstream buffer;
            buffer << test_index << " : " << t.test_case;
            const std::string test_case_info = buffer.str();

            DoRegularTest(t.t_in, t.t_exp, test_case_info);

            ++test_index;
        }
    }
};

TEST_F(VPULayerCostModelTest, LayerCostModelVPU_2_0) {
    auto layer = generate_helper_layer(16, 64);
    auto vpu20_layer_cost = model_2_0.Layer(layer, 5, 4);

    // Basic expectations
    EXPECT_GT(vpu20_layer_cost, 0u);
}

TEST_F(VPULayerCostModelTest, LayerCostModelVPU_2_7_shv) {
    auto layer = generate_helper_sw_layer(16, 64);
    auto vpu20_layer_cost = model_2_7.Layer(layer, 5, 4);

    // Basic expectations
    EXPECT_GT(vpu20_layer_cost, 0u);
}

TEST_F(VPULayerCostModelTest, ELTWISE_Concrete_Add14_VPU27) {
    /*     const VPUNN::DPUWorkload wl1{
                VPUNN::VPUDevice::VPU_2_7,
                VPUNN::Operation::ELTWISE,
                {VPUNN::VPUTensor(56, 56, 256, 1, VPUNN::DataType::UINT8)},  // input dimensions
                {VPUNN::VPUTensor(56, 56, 256, 1, VPUNN::DataType::UINT8)},  // output dimensions
                {1, 1},                                                      // kernels
                {1, 1},                                                      // strides
                {0, 0, 0, 0},                                                // padding
                VPUNN::ExecutionMode::CUBOID_16x16,                          // execution mode
                VPUNN::ActivationFunction::NONE,                             // activation
                0.0F,                                                        // act_sparsity
                0.0F,                                                        // weight_sparsity
                {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},          // input_swizzling
                {VPUNN::Swizzling::KEY_0},                                   // output_swizzling
                1,                                                           // output_write_tiles
                {0, 0, 0, 0},                                                // offsets
                VPUNN::ISIStrategy::CLUSTERING,                              // isi_strategy
                false,                                                       // weight_sparsity_enabled
        }; */

    const VPUNN::DPULayer tst_layer(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::ELTWISE,
                                    {VPUNN::VPUTensor(56, 56, 256, 1, VPUNN::DataType::UINT8)},  // input dimensions
                                    {VPUNN::VPUTensor(56, 56, 256, 1, VPUNN::DataType::UINT8)},  // output dimensions
                                    {1, 1},                                                      // kernels
                                    {1, 1},                                                      // strides
                                    {0, 0, 0, 0}                                                 // padding
    );
    const VPUNN::VPULayerStrategy tst_strategy{1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true};
    // prefetching  true = only DPU data, no DMA.

    {  // clustering
        TestInput tin{tst_layer, tst_strategy};
        tin.strategy.tiling_strategy = VPUNN::VPUTilingStrategy::NONE;

        TestExpectations texp{NO_ERROR_EXPECTED, false, 24000, 24100};

        DoRegularTest(tin, texp, "Clustering");
    }

    {  // SOH
        TestInput tin{tst_layer, tst_strategy};
        tin.strategy.tiling_strategy = VPUNN::VPUTilingStrategy::SOH;

        TestExpectations texp{NO_ERROR_EXPECTED, false, 6100, 6200U};  // before isistrategy fix:  10000U, 11000U

        DoRegularTest(tin, texp, "SOH");
    }
    {  // SOK
        TestInput tin{tst_layer, tst_strategy};
        tin.strategy.tiling_strategy = VPUNN::VPUTilingStrategy::SOK;  // SOK is not allowed for element wise

        TestExpectations texp{VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 9500, 9600};  //

        DoRegularTest(tin, texp, "SOK tentative on elementwise");
    }
}

TEST_F(VPULayerCostModelTest, CONVOLUTION_Concrete_Multiply8641_VPU27) {
    const VPUNN::DPULayer tst_layer(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
                                    {VPUNN::VPUTensor(14, 14, 512, 1, VPUNN::DataType::UINT8)},  // input dimensions
                                    {VPUNN::VPUTensor(7, 7, 512, 1, VPUNN::DataType::UINT8)},    // output dimensions
                                    {3, 3},                                                      // kernels
                                    {2, 2},                                                      // strides
                                    {1, 0, 1, 0}                                                 // padding
    );

    const VPUNN::VPULayerStrategy tst_strategy{1U,    1U,    2U /*tiles*/, VPUNN::VPUTilingStrategy::NONE,
                                               false, false, true};  // prefetching  true = only DPU data, no DMA.

    {  // SOH
        TestInput tin{tst_layer, tst_strategy};
        tin.strategy.tiling_strategy = VPUNN::VPUTilingStrategy::SOH;

        TestExpectations texp{VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 1U, 1U};

        DoRegularTest(tin, texp, "SOH with errors");
    }

    const std::vector<TestCase> tests{
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
             {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 1U, 1U},
             "CLUSTERING , no memmove, with errors"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
             {NO_ERROR_EXPECTED, false, 19974, 19974},
             "SOK , no memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
             {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 1U, 1U},
             "SOH ,no memmove, with errors"},
    };

    executeTests(tests);
}

TEST_F(VPULayerCostModelTest, AVEPOOL_Concrete_GlobalAveragePool_172_VPU27) {
    const VPUNN::DPULayer tst_layer(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::AVEPOOL,       // goes to DW Conv
                                    {VPUNN::VPUTensor(7, 7, 2048, 1, VPUNN::DataType::UINT8)},  // input dimensions
                                    {VPUNN::VPUTensor(1, 1, 2048, 1, VPUNN::DataType::UINT8)},  // output dimensions
                                    {7, 7},                                                     // kernels
                                    {1, 1},                                                     // strides
                                    {0, 0, 0, 0}                                                // padding
    );
    const VPUNN::VPULayerStrategy tst_strategy{1U,    1U,    2U /*tiles*/, VPUNN::VPUTilingStrategy::NONE,
                                               false, false, true};  // prefetching  true = only DPU data, no DMA.

    {  // SOK
        TestInput tin{tst_layer, tst_strategy};
        tin.strategy.tiling_strategy = VPUNN::VPUTilingStrategy::SOK;

        TestExpectations texp{NO_ERROR_EXPECTED, false, 7600, 7600 + 100};
        // before isistrategy fix :322000U, 323000, and before Z tiling fix 142000

        DoRegularTest(tin, texp, "SOK ok");
    }

    const std::vector<TestCase> tests{
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
             {VPUNN::Cycles::NO_ERROR, false, 51000, 51000 + 1000},
             "CLUSTERING , no memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
             {NO_ERROR_EXPECTED, false, 7600, 7600 + 100},
             "SOK , no memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
             {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 52500, 52500 + 1000},  // was 1824000
             "SOH ,no memmove"},
    };

    executeTests(tests);
}

TEST_F(VPULayerCostModelTest, AVEPOOL_Concrete_GlobalAveragePool_172_VPU27_SOH) {
    const VPUNN::DPULayer tst_layer(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::AVEPOOL,        // goes to DW Conv
                                    {VPUNN::VPUTensor(7, 15, 2048, 1, VPUNN::DataType::UINT8)},  // input dimensions
                                    {VPUNN::VPUTensor(1, 9, 2048, 1, VPUNN::DataType::UINT8)},   // output dimensions
                                    {7, 7},                                                      // kernels
                                    {1, 1},                                                      // strides
                                    {0, 0, 0, 0}                                                 // padding
    );

    const std::vector<TestCase> tests{
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
             {VPUNN::Cycles::NO_ERROR, false, 51500, 51500 + 1000},
             "CLUSTERING , no memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
             {NO_ERROR_EXPECTED, false, 12500, 12500 + 100},
             "SOK , no memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
             {VPUNN::Cycles::NO_ERROR, true, 52000, 52000 + 1000},  // SOH us possible
             "SOH ,no memmove"},
    };

    executeTests(tests);
}

TEST_F(VPULayerCostModelTest, CONVOLUTION_Concrete_Multiply8648_VPU27) {
    const VPUNN::DPULayer tst_layer(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
                                    {VPUNN::VPUTensor(7, 7, 512, 1, VPUNN::DataType::UINT8)},   // input dimensions
                                    {VPUNN::VPUTensor(7, 7, 2048, 1, VPUNN::DataType::UINT8)},  // output dimensions
                                    {1, 1},                                                     // kernels
                                    {1, 1},                                                     // strides
                                    {0, 0, 0, 0}                                                // padding
    );
    const VPUNN::VPULayerStrategy tst_strategy{1U,    1U,    2U /*tiles*/, VPUNN::VPUTilingStrategy::NONE,
                                               false, false, true};  // prefetching  true = only DPU data, no DMA.

    {  // SOK
        TestInput tin{tst_layer, tst_strategy};
        tin.strategy.tiling_strategy = VPUNN::VPUTilingStrategy::SOK;

        TestExpectations texp{NO_ERROR_EXPECTED, false, 9100, 9200};  // before isistrategy fix :17000U, 18000

        DoRegularTest(tin, texp, "SOK convolution");
    }

    const std::vector<TestCase> tests{
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
             {NO_ERROR_EXPECTED, false, 34000U, 34500U},
             "CLUSTERING , no memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
             {NO_ERROR_EXPECTED, false, 9100, 9200},  // before isistrategy fix :17000U, 18000
             "SOK , no memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
             {NO_ERROR_EXPECTED, false, 11700, 11800},  // before isistrategy fix :19500U, 20000
             "SOK , no prefetch"},
    };

    executeTests(tests);
}

TEST_F(VPULayerCostModelTest, MAXPOOL_avgpoolBased_172_VPU27) {
    const VPUNN::DPULayer tst_layer(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::MAXPOOL,
                                    {VPUNN::VPUTensor(7, 7, 2048, 1, VPUNN::DataType::UINT8)},  // input dimensions
                                    {VPUNN::VPUTensor(1, 1, 2048, 1, VPUNN::DataType::UINT8)},  // output dimensions
                                    {7, 7},                                                     // kernels
                                    {1, 1},                                                     // strides
                                    {0, 0, 0, 0}                                                // padding
    );

    const std::vector<TestCase> tests{
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
             {NO_ERROR_EXPECTED, false, 46000, 46000 + 1000},
             "CLUSTERING , no mem"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
             {NO_ERROR_EXPECTED, false, 6900, 6900 + 100},
             "SOK , no mem"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
             {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, false, 45000,
              45000 + 1000},  // 7 cannot be split, must be 14
             "SOH ,no mem"},
    };

    executeTests(tests);
}

TEST_F(VPULayerCostModelTest, MAXPOOL_avgpoolBased_172_VPU27_SOH) {  // SOH Split possible at limit
    const VPUNN::DPULayer tst_layer(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::MAXPOOL,
                                    {VPUNN::VPUTensor(7, 14, 2048, 1, VPUNN::DataType::UINT8)},  // input dimensions
                                    {VPUNN::VPUTensor(1, 8, 2048, 1, VPUNN::DataType::UINT8)},   // output dimensions
                                    {7, 7},                                                      // kernels
                                    {1, 1},                                                      // strides
                                    {0, 0, 0, 0}                                                 // padding
    );

    const std::vector<TestCase> tests{
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
             {NO_ERROR_EXPECTED, false, 48000, 48000 + 1000},
             "CLUSTERING , no mem"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
             {NO_ERROR_EXPECTED, false, 10100, 10100 + 100},
             "SOK , no mem"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
             {NO_ERROR_EXPECTED, false, 45000, 45000 + 1000},
             "SOH ,output H ?"},
    };

    executeTests(tests);
}

TEST_F(VPULayerCostModelTest, AVEPOOL_Concrete_GlobalAveragePool_172_BAD_CHANNELS_VPU27) {
    const VPUNN::DPULayer tst_layer(
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::AVEPOOL,  // replace by DW conv , only outputs ch of 16-32-64 in a workload alowwed
            {VPUNN::VPUTensor(7, 7, 2047, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(1, 1, 2047, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {7, 7},                                                     // kernels
            {1, 1},                                                     // strides
            {0, 0, 0, 0}                                                // padding
    );
    // 2047 is not MOD16, split by 2 also is not MOD 16, so no intra tile z split is possible

    const std::vector<TestCase> tests{
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
             {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 2469000, 2469000 + 1000},
             "CLUSTERING no MOD16 , no memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
             {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 142000, 142000 + 1000},
             "SOK no MOD16, no memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
             {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 1821000, 1821000 + 1000},
             "SOH no MOD16,no memmove"},
    };

    executeTests(tests);
}

TEST_F(VPULayerCostModelTest, AVEPOOL_Concrete_GlobalAveragePool_172_MOD16_NOMOD32_CH_VPU27) {
    const VPUNN::DPULayer tst_layer(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::AVEPOOL,
            {VPUNN::VPUTensor(7, 7, 2048 - 16, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(1, 1, 2048 - 16, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {7, 7},                                                          // kernels
            {1, 1},                                                          // strides
            {0, 0, 0, 0}                                                     // padding
    );
    const VPUNN::VPULayerStrategy tst_strategy{1U,    1U,    2U /*tiles*/, VPUNN::VPUTilingStrategy::NONE,
                                               false, false, true};  // prefetching  true = only DPU data, no DMA.

    {  // CLUSTERING
        TestInput tin{tst_layer, tst_strategy};
        tin.strategy.tiling_strategy = VPUNN::VPUTilingStrategy::NONE;

        // Very large because 2032 is  mod 16, but 31X64 +1X48 is not accepted, 63x32+1X16 is not reached (limit 50)
        // and we cannot reach a split like: 31x64 = 1984  , 1x32=32,  1x16=16  =>2032 split in 33 workloads
        // the 1x2032 is not accepted because ch not 16/32/64
        TestExpectations texp{VPUNN::Cycles::ERROR_TILE_OUTPUT, true, 2408000, 2408000 + 1000};

        DoRegularTest(tin, texp, "CLUST not valid 1 workload");
    }

    {  // SOK
        TestInput tin{tst_layer, tst_strategy};
        tin.strategy.tiling_strategy = VPUNN::VPUTilingStrategy::SOK;

        // 2032 split to tile like 1024 and 1008, both div by 16
        // 1024 = 16x64
        // 1008 :   15x64 , 1x48 : not allowed;   31x32, 1x16  reached,    but  15x64 , 1x32, 1x16 : not possible
        // but best.   due to not possible one we get 14000 instead of 8000
        TestExpectations texp{NO_ERROR_EXPECTED, false, 14900, 14900 + 100};

        DoRegularTest(tin, texp, "SOK not optimum: 15x64 , 1x32, 1x16  = 1008 ");
    }

    const std::vector<TestCase> tests{
            //         {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
            //          {NO_ERROR_EXPECTED, false, 2473000, 2474000},
            //          "CLUSTERING , no memmove"},
            //         {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
            //          {NO_ERROR_EXPECTED, false, 142000, 143000},
            //          "SOK , no memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
             {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 1824000, 1825000},
             "SOH ,no memmove"},
    };

    executeTests(tests);
}

TEST_F(VPULayerCostModelTest, AVEPOOL_Concrete_GlobalAveragePool_172_MOD16_NOMOD32_CH_VPU27_SOH) {
    const VPUNN::DPULayer tst_layer(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::AVEPOOL,
            {VPUNN::VPUTensor(7, 14, 2048 - 16, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(1, 8, 2048 - 16, 1, VPUNN::DataType::UINT8)},   // output dimensions
            {7, 7},                                                           // kernels
            {1, 1},                                                           // strides
            {0, 0, 0, 0}                                                      // padding
    );
    const VPUNN::VPULayerStrategy tst_strategy{1U,    1U,    2U /*tiles*/, VPUNN::VPUTilingStrategy::NONE,
                                               false, false, true};  // prefetching  true = only DPU data, no DMA.

    {  // CLUSTERING
        TestInput tin{tst_layer, tst_strategy};
        tin.strategy.tiling_strategy = VPUNN::VPUTilingStrategy::NONE;

        // Very large because 2032 is  mod 16, but 31X64 +1X48 is not accepted, 63x32+1X16 is not reached (limit 50)
        // and we cannot reach a split like: 31x64 = 1984  , 1x32=32,  1x16=16  =>2032 split in 33 workloads
        // the 1x2032 is not accepted because ch not 16/32/64
        TestExpectations texp{VPUNN::Cycles::ERROR_TILE_OUTPUT, true, 2408000, 2408000 + 1000};

        DoRegularTest(tin, texp, "CLUST not valid 1 workload");
    }

    {  // SOK
        TestInput tin{tst_layer, tst_strategy};
        tin.strategy.tiling_strategy = VPUNN::VPUTilingStrategy::SOK;

        // 2032 split to tile like 1024 and 1008, both div by 16
        // 1024 = 16x64
        // 1008 :   15x64 , 1x48 : not allowed;   31x32, 1x16  reached,    but  15x64 , 1x32, 1x16 : not possible
        // but best.   due to not possible one we get 14000 instead of 8000
        TestExpectations texp{NO_ERROR_EXPECTED, false, 22000, 22000 + 1000};

        DoRegularTest(tin, texp, "SOK not optimum: 15x64 , 1x32, 1x16  = 1008 ");
    }

    const std::vector<TestCase> tests{
            //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
            // {NO_ERROR_EXPECTED, false, 2473000, 2474000},
            // "CLUSTERING , no memmove"},
            //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
            // {NO_ERROR_EXPECTED, false, 142000, 143000},
            // "SOK , no memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
             {VPUNN::Cycles::ERROR_TILE_OUTPUT, true, 1824000,
              1825000},  // but is 31000 id allowed fro split to 64 workloads
             "SOH ,no memmove"},
    };

    executeTests(tests);
}
TEST_F(VPULayerCostModelTest, CONVOLUTION_Concrete_Multiply6326_FLOAT_FLOAT_VPU27) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(60, 7, 512, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(60, 6, 512, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            {3, 3},                                                       // kernels
            {1, 1},                                                       // strides
            {1, 0, 1, 1}                                                  // padding
    );
    {
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = true;
        tst_layer.weight_sparsity = 0.723566F;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                 "CLUSTERING ,flt , sparse "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, false, 128000, 128000 + 1000},
                 "SOK , no memmove, sparse"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 327500, 327500 + 1000},
                 "SOH , no memmove, sparse"},
        };

        executeTests(tests);
    }
}
TEST_F(VPULayerCostModelTest, CONVOLUTION_Concrete_Multiply6326_FLOAT_INT_VPU27) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(60, 7, 512, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(60, 6, 512, 1, VPUNN::DataType::UINT8)},    // output dimensions
            {3, 3},                                                       // kernels
            {1, 1},                                                       // strides
            {1, 0, 1, 1}                                                  // padding
    );
    {
        const auto tst_layer = tst_layer_ref;
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                 "CLUSTERING , flt, dense "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                 "SOK , no memmove, dense"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                 "SOH , no memmove, dense"},
        };

        executeTests(tests);
    }
    {
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = true;
        tst_layer.weight_sparsity = 0.723566F;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                 "CLUSTERING ,flt , sparse "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, false, 128000, 128000 + 1000},
                 "SOK , no memmove, sparse"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, false, 329000, 329000 + 1000},
                 "SOH , no memmove, sparse"},
        };

        executeTests(tests);
    }
}
TEST_F(VPULayerCostModelTest, CONVOLUTION_Concrete_Multiply6326_INT_FLOAT_VPU27) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(60, 7, 512, 1, VPUNN::DataType::UINT8)},    // input dimensions
            {VPUNN::VPUTensor(60, 6, 512, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            {3, 3},                                                       // kernels
            {1, 1},                                                       // strides
            {1, 0, 1, 1}                                                  // padding
    );

    {
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = true;
        tst_layer.weight_sparsity = 0.723566F;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, false, 266000, 266000 + 1000},
                 "CLUSTERING ,flt , sparse "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, false, 68000, 68000 + 1000},
                 "SOK , no memmove, sparse"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, false, 168000, 168000 + 1000},
                 "SOH , no memmove, sparse"},
        };

        executeTests(tests);
    }
}
TEST_F(VPULayerCostModelTest, CONVOLUTION_Concrete_Multiply6326_INT_INT_VPU27) {
    const VPUNN::DPULayer tst_layer_ref(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
                                        {VPUNN::VPUTensor(60, 7, 512, 1, VPUNN::DataType::UINT8)},  // input dimensions
                                        {VPUNN::VPUTensor(60, 6, 512, 1, VPUNN::DataType::UINT8)},  // output dimensions
                                        {3, 3},                                                     // kernels
                                        {1, 1},                                                     // strides
                                        {1, 0, 1, 1}                                                // padding
    );
    {
        const auto tst_layer = tst_layer_ref;
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                 "CLUSTERING , int "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {NO_ERROR_EXPECTED, false, 139000, 139000 + 1000},
                 "SOK , no memmove"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                 "SOH , no memmove"},
        };

        executeTests(tests);
    }

    {
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = true;
        tst_layer.weight_sparsity = 0.723566F;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, false, 266000, 266000 + 1000},
                 "CLUSTERING , int "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {NO_ERROR_EXPECTED, false, 68000, 68000 + 1000},
                 "SOK , no memmove"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, false, 167000, 167000 + 1000},
                 "SOH , no memmove"},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, CONVOLUTION_Multiply_6326_FI_VPU27) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(60, 8, 512, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(60, 6, 512, 1, VPUNN::DataType::UINT8)},    // output dimensions
            {3, 3},                                                       // kernels
            {1, 1},                                                       // strides
            {0, 0, 1, 1}                                                  // padding
    );
    //{
    //    const auto tst_layer = tst_layer_ref;
    //    const std::vector<TestCase> tests{
    //            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
    //             {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
    //             "CLUSTERING , flt, dense "},
    //            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
    //             {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
    //             "SOK , no memmove, dense"},
    //            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
    //             {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
    //             "SOH , no memmove, dense"},
    //    };

    //    executeTests(tests);
    //}
    {
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = true;
        tst_layer.weight_sparsity = 0.723158f;

        const std::vector<TestCase> tests{
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                // {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                // "CLUSTERING ,flt , sparse "},
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                // {VPUNN::Cycles::NO_ERROR, false, 132000, 132000 + 1000},//132831
                // "SOK , no memmove, sparse"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 330000, 330000 + 1000},  // due to cmx overhead
                 "SOH , no memmove, sparse"},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, Fused_234_3xELEMENTWISE) {
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
        auto tst_layer = l1F_ref;
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                 t + "CLU 0m "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 0, 0 + 0},
                 t + "SOK ,0m"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 24000, 24000 + 1000},  // 24324
                 t + "SOH ,0m"},
        };
        executeTests(tests);
    }

    {  // UINT8 all, Layer 1 changed
        std::string t{"All UINT8 @ Layer 1 "};
        auto tst_layer = l1Int_ref;
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 43000, 43000 + 1000},  // 43169
                 t + "CLU 0m "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 0, 0 + 0},
                 t + "SOK ,0m"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 12000, 12000 + 1000},  // 12380
                 t + "SOH ,0m"},
        };
        executeTests(tests);
    }

    {  // FLOAT to INT all, Layer 1 changed
        std::string t{"Reversed F toI @ Layer 1  mixed"};
        auto tst_layer = l1FI_ref;
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0 + 0},  //
                 t + "CLU 0m "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 0, 0 + 0},
                 t + "SOK ,0m"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 25000, 25000 + 1000},  // 25140
                 t + "SOH ,0m"},
        };
        executeTests(tests);
    }

    {  // L1&2 int -float. Has to be IMPROVED
        std::string t{"Original Layer 1, mixed "};
        auto tst_layer = l1and2_ref;
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 44000, 44000 + 1000},  // 44608
                 t + "CLU 0m "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 0, 0 + 0},
                 t + "SOK ,0m"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 12000, 12000 + 1000},  // 12491
                 t + "SOH ,0m"},
        };
        executeTests(tests);
    }

    {  // L3  int -float. Has to be IMPROVED
        std::string t{"Original Layer 3, mixed "};
        auto tst_layer = l3_ref;
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 44000, 44000 + 1000},  // 44389
                 t + "CLU 0m "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 0, 0 + 0},
                 t + "SOK ,0m"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 12000, 12000 + 1000},  // 12220
                 t + "SOH ,0m"},
        };
        executeTests(tests);
    }
    // ASSERT_TRUE(false);
}

TEST_F(VPULayerCostModelTest, AVEPOOL_Concrete_GlobalAveragePool_172_MaxWorkloadSPlitAndDetails) {
    const VPUNN::DPULayer tst_layer(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::AVEPOOL,
            {VPUNN::VPUTensor(7, 7, 2048 - 16, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(1, 1, 2048 - 16, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {7, 7},                                                          // kernels
            {1, 1},                                                          // strides
            {0, 0, 0, 0}                                                     // padding
    );
    const VPUNN::VPULayerStrategy tst_strategy{1U,    1U,    2U /*tiles*/, VPUNN::VPUTilingStrategy::NONE,
                                               false, false, true};  // prefetching  true = only DPU data, no DMA.

    {  // CLUSTERING
        TestInput tin{tst_layer, tst_strategy};
        tin.strategy.tiling_strategy = VPUNN::VPUTilingStrategy::NONE;

        const auto maxWSplit = model_2_7.get_maxWorkloadsPerIntraTileSplit();

        EXPECT_EQ(50U, maxWSplit) << "max workloads split must be default";

        // Very large because 2032 is  mod 16, but 31X64 +1X48 is not accepted, 63x32+1X16 is not reached (limit 50)
        // and we cannot reach a split like: 31x64 = 1984  , 1x32=32,  1x16=16  =>2032 split in 33 workloads
        // the 1x2032 is not accepted because ch not 16/32/64
        TestExpectations texp1{VPUNN::Cycles::ERROR_TILE_OUTPUT, true, 2408000, 2408000 + 1000};

        DoRegularTest(tin, texp1, "CLUST not valid , cannot split");

        model_2_7.set_maxWorkloadsPerIntraTileSplit(64U);
        EXPECT_EQ(model_2_7.get_maxWorkloadsPerIntraTileSplit(), 64U) << "max workloads split must be set";

        // 63x32+1X16 is reached (limit 64)
        TestExpectations texp2{VPUNN::Cycles::NO_ERROR, true, 31000, 31000 + 1000};

        DoRegularTest(tin, texp2, "CLUST must be split to 64");
    }
    {
        TestInput tin{tst_layer, tst_strategy};
        tin.strategy.tiling_strategy = VPUNN::VPUTilingStrategy::NONE;
        LayerSplitInfo splitInfo{};

        auto t = model_2_7.Layer(tin.l1, tin.strategy.tiling_strategy, 1U, 2, false, false, true, splitInfo);

        EXPECT_FALSE(Cycles::isErrorCode(t));

        ASSERT_EQ(splitInfo.size(), 2U) << "Must be 2 tiles!";
        EXPECT_EQ(splitInfo[0].best_intra_tile_split.second.size(), 64U) << "Tile 1 Must be 64 workloads!";
        EXPECT_EQ(splitInfo[1].best_intra_tile_split.second.size(), 64U) << "Tile 2 Must be 64 workloads!";
        EXPECT_EQ(splitInfo[0].best_intra_tile_split.first, splitInfo[1].best_intra_tile_split.first)
                << "Tiles must be equal in cycles";
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
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = true;
        tst_layer.weight_sparsity = 0.950016f;

        const std::vector<TestCase> tests{
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                // {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                // "CLUSTERING ,flt , sparse "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 178000, 178000 + 1000},  // 132831 // Out chnannels %32
                 "SOK , no memmove, sparse"},
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
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
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = true;
        tst_layer.weight_sparsity = 0.880725f;

        const std::vector<TestCase> tests{
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                // {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                // "CLUSTERING ,flt , sparse "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 153000, 153000 + 1000},  //
                 "SOK , no memmove, sparse"},
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
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
        auto tst_layer = tst_layer_ref2;
        tst_layer.weight_sparsity_enabled = true;
        tst_layer.weight_sparsity = 0.880725f;

        const std::vector<TestCase> tests{
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                // {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                // "CLUSTERING ,flt , sparse "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 154000, 154000 + 1000},  //
                 "SOK , no memmove, sparse , padding 00"},
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
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
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = true;
        tst_layer.weight_sparsity = 0.36438f;

        const std::vector<TestCase> tests{
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                // {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                // "CLUSTERING ,flt , sparse "},
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                // {VPUNN::Cycles::NO_ERROR, true, 132000, 132000 + 1000},  //
                // "SOK , no memmove, sparse"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 182000, 182000 + 1000},  //
                 "SOH , no memmove, sparse"},
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
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = true;
        tst_layer.weight_sparsity = 0.0664062f;

        const std::vector<TestCase> tests{
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                // {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                // "CLUSTERING ,flt , sparse "},
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                // {VPUNN::Cycles::NO_ERROR, true, 132000, 132000 + 1000},  //
                // "SOK , no memmove, sparse"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 122000, 122000 + 1000},  //
                 "SOH , no memmove, sparse"},
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
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 29900, 29900 + 1000},
                 "CLUSTERING , ,  "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 13300, 13300 + 1000},  //
                 "SOK , no memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 7900, 7900 + 1000},  //
                 "SOH , no memmove, "},
        };

        executeTests(tests);
    }
}

}  // namespace VPUNN_unit_tests
