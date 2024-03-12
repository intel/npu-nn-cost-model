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
using namespace VPUNN;

class VPULayerCostModelTest : public ::testing::Test {
public:
protected:
    VPUNN::VPULayerCostModel model_invalid{""};

    VPUNN::VPULayerCostModel model_2_0{VPU_2_0_MODEL_PATH};
    VPUNN::VPULayerCostModel model_2_7{VPU_2_7_MODEL_PATH};

    void SetUp() override {
        VPUNN::Logger::clear2ndlog();
        VPUNN::Logger::activate2ndlog();
    }
    void TearDown() override {
        VPUNN::Logger::clear2ndlog();
        VPUNN::Logger::deactivate2ndlog();
    }

    VPULayerCostModel& getModel(const VPUDevice device) {
        switch (device) {
        case VPUDevice::VPU_2_0:
        case VPUDevice::VPU_2_1:
            return model_2_0;
        case VPUDevice::VPU_2_7:
            return model_2_7;
        default:
            return model_invalid;
            break;
        }
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
        ASSERT_NO_THROW(cost_cyc = getModel(l1.device).Layer(l1, strategy)) << t_header << l1 << strategy;

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
TEST_F(VPULayerCostModelTest, LayerLoadModels) {
    EXPECT_EQ(model_2_7.nn_initialized(), true);
    EXPECT_EQ(model_2_0.nn_initialized(), true);
}

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

        TestExpectations texp{NO_ERROR_EXPECTED, false, 23000, 24100};

        DoRegularTest(tin, texp, "Clustering");
    }

    {  // SOH
        TestInput tin{tst_layer, tst_strategy};
        tin.strategy.tiling_strategy = VPUNN::VPUTilingStrategy::SOH;

        TestExpectations texp{NO_ERROR_EXPECTED, false, 10000, 10000 + 1000U};

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
             {NO_ERROR_EXPECTED, true, 37000, 38000},
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

        TestExpectations texp{NO_ERROR_EXPECTED, false, 14000, 14000 + 1000};
        // before isistrategy fix :322000U, 323000, and before Z tiling fix 142000

        DoRegularTest(tin, texp, "SOK ok");
    }

    const std::vector<TestCase> tests{
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
             {VPUNN::Cycles::NO_ERROR, false, 50000, 50000 + 2000},
             "CLUSTERING , no memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
             {NO_ERROR_EXPECTED, false, 14000, 14000 + 1000},
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
             {VPUNN::Cycles::NO_ERROR, false, 51500, 51500 + 2000},
             "CLUSTERING , no memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
             {NO_ERROR_EXPECTED, false, 20000, 20000 + 1000},
             "SOK , no memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
             {VPUNN::Cycles::NO_ERROR, true, 51000, 51000 + 2000},  // SOH us possible
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

        TestExpectations texp{NO_ERROR_EXPECTED, false, 18000, 18000 + 1000};  // before isistrategy fix :17000U, 18000

        DoRegularTest(tin, texp, "SOK convolution");
    }

    const std::vector<TestCase> tests{
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
             {NO_ERROR_EXPECTED, false, 34000, 34000 + 2000},
             "CLUSTERING , no memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
             {NO_ERROR_EXPECTED, false, 88500, 88500 + 2000},  // fetching big data
             "CLUSTERING ,with fetch required"},

            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
             {NO_ERROR_EXPECTED, false, 18000, 18000 + 1000},  // before isistrategy fix :17000U, 18000
             "SOK , no memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
             {NO_ERROR_EXPECTED, false, 45000, 45000 + 1000},  // fetching big data
             "SOK , with fetch required"},
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
             {NO_ERROR_EXPECTED, false, 52000, 52000 + 1000},  // before was 46k, why big change?
             "CLUSTERING , no mem"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
             {NO_ERROR_EXPECTED, false, 13000, 13000 + 1000},
             "SOK , no mem"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
             {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 45000,
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
             {NO_ERROR_EXPECTED, false, 53000, 54000},
             "CLUSTERING , no mem"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
             {NO_ERROR_EXPECTED, false, 21000, 22000},
             "SOK , no mem"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
             {NO_ERROR_EXPECTED, false, 51000, 52000},
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
        TestExpectations texp{NO_ERROR_EXPECTED, false, 14000, 15000};

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
        TestExpectations texp{NO_ERROR_EXPECTED, false, 20000, 21000 + 3000};  // big range

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
                 {VPUNN::Cycles::NO_ERROR, false, 260000, 260000 + 10000},
                 "SOK , no memmove, sparse"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 327500, 327500 + 1000},  // was SOH no overlap  : 327500
                 "SOH , no memmove, sparse"},
        };

        executeTests(tests);
    }
}
TEST_F(VPULayerCostModelTest, CONVOLUTION_Concrete_Multiply6326_FLOAT_INT_VPU27_EISW_76882) {
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
                 {VPUNN::Cycles::NO_ERROR, false, 260000, 260000 + 10000},
                 "SOK , no memmove, sparse"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, false, 350000, 350000 + 10000},  // MAIN TEST CASE
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
                 {VPUNN::Cycles::NO_ERROR, false, 270000, 270000 + 10000},  // why changed so much from 266K
                 "CLUSTERING ,flt , sparse "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, false, 130000, 130000 + 10000},
                 "SOK , no memmove, sparse"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, false, 170000, 170000 + 10000},
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
                 {NO_ERROR_EXPECTED, false, 270000, 270000 + 10000},
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
                 {VPUNN::Cycles::NO_ERROR, false, 270000, 270000 + 10000},
                 "CLUSTERING , int "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {NO_ERROR_EXPECTED, false, 130000, 130000 + 10000},
                 "SOK , no memmove"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, false, 170000, 170000 + 10000},
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
                 {VPUNN::Cycles::NO_ERROR, true, 340000, 340000 + 15000},  //
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
                 {VPUNN::Cycles::NO_ERROR, true, 30000, 35000},
                 t + "SOH ,0m"},
        };
        executeTests(tests);
    }

    {  // UINT8 all, Layer 1 changed
        std::string t{"All UINT8 @ Layer 1 "};
        auto tst_layer = l1Int_ref;
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 37000, 37000 + 3000},  // 43169 before
                 t + "CLU 0m "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 0, 0 + 0},
                 t + "SOK ,0m"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 21500, 21500 + 2000},  // 23kbefore
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
                 {VPUNN::Cycles::NO_ERROR, true, 28000, 28000 + 1000},  // 25140
                 t + "SOH ,0m"},
        };
        executeTests(tests);
    }

    {  // L1&2 int -float. Has to be IMPROVED
        std::string t{"Original Layer 1, mixed "};
        auto tst_layer = l1and2_ref;
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 40000, 40000 + 5000},  // 44608
                 t + "CLU 0m "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 0, 0 + 0},
                 t + "SOK ,0m"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 23500, 23500 + 1000},  // 12491
                 t + "SOH ,0m"},
        };
        executeTests(tests);
    }

    {  // L3  int -float. Has to be IMPROVED
        std::string t{"Original Layer 3, mixed "};
        auto tst_layer = l3_ref;
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 40000, 40000 + 5000},  // 44389
                 t + "CLU 0m "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 0, 0 + 0},
                 t + "SOK ,0m"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 23000, 23000 + 2000},  // 12220
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
        TestExpectations texp2{VPUNN::Cycles::NO_ERROR, true, 30000, 30000 + 2000};

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
                 {VPUNN::Cycles::NO_ERROR, true, 370000, 370000 + 120000},  // 132831 // Out chnannels %32
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
                 {VPUNN::Cycles::NO_ERROR, true, 280000, 280000 + 15000},  //
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
                 {VPUNN::Cycles::NO_ERROR, true, 280000, 280000 + 15000},  //
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
                 {VPUNN::Cycles::NO_ERROR, true, 290000, 290000 + 10000},  //
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
                 {VPUNN::Cycles::NO_ERROR, true, 196000, 196000 + 10000},  //
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
                 {VPUNN::Cycles::NO_ERROR, true, 28000, 28000 + 2000},
                 "CLUSTERING , ,  "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 23000, 23000 + 2000},  //
                 "SOK , no memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 14000, 14000 + 2000},  //
                 "SOH , no memmove, "},
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
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 4000, 4000 + 1000},
                 "CLUSTERING , ,  "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 3100, 3100 + 1000},  //
                 "SOK , no memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 4000, 4000 + 1000},  //
                 "SOH , no memmove, "},
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
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 5400, 5400 + 1000},
                 "CLUSTERING , ,  "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 2000, 2000 + 1000},  //
                 "SOK , no memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
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
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
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
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 18000, 20000},
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
    {
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 92000, 92000 + 3000},
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
    {
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 92000, 92000 + 3000},
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
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
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
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 57700, 57700 + 1000},
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
    {
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 100000, 100000 + 10000},
                 "SOH , no memmove, "},
        };

        executeTests(tests);
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
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
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
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 13000, 13000 + 2000},
                 "SOK , no memmove, "},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest,
       face_detection_adas_0001_caffe_dense_IRv11_FP16_INT8_THROUGHPUT_NCHW_NCHW_U8_FP16_API20_MLIR_Set_1) {
    const VPUNN::DPULayer tst_layer_ref(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::DW_CONVOLUTION,
                                        {VPUNN::VPUTensor(6, 3, 128, 1, VPUNN::DataType::UINT8)},  // input dimensions
                                        {VPUNN::VPUTensor(3, 2, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
                                        {3, 3},                                                    // kernels
                                        {2, 2},                                                    // strides
                                        {1, 1, 1, 0}                                               // padding
    );
    {
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 500, 500 + 1000},
                 "SOH , no memmove, "},
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
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 57000, 57000 + 3000},
                 "SOH , no memmove, "},
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
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 57000, 57000 + 3000},
                 "SOH , no memmove, "},
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
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 114000, 114000 + 15000},
                 "SOH , no memmove, "},
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
    {
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 3000, 3000 + 1000},
                 "SOH , no memmove, "},
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
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 7000, 7000 + 2000},
                 "SOH , no memmove, "},
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
    {
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 4000, 4000 + 1000},
                 "SOH , no memmove, "},
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
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 1, VPUNN::Cycles::START_ERROR_RANGE - 1},
                 "SOH , no memmove, "},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, CONVOLUTION_Concrete_Multiply8641_VPU27_Prefetch) {
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
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
             {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 1U, 1U},
             "CLUSTERING , no memmove, with errors"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
             {NO_ERROR_EXPECTED, true, 94000, 97000},
             "SOK , no memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, false}},
             {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 1U, 1U},
             "SOH ,no memmove, with errors"},
    };

    executeTests(tests);
}

TEST_F(VPULayerCostModelTest, CONVOLUTION_Concrete_Multiply6326_FLOAT_FLOAT_VPU27_Prefetch) {
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
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                 "CLUSTERING ,flt , sparse "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, false, 300000, 300000 + 10000},
                 "SOK , no memmove, sparse"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, false}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 327500, 327500 + 1000},  // was SOH no overlap  : 327500
                 "SOH , no memmove, sparse"},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, ConvolutionPrefetchTest) {
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
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, false, 315000,
                  315000 + 10000},  // without prefetching:270000, 270000 + 10000
                 "CLUSTERING ,flt , sparse "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, false, 155000,
                  155000 + 10000},  // without prefetching 130000, 130000 + 10000
                 "SOK , no memmove, sparse"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, false, 220000,
                  220000 + 10000},  // without prefetching 170000, 170000 + 10000
                 "SOH , no memmove, sparse"},
        };

        executeTests(tests);
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
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 5500, 5500 + 1000},
                 "CLUSTERING , ,  "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 4500, 4500 + 1000},  //
                 "SOK , no memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 5500, 5500 + 1000},  //
                 "SOH , no memmove, "},
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
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 28000, 28000 + 2000},  // without prefetch 28000, 28000 + 2000
                 "CLUSTERING , ,  "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 24000, 24000 + 2000},  // without prefetch 23000, 23000 + 2000
                 "SOK , no memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 15000, 15000 + 2000},  // without prefetch 14000, 14000 + 2000
                 "SOH , no memmove, "},
        };

        executeTests(tests);
    }
}
TEST_F(VPULayerCostModelTest, Fused_234_3xELEMENTWISE_Prefetch) {
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
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                 t + "CLU 0m "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 0, 0 + 0},
                 t + "SOK ,0m"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 77000, 77000 + 5000},
                 t + "SOH ,0m"},
        };
        executeTests(tests);
    }

    {  // UINT8 all, Layer 1 changed
        std::string t{"All UINT8 @ Layer 1 "};
        auto tst_layer = l1Int_ref;
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 83000, 83000 + 5000},
                 t + "CLU 0m "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 0, 0 + 0},
                 t + "SOK ,0m"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 45000, 45000 + 5000},
                 t + "SOH ,0m"},
        };
        executeTests(tests);
    }

    {  // FLOAT to INT all, Layer 1 changed
        std::string t{"Reversed F toI @ Layer 1  mixed"};
        auto tst_layer = l1FI_ref;
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0 + 0},  //
                 t + "CLU 0m "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 0, 0 + 0},
                 t + "SOK ,0m"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 73000, 73000 + 5000},
                 t + "SOH ,0m"},
        };
        executeTests(tests);
    }

    {  // L1&2 int -float. Has to be IMPROVED
        std::string t{"Original Layer 1, mixed "};
        auto tst_layer = l1and2_ref;
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 85000, 85000 + 5000},
                 t + "CLU 0m "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 0, 0 + 0},
                 t + "SOK ,0m"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 47000, 47000 + 3000},
                 t + "SOH ,0m"},
        };
        executeTests(tests);
    }

    {  // L3  int -float. Has to be IMPROVED
        std::string t{"Original Layer 3, mixed "};
        auto tst_layer = l3_ref;
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 85000, 85000 + 5000},
                 t + "CLU 0m "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 0, 0 + 0},
                 t + "SOK ,0m"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 46000, 46000 + 3000},
                 t + "SOH ,0m"},
        };
        executeTests(tests);
    }
    // ASSERT_TRUE(false);
}

TEST_F(VPULayerCostModelTest, ELTWISEPrefetchTest) {
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

        TestExpectations texp{NO_ERROR_EXPECTED, false, 62000, 62000 + 2000};  // without prefetching:  24000 , 25100

        DoRegularTest(tin, texp, "Clustering");
    }

    {  // SOH
        TestInput tin{tst_layer, tst_strategy};
        tin.strategy.tiling_strategy = VPUNN::VPUTilingStrategy::SOH;

        TestExpectations texp{NO_ERROR_EXPECTED, false, 30000,
                              30000 + 2000U};  // without prefetching: 11000, 11000 + 1000U

        DoRegularTest(tin, texp, "SOH");
    }
    {  // SOK
        TestInput tin{tst_layer, tst_strategy};
        tin.strategy.tiling_strategy = VPUNN::VPUTilingStrategy::SOK;  // SOK is not allowed for element wise

        TestExpectations texp{VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 9500, 9600};

        DoRegularTest(tin, texp, "SOK tentative on elementwise");
    }
}
TEST_F(VPULayerCostModelTest, MAXPOOLPrefetchTest) {
    const VPUNN::DPULayer tst_layer(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::MAXPOOL,
                                    {VPUNN::VPUTensor(7, 7, 2048, 1, VPUNN::DataType::UINT8)},  // input dimensions
                                    {VPUNN::VPUTensor(1, 1, 2048, 1, VPUNN::DataType::UINT8)},  // output dimensions
                                    {7, 7},                                                     // kernels
                                    {1, 1},                                                     // strides
                                    {0, 0, 0, 0}                                                // padding
    );

    const std::vector<TestCase> tests{
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
             {NO_ERROR_EXPECTED, false, 55000, 55000 + 1000},  // without prefetch 52000, 52000 + 1000
             "CLUSTERING , no mem"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
             {NO_ERROR_EXPECTED, false, 15000, 15000 + 1000},  // without prefetch 13000, 13000 + 1000
             "SOK , no mem"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, false}},
             {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 45000,
              45000 + 1000},  // 7 cannot be split, must be 14
             "SOH ,no mem"},
    };

    executeTests(tests);
}

TEST_F(VPULayerCostModelTest, AVEPOOLPrefetchTest) {
    const VPUNN::DPULayer tst_layer(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::AVEPOOL,        // goes to DW Conv
                                    {VPUNN::VPUTensor(7, 15, 2048, 1, VPUNN::DataType::UINT8)},  // input dimensions
                                    {VPUNN::VPUTensor(1, 9, 2048, 1, VPUNN::DataType::UINT8)},   // output dimensions
                                    {7, 7},                                                      // kernels
                                    {1, 1},                                                      // strides
                                    {0, 0, 0, 0}                                                 // padding
    );

    const std::vector<TestCase> tests{
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
             {VPUNN::Cycles::NO_ERROR, false, 61500, 61500 + 2000},
             "CLUSTERING , no memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
             {NO_ERROR_EXPECTED, false, 25000, 25000 + 1000},
             "SOK , no memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, false}},
             {VPUNN::Cycles::NO_ERROR, true, 60000, 60000 + 2000},  // SOH us possible
             "SOH ,no memmove"},
    };

    executeTests(tests);
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
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 60000, 60000 + 3000},
                 "NONE , no memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 30000, 30000 + 3000},
                 "SOH , no memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
                 {VPUNN::Cycles::NO_ERROR, true, 60000, 60000 + 3000},
                 "SOK , no memmove, "},
        };

        executeTests(tests);
    }
}

inline std::string toStringLayerSplitInfo(const LayerSplitInfo& d) {
    std::stringstream stream;
    stream << "\nVPU LayerSplitInfo : \n";
    int i{1};
    for (const auto& l : d) {
        stream << "\nTile: #" << i << " of:" << d.size() << "\n-Best Cost : " << l.best_intra_tile_split.first << " = "
               << Cycles::toErrorText(l.best_intra_tile_split.first)
               << "\n-ON this amount of workloads (intra-tile): " << l.best_intra_tile_split.second.size();
        stream << "\n--First WORKLOAD: \n" << l.best_intra_tile_split.second[0];
        stream << "\n-Tile LAYER: \n" << l.inter_tile_split_layer;
        // stream << "\n ***** ALT FORMAT Tile LAYER: ******** \n" <<
        // WLHelp::toDictString(l.inter_tile_split_layer);//dictionary style output for wl
        stream << "\n-END TILE no: " << i << " of:" << d.size();
        i++;
    }
    stream << "\nVPU LayerSplitInfo : END---------------- \n";
    return stream.str();
}

TEST_F(VPULayerCostModelTest, Unet_perf_SOH_SOK_after_SOHO) {
    const VPUNN::DPUWorkload wl_h5{
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(60, 5, 1024, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(60, 4, 176, 1, VPUNN::DataType::FLOAT16)},   // output dimensions
            {3, 3},                                                        // kernels
            {1, 1},                                                        // strides
            {1, 0, 1, 1},                                                  // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                            // execution mode
            VPUNN::ActivationFunction::NONE,                               // activation
            0.0F,                                                          // act_sparsity
            0.624613F,                                                     // weight_sparsity
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},            // input_swizzling
            {VPUNN::Swizzling::KEY_0},                                     // output_swizzling
            1,                                                             // output_write_tiles
            {0, 0, 0, 0},                                                  // offsets
            VPUNN::ISIStrategy::CLUSTERING,                                // isi_strategy
            true,                                                          // weight_sparsity_enabled
    };

    const VPUNN::DPUWorkload wl_h3{
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(60, 3, 1024, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(60, 2, 176, 1, VPUNN::DataType::FLOAT16)},   // output dimensions
            {3, 3},                                                        // kernels
            {1, 1},                                                        // strides
            {1, 0, 1, 1},                                                  // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                            // execution mode
            VPUNN::ActivationFunction::NONE,                               // activation
            0.0F,                                                          // act_sparsity
            0.624613F,                                                     // weight_sparsity
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},            // input_swizzling
            {VPUNN::Swizzling::KEY_0},                                     // output_swizzling
            1,                                                             // output_write_tiles
            {0, 0, 0, 0},                                                  // offsets
            VPUNN::ISIStrategy::CLUSTERING,                                // isi_strategy
            true,                                                          // weight_sparsity_enabled
    };

    // const std::string altModelName(NameHelperNN::get_model_root() + "vpu_2_7-150.vpunn");
    // VPULayerCostModel model_alternative{altModelName};
    const bool prefetch{true};
    VPULayerCostModel& theModel = model_2_7;

    {
        const VPUNN::DPULayer tst_layer(wl_h5);

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 262000, 262000 + 4000},  // 332898 ,   263125
                 "SOH , no memmove, "},
        };

        executeTests(tests);
    }
    //{
    //    VPUNN::DPULayer tst_layer(wl_h5);
    //    const VPULayerStrategy strategy{1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, prefetch};

    //    Logger::clear2ndlog();
    //    unsigned cost_cyc{};
    //    LayerSplitInfo detailed_split;
    //    ASSERT_NO_THROW(cost_cyc = theModel.Layer(tst_layer, VPUNN::VPUTilingStrategy::SOH, 1U, 2U, false, false,
    //                                              prefetch, detailed_split))
    //            << tst_layer << strategy;

    //    EXPECT_EQ(cost_cyc, 1) << tst_layer << toStringLayerSplitInfo(detailed_split) << Logger::get2ndlog();

    //    std::string err_info;
    //    CyclesInterfaceType dpu_cost = theModel.DPU(tst_layer, err_info);
    //    EXPECT_EQ(dpu_cost, 2) << dpu_cost << "=" << Cycles::toErrorText(dpu_cost) << tst_layer << err_info;
    //}

    {
        VPUNN::DPULayer tst_layer(wl_h3);
        const VPULayerStrategy strategy{1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, prefetch};

        Logger::clear2ndlog();
        unsigned cost_cyc{};
        LayerSplitInfo detailed_split;
        ASSERT_NO_THROW(cost_cyc = theModel.Layer(tst_layer, VPUNN::VPUTilingStrategy::SOH, 1U, 2U, false, false,
                                                  prefetch, detailed_split))
                << tst_layer << strategy;
        // EXPECT_EQ(cost_cyc, 3) << tst_layer << toStringLayerSplitInfo(detailed_split) << Logger::get2ndlog();

        std::string err_info;
        CyclesInterfaceType dpu_cost = theModel.DPU(tst_layer, err_info);
        // EXPECT_EQ(dpu_cost, 4) << dpu_cost << "=" << Cycles::toErrorText(dpu_cost) << tst_layer << err_info
        //                        << "END ERR";

        EXPECT_LE(cost_cyc, dpu_cost) << tst_layer << toStringLayerSplitInfo(detailed_split) << Logger::get2ndlog()
                                      << dpu_cost << "=" << Cycles::toErrorText(dpu_cost) << err_info << "END ERR";
    }
}

TEST_F(VPULayerCostModelTest, Layer_PRE_split_CLUSTERING) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(60, 7, 512, 1, VPUNN::DataType::UINT8)},    // input dimensions
            {VPUNN::VPUTensor(60, 6, 512, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            {3, 3},                                                       // kernels
            {1, 1},                                                       // strides
            {1, 0, 1, 1}                                                  // padding
    );

    const bool prefetch{true};  // prefetch was done
    VPULayerCostModel& theModel = model_2_7;

    {
        VPUNN::DPULayer tst_layer(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = true;
        tst_layer.weight_sparsity = 0.723566F;

        std::vector<DPULayer> splitLayers1{tst_layer};

        const VPUTilingStrategy strategy{VPUNN::VPUTilingStrategy::NONE};  // clustering 2T, no memo

        Logger::clear2ndlog();
        CyclesInterfaceType cost_cyc_Layer{};
        LayerSplitInfo detailed_split_layer;
        ASSERT_NO_THROW(cost_cyc_Layer = theModel.Layer(tst_layer, strategy, 1U, 2U, false, false, prefetch,
                                                        detailed_split_layer))
                << tst_layer << (int)strategy << " : " << VPUTilingStrategy_ToText.at(static_cast<int>(strategy));

        Logger::clear2ndlog();
        CyclesInterfaceType pre_split_cost1{};
        LayerSplitInfo detailed_split_pre_layer;
        ASSERT_NO_THROW(pre_split_cost1 = theModel.LayersPreSplit(splitLayers1, 1U, false, false, prefetch,
                                                                  detailed_split_pre_layer))
                << toStringLayerSplitInfo(detailed_split_pre_layer);

        EXPECT_EQ(cost_cyc_Layer, pre_split_cost1)
                << toStringLayerSplitInfo(detailed_split_layer) << toStringLayerSplitInfo(detailed_split_pre_layer)
                << Logger::get2ndlog() << "END ERR";

        Logger::clear2ndlog();
        CyclesInterfaceType pre_split_cost2{};
        LayerSplitInfo detailed_split_pre_layer2;
        std::vector<DPULayer> splitLayers2{};
        splitLayers2.push_back(detailed_split_layer[0].inter_tile_split_layer);
        splitLayers2.push_back(detailed_split_layer[1].inter_tile_split_layer);

        ASSERT_NO_THROW(pre_split_cost2 = theModel.LayersPreSplit(splitLayers2, 1U, false, false, prefetch,
                                                                  detailed_split_pre_layer2))
                << toStringLayerSplitInfo(detailed_split_pre_layer2);

        EXPECT_EQ(pre_split_cost2, pre_split_cost1)
                << toStringLayerSplitInfo(detailed_split_pre_layer2) << toStringLayerSplitInfo(detailed_split_pre_layer)
                << Logger::get2ndlog() << "END ERR";
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
    VPULayerCostModel& theModel = model_2_7;

    {
        VPUNN::DPULayer tst_layer(tst_layer_ref);
        tst_layer.weight_sparsity_enabled = true;
        tst_layer.weight_sparsity = 0.723566F;

        std::vector<DPULayer> splitLayers1{tst_layer};

        const VPUTilingStrategy strategy{VPUNN::VPUTilingStrategy::NONE};  // clustering 2T, no memo

        Logger::clear2ndlog();
        LayerSplitInfo detailed_split_layer;
        CyclesInterfaceType cost_cyc_LayerFromDDR{};
        ASSERT_NO_THROW(cost_cyc_LayerFromDDR = theModel.Layer(tst_layer, strategy, 1U, 2U, true, false, prefetch,
                                                               detailed_split_layer))
                << tst_layer << (int)strategy << " : " << VPUTilingStrategy_ToText.at(static_cast<int>(strategy))
                << toStringLayerSplitInfo(detailed_split_layer) << Logger::get2ndlog();

        Logger::clear2ndlog();
        detailed_split_layer.clear();

        CyclesInterfaceType cost_cyc_LayerFromCMX{};
        ASSERT_NO_THROW(cost_cyc_LayerFromCMX = theModel.Layer(tst_layer, strategy, 1U, 2U, false, true, prefetch,
                                                               detailed_split_layer))
                << tst_layer << (int)strategy << " : " << VPUTilingStrategy_ToText.at(static_cast<int>(strategy))
                << toStringLayerSplitInfo(detailed_split_layer) << Logger::get2ndlog();

        EXPECT_EQ(cost_cyc_LayerFromDDR, cost_cyc_LayerFromCMX) << tst_layer << Logger::get2ndlog() << "END ERR";

        Logger::clear2ndlog();
        detailed_split_layer.clear();

        CyclesInterfaceType cost_cyc_LayerNoMem{};
        ASSERT_NO_THROW(cost_cyc_LayerNoMem = theModel.Layer(tst_layer, strategy, 1U, 2U, false, false, prefetch,
                                                             detailed_split_layer))
                << tst_layer << (int)strategy << " : " << VPUTilingStrategy_ToText.at(static_cast<int>(strategy))
                << toStringLayerSplitInfo(detailed_split_layer) << Logger::get2ndlog();

        EXPECT_NE(cost_cyc_LayerFromDDR, cost_cyc_LayerNoMem) << tst_layer << Logger::get2ndlog() << "END ERR";

        EXPECT_GT(cost_cyc_LayerFromDDR, cost_cyc_LayerNoMem) << tst_layer << Logger::get2ndlog() << "END ERR";
        EXPECT_GT(cost_cyc_LayerFromCMX, cost_cyc_LayerNoMem) << tst_layer << Logger::get2ndlog() << "END ERR";

        EXPECT_EQ(cost_cyc_LayerFromDDR, cost_cyc_LayerFromCMX) << cost_cyc_LayerFromDDR << "\n"
                                                                << cost_cyc_LayerFromCMX << "\n"
                                                                << cost_cyc_LayerNoMem << "\n"
                                                                << tst_layer << Logger::get2ndlog() << "END ERR";
        const auto& outT{tst_layer.outputs[0]};
        auto cmxddr_dma = theModel.DMA(tst_layer.device, outT, outT, MemoryLocation::CMX, MemoryLocation::DRAM);
        auto ddrcmx_dma = theModel.DMA(tst_layer.device, outT, outT, MemoryLocation::DRAM, MemoryLocation::CMX);
        EXPECT_EQ(ddrcmx_dma, cmxddr_dma) << ddrcmx_dma << "\n"
                                          << cmxddr_dma << "\n"
                                          << cost_cyc_LayerFromDDR << "\n"
                                          << cost_cyc_LayerFromCMX << "\n"
                                          << cost_cyc_LayerNoMem << "\n";
    }
}

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

TEST_F(VPULayerCM_InvestigationTest, DWConv_SOK_SOH_Comparison_EISW_92399) {
    const VPUNN::DPUWorkload wl_h17{
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::DW_CONVOLUTION,
            {VPUNN::VPUTensor(17, 17, 288, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(17, 17, 288, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            {9, 9},                                                        // kernels
            {1, 1},                                                        // strides
            {4, 4, 4, 4},                                                  // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                            // execution mode
            VPUNN::ActivationFunction::NONE,                               // activation
            0.0F,                                                          // act_sparsity
            0.0F,                                                          // weight_sparsity
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},            // input_swizzling
            {VPUNN::Swizzling::KEY_0},                                     // output_swizzling
            1,                                                             // output_write_tiles
            {0, 0, 0, 0},                                                  // offsets
            VPUNN::ISIStrategy::CLUSTERING,                                // isi_strategy
            false,                                                         // weight_sparsity_enabled
    };

    // const VPUNN::DPUWorkload wl_h17_Top_1{
    //         VPUNN::VPUDevice::VPU_2_7,
    //         VPUNN::Operation::DW_CONVOLUTION,
    //         {VPUNN::VPUTensor(17, 13, 32, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
    //         {VPUNN::VPUTensor(17, 9, 32, 1, VPUNN::DataType::FLOAT16)},   // output dimensions
    //         {9, 9},                                                       // kernels
    //         {1, 1},                                                       // strides
    //         {4, 0, 4, 4},                                                 // padding
    //         VPUNN::ExecutionMode::CUBOID_16x16,                           // execution mode
    //         VPUNN::ActivationFunction::NONE,                              // activation
    //         0.0F,                                                         // act_sparsity
    //         0.0F,                                                         // weight_sparsity
    //         {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},           // input_swizzling
    //         {VPUNN::Swizzling::KEY_0},                                    // output_swizzling
    //         1,                                                            // output_write_tiles
    //         {0, 0, 0, 0},                                                 // offsets
    //         VPUNN::ISIStrategy::CLUSTERING,                               // isi_strategy
    //         false,                                                        // weight_sparsity_enabled
    // };
    // const VPUNN::DPUWorkload wl_h17_Bot_1{
    //         VPUNN::VPUDevice::VPU_2_7,
    //         VPUNN::Operation::DW_CONVOLUTION,
    //         {VPUNN::VPUTensor(17, 12, 32, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
    //         {VPUNN::VPUTensor(17, 8, 32, 1, VPUNN::DataType::FLOAT16)},   // output dimensions
    //         {9, 9},                                                       // kernels
    //         {1, 1},                                                       // strides
    //         {0, 4, 4, 4},                                                 // padding
    //         VPUNN::ExecutionMode::CUBOID_16x16,                           // execution mode
    //         VPUNN::ActivationFunction::NONE,                              // activation
    //         0.0F,                                                         // act_sparsity
    //         0.0F,                                                         // weight_sparsity
    //         {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},           // input_swizzling
    //         {VPUNN::Swizzling::KEY_0},                                    // output_swizzling
    //         1,                                                            // output_write_tiles
    //         {0, 0, 0, 0},                                                 // offsets
    //         VPUNN::ISIStrategy::CLUSTERING,                               // isi_strategy
    //         false,                                                        // weight_sparsity_enabled
    // };
    // const VPUNN::DPUWorkload wl_h17_Bot_2{
    //         VPUNN::VPUDevice::VPU_2_7,
    //         VPUNN::Operation::DW_CONVOLUTION,
    //         {VPUNN::VPUTensor(17, 12, 16, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
    //         {VPUNN::VPUTensor(17, 8, 16, 1, VPUNN::DataType::FLOAT16)},   // output dimensions
    //         {9, 9},                                                       // kernels
    //         {1, 1},                                                       // strides
    //         {0, 4, 4, 4},                                                 // padding
    //         VPUNN::ExecutionMode::CUBOID_16x16,                           // execution mode
    //         VPUNN::ActivationFunction::NONE,                              // activation
    //         0.0F,                                                         // act_sparsity
    //         0.0F,                                                         // weight_sparsity
    //         {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},           // input_swizzling
    //         {VPUNN::Swizzling::KEY_0},                                    // output_swizzling
    //         1,                                                            // output_write_tiles
    //         {0, 0, 0, 0},                                                 // offsets
    //         VPUNN::ISIStrategy::CLUSTERING,                               // isi_strategy
    //         false,                                                        // weight_sparsity_enabled
    // };

    // const VPUNN::DPUWorkload wl_h17_SOKHalf_1{
    //         VPUNN::VPUDevice::VPU_2_7,
    //         VPUNN::Operation::DW_CONVOLUTION,
    //         {VPUNN::VPUTensor(17, 17, 32, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
    //         {VPUNN::VPUTensor(17, 17, 32, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
    //         {9, 9},                                                       // kernels
    //         {1, 1},                                                       // strides
    //         {4, 4, 4, 4},                                                 // padding
    //         VPUNN::ExecutionMode::CUBOID_16x16,                           // execution mode
    //         VPUNN::ActivationFunction::NONE,                              // activation
    //         0.0F,                                                         // act_sparsity
    //         0.0F,                                                         // weight_sparsity
    //         {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},           // input_swizzling
    //         {VPUNN::Swizzling::KEY_0},                                    // output_swizzling
    //         2,                                                            // output_write_tiles
    //         {0, 0, 0, 0},                                                 // offsets
    //         VPUNN::ISIStrategy::SPLIT_OVER_K,                             // isi_strategy
    //         false,                                                        // weight_sparsity_enabled
    // };
    // const VPUNN::DPUWorkload wl_h17_SOKHalf_2{
    //         VPUNN::VPUDevice::VPU_2_7,
    //         VPUNN::Operation::DW_CONVOLUTION,
    //         {VPUNN::VPUTensor(17, 17, 16, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
    //         {VPUNN::VPUTensor(17, 17, 16, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
    //         {9, 9},                                                       // kernels
    //         {1, 1},                                                       // strides
    //         {4, 4, 4, 4},                                                 // padding
    //         VPUNN::ExecutionMode::CUBOID_16x16,                           // execution mode
    //         VPUNN::ActivationFunction::NONE,                              // activation
    //         0.0F,                                                         // act_sparsity
    //         0.0F,                                                         // weight_sparsity
    //         {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},           // input_swizzling
    //         {VPUNN::Swizzling::KEY_0},                                    // output_swizzling
    //         2,                                                            // output_write_tiles
    //         {0, 0, 0, 0},                                                 // offsets
    //         VPUNN::ISIStrategy::SPLIT_OVER_K,                             // isi_strategy
    //         false,                                                        // weight_sparsity_enabled
    // };

    // const std::string altModelName(NameHelperNN::get_model_root() + "vpu_2_7-150.vpunn");
    // VPULayerCostModel model_alternative{altModelName};
    const bool prefetch{true};
    VPULayerCostModel& theModel = model_2_7;

    {
        const VPUNN::DPULayer tst_layer(wl_h17);

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 95000, 95000 + 10000},  //
                 "SOH , no memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 70000, 70000 + 10000},  //
                 "SOK , no memmove, "},
        };
        executeTests(tests);
    }

    {
        VPUNN::DPULayer tst_layer(wl_h17);
        const VPULayerStrategy strategy{1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, prefetch};

        Logger::clear2ndlog();
        CyclesInterfaceType cost_cyc{};
        LayerSplitInfo detailed_split;
        ASSERT_NO_THROW(cost_cyc = theModel.Layer(tst_layer, strategy.tiling_strategy, strategy.nDPUs, strategy.nTiles,
                                                  false, false, prefetch, detailed_split))
                << tst_layer << strategy << cost_cyc;
        // EXPECT_EQ(cost_cyc, 1) << tst_layer << toStringLayerSplitInfo(detailed_split);  //<<
        //  Logger::get2ndlog();//99639
    }
    {
        VPUNN::DPULayer tst_layer(wl_h17);
        const VPULayerStrategy strategy{1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, prefetch};

        Logger::clear2ndlog();
        CyclesInterfaceType cost_cyc{};
        LayerSplitInfo detailed_split;
        ASSERT_NO_THROW(cost_cyc = theModel.Layer(tst_layer, strategy.tiling_strategy, strategy.nDPUs, strategy.nTiles,
                                                  false, false, prefetch, detailed_split))
                << tst_layer << strategy << cost_cyc;
        // EXPECT_EQ(cost_cyc, 2) << tst_layer << toStringLayerSplitInfo(detailed_split);  //<<
        //  Logger::get2ndlog();//76005
    }

    //{
    //    VPUNN::DPULayer tst_layer(wl_h17);
    //    const VPULayerStrategy strategy{1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, prefetch};

    //    Logger::clear2ndlog();
    //    unsigned cost_cyc{};
    //    LayerSplitInfo detailed_split;
    //    ASSERT_NO_THROW(cost_cyc = theModel.Layer(tst_layer, strategy.tiling_strategy, strategy.nDPUs,
    //    strategy.nTiles,
    //                                              false, false, prefetch, detailed_split))
    //            << tst_layer << strategy;
    //     EXPECT_EQ(cost_cyc, 3) << tst_layer << toStringLayerSplitInfo(detailed_split);  //<< Logger::get2ndlog();
    //}

    /* {
         Logger::clear2ndlog();
         std::string err_info;

         {
             std::string whatTest{"\nTEST of: SOH TOP K=32 \n"};
             VPUNN::DPULayer tst_layer(wl_h17_Top_1);

             CyclesInterfaceType dpu_cost = theModel.DPU(tst_layer, err_info);
             EXPECT_EQ(dpu_cost, 10) << whatTest << dpu_cost << "=" << Cycles::toErrorText(dpu_cost) << "\n"
                                     << tst_layer << err_info << "END ERR";
         }
         {
             std::string whatTest{"\nTEST of:SOH BOT K=32 \n"};
             VPUNN::DPULayer tst_layer(wl_h17_Bot_1);

             CyclesInterfaceType dpu_cost = theModel.DPU(tst_layer, err_info);
             EXPECT_EQ(dpu_cost, 11) << whatTest << dpu_cost << "=" << Cycles::toErrorText(dpu_cost) << "\n"
                                     << tst_layer << err_info << "END ERR";
         }
         {
             std::string whatTest{"\nTEST of:SOH BOT K=16 \n"};
             VPUNN::DPULayer tst_layer(wl_h17_Bot_2);

             CyclesInterfaceType dpu_cost = theModel.DPU(tst_layer, err_info);
             EXPECT_EQ(dpu_cost, 12) << whatTest << dpu_cost << "=" << Cycles::toErrorText(dpu_cost) << "\n"
                                     << tst_layer << err_info << "END ERR";
         }
         {
             std::string whatTest{"\nTEST of:SOK K=32 \n"};
             VPUNN::DPULayer tst_layer(wl_h17_SOKHalf_1);

             CyclesInterfaceType dpu_cost = theModel.DPU(tst_layer, err_info);
             EXPECT_EQ(dpu_cost, 13) << whatTest << dpu_cost << "=" << Cycles::toErrorText(dpu_cost) << "\n"
                                     << tst_layer << err_info << "END ERR";
         }
         {
             std::string whatTest{"\nTEST of:SOK K=16 \n"};
             VPUNN::DPULayer tst_layer(wl_h17_SOKHalf_2);

             CyclesInterfaceType dpu_cost = theModel.DPU(tst_layer, err_info);
             EXPECT_EQ(dpu_cost, 14) << whatTest << dpu_cost << "=" << Cycles::toErrorText(dpu_cost) << "\n"
                                     << tst_layer << err_info << "END ERR";
         }
     }*/
}

TEST_F(VPULayerCM_InvestigationTest, CONV_TILE_Mineeva_EISW_9xxxxx) {
    // Investigation on October 2023, older VPU version used. refresh provided
    const DPUWorkload wl_T1_3{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(512, 172, 4, 1, DataType::UINT8)},        // input dimensions
            {VPUTensor(512, 171, 16, 1, DataType::FLOAT16)},     // output dimensions
            {3, 3},                                              // kernels
            {1, 1},                                              // strides
            {1, 0, 1, 1},                                        // padding Top, Bottom, Left,  Right
            VPUNN::ExecutionMode::CUBOID_16x16,                  // execution mode
            VPUNN::ActivationFunction::NONE,                     // activation
            0.0F,                                                // act_sparsity
            0.0F,                                                // weight_sparsity
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},  // input_swizzling
            {VPUNN::Swizzling::KEY_0},                           // output_swizzling
            1,                                                   // output_write_tiles
            {0, 0, 0, 0},                                        // offsets
            VPUNN::ISIStrategy::CLUSTERING,                      // isi_strategy
            false,                                               // weight_sparsity_enabled
    };
    const DPUWorkload wl_T2_3{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(512, 173, 4, 1, DataType::UINT8)},        // input dimensions
            {VPUTensor(512, 171, 16, 1, DataType::FLOAT16)},     // output dimensions
            {3, 3},                                              // kernels
            {1, 1},                                              // strides
            {0, 0, 1, 1},                                        // padding Top, Bottom, Left,  Right
            VPUNN::ExecutionMode::CUBOID_16x16,                  // execution mode
            VPUNN::ActivationFunction::NONE,                     // activation
            0.0F,                                                // act_sparsity
            0.0F,                                                // weight_sparsity
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},  // input_swizzling
            {VPUNN::Swizzling::KEY_0},                           // output_swizzling
            1,                                                   // output_write_tiles
            {0, 0, 0, 0},                                        // offsets
            VPUNN::ISIStrategy::CLUSTERING,                      // isi_strategy
            false,                                               // weight_sparsity_enabled
    };
    const DPUWorkload wl_T3_3{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(512, 171, 4, 1, DataType::UINT8)},        // input dimensions
            {VPUTensor(512, 170, 16, 1, DataType::FLOAT16)},     // output dimensions
            {3, 3},                                              // kernels
            {1, 1},                                              // strides
            {0, 1, 1, 1},                                        // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                  // execution mode
            VPUNN::ActivationFunction::NONE,                     // activation
            0.0F,                                                // act_sparsity
            0.0F,                                                // weight_sparsity
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},  // input_swizzling
            {VPUNN::Swizzling::KEY_0},                           // output_swizzling
            1,                                                   // output_write_tiles
            {0, 0, 0, 0},                                        // offsets
            VPUNN::ISIStrategy::CLUSTERING,                      // isi_strategy
            false,                                               // weight_sparsity_enabled
    };

    ///////////////////
    const DPUWorkload wl_T1_4{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(512, 129, 4, 1, DataType::UINT8)},        // input dimensions
            {VPUTensor(512, 128, 16, 1, DataType::FLOAT16)},     // output dimensions
            {3, 3},                                              // kernels
            {1, 1},                                              // strides
            {1, 0, 1, 1},                                        // padding Top, Bottom, Left,  Right
            VPUNN::ExecutionMode::CUBOID_16x16,                  // execution mode
            VPUNN::ActivationFunction::NONE,                     // activation
            0.0F,                                                // act_sparsity
            0.0F,                                                // weight_sparsity
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},  // input_swizzling
            {VPUNN::Swizzling::KEY_0},                           // output_swizzling
            1,                                                   // output_write_tiles
            {0, 0, 0, 0},                                        // offsets
            VPUNN::ISIStrategy::CLUSTERING,                      // isi_strategy
            false,                                               // weight_sparsity_enabled
    };
    const DPUWorkload wl_T2_4{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(512, 130, 4, 1, DataType::UINT8)},        // input dimensions
            {VPUTensor(512, 128, 16, 1, DataType::FLOAT16)},     // output dimensions
            {3, 3},                                              // kernels
            {1, 1},                                              // strides
            {0, 0, 1, 1},                                        // padding Top, Bottom, Left,  Right
            VPUNN::ExecutionMode::CUBOID_16x16,                  // execution mode
            VPUNN::ActivationFunction::NONE,                     // activation
            0.0F,                                                // act_sparsity
            0.0F,                                                // weight_sparsity
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},  // input_swizzling
            {VPUNN::Swizzling::KEY_0},                           // output_swizzling
            1,                                                   // output_write_tiles
            {0, 0, 0, 0},                                        // offsets
            VPUNN::ISIStrategy::CLUSTERING,                      // isi_strategy
            false,                                               // weight_sparsity_enabled
    };
    const DPUWorkload wl_T3_4{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(512, 130, 4, 1, DataType::UINT8)},        // input dimensions
            {VPUTensor(512, 128, 16, 1, DataType::FLOAT16)},     // output dimensions
            {3, 3},                                              // kernels
            {1, 1},                                              // strides
            {0, 0, 1, 1},                                        // padding Top, Bottom, Left,  Right
            VPUNN::ExecutionMode::CUBOID_16x16,                  // execution mode
            VPUNN::ActivationFunction::NONE,                     // activation
            0.0F,                                                // act_sparsity
            0.0F,                                                // weight_sparsity
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},  // input_swizzling
            {VPUNN::Swizzling::KEY_0},                           // output_swizzling
            1,                                                   // output_write_tiles
            {0, 0, 0, 0},                                        // offsets
            VPUNN::ISIStrategy::CLUSTERING,                      // isi_strategy
            false,                                               // weight_sparsity_enabled
    };
    const DPUWorkload wl_T4_4{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(512, 129, 4, 1, DataType::UINT8)},        // input dimensions
            {VPUTensor(512, 128, 16, 1, DataType::FLOAT16)},     // output dimensions
            {3, 3},                                              // kernels
            {1, 1},                                              // strides
            {1, 0, 1, 1},                                        // padding Top, Bottom, Left,  Right
            VPUNN::ExecutionMode::CUBOID_16x16,                  // execution mode
            VPUNN::ActivationFunction::NONE,                     // activation
            0.0F,                                                // act_sparsity
            0.0F,                                                // weight_sparsity
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},  // input_swizzling
            {VPUNN::Swizzling::KEY_0},                           // output_swizzling
            1,                                                   // output_write_tiles
            {0, 1, 0, 0},                                        // offsets
            VPUNN::ISIStrategy::CLUSTERING,                      // isi_strategy
            false,                                               // weight_sparsity_enabled
    };

    // const std::string altModelName(NameHelperNN::get_model_root() + "vpu_2_7-150.vpunn");
    // VPULayerCostModel model_alternative{altModelName};
    // const bool prefetch{true};
    VPULayerCostModel& theModel = model_2_7;

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
        const VPULayerStrategy strategy{1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, true /*prefetch*/};
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
        const VPULayerStrategy strategy{1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, false /*prefetch*/};
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

TEST_F(VPULayerCM_InvestigationTest, RuntimeELT_CONV_SOH_SOK_EISW_98656) {
    const VPUNN::DPUWorkload wl_elm_layer{
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::ELTWISE,
            {VPUNN::VPUTensor(14, 14, 1024, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(14, 14, 1024, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {1, 1},                                                       // kernels
            {1, 1},                                                       // strides
            {0, 0, 0, 0},                                                 // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                           // execution mode
            VPUNN::ActivationFunction::NONE,                              // activation
            0.0F,                                                         // act_sparsity
            0.0F,                                                         // weight_sparsity
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},           // input_swizzling
            {VPUNN::Swizzling::KEY_0},                                    // output_swizzling
            1,                                                            // output_write_tiles
            {0, 0, 0, 0},                                                 // offsets
            VPUNN::ISIStrategy::CLUSTERING,                               // isi_strategy
            false,                                                        // weight_sparsity_enabled
    };

    const VPUNN::DPUWorkload wl_conv_layer{
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(14, 14, 1024, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(14, 14, 256, 1, VPUNN::DataType::UINT8)},   // output dimensions
            {1, 1},                                                       // kernels
            {1, 1},                                                       // strides
            {0, 0, 0, 0},                                                 // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                           // execution mode
            VPUNN::ActivationFunction::NONE,                              // activation
            0.0F,                                                         // act_sparsity
            0.259766F,                                                    // weight_sparsity
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},           // input_swizzling
            {VPUNN::Swizzling::KEY_0},                                    // output_swizzling
            1,                                                            // output_write_tiles
            {0, 0, 0, 0},                                                 // offsets
            VPUNN::ISIStrategy::CLUSTERING,                               // isi_strategy
            true,                                                         // weight_sparsity_enabled
    };

    // const std::string altModelName(NameHelperNN::get_model_root() + "vpu_2_7-150.vpunn");
    // VPULayerCostModel model_alternative{altModelName};
    const bool prefetch{true};
    VPULayerCostModel& theModel = model_2_7;

    //{
    //    const VPUNN::DPULayer tst_layer(wl_elm_layer);

    //    const std::vector<TestCase> tests{
    //            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, prefetch}},
    //             {VPUNN::Cycles::NO_ERROR, true, 95000, 95000 + 10000},  //
    //             "SOH , no memmove, "},
    //            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, prefetch}},
    //             {VPUNN::Cycles::NO_ERROR, true, 70000, 70000 + 10000},  //
    //             "Clustering , no memmove, "},
    //    };
    //    executeTests(tests);
    //}
    const std::string nline{"\n ------------- NEW TEST------------------------------------ ------------------"};

    // element wise
    {
        std::cout << nline;
        VPUNN::DPULayer tst_layer(wl_elm_layer);
        const VPULayerStrategy strategy{1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, prefetch};

        Logger::clear2ndlog();
        CyclesInterfaceType cost_cyc{};
        LayerSplitInfo detailed_split;
        ASSERT_NO_THROW(cost_cyc = theModel.Layer(tst_layer, strategy.tiling_strategy, strategy.nDPUs, strategy.nTiles,
                                                  false, false, prefetch, detailed_split))
                << tst_layer << strategy << cost_cyc;
        //      EXPECT_EQ(cost_cyc, 1) << tst_layer << toStringLayerSplitInfo(detailed_split);  //<<
        //  Logger::get2ndlog();//99639
    }
    {
        std::cout << nline;
        VPUNN::DPULayer tst_layer(wl_elm_layer);
        const VPULayerStrategy strategy{1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, prefetch};

        Logger::clear2ndlog();
        CyclesInterfaceType cost_cyc{};
        LayerSplitInfo detailed_split;
        ASSERT_NO_THROW(cost_cyc = theModel.Layer(tst_layer, strategy.tiling_strategy, strategy.nDPUs, strategy.nTiles,
                                                  false, false, prefetch, detailed_split))
                << tst_layer << strategy << cost_cyc;
        //        EXPECT_EQ(cost_cyc, 2) << tst_layer << toStringLayerSplitInfo(detailed_split);  //<<
    }

    // CONV
    {
        std::cout << nline;
        VPUNN::DPULayer tst_layer(wl_conv_layer);
        const VPULayerStrategy strategy{1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH, false, false, prefetch};

        Logger::clear2ndlog();
        CyclesInterfaceType cost_cyc{};
        LayerSplitInfo detailed_split;
        ASSERT_NO_THROW(cost_cyc = theModel.Layer(tst_layer, strategy.tiling_strategy, strategy.nDPUs, strategy.nTiles,
                                                  false, false, prefetch, detailed_split))
                << tst_layer << strategy << cost_cyc;
        //      EXPECT_EQ(cost_cyc, 3) << tst_layer << toStringLayerSplitInfo(detailed_split);  //<<
    }
    {
        std::cout << nline;
        VPUNN::DPULayer tst_layer(wl_conv_layer);
        const VPULayerStrategy strategy{1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, prefetch};

        Logger::clear2ndlog();
        CyclesInterfaceType cost_cyc{};
        LayerSplitInfo detailed_split;
        ASSERT_NO_THROW(cost_cyc = theModel.Layer(tst_layer, strategy.tiling_strategy, strategy.nDPUs, strategy.nTiles,
                                                  false, false, prefetch, detailed_split))
                << tst_layer << strategy << cost_cyc;
        //        EXPECT_EQ(cost_cyc, 4) << tst_layer << toStringLayerSplitInfo(detailed_split);  //<<
    }
    {
        std::cout << nline;
        VPUNN::DPULayer tst_layer(wl_conv_layer);
        const VPULayerStrategy strategy{1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, prefetch};

        Logger::clear2ndlog();
        CyclesInterfaceType cost_cyc{};
        LayerSplitInfo detailed_split;
        ASSERT_NO_THROW(cost_cyc = theModel.Layer(tst_layer, strategy.tiling_strategy, strategy.nDPUs, strategy.nTiles,
                                                  false, false, prefetch, detailed_split))
                << tst_layer << strategy << cost_cyc;
        //       EXPECT_EQ(cost_cyc, 5) << tst_layer << toStringLayerSplitInfo(detailed_split);  //<<
    }
}

}  // namespace VPUNN_unit_tests
