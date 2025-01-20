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
#include "vpu_layer_cost_model.h"

#include "vpu/shave/layers.h"

namespace VPUNN_unit_tests {
using namespace VPUNN;

inline std::string toStringLayerSplitInfo(const LayerSplitInfo& d) {
    std::stringstream stream;
    stream << "\nVPU LayerSplitInfo : \n";
    int i{1};
    for (const auto& l : d) {
        const auto& wls{l.best_intra_tile_split.second};       // the winning workloads container
        const auto& best_time{l.best_intra_tile_split.first};  // best workload time. the winner
        stream << "\nTile: #" << i << " of:" << d.size() << "\n-Best Cost : " << best_time << " = "
               << Cycles::toErrorText(best_time) << "\n-ON this amount of workloads (intra-tile): " << wls.size();
        stream << "\n--First WORKLOAD: \n";
        ((wls.size() > 0) ? (stream << wls[0] << " Cycles:  NA in info") : (stream << " NO WORKLOADS EXIST!"));
        stream << "\n-Tile LAYER: \n" << l.inter_tile_split_layer;
        // stream << "\n ***** ALT FORMAT Tile LAYER: ******** \n" <<
        // WLHelp::toDictString(l.inter_tile_split_layer);//dictionary style output for wl
        stream << "\n-END TILE no: " << i << " of:" << d.size();
        i++;
    }
    stream << "\nVPU LayerSplitInfo : END---------------- \n";
    return stream.str();
}

class VPULayerCostModelTest : public ::testing::Test {
public:
protected:
    DMACostModel<DMANNWorkload_NPU27> dma_model_invalid{""};
    DMACostModel<DMANNWorkload_NPU40> dma_model_4_0{VPU_DMA_4_0_MODEL_PATH};
    DMACostModel<DMANNWorkload_NPU27> dma_model_2_7{VPU_DMA_2_7_MODEL_PATH};
    DMACostModel<DMANNWorkload_NPU40> dma_model_5_0{NPU_DMA_5_0_MODEL_PATH};

    VPULayerCostModel model_invalid_without_dma{""};
    VPULayerCostModel model_invalid{&dma_model_invalid, ""};
    VPULayerCostModel model_2_0{&dma_model_2_7, VPU_2_0_MODEL_PATH};
    VPULayerCostModel model_2_7_no_dma{VPU_2_7_MODEL_PATH};
    VPULayerCostModel model_2_7{&dma_model_2_7, VPU_2_7_MODEL_PATH};
    VPULayerCostModel model_4_0{&dma_model_4_0, VPU_4_0_MODEL_PATH};

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
        case VPUDevice::VPU_4_0:
            return model_4_0;
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

    bool show_split{false};  ///> controls  show in DoRegularTest, false is more quiet.

    std::string current_test_name() const {
        const ::testing::TestInfo* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        return test_info->name();
    }

    std::string current_test_fixture_name() const {
        const ::testing::TestInfo* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        return test_info->test_suite_name();
    }

    void DoRegularTest(const TestInput& t_in, const TestExpectations& t_exp, const std::string& test_case = "") {
        DPULayer l1{t_in.l1};
        VPULayerStrategy strategy{t_in.strategy};

        std::string t_header{"** Test Case: " + test_case + "\n"};
        std::cout << ">> " << t_header << std::endl;

        Logger::clear2ndlog();
        unsigned cost_cyc{};

        std::string test_info_str = test_case;
        std::replace(test_info_str.begin(), test_info_str.end(), ',', '_');
        test_info_str.erase(std::remove(test_info_str.begin(), test_info_str.end(), '\n'), test_info_str.end());
        // TO BE REFACTORED
        getModel(l1.device).get_serializer().serialize(SerializableField<std::string>(
                "info", current_test_fixture_name() + "::" + current_test_name() + "::" + test_info_str +
                                "::" + std::to_string(t_exp.min_cyc) + "<->" + std::to_string(t_exp.max_cyc)));

        LayerSplitInfo detailed_split;
        ASSERT_NO_THROW(cost_cyc = getModel(l1.device).Layer(l1, strategy, detailed_split))
                << t_header << l1 << strategy;

        const auto err_expected{t_exp.err_expected};
        if (Cycles::isErrorCode(err_expected)) {  // error code is expected
            EXPECT_TRUE(Cycles::isErrorCode(cost_cyc)) << t_header << "Expected ERROR, but value received: " << cost_cyc
                                                       << " : " << Cycles::toErrorText(cost_cyc) << std::endl
                                                       << "Expected ERROR code: " << err_expected << " : "
                                                       << Cycles::toErrorText(err_expected) << std::endl
                                                       << l1 << strategy << "\n 2ndLOG:\n"
                                                       << Logger::get2ndlog();

            if (Cycles::isErrorCode(cost_cyc) &&
                t_exp.strict_err_check) {  // in case is error code AND we want to have exact value
                EXPECT_EQ(cost_cyc, err_expected) << t_header << "ERROR code received: " << cost_cyc << " : "
                                                  << Cycles::toErrorText(cost_cyc) << std::endl
                                                  << "Expected ERROR code: " << err_expected << " : "
                                                  << Cycles::toErrorText(err_expected) << std::endl
                                                  << l1 << strategy << "\n 2ndLOG:\n"
                                                  << Logger::get2ndlog();
            }

        } else {  // regular cycle value expected
            EXPECT_FALSE(Cycles::isErrorCode(cost_cyc))
                    << t_header << "Unexpected ERROR code: " << cost_cyc << " : " << Cycles::toErrorText(cost_cyc)
                    << "\n  expected in [ " << t_exp.min_cyc << " , " << t_exp.max_cyc << " ] \n"
                    << l1 << strategy << "\n 2ndLOG:\n"
                    << Logger::get2ndlog() << (show_split ? toStringLayerSplitInfo(detailed_split) : "\n");

            if (!Cycles::isErrorCode(cost_cyc)) {  // if cycle value check against other ranges
                EXPECT_GT(cost_cyc, 0u) << t_header << l1 << strategy;
                EXPECT_LT(cost_cyc, MAX_COST) << t_header << l1 << strategy;  // 1 million

                EXPECT_TRUE(cost_cyc >= t_exp.min_cyc && cost_cyc <= t_exp.max_cyc)
                        << t_header << " Cost not in interval. cost: " << cost_cyc << ",  expected in [ "
                        << t_exp.min_cyc << " , " << t_exp.max_cyc << " ] \n"
                        << l1 << strategy << "\n 2ndLOG:\n"
                        << Logger::get2ndlog() << (show_split ? toStringLayerSplitInfo(detailed_split) : "\n");
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

    VPUNN::SHAVEWorkload generate_helper_shave_wl_layer(const VPUNN::VPUDevice device, const unsigned int dim,
                                                        const unsigned int channels) {
        return VPUNN::SHAVEWorkload(
                "sigmoid",                                                            // name
                device,                                                               // VPU device
                {VPUNN::VPUTensor(dim, dim, channels, 1, VPUNN::DataType::FLOAT16)},  // Input tensor
                {VPUNN::VPUTensor(dim, dim, channels, 1, VPUNN::DataType::FLOAT16)}   // Output tensor
        );
    }

    /// @brief lambda function: set sparsities for an wl, they are given as parameters
    ///
    /// @param wl: The DPU Workload, we set its sparsities
    /// @param input_sparsity_enable: activate/deactivate input sparsity, it's value could be true (that means input
    /// sparsity is activate) or false (that means input sparsity is deactivate)
    /// @param act_sparsity: value for input sparsity; should be [0, 1]
    /// @param weight_sparsity_enable: activate/deactivate weight sparsity , it's value could be true (that means weight
    /// sparsity is activate) or false (that means weight sparsity is deactivate)
    /// @param weight_sparsity: value for weight sparsity; should be [0, 1]
    ///
    /// @return the wl with new sparsities
    DPUWorkload wl_sparsity_initialization(const DPUWorkload& wl, bool input_sparsity_enable, float act_sparsity,
                                           bool weight_sparsity_enable, float weight_sparsity) const {
        DPUWorkload wl_ref{wl};
        // input sparsity
        wl_ref.inputs[0].set_sparsity(input_sparsity_enable);
        wl_ref.act_sparsity = act_sparsity;

        // weight sparsity
        wl_ref.weight_sparsity_enabled = weight_sparsity_enable;
        wl_ref.weight_sparsity = weight_sparsity;

        return wl_ref;
    };
};
TEST_F(VPULayerCostModelTest, LayerLoadModels) {
    EXPECT_EQ(model_2_0.nn_initialized(), true);
    EXPECT_EQ(model_2_7.nn_initialized(), true);
    EXPECT_EQ(model_2_7_no_dma.nn_initialized(), true);
    EXPECT_EQ(model_4_0.nn_initialized(), true);
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

TEST_F(VPULayerCostModelTest, LayerCostModelVPU_2_7_shv_workload) {
    auto layer = generate_helper_shave_wl_layer(VPUNN::VPUDevice::VPU_2_7, 16, 64);
    auto vpu20_layer_cost = model_2_7.Layer(layer, 5, 4);

    // Basic expectations
    EXPECT_GT(vpu20_layer_cost, 0u);
}

TEST_F(VPULayerCostModelTest, LayerCostModelVPU_2_7_shv_wl_bad_name) {
    auto layer = VPUNN::SHAVEWorkload("bad_wl",                                                     // name
                                      VPUNN::VPUDevice::VPU_2_7,                                    // VPU device
                                      {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::FLOAT16)},  // Input tensor
                                      {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::FLOAT16)}   // Output tensor
    );
    auto vpu20_layer_cost = model_2_7.Layer(layer, 5, 4);

    // Basic expectations
    EXPECT_EQ(vpu20_layer_cost, Cycles::ERROR_SHAVE);
}

TEST_F(VPULayerCostModelTest, ELTWISE_Concrete_Add14_VPU27) {
    const VPUNN::DPULayer tst_layer(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::ELTWISE,
                                    {VPUNN::VPUTensor(56, 56, 256, 1, VPUNN::DataType::UINT8)},  // input dimensions
                                    {VPUNN::VPUTensor(56, 56, 256, 1, VPUNN::DataType::UINT8)},  // output dimensions
                                    {1, 1},                                                      // kernels
                                    {1, 1},                                                      // strides
                                    {0, 0, 0, 0}                                                 // padding
    );

    // EXPECT_TRUE(false);

    {
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUTilingStrategy::NONE, false, false, true}},
                 {NO_ERROR_EXPECTED, true, 22500, 22500 + 2250},  // v16 23774, v17 22844   //v150 23213
                 "Clustering, no mem move"},
                {{tst_layer, {1U, 1U, 2U, VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {NO_ERROR_EXPECTED, true, 10000, 10000 + 2500},  // v16 11763, v17 12193   //v150 10893
                 "SOHO, no mem move"},
                {{tst_layer,
                  {1U, 1U, 2U, VPUTilingStrategy::SOK, false, false,
                   true}},                                        // SOK is now allowed for element wise (not trained)
                 {NO_ERROR_EXPECTED, true, 11000, 11000 + 2000},  // v16 17458, v17 16994  //v150 11751    GT??
                                                                  // goes back to CLU OWT=1
                 "SOK tentative on elementwise, no mem move"},
        };

        executeTests(tests);
    }
    {  // what is the SOK eqiovalent if OWT is 1
        DPUWorkload wl{tst_layer};
        wl.inputs[0] = VPUTensor(56, 56, 256 / 2, 1, VPUNN::DataType::UINT8);
        wl.outputs[0] = VPUTensor(56, 56, 256 / 2, 1, VPUNN::DataType::UINT8);  // SOK result
        wl.isi_strategy = ISIStrategy::CLUSTERING;
        const DPULayer tst_layer2{wl};

        wl.output_write_tiles = 2;
        const DPULayer tst_layer3{wl};  // CLU +OWT2 will become SOK, but elementwise will become CLU owt=1

        const std::vector<TestCase> tests{
                {{tst_layer2, {1U, 1U, 2U, VPUTilingStrategy::NONE, false, false, true}},
                 {NO_ERROR_EXPECTED, true, 11000, 11000 + 2000},  // v16 11844 , v17 11350 //v150 ???
                 "Clustering equivalent of prev SOK, but OWT =1, no mem move"},
                {{tst_layer3, {1U, 1U, 2U, VPUTilingStrategy::NONE, false, false, true}},
                 {NO_ERROR_EXPECTED, true, 11000, 11000 + 2000},  // v16 18902 , v17 18364  //v150 ???  not trained!
                                                                  //, should = first case CLU+OWT=1
                 "Clustering equivalent of prev SOK, but OWT =2 (+ ELM), no mem move"},

        };
        executeTests(tests);
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
        tin.strategy.tiling_strategy = VPUNN::VPUTilingStrategy::SOH_Overlapped;

        TestExpectations texp{VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0};

        DoRegularTest(tin, texp, "SOH with errors");
    }

    const std::vector<TestCase> tests{
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
             {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
             "CLUSTERING , no memmove, with errors"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
             {NO_ERROR_EXPECTED, true, 37000, 37000 + 3500},
             "SOK , no memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
             {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
             "SOH ,no memmove, with errors"},
    };

    executeTests(tests);
}

TEST_F(VPULayerCostModelTest, CONVOLUTION_Concrete_Multiply8641_NPU40_mock) {
    const VPUNN::DPULayer tst_layer(VPUNN::VPUDevice::VPU_4_0, VPUNN::Operation::CONVOLUTION,
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
        tin.strategy.tiling_strategy = VPUNN::VPUTilingStrategy::SOH_Overlapped;

        TestExpectations texp{VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 1U, 1U};

        DoRegularTest(tin, texp, "SOH with errors");
    }
    // same as for VPU 2.7

    show_split = true;
    const std::vector<TestCase> tests{
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
             {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
             "CLUSTERING , no memmove, with errors"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
             {NO_ERROR_EXPECTED, true, 28500, 37313 + 3700},  // NEED ground truth on this NPU40.  GTL 37313
             "SOK , no memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
             {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
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
    const auto origMax = getModel(VPUDevice::VPU_2_7).get_maxWorkloadsPerIntraTileSplit();
    getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(50);  // original test context was 50

    {  // SOK
        TestInput tin{tst_layer, tst_strategy};
        tin.strategy.tiling_strategy = VPUNN::VPUTilingStrategy::SOK;

        TestExpectations texp{NO_ERROR_EXPECTED, false, 14000, 14000 + 2500};  // v16:16032,  GT??

        DoRegularTest(tin, texp, "SOK ok");
    }

    const std::vector<TestCase> tests{
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
             {VPUNN::Cycles::NO_ERROR, false, 50000, 50000 + 5000},
             "CLUSTERING , no memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
             {NO_ERROR_EXPECTED, false, 14000, 14000 + 2300},  // v16 16300
             "SOK , no memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
             {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 52500, 52500 + 1000},  // GT ???
             "SOHO ,no memmove"},
    };

    executeTests(tests);

    getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(origMax);
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
             {VPUNN::Cycles::NO_ERROR, false, 52000, 52000 + 7000},
             "CLUSTERING , no memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
             {NO_ERROR_EXPECTED, false, 20000, 20000 + 5000},  // GT ???
             "SOK , no memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
             {VPUNN::Cycles::NO_ERROR, true, 51000, 51000 + 6000},  //
             "SOHO ,no memmove"},
    };

    const auto origMax = getModel(VPUDevice::VPU_2_7).get_maxWorkloadsPerIntraTileSplit();
    getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(50);  // original test context was 50
    executeTests(tests);
    getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(origMax);
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

        TestExpectations texp{NO_ERROR_EXPECTED, false, 16000, 16000 + 3000};  //

        DoRegularTest(tin, texp, "SOK convolution");
    }

    const std::vector<TestCase> tests{
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
             {NO_ERROR_EXPECTED, false, 34000, 34000 + 2000},
             "CLUSTERING , no memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
             {NO_ERROR_EXPECTED, false, 79000, 79000 + 4000},  // fetching big data
             "CLUSTERING ,with fetch required"},

            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
             {NO_ERROR_EXPECTED, false, 16000, 16000 + 3000},  //
             "SOK , no memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
             {NO_ERROR_EXPECTED, false, 16000 + 22000, 16000 + 22000 + 3000},  // fetching big data
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
             {NO_ERROR_EXPECTED, false, 52000, 52000 + 18000},  // huge change
             "CLUSTERING , no mem"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
             {NO_ERROR_EXPECTED, false, 13000, 13000 + 7000},  // huge
             "SOK , no mem"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
             {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 0, 0 + 0},  // 7 cannot be split, must be 14
             "SOH ,no mem"},
    };

    const auto origMax = getModel(VPUDevice::VPU_2_7).get_maxWorkloadsPerIntraTileSplit();
    getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(50);  // original test context was 50
    executeTests(tests);
    getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(origMax);
}

// this test is used to split a layer into tiles
// we use these tiles in some regression tests, in class Regression_tests_MAXPOOL_NPU40
TEST_F(VPULayerCostModelTest, DISABLED_Maxpool_layer_split_NPU40) {
    const VPUNN::DPUWorkload wl_layer{
            VPUNN::VPUDevice::VPU_4_0,
            VPUNN::Operation::MAXPOOL,
            {VPUNN::VPUTensor(54, 54, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(53, 53, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                     // kernels
            {1, 1},                                                     // strides
            {1, 0, 1, 0},                                               // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                         // execution mode
            VPUNN::ActivationFunction::NONE,                            // activation
            0.0F,                                                       // act_sparsity
            0.0F,                                                       // weight_sparsity
            {swz_def, swz_def},                                         // input_swizzling
            {swz_def},                                                  // output_swizzling
            1,                                                          // output_write_tiles
            {0, 0, 0, 0},                                               // offsets
            VPUNN::ISIStrategy::CLUSTERING,                             // isi_strategy
            false,                                                      // weight_sparsity_enabled
    };

    bool prefetch{true};
    show_split = true;
    unsigned int no_fail = 1;
    // VPULayerCostModel& theModel = model_4_0;
    {
        const VPUNN::DPULayer tst_layer(wl_layer);
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 1U, VPUNN::VPUTilingStrategy::NONE, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 14000U, 14000U * no_fail + 1000U},  // 14492 V17:
                 "Full, no memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 5500U, 5500U * no_fail + 1000U},  // 6062 V17:
                 "SOHO /2, no memmove, "},
                {{tst_layer, {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 2500U, 2500U * no_fail + 1000U},  // 3168 v17:
                 "SOHO /4 , no memmove, "},

                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 7500U, 7500U * no_fail + 1000U},  // 8228 v17:
                 "SOK , no memmove, "},

        };
        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, MAXPOOL_avgpoolBased_172_VPU27_SOH) {  // SOH Split possible at limit
    const VPUNN::DPULayer tst_layer(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::MAXPOOL,
                                    {VPUNN::VPUTensor(7, 14, 2048, 1, VPUNN::DataType::UINT8)},  // input dimensions
                                    {VPUNN::VPUTensor(1, 8, 2048, 1, VPUNN::DataType::UINT8)},   // output dimensions
                                    {7, 7},                                                      // kernels
                                    {1, 1},                                                      // strides
                                    {0, 0, 0, 0}                                                 // padding
    );
    // EXPECT_TRUE(false);

    const std::vector<TestCase> tests{
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
             {NO_ERROR_EXPECTED, false, 51000, 51000 + 24000},  // huge  GT??
             "CLUSTERING , no mem"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
             {NO_ERROR_EXPECTED, false, 21000, 21000 + 3000},
             "SOK , no mem"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
             {NO_ERROR_EXPECTED, false, 51000, 51000 + 20000},  // huge GT??
             "SOH ,output H ?"},
    };

    const auto origMax = getModel(VPUDevice::VPU_2_7).get_maxWorkloadsPerIntraTileSplit();
    getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(50);  // original test context was 50
    executeTests(tests);
    getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(origMax);
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
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
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

    const auto origMax = getModel(VPUDevice::VPU_2_7).get_maxWorkloadsPerIntraTileSplit();
    getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(50);  // original test context was 50

    {  // CLUSTERING
        TestInput tin{tst_layer, tst_strategy};
        tin.strategy.tiling_strategy = VPUNN::VPUTilingStrategy::NONE;

        // Very large because 2032 is  mod 16, but 31X64 +1X48 is not accepted, 63x32+1X16 is not reached (limit 50)
        // and we cannot reach a split like: 31x64 = 1984  , 1x32=32,  1x16=16  =>2032 split in 33 workloads
        // the 1x2032 is not accepted because ch not 16/32/64
        TestExpectations texp{VPUNN::Cycles::ERROR_TILE_OUTPUT, true, 0, 0 + 0};

        DoRegularTest(tin, texp, "CLUST not valid 1 workload");
    }

    {  // SOK
        TestInput tin{tst_layer, tst_strategy};
        tin.strategy.tiling_strategy = VPUNN::VPUTilingStrategy::SOK;

        // 2032 split to tile like 1024 and 1008, both div by 16
        // 1024 = 16x64
        // 1008 :   15x64 , 1x48 : not allowed;   31x32, 1x16  reached,    but  15x64 , 1x32, 1x16 : not possible
        // but best.   due to not possible one we get 14000(or 16k) instead of 8000
        TestExpectations texp{NO_ERROR_EXPECTED, false, 14000, 14000 + 2300};

        DoRegularTest(tin, texp, "SOK not optimum: 15x64 , 1x32, 1x16  = 1008 ");
    }

    const std::vector<TestCase> tests{
            //         {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
            //          {NO_ERROR_EXPECTED, false, 2473000, 2474000},
            //          "CLUSTERING , no memmove"},
            //         {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
            //          {NO_ERROR_EXPECTED, false, 142000, 143000},
            //          "SOK , no memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
             {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 0, 0},
             "SOHO ,no memmove"},
    };

    executeTests(tests);

    getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(origMax);
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
    const auto origMax = getModel(VPUDevice::VPU_2_7).get_maxWorkloadsPerIntraTileSplit();
    getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(50);  // original test context was 50

    {  // CLUSTERING
        TestInput tin{tst_layer, tst_strategy};
        tin.strategy.tiling_strategy = VPUNN::VPUTilingStrategy::NONE;

        // Very large because 2032 is  mod 16, but 31X64 +1X48 is not accepted, 63x32+1X16 is not reached (limit 50)
        // and we cannot reach a split like: 31x64 = 1984  , 1x32=32,  1x16=16  =>2032 split in 33 workloads
        // the 1x2032 is not accepted because ch not 16/32/64
        TestExpectations texp{VPUNN::Cycles::ERROR_TILE_OUTPUT, true, 0, 0 + 0};

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
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
             {VPUNN::Cycles::ERROR_TILE_OUTPUT, true, 0, 0},  // but is 31000 if allowed fro split to 64 workloads
             "SOH ,no memmove"},
    };

    executeTests(tests);

    getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(origMax);
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
                 {VPUNN::Cycles::NO_ERROR, false, 260000, 260000 + 55000},
                 "SOK , no memmove, sparse"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0 + 1000},  // was SOH no overlap  : 327500
                 "SOH , no memmove, sparse"},
        };

        executeTests(tests);
    }
}
TEST_F(VPULayerCostModelTest, CONVOLUTION_Concrete_Multiply6326_FLOAT_INT_VPU27_EISXW_76882) {
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
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
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
                 "CLUSTERING ,flt int, sparse "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, false, 250000, 250000 + 60000},
                 "SOK , no memmove, sparse"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, false, 300000,
                  300000 + 60000},  // MAIN TEST CASE, v16 309k, pre v16 350k, GT ???
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
                 {VPUNN::Cycles::NO_ERROR, false, 240000, 240000 + 33000},  // //v150 272k
                 "CLUSTERING ,flt , sparse "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, false, 130000, 130000 + 20000},  //    //v150 134k
                 "SOK , no memmove, sparse"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, false, 160000, 160000 + 20000},  //    //v150 175k
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
                 {NO_ERROR_EXPECTED, false, 270000, 270000 + 20000},
                 "SOK , no memmove"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
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
                 {VPUNN::Cycles::NO_ERROR, false, 250000, 250000 + 30000},
                 "CLUSTERING , int "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {NO_ERROR_EXPECTED, false, 130000, 130000 + 20000},
                 "SOK , no memmove"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, false, 160000, 160000 + 20000},
                 "SOHO , no memmove"},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, CONVOLUTION_Concrete_Multiply6326_FI_VPU27) {
    const VPUNN::DPULayer tst_layer_ref(
            VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(60, 8, 512, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(60, 6, 512, 1, VPUNN::DataType::UINT8)},    // output dimensions
            {3, 3},                                                       // kernels
            {1, 1},                                                       // strides
            {0, 0, 1, 1}                                                  // padding
    );
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
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 300000, 300000 + 55000},  //
                 "SOH , no memmove, sparse"},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, Fused_234_3xELEMENTWISE) {
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

    // EXPECT_TRUE(false);

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
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 30000, 30000 + 10000},
                 t + "SOH ,0m"},
        };
        executeTests(tests);
    }

    {  // UINT8 all, Layer 1 changed
        std::string t{"All UINT8 @ Layer 1 "};
        auto tst_layer = l1Int_ref;
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 36000, 36000 + 4000},  //
                 t + "CLU 0m "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 0, 0 + 0},
                 t + "SOK ,0m"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 21000, 21000 + 4000},  //
                 t + "SOH ,0m"},
        };
        executeTests(tests);
    }

    {  // FLOAT to INT all, Layer 1 changed, no more in place output possible , output is stand alone
        std::string t{"Reversed F toI @ Layer 1  mixed "};
        auto tst_layer = l1FI_ref;
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0 + 0},  //
                 t + "CLU 0m "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 0, 0 + 0},
                 t + "SOK ,0m"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 22000, 22000 + 6500},  // huge v16 24k v17 22323   GT??  v159NN:28370
                 t + "SOHO ,0m"},
        };
        executeTests(tests);
    }

    {  // L1&2 int -float. Has to be IMPROVED,  no more in place output possible
        std::string t{"Original Layer 1, mixed "};
        auto tst_layer = l1and2_ref;
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 40000, 40000 + 5000},  // 44608
                 t + "CLU 0m "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 0, 0 + 0},
                 t + "SOK ,0m"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 35000, 35000 + 5000},  // v16 38k , v17 37k   //v159 24k  GT??
                 t + "SOHO ,0m"},
        };
        executeTests(tests);
    }

    {  // L3  int -float. Has to be IMPROVED , no more in place output possible
        std::string t{"Original Layer 3, mixed "};
        auto tst_layer = l3_ref;
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 40000, 40000 + 5000},  // 44389
                 t + "CLU 0m "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 0, 0 + 0},
                 t + "SOK ,0m"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 35000, 35000 + 5000},  // v16 37k  v17 37k //v159 24k GT??
                 t + "SOHO ,0m"},
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

    const auto origMax = getModel(VPUDevice::VPU_2_7).get_maxWorkloadsPerIntraTileSplit();
    getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(50);  // original test context was 50

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

    getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(origMax);
}

TEST_F(VPULayerCostModelTest, Default_MaxWorkloadSPlitAndDetails_Test) {
    std::vector<const VPULayerCostModel*> all_models{
            &model_2_0, &model_2_7, &model_2_7_no_dma, &model_4_0,
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
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = true;
        tst_layer.weight_sparsity = 0.950016f;

        const std::vector<TestCase> tests{
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                // {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                // "CLUSTERING ,flt , sparse "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
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
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = true;
        tst_layer.weight_sparsity = 0.880725f;

        const std::vector<TestCase> tests{
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                // {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                // "CLUSTERING ,flt , sparse "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
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
        auto tst_layer = tst_layer_ref2;
        tst_layer.weight_sparsity_enabled = true;
        tst_layer.weight_sparsity = 0.880725f;

        const std::vector<TestCase> tests{
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
                // {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                // "CLUSTERING ,flt , sparse "},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
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
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
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
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
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
        auto tst_layer = tst_layer_ref;
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
        auto tst_layer = tst_layer_ref;
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
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
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
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
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
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
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
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 100000, 100000 + 17000},  // v16 104000 , v17 114995
                 "SOH , no memmove, "},
        };

        executeTests(tests);
    }
}

TEST_F(VPULayerCostModelTest, ELTWISE_diff_swizz_NPU40) {
    auto generate_wl = [](unsigned int ch, ExecutionMode exec, Swizzling in0, Swizzling in1, Swizzling out0) {
        DPUWorkload wl{
                VPUNN::VPUDevice::VPU_4_0,
                Operation::ELTWISE,
                {VPUTensor(112, ch, 32, 1, DataType::UINT8, Layout::ZXY)},  // input dimensions
                {VPUTensor(112, ch, 32, 1, DataType::UINT8, Layout::ZXY)},  // output dimensions
                {1, 1},                                                     // kernels
                {1, 1},                                                     // strides
                {0, 0, 0, 0},                                               // padding
                exec,                                                       // execution mode
                ActivationFunction::RELU,                                   // activation
                0.0F,                                                       // act_sparsity
                0.0F,                                                       // weight_sparsity
                {in0, in1},                                                 // input_swizzling
                {out0},                                                     // output_swizzling
                1,                                                          // output_write_tiles
                {0, 0, 0, 0},                                               // offsets
                ISIStrategy::CLUSTERING,                                    // isi_strategy
                false,                                                      // weight_sparsity_enabled
        };
        return wl;
    };
    auto test_message = [](DPUWorkload wl, std::string text) {
        // clang-format off
        std::string message = text +
                              " Operation:" + Operation_ToText.at(static_cast<int>(wl.op)) + 
                              " input_swizzling: {" +
                               Swizzling_ToText.at(static_cast<int>(wl.input_swizzling[0])) + ", "
                             + Swizzling_ToText.at(static_cast<int>(wl.input_swizzling[1])) + "} ;\n"  

                             + " output_swizzling: " 
                             + " :  {" + Swizzling_ToText.at(static_cast<int>(wl.output_swizzling[0])) + "} \n" ;

        // clang-format on

        return message;
    };

    DPUWorkload wl_swizz_000{
            generate_wl(16, ExecutionMode::CUBOID_16x16, Swizzling::KEY_0, Swizzling::KEY_0, Swizzling::KEY_0)};
    DPUWorkload wl_swizz_555{
            generate_wl(16, ExecutionMode::CUBOID_16x16, Swizzling::KEY_5, Swizzling::KEY_5, Swizzling::KEY_5)};
    DPUWorkload wl_swizz_005{
            generate_wl(16, ExecutionMode::CUBOID_16x16, Swizzling::KEY_0, Swizzling::KEY_0, Swizzling::KEY_5)};
    DPUWorkload wl_swizz_550{
            generate_wl(16, ExecutionMode::CUBOID_16x16, Swizzling::KEY_5, Swizzling::KEY_5, Swizzling::KEY_0)};
    DPUWorkload wl_swizz_055{
            generate_wl(16, ExecutionMode::CUBOID_16x16, Swizzling::KEY_0, Swizzling::KEY_5, Swizzling::KEY_5)};

    const bool prefetch{true};
    VPULayerCostModel& theModel = model_4_0;
    const std::string nline{
            "\n ------------------------------------------- TEST ------------------------------------------- \n"};
    const bool force_fail{false};
    show_split = false;

    auto run_layer = [=, &theModel](const DPUWorkload& wl, const VPUTilingStrategy tilStrtgy, std::string text) {
        std::cout << nline << " " << text;
        DPULayer tst_layer(wl);
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
    run_layer(wl_swizz_000, VPUTilingStrategy::SOH_Overlapped, test_message(wl_swizz_000, "SOHO "));  // 878
    run_layer(wl_swizz_000, VPUTilingStrategy::NONE, test_message(wl_swizz_000, "NONE "));            // 1914

    run_layer(wl_swizz_555, VPUTilingStrategy::SOH_Overlapped, test_message(wl_swizz_555, "SOHO "));  // 954
    run_layer(wl_swizz_555, VPUTilingStrategy::NONE, test_message(wl_swizz_555, "NONE "));            // 1782

    run_layer(wl_swizz_005, VPUTilingStrategy::SOH_Overlapped, test_message(wl_swizz_005, "SOHO "));  // 891
    run_layer(wl_swizz_005, VPUTilingStrategy::NONE, test_message(wl_swizz_005, "NONE "));            // 1775

    run_layer(wl_swizz_550, VPUTilingStrategy::SOH_Overlapped, test_message(wl_swizz_550, "SOHO "));  // 857
    run_layer(wl_swizz_550, VPUTilingStrategy::NONE, test_message(wl_swizz_550, "NONE "));            // 1767

    run_layer(wl_swizz_055, VPUTilingStrategy::SOH_Overlapped, test_message(wl_swizz_055, "SOHO "));  // 891
    run_layer(wl_swizz_055, VPUTilingStrategy::NONE, test_message(wl_swizz_055, "NONE "));            // 1775

    // tiles
    DPUWorkload tile_swizz_000{
            generate_wl(8, ExecutionMode::CUBOID_8x16, Swizzling::KEY_0, Swizzling::KEY_0, Swizzling::KEY_0)};
    DPUWorkload tile_swizz_555{
            generate_wl(8, ExecutionMode::CUBOID_8x16, Swizzling::KEY_5, Swizzling::KEY_5, Swizzling::KEY_5)};
    DPUWorkload tile_swizz_005{
            generate_wl(8, ExecutionMode::CUBOID_8x16, Swizzling::KEY_0, Swizzling::KEY_0, Swizzling::KEY_5)};
    DPUWorkload tile_swizz_550{
            generate_wl(8, ExecutionMode::CUBOID_8x16, Swizzling::KEY_5, Swizzling::KEY_5, Swizzling::KEY_0)};
    DPUWorkload tile_swizz_055{
            generate_wl(8, ExecutionMode::CUBOID_8x16, Swizzling::KEY_0, Swizzling::KEY_5, Swizzling::KEY_5)};

    auto run_dpu_wl = [=, &theModel](const DPUWorkload& wl, std::string text) {
        std::cout << nline << " " << text;
        std::string err_info;
        DPUWorkload tst_wl{wl};
        CyclesInterfaceType dpu_cost = theModel.DPU(tst_wl, err_info);

        if (force_fail) {
            EXPECT_EQ(dpu_cost, 10) << dpu_cost << "=" << Cycles::toErrorText(dpu_cost) << tst_wl << err_info;
        }

        std::cout << " \n:" << text << " ,cyc: " << dpu_cost << " = " << Cycles::toErrorText(dpu_cost);
    };

    run_dpu_wl(tile_swizz_000, test_message(tile_swizz_000, "CLU "));  // 878
    run_dpu_wl(tile_swizz_555, test_message(tile_swizz_555, "CLU "));  // 954
    run_dpu_wl(tile_swizz_005, test_message(tile_swizz_005, "CLU "));  // 891
    run_dpu_wl(tile_swizz_550, test_message(tile_swizz_550, "CLU "));  // 857
    run_dpu_wl(tile_swizz_055, test_message(tile_swizz_055, "CLU "));  // 954
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
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
                 "SOHO , no memmove, "},
        };

        executeTests(tests);
    }
    {
        auto tst_layer = tst_layer_ref;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_HaloRead, false, false, true}},
                 {Cycles::NO_ERROR, true, 600000, 600000 + 500000 + 350000},  // huge delta , what is the GT
                 // v17: :newmemtens 1390740
                 "SOH w in Halo , no memmove, "},
        };

        executeTests(tests);
    }
    {  // details
        DPULayer tst_layer(tst_layer_ref);
        const VPULayerStrategy strategy{1U, 1U, 2U, VPUTilingStrategy::SOH_HaloRead, false, false, true};

        VPULayerCostModel& theModel{getModel(tst_layer.device)};

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
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                 {Cycles::NO_ERROR, true, 19000, 19000 + 2000},
                 "SOHO , no memmove, "},
        };

        executeTests(tests);
    }
    {
        auto tst_layer = tst_layer_ref;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_HaloRead, false, false, true}},
                 {Cycles::NO_ERROR, true, 22000, 22000 + 1200},  // 604935 split 15
                 "SOH w in Halo , no memmove, "},
        };

        executeTests(tests);
    }
    {  // details
        DPULayer tst_layer(tst_layer_ref);
        const VPULayerStrategy strategy{1U, 1U, 2U, VPUTilingStrategy::SOH_HaloRead, false, false, true};

        VPULayerCostModel& theModel{getModel(tst_layer.device)};

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
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
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
        auto tst_layer = tst_layer_ref;
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
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
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
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, true}},
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
        auto tst_layer = tst_layer_ref;
        tst_layer.weight_sparsity_enabled = false;
        tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
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
        auto tst_layer = tst_layer_ref;
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
        auto tst_layer = tst_layer_ref;
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
        auto tst_layer = tst_layer_ref;
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
        auto tst_layer = tst_layer_ref;
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
        tin.strategy.tiling_strategy = VPUNN::VPUTilingStrategy::SOH_Overlapped;

        TestExpectations texp{VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0};

        DoRegularTest(tin, texp, "SOH with errors");
    }

    const std::vector<TestCase> tests{
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
             {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
             "CLUSTERING , + memmove, with errors"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
             {NO_ERROR_EXPECTED, true, 30000 + 57000, 30000 + 57000 + 3500},
             "SOK , + memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, false}},
             {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0},
             "SOH ,+ memmove, with errors"},
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
                 {VPUNN::Cycles::NO_ERROR, false, 260000 + 20000, 260000 + 20000 + 55000},
                 "SOK , no memmove, sparse"},
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, false}},
                 {VPUNN::Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0 + 1000},  // was SOH no overlap  : 327500
                 "SOH , no memmove, sparse"},
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
        auto tst_layer = tst_layer_ref;
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
        auto tst_layer = tst_layer_ref;
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
        auto tst_layer = tst_layer_ref;
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
        auto tst_layer = l1F_ref;
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
        auto tst_layer = l1Int_ref;
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
        auto tst_layer = l1FI_ref;
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
        auto tst_layer = l1and2_ref;
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
        auto tst_layer = l3_ref;
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
        TestInput tin{tst_layer, tst_strategy};
        tin.strategy.tiling_strategy = VPUNN::VPUTilingStrategy::SOK;  // SOK is now allowed for element wise

        TestExpectations texp{NO_ERROR_EXPECTED, true, 26000, 26000 + 8000};

        DoRegularTest(tin, texp, "SOK tentative on elementwise, memove");
    }
}
TEST_F(VPULayerCostModelTest, MAXPOOLPrefetchTest_NPU27) {
    const VPUNN::DPULayer tst_layer(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::MAXPOOL,
                                    {VPUNN::VPUTensor(7, 7, 2048, 1, VPUNN::DataType::UINT8)},  // input dimensions
                                    {VPUNN::VPUTensor(1, 1, 2048, 1, VPUNN::DataType::UINT8)},  // output dimensions
                                    {7, 7},                                                     // kernels
                                    {1, 1},                                                     // strides
                                    {0, 0, 0, 0}                                                // padding
    );
    show_split = true;
    const std::vector<TestCase> tests{
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
             {NO_ERROR_EXPECTED, false, 52000 + 3000, 52000 + 3000 + 18000},
             "CLUSTERING , no mem"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
             {NO_ERROR_EXPECTED, false, 13000 + 2000, 13000 + 2000 + 7000},  // without prefetch 13000, 13000 + 1000
             "SOK , no mem"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, false}},
             {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 0, 0 + 0},  // 7 cannot be split, must be 14
             "SOH ,no mem"},
    };

    const auto origMax = getModel(VPUDevice::VPU_2_7).get_maxWorkloadsPerIntraTileSplit();
    getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(50);  // original test context was 50
    executeTests(tests);
    getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(origMax);
}

TEST_F(VPULayerCostModelTest, AVEPOOLPrefetchTest_172_NPU27) {
    const VPUNN::DPULayer tst_layer(VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::AVEPOOL,        // goes to DW Conv
                                    {VPUNN::VPUTensor(7, 15, 2048, 1, VPUNN::DataType::UINT8)},  // input dimensions
                                    {VPUNN::VPUTensor(1, 9, 2048, 1, VPUNN::DataType::UINT8)},   // output dimensions
                                    {7, 7},                                                      // kernels
                                    {1, 1},                                                      // strides
                                    {0, 0, 0, 0}                                                 // padding
    );

    const std::vector<TestCase> tests{
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, false}},
             {VPUNN::Cycles::NO_ERROR, false, 52000 + 10000, 52000 + 10000 + 7000},
             "CLUSTERING , + memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, false}},
             {NO_ERROR_EXPECTED, false, 20000 + 5000, 20000 + 5000 + 5000},
             "SOK , + memmove"},
            {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, false}},
             {VPUNN::Cycles::NO_ERROR, true, 51000 + 9000, 51000 + 9000 + 6000},  //
             "SOHO ,+ memmove"},
    };
    const auto origMax = getModel(VPUDevice::VPU_2_7).get_maxWorkloadsPerIntraTileSplit();
    getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(50);  // original test context was 50
    executeTests(tests);
    getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(origMax);
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
            {swz_def, swz_def},                                            // input_swizzling
            {swz_def},                                                     // output_swizzling
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
            {swz_def, swz_def},                                            // input_swizzling
            {swz_def},                                                     // output_swizzling
            1,                                                             // output_write_tiles
            {0, 0, 0, 0},                                                  // offsets
            VPUNN::ISIStrategy::CLUSTERING,                                // isi_strategy
            true,                                                          // weight_sparsity_enabled
    };

    // const std::string altModelName(NameHelperNN::get_model_root() + "vpu_2_7-150.vpunn");
    // VPULayerCostModel model_alternative{altModelName};
    const bool prefetch{true};
    VPULayerCostModel& theModel = model_2_7;
    // EXPECT_TRUE(false);
    //  GT??  are these corner cases?
    {
        const VPUNN::DPULayer tst_layer(wl_h5);
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 250000,
                  250000 + 30000},  // v16 251k  //v17 279k, same as CLU H3 below, v159NN:263125
                 "small H(5) SOHO , no memmove, "},

                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_HaloRead, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 300000,
                  320000 + 100000},  // huge  GT??  v16 324k    v17 373k(!!!): 396149  //v159 (SOHO) 260k  out of
                                     // reasonable range()
                 "small H(5) SOH H , no memmove, "},

                {{tst_layer, {1U, 1U, 2U, VPUTilingStrategy::NONE, false, false, prefetch}},
                 {Cycles::ERROR_INPUT_TOO_BIG, true, 0, 0 + 0},  //
                 " small H(5) CLUSTERING , no memmove, "},

                {{tst_layer, {1U, 1U, 2U, VPUTilingStrategy::SOK, false, false, prefetch}},
                 {Cycles::NO_ERROR, true, 137000, 137000 + 11000},  // v17:146427    v159NN:138370
                 " small H(5) SOK (added for  completion), no memmove, "},

                // Note : SOHO reverses with SOHH with v17 ?,
                // note: SOK wins always ahead of SOHO (stability OK), huge deltas
        };
        executeTests(tests);
    }
    {
        const DPULayer tst_layer(wl_h3);
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {Cycles::NO_ERROR, true, 240000, 240000 + 48000},  // v16 249k,   v17 280K   GT?? v159nn:260k
                 "very small H(3) SOHO , no memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUTilingStrategy::SOH_HaloRead, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 300000,
                  300000 + 100000},  // v16 314k,   v17 360k:388k  //v159NN (SOHO) 253k  huge Gt??
                 "very small H(3) SOH H , no memmove, "},
                {{tst_layer, {1U, 1U, 2U, VPUTilingStrategy::NONE, false, false, prefetch}},
                 {Cycles::NO_ERROR, true, 250000, 250000 + 30000},  // v16 251k,   v17 276k    //v159 261k
                 "very small H(3) CLUSTERING , no memmove, "},

                {{tst_layer, {1U, 1U, 2U, VPUTilingStrategy::SOK, false, false, prefetch}},
                 {Cycles::NO_ERROR, true, 131000, 131000 + 15000},  // v17 144K  //v159NN:133245 Gt??
                 "very small H(3) SOK (added extra) , no memmove, "},

                // Note: SOK wins always ahead of SOHO (stability OK)
        };
        executeTests(tests);
    }
    //{
    //    VPUNN::DPULayer tst_layer(wl_h5);
    //    const VPULayerStrategy strategy{1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false,
    //    prefetch};

    //    Logger::clear2ndlog();
    //    unsigned cost_cyc{};
    //    LayerSplitInfo detailed_split;
    //    ASSERT_NO_THROW(cost_cyc = theModel.Layer(tst_layer, VPUNN::VPUTilingStrategy::SOH_Overlapped, 1U, 2U,
    //    false, false,
    //                                              prefetch, detailed_split))
    //            << tst_layer << strategy;

    //    EXPECT_EQ(cost_cyc, 1) << tst_layer << toStringLayerSplitInfo(detailed_split) << Logger::get2ndlog();

    //    std::string err_info;
    //    CyclesInterfaceType dpu_cost = theModel.DPU(tst_layer, err_info);
    //    EXPECT_EQ(dpu_cost, 2) << dpu_cost << "=" << Cycles::toErrorText(dpu_cost) << tst_layer << err_info;
    //}

    {
        VPUNN::DPULayer tst_layer(wl_h3);
        const VPULayerStrategy strategy{1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch};

        Logger::clear2ndlog();
        unsigned cost_cyc{};
        LayerSplitInfo detailed_split;
        ASSERT_NO_THROW(cost_cyc = theModel.Layer(tst_layer, VPUNN::VPUTilingStrategy::SOH_Overlapped, 1U, 2U, false,
                                                  false, prefetch, detailed_split))
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

// use this as a template for investigation tests
TEST_F(VPULayerCostModelTest, DISABLED_Z_InvestigationTest) {
    const VPUNN::DPUWorkload wl_1{
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(1019, 5, 16, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(127, 1, 768, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            {11, 5},                                                       // kernels
            {8, 8},                                                        // strides
            {0, 0, 0, 0},                                                  // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                            // execution mode
            VPUNN::ActivationFunction::NONE,                               // activation
            0.0F,                                                          // act_sparsity
            0.0F,                                                          // weight_sparsity
            {swz_def, swz_def},                                            // input_swizzling
            {swz_def},                                                     // output_swizzling
            1,                                                             // output_write_tiles
            {0, 0, 0, 0},                                                  // offsets
            VPUNN::ISIStrategy::CLUSTERING,                                // isi_strategy
            false,                                                         // weight_sparsity_enabled
    };

    const VPUNN::DPULayer tst_layer_ref(
            wl_1
            // VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
            //{VPUNN::VPUTensor(1019, 5, 16, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            //{VPUNN::VPUTensor(127, 1, 768, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            //{11, 5},                                                       // kernels
            //{8, 8},                                                        // strides
            //{0, 0, 0, 0}                                                   // padding
    );

    VPULayerCostModel& theModel = model_2_7;
    Logger::clear2ndlog();

    {
        auto tst_layer = tst_layer_ref;
        // tst_layer.weight_sparsity_enabled = false;
        // tst_layer.weight_sparsity = 0.0f;

        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                 {VPUNN::Cycles::NO_ERROR, true, 0, 0 + 0},
                 "CLUSTERING , ,  "},
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true}},
                // {VPUNN::Cycles::NO_ERROR, true, 2000, 2000 + 1000},  //
                // "SOK , no memmove, "},
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true}},
                // {VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, true, 7900,
                //  7900 + 1000},  // H=8 with K=8 , no padding cannot be split on h
                // "SOH , no memmove, "},
        };

        executeTests(tests);
    }

    // layer aspects
    {
        DPUWorkload tst_wl{wl_1};
        {
            DPULayer tst_layer(tst_wl);
            const VPULayerStrategy strategy{1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOK, false, false, true};

            Logger::clear2ndlog();
            unsigned cost_cyc{};
            LayerSplitInfo detailed_split;
            ASSERT_NO_THROW(cost_cyc = theModel.Layer(tst_layer, VPUNN::VPUTilingStrategy::SOK, 1U, 2U, false, false,
                                                      true, detailed_split))
                    << tst_layer << strategy;

            EXPECT_EQ(cost_cyc, 1) << tst_layer << toStringLayerSplitInfo(detailed_split) << "\n 2ndLOG:\n"
                                   << Logger::get2ndlog();
        }
        Logger::clear2ndlog();
        std::string err_info;
        CyclesInterfaceType dpu_cost = theModel.DPU(tst_wl, err_info);
        EXPECT_EQ(dpu_cost, 2) << dpu_cost << "=" << Cycles::toErrorText(dpu_cost) << tst_wl << err_info
                               << "\n 2ndLOG:\n"
                               << Logger::get2ndlog();
    }
}

TEST_F(VPULayerCostModelTest, Extreme_values_Layer_Test_NPU40) {
    const VPUNN::DPUWorkload wl = {
            VPUNN::VPUDevice::VPU_4_0,
            VPUNN::Operation::ELTWISE,
            {VPUNN::VPUTensor(0, 0, 0, 1, VPUNN::DataType::UINT8, Layout::ZXY)},  // input dimensions
            {VPUNN::VPUTensor(0, 0, 0, 1, VPUNN::DataType::UINT8, Layout::ZXY)},  // output dimensions
            {0, 0},                                                               // kernels
            {0, 0},                                                               // strides
            {0, 0, 0, 0},                                                         // padding
            VPUNN::ExecutionMode::CUBOID_8x16,                                    // execution mode
            VPUNN::ActivationFunction::NONE,                                      // activation
            0.0F,                                                                 // act_sparsity
            0.0F,                                                                 // weight_sparsity
            {Swizzling::KEY_5, Swizzling::KEY_5},                                 // input_swizzling
            {Swizzling::KEY_5},                                                   // output_swizzling
            0,                                                                    // output_write_tiles
            {0, 0, 0, 0},                                                         // offsets
            VPUNN::ISIStrategy::CLUSTERING,                                       // isi_strategy
            true,                                                                 // weight_sparsity_enabled

    };

    VPULayerCostModel& theModel = model_4_0;
    Logger::clear2ndlog();
    DPULayer tst_layer(wl);

    unsigned cyc{};
    LayerSplitInfo detailed_split;

    cyc = theModel.Layer(tst_layer, VPUNN::VPUTilingStrategy::SOH_Overlapped, 1U, 2U, false, false, true,
                         detailed_split);
    EXPECT_TRUE(Cycles::isErrorCode(cyc));
    // EXPECT_TRUE(false) << Cycles::toErrorText(cyc);
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
    for (const auto theModel :
         std::vector<VPULayerCostModel*>{&model_2_7, &model_2_7_no_dma})  // check fallback to DMA theoretical as well
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
        auto cmxddr_dma = theModel->DMA(tst_layer.device, outT, outT, MemoryLocation::CMX, MemoryLocation::DRAM);
        auto ddrcmx_dma = theModel->DMA(tst_layer.device, outT, outT, MemoryLocation::DRAM, MemoryLocation::CMX);
        EXPECT_EQ(ddrcmx_dma, cmxddr_dma) << ddrcmx_dma << "\n"
                                          << cmxddr_dma << "\n"
                                          << cost_cyc_LayerFromDDR << "\n"
                                          << cost_cyc_LayerFromCMX << "\n"
                                          << cost_cyc_LayerNoMem << "\n";
    }
}

TEST_F(VPULayerCostModelTest, Dual_Sparsity_Active_Layer_Test_NPU40) {
    // VPUNN::DPUWorkload wl_ref_2_7 = {
    //         VPUNN::VPUDevice::VPU_2_7,
    //         VPUNN::Operation::CONVOLUTION,
    //         {VPUNN::VPUTensor(5, 5, 100, 1, VPUNN::DataType::UINT8)},  // input dimensions
    //         {VPUNN::VPUTensor(3, 3, 50, 1, VPUNN::DataType::UINT8)},   // output dimensions
    //         {3, 3},                                                    // kernels
    //         {1, 1},                                                    // strides
    //         {0, 0, 0, 0},                                              // padding
    //         VPUNN::ExecutionMode::CUBOID_16x16,                        // execution mode
    //         VPUNN::ActivationFunction::NONE,                           // activation
    //         0.0F,                                                      // act_sparsity
    //         0.0F,                                                      // weight_sparsity
    //         {swz_def, swz_def},                                        // input_swizzling
    //         {swz_def},                                                 // output_swizzling
    //         1,                                                         // output_write_tiles
    //         {0, 0, 0, 0},                                              // offsets
    //         VPUNN::ISIStrategy::CLUSTERING,                            // isi_strategy
    //         false,                                                     // weight_sparsity_enabled
    // };

    VPUNN::DPUWorkload wl_ref_4_0{
            VPUNN::VPUDevice::VPU_4_0,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(64, 64, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(21, 21, 16, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                     // kernels
            {3, 3},                                                     // strides
            {0, 0, 0, 0},                                               // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                         // execution mode
            VPUNN::ActivationFunction::NONE,                            // activation
            0.0F,                                                       // act_sparsity
            0.0F,                                                       // weight_sparsity
            {swz_def, swz_def},                                         // input_swizzling
            {swz_def},                                                  // output_swizzling
            1,                                                          // output_write_tiles
            {0, 0, 0, 0},                                               // offsets
            VPUNN::ISIStrategy::CLUSTERING,                             // isi_strategy
            false                                                       // weight_sparsity_enabled
    };

    struct TestInput {
        VPUNN::DPUWorkload wl;  // the wl for which we compute memory
    };

    struct TestCase {
        TestInput t_in;
        std::string info = "";
    };

    const VPULayerStrategy strategy{1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, true};
    auto verify_sparsity_influence = [&strategy](const TestCase& t, VPULayerCostModel& theModel) {
        DPULayer tst_layer(t.t_in.wl);
        Logger::clear2ndlog();
        CyclesInterfaceType cost_cyc{};
        LayerSplitInfo detailed_split;

        ASSERT_NO_THROW(cost_cyc = theModel.Layer(tst_layer, strategy.tiling_strategy, strategy.nDPUs, strategy.nTiles,
                                                  false, false, true, detailed_split))
                << tst_layer << strategy << cost_cyc;

        EXPECT_FALSE(Cycles::isErrorCode(cost_cyc)) << t.info;
    };

    ///@brief this lambda function executes a given lambda function as a parameter on each test case in a test vector
    ///@param tests a test vector
    ///@param testChecker is a lambda function
    auto run_Tests = [](const std::vector<TestCase>& tests, VPULayerCostModel& theModel, auto testCheck) {
        for (const auto& t : tests) {
            testCheck(t, theModel);
        }
    };

    {
        const std::vector<TestCase> tests4_0 = {
                // clang-format off
            /************************************************ TABLE HEADER ********************************************************/
        /*  ||                              workload                          ||         test info         ||   */

            {{wl_sparsity_initialization(wl_ref_4_0,  false, 0.0F, false, 0.0F) }, "Device 4_0: No sparsity active" },
            {{wl_sparsity_initialization(wl_ref_4_0,  true, 0.7F, false, 0.0F)}, "Device 4_0: Input sparsity active"},
            {{wl_sparsity_initialization(wl_ref_4_0,  false, 0.0F, true, 0.6F)}, "Device 4_0: Weight sparsity active" },
            {{wl_sparsity_initialization(wl_ref_4_0, true, 0.2F, true, 0.4F)}, "Device 4_0: Input + Weight sparsity"},
            {{wl_sparsity_initialization(wl_ref_4_0, true, 0.2F, true, 0.4F)}, "Device 4_0: Input + Weight sparsity"},

                // clang-format on
        };

        VPULayerCostModel& theModel = model_4_0;
        run_Tests(tests4_0, theModel, verify_sparsity_influence);
    }
}

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

TEST_F(VPULayerCM_InvestigationTest, MEXP_C2_ELTWISE_1662_EISXW_126389_NPU40) {
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

    VPULayerCostModel& theModel = this->model_4_0;

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
TEST_F(VPULayerCM_InvestigationTest, MEXP_C2_CONV_4634_4662_EISXW_126389_NPU40) {
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

    VPULayerCostModel& theModel = this->model_4_0;
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

        CyclesInterfaceType dpu_cost = theModel.DPU(tst_layer, err_info);
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
TEST_F(VPULayerCM_InvestigationTest, MEXP_C2_CONV_4648_4676_EISXW_126389_NPU40) {
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

    VPULayerCostModel& theModel = this->model_4_0;

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
TEST_F(VPULayerCM_InvestigationTest, DWConv_SOK_SOH_Comparison_EISXW_92399) {
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

    VPULayerCostModel& theModel = model_2_7;

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

        CyclesInterfaceType dpu_cost = theModel.DPU(tst_layer, err_info);
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

TEST_F(VPULayerCM_InvestigationTest, CONV_TILE_Teammate_EISXW_9xxxxx) {
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

TEST_F(VPULayerCM_InvestigationTest, RuntimeELT_CONV_SOH_SOK_EISXW_98656) {
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
        return wl_elm_layer;
    };

    // element wise SWIZZ in 0 at input and 5 at output in the real world.Cannot simulate mix swizzlings, not
    // trained
    const DPUWorkload wl_elm_layer{gen_eltwise(14, 14, 1024, ISIStrategy::CLUSTERING, 1, ExecutionMode::CUBOID_16x16)};
    const DPUWorkload wl_conv_layer{
            gen_conv(14, 14, 1024, 256, ISIStrategy::CLUSTERING, 1, ExecutionMode::CUBOID_16x16)};

    const bool prefetch{true};
    VPULayerCostModel& theModel = model_2_7;
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
        std::cout << nline << " " << text;
        std::string err_info;
        DPUWorkload tst_wl{wl};
        CyclesInterfaceType dpu_cost = theModel.DPU(tst_wl, err_info);

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

TEST_F(VPULayerCM_InvestigationTest, MoreTiles_MAXP_EISXW_99246_NPU40) {
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
    VPULayerCostModel& theModel = model_4_0;

    {
        const VPUNN::DPULayer tst_layer(wl_MXP_layer);

        const std::vector<TestCase> tests{
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::NONE, false, false, prefetch}},
                // {VPUNN::Cycles::NO_ERROR, true, 21000, 21000 + 1100},  // v17 21341   v159NN:22003
                // "CLU /2, no memmove, "},
                //{{tst_layer, {1U, 1U, 2U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                // {VPUNN::Cycles::NO_ERROR, true, 11000, 11000 + 1000},  // v17 11164    v159NN:11612
                // "SOHO /2, no memmove, "},

                {{tst_layer, {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
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
    auto verify_cost_cyc = [nline, &wl_MXP_layer, prefetch, &theModel](unsigned int nTiles,
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

TEST_F(VPULayerCM_InvestigationTest, CONV_Act_sparsity_EISXW_117195_INT_NPU40) {
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

    VPULayerCostModel& theModel = this->model_4_0;

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

TEST_F(VPULayerCM_InvestigationTest, DWConv_SOK_SOH_decision_EISXW_117314_2d_3_NPU40) {
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

    VPULayerCostModel& theModel = this->model_4_0;

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
TEST_F(VPULayerCM_InvestigationTest, SOHO_ELTWISE_EISXW_127594_Teammate10June_NPU40) {
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
                {{tst_layer_28, {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 2655 - 260, fail * (3200 + 384)},  //  4T , 4x8: 3484 GTLNL:
                 "SOHO 28 , no memmove, "},
                {{tst_layer_27, {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 2655 - 260, fail * (3498 + 349)},  //  4T , 3x7+1x6:3498  GTLNL:
                 "SOHO 27 , no memmove, "},

                // note:?? wins
        };
        executeTests(tests);
    }

    VPULayerCostModel& theModel = getModel(VPUDevice::VPU_4_0);

    auto case_run = [=, &theModel](const DPUWorkload& workload, const std::string& whatTest) {
        DPUWorkload tst_layer{workload};
        std::string err_info;

        CyclesInterfaceType dpu_cost = theModel.DPU(tst_layer, err_info);
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

///
///
TEST_F(VPULayerCM_InvestigationTest, Model_N_v1_CONV_EISXW_127644_NPU40) {
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
                {{tst_layer, {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 3500,
                  fail * 3501 + 1000},  // V17: 3889 *2 = 7778    v159NN: 4234 *2=8468 GTVPUX:8400ns (2x 4200ns)
                                        // GTL:3811 *2 = 7622
                 "SOHO half K"},

        };
        executeTests(tests);
    }
    // note:v17: SOHO wins,  (not split in half)
    // v159 SOHO wins also in the split way K80x2

    VPULayerCostModel& theModel = this->model_4_0;

    auto case_run = [=, &theModel](const DPUWorkload& workload, const std::string& whatTest) {
        DPUWorkload tst_layer(workload);
        std::string err_info;

        CyclesInterfaceType dpu_cost = theModel.DPU(tst_layer, err_info);
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
TEST_F(VPULayerCM_InvestigationTest, Model_E_v9_CONV_EISXW_127649_NPU40) {
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

    VPULayerCostModel& theModel = this->model_4_0;

    auto case_run = [=, &theModel](const DPUWorkload& workload, const std::string& whatTest) {
        DPUWorkload tst_layer(workload);
        std::string err_info;

        CyclesInterfaceType dpu_cost = theModel.DPU(tst_layer, err_info);
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
        DPUWorkload conv4_SOHO_T{conv4s};
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
        DPUWorkload conv8_SOK_K16{conv8};
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
            {VPUTensor(11, 3, 128, 1, DataType::FLOAT16)},  // input dimensions
            {VPUTensor(11, 3, 13872, 1, DataType::FLOAT16)},  // output dimensions
            {1, 1},                                        // kernels
            {1, 1},                                        // strides
            {0, 0, 0, 0},                                  // padding
            ExecutionMode::CUBOID_16x16,                   // execution mode
            ActivationFunction::NONE,                      // activation
            0.0F,                                          // act_sparsity
            0.0F,                                          // weight_sparsity
            {Swizzling::KEY_0, Swizzling::KEY_0},             // input_swizzling
            {Swizzling::KEY_0},                               // output_swizzling
            1,                                             // output_write_tiles
            {0, 0, 0, 0},                                  // offsets
            ISIStrategy::CLUSTERING,                       // isi_strategy
            false,                                         // weight_sparsity_enabled
    };

    const bool prefetch{true};

    {
        const DPULayer tst_layer(wl_);
        const std::vector<TestCase> tests{
                {{tst_layer, {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOK, false, false, prefetch}},
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

TEST_F(VPULayerCM_InvestigationTest, WhisperFP16_BIG_CONV_EISXW_131119_NPU40) {
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
                {{tst_layer, {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOK_NO_BROADCAST, false, false, prefetch}},
                 {Cycles::NO_ERROR, true, 335000, fail * 370000 + 30000},  // 2// V17:384288   v159NN: ?   GTL: ?? todo
                 "SOK no broadcast , no memmove, "},

                // note:?? wins
        };
        executeTests(tests);
    }

    VPULayerCostModel& theModel = this->model_4_0;

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
                DPUWorkload wl = wl_SOK_TOP_CLU_16x;
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
                DPUWorkload wl = wl_SOK_TOP_halo_16x;
                EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;
                EXPECT_GE(mem.cmx, (1024 * 1024 + 1024 * 512)) << "\nMemory size: " << mem << wl << std::endl;
                std::cout << mem << std::endl;
            }
            {
                std::cout << "\n   ---- FULL LAYER  no halo  ------- ";
                DPUWorkload wl = wl_;
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
                        {{tst_layer, {1U, 1U, 4U, VPUNN::VPUTilingStrategy::NONE, false, false, prefetch}},
                         {Cycles::NO_ERROR, true, 340000, fail * 370000 + 30000},
                         "Tile split + CLU+ broadcast , no memmove, "},

                };
                executeTests(tests);
            }
        }
    }
}
TEST_F(VPULayerCM_InvestigationTest, Layer_EISXW_132141_SEP_split_Qualitative_NPU40) {
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

    VPULayerCostModel& theModel = this->model_4_0;

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

    auto verify_SEP_split_equiv = [](SEP_TestCase t) {
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

    auto verify_SEP_split_equiv = [this](SEP_TestCase t) {
        DPULayer layer(t.t_in.wl);

        VPULayerStrategy strategy{t.t_in.strategy};
        LayerSplitInfo detailed_split;
        CyclesInterfaceType cost_cyc = getModel(layer.device).Layer(layer, strategy, detailed_split);

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

// this is a test fro the situation of K=4096 ans when doing intratile splits the N=50 limit of max splits does not
// allow the algo to reach a workload with K=64. Reason 4096/64>50
TEST_F(VPULayerCM_InvestigationTest, Layer_MAXP_EISXW_na_MINGQI_NPU27) {
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

        const auto origMax = getModel(VPUDevice::VPU_2_7).get_maxWorkloadsPerIntraTileSplit();
        getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(64);
        executeTests(tests);
        getModel(VPUDevice::VPU_2_7).set_maxWorkloadsPerIntraTileSplit(origMax);
    }

    VPULayerCostModel& theModel = this->model_2_7;

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

/// a simple template for investigation of Layer splits ops
TEST_F(VPULayerCM_InvestigationTest, zTemplate_Layer_EISXW_xxxxxxx_NPUXX) {
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
                {{tst_layer, {1U, 1U, 4U, VPUNN::VPUTilingStrategy::SOH_Overlapped, false, false, prefetch}},
                 {VPUNN::Cycles::NO_ERROR, true, 1000, fail * 5001 + 1500},  // 2// V17: x   v159NN: ? GTVPUX:   GTL: ??
                 "SOHO , no memmove, "},

                // note:?? wins
        };
        executeTests(tests);
    }

    VPULayerCostModel& theModel = this->model_4_0;

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

class VPULayerInvstgt_EISXW_119193_Deeplab_v3 : public VPULayerCM_InvestigationTest {
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

    VPULayerCostModel& theModel{this->model_2_7};
    bool force_fail{false};  // controls force failing assertion
    bool simple_fail_all{false};

    auto case_run(const DPUWorkload& workload, const std::string& whatTest) {
        DPUWorkload tst_layer(workload);
        std::string err_info;

        CyclesInterfaceType dpu_cost = theModel.DPU(tst_layer, err_info);
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
}  // namespace VPUNN_unit_tests
