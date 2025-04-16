// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_UT_LAYER_TEST_H
#define VPUNN_UT_LAYER_TEST_H

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

        stream << "All intra splits Costs explored: # " << l.all_intra_tile_splits.size() << "\n";
        {
            int j = 1;
            for (const auto& s : l.all_intra_tile_splits) {
                CyclesInterfaceType sum{std::accumulate(s.cycles.cbegin(), s.cycles.cend(), Cycles::NO_ERROR, Cycles::cost_adder)};
                stream << "# " << j << ", Cost: = " << sum << " [ " << Cycles::toErrorText(sum) << "] "
                       << "  Workloads number: " << s.workloads.size() << "\n";
                j++;
            }
        }

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

}  // namespace VPUNN_unit_tests

#endif  // VPUNN_UT_LAYER_TEST_H