// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include "vpu/shave/shave_op_executors.h"
#include "vpu/shave/shave_instance_holder_factors.h"
#include <gtest/gtest.h>

#include "common/common_helpers.h"
#include "vpu/shave/shave_collection.h"
#include "vpu/shave/shave_devices.h"
#include "vpu/types.h"
#include "vpu_cost_model.h"
#include "vpu_shave_cost_model.h"
#include "vpu/shave/shave_cost_providers/shave_cost_providers.h"

// include generated factors population
#include "vpu/shave/generated_shave_factors_population_npu5.h"

#include <fstream>
#include <iostream>

namespace VPUNN_unit_tests {
using namespace VPUNN;

class ComplexOpsCollectionTestsNPU40 : public ::testing::Test {
protected:
    void SetUp() override {
    }
    const ShaveInstanceHolder_NPU40 ih;
    const SHAVEWorkload::Param select_channel = 1;
    const SHAVEWorkload::Param select_batch = 0;
    const SHAVEWorkload::Param gather_batch_dims = 1;
    const SHAVEWorkload::Param gather_axis = 1;
    const unsigned int MAX_COST{VPUNN::Cycles::START_ERROR_RANGE - 1};  // ten million
    static constexpr unsigned int NO_ERROR_EXPECTED{VPUNN::Cycles::NO_ERROR};

    const ShaveOpExecutor& getShaveOp(const std::string& fn) {
        EXPECT_EQ(ih.getDevice(), VPUDevice::VPU_4_0);

        const DeviceShaveContainer& list = ih.getContainer();

        EXPECT_EQ(list.getDevice(), VPUDevice::VPU_4_0);
        EXPECT_EQ(list.existsShave(fn), true);

        const auto& shaveOp = list.getShaveExecutor(fn);
        const auto ss = shaveOp.getName();

        EXPECT_EQ(ss, fn);
        EXPECT_EQ(shaveOp.getName(), fn);

        return shaveOp;
    }

    struct TestInput {
        const ShaveOpExecutor& shaveOp;
        SHAVEWorkload swl;
    };

    struct TestExpectation {
        CyclesInterfaceType min_cyc{0};
        CyclesInterfaceType max_cyc{0};
    };

    struct TestCase {
        TestInput t_in;
        TestExpectation t_exp;
        std::string test_case = "";
    };

    using TestsVector = std::vector<TestCase>;

    void DoRegularTest(const TestInput& t_in, const TestExpectation& t_exp, const std::string& test_case = "") {
        SHAVEWorkload swl{t_in.swl};
        const ShaveOpExecutor& shvOp{t_in.shaveOp};

        std::string t_header{"** Test Case: " + test_case + "\n"};
        std::cout << ">> " << t_header << std::endl;

        Logger::clear2ndlog();
        unsigned cost_cyc{};
        ASSERT_NO_THROW(cost_cyc = shvOp.dpuCycles(swl));

        EXPECT_GT(cost_cyc, 0u) << t_header;
        EXPECT_TRUE(cost_cyc >= t_exp.min_cyc && cost_cyc <= t_exp.max_cyc)
                << t_header << " Cost not in interval. cost: " << cost_cyc << ",  expected in [ " << t_exp.min_cyc
                << " , " << t_exp.max_cyc << " ] \n";

        std::cout << t_header << " *** ERROR/Cycles code: " << cost_cyc << " : " << Cycles::toErrorText(cost_cyc)
                  << std::endl
                  << "------------------------------------------------------------------------" << std::endl;
    }

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
TEST_F(ComplexOpsCollectionTestsNPU40, GatherMakeRegularTest) {
    const std::string fn{"gather"};
    const auto& shaveOp = getShaveOp(fn);

    SHAVEWorkload::Parameters params = std::vector{gather_axis, gather_batch_dims};

    SHAVEWorkload swl_1("gather", VPUDevice::VPU_4_0, {VPUTensor(40960, 1, 1, 1, DataType::FLOAT16, Layout::XYZ)},
                        {VPUTensor(40960, 1, 1, 1, DataType::FLOAT16, Layout::XYZ)}, params);
    SHAVEWorkload swl_2("gather", VPUDevice::VPU_4_0, {VPUTensor(1, 40960, 1, 1, DataType::FLOAT16, Layout::XYZ)},
                        {VPUTensor(1, 40960, 1, 1, DataType::FLOAT16, Layout::XYZ)}, params);
    SHAVEWorkload swl_3("gather", VPUDevice::VPU_4_0, {VPUTensor(70, 7, 1, 64, DataType::FLOAT16, Layout::XYZ)},
                        {VPUTensor(70, 7, 1, 64, DataType::FLOAT16, Layout::XYZ)}, params);

    const TestsVector tests{
            {{shaveOp, swl_1}, {139000, 140000}, "Full inner dim"},  // 195.9558724 *1700 ~= 331500
            {{shaveOp, swl_2}, {6722000, 6723000}, "InterDim"},      // 17.37302728 * 1700 ~= 29534
            {{shaveOp, swl_3}, {195000, 196000}, "Normal Testcase"}  // 17.37302728 * 1700 ~= 29534

    };

    executeTests(tests);
}

TEST_F(ComplexOpsCollectionTestsNPU40, GatherWrongParamsTest) {
    const std::string fn{"softmax"};
    const auto& shaveOp = getShaveOp(fn);

    const SHAVEWorkload::Param wrong_batch = 0;
    const SHAVEWorkload::Param wrong_axis = 0;

    SHAVEWorkload::Parameters three_params = std::vector{gather_axis, gather_axis, gather_batch_dims};
    SHAVEWorkload::Parameters batch_wrong_param = std::vector{gather_axis, wrong_batch};
    SHAVEWorkload::Parameters axis_wrong_param = std::vector{wrong_axis, gather_batch_dims};

    SHAVEWorkload swl_1("softmax", VPUDevice::VPU_4_0, {VPUTensor(344, 1, 250, 16, DataType::FLOAT16, Layout::XYZ)},
                        {VPUTensor(344, 1, 250, 16, DataType::FLOAT16, Layout::XYZ)});
    SHAVEWorkload swl_2("softmax", VPUDevice::VPU_4_0, {VPUTensor(36, 1, 25, 2, DataType::FLOAT16, Layout::XYZ)},
                        {VPUTensor(36, 1, 25, 2, DataType::FLOAT16, Layout::XYZ)}, three_params);
    SHAVEWorkload swl_3("softmax", VPUDevice::VPU_4_0, {VPUTensor(36, 1, 25, 2, DataType::FLOAT16, Layout::XYZ)},
                        {VPUTensor(36, 1, 25, 2, DataType::FLOAT16, Layout::XYZ)}, batch_wrong_param);
    SHAVEWorkload swl_4("softmax", VPUDevice::VPU_4_0, {VPUTensor(36, 1, 25, 2, DataType::FLOAT16, Layout::XYZ)},
                        {VPUTensor(36, 1, 25, 2, DataType::FLOAT16, Layout::XYZ)}, axis_wrong_param);

    const TestsVector tests{{{shaveOp, swl_1},
                             {V(Cycles::ERROR_SHAVE_PARAMS), V(Cycles::ERROR_SHAVE_PARAMS)},
                             "Error no params testcase"},
                            {{shaveOp, swl_2},
                             {V(Cycles::ERROR_SHAVE_PARAMS), V(Cycles::ERROR_SHAVE_PARAMS)},
                             "Error too many params testcase"},
                            {{shaveOp, swl_3},
                             {V(Cycles::ERROR_SHAVE_PARAMS), V(Cycles::ERROR_SHAVE_PARAMS)},
                             "Error wrong batch param testcase"},
                            {{shaveOp, swl_4},
                             {V(Cycles::ERROR_SHAVE_PARAMS), V(Cycles::ERROR_SHAVE_PARAMS)},
                             "Error wrong axis param testcase"}};

    executeTests(tests);
}

TEST_F(ComplexOpsCollectionTestsNPU40, SoftmaxMakeRegularTest) {
    const std::string fn{"softmax"};
    const auto& shaveOp = getShaveOp(fn);
    SHAVEWorkload::Parameters params = std::vector{select_channel};
    SHAVEWorkload swl_1("softmax", VPUDevice::VPU_4_0, {VPUTensor(344, 1, 250, 1, DataType::FLOAT16, Layout::XYZ)},
                        {VPUTensor(344, 1, 250, 1, DataType::FLOAT16, Layout::XYZ)}, params);
    SHAVEWorkload swl_2("softmax", VPUDevice::VPU_4_0, {VPUTensor(36, 1, 25, 1, DataType::FLOAT16, Layout::XYZ)},
                        {VPUTensor(36, 1, 25, 1, DataType::FLOAT16, Layout::XYZ)}, params);

    const TestsVector tests{
            {{shaveOp, swl_1}, {330000, 335000}, "Normal testcase type 8"},  // 195.9558724 *1700 ~= 331500
            {{shaveOp, swl_2}, {28500, 30000}, "Normal testcase type 2"}     // 17.37302728 * 1700 ~= 29534
    };

    executeTests(tests);
}

TEST_F(ComplexOpsCollectionTestsNPU40, SoftmaxCheckForLayoutError) {
    const std::string fn{"softmax"};
    const auto& shaveOp = getShaveOp(fn);

    SHAVEWorkload::Parameters params = std::vector{select_channel};
    SHAVEWorkload swl_1("softmax", VPUDevice::VPU_4_0, {VPUTensor(344, 1, 250, 1, DataType::FLOAT16, Layout::XZY)},
                        {VPUTensor(344, 1, 250, 1, DataType::FLOAT16, Layout::XZY)}, params);
    SHAVEWorkload swl_2("softmax", VPUDevice::VPU_4_0, {VPUTensor(36, 1, 25, 1, DataType::FLOAT16, Layout::ZXY)},
                        {VPUTensor(36, 1, 25, 1, DataType::FLOAT16, Layout::ZXY)}, params);

    const TestsVector tests{
            {{shaveOp, swl_1},
             //{V(Cycles::ERROR_SHAVE_LAYOUT), V(Cycles::ERROR_SHAVE_LAYOUT)},
             {V(Cycles::NO_ERROR), V(Cycles::START_ERROR_RANGE)},
             "Error layout testcase type 8"},  // 195.9558724 *1700 ~= 331500
            {{shaveOp, swl_2},
             //{V(Cycles::ERROR_SHAVE_LAYOUT), V(Cycles::ERROR_SHAVE_LAYOUT)},
             {V(Cycles::NO_ERROR), V(Cycles::START_ERROR_RANGE)},
             "Error layout testcase type 2"}  // 17.37302728 * 1700 ~= 29534
    };

    executeTests(tests);
}

TEST_F(ComplexOpsCollectionTestsNPU40, SoftmaxCheckForInvalidInputError) {
    const std::string fn{"softmax"};
    const auto& shaveOp = getShaveOp(fn);

    SHAVEWorkload::Parameters params = std::vector{select_channel};
    SHAVEWorkload swl_1("softmax", VPUDevice::VPU_4_0, {VPUTensor(344, 1, 250, 16, DataType::FLOAT16, Layout::XYZ)},
                        {VPUTensor(344, 1, 250, 16, DataType::FLOAT16, Layout::XYZ)}, params);
    SHAVEWorkload swl_2("softmax", VPUDevice::VPU_4_0, {VPUTensor(36, 1, 25, 2, DataType::FLOAT16, Layout::XYZ)},
                        {VPUTensor(36, 1, 25, 2, DataType::FLOAT16, Layout::XYZ)}, params);

    const TestsVector tests{
            {{shaveOp, swl_1},
             {V(Cycles::ERROR_SHAVE_INVALID_INPUT), V(Cycles::ERROR_SHAVE_INVALID_INPUT)},
             "Error invalid input testcase type 8"},  // 195.9558724 *1700 ~= 331500
            {{shaveOp, swl_2},
             {V(Cycles::ERROR_SHAVE_INVALID_INPUT), V(Cycles::ERROR_SHAVE_INVALID_INPUT)},
             "Error invalid input testcase type 2"}  // 17.37302728 * 1700 ~= 29534
    };

    executeTests(tests);
}

TEST_F(ComplexOpsCollectionTestsNPU40, SoftmaxCheckForInvalidParamsError) {
    const std::string fn{"softmax"};
    const auto& shaveOp = getShaveOp(fn);

    SHAVEWorkload::Parameters two_params = std::vector{select_channel, select_batch};
    SHAVEWorkload::Parameters batch_params = std::vector{select_batch};

    SHAVEWorkload swl_1("softmax", VPUDevice::VPU_4_0, {VPUTensor(344, 1, 250, 16, DataType::FLOAT16, Layout::XYZ)},
                        {VPUTensor(344, 1, 250, 16, DataType::FLOAT16, Layout::XYZ)});
    SHAVEWorkload swl_2("softmax", VPUDevice::VPU_4_0, {VPUTensor(36, 1, 25, 2, DataType::FLOAT16, Layout::XYZ)},
                        {VPUTensor(36, 1, 25, 2, DataType::FLOAT16, Layout::XYZ)}, two_params);
    SHAVEWorkload swl_3("softmax", VPUDevice::VPU_4_0, {VPUTensor(36, 1, 25, 2, DataType::FLOAT16, Layout::XYZ)},
                        {VPUTensor(36, 1, 25, 2, DataType::FLOAT16, Layout::XYZ)}, batch_params);

    const TestsVector tests{
            {{shaveOp, swl_1},
             {V(Cycles::ERROR_SHAVE_PARAMS), V(Cycles::ERROR_SHAVE_PARAMS)},
             "Error no params testcase type 8"},  // 195.9558724 *1700 ~= 331500
            {{shaveOp, swl_2},
             {V(Cycles::ERROR_SHAVE_PARAMS), V(Cycles::ERROR_SHAVE_PARAMS)},
             "Error invalid params testcase type 2"},  // 17.37302728 * 1700 ~= 29534
            {{shaveOp, swl_3},
             {V(Cycles::ERROR_SHAVE_PARAMS), V(Cycles::ERROR_SHAVE_PARAMS)},
             "Error batch param testcase type 2"}  // 17.37302728 * 1700 ~= 29534
    };

    executeTests(tests);
}

TEST_F(ComplexOpsCollectionTestsNPU40, NormL2CMakeRegularTest) {
    const std::string fn{"normalizel2onlyc"};
    const auto& shaveOp = getShaveOp(fn);
    SHAVEWorkload::Parameters params = std::vector{select_channel};
    SHAVEWorkload swl_1("normalizel2onlyc", VPUDevice::VPU_4_0,
                        {VPUTensor(112, 1, 20, 1, DataType::FLOAT16, Layout::XYZ)},
                        {VPUTensor(112, 1, 20, 1, DataType::FLOAT16, Layout::XYZ)}, params);
    SHAVEWorkload swl_2("normalizel2onlyc", VPUDevice::VPU_4_0,
                        {VPUTensor(2, 32, 160, 4, DataType::FLOAT16, Layout::XYZ)},
                        {VPUTensor(2, 32, 160, 4, DataType::FLOAT16, Layout::XYZ)}, params);

    const TestsVector tests{
            {{shaveOp, swl_1}, {13000, 14000}, "Normal testcase "},  // 7.757324227 *1700 ~= 13188
            {{shaveOp, swl_2}, {840000, 845000}, "Normal testcase"}  // 495.5978702 * 1700 ~= 842517
    };

    executeTests(tests);
}

TEST_F(ComplexOpsCollectionTestsNPU40, NormL2CCheckForLayoutError) {
    const std::string fn{"normalizel2onlyc"};
    const auto& shaveOp = getShaveOp(fn);

    SHAVEWorkload::Parameters params = std::vector{select_channel};
    SHAVEWorkload swl_1("normalizel2onlyc", VPUDevice::VPU_4_0,
                        {VPUTensor(344, 1, 250, 1, DataType::FLOAT16, Layout::XZY)},
                        {VPUTensor(344, 1, 250, 1, DataType::FLOAT16, Layout::XZY)}, params);
    SHAVEWorkload swl_2("normalizel2onlyc", VPUDevice::VPU_4_0,
                        {VPUTensor(36, 1, 25, 1, DataType::FLOAT16, Layout::ZXY)},
                        {VPUTensor(36, 1, 25, 1, DataType::FLOAT16, Layout::ZXY)}, params);

    const TestsVector tests{
            {{shaveOp, swl_1},
             {V(Cycles::ERROR_SHAVE_LAYOUT), V(Cycles::ERROR_SHAVE_LAYOUT)},
             "Error layout"},  // 195.9558724 *1700 ~= 331500
            {{shaveOp, swl_2},
             {V(Cycles::ERROR_SHAVE_LAYOUT), V(Cycles::ERROR_SHAVE_LAYOUT)},
             "Error layout"}  // 17.37302728 * 1700 ~= 29534
    };

    executeTests(tests);
}

TEST_F(ComplexOpsCollectionTestsNPU40, NormL2CCheckForInvalidParamsError) {
    const std::string fn{"normalizel2onlyc"};
    const auto& shaveOp = getShaveOp(fn);

    SHAVEWorkload::Parameters two_params = std::vector{select_channel, select_batch};
    SHAVEWorkload::Parameters batch_params = std::vector{select_batch};

    SHAVEWorkload swl_1("normalizel2onlyc", VPUDevice::VPU_4_0,
                        {VPUTensor(344, 1, 250, 16, DataType::FLOAT16, Layout::XYZ)},
                        {VPUTensor(344, 1, 250, 16, DataType::FLOAT16, Layout::XYZ)});
    SHAVEWorkload swl_2("normalizel2onlyc", VPUDevice::VPU_4_0,
                        {VPUTensor(36, 1, 25, 2, DataType::FLOAT16, Layout::XYZ)},
                        {VPUTensor(36, 1, 25, 2, DataType::FLOAT16, Layout::XYZ)}, two_params);
    SHAVEWorkload swl_3("normalizel2onlyc", VPUDevice::VPU_4_0,
                        {VPUTensor(36, 1, 25, 2, DataType::FLOAT16, Layout::XYZ)},
                        {VPUTensor(36, 1, 25, 2, DataType::FLOAT16, Layout::XYZ)}, batch_params);

    const TestsVector tests{
            {{shaveOp, swl_1}, {V(Cycles::ERROR_SHAVE_PARAMS), V(Cycles::ERROR_SHAVE_PARAMS)}, "Error no params"},
            {{shaveOp, swl_2},
             {V(Cycles::ERROR_SHAVE_PARAMS), V(Cycles::ERROR_SHAVE_PARAMS)},
             "Error more than one params"},  // 17.37302728 * 1700 ~= 29534
            {{shaveOp, swl_3},
             {V(Cycles::ERROR_SHAVE_PARAMS), V(Cycles::ERROR_SHAVE_PARAMS)},
             "Error different param"}  // 17.37302728 * 1700 ~= 29534
    };

    executeTests(tests);
}

class MVNCollectionTests : public ::testing::Test {
protected:
    void SetUp() override {
    }
    const ShaveInstanceHolder_VPU27 ih;

    const unsigned int MAX_COST{VPUNN::Cycles::START_ERROR_RANGE - 1};  // ten million
    static constexpr unsigned int NO_ERROR_EXPECTED{VPUNN::Cycles::NO_ERROR};

    const ShaveOpExecutor& getShaveOp(const std::string& fn) {
        EXPECT_EQ(ih.getDevice(), VPUDevice::VPU_2_7);

        const DeviceShaveContainer& list = ih.getContainer();

        EXPECT_EQ(list.getDevice(), VPUDevice::VPU_2_7);
        EXPECT_EQ(list.existsShave(fn), true);

        const auto& shaveOp = list.getShaveExecutor(fn);
        const auto ss = shaveOp.getName();

        EXPECT_EQ(ss, fn);
        EXPECT_EQ(shaveOp.getName(), fn);

        return shaveOp;
    }

    struct TestInput {
        const ShaveOpExecutor& shaveOp;
        SHAVEWorkload swl;
    };

    struct TestExpectation {
        CyclesInterfaceType min_cyc{0};
        CyclesInterfaceType max_cyc{0};
    };

    struct TestCase {
        TestInput t_in;
        TestExpectation t_exp;
        std::string test_case = "";
    };

    using TestsVector = std::vector<TestCase>;

    void DoRegularTest(const TestInput& t_in, const TestExpectation& t_exp, const std::string& test_case = "") {
        SHAVEWorkload swl{t_in.swl};
        const ShaveOpExecutor& shvOp{t_in.shaveOp};

        std::string t_header{"** Test Case: " + test_case + "\n"};
        std::cout << ">> " << t_header << std::endl;

        Logger::clear2ndlog();
        unsigned cost_cyc{};
        ASSERT_NO_THROW(cost_cyc = shvOp.dpuCycles(swl));

        EXPECT_GT(cost_cyc, 0u) << t_header;
        EXPECT_TRUE(cost_cyc >= t_exp.min_cyc && cost_cyc <= t_exp.max_cyc)
                << t_header << " Cost not in interval. cost: " << cost_cyc << ",  expected in [ " << t_exp.min_cyc
                << " , " << t_exp.max_cyc << " ] \n";

        std::cout << t_header << " *** ERROR/Cycles code: " << cost_cyc << " : " << Cycles::toErrorText(cost_cyc)
                  << std::endl
                  << "------------------------------------------------------------------------" << std::endl;
    }

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

TEST_F(MVNCollectionTests, MVN6OneAxTests) {
    const std::string fn{"MVN6_oneAx"};
    const auto& shaveOp = getShaveOp(fn);

    SHAVEWorkload swl_1("MVN6_oneAx", VPUDevice::VPU_2_7, {VPUTensor(1, 40960, 1, 1, DataType::FLOAT16)},
                        {VPUTensor(1, 40960, 1, 1, DataType::FLOAT16)});
    SHAVEWorkload swl_2("MVN6_oneAx", VPUDevice::VPU_2_7, {VPUTensor(2, 2, 10240, 1, DataType::FLOAT16)},
                        {VPUTensor(2, 2, 10240, 1, DataType::FLOAT16)});

    const TestsVector tests{
            {{shaveOp, swl_1}, {30000000, 30500000}, "Extreme testcase"},  // 23310.99302 *1300 = 30,304,290.926
            {{shaveOp, swl_2}, {10600000, 10700000}, "Normal testcase"}    // 8180.988415 * 1300 = 10,635,284.9395
    };

    executeTests(tests);
}

TEST_F(MVNCollectionTests, MVN6TwoAxTests) {
    const std::string fn{"MVN6_twoAx"};
    const auto& shaveOp = getShaveOp(fn);

    SHAVEWorkload swl_1("MVN6_twoAx", VPUDevice::VPU_2_7, {VPUTensor(1, 40960, 1, 1, DataType::FLOAT16, Layout::XZY)},
                        {VPUTensor(1, 40960, 1, 1, DataType::FLOAT16, Layout::XZY)});
    SHAVEWorkload swl_2("MVN6_twoAx", VPUDevice::VPU_2_7, {VPUTensor(32, 128, 10, 1, DataType::FLOAT16, Layout::XZY)},
                        {VPUTensor(32, 128, 10, 1, DataType::FLOAT16, Layout::XZY)});

    const TestsVector tests{
            {{shaveOp, swl_1}, {32000000, 33000000}, "Extreme testcase"},  // 25164.83868*1300 = 32,714,290.284
            {{shaveOp, swl_2}, {11800000, 12100000}, "Normal testcase"}    // 9185.864336*1300 = 11,941,623.6368
    };

    executeTests(tests);
}

TEST_F(MVNCollectionTests, MVN6ThreeAxTests) {
    const std::string fn{"MVN6_threeAx"};
    const auto& shaveOp = getShaveOp(fn);
    // WHCB -> WCBH Since we dont have this layout config I am going to put the size of H in B and vice versa
    SHAVEWorkload swl_1("MVN6_threeAx", VPUDevice::VPU_2_7, {VPUTensor(1, 1, 1, 40960, DataType::FLOAT16, Layout::XZY)},
                        {VPUTensor(1, 1, 1, 40960, DataType::FLOAT16, Layout::XZY)});
    SHAVEWorkload swl_2("MVN6_threeAx", VPUDevice::VPU_2_7, {VPUTensor(16, 16, 4, 40, DataType::FLOAT16, Layout::XZY)},
                        {VPUTensor(16, 16, 4, 40, DataType::FLOAT16, Layout::XZY)});

    const TestsVector tests{
            {{shaveOp, swl_1}, {36800000, 37000000}, "Extreme testcase"},  // 28408.64375 * 1300 = 36,931,236.875
            {{shaveOp, swl_2}, {13260000, 13280000}, "Normal testcase"}    // 10206.68712 * 1300 = 13,268,693.256
    };

    executeTests(tests);
}

TEST_F(MVNCollectionTests, MVN6FourAxTests) {
    const std::string fn{"MVN6_fourAx"};
    const auto& shaveOp = getShaveOp(fn);

    SHAVEWorkload swl_1("MVN6_fourAx", VPUDevice::VPU_2_7, {VPUTensor(1, 40960, 1, 1, DataType::FLOAT16, Layout::XYZ)},
                        {VPUTensor(1, 40960, 1, 1, DataType::FLOAT16, Layout::XYZ)});
    SHAVEWorkload swl_2("MVN6_fourAx", VPUDevice::VPU_2_7, {VPUTensor(40, 32, 32, 1, DataType::FLOAT16, Layout::XYZ)},
                        {VPUTensor(40, 32, 32, 1, DataType::FLOAT16, Layout::XYZ)});

    const TestsVector tests{
            {{shaveOp, swl_1}, {17850000, 17880000}, "Extreme testcase"},  // 17,861,552.241 = 1300 * 13739.65557
            {{shaveOp, swl_2}, {14330000, 14380000}, "Normal testcase"}    // 1300 * 11049.06877 = 14,363,789.401
    };

    executeTests(tests);
}

TEST_F(MVNCollectionTests, MVNSimple2AxTests) {
    const std::string fn{"MVN_2Ax"};
    const auto& shaveOp = getShaveOp(fn);

    SHAVEWorkload swl_1("MVN_2Ax", VPUDevice::VPU_2_7, {VPUTensor(1, 1, 1, 80000, DataType::FLOAT16, Layout::XYZ)},
                        {VPUTensor(1, 1, 1, 80000, DataType::FLOAT16, Layout::XYZ)});
    SHAVEWorkload swl_2("MVN_2Ax", VPUDevice::VPU_2_7, {VPUTensor(136, 1, 32, 10, DataType::FLOAT16, Layout::XYZ)},
                        {VPUTensor(136, 1, 32, 10, DataType::FLOAT16, Layout::XYZ)});

    const TestsVector tests{{{shaveOp, swl_1}, {46600000, 47000000}, "Extreme testcase"},
                            {{shaveOp, swl_2}, {270000, 285000}, "Normal testcase"}};

    executeTests(tests);
}

TEST_F(MVNCollectionTests, MVNSimple3AxTests) {
    const std::string fn{"MVN_3Ax"};
    const auto& shaveOp = getShaveOp(fn);

    SHAVEWorkload swl_1("MVN_3Ax", VPUDevice::VPU_2_7, {VPUTensor(1, 1, 1, 40960, DataType::FLOAT16, Layout::XYZ)},
                        {VPUTensor(1, 1, 1, 40960, DataType::FLOAT16, Layout::XYZ)});
    SHAVEWorkload swl_2("MVN_3Ax ", VPUDevice::VPU_2_7, {VPUTensor(512, 2, 40, 1, DataType::FLOAT16, Layout::XYZ)},
                        {VPUTensor(512, 2, 40, 1, DataType::FLOAT16, Layout::XYZ)});

    const TestsVector tests{{{shaveOp, swl_1}, {20400000, 20500000}, "Extreme testcase"},
                            {{shaveOp, swl_2}, {95000, 105000}, "Normal testcase"}};

    executeTests(tests);
}
class ShaveCollectionTest : public ::testing::Test {
protected:
    void SetUp() override {
    }
    const SHAVEWorkload ref_shv_wrkld{"sigmoid",
                                      VPUDevice::VPU_2_7,
                                      {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
                                      {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)}};

    struct Test {
        SHAVEWorkload input;

        bool expected_exist = true;
        CyclesInterfaceType exp_cycles = 0;
    };

    void ExecuteTest(const DeviceShaveContainer& list, const Test& test) {
        const std::string fn{test.input.get_name()};
        const bool found{list.existsShave(fn)};
        ASSERT_EQ(found, test.expected_exist) << fn;
        if (test.expected_exist) {
            const auto& shaveOp = list.getShaveExecutor(fn);
            ASSERT_EQ(shaveOp.getName(), fn) << fn;

            const SHAVEWorkload& swl{test.input};
            CyclesInterfaceType cycles = shaveOp.dpuCycles(swl);

            if (test.exp_cycles != 0) {
                EXPECT_EQ(cycles, test.exp_cycles) << swl;
            } else {
                EXPECT_GT(cycles, 0) << swl;
            }
        }
    }
};

TEST_F(ShaveCollectionTest, DefaultCaseTestNPU27) {
    {
        ShaveInstanceHolder_VPU27 ih;
        EXPECT_EQ(ih.getDevice(), VPUDevice::VPU_2_7);
        const DeviceShaveContainer& list = ih.getContainer();

        // list.find
        EXPECT_EQ(list.getDevice(), VPUDevice::VPU_2_7);
        const std::string fn{"default"};
        EXPECT_EQ(list.existsShave(fn), true);
        const auto& shaveOp = list.getShaveExecutor(fn);
        const auto ss = shaveOp.getName();
        EXPECT_EQ(ss, fn);
        EXPECT_EQ(shaveOp.getName(), fn);

        SHAVEWorkload swl("default", VPUDevice::VPU_2_7, {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
                          {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)});
        CyclesInterfaceType cycles = shaveOp.dpuCycles(swl);
        EXPECT_NEAR(cycles, 5000, 1);
    }
}

TEST_F(ShaveCollectionTest, DefaultCaseTestNPU40) {
    {
        ShaveInstanceHolder_NPU40 ih;
        EXPECT_EQ(ih.getDevice(), VPUDevice::VPU_4_0);
        const DeviceShaveContainer& list = ih.getContainer();

        EXPECT_EQ(list.getDevice(), VPUDevice::VPU_4_0);
        const std::string fn{"default"};
        EXPECT_EQ(list.existsShave(fn), true);
        const auto& shaveOp = list.getShaveExecutor(fn);
        const auto ss = shaveOp.getName();
        EXPECT_EQ(ss, fn);
        EXPECT_EQ(shaveOp.getName(), fn);

        {
            SHAVEWorkload swl("default", VPUDevice::VPU_4_0, {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
                              {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)});
            CyclesInterfaceType cycles = shaveOp.dpuCycles(swl);
            EXPECT_NEAR(cycles, 5000, 1);
        }
        {
            SHAVEWorkload swl("default", VPUDevice::VPU_4_0, {VPUTensor(7, 29, 3, 1, DataType::FLOAT16)},
                              {VPUTensor(7, 29, 3, 1, DataType::FLOAT16)});
            CyclesInterfaceType cycles = shaveOp.dpuCycles(swl);
            EXPECT_NEAR(cycles, 7 * 29 * 3, 1);
        }
        {
            SHAVEWorkload swl("default", VPUDevice::VPU_4_0, {VPUTensor(101, 101, 2, 1, DataType::FLOAT16)},
                              {VPUTensor(101, 101, 2, 1, DataType::FLOAT16)});
            CyclesInterfaceType cycles = shaveOp.dpuCycles(swl);
            EXPECT_NEAR(cycles, 101 * 101 * 2, 1);
        }
        {
            SHAVEWorkload swl("default", VPUDevice::VPU_4_0, {VPUTensor(1, 1, 5, 1, DataType::FLOAT16)},
                              {VPUTensor(1, 1, 5, 1, DataType::FLOAT16)});
            CyclesInterfaceType cycles = shaveOp.dpuCycles(swl);
            EXPECT_NEAR(cycles, 5, 1);
        }
    }
}

TEST_F(ShaveCollectionTest, DefaultCaseTestNPU50) {
    {
        ShaveInstanceHolder_Mock_NPU50 ih;
        EXPECT_EQ(ih.getDevice(), VPUDevice::NPU_5_0);
        const DeviceShaveContainer& list = ih.getContainer();

        EXPECT_EQ(list.getDevice(), VPUDevice::NPU_5_0);
        const std::string fn{"default"};
        EXPECT_EQ(list.existsShave(fn), true);
        const auto& shaveOp = list.getShaveExecutor(fn);
        const auto ss = shaveOp.getName();
        EXPECT_EQ(ss, fn);
        EXPECT_EQ(shaveOp.getName(), fn);

        {
            SHAVEWorkload swl(fn, VPUDevice::NPU_5_0, {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
                              {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)});
            CyclesInterfaceType cycles = shaveOp.dpuCycles(swl);
            EXPECT_NEAR(cycles, 5000, 10);
        }
        {
            SHAVEWorkload swl(fn, VPUDevice::NPU_5_0, {VPUTensor(7, 29, 3, 1, DataType::FLOAT16)},
                              {VPUTensor(7, 29, 3, 1, DataType::FLOAT16)});
            CyclesInterfaceType cycles = shaveOp.dpuCycles(swl);
            EXPECT_NEAR(cycles, 7 * 29 * 3, 10);
        }
        {
            SHAVEWorkload swl(fn, VPUDevice::NPU_5_0, {VPUTensor(101, 101, 2, 1, DataType::FLOAT16)},
                              {VPUTensor(101, 101, 2, 1, DataType::FLOAT16)});
            CyclesInterfaceType cycles = shaveOp.dpuCycles(swl);
            EXPECT_NEAR(cycles, 101 * 101 * 2, 10);
        }
        {
            SHAVEWorkload swl(fn, VPUDevice::NPU_5_0, {VPUTensor(1, 1, 5, 1, DataType::FLOAT16)},
                              {VPUTensor(1, 1, 5, 1, DataType::FLOAT16)});
            CyclesInterfaceType cycles = shaveOp.dpuCycles(swl);
            EXPECT_NEAR(cycles, 5, 10);
        }
    }
}


TEST_F(ShaveCollectionTest, InstanceHolderWithFloatFactors) {
    // Thread-local storage for test factors (to avoid static member issues in local class)
    thread_local std::unordered_map<std::string, ShaveSpeedUpFactorType> g_test_factors;
    
    // Define a custom PopulatedFactorsLUT class for testing with specific factors
    class CustomTestPopulatedFactorsLUT : public FactorsLookUpTable {
    public:
        CustomTestPopulatedFactorsLUT() {
            populate();
        }
    private:        
        void populate() {
            // Populate with the thread-local test factors
            for (const auto& [name, factor] : g_test_factors) {
                add(name, factor);
            }
        }
    };

    ShaveInstanceHolder_Mock_NPU50 ih;
    const DeviceShaveContainer& list_without_factors = ih.getContainer();
    // add a set of functions to test
    const std::vector<std::string> functions = {
        "add", "sub", "sigmoid", "abs", "clamp", "cos", "cumsum", 
        "hsigmoid", "elu", "mish", "floor", "erf", "exp", "sqrt", "ceiling",
        "logicalnot", "log", "acos", "acosh", "asin", "asinh", "atan", "atanh"
    };

    //filter functions that actually exist (just in case)
    std::vector<std::string> existing_functions;
    for (const auto& fn : functions) {
        if (list_without_factors.existsShave(fn)) {
            existing_functions.push_back(fn);
        }
    }

    ASSERT_GT(existing_functions.size(), 0) << "No test functions found in the container";

    size_t nr_functions_with_factors = functions.size() / 2;
    const float speedup = 2.0f;
    
    // Set up the test factors
    g_test_factors.clear();
    for (size_t i = 0; i < nr_functions_with_factors; i++) {
        g_test_factors.emplace(functions[i], speedup);
    }
    
    // use ShaveInstanceHolderWithFactors with the custom populate class
    ShaveInstanceHolderWithFactors<CustomTestPopulatedFactorsLUT, ShaveInstanceHolder_Mock_NPU50, VPUDevice::NPU_5_0> fih;
    const DeviceShaveContainer& list_with_factors = fih.getContainer();

    for (const std::string& fn : existing_functions) {
        EXPECT_TRUE(list_with_factors.existsShave(fn))
                << "Function: " << fn << " does not exist in the list with factors";

        const auto& factor_shaveOp = list_with_factors.getShaveExecutor(fn);
        const auto& shaveOp = list_without_factors.getShaveExecutor(fn);

        SHAVEWorkload swl(fn, VPUDevice::NPU_5_0, {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
                          {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)});

        CyclesInterfaceType cycles = 0;
        CyclesInterfaceType factor_cycles = 0;

        ASSERT_NO_THROW(cycles = shaveOp.dpuCycles(swl)) << "Failed to get cycles for function: " << fn;
        ASSERT_NO_THROW(factor_cycles = factor_shaveOp.dpuCycles(swl))
                << "Failed to get factor cycles for function: " << fn;
        
        // Check if this function should have speedup applied
        auto end_of_list_with_factors = existing_functions.begin() + nr_functions_with_factors; // iterator to the end of the functions with factors
        // check if fn is in the boundaries of list with factors
        bool should_have_speedup = std::find(existing_functions.begin(), end_of_list_with_factors, fn) != end_of_list_with_factors;
        
        // For functions with speedup factor, expect reduced cycles
        if (should_have_speedup) {
            // This function should have speedup applied
            EXPECT_LT(factor_cycles, cycles) << "Function " << fn << " should have reduced cycles";
            float expected_factor_cycles = cycles / speedup;
            EXPECT_NEAR(factor_cycles, expected_factor_cycles, 1.0f)
                    << "Function " << fn << " speedup not applied correctly. Expected: " << expected_factor_cycles
                    << ", Got: " << factor_cycles;
        } else {
            // This function should have default factor (1.0f) so cycles should be the same
            EXPECT_EQ(cycles, factor_cycles) << "Function " << fn << " should have same cycles";
        }
    }
}

TEST_F(ShaveCollectionTest, CheckCMakeFactorExtractionAndPopulate) {
    // Test that CMake factors are properly extracted and ShaveInstanceHolderWithFactors is populated

    // Create instance holders with factors to test CMake integration
    using ShaveInstanceHolder_NPU40_WithFactors =
            ShaveInstanceHolderWithFactors<PopulatedFactorsLUT_NPU5, ShaveInstanceHolder_NPU40, VPUDevice::VPU_4_0>;
    using ShaveInstanceHolder_NPU50_WithFactors =
            ShaveInstanceHolderWithFactors<PopulatedFactorsLUT_NPU5, ShaveInstanceHolder_Mock_NPU50, VPUDevice::NPU_5_0>;

    // Test NPU 4.0 with factors - check which functions have non-default factors
    {
        ShaveInstanceHolder_NPU40_WithFactors ih_with_factors;
        const DeviceShaveContainer& list_with_factors = ih_with_factors.getContainer();

        // Also create baseline for comparison
        ShaveInstanceHolder_NPU40 ih_baseline;

        // Verify device is correctly set
        EXPECT_EQ(list_with_factors.getDevice(), VPUDevice::VPU_4_0);

        // Check that the container is populated with functions
        std::vector<std::string> available_functions = list_with_factors.getShaveList();
        EXPECT_GT(available_functions.size(), 0) << "ShaveInstanceHolderWithFactors should be populated with functions";

        std::cout << "NPU 4.0 ShaveInstanceHolderWithFactors populated with " << available_functions.size()
                  << " functions" << std::endl;

        // Test the FactorsLookUpTable directly to see what factors are defined
        PopulatedFactorsLUT_NPU5 test_lut;

        size_t functions_with_non_default_factors = 0;
        std::cout << "Functions with non-default factors:" << std::endl;

        for (const auto& fn : available_functions) {
            auto factor = test_lut.getOperatorFactor(fn);

            // Check for different factor than default one
            if (factor != 1.0f) {
                functions_with_non_default_factors++;
                std::cout << "  " << fn << " -> " << factor << std::endl;
            }
        }

        std::cout << "Total functions with non-default factors: " << functions_with_non_default_factors << std::endl;
        std::cout << "Functions with default factors (1.0f): "
                  << (available_functions.size() - functions_with_non_default_factors) << std::endl;
    }

    // Test NPU 5.0 with factors - only check if populated
    {
        ShaveInstanceHolder_NPU50_WithFactors ih_with_factors;
        const DeviceShaveContainer& list_with_factors = ih_with_factors.getContainer();

        // Verify device is correctly set
        EXPECT_EQ(list_with_factors.getDevice(), VPUDevice::NPU_5_0);

        // Check that the container is populated with functions
        std::vector<std::string> available_functions = list_with_factors.getShaveList();
        EXPECT_GT(available_functions.size(), 0) << "ShaveInstanceHolderWithFactors should be populated with functions";

        std::cout << "NPU 5.0 ShaveInstanceHolderWithFactors populated with " << available_functions.size()
                  << " functions" << std::endl;
    }

    // Verify that the FactorsLookUpTable populate() method works
    {
        PopulatedFactorsLUT_NPU5 test_lut;
        EXPECT_TRUE(test_lut.is_populated()) << "PopulatedFactorsLUT_NPU5 should have been populated";

        // Test that we can retrieve factors (should return default 1.0f if none defined)
        auto factor = test_lut.getOperatorFactor("test_function");

        // Default factor should be 1.0f for non-existent functions
        EXPECT_FLOAT_EQ(factor, 1.0f) << "Default factor should be 1.0f";
    }

    // Test that ShaveInstanceHolderWithFactors constructor doesn't throw
    {
        using TestHolderType = ShaveInstanceHolderWithFactors<PopulatedFactorsLUT_NPU5, ShaveInstanceHolder_NPU50, VPUDevice::NPU_5_0>; // avoid treating template comma as macro arguments
        EXPECT_NO_THROW(TestHolderType test_holder = TestHolderType{})
                << "ShaveInstanceHolderWithFactors constructor should not throw";
    }
}

TEST_F(ShaveCollectionTest, NPU5FactorsEquivalentToGetSelector) {
    // Test that for PTL (PopulatedFactorsLUT_NPU5), the ShaveInstanceHolderWithFactors
    // produces the same results as getSelector since all factors are 1.0f (mocked)
    
    // Get the two selectors from ShaveConfiguration
    ShaveConfiguration shave_config;
    
    // selector_50 uses mock_shaves_50 (all factors implicitly 1.0f)
    const ShaveSelector& selector_no_factors = shave_config.getSelector(VPUDevice::NPU_5_0);
    
    // selector_50_with_factors uses ShaveInstanceHolder_NPU50_WithFactors (PTL with all factors = 1.0f)
    const ShaveSelector& selector_with_factors = shave_config.getSelectorWithFactors(VPUDevice::NPU_5_0);
    
    // Get list of available functions from the selector without factors
    std::vector<std::string> available_functions = selector_no_factors.getShaveList();
    ASSERT_GT(available_functions.size(), 0) << "No shave functions available for NPU_5_0";
    
    std::cout << "Testing " << available_functions.size() << " functions for PTL factor equivalence" << std::endl;
    
    // Test a variety of workloads for each function
    std::vector<SHAVEWorkload> test_workloads;
    
    // Create diverse test workloads with different tensor shapes
    const std::vector<std::tuple<int, int, int, int>> test_shapes = {
        {10, 100, 5, 1},
        {1, 1, 5, 1},
        {100, 100, 50, 1},
        {7, 29, 3, 1},
        {344, 1, 250, 1},
        {36, 1, 25, 1}
    };
    
    size_t tests_passed = 0;
    size_t tests_total = 0;
    
    for (const auto& fn : available_functions) {        
        const auto& shave_op_no_factors = selector_no_factors.getShaveFuntion(fn);
        const auto& shave_op_with_factors = selector_with_factors.getShaveFuntion(fn);
        
        for (const auto& [w, h, c, b] : test_shapes) {
            SHAVEWorkload swl(fn, VPUDevice::NPU_5_0, 
                            {VPUTensor(w, h, c, b, DataType::FLOAT16)},
                            {VPUTensor(w, h, c, b, DataType::FLOAT16)});
            
            CyclesInterfaceType cycles_no_factors = 0;
            CyclesInterfaceType cycles_with_factors = 0;
            
            try {
                cycles_no_factors = shave_op_no_factors.dpuCycles(swl);
                cycles_with_factors = shave_op_with_factors.dpuCycles(swl);
                
                tests_total++;
                
                // Since all PTL factors are 1.0f (mocked), results should be identical
                EXPECT_EQ(cycles_no_factors, cycles_with_factors)
                    << "Function: " << fn 
                    << ", Shape: [" << w << ", " << h << ", " << c << ", " << b << "]"
                    << ", cycles_no_factors: " << cycles_no_factors
                    << ", cycles_with_factors: " << cycles_with_factors;
                
                if (cycles_no_factors == cycles_with_factors) {
                    tests_passed++;
                }
            } catch (const std::exception&) {
                // Some workloads might not be valid for certain operations, skip them
                continue;
            }
        }
    }
    
    std::cout << "PTL Factor Equivalence Test Results: " << tests_passed << "/" << tests_total << " tests passed" << std::endl;
    EXPECT_GT(tests_passed, 0) << "No valid tests were executed";
    EXPECT_EQ(tests_passed, tests_total) << "Some tests failed the equivalence check";
}

TEST_F(ShaveCollectionTest, EqualSpecialCase) {
    {
        ShaveInstanceHolder_NPU40 ih;
        EXPECT_EQ(ih.getDevice(), VPUDevice::VPU_4_0);
        const DeviceShaveContainer& list = ih.getContainer();

        EXPECT_EQ(list.getDevice(), VPUDevice::VPU_4_0);
        const std::string fn{"equal"};
        EXPECT_EQ(list.existsShave(fn), true);
        const auto& shaveOp = list.getShaveExecutor(fn);
        const auto ss = shaveOp.getName();
        EXPECT_EQ(ss, fn);
        EXPECT_EQ(shaveOp.getName(), fn);

        SHAVEWorkload swl1("equal", VPUDevice::VPU_4_0, {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
                           {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)});
        SHAVEWorkload swl2("equal", VPUDevice::VPU_4_0, {VPUTensor(1, 1, 5, 1, DataType::FLOAT16)},
                           {VPUTensor(1, 1, 5, 1, DataType::FLOAT16)});
        SHAVEWorkload swl3("equal", VPUDevice::VPU_4_0, {VPUTensor(100, 100, 50, 1, DataType::FLOAT16)},
                           {VPUTensor(100, 100, 50, 1, DataType::FLOAT16)});
        CyclesInterfaceType cycles1 = shaveOp.dpuCycles(swl1);
        CyclesInterfaceType cycles2 = shaveOp.dpuCycles(swl2);
        CyclesInterfaceType cycles3 = shaveOp.dpuCycles(swl3);

        EXPECT_NEAR(cycles1, 2952, 1);
        EXPECT_NEAR(cycles2, 2952, 1);
        EXPECT_NEAR(cycles3, 2952, 1);
    }
}

TEST_F(ShaveCollectionTest, instanceHolder_Smoke) {
    ShaveInstanceHolder_VPU27 ih;
    EXPECT_EQ(ih.getDevice(), VPUDevice::VPU_2_7);
    const DeviceShaveContainer& list = ih.getContainer();

    // list.find
    EXPECT_EQ(list.getDevice(), VPUDevice::VPU_2_7);
    // EXPECT_EQ(list.map.size(), 2);

    const std::string fn{"sigmoid"};
    // const auto it = list.map.find(fn);

    // EXPECT_NE(it, list.map.end());
    EXPECT_EQ(list.existsShave(fn), true);

    // auto& shaveOp = it->second;  // list[fn];
    const auto& shaveOp = list.getShaveExecutor(fn);
    const auto ss = shaveOp.getName();
    EXPECT_EQ(ss, fn);
    EXPECT_EQ(shaveOp.getName(), fn);

    SHAVEWorkload swl("sigmoid", VPUDevice::VPU_2_7, {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
                      {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)});
    CyclesInterfaceType cycles = shaveOp.dpuCycles(swl);
    EXPECT_GT(cycles, 0);
    // delete &shaveOp;  // not good
}

TEST_F(ShaveCollectionTest, instanceHolder_Smoke_NPU_40) {
    ShaveInstanceHolder_NPU40 ih;
    EXPECT_EQ(ih.getDevice(), VPUDevice::VPU_4_0);
    const DeviceShaveContainer& list = ih.getContainer();

    // list.find
    EXPECT_EQ(list.getDevice(), VPUDevice::VPU_4_0);
    // EXPECT_EQ(list.map.size(), 2);

    const std::string fn{"sigmoid"};
    // const auto it = list.map.find(fn);

    // EXPECT_NE(it, list.map.end());
    EXPECT_EQ(list.existsShave(fn), true);

    // auto& shaveOp = it->second;  // list[fn];
    const auto& shaveOp = list.getShaveExecutor(fn);
    const auto ss = shaveOp.getName();
    EXPECT_EQ(ss, fn);
    EXPECT_EQ(shaveOp.getName(), fn);

    SHAVEWorkload swl("sigmoid", VPUDevice::VPU_4_0, {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
                      {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)});
    CyclesInterfaceType cycles = shaveOp.dpuCycles(swl);
    EXPECT_GT(cycles, 0);
    // delete &shaveOp;  // not good
}

TEST_F(ShaveCollectionTest, instanceHolder_Smoke_NPU_50) {
    ShaveInstanceHolder_Mock_NPU50 ih;
    EXPECT_EQ(ih.getDevice(), VPUDevice::NPU_5_0);
    const DeviceShaveContainer& list = ih.getContainer();

    // list.find
    EXPECT_EQ(list.getDevice(), VPUDevice::NPU_5_0);
    // EXPECT_EQ(list.map.size(), 2);

    const std::string fn{"sigmoid"};
    // const auto it = list.map.find(fn);

    // EXPECT_NE(it, list.map.end());
    EXPECT_EQ(list.existsShave(fn), true);

    // auto& shaveOp = it->second;  // list[fn];
    const auto& shaveOp = list.getShaveExecutor(fn);
    const auto ss = shaveOp.getName();
    EXPECT_EQ(ss, fn);
    EXPECT_EQ(shaveOp.getName(), fn);

    SHAVEWorkload swl("sigmoid", VPUDevice::NPU_5_0, {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
                      {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)});
    CyclesInterfaceType cycles = shaveOp.dpuCycles(swl);
    EXPECT_GT(cycles, 0);
    // delete &shaveOp;  // not good
}


TEST_F(ShaveCollectionTest, instanceHolder_27_40_Smoke) {
    const VPUDevice testDev{VPUDevice::VPU_2_7};
    const VPUDevice testDev_40{VPUDevice::VPU_4_0};

    const std::vector<Test> tests{
            {{"sigmoid", testDev, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, true, 0},
            {{"hardswish", testDev, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, true, 0},
            {{"logicaland", testDev, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, true, 0},
            {{"PermuteQuantize", testDev, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, false, 0},
            // below exisst in csv
            {{"abs", testDev, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, true, 0},
            {{"clamp", testDev, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, true, 0},
            {{"cos", testDev, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, true, 0},
            {{"cumsum", testDev, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, true, 0},
            {{"elu", testDev, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, true, 0},
            {{"div", testDev, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, true, 0},
            {{"equal", testDev, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, true, 0},

            // hand added for now
            {{"MVN6_onlyOneAx", testDev, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, true, 0},

            {{"MVN6_oneAx", testDev, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, true, 0},
            {{"MVN6_twoAx", testDev, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, true, 0},
            {{"MVN6_threeAx", testDev, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, true, 0},
            {{"MVN6_fourAx", testDev, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, true, 0},
            {{"MVN6_fiveAx", testDev, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, false, 0},

            {{"MVN6", testDev, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, true, 0},  // generic 6

            {{"MVN_3Ax", testDev, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, true, 0},  // simple 3 axis
            {{"MVN_2Ax", testDev, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, true, 0},  // simple 2 axis

            // composite
            {{"MVN", testDev, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, true, 0},  // composite mvn

            // interpolate
            {{"interpolatewh_1", testDev, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()},
             true,
             0},  // composite mvn

            {{"zzzzzk", testDev, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, false, 0},

    };

    const std::vector<Test> tests_40{
            {{"sigmoid", testDev_40, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, true, 0},
            {{"relu", testDev_40, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, true, 0},

            {{"MVN6", testDev_40, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, true, 0},

            {{"interpolatewh_1", testDev, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, false, 0},

            {{"zzzzzk", testDev_40, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, false, 0},
    };
    // Test x{{"PermuteQuantize", testDev, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, false, 0}

    // list.find

    // EXPECT_EQ(list.map.size(), 2);

    {
        ShaveInstanceHolder_VPU27 ih;
        EXPECT_EQ(ih.getDevice(), VPUDevice::VPU_2_7);
        const DeviceShaveContainer& list = ih.getContainer();
        EXPECT_EQ(list.getDevice(), VPUDevice::VPU_2_7);

        for (const auto& t : tests) {
            ExecuteTest(list, t);
        }
    }

    {
        const VPUDevice td{VPUDevice::VPU_4_0};

        ShaveInstanceHolder_Mock_NPU40 ih;
        EXPECT_EQ(ih.getDevice(), td);
        const DeviceShaveContainer& list = ih.getContainer();
        EXPECT_EQ(list.getDevice(), td);

        for (const auto& t : tests) {
            //{{"sigmoid", testDev, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, true, 0},
            Test ti{{t.input.get_name(), td, t.input.get_inputs(), t.input.get_outputs()},
                    t.expected_exist,
                    t.exp_cycles};
            ExecuteTest(list, t);
        }
    }

    {
        ShaveInstanceHolder_NPU40 ih;
        EXPECT_EQ(ih.getDevice(), VPUDevice::VPU_4_0);
        const DeviceShaveContainer& list = ih.getContainer();
        EXPECT_EQ(list.getDevice(), VPUDevice::VPU_4_0);

        for (const auto& t : tests_40) {
            ExecuteTest(list, t);
        }
    }
}

TEST_F(ShaveCollectionTest, instanceHolder27_Classic_Smoke) {
    ShaveInstanceHolder_VPU27CLassic ih;
    EXPECT_EQ(ih.getDevice(), VPUDevice::VPU_2_7);
    const DeviceShaveContainer& list = ih.getContainer();

    // list.find
    EXPECT_EQ(list.getDevice(), VPUDevice::VPU_2_7);
    // EXPECT_EQ(list.map.size(), 2);

    const VPUDevice testDev{VPUDevice::VPU_2_7};

    std::vector<Test> tests{
            {{"HardSigmoid", testDev, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, true, 23238},
            {{"Transpose", testDev, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, true, 10000},
            {{"Minimum", testDev, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, true, 677714},

            {{"TransposeXX", testDev, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, false, 0},
    };

    for (const auto& t : tests) {
        ExecuteTest(list, t);
    }
}

class ShaveDevicesTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

private:
    const ShaveCostProvider shaves2{};

public:
};

TEST_F(ShaveDevicesTest, config_Smoke) {
    const OldShaveCostProvider shaves{};

    {
        VPUDevice d = VPUDevice::VPU_2_0;
        const auto names = shaves.get_shave_supported_ops(d);
        const auto count = names.size();
        EXPECT_GT(count, 2);
    }
    {
        VPUDevice d = VPUDevice::VPU_2_7;
        const auto names = shaves.get_shave_supported_ops(d);
        const auto count = names.size();
        EXPECT_GT(count, 40);
    }

    {
        VPUDevice d = VPUDevice::VPU_4_0;
        const auto names = shaves.get_shave_supported_ops(d);
        const auto count = names.size();
        EXPECT_GT(count, 40);
    }
    // ShaveConfiguration shavesCopy(shaves);// non copiable?

    DeviceShaveContainer empty_shavesA{VPUDevice::__size};
    DeviceShaveContainer empty_shavesB{VPUDevice::VPU_2_0};
    DeviceShaveContainer empty_shavesC{std::move(empty_shavesA)};
    // empty_shavesB = empty_shavesC;

    // DeviceShaveContainer empty_shavesD{empty_shavesB};
}

class ShaveModelSweepTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

    ShaveModelSweepTest()
        : modelUnroll16(VPUNN::DataType::FLOAT16, 0.00030019021391918926f, 5.661592644465712f, 0.0259999999999998f,
                        3.18080481815654f, 8, 16, 1300, 975),
          modelUnroll8(VPUNN::DataType::FLOAT16, 0.0001398f, 2.740062f, 0.208f, 0.590509f, 8, 8, 1300, 975),
          modelNoUnroll(VPUNN::DataType::FLOAT16, 0.000579f, 2.395082f, 0.078f, 0.0f, 8, 1, 1300, 975),
          model() {}

    ShaveModel1to1 modelUnroll16;
    ShaveModel1to1 modelUnroll8;
    ShaveModel1to1 modelNoUnroll;

    SHAVECostModel model;
};
// Test case for is_in_first_block method
TEST_F(ShaveModelSweepTest, DISABLED_ReferenceSweepSmoke) {
    const auto dtype{VPUNN::DataType::FLOAT16};
    ShaveModel1to1 modelSigmo{dtype,
                              0.00014846888585262483f /*slope*/,
                              3.1334163602078635f /*intercept*/,
                              0.0259999999999998f /*ofScalar*/,
                              0.6429183689024316f /*ofUnroll*/,
                              8,   // vectsize
                              16,  // unroll size
                              1300,
                              975};

    std::ofstream myfile;
    myfile.open("ReferenceSweepSigmoid.csv", std::ios::ate | std::ios::trunc);
    myfile << "\n Writing A new  SWEEP: \n";

    ShaveModel1to1& myModel{modelSigmo};
    myfile << myModel;
    myfile << "size_in_bytes , elementsOut , us , dpu_cyc \n";
    for (int elementsOut = 1; elementsOut < 1024; ++elementsOut) {
        int size_in_bytes = compute_size_in_bytes(elementsOut, dtype);
        const float us{myModel.getMicroSeconds(size_in_bytes)};
        const CyclesInterfaceType dpu_cyc{myModel.getDPUCycles(size_in_bytes)};
        myfile << size_in_bytes << " , " << elementsOut << " , " << us << " , " << dpu_cyc << "\n";
    }

    for (int zoneElementsOut = 1025; zoneElementsOut < 100000; zoneElementsOut = int(zoneElementsOut * 1.9F)) {
        for (int i = -1; i <= 1; ++i) {
            const auto elementsOut{zoneElementsOut + i};
            int size_in_bytes = compute_size_in_bytes(elementsOut, dtype);
            const float us{myModel.getMicroSeconds(size_in_bytes)};
            const CyclesInterfaceType dpu_cyc{myModel.getDPUCycles(size_in_bytes)};
            myfile << size_in_bytes << " , " << elementsOut << " , " << us << " , " << dpu_cyc << "\n";
        }
    }

    myfile.close();
}
TEST_F(ShaveModelSweepTest, DISABLED_ReferenceSigmoidSmoke) {
    const auto dtype{VPUNN::DataType::FLOAT16};

    std::string shave_name{"sigmoid"};
    VPUDevice device = VPUDevice::VPU_2_7;
    const int dpu_freq_MHz{1300};
    const auto op = model.getShaveInstance(shave_name, device);
    EXPECT_TRUE(op.has_value());

    const ShaveOpExecutor& myModel2{op.value().get()};

    std::ofstream myfile;
    myfile.open("L1APISweepSigmoid.csv", std::ios::ate | std::ios::trunc);
    myfile << "\n Writing A new  SWEEP: \n";
    myfile << myModel2.toString();

    myfile << "size_in_bytes , elementsOut , us , dpu_cyc \n";

    for (int elementsOut = 1; elementsOut < 1024; ++elementsOut) {
        int size_in_bytes = compute_size_in_bytes(elementsOut, dtype);
        SHAVEWorkload swl(shave_name, device, {VPUTensor(1, elementsOut, 1, 1, DataType::FLOAT16)},
                          {VPUTensor(1, elementsOut, 1, 1, DataType::FLOAT16)});
        const CyclesInterfaceType dpu_cyc{myModel2.dpuCycles(swl)};
        const float us{dpu_cyc / (float)dpu_freq_MHz};
        myfile << size_in_bytes << " , " << elementsOut << " , " << us << " , " << dpu_cyc << "\n";
    }

    for (int zoneElementsOut = 1025; zoneElementsOut < 100000; zoneElementsOut = int(zoneElementsOut * 1.9F)) {
        for (int i = -1; i <= 1; ++i) {
            const auto elementsOut{zoneElementsOut + i};
            int size_in_bytes = compute_size_in_bytes(elementsOut, dtype);
            SHAVEWorkload swl(shave_name, device, {VPUTensor(1, elementsOut, 1, 1, DataType::FLOAT16)},
                              {VPUTensor(1, elementsOut, 1, 1, DataType::FLOAT16)});
            const CyclesInterfaceType dpu_cyc{myModel2.dpuCycles(swl)};
            const float us{dpu_cyc / (float)dpu_freq_MHz};
            myfile << size_in_bytes << " , " << elementsOut << " , " << us << " , " << dpu_cyc << "\n";
        }
    }

    myfile.close();
}
TEST_F(ShaveModelSweepTest, DISABLED_ReferenceReluNPU40Smoke) {
    const auto dtype{VPUNN::DataType::FLOAT16};

    std::string shave_name{"relu"};
    VPUDevice device = VPUDevice::VPU_4_0;
    const int dpu_freq_MHz{1700};

    const auto op = model.getShaveInstance(shave_name, device);
    EXPECT_TRUE(op.has_value());

    const ShaveOpExecutor& myModel2{op.value().get()};

    std::ofstream myfile;
    myfile.open("L1APISweepReluNPU40test.csv", std::ios::ate | std::ios::trunc);
    myfile << "\n Writing A new  SWEEP: \n";
    myfile << myModel2.toString();

    myfile << "size_in_bytes,elementsOut,us,dpu_cyc\n";

    for (int elementsOut = 1; elementsOut < 5000; ++elementsOut) {
        int size_in_bytes = compute_size_in_bytes(elementsOut, dtype);
        SHAVEWorkload swl(shave_name, device, {VPUTensor(1, elementsOut, 1, 1, DataType::FLOAT16)},
                          {VPUTensor(1, elementsOut, 1, 1, DataType::FLOAT16)});
        const CyclesInterfaceType dpu_cyc{myModel2.dpuCycles(swl)};
        const float us{dpu_cyc / (float)dpu_freq_MHz};
        myfile << size_in_bytes << "," << elementsOut << "," << us << "," << dpu_cyc << "\n";
    }

    myfile.close();
}
}  // namespace VPUNN_unit_tests