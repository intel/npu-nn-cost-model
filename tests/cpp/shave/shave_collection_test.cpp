// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include "vpu/shave/shave_op_executors.h"

#include <gtest/gtest.h>
#include "common/common_helpers.h"
#include "vpu/shave/shave_collection.h"
#include "vpu/shave/shave_devices.h"
#include "vpu/types.h"
#include "vpu_cost_model.h"
#include "vpu_shave_cost_model.h"
#include "vpu/shave/shave_cost_providers/shave_cost_providers.h"

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

#ifdef INTEL_EMBARGO_NPU5
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
#endif


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

#ifdef INTEL_EMBARGO_NPU5
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
#endif


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