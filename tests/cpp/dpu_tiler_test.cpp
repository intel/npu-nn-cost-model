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
// #include "vpu_layer_cost_model.h"
#include "core/logger.h"
#include "vpu/optimization/dimension_tiler.h"

namespace VPUNN_unit_tests {
using namespace VPUNN;
// using namespace std::placeholders;

class DPUTilerTest : public ::testing::Test {
public:
protected:
    void SetUp() override {
        Logger::clear2ndlog();
        Logger::activate2ndlog();
    }
    void TearDown() override {
        Logger::clear2ndlog();
        Logger::deactivate2ndlog();
    }

public:
    struct TestInput {
        int dimension{0};  ///< Dimension to split
        int nTiles{0};     ///< Number of tiles
    };

    struct TestExpectations {
        std::vector<int> expected{};
    };

protected:
    struct TestCase {
        TestInput t_in;
        TestExpectations t_exp;
        const std::string test_case = "";
    };
    using TestsVector = std::vector<TestCase>;
    using container = SplitDimension::SplitContainer;

    auto execute(const TestCase& t, const SplitDimension& dut) const {
        container result{};
        if (t.t_exp.expected.size() > 0) {
            EXPECT_TRUE(dut.divideBalanced(t.t_in.dimension, t.t_in.nTiles, result))
                    << t.test_case << " Dimension: " << t.t_in.dimension << " , split in: " << t.t_in.nTiles << ".";
            EXPECT_EQ(result, t.t_exp.expected)
                    << t.test_case << " Dimension: " << t.t_in.dimension << " , split in: " << t.t_in.nTiles << ".\n"
                    << " Size result: " << result.size() << "  Expected Size: " << t.t_exp.expected.size();
        } else {
            EXPECT_FALSE(dut.divideBalanced(t.t_in.dimension, t.t_in.nTiles, result))
                    << t.test_case << " Dimension: " << t.t_in.dimension << " , split in: " << t.t_in.nTiles << ".";
        }
    };

public:
protected:
    // void executeT(std::function<void(const TestInput&, const TestExpectations&, const std::string&)>& f,
    //               const TestsVector& tests, std::string h = "") {
    //     int test_index = 0;
    //     for (const auto& t : tests) {
    //         std::stringstream buffer;
    //         buffer << test_index << " : " << t.test_case;
    //         const std::string test_case_info = buffer.str();

    //        f(t.t_in, t.t_exp, h + test_case_info);

    //        ++test_index;
    //    }
    //}
};

TEST_F(DPUTilerTest, SmokeTest) {
    SmartRanges rangeN{1, 8192};
    SplitDimension dut{rangeN};

    {
        container result{};
        EXPECT_FALSE(dut.divideBalanced(10, 0, result));  // no splits is  invalid for naturals
        EXPECT_EQ(result.size(), 0);
    }
    {
        container result{};
        EXPECT_TRUE(dut.divideBalanced(0, 0, result));  // 0/0 OK  by definition
        EXPECT_EQ(result.size(), 0);
    }

    {
        container result{};
        EXPECT_FALSE(dut.divideBalanced(1, 0, result));  // /0
        EXPECT_EQ(result.size(), 0);
    }
    {
        container result{};
        EXPECT_FALSE(dut.divideBalanced(1, 2, result));  // no solution
        EXPECT_LE(result.size(), 1);
    }

    {
        container result{};
        EXPECT_TRUE(dut.divideBalanced(1, 1, result));
        EXPECT_EQ(result.size(), 1);
        container result_e{1};
        EXPECT_EQ(result, result_e);
    }

    {
        container result{};
        EXPECT_TRUE(dut.divideBalanced(16, 1, result));
        EXPECT_EQ(result.size(), 1);
        container result_e{16};
        EXPECT_EQ(result, result_e);
    }

    {
        container result{};
        EXPECT_TRUE(dut.divideBalanced(10, 10, result));
        EXPECT_EQ(result.size(), 10);
        container result_e{1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        EXPECT_EQ(result, result_e);
    }
    {
        container result{};
        EXPECT_TRUE(dut.divideBalanced(10, 1, result));
        EXPECT_EQ(result.size(), 1);
        container result_e{10};
        EXPECT_EQ(result, result_e);
    }

    {
        container result{};
        EXPECT_TRUE(dut.divideBalanced(10, 2, result));
        EXPECT_EQ(result.size(), 2);
        container result_e{5, 5};
        EXPECT_EQ(result, result_e);
    }
    {
        container result{};
        EXPECT_TRUE(dut.divideBalanced(10, 3, result));
        EXPECT_EQ(result.size(), 3);
        container result_e{4, 3, 3};
        EXPECT_EQ(result, result_e);
    }
    {
        container result{};
        EXPECT_TRUE(dut.divideBalanced(10, 1, result));
        EXPECT_EQ(result.size(), 1);
        container result_e{10};
        EXPECT_EQ(result, result_e);
    }

    {
        container result{};
        EXPECT_TRUE(dut.divideBalanced(3, 2, result));
        EXPECT_EQ(result.size(), 2);
        container result_e{2, 1};
        EXPECT_EQ(result, result_e);
    }
    {
        container result{};
        EXPECT_TRUE(dut.divideBalanced(7, 3, result));
        EXPECT_EQ(result.size(), 3);
        container result_e{3, 2, 2};
        EXPECT_EQ(result, result_e);
    }
}

TEST_F(DPUTilerTest, Div16Range_Neg_Test) {
    SmartRanges rangeN{1, 8192, 16};
    SplitDimension dut{rangeN};

    {
        container result{};
        EXPECT_FALSE(dut.divideBalanced(10, 0, result));  // no splits is  invalid for naturals
        EXPECT_EQ(result.size(), 0);
    }
    {
        container result{};
        EXPECT_TRUE(dut.divideBalanced(0, 0, result));  // 0/0 OK  by definition
        EXPECT_EQ(result.size(), 0);
    }

    {
        container result{};
        EXPECT_FALSE(dut.divideBalanced(1, 0, result));  // /0
        EXPECT_EQ(result.size(), 0);
    }
    {
        container result{};
        EXPECT_FALSE(dut.divideBalanced(1, 2, result));  // no solution
        EXPECT_LE(result.size(), 1);
    }
    {
        container result{};
        EXPECT_TRUE(dut.divideBalanced(16, 1, result));
        EXPECT_EQ(result.size(), 1);
        container result_e{16};
        EXPECT_EQ(result, result_e);
    }

    {
        container result{};
        EXPECT_FALSE(dut.divideBalanced(15, 1, result));  // no solution
        EXPECT_LE(result.size(), 1);
    }
    {
        container result{};
        EXPECT_FALSE(dut.divideBalanced(17, 1, result));  // no solution
        EXPECT_LE(result.size(), 1);
    }
    {
        container result{};
        EXPECT_FALSE(dut.divideBalanced(49, 1, result));  // no solution
        EXPECT_LE(result.size(), 1);
    }

    {
        container result{};
        EXPECT_FALSE(dut.divideBalanced(1, 1, result));
        // EXPECT_EQ(result, result_e);
    }

    {
        container result{};
        EXPECT_FALSE(dut.divideBalanced(10, 10, result));
    }
    {
        container result{};
        EXPECT_FALSE(dut.divideBalanced(10, 1, result));
    }

    {
        container result{};
        EXPECT_FALSE(dut.divideBalanced(10, 2, result));
    }
    {
        container result{};
        EXPECT_FALSE(dut.divideBalanced(10, 3, result));
    }
    {
        container result{};
        EXPECT_FALSE(dut.divideBalanced(10, 1, result));
    }

    {
        container result{};
        EXPECT_FALSE(dut.divideBalanced(3, 2, result));
    }
    {
        container result{};
        EXPECT_FALSE(dut.divideBalanced(7, 3, result));
    }
}

TEST_F(DPUTilerTest, Div16Range_Norm_Test) {
    SmartRanges rangeN{1, 8192, 16};
    SplitDimension dut_local{rangeN};

    std::vector<TestCase> tests{
            {{10, 2}, {{}}, "10/2"},             //
            {{16, 1}, {{16}}, "16/1"},           //
            {{48, 1}, {{48}}, "Div16"},          //
            {{48, 2}, {{32, 16}}, "Div16"},      //
            {{48, 3}, {{16, 16, 16}}, "Div16"},  //
            {{48, 4}, {{}}, "Div16"},            //

            {{80, 2}, {{48, 32}}, "Div16"},                                   //
            {{80, 3}, {{32, 32, 16}}, "Div16"},                               //
            {{8192, 2}, {{8192 / 2, 8192 / 2}}, "Div16 8192."},               //
            {{8192 + 16, 2}, {{8192 / 2 + 16, 8192 / 2}}, "Div16 8192+16."},  //
            {{8192 * 2, 2}, {{8192, 8192}}, "Div16 8192*2."},                 //
            {{8192 * 2 + 16, 2}, {{}}, "Div16 8192*2+16."},                   //
    };

    {
        container result{};
        EXPECT_FALSE(dut_local.divideBalanced(1, 2, result));  // no solution
        EXPECT_LE(result.size(), 1);
    }
    {
        container result{};
        EXPECT_TRUE(dut_local.divideBalanced(16, 1, result));
        EXPECT_EQ(result.size(), 1);
        container result_e{16};
        EXPECT_EQ(result, result_e);
    }

    for (const auto& t : tests) {
        execute(t, dut_local);
    }
}

TEST_F(DPUTilerTest, Range16_32_64_BasicTest) {
    SmartRanges rangeN{16, 64, 16, 32};
    SplitDimension dut_local{rangeN};

    std::vector<TestCase> tests{
            {{10, 2}, {{}}, "10/2"},  //

            {{16, 1}, {{16}}, "16/1"},  //
            {{16, 2}, {{}}, "16/2"},    //

            {{32, 1}, {{32}}, ""},      //
            {{32, 2}, {{16, 16}}, ""},  //
            {{32, 3}, {{}}, ""},        //

            {{48, 1}, {{}}, "D32"},           //
            {{48, 2}, {{32, 16}}, "D32/16"},  //
            {{48, 3}, {{16, 16, 16}}, ""},    //
            {{48, 4}, {{}}, ""},              //

            {{64, 1}, {{64}}, "64"},            //
            {{64, 2}, {{32, 32}}, ""},          //
            {{64, 3}, {{32, 16, 16}}, ""},      //
            {{64, 4}, {{16, 16, 16, 16}}, ""},  //

            {{80, 1}, {{}}, ""},        //
            {{80, 2}, {{64, 16}}, ""},  //

            {{96, 2}, {{64, 32}}, ""},                  //
            {{96, 3}, {{32, 32, 32}}, ""},              //
            {{96, 4}, {{32, 32, 16, 16}}, ""},          //
            {{96, 5}, {{32, 16, 16, 16, 16}}, ""},      //
            {{96, 6}, {{16, 16, 16, 16, 16, 16}}, ""},  //

            {{112, 3}, {{64, 32, 16}}, ""},      //
            {{176, 4}, {{64, 64, 32, 16}}, ""},  //
            {{144, 4}, {{64, 32, 32, 16}}, ""},  //
            {{128, 4}, {{32, 32, 32, 32}}, ""},  //

            {{176, 6}, {{32, 32, 32, 32, 32, 16}}, ""},  //
            {{160, 6}, {{32, 32, 32, 32, 16, 16}}, ""},  //
            {{144, 6}, {{32, 32, 32, 16, 16, 16}}, ""},  //

            {{208, 6}, {{64, 32, 32, 32, 32, 16}}, ""},  //
            {{224, 6}, {{64, 32, 32, 32, 32, 32}}, ""},  //

            {{1200, 20},
             {{64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 32, 16}},
             "1200/20TEST"},  //

    };

    {
        container result{};
        EXPECT_FALSE(dut_local.divideBalanced(1, 2, result));  // no solution
        EXPECT_LE(result.size(), 1);
    }
    {
        container result{};
        EXPECT_TRUE(dut_local.divideBalanced(16, 1, result));
        EXPECT_EQ(result.size(), 1);
        container result_e{16};
        EXPECT_EQ(result, result_e);
    }

    for (const auto& t : tests) {
        execute(t, dut_local);
    }
}

TEST_F(DPUTilerTest, Range16_32_64_BIGTest) {
    // std::vector<TestCase> tests{
    //         {{1200, 20},
    //          {{64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 32, 16}},
    //          "1200/20TEST"},  //

    //};

    for (int i = 1; i <= 255; ++i) {
        container result{};
        const SmartRanges rangeN{16, 64, 16, 32};
        const SplitDimension dut_local{rangeN};
        EXPECT_NO_THROW(dut_local.divideBalanced(1200, i, result));
    }
}

}  // namespace VPUNN_unit_tests
