// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include "inference/model.h"

#include <gtest/gtest.h>

#include "inference/model_version.h"
#include "inference/post_process.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

namespace VPUNN_unit_tests {
using namespace VPUNN;

///@ future tests on model to be here
class TestModel : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }

private:
};

class TestModelVersion : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }

private:
};

/// Test basic cases of normal versions
TEST_F(TestModelVersion, SimpleVersionExtraction) {
    struct Test {
        std::string in_name;
        std::string exp_name;
        int in_v;
        int out_v;
    };

    auto toString = [](const Test& t) {
        std::stringstream buffer;
        buffer << "( " << t.in_name << " , " << t.exp_name << " , " << t.in_v << " , " << t.out_v << " )";
        return buffer.str();
    };
    //to do extend the tests to cover also Nickname and version
    std::vector<Test> test_vector{
            {"VPUNN-10-1", "VPUNN", 10, 1},
            {"VPUNN", "VPUNN", 1, 1},
            {"VPUNN-00-0", "VPUNN", 0, 0},
            {"VPUNN-10-2", "VPUNN", 10, 2},
            {"VPUNN-10-02", "VPUNN", 10, 2},
            {"VPUNN-01-1", "VPUNN", 1, 1},
            {"VPUNN-01-2", "VPUNN", 1, 2},
            {"VPUNN-44-3", "VPUNN", 44, 3},
            {"VPUNN-44-", "VPUNN", 44, 1},
            {"VPUNN-4011-02", "VPUNN", 4011, 2},
            {"Hello dear -100-13 special version", "Hello dear ", 100, 13},
            //{"--22", "none", 1, 22}, throws....
            {"-22-5", "none", 22, 5},
            {"", "none", 00, 1},  // latest if empty
            {"VPUNN-44-3Nickmname and version here, an-y lengths", "VPUNN", 44, 3},
            {"VPUNN-44-3Nickname-22-11blah and version here, an-y lengths66-77", "VPUNN", 44, 3},
            {"VPUNN-44-3 Nickname-22-11blah and version here, an-y lengths66-77", "VPUNN", 44, 3},
            {"VPUNN-44-3-Nickname-22-11blah and version here, an-y lengths66-77", "VPUNN", 44, 3},
            {"VPUNN-45-32 $v0000.0000 Nickname26chars$", "VPUNN", 45, 32},

    };

    VPUNN::ModelVersion mv;

    EXPECT_EQ(mv.get_NN_name(), ("none"));
    EXPECT_EQ(mv.get_input_interface_version(), 1);
    EXPECT_EQ(mv.get_output_interface_version(), 1);

    for (const auto& tst : test_vector) {
        mv.parse_name(tst.in_name);

        EXPECT_EQ(mv.get_NN_name(), tst.exp_name) << "name is not matching." << toString(tst);
        EXPECT_EQ(mv.get_input_interface_version(), tst.in_v) << "VI is not matching." << toString(tst);
        EXPECT_EQ(mv.get_output_interface_version(), tst.out_v) << "VO is not matching." << toString(tst);
    }

    {  // malformed
        std::string in_name{"--22"};
        EXPECT_THROW(mv.parse_name(in_name), std::invalid_argument);

        in_name = "a name-jjjj-1";
        EXPECT_THROW(mv.parse_name(in_name), std::invalid_argument);
    }
}

class PostProcessChecker : public ::testing::Test {
protected:
    void SetUp() override {
    }
};
/// @brief Basic test to check the versions that are supported or not
TEST_F(PostProcessChecker, BasicOutputSupport) {
    struct Test {
        std::string info;
        int output_version;
        bool support;
    };

    std::vector<Test> test_vector{
            {"OUT_LATEST", 0, true},  //
            {"OUT_HW_OVERHEAD_BOUNDED", 1, false},
            {"OUT_CYCLES", 2, true},
            {"OUT_HW_OVERHEAD_UNBOUNDED", 3, false},
            {"MOck for post 2.7", 4, true},
            {"NPU40 post proc", 5, true},
            {"Other versions", 6, false},
            {"Other versions", 199, false},
            {"Other versions", 22, false},
    };

    for (const auto& tst : test_vector) {
        VPUNN::PostProcessSupport support_config(tst.output_version);
        EXPECT_EQ(support_config.is_output_supported(), tst.support) << tst.info << " is expected as " << tst.support;
    }
}
}  // namespace VPUNN_unit_tests
