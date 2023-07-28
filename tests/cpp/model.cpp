// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include "inference/model.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

namespace VPUNN_unit_tests {

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
            {"Hello dear -100-13 special version", "Hello dear ", 100, 13},
            //{"--22", "none", 1, 22}, throws....
            {"-22-5", "none", 22, 5},
            {"", "none", 00, 1},  // latest if empty

    };

    VPUNN::ModelVersion mv;

    EXPECT_EQ(mv.get_NN_name(), ("none"));
    EXPECT_EQ(mv.get_input_interface_version(), 1);
    EXPECT_EQ(mv.get_output_interface_version(), 1);

    for (auto tst : test_vector) {
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

}  // namespace VPUNN_unit_tests
