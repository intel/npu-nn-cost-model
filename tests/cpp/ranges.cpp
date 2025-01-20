// Copyright � 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the �Software Package�)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the �third-party-programs.txt� or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/ranges.h"
#include <gtest/gtest.h>
#include <sstream>  // for error formating
#include "common_helpers.h"

namespace VPUNN_unit_tests {
using namespace VPUNN;
class SmartRangesTest : public ::testing::Test {};

TEST_F(SmartRangesTest, Is_Value_In_Interval_Test) {
    std::string text{""};

    EXPECT_TRUE(SmartRanges(20, 100).is_in(50, text));
    EXPECT_FALSE(SmartRanges(20, 100, 16).is_in(50, text));
    EXPECT_FALSE(SmartRanges(20, 100).is_in(10, text));
    EXPECT_FALSE(SmartRanges(20, 100, 16).is_in(256, text));
    EXPECT_TRUE(SmartRanges(20, 100, 12).is_in(24, text));
    EXPECT_TRUE(SmartRanges(20, 100).is_in(20, text));
    EXPECT_TRUE(SmartRanges(20, 100).is_in(100, text));

    EXPECT_TRUE(SmartRanges(20, 100, 13, 2).is_in(26, text));
    EXPECT_FALSE(SmartRanges(20, 100, 13, 2).is_in(27, text));

    EXPECT_TRUE(SmartRanges(16, 64, 16, 32).is_in(16, text));
    EXPECT_TRUE(SmartRanges(16, 64, 16, 32).is_in(32, text));
    EXPECT_FALSE(SmartRanges(16, 64, 16, 32).is_in(48, text));
    EXPECT_TRUE(SmartRanges(16, 64, 16, 32).is_in(64, text));

}

TEST_F(SmartRangesTest, Multiply_UpperBound_Test) {
    SmartRanges range{20, 100, 16};
    auto result = range.multiply_upper(10);

    EXPECT_EQ(result.getUpperBound(), range.getUpperBound() * 10);
    EXPECT_EQ(result.getUpperBound(), 1000);
}

TEST_F(SmartRangesTest, Multiply_LowerBound_Test) {
    SmartRanges range{20, 1000, 16};
    auto result = range.multiply_lower(10);

    EXPECT_EQ(result.getLowerBound(), range.getLowerBound() * 10);
    EXPECT_EQ(result.getLowerBound(), 200);
}

TEST_F(SmartRangesTest, Add_UpperBound_Test) {
    SmartRanges range{20, 100, 16};
    auto result = range.add_upper(10);

    EXPECT_EQ(result.getUpperBound(), range.getUpperBound() + 10);
    EXPECT_EQ(result.getUpperBound(), 110);
}

TEST_F(SmartRangesTest, Add_LowerBound_Test) {
    SmartRanges range{20, 100, 16};
    auto result = range.add_lower(10);

    EXPECT_EQ(result.getLowerBound(), range.getLowerBound() + 10);
    EXPECT_EQ(result.getLowerBound(), 30);
}

//TEST_F(SmartRangesTest, RangeSize_Test) {
//    EXPECT_EQ(SmartRanges(16, 64, 16, 32).range_size(), 3);
//    EXPECT_EQ(SmartRanges(16, 64, 16).range_size(), 4);
//    EXPECT_EQ(SmartRanges(10, 64, 2, 10).range_size(), 6);
//    EXPECT_EQ(SmartRanges(10, 64, 16, 3).range_size(), 4);
//    EXPECT_EQ(SmartRanges(10, 64, 2, 32).range_size(), 13);
//    EXPECT_EQ(SmartRanges(1, 20, 2, 5).range_size(), 4);
//    EXPECT_EQ(SmartRanges(1, 20, 5).range_size(), 4);
//}

}  // namespace VPUNN_unit_tests