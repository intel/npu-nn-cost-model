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

// TEST_F(SmartRangesTest, RangeSize_Test) {
//     EXPECT_EQ(SmartRanges(16, 64, 16, 32).range_size(), 3);
//     EXPECT_EQ(SmartRanges(16, 64, 16).range_size(), 4);
//     EXPECT_EQ(SmartRanges(10, 64, 2, 10).range_size(), 6);
//     EXPECT_EQ(SmartRanges(10, 64, 16, 3).range_size(), 4);
//     EXPECT_EQ(SmartRanges(10, 64, 2, 32).range_size(), 13);
//     EXPECT_EQ(SmartRanges(1, 20, 2, 5).range_size(), 4);
//     EXPECT_EQ(SmartRanges(1, 20, 5).range_size(), 4);
// }

TEST_F(SmartRangesTest, IsInSimple_Test) {
    EXPECT_TRUE(SmartRanges(20, 100).is_in(50));
    EXPECT_FALSE(SmartRanges(20, 100, 16).is_in(50));
    EXPECT_FALSE(SmartRanges(20, 100).is_in(10));
    EXPECT_FALSE(SmartRanges(20, 100, 16).is_in(256));
    EXPECT_TRUE(SmartRanges(20, 100, 12).is_in(24));
    EXPECT_FALSE(SmartRanges(20, 100, 12).is_in(25));
    EXPECT_TRUE(SmartRanges(20, 100).is_in(20));
    EXPECT_TRUE(SmartRanges(20, 100).is_in(100));
    EXPECT_TRUE(SmartRanges(20, 100).is_in(31));

    EXPECT_TRUE(SmartRanges(20, 100, 13, 2).is_in(26));
    EXPECT_FALSE(SmartRanges(20, 100, 13, 2).is_in(27));

    EXPECT_TRUE(SmartRanges(16, 64, 16, 32).is_in(16));
    EXPECT_TRUE(SmartRanges(16, 64, 16, 32).is_in(32));
    EXPECT_FALSE(SmartRanges(16, 64, 16, 32).is_in(48));
    EXPECT_TRUE(SmartRanges(16, 64, 16, 32).is_in(64));
}

TEST_F(SmartRangesTest, RoundToNext_Test) {
    EXPECT_TRUE(SmartRanges(20, 100).roundToNextLarger(50));
    EXPECT_EQ(SmartRanges(20, 100).roundToNextLarger(50), 50);
    EXPECT_EQ(SmartRanges(20, 100).roundToNextLarger(51), 51);

     EXPECT_FALSE(SmartRanges(20, 100).roundToNextLarger(19));
    EXPECT_FALSE(SmartRanges(20, 100).roundToNextLarger(101));

    EXPECT_EQ(SmartRanges(20, 100, 16).roundToNextLarger(50), 64);
    EXPECT_EQ(SmartRanges(20, 100, 16).roundToNextLarger(51), 64);
    EXPECT_EQ(SmartRanges(20, 100, 16).roundToNextLarger(64), 64);
    EXPECT_EQ(SmartRanges(20, 100, 16).roundToNextLarger(20), 32);
    EXPECT_EQ(SmartRanges(20, 100, 16).roundToNextLarger(21), 32);
    EXPECT_EQ(SmartRanges(20, 100, 16).roundToNextLarger(96), 96);
    EXPECT_FALSE(SmartRanges(20, 100, 16).roundToNextLarger(97));
    EXPECT_FALSE(SmartRanges(20, 100, 16).roundToNextLarger(98));
    EXPECT_FALSE(SmartRanges(20, 100, 16).roundToNextLarger(99));
    EXPECT_FALSE(SmartRanges(20, 100, 16).roundToNextLarger(100));
    EXPECT_FALSE(SmartRanges(20, 100, 16).roundToNextLarger(101));
    EXPECT_FALSE(SmartRanges(20, 100, 16).roundToNextLarger(19));

    // EXPECT_TRUE(SmartRanges(20, 100, 13, 2).is_in(26));
    // EXPECT_FALSE(SmartRanges(20, 100, 13, 2).is_in(27));

    EXPECT_FALSE(SmartRanges(16, 64, 16, 32).roundToNextLarger(0));

    EXPECT_FALSE(SmartRanges(16, 64, 16, 32).roundToNextLarger(15));

    EXPECT_EQ(SmartRanges(16, 64, 16, 32).roundToNextLarger(16), 16);
    EXPECT_EQ(SmartRanges(16, 64, 16, 32).roundToNextLarger(17), 32);
    EXPECT_EQ(SmartRanges(16, 64, 16, 32).roundToNextLarger(31), 32);
    EXPECT_EQ(SmartRanges(16, 64, 16, 32).roundToNextLarger(32), 32);
    EXPECT_EQ(SmartRanges(16, 64, 16, 32).roundToNextLarger(33), 64);
    EXPECT_EQ(SmartRanges(16, 64, 16, 32).roundToNextLarger(47), 64);
    EXPECT_EQ(SmartRanges(16, 64, 16, 32).roundToNextLarger(48), 64);
    EXPECT_EQ(SmartRanges(16, 64, 16, 32).roundToNextLarger(49), 64);
    EXPECT_EQ(SmartRanges(16, 64, 16, 32).roundToNextLarger(63), 64);
    EXPECT_EQ(SmartRanges(16, 64, 16, 32).roundToNextLarger(64), 64);

    EXPECT_FALSE(SmartRanges(16, 64, 16, 32).roundToNextLarger(65));
}

TEST_F(SmartRangesTest, RoundToNext_Pathological_Test) {

    EXPECT_EQ(SmartRanges(1, 100, 16).roundToNextLarger(1), 16);
    EXPECT_EQ(SmartRanges(1, 100, 16).roundToNextLarger(2), 16);
   


//    EXPECT_EQ(SmartRanges(16, 64, 16, 32).roundToNextLarger(16), 16);


    EXPECT_FALSE(SmartRanges(1, 65, 16, 32).roundToNextLarger(65));
    EXPECT_EQ(SmartRanges(1, 65, 16, 32).roundToNextLarger(1), 16);
}

TEST_F(SmartRangesTest, DefaultConstructorAndAssignment) {
    SmartRanges r_default;
    EXPECT_EQ(r_default.getLowerBound(), 0);
    EXPECT_EQ(r_default.getUpperBound(), 0);
    EXPECT_TRUE(r_default.is_in(0));
    EXPECT_FALSE(r_default.is_in(1));

    SmartRanges r1(10, 20, 2);
    SmartRanges r2;
    r2 = r1;
    EXPECT_EQ(r2.getLowerBound(), 10);
    EXPECT_EQ(r2.getUpperBound(), 20);
    EXPECT_TRUE(r2.is_in(12));
    EXPECT_FALSE(r2.is_in(13));
}

TEST_F(SmartRangesTest, CopyAndMoveConstructor) {
    SmartRanges r1(5, 15, 2);
    SmartRanges r2(r1);  // copy
    EXPECT_EQ(r2.getLowerBound(), 5);
    EXPECT_EQ(r2.getUpperBound(), 15);

    SmartRanges r3(std::move(r2));  // move
    EXPECT_EQ(r3.getLowerBound(), 5);
    EXPECT_EQ(r3.getUpperBound(), 15);
}

TEST_F(SmartRangesTest, CopyAndMoveAssignment) {
    SmartRanges r1(1, 9, 2);
    SmartRanges r2;
    r2 = r1;
    EXPECT_EQ(r2.getLowerBound(), 1);
    EXPECT_EQ(r2.getUpperBound(), 9);

    SmartRanges r3;
    r3 = std::move(r2);
    EXPECT_EQ(r3.getLowerBound(), 1);
    EXPECT_EQ(r3.getUpperBound(), 9);
}

class MultiSmartRangesTest : public ::testing::Test {};

TEST_F(MultiSmartRangesTest, ConstructionAndCopyMove) {
    SmartRanges r1(0, 10, 2);
    SmartRanges r2(20, 30, 5);

    MultiSmartRanges msr({r1, r2});
    EXPECT_EQ(msr.get_range(0).getLowerBound(), 0);
    EXPECT_EQ(msr.get_range(1).getUpperBound(), 30);

    // Copy constructor
    MultiSmartRanges msr_copy(msr);
    EXPECT_EQ(msr_copy.get_range(0).getLowerBound(), 0);

    // Move constructor
    MultiSmartRanges msr_move(std::move(msr_copy));
    EXPECT_EQ(msr_move.get_range(1).getUpperBound(), 30);

    // Copy assignment
    MultiSmartRanges msr_assign = msr;
    EXPECT_EQ(msr_assign.get_range(1).getLowerBound(), 20);

    // Move assignment
    MultiSmartRanges msr_assign2 = std::move(msr_move);
    EXPECT_EQ(msr_assign2.get_range(0).getUpperBound(), 10);
}

TEST_F(MultiSmartRangesTest, IsInWithoutMask) {
    SmartRanges r1(0, 10, 2);   // even numbers 0-10
    SmartRanges r2(20, 30, 5);  // multiples of 5, 20-30
    MultiSmartRanges msr({r1, r2});

    EXPECT_TRUE(msr.is_in(4));    // in r1
    EXPECT_FALSE(msr.is_in(3));   // not in r1 or r2
    EXPECT_TRUE(msr.is_in(25));   // in r2
    EXPECT_FALSE(msr.is_in(15));  // not in r1 or r2
}

TEST_F(MultiSmartRangesTest, IsInWithMask) {
    SmartRanges r1(0, 10, 2);
    SmartRanges r2(20, 30, 5);
    MultiSmartRanges msr({r1, r2});

    std::vector<bool> mask1 = {true, false};
    EXPECT_TRUE(msr.is_in(4, mask1));    // in r1
    EXPECT_FALSE(msr.is_in(25, mask1));  // not in r1

    std::vector<bool> mask2 = {false, true};
    EXPECT_FALSE(msr.is_in(4, mask2));  // not in r2
    EXPECT_TRUE(msr.is_in(25, mask2));  // in r2

    std::vector<bool> mask3 = {false, false};
    EXPECT_FALSE(msr.is_in(4, mask3));   // none enabled
    EXPECT_FALSE(msr.is_in(25, mask3));  // none enabled
}

TEST_F(MultiSmartRangesTest, MaskAutoResize) {
    SmartRanges r1(0, 10, 2);
    SmartRanges r2(20, 30, 5);
    MultiSmartRanges msr({r1, r2});

    // mask smaller than ranges: should pad with false
    std::vector<bool> mask = {true};
    EXPECT_TRUE(msr.is_in(4, mask));    // only r1 enabled
    EXPECT_FALSE(msr.is_in(25, mask));  // only r1 enabled

    // mask larger than ranges: extra entries ignored
    std::vector<bool> mask_large = {false, true, true, false};
    EXPECT_TRUE(msr.is_in(25, mask_large));  // only r2 enabled
}

TEST_F(MultiSmartRangesTest, GetRangeAndBounds) {
    SmartRanges r1(0, 10, 2);
    SmartRanges r2(20, 30, 5);
    MultiSmartRanges msr({r1, r2});

    EXPECT_EQ(msr.get_range(0).getLowerBound(), 0);
    EXPECT_EQ(msr.get_range(1).getLowerBound(), 20);

    EXPECT_EQ(msr.getUpperBound(), 30);
    EXPECT_EQ(msr.getLowerBound(), 0);
}

TEST_F(MultiSmartRangesTest, MultiplyLowerAndUpper) {
    SmartRanges r1(1, 5, 1);
    SmartRanges r2(10, 20, 2);
    MultiSmartRanges msr({r1, r2});

    MultiSmartRanges msr_lower = msr.multiply_lower(2);
    EXPECT_EQ(msr_lower.get_range(0).getLowerBound(), 2);
    EXPECT_EQ(msr_lower.get_range(1).getLowerBound(), 20);

    MultiSmartRanges msr_upper = msr.multiply_upper(3);
    EXPECT_EQ(msr_upper.get_range(0).getUpperBound(), 15);
    EXPECT_EQ(msr_upper.get_range(1).getUpperBound(), 60);
}

TEST_F(MultiSmartRangesTest, OutOfRangeThrows) {
    SmartRanges r1(0, 10, 2);
    MultiSmartRanges msr({r1});
    EXPECT_THROW(msr.get_range(1), std::out_of_range);
}

}  // namespace VPUNN_unit_tests