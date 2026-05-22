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
#include <algorithm>
#include <iterator>
#include <sstream>  // for error formating
#include <type_traits>
#include <utility>
#include "common/common_helpers.h"

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

    EXPECT_FALSE(SmartRanges(1, 65, 16, 32).roundToNextLarger(65));
    EXPECT_EQ(SmartRanges(1, 65, 16, 32).roundToNextLarger(1), 16);

    // Threshold / LCM alignment gap: a larger valid value exists, but candidate-based rounding misses it.
    {
        const SmartRanges r{0, 100, 6, 15};
        EXPECT_TRUE(r.is_in(60));
        std::optional<int> result = r.roundToNextLarger(31);
        EXPECT_EQ(result, 60);
    }

    // 1) Idempotence: input already valid (below threshold and at/above threshold)
    {
        const SmartRanges r{0, 100, 6, 15};
        EXPECT_EQ(r.roundToNextLarger(6), 6);    // valid below threshold
        EXPECT_EQ(r.roundToNextLarger(60), 60);  // valid in LCM-domain
    }

    // 2) Crossing threshold: below threshold but next valid must be LCM-aligned at/after threshold
    {
        const SmartRanges r{0, 200, 6, 15};  // LCM=30, threshold=15
        // 14 is in-range but below threshold; next valid >=14 is 30 (not 18)
        EXPECT_EQ(r.roundToNextLarger(14), 30);
    }

    // 3) "Next valid exists" but is upperBound (edge of range)
    {
        const SmartRanges r{0, 60, 6, 15};  // valid values include 60
        EXPECT_EQ(r.roundToNextLarger(31), 60);
        EXPECT_FALSE(r.roundToNextLarger(61));
    }

    // 3b) Next LCM-aligned value exists conceptually but is out of range => nullopt
    {
        const SmartRanges r{0, 59, 6, 15};  // next LCM alignment is 60, but UB=59
        EXPECT_FALSE(r.roundToNextLarger(31));
    }

    // 4) Threshold above upperBound => should behave like divisor-only rounding
    {
        const SmartRanges r{0, 20, 4, 100};  // threshold never applies
        EXPECT_EQ(r.roundToNextLarger(1), 4);
        EXPECT_EQ(r.roundToNextLarger(17), 20);
    }

    // 5) Negative second_divisor is abs()-normalized at construction: SmartRanges(0, 40, 6, -12) becomes
    //    SmartRanges(0, 40, 6, 12). Threshold=12, LCM=12. Below-threshold domain [0,12) steps by divisor=6,
    //    so valid values are: {0, 6, 12, 24, 36}.
    {
        const SmartRanges r{0, 40, 6, -12};      // normalized to {0, 40, 6, 12}: threshold=12, LCM=12
        EXPECT_EQ(r.roundToNextLarger(1), 6);    // below threshold, next divisor-aligned is 6
        EXPECT_EQ(r.roundToNextLarger(6), 6);    // already valid below threshold
        EXPECT_EQ(r.roundToNextLarger(7), 12);   // crosses threshold, next LCM-aligned is 12
        EXPECT_EQ(r.roundToNextLarger(12), 12);  // at threshold, already valid
        EXPECT_EQ(r.roundToNextLarger(13), 24);  // above threshold, next LCM-aligned is 24
    }

    // 6) Negative bounds are clamped to 0 and negative second_divisor is abs()-normalized:
    //    SmartRanges(-20, 20, 4, -6) becomes SmartRanges(0, 20, 4, 6). Threshold=6, LCM=12.
    //    Below-threshold domain [0,6) steps by divisor=4, above-threshold steps by LCM=12.
    //    Valid values: {0, 4, 12}.
    {
        const SmartRanges r{-20, 20, 4, -6};  // normalized to (0, 20, 4, 6): threshold=6, LCM=12
        // Negative inputs are out of [0, 20] => nullopt
        EXPECT_FALSE(r.roundToNextLarger(-19));
        EXPECT_FALSE(r.roundToNextLarger(-7));
        EXPECT_FALSE(r.roundToNextLarger(-1));
        // Non-negative inputs within [0, 20]
        EXPECT_EQ(r.roundToNextLarger(0), 0);    // 0 is valid (divisible by 4, below threshold 6)
        EXPECT_EQ(r.roundToNextLarger(1), 4);    // below threshold, next divisor-aligned is 4
        EXPECT_EQ(r.roundToNextLarger(4), 4);    // already valid below threshold
        EXPECT_EQ(r.roundToNextLarger(5), 12);   // crosses threshold, next LCM-aligned is 12
        EXPECT_EQ(r.roundToNextLarger(12), 12);  // at/above threshold, already valid
        EXPECT_FALSE(r.roundToNextLarger(13));   // next LCM-aligned would be 24, but 24 > 20
    }

    // 7) No valid elements in-range (tiny span)
    {
        const SmartRanges r{1, 3, 16};
        EXPECT_FALSE(r.roundToNextLarger(1));
        EXPECT_FALSE(r.roundToNextLarger(2));
        EXPECT_FALSE(r.roundToNextLarger(3));
    }

    // 7b) Single valid element: round to it or nullopt if above
    {
        const SmartRanges r{10, 10, 2};
        EXPECT_EQ(r.roundToNextLarger(0), std::optional<int>{});  // out of range (should stay nullopt)
        EXPECT_EQ(r.roundToNextLarger(10), 10);
    }

    // 8) Negative divisor is abs()-normalized at construction: SmartRanges(0, 20, -4) becomes SmartRanges(0, 20, 4).
    //    Divisibility by -4 is identical to divisibility by 4.
    {
        const SmartRanges r{0, 20, -4};
        EXPECT_EQ(r.roundToNextLarger(1), 4);
        EXPECT_EQ(r.roundToNextLarger(4), 4);
        EXPECT_EQ(r.roundToNextLarger(5), 8);
    }
}
TEST_F(SmartRangesTest, DefaultConstructor) {
    SmartRanges r_default;
    EXPECT_EQ(r_default.getLowerBound(), 0);
    EXPECT_EQ(r_default.getUpperBound(), 0);
    EXPECT_TRUE(r_default.is_in(0));
    EXPECT_FALSE(r_default.is_in(1));
}

TEST_F(SmartRangesTest, CopyAndMoveAssignment_Disabled) {
    // SmartRanges is intentionally immutable after construction:
    // - Its members are const.
    // - Copy/move assignment would need to overwrite const members, so operator= is deleted.
    // Any change must be expressed as constructing a new instance (copy/move construction) or via methods that return
    // a new range (e.g. multiply_* / add_*).
    static_assert(!std::is_copy_assignable_v<SmartRanges>);
    static_assert(!std::is_move_assignable_v<SmartRanges>);
}

TEST_F(SmartRangesTest, IncrementAssumingIsValid_Test) {
    // Basic divisor-only stepping: no second_divisor, simple step-by-divisor progression
    {
        const SmartRanges r{0, 20, 4};
        std::optional<int> val = 0;

        val = r.incrementAssumingIsValid(val);
        EXPECT_EQ(val, 4);
        val = r.incrementAssumingIsValid(val);
        EXPECT_EQ(val, 8);
        val = r.incrementAssumingIsValid(val);
        EXPECT_EQ(val, 12);
        val = r.incrementAssumingIsValid(val);
        EXPECT_EQ(val, 16);
        val = r.incrementAssumingIsValid(val);
        EXPECT_EQ(val, 20);
        val = r.incrementAssumingIsValid(val);
        EXPECT_FALSE(val.has_value());  // 24 > 20 => nullopt
    }

    // Incrementing nullopt is a no-op (because nullopt represents end of range)
    {
        const SmartRanges r{0, 20, 4};
        std::optional<int> val = std::nullopt;
        val = r.incrementAssumingIsValid(val);
        EXPECT_FALSE(val.has_value());
    }

    // With second_divisor: below threshold step by divisor, crossing threshold triggers LCM realignment
    {
        const SmartRanges r{0, 40, 6, 12};  // divisor=6, second_divisor=12, LCM=12
        // Valid values: 0, 6, 12, 24, 36
        std::optional<int> val = 0;

        val = r.incrementAssumingIsValid(val);
        EXPECT_EQ(val, 6);  // below threshold, step by divisor=6
        val = r.incrementAssumingIsValid(val);
        EXPECT_EQ(val, 12);  // crosses threshold at 12, alignToNextValid => 12
        val = r.incrementAssumingIsValid(val);
        EXPECT_EQ(val, 24);  // above threshold, step by LCM=12
        val = r.incrementAssumingIsValid(val);
        EXPECT_EQ(val, 36);
        val = r.incrementAssumingIsValid(val);
        EXPECT_FALSE(val.has_value());  // 48 > 40
    }

    // Threshold crossing with non-coprime divisors: candidate after crossing needs LCM realignment
    {
        const SmartRanges r{0, 30, 2, 3};  // divisor=2, second_divisor=3, LCM=6
        // Valid values: 0, 2, 6, 12, 18, 24, 30
        std::optional<int> val = 0;

        val = r.incrementAssumingIsValid(val);
        EXPECT_EQ(val, 2);  // below threshold=3, step by 2
        val = r.incrementAssumingIsValid(val);
        EXPECT_EQ(val, 6);  // 2+2=4 crosses threshold, alignToNextValid(4) => 6
        val = r.incrementAssumingIsValid(val);
        EXPECT_EQ(val, 12);  // above threshold, step by LCM=6
        val = r.incrementAssumingIsValid(val);
        EXPECT_EQ(val, 18);
        val = r.incrementAssumingIsValid(val);
        EXPECT_EQ(val, 24);
        val = r.incrementAssumingIsValid(val);
        EXPECT_EQ(val, 30);
        val = r.incrementAssumingIsValid(val);
        EXPECT_FALSE(val.has_value());
    }

    // Single valid element: next increment yields nullopt
    {
        const SmartRanges r{10, 10, 2};
        std::optional<int> val = 10;
        val = r.incrementAssumingIsValid(val);
        EXPECT_FALSE(val.has_value());  // 12 > 10
    }

    // Stepping lands exactly on upperBound before going past
    {
        const SmartRanges r{0, 8, 4};
        std::optional<int> val = 4;

        val = r.incrementAssumingIsValid(val);
        EXPECT_EQ(val, 8);  // exactly on upperBound
        val = r.incrementAssumingIsValid(val);
        EXPECT_FALSE(val.has_value());  // 12 > 8
    }

    // Negative bounds are clamped to 0: SmartRanges(-8, 4, 4) becomes SmartRanges(0, 4, 4).
    // Valid values: {0, 4}.
    {
        const SmartRanges r{-8, 4, 4};  // normalized to (0, 4, 4)
        std::optional<int> val = 0;

        val = r.incrementAssumingIsValid(val);
        EXPECT_EQ(val, 4);
        val = r.incrementAssumingIsValid(val);
        EXPECT_FALSE(val.has_value());  // 8 > 4
    }

    // Negative second_divisor is abs()-normalized at construction: SmartRanges(0, 40, 6, -12) becomes
    // SmartRanges(0, 40, 6, 12). Threshold=12, LCM=12. Below-threshold domain [0,12) steps by divisor=6.
    // Valid values: {0, 6, 12, 24, 36}.
    {
        const SmartRanges r{0, 40, 6, -12};  // normalized to (0, 40, 6, 12): threshold=12, LCM=12
        std::optional<int> val = 0;

        val = r.incrementAssumingIsValid(val);
        EXPECT_EQ(val, 6);  // below threshold=12, step by divisor=6
        val = r.incrementAssumingIsValid(val);
        EXPECT_EQ(val, 12);  // 6+6=12 crosses threshold, next LCM-aligned is 12
        val = r.incrementAssumingIsValid(val);
        EXPECT_EQ(val, 24);  // above threshold, step by LCM=12
        val = r.incrementAssumingIsValid(val);
        EXPECT_EQ(val, 36);
        val = r.incrementAssumingIsValid(val);
        EXPECT_FALSE(val.has_value());  // 48 > 40
    }

    // Large LCM gap: crossing threshold skips past intermediate values
    {
        const SmartRanges r{0, 35, 6, 15};  // divisor=6, second_divisor=15, LCM=30
        // Valid values: 0, 6, 12, 30
        std::optional<int> val = 12;

        val = r.incrementAssumingIsValid(val);
        EXPECT_EQ(val, 30);  // 12+6=18 crosses threshold, alignToNextValid(18) => 30
        val = r.incrementAssumingIsValid(val);
        EXPECT_FALSE(val.has_value());  // 60 > 35
    }

    // Validation against is_in: walk via incrementAssumingIsValid and compare to is_in.
    // Negative bounds are clamped to 0, negative divisors/second_divisors are abs()-normalized.
    {
        const std::vector<SmartRanges> test_ranges{
                SmartRanges{0, 10, 2},      SmartRanges{0, 30, 2, 3},    SmartRanges{16, 64, 16, 32},
                SmartRanges{0, 100, 6, 15}, SmartRanges{0, 40, 6, -12},   // normalized to (0, 40, 6, 12)
                SmartRanges{-10, 10, 5},                                  // normalized to (0, 10, 5)
                SmartRanges{0, 150, 7, 9},  SmartRanges{0, 20, -4},       // normalized to (0, 20, 4)
                SmartRanges{10, 10, 2},     SmartRanges{-20, 20, 4, -6},  // normalized to (0, 20, 4, 6)
        };

        for (const auto& r : test_ranges) {
            std::cout << "Testing range [" << r.getLowerBound() << ", " << r.getUpperBound()
                      << "],  begin is: " << ((r.begin() != r.end()) ? (*r.begin()) : -100000000) << "\n";
            // Build expected sequence by brute-force is_in scan
            std::vector<int> expected;
            for (int v = r.getLowerBound(); v <= r.getUpperBound(); ++v) {
                // std::cout << "Checking value " << v << ": is_in=" << r.is_in(v) << "\n";
                if (r.is_in(v)) {
                    expected.push_back(v);
                }
            }
            // std::cout << "Expected valid values DONE \n ";

            // Build actual sequence via incrementAssumingIsValid
            std::vector<int> actual;
            auto val = r.roundToNextLarger(r.getLowerBound());
            // std::cout << "First valid value is: " << (val.has_value() ? std::to_string(*val) : "nullopt") << "\n";
            while (val.has_value()) {
                actual.push_back(*val);
                val = r.incrementAssumingIsValid(val);
                // std::cout << "Next valid value is: " << (val.has_value() ? std::to_string(*val) : "nullopt") << "\n";
            }

            EXPECT_EQ(actual, expected) << "Mismatch for range [" << r.getLowerBound() << ", " << r.getUpperBound()
                                        << "]";
        }
    }
}

TEST_F(SmartRangesTest, TransformToVector_Test) {
    // We check that the transformSmartRangetoVector() output matches the is_in() results for all values in the range
    auto check_transform_matches_is_in = [](const SmartRanges& range) {
        std::vector<int> expected;
        for (int val = range.getLowerBound(); val <= range.getUpperBound(); ++val) {
            if (range.is_in(val)) {
                expected.push_back(val);
            }
        }

        EXPECT_EQ(range.transformSmartRangetoVector(), expected);
    };

    struct TestCase {
        SmartRanges range;
    };

    // SmartRanges structure: lowerBound, upperBound, divisor, (optional)second_divisor
    const std::vector<TestCase> tests{
            {SmartRanges{0, 10, 2}},        // basic step
            {SmartRanges{0, 30, 2, 3}},     // thresholded
            {SmartRanges{10, 10, 2}},       // single valid
            {SmartRanges{1, 1, 2}},         // single invalid
            {SmartRanges{-10, 10, 5}},      // negative bounds
            {SmartRanges{0, 10, 20}},       // divisor > span
            {SmartRanges{0, 9, 2}},         // upper not hit
            {SmartRanges{10, 0, 2}},        // empty (lb > ub)
            {SmartRanges{16, 64, 16, 32}},  // threshold 32
            {SmartRanges{0, 40, 6, -12}},   // negative threshold
            {SmartRanges{-7, 7, 4}},        // align from neg
            {SmartRanges{5, 6, 4}},         // empty after align
            {SmartRanges{12, 40, 6, 12}},   // thr == lb
            {SmartRanges{0, 12, 6, 12}},    // thr == ub
            {SmartRanges{50, 80, 6, 12}},   // thr < lb
            {SmartRanges{-20, 20, 4, -6}},  // neg thr span
            {SmartRanges{0, 100, 6, 15}},   // non-coprime LCM
            {SmartRanges{0, 150, 7, 9}},    // coprime LCM
            {SmartRanges{1, 2, 3}},          {SmartRanges{-30, 30, 3, -5}},  {SmartRanges{-30, 60, -6, -10}},
            {SmartRanges{-20, 40, 4, -20}},  {SmartRanges{-50, 50, 4, -40}}, {SmartRanges{-100, 100, 6, -90}},
            {SmartRanges{-80, 80, 10, -60}},
    };

    for (const auto& t : tests) {
        check_transform_matches_is_in(t.range);
    }
}

TEST_F(SmartRangesTest, IteratorTest) {
    const SmartRanges r{0, 30, 2, 3};
    const std::vector<int> expected{0, 2, 6, 12, 18, 24, 30};

    // for iterators
    std::vector<int> got_for;
    for (auto it = r.begin(); it != r.end(); ++it) {
        got_for.push_back(*it);
    }
    EXPECT_EQ(got_for, expected);

    // while iterators
    std::vector<int> got_while;
    auto it = r.begin();
    while (it != r.end()) {
        got_while.push_back(*it);
        ++it;
    }
    EXPECT_EQ(got_while, expected);

    // range-based for
    std::vector<int> got_range_for;
    for (const int v : r) {
        got_range_for.push_back(v);
    }
    EXPECT_EQ(got_range_for, expected);

    // for each
    std::vector<int> got_for_each;
    std::for_each(r.begin(), r.end(), [&](int v) {
        got_for_each.push_back(v);
    });
    EXPECT_EQ(got_for_each, expected);

    // Iterator dereference returns by value
    static_assert(std::is_same_v<decltype(*r.begin()), SmartRanges::value_type>, "Iterator must dereference to int");

    // Input iterator: single-pass; algorithms must not rely on multi-pass behavior.
    static_assert(std::is_same_v<std::iterator_traits<SmartRanges::const_iterator>::iterator_category,
                                 std::input_iterator_tag>);

    // Shouldn't be default constructible.
    static_assert(!std::is_default_constructible_v<SmartRanges::const_iterator>);

    // Copy/move semantics of the iterator itself.
    // Copy/move construction is supported; assignment is intentionally disabled (reference member).
    // This limits STL algorithm compatibility -> see the copy assignability section in the iterator's documentation.
    static_assert(std::is_copy_constructible_v<SmartRanges::const_iterator>);
    static_assert(std::is_move_constructible_v<SmartRanges::const_iterator>);

    // Copy/move assignment is supported via a custom operator= that enforces same-range identity.
    // Only the position (current value) is updated; the range reference is never reseated.
    // Cross-range assignment throws std::logic_error.
    static_assert(std::is_copy_assignable_v<SmartRanges::const_iterator>);
    static_assert(std::is_move_assignable_v<SmartRanges::const_iterator>);

    // Pre-increment returns iterator&
    static_assert(
            std::is_same_v<decltype(++std::declval<SmartRanges::const_iterator&>()), SmartRanges::const_iterator&>);

    // Iterator dereferences by value, so iterator_traits<It>::reference and decltype(*it) are value_type and not a
    // reference
    static_assert(
            std::is_same_v<std::iterator_traits<SmartRanges::const_iterator>::reference, SmartRanges::value_type>);
    static_assert(std::is_same_v<decltype(*std::declval<SmartRanges::const_iterator&>()), SmartRanges::value_type>);

    // Pre-increment chaining returns same object
    {
        auto i = r.begin();
        auto& same = ++i;
        EXPECT_EQ(&same, &i);
    }

    // *it stability after increment (by-value dereference => old value copy remains valid)
    {
        auto i = r.begin();
        const int v0 = *i;
        ++i;
        EXPECT_EQ(v0, expected[0]);
        ASSERT_NE(i, r.end());
        EXPECT_EQ(*i, expected[1]);
    }

    // Consume exactly N values and land at end()
    {
        size_t count = 0;
        for (auto i = r.begin(); i != r.end(); ++i) {
            ++count;
        }
        EXPECT_EQ(count, expected.size());
    }
    {
        auto i = r.begin();
        for (size_t k = 0; k < expected.size(); ++k) {
            ASSERT_NE(i, r.end());
            ++i;
        }
        EXPECT_EQ(i, r.end());
    }

    // end() behavior: incrementing end() is a no-op and remains end()
    {
        auto e = r.end();
        ++e;
        EXPECT_EQ(e, r.end());
    }

    // Iterators from different ranges must not be equal
    {
        SmartRanges r2{0, 30, 2, 3};
        EXPECT_NE(r.begin(), r2.begin());
        EXPECT_NE(r.end(), r2.end());
    }

    // Equality/inequality invariants (basic laws inside same range domain)
    {
        auto a = r.begin();
        auto b = r.begin();
        EXPECT_TRUE(a == a);
        EXPECT_TRUE(b == b);
        EXPECT_TRUE(a == b);
        EXPECT_FALSE(a != b);

        ++a;
        EXPECT_TRUE(a != b);
        EXPECT_FALSE(a == b);
    }

    // begin() must align to the first valid value >= lowerBound
    {
        const SmartRanges r_align{1, 10, 2};
        ASSERT_NE(r_align.begin(), r_align.end());
        EXPECT_EQ(*r_align.begin(), 2);
    }

    // Empty iteration when lowerBound > upperBound
    {
        const SmartRanges r_empty{10, 0, 2};
        EXPECT_EQ(r_empty.begin(), r_empty.end());
    }

    // Single-element range where the single value is valid
    {
        const SmartRanges r_one{10, 10, 2};
        ASSERT_NE(r_one.begin(), r_one.end());
        EXPECT_EQ(*r_one.begin(), 10);
        auto i = r_one.begin();
        ++i;
        EXPECT_EQ(i, r_one.end());
    }

    // Single-element range where the single value is invalid
    {
        const SmartRanges r_none{11, 11, 2};
        EXPECT_EQ(r_none.begin(), r_none.end());
    }

    // Negative bounds are clamped to 0: SmartRanges(-9, 3, 4) becomes SmartRanges(0, 3, 4).
    // Valid values in [0, 3] divisible by 4: {0}.
    {
        const SmartRanges r_neg{-9, 3, 4};  // normalized to (0, 3, 4)
        const std::vector<int> expected_neg{0};
        EXPECT_EQ(r_neg.transformSmartRangetoVector(), expected_neg);
        ASSERT_NE(r_neg.begin(), r_neg.end());
        EXPECT_EQ(*r_neg.begin(), 0);
    }

    // Threshold transition behavior
    {
        const SmartRanges r_thresh{0, 40, 6, 12};
        const std::vector<int> expected_thresh{0, 6, 12, 24, 36};

        std::vector<int> got;
        for (int v : r_thresh) {
            got.push_back(v);
        }
        EXPECT_EQ(got, expected_thresh);
    }

    // second_divisor exists but threshold is above upperBound => acts like divisor-only within range
    {
        const SmartRanges r_hi_thresh{0, 10, 2, 100};
        const std::vector<int> expected_hi_thresh{0, 2, 4, 6, 8, 10};
        EXPECT_EQ(r_hi_thresh.transformSmartRangetoVector(), expected_hi_thresh);
    }

    // lowerBound starts above threshold => immediate LCM stepping
    {
        const SmartRanges r_from_above{30, 100, 6, 12};
        const std::vector<int> expected_from_above{36, 48, 60, 72, 84, 96};
        EXPECT_EQ(r_from_above.transformSmartRangetoVector(), expected_from_above);
    }

    // Threshold not divisible by divisor => below-threshold values possible, above-threshold may be empty via LCM
    {
        const SmartRanges r_mismatch{0, 50, 8, 18};
        const std::vector<int> expected_mismatch{0, 8, 16};
        EXPECT_EQ(r_mismatch.transformSmartRangetoVector(), expected_mismatch);
    }

    // cbegin() and cend() equality with begin() and end()
    {
        EXPECT_EQ(r.begin(), r.cbegin());
        EXPECT_EQ(r.end(), r.cend());
    }

    // Negative divisor is abs()-normalized at construction: SmartRanges(0, 20, -4) becomes SmartRanges(0, 20, 4).
    // Divisibility by -4 is identical to divisibility by 4.
    {
        const SmartRanges r_neg_div{0, 20, -4};  // normalized to (0, 20, 4)
        const std::vector<int> expected_neg_div{0, 4, 8, 12, 16, 20};
        EXPECT_EQ(r_neg_div.transformSmartRangetoVector(), expected_neg_div);
        EXPECT_TRUE(r_neg_div.is_in(8));
        EXPECT_FALSE(r_neg_div.is_in(10));
    }

    // Negative second_divisor is abs()-normalized at construction: SmartRanges(0, 40, 6, -12) becomes
    // SmartRanges(0, 40, 6, 12). Threshold=12, LCM=12. Below-threshold domain [0,12) steps by divisor=6,
    // so valid values are: {0, 6, 12, 24, 36}.
    {
        const SmartRanges r_neg_second{0, 40, 6, -12};  // normalized to (0, 40, 6, 12)
        const std::vector<int> expected_neg_second{0, 6, 12, 24, 36};
        EXPECT_EQ(r_neg_second.transformSmartRangetoVector(), expected_neg_second);
        EXPECT_TRUE(r_neg_second.is_in(6));    // valid below threshold=12
        EXPECT_TRUE(r_neg_second.is_in(12));   // valid at threshold
        EXPECT_FALSE(r_neg_second.is_in(18));  // not LCM-aligned (LCM=12)
    }
}

TEST_F(SmartRangesTest, IteratorAssignment_Test) {
    const SmartRanges r{0, 30, 2, 3};
    const std::vector<int> expected{0, 2, 6, 12, 18, 24, 30};

    // Same-range copy assignment: updates position only
    {
        auto a = r.begin();
        auto b = r.begin();
        ++b;
        ++b;
        a = b;
        EXPECT_EQ(*a, *b);
    }

    // Same-range move assignment
    {
        auto a = r.begin();
        auto b = r.begin();
        ++b;
        a = std::move(b);
        EXPECT_EQ(*a, expected[1]);
    }

    // Self-assignment is a no-op
    {
        auto a = r.begin();
        // necessary because clang gives a warning for self-move-assignment even if it's well-defined and safe in this
        // case (iterator remains valid and unchanged)
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wself-assign-overloaded"
#endif
        a = a;
#ifdef __clang__
#pragma clang diagnostic pop
#endif

        EXPECT_EQ(*a, expected[0]);
    }

    // Assign end() within the same range
    {
        auto a = r.begin();
        a = r.end();
        EXPECT_EQ(a, r.end());
    }

    // STL algorithm compatibility: std::find and std::count
    {
        auto it = std::find(r.begin(), r.end(), 12);
        ASSERT_NE(it, r.end());
        EXPECT_EQ(*it, 12);

        EXPECT_EQ(std::find(r.begin(), r.end(), 99), r.end());
        EXPECT_EQ(std::count(r.begin(), r.end(), 6), 1);
    }

    // Cross-range assignment must throw std::logic_error
    {
        const SmartRanges r2{0, 30, 2, 3};  // same config, different instance

        auto a = r.begin();
        auto b = r2.begin();
        EXPECT_THROW(a = b, std::logic_error);
        EXPECT_THROW(a = std::move(b), std::logic_error);

        // Iterator state must remain unchanged after a failed assignment
        EXPECT_EQ(*a, expected[0]);
        EXPECT_EQ(a, r.begin());
    }
}

class MultiSmartRangesTest : public ::testing::Test {};

TEST_F(MultiSmartRangesTest, ConstructionAndCopyMove) {
    SmartRanges r1(0, 10, 2);
    SmartRanges r2(20, 30, 5);

    MultiSmartRanges msr({r1, r2});
    EXPECT_EQ(msr.get_range(0).getLowerBound(), 0);
    EXPECT_EQ(msr.get_range(1).getUpperBound(), 30);

    // Copy constructor: still supported (SmartRanges is copy constructible).
    MultiSmartRanges msr_copy(msr);
    EXPECT_EQ(msr_copy.get_range(0).getLowerBound(), 0);

    // Move constructor: still supported (vector moves by moving elements / buffers).
    MultiSmartRanges msr_move(std::move(msr_copy));
    EXPECT_EQ(msr_move.get_range(1).getUpperBound(), 30);

    // NOTE: MultiSmartRanges is intentionally non-assignable because it owns a std::vector<SmartRanges>
    // and SmartRanges is non-assignable. Allowing MultiSmartRanges::operator= would require
    // std::vector<SmartRanges>::operator=, which relies on SmartRanges assignment and would not compile.
    // The declarations below use initialization (copy/move construction), not assignment.
    /* coverity[copy_instead_of_move] */
    MultiSmartRanges msr_assign = msr;  // copy construction
    EXPECT_EQ(msr_assign.get_range(1).getLowerBound(), 20);

    MultiSmartRanges msr_assign2 = std::move(msr_move);  // move construction
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
    EXPECT_TRUE(msr.is_in(4, mask1));               // in r1
    EXPECT_FALSE(msr.is_in(25, std::move(mask1)));  // not in r1

    std::vector<bool> mask2 = {false, true};
    EXPECT_FALSE(msr.is_in(4, mask2));             // not in r2
    EXPECT_TRUE(msr.is_in(25, std::move(mask2)));  // in r2

    std::vector<bool> mask3 = {false, false};
    EXPECT_FALSE(msr.is_in(4, mask3));              // none enabled
    EXPECT_FALSE(msr.is_in(25, std::move(mask3)));  // none enabled
}

TEST_F(MultiSmartRangesTest, MaskAutoResize) {
    SmartRanges r1(0, 10, 2);
    SmartRanges r2(20, 30, 5);
    MultiSmartRanges msr({r1, r2});

    // mask smaller than ranges: should pad with false
    std::vector<bool> mask = {true};
    EXPECT_TRUE(msr.is_in(4, mask));               // only r1 enabled
    EXPECT_FALSE(msr.is_in(25, std::move(mask)));  // only r1 enabled

    // mask larger than ranges: extra entries ignored
    std::vector<bool> mask_large = {false, true, true, false};
    EXPECT_TRUE(msr.is_in(25, std::move(mask_large)));  // only r2 enabled
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