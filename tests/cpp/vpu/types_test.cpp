// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include "vpu/types.h"

#include <gtest/gtest.h>

#include "common/common_helpers.h"
#include "vpu/cycles_interface_types.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

namespace VPUNN_unit_tests {
using namespace VPUNN;

class TestTypes : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }

private:
};

/// Test cases covering the creation of the Tensor via its simple constructor, no init value
TEST_F(TestTypes, layout_to_order_test) {
    EXPECT_EQ(VPUNN::layout_to_order(VPUNN::Layout::ZMAJOR), VPUNN::layout_to_order(VPUNN::Layout::ZXY))
            << "must be equivalent layouts";
    EXPECT_EQ(VPUNN::layout_to_order(VPUNN::Layout::CMAJOR), VPUNN::layout_to_order(VPUNN::Layout::XYZ))
            << "must be equivalent layouts";

    // different
    EXPECT_NE(VPUNN::layout_to_order(VPUNN::Layout::CMAJOR), VPUNN::layout_to_order(VPUNN::Layout::ZMAJOR))
            << "must be different";

    EXPECT_NE(VPUNN::layout_to_order(VPUNN::Layout::ZXY), VPUNN::layout_to_order(VPUNN::Layout::XYZ))
            << "must be different";
}

TEST_F(TestTypes, CompareValues_isEqualTest) {
    uint16_t v1 = 3;
    uint16_t v2 = 9;
    uint16_t diff1 = 8;

    CyclesInterfaceType v3 = 20;
    CyclesInterfaceType v4 = 22;
    CyclesInterfaceType diff2 = 5;
    CyclesInterfaceType diff3 = 1;

    // case values are equal because their difference is smaller than the given value 20-10<15
    ASSERT_TRUE(CompareValues::isEqual(10, 20, 15, 5.0F));

    // case values are not equal because their difference is greater than the given value 20-15>5 and also the tolerance
    // is too small to claim that the values are equal 2%*10=0,2
    ASSERT_FALSE(CompareValues::isEqual(10, 20, 5, 2.0F));

    // case values are equal because of tolerance
    ASSERT_TRUE(CompareValues::isEqual(1000, 1050, 10, 6.0F));

    // case we test unsigned values, they are equal because of the given differnece value
    ASSERT_TRUE(CompareValues::isEqual(v1, v2, diff1, 3.0F));

    // case we test float values, they are equal because of tolerance
    ASSERT_TRUE(CompareValues::isEqual(15.7F, 15.9F, 0.0F, 3.0F));

    // case we test CycleInterfaceType (uint32), they are equal because of the given difference value (v4-v3<diff3 =>
    // 22-20<5)
    ASSERT_TRUE(CompareValues::isEqual(v3, v4, diff2, 5.0F));

    // case we test CycleInterfaceType (uint32), they are equal because of tolerance (15%*20=3 => 22-20<3)
    ASSERT_TRUE(CompareValues::isEqual(v3, v4, diff3, 15.0F));

    // case where the tolerance is too small to claim that the values are equal
    ASSERT_FALSE(CompareValues::isEqual(10, 20, 5, 60.0F));

    // case where the tolerance is very close to the difference between values, but not enough to consider them equal
    ASSERT_FALSE(CompareValues::isEqual(901, 1000, 5, 10.0F));

    // case we have negative values. Only debug compile will have the assert throwing
    // EXPECT_DEATH(CompareValues::isEqual(10, -20, 5, 60.0F), "Assertion.* failed");
}

}  // namespace VPUNN_unit_tests
