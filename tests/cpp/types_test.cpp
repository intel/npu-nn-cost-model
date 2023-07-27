// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include "vpu/types.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

namespace VPUNN_unit_tests {

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

}  // namespace VPUNN_unit_tests
