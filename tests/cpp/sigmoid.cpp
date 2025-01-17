// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "kernels/sigmoid.h"
#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>
#include "core/tensors.h"

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {
class TestSigmoidLayer : public testing::Test {
public:
protected:
    void SetUp() override {
    }
    void sigmoid_test(unsigned int input_channels, unsigned int batch_size = 1) {
        auto input = VPUNN::random_uniform<float>({batch_size, input_channels}, -10.0, 10.0);
        auto expected_output = VPUNN::Tensor<float>({batch_size, input_channels}, 0);

        // Compute the expected output
        for (unsigned int idx = 0; idx < batch_size * input_channels; idx++) {
            expected_output[idx] = 1 / (1 + std::exp(-input[idx]));
        }

        Sigmoid(&input);

        for (unsigned int idx = 0; idx < batch_size * input_channels; idx++) {
            EXPECT_FLOAT_EQ(input[idx], expected_output[idx]);
        }
    }
};

// Demonstrate some basic assertions.
TEST_F(TestSigmoidLayer, BasicAssertions) {
    for (auto batch_size = 1; batch_size <= 10; batch_size++) {
        for (auto input_channels = 1; input_channels <= 500; input_channels += 10) {
            sigmoid_test(input_channels, batch_size);
        }
    }
}

}  // namespace VPUNN_unit_tests
