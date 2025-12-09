// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "kernels/fully_connected.h"

#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>
#include "core/tensors.h"

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {
class TestFCLayer : public testing::Test {
public:
protected:
    void SetUp() override {
    }
    void fc_test(unsigned int input_channels, unsigned int output_channels, unsigned int batch_size = 1) {
        auto weights = VPUNN::random_uniform<float>({output_channels, input_channels}, -10.0f, 10.0f);
        auto input = VPUNN::random_uniform<float>({batch_size, input_channels}, -10.0f, 10.0f);
        auto output = VPUNN::zeros<float>({batch_size, output_channels});
        auto expected_output = VPUNN::zeros<float>({batch_size, output_channels});

        VPUNN::Dense(&weights, &input, &output);

        for (unsigned int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            for (unsigned int outC = 0; outC < output_channels; outC++) {
                for (unsigned int inC = 0; inC < input_channels; inC++) {
                    expected_output[outC + batch_idx * output_channels] +=
                            weights[inC + input_channels * outC] * input[inC + batch_idx * input_channels];
                }
                // Check
                auto idx = outC + batch_idx * output_channels;
                if (output[idx] != 0) {
                    // Relative error less than 1%
                    EXPECT_LE(std::abs(expected_output[idx] - output[idx]) / output[idx], 0.01f);
                } else {
                    EXPECT_FLOAT_EQ(output[idx], 0.0);
                }
            }
        }
    }
};
// Demonstrate some basic assertions.
TEST_F(TestFCLayer, BasicAssertions) {
    for (auto batch_size = 1; batch_size <= 10; batch_size++) {
        for (auto output_channels : {1, 10, 20, 50, 100, 250, 500}) {
            for (auto input_channels : {1, 10, 20, 50, 100, 250, 500}) {
                fc_test(input_channels, output_channels, batch_size);
            }
        }
    }
}
}  // namespace VPUNN_unit_tests
