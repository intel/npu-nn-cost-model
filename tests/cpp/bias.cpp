// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "kernels/bias.h"
#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include "core/tensors.h"

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {
using namespace VPUNN;

class TestBiasLayer : public testing::Test {
public:
protected:
    void SetUp() override {
    }
    void bias_test(unsigned int vector_size, unsigned int batch_size = 1) {
        auto bias = VPUNN::Tensor<float>({1, vector_size}, 0);
        auto output = VPUNN::Tensor<float>({batch_size, vector_size}, 0);
        auto input = VPUNN::Tensor<float>({batch_size, vector_size}, 0);
        std::random_device rd;   // create a random device to obtain a seed for the random number generator
        std::mt19937 gen(rd());  // initialize the random number generator with the random seed
        std::uniform_real_distribution<float> distrib(0.0, 1.0);  // uniform distribution, range{0.0, 1.0}, we chose
                                                                  // this range because of the old code: rand()/RAND_MAX

        for (unsigned int idx = 0; idx < batch_size * vector_size; idx++) {
            auto random_number = distrib(gen);  // old code: static_cast<float>(rand()) /
                                                // (static_cast<float>(RAND_MAX));
            input[idx] = random_number;
            output[idx] = random_number;
        }

        for (unsigned int idx = 0; idx < vector_size; idx++) {
            bias[idx] = distrib(gen);  // old code: static_cast<float>(rand()) / (static_cast<float>(RAND_MAX));
        }

        BiasOp biasOperator;
        BiasOpBuffer biasBuffer;

        biasBuffer.reserve_bias_space(batch_size);
        biasOperator.Bias(&bias, &output, biasBuffer);

        for (unsigned int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            for (unsigned int vector_idx = 0; vector_idx < vector_size; vector_idx++) {
                auto idx = vector_idx + batch_idx * vector_size;  // [B, C]
                EXPECT_FLOAT_EQ(input[idx] + bias[vector_idx], output[idx]);
            }
        }
    }
};
// Demonstrate some basic assertions.
TEST_F(TestBiasLayer, BasicAssertions) {
    for (auto batch_size = 1; batch_size <= 10; batch_size++) {
        for (auto vector_size = 1; vector_size <= 500; vector_size += 10) {
            bias_test(vector_size, batch_size);
        }
    }
}
}  // namespace VPUNN_unit_tests
