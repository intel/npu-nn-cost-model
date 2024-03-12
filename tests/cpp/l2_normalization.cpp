// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "kernels/l2_normalization.h"
#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>
#include "core/tensors.h"

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {
class TestL2NormLayer : public testing::Test {
public:
protected:
    void SetUp() override {
    }

    void l2_test(unsigned int vector_size, unsigned int batch_size = 1) {
        auto input = VPUNN::Tensor<float>({batch_size, vector_size}, 0);
        auto output = VPUNN::Tensor<float>({batch_size, vector_size}, 0);
        for (unsigned int idx = 0; idx < batch_size * vector_size; idx++) {
            /* coverity[dont_call] */
            input[idx] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX));
        }

        L2Normalization(&input, &output);

        float* output_data = output.data();

        for (unsigned int idx = 0; idx < batch_size; idx++) {
            auto start_idx = output_data + idx * vector_size;
            float l2_norm = vectorNorm(start_idx, vector_size);
            EXPECT_NEAR(l2_norm, 1.0, 0.0001);
        }
    }

    template <typename Iter_T>
    float vectorNorm(Iter_T iter, size_t size) {
        float acc = 0;
        for (unsigned int idx = 0; idx < size; idx++) {
            acc += iter[idx] * iter[idx];
        }
        return std::sqrt(acc);
    }
};

// Demonstrate some basic assertions.
TEST_F(TestL2NormLayer, BasicAssertions) {
    for (auto batch_size = 1; batch_size <= 10; batch_size++) {
        for (auto vector_size = batch_size; vector_size <= 500; vector_size += 10) {
            l2_test(vector_size, batch_size);
        }
    }
}

}  // namespace VPUNN_unit_tests
