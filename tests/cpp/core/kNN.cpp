// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "kernels/kNN.h"

#include <gtest/gtest.h>
#include <iostream>
#include "core/tensors.h"

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {
class TestkNNOneHot : public testing::Test {
public:
protected:
    void SetUp() override {
    }
    void one_hot_test(unsigned int db_items, unsigned int embedding_size, unsigned int item_index = 0,
                      unsigned int batch_size = 1) {
        unsigned int output_size = 1;
        unsigned int offset = 42;

        auto weights = VPUNN::Tensor<float>({db_items, embedding_size}, 0);
        auto targets = VPUNN::Tensor<float>({db_items, output_size}, 0);
        for (unsigned int idx = 0; idx < db_items; idx++) {
            targets[idx] = static_cast<float>(idx + offset);
            for (unsigned int embedding_idx = 0; embedding_idx < embedding_size; embedding_idx++) {
                weights.data()[embedding_idx + embedding_size * idx] = embedding_idx == idx ? 1.0f : 0.0f;
            }
        }

        auto input = VPUNN::Tensor<float>({batch_size, embedding_size}, 0);
        input[item_index] = 1;
        auto output = VPUNN::Tensor<float>({batch_size, output_size}, 0);

        kNN(&weights, &targets, &input, &output);

        unsigned int expected = item_index + offset;
        EXPECT_EQ(roundf(output[0]), expected);
    }
};
// Demonstrate some basic assertions.
TEST_F(TestkNNOneHot, BasicAssertions) {
    auto max_items = 20;

    for (auto items = 1; items <= max_items; items++) {
        for (auto item_index = 0; item_index < items; item_index++) {
            for (auto batch_size = 1; batch_size <= 10; batch_size++) {
                one_hot_test(items, items, item_index, batch_size);
            }
        }
    }
}

}  // namespace VPUNN_unit_tests
