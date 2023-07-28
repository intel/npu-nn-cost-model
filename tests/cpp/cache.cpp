// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "core/cache.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <ctime>
#include <random>
#include <vector>
#include "vpu_cost_model.h"

auto small_cache_model = VPUNN::VPUCostModel(std::string(""), false, 10);
auto no_cache_model = VPUNN::VPUCostModel(std::string(""), false, 0);

// Demonstrate some basic assertions.
TEST(VPUNNCacheTest, BasicAssertions) {
    VPUNN::DPUWorkload wl = {
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                     // kernels
            {1, 1},                                                     // strides
            {1, 1},                                                     // padding
            VPUNN::ExecutionMode::CUBOID_16x16                          // execution mode
    };

    auto dpu_cycles = small_cache_model.DPU(wl);

    for (auto idx = 0; idx < 100; idx++) {
        // Testing caching
        EXPECT_EQ(dpu_cycles, small_cache_model.DPU(wl));
        // Testing correctness
        EXPECT_EQ(dpu_cycles, no_cache_model.DPU(wl));
    }
}

TEST(CacheBasicTest, BasicAssertions) {
    VPUNN::LRUCache<float> cache(1);

    std::srand(unsigned(std::time(nullptr)));
    std::vector<float> v1(100), v2(100);

    for (auto idx = 0; idx < 100; idx++) {
        // Generate a random vector and val
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> rand_gen(-1.0f, 1.0f);
        auto random_float = [&]() {
            return rand_gen(gen);
        };
        std::generate(v1.begin(), v1.end(), random_float);
        std::generate(v2.begin(), v2.end(), random_float);
        auto val1 = random_float();
        auto val2 = random_float();

        // Testing that there is no vector
        EXPECT_EQ(cache.get(v1), nullptr);
        EXPECT_EQ(cache.get(v2), nullptr);

        cache.add(v1, val1);

        EXPECT_EQ(*cache.get(v1), val1);
        EXPECT_EQ(cache.get(v2), nullptr);

        cache.add(v2, val2);
        EXPECT_EQ(cache.get(v1), nullptr);
        EXPECT_EQ(*cache.get(v2), val2);
    }
}
