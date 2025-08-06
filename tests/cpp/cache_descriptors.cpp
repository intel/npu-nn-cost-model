// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "core/cache_descriptors.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <ctime>
#include <random>
#include <vector>
// #include "core/cache.h"
//  #include "vpu_cost_model.h"
#include "vpu/types.h"

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {
using namespace VPUNN;

class SHAVECacheDescriptorTest : public testing::Test {
public:
protected:
    void SetUp() override {
    }
    //    VPUNN::VPUCostModel small_cache_model{std::string(""), false, 10};
    //    VPUNN::VPUCostModel no_cache_model{std::string(""), false, 0};
};
// Demonstrate some basic assertions.
TEST_F(SHAVECacheDescriptorTest, DISABLED_BasicAssertionsWL) {
    // const DPUWorkload wl = {
    //         VPUDevice::VPU_2_7,
    //         Operation::CONVOLUTION,
    //         {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // input dimensions
    //         {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // output dimensions
    //         {3, 3},                                       // kernels
    //         {1, 1},                                       // strides
    //         {1, 1, 1, 1},                                 // padding
    //         ExecutionMode::CUBOID_16x16                   // execution mode
    // };
    // DPUWorkload wl2{wl};
    // wl2.device = VPUDevice::VPU_4_0;

    // LRUKeyCache<float, DPUWorkload> mycache(1000);

    // const float v_1{22.23f};
    // mycache.add(wl, v_1);
    // const auto res = mycache.get(wl);

    // EXPECT_TRUE(res != nullptr);
    // EXPECT_EQ(res, v_1);
}

TEST_F(SHAVECacheDescriptorTest, BasicAssertionsSHAVE) {
    {
        const SHAVEWorkload::Param p2{2.9f};
        // p2.f = 2.9f;
        const SHAVEWorkload::Param p3{p2};

        const SHAVEWorkload wl{"sigmoid",
                               VPUDevice::VPU_2_7,
                               {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
                               {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
                               {{1}, p2, p3}};
        const SHAVEWorkload wl2{"sigmoid",
                                VPUDevice::VPU_4_0,
                                {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
                                {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
                                {{1}, p2, p3}};
        // std::hash<int> k;
        // std::cout << k(7);

        //// std::hash<SHAVEWorkload> k1;
        ////  std::cout << k1(wl);

        // p3 < p2;

        // std::map<std::string, int> m;
        // std::map<std::vector<std::string>, int> mvct;
        //(void)mvct.count({"unu", "doi"});

        // std::map<VPUDevice, int> mdv;
        //(void)mdv.count(wl.get_device());

        // std::map<SHAVEWorkload::Param, int> mp;
        //(void)mp.count(p3);

        // std::map<SHAVEWorkload::Parameters, int> mpvct;

        // std::map<VPUTensor, int> mt;
        //(void)mt.count(VPUTensor(10, 100, 5, 1, DataType::FLOAT16));

        // VPUTensor t1(10, 100, 5, 1, DataType::FLOAT16);
        // VPUTensor t2(10, 100, 5, 1, DataType::FLOAT16);
        // t1 < t2;

        // std::map<std::vector<VPUTensor>, int> mtvct;

        // wl2 < wl;
        // std::map<SHAVEWorkload, float> mshv;
        //(void)mshv.count(wl);

        LRUKeyCache<CyclesInterfaceType, SHAVEWorkload> mycache(1000);

        {
            const CyclesInterfaceType v_1{425};
            mycache.add(wl, v_1);
            const auto res = mycache.get(wl);

            EXPECT_TRUE(res.has_value());
            EXPECT_EQ(*res, v_1);
        }
        {
            const CyclesInterfaceType v_2{800};
            mycache.add(wl2, v_2);
            const auto res2 = mycache.get(wl2);
            EXPECT_TRUE(res2.has_value());
            EXPECT_EQ(*res2, v_2);
        }

        EXPECT_EQ(*mycache.get(wl2), 800);
        EXPECT_EQ(*mycache.get(wl), 425);
    }
}

TEST_F(SHAVECacheDescriptorTest, BasicAssertionsSHAVE_LUT) {
    {
        const SHAVEWorkload::Param p2{2.9f};
        // p2.f = 2.9f;
        const SHAVEWorkload::Param p3{p2};

        const SHAVEWorkload wl{"sigmoid",
                               VPUDevice::VPU_2_7,
                               {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
                               {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
                               {{1}, p2, p3}};
        const SHAVEWorkload wl2{"sigmoid",
                                VPUDevice::VPU_4_0,
                                {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
                                {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
                                {{1}, p2, p3}};
        const SHAVEWorkload wl3{
                "sigmoid",
                VPUDevice::VPU_4_0,
                {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
                {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
        };

        /*SimpleLUTKeyCache*/ LRUKeyCache<CyclesInterfaceType, SHAVEWorkload> mycache(1000);

        {
            const CyclesInterfaceType v_1{425};
            mycache.add(wl, v_1);
            const auto res = mycache.get(wl);

            EXPECT_TRUE(res.has_value());
            EXPECT_EQ(*res, v_1);
        }
        {
            const CyclesInterfaceType v_2{800};
            mycache.add(wl2, v_2);
            const auto res2 = mycache.get(wl2);
            EXPECT_TRUE(res2);
            EXPECT_EQ(*res2, v_2);
        }
        EXPECT_TRUE(mycache.get(wl2));
        EXPECT_EQ(*mycache.get(wl2), 800);

        EXPECT_TRUE(mycache.get(wl));
        EXPECT_EQ(*mycache.get(wl), 425);

        EXPECT_FALSE(mycache.get(wl3));
    }
}

// Demonstrate some basic assertions.
TEST_F(SHAVECacheDescriptorTest, CacheBasicTest) {
    // VPUNN::LRUCache<float> cache(1);
    // std::srand(unsigned(std::time(nullptr)));
    // std::vector<float> v1(100), v2(100);
    // for (auto idx = 0; idx < 100; idx++) {
    //     // Generate a random vector and val
    //     std::random_device rd;
    //     std::mt19937 gen(rd());
    //     std::uniform_real_distribution<float> rand_gen(-1.0f, 1.0f);
    //     auto random_float = [&]() {
    //         return rand_gen(gen);
    //     };
    //     std::generate(v1.begin(), v1.end(), random_float);
    //     std::generate(v2.begin(), v2.end(), random_float);
    //     auto val1 = random_float();
    //     auto val2 = random_float();

    //    // Testing that there is no vector
    //    EXPECT_EQ(cache.get(v1), nullptr);
    //    EXPECT_EQ(cache.get(v2), nullptr);

    //    cache.add(v1, val1);

    //    EXPECT_EQ(*cache.get(v1), val1);
    //    EXPECT_EQ(cache.get(v2), nullptr);

    //    cache.add(v2, val2);
    //    EXPECT_EQ(cache.get(v1), nullptr);
    //    EXPECT_EQ(*cache.get(v2), val2);
    //}
}
}  // namespace VPUNN_unit_tests
