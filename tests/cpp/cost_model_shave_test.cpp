// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include "vpu/shave/activation.h"
#include "vpu/shave/data_movement.h"
#include "vpu/shave/elementwise.h"

#include "vpu/shave/layers.h"

#include <gtest/gtest.h>
#include "common_helpers.h"
#include "vpu_cost_model.h"

#include <unordered_map>

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {
using namespace VPUNN;

/// @brief Tests that the Shave objects can be created. Not covering every functionality/shave
class TestSHAVE : public ::testing::Test {
public:
protected:
    VPUNN::VPUCostModel empty_model{};

    const VPUNN::VPUTensor input_0{56, 56, 16, 1, VPUNN::DataType::FLOAT16};   // input dimensions
    const VPUNN::VPUTensor output_0{56, 56, 16, 1, VPUNN::DataType::FLOAT16};  // output dimensions

    void SetUp() override {
    }

private:
};

/// @brief tests that an activation can be instantiated
TEST_F(TestSHAVE, BasicAssertionsActivationCategory) {
    constexpr unsigned int efficiencyx1K{2000};
    constexpr unsigned int latency{1000};
    auto swwl = VPUNN::SHVActivation<efficiencyx1K, latency>(VPUNN::VPUDevice::VPU_2_7,
                                                             input_0,  // input dimensions
                                                             output_0  // output dimensions
    );
    // init part
    EXPECT_EQ(swwl.device, VPUNN::VPUDevice::VPU_2_7);
    ASSERT_EQ(swwl.inputs.size(), 1);
    EXPECT_EQ(swwl.inputs[0].size(), input_0.size()) << " clearly they are not the same" << std::endl;

    ASSERT_EQ(swwl.outputs.size(), 1);
    EXPECT_EQ(swwl.outputs[0].size(), output_0.size()) << " clearly they are not the same" << std::endl;

    // behavioral part
    EXPECT_NEAR(swwl.getKernelEfficiency(), efficiencyx1K / 1000.0F, 0.001F);
    EXPECT_EQ(swwl.getLatency(), latency);

    const auto shave_cycles_sigmoid = empty_model.SHAVE(swwl);
    const auto op_count = swwl.outputs[0].size();

    EXPECT_NEAR(shave_cycles_sigmoid, std::round(op_count / swwl.getKernelEfficiency()) + swwl.getLatency(), 1.0F);
}

/// @brief tests that an Element wise can be instantiated
TEST_F(TestSHAVE, BasicAssertionsELMWiseCategory) {
    constexpr unsigned int efficiencyx1K{800};
    constexpr unsigned int latency{1300};
    auto swwl = VPUNN::SHVElementwise<efficiencyx1K, latency>(VPUNN::VPUDevice::VPU_2_7, {input_0},  // input dimensions
                                                              output_0  // output dimensions
    );
    // init part
    EXPECT_EQ(swwl.device, VPUNN::VPUDevice::VPU_2_7);
    ASSERT_EQ(swwl.inputs.size(), 1);
    EXPECT_EQ(swwl.inputs[0].size(), input_0.size()) << " clearly they are not the same" << std::endl;

    ASSERT_EQ(swwl.outputs.size(), 1);
    EXPECT_EQ(swwl.outputs[0].size(), output_0.size()) << " clearly they are not the same" << std::endl;

    // behavioral part
    EXPECT_NEAR(swwl.getKernelEfficiency(), efficiencyx1K / 1000.0F, 0.001F);
    EXPECT_EQ(swwl.getLatency(), latency);

    const auto shave_cycles_sigmoid = empty_model.SHAVE(swwl);
    const auto op_count = swwl.outputs[0].size();

    EXPECT_NEAR(shave_cycles_sigmoid, std::round(op_count / swwl.getKernelEfficiency()) + swwl.getLatency(), 1.0F);
}

/// @brief tests that an Data Movement can be instantiated
TEST_F(TestSHAVE, BasicAssertionsDataMovementCategory) {
    constexpr unsigned int efficiencyx1K{2050};
    constexpr unsigned int latency{3000};
    auto swwl = VPUNN::SHVDataMovement<efficiencyx1K, latency>(VPUNN::VPUDevice::VPU_2_7,
                                                               input_0,  // input dimensions
                                                               output_0  // output dimensions
    );
    // init part
    EXPECT_EQ(swwl.device, VPUNN::VPUDevice::VPU_2_7);
    ASSERT_EQ(swwl.inputs.size(), 1);
    EXPECT_EQ(swwl.inputs[0].size(), input_0.size()) << " clearly they are not the same" << std::endl;

    ASSERT_EQ(swwl.outputs.size(), 1);
    EXPECT_EQ(swwl.outputs[0].size(), output_0.size()) << " clearly they are not the same" << std::endl;

    // behavioral part
    EXPECT_NEAR(swwl.getKernelEfficiency(), efficiencyx1K / 1000.0F, 0.001F);
    EXPECT_EQ(swwl.getLatency(), latency);

    const auto shave_cycles_sigmoid = empty_model.SHAVE(swwl);
    const auto op_count = swwl.outputs[0].size();

    EXPECT_NEAR(shave_cycles_sigmoid, std::round(op_count / swwl.getKernelEfficiency()) + swwl.getLatency(), 1.0F);
}

/// @brief tests that an Sigmoid can be instantiated
TEST_F(TestSHAVE, BasicAssertionsActivationSigmoid) {
    auto swwl = VPUNN::SHVSigmoid(VPUNN::VPUDevice::VPU_2_7,
                                  input_0,  // input dimensions
                                  output_0  // output dimensions
    );
    EXPECT_EQ(swwl.device, VPUNN::VPUDevice::VPU_2_7);
    ASSERT_EQ(swwl.inputs.size(), 1);
    EXPECT_EQ(swwl.inputs[0].size(), input_0.size()) << " clearly they are not the same" << std::endl;

    ASSERT_EQ(swwl.outputs.size(), 1);
    EXPECT_EQ(swwl.outputs[0].size(), output_0.size()) << " clearly they are not the same" << std::endl;

    auto shave_cycles_sigmoid = empty_model.SHAVE(swwl);
    // Expect equality.
    EXPECT_GE(shave_cycles_sigmoid, 0u);
}

/// @brief tests that V2 prototypeinterface s usable
TEST_F(TestSHAVE, SHAVE_v2_Smoke) {
    {
        SHAVEWorkload swwl{
                "UnspecifiedName",
                VPUDevice::VPU_2_7,
                {input_0},
                {output_0},
        };
        EXPECT_EQ(swwl.get_device(), VPUNN::VPUDevice::VPU_2_7);
        ASSERT_EQ(swwl.get_inputs().size(), 1);
        ASSERT_EQ(swwl.get_outputs().size(), 1);
        ASSERT_EQ(swwl.get_params().size(), 0);

        std::string info;
        auto shave_cycles = empty_model.SHAVE_2(swwl, info);
        EXPECT_EQ(shave_cycles, V(Cycles::ERROR_SHAVE));
    }

    {
        const SHAVEWorkload::Param p2{2.9f};
        // p2.f = 2.9f;
        const SHAVEWorkload::Param p3{p2};

        // SHAVEWorkload::Parameters p{{1}, {.f=2.1f}};//only C++20

        SHAVEWorkload swwl{"UnspecifiedName2", VPUDevice::VPU_4_0, {input_0}, {output_0}, {{1}, p2, p3}};
        EXPECT_EQ(swwl.get_device(), VPUNN::VPUDevice::VPU_4_0);
        ASSERT_EQ(swwl.get_inputs().size(), 1);
        ASSERT_EQ(swwl.get_outputs().size(), 1);
        ASSERT_EQ(swwl.get_params().size(), 3);

        std::string info;
        auto shave_cycles = empty_model.SHAVE_2(swwl, info);
        EXPECT_EQ(shave_cycles, V(Cycles::ERROR_SHAVE));
    }
}

TEST_F(TestSHAVE, SHAVE_v2_Cache_Smoke) {
    const SHAVEWorkload wlp7{
            "sigmoid",
            VPUDevice::VPU_4_0,
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {{7}},
    };

    const SHAVEWorkload wlp10{
            "sigmoid",
            VPUDevice::VPU_4_0,
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {{10}},
    };

    const SHAVEWorkload wls{
            "sigmoid",
            VPUDevice::VPU_4_0,
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},

    };

    const std::string temp_cache_file{"SHAVE_v2_Cache_Smoke.cache_bin"};
    FixedCache shave_cacheToBe{temp_cache_file};

    shave_cacheToBe.insert(wlp7.hash(), 1.0f);
    shave_cacheToBe.insert(wlp10.hash(), 2.0f);

    shave_cacheToBe.write_cache(temp_cache_file);

    // need to load a cache that contains 1 & 2  values already cached
    VPUCostModel model_shave_cache{"", "", temp_cache_file};

    {
        std::string info;
        auto shave_cycles = model_shave_cache.SHAVE_2(wlp7, info);
        EXPECT_EQ(shave_cycles, V(1));
    }
    {
        std::string info;
        auto shave_cycles = model_shave_cache.SHAVE_2(wlp10, info);
        EXPECT_EQ(shave_cycles, V(2));
    }
    {  // not in oracolo cache
        std::string info;
        auto shave_cycles = model_shave_cache.SHAVE_2(wls, info);
        EXPECT_GT(shave_cycles, V(100)) << wls << info;
    }
    {  // not in oracolo cache
        std::string info;
        auto shave_cycles = empty_model.SHAVE_2(wls, info);
        EXPECT_GT(shave_cycles, V(100)) << wls << info;
    }
}

TEST_F(TestSHAVE, SHAVE_v2_ListOfOperators) {
    // EXPECT_TRUE(false);
    {
        const auto d{VPUDevice::VPU_2_7};
        auto ops = empty_model.getShaveSupportedOperations(d);

        EXPECT_EQ(ops.size(), 71 + 3);

        std::cout << "\n -------------------------- DEVICE : " << VPUDevice_ToText.at((int)d)
                  << "  has # SHAVE operators : " << ops.size() << " -----------------------------------";
        for (const auto& o : ops) {
            std::cout << "\n  : " << o;
        }
    }

    {
        const auto d{VPUDevice::VPU_4_0};
        auto ops = empty_model.getShaveSupportedOperations(d);

        EXPECT_EQ(ops.size(), 80);

        std::cout << "\n -------------------------- DEVICE : " << VPUDevice_ToText.at((int)d)
                  << "  has # SHAVE operators : " << ops.size() << " -----------------------------------";
        for (const auto& o : ops) {
            std::cout << "\n  : " << o;
        }
    }

    {  // special in-existing
        const auto d{VPUDevice::__size};
        auto ops = empty_model.getShaveSupportedOperations(d);
        EXPECT_EQ(ops.size(), 0);

        std::cout << "\n -------------------------- DEVICE : OUT of RANGE "
                  << "  has # SHAVE operators : " << ops.size() << " -----------------------------------";
        for (const auto& o : ops) {
            std::cout << "\n  : " << o;
        }
    }
}

TEST_F(TestSHAVE, SHAVE_v2_ListOfOperatorsDetails_27) {
    //  EXPECT_TRUE(false);
    {
        const auto d{VPUDevice::VPU_2_7};
        auto ops = empty_model.getShaveSupportedOperations(d);

        const auto ops_cnt{ops.size()};
        EXPECT_GT(ops_cnt, 1);

        std::cout << "\n -------------------------- DEVICE : " << VPUDevice_ToText.at((int)d)
                  << "  has # SHAVE operators : " << ops_cnt << " -----------------------------------";
        int i{1};
        for (const auto& o : ops) {
            std::cout << "\n*** ----------- " << i++ << " of " << ops_cnt << " : " << o << " ---------\n";
            const auto& shv = empty_model.getShaveInstance(o, d);
            std::cout << shv.toString();
            EXPECT_GT(shv.toString().length(), 50);
        }
    }
}
TEST_F(TestSHAVE, SHAVE_v2_ListOfOperatorsDetails_40) {
    // EXPECT_TRUE(false);
    {
        const auto d{VPUDevice::VPU_4_0};
        auto ops = empty_model.getShaveSupportedOperations(d);

        const auto ops_cnt{ops.size()};
        EXPECT_GT(ops_cnt, 1);

        std::cout << "\n -------------------------- DEVICE : " << VPUDevice_ToText.at((int)d)
                  << "  has # SHAVE operators : " << ops_cnt << " -----------------------------------";
        int i{1};
        for (const auto& o : ops) {
            std::cout << "\n*** ----------- " << i++ << " of " << ops_cnt << " : " << o << " ---------\n";
            const auto& shv = empty_model.getShaveInstance(o, d);
            std::cout << shv.toString();
            EXPECT_GT(shv.toString().length(), 50);
        }
    }
}

}  // namespace VPUNN_unit_tests