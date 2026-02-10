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
#include "common/common_helpers.h"
#include "vpu_cost_model.h"
#include "vpu_shave_cost_model.h"

#include <unordered_map>

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {
using namespace VPUNN;

/// @brief Tests that the Shave objects can be created. Not covering every functionality/shave
class TestSHAVE : public ::testing::Test {
public:
protected:
    VPUNN::SHAVECostModel empty_model{};
    VPUNN::SHAVECostModel model_with_cache{std::string{"../../../models/shave_5_1.cachebin"}, 16384, true};

    std::shared_ptr<VPUNN::PriorityShaveCostProvider> only_new_provider{std::make_shared<VPUNN::PriorityShaveCostProvider>(ShaveCostProviderBundles::createNewShaveOnlyProviders())};
    
    class SHAVECostModelTest : public VPUNN::SHAVECostModel {
        public:
            SHAVECostModelTest(std::shared_ptr<IShaveCostProvider> provider) 
                : VPUNN::SHAVECostModel(provider) {}
    };

    SHAVECostModelTest model_with_new_provider{only_new_provider};

    const VPUNN::VPUTensor input_0{56, 56, 16, 1, VPUNN::DataType::FLOAT16};   // input dimensions
    const VPUNN::VPUTensor output_0{56, 56, 16, 1, VPUNN::DataType::FLOAT16};  // output dimensions

    void SetUp() override {
    }

private:
};

TEST_F(TestSHAVE, DISABLED_CheckTheCacheTest){
    
    const VPUNN::VPUTensor input_tensor{128,64,64,1, VPUNN::DataType::FLOAT16, Layout::XYZ};
    const VPUNN::VPUTensor out_tensor{128,64,64,1, VPUNN::DataType::FLOAT16, Layout::XYZ};
    
    SHAVEWorkload::Param param = 2;
    SHAVEWorkload::Parameters params = {param};
    SHAVEWorkload::ExtraParameters extra_params = {};
    
    std::string false_string = "False";
    std::string true_string = "True";
    std::string level = "VPU";
    float eps = 0.000010f;
    
    extra_params["across_channels"] = false_string;
    extra_params["eps"] = eps;
    extra_params["high_precision_normalize"] = false_string;
    extra_params["normalize_variance"] = true_string;

    SHAVEWorkload swwl {
        "MVN_2Ax",
        VPUDevice::NPU_5_0,
        {input_tensor},
        {out_tensor},
        params,
        extra_params
    };

    std::string info;
    auto shave_cycles = model_with_cache.computeCycles(swwl, info);
    EXPECT_GT(shave_cycles, 251000);
    EXPECT_LT(shave_cycles, 252000);
}

TEST_F(TestSHAVE, ShavePresentOldNotNew) {
    SHAVEWorkload swwl {
        "HardSigmoid",
        VPUDevice::VPU_4_0,
        {input_0},
        {output_0},
    };
    EXPECT_EQ(swwl.get_device(), VPUNN::VPUDevice::VPU_4_0);
    ASSERT_EQ(swwl.get_inputs().size(), 1);
    ASSERT_EQ(swwl.get_outputs().size(), 1);
    ASSERT_EQ(swwl.get_params().size(), 0);

    std::string info;
    auto shave_cycles = empty_model.computeCycles(swwl, info);
    EXPECT_GT(shave_cycles, Cycles::NO_ERROR);
    EXPECT_LE(shave_cycles, Cycles::START_ERROR_RANGE);

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
        auto shave_cycles = empty_model.computeCycles(swwl, info);
        EXPECT_EQ(shave_cycles, V(Cycles::ERROR_SHAVE_OPERATOR_MISSING));
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
        auto shave_cycles = empty_model.computeCycles(swwl, info);
        EXPECT_EQ(shave_cycles, V(Cycles::ERROR_SHAVE_OPERATOR_MISSING));
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
    SHAVECostModel model_shave_cache{temp_cache_file, 16384, true};

    {
        std::string info;
        auto shave_cycles = model_shave_cache.computeCycles(wlp7, info);
        EXPECT_EQ(shave_cycles, V(1));
    }
    {
        std::string info;
        auto shave_cycles = model_shave_cache.computeCycles(wlp10, info);
        EXPECT_EQ(shave_cycles, V(2));
    }
    {  // not in oracolo cache
        std::string info;
        auto shave_cycles = model_shave_cache.computeCycles(wls, info);
        EXPECT_GT(shave_cycles, V(100)) << wls << info;
    }
    {  // not in oracolo cache
        std::string info;
        auto shave_cycles = empty_model.computeCycles(wls, info);
        EXPECT_GT(shave_cycles, V(100)) << wls << info;
    }
}

TEST_F(TestSHAVE, SHAVE_v2_ListOfOperators) {
    // EXPECT_TRUE(false);
    {
        const auto d{VPUDevice::VPU_2_7};
        auto ops = model_with_new_provider.getShaveSupportedOperations(d);

        EXPECT_EQ(ops.size(), 71 + 3);

        std::cout << "\n -------------------------- DEVICE : " << VPUDevice_ToText.at((int)d)
                  << "  has # SHAVE operators : " << ops.size() << " -----------------------------------";
        for (const auto& o : ops) {
            std::cout << "\n  : " << o;
        }
    }

    {
        const auto d{VPUDevice::VPU_4_0};
        auto ops = model_with_new_provider.getShaveSupportedOperations(d);

        EXPECT_EQ(ops.size(), 80);

        std::cout << "\n -------------------------- DEVICE : " << VPUDevice_ToText.at((int)d)
                  << "  has # SHAVE operators : " << ops.size() << " -----------------------------------";
        for (const auto& o : ops) {
            std::cout << "\n  : " << o;
        }
    }
    {
        const auto d{VPUDevice::NPU_5_0};
        auto ops = model_with_new_provider.getShaveSupportedOperations(d);

        EXPECT_EQ(ops.size(), 80);

        std::cout << "\n -------------------------- DEVICE : " << VPUDevice_ToText.at((int)d)
                  << "  has # SHAVE operators : " << ops.size() << " -----------------------------------";
        for (const auto& o : ops) {
            std::cout << "\n  : " << o;
        }
    }
    {  // special in-existing
        const auto d{VPUDevice::__size};
        auto ops = model_with_new_provider.getShaveSupportedOperations(d);
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
            EXPECT_TRUE(shv.has_value());
            const auto& shave_instance = shv.value().get();
            std::cout << "Shave instance details: " << shave_instance.toString() << "\n";
            
            EXPECT_GT(shave_instance.toString().length(), 50);
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
            
            EXPECT_TRUE(shv.has_value());
            const auto& shave_instance = shv.value().get();
            std::cout << "Shave instance details: " << shave_instance.toString() << "\n";
            
            EXPECT_GT(shave_instance.toString().length(), 50);
        }
    }
}
TEST_F(TestSHAVE, SHAVE_v2_ListOfOperatorsDetails_5x) {
    {
        const auto d{VPUDevice::NPU_5_0};
        auto ops = empty_model.getShaveSupportedOperations(d);

        const auto ops_cnt{ops.size()};
        EXPECT_GT(ops_cnt, 1);

        std::cout << "\n -------------------------- DEVICE : " << VPUDevice_ToText.at((int)d)
                  << "  has # SHAVE operators : " << ops_cnt << " -----------------------------------";
        int i{1};
        for (const auto& o : ops) {
            std::cout << "\n*** ----------- " << i++ << " of " << ops_cnt << " : " << o << " ---------\n";
            const auto& shv = empty_model.getShaveInstance(o, d);
            EXPECT_TRUE(shv.has_value());
            const auto& shave_instance = shv.value().get();
            std::cout << "Shave instance details: " << shave_instance.toString() << "\n";
            
            EXPECT_GT(shave_instance.toString().length(), 50);
        }
    }

}

}  // namespace VPUNN_unit_tests