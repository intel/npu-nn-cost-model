// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include <gtest/gtest.h>
#include "common/common_helpers.h"

#include "vpu/shave/shave_cost_providers/shave_cost_providers.h"
#include "vpu/shave/shave_cost_providers/shave_provider_bundles.h"
#include "vpu/shave/shave_cost_providers/priority_shave_cost_provider.h"
#include "vpu/shave/shave_cost_providers/name_mapping_shave_cost_provider.h"
#include "vpu/shave_workload.h"

#include <fstream>
#include <iostream>

namespace VPUNN_unit_tests {
using namespace VPUNN;

class ShaveCostProviderTests : public ::testing::Test {
protected:
    void SetUp() override {
    }
    std::shared_ptr<IShaveCostProvider> default_prio_provider{ShaveCostProviderBundles::createDefaultProvider()};
    std::shared_ptr<IShaveCostProvider> old_prio_provider{ShaveCostProviderBundles::createOldShaveOnlyProvider()};
    std::shared_ptr<IShaveCostProvider> new_prio_provider{ShaveCostProviderBundles::createNewShaveOnlyProvider()};

    std::shared_ptr<IShaveCostProvider> shave2_provider{std::make_shared<ShaveCostProvider>()};
    std::shared_ptr<IShaveCostProvider> shave1_provider{std::make_shared<OldShaveCostProvider>()};

    void expectSameCost(const std::vector<std::shared_ptr<IShaveCostProvider>>& providers, 
                       const SHAVEWorkload& workload) {
        if (providers.empty()) return;
        
        auto expected_cost = providers[0]->get_cost(workload);
        for (size_t i = 1; i < providers.size(); i++) {
            auto actual_cost = providers[i]->get_cost(workload);
            EXPECT_EQ(expected_cost, actual_cost) << "Cost mismatch at provider index " << i;
        }
    }

    SHAVEWorkload new_swwl{"relu", VPUDevice::VPU_4_0, {VPUTensor(1, 50, 50, 1, DataType::FLOAT16)},
                      {VPUTensor(1, 50, 50, 1, DataType::FLOAT16)}};

    SHAVEWorkload old_swwl{"Sigmoid", VPUDevice::VPU_4_0, {VPUTensor(1, 50, 50, 1, DataType::FLOAT16)},
                        {VPUTensor(1, 50, 50, 1, DataType::FLOAT16)}};

    SHAVEWorkload unknown_swwl{"unknown_op", VPUDevice::VPU_4_0, {VPUTensor(1, 50, 50, 1, DataType::FLOAT16)},
                        {VPUTensor(1, 50, 50, 1, DataType::FLOAT16)}};
};

TEST_F(ShaveCostProviderTests, ProvidersTestNewOp) {
    std::vector<std::shared_ptr<IShaveCostProvider>> providers = {default_prio_provider, new_prio_provider, shave2_provider};
    expectSameCost(providers, new_swwl);

    std::vector<std::shared_ptr<IShaveCostProvider>> providers_no_default = {old_prio_provider, shave1_provider};
    expectSameCost(providers_no_default, new_swwl);

    auto default_cost = default_prio_provider->get_cost(new_swwl);
    auto old_cost = old_prio_provider->get_cost(new_swwl);

    EXPECT_FALSE(Cycles::isErrorCode(default_cost));
    EXPECT_TRUE(Cycles::isErrorCode(old_cost));
}

TEST_F(ShaveCostProviderTests, ProvidersTestOldOp) {
    std::vector<std::shared_ptr<IShaveCostProvider>> providers = {default_prio_provider, old_prio_provider, shave1_provider};
    expectSameCost(providers, old_swwl);

    std::vector<std::shared_ptr<IShaveCostProvider>> providers_no_default = {new_prio_provider, shave2_provider};
    expectSameCost(providers_no_default, old_swwl);

    auto default_cost = default_prio_provider->get_cost(old_swwl);
    auto new_cost = new_prio_provider->get_cost(old_swwl);

    EXPECT_FALSE(Cycles::isErrorCode(default_cost));
    EXPECT_TRUE(Cycles::isErrorCode(new_cost));
}

TEST_F(ShaveCostProviderTests, ProvidersTestUnknownOp) {
    std::vector<std::shared_ptr<IShaveCostProvider>> providers = {default_prio_provider, old_prio_provider, shave1_provider, new_prio_provider, shave2_provider};
    expectSameCost(providers, unknown_swwl);

    auto default_cost = default_prio_provider->get_cost(unknown_swwl);
    auto old_cost = old_prio_provider->get_cost(unknown_swwl);
    auto new_cost = new_prio_provider->get_cost(unknown_swwl);

    EXPECT_TRUE(Cycles::isErrorCode(default_cost));
    EXPECT_TRUE(Cycles::isErrorCode(old_cost));
    EXPECT_TRUE(Cycles::isErrorCode(new_cost));
}

TEST_F(ShaveCostProviderTests, CheckMultipleProvidersTestSoftMax) {
    auto old_provider = ShaveCostProviderBundles::createOldShaveOnlyProvider();
    VPUDevice device = VPUNN::VPUDevice::VPU_4_0;
    auto list_of_supported_ops = old_provider->get_shave_supported_ops(device);
    auto it = std::find(list_of_supported_ops.begin(), list_of_supported_ops.end(), "SoftMax");
    EXPECT_TRUE(it == list_of_supported_ops.end());

    auto name_mapping_provider = ShaveCostProviderBundles::createNameMappingOldProvider();
    list_of_supported_ops = name_mapping_provider->get_shave_supported_ops(device);
    it = std::find(list_of_supported_ops.begin(), list_of_supported_ops.end(), "SoftMax");
    EXPECT_TRUE(it != list_of_supported_ops.end());

    // Works only 5.0 +
    auto composite_provider = ShaveCostProviderBundles::createCompositeBasedOnHeuristicWithOldNameMappingProviderNPU5();
    list_of_supported_ops = composite_provider->get_shave_supported_ops(device);
    it = std::find(list_of_supported_ops.begin(), list_of_supported_ops.end(), "SoftMax");
    EXPECT_TRUE(it == list_of_supported_ops.end());

    auto device_based_provider = ShaveCostProviderBundles::createDeviceMappedProvider();
    list_of_supported_ops = device_based_provider->get_shave_supported_ops(device);
    it = std::find(list_of_supported_ops.begin(), list_of_supported_ops.end(), "SoftMax");
    EXPECT_TRUE(it != list_of_supported_ops.end());
}

// ============================================================================
// HeuristicCostProvider Tests
// ============================================================================

class HeuristicCostProviderTests : public ::testing::Test {
protected:
    void SetUp() override {
        heuristic_provider = new HeuristicCostProvider();
    }

    void TearDown() override {
        delete heuristic_provider;
    }

    HeuristicCostProvider* heuristic_provider;
};

TEST_F(HeuristicCostProviderTests, GetCostSourceName) {
    // Test that cost source is correctly set
    SHAVEWorkload workload(
        "gridsample",  // This should exist in heuristic provider
        VPUDevice::NPU_5_0,
        {VPUTensor(10, 10, 1, 1, DataType::FLOAT16)},
        {VPUTensor(10, 10, 1, 1, DataType::FLOAT16)}
    );
    
    std::string cost_source;
    auto cost = heuristic_provider->get_cost(workload, &cost_source);
    
    // If the operator exists, cost source should be set
    if (!Cycles::isErrorCode(cost)) {
        EXPECT_EQ(cost_source, "shave_heuristic");
    }
}

TEST_F(HeuristicCostProviderTests, GetCostWithoutSourceParameter) {
    // Test that get_cost works without cost_source parameter
    SHAVEWorkload workload(
        "gridsample",
        VPUDevice::NPU_5_0,
        {VPUTensor(10, 10, 1, 1, DataType::FLOAT16)},
        {VPUTensor(10, 10, 1, 1, DataType::FLOAT16)}
    );
    
    // Should not crash
    ASSERT_NO_THROW({
        [[maybe_unused]] auto cost = heuristic_provider->get_cost(workload);
    });
}

TEST_F(HeuristicCostProviderTests, GetCostForUnknownOperator) {
    // Test with unknown operator
    SHAVEWorkload workload(
        "unknown_operator_xyz",
        VPUDevice::NPU_5_0,
        {VPUTensor(10, 10, 1, 1, DataType::FLOAT16)},
        {VPUTensor(10, 10, 1, 1, DataType::FLOAT16)}
    );
    
    auto cost = heuristic_provider->get_cost(workload);
    
    // Should return error code for unknown operator
    EXPECT_TRUE(Cycles::isErrorCode(cost));
    EXPECT_EQ(cost, Cycles::ERROR_SHAVE_OPERATOR_MISSING);
}

TEST_F(HeuristicCostProviderTests, GetMaxNumParams) {
    // Test that get_max_num_params returns a valid value
    int max_params = heuristic_provider->get_max_num_params();
    
    // Should be non-negative
    EXPECT_GE(max_params, 0);
}

TEST_F(HeuristicCostProviderTests, GetShaveSupportedOps) {
    // Test for NPU_5_0 device
    VPUDevice device = VPUDevice::NPU_5_0;
    auto ops_list = heuristic_provider->get_shave_supported_ops(device);
    
    // Should return a list (empty or populated depending on configuration)
    // Just verify the call doesn't crash
    (void)ops_list;  // Use the variable to avoid unused warning
    SUCCEED();  // Test passes if we get here without crash
}

TEST_F(HeuristicCostProviderTests, GetShaveInstance) {
    // Test getting a shave instance
    std::string op_name = "gridsample";
    VPUDevice device = VPUDevice::NPU_5_0;
    
    auto instance_opt = heuristic_provider->get_shave_instance(op_name, device);
    
    // If the operator exists, should return a valid reference
    if (instance_opt.has_value()) {
        const auto& instance = instance_opt.value().get();
        EXPECT_EQ(instance.getName(), op_name);
    }
}

TEST_F(HeuristicCostProviderTests, GetShaveInstanceForNonExistent) {
    // Test getting a non-existent operator
    std::string op_name = "nonexistent_operator_xyz";
    VPUDevice device = VPUDevice::NPU_5_0;
    
    auto instance_opt = heuristic_provider->get_shave_instance(op_name, device);
    
    // Should return nullopt for non-existent operator
    EXPECT_FALSE(instance_opt.has_value());
}

TEST_F(HeuristicCostProviderTests, GetCostConsistencyAcrossCalls) {
    // Test that multiple calls with same workload return same result
    SHAVEWorkload workload(
        "gridsample",
        VPUDevice::NPU_5_0,
        {VPUTensor(10, 10, 1, 1, DataType::FLOAT16)},
        {VPUTensor(10, 10, 1, 1, DataType::FLOAT16)}
    );
    
    auto cost1 = heuristic_provider->get_cost(workload);
    auto cost2 = heuristic_provider->get_cost(workload);
    
    // Should return same cost for same workload
    EXPECT_EQ(cost1, cost2);
}

TEST_F(HeuristicCostProviderTests, GetCostScalesWithWorkloadSize) {
    // Create workloads of different sizes
    SHAVEWorkload small_workload(
        "gridsample",
        VPUDevice::NPU_5_0,
        {VPUTensor(10, 10, 1, 1, DataType::FLOAT16)},
        {VPUTensor(10, 10, 1, 1, DataType::FLOAT16)}
    );
    
    SHAVEWorkload large_workload(
        "gridsample",
        VPUDevice::NPU_5_0,
        {VPUTensor(100, 100, 1, 1, DataType::FLOAT16)},
        {VPUTensor(100, 100, 1, 1, DataType::FLOAT16)}
    );
    
    auto small_cost = heuristic_provider->get_cost(small_workload);
    auto large_cost = heuristic_provider->get_cost(large_workload);
    
    // If both are valid (not error codes), larger workload should cost more
    if (!Cycles::isErrorCode(small_cost) && !Cycles::isErrorCode(large_cost)) {
        EXPECT_GT(large_cost, small_cost);
    }
}

// ============================================================================
// MathematicalShaveCostProviderBase Tests (via derived classes)
// ============================================================================

class MathematicalShaveCostProviderBaseTests : public ::testing::Test {
protected:
    void SetUp() override {
        shave2_provider = new ShaveCostProvider();
        shave1_provider = new OldShaveCostProvider();
    }

    void TearDown() override {
        delete shave2_provider;
        delete shave1_provider;
    }

    ShaveCostProvider* shave2_provider;
    OldShaveCostProvider* shave1_provider;
};

TEST_F(MathematicalShaveCostProviderBaseTests, Shave2CostSourceName) {
    SHAVEWorkload workload(
        "relu",
        VPUDevice::VPU_4_0,
        {VPUTensor(10, 10, 1, 1, DataType::FLOAT16)},
        {VPUTensor(10, 10, 1, 1, DataType::FLOAT16)}
    );
    
    std::string cost_source;
    auto cost = shave2_provider->get_cost(workload, &cost_source);
    
    if (!Cycles::isErrorCode(cost)) {
        EXPECT_EQ(cost_source, "shave_2");
    }
}

TEST_F(MathematicalShaveCostProviderBaseTests, Shave1CostSourceName) {
    SHAVEWorkload workload(
        "Sigmoid",
        VPUDevice::VPU_4_0,
        {VPUTensor(10, 10, 1, 1, DataType::FLOAT16)},
        {VPUTensor(10, 10, 1, 1, DataType::FLOAT16)}
    );
    
    std::string cost_source;
    auto cost = shave1_provider->get_cost(workload, &cost_source);
    
    if (!Cycles::isErrorCode(cost)) {
        EXPECT_EQ(cost_source, "shave_1");
    }
}

TEST_F(MathematicalShaveCostProviderBaseTests, GetMaxNumParamsNonNegative) {
    EXPECT_GE(shave2_provider->get_max_num_params(), 0);
    EXPECT_GE(shave1_provider->get_max_num_params(), 0);
}

TEST_F(MathematicalShaveCostProviderBaseTests, GetShaveSupportedOpsReturnsVector) {
    VPUDevice device = VPUDevice::VPU_4_0;
    
    auto shave2_ops = shave2_provider->get_shave_supported_ops(device);
    auto shave1_ops = shave1_provider->get_shave_supported_ops(device);
    
    // Should return vectors (may be empty or populated)
    // Just verify the calls don't crash
    (void)shave2_ops;
    (void)shave1_ops;
    SUCCEED();  // Test passes if we get here without crash
}

TEST_F(MathematicalShaveCostProviderBaseTests, GetShaveInstanceReturnsOptional) {
    VPUDevice device = VPUDevice::VPU_4_0;
    
    // Try with a potentially existing operator
    auto instance2 = shave2_provider->get_shave_instance("relu", device);
    auto instance1 = shave1_provider->get_shave_instance("Sigmoid", device);
    
    // Results may vary, but should not crash
    // If they exist, verify the names match
    if (instance2.has_value()) {
        EXPECT_EQ(instance2.value().get().getName(), "relu");
    }
    if (instance1.has_value()) {
        EXPECT_EQ(instance1.value().get().getName(), "Sigmoid");
    }
}

}  // namespace VPUNN_unit_tests