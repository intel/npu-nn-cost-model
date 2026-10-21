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

#include "vpu/shave/shave_cost_providers/name_mapping_shave_cost_provider.h"
#include "vpu/shave/shave_cost_providers/op_set_override_shave_cost_provider.h"
#include "vpu/shave/shave_cost_providers/priority_shave_cost_provider.h"
#include "vpu/shave/shave_cost_providers/shave_cost_providers.h"
#include "vpu/shave/shave_cost_providers/shave_provider_bundles.h"
#include "vpu/shave_workload.h"

#include <fstream>
#include <iostream>
#include <set>

namespace VPUNN_unit_tests {
using namespace VPUNN;

class FixedCostProvider final : public IShaveCostProvider {
private:
    CyclesInterfaceType fixed_cost;
    int max_params;
    std::vector<std::string> supported_ops;

public:
    FixedCostProvider(CyclesInterfaceType fixed_cost, int max_params, std::vector<std::string> supported_ops)
            : fixed_cost(fixed_cost), max_params(max_params), supported_ops(std::move(supported_ops)) {
    }

    CyclesInterfaceType get_cost(const SHAVEWorkload&, std::string* cost_source = nullptr) const override {
        if (cost_source) {
            *cost_source = "fixed_cost_provider";
        }
        return fixed_cost;
    }

    int get_max_num_params() const override {
        return max_params;
    }

    std::vector<std::string> get_shave_supported_ops(VPUDevice&) const override {
        return supported_ops;
    }

    std::optional<std::reference_wrapper<const ShaveOpExecutor>> get_shave_instance(const std::string&,
                                                                                    VPUDevice&) const override {
        return std::nullopt;
    }
};

class ShaveCostProviderTests : public ::testing::Test {
protected:
    void SetUp() override {
    }
    Shared_IShaveCostProvider default_prio_provider{ShaveCostProviderBundles::createDefaultProvider()};
    Shared_IShaveCostProvider old_prio_provider{ShaveCostProviderBundles::createOldShaveOnlyProvider()};
    Shared_IShaveCostProvider new_prio_provider{ShaveCostProviderBundles::createNewShaveOnlyProvider()};

    Shared_IShaveCostProvider shave2_provider{std::make_shared<ShaveCostProvider>()};
    Shared_IShaveCostProvider shave1_provider{std::make_shared<OldShaveCostProvider>()};

    void expectSameCost(const ShaveCostProviderList& providers, const SHAVEWorkload& workload) {
        if (providers.empty())
            return;

        auto expected_cost = providers[0]->get_cost(workload);
        for (size_t i = 1; i < providers.size(); i++) {
            auto actual_cost = providers[i]->get_cost(workload);
            EXPECT_EQ(expected_cost, actual_cost) << "Cost mismatch at provider index " << i;
        }
    }

    SHAVEWorkload new_swwl{"relu",
                           VPUDevice::VPU_4_0,
                           {VPUTensor(1, 50, 50, 1, DataType::FLOAT16)},
                           {VPUTensor(1, 50, 50, 1, DataType::FLOAT16)}};

    SHAVEWorkload old_swwl{"Sigmoid",
                           VPUDevice::VPU_4_0,
                           {VPUTensor(1, 50, 50, 1, DataType::FLOAT16)},
                           {VPUTensor(1, 50, 50, 1, DataType::FLOAT16)}};

    SHAVEWorkload unknown_swwl{"unknown_op",
                               VPUDevice::VPU_4_0,
                               {VPUTensor(1, 50, 50, 1, DataType::FLOAT16)},
                               {VPUTensor(1, 50, 50, 1, DataType::FLOAT16)}};

    static void expectSameSupportedOps(const std::vector<std::string>& lhs, const std::vector<std::string>& rhs) {
        const std::set<std::string> lhs_set(lhs.begin(), lhs.end());
        const std::set<std::string> rhs_set(rhs.begin(), rhs.end());
        EXPECT_EQ(lhs_set, rhs_set);
    }
};

TEST_F(ShaveCostProviderTests, ProvidersTestNewOp) {
    ShaveCostProviderList providers = {default_prio_provider, new_prio_provider, shave2_provider};
    expectSameCost(providers, new_swwl);

    ShaveCostProviderList providers_no_default = {old_prio_provider, shave1_provider};
    expectSameCost(providers_no_default, new_swwl);

    auto default_cost = default_prio_provider->get_cost(new_swwl);
    auto old_cost = old_prio_provider->get_cost(new_swwl);

    EXPECT_FALSE(Cycles::isErrorCode(default_cost));
    EXPECT_TRUE(Cycles::isErrorCode(old_cost));
}

TEST_F(ShaveCostProviderTests, ProvidersTestOldOp) {
    ShaveCostProviderList providers = {default_prio_provider, old_prio_provider, shave1_provider};
    expectSameCost(providers, old_swwl);

    ShaveCostProviderList providers_no_default = {new_prio_provider, shave2_provider};
    expectSameCost(providers_no_default, old_swwl);

    auto default_cost = default_prio_provider->get_cost(old_swwl);
    auto new_cost = new_prio_provider->get_cost(old_swwl);

    EXPECT_FALSE(Cycles::isErrorCode(default_cost));
    EXPECT_TRUE(Cycles::isErrorCode(new_cost));
}

TEST_F(ShaveCostProviderTests, ProvidersTestUnknownOp) {
    ShaveCostProviderList providers = {default_prio_provider, old_prio_provider, shave1_provider, new_prio_provider,
                                       shave2_provider};
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
    auto op_override_provider = ShaveCostProviderBundles::createHeuristicWithOldNameMappingExtendedOpsOverride();
    list_of_supported_ops = op_override_provider->get_shave_supported_ops(device);
    it = std::find(list_of_supported_ops.begin(), list_of_supported_ops.end(), "SoftMax");
    EXPECT_TRUE(it == list_of_supported_ops.end());

    auto device_based_provider = ShaveCostProviderBundles::createDeviceMappedProvider();
    list_of_supported_ops = device_based_provider->get_shave_supported_ops(device);
    it = std::find(list_of_supported_ops.begin(), list_of_supported_ops.end(), "SoftMax");
    EXPECT_TRUE(it != list_of_supported_ops.end());
}

TEST_F(ShaveCostProviderTests, LightweightQuerySupportedOperationsMatchesDeviceMappedProvider) {
    auto expected_provider = ShaveCostProviderBundles::createDeviceMappedProvider();

    for (int i = 0; i < static_cast<int>(VPUDevice::__size); ++i) {
        VPUDevice device = static_cast<VPUDevice>(i);
        VPUDevice mutable_device = device;
        const auto expected_ops = expected_provider->get_shave_supported_ops(mutable_device);
        const auto queried_ops = ShaveCostProviderBundles::queryDeviceMappedSupportedOperations(device);
        expectSameSupportedOps(expected_ops, queried_ops);
    }
}

TEST_F(ShaveCostProviderTests, LightweightQueryAllDevicesInRangeAreMappedOrKnownUnmapped) {
    // Devices that are intentionally unmapped when their embargo flag is absent.
    // Update this set when a new device is added to VPUDevice and routing policy is decided.
    const std::set<VPUDevice> known_unmapped = {
            VPUDevice::NPU_RESERVED,
            VPUDevice::NPU_RESERVED_1,
    };

    for (int i = 0; i < static_cast<int>(VPUDevice::__size); ++i) {
        const VPUDevice device = static_cast<VPUDevice>(i);
        const auto queried_ops = ShaveCostProviderBundles::queryDeviceMappedSupportedOperations(device);

        if (known_unmapped.count(device)) {
            EXPECT_TRUE(queried_ops.empty())
                    << "Expected device " << i << " to be known-unmapped, but lightweight query returned ops";
        } else {
            EXPECT_FALSE(queried_ops.empty()) << "Device " << i
                                              << " has no mapped lightweight query result. Add routing in "
                                                 "createProviderForDevice or add to known_unmapped intentionally.";
        }
    }
}

TEST_F(ShaveCostProviderTests, CreateDeviceMappedProviderDoesNotThrow) {
    Shared_IShaveCostProvider provider;
    EXPECT_NO_THROW(provider = ShaveCostProviderBundles::createDeviceMappedProvider());
    EXPECT_NE(provider, nullptr);
}

TEST_F(ShaveCostProviderTests, AllDevicesInRangeAreMappedOrKnownUnmapped) {
    // Devices that are intentionally unmapped when their embargo flag is absent.
    // Update this set when a new device is added to VPUDevice and routed in createProviderForDevice.
    const std::set<VPUDevice> known_unmapped = {
            VPUDevice::NPU_RESERVED,
            VPUDevice::NPU_RESERVED_1,
    };

    auto provider = ShaveCostProviderBundles::createDeviceMappedProvider();

    for (int i = 0; i < static_cast<int>(VPUDevice::__size); ++i) {
        VPUDevice device = static_cast<VPUDevice>(i);
        VPUDevice mutable_device = device;
        const auto ops = provider->get_shave_supported_ops(mutable_device);

        if (known_unmapped.count(device)) {
            EXPECT_TRUE(ops.empty()) << "Expected device " << i << " to be unmapped but it returned ops";
        } else {
            EXPECT_FALSE(ops.empty())
                    << "Device " << i
                    << " has no mapped provider — add it to createProviderForDevice or to known_unmapped";
        }
    }
}

TEST_F(ShaveCostProviderTests, DeviceMappedProviderUnmappedDeviceStrictRouting) {
    auto device_based_provider = ShaveCostProviderBundles::createDeviceMappedProvider();

    VPUDevice unmapped_device = VPUDevice::__size;
    const auto ops_from_provider = device_based_provider->get_shave_supported_ops(unmapped_device);
    const auto ops_from_static_query = ShaveCostProviderBundles::queryDeviceMappedSupportedOperations(unmapped_device);

    EXPECT_TRUE(ops_from_provider.empty());
    EXPECT_TRUE(ops_from_static_query.empty());

    const SHAVEWorkload wl_unmapped_device("Sigmoid", VPUDevice::__size, {VPUTensor(1, 50, 50, 1, DataType::FLOAT16)},
                                           {VPUTensor(1, 50, 50, 1, DataType::FLOAT16)});

    std::string cost_source;
    const auto cost = device_based_provider->get_cost(wl_unmapped_device, &cost_source);
    EXPECT_EQ(cost, Cycles::ERROR_SHAVE_OPERATOR_MISSING);
    EXPECT_EQ(cost_source, "no_provider");
}

TEST_F(ShaveCostProviderTests, RoutesOnlyConfiguredOperationsToOverrideProvider) {
    auto default_provider = std::make_shared<FixedCostProvider>(100, 2, std::vector<std::string>{"relu", "MVN"});
    auto override_provider = std::make_shared<FixedCostProvider>(200, 5, std::vector<std::string>{"SoftMax"});
    OpSetOverrideShaveCostProvider provider(default_provider, override_provider, {"MVN"});

    const SHAVEWorkload override_op_workload("MVN", VPUDevice::NPU_5_0, {VPUTensor(1, 8, 8, 1, DataType::FLOAT16)},
                                             {VPUTensor(1, 8, 8, 1, DataType::FLOAT16)});
    const SHAVEWorkload default_op_workload("relu", VPUDevice::NPU_5_0, {VPUTensor(1, 8, 8, 1, DataType::FLOAT16)},
                                            {VPUTensor(1, 8, 8, 1, DataType::FLOAT16)});

    EXPECT_EQ(provider.get_cost(override_op_workload), 200);
    EXPECT_EQ(provider.get_cost(default_op_workload), 100);
}

TEST_F(ShaveCostProviderTests, SupportedOpsComeFromDefaultProviderAndMaxParamsUsesMaximum) {
    auto default_provider = std::make_shared<FixedCostProvider>(100, 3, std::vector<std::string>{"A", "B"});
    auto override_provider = std::make_shared<FixedCostProvider>(200, 7, std::vector<std::string>{"C"});
    OpSetOverrideShaveCostProvider provider(default_provider, override_provider, {"B"});

    VPUDevice device = VPUDevice::NPU_5_0;
    const auto supported_ops = provider.get_shave_supported_ops(device);

    EXPECT_EQ(provider.get_max_num_params(), 7);
    EXPECT_EQ(std::set<std::string>(supported_ops.begin(), supported_ops.end()), std::set<std::string>({"A", "B"}));
}

TEST_F(ShaveCostProviderTests, NullOverrideProviderFallsBackToDefaultProvider) {
    auto default_provider = std::make_shared<FixedCostProvider>(123, 4, std::vector<std::string>{"relu", "MVN"});
    Shared_IShaveCostProvider override_provider;
    OpSetOverrideShaveCostProvider provider(default_provider, override_provider, {"MVN"});

    const SHAVEWorkload overridden_op_workload("MVN", VPUDevice::NPU_5_0, {VPUTensor(1, 8, 8, 1, DataType::FLOAT16)},
                                               {VPUTensor(1, 8, 8, 1, DataType::FLOAT16)});
    const SHAVEWorkload default_op_workload("relu", VPUDevice::NPU_5_0, {VPUTensor(1, 8, 8, 1, DataType::FLOAT16)},
                                            {VPUTensor(1, 8, 8, 1, DataType::FLOAT16)});

    EXPECT_EQ(provider.get_cost(overridden_op_workload), 123);
    EXPECT_EQ(provider.get_cost(default_op_workload), 123);
    EXPECT_EQ(provider.get_max_num_params(), 4);
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
    SHAVEWorkload workload("gridsample",  // This should exist in heuristic provider
                           VPUDevice::NPU_5_0, {VPUTensor(10, 10, 1, 1, DataType::FLOAT16)},
                           {VPUTensor(10, 10, 1, 1, DataType::FLOAT16)});

    std::string cost_source;
    auto cost = heuristic_provider->get_cost(workload, &cost_source);

    // If the operator exists, cost source should be set
    if (!Cycles::isErrorCode(cost)) {
        EXPECT_EQ(cost_source, "shave_heuristic");
    }
}

TEST_F(HeuristicCostProviderTests, GetCostWithoutSourceParameter) {
    // Test that get_cost works without cost_source parameter
    SHAVEWorkload workload("gridsample", VPUDevice::NPU_5_0, {VPUTensor(10, 10, 1, 1, DataType::FLOAT16)},
                           {VPUTensor(10, 10, 1, 1, DataType::FLOAT16)});

    // Should not crash
    ASSERT_NO_THROW({ [[maybe_unused]] auto cost = heuristic_provider->get_cost(workload); });
}

TEST_F(HeuristicCostProviderTests, GetCostForUnknownOperator) {
    // Test with unknown operator
    SHAVEWorkload workload("unknown_operator_xyz", VPUDevice::NPU_5_0, {VPUTensor(10, 10, 1, 1, DataType::FLOAT16)},
                           {VPUTensor(10, 10, 1, 1, DataType::FLOAT16)});

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
    SUCCEED();       // Test passes if we get here without crash
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
    SHAVEWorkload workload("gridsample", VPUDevice::NPU_5_0, {VPUTensor(10, 10, 1, 1, DataType::FLOAT16)},
                           {VPUTensor(10, 10, 1, 1, DataType::FLOAT16)});

    auto cost1 = heuristic_provider->get_cost(workload);
    auto cost2 = heuristic_provider->get_cost(workload);

    // Should return same cost for same workload
    EXPECT_EQ(cost1, cost2);
}

TEST_F(HeuristicCostProviderTests, GetCostScalesWithWorkloadSize) {
    // Create workloads of different sizes
    SHAVEWorkload small_workload("gridsample", VPUDevice::NPU_5_0, {VPUTensor(10, 10, 1, 1, DataType::FLOAT16)},
                                 {VPUTensor(10, 10, 1, 1, DataType::FLOAT16)});

    SHAVEWorkload large_workload("gridsample", VPUDevice::NPU_5_0, {VPUTensor(100, 100, 1, 1, DataType::FLOAT16)},
                                 {VPUTensor(100, 100, 1, 1, DataType::FLOAT16)});

    auto small_cost = heuristic_provider->get_cost(small_workload);
    auto large_cost = heuristic_provider->get_cost(large_workload);

    // If both are valid (not error codes), larger workload should cost more
    if (!Cycles::isErrorCode(small_cost) && !Cycles::isErrorCode(large_cost)) {
        EXPECT_GT(large_cost, small_cost);
    }
}

TEST_F(HeuristicCostProviderTests, LegacyShvCompatScalesHeuristicCostsOnly) {
    auto base_provider = ShaveCostProviderBundles::createHeuristicWithFactorsOnlyProvider();
    auto wrapped_provider = ShaveCostProviderBundles::createDeviceMappedProvider();

    // NPU_5_0_W is excluded: it routes to the legacy old selector (no LegacyShvCompatProvider wrapper applied)
    std::vector<VPUDevice> wrapped_devices = {VPUDevice::NPU_5_0};

    for (const auto device : wrapped_devices) {
        SHAVEWorkload heuristic_workload("GridSample", device, {VPUTensor(80, 80, 4, 1, DataType::FLOAT16)},
                                         {VPUTensor(80, 80, 4, 1, DataType::FLOAT16)});

        std::string base_source;
        std::string wrapped_source;

        const auto base_cost = base_provider->get_cost(heuristic_workload, &base_source);
        const auto wrapped_cost = wrapped_provider->get_cost(heuristic_workload, &wrapped_source);

        ASSERT_FALSE(Cycles::isErrorCode(base_cost))
                << "Base provider returned error code on device " << static_cast<int>(device) << ": "
                << Cycles::toErrorText(base_cost);
        ASSERT_FALSE(Cycles::isErrorCode(wrapped_cost))
                << "Wrapped provider returned error code on device " << static_cast<int>(device) << ": "
                << Cycles::toErrorText(wrapped_cost);

        EXPECT_EQ(base_source, wrapped_source);
        EXPECT_LT(wrapped_cost, base_cost);
    }
}

TEST_F(HeuristicCostProviderTests, DeviceMappedProviderDoesNotAlterNonHeuristicMappedPath) {
    auto composite_provider = ShaveCostProviderBundles::createDeviceMappedProvider();
    auto old_mapped_provider = ShaveCostProviderBundles::createNameMappingOldProvider();

    SHAVEWorkload mapped_old_path("SoftMax", VPUDevice::NPU_5_0, {VPUTensor(32, 1, 1, 1, DataType::FLOAT16)},
                                  {VPUTensor(32, 1, 1, 1, DataType::FLOAT16)});

    std::string composite_source;
    std::string old_source;

    const auto composite_cost = composite_provider->get_cost(mapped_old_path, &composite_source);
    const auto old_cost = old_mapped_provider->get_cost(mapped_old_path, &old_source);

    EXPECT_EQ(composite_cost, old_cost);
    EXPECT_EQ(composite_source, old_source);
}

TEST_F(HeuristicCostProviderTests, LegacyShvCompatPreservesErrors) {
    auto base_provider = ShaveCostProviderBundles::createHeuristicWithFactorsOnlyProvider();
    auto wrapped_provider = ShaveCostProviderBundles::createDeviceMappedProvider();

    SHAVEWorkload unknown_workload("definitely_unknown_op", VPUDevice::NPU_5_0,
                                   {VPUTensor(8, 8, 1, 1, DataType::FLOAT16)},
                                   {VPUTensor(8, 8, 1, 1, DataType::FLOAT16)});

    const auto base_cost = base_provider->get_cost(unknown_workload);
    const auto wrapped_cost = wrapped_provider->get_cost(unknown_workload);

    EXPECT_TRUE(Cycles::isErrorCode(base_cost));
    EXPECT_EQ(base_cost, wrapped_cost);
}

TEST_F(HeuristicCostProviderTests, LegacyShvCompatMatchesGetCyclesEquivalentPath) {
    auto base_provider = ShaveCostProviderBundles::createHeuristicWithFactorsOnlyProvider();
    auto wrapped_provider = ShaveCostProviderBundles::createDeviceMappedProvider();

    // NPU_5_0_W is excluded: it routes to the legacy old selector (no LegacyShvCompatProvider wrapper applied).
    std::vector<VPUDevice> wrapped_devices = {VPUDevice::NPU_5_0};

    for (const auto device : wrapped_devices) {
        SHAVEWorkload workload("GridSample", device, {VPUTensor(20, 20, 5, 1, DataType::FLOAT16)},
                               {VPUTensor(20, 20, 5, 1, DataType::FLOAT16)});

        const auto wrapped_cost = wrapped_provider->get_cost(workload);
        ASSERT_FALSE(Cycles::isErrorCode(wrapped_cost))
                << "Wrapped provider returned error code on device " << static_cast<int>(device) << ": "
                << Cycles::toErrorText(wrapped_cost);

        VPUDevice mutable_device = device;
        auto shave_instance = base_provider->get_shave_instance(workload.get_name(), mutable_device);
        ASSERT_TRUE(shave_instance.has_value());

        const auto expected = shave_instance.value().get().dpuCycles(workload, 1, 1);

        ASSERT_FALSE(Cycles::isErrorCode(expected))
                << "Expected value returned error code on device " << static_cast<int>(device) << ": "
                << Cycles::toErrorText(expected);
        EXPECT_NEAR(static_cast<double>(wrapped_cost), static_cast<double>(expected), 1.0);
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

class ShaveCostProviderConfigInspector final : public ShaveCostProvider {
public:
    const ShaveConfiguration* get_config_address() const {
        return &getShaveConfigurator();
    }
};

class OldShaveCostProviderConfigInspector final : public OldShaveCostProvider {
public:
    const ShaveConfiguration* get_config_address() const {
        return &getShaveConfigurator();
    }
};

TEST_F(MathematicalShaveCostProviderBaseTests, Shave2CostSourceName) {
    SHAVEWorkload workload("relu", VPUDevice::VPU_4_0, {VPUTensor(10, 10, 1, 1, DataType::FLOAT16)},
                           {VPUTensor(10, 10, 1, 1, DataType::FLOAT16)});

    std::string cost_source;
    auto cost = shave2_provider->get_cost(workload, &cost_source);

    if (!Cycles::isErrorCode(cost)) {
        EXPECT_EQ(cost_source, "shave_2");
    }
}

TEST_F(MathematicalShaveCostProviderBaseTests, Shave1CostSourceName) {
    SHAVEWorkload workload("Sigmoid", VPUDevice::VPU_4_0, {VPUTensor(10, 10, 1, 1, DataType::FLOAT16)},
                           {VPUTensor(10, 10, 1, 1, DataType::FLOAT16)});

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

TEST_F(MathematicalShaveCostProviderBaseTests, IndependentShaveConfigurationInstanceAcrossProviders) {
    ShaveCostProviderConfigInspector shave2_provider_inspector;
    OldShaveCostProviderConfigInspector shave1_provider_inspector;

    EXPECT_NE(shave2_provider_inspector.get_config_address(), shave1_provider_inspector.get_config_address());
}

}  // namespace VPUNN_unit_tests