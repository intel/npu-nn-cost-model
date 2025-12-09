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
#include "vpu/shave_workload.h"

#include <fstream>
#include <iostream>

namespace VPUNN_unit_tests {
using namespace VPUNN;

class ShaveCostProviderTests : public ::testing::Test {
protected:
    void SetUp() override {
    }
    PriorityShaveCostProvider default_prio_provider{ShaveCostProviderBundles::createDefaultShaveCostProviders()};
    PriorityShaveCostProvider old_prio_provider{ShaveCostProviderBundles::createOldShaveOnlyProviders()};
    PriorityShaveCostProvider new_prio_provider{ShaveCostProviderBundles::createNewShaveOnlyProviders()};

    ShaveCostProvider shave2_provider{};
    OldShaveCostProvider shave1_provider{};

    void expectSameCost(const std::vector<std::reference_wrapper<IShaveCostProvider>>& providers, 
                       const SHAVEWorkload& workload) {
        if (providers.empty()) return;
        
        auto expected_cost = providers[0].get().get_cost(workload);
        for (size_t i = 1; i < providers.size(); i++) {
            auto actual_cost = providers[i].get().get_cost(workload);
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
    std::vector<std::reference_wrapper<IShaveCostProvider>> providers = {default_prio_provider, new_prio_provider, shave2_provider};
    expectSameCost(providers, new_swwl);

    std::vector<std::reference_wrapper<IShaveCostProvider>> providers_no_default = {old_prio_provider, shave1_provider};
    expectSameCost(providers_no_default, new_swwl);

    auto default_cost = default_prio_provider.get_cost(new_swwl);
    auto old_cost = old_prio_provider.get_cost(new_swwl);

    EXPECT_FALSE(Cycles::isErrorCode(default_cost));
    EXPECT_TRUE(Cycles::isErrorCode(old_cost));
}

TEST_F(ShaveCostProviderTests, ProvidersTestOldOp) {
    std::vector<std::reference_wrapper<IShaveCostProvider>> providers = {default_prio_provider, old_prio_provider, shave1_provider};
    expectSameCost(providers, old_swwl);

    std::vector<std::reference_wrapper<IShaveCostProvider>> providers_no_default = {new_prio_provider, shave2_provider};
    expectSameCost(providers_no_default, old_swwl);

    auto default_cost = default_prio_provider.get_cost(old_swwl);
    auto new_cost = new_prio_provider.get_cost(old_swwl);

    EXPECT_FALSE(Cycles::isErrorCode(default_cost));
    EXPECT_TRUE(Cycles::isErrorCode(new_cost));
}

TEST_F(ShaveCostProviderTests, ProvidersTestUnknownOp) {
    std::vector<std::reference_wrapper<IShaveCostProvider>> providers = {default_prio_provider, old_prio_provider, shave1_provider, new_prio_provider, shave2_provider};
    expectSameCost(providers, unknown_swwl);

    auto default_cost = default_prio_provider.get_cost(unknown_swwl);
    auto old_cost = old_prio_provider.get_cost(unknown_swwl);
    auto new_cost = new_prio_provider.get_cost(unknown_swwl);

    EXPECT_TRUE(Cycles::isErrorCode(default_cost));
    EXPECT_TRUE(Cycles::isErrorCode(old_cost));
    EXPECT_TRUE(Cycles::isErrorCode(new_cost));
}


}  // namespace VPUNN_unit_tests