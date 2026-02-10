// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include "vpu/dma_cost_providers/priority_dma_cost_provider.h"
#include "vpu/dma_cost_providers/dma_theoretical_cost_provider.h"

#include <gtest/gtest.h>
#include "common/common_helpers.h"
#include "vpu/dma_cost_providers/dma_cost_provider_interface.h"
#include "vpu/dma_types.h"
#include "vpu/dma_cost_providers/dmann_cost_provider.h"
#include "vpu/dma_cost_providers/dmann_adapter.h"

#include <fstream>
#include <iostream>

namespace VPUNN_unit_tests {
using namespace VPUNN;

class DMACostProviderTests : public ::testing::Test {
protected:
    const DMANNWorkload_NPU40 wl_40{
            VPUNN::VPUDevice::VPU_4_0,  // VPUDevice device;  ///< NPU device

            8192,  // int src_width;
            8192,  // int dst_width;

            0,  // int num_dim;
            {{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}},
            Num_DMA_Engine::Num_Engine_1,
            MemoryDirection::CMX2CMX  // MemoryDirection transfer_direction;
    };

    const DMANNWorkload_NPU50 wl_50{
            VPUNN::VPUDevice::NPU_5_0,  // VPUDevice device;  ///< NPU device

            8192,  // int src_width;
            8192,  // int dst_width;

            0,  // int num_dim;
            {{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}},
            Num_DMA_Engine::Num_Engine_1,
            MemoryDirection::CMX2CMX  // MemoryDirection transfer_direction;
    };

    DMAWorkload dma_wl {
            VPUNN::VPUDevice::NPU_5_0,  // device
            {VPUTensor(4, 1, 64, 4 /*INT32*/, DataType::INT8, Layout::ZXY)},  // input dimensions
            {VPUTensor(4, 1, 64, 4 /*INT32*/, DataType::INT8, Layout::ZXY)},  // output dimensions
            MemoryLocation::DRAM,                                             // src
            MemoryLocation::CMX,                                              // dst
            2,                                                                // owt
        };

    const DMANNWorkload_NPU40_50 wl_40_50{
        VPUNN::VPUDevice::VPU_4_0,  // VPUDevice device;  ///< NPU device

        8192,  // int src_width;
        8192,  // int dst_width;

        0,  // int num_dim;
        {{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}},
        Num_DMA_Engine::Num_Engine_1,
        MemoryDirection::CMX2CMX  // MemoryDirection transfer_direction;
    };

    /**
     * @brief Helper function to compare costs from PriorityDMACostProvider and a reference IDMACostProvider
     * @tparam PrioWlT Workload type for PriorityDMACostProvider
     * @tparam RefWlT Workload type for reference IDMACostProvider
     * @tparam TestWl Workload type for the test workload
     * @param prio_provider The PriorityDMACostProvider to test
     * @param ref_provider The reference IDMACostProvider to compare against
     * @param wl The workload to test
     * @param error_message Optional error message for assertions
     * @param cost_source Optional pointer to a string to store the source of the cost
     */
    template <typename PrioWlT, typename RefWlT, typename TestWl>
    void expectSameCost(PriorityDMACostProvider<PrioWlT>& prio_provider, 
                    const IDMACostProvider<RefWlT>& ref_provider, 
                    const TestWl& wl, 
                    const std::string& error_message = "", 
                    std::string* cost_source = nullptr) {
        // Convert workload to PrioWlT type if needed for priority provider
        PrioWlT prio_wl = DMAWorkloadTransformer::create_workload<PrioWlT, TestWl>(wl);
        auto cost_from_priority = prio_provider.get_cost(prio_wl, cost_source);
        
        // Convert workload to RefWlT type if needed for reference provider
        // because we use the raw IDMACostProviders we need to do the conversion 
        // that PriorityDMACostProvider does inside when it calls another cost provider
        RefWlT ref_wl = DMAWorkloadTransformer::create_workload<RefWlT, TestWl>(wl);
        auto cost_from_ref = ref_provider.get_cost(ref_wl, cost_source);

        EXPECT_FALSE(Cycles::isErrorCode(cost_from_priority)) << "Priority provider failed for workload";
        EXPECT_FALSE(Cycles::isErrorCode(cost_from_ref)) << "Reference provider failed for workload";
        EXPECT_EQ(cost_from_priority, cost_from_ref) << error_message;
    }
};

TEST_F(DMACostProviderTests, PriorityProviders_DMANNWorkload_NPU40) {
    // for DMANNWorkload_NPU40
    // Create shared_ptr providers for adapter and reference
    auto nn_provider_ref_npu40 = std::make_shared<DMANNCostProvider<DMANNWorkload_NPU40>>(VPU_DMA_4_0_MODEL_PATH);
    auto theoretical_provider_ref_npu40 = std::make_shared<DMATheoreticalCostProvider>();

    // For the PriorityDMACostProvider the Workload can be specified of the latest version
    // because it will convert anyway to the one of the providers inside
    // So it is easier to have one workload that decays (can convert) to any other 
    PriorityDMACostProvider<DMANNWorkload_NPU40> dmann_prio_first_npu40{
        DMACostProviderList<DMANNWorkload_NPU40> {
            std::make_shared<DMANNAdapter<DMANNWorkload_NPU40>>(
                std::static_pointer_cast<const IDMACostProvider<DMANNWorkload_NPU40>>(
                    std::make_shared<DMANNCostProvider<DMANNWorkload_NPU40>>(VPU_DMA_4_0_MODEL_PATH))),
            std::make_shared<DMANNAdapter<DMANNWorkload_NPU40>>(
                std::static_pointer_cast<const IDMACostProvider<DMAWorkload>>(
                    std::make_shared<DMATheoreticalCostProvider>()))
        }
    };

    PriorityDMACostProvider<DMANNWorkload_NPU40> theoretical_prio_first_npu40{
        DMACostProviderList<DMANNWorkload_NPU40> {
            std::make_shared<DMANNAdapter<DMANNWorkload_NPU40>>(
                std::static_pointer_cast<const IDMACostProvider<DMAWorkload>>(
                    std::make_shared<DMATheoreticalCostProvider>())),
            std::make_shared<DMANNAdapter<DMANNWorkload_NPU40>>(
                std::static_pointer_cast<const IDMACostProvider<DMANNWorkload_NPU40>>(
                    std::make_shared<DMANNCostProvider<DMANNWorkload_NPU40>>(VPU_DMA_4_0_MODEL_PATH)))
        }
    };
    
    // check
    expectSameCost(dmann_prio_first_npu40, *nn_provider_ref_npu40, wl_40, "DMANN provider inside priority provider should have had same value as DMANN direct call");
    expectSameCost(theoretical_prio_first_npu40, *theoretical_provider_ref_npu40, wl_40, "Theoretical provider should have had same value as Theoretical direct call");
}

TEST_F(DMACostProviderTests, PriorityProviders_DMANNWorkload_NPU50) {
    // for DMANNWorkload_NPU50
    auto nn_provider_ref_npu50 = std::make_shared<DMANNCostProvider<DMANNWorkload_NPU50>>(NPU_DMA_5_0_MODEL_PATH);
    auto theoretical_provider_ref_npu50 = std::make_shared<DMATheoreticalCostProvider>();

    PriorityDMACostProvider<DMANNWorkload_NPU50> dmann_prio_first_npu50{
        DMACostProviderList<DMANNWorkload_NPU50> {
            std::make_shared<DMANNAdapter<DMANNWorkload_NPU50>>(
                std::static_pointer_cast<const IDMACostProvider<DMANNWorkload_NPU50>>(
                    std::make_shared<DMANNCostProvider<DMANNWorkload_NPU50>>(NPU_DMA_5_0_MODEL_PATH))),
            std::make_shared<DMANNAdapter<DMANNWorkload_NPU50>>(
                std::static_pointer_cast<const IDMACostProvider<DMAWorkload>>(
                    std::make_shared<DMATheoreticalCostProvider>()))
        }
    };

    PriorityDMACostProvider<DMANNWorkload_NPU50> theoretical_prio_first_npu50{
        DMACostProviderList<DMANNWorkload_NPU50> {
            std::make_shared<DMANNAdapter<DMANNWorkload_NPU50>>(
                std::static_pointer_cast<const IDMACostProvider<DMAWorkload>>(
                    std::make_shared<DMATheoreticalCostProvider>())),
            std::make_shared<DMANNAdapter<DMANNWorkload_NPU50>>(
                std::static_pointer_cast<const IDMACostProvider<DMANNWorkload_NPU50>>(
                    std::make_shared<DMANNCostProvider<DMANNWorkload_NPU50>>(NPU_DMA_5_0_MODEL_PATH)))
        }
    };

    // check
    expectSameCost(dmann_prio_first_npu50, *nn_provider_ref_npu50, wl_50, "DMANN provider inside priority provider should have had same value as DMANN direct call");
    expectSameCost(theoretical_prio_first_npu50, *theoretical_provider_ref_npu50, wl_50, "Theoretical provider should have had same value as Theoretical direct call");
}

TEST_F(DMACostProviderTests, PriorityProviders_DMANNWorkload_NPU40_50) {
    // for DMANNWorkload_NPU40_50
    auto nn_provider_ref_npu40_50 = std::make_shared<DMANNCostProvider<DMANNWorkload_NPU40_50>>(NPU_DMA_5_0_MODEL_PATH);
    auto theoretical_provider_ref_npu40_50 = std::make_shared<DMATheoreticalCostProvider>();

    PriorityDMACostProvider<DMANNWorkload_NPU50> dmann_prio_first_npu40_50{
        DMACostProviderList<DMANNWorkload_NPU50> {
            std::make_shared<DMANNAdapter<DMANNWorkload_NPU50>>(
                std::static_pointer_cast<const IDMACostProvider<DMANNWorkload_NPU40_50>>(
                    std::make_shared<DMANNCostProvider<DMANNWorkload_NPU40_50>>(NPU_DMA_5_0_MODEL_PATH))),
            std::make_shared<DMANNAdapter<DMANNWorkload_NPU50>>(
                std::static_pointer_cast<const IDMACostProvider<DMAWorkload>>(
                    std::make_shared<DMATheoreticalCostProvider>()))
        }
    };

    PriorityDMACostProvider<DMANNWorkload_NPU50> theoretical_prio_first_npu40_50{
        DMACostProviderList<DMANNWorkload_NPU50> {
            std::make_shared<DMANNAdapter<DMANNWorkload_NPU50>>(
                std::static_pointer_cast<const IDMACostProvider<DMAWorkload>>(
                    std::make_shared<DMATheoreticalCostProvider>())),
            std::make_shared<DMANNAdapter<DMANNWorkload_NPU50>>(
                std::static_pointer_cast<const IDMACostProvider<DMANNWorkload_NPU40_50>>(
                    std::make_shared<DMANNCostProvider<DMANNWorkload_NPU40_50>>(NPU_DMA_5_0_MODEL_PATH)))
        }
    };

    // check
    expectSameCost(dmann_prio_first_npu40_50, *nn_provider_ref_npu40_50, wl_40_50, "DMANN provider inside priority provider should have had same value as DMANN direct call");
    expectSameCost(theoretical_prio_first_npu40_50, *theoretical_provider_ref_npu40_50, wl_40_50, "Theoretical provider should have had same value as Theoretical direct call");
}


// TODO: This test should not exist theoretically
TEST_F(DMACostProviderTests, PriorityProviders_DMANNWorkload) {
    // for DMAWorkload
    auto nn_provider_ref = std::make_shared<DMANNCostProvider<DMANNWorkload_NPU50>>(NPU_DMA_5_0_MODEL_PATH);
    auto theoretical_provider_ref = std::make_shared<DMATheoreticalCostProvider>();

    PriorityDMACostProvider<DMANNWorkload_NPU50> dmann_prio_first_wl{
        DMACostProviderList<DMANNWorkload_NPU50> {
            std::make_shared<DMANNAdapter<DMANNWorkload_NPU50>>(
                std::static_pointer_cast<const IDMACostProvider<DMANNWorkload_NPU50>>(
                    std::make_shared<DMANNCostProvider<DMANNWorkload_NPU50>>(NPU_DMA_5_0_MODEL_PATH))),
            std::make_shared<DMANNAdapter<DMANNWorkload_NPU50>>(
                std::static_pointer_cast<const IDMACostProvider<DMAWorkload>>(
                    std::make_shared<DMATheoreticalCostProvider>()))
        }
    };

    PriorityDMACostProvider<DMANNWorkload_NPU50> theoretical_prio_first_wl{
        DMACostProviderList<DMANNWorkload_NPU50> {
            std::make_shared<DMANNAdapter<DMANNWorkload_NPU50>>(
                std::static_pointer_cast<const IDMACostProvider<DMAWorkload>>(
                    std::make_shared<DMATheoreticalCostProvider>())),
            std::make_shared<DMANNAdapter<DMANNWorkload_NPU50>>(
                std::static_pointer_cast<const IDMACostProvider<DMANNWorkload_NPU50>>(
                    std::make_shared<DMANNCostProvider<DMANNWorkload_NPU50>>(NPU_DMA_5_0_MODEL_PATH)))
        }
    };

    // check
    expectSameCost(dmann_prio_first_wl, *nn_provider_ref, dma_wl, "DMANN provider inside priority provider should have had same value as DMANN direct call");
    expectSameCost(theoretical_prio_first_wl, *theoretical_provider_ref, dma_wl, "Theoretical provider should have had same value as Theoretical direct call");
}

TEST_F(DMACostProviderTests, PriorityProvidersTestMultipleProvidersType_WithFailures) {
    // Create a mock provider that always fails
    class FailingProvider : public IDMACostProvider<DMANNWorkload_NPU50> {
    public:
        CyclesInterfaceType get_cost(const DMANNWorkload_NPU50&, std::string* cost_source = nullptr) const override {
            if (cost_source) {
                *cost_source = "FailingProvider";
            }
            return Cycles::ERROR_INPUT_TOO_BIG;  // Return an error code
        }
    };
    
    // Create a mock provider that always succeeds with a known value
    class SuccessProvider : public IDMACostProvider<DMAWorkload> {
    public:
        CyclesInterfaceType get_cost(const DMAWorkload&, std::string* cost_source = nullptr) const override {
            if (cost_source) {
                *cost_source = "SuccessProvider";
            }
            return 12345;  // Return a known value
        }
    };

    FailingProvider failing_provider_1;
    FailingProvider failing_provider_2;
    SuccessProvider success_provider;
    
    // Create a priority provider where first two providers fail, third succeeds
    // This tests that:
    // 1. The priority mechanism tries providers in order
    // 2. DMANNAdapter correctly converts NPU50 -> DMAWorkload for the success provider
    // 3. The conversion system works end-to-end through the priority chain
    PriorityDMACostProvider<DMANNWorkload_NPU50> priority_with_failures{
        DMACostProviderList<DMANNWorkload_NPU50> {
            std::make_shared<DMANNAdapter<DMANNWorkload_NPU50>>(
                std::static_pointer_cast<const IDMACostProvider<DMANNWorkload_NPU50>>(
                    std::make_shared<FailingProvider>(failing_provider_1))),  // Will fail
            std::make_shared<DMANNAdapter<DMANNWorkload_NPU50>>(
                std::static_pointer_cast<const IDMACostProvider<DMANNWorkload_NPU50>>(
                    std::make_shared<FailingProvider>(failing_provider_2))),  // Will fail
            std::make_shared<DMANNAdapter<DMANNWorkload_NPU50>>(
                std::static_pointer_cast<const IDMACostProvider<DMAWorkload>>(
                    std::make_shared<SuccessProvider>(success_provider)))     // Will succeed after NPU50->DMAWorkload conversion
        }
    };

    std::string cost_source;
    // Test with a DMANNWorkload_NPU50 workload
    auto cost = priority_with_failures.get_cost(wl_50, &cost_source);
    
    // Should get the value from the third provider (success_provider)
    EXPECT_FALSE(Cycles::isErrorCode(cost)) << "Priority provider should have succeeded on third provider";
    EXPECT_EQ(cost, 12345u) << "Should get the known value from success_provider";
    
    // Test with NPU40 workload (alias of NPU50) - should work identically
    auto cost_40 = priority_with_failures.get_cost(wl_40, &cost_source);
    EXPECT_EQ(cost_40, 12345u) << "Should work with NPU40 workload (alias of NPU50)";
    
    // Test with NPU40_50 workload - should work identically  
    auto cost_40_50 = priority_with_failures.get_cost(wl_40_50, &cost_source);
    EXPECT_EQ(cost_40_50, 12345u) << "Should work with NPU40_50 workload";
    
    // Create NPU27 workload for testing cross-generation conversion
    DMANNWorkload_NPU27 wl_27{
        VPUNN::VPUDevice::VPU_2_7,  // device
        0,      // num_planes
        8192,   // length
        8192,   // src_width
        8192,   // dst_width
        0,      // src_stride
        0,      // dst_stride
        0,      // src_plane_stride
        0,      // dst_plane_stride
        MemoryDirection::CMX2CMX
    };
    
    // Create a priority provider that accepts NPU27, tests NPU27 -> NPU50 -> DMAWorkload conversion chain
    PriorityDMACostProvider<DMANNWorkload_NPU27> priority_npu27{
        DMACostProviderList<DMANNWorkload_NPU27> {
            std::make_shared<DMANNAdapter<DMANNWorkload_NPU27>>(
                std::static_pointer_cast<const IDMACostProvider<DMAWorkload>>(
                    std::make_shared<SuccessProvider>(success_provider)))  // NPU27 -> DMAWorkload conversion
        }
    };
    
    auto cost_27 = priority_npu27.get_cost(wl_27, &cost_source);
    EXPECT_EQ(cost_27, 12345u) << "Should work with NPU27 workload after conversion to DMAWorkload";
}

}  // namespace VPUNN_unit_tests