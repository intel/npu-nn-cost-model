// Copyright © 2025 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#include <gtest/gtest.h>
#include "vpu/shave/shave_op_executors.h"
#include "vpu/shave/shave_collection.h"
#include "vpu/shave_workload.h"
#include "vpu/types.h"
#include <sstream>
#include <algorithm>

using namespace VPUNN;

// ============================================================================
// ShaveSimpleHeuristicModelActivation Tests
// ============================================================================
namespace VPUNN_unit_tests {
class ShaveSimpleHeuristicModelActivationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create activations with different parameters
        default_activation = new ShaveSimpleHeuristicModelActivation(
            "test_simple_default",
            1.0f,  // elements_per_cycle
            0.8f,  // code_derate
            0.7f,  // bw_derate
            2000.0f  // entry_cost_cycles
        );
        
        custom_activation = new ShaveSimpleHeuristicModelActivation(
            "test_simple_custom",
            2.0f,    // 2 cycles per element
            0.5f,    // 50% code derate
            0.6f,    // 40% bandwidth derate
            1000.0f  // 1000 cycles entry cost
        );
    }

    void TearDown() override {
        delete default_activation;
        delete custom_activation;
    }

    ShaveSimpleHeuristicModelActivation* default_activation;
    ShaveSimpleHeuristicModelActivation* custom_activation;
};

TEST_F(ShaveSimpleHeuristicModelActivationTest, NameIsCorrect) {
    EXPECT_EQ(default_activation->getName(), "test_simple_default");
    EXPECT_EQ(custom_activation->getName(), "test_simple_custom");
}

TEST_F(ShaveSimpleHeuristicModelActivationTest, BasicWorkload) {
    // Create a simple workload with 1000 elements (1x10x10x10)
    SHAVEWorkload workload(
        "test_simple_default",
        VPUDevice::VPU_4_0,
        {VPUTensor(10, 10, 10, 1, DataType::FLOAT16)},  // input: 1000 elements
        {VPUTensor(10, 10, 10, 1, DataType::FLOAT16)}   // output: 1000 elements
    );
    
    CyclesInterfaceType cycles = default_activation->dpuCycles(workload);
    
    // total_number_of_elements() returns inputs + outputs = 1000 + 1000 = 2000
    // Expected: 2000 + (2000 / (1.0 * 0.8 * 0.7)) = 2000 + 3571.43 = 5572
    EXPECT_NEAR(cycles, 5572, 2);
}

TEST_F(ShaveSimpleHeuristicModelActivationTest, CustomParametersWorkload) {
    // Create a workload with 500 elements per tensor
    SHAVEWorkload workload(
        "test_simple_custom",
        VPUDevice::VPU_4_0,
        {VPUTensor(5, 10, 10, 1, DataType::FLOAT16)},  // 500 elements
        {VPUTensor(5, 10, 10, 1, DataType::FLOAT16)}   // 500 elements
    );
    
    CyclesInterfaceType cycles = custom_activation->dpuCycles(workload);
    
    // total_number_of_elements() = 500 + 500 = 1000
    // Expected: 1000 + (1000 / (2.0 * 0.5 * 0.6)) = 1000 + 1666.67 = 2667
    EXPECT_NEAR(cycles, 2667, 2);
}

TEST_F(ShaveSimpleHeuristicModelActivationTest, DpuCyclesWithFrequency) {
    // Test the overloaded dpuCycles that takes frequency parameters
    SHAVEWorkload workload(
        "test_simple_default",
        VPUDevice::VPU_4_0,
        {VPUTensor(10, 10, 10, 1, DataType::FLOAT16)},
        {VPUTensor(10, 10, 10, 1, DataType::FLOAT16)}
    );
    
    // Should ignore frequency parameters and return same result
    CyclesInterfaceType cycles1 = default_activation->dpuCycles(workload);
    CyclesInterfaceType cycles2 = default_activation->dpuCycles(workload, 700, 1300);
    
    EXPECT_EQ(cycles1, cycles2);
}

TEST_F(ShaveSimpleHeuristicModelActivationTest, ZeroElementsWorkload) {
    // Workload with zero volume
    SHAVEWorkload workload(
        "test_simple_default",
        VPUDevice::VPU_4_0,
        {VPUTensor(0, 0, 0, 1, DataType::FLOAT16)},
        {VPUTensor(0, 0, 0, 1, DataType::FLOAT16)}
    );
    
    CyclesInterfaceType cycles = default_activation->dpuCycles(workload);
    
    // Should only have entry cost
    EXPECT_EQ(cycles, 2000);
}

TEST_F(ShaveSimpleHeuristicModelActivationTest, ToStringOutput) {
    std::string str = default_activation->toString();
    
    EXPECT_TRUE(str.find("ShaveSimpleHeuristicModelActivation") != std::string::npos);
    EXPECT_TRUE(str.find("test_simple_default") != std::string::npos);
    EXPECT_TRUE(str.find("ShaveSimpleHeuristicModel") != std::string::npos);
}

TEST_F(ShaveSimpleHeuristicModelActivationTest, MultipleOutputsWorkload) {
    // Test with multiple outputs (should sum their elements)
    SHAVEWorkload workload(
        "test_simple_default",
        VPUDevice::VPU_4_0,
        {VPUTensor(10, 10, 1, 1, DataType::FLOAT16)},  // 100 elements input
        {
            VPUTensor(10, 10, 1, 1, DataType::FLOAT16),  // 100 elements output
            VPUTensor(5, 5, 1, 1, DataType::FLOAT16)     // 25 elements output
        }
    );
    
    CyclesInterfaceType cycles = default_activation->dpuCycles(workload);
    
    // Total elements: input(100) + output1(100) + output2(25) = 225
    // Expected: 2000 + (225 / (1.0 * 0.8 * 0.7)) = 2000 + 401.79 = 2402
    EXPECT_NEAR(cycles, 2402, 2);
}

// ============================================================================
// ShaveRooflineHeuristicModelActivation Tests
// ============================================================================

class ShaveRooflineHeuristicModelActivationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Elementwise operation (BW-bound)
        elementwise_activation = new ShaveRooflineHeuristicModelActivation(
            "test_roofline_eltwise",
            1.0f,    // arithmetic_ops_per_32_outputs
            3.0f,    // memory_ops_per_32_outputs
            false,   // not unaligned
            0.7f,    // bw_derate
            0.8f,    // code_derate
            2000.0f, // entry_cost_cycles
            0.0f,    // scalar_cost_per_channel
            2.0f     // unalignment_derate
        );
        
        // Compute-intensive operation
        compute_activation = new ShaveRooflineHeuristicModelActivation(
            "test_roofline_compute",
            32.0f,   // arithmetic_ops_per_32_outputs
            2.0f,    // memory_ops_per_32_outputs
            false,   // not unaligned
            0.7f,    // bw_derate
            0.8f,    // code_derate
            2000.0f, // entry_cost_cycles
            0.0f,    // scalar_cost_per_channel
            2.0f     // unalignment_derate
        );
        
        // Operation with scalar cost per channel
        scalar_cost_activation = new ShaveRooflineHeuristicModelActivation(
            "test_roofline_scalar",
            5.0f,    // arithmetic_ops_per_32_outputs
            5.0f,    // memory_ops_per_32_outputs
            false,   // not unaligned
            0.7f,    // bw_derate
            0.8f,    // code_derate
            1000.0f, // entry_cost_cycles
            50.0f,   // scalar_cost_per_channel
            2.0f     // unalignment_derate
        );

        // Unaligned operation
        unaligned_activation = new ShaveRooflineHeuristicModelActivation(
            "test_roofline_unaligned",
            1.0f,    // arithmetic_ops_per_32_outputs
            2.0f,    // memory_ops_per_32_outputs
            true,    // unaligned by nature
            0.7f,    // bw_derate
            0.8f,    // code_derate
            2000.0f, // entry_cost_cycles
            0.0f,    // scalar_cost_per_channel
            2.0f     // unalignment_derate
        );
    }

    void TearDown() override {
        delete elementwise_activation;
        delete compute_activation;
        delete scalar_cost_activation;
        delete unaligned_activation;
    }

    ShaveRooflineHeuristicModelActivation* elementwise_activation;
    ShaveRooflineHeuristicModelActivation* compute_activation;
    ShaveRooflineHeuristicModelActivation* scalar_cost_activation;
    ShaveRooflineHeuristicModelActivation* unaligned_activation;
};

TEST_F(ShaveRooflineHeuristicModelActivationTest, NameIsCorrect) {
    EXPECT_EQ(elementwise_activation->getName(), "test_roofline_eltwise");
    EXPECT_EQ(compute_activation->getName(), "test_roofline_compute");
    EXPECT_EQ(scalar_cost_activation->getName(), "test_roofline_scalar");
}

TEST_F(ShaveRooflineHeuristicModelActivationTest, BasicElementwiseWorkload) {
    // Create workload with 32 fp16 elements = 512 bits per tensor
    SHAVEWorkload workload(
        "test_roofline_eltwise",
        VPUDevice::VPU_4_0,
        {VPUTensor(32, 1, 1, 1, DataType::FLOAT16)},  // 32 elements, 512 bits
        {VPUTensor(32, 1, 1, 1, DataType::FLOAT16)}   // 32 elements, 512 bits
    );
    
    CyclesInterfaceType cycles = elementwise_activation->dpuCycles(workload);
    
    // total_size_in_bits = output bits = 512
    // The model uses output bits, not input+output
    // Should be close to 2006-2011 cycles based on 512 bits (rounding differences)
    EXPECT_NEAR(cycles, 2006, 10);
}

TEST_F(ShaveRooflineHeuristicModelActivationTest, ComputeIntensiveWorkload) {
    // Create workload with 32 fp16 elements = 512 bits per tensor
    SHAVEWorkload workload(
        "test_roofline_compute",
        VPUDevice::VPU_4_0,
        {VPUTensor(32, 1, 1, 1, DataType::FLOAT16)},  // 512 bits
        {VPUTensor(32, 1, 1, 1, DataType::FLOAT16)}   // 512 bits
    );
    
    CyclesInterfaceType cycles = compute_activation->dpuCycles(workload);
    
    // Should be around 2058-2115 cycles (compute-bound) based on 512 bits output
    EXPECT_NEAR(cycles, 2058, 60);
}

TEST_F(ShaveRooflineHeuristicModelActivationTest, WorkloadWithChannels) {
    // Create workload with multiple channels
    SHAVEWorkload workload(
        "test_roofline_scalar",
        VPUDevice::VPU_4_0,
        {VPUTensor(32, 1, 4, 1, DataType::FLOAT16)},  // 4 channels
        {VPUTensor(32, 1, 4, 1, DataType::FLOAT16)}   // 4 channels
    );
    
    CyclesInterfaceType cycles = scalar_cost_activation->dpuCycles(workload);
    
    // Should include scalar cost: 50 * 4 = 200 cycles
    EXPECT_GT(cycles, 1000 + 200);  // At least entry cost + scalar cost
}

TEST_F(ShaveRooflineHeuristicModelActivationTest, MultipleOutputsWithChannels) {
    // Test with multiple outputs (channels should sum)
    SHAVEWorkload workload(
        "test_roofline_scalar",
        VPUDevice::VPU_4_0,
        {VPUTensor(32, 1, 3, 1, DataType::FLOAT16)},
        {
            VPUTensor(32, 1, 3, 1, DataType::FLOAT16),  // 3 channels
            VPUTensor(16, 1, 2, 1, DataType::FLOAT16)   // 2 channels
        }
    );
    
    CyclesInterfaceType cycles = scalar_cost_activation->dpuCycles(workload);
    
    // Total channels: 3 + 2 = 5
    // Scalar cost: 50 * 5 = 250
    EXPECT_GT(cycles, 1000 + 250);
}

TEST_F(ShaveRooflineHeuristicModelActivationTest, UnalignedWorkload) {
    // Create workload with 32 fp16 elements = 512 bits per tensor
    SHAVEWorkload workload(
        "test_roofline_unaligned",
        VPUDevice::VPU_4_0,
        {VPUTensor(32, 1, 1, 1, DataType::FLOAT16)},  // 512 bits
        {VPUTensor(32, 1, 1, 1, DataType::FLOAT16)}   // 512 bits
    );
    
    CyclesInterfaceType cycles = unaligned_activation->dpuCycles(workload);
    
    // Should be close to 2008 cycles (with unalignment derate) based on 512 bits output (allow some rounding tolerance)
    EXPECT_NEAR(cycles, 2008, 10);
}

TEST_F(ShaveRooflineHeuristicModelActivationTest, DpuCyclesWithFrequency) {
    // Test the overloaded dpuCycles that takes frequency parameters
    SHAVEWorkload workload(
        "test_roofline_eltwise",
        VPUDevice::VPU_4_0,
        {VPUTensor(32, 1, 1, 1, DataType::FLOAT16)},
        {VPUTensor(32, 1, 1, 1, DataType::FLOAT16)}
    );
    
    // Should ignore frequency parameters and return same result
    CyclesInterfaceType cycles1 = elementwise_activation->dpuCycles(workload);
    CyclesInterfaceType cycles2 = elementwise_activation->dpuCycles(workload, 700, 1300);
    
    EXPECT_EQ(cycles1, cycles2);
}

TEST_F(ShaveRooflineHeuristicModelActivationTest, ZeroElementsWorkload) {
    // Workload with zero volume
    SHAVEWorkload workload(
        "test_roofline_eltwise",
        VPUDevice::VPU_4_0,
        {VPUTensor(0, 0, 0, 1, DataType::FLOAT16)},
        {VPUTensor(0, 0, 0, 1, DataType::FLOAT16)}
    );
    
    CyclesInterfaceType cycles = elementwise_activation->dpuCycles(workload);
    
    // Should only have entry cost
    EXPECT_EQ(cycles, 2000);
}

TEST_F(ShaveRooflineHeuristicModelActivationTest, ToStringOutput) {
    std::string str = elementwise_activation->toString();
    
    EXPECT_TRUE(str.find("ShaveRooflineHeuristicModelActivation") != std::string::npos);
    EXPECT_TRUE(str.find("test_roofline_eltwise") != std::string::npos);
    EXPECT_TRUE(str.find("ShaveRooflineHeuristicModel") != std::string::npos);
}

TEST_F(ShaveRooflineHeuristicModelActivationTest, LargeWorkload) {
    // Test with larger workload
    SHAVEWorkload workload(
        "test_roofline_eltwise",
        VPUDevice::VPU_4_0,
        {VPUTensor(100, 100, 1, 1, DataType::FLOAT16)},  // 10000 elements
        {VPUTensor(100, 100, 1, 1, DataType::FLOAT16)}
    );
    
    CyclesInterfaceType cycles = elementwise_activation->dpuCycles(workload);
    
    // Should be significantly larger than entry cost
    EXPECT_GT(cycles, 2000);
    // But reasonable (not overflowing)
    EXPECT_LT(cycles, 1000000);
}

// ============================================================================
// Integration Tests for Executors
// ============================================================================

TEST(ShaveHeuristicExecutorsIntegration, MultipleExecutorsWorkTogether) {
    // Create multiple executors
    ShaveSimpleHeuristicModelActivation simple1("op1", 1.0f, 0.8f, 0.7f, 2000.0f);
    ShaveSimpleHeuristicModelActivation simple2("op2", 1.0f, 0.8f, 0.7f, 2000.0f);
    ShaveRooflineHeuristicModelActivation roofline1("op3", 1.0f, 3.0f, false, 0.7f, 0.9f, 2000.0f, 0.0f, 2.0f);
    
    // Create same workload for all
    SHAVEWorkload workload(
        "test",
        VPUDevice::VPU_4_0,
        {VPUTensor(32, 1, 1, 1, DataType::FLOAT16)},
        {VPUTensor(32, 1, 1, 1, DataType::FLOAT16)}
    );
    
    // All should return valid results
    EXPECT_GT(simple1.dpuCycles(workload), 2000);
    EXPECT_GT(simple2.dpuCycles(workload), 2000);
    EXPECT_GT(roofline1.dpuCycles(workload), 2000);
    
    // Simple models with same parameters should return same results
    EXPECT_EQ(simple1.dpuCycles(workload), simple2.dpuCycles(workload));
}

TEST(ShaveHeuristicExecutorsIntegration, DifferentWorkloadSizes) {
    ShaveSimpleHeuristicModelActivation executor("test", 1.0f, 0.8f, 0.7f, 2000.0f);
    
    // Small workload
    SHAVEWorkload small(
        "test",
        VPUDevice::VPU_4_0,
        {VPUTensor(10, 1, 1, 1, DataType::FLOAT16)},
        {VPUTensor(10, 1, 1, 1, DataType::FLOAT16)}
    );
    
    // Large workload
    SHAVEWorkload large(
        "test",
        VPUDevice::VPU_4_0,
        {VPUTensor(100, 1, 1, 1, DataType::FLOAT16)},
        {VPUTensor(100, 1, 1, 1, DataType::FLOAT16)}
    );
    
    auto small_cycles = executor.dpuCycles(small);
    auto large_cycles = executor.dpuCycles(large);
    
    // Larger workload should cost more
    EXPECT_GT(large_cycles, small_cycles);
}

TEST(ShaveHeuristicExecutorsIntegration, RooflineVsSimpleComparison) {
    ShaveSimpleHeuristicModelActivation simple_exec("simple", 1.0f, 0.8f, 0.7f, 2000.0f);
    ShaveRooflineHeuristicModelActivation roofline_exec("roofline", 1.0f, 3.0f, false, 0.7f, 0.8f, 2000.0f, 0.0f, 2.0f);
    
    // Create workload
    SHAVEWorkload workload(
        "test",
        VPUDevice::VPU_4_0,
        {VPUTensor(100, 1, 1, 1, DataType::FLOAT16)},
        {VPUTensor(100, 1, 1, 1, DataType::FLOAT16)}
    );
    
    auto simple_cycles = simple_exec.dpuCycles(workload);
    auto roofline_cycles = roofline_exec.dpuCycles(workload);
    
    // Both should return valid results
    EXPECT_GT(simple_cycles, 2000);
    EXPECT_GT(roofline_cycles, 2000);
    
    // Results may differ based on model type
    // Just verify both are reasonable
    EXPECT_LT(simple_cycles, 1000000);
    EXPECT_LT(roofline_cycles, 1000000);
}

}  // namespace VPUNN_unit_tests