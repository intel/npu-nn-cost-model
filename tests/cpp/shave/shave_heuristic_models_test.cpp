// Copyright © 2025 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#include <gtest/gtest.h>
#include "vpu/shave/shave_heuristic_models.h"
#include <cmath>

using namespace VPUNN;
namespace VPUNN_unit_tests {

class ShaveSimpleHeuristicModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Default model with standard parameters
        default_model = new ShaveSimpleHeuristicModel();
        
        // Custom model with specific parameters
        custom_model = new ShaveSimpleHeuristicModel(
            2.0f,    // 2 cycles per element
            0.5f,    // 50% code derate
            0.6f,    // 40% bandwidth derate
            1000.0f  // 1000 cycles entry cost
        );
    }

    void TearDown() override {
        delete default_model;
        delete custom_model;
    }

    ShaveSimpleHeuristicModel* default_model;
    ShaveSimpleHeuristicModel* custom_model;
};

TEST_F(ShaveSimpleHeuristicModelTest, DefaultParameters) {
    // Test with default parameters: 1 elem/cycle, 0.8 derate, 2000 entry cost
    // For 100 elements: cycles = 2000 + (100 / (1.0 * 0.8 * 0.7)) = 2000 + 178.57 = 2179
    CyclesInterfaceType cycles = default_model->getCycles(100);
    EXPECT_EQ(cycles, 2179);
}

TEST_F(ShaveSimpleHeuristicModelTest, ZeroElements) {
    // With 0 elements, only entry cost should be charged
    CyclesInterfaceType cycles = default_model->getCycles(0);
    EXPECT_EQ(cycles, 2000);
}

TEST_F(ShaveSimpleHeuristicModelTest, SmallWorkload) {
    // Test with small workload (10 elements)
    // cycles = 2000 + (10 / (1.0 * 0.8 * 0.7)) = 2000 + 17.86 = 2018
    CyclesInterfaceType cycles = default_model->getCycles(10);
    EXPECT_EQ(cycles, 2018);
}

TEST_F(ShaveSimpleHeuristicModelTest, LargeWorkload) {
    // Test with large workload (10000 elements)
    // cycles = 2000 + (10000 / (1.0 * 0.8 * 0.7)) = 2000 + 17857.14 = 19858
    CyclesInterfaceType cycles = default_model->getCycles(10000);
    EXPECT_EQ(cycles, 19858);
}

TEST_F(ShaveSimpleHeuristicModelTest, CustomParameters) {
    // Custom model: 2 cycles/elem, 0.5 derate, 1000 entry cost
    // For 100 elements: cycles = 1000 + (100 / (2.0 * 0.5 * 0.6)) = 1000 + 166.67 = 1167
    CyclesInterfaceType cycles = custom_model->getCycles(100);
    EXPECT_EQ(cycles, 1167);
}

TEST_F(ShaveSimpleHeuristicModelTest, CustomParametersLargeWorkload) {
    // For 5000 elements: cycles = 1000 + (5000 / (2.0 * 0.5 * 0.6)) = 1000 + 8333.33 = 9334
    CyclesInterfaceType cycles = custom_model->getCycles(5000);
    EXPECT_EQ(cycles, 9334);
}

TEST_F(ShaveSimpleHeuristicModelTest, FractionalCyclesRoundedUp) {
    // Test that fractional cycles are rounded up
    // For 101 elements with default: 2000 + (101 / (1.0 * 0.8 * 0.7)) = 2000 + 179.46 = 2179.46 → 2181
    CyclesInterfaceType cycles = default_model->getCycles(101);
    EXPECT_EQ(cycles, 2181);
}

class ShaveRooflineHeuristicModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Model for simple elementwise operation (e.g., Add)
        // 1 arithmetic op, 3 memory ops per 32 outputs
        elementwise_model = new ShaveRooflineHeuristicModel(
            1.0f,    // arithmetic_ops_per_32_outputs
            3.0f,    // memory_ops_per_32_outputs
            false,   // not unaligned
            0.7f,    // 30% BW derate
            0.8f,    // 20% code derate
            2000.0f  // entry cost
        );
        
        // Model for compute-intensive operation (e.g., complex math)
        // 32 arithmetic ops, 2 memory ops per 32 outputs
        compute_intensive_model = new ShaveRooflineHeuristicModel(
            32.0f,   // arithmetic_ops_per_32_outputs
            2.0f,    // memory_ops_per_32_outputs
            false,   // not unaligned
            0.7f,    // 30% BW derate
            0.8f,    // 20% code derate
            2000.0f  // entry cost
        );
        
        // Model for unaligned operation
        unaligned_model = new ShaveRooflineHeuristicModel(
            1.0f,    // arithmetic_ops_per_32_outputs
            2.0f,    // memory_ops_per_32_outputs
            true,    // unaligned by nature
            0.7f,    // 30% BW derate
            0.8f,    // 20% code derate
            2000.0f, // entry cost
            0.0f,    // no scalar cost
            2.0f     // 2x unalignment derate
        );
        
        // Model with per-channel scalar cost 
        scalar_cost_model = new ShaveRooflineHeuristicModel(
            5.0f,    // arithmetic_ops_per_32_outputs
            5.0f,    // memory_ops_per_32_outputs
            false,   // not unaligned
            0.7f,    // 30% BW derate
            0.8f,    // 20% code derate
            1000.0f, // entry cost
            50.0f    // 50 cycles per channel scalar cost
        );
    }

    void TearDown() override {
        delete elementwise_model;
        delete compute_intensive_model;
        delete unaligned_model;
        delete scalar_cost_model;
    }

    ShaveRooflineHeuristicModel* elementwise_model;
    ShaveRooflineHeuristicModel* compute_intensive_model;
    ShaveRooflineHeuristicModel* unaligned_model;
    ShaveRooflineHeuristicModel* scalar_cost_model;
};

TEST_F(ShaveRooflineHeuristicModelTest, ElementwiseOperationBWBound) {
    // Elementwise: 1 arith op, 3 mem ops per 32 outputs
    // BW throughput: 32/3 = 10.67 fp16/cycle → 170.67 bits/cycle
    // Compute throughput: 32/1 = 32 fp16/cycle → 512 bits/cycle
    // Bottleneck: BW (170.67 bits/cycle)
    // After derates: 170.67 * 0.7 * 0.8 = 95.57 bits/cycle
    // For 512 bits (32 fp16): 512/95.57 = 5.36 cycles → 6 cycles (rounded up)
    // Total: 2000 + 6 = 2006 cycles
    
    CyclesInterfaceType cycles = elementwise_model->getCycles(512, 0);
    EXPECT_NEAR(cycles, 2006, 1); // Allow 1 cycle tolerance for rounding
}

TEST_F(ShaveRooflineHeuristicModelTest, ComputeIntensiveOperationComputeBound) {
    // Compute-intensive: 32 arith ops, 2 mem ops per 32 outputs
    // BW throughput: 32/2 = 16 fp16/cycle → 256 bits/cycle
    // Compute throughput: 32/32 = 1 fp16/cycle → 16 bits/cycle
    // Bottleneck: Compute (16 bits/cycle)
    // After derates: 16 * 0.7 * 0.8 = 8.96 bits/cycle
    // For 512 bits: 512/8.96 = 57.14 cycles → 58 cycles (rounded up)
    // Total: 2000 + 58 = 2058 cycles
    
    CyclesInterfaceType cycles = compute_intensive_model->getCycles(512, 0);
    EXPECT_NEAR(cycles, 2058, 1);
}

TEST_F(ShaveRooflineHeuristicModelTest, UnalignedOperation) {
    // Unaligned: 1 arith op, 2 mem ops per 32 outputs
    // BW throughput: 32/2 = 16 fp16/cycle → 256 bits/cycle
    // Compute throughput: 32/1 = 32 fp16/cycle → 512 bits/cycle
    // Bottleneck: BW (256 bits/cycle)
    // After BW and code derates: 256 * 0.7 * 0.8 = 143.36 bits/cycle
    // After unalignment derate: 143.36 / 2.0 = 71.68 bits/cycle
    // For 512 bits: 512/71.68 = 7.14 cycles → 8 cycles (rounded up)
    // Total: 2000 + 8 = 2008 cycles
    
    CyclesInterfaceType cycles = unaligned_model->getCycles(512, 0);
    EXPECT_NEAR(cycles, 2008, 1);
}

TEST_F(ShaveRooflineHeuristicModelTest, WithScalarCostPerChannel) {
    // Scalar cost model: 5 arith ops, 5 mem ops per 32 outputs
    // BW throughput: 32/5 = 6.4 fp16/cycle → 102.4 bits/cycle
    // Compute throughput: 32/5 = 6.4 fp16/cycle → 102.4 bits/cycle
    // Bottleneck: both equal (102.4 bits/cycle)
    // After derates: 102.4 * 0.7 * 0.8 = 57.34 bits/cycle
    // For 512 bits: 512/57.34 = 8.93 cycles → 9 cycles (rounded up)
    // With 4 channels: scalar_cost = 50 * 4 = 200 cycles
    // Total: 1000 + 9 + 200 = 1209 cycles
    
    CyclesInterfaceType cycles = scalar_cost_model->getCycles(512, 512 / 16, 4);
    EXPECT_NEAR(cycles, 1209, 2);
}

TEST_F(ShaveRooflineHeuristicModelTest, ZeroChannelsNoScalarCost) {
    // Same model but with 0 channels
    // Total: 1000 + 9 + 0 = 1009 cycles
    
    CyclesInterfaceType cycles = scalar_cost_model->getCycles(512, 0);
    EXPECT_NEAR(cycles, 1009, 2);
}

TEST_F(ShaveRooflineHeuristicModelTest, ZeroBitsOnlyEntryCost) {
    // With 0 bits, only entry cost should be charged
    CyclesInterfaceType cycles = elementwise_model->getCycles(0, 0);
    EXPECT_EQ(cycles, 2000);
}

TEST_F(ShaveRooflineHeuristicModelTest, StreamOutput) {
    // Test that stream output works without crashing
    std::ostringstream oss;
    oss << *elementwise_model;
    std::string output = oss.str();
    
    EXPECT_TRUE(output.find("ShaveRooflineHeuristicModel") != std::string::npos);
    EXPECT_TRUE(output.find("arithmetic_ops_per_32_outputs") != std::string::npos);
    EXPECT_TRUE(output.find("memory_ops_per_32_outputs") != std::string::npos);
    EXPECT_TRUE(output.find("bw_derate") != std::string::npos);
    EXPECT_TRUE(output.find("code_derate") != std::string::npos);
}

class ShaveHeuristicModelsIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Models based on CSV data
        
        // GridSample - non-vectorizable
        gridsample_model = new ShaveSimpleHeuristicModel(1.0f, 0.8f, 0.7f, 2000.0f);
        
        // Interpolate - non-vectorizable
        interpolate_model = new ShaveSimpleHeuristicModel(1.0f, 0.8f, 0.7f, 2000.0f);
        
        // Add - vectorizable, elementwise
        add_model = new ShaveRooflineHeuristicModel(1.0f, 3.0f, false, 0.7f, 0.9f, 2000.0f);
        
        // Multiply - vectorizable, elementwise
        multiply_model = new ShaveRooflineHeuristicModel(1.0f, 3.0f, false, 0.7f, 0.9f, 2000.0f);
        
        // Sqrt - compute intensive
        sqrt_model = new ShaveRooflineHeuristicModel(32.0f, 2.0f, false, 0.7f, 0.8f, 2000.0f);
        
        // SoftMax - with per-channel cost
        softmax_model = new ShaveRooflineHeuristicModel(5.0f, 5.0f, false, 0.7f, 0.8f, 1000.0f, 50.0f);
        
        // MVN - with per-channel cost
        mvn_model = new ShaveRooflineHeuristicModel(5.0f, 3.0f, false, 0.7f, 0.8f, 2000.0f, 100.0f);
    }

    void TearDown() override {
        delete gridsample_model;
        delete interpolate_model;
        delete add_model;
        delete multiply_model;
        delete sqrt_model;
        delete softmax_model;
        delete mvn_model;
    }

    ShaveSimpleHeuristicModel* gridsample_model;
    ShaveSimpleHeuristicModel* interpolate_model;
    ShaveRooflineHeuristicModel* add_model;
    ShaveRooflineHeuristicModel* multiply_model;
    ShaveRooflineHeuristicModel* sqrt_model;
    ShaveRooflineHeuristicModel* softmax_model;
    ShaveRooflineHeuristicModel* mvn_model;
};

TEST_F(ShaveHeuristicModelsIntegrationTest, GridSampleTypicalWorkload) {
    // GridSample with 1024 elements
    // Non-vectorizable: 1 elem/cycle with 0.8 derate
    // cycles = 2000 + (1024 / (1.0 * 0.8 * 0.7)) = 2000 + 1828.57 = 3829 (rounded up)
    CyclesInterfaceType cycles = gridsample_model->getCycles(1024);
    EXPECT_NEAR(cycles, 3829, 1);
}

TEST_F(ShaveHeuristicModelsIntegrationTest, InterpolateTypicalWorkload) {
    // Interpolate with 2048 elements
    // Non-vectorizable: 1 elem/cycle with 0.8 derate
    // cycles = 2000 + (2048 / (1.0 * 0.8 * 0.7)) = 2000 + 3657.14 = 5658 (rounded up)
    CyclesInterfaceType cycles = interpolate_model->getCycles(2048);
    EXPECT_NEAR(cycles, 5658, 1);
}

TEST_F(ShaveHeuristicModelsIntegrationTest, AddVsMultiply) {
    // Add and Multiply should have similar performance (both elementwise)
    int size_bits = 8192; // 512 fp16 elements
    CyclesInterfaceType add_cycles = add_model->getCycles(size_bits, 0);
    CyclesInterfaceType mul_cycles = multiply_model->getCycles(size_bits, 0);
    
    // Should be very close (both use same parameters)
    EXPECT_EQ(add_cycles, mul_cycles);
}

TEST_F(ShaveHeuristicModelsIntegrationTest, SqrtSlowerThanElementwise) {
    // Sqrt (compute intensive) should be slower than Add (memory bound)
    int size_bits = 8192; // 512 fp16 elements
    CyclesInterfaceType add_cycles = add_model->getCycles(size_bits, 0);
    CyclesInterfaceType sqrt_cycles = sqrt_model->getCycles(size_bits, 0);
    
    // Sqrt should take more cycles due to 32 arithmetic ops per 32 outputs
    EXPECT_GT(sqrt_cycles, add_cycles);
}

TEST_F(ShaveHeuristicModelsIntegrationTest, SoftMaxScalesWithChannels) {
    // SoftMax with increasing channels should increase cost linearly
    int size_bits = 4096;
    int size_elements = size_bits / 16; // fp16 elements
    CyclesInterfaceType cycles_1ch = softmax_model->getCycles(size_bits, size_elements, 1);
    CyclesInterfaceType cycles_4ch = softmax_model->getCycles(size_bits, size_elements, 4);
    CyclesInterfaceType cycles_8ch = softmax_model->getCycles(size_bits, size_elements, 8);
    
    // Difference should be scalar_cost * num_channels
    EXPECT_NEAR(cycles_4ch - cycles_1ch, 50.0f * 3, 1); // 3 additional channels
    EXPECT_NEAR(cycles_8ch - cycles_1ch, 50.0f * 7, 1); // 7 additional channels
}

TEST_F(ShaveHeuristicModelsIntegrationTest, MVNWithMultipleChannels) {
    // MVN (Mean-Variance Normalization) with per-channel cost
    int size_bits = 16384; // 1024 fp16 elements
    int num_channels = 16;
    
    CyclesInterfaceType cycles = mvn_model->getCycles(size_bits, size_bits / 16, num_channels);
    
    // Should include entry cost + computation + (100 * 16) scalar cost
    // Entry: 2000
    // Scalar: 100 * 16 = 1600
    // Minimum total: 2000 + 1600 = 3600
    EXPECT_GT(cycles, 3600);
}

TEST_F(ShaveHeuristicModelsIntegrationTest, NonVectorizableVsVectorizable) {
    // Compare non-vectorizable (GridSample) vs vectorizable (Add)
    // For same number of elements
    int total_elements = 1024;
    int size_bits = total_elements * 16; // fp16
    
    CyclesInterfaceType gridsample_cycles = gridsample_model->getCycles(total_elements);
    CyclesInterfaceType add_cycles = add_model->getCycles(size_bits, 0);
    
    // Vectorizable operations should generally be faster for large workloads
    // (though this depends on specific parameters)
    // At minimum, both should have valid results
    EXPECT_GT(gridsample_cycles, 2000); // More than entry cost
    EXPECT_GT(add_cycles, 2000);
}

TEST_F(ShaveHeuristicModelsIntegrationTest, ScalingWithWorkloadSize) {
    // Test that both models scale appropriately with workload size
    int small_elements = 100;
    int large_elements = 10000;
    
    CyclesInterfaceType small_cycles = gridsample_model->getCycles(small_elements);
    CyclesInterfaceType large_cycles = gridsample_model->getCycles(large_elements);
    
    // Ratio should be approximately (large_elements / small_elements)
    float ratio = static_cast<float>(large_cycles - 2000) / static_cast<float>(small_cycles - 2000);
    EXPECT_NEAR(ratio, static_cast<float>(large_elements) / small_elements, 1.0f);
}

TEST(ShaveHeuristicModelsEdgeCases, SimpleModelVeryLargeWorkload) {
    ShaveSimpleHeuristicModel model;
    
    // Very large workload (100M elements)
    // Should not overflow or crash
    CyclesInterfaceType cycles = model.getCycles(100000000);
    EXPECT_GT(cycles, 2000);
    EXPECT_LT(cycles, std::numeric_limits<CyclesInterfaceType>::max());
}

TEST(ShaveHeuristicModelsEdgeCases, RooflineModelVeryLargeWorkload) {
    ShaveRooflineHeuristicModel model(1.0f, 2.0f);
    
    // Very large workload (1GB in bits = 8 billion bits)
    // Should not overflow or crash
    int size_elements = 8000000000 / 16;
    CyclesInterfaceType cycles = model.getCycles(8000000000, size_elements, 0);
    EXPECT_GT(cycles, 2000);
    EXPECT_LT(cycles, std::numeric_limits<CyclesInterfaceType>::max());
}

TEST(ShaveHeuristicModelsEdgeCases, RooflineZeroOpsPerOutput) {
    // Edge case: 0 ops per 32 outputs (should fall back to simple heuristic)
    ShaveRooflineHeuristicModel model(0.0f, 0.0f);
    
    int size_bits = 512; // 32 fp16 elements
    int size_elements = size_bits / 16;
    CyclesInterfaceType cycles = model.getCycles(size_bits, size_elements, 0);
    
    // Should use simple heuristic: entry_cost + (elements / (code_derate * bw_derate))
    // elements = 512 / 16 = 32
    // computation = 32 / (0.8 * 0.7) = 32 / 0.56 = 57.14 ≈ 58
    // total = 2000 + 58 = 2058
    EXPECT_NEAR(cycles, 2058, 2);
}

TEST(ShaveHeuristicModelsEdgeCases, RooflineOnlyMemoryOps) {
    // Only memory ops specified, arithmetic should be set to same value
    ShaveRooflineHeuristicModel model(0.0f, 2.0f);
    
    int size_bits = 512; // 32 fp16 elements
    int size_elements = size_bits / 16;
    CyclesInterfaceType cycles = model.getCycles(size_bits, size_elements, 0);
    
    // Both ops should be 2.0, so throughput = 32/2 = 16 elem/cycle = 256 bits/cycle
    // With derates: 256 * 0.8 * 0.7 = 143.36 bits/cycle
    // Cycles = 2000 + (512 / 143.36) = 2000 + 3.57 ≈ 2004
    EXPECT_NEAR(cycles, 2004, 2);
}

TEST(ShaveHeuristicModelsEdgeCases, RooflineOnlyArithmeticOps) {
    // Only arithmetic ops specified, memory should be set to same value
    ShaveRooflineHeuristicModel model(4.0f, 0.0f);
    
    int size_bits = 512; // 32 fp16 elements
    int size_elements = size_bits / 16;
    CyclesInterfaceType cycles = model.getCycles(size_bits, size_elements, 0);
    
    // Both ops should be 4.0, so throughput = 32/4 = 8 elem/cycle = 128 bits/cycle
    // With derates: 128 * 0.8 * 0.7 = 71.68 bits/cycle
    // Cycles = 2000 + (512 / 71.68) = 2000 + 7.14 ≈ 2008
    EXPECT_NEAR(cycles, 2008, 2);
}

TEST(ShaveHeuristicModelsEdgeCases, RooflineBothOpsNegative) {
    // Negative ops should fall back to simple heuristic
    ShaveRooflineHeuristicModel model(-1.0f, -5.0f);
    
    int size_bits = 512; // 32 fp16 elements
    int size_elements = size_bits / 16;
    CyclesInterfaceType cycles = model.getCycles(size_bits, size_elements, 0);
    
    // Should use simple heuristic like zero case
    EXPECT_NEAR(cycles, 2058, 2);
}

TEST(ShaveHeuristicModelsEdgeCases, SimpleModelZeroDerate) {
    // Edge case: zero derate (infinite efficiency - unrealistic but tests edge)
    ShaveSimpleHeuristicModel model(1.0f, 0.0f, 2000.0f);
    
    // With 0 derate, computation cycles should be 0
    CyclesInterfaceType cycles = model.getCycles(1000);
    EXPECT_NEAR(cycles, 2000, 1); // Only entry cost
}

TEST(ShaveHeuristicModelsEdgeCases, RooflineModelZeroEntryCost) {
    // Edge case: zero entry cost
    ShaveRooflineHeuristicModel model(1.0f, 2.0f, false, 0.7f, 0.8f, 0.0f);
    
    CyclesInterfaceType cycles = model.getCycles(512, 0);
    EXPECT_GT(cycles, 0); // Should still have computation cost
}

TEST(ShaveHeuristicModelsEdgeCases, RooflineModelMaximumChannels) {
    // Test with very large number of channels
    ShaveRooflineHeuristicModel model(5.0f, 5.0f, false, 0.7f, 0.8f, 1000.0f, 50.0f);
    
    CyclesInterfaceType cycles = model.getCycles(512, 512 / 16, 10000);
    
    // Should include: 1000 + computation + (50 * 10000) = 1000 + computation + 500000
    EXPECT_GT(cycles, 500000);
}

// ============================================================================
// Performance Comparison Tests
// ============================================================================

TEST(ShaveHeuristicModelsComparison, SimpleVsRooflineForSmallWorkload) {
    // For very small workloads, entry cost dominates
    ShaveSimpleHeuristicModel simple_model;
    ShaveRooflineHeuristicModel roofline_model(1.0f, 2.0f);
    
    int small_elements = 10;
    int small_bits = small_elements * 16;
    
    CyclesInterfaceType simple_cycles = simple_model.getCycles(small_elements);
    CyclesInterfaceType roofline_cycles = roofline_model.getCycles(small_bits, 0);
    
    // Both should be close to entry cost for very small workloads
    EXPECT_NEAR(simple_cycles, 2000, 100);
    EXPECT_NEAR(roofline_cycles, 2000, 100);
}

TEST(ShaveHeuristicModelsComparison, ComputeVsBandwidthBound) {
    // Create two models: one BW-bound, one compute-bound
    ShaveRooflineHeuristicModel bw_bound(1.0f, 10.0f);  // More memory ops
    ShaveRooflineHeuristicModel compute_bound(100.0f, 2.0f);  // More arithmetic ops
    
    int size_bits = 8192;
    
    CyclesInterfaceType bw_cycles = bw_bound.getCycles(size_bits, 0);
    CyclesInterfaceType compute_cycles = compute_bound.getCycles(size_bits, 0);
    
    // Compute-bound should take more cycles
    EXPECT_GT(compute_cycles, bw_cycles);
}

} // namespace VPUNN_unit_tests