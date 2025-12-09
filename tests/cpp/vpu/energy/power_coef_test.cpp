// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the ?Software Package?)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the ?third-party-programs.txt? or other similarly-named text file included with the
// Software Package for additional details.
#include "vpu/vpu_performance_model.h"

#include <gtest/gtest.h>

#include "vpu/dpu_workload.h"
#include "vpu/dpu_types.h"
#include "vpu/performance.h"


namespace VPUNN_unit_tests {
using namespace VPUNN;

/**
 * @brief Test suite for native computation type detection in VPU performance models
 *
 * This test suite validates the correct classification of DPU workloads based on their
 * data types into different native computation categories (FP16, FP8, I8, etc.).
 * The classification affects how the VPU hardware processes the workload and impacts
 * performance characteristics.
 */
class NativeComputationTest : public ::testing::Test {
protected:
    HWPerformanceModel perf_model;

    /**
     * @brief Structure defining data type combinations for test workloads
     *
     * Represents different scenarios of input, output, and optional weight data types
     * that can occur in real neural network operations.
     */

    struct TestInputs {
        DataType in_type;                                ///< Input tensor data type
        DataType out_type;                               ///< Output tensor data type
        std::optional<DataType> wts_type = std::nullopt; ///< Optional weights data type (if different from input)
    };

    struct TestCases {
        TestInputs t_in;
        bool t_exp;
    };

    using TestVector = std::vector<TestCases>;

protected:

    /**
     * @brief Executes test function on generated workloads and validates results
     *
     * @tparam Func Function type that accepts a DPUWorkload and returns bool
     * @param tests Vector of test cases with data type specs and expected results
     * @param test_func Function under test (e.g., native_comp_is_any_fp, native_comp_on_fp16)
     *
     * Generates workloads from test specs, runs the test function on each,
     * and compares results with expectations. Provides detailed error messages
     * including data type information on test failures.
     */
    template <typename Func>
    void check_types(const TestVector& tests, Func test_func) {
        std::vector<DPUWorkload> wls = generate_test_wls(tests);
        // Ensure we have test data
        ASSERT_FALSE(wls.empty());
        ASSERT_EQ(wls.size(), tests.size());

        // Test each workload and compare with expected result
        for (size_t i = 0; i < wls.size(); i++) {
            EXPECT_EQ(test_func(wls[i]), tests[i].t_exp)
                    << "Test case " << i << " failed: expected " << tests[i].t_exp << " but got "
                    << test_func(wls[i])
                    // Detailed error information showing the data types involved
                    << "\n\tInput type: " << mapToText<DataType>().at(static_cast<int>(wls[i].inputs[0].get_dtype()))
                    << "\n\tOutput type: " << mapToText<DataType>().at(static_cast<int>(wls[i].outputs[0].get_dtype()))
                    << "\n\tWeights type: "
                    << ((wls[i].weight_type) ? mapToText<DataType>().at(static_cast<int>(wls[i].weight_type.value()))
                                             : "N/A");
        }
    }

private:
    /**
     * @brief Creates standardized convolution workloads with specified data types
     *
     * @param tests Vector of test cases containing data type specifications
     * @return Vector of DPUWorkload objects with configured input/output/weight types
     *
     * Generates uniform 16x16x64 convolution workloads with 1x1 kernels on NPU_5_0
     * for isolated data type testing. Sets optional weight types for mixed-precision scenarios.
     */
    std::vector<DPUWorkload> generate_test_wls(const TestVector& tests) {
        std::vector<DPUWorkload> ret_wls;
        for (const auto& test : tests) {
            // Create a standard convolution workload with specified data types
            DPUWorkload wl = {
                    VPUDevice::NPU_5_0,                              // Target VPU device
                    Operation::CONVOLUTION,                          // Convolution operation
                    {VPUTensor(16, 16, 64, 1, test.t_in.in_type)},   // Input tensor: 16x16x64 with specified type
                    {VPUTensor(16, 16, 64, 1, test.t_in.out_type)},  // Output tensor: same shape, specified type
                    {1, 1},                                          // 1x1 kernel (minimal convolution)
                    {1, 1},                                          // Stride 1x1
                    {0, 0, 0, 0},                                    // No padding
                    ExecutionMode::CUBOID_16x16,                     // Execution mode optimized for NPU_5_0
            };

            // Set explicit weight type if provided (for mixed-precision scenarios)
            if (test.t_in.wts_type)
                wl.weight_type = test.t_in.wts_type.value();

            ret_wls.push_back(wl);
        }
        return ret_wls;
    }
};

/**
 * @brief Test case: Detects if any data type in the workload is floating point
 *
 * Tests the native_comp_is_any_fp() function which should return true if:
 * - Input tensor is any floating point type (FLOAT16, BFLOAT16, FLOAT32, HF8, BF8)
 * - Output tensor is any floating point type
 * - Weight tensor is any floating point type
 *
 * Expected results correspond to test cases in t_in array.
 */
TEST_F(NativeComputationTest, DetectAnyFloatingPointInputType) {
    TestVector any_fp = {
            //   Input Type     |    Output Type    |    Weights Type    | Expectation
            {{DataType::INT32, DataType::INT32}, false},                    // INT32 -> INT32: No FP types
            {{DataType::UINT8, DataType::UINT8}, false},                    // UINT8 -> UINT8: No FP types
            {{DataType::UINT8, DataType::UINT8, DataType::FLOAT16}, false}, // UINT8 -> UINT8 (FP16 weights): Input/output are not FP
            {{DataType::FLOAT16, DataType::UINT8}, true},                   // FLOAT16 -> UINT8: Input is FP
            {{DataType::BFLOAT16, DataType::UINT8}, true},                  // BFLOAT16 -> UINT8: Input is FP
            {{DataType::FLOAT32, DataType::UINT8}, true},                   // FLOAT32 -> UINT8: Input is FP
            {{DataType::HF8, DataType::UINT8}, true},                       // HF8 -> UINT8: Input is FP
            {{DataType::HF8, DataType::UINT8, DataType::FLOAT16}, true},    // HF8 -> UINT8 (FP16 weights): Input is FP
            {{DataType::BF8, DataType::UINT8}, true},                       // BF8 -> UINT8: Input is FP
            {{DataType::FLOAT32, DataType::INT8, DataType::INT4}, true},    // FLOAT32 -> INT8 (INT4 weights): Input is FP
            {{DataType::UINT8, DataType::BF8, DataType::UINT8}, false},     // UINT8 -> BF8 (UINT8 weights): Input is not FP
            {{DataType::FLOAT16, DataType::BFLOAT16, DataType::HF8}, true}, // FLOAT16 -> BFLOAT16 (HF8 weigts): Input is FP

    };
    check_types(any_fp, 
                [&](const DPUWorkload& wl) { 
                    return perf_model.native_comp_is_any_fp(wl); 
                });
}

/**
 * @brief Test case: Detects if computation uses FP16 precision
 *
 * Tests the native_comp_on_fp16() function which should return true for:
 * - FLOAT16, BFLOAT16, FLOAT32 input types (16-bit or higher precision FP)
 * - Should return false for FP8 types (HF8, BF8) as they use different precision
 */
TEST_F(NativeComputationTest, DetectFP16FamilyInputTypes) {

    TestVector fp16 = {
            //   Input Type     |    Output Type    |    Weights Type    | Expectation
            {{DataType::INT32, DataType::INT32}, false},                    // INT32 -> INT32: Not FP16
            {{DataType::UINT8, DataType::UINT8}, false},                    // UINT8 -> UINT8: Not FP16
            {{DataType::UINT8, DataType::UINT8, DataType::FLOAT16}, false}, // UINT8 -> UINT8 (FP16 weights): Input/output not FP16
            {{DataType::FLOAT16, DataType::UINT8}, true},                   // FLOAT16 -> UINT8: Input is FP16
            {{DataType::BFLOAT16, DataType::UINT8}, true},                  // BFLOAT16 -> UINT8: Input is FP16-class
            {{DataType::FLOAT32, DataType::UINT8}, true},                   // FLOAT32 -> UINT8: Input is higher precision FP
            {{DataType::HF8, DataType::UINT8}, false},                      // HF8 -> UINT8: Input is FP8, not FP16
            {{DataType::HF8, DataType::UINT8, DataType::FLOAT16}, false},   // HF8 -> UINT8 (FP16 weights): Input is FP8, not FP16
            {{DataType::BF8, DataType::UINT8}, false},                      // BF8 -> UINT8: Input is FP8, not FP16
            {{DataType::FLOAT32, DataType::INT8, DataType::INT4}, true},    // FLOAT32 -> INT8 (INT4 weights): Input is FP32
            {{DataType::UINT8, DataType::BF8, DataType::UINT8}, false},     // UINT8 -> BF8 (UINT8 weights): Input is not FP
            {{DataType::FLOAT16, DataType::BFLOAT16, DataType::HF8}, true}, // FLOAT16 -> BFLOAT16 (HF8 weigts): Input is FP16
    };

    check_types(fp16, 
                [&](const DPUWorkload& wl) { 
                    return perf_model.native_comp_on_fp16(wl);
                });
}

/**
 * @brief Test case: Detects if computation uses FP8 precision
 *
 * Tests the native_comp_on_fp8() function which should return true for:
 * - HF8 (Half-precision Float 8-bit) input types
 * - BF8 (Brain Float 8-bit) input types
 * - Should return false for higher precision FP types
 */
TEST_F(NativeComputationTest, DetectFP8InputWithNonFP16Weights) {
    TestVector fp8 = {
            //   Input Type     |    Output Type    |    Weights Type    | Expectation
            {{DataType::INT32, DataType::INT32}, false},                     // INT32 -> INT32: Not FP8
            {{DataType::UINT8, DataType::UINT8}, false},                     // UINT8 -> UINT8: Not FP8
            {{DataType::UINT8, DataType::UINT8, DataType::FLOAT16}, false},  // UINT8 -> UINT8 (FP16 weights): Not FP8
            {{DataType::FLOAT16, DataType::UINT8}, false},                   // FLOAT16 -> UINT8: FP16, not FP8
            {{DataType::BFLOAT16, DataType::UINT8}, false},                  // BFLOAT16 -> UINT8: FP16-class, not FP8
            {{DataType::FLOAT32, DataType::UINT8}, false},                   // FLOAT32 -> UINT8: FP32, not FP8
            {{DataType::HF8, DataType::UINT8}, true},                        // HF8 -> UINT8: Input is FP8
            {{DataType::HF8, DataType::UINT8, DataType::FLOAT16}, false},    // HF8 -> UINT8 (FP16 weights): Input is FP8 but weights are FP16
            {{DataType::BF8, DataType::UINT8}, true},                        // BF8 -> UINT8: Input is FP8
            {{DataType::FLOAT32, DataType::INT8, DataType::INT4}, false},    // FLOAT32 -> INT8 (INT4 weights): Input not FP8
            {{DataType::UINT8, DataType::BF8, DataType::UINT8}, false},      // UINT8 -> BF8 (UINT8 weights): Input not FP
            {{DataType::FLOAT16, DataType::BFLOAT16, DataType::HF8}, false}, // FLOAT16 -> BFLOAT16 (HF8 weigts): Input not FP8
            {{DataType::BF8, DataType::INT8, DataType::BF8}, true},          // BF8 -> INT8 (BF8 weights): Input FP8 and weights BF8
            {{DataType::BF8, DataType::INT8, DataType::UINT2}, true},        // BF8 -> INT8 (UINT2 weights): Input FP8 and weights UINT2
    };

    check_types(fp8, 
                [&](const DPUWorkload& wl) {
                    return perf_model.native_comp_on_fp8(wl);
                });
}

/**
 * @brief Test case: Detects if computation uses 8-bit integer precision
 *
 * Tests the native_comp_on_i8() function which should return true for:
 * - Operations where both input and output are 8-bit integer types
 * - Should consider weight types for mixed-precision scenarios
 *
 * Note: The corrected expectation for case 1 (UINT8->UINT8) should be true.
 */
TEST_F(NativeComputationTest, DetectI8InputWithNonFloatWeights) {
    TestVector i8 = {
            //   Input Type     |    Output Type    |    Weights Type    | Expectation
            {{DataType::INT32, DataType::INT32}, true},                     // INT32 -> INT32: 32-bit integer (may be treated as I8-compatible)
            {{DataType::UINT8, DataType::UINT8}, true},                     // UINT8 -> UINT8: Pure 8-bit integer computation
            {{DataType::UINT8, DataType::UINT8, DataType::FLOAT16}, false}, // UINT8 -> UINT8 (FP16 weights): Mixed precision with FP weights
            {{DataType::FLOAT16, DataType::UINT8}, false},                  // FLOAT16 -> UINT8: Input is FP, not I8
            {{DataType::BFLOAT16, DataType::UINT8}, false},                 // BFLOAT16 -> UINT8: Input is FP, not I8
            {{DataType::FLOAT32, DataType::UINT8}, false},                  // FLOAT32 -> UINT8: Input is FP, not I8
            {{DataType::HF8, DataType::UINT8}, false},                      // HF8 -> UINT8: Input is FP8, not I8 
            {{DataType::HF8, DataType::UINT8, DataType::FLOAT16}, false},   // HF8 -> UINT8 (FP16 weights): Input is FP8, not I8
            {{DataType::BF8, DataType::UINT8}, false},                      // BF8 -> UINT8: Input is FP8, not I8
            {{DataType::INT4, DataType::BF8, DataType::FLOAT4}, true},      // INT4 -> BF8 (FLOAT4 weights): Input I4, weights FP
            {{DataType::UINT2, DataType::BFLOAT16, DataType::HF8}, false},  // UINT2 -> BFLOAT16 (HF8 weights): Input I2, weights FP
            {{DataType::INT1, DataType::INT32, DataType::UINT4}, true}      // INT1 -> INT32 (UINT4 weights): Input I1, weights UINT4
    };

    check_types(i8, 
        [&](const DPUWorkload& wl) {
            return perf_model.native_comp_on_i8(wl);
        });
}

/**
 * @brief Test case: Validates I8 family data type classification for tensors
 *
 * Tests the `is_i8family()` method which determines if a tensor's data type belongs
 * to the integer family (I8, I4, I2, I1, etc.) vs floating-point types (FP16, FP8).
 * This classification affects VPU hardware execution path selection and memory optimization.
 *
 * Validates classification for input, output, and weight tensors across various integer
 * and floating-point data types.
 */
TEST_F(NativeComputationTest, DetectI8ForTensors) {
    TestVector test_input_output = {
        {{DataType::BF8, DataType::BF8}, false},        // BF8 input is not I8
        {{DataType::UINT8, DataType::UINT8}, true},     // UINT8 input is I8
        {{DataType::INT8, DataType::INT8}, true},       // INT8 input is I8
        {{DataType::UINT4, DataType::UINT4}, true},     // UINT4 input is I8
        {{DataType::INT4, DataType::INT4}, true},       // INT4 input is I8
        {{DataType::UINT2, DataType::UINT2}, true},     // UINT2 input is I8
        {{DataType::INT2, DataType::INT2}, true},       // INT2 input is I8
        {{DataType::UINT1, DataType::UINT1}, true},     // UINT1 input is I8
        {{DataType::INT1, DataType::INT1}, true},       // INT1 input is I8
        {{DataType::INT32, DataType::INT32}, true},     // INT32 input is considered I8 family here
        {{DataType::UINT16, DataType::UINT16}, true},   // UINT16 input is considered I8 family here
        {{DataType::INT16, DataType::INT16}, true}      // INT16 input is considered I8 family here
    };

    // Test I8 family classification for input tensors
    check_types(test_input_output, [](const DPUWorkload& wl) {
        return wl.inputs[0].is_i8family();
    });

    // Test I8 family classification for output tensors
    check_types(test_input_output, [](const DPUWorkload& wl) {
        return wl.outputs[0].is_i8family();
    });

    // Test I8 family classification for weight tensors (with explicit weight types)
    TestVector test_weights = {
        {{DataType::BF8, DataType::BF8, DataType::BF8}, false},       // BF8 weights is not I8
        {{DataType::UINT8, DataType::UINT8, DataType::UINT8}, true},  // UINT8 weights is I8
        {{DataType::INT8, DataType::INT8, DataType::INT8}, true},     // INT8 weights is I8
        {{DataType::UINT4, DataType::UINT4, DataType::UINT4}, true},  // UINT4 weights is I8
        {{DataType::INT4, DataType::INT4, DataType::INT4}, true},     // INT4 weights is I8
        {{DataType::UINT2, DataType::UINT2, DataType::UINT2}, true},  // UINT2 weights is I8
        {{DataType::INT2, DataType::INT2, DataType::INT2}, true},     // INT2 weights is I8
        {{DataType::UINT1, DataType::UINT1, DataType::UINT1}, true},  // UINT1 weights is I8
        {{DataType::INT1, DataType::INT1, DataType::INT1}, true},     // INT1 weights is I8
    };

    // Test I8 family classification for dynamically created weight tensors
    check_types(test_weights, [](const DPUWorkload& wl) {
        const VPUTensor wts({1, 1, 1, 1}, wl.get_weight_type());
        return wts.is_i8family();
    });
}

}