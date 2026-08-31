// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include <gtest/gtest.h>

#include "vpu/validation/serializable_dpu.h"

#include <algorithm>
#include <optional>
#include <unordered_map>
#include <variant>

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {
using namespace VPUNN;

class SerializableDPU_Test : public ::testing::Test {
public:
    void set_member_map_value(SerializableDPU& serializable_wl, const std::string& key, const std::string& value) {
        auto& member_map = serializable_wl.get_member_map();
        auto it = member_map.find(key);
        ASSERT_NE(it, member_map.end()) << "Member map key '" << key
                                        << "' not found. "
                                           "Check member name spelling or if it has been renamed.";

        auto* set_get = std::get_if<SetGet_MemberMapValues>(&it->second);
        ASSERT_NE(set_get, nullptr) << "Member map entry '" << key << "' is not a set/get callback";
        (*set_get)(true, value);
    }
};

TEST_F(SerializableDPU_Test, InPlaceOutputMemory_SetterHeuristic_AppliesCorrectly) {
    // Elementwise + matching layout/datatype + non-numeric field => TRUE
    {
        DPUWorkload wl = {VPUDevice::NPU_5_0,
                          Operation::ELTWISE,
                          {VPUTensor(56, 56, 64, 1, DataType::INT8, Layout::ZXY)},
                          {VPUTensor(56, 56, 64, 1, DataType::INT8, Layout::ZXY)},
                          {1, 1},
                          {1, 1},
                          {0, 0, 0, 0},
                          ExecutionMode::CUBOID_16x16};
        SerializableDPU serializable_dpu(wl);
        set_member_map_value(serializable_dpu, "in_place_output", "");
        EXPECT_TRUE(serializable_dpu.dpu_operation_data().in_place_output_memory)
            << "TRUE for elementwise with matching layouts";
    }

    // Elementwise + different layout + non-numeric field => FALSE
    {
        DPUWorkload wl = {VPUDevice::NPU_5_0,
                          Operation::ELTWISE,
                          {VPUTensor(56, 56, 64, 1, DataType::INT8, Layout::ZXY)},
                          {VPUTensor(56, 56, 64, 1, DataType::INT8, Layout::XYZ)},
                          {1, 1},
                          {1, 1},
                          {0, 0, 0, 0},
                          ExecutionMode::CUBOID_16x16};
        SerializableDPU serializable_dpu(wl);
        set_member_map_value(serializable_dpu, "in_place_output", "");
        EXPECT_FALSE(serializable_dpu.dpu_operation_data().in_place_output_memory) << "FALSE when layouts differ";
    }

    // Non-elementwise + non-numeric field => FALSE
    {
        DPUWorkload wl = {VPUDevice::NPU_5_0,
                          Operation::CONVOLUTION,
                          {VPUTensor(56, 56, 64, 1, DataType::INT8, Layout::ZXY)},
                          {VPUTensor(56, 56, 64, 1, DataType::INT8, Layout::ZXY)},
                          {1, 1},
                          {1, 1},
                          {0, 0, 0, 0},
                          ExecutionMode::CUBOID_16x16};
        SerializableDPU serializable_dpu(wl);
        set_member_map_value(serializable_dpu, "in_place_output", "");
        EXPECT_FALSE(serializable_dpu.dpu_operation_data().in_place_output_memory) << "FALSE for non-elementwise ops";
    }

    // Explicit numeric value 1 overrides heuristic
    {
        DPUWorkload wl = {VPUDevice::NPU_5_0,
                          Operation::CONVOLUTION,
                          {VPUTensor(56, 56, 64, 1, DataType::INT8, Layout::ZXY)},
                          {VPUTensor(56, 56, 64, 1, DataType::INT8, Layout::ZXY)},
                          {1, 1},
                          {1, 1},
                          {0, 0, 0, 0},
                          ExecutionMode::CUBOID_16x16};
        SerializableDPU serializable_dpu(wl);
        set_member_map_value(serializable_dpu, "in_place_output", "1");
        EXPECT_TRUE(serializable_dpu.dpu_operation_data().in_place_output_memory) << "Explicit '1' overrides heuristic";
    }

    // Explicit numeric value 0 overrides heuristic
    {
        DPUWorkload wl = {VPUDevice::NPU_5_0,
                          Operation::ELTWISE,
                          {VPUTensor(56, 56, 64, 1, DataType::INT8, Layout::ZXY)},
                          {VPUTensor(56, 56, 64, 1, DataType::INT8, Layout::ZXY)},
                          {1, 1},
                          {1, 1},
                          {0, 0, 0, 0},
                          ExecutionMode::CUBOID_16x16};
        SerializableDPU serializable_dpu(wl);
        set_member_map_value(serializable_dpu, "in_place_output", "0");
        EXPECT_FALSE(serializable_dpu.dpu_operation_data().in_place_output_memory) << "Explicit '0' overrides heuristic";
    }
}

TEST_F(SerializableDPU_Test, ConstructorFromWorkload_AndFromOperation_AreEquivalent) {
    DPUWorkload wl = {VPUDevice::NPU_RESERVED,
                      Operation::DW_CONVOLUTION,
                      {VPUTensor(28, 28, 64, 1, DataType::INT8, Layout::ZXY)},
                      {VPUTensor(28, 28, 64, 1, DataType::UINT8, Layout::ZXY)},
                      {5, 5},
                      {2, 2},
                      {2, 2, 1, 1},
                      ExecutionMode::CUBOID_8x16};

    wl.activation_function = ActivationFunction::RELU;
    wl.weight_type = DataType::INT8;
    wl.input_swizzling = {Swizzling::KEY_1, Swizzling::KEY_2};
    wl.output_swizzling = {Swizzling::KEY_3};
    wl.weight_sparsity_enabled = true;
    wl.weight_sparsity = 0.2F;
    wl.act_sparsity = 0.1F;
    wl.output_write_tiles = 4;
    wl.isi_strategy = ISIStrategy::SPLIT_OVER_K;
    wl.halo.input_0_halo.top = 1;
    wl.sep_activators.sep_activators = true;
    wl.input_autopad = true;
    wl.output_autopad = false;
    wl.reduce_minmax_op = true;

    const DPUOperation op{wl};
    const SerializableDPU from_workload{wl};
    const SerializableDPU from_operation{op};

    EXPECT_EQ(from_workload.hash(), from_operation.hash());
    EXPECT_EQ(from_workload.to_DPUWorkload().hash(), from_operation.to_DPUWorkload().hash());
}

TEST_F(SerializableDPU_Test, WeightlessOperation_SetterHeuristic_AppliesCorrectly) {
    // Elementwise + different layout + non-numeric field => TRUE
    {
        DPUWorkload wl = {VPUDevice::NPU_5_0,
                          Operation::ELTWISE,
                          {VPUTensor(56, 56, 64, 1, DataType::INT8, Layout::ZXY)},
                          {VPUTensor(56, 56, 64, 1, DataType::INT8, Layout::XYZ)},
                          {1, 1},
                          {1, 1},
                          {0, 0, 0, 0},
                          ExecutionMode::CUBOID_16x16};
        SerializableDPU serializable_dpu(wl);
        set_member_map_value(serializable_dpu, "in_place_input1", "");
        EXPECT_TRUE(serializable_dpu.dpu_operation_data().weightless_operation)
            << "TRUE for elementwise with different layouts";
    }

    // Elementwise + matching layout/datatype + non-numeric field => FALSE
    {
        DPUWorkload wl = {VPUDevice::NPU_5_0,
                          Operation::ELTWISE,
                          {VPUTensor(56, 56, 64, 1, DataType::INT8, Layout::ZXY)},
                          {VPUTensor(56, 56, 64, 1, DataType::INT8, Layout::ZXY)},
                          {1, 1},
                          {1, 1},
                          {0, 0, 0, 0},
                          ExecutionMode::CUBOID_16x16};
        SerializableDPU serializable_dpu(wl);
        set_member_map_value(serializable_dpu, "in_place_input1", "");
        EXPECT_FALSE(serializable_dpu.dpu_operation_data().weightless_operation) << "FALSE when layouts and datatypes match";
    }

    // Non-elementwise + non-numeric field => FALSE
    {
        DPUWorkload wl = {VPUDevice::NPU_5_0,
                          Operation::CONVOLUTION,
                          {VPUTensor(56, 56, 64, 1, DataType::INT8, Layout::ZXY)},
                          {VPUTensor(56, 56, 64, 1, DataType::INT8, Layout::XYZ)},
                          {1, 1},
                          {1, 1},
                          {0, 0, 0, 0},
                          ExecutionMode::CUBOID_16x16};
        SerializableDPU serializable_dpu(wl);
        set_member_map_value(serializable_dpu, "in_place_input1", "");
        EXPECT_FALSE(serializable_dpu.dpu_operation_data().weightless_operation) << "FALSE for non-elementwise ops";
    }

    // Explicit numeric value 1 overrides heuristic
    {
        DPUWorkload wl = {VPUDevice::NPU_5_0,
                          Operation::CONVOLUTION,
                          {VPUTensor(56, 56, 64, 1, DataType::INT8, Layout::ZXY)},
                          {VPUTensor(56, 56, 64, 1, DataType::INT8, Layout::XYZ)},
                          {1, 1},
                          {1, 1},
                          {0, 0, 0, 0},
                          ExecutionMode::CUBOID_16x16};
        SerializableDPU serializable_dpu(wl);
        set_member_map_value(serializable_dpu, "in_place_input1", "1");
        EXPECT_TRUE(serializable_dpu.dpu_operation_data().weightless_operation) << "Explicit '1' overrides heuristic";
    }

    // Explicit numeric value 0 overrides heuristic
    {
        DPUWorkload wl = {VPUDevice::NPU_5_0,
                          Operation::ELTWISE,
                          {VPUTensor(56, 56, 64, 1, DataType::INT8, Layout::ZXY)},
                          {VPUTensor(56, 56, 64, 1, DataType::INT8, Layout::XYZ)},
                          {1, 1},
                          {1, 1},
                          {0, 0, 0, 0},
                          ExecutionMode::CUBOID_16x16};
        SerializableDPU serializable_dpu(wl);
        set_member_map_value(serializable_dpu, "in_place_input1", "0");
        EXPECT_FALSE(serializable_dpu.dpu_operation_data().weightless_operation) << "Explicit '0' overrides heuristic";
    }
}

TEST_F(SerializableDPU_Test, WorkloadConversion_PreservesSemantics) {
    // Create a complex DPUWorkload with non-default values
    const DPUWorkload original_wl = {VPUDevice::NPU_5_0,
                                     Operation::DW_CONVOLUTION,
                                     {VPUTensor(28, 28, 64, 1, DataType::INT8, Layout::XYZ)},
                                     {VPUTensor(28, 28, 128, 1, DataType::UINT8, Layout::ZXY)},
                                     {3, 3},        // kernel
                                     {1, 1},        // stride
                                     {1, 1, 1, 1},  // padding
                                     ExecutionMode::CUBOID_16x16};

    // Wrap in SerializableDPU
    SerializableDPU serializable_dpu(original_wl);

    // Verify key fields are preserved
    EXPECT_EQ(serializable_dpu.dpu_operation_data().device, original_wl.device);
    EXPECT_EQ(serializable_dpu.dpu_operation_data().operation, original_wl.op);
    EXPECT_EQ(serializable_dpu.dpu_operation_data().execution_order, original_wl.execution_order);
    EXPECT_EQ(serializable_dpu.dpu_operation_data().kernel.width, original_wl.kernels[Dim::Grid::W]);
    EXPECT_EQ(serializable_dpu.dpu_operation_data().kernel.height, original_wl.kernels[Dim::Grid::H]);

    // Convert back to DPUWorkload
    DPUWorkload recovered_wl = serializable_dpu.to_DPUWorkload();

    // Verify core semantics match
    EXPECT_EQ(recovered_wl.device, original_wl.device);
    EXPECT_EQ(recovered_wl.op, original_wl.op);
    EXPECT_EQ(recovered_wl.execution_order, original_wl.execution_order);
    EXPECT_EQ(recovered_wl.kernels[Dim::Grid::W], original_wl.kernels[Dim::Grid::W]);
    EXPECT_EQ(recovered_wl.kernels[Dim::Grid::H], original_wl.kernels[Dim::Grid::H]);
    EXPECT_EQ(recovered_wl.strides[Dim::Grid::W], original_wl.strides[Dim::Grid::W]);
    EXPECT_EQ(recovered_wl.strides[Dim::Grid::H], original_wl.strides[Dim::Grid::H]);
    EXPECT_EQ(recovered_wl.activation_function, original_wl.activation_function);
}

TEST_F(SerializableDPU_Test, Hash_ChangesWhenSerializedFieldsChange) {
    const DPUWorkload base_wl = {VPUDevice::VPU_4_0,
                                 Operation::CONVOLUTION,
                                 {VPUTensor(16, 16, 64, 1, DataType::UINT8)},
                                 {VPUTensor(16, 16, 64, 1, DataType::UINT8)},
                                 {1, 1},
                                 {1, 1},
                                 {0, 0, 0, 0},
                                 ExecutionMode::CUBOID_16x16};

    SerializableDPU base_dpu(base_wl);
    const auto base_hash = base_dpu.hash();

    // Change device
    {
        auto wl = base_wl;
        wl.device = VPUDevice::NPU_5_0;
        SerializableDPU dpu(wl);
        EXPECT_NE(dpu.hash(), base_hash) << "Hash should change when device changes";
    }

    // Change operation
    {
        auto wl = base_wl;
        wl.op = Operation::ELTWISE;
        SerializableDPU dpu(wl);
        EXPECT_NE(dpu.hash(), base_hash) << "Hash should change when operation changes";
    }

    // Change input_0 shape
    {
        auto wl = base_wl;
        wl.inputs[0] = VPUTensor(32, 32, 64, 1, DataType::UINT8);
        SerializableDPU dpu(wl);
        EXPECT_NE(dpu.hash(), base_hash) << "Hash should change when input_0 shape changes";
    }

    // Change input_1 datatype (weight_type)
    {
        auto wl = base_wl;
        wl.weight_type = DataType::INT8;
        SerializableDPU dpu(wl);
        EXPECT_NE(dpu.hash(), base_hash) << "Hash should change when weight_type changes";
    }

    // Change kernel width
    {
        auto wl = base_wl;
        wl.kernels[Dim::Grid::W] = 3;
        SerializableDPU dpu(wl);
        EXPECT_NE(dpu.hash(), base_hash) << "Hash should change when kernel width changes";
    }

    // Change activation function
    {
        auto wl = base_wl;
        wl.activation_function = ActivationFunction::RELU;
        SerializableDPU dpu(wl);
        EXPECT_NE(dpu.hash(), base_hash) << "Hash should change when activation function changes";
    }

    // Change sparsity_enabled flags
    {
        auto wl = base_wl;
        wl.weight_sparsity_enabled = true;
        SerializableDPU dpu(wl);
        EXPECT_NE(dpu.hash(), base_hash) << "Hash should change when weight_sparsity_enabled changes";
    }
}

TEST_F(SerializableDPU_Test, Hash_IsStable_ConsistentAcrossMultipleCalls) {
    const DPUWorkload wl = {VPUDevice::NPU_5_0,
                            Operation::DW_CONVOLUTION,
                            {VPUTensor(28, 28, 64, 1, DataType::INT8)},
                            {VPUTensor(28, 28, 64, 1, DataType::UINT8)},
                            {3, 3},
                            {1, 1},
                            {1, 1, 1, 1},
                            ExecutionMode::CUBOID_16x16};

    SerializableDPU dpu(wl);

    // Hash should be deterministic across multiple calls
    const auto hash1 = dpu.hash();
    const auto hash2 = dpu.hash();
    const auto hash3 = dpu.hash();

    EXPECT_EQ(hash1, hash2) << "Hash must be stable across calls on same object";
    EXPECT_EQ(hash2, hash3) << "Hash must be stable across calls on same object";

    // Same workload in new object should produce same hash
    SerializableDPU dpu2(wl);
    EXPECT_EQ(hash1, dpu2.hash()) << "Identical workloads must produce identical hashes";
}

}  // namespace VPUNN_unit_tests
