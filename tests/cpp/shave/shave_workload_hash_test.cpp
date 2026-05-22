// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/shave_workload.h"
#include "vpu/validation/serializable_shave.h"
#include "core/serializer.h"
#include "common/common_helpers.h"

#include <gtest/gtest.h>
#include <filesystem>
#include <map>
#include <string>
#include <variant>
#include <vector>
#include <type_traits>
#include <chrono>
#include <cstdint>

namespace VPUNN_unit_tests {
using namespace VPUNN;

class SHAVEWorkloadHashTest : public ::testing::Test {
protected:
    // Helper to create a basic workload for reuse
    SHAVEWorkload makeBasicWorkload() const {
        return SHAVEWorkload{
                "sigmoid",
                VPUDevice::VPU_4_0,
                {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
                {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
        };
    }
};

// ===== Determinism =====

/// Calling hash() multiple times on the same workload must return the same value.
TEST_F(SHAVEWorkloadHashTest, Deterministic_SameObjectSameHash) {
    const auto wl = makeBasicWorkload();
    const auto h1 = wl.hash();
    const auto h2 = wl.hash();
    EXPECT_EQ(h1, h2) << "hash() must be deterministic for the same object";
}

/// Two identically-constructed workloads must produce the same hash.
TEST_F(SHAVEWorkloadHashTest, Deterministic_IdenticalWorkloadsMatch) {
    const auto wl1 = makeBasicWorkload();
    const auto wl2 = makeBasicWorkload();
    EXPECT_EQ(wl1.hash(), wl2.hash()) << "Identical workloads must hash to the same value";
}

// ===== Sensitivity to each field =====

/// Changing the operation name must change the hash.
TEST_F(SHAVEWorkloadHashTest, Sensitive_OperationName) {
    const SHAVEWorkload wl1{
            "sigmoid",
            VPUDevice::VPU_4_0,
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
    };
    const SHAVEWorkload wl2{
            "tanh",
            VPUDevice::VPU_4_0,
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
    };
    EXPECT_NE(wl1.hash(), wl2.hash()) << "Different operation names must produce different hashes";
}

/// Changing the device must change the hash.
TEST_F(SHAVEWorkloadHashTest, Sensitive_Device) {
    const SHAVEWorkload wl1{
            "sigmoid",
            VPUDevice::VPU_4_0,
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
    };
    const SHAVEWorkload wl2{
            "sigmoid",
            VPUDevice::VPU_2_7,
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
    };
    EXPECT_NE(wl1.hash(), wl2.hash()) << "Different devices must produce different hashes";
}

/// Changing input tensor shape must change the hash.
TEST_F(SHAVEWorkloadHashTest, Sensitive_InputTensorShape) {
    const SHAVEWorkload wl1{
            "sigmoid",
            VPUDevice::VPU_4_0,
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
    };
    const SHAVEWorkload wl2{
            "sigmoid",
            VPUDevice::VPU_4_0,
            {VPUTensor(20, 100, 5, 1, DataType::FLOAT16)},  // different width
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
    };
    EXPECT_NE(wl1.hash(), wl2.hash()) << "Different input tensor shapes must produce different hashes";
}

/// Changing input and output tensor shape must change the hash.
TEST_F(SHAVEWorkloadHashTest, Sensitive_InputOutputTensorShape) {
    const SHAVEWorkload wl1{ 
        "sigmoid", 
        VPUDevice::VPU_4_0, 
        {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)}, 
        {VPUTensor(20, 100, 5, 1, DataType::FLOAT16)}, // output bigger here 
    }; 
    const SHAVEWorkload wl2{ 
        "sigmoid", 
        VPUDevice::VPU_4_0, 
        {VPUTensor(20, 100, 5, 1, DataType::FLOAT16)}, // input bigger here 
        {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)}, 
    };
    EXPECT_NE(wl1.hash(), wl2.hash()) << "Different input and output tensor shapes must produce different hashes";
}

/// Changing input tensor datatype must change the hash.
TEST_F(SHAVEWorkloadHashTest, Sensitive_InputTensorDatatype) {
    const SHAVEWorkload wl1{
            "sigmoid",
            VPUDevice::VPU_4_0,
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
    };
    const SHAVEWorkload wl2{
            "sigmoid",
            VPUDevice::VPU_4_0,
            {VPUTensor(10, 100, 5, 1, DataType::UINT8)},  // different datatype
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
    };
    EXPECT_NE(wl1.hash(), wl2.hash()) << "Different input tensor datatypes must produce different hashes";
}

/// Changing output tensor shape must change the hash.
TEST_F(SHAVEWorkloadHashTest, Sensitive_OutputTensorShape) {
    const SHAVEWorkload wl1{
            "sigmoid",
            VPUDevice::VPU_4_0,
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
    };
    const SHAVEWorkload wl2{
            "sigmoid",
            VPUDevice::VPU_4_0,
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {VPUTensor(10, 200, 5, 1, DataType::FLOAT16)},  // different output height
    };
    EXPECT_NE(wl1.hash(), wl2.hash()) << "Different output tensor shapes must produce different hashes";
}

/// Changing output tensor datatype must change the hash.
TEST_F(SHAVEWorkloadHashTest, Sensitive_OutputTensorDatatype) {
    const SHAVEWorkload wl1{
            "sigmoid",
            VPUDevice::VPU_4_0,
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
    };
    const SHAVEWorkload wl2{
            "sigmoid",
            VPUDevice::VPU_4_0,
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {VPUTensor(10, 100, 5, 1, DataType::UINT8)},  // different output datatype
    };
    EXPECT_NE(wl1.hash(), wl2.hash()) << "Different output tensor datatypes must produce different hashes";
}

// ===== Sensitivity to call_params =====

/// Changing a call parameter value must change the hash.
TEST_F(SHAVEWorkloadHashTest, Sensitive_CallParamValue) {
    const SHAVEWorkload wl1{
            "sigmoid",
            VPUDevice::VPU_4_0,
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {{7}},
    };
    const SHAVEWorkload wl2{
            "sigmoid",
            VPUDevice::VPU_4_0,
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {{10}},
    };
    EXPECT_NE(wl1.hash(), wl2.hash()) << "Different call parameter values must produce different hashes";
}

/// Presence vs absence of call parameters must change the hash.
TEST_F(SHAVEWorkloadHashTest, Sensitive_CallParamPresence) {
    const SHAVEWorkload wl_no_params{
            "sigmoid",
            VPUDevice::VPU_4_0,
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
    };
    const SHAVEWorkload wl_with_params{
            "sigmoid",
            VPUDevice::VPU_4_0,
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {{7}},
    };
    EXPECT_NE(wl_no_params.hash(), wl_with_params.hash())
            << "Workload with params must differ from workload without params";
}

// ===== Sensitivity to extra_params =====

/// Changing an extra parameter value must change the hash.
TEST_F(SHAVEWorkloadHashTest, Sensitive_ExtraParamValue) {
    SHAVEWorkload::ExtraParameters ep1{{"axis", SHAVEWorkload::Param{0}}};
    SHAVEWorkload::ExtraParameters ep2{{"axis", SHAVEWorkload::Param{1}}};

    const SHAVEWorkload wl1{
            "softmax",
            VPUDevice::VPU_4_0,
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {},
            ep1,
    };
    const SHAVEWorkload wl2{
            "softmax",
            VPUDevice::VPU_4_0,
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {},
            ep2,
    };
    EXPECT_NE(wl1.hash(), wl2.hash()) << "Different extra parameter values must produce different hashes";
}

/// Changing an extra parameter key must change the hash.
TEST_F(SHAVEWorkloadHashTest, Sensitive_ExtraParamKey) {
    SHAVEWorkload::ExtraParameters ep1{{"axis", SHAVEWorkload::Param{0}}};
    SHAVEWorkload::ExtraParameters ep2{{"dim", SHAVEWorkload::Param{0}}};

    const SHAVEWorkload wl1{
            "softmax",
            VPUDevice::VPU_4_0,
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {},
            ep1,
    };
    const SHAVEWorkload wl2{
            "softmax",
            VPUDevice::VPU_4_0,
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {},
            ep2,
    };
    EXPECT_NE(wl1.hash(), wl2.hash()) << "Different extra parameter keys must produce different hashes";
}

/// Presence vs absence of extra parameters must change the hash.
TEST_F(SHAVEWorkloadHashTest, Sensitive_ExtraParamPresence) {
    SHAVEWorkload::ExtraParameters ep{{"axis", SHAVEWorkload::Param{0}}};

    const SHAVEWorkload wl_no_extra{
            "softmax",
            VPUDevice::VPU_4_0,
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
    };
    const SHAVEWorkload wl_with_extra{
            "softmax",
            VPUDevice::VPU_4_0,
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {},
            ep,
    };
    EXPECT_NE(wl_no_extra.hash(), wl_with_extra.hash())
            << "Workload with extra params must differ from workload without";
}

// ===== Multiple inputs/outputs =====

/// Different number of input tensors must change the hash.
TEST_F(SHAVEWorkloadHashTest, Sensitive_NumberOfInputs) {
    const SHAVEWorkload wl_one_input{
            "eltwise_add",
            VPUDevice::VPU_4_0,
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
    };
    const SHAVEWorkload wl_two_inputs{
            "eltwise_add",
            VPUDevice::VPU_4_0,
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16), VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
    };
    EXPECT_NE(wl_one_input.hash(), wl_two_inputs.hash())
            << "Different number of inputs must produce different hashes";
}

/// Different number of output tensors must change the hash.
TEST_F(SHAVEWorkloadHashTest, Sensitive_NumberOfOutputs) {
    const SHAVEWorkload wl_one_output{
            "split",
            VPUDevice::VPU_4_0,
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
    };
    const SHAVEWorkload wl_two_outputs{
            "split",
            VPUDevice::VPU_4_0,
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {VPUTensor(10, 50, 5, 1, DataType::FLOAT16), VPUTensor(10, 50, 5, 1, DataType::FLOAT16)},
    };
    EXPECT_NE(wl_one_output.hash(), wl_two_outputs.hash())
            << "Different number of outputs must produce different hashes";
}

/// Workload with multiple outputs (using the new 4-output staging) must hash consistently.
TEST_F(SHAVEWorkloadHashTest, MultipleOutputs_Deterministic) {
    const SHAVEWorkload wl{
            "split",
            VPUDevice::VPU_4_0,
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {VPUTensor(10, 25, 5, 1, DataType::FLOAT16), VPUTensor(10, 25, 5, 1, DataType::FLOAT16),
             VPUTensor(10, 25, 5, 1, DataType::FLOAT16), VPUTensor(10, 25, 5, 1, DataType::FLOAT16)},
    };
    EXPECT_EQ(wl.hash(), wl.hash()) << "Multi-output workload hash must be deterministic";
}

/// Two multi-output workloads that differ only in one output tensor shape must differ in hash.
TEST_F(SHAVEWorkloadHashTest, MultipleOutputs_Sensitive) {
    const SHAVEWorkload wl1{
            "split",
            VPUDevice::VPU_4_0,
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {VPUTensor(10, 25, 5, 1, DataType::FLOAT16), VPUTensor(10, 25, 5, 1, DataType::FLOAT16),
             VPUTensor(10, 25, 5, 1, DataType::FLOAT16), VPUTensor(10, 25, 5, 1, DataType::FLOAT16)},
    };
    const SHAVEWorkload wl2{
            "split",
            VPUDevice::VPU_4_0,
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {VPUTensor(10, 25, 5, 1, DataType::FLOAT16), VPUTensor(10, 25, 5, 1, DataType::FLOAT16),
             VPUTensor(10, 25, 5, 1, DataType::FLOAT16), VPUTensor(10, 30, 5, 1, DataType::FLOAT16)},  // last differs
    };
    EXPECT_NE(wl1.hash(), wl2.hash())
            << "Multi-output workloads differing in one output must produce different hashes";
}

// ===== Copy / move semantics =====

/// A copied workload must have the same hash as the original.
TEST_F(SHAVEWorkloadHashTest, CopyPreservesHash) {
    const auto wl_original = makeBasicWorkload();
    const SHAVEWorkload wl_copy(wl_original);
    EXPECT_EQ(wl_original.hash(), wl_copy.hash()) << "Copy-constructed workload must hash identically";
}

/// A copy-assigned workload must have the same hash as the original.
TEST_F(SHAVEWorkloadHashTest, CopyAssignPreservesHash) {
    const auto wl_original = makeBasicWorkload();
    SHAVEWorkload wl_assigned;
    wl_assigned = wl_original;
    EXPECT_EQ(wl_original.hash(), wl_assigned.hash()) << "Copy-assigned workload must hash identically";
}

/// A moved workload must produce the same hash value as the original before move.
TEST_F(SHAVEWorkloadHashTest, MovePreservesHash) {
    auto wl_source = makeBasicWorkload();
    const auto expected_hash = wl_source.hash();
    const SHAVEWorkload wl_moved(std::move(wl_source));
    EXPECT_EQ(expected_hash, wl_moved.hash()) << "Move-constructed workload must hash identically to original";
}

// ===== Non-zero hash =====

/// Hash of a non-trivial workload should not be zero (extremely unlikely with FNV-1a).
TEST_F(SHAVEWorkloadHashTest, NonZeroHash) {
    const auto wl = makeBasicWorkload();
    EXPECT_NE(wl.hash(), 0u) << "Hash of a real workload should not be zero";
}

// ===== Param type sensitivity =====

/// An int param vs a float param with same numeric value should produce different hashes
/// because FNV hashing paths differ for int vs float.
TEST_F(SHAVEWorkloadHashTest, Sensitive_ParamType_IntVsFloat) {
    // int param = 7
    const SHAVEWorkload wl_int{
            "sigmoid",
            VPUDevice::VPU_4_0,
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {{SHAVEWorkload::Param{7}}},
    };
    // float param = 7.0f
    const SHAVEWorkload wl_float{
            "sigmoid",
            VPUDevice::VPU_4_0,
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {{SHAVEWorkload::Param{7.0f}}},
    };
    // These should likely differ because variant index and hash path differ
    EXPECT_NE(wl_int.hash(), wl_float.hash())
            << "Int param vs float param (same numeric value) should produce different hashes";
}

/// A string param must produce a different hash than an int param.
TEST_F(SHAVEWorkloadHashTest, Sensitive_ParamType_StringVsInt) {
    const SHAVEWorkload wl_str{
            "sigmoid",
            VPUDevice::VPU_4_0,
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {{SHAVEWorkload::Param{std::string("hello")}}},
    };
    const SHAVEWorkload wl_int{
            "sigmoid",
            VPUDevice::VPU_4_0,
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {{SHAVEWorkload::Param{42}}},
    };
    EXPECT_NE(wl_str.hash(), wl_int.hash())
            << "String param vs int param should produce different hashes";
}

// ===== Default-constructed workload =====

/// Default-constructed workload hash should be deterministic.
TEST_F(SHAVEWorkloadHashTest, DefaultConstructed_Deterministic) {
    const SHAVEWorkload wl1;
    const SHAVEWorkload wl2;
    EXPECT_EQ(wl1.hash(), wl2.hash()) << "Two default-constructed workloads must hash identically";
}

/// Default-constructed workload must differ from a non-trivial workload.
TEST_F(SHAVEWorkloadHashTest, DefaultConstructed_DiffersFromNonTrivial) {
    const SHAVEWorkload wl_default;
    const auto wl_real = makeBasicWorkload();
    EXPECT_NE(wl_default.hash(), wl_real.hash()) << "Default workload hash must differ from real workload hash";
}

/// ===== Serialization consistency =====

/// Serialize and deserialize a workload should preserve the hash.
TEST_F(SHAVEWorkloadHashTest, SerializationPreservesHash) {
    SHAVEWorkload::ExtraParameters ep{{"axis", SHAVEWorkload::Param{1}}};
    const SHAVEWorkload original{
            "softmax",
            VPUDevice::VPU_4_0,
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {{SHAVEWorkload::Param{42}}},
            ep,
    };
    const auto original_hash = original.hash();

    // Build unique temporary file path for this test instance
    const auto temp_dir = std::filesystem::temp_directory_path();
    const auto now_ticks = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto unique_suffix = std::to_string(now_ticks);
    const std::filesystem::path temp_path = temp_dir / ("shave_hash_roundtrip_test_" + unique_suffix + ".csv");
    ScopedFileCleanup temp_file_guard(temp_path);  // Ensure temp file is cleaned up after test

    // Convert to the serializable form and write to CSV
    SerializableSHAVE serializable(original);
    Serializer<FileFormat::CSV> serializer{true};
    serializer.initialize(temp_path.string(), FileMode::READ_WRITE,
                          SerializableSHAVE::_get_member_names());
    serializer.serialize(serializable);
    serializer.end();

    // Deserialize back into a fresh SerializableSHAVE, then convert to SHAVEWorkload
    serializer.jump_to_beginning();
    SerializableSHAVE deserialized;
    const bool ok = serializer.deserialize(deserialized);
    ASSERT_TRUE(ok) << "Deserialization must succeed";
    const SHAVEWorkload restored = deserialized.clone_as_SHAVEWorkload();

    EXPECT_EQ(original_hash, restored.hash())
            << "Hash must be preserved across CSV serialization round-trip";
}


}  // namespace VPUNN_unit_tests
