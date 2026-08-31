// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/shave/generated_shave_prefetch_cost_population.h"
#include "vpu/shave/shave_cost_providers/shave_provider_bundles.h"

#include <gtest/gtest.h>
#include "common/common_helpers.h"
#include "vpu_cost_model.h"
#include "vpu_shave_cost_model.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <set>
#include <unordered_set>
#include <utility>
#include <vector>

namespace VPUNN_unit_tests {
using namespace VPUNN;

class TestSHAVEPrefetch : public ::testing::Test {
protected:
    VPUNN::SHAVECostModel empty_model{};

    const VPUNN::VPUTensor input_0{56, 56, 16, 1, VPUNN::DataType::FLOAT16};
    const VPUNN::VPUTensor output_0{56, 56, 16, 1, VPUNN::DataType::FLOAT16};

    // Returns all NPU-generation devices that have a non-empty prefetch LUT in the current build.
    // Iterates the enum from NPU_5_0 forward so new devices are covered automatically.
    static std::vector<VPUDevice> coveredPrefetchDevices() {
        std::vector<VPUDevice> devices;
        for (int d = static_cast<int>(VPUDevice::NPU_5_0); d < static_cast<int>(VPUDevice::__size); ++d) {
            const auto device = static_cast<VPUDevice>(d);
            if (!PopulatedShavePrefetchCostLUT::get_lut_for_device(device).empty()) {
                devices.push_back(device);
            }
        }
        return devices;
    }
};

// Verifies that the generated-table factory loads all generated entries correctly.
// Asserts unknown operators on covered devices return ERROR_SHAVE_PREFETCH_NOT_FOUND,
// and uncovered devices return ERROR_SHAVE_PREFETCH_DEVICE_NOT_COVERED.
TEST_F(TestSHAVEPrefetch, PrefetchFactory_LoadsGeneratedEntriesAndReturnsNotFoundForUnknownOp) {
    auto strategy = ShaveCostProviderBundles::createDeviceMappedPrefetchStrategy();

    const auto devices = coveredPrefetchDevices();
    ASSERT_FALSE(devices.empty()) << "No prefetch-covered devices found — NPU_5_0 LUT must always be present";

    for (const auto device : devices) {
        const auto& lut = PopulatedShavePrefetchCostLUT::get_lut_for_device(device);
        for (const auto& op_entry : lut) {
            SHAVEWorkload swl{op_entry.first, device, {input_0}, {output_0}};
            EXPECT_EQ(strategy.get_code_prefetch_cost(swl), op_entry.second)
                    << "Generated prefetch entry mismatch for op: " << op_entry.first;
        }
    }

    std::unordered_set<std::string> known;
    for (const auto device : devices) {
        for (const auto& op_entry : PopulatedShavePrefetchCostLUT::get_lut_for_device(device)) {
            known.insert(op_entry.first);
        }
    }

    // Unknown op on a covered device must return NOT_FOUND (device is present, op is absent).
    std::string unknown_op{"__generated_prefetch_unknown_op"};
    while (known.count(unknown_op) > 0U) {
        unknown_op += "_x";
    }
    const VPUDevice covered_device = VPUDevice::NPU_5_0;
    SHAVEWorkload covered_unknown{unknown_op, covered_device, {input_0}, {output_0}};
    EXPECT_EQ(strategy.get_code_prefetch_cost(covered_unknown), V(Cycles::ERROR_SHAVE_PREFETCH_NOT_FOUND));

    // Uncovered device must return DEVICE_NOT_COVERED regardless of op name.
    SHAVEWorkload uncovered_unknown{unknown_op, VPUDevice::VPU_4_0, {input_0}, {output_0}};
    EXPECT_EQ(strategy.get_code_prefetch_cost(uncovered_unknown), V(Cycles::ERROR_SHAVE_PREFETCH_DEVICE_NOT_COVERED));
}

// Coverage invariant: for each covered device, generated prefetch CSV operations
// must be exactly equal to getShaveSupportedOperations (both directions).
// Any missing or extra operation in either source must fail loudly.
TEST_F(TestSHAVEPrefetch, PrefetchCoverage_AllDeviceMappedOpsHaveMeasuredEntry_ForAllDevices) {
    const auto devices = coveredPrefetchDevices();
    ASSERT_FALSE(devices.empty()) << "No prefetch-covered devices found — NPU_5_0 LUT must always be present";

    for (const auto device : devices) {
        std::set<std::string> prefetch_ops;
        for (const auto& op_entry : PopulatedShavePrefetchCostLUT::get_lut_for_device(device)) {
            prefetch_ops.insert(op_entry.first);
        }

        const auto ops = empty_model.getShaveSupportedOperations(device);
        std::set<std::string> supported_ops;
        for (const auto& op : ops) {
            supported_ops.insert(op);
        }

        ASSERT_FALSE(supported_ops.empty())
                << "Expected device-mapped operations for device: " << VPUDevice_ToText.at((int)device);
        ASSERT_FALSE(prefetch_ops.empty())
                << "Expected generated prefetch operations for device: " << VPUDevice_ToText.at((int)device);

        EXPECT_EQ(prefetch_ops, supported_ops)
                << "Prefetch CSV operations must match exactly the supported operations for device: "
                << VPUDevice_ToText.at((int)device);

        for (const auto& op : ops) {
            SHAVEWorkload swl{op, device, {input_0}, {output_0}};
            const auto prefetch = empty_model.getCodePrefetchCost(swl);
            EXPECT_NE(prefetch, V(Cycles::ERROR_SHAVE_PREFETCH_NOT_FOUND))
                    << "Missing prefetch coverage for op: " << op << " on device " << VPUDevice_ToText.at((int)device);
            EXPECT_FALSE(Cycles::isErrorCode(prefetch)) << "Prefetch lookup returned unexpected error for op: " << op
                                                        << " on device " << VPUDevice_ToText.at((int)device);
        }
    }
}

// Unknown operators are reported by the standalone prefetch strategy and interpreted by SHAVECostModel.
// This test validates the public prefetch query path for clearly invalid operators.
// Expected behavior is a direct PREFETCH_NOT_FOUND signal on every covered device.
TEST_F(TestSHAVEPrefetch, GetCodePrefetchCost_UnknownOperator_ReturnsNotFound) {
    const auto devices = coveredPrefetchDevices();
    ASSERT_FALSE(devices.empty()) << "No prefetch-covered devices found — NPU_5_0 LUT must always be present";

    for (const auto device : devices) {
        SHAVEWorkload swl{"not_a_real_op", device, {input_0}, {output_0}};
        const CyclesInterfaceType cost = empty_model.getCodePrefetchCost(swl);
        EXPECT_EQ(cost, V(Cycles::ERROR_SHAVE_PREFETCH_NOT_FOUND));
    }
}

// Per-device prefetch table coverage: NPU 5.0 has all supported ops measured.
TEST_F(TestSHAVEPrefetch, PrefetchCoverage_NPU5_TablePresent_AllOpsIncluded) {
    const VPUDevice device = VPUDevice::NPU_5_0;
    const auto& lut = PopulatedShavePrefetchCostLUT::get_lut_for_device(device);

    ASSERT_FALSE(lut.empty()) << "Expected non-empty prefetch table for device: " << VPUDevice_ToText.at((int)device);

    const auto supported_ops = empty_model.getShaveSupportedOperations(device);
    ASSERT_FALSE(supported_ops.empty()) << "Expected supported operations for device: "
                                        << VPUDevice_ToText.at((int)device);

    std::set<std::string> prefetch_ops;
    for (const auto& op_entry : lut) {
        prefetch_ops.insert(op_entry.first);
    }

    std::set<std::string> supported_ops_exact;
    for (const auto& op : supported_ops) {
        supported_ops_exact.insert(op);
    }

    EXPECT_EQ(prefetch_ops, supported_ops_exact)
            << "Prefetch table must contain exactly all supported operations for device: "
            << VPUDevice_ToText.at((int)device);
}



// Prefetch lookup now requires exact operation names; aliases are not remapped.
TEST_F(TestSHAVEPrefetch, GetCodePrefetchCost_AliasNameIsNotRemapped_ReturnsNotFound) {
    const VPUDevice device = VPUDevice::NPU_5_0;

    OperationPrefetchLUT lut;
    lut["Softmax"] = static_cast<CyclesInterfaceType>(23);

    auto measured = std::make_shared<SingleDevicePrefetchCostStrategy>(lut);

    SHAVEWorkload canonical_wl{"Softmax", device, {input_0}, {output_0}};
    SHAVEWorkload alias_wl{"SoftMax", device, {input_0}, {output_0}};

    EXPECT_EQ(measured->get_code_prefetch_cost(canonical_wl), static_cast<CyclesInterfaceType>(23));
    EXPECT_EQ(measured->get_code_prefetch_cost(alias_wl), V(Cycles::ERROR_SHAVE_PREFETCH_NOT_FOUND));
}

// Validates that SHAVECostModel forwards prefetch queries to the generated-table strategy.
// We smoke-check one generated row (if available) and independently verify unknown-op behavior.
// This avoids duplicating full factory-table validation in model-level tests.
TEST_F(TestSHAVEPrefetch, ModelPrefetchQuery_MatchesGeneratedTableEntriesAndNotFound) {
    const auto& lut = PopulatedShavePrefetchCostLUT::get_lut_for_device(VPUDevice::NPU_5_0);
    ASSERT_FALSE(lut.empty());
    const auto& op_entry = *lut.begin();
    SHAVEWorkload sampled_wl{op_entry.first, VPUDevice::NPU_5_0, {input_0}, {output_0}};
    EXPECT_EQ(empty_model.getCodePrefetchCost(sampled_wl), op_entry.second)
            << "Model prefetch query mismatch for generated CSV op: " << op_entry.first;

    std::unordered_set<std::string> known;
    for (const auto& op : lut) {
        known.insert(op.first);
    }

    std::string unknown_op{"__generated_prefetch_model_unknown_op"};
    while (known.count(unknown_op) > 0U) {
        unknown_op += "_x";
    }

    // Unknown op on a covered device → NOT_FOUND; uncovered device → DEVICE_NOT_COVERED.
    const VPUDevice covered_device = VPUDevice::NPU_5_0;
    SHAVEWorkload covered_unknown{unknown_op, covered_device, {input_0}, {output_0}};
    EXPECT_EQ(empty_model.getCodePrefetchCost(covered_unknown), V(Cycles::ERROR_SHAVE_PREFETCH_NOT_FOUND));
    SHAVEWorkload uncovered_unknown{unknown_op, VPUDevice::VPU_4_0, {input_0}, {output_0}};
    EXPECT_EQ(empty_model.getCodePrefetchCost(uncovered_unknown), V(Cycles::ERROR_SHAVE_PREFETCH_DEVICE_NOT_COVERED));
}

// Tests the single-device prefetch strategy in isolation from SHAVECostModel.
// Confirms explicit entry lookup works and missing entries return NOT_FOUND.
// Also confirms exact-name matching semantics (no alias remapping).
TEST_F(TestSHAVEPrefetch, SingleDevicePrefetchCostStrategy_LookupAndNotFound) {
    const std::string op{"Sigmoid"};
    const VPUDevice device = VPUDevice::NPU_5_0;

    OperationPrefetchLUT lut;
    lut[op] = static_cast<CyclesInterfaceType>(37);
    lut["Softmax"] = static_cast<CyclesInterfaceType>(23);

    auto measured = std::make_shared<SingleDevicePrefetchCostStrategy>(lut);

    SHAVEWorkload exact_wl{op, device, {input_0}, {output_0}};
    SHAVEWorkload missing_wl{"not_measured", device, {input_0}, {output_0}};
    SHAVEWorkload other_device_same_op{op, VPUDevice::VPU_2_7, {input_0}, {output_0}};
    SHAVEWorkload alias_wl{"SoftMax", device, {input_0}, {output_0}};

    EXPECT_EQ(measured->get_code_prefetch_cost(exact_wl), static_cast<CyclesInterfaceType>(37));
    EXPECT_EQ(measured->get_code_prefetch_cost(missing_wl), V(Cycles::ERROR_SHAVE_PREFETCH_NOT_FOUND));
    EXPECT_EQ(measured->get_code_prefetch_cost(other_device_same_op), static_cast<CyclesInterfaceType>(37));

    EXPECT_EQ(measured->get_code_prefetch_cost(alias_wl), V(Cycles::ERROR_SHAVE_PREFETCH_NOT_FOUND));
}

// Baseline behavior check: with include_code_prefetch disabled, computeCycles()
// must return the warm/base execution cost only.
// This ensures prefetch logic is opt-in and never applied implicitly.
TEST_F(TestSHAVEPrefetch, ComputeCycles_DefaultPrefetchFlagFalse_DoesNotAddPrefetch) {
    const VPUDevice device = VPUDevice::NPU_5_0;
    const auto ops = empty_model.getShaveSupportedOperations(device);

    std::optional<std::string> selected_op{};

    for (const auto& op : ops) {
        SHAVEWorkload probe{op, device, {input_0}, {output_0}};
        std::string info;
        const auto base_cycles = empty_model.computeCycles(probe, info);
        if (!Cycles::isErrorCode(base_cycles)) {
            selected_op = op;
            break;
        }
    }

    ASSERT_TRUE(selected_op.has_value()) << "No valid SHAVE operator found for NPU_5_0.";

    SHAVEWorkload swl{*selected_op, device, {input_0}, {output_0}};

    std::string info;
    const auto cycles = empty_model.computeCycles(swl, info);
    ASSERT_FALSE(Cycles::isErrorCode(cycles)) << info;

    const auto instance = empty_model.getShaveInstance(*selected_op, device);
    ASSERT_TRUE(instance.has_value());
    const auto expected_base = instance->get().dpuCycles(swl);
    EXPECT_EQ(cycles, expected_base);
}

// With include_code_prefetch enabled on a covered device, the model must add the measured
// prefetch term (which may be 0) and must not receive PREFETCH_NOT_FOUND for supported operations.
// This asserts strict prefetch coverage and additive behavior simultaneously.
TEST_F(TestSHAVEPrefetch, ComputeCycles_IncludePrefetchFlag_AddsMeasuredPrefetchForCoveredDevice) {
    const VPUDevice device = VPUDevice::NPU_5_0;
    const auto ops = empty_model.getShaveSupportedOperations(device);

    std::optional<std::string> selected_op{};
    CyclesInterfaceType selected_base{0};

    for (const auto& op : ops) {
        SHAVEWorkload probe{op, device, {input_0}, {output_0}};
        std::string info;
        const auto base_cycles = empty_model.computeCycles(probe, info);
        if (!Cycles::isErrorCode(base_cycles)) {
            selected_op = op;
            selected_base = base_cycles;
            break;
        }
    }

    ASSERT_TRUE(selected_op.has_value()) << "No valid SHAVE operator found for NPU_5_0.";

    SHAVEWorkload swl{*selected_op, device, {input_0}, {output_0}};
    swl.include_code_prefetch = true;

    std::string info;
    const auto with_prefetch = empty_model.computeCycles(swl, info);
    ASSERT_FALSE(Cycles::isErrorCode(with_prefetch)) << info;

    const auto expected_prefetch = empty_model.getCodePrefetchCost(swl);
    ASSERT_NE(expected_prefetch, V(Cycles::ERROR_SHAVE_PREFETCH_NOT_FOUND));
    ASSERT_FALSE(Cycles::isErrorCode(expected_prefetch));
    const auto expected_cycles = selected_base + expected_prefetch;
    EXPECT_EQ(with_prefetch, expected_cycles) << "Prefetch-enabled execution must either add the reported prefetch "
                                                 "cost.";
}

// Verifies prefetch application semantics on cache-hit path specifically.
// Even when base cycles come from cache, prefetch handling must remain identical:
// add measured prefetch (which may be 0) and never yield PREFETCH_NOT_FOUND on covered ops.
TEST_F(TestSHAVEPrefetch, ComputeCycles_IncludePrefetchFlag_CacheHitStillAddsPrefetch) {
    const VPUDevice device = VPUDevice::NPU_5_0;
    const auto ops = empty_model.getShaveSupportedOperations(device);

    std::optional<std::string> selected_op{};
    CyclesInterfaceType selected_base{0};

    for (const auto& op : ops) {
        SHAVEWorkload probe{op, device, {input_0}, {output_0}};
        std::string info;
        const auto base_cycles = empty_model.computeCycles(probe, info);
        if (!Cycles::isErrorCode(base_cycles)) {
            selected_op = op;
            selected_base = base_cycles;
            break;
        }
    }

    ASSERT_TRUE(selected_op.has_value()) << "No valid SHAVE operator found for NPU_5_0.";

    const std::string temp_cache_file{"SHAVE_Prefetch_CacheHit.cache_bin"};
    const ScopedFileCleanup cleanup{temp_cache_file};
    std::error_code ec;
    std::filesystem::remove(temp_cache_file, ec);

    SHAVEWorkload base_wl{*selected_op, device, {input_0}, {output_0}};
    FixedCache cache_to_be{temp_cache_file};
    cache_to_be.insert(base_wl.hash(), static_cast<float>(selected_base));
    cache_to_be.write_cache(temp_cache_file);

    SHAVECostModel model_with_fixed_cache{temp_cache_file, 16384};

    SHAVEWorkload swl{*selected_op, device, {input_0}, {output_0}};
    swl.include_code_prefetch = true;

    std::string info;
    const auto with_prefetch = model_with_fixed_cache.computeCycles(swl, info);
    ASSERT_FALSE(Cycles::isErrorCode(with_prefetch)) << info;

    const auto expected_prefetch = model_with_fixed_cache.getCodePrefetchCost(swl);
    ASSERT_NE(expected_prefetch, V(Cycles::ERROR_SHAVE_PREFETCH_NOT_FOUND));
    ASSERT_FALSE(Cycles::isErrorCode(expected_prefetch));
    const auto expected_cycles = selected_base + expected_prefetch;
    EXPECT_EQ(with_prefetch, expected_cycles)
            << "Expected cache-hit path to add reported prefetch for op: " << *selected_op;
}

// With strict prefetch policy enabled in SHAVECostModel, unknown operators with include_code_prefetch=true
// must return ERROR_SHAVE_PREFETCH_NOT_FOUND if the warm/base path succeeds (forced here via cache hit).
// This confirms prefetch errors are propagated and not ignored.
TEST_F(TestSHAVEPrefetch, ComputeCycles_IncludePrefetchFlag_UnknownOperatorReturnsPrefetchNotFound) {
    const std::string unknown_op{"__prefetch_unknown_op_error_path"};
    const CyclesInterfaceType cached_base_cycles{static_cast<CyclesInterfaceType>(1234)};

    SHAVEWorkload cached_wl{
            unknown_op,
            VPUDevice::NPU_5_0,
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
            {{7}},
    };
    cached_wl.include_code_prefetch = true;

    EXPECT_EQ(empty_model.getCodePrefetchCost(cached_wl), V(Cycles::ERROR_SHAVE_PREFETCH_NOT_FOUND));

    const std::string temp_cache_file{"SHAVE_Prefetch_UnknownOp_StrictPolicy.cache_bin"};
    const ScopedFileCleanup cleanup{temp_cache_file};
    std::error_code ec;
    std::filesystem::remove(temp_cache_file, ec);

    FixedCache cache_to_be{temp_cache_file};
    cache_to_be.insert(cached_wl.hash(), static_cast<float>(cached_base_cycles));
    cache_to_be.write_cache(temp_cache_file);

    SHAVECostModel model_with_fixed_cache{temp_cache_file, 16384};

    std::string info;
    const auto cycles = model_with_fixed_cache.computeCycles(cached_wl, info);
    EXPECT_EQ(cycles, V(Cycles::ERROR_SHAVE_PREFETCH_NOT_FOUND));
}

// include_code_prefetch is intentionally metadata, not workload identity.
// Hash and ordering must remain unchanged between flag=false and flag=true workloads.
// Runtime delta is the additive measured prefetch term on covered devices.
TEST_F(TestSHAVEPrefetch, ComputeCycles_PrefetchFlag_IsMetadataNotWorkloadIdentity) {
    const VPUDevice device = VPUDevice::NPU_5_0;
    const auto ops = empty_model.getShaveSupportedOperations(device);

    std::optional<std::string> selected_op{};

    for (const auto& op : ops) {
        SHAVEWorkload probe{op, device, {input_0}, {output_0}};
        std::string info;
        const auto base_cycles = empty_model.computeCycles(probe, info);
        if (!Cycles::isErrorCode(base_cycles)) {
            selected_op = op;
            break;
        }
    }

    ASSERT_TRUE(selected_op.has_value()) << "No valid SHAVE operator found for NPU_5_0.";

    SHAVEWorkload without_prefetch{*selected_op, device, {input_0}, {output_0}};
    SHAVEWorkload with_prefetch{without_prefetch};
    with_prefetch.include_code_prefetch = true;

    EXPECT_EQ(without_prefetch.hash(), with_prefetch.hash());
    EXPECT_FALSE(without_prefetch < with_prefetch);
    EXPECT_FALSE(with_prefetch < without_prefetch);

    std::string info_base;
    const auto base_cycles = empty_model.computeCycles(without_prefetch, info_base);
    ASSERT_FALSE(Cycles::isErrorCode(base_cycles)) << info_base;

    std::string info_with;
    const auto with_cycles = empty_model.computeCycles(with_prefetch, info_with);
    ASSERT_FALSE(Cycles::isErrorCode(with_cycles)) << info_with;

    const auto prefetch = empty_model.getCodePrefetchCost(with_prefetch);
    ASSERT_NE(prefetch, V(Cycles::ERROR_SHAVE_PREFETCH_NOT_FOUND));
    ASSERT_FALSE(Cycles::isErrorCode(prefetch));
    const auto expected_cycles = base_cycles + prefetch;
    EXPECT_EQ(with_cycles, expected_cycles);
}

}  // namespace VPUNN_unit_tests