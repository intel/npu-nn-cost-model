// Copyright © 2025 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#ifndef VPUNN_SHAVE_PROVIDER_BUNDLES_H
#define VPUNN_SHAVE_PROVIDER_BUNDLES_H

#include <initializer_list>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "vpu/performance.h"
#include "vpu/shave/generated_shave_prefetch_cost_population.h"
#include "vpu/shave/shave_cost_providers/composite_shave_cost_provider.h"
#include "vpu/shave/shave_cost_providers/device_mapping_shave_cost_provider.h"
#include "vpu/shave/shave_cost_providers/legacy_shv_compat_provider.h"
#include "vpu/shave/shave_cost_providers/name_mapping_shave_cost_provider.h"
#include "vpu/shave/shave_cost_providers/op_set_override_shave_cost_provider.h"
#include "vpu/shave/shave_cost_providers/priority_shave_cost_provider.h"
#include "vpu/shave/shave_cost_providers/shave_cost_provider_interface.h"
#include "vpu/shave/shave_cost_providers/shave_cost_providers.h"
#include "vpu/shave/shave_prefetch_cost_strategy.h"
#include "vpu/types.h"

namespace VPUNN {
/**
 * @brief Utility class for creating precomposed SHAVE cost provider bundles
 *
 * This class provides static factory methods for creating different combinations
 * of SHAVE cost providers without requiring class instantiation. All methods
 * return a single IShaveCostProvider pointer, creating PriorityShaveCostProvider
 * internally when multiple providers are needed.
 */
class ShaveCostProviderBundles {
private:
    // Delete constructors to prevent instantiation
    ShaveCostProviderBundles() = delete;
    ShaveCostProviderBundles(const ShaveCostProviderBundles&) = delete;
    ShaveCostProviderBundles& operator=(const ShaveCostProviderBundles&) = delete;

    ShaveCostProviderBundles(ShaveCostProviderBundles&&) = delete;
    ShaveCostProviderBundles& operator=(ShaveCostProviderBundles&&) = delete;

    // Delete the destructor because this class exposes only a bunch of static functions and forbid any attempt to
    // create an instance of that class
    ~ShaveCostProviderBundles() = delete;

    /**
     * @brief Get standard name mappings for legacy operation names
     *
     * @return std::unordered_map<std::string, std::string> Name mapping dictionary
     */
    static std::unordered_map<std::string, std::string> getStandardNameMappings() {
        return {{"Elu", "ELU"},
                {"Quantize", "QuantizeCast"},
                {"SpaceToDepth", "SpaceToDepthOp"},
                {"SoftMax", "Softmax"},
                {"SquaredDifference", "SquaredDiff"}};
    }

    /// @brief Creates the legacy route provider used for explicitly mapped legacy devices.
    /// @return Shared_IShaveCostProvider The legacy route provider instance
    static Shared_IShaveCostProvider createLegacyRouteProvider() {
        return createOldSelectorWithNameMappingProvider();
    }

    /// @brief Creates the provider for one device in the device-mapped composition.
    static Shared_IShaveCostProvider createProviderForDevice(VPUDevice device) {
        switch (device) {
        case VPUDevice::VPU_2_0:
        case VPUDevice::VPU_2_1:
            return createLegacyRouteProvider();
        case VPUDevice::VPU_2_7:
            return createLegacyRouteProvider();
        case VPUDevice::VPU_4_0:
            return createLegacyRouteProvider();
        case VPUDevice::NPU_5_0:
            // Providing frequencies enables the legacy compatibility wrapper, converting
            // heuristic DPU-cycle outputs back to SHV-like cycles for this path.
            return createHeuristicWithOldNameMappingExtendedOpsOverride(
                    std::make_pair(GlobalHarwdwareCharacteristics::get_dpu_fclk(device),
                                   GlobalHarwdwareCharacteristics::get_cmx_fclk(device)));
        case VPUDevice::NPU_5_0_W:
            // no need to use wrapper here, since WCL uses old selector
            return createHeuristicWithOldNameMappingExtendedOpsOverride();
        default: {
            const auto key = static_cast<int>(device);
            const auto it = VPUDevice_ToText.find(key);
            const std::string dev =
                    (it != VPUDevice_ToText.end()) ? it->second : ("UNKNOWN(" + std::to_string(key) + ")");
            throw std::invalid_argument("Unsupported device for SHAVE cost provider: " + dev);
        }
        }
    }

    /// @brief Creates the prefetch strategy used for one device in the device-mapped composition.
    static SingleDevicePrefetchCostStrategy createPrefetchStrategyForDevice(VPUDevice device) {
        return SingleDevicePrefetchCostStrategy(PopulatedShavePrefetchCostLUT::get_lut_for_device(device));
    }

    /// @brief Helper function to create an op-set override provider with name mapping for specific operations
    /// @param ops List of operation names to route through the name mapping provider
    /// @param frequencies Optional DPU/SHV frequency pair, that when provided, the heuristic base provider is wrapped
    ///        with the legacy compatibility adapter that converts DPU cycle outputs back to SHV cycles.
    /// @return Shared_IShaveCostProvider The configured provider
    static Shared_IShaveCostProvider createOpSetOverrideWithNameMappingForOps(
            std::initializer_list<const char*> ops,
            std::optional<std::pair<unsigned int, unsigned int>> frequencies = std::nullopt) {
        const auto base_provider = frequencies.has_value()
                                           ? createLegacyShvCompatibilityWrapper(
                                                     createHeuristicWithFactorsOnlyProvider(), frequencies.value())
                                           : createHeuristicWithFactorsOnlyProvider();
        const auto name_mapping_provider = createNameMappingOldProvider();

        std::set<std::string> override_operations;
        for (const auto* op : ops) {
            override_operations.emplace(op);
        }

        return std::make_shared<OpSetOverrideShaveCostProvider>(base_provider, name_mapping_provider,
                                                                std::move(override_operations));
    }

    /// @brief Factory for legacy SHV-like compatibility behavior.
    static Shared_IShaveCostProvider createLegacyShvCompatibilityWrapper(
            const Shared_IShaveCostProvider& provider, std::pair<unsigned int, unsigned int> frequencies) {
        return std::make_shared<LegacyShvCompatProvider>(provider, frequencies);
    }

public:
    /**
     * @brief Create the default SHAVE cost provider with priority-based fallback
     *
     * Creates a PriorityShaveCostProvider with:
     * - Priority 0: ShaveCostProvider (with cache and Shave2 API)
     * - Priority 1: HeuristicCostProviderWithFactors (fallback with heuristic calculations)
     *
     * @return Shared_IShaveCostProvider The configured provider
     */
    static Shared_IShaveCostProvider createDefaultProvider() {
        ShaveCostProviderList providers;
        providers.push_back(std::make_shared<ShaveCostProvider>());
        providers.push_back(std::make_shared<HeuristicCostProviderWithFactors>());
        return std::make_shared<PriorityShaveCostProvider>(std::move(providers));
    }

    /**
     * @brief Create a provider using only the new SHAVE API (Shave2)
     *
     * @return Shared_IShaveCostProvider ShaveCostProvider instance
     */
    static Shared_IShaveCostProvider createNewShaveOnlyProvider() {
        return std::make_shared<ShaveCostProvider>();
    }

    /**
     * @brief Create a provider using only the old SHAVE API (Shave1)
     *
     * @return Shared_IShaveCostProvider OldShaveCostProvider instance
     */
    static Shared_IShaveCostProvider createOldShaveOnlyProvider() {
        return std::make_shared<OldShaveCostProvider>();
    }

    /**
     * @brief Create a provider using only heuristic cost calculations
     *
     * @return Shared_IShaveCostProvider HeuristicCostProvider instance
     */
    static Shared_IShaveCostProvider createHeuristicOnlyProvider() {
        return std::make_shared<HeuristicCostProvider>();
    }

    /**
     * @brief Create a provider using heuristic cost calculations with correction factors
     *
     * @return Shared_IShaveCostProvider HeuristicCostProviderWithFactors instance
     */
    static Shared_IShaveCostProvider createHeuristicWithFactorsOnlyProvider() {
        return std::make_shared<HeuristicCostProviderWithFactors>();
    }

    /**
     * @brief Create a name mapping provider using the old SHAVE API
     *
     * @return Shared_IShaveCostProvider Name mapping provider
     */
    static Shared_IShaveCostProvider createNameMappingOldProvider() {
        return std::make_shared<NameMappingShaveCostProvider>(std::make_shared<OldShaveCostProvider>(),
                                                              getStandardNameMappings());
    }

    /**
     * @brief Create a heuristic provider with legacy-name op-set override for core operations
     *
     * This creates an OpSetOverrideShaveCostProvider that:
     * - Uses HeuristicCostProviderWithFactors as the base provider
     * - Uses NameMappingShaveCostProvider(OldShaveCostProvider) as the override provider
     * - Routes only {DepthToSpace, MVN, SoftMax} through the override provider
     * - Optionally wraps the base with LegacyShvCompatProvider when a frequency pair is provided
     *
     * @param frequencies Optional DPU/SHV frequency pair that enables the legacy compatibility wrapper
     * @return Shared_IShaveCostProvider Configured op-set override provider
     */
    static Shared_IShaveCostProvider createHeuristicWithOldNameMappingCoreOpsOverride(
            std::optional<std::pair<unsigned int, unsigned int>> frequencies = std::nullopt) {
        return createOpSetOverrideWithNameMappingForOps({"DepthToSpace", "MVN", "SoftMax"}, frequencies);
    }

    /// @brief Same as the core-op override variant, but with the extended operation set.
    /// @param frequencies Optional DPU/SHV frequency pair that enables the legacy compatibility wrapper.
    static Shared_IShaveCostProvider createHeuristicWithOldNameMappingExtendedOpsOverride(
            std::optional<std::pair<unsigned int, unsigned int>> frequencies = std::nullopt) {
        return createOpSetOverrideWithNameMappingForOps(
                {"DepthToSpace", "MVN", "SoftMax", "Gelu", "Multiply", "Cos", "Sin"}, frequencies);
    }

    /**
     * @brief Create a provider using old selector with name mapping and priority fallback
     *
     * Creates a PriorityShaveCostProvider with:
     * - Priority 0: OldShaveCostProvider (base)
     * - Priority 1: NameMappingShaveCostProvider (with legacy name translations)
     *
     * @return Shared_IShaveCostProvider Priority provider with name mapping
     */
    static Shared_IShaveCostProvider createOldSelectorWithNameMappingProvider() {
        ShaveCostProviderList providers;
        providers.push_back(createOldShaveOnlyProvider());
        providers.push_back(createNameMappingOldProvider());
        return std::make_shared<PriorityShaveCostProvider>(std::move(providers));
    }

    /**
     * @brief Create a device-specific provider with different implementations per device
     *
     * This creates a DeviceMappingShaveCostProvider that:
     * - Uses heuristic-with-factors as the base provider for NPU 5.0 and newer devices, with a
     *   name-mapped old selector applied only for a fixed op-set override; the overridden op set
     *   differs per generation
     * - Uses legacy priority provider (old selector + name mapping fallback) for legacy devices
     * - Uses strict map lookup; unmapped devices are not routed
     *
     * @return Shared_IShaveCostProvider Device-mapped provider
     */
    static Shared_IShaveCostProvider createDeviceMappedProvider() {
        // If a new device is added to the map below but its corresponding case is
        // missing in createProviderForDevice, construction fails loudly here.
        try {
            const auto legacy_provider = createLegacyRouteProvider();
            const auto new_device_provider_npu5 = createProviderForDevice(VPUDevice::NPU_5_0);

            // Map devices to specific providers
            const std::unordered_map<VPUDevice, Shared_IShaveCostProvider> device_to_provider_map = {
                    {VPUDevice::VPU_2_0, legacy_provider},
                    {VPUDevice::VPU_2_1, legacy_provider},
                    {VPUDevice::VPU_2_7, legacy_provider},
                    {VPUDevice::VPU_4_0, legacy_provider},
                    {VPUDevice::NPU_5_0, new_device_provider_npu5},
                    {VPUDevice::NPU_5_0_W, new_device_provider_npu5},
            };

            return std::make_shared<DeviceMappingShaveCostProvider>(device_to_provider_map);
        } catch (const std::invalid_argument& e) {
            throw std::logic_error("Failed to create SHAVE device-mapped provider: " + std::string(e.what()));
        }
    }

    /// @brief Create a device-specific prefetch strategy with different implementations per device.
    static DeviceMappedPrefetchCostStrategy createDeviceMappedPrefetchStrategy() {
        // the objects in the map are just smart wrapper on top of teh static LUT specific for each device
        static const DeviceMappedPrefetchCostStrategy::DeviceToStrategyMap device_to_strategy_map = {
                {VPUDevice::NPU_5_0, createPrefetchStrategyForDevice(VPUDevice::NPU_5_0)},
                {VPUDevice::NPU_5_0_W, createPrefetchStrategyForDevice(VPUDevice::NPU_5_0_W)},
        };

        return DeviceMappedPrefetchCostStrategy{
                device_to_strategy_map};  // creates an instance that references the static map created above.
    }

    /**
     * @brief Lightweight capability query that does not require cost model construction.
     *
     * Returns supported SHAVE operation names for the requested device using
     * the device-mapped provider composition (matching default VPUCostModel behavior).
     *
     * @param device Device to query
     * @return std::vector<std::string> Supported operation names
     */
    static std::vector<std::string> queryDeviceMappedSupportedOperations(VPUDevice device) {
        try {
            const auto provider = createProviderForDevice(device);
            return provider->get_shave_supported_ops(device);
        } catch (const std::invalid_argument&) {
            // If the device is not mapped, return an empty list
            return {};
        }
    }
};
}  // namespace VPUNN
#endif  // VPUNN_SHAVE_PROVIDER_BUNDLES_H
