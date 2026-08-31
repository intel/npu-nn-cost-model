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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "vpu/shave/generated_shave_prefetch_cost_population.h"
#include "vpu/shave/shave_cost_providers/composite_shave_cost_provider.h"
#include "vpu/shave/shave_cost_providers/device_mapping_shave_cost_provider.h"
#include "vpu/shave/shave_cost_providers/name_mapping_shave_cost_provider.h"
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

    /// @brief  Creates the default route provider for devices that do not have a specific provider defined.
    /// @return std::shared_ptr<IShaveCostProvider> The default route provider instance
    static std::shared_ptr<IShaveCostProvider> createDefaultRouteProvider() {
        return createOldSelectorWithNameMappingProvider();
    }

    /// @brief Creates the provider used for one device in the device-mapped composition.
    static std::shared_ptr<IShaveCostProvider> createProviderForDevice(VPUDevice device) {
        switch (device) {
        case VPUDevice::NPU_5_0:
            return createCompositeBasedOnHeuristicWithOldNameMappingExtendedOps();
        case VPUDevice::NPU_5_0_W:
            return createCompositeBasedOnHeuristicWithOldNameMappingExtendedOps();

        default:
            return createDefaultRouteProvider();
        }
    }

    /// @brief Creates the prefetch strategy used for one device in the device-mapped composition.
    static SingleDevicePrefetchCostStrategy createPrefetchStrategyForDevice(VPUDevice device) {
        return SingleDevicePrefetchCostStrategy(PopulatedShavePrefetchCostLUT::get_lut_for_device(device));
    }

    /// @brief Helper function to create a composite provider with name mapping for specific operations
    /// @param ops List of operation names to route through the name mapping provider
    /// @return std::shared_ptr<IShaveCostProvider> The configured composite provider
    static std::shared_ptr<IShaveCostProvider> createCompositeWithNameMappingForOps(
            std::initializer_list<const char*> ops) {
        const auto base_provider = createHeuristicWithFactorsOnlyProvider();
        const auto name_mapping_provider = createNameMappingOldProvider();

        std::unordered_map<std::string, std::shared_ptr<IShaveCostProvider>> op_to_provider_map;
        op_to_provider_map.reserve(ops.size());
        for (const auto* op : ops) {
            op_to_provider_map.emplace(op, name_mapping_provider);
        }

        return std::make_shared<CompositeShaveCostProvider>(base_provider, std::move(op_to_provider_map));
    }

public:
    /**
     * @brief Create the default SHAVE cost provider with priority-based fallback
     *
     * Creates a PriorityShaveCostProvider with:
     * - Priority 0: ShaveCostProvider (with cache and Shave2 API)
     * - Priority 1: HeuristicCostProviderWithFactors (fallback with heuristic calculations)
     *
     * @return std::shared_ptr<IShaveCostProvider> The configured provider
     */
    static std::shared_ptr<IShaveCostProvider> createDefaultProvider() {
        ShaveCostProviderList providers;
        providers.push_back(std::make_shared<ShaveCostProvider>());
        providers.push_back(std::make_shared<HeuristicCostProviderWithFactors>());
        return std::make_shared<PriorityShaveCostProvider>(std::move(providers));
    }

    /**
     * @brief Create a provider using only the new SHAVE API (Shave2)
     *
     * @return std::shared_ptr<IShaveCostProvider> ShaveCostProvider instance
     */
    static std::shared_ptr<IShaveCostProvider> createNewShaveOnlyProvider() {
        return std::make_shared<ShaveCostProvider>();
    }

    /**
     * @brief Create a provider using only the old SHAVE API (Shave1)
     *
     * @return std::shared_ptr<IShaveCostProvider> OldShaveCostProvider instance
     */
    static std::shared_ptr<IShaveCostProvider> createOldShaveOnlyProvider() {
        return std::make_shared<OldShaveCostProvider>();
    }

    /**
     * @brief Create a provider using only heuristic cost calculations
     *
     * @return std::shared_ptr<IShaveCostProvider> HeuristicCostProvider instance
     */
    static std::shared_ptr<IShaveCostProvider> createHeuristicOnlyProvider() {
        return std::make_shared<HeuristicCostProvider>();
    }

    /**
     * @brief Create a provider using heuristic cost calculations with correction factors
     *
     * @return std::shared_ptr<IShaveCostProvider> HeuristicCostProviderWithFactors instance
     */
    static std::shared_ptr<IShaveCostProvider> createHeuristicWithFactorsOnlyProvider() {
        return std::make_shared<HeuristicCostProviderWithFactors>();
    }

    /**
     * @brief Create a name mapping provider using the old SHAVE API
     *
     * @return std::shared_ptr<IShaveCostProvider> Name mapping provider
     */
    static std::shared_ptr<IShaveCostProvider> createNameMappingOldProvider() {
        return std::make_shared<NameMappingShaveCostProvider>(std::make_shared<OldShaveCostProvider>(),
                                                              getStandardNameMappings());
    }

    /**
     * @brief Create a composite provider with name mapping for legacy operations
     *
     * This creates a CompositeShaveCostProvider that:
     * - Uses HeuristicCostProviderWithFactors as the default
     * - Applies name mapping (e.g., "Elu" → "ELU") for specific operations
     * - Routes certain operations (DepthToSpace, MVN, SoftMax) through the name mapping provider
     *
     * @return std::shared_ptr<IShaveCostProvider> Configured composite provider
     */
    static std::shared_ptr<IShaveCostProvider> createCompositeBasedOnHeuristicWithOldNameMappingCoreOps() {
        return createCompositeWithNameMappingForOps({"DepthToSpace", "MVN", "SoftMax"});
    }

    static std::shared_ptr<IShaveCostProvider> createCompositeBasedOnHeuristicWithOldNameMappingExtendedOps() {
        return createCompositeWithNameMappingForOps(
                {"DepthToSpace", "MVN", "SoftMax", "Gelu", "Multiply", "Cos", "Sin"});
    }

    /**
     * @brief Create a provider using old selector with name mapping and priority fallback
     *
     * Creates a PriorityShaveCostProvider with:
     * - Priority 0: OldShaveCostProvider (base)
     * - Priority 1: NameMappingShaveCostProvider (with legacy name translations)
     *
     * @return std::shared_ptr<IShaveCostProvider> Priority provider with name mapping
     */
    static std::shared_ptr<IShaveCostProvider> createOldSelectorWithNameMappingProvider() {
        ShaveCostProviderList providers;
        providers.push_back(createOldShaveOnlyProvider());
        providers.push_back(createNameMappingOldProvider());
        return std::make_shared<PriorityShaveCostProvider>(std::move(providers));
    }

    /**
     * @brief Create a device-specific provider with different implementations per device
     *
     * This creates a DeviceMappingShaveCostProvider that:
     * - Uses composite with name mapping for NPU 5.0, 5.0_W, and 6.0
     * - Falls back to old selector with name mapping for other devices
     *
     * @return std::shared_ptr<IShaveCostProvider> Device-mapped provider
     */
    static std::shared_ptr<IShaveCostProvider> createDeviceMappedProvider() {
        // Provider for newer devices (NPU 5.0+)
        const auto default_provider = createDefaultRouteProvider();
        const auto new_device_provider_npu5 = createProviderForDevice(VPUDevice::NPU_5_0);

        // Map devices to specific providers
        std::unordered_map<VPUDevice, std::shared_ptr<IShaveCostProvider>> device_to_provider_map = {
                {VPUDevice::VPU_2_0, default_provider},           {VPUDevice::VPU_2_7, default_provider},
                {VPUDevice::VPU_4_0, default_provider},           {VPUDevice::NPU_5_0, new_device_provider_npu5},
                {VPUDevice::NPU_5_0_W, new_device_provider_npu5},
        };

        return std::make_shared<DeviceMappingShaveCostProvider>(default_provider, device_to_provider_map);
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
        const auto provider = createProviderForDevice(device);
        return provider->get_shave_supported_ops(device);
    }
};
}  // namespace VPUNN
#endif  // VPUNN_SHAVE_PROVIDER_BUNDLES_H
