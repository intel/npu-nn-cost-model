// Copyright © 2025 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#ifndef SHAVE_PROVIDER_BUNDLES_H
#define SHAVE_PROVIDER_BUNDLES_H

#include <vpu/shave/shave_cost_providers/shave_cost_provider_interface.h>
#include <vpu/shave/shave_cost_providers/shave_cost_providers.h>
#include <vpu/shave/shave_cost_providers/priority_shave_cost_provider.h>
#include <vpu/shave/shave_cost_providers/composite_shave_cost_provider.h>
#include <vpu/shave/shave_cost_providers/name_mapping_shave_cost_provider.h>
#include <vpu/shave/shave_cost_providers/device_mapping_shave_cost_provider.h>
#include <vpu/types.h>
#include <vector>
#include <memory>
#include <unordered_map>
#include <string>

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

    // Delete the destructor because this class exposes only a bunch of static functions and forbid any attempt to create an instance of that class
    ~ShaveCostProviderBundles() = delete;

    /**
     * @brief Get standard name mappings for legacy operation names
     * 
     * @return std::unordered_map<std::string, std::string> Name mapping dictionary
     */
    static std::unordered_map<std::string, std::string> getStandardNameMappings() {
        return {
            {"Elu", "ELU"},
            {"Quantize", "QuantizeCast"},
            {"SpaceToDepth", "SpaceToDepthOp"},
            {"SoftMax", "Softmax"},
            {"SquaredDifference", "SquaredDiff"}  
        };
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
        return std::make_shared<NameMappingShaveCostProvider>(
            std::make_shared<OldShaveCostProvider>(), 
            getStandardNameMappings()
        );
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
    static std::shared_ptr<IShaveCostProvider> createCompositeBasedOnHeuristicWithOldNameMappingProviderNPU_RESERVED_1() {
        const auto base_provider = createHeuristicWithFactorsOnlyProvider();
        const auto name_mapping_provider = createNameMappingOldProvider();

        // Map specific operations to use the name mapping provider
        const std::unordered_map<std::string, std::shared_ptr<IShaveCostProvider>> op_to_provider_map = {
            {"DepthToSpace", name_mapping_provider},
            {"MVN", name_mapping_provider},
            {"SoftMax", name_mapping_provider}
        };

        return std::make_shared<CompositeShaveCostProvider>(
            base_provider, 
            op_to_provider_map
        );
    }

    static std::shared_ptr<IShaveCostProvider> createCompositeBasedOnHeuristicWithOldNameMappingProviderNPU5() {
        const auto base_provider = createHeuristicWithFactorsOnlyProvider();
        const auto name_mapping_provider = createNameMappingOldProvider();

        // Map specific operations to use the name mapping provider
        const std::unordered_map<std::string, std::shared_ptr<IShaveCostProvider>> op_to_provider_map = {
            {"DepthToSpace", name_mapping_provider},
            {"MVN", name_mapping_provider},
            {"SoftMax", name_mapping_provider},
            {"Gelu", name_mapping_provider},
            {"Multiply", name_mapping_provider},
            {"Cos", name_mapping_provider},
            {"Sin", name_mapping_provider}
        };

        return std::make_shared<CompositeShaveCostProvider>(
            base_provider, 
            op_to_provider_map
        );
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
     * - Uses composite with name mapping for NPU 5.0, RESERVED, and 6.0
     * - Falls back to old selector with name mapping for other devices
     * 
     * @return std::shared_ptr<IShaveCostProvider> Device-mapped provider
     */
    static std::shared_ptr<IShaveCostProvider> createDeviceMappedProvider() {
        // Provider for newer devices (NPU 5.0+)
        const auto new_device_provider_npu5 = createCompositeBasedOnHeuristicWithOldNameMappingProviderNPU5();
        const auto new_device_provider_NPU_RESERVED_1 = createCompositeBasedOnHeuristicWithOldNameMappingProviderNPU_RESERVED_1();
        // Default provider for older devices
        const auto default_provider = createOldSelectorWithNameMappingProvider();

        // Map devices to specific providers
        std::unordered_map<VPUDevice, std::shared_ptr<IShaveCostProvider>> device_to_provider_map = {
            {VPUDevice::VPU_2_0, default_provider},
            {VPUDevice::VPU_2_7, default_provider},
            {VPUDevice::VPU_4_0, default_provider},
            {VPUDevice::NPU_5_0, new_device_provider_npu5},
        };

        return std::make_shared<DeviceMappingShaveCostProvider>(
            default_provider, 
            device_to_provider_map
        );
    }
};
}  // namespace VPUNN
#endif  // SHAVE_PROVIDER_BUNDLES_H
