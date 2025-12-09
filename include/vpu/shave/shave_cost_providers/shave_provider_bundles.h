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
#include <vector>
#include <memory>

namespace VPUNN {
/**
 * @brief Utility class for creating precomposed SHAVE cost provider bundles
 * 
 * This class provides static factory methods for creating different combinations
 * of SHAVE cost providers without requiring class instantiation.
 */
class ShaveCostProviderBundles {
public:
    /**
     * @brief Create a prebuilt list of SHAVE cost providers with default priority order
     * 
     * Creates a priority list with:
     * - Priority 0: ShaveCostProvider (with cache and Shave2 API)
     * - Priority 1: OldShaveCostProvider (fallback to Shave1 API)
     * 
     * @return ShaveCostProviderList The prebuilt provider list
     */
    static ShaveCostProviderList createDefaultShaveCostProviders() {
        ShaveCostProviderList providers;
        
        // Priority 0: ShaveCostProvider (with cache and Shave2 API)
        providers.push_back(std::make_shared<ShaveCostProvider>());
        
        // Priority 1: OldShaveCostProvider (fallback to Shave1 API)
        providers.push_back(std::make_shared<OldShaveCostProvider>());
        
        return providers;
    }
    
    /**
     * @brief Create a bundle with only the new SHAVE cost provider
     * 
     * @return ShaveCostProviderList Provider list with only ShaveCostProvider
     */
    static ShaveCostProviderList createNewShaveOnlyProviders() {
        ShaveCostProviderList providers;
        providers.push_back(std::make_shared<ShaveCostProvider>());
        return providers;
    }

    /**
     * @brief Create a bundle with only the old SHAVE cost provider
     * 
     * @return ShaveCostProviderList Provider list with only OldShaveCostProvider
     */
    static ShaveCostProviderList createOldShaveOnlyProviders() {
        ShaveCostProviderList providers;
        providers.push_back(std::make_shared<OldShaveCostProvider>());
        return providers;
    }
};

}  // namespace VPUNN
#endif  // SHAVE_PROVIDER_BUNDLES_H
