// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.
#ifndef VPUNN_SHAVE_COST_PROVIDERS_H
#define VPUNN_SHAVE_COST_PROVIDERS_H

#include <vpu/shave/shave_cost_providers/mathematical_shave_cost_provider_base.h>

namespace VPUNN {

/**
 * @brief SHAVE cost provider using Shave2 API (newer version)
 *
 * This class provides cost estimation for SHAVE workloads using the Shave2 API,
 * which includes the latest operator implementations and optimizations.
 * It inherits common functionality from MathematicalShaveCostProviderBase and
 * implements selector retrieval specific to Shave2.
 */
class ShaveCostProvider : public MathematicalShaveCostProviderBase<ShaveCostProvider> {
    friend class MathematicalShaveCostProviderBase<ShaveCostProvider>;

    const ShaveSelector& getSelectorImpl(VPUDevice& device) const {
        return getShaveConfigurator().getSelector(device);
    }

    static constexpr std::string_view cost_source_name = "shave_2";
};

/**
 * @brief SHAVE cost provider using Shave1 API (legacy version)
 *
 * This class provides cost estimation for SHAVE workloads using the legacy Shave1 API,
 * maintained for backward compatibility with older operator implementations.
 */
class OldShaveCostProvider : public MathematicalShaveCostProviderBase<OldShaveCostProvider> {
    friend class MathematicalShaveCostProviderBase<OldShaveCostProvider>;

    const ShaveSelector& getSelectorImpl(VPUDevice& device) const {
        return getShaveConfigurator().getOldSelector(device);
    }

    static constexpr std::string_view cost_source_name = "shave_1";
};

/**
 * @brief SHAVE cost provider using heuristic models for certain operations
 *
 * This class provides cost estimation for SHAVE workloads using simple heuristic models,
 * particularly for operations that are inherently non-vectorizable. It leverages
 * predefined heuristic models to estimate execution costs.
 */
class HeuristicCostProvider : public MathematicalShaveCostProviderBase<HeuristicCostProvider> {
    friend class MathematicalShaveCostProviderBase<HeuristicCostProvider>;

    const ShaveSelector& getSelectorImpl(VPUDevice& device) const {
        return getShaveConfigurator().getHeuristicSelector(device);
    }

    static constexpr std::string_view cost_source_name = "shave_heuristic";
};

class HeuristicCostProviderWithFactors : public MathematicalShaveCostProviderBase<HeuristicCostProviderWithFactors> {
    friend class MathematicalShaveCostProviderBase<HeuristicCostProviderWithFactors>;

    const ShaveSelector& getSelectorImpl(VPUDevice& device) const {
        return getShaveConfigurator().getHeuristicSelectorWithFactors(device);
    }

    static constexpr std::string_view cost_source_name = "shave_heuristic_with_factors";
};

}  // namespace VPUNN
#endif  // VPUNN_SHAVE_COST_PROVIDERS_H
