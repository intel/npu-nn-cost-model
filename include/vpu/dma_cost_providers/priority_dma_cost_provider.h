// Copyright © 2025 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#ifndef PRIORITY_DMA_COST_PROVIDER_H
#define PRIORITY_DMA_COST_PROVIDER_H

#include "dma_cost_provider_interface.h"
#include "dmann_adapter.h"
#include <memory>
#include <type_traits>

namespace VPUNN {
/**
 * @brief A list of cost providers in priority order
 * Providers are shared_ptr to allow co-ownership. If
 * there will be different ways to composite the Providers the
 * the references will be shared.
 * @tparam WlType Workload type
 */
template<typename WlType>
using DMACostProviderList = std::vector<std::shared_ptr<DMANNAdapter<WlType>>>;

/**
 * @brief Priority-based DMA cost provider. It can be initialized with a list of cost providers and queries the first available one 
 * depending on priority order.
 * @tparam WlT The workload type
 */
template <typename WlT>
class PriorityDMACostProvider : public IDMACostProvider<WlT> {
private:
    DMACostProviderList<WlT> cost_providers;  ///< List of cost providers in priority order
public:
    explicit PriorityDMACostProvider(const DMACostProviderList<WlT>& providers)
        : cost_providers(providers) {}

    /**
     * @brief Get the cost of a workload by querying providers in priority order
     * @param workload The workload to evaluate
     * @param cost_source Optional pointer to a string to store the source of the cost
     * @return The cost in cycles, or an error code if all providers fail
     */
    CyclesInterfaceType get_cost(const WlT& workload, std::string* cost_source = nullptr) const override {
        CyclesInterfaceType cycles = Cycles::ERROR_NO_VALID_DMA_COST_PROVIDER;

        for(const auto& prov_ref : cost_providers) {
            if (!prov_ref) continue;

            CyclesInterfaceType result = prov_ref->get_cost(workload, cost_source);

            // If the provider succeeded (no error), return the result
            if (!Cycles::isErrorCode(result)) {
                return result;
            }
        }

        // All providers failed, return the last error
        if(cost_source) {
            *cost_source = "unknown";
        }
        return cycles;
    }

    /**
     * @brief Check if any of the underlying providers is initialized. We come with assumption
     * that only DMANN based providers will have initialization status. So for theoretical providers
     * this should return false, that's why this method specifically checks if nn is initialized.
     * @return true if at least one provider is initialized, false otherwise
     */
    bool has_nn_initialized() const {
        for(const auto& prov_ref : cost_providers) {
            if (prov_ref && prov_ref->is_initialized()) {
                return true;
            }
        }
        return false;
    }
};
    
}

#endif
