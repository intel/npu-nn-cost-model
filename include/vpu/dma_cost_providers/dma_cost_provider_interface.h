// Copyright © 2025 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#ifndef DMA_COST_PROVIDER_INTERFACE_H
#define DMA_COST_PROVIDER_INTERFACE_H

#include "vpu/dma_types.h"

namespace VPUNN {
/**
 * @brief Interface for DMA cost providers
 *
 * This interface is used by classes that provide cost estimation
 * for DMA workloads. Implementations of this interface should provide
 * specific algorithms for calculating the execution cost of
 * DMA operations.
 */

template <typename WlT>
class IDMACostProvider {
public:
    virtual ~IDMACostProvider() = default;
    /**
     * @brief Calculate the cost of a given DMA workload
     * @param workload The DMA workload to evaluate
     * @param cost_source Optional pointer to a string to store the source of the cost
     * @return The cost in cycles, or an error code if the cost cannot be determined
     */
    virtual CyclesInterfaceType get_cost(const WlT& workload, std::string* cost_source) const = 0;

    /**
     * @brief Check if the provider is initialized, this method should be overridden only by
     * DMANN classes, which have a model to be loaded, so there is no point in having it in theoretical cost providers,
     * so for theoretical cost provider it will always return false, because it has no model handling.
     * @return true if the provider is initialized, false otherwise
     */
    virtual bool is_initialized() const {
        return false;
    }
};
}

#endif
