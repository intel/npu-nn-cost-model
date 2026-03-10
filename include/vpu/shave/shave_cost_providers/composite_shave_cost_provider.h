// Copyright © 2025 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.
#ifndef COMPOSITE_SHAVE_COST_PROVIDER_H
#define COMPOSITE_SHAVE_COST_PROVIDER_H

#include <vpu/shave/shave_cost_providers/shave_cost_provider_interface.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <set>
#include <optional>
#include <functional>

namespace VPUNN {

/**
 * @brief Composite SHAVE cost provider that supports a default provider and operation-specific overrides
 * 
 * This class implements a flexible cost provider mechanism where:
 * - A default provider handles all operations unless overridden
 * - Specific operations can be mapped to alternative providers
 * - Multiple operations can share the same provider (by referencing the same shared_ptr)
 * 
 * ATTENTION: This Composite provider will not add new operations. It will simply override existing ones form the
 * default provider.
 * 
 * This allows for fine-grained control over which provider handles which operation,
 * enabling hybrid approaches where some operations use mathematical models while
 * others use neural network models in the.
 */
class CompositeShaveCostProvider : public IShaveCostProvider {
private:
    const std::shared_ptr<IShaveCostProvider> default_provider;  ///< Default provider for all operations
    const std::unordered_map<std::string, std::shared_ptr<IShaveCostProvider>> operation_to_provider;  ///< Maps operation names to specific providers

public:
    /**
     * @brief Construct a new Composite Shave Cost Provider with just a default provider
     * 
     * @param default_provider Shared pointer to the default IShaveCostProvider instance
     */
    explicit CompositeShaveCostProvider(std::shared_ptr<IShaveCostProvider> default_provider)
        : default_provider(std::move(default_provider)) {
        if (!this->default_provider) {
            throw std::invalid_argument("Default provider cannot be null");
        }
    }

    /**
     * @brief Construct a new Composite Shave Cost Provider with operation-specific overrides
     * 
     * @param default_provider Shared pointer to the default provider
     * @param op_to_provider_map Map of operation names to specific provider instances
     */
    CompositeShaveCostProvider(
        std::shared_ptr<IShaveCostProvider> default_provider,
        std::unordered_map<std::string, std::shared_ptr<IShaveCostProvider>> op_to_provider_map = {})
        : default_provider(std::move(default_provider)),
          operation_to_provider(std::move(op_to_provider_map)) {
        
        if (!this->default_provider) {
            throw std::invalid_argument("Default provider cannot be null");
        }
    }

    /**
     * @brief Calculate the execution cost for a SHAVE workload using composite provider strategy
     * 
     * This function checks if there's a specific provider mapped for the operation.
     * If found, uses that provider; otherwise, uses the default provider.
     * 
     * @param workload The SHAVE workload for which to calculate the cost
     * @param cost_source Optional pointer to store the source of the cost
     * @return CyclesInterfaceType The estimated execution cost/cycles for the workload
     */
    CyclesInterfaceType get_cost(const SHAVEWorkload& workload, std::string* cost_source = nullptr) const override {
        // Get the operation name from the workload
        const std::string operation_name = workload.get_name();
        
        // Check if there's a specific provider for this operation
        auto it = operation_to_provider.find(operation_name);
        if (it != operation_to_provider.end() && it->second) {
            return it->second->get_cost(workload, cost_source);
        }
        
        // Use the default provider
        return default_provider->get_cost(workload, cost_source);
    }

    /**
     * @brief Get the maximum number of parameters across all providers
     * 
     * @return int The maximum number of parameters
     */
    int get_max_num_params() const override {
        int max_params = default_provider->get_max_num_params();
        
        // Check all mapped providers for their max params
        for (const auto& [op_name, provider] : operation_to_provider) {
            if (provider) {
                max_params = std::max(max_params, provider->get_max_num_params());
            }
        }
        
        return max_params;
    }

    /**
     * @brief Get the list of supported operators on a specified device
     * 
     * Returns the supported operations from the default provider.
     * 
     * @param device Specified device by caller
     * @return std::vector<std::string> Container with the names of operators
     */
    std::vector<std::string> get_shave_supported_ops(VPUDevice& device) const override {
        return default_provider->get_shave_supported_ops(device);
    }

    /**
     * @brief Provides a reference to the operator executor specified by name
     * 
     * Checks for a specific provider mapping first, then falls back to default provider.
     * 
     * @param name Name of the operator
     * @param device Device name
     * @return std::optional<std::reference_wrapper<const ShaveOpExecutor>> Reference to the executor if found
     */
    std::optional<std::reference_wrapper<const ShaveOpExecutor>> get_shave_instance(const std::string& name, VPUDevice& device) const override {
        // Check if there's a specific provider for this operation
        auto it = operation_to_provider.find(name);
        if (it != operation_to_provider.end() && it->second) {
            return it->second->get_shave_instance(name, device);
        }
        
        // Use the default provider
        return default_provider->get_shave_instance(name, device);
    }
};

}  // namespace VPUNN
#endif  // COMPOSITE_SHAVE_COST_PROVIDER_H
