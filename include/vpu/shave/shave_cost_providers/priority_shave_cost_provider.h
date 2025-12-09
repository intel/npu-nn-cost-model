// Copyright © 2025 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#ifndef PRIORITY_SHAVE_COST_PROVIDER_H
#define PRIORITY_SHAVE_COST_PROVIDER_H

#include <vpu/shave/shave_cost_providers/shave_cost_provider_interface.h>
#include <vpu/shave/shave_cost_providers/shave_cost_providers.h>
#include <vector>
#include <memory>

namespace VPUNN {

/**
 * @brief Type alias for a list of SHAVE cost providers.
 * Currently we keep it as a shared_ptr to allow co-ownership in future.
 * If we are going to add a different way to composite the Providers we can
 * simply share the references.
 */
using ShaveCostProviderList = std::vector<std::shared_ptr<IShaveCostProvider>>;

/**
 * @brief Priority-based SHAVE cost provider that tries multiple providers in order
 * 
 * This class implements a priority-based approach to SHAVE cost calculation by
 * maintaining a list of cost providers and trying them in order until one succeeds,
 * otherwise proper error code is returned.
 */
class PriorityShaveCostProvider : public IShaveCostProvider {
private:
    ShaveCostProviderList cost_providers;  ///< List of cost providers in priority order

public:
    /**
     * @brief Construct a new Priority Shave Cost Provider
     * 
     * @param providers Vector of shared pointers to IShaveCostProvider instances,
     *                  ordered by priority (first provider has highest priority)
     */
    explicit PriorityShaveCostProvider(const ShaveCostProviderList& providers)
        : cost_providers(providers) {}

    /**
     * @brief Calculate the execution cost for a SHAVE workload using priority-based providers
     * 
     * This function tries each cost provider in the priority list until one succeeds
     * (returns a value other than error codes). If all providers fail, it returns
     * the last error encountered.
     * 
     * @param workload The SHAVE workload for which to calculate the cost
     * @param cost_source Optional pointer to store the source of the cost. This is an output parameter only
     * @return CyclesInterfaceType The estimated execution cost/cycles for the workload,
     *         or an appropriate error code if all providers fail
     */
    CyclesInterfaceType get_cost(const SHAVEWorkload& workload, std::string* cost_source = nullptr) const override {
        // set it from start to error code in case no provider is in list
        CyclesInterfaceType cycles = Cycles::ERROR_SHAVE_OPERATOR_MISSING;

        for (const auto& prov_ref : cost_providers) {
            if (!prov_ref) continue;

            CyclesInterfaceType result = prov_ref->get_cost(workload, cost_source);

            // If the provider succeeded (no error), return the result
            if (!Cycles::isErrorCode(result)) {
                return result;
            }
        }
        
        // All providers failed, return the last error
        if (cost_source) {
            *cost_source = "unknown";
        }
        return cycles;
    }

    /// @brief Get the maximum number of parameters across all SHAVE functions and all CostProviders 
    /// @return the max number found
    int get_max_num_params() const override {
        std::vector<int> all_num_params{};

        for (const auto& prov_reference : cost_providers) {
            if (!prov_reference) continue;

            all_num_params.push_back(prov_reference->get_max_num_params());
        }

        return *std::max_element(all_num_params.begin(), all_num_params.end());
    }

    /** @brief the list of names of supported operators on a specified device. Each device has own operators
     *
     * @param device specified device by caller
     * @returns container with the name of operators
     */
    std::vector<std::string> get_shave_supported_ops(VPUDevice& device) const override {
        std::set<std::string> unique_ops_set;

        for (const auto& prov_ref : cost_providers) {
            if (!prov_ref) continue;

            auto ops = prov_ref->get_shave_supported_ops(device);
            unique_ops_set.insert(ops.begin(), ops.end());
        }

        // Convert set to vector
        return std::vector<std::string>(unique_ops_set.begin(), unique_ops_set.end());
    };

    /** 
     * @brief provides a reference to the operator executor specified by name
     * The executor can be used to execute (run) the runtime prediction on different tensors/parameters
     * Or can be asked to print information about its implementation parameters
     *
     * @param name of the operator
     * @param device name
     * @returns a ref (no ownership transfered. exists as long as this VPUCostModel instance exists)
     */
    std::optional<std::reference_wrapper<const ShaveOpExecutor>> get_shave_instance(const std::string& name, VPUDevice& device) const override {
        for (const auto& prov_reference : cost_providers) {
            if (!prov_reference) continue;

            auto result = prov_reference->get_shave_instance(name, device);
            if (result.has_value()) {
                return result;
            }
        }

        return std::nullopt;
    };

};

}  // namespace VPUNN
#endif  // PRIORITY_SHAVE_COST_PROVIDER_H
