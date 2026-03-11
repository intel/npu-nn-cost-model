// Copyright © 2025 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.
#ifndef NAME_MAPPING_SHAVE_COST_PROVIDER_H
#define NAME_MAPPING_SHAVE_COST_PROVIDER_H

#include <vpu/shave/shave_cost_providers/shave_cost_provider_interface.h>
#include <memory>
#include <string>
#include <unordered_map>

namespace VPUNN {

/**
 * @brief Name-mapping SHAVE cost provider that translates operation names before delegation
 * 
 * This provider acts as a wrapper around another provider, allowing operation names
 * to be remapped. For example, if the workload uses "OpName" but the underlying
 * provider expects "op_name", this provider can perform the translation.
 * 
 * Use cases:
 * - Handling naming convention differences between caller and provider
 * - Supporting multiple aliases for the same operation
 * - Gradual migration of operation naming schemes
 */
class NameMappingShaveCostProvider : public IShaveCostProvider {
private:
    const std::shared_ptr<IShaveCostProvider> underlying_provider;  ///< The wrapped provider
    const std::unordered_map<std::string, std::string> name_mapping;  ///< Maps input name to provider name
    const std::unordered_map<std::string, std::string> reverse_name_mapping;  ///< Maps provider name to input name
                                                                              ///< Built so we can return correct names 
                                                                              ///< in get_shave_supported_ops

    /**
     * @brief Build the reverse mapping from provider names to input names
     * 
     * @param forward_mapping The forward mapping (input name -> provider name)
     * @return std::unordered_map<std::string, std::string> The reverse mapping (provider name -> input name)
     */
    static std::unordered_map<std::string, std::string> build_reverse_mapping(
        const std::unordered_map<std::string, std::string>& forward_mapping) {
        std::unordered_map<std::string, std::string> reverse;
        for (const auto& [input_name, provider_name] : forward_mapping) {
            reverse[provider_name] = input_name;
        }
        return reverse;
    }

public:
    /**
     * @brief Construct a new Name Mapping Shave Cost Provider
     * 
     * @param provider The underlying provider to delegate to
     * @param mappings Map from input operation names to provider operation names
     */
    explicit NameMappingShaveCostProvider(
        std::shared_ptr<IShaveCostProvider> provider,
        std::unordered_map<std::string, std::string> mappings = {})
        : underlying_provider(std::move(provider)),
          name_mapping(std::move(mappings)),
          reverse_name_mapping(build_reverse_mapping(name_mapping)) {
        
        if (!underlying_provider) {
            throw std::invalid_argument("Underlying provider cannot be null");
        }
    }

    /**
     * @brief Calculate the execution cost for a SHAVE workload with name translation
     * 
     * Translates the operation name if a mapping exists, then delegates to the
     * underlying provider.
     * 
     * @param workload The SHAVE workload for which to calculate the cost
     * @param cost_source Optional pointer to store the source of the cost
     * @return CyclesInterfaceType The estimated execution cost/cycles for the workload
     */
    CyclesInterfaceType get_cost(const SHAVEWorkload& workload, std::string* cost_source = nullptr) const override {
        auto it = name_mapping.find(workload.get_name());
        
        // If name needs translation, create a new workload with the translated name
        if (it != name_mapping.end()) {
            SHAVEWorkload translated_workload(
                it->second,
                workload.get_device(),
                workload.get_inputs(),
                workload.get_outputs(),
                workload.get_params(),
                workload.get_extra_params(),
                workload.get_loc_name()
            );
            return underlying_provider->get_cost(translated_workload, cost_source);
        }
        
        // No translation needed, use original workload
        return underlying_provider->get_cost(workload, cost_source);
    }

    /**
     * @brief Get the maximum number of parameters from the underlying provider
     * 
     * @return int The maximum number of parameters
     */
    int get_max_num_params() const override {
        return underlying_provider->get_max_num_params();
    }

    /**
     * @brief Get the list of supported operators on a specified device
     * 
     * Returns the operations supported by the underlying provider.
     * Note: The returned names are as the underlying provider knows them,
     * not necessarily as they appear in the workload.
     * 
     * @param device Specified device by caller
     * @return std::vector<std::string> Container with the names of operators
     */
    std::vector<std::string> get_shave_supported_ops(VPUDevice& device) const override {
        auto ops = underlying_provider->get_shave_supported_ops(device);
        
        // Replace provider names with input names where mapping exists
        for (auto& op_name : ops) {
            auto it = reverse_name_mapping.find(op_name);
            if (it != reverse_name_mapping.end()) {
                op_name = it->second;  // Replace with the input name (key)
            }
        }
        
        return ops;
    }

    /**
     * @brief Provides a reference to the operator executor specified by name
     * 
     * Translates the name before querying the underlying provider.
     * 
     * @param name Name of the operator (will be translated if mapping exists)
     * @param device Device name
     * @return std::optional<std::reference_wrapper<const ShaveOpExecutor>> Reference to the executor if found
     */
    std::optional<std::reference_wrapper<const ShaveOpExecutor>> get_shave_instance(const std::string& name, VPUDevice& device) const override {
        auto it = name_mapping.find(name);
        const std::string& translated_name = (it != name_mapping.end()) ? it->second : name;
        return underlying_provider->get_shave_instance(translated_name, device);
    }
};

}  // namespace VPUNN
#endif  // NAME_MAPPING_SHAVE_COST_PROVIDER_H
