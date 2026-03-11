// Copyright © 2025 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.
#ifndef DEVICE_MAPPING_SHAVE_COST_PROVIDER_H
#define DEVICE_MAPPING_SHAVE_COST_PROVIDER_H

#include <vpu/shave/shave_cost_providers/shave_cost_provider_interface.h>
#include <memory>
#include <unordered_map>
#include <set>

namespace VPUNN {

/**
 * @brief Device-mapping SHAVE cost provider that routes to different providers based on device
 * 
 * This provider selects which underlying provider to use based on the device specified
 * in the workload. Different devices can use different cost models or calibrations.
 * 
 * Use cases:
 * - Device-specific cost models (VPU 2.7 uses one model, VPU 4.0 uses another)
 * - Gradual rollout of new models per device
 * - Using specialized providers optimized for specific hardware
 */
class DeviceMappingShaveCostProvider : public IShaveCostProvider {
private:
    const std::shared_ptr<IShaveCostProvider> default_provider;  ///< Fallback provider for unmapped devices
    const std::unordered_map<VPUDevice, std::shared_ptr<IShaveCostProvider>> device_providers;  ///< Device-specific providers

    /**
     * @brief Get the provider for a specific device
     * 
     * @param device The device to get the provider for
     * @return std::shared_ptr<IShaveCostProvider> The provider for this device, or default if not mapped
     */
    std::shared_ptr<IShaveCostProvider> get_provider_for_device(VPUDevice device) const {
        auto it = device_providers.find(device);
        return (it != device_providers.end() && it->second) ? it->second : default_provider;
    }

public:
    /**
     * @brief Construct a new Device Mapping Shave Cost Provider
     * 
     * @param default_prov The default provider for unmapped devices
     * @param device_mappings Map from devices to their specific providers
     */
    explicit DeviceMappingShaveCostProvider(
        std::shared_ptr<IShaveCostProvider> default_prov,
        std::unordered_map<VPUDevice, std::shared_ptr<IShaveCostProvider>> device_mappings = {})
        : default_provider(std::move(default_prov)),
          device_providers(std::move(device_mappings)) {
        
        if (!default_provider) {
            throw std::invalid_argument("Default provider cannot be null");
        }
    }

    /**
     * @brief Calculate the execution cost for a SHAVE workload using device-based routing
     * 
     * Routes the workload to the appropriate provider based on the device in the workload.
     * 
     * @param workload The SHAVE workload for which to calculate the cost
     * @param cost_source Optional pointer to store the source of the cost
     * @return CyclesInterfaceType The estimated execution cost/cycles for the workload
     */
    CyclesInterfaceType get_cost(const SHAVEWorkload& workload, std::string* cost_source = nullptr) const override {
        VPUDevice device = workload.get_device();
        auto provider = get_provider_for_device(device);
        
        if (provider) {
            return provider->get_cost(workload, cost_source);
        }
        
        // No provider available for device then use default one
        if (default_provider) {
            return default_provider->get_cost(workload, cost_source);
        }

        // No provider at all
        if (cost_source) {
            *cost_source = "no_provider";
        }
        return Cycles::ERROR_SHAVE_OPERATOR_MISSING;
    }

    /**
     * @brief Get the maximum number of parameters across all providers
     * 
     * @return int The maximum number of parameters
     */
    int get_max_num_params() const override {
        int max_params = 0;
        
        if (default_provider) {
            max_params = default_provider->get_max_num_params();
        }
        
        for (const auto& [device, provider] : device_providers) {
            if (provider) {
                max_params = std::max(max_params, provider->get_max_num_params());
            }
        }
        
        return max_params;
    }

    /**
     * @brief Get the list of supported operators on a specified device
     * 
     * Returns operations supported by the provider mapped to this device.
     * 
     * @param device Specified device by caller
     * @return std::vector<std::string> Container with the names of operators
     */
    std::vector<std::string> get_shave_supported_ops(VPUDevice& device) const override {
        auto provider = get_provider_for_device(device);
        if (provider) {
            return provider->get_shave_supported_ops(device);
        }
        return std::vector<std::string>();
    }

    /**
     * @brief Provides a reference to the operator executor specified by name
     * 
     * Routes to the provider mapped to the specified device.
     * 
     * @param name Name of the operator
     * @param device Device name
     * @return std::optional<std::reference_wrapper<const ShaveOpExecutor>> Reference to the executor if found
     */
    std::optional<std::reference_wrapper<const ShaveOpExecutor>> get_shave_instance(
        const std::string& name, VPUDevice& device) const override {
        auto provider = get_provider_for_device(device);
        if (provider) {
            return provider->get_shave_instance(name, device);
        }
        return std::nullopt;
    }
};

}  // namespace VPUNN
#endif  // DEVICE_MAPPING_SHAVE_COST_PROVIDER_H
