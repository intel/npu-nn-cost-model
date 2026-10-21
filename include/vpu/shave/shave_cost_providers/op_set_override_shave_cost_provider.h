// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.
#ifndef VPUNN_OP_SET_OVERRIDE_SHAVE_COST_PROVIDER_H
#define VPUNN_OP_SET_OVERRIDE_SHAVE_COST_PROVIDER_H

#include <vpu/shave/shave_cost_providers/shave_cost_provider_interface.h>

#include <algorithm>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>

namespace VPUNN {

/**
 * @brief Routes only a selected operation set to an override provider.
 *
 * This provider is intentionally narrow and policy-focused:
 * - Operations listed in override_operations are delegated to override_provider
 * - All other operations are delegated to default_provider
 * - Exactly one override provider is supported by design
 * - If override_provider is null, all operations fall back to default_provider
 * - Supported-op queries are delegated to default_provider to preserve current
 *   device-capability reporting semantics
 *
 * Use this class for the common "default provider + one override provider for
 * a fixed op set" composition. For per-op routing to multiple different override
 * providers, use CompositeShaveCostProvider instead.
 */
class OpSetOverrideShaveCostProvider : public IShaveCostProvider {
private:
    const Shared_IShaveCostProvider default_provider;
    const Shared_IShaveCostProvider override_provider;
    const std::set<std::string> override_operations;

    bool should_use_override(const std::string& operation_name) const {
        return override_operations.find(operation_name) != override_operations.end();
    }

public:
    OpSetOverrideShaveCostProvider(Shared_IShaveCostProvider default_provider,
                                   Shared_IShaveCostProvider override_provider,
                                   std::set<std::string> override_operations = {})
            : default_provider(std::move(default_provider)),
              override_provider(std::move(override_provider)),
              override_operations(std::move(override_operations)) {
        if (!this->default_provider) {
            throw std::invalid_argument("Default provider cannot be null");
        }
    }

    CyclesInterfaceType get_cost(const SHAVEWorkload& workload, std::string* cost_source = nullptr) const override {
        if (override_provider && should_use_override(workload.get_name())) {
            return override_provider->get_cost(workload, cost_source);
        }
        return default_provider->get_cost(workload, cost_source);
    }

    int get_max_num_params() const override {
        if (!override_provider) {
            return default_provider->get_max_num_params();
        }
        return std::max(default_provider->get_max_num_params(), override_provider->get_max_num_params());
    }

    std::vector<std::string> get_shave_supported_ops(VPUDevice& device) const override {
        return default_provider->get_shave_supported_ops(device);
    }

    std::optional<std::reference_wrapper<const ShaveOpExecutor>> get_shave_instance(const std::string& name,
                                                                                    VPUDevice& device) const override {
        if (override_provider && should_use_override(name)) {
            return override_provider->get_shave_instance(name, device);
        }
        return default_provider->get_shave_instance(name, device);
    }
};

}  // namespace VPUNN

#endif  // VPUNN_OP_SET_OVERRIDE_SHAVE_COST_PROVIDER_H
