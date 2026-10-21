// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.
#ifndef VPUNN_MATHEMATICAL_SHAVE_COST_PROVIDER_BASE_H
#define VPUNN_MATHEMATICAL_SHAVE_COST_PROVIDER_BASE_H

#include <vpu/shave/shave_cost_providers/shave_cost_provider_interface.h>
#include <vpu/shave/shave_devices.h>

namespace VPUNN {

/**
 * @brief Template base class implementing common SHAVE workload cost computation logic.
 *
 * This class provides shared behavior for mathematical SHAVE providers, while the
 * derived class supplies device-specific selector retrieval via CRTP.
 *
 * @tparam CostProviderImpl Derived provider type implementing getSelectorImpl() and cost_source_name
 */
template <typename CostProviderImpl>
class MathematicalShaveCostProviderBase : public IShaveCostProvider {
private:
    /// @brief Independent SHAVE configuration for device-specific selector access.
    /// Constructing this provider does not create per-device selectors.
    /// Selectors are lazily created on first call to getSelector/getOldSelector/
    /// getHeuristicSelector/getHeuristicSelectorWithFactors, then cached.
    /// Each provider instance owns its own selector cache, reducing cross-instance contention.
    /// ShaveConfiguration synchronizes lazy creation; additional per-instance thread-safety
    /// guarantees are unspecified.
    const ShaveConfiguration shave_configurator;

protected:
    /// @brief Provides access to the SHAVE configuration instance used internally for CRTP.
    /// @return A const reference to the configuration instance.
    const ShaveConfiguration& getShaveConfigurator() const {
        return shave_configurator;
    }

public:
    MathematicalShaveCostProviderBase() = default;

    /**
     * @brief Calculate execution cost for a SHAVE workload using the selected SHAVE executor.
     *
     * This function directly fetches the cost using the Shave API without checking cache.
     * On the first call that targets a given device (through this provider instance),
     * selector resolution may trigger lazy per-device selector initialization in
     * ShaveConfiguration. Subsequent calls for that device reuse the cached selector.
     * If an exception occurs during selector/function resolution, it returns
     * ERROR_SHAVE_OPERATOR_MISSING.
     *
     * @param workload SHAVE workload for which to calculate cost.
     * @param cost_source Optional pointer to store the source of the cost (e.g., "shave_1").
     * @return Estimated execution cycles or ERROR_SHAVE_OPERATOR_MISSING.
     */
    CyclesInterfaceType get_cost(const SHAVEWorkload& workload, std::string* cost_source = nullptr) const override {
        VPUDevice device = workload.get_device();
        auto shaveInstance = get_shave_instance(workload.get_name(), device);

        // Operator not found for this selector/device pair.
        if (!shaveInstance.has_value()) {
            return Cycles::ERROR_SHAVE_OPERATOR_MISSING;
        }

        if (cost_source)
            *cost_source = CostProviderImpl::cost_source_name;

        return shaveInstance.value().get().dpuCycles(workload);
    }

    /// @brief Get the maximum number of parameters across all SHAVE functions.
    /// @return The max number found.
    int get_max_num_params() const override {
        return getShaveConfigurator().get_max_num_params();
    }

    /** @brief List names of supported operators on a specified device.
     *
     * @param device Specified device by caller.
     * @return Container with operator names.
     */
    std::vector<std::string> get_shave_supported_ops(VPUDevice& device) const override {
        const auto& sel = static_cast<const CostProviderImpl*>(this)->getSelectorImpl(device);
        const std::vector<std::string> list = sel.getShaveList();
        // clang and gcc do not support std::move optimization in this case.
        /* coverity[copy_instead_of_move] */
        return list;
    }

    /**
     * @brief Provide reference to operator executor specified by name.
     *
     * @param name Operator name.
     * @param device Device name.
     * @return Reference wrapper to executor if found; std::nullopt otherwise.
     */
    std::optional<std::reference_wrapper<const ShaveOpExecutor>> get_shave_instance(const std::string& name,
                                                                                     VPUDevice& device) const override {
        try {
            const auto& sel = static_cast<const CostProviderImpl*>(this)->getSelectorImpl(device);
            const auto& shaveInstance = sel.getShaveFunction(name);
            return shaveInstance;
        } catch (...) {
            return std::nullopt;
        }
    }
};

}  // namespace VPUNN

#endif  // VPUNN_MATHEMATICAL_SHAVE_COST_PROVIDER_BASE_H
