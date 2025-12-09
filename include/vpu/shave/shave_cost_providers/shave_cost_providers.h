// Copyright © 2025 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.
#ifndef SHAVE_COST_PROVIDERS_H
#define SHAVE_COST_PROVIDERS_H

#include <vpu/shave/shave_cost_providers/shave_cost_provider_interface.h>
#include <core/cache.h>
#include <vpu/validation/shave_workloads_sanitizer.h>
#include <vpu/shave/shave_devices.h>

namespace VPUNN {

/**
 * @brief template base class that implements the common logic for calculating SHAVE workload costs using 
 * mathematical models (as opposed to NN-based models). It provides a unified implementation for two different 
 * SHAVE API versions (Shave1 and Shave2) that differ only in which selector method they call.
 * 
 * Using curiously recurring template pattern (CRTP) provides static polymorphism
 * @param CostProviderImpl The derived class implementing specific selector retrieval logic and cost source name.
 */
template<typename CostProviderImpl>
class MathematicalShaveCostProviderBase : public IShaveCostProvider {
private:
    ShaveConfiguration shave_configurator;   ///< shave config generator

protected:
    /// @brief provides access to the shave configurator instance used internally for CRTP
    /// @returns a const ref to the instance
    const ShaveConfiguration& getShaveConfigurator() const {
        return shave_configurator; 
    }

public:
    MathematicalShaveCostProviderBase() = default;

    /**
     * @brief Calculate the execution cost for a SHAVE workload using the Shave API
     * 
     * This function directly fetches the cost using the Shave API without checking cache.
     * If an exception occurs during fetchCyclesFromShave, it returns ERROR_SHAVE_OPERATOR_MISSING.
     * 
     * @param workload The SHAVE workload for which to calculate the cost
     * @param cost_source Optional pointer to store the source of the cost (e.g., "shave_1"). This is an output parameter only
     * @return CyclesInterfaceType The estimated execution cost/cycles for the workload,
     *         or ERROR_SHAVE_OPERATOR_MISSING if operator is missing or computation fails
     */
    CyclesInterfaceType get_cost(const SHAVEWorkload& workload, std::string* cost_source = nullptr) const override {    

        VPUDevice device = workload.get_device();
        auto shaveInstance = get_shave_instance(workload.get_name(), device);

        // operator not found
        if (!shaveInstance.has_value()) {
            return Cycles::ERROR_SHAVE_OPERATOR_MISSING;
        }

        if (cost_source) *cost_source = CostProviderImpl::cost_source_name;

        return shaveInstance.value().get().dpuCycles(workload);
    }

    /// @brief Get the maximum number of parameters across all SHAVE functions
    /// @return the max number found
    int get_max_num_params() const override {
        return shave_configurator.get_max_num_params();
    }

    
    /** @brief the list of names of supported operators on a specified device. Each device has own operators
     *
     * @param device specified device by caller
     * @returns container with the name of operators
     */
    std::vector<std::string> get_shave_supported_ops(VPUDevice& device) const override {
        const auto& sel = static_cast<const CostProviderImpl*>(this)->getSelectorImpl(device);
        const std::vector<std::string> list = sel.getShaveList();
        // clang and gcc does not support to use std::move here, so we need suppression
        /* coverity[copy_instead_of_move] */
        return list;
    }

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
        try {
            const auto& sel = static_cast<const CostProviderImpl*>(this)->getSelectorImpl(device);
            const auto& shaveInstance = sel.getShaveFuntion(name);
            return shaveInstance;
        } catch (...) {
            return std::nullopt;
        }
    }
};

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

} // namespace VPUNN
#endif //SHAVE_COST_PROVIDERS_H
