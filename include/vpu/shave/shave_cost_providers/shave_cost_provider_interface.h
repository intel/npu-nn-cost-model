// Copyright © 2025 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#ifndef SHAVE_COST_PROVIDER_INTERFACE_H
#define SHAVE_COST_PROVIDER_INTERFACE_H

#include "vpu/shave_workload.h"
#include "vpu/cycles_interface_types.h"
#include <optional>
#include <functional>
#include <vector>
#include <string>
#include "vpu/shave/shave_op_executors.h"


namespace VPUNN {

/**
 * @brief Interface for SHAVE cost providers
 * 
 * This interface defines the contract for classes that provide cost estimation
 * for SHAVE workloads. Implementations of this interface should provide
 * specific algorithms or models for calculating the execution cost of
 * SHAVE operations.
 */
class IShaveCostProvider {
public:
    /**
     * @brief Virtual destructor for proper cleanup of derived classes
     */
    virtual ~IShaveCostProvider() = default;

    /**
     * @brief Calculate the execution cost for a SHAVE workload
     * 
     * This pure virtual function must be implemented by derived classes to
     * provide cost estimation for the given SHAVE workload. The cost is
     * typically measured in cycles or other performance metrics.
     * 
     * @param workload The SHAVE workload for which to calculate the cost
     * @param cost_source The source of cost under a string to be put in serializer
     * @return CyclesInterfaceType The estimated execution cost/cycles for the workload
     */
    
    virtual CyclesInterfaceType get_cost(const SHAVEWorkload& workload, std::string* cost_source = nullptr) const = 0;

    /// @brief Get the maximum number of parameters across all SHAVE functions
    /// @return the max number found
    virtual int get_max_num_params() const = 0;

    /** @brief the list of names of supported operators on a specified device. Each device has own operators
     *
     * @param device specified device by caller
     * @returns container with the name of operators
     */
    virtual std::vector<std::string> get_shave_supported_ops(VPUDevice& device) const = 0;


    /** 
     * @brief provides a reference to the operator executor specified by name
     * The executor can be used to execute (run) the runtime prediction on different tensors/parameters
     * Or can be asked to print information about its implementation parameters
     *
     * @param name of the operator
     * @param device name
     * @returns a ref (no ownership transfered. exists as long as this VPUCostModel instance exists)
     */
    virtual std::optional<std::reference_wrapper<const ShaveOpExecutor>> get_shave_instance(const std::string& name, VPUDevice& device) const = 0;

};

}  // namespace VPUNN
#endif  // SHAVE_COST_PROVIDER_INTERFACE_H
