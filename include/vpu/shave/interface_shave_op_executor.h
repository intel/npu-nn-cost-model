// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef INTERFACE_SHAVE_OP_EXECUTOR_H
#define INTERFACE_SHAVE_OP_EXECUTOR_H

#include "vpu/cycles_interface_types.h"
#include "vpu/types.h"

namespace VPUNN {

/// @brief Interface for a shave model from the execution(obtaining the runtime) perspective.
/// Instances are properly configured Shave models, for a particular model.
/// Cannot be deleted by the user.
class ShaveOpExecutor {
public:
    /**
     * @brief Return the number of cycles of the sw operation
     *
     * @param w the workload descriptor. Some fields may be ignored . Eg even if the Device is not matching the one that
     * the model was built for, it will run the estimation as if it had a good device in workload
     *
     * @return cycles in dpu frequency
     */
    virtual CyclesInterfaceType dpuCycles(const SHAVEWorkload& w) const = 0;

    /// @brief the name of the modeled SHAVE function
    virtual std::string getName() const {
        return operation_name;
    }

    /// provides a string with details of implementation, name parameters, frequencies, etc
    virtual std::string toString() const = 0;

protected:
    /// @brief copy ctor, default like implementation
    ShaveOpExecutor(const std::string& name, const unsigned int num_expected_params = 0): operation_name(name), num_expected_params(num_expected_params) {
    }

    /// Destructor is not public, so no user can delete this object of it has a naked pointer to it
    virtual ~ShaveOpExecutor() = default;
    
private:
    std::string operation_name;  ///< shave function name

    const unsigned int num_expected_params;  ///< number of expected (variadic) parameters passed through SHAVEWorkload

    /// @brief the friend class can delete the instance
    friend class DeviceShaveContainer;

public:  // extended interface
    /**
     * @brief Return the number of cycles of the sw operation considering that the nominal freq were altered with the
     * passed ones as params
     *
     * @param w the workload descriptor. Some fields may be ignored . Eg even if the Device is not matching the one that
     * the model was built for, it will run the estimation as if it had a good device in workload
     * @param present_dpu_frq  the new DPU freq
     * @param present_shv_frq the new shave (VPU) freq
     *
     * @return cycles in dpu frequency as specified by present_dpu_frq
     */
    virtual CyclesInterfaceType dpuCycles(const SHAVEWorkload& w, const int present_dpu_frq,
                                          const int present_shv_frq) const = 0;

    unsigned int getNumExpectedParams() const {
        return num_expected_params;
    }
};

}  // namespace VPUNN
#endif