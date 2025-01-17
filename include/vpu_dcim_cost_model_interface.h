// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_VPU_DCIM_COST_MODEL_INTERFACE_H
#define VPUNN_VPU_DCIM_COST_MODEL_INTERFACE_H

#include <string>

#include "vpu/dpu_dcim_workload.h"

namespace VPUNN {

/// @brief The interfacefor DCim COst model operations at  workload level (L1/variant)
/// DCiM_Workload_Alias can be a DCIMWorkload or a variant of it, or a DPUWorkload if reused the datastructure.
template <class DCiM_Workload_Alias>
class DCiMCostModelInterface {
public:
    // future potential functionalities
    // DCIMInfoPack DCiMInfo(const DCiM_Workload_Alias& workload)
    // float DCiMEnergy(const DCiM_Workload_Alias& wl)
    // float DCiMActivityFactorXXXX(const DCiM_Workload_Alias& wl)   //more methods here

    // first interface
    CyclesInterfaceType dCiM(const DCiM_Workload_Alias& wl, std::string& info) {
        info = "";
        if (wl.kernels[0])
            return Cycles::ERROR_INFERENCE_NOT_POSSIBLE;
        else
            return Cycles::ERROR_INVALID_INPUT_CONFIGURATION;
    }

    // for python binding
    std::tuple<CyclesInterfaceType, std::string> dCiM_Msg(DCiM_Workload_Alias wl) {
        std::string info;
        CyclesInterfaceType cycles = dCiM(wl, info);
        return std::make_tuple(cycles, info);
    }

private:
};

}  // namespace VPUNN

#endif  // VPUNN_TYPES_H
