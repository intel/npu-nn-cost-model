// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_VPU_SANITY_REPORT_H
#define VPUNN_VPU_SANITY_REPORT_H

#include "vpu/cycles_interface_types.h"
#include "vpu/types.h"

namespace VPUNN {

/// @brief Post sanity analysis.
///
struct SanityReport {
private:
    CyclesInterfaceType abnormal_return{Cycles::NO_ERROR};  ///< result code

public:
    std::string info{};  ///< accumulates textual information about problems or information

    /// is the workload usable for NN run
    bool is_usable() const {
        return abnormal_return == Cycles::NO_ERROR;
    };
    bool has_error() const {
        return !is_usable();
    };
    void resetOK() {
        abnormal_return = Cycles::NO_ERROR;
        info.clear();
    }
    CyclesInterfaceType value() const {
        return abnormal_return;
    }

    void mark_size_too_big() {
        abnormal_return = Cycles::ERROR_INPUT_TOO_BIG;
    }
    void mark_unknown_device() {
        abnormal_return = Cycles::ERROR_INVALID_INPUT_DEVICE;
    }
    void mark_unknown_operation() {
        abnormal_return = Cycles::ERROR_INVALID_INPUT_OPERATION;
    }
    void mark_invalid_NN_response() {
        abnormal_return = Cycles::ERROR_INVALID_OUTPUT_RANGE;
    }
    void mark_invalid_DPU_workload() {
        abnormal_return = Cycles::ERROR_INVALID_INPUT_CONFIGURATION;
    }
    void mark_invalid_SHAVE_workload(){
        abnormal_return = Cycles::ERROR_SHAVE_INVALID_INPUT;
    }
    void mark_invalid_LayerConfiguration() {
        abnormal_return = Cycles::ERROR_INVALID_LAYER_CONFIGURATION;
    }
    void mark_split_error() {
        abnormal_return = Cycles::ERROR_TILE_OUTPUT;
    }
};

}  // namespace VPUNN

#endif  //
