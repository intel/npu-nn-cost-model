// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_SHAVE_WORKLOADS_SANITIZER_H
#define VPUNN_SHAVE_WORKLOADS_SANITIZER_H

#include <vpu/shave_workload.h>
#include "sanity_report.h"
#include "vpu/types.h"
#include "memory_calculator.h"
#include "dpu_operations_validator.h"

namespace VPUNN {
class SHAVE_Workloads_Sanitizer : private SHAVE_OperationValidator {

public:
    void check_and_sanitize(const SHAVEWorkload& swl, SanityReport& result) const {
        result.resetOK();

        // Check the datatype of the input and output tensors
        {
            const auto intype_0{swl.get_inputs()[0].get_dtype()};
            const auto outtype_0{swl.get_outputs()[0].get_dtype()};
            if (intype_0 != DataType::FLOAT16 || outtype_0 != DataType::FLOAT16) {
                result.info = "SHAVE workload input/output tensor datatype can only be FLOAT16 for profiled regressions";
                result.mark_invalid_SHAVE_workload();
                return;
            }
        };
        // Check if it fits in CMX
        const auto& config = get_config(swl.get_device());
        const auto cmx_memory = memory_calculator.compute_memory(swl);
        const int avaialable_cmx_memo{config.get_cmx_size(swl.get_device())};
        const auto necesarry_cmx_memo = cmx_memory.cmx;

        if (avaialable_cmx_memo < necesarry_cmx_memo) {
            result.mark_size_too_big();
            return;
        }
    }
};
}
#endif  //
