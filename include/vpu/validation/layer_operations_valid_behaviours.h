// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_VPU_LAYER_OPERATIONS_VALID_BEHAVIOURS_H
#define VPUNN_VPU_LAYER_OPERATIONS_VALID_BEHAVIOURS_H

#include "vpu/types.h"

#include "checker_utils.h"
#include "dpu_operations_valid_behaviours.h"

namespace VPUNN {

class CONVOLUTION_Constraints_Layer : public CONVOLUTION_Constraints {
protected:
    bool check_sparsity_rules(const IDeviceValidValues& config, const DPUOperation& dpu,
                              std::string& info) const override {
        std::string local_info{};
        Checker checker;
        // if (!check_sparsity_layer_SOK(config, dpu, local_info)) {
        //     checker.add_check_failed(local_info);
        // }
        local_info = "";
        if (!CONVOLUTION_Constraints::check_sparsity_rules(config, dpu, local_info)) {
            checker.add_check_failed(local_info);
        }
        info = checker.findings();
        return checker.is_clean();
    }
};

class DW_CONVOLUTION_Constraints_Layer : public DW_CONVOLUTION_Constraints {
protected:
};

class CM_CONVOLUTION_Constraints_Layer : public CM_CONVOLUTION_Constraints {
protected:
};

class ELTWISE_Constraints_Layer : public ELTWISE_Constraints {
protected:
};

class MAXPOOL_Constraints_Layer : public MAXPOOL_Constraints {
protected:
};

}  // namespace VPUNN

#endif  //
