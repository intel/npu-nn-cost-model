// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/validation/data_dpu_operation.h"
#include "vpu/validation/interface_valid_values.h"
#include "vpu/validation/interface_operations_behavior.h"

namespace VPUNN {

DPUOperation::DPUOperation(const DPUWorkload& w, const IDeviceValidValues& config)
        : DPUOperation{w}  // delegate to the main constructor
{
    auto& operation_behaviour = config.get_specific_behaviour(this->operation);// may throw
    operation_behaviour.deduce_input_1_shape_and_layout(input_0, output_0, config, kernel, input_1);
}

}  // namespace VPUNN