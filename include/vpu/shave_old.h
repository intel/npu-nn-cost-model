// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_SHAVE_OLD_H
#define VPUNN_SHAVE_OLD_H

#include <vector>

#include "dpu_types.h"
#include "vpu_tensor.h"

namespace VPUNN {

/**
 * @brief The base structure that encodes a Software layer
 * \deprecated
 *
 */
struct SWOperation {
    VPUDevice device;                      ///< The VPU device
    const std::vector<VPUTensor> inputs;   ///< The input tensors
    const std::vector<VPUTensor> outputs;  ///< The output tensors

    /// @brief ctor must exist since we have aggregate initialization possible on this type (abstract type)
    SWOperation(const VPUDevice& device, const std::vector<VPUTensor>& inputs, const std::vector<VPUTensor>& outputs)
            : device{device}, inputs{inputs}, outputs{outputs} {
    }

    /**
     * @brief Return the number of cycles of the sw operation
     *
     * @return unsigned int
     */
    virtual unsigned int cycles() const = 0;

    /**
     * @brief Destroy the SWOperation object
     *
     */
    virtual ~SWOperation(){};
};

}  // namespace VPUNN

#endif  // VPUNN_TYPES_H
