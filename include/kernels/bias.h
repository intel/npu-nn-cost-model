// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef KERNELS_BIAS_H
#define KERNELS_BIAS_H

#include "core/tensors.h"

namespace VPUNN {

/// @brief Floating point bias layer (float). The instance has a helper memory for the constant buffer
class VPUNN_API BiasOpBuffer {
private:
    std::vector<float> batch_buffer{1.0F};
    friend class BiasOp;

public:
    /// @brief ensures a minimim allocated space for the constant bias buffer
    void reserve_bias_space(int space_required, float fill_value = 1.0F) {
        batch_buffer.reserve(space_required);
        batch_buffer.resize(0);  // for ensuring the fill value everywhere
        batch_buffer.resize(space_required, fill_value);
    }
};

/// @brief Floating point bias layer (float). The instance has a helper memory for the constant buffer
class VPUNN_API BiasOp {
private:
public:
    /**
     * @brief Floating point bias layer (float). The operation is done implace in the output tensor
     *
     * @param bias a VPUNN::Tensor containing the bias
     * @param output the input/output tensor
     */
    static void Bias(const VPUNN::Tensor<float>* bias, VPUNN::Tensor<float>* output, BiasOpBuffer& batch_buff);
};

}  // namespace VPUNN

#endif  // KERNELS_BIAS_H
