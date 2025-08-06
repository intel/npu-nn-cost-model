// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef KERNELS_SIGMOID_H
#define KERNELS_SIGMOID_H

#include "core/tensors.h"

namespace VPUNN {

/**
 * @brief Floating point sigmoid layer (float). The operation is done implace
 *
 * @param output the input/output tensor
 */
VPUNN_API void Sigmoid(VPUNN::Tensor<float>* output);

}  // namespace VPUNN

#endif  // KERNELS_SIGMOID_H
