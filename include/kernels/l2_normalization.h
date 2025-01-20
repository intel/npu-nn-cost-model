// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef KERNELS_L2NORM_H
#define KERNELS_L2NORM_H

#include "core/tensors.h"

namespace VPUNN {

/**
 * @brief Floating point L2 normalization layer (float)
 *
 * @param activations the input tensor
 * @param output the output tensor
 */
VPUNN_API(void) L2Normalization(VPUNN::Tensor<float>* activations, VPUNN::Tensor<float>* output);

}  // namespace VPUNN

#endif  // KERNELS_L2NORM_H