// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef KERNELS_FC_H
#define KERNELS_FC_H

#include "core/tensors.h"

namespace VPUNN {

/**
 * @brief Floating point FC layer (float)
 *
 * @param weights a VPUNN::Tensor containing the FC layer weights
 * @param activations the input tensor
 * @param output the output tensor
 */
VPUNN_API(void)
Dense(const VPUNN::Tensor<float>* weights, const VPUNN::Tensor<float>* activations, VPUNN::Tensor<float>* output);

}  // namespace VPUNN

#endif  // KERNELS_FC_H