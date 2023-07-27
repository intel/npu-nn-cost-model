// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef KERNELS_KNN_H
#define KERNELS_KNN_H

#include "core/tensors.h"

namespace VPUNN {

/**
 * @brief Floating point k-Nearest-Neighbor (kNN) layer (float)
 *
 * @param weights a VPUNN::Tensor containing the kNN layer weights
 * @param targets a VPUNN::Tensor containing the kNN layer targets
 * @param activations the input tensor
 * @param output the output tensor
 * @param n_neighbours number of neighbors to consider. must be >=1
 */
VPUNN_API(void)
kNN(VPUNN::Tensor<float>* weights, VPUNN::Tensor<float>* targets, VPUNN::Tensor<float>* activations,
    VPUNN::Tensor<float>* output, unsigned int n_neighbours = 1);

}  // namespace VPUNN

#endif  // KERNELS_KNN_H