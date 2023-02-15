// Copyright © 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "kernels/l2_normalization.h"
#include "kernels/vpunn_blas.h"

void VPUNN::L2Normalization(VPUNN::Tensor<float>* activations, VPUNN::Tensor<float>* output) {
    // Use cblas_sgemm to compute C <- alpha A * B + beta C

    int channels = activations->shape()[1];
    int batch_size = activations->shape()[0];

    for (auto batch = 0; batch < batch_size; batch++) {
        // Compute the norm of the vector
        auto norm = cblas_snrm2(channels, activations->data() + channels * batch, 1);
        if (norm > 0)
            // Scale the tensor by the inverse of the norm
            cblas_sscal(channels, 1.0f / norm, activations->data() + channels * batch, 1);
    }

    // Assign to the output
    output->assign(activations->data(), activations->size() * sizeof(float));
}
