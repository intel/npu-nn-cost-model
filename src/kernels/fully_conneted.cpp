// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "kernels/fully_connected.h"
#include "kernels/vpunn_blas.h"

void VPUNN::Dense(const VPUNN::Tensor<float>* weights, const VPUNN::Tensor<float>* activations,
                  VPUNN::Tensor<float>* output) {
    // Use cblas_sgemm to compute C <- alpha A * B + beta C

    int output_channels = output->shape()[1];
    int input_channels = activations->shape()[1];
    int batch_size = activations->shape()[0];

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch_size, output_channels, input_channels, 1.0F,
                activations->c_ptr(), input_channels, weights->c_ptr(), input_channels, 0.0F, output->data(),
                output_channels);
}
