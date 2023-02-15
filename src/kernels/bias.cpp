// Copyright © 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "kernels/bias.h"
#include "kernels/vpunn_blas.h"

void VPUNN::Bias(VPUNN::Tensor<float>* bias, VPUNN::Tensor<float>* output) {
    // Use cblas_sgemm to compute C <- alpha A * B + beta C

    int batch_size = output->shape()[0];
    int output_channels = output->shape()[1];

    const int input_channels = 1;

    // Create a bias multiplier for batching
    float* bias_multiplier = new float[batch_size];
    for (int idx = 0; idx < batch_size; ++idx)
        bias_multiplier[idx] = 1.0;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch_size, output_channels, input_channels, 1.0,
                bias_multiplier, input_channels, bias->c_ptr(), output_channels, 1.0, output->c_ptr(), output_channels);

    delete[] bias_multiplier;
}
