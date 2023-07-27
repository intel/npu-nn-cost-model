// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "kernels/bias.h"
#include "kernels/vpunn_blas.h"

namespace VPUNN {

void BiasOp::Bias(const VPUNN::Tensor<float>* bias, VPUNN::Tensor<float>* output) const {
    // Use cblas_sgemm to compute C <- alpha A * B + beta C

    const int batch_size = output->shape()[0];
    const int output_channels = output->shape()[1];

    constexpr int input_channels = 1;

    // use bias multiplier for batching,  should be preallocated and all 1.0
    const float* bias_multiplier = batch_buffer.data();

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch_size, output_channels, input_channels, 1.0,
                bias_multiplier, input_channels, bias->c_ptr(), output_channels, 1.0, output->data(), output_channels);
}

}  // namespace VPUNN
