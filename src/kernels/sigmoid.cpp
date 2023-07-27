// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "kernels/sigmoid.h"

void VPUNN::Sigmoid(VPUNN::Tensor<float>* output) {
    for (auto idx = 0; idx < output->size(); idx++) {
        float exp_x = std::exp(-(*output)[idx]);
        (*output)[idx] = 1 / (1 + exp_x);
    }
}
