// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_DMA_COST_MODEL_VARIANT_H
#define VPUNN_DMA_COST_MODEL_VARIANT_H


#include <variant>

#include "vpu_dma_cost_model.h"
#include "vpu/dma_types.h"

namespace VPUNN {

/// Variant used to hold any available DMACostModel<> - Currently DMACostModel<DMANNWorkload_NPU27> and
/// DMACostModel<DMANNWorkload_NPU40_50> The ownership is not transferred, the client must ensure the lifetime of the
/// model
using DMACostModelVariant = std::variant<DMACostModel<DMANNWorkload_NPU27>*, DMACostModel<DMANNWorkload_NPU40_50>*>;



}  // namespace VPUNN

#endif  // VPUNN_DMA_COST_MODEL_VARIANT_H
