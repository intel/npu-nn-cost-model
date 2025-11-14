// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.


#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>  // for SWOperation implementations . SHAVE v1
#include <vpu_dma_cost_model.h>

template class VPUNN::DMACostModel<VPUNN::DMANNWorkload_NPU27>;  ///< explicit instantiation 
template class VPUNN::DMACostModel<VPUNN::DMANNWorkload_NPU40_RESERVED>;  ///< explicit instantiation 
