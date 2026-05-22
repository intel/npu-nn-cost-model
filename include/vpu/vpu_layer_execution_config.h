// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_VPU_LAYER_EXECUTION_CONFIG_H
#define VPUNN_VPU_LAYER_EXECUTION_CONFIG_H

namespace VPUNN {

/// @brief General execution strategy for DPU 
struct VPULayerExecutionConfig {
    bool input_fetching{false};   ///< true if the layer input is in DDR
    bool output_spilling{false};  ///< true if the layer output is in DDR
    bool prefetching{true};       ///< If layer parameters are prefetched with previous layers. If true it considers the
                                  ///< weights are prefetched, if false will fetch the weights considering also sparsity
};

} // namespace VPUNN

#endif
