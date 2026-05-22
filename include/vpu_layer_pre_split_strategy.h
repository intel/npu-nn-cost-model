// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_LAYER_PRE_SPLIT_STRATEGY_H
#define VPUNN_LAYER_PRE_SPLIT_STRATEGY_H

#include <iostream>
#include <optional>
#include <string>

#include "vpu/vpu_layer_execution_config.h"
#include "vpu/vpu_tiling_strategy.h"
#include "vpu/dpu_defaults.h"

namespace VPUNN {

/// @brief A subset of VPU layer strategy information, focused on pre-split layers where tiling strategy may be less relevant
struct VPULayersPreSplitStrategy : public VPULayerExecutionConfig {
    unsigned int nDPUs{1};                               ///< Number of DPUs per tile
    std::optional<VPUTilingStrategy> tiling_strategy{};  ///< tiling strategy

    explicit VPULayersPreSplitStrategy(unsigned int nDPUs = 1, std::optional<VPUTilingStrategy> tiling_strategy = std::nullopt,
                                   bool input_fetching = false, bool output_spilling = false, bool prefetching = true)
        : VPULayerExecutionConfig{input_fetching, output_spilling, prefetching},
          nDPUs(nDPUs), tiling_strategy(tiling_strategy) {}
};

inline std::ostream& operator<<(std::ostream& stream, const VPULayersPreSplitStrategy& d) {
    stream << "\nVPULayersPreSplitStrategy : \n"
           << " n DPUs: \t" << d.nDPUs << "\n"
           << " Tiling Strategy: \t" << (d.tiling_strategy.has_value() ? std::to_string(static_cast<int>(d.tiling_strategy.value())) : "N/A") << " : "
           << (d.tiling_strategy.has_value() ? VPUTilingStrategy_ToText.at(static_cast<int>(d.tiling_strategy.value())) : "N/A") << " ;\n"
           << " input_fetching : \t" << (int)d.input_fetching << " : " << (d.input_fetching ? "true" : "false")
           << " ;\n"  //
           << " output_spilling: \t" << (int)d.output_spilling << " : " << (d.output_spilling ? "true" : "false")
           << " ;\n"                                                                                                 //
           << " prefetching    : \t" << (int)d.prefetching << " : " << (d.prefetching ? "true" : "false") << " ;\n"  //
           << out_terminator() << "VPULayersPreSplitStrategy "  // terminator
            ;
    return stream;
}

} // namespace VPUNN

#endif
