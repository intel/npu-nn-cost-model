// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_LAYER_STRATEGY_H
#define VPUNN_LAYER_STRATEGY_H

#include <iostream>
#include <optional>
#include <string>

#include "vpu/dpu_defaults.h"
#include "vpu/vpu_tiling_strategy.h"
#include "vpu/vpu_layer_execution_config.h"

namespace VPUNN {

/// @brief A VPU layer strategy with additional information on tiling and SHV distribution, used for non pre-split layers where the tiling strategy is relevant
struct VPULayerStrategy : public VPULayerExecutionConfig {
    unsigned int nDPUs{1};   ///< Number of DPUs per tile
    unsigned int nSHVs{1};   ///< Number of Shaves per tile
    unsigned int nTiles{1};  ///< Number of tiles

    VPUTilingStrategy tiling_strategy{VPUTilingStrategy::NONE};  ///< tiling strategy
    
    explicit VPULayerStrategy(unsigned int nDPUs = 1, unsigned int nSHVs = 1, unsigned int nTiles = 1,
                     VPUTilingStrategy tiling_strategy = VPUTilingStrategy::NONE,
                     bool input_fetching = false, bool output_spilling = false, bool prefetching = true)
        : VPULayerExecutionConfig{input_fetching, output_spilling, prefetching},
          nDPUs(nDPUs), nSHVs(nSHVs), nTiles(nTiles), tiling_strategy(tiling_strategy) {}
};

// struct LayerMetaInfo {         // info about the layer/contextual info (not content oflayer)
//     std::string layer_name{};  ///< The name of the Layer (if available) - as depicted in compiler graph
//     std::string info{};        ///< The name of the compiler pass that generated this layer
// };

inline std::ostream& operator<<(std::ostream& stream, const VPULayerStrategy& d) {
    stream << "\nVPULayerStrategy : \n"
           << " n DPUs: \t" << d.nDPUs << "\n"
           << " n SHVs: \t" << d.nSHVs << "\n"
           << " n Tiles: \t" << d.nTiles << "\n"
           << " Tiling Strategy: \t" << (int)d.tiling_strategy << " : "
           << VPUTilingStrategy_ToText.at(static_cast<int>(d.tiling_strategy)) << " ;\n"
           << " input_fetching : \t" << (int)d.input_fetching << " : " << (d.input_fetching ? "true" : "false")
           << " ;\n"  //
           << " output_spilling: \t" << (int)d.output_spilling << " : " << (d.output_spilling ? "true" : "false")
           << " ;\n"                                                                                                 //
           << " prefetching    : \t" << (int)d.prefetching << " : " << (d.prefetching ? "true" : "false") << " ;\n"  //
           << out_terminator() << "VPULayerStrategy "  // terminator
            ;
    return stream;
}

}  // namespace VPUNN

#endif  // VPUNN_LAYER_STRATEGY_H
