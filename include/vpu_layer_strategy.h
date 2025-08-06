// Copyright © 2024 Intel Corporation
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

#include "vpu/types.h"
#include "vpu/dpu_defaults.h"
#include "vpu/vpu_tiling_strategy.h"

namespace VPUNN {

/// @brief A VPU layer strategy
struct VPULayerStrategy {
    unsigned int nDPUs{1};   ///< Number of DPUs per tile
    unsigned int nSHVs{1};   ///< Number of Shaves per tile
    unsigned int nTiles{1};  ///< Number of tiles

    VPUTilingStrategy tiling_strategy{VPUTilingStrategy::NONE};  ///< tiling strategy

    bool input_fetching{false};   ///< true if the layer input is in DDR
    bool output_spilling{false};  ///< true if the layer output is in DDR

    bool prefetching{true};  ///< If layer parameters are prefetched with previous layers. If true it considers the
                             ///< weights are prefetched, if false will fetch the weights considering also sparsity
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
