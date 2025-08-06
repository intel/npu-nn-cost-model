// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_LAYER_SPLIT_INFO_H
#define VPUNN_LAYER_SPLIT_INFO_H

#include <cmath>
#include "core/logger.h"
#include "vpu/types.h"
#include "vpu/utils.h"

#include "vpu/vpu_tiling_strategy.h"
#include "vpu/layer.h" //for DPULayer 

namespace VPUNN {


struct DMA_CyclesInfo {
    CyclesInterfaceType cycles{Cycles::NO_ERROR};  ///< cycles
    // pipelined? y/n or 0,1,2,3
};
struct DMALayerInfo {
    DMA_CyclesInfo w_tensor{};
    DMA_CyclesInfo input_tensor{};
    DMA_CyclesInfo output_tensor{};
};

/// container of workloads
using DPUWorkloads = std::vector<DPUWorkload>;

///  describes a pair of cost and the associated DPUWorkloads.
/// the cost normally represents the runtime of the workloads sequence on a tile, considering also pipelining on nDPUs
/// (not mentioned here)
using DPUWorkloadsCost = std::pair<CyclesInterfaceType, DPUWorkloads>;  ///> interface VPUX

/// container of DPUWorkload (order is relevant). Normally it stores the DPUworkloads associated to a tile plus their
/// predicted  runtime
struct DPUWorkloadsWithCyclesSplit {
    std::vector<CyclesInterfaceType> cycles{};
    std::vector<DPUWorkload> workloads{};
};

using DPUWorkloadsWithCycleCost = std::pair<CyclesInterfaceType, DPUWorkloadsWithCyclesSplit>;  ///>internal

/// details about a tile split strategy
struct OneTileLayerInfo {
    DPULayer inter_tile_split_layer{};  ///<  layer resulted by splitting the orginalLayer to one tile using requested
                                        ///<  strategy
    DPUWorkloadsCost best_intra_tile_split{Cycles::NO_ERROR, {}};  ///< the cost and list of workloads that were
                                                                   ///< inferred to be the best after
    ///< performing the intra-tile split algorithm
    std::vector<DPUWorkloadsWithCyclesSplit>
            all_intra_tile_splits{};  ///< all intra tile splits generated. one pair() is a split

    DMALayerInfo DMA_info{};  //< layers detailed DMA info (zero if not requested)
};

/// info on how are the splits on each tile
/// For each tile a OneTileLayerInfo is allocated.
using LayerSplitInfo = std::vector<OneTileLayerInfo>;


}  // namespace VPUNN

#endif  // VPUNN_LAYER_SPLIT_INFO_H
