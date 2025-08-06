// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_SPLITS_H
#define VPUNN_SPLITS_H

#include <list>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "vpu/layer.h"
// #include "vpu/optimization/workload_optimization_types.h"
#include "vpu/types.h"
#include "vpu/utils.h"

#include "vpu/layer_split_info.h"

namespace VPUNN {

/**
 * @brief Split a DPULayer over the Z dimension, appending the result to the splitPool list
 *
 * HALO ASPECTS: all HW direction halos are passed down to split
 * Output_inbound halo for Z/K  direction technically should be adjusted to fit always the full layer memory, but this
 * is not really necessary for the DPUWorload level because is not used for NN, just for fit to memory check. If the
 * layer fits to memory the intra tile split will also fit.
 * Decision(for now) is to pass and leave unaltered also the output_inbound_halo Z direction.
 *
 * @param layer a DPULayer to be split
 * @param splitPool [out] the pool of valid split
 * @param mode the valid ExecutionMode
 * @param nWorkloads number of splits to apply
 * @param validZTiles valid Z dimension for the splits
 */
void splitOverZ(const DPULayer& layer, std::list<DPUWorkloadsWithCyclesSplit>& splitPool, const ExecutionMode mode,
                const unsigned int nWorkloads, const std::vector<unsigned int>& validZTiles);

/**
 * @brief Split a DPULayer over the H and W dimensions, appending the result to the splitPool list
 *
 * HALO ASPECTS: @todo:
 * Pass halo to intra tile splits. Especially edge tiles are influenced by halo. Important are what NN uses and 2nd the
 * memory aspects
 * Input halo influences NN up to v2.7. To be ported if positive to edge tiles.
 * Output halo + cnt influences NN v4.0 and onwards. To be ported to edge tiles (+ inner tiles if larger than edge tile)
 * output_inbound halo is not used by NN, only for memory, if left unchanged increases memory but the layer was already
 * checked to fit in memory.
 *
 * @param layer a DPULayer
 * @param splitPool the pool of valid split
 * @param widthFactor the number of splits in the X dimension
 * @param heightFactor  the number of splits in the Y dimension
 * @param mode the valid ExecutionMode
 */
void splitOverHW(const DPULayer& layer, std::list<DPUWorkloadsWithCyclesSplit>& splitPool,
                 const unsigned int widthFactor, const unsigned int heightFactor, const ExecutionMode mode);

DPUWorkload createTileHW(const DPULayer& layer, const unsigned int width, const unsigned int height,
                         const unsigned int offset_width, const unsigned int offset_height, HaloWorkload halo);

DPUWorkload createTileZ(const DPULayer& layer, const unsigned int channels, const unsigned int offset_channels);

}  // namespace VPUNN

#endif  // VPUNN_TILER_H
