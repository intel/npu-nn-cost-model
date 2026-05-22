// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/optimization/ztiling.h"
#include "vpu/optimization/tiling_primitives.h"

#include <algorithm>
#include <cmath>
#include <list>
#include <set>
#include <utility>
#include <vector>

#include "vpu/cycles_interface_types.h"
#include "vpu/dim_enum.h"
#include "vpu/dpu_types.h"
#include "vpu/utils.h"

namespace VPUNN {

std::set<unsigned int> ZTiling::generateSplitPool(const unsigned int numDPU,
                                                  const ExecutionMode& valid_execution_mode) const {
    if (numDPU == 1 && TilingPrimitives::isNoSplitOperation(this->layer_on_tile.op)) {
        return {1U};  // why? CONV and ELEMwisae are not split?
    }

    // Enable ZTiling only for VPU2.0 in vector mode for non conv layers
    // This is a VPUX specific behavior we need to keep
    if ((this->layer_on_tile.device == VPUDevice::VPU_2_0) &&
        ((valid_execution_mode != ExecutionMode::VECTOR) &&
         (this->layer_on_tile.op != VPUNN::Operation::CONVOLUTION))) {
        return {1U};
    }

    // The max number of splits in the Z dimension
    std::vector<unsigned int> maxSplitsInZ;
    {
        std::vector<unsigned int> validZTiles;
        // 2^4 equals to the CMX word size in bytes,  2^8 is an up bound to limit the number of splits
        for (unsigned int i = MIN_VALID_ZTILE_EXPONENT; i < MAX_VALID_ZTILE_EXPONENT; ++i) {
            validZTiles.push_back(static_cast<unsigned int>(std::pow(2, i)));
            validZTiles.push_back(validZTiles.back() + DEFAULT_ZTILE_VALUE);
        }

        for (const auto& zTile : validZTiles) {
            maxSplitsInZ.push_back(
                    static_cast<unsigned int>(std::ceil(layer_on_tile.outputs[0].z() / static_cast<double>(zTile))));
        }
    }

    const auto maxZ = *std::max_element(maxSplitsInZ.begin(), maxSplitsInZ.end());
    const auto maxSplits = std::min(maxWorkloads, maxZ);

    std::set<unsigned int> dpuMulSplits;  //< unique values, sorted!
    for (auto i = numDPU; i < maxSplits + 1U; i += numDPU) {
        dpuMulSplits.insert(i);
    }
    for (auto splitsZ : maxSplitsInZ) {
        auto zRanges = TilingPrimitives::getSplitsFromRange(splitsZ, maxSplits);
        dpuMulSplits.insert(zRanges.begin(), zRanges.end());
    }
    dpuMulSplits.insert(1U);

    return dpuMulSplits;
}

void ZTiling::tileMultipleWl(std::list<DPUWorkloadsWithCyclesSplit>& splitPool, const ExecutionMode mode,
                             const unsigned int nWorkloads) {
    if (layer_on_tile.device < VPUDevice::NPU_5_0  //
        || (force_LegacyZTiling)                   // Every device runs as before
    ) {                                            // Some layers have a max size in Z by specification
        const std::vector<unsigned int> validZTiles{
                TilingPrimitives::requireMaxZTile(layer_on_tile)
                                                            ? getValidIntraTilesChannelsOptions(layer_on_tile.device)
                                                                      .transformSmartRangetoVector<unsigned int>()
                        : std::vector<unsigned int>({})};

        splitOverZ(layer_on_tile, splitPool, mode, nWorkloads, validZTiles);
    } else {  // experimental for new devices
        const SplitDimension splitter{
                TilingPrimitives::requireMaxZTile(layer_on_tile)
                        ? getValidIntraTilesChannelsOptions(layer_on_tile.device)
                                                                 : SmartRanges{1, 8192, 16}  // div 16
        };
        splitInNOverZ(layer_on_tile, splitPool, mode, nWorkloads, splitter);
    }
}

void ZTiling::splitInNOverZ(const DPULayer& layer, std::list<DPUWorkloadsWithCyclesSplit>& splitPool,
                            const ExecutionMode mode, const unsigned int nWorkloads, const SplitDimension& theSpliter) {
    SplitDimension::SplitContainer split_bins{};  // empty
    const int dimensionToSplit{static_cast<int>(layer.outputs[0].channels())};
    const bool split_status = theSpliter.divideBalanced(dimensionToSplit, nWorkloads, split_bins);

    if (!split_status) {
        return;  // no valid split
    }
    if (split_bins.size() != nWorkloads) {
        return;  // no valid split, ERROR in splitter
    }

    DPUWorkloadsWithCyclesSplit workloads_split;
    int offset_channels{0};  // used/incremented  in  loop
    for (const auto& split : split_bins) {
        const auto actual_channels{split};

        if (actual_channels <= 0) {
            return;  // invalid split
        }

        workloads_split.workloads.emplace_back(
                createTileZ(layer, static_cast<unsigned>(actual_channels), static_cast<unsigned>(offset_channels)));
        workloads_split.cycles.emplace_back(Cycles::NO_ERROR);

        offset_channels += static_cast<unsigned>(actual_channels);  // for next iteration
    }

    ITilerAlgorithm::setWorkloadsModeAndInfereInputShape(workloads_split, mode,
                                                         layer);  // computes also input tensor
    splitPool.push_back(std::move(workloads_split));
}

DPUWorkload ZTiling::createTileZ(const DPULayer& layer, const unsigned int channels,
                                 const unsigned int offset_channels) {
    DPUWorkload wl{TilingPrimitives::createIncompleteTile(
            layer, {layer.outputs[0].x(), layer.outputs[0].y(), channels, 1}, {0, 0, offset_channels, 0})};
    // handle halo for intra-tiles. OWT is ignored here, in case some OWT broadcast exists, will override this info?
    // what happens in case of owt?
    {
        const HaloWorkload& tileHalo{layer.halo};
        // @offset_channels: represent the number of channels that was produced by previous intra-tiles (before current
        // one)
        wl.halo.output_0_inbound_halo.front = tileHalo.output_0_inbound_halo.front + offset_channels;

        // @channels: is the number of channels produced by this intra-tile
        // @layer.outputs[0].channels(): is the number of channels of this entire tile
        const int remaining_channels_to_be_produced_after_current_tile =
                layer.outputs[0].channels() - channels - offset_channels;

        wl.halo.output_0_inbound_halo.back =
                tileHalo.output_0_inbound_halo.back + remaining_channels_to_be_produced_after_current_tile;
    }
    return wl;
}

bool ZTiling::isValidZ(unsigned int channels, const std::vector<unsigned int>& validZTiles) {
    if (validZTiles.size() == 0)
        return true;
    return std::find(validZTiles.begin(), validZTiles.end(), channels) != validZTiles.end();
}

// @todo redesign at least here, if not also above!
void ZTiling::splitOverZ(const DPULayer& layer, std::list<DPUWorkloadsWithCyclesSplit>& splitPool,
                         const ExecutionMode mode, const unsigned int nWorkloads,
                         const std::vector<unsigned int>& validZTiles) {
    DPUWorkloadsWithCyclesSplit workloads_split;
    const auto gridSize = mpe_mode_to_grid(mode);
    const auto gridSize_Z = gridSize[Dim::Act::Z];  // typically 16?

    const auto all_channels = layer.outputs[0].z();
    if ((all_channels < gridSize_Z)            // smaller
        || ((all_channels % gridSize_Z) != 0)  // must be a multiple of grid
    )
        return;  // nothing , cannot split

    const auto max_Z = round_up(ceil_division(all_channels, nWorkloads), gridSize_Z);

    auto channels = all_channels;  // channels remaining to split
    for (unsigned int idx = 0; idx < nWorkloads; idx++) {
        const auto actual_channels{(channels > max_Z) ? max_Z : channels};
        const auto offset_channels{idx * max_Z};

        if ((actual_channels % gridSize_Z) != 0) {
            return;  // error , not a multiple
        }

        // Invalid split
        if (!isValidZ(actual_channels, validZTiles))  // can be zero before nWorkloads reached.
            return;

        workloads_split.workloads.emplace_back(createTileZ(layer, actual_channels, offset_channels));
        workloads_split.cycles.emplace_back(Cycles::NO_ERROR);

        channels -= actual_channels;
    }
    ITilerAlgorithm::setWorkloadsModeAndInfereInputShape(workloads_split, mode, layer);  // computes also input tensor
    splitPool.push_back(std::move(workloads_split));
}

}  // namespace VPUNN