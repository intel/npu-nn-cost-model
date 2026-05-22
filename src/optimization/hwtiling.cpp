// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/optimization/hwtiling.h"
#include "vpu/optimization/tiling_primitives.h"

#include <algorithm>
#include <cmath>
#include <list>
#include <set>
#include <utility>
#include <vector>

#include "vpu/cycles_interface_types.h"
#include "vpu/dim_enum.h"
#include "vpu/utils.h"

namespace VPUNN {

std::set<unsigned int> HWTiling::generateSplitPool(const unsigned int numDPU,
                                                   const ExecutionMode& valid_execution_mode) const {
    std::set<unsigned int> dpuMulSplits{1};  //< unique values, sorted!
    if (numDPU == 1) {
        return dpuMulSplits;
    }
    // Get the min grid size from valid MPE grid size
    const auto grid = mpe_mode_to_grid(valid_execution_mode);
    const unsigned int grid_x{grid[Dim::Grid::W]};
    const unsigned int grid_y{grid[Dim::Grid::H]};

    std::vector<unsigned int> maxSplitsInXY;
    maxSplitsInXY.push_back(static_cast<unsigned int>(std::ceil(layer_on_tile.outputs[0].y() / grid_x) *
                                                      std::ceil(layer_on_tile.outputs[0].x() / grid_y)));

    const auto maxXY = *std::max_element(maxSplitsInXY.begin(), maxSplitsInXY.end());
    const auto maxSplits = std::min(maxWorkloads, maxXY);

    for (auto i = numDPU; i < maxSplits + 1; i = i + numDPU) {
        dpuMulSplits.insert(static_cast<uint32_t>(i));
    }

    for (auto splitsXY : maxSplitsInXY) {
        const auto xyRanges = TilingPrimitives::getSplitsFromRange(splitsXY, maxSplits);
        dpuMulSplits.insert(xyRanges.begin(), xyRanges.end());
    }

    return dpuMulSplits;
}

void HWTiling::tileMultipleWl(std::list<DPUWorkloadsWithCyclesSplit>& splitPool, const ExecutionMode mode,
                              const unsigned int nWorkloads) {
    // Get each pair of factor of nWorkloads (largest, smallest)
    for (const auto& factor : getFactors(nWorkloads)) {
        // Map factor.first , factor.second -> width, height
        if (factor.first <= layer_on_tile.outputs[0].x() && factor.second <= layer_on_tile.outputs[0].y()) {
            tileOverHW(splitPool, factor.first, factor.second, mode);
        }
    }
}

void HWTiling::tileOverHW(std::list<DPUWorkloadsWithCyclesSplit>& splitPool, const unsigned int widthFactor,
                          const unsigned int heightFactor, const ExecutionMode mode) {
    splitOverHW(layer_on_tile, splitPool, widthFactor, heightFactor, mode);
}

std::list<std::pair<unsigned int, unsigned int>> HWTiling::getFactors(unsigned int n) {
    std::list<std::pair<unsigned int, unsigned int>> factors;
    for (unsigned int i = 1; i <= sqrt(n); i++) {
        if (n % i == 0) {
            factors.emplace_back(n / i, i);  // larger, smaller
            factors.emplace_back(i, n / i);  // smaller, larger
        }
    }
    return factors;
}

void HWTiling::splitOverHW(const DPULayer& layer, std::list<DPUWorkloadsWithCyclesSplit>& splitPool,
                           const unsigned int widthFactor, const unsigned int heightFactor, const ExecutionMode mode) {
    DPUWorkloadsWithCyclesSplit workloads_split;

    const auto gridSize = mpe_mode_to_grid(mode);

    const auto width = layer.outputs[0].width();
    const auto height = layer.outputs[0].height();

    // Compute max width and height considering the grid
    const auto maxWidth = round_up(ceil_division(width, widthFactor), gridSize[Dim::Act::X]);
    const auto maxHeight = round_up(ceil_division(height, heightFactor), gridSize[Dim::Act::Y]);

    // And the actual split numbers
    const auto actualWidthSplitsNum = ceil_division(width, maxWidth);
    const auto actualHeightSplitsNum = ceil_division(height, maxHeight);
    const auto& halo{layer.halo};

    auto remainedHeight = height;
    for (unsigned int idx = 0; idx < actualHeightSplitsNum; idx++) {
        const auto currentHeightStep = remainedHeight > maxHeight ? maxHeight : remainedHeight;

        HaloWorkload halo_now{halo};  // full pass down

        {
            if (idx == 0)  // first
            {
                halo_now.input_0_halo.top = halo.input_0_halo.top;
                halo_now.output_0_halo.top = halo.output_0_halo.top;
                if (halo_now.output_0_halo.top > (int)currentHeightStep) {  // error
                    halo_now.output_0_halo.top = (int)currentHeightStep;
                }
            } else {
                halo_now.input_0_halo.top = 0;
                halo_now.output_0_halo.top = 0;
                halo_now.output_0_halo_broadcast_cnt.top = 0;
            }

            if (idx == (actualHeightSplitsNum - 1))  // last
            {
                halo_now.input_0_halo.bottom = halo.input_0_halo.bottom;
                halo_now.output_0_halo.bottom = halo.output_0_halo.bottom;
                if (halo_now.output_0_halo.bottom > (int)currentHeightStep) {  // error
                    halo_now.output_0_halo.bottom = (int)currentHeightStep;
                }
            } else {
                halo_now.input_0_halo.bottom = 0;
                halo_now.output_0_halo.bottom = 0;
                halo_now.output_0_halo_broadcast_cnt.bottom = 0;
            }
        }

        auto remainedWidth = width;
        for (unsigned int idy = 0; idy < actualWidthSplitsNum; idy++) {
            // Create a new output tile tensor from the original one
            const auto tile_width = remainedWidth > maxWidth ? maxWidth : remainedWidth;

            {
                halo_now.input_0_halo.left = 0;
                halo_now.output_0_halo.left = 0;
                if (idy == 0)  // first
                {
                    halo_now.input_0_halo.left = halo.input_0_halo.left;
                    halo_now.output_0_halo.left = halo.output_0_halo.left;
                    if (halo_now.output_0_halo.left > (int)tile_width) {  // error
                        halo_now.output_0_halo.left = (int)tile_width;
                    }
                } else {
                    halo_now.input_0_halo.left = 0;
                    halo_now.output_0_halo.left = 0;
                    halo_now.output_0_halo_broadcast_cnt.left = 0;
                }

                if (idy == (actualWidthSplitsNum - 1))  // last
                {
                    halo_now.input_0_halo.right = halo.input_0_halo.right;
                    halo_now.output_0_halo.right = halo.output_0_halo.right;
                    if (halo_now.output_0_halo.right > (int)tile_width) {  // error
                        halo_now.output_0_halo.right = (int)tile_width;
                    }
                } else {
                    halo_now.input_0_halo.right = 0;
                    halo_now.output_0_halo.right = 0;
                    halo_now.output_0_halo_broadcast_cnt.right = 0;
                }
            }

            const auto tile_height = currentHeightStep;
            const auto offset_width = idy * maxWidth;
            const auto offset_height = idx * maxHeight;

            remainedWidth -= tile_width;

            // Generate the workload and assign the new shape
            workloads_split.workloads.emplace_back(
                    createTileHW(layer, tile_width, tile_height, offset_width, offset_height, halo_now));
            workloads_split.cycles.emplace_back(Cycles::NO_ERROR);
        }
        remainedHeight -= currentHeightStep;
    }

    ITilerAlgorithm::setWorkloadsModeAndInfereInputShape(
            workloads_split, mode, layer);  // computes also input tensor, and halo inoput sanitisation
    splitPool.push_back(std::move(workloads_split));
}

DPUWorkload HWTiling::createTileHW(const DPULayer& layer, const unsigned int width, const unsigned int height,
                                   const unsigned int offset_width, const unsigned int offset_height,
                                   HaloWorkload halo) {
    DPUWorkload wl{TilingPrimitives::createIncompleteTile(layer, {width, height, layer.outputs[0].z(), 1},
                                                          {offset_width, offset_height, 0, 0})};

    const HaloWorkload& tileHalo{layer.halo};

    ///////////////////////////// HALO for Height /////////////////////////////

    // @offset_height: represent the number of rows that was produced by previous intra-tiles (before current
    // one)
    halo.output_0_inbound_halo.top = tileHalo.output_0_inbound_halo.top + offset_height;

    // @height: is the number of rows produced by this intra-tile
    // @layer.outputs[0].height(): is the number of rows of this entire tile
    const int remaining_rows_to_be_produced_after_current_tile = layer.outputs[0].height() - height - offset_height;
    halo.output_0_inbound_halo.bottom =
            tileHalo.output_0_inbound_halo.bottom + remaining_rows_to_be_produced_after_current_tile;

    ///////////////////////////// HALO for Width /////////////////////////////
    // @offset_width: represent the number of columns that was produced by previous intra-tiles (before current
    // one)
    halo.output_0_inbound_halo.left = tileHalo.output_0_inbound_halo.left + offset_width;

    // @width: is the number of columns produced by this intra-tile
    // @layer.outputs[0].width(): is the number of columns of this entire tile
    const int remaining_columns_to_be_produced_after_current_tile = layer.outputs[0].width() - width - offset_width;
    halo.output_0_inbound_halo.right =
            tileHalo.output_0_inbound_halo.right + remaining_columns_to_be_produced_after_current_tile;

    wl.halo = halo;

    return wl;
}

}  // namespace VPUNN