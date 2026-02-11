// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/optimization/splits.h"
#include "vpu/optimization/tiler.h"

namespace VPUNN {

/// @brief creates a Workload starting from layer. modifies the output tensor and shape only, input tensor remains the
/// same, kernels remain the same. This stage does not contain a valid Input output correlation.
static DPUWorkload createIncompleteTile(const DPULayer& layer, const std::array<unsigned int, 4>& new_shape,
                                        const std::array<unsigned int, 4>& offsets = {0, 0, 0, 0}) {
    const auto outTile = VPUTensor(new_shape, layer.outputs[0]);  // new shape rest is the same
    DPUWorkload split(layer);
    split.outputs[0] = outTile;
    split.offsets = offsets;
    return split;
}

DPUWorkload createTileHW(const DPULayer& layer, const unsigned int width, const unsigned int height,
                         const unsigned int offset_width, const unsigned int offset_height, HaloWorkload halo) {
    DPUWorkload wl{
            createIncompleteTile(layer, {width, height, layer.outputs[0].z(), 1}, {offset_width, offset_height, 0, 0})};

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

DPUWorkload createTileZ(const DPULayer& layer, const unsigned int channels, const unsigned int offset_channels) {
    DPUWorkload wl{createIncompleteTile(layer, {layer.outputs[0].x(), layer.outputs[0].y(), channels, 1},
                                        {0, 0, offset_channels, 0})};
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

static bool isValidZ(unsigned int channels, const std::vector<unsigned int>& validZTiles) {
    if (validZTiles.size() == 0)
        return true;
    return std::find(validZTiles.begin(), validZTiles.end(), channels) != validZTiles.end();
}

// @todo redesign at least here, if not also above!
void splitOverZ(const DPULayer& layer, std::list<DPUWorkloadsWithCyclesSplit>& splitPool, const ExecutionMode mode,
                const unsigned int nWorkloads, const std::vector<unsigned int>& validZTiles) {
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

void splitOverHW(const DPULayer& layer, std::list<DPUWorkloadsWithCyclesSplit>& splitPool,
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

}  // namespace VPUNN
