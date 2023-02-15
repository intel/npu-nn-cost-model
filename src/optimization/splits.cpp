// Copyright © 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/optimization/tiler.h"

namespace VPUNN {

DPUWorkload createTile(const DPULayer& layer, const std::array<unsigned int, 4>& new_shape,
                       const std::array<unsigned int, 4>& offsets = {0, 0, 0, 0}) {
    auto dtype = layer.outputs[0].get_dtype();
    auto layout = layer.outputs[0].get_layout();
    auto outTile = VPUTensor(new_shape, dtype, layout);
    DPUWorkload split = DPUWorkload(layer);
    split.outputs[0] = outTile;
    split.offsets = offsets;
    return split;
}

DPUWorkload createTileHW(const DPULayer& layer, const unsigned int width, const unsigned int height,
                         const unsigned int offset_width, const unsigned int offset_height) {
    return createTile(layer, {width, height, layer.outputs[0].z(), 1}, {offset_width, offset_height, 0, 0});
}

DPUWorkload createTileZ(const DPULayer& layer, const unsigned int channels, const unsigned int offset_channels) {
    return createTile(layer, {layer.outputs[0].x(), layer.outputs[0].y(), channels, 1}, {0, 0, offset_channels, 0});
}

bool isValidZ(unsigned int channels, std::vector<unsigned int>& validZTiles) {
    if (validZTiles.size() == 0)
        return true;
    return std::find(validZTiles.begin(), validZTiles.end(), channels) != validZTiles.end();
}

void splitOverZ(const DPULayer& layer, std::list<DPUWorkloads>& splitPool, const ExecutionMode mode,
                const unsigned int nWorkloads, std::vector<unsigned int>& validZTiles) {
    DPUWorkloads workloads;
    const auto gridSize = mpe_mode_to_grid(mode);

    if ((layer.outputs[0].z() < gridSize[Dim::Act::Z]) || (layer.outputs[0].z() % gridSize[Dim::Act::Z] != 0))
        return;

    auto channels = layer.outputs[0].z();
    auto max_Z = round_up(ceil_division(channels, nWorkloads), gridSize[Dim::Act::Z]);

    for (unsigned int idx = 0; idx < nWorkloads; idx++) {
        auto actual_channels = channels > max_Z ? max_Z : channels;
        auto offset_channels = idx * max_Z;

        if (actual_channels % gridSize[Dim::Act::Z] != 0) {
            return;
        }

        // Invalid split
        if (!isValidZ(actual_channels, validZTiles))
            return;

        workloads.emplace_back(createTileZ(layer, actual_channels, offset_channels));

        channels -= actual_channels;
    }

    Tiler::setWorkloadsMode(workloads, mode, layer);
    splitPool.push_back(workloads);
}

void splitOverHW(const DPULayer& layer, std::list<DPUWorkloads>& splitPool, const unsigned int widthFactor,
                 const unsigned int heightFactor, const ExecutionMode mode) {
    DPUWorkloads workloads;
    auto shape = layer.outputs[0].get_shape();

    const auto gridSize = mpe_mode_to_grid(mode);

    const auto width = shape[Dim::Act::X];
    const auto height = shape[Dim::Act::Y];

    // Compute max width and height considering the grid
    const auto maxWidth = round_up(ceil_division(width, widthFactor), gridSize[Dim::Act::X]);
    const auto maxHeight = round_up(ceil_division(height, heightFactor), gridSize[Dim::Act::Y]);

    // And the actual split numbers
    const auto actualWidthSplitsNum = ceil_division(width, maxWidth);
    auto actualHeightSplitsNum = ceil_division(height, maxHeight);

    auto remainedHeight = height;
    for (unsigned int idx = 0; idx < actualHeightSplitsNum; idx++) {
        auto currentHeightStep = remainedHeight > maxHeight ? maxHeight : remainedHeight;

        auto remainedWidth = width;
        for (unsigned int idy = 0; idy < actualWidthSplitsNum; idy++) {
            // Create a new output tile tensor from the original one
            auto tile_width = remainedWidth > maxWidth ? maxWidth : remainedWidth, tile_height = currentHeightStep;
            auto offset_width = idy * maxWidth, offset_height = idx * maxHeight;

            // Generate the workload and assign the new shape
            remainedWidth -= tile_width;

            // Generate the split
            workloads.emplace_back(createTileHW(layer, tile_width, tile_height, offset_width, offset_height));
        }
        remainedHeight -= currentHeightStep;
    }

    Tiler::setWorkloadsMode(workloads, mode, layer);
    splitPool.push_back(workloads);
}

}  // namespace VPUNN
