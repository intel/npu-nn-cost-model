// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/optimization/tiling_primitives.h"
#include "vpu/utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "vpu/dim_enum.h"
#include "vpu/dpu_types.h"

namespace VPUNN {

DPUWorkload TilingPrimitives::createIncompleteTile(const DPULayer& layer, const std::array<unsigned int, 4>& new_shape,
                                                   const std::array<unsigned int, 4>& offsets) {
    const auto outTile = VPUTensor(new_shape, layer.outputs[0]);  // new shape rest is the same
    DPUWorkload split(layer);
    split.outputs[0] = outTile;
    split.offsets = offsets;
    return split;
}

bool TilingPrimitives::requireMaxZTile(const DPULayer& layer) {
    if (layer.device >= VPUDevice::VPU_2_7) {
        if (is_dwconv_family_operation(layer.op)) {
            return true;
        }
    }
    return false;
}

bool TilingPrimitives::isNoSplitOperation(Operation op) {
    return (op == Operation::CONVOLUTION        //
            || op == Operation::ELTWISE         //
            || op == Operation::CM_CONVOLUTION  //
            || op == Operation::LAYER_NORM      //
            || op == Operation::ELTWISE_MUL);
}

bool TilingPrimitives::isHWTilingAllowed(const DPULayer& layer, const SplitOptions& options) {
    const auto& strategies = options.availableStrategies;
    if (options.nDPU == 1 &&
        std::find(strategies.begin(), strategies.end(), VPUSplitStrategy::Z_TILING) != strategies.end()) {
        return false;  // WHY? ONly K tiling allowed on one DPU, no parallelism?
    }

    // If the layer require a max tile size in Z then only ZTiling is allowed
    return !requireMaxZTile(layer);
}

void TilingPrimitives::inferInputTensorShape(DPUWorkload& wl, const DPULayer& originalLayer) {
    // const auto input_width = (wl.outputs[0].width() - 1) * wl.strides[Dim::Grid::W] + wl.kernels[Dim::Grid::W] -
    //                         wl.padding[Dim::Padding::LEFT] - wl.padding[Dim::Padding::RIGHT];
    // const auto input_height = (wl.outputs[0].height() - 1) * wl.strides[Dim::Grid::H] + wl.kernels[Dim::Grid::H] -
    //                          wl.padding[Dim::Padding::TOP] - wl.padding[Dim::Padding::BOTTOM];

    const auto input_width = helper_input_dim(wl.outputs[0].width(), wl.kernels[Dim::Grid::W],
                                              wl.padding[Dim::Padding::LEFT] + wl.padding[Dim::Padding::RIGHT],
                                              wl.strides[Dim::Grid::W]);

    const auto input_height = helper_input_dim(wl.outputs[0].height(), wl.kernels[Dim::Grid::H],
                                               wl.padding[Dim::Padding::TOP] + wl.padding[Dim::Padding::BOTTOM],
                                               wl.strides[Dim::Grid::H]);

    // the workload can be a result of HW split or Z split
    // for HW split the input channels remain unaffected (operation irrelevant)
    //
    // for Z split: the output (of 1 DPU workload) has only a fraction of the original z. 
    // For the operations that use the full input channels in their kernel (e.g. CONV, CM_CONV)  
    // - the input's Z should be again the full original input channels, there was no split of input Z
    // For the operation that are not having a kernel with depth (e.g. ELEMENTWISE)
    // - the input's Z should be equal to output's Z  (as a more general rule)
    const auto input_channel =
            ((wl.op == Operation::CONVOLUTION) || (wl.op == Operation::CM_CONVOLUTION) ||
             wl.is_output_autopad())    // kernels need all input Z
                    ? wl.inputs[0].z()  // use what the split left here by default (maybe original z, maybe a Z split of
                                        // inputs(future?))
                    : wl.outputs[0].z();  // for elementwise operations the in-out channels should match

    const auto input_batch = wl.outputs[0].batches();

    const auto inputTensor =
            VPUTensor({input_width, input_height, input_channel, input_batch}, originalLayer.inputs[0]);
    wl.inputs[0] = inputTensor;

    // sanitary limitation
    {  // limit halo height
        if (wl.halo.input_0_halo.top > (int)wl.inputs[0].height()) {
            wl.halo.input_0_halo.top = (int)wl.inputs[0].height();
        }
        if (wl.halo.input_0_halo.bottom > (int)wl.inputs[0].height()) {
            wl.halo.input_0_halo.bottom = (int)wl.inputs[0].height();
        }
    }
    {  // limit width
        if (wl.halo.input_0_halo.left > (int)wl.inputs[0].width()) {
            wl.halo.input_0_halo.left = (int)wl.inputs[0].width();
        }
        if (wl.halo.input_0_halo.right > (int)wl.inputs[0].width()) {
            wl.halo.input_0_halo.right = (int)wl.inputs[0].width();
        }
    }

    {  // limit halo channels/depth
        if (wl.halo.input_0_halo.front > (int)wl.inputs[0].channels()) {
            wl.halo.input_0_halo.front = (int)wl.inputs[0].channels();
        }
        if (wl.halo.input_0_halo.back > (int)wl.inputs[0].channels()) {
            wl.halo.input_0_halo.back = (int)wl.inputs[0].channels();
        }
    }
}

std::vector<unsigned int> TilingPrimitives::getSplitsFromRange(const unsigned int maxSplitRange,
                                                               const unsigned int maxLimit) {
    std::vector<unsigned int> splits;
    for (unsigned int idx = 0; idx < std::log2(maxSplitRange); idx++) {
        auto powIdx = static_cast<unsigned int>(std::pow(2, idx));
        auto splitCandidate = maxSplitRange / powIdx;
        if (maxSplitRange % powIdx == 0 && splitCandidate <= maxLimit) {
            splits.push_back(splitCandidate);
        }
    }
    return splits;
}

DPUWorkload TilingPrimitives::createTileZ(const DPULayer& layer, unsigned int channels,
                                           unsigned int offset_channels) {
    DPUWorkload wl{createIncompleteTile(layer, {layer.outputs[0].x(), layer.outputs[0].y(), channels, 1},
                                        {0, 0, offset_channels, 0})};

    // Handle halo for intra-tiles. OWT is ignored here.
    // In case some OWT broadcast exists, it will override this info.
    {
        const HaloWorkload& tileHalo{layer.halo};
        // offset_channels: represents the number of channels produced by previous intra-tiles
        wl.halo.output_0_inbound_halo.front = tileHalo.output_0_inbound_halo.front + offset_channels;

        // channels: is the number of channels produced by this intra-tile
        // layer.outputs[0].channels(): is the number of channels of this entire tile
        const int remaining_channels_to_be_produced_after_current_tile =
                layer.outputs[0].channels() - channels - offset_channels;

        wl.halo.output_0_inbound_halo.back =
                tileHalo.output_0_inbound_halo.back + remaining_channels_to_be_produced_after_current_tile;
    }

    return wl;
}

}  // namespace VPUNN