// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/optimization/zefftiling.h"
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
#include "vpu/vpu_tensor.h"

namespace VPUNN {

ZEffTiling::ZEffTiling(const DPULayer& layer, const unsigned int maxWorkloads_, double overhead)
        : ITilerAlgorithm(layer, maxWorkloads_)
        , basic_blocks_()
        , remainder_blocks_()
        , block_overhead_(overhead) {
}

ZEffTiling::ZEffTiling(const DPULayer& layer, const unsigned int maxWorkloads_,
                       const std::vector<SplitBlock>& basic_blocks, double overhead)
        : ITilerAlgorithm(layer, maxWorkloads_)
        , basic_blocks_(basic_blocks)
        , remainder_blocks_()
        , block_overhead_(overhead) {
}

void ZEffTiling::setBlocks(const std::vector<SplitBlock>& basic_blocks) {
    basic_blocks_ = basic_blocks;
}

void ZEffTiling::setRemainderBlocks(const std::vector<SplitBlock>& remainder_blocks) {
    remainder_blocks_ = remainder_blocks;
}
// For now name is unrepresentative but will be fixed when we go deeper into redesigning the entire system
std::set<unsigned int> ZEffTiling::generateSplitPool( const unsigned int numDPU,
                                                     [[maybe_unused]] const ExecutionMode& valid_execution_mode) const {
    if (numDPU == 1 && TilingPrimitives::isNoSplitOperation(this->layer_on_tile.op)) {
        return {1U};  // why? CONV and ELEMwise are not split?
    }

    // If we have a case with under 16 channels, either input or output and we see input/output autopad
    // We return 1 as we should not split it.
    if ((layer_on_tile.inputs[0].channels() < 16 || layer_on_tile.outputs[0].channels() < 16) &&
        (layer_on_tile.is_input_autopad() || layer_on_tile.is_output_autopad())) {
        return {1U};
    }
    // If we dont have blocking points then we can let it on 0 so it will try all combinations possible
    return {0U};
}

void ZEffTiling::tileMultipleWl(std::list<DPUWorkloadsWithCyclesSplit>& splitPool, const ExecutionMode mode,
                                [[maybe_unused]] const unsigned int nWorkloads) {
    // Use efficiency-based algorithm for all devices
    // Note: nWorkloads is ignored - the Dynamic Programming algorithm determines optimal split count
    splitOverZEfficiency(layer_on_tile, splitPool, mode);
}

void ZEffTiling::splitOverZEfficiency(const DPULayer& layer, std::list<DPUWorkloadsWithCyclesSplit>& splitPool,
                                       const ExecutionMode mode) {
    // Validate that blocks are available
    if (basic_blocks_.empty()) {
        return;  // No blocks available - cannot perform split
    }

    const unsigned int oc = layer.outputs[0].channels();
    const auto is_unaligned = layer.is_output_unaligned();
    // Build the splitter.  For aligned targets the remainder vector is empty
    // and ignored; for unaligned targets the DP appends exactly one compatible
    // remainder block as the last element of the result.
    EfficiencyBasedSplitter splitter(basic_blocks_, remainder_blocks_, block_overhead_);

    EfficiencyBasedSplitter::SplitContainer split_bins;
    if (!splitter.divideEfficiency(static_cast<int>(oc), split_bins) || split_bins.empty()) {
        return;
    }

    // Create workload tiles from the optimal partition
    DPUWorkloadsWithCyclesSplit workloads_split;
    unsigned int offset_channels = 0;

    for (size_t i = 0; i < split_bins.size(); ++i) {
        const auto& block_size = split_bins[i];
        if (block_size <= 0) {
            return;  // Invalid block size
        }

        auto tile = TilingPrimitives::createTileZ(layer, static_cast<unsigned>(block_size), offset_channels);
        // Use explicit index comparison to detect the last tile for robust remainder-tile autopad handling
        if (is_unaligned && (i == split_bins.size() - 1)) {
            tile.set_output_autopad(true);  // Mark the remainder tile with autopad
        } else {
            tile.set_output_autopad(false);  // Aligned tiles should not have autopad
        }
        workloads_split.workloads.push_back(std::move(tile));
        workloads_split.cycles.push_back(Cycles::NO_ERROR);
        offset_channels += static_cast<unsigned>(block_size);
    }

    // Verify we used all channels
    if (offset_channels != oc) {
        return;  // Split didn't cover all channels
    }

    // Set execution mode and infer input shapes for all tiles.
    ITilerAlgorithm::setWorkloadsModeAndInfereInputShape(workloads_split, mode, layer);

    // Fix input channels for the remainder tile of DW-like ops.
    //
    // inferInputTensorShape preserves the full layer IC for any output_autopad tile, which is
    // correct for CONVOLUTION (the kernel consumes all input depth regardless of output split).
    // However, for DW_CONVOLUTION, AVEPOOL, and MAXPOOL the in_ch is coupled to out_ch:
    // the hardware reads only the channels that correspond to the (hardware-aligned) tile OC.
    // measureCandidateBlocks already models this via createTestLayer (((oc/32)+1)*32 alignment),
    // so the actual tile must use the same formula — otherwise the validator rejects the tile
    // with ERROR_TILE_OUTPUT because in_ch (full layer IC) != aligned out_ch.
    if (is_unaligned && !workloads_split.workloads.empty()) {
        auto& rem_wl = workloads_split.workloads.back();
        if (rem_wl.is_output_autopad() &&
            is_dwconv_family_operation(rem_wl.op)) {
            // Retrieve the output channel count from the remainder workload
            const unsigned int out_c = rem_wl.outputs[0].channels();
            
            // Calculate the aligned input channels to match hardware requirements.
            // The alignment follows createTestLayer logic:
            // - If already aligned to 16: keep as is
            // - If less than 16: use 16 (minimum alignment)
            // - Otherwise: round up to next multiple of 32 above out_c
            const unsigned int aligned_ic = dw_channel_align(out_c);

            rem_wl.inputs[0] = VPUTensor(
                    {rem_wl.inputs[0].width(), rem_wl.inputs[0].height(), aligned_ic,
                     rem_wl.inputs[0].batches()},
                    layer.inputs[0]);
        }
    }

    // Add to split pool
    splitPool.push_back(std::move(workloads_split));
}

}  // namespace VPUNN
