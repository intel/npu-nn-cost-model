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

namespace VPUNN {

ZEffTiling::ZEffTiling(const DPULayer& layer, const unsigned int maxWorkloads_, double overhead)
        : ITilerAlgorithm(layer, maxWorkloads_)
        , basic_blocks_()
        , block_overhead_(overhead) {
}

ZEffTiling::ZEffTiling(const DPULayer& layer, const unsigned int maxWorkloads_,
                       const std::vector<SplitBlock>& basic_blocks, double overhead)
        : ITilerAlgorithm(layer, maxWorkloads_)
        , basic_blocks_(basic_blocks)
        , block_overhead_(overhead) {
}

void ZEffTiling::setBlocks(const std::vector<SplitBlock>& basic_blocks) {
    basic_blocks_ = basic_blocks;
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
    // Create efficiency-based splitter with our blocks
    EfficiencyBasedSplitter splitter(basic_blocks_, block_overhead_);

    // Find optimal split using Dynamic Programming
    EfficiencyBasedSplitter::SplitContainer split_bins;
    const bool split_status = splitter.divideEfficiency(layer.outputs[0].channels(), split_bins);

    if (!split_status) {
        return;  // No valid split found
    }

    if (split_bins.empty()) {
        return;  // Empty split
    }

    // Create workloads from the optimal split
    DPUWorkloadsWithCyclesSplit workloads_split;
    unsigned int offset_channels = 0;

    for (const auto& block_size : split_bins) {
        if (block_size <= 0) {
            return;  // Invalid block size
        }

        // Create workload for this channel block
        workloads_split.workloads.emplace_back(TilingPrimitives::createTileZ(
                layer, static_cast<unsigned>(block_size), static_cast<unsigned>(offset_channels)));
        workloads_split.cycles.emplace_back(Cycles::NO_ERROR);

        offset_channels += block_size;
    }

    // Verify we used all channels
    if (offset_channels != layer.outputs[0].channels()) {
        return;  // Split didn't cover all channels
    }

    // Set execution mode and infer input shapes
    ITilerAlgorithm::setWorkloadsModeAndInfereInputShape(workloads_split, mode, layer);

    // Add to split pool
    splitPool.push_back(std::move(workloads_split));
}

}  // namespace VPUNN
