// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/optimization/efficiency_based_splitter.h"
#include <cmath>
#include <limits>

namespace VPUNN {

SplitBlock::SplitBlock(int s, CyclesInterfaceType t) 
    : size(s), time(t), efficiency(static_cast<double>(time) / static_cast<double>(size)) {
}

EfficiencyBasedSplitter::DynProgState::DynProgState() 
    : total_cost(std::numeric_limits<double>::infinity()), count(std::numeric_limits<int>::max()) {
}

EfficiencyBasedSplitter::DynProgState::DynProgState(double cost, int cnt) 
    : total_cost(cost), count(cnt) {
}

EfficiencyBasedSplitter::EfficiencyBasedSplitter(const std::vector<SplitBlock>& blocks_, double overhead)
    : blocks(blocks_), block_overhead(overhead) {
}

bool EfficiencyBasedSplitter::divideEfficiency(int targetSize, SplitContainer& result) const {
    // Validate input: only multiples of 16 are valid (hardware constraint)
    if (targetSize % 16 != 0 || targetSize <= 0) {
        return false;
    }

    // Scale down by 16 to reduce state space (16 -> 1, 32 -> 2, etc.)
    // This optimization reduces memory and computation by 16x
    int scaled_target = targetSize / 16;
    const double EPSILON = 1e-9;  // For floating-point comparison tolerance

    // Dynamic Programming table: dynProg[i] = optimal solution for partitioning i*16 channels
    std::vector<DynProgState> dynProg(scaled_target + 1);
    // Choice table: choice[i] = which block size was used to optimally reach state i
    std::vector<int> choice(scaled_target + 1, -1);

    // Base case: partitioning 0 channels costs 0 and uses 0 blocks
    dynProg[0] = DynProgState(0.0, 0);

    // Dynamic Programming forward pass: build optimal solutions for increasing channel counts
    // For each reachable state (i), try extending it with each available block
    for (int i = 0; i < scaled_target; ++i) {
        // Skip unreachable states (never found a valid partition to reach here)
        if (dynProg[i].total_cost >= std::numeric_limits<double>::infinity() - EPSILON)
            continue;

        // Try adding each block type to current state
        for (const auto& block : blocks) {
            if (block.size % 16 != 0)
                continue;  // Safety check: blocks must be multiples of 16

            int scaled_block_size = block.size / 16;
            int next_size = i + scaled_block_size;

            // Check if adding this block stays within target
            if (next_size <= scaled_target) {
                // Calculate cost of adding this block
                // Cost = efficiency metric (time/size * size = time) + fixed overhead
                double block_cost = (block.efficiency * block.size) + block_overhead;
                double new_cost = dynProg[i].total_cost + block_cost;
                int new_count = dynProg[i].count + 1;

                // Update if this path is better than previous best for next_size
                // Primary criterion: lower cost
                bool is_better_cost = new_cost < (dynProg[next_size].total_cost - EPSILON);
                // Secondary criterion: same cost but fewer blocks (reduces fragmentation)
                bool is_same_cost_fewer_blocks =
                        (std::abs(new_cost - dynProg[next_size].total_cost) <= EPSILON) &&
                        (new_count < dynProg[next_size].count);

                if (is_better_cost || is_same_cost_fewer_blocks) {
                    dynProg[next_size] = DynProgState(new_cost, new_count);
                    choice[next_size] = block.size;  // Remember which block led to this optimal state
                }
            }
        }
    }

    // Check if we found a valid partition for the full target
    if (dynProg[scaled_target].total_cost >= std::numeric_limits<double>::infinity() - EPSILON) {
        return false;  // No valid partition exists (e.g., target=80 but only {96, 64} available)
    }

    // Reconstruct solution by backtracking through choice table
    // Start from target and work backwards, collecting the blocks used
    result.clear();
    int current_scaled_size = scaled_target;

    while (current_scaled_size > 0) {
        int block_used = choice[current_scaled_size];
        if (block_used <= 0)
            return false;  // Invalid state: should never happen if Dynamic Programming succeeded

        result.push_back(block_used);  // Add this block to result
        current_scaled_size -= (block_used / 16);  // Move backwards by block size
    }

    return true;
}

}  // namespace VPUNN
