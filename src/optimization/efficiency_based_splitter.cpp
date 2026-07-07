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

EfficiencyBasedSplitter::EfficiencyBasedSplitter(const std::vector<SplitBlock>& blocks_,
                                                  const std::vector<SplitBlock>& rem_blocks_,
                                                  double overhead)
    : blocks(blocks_), remainder_blocks(rem_blocks_), block_overhead(overhead) {
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------
bool EfficiencyBasedSplitter::divideEfficiency(int targetSize, SplitContainer& result) const {
    // Validate input: targetSize must be positive. Aligned targets (multiples
    // of 16) use the regular DP path; unaligned targets may still be handled
    // when a compatible remainder block is available.
    if (targetSize % 16 != 0 || targetSize <= 0) {
        // ----------------------------------------------------------------
        // UNALIGNED TARGET HANDLER
        //
        // A target that is not a multiple of 16 cannot be covered by the
        // regular blocks alone (all regular blocks are multiples of 16).
        // The only way to reach such a target is to place exactly ONE
        // special "remainder block" as the very last piece, so that its
        // lower bits cancel the residue, and everything before it is an
        // aligned prefix that the normal DP can solve.
        //
        // Strategy:
        //   for each candidate remainder block R:
        //     1. check compatibility   (R % 16 == target % 16  &&  R <= target)
        //     2. the prefix is         (target - R), which is always a multiple
        //        of 16, so we can recurse into this same function and it will
        //        follow the aligned DP path below.
        //     3. compute the total cost of (prefix solution + R)
        //     4. keep the cheapest combination; break ties by block count.
        // ----------------------------------------------------------------

        if (targetSize > 0 && !remainder_blocks.empty()) {
            // residue: the lower bits (1-15) that no regular block can cover.
            // A remainder block R is compatible iff R % 16 equals this value.
            const int residue = targetSize % 16;
            const double EPSILON = 1e-9;

            // Track the best solution found across all compatible remainder blocks.
            double best_cost = std::numeric_limits<double>::infinity();
            int best_count = std::numeric_limits<int>::max();
            bool found = false;

            for (const auto& rem : remainder_blocks) {
                // Skip incompatible remainder blocks:
                //   - size must be positive (sanity)
                //   - lower bits of rem.size must match the residue so that
                //     (prefix + rem.size) == targetSize exactly
                //   - rem.size must not exceed the target (can't overshoot)
                if (rem.size <= 0 || rem.size % 16 != residue || rem.size > targetSize)
                    continue;

                // The channels before the remainder block: always a multiple of 16
                // (or 0 when the remainder block alone covers the whole target).
                const int prefix = targetSize - rem.size;
                SplitContainer prefix_result;

                if (prefix > 0) {
                    // Recurse: prefix is aligned, so this call follows the DP
                    // path below and fills prefix_result with regular blocks.
                    if (!divideEfficiency(prefix, prefix_result))
                        continue;  // no valid DP solution for this prefix; try next rem
                }
                // If prefix == 0, prefix_result stays empty — the remainder
                // block alone covers the whole target, which is valid.

                // Compute the total execution cost of the prefix solution.
                // Each entry in prefix_result is a block size; look it up in
                // the regular-block table to retrieve its time value.
                // Cost of one block = block.time + block_overhead (fixed dispatch cost).
                double prefix_cost = 0.0;
                for (int sz : prefix_result) {
                    for (const auto& b : blocks) {
                        if (b.size == sz) {
                            prefix_cost += b.time + block_overhead;
                            break;  // found the matching block descriptor; move on
                        }
                    }
                }

                // Cost of the remainder block itself (same formula as regular blocks).
                double rem_cost = rem.time + block_overhead;

                // Total cost and block count for this (prefix + remainder) combination.
                double total_cost = prefix_cost + rem_cost;
                int total_count = static_cast<int>(prefix_result.size()) + 1;  // +1 for rem

                // Primary criterion: prefer lower total cost.
                bool is_better = total_cost < (best_cost - EPSILON);
                // Tie-break: equal cost → prefer fewer blocks (less fragmentation).
                bool is_tie_fewer =
                        (std::abs(total_cost - best_cost) <= EPSILON) && (total_count < best_count);

                if (is_better || is_tie_fewer) {
                    best_cost = total_cost;
                    best_count = total_count;
                    // Store the winning solution: prefix blocks first, then the
                    // remainder block appended at the end (hardware requires it last).
                    result = std::move(prefix_result);
                    result.push_back(rem.size);
                    found = true;
                }
            }

            // If any compatible remainder block produced a valid solution, return it.
            if (found)
                return true;
        }

        // No compatible remainder block found (or none provided): target is
        // unsolvable with the current block set.
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
