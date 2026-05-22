// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_EFFICIENCY_BASED_SPLITTER_H
#define VPUNN_EFFICIENCY_BASED_SPLITTER_H

#include <vector>
#include "vpu/cycles_interface_types.h"

namespace VPUNN {

/**
 * @brief Represents a candidate block size with its execution cost
 *
 * Each block represents one possible intratile split size (e.g., 16, 32, 64... channels).
 * The cost model provides the actual hardware execution time for each block size.
 */
struct SplitBlock {
    int size;                  ///< Block size in channels (must be multiple of 16)
    CyclesInterfaceType time;  ///< Hardware execution time in cycles for this block size
    double efficiency;         ///< Cost per channel: time / size (smaller = more efficient)

    SplitBlock(int s, CyclesInterfaceType t);
};

/**
 * @brief Efficiency-based dynamic programming splitter for optimal intratile channel splits
 *
 * Problem: Given a total number of channels to split and a set of allowed block sizes,
 * find the optimal partition that minimizes total execution cost.
 *
 * - Why Dynamic Programming beats greedy:
 * - Greedy-by-size may leave impossible remainders (e.g., pick 96, leaving 16 when no 16 available)
 * - Greedy-by-efficiency may create too many small blocks, increasing overhead
 * - DP explores all valid combinations to find globally optimal solution
 *
 * Algorithm: Classic unbounded knapsack variant with cost minimization
 * - State: dynProg[i] = minimum cost to partition i*16 channels
 * - Transition: For each reachable state, try adding each available block
 * - Optimization: Minimize cost, then minimize block count (tie-breaker)
 * - Time complexity: O(target/16 * num_blocks)
 * - Space complexity: O(target/16)
 */
class EfficiencyBasedSplitter {
private:
    /**
     * @brief DP state tracking optimal cost and block count for a given channel count
     *
     * We track both cost and count to implement tie-breaking:
     * - Primary: Minimize total_cost (execution time is most important)
     * - Secondary: Minimize count (fewer blocks reduce scheduling overhead and fragmentation)
     */
    struct DynProgState {
        double total_cost;  ///< Accumulated cost (efficiency * size + overhead) for optimal partition
        int count;          ///< Number of blocks used (for tie-breaking when costs are equal)

        DynProgState();
        DynProgState(double cost, int cnt);
    };

    std::vector<SplitBlock> blocks;  ///< Available block sizes with their costs
    double block_overhead;           ///< Fixed per-block overhead (e.g., scheduling, context switch)

public:
    using SplitContainer = std::vector<int>;

    /**
     * @brief Construct splitter with available block options and optional per-block overhead
     * @param blocks_ Vector of available block sizes with execution costs
     * @param overhead Fixed cost added per block (for modeling scheduling overhead)
     */
    EfficiencyBasedSplitter(const std::vector<SplitBlock>& blocks_, double overhead = 0.0);

    /**
     * @brief Find optimal partition of targetSize channels using dynamic programming
     *
     * Algorithm phases:
     * 1. Validation: Check targetSize is positive and multiple of 16
     * 2. Initialization: Scale target by dividing by 16 to reduce state space
     * 3. Dynamic Programming forward pass: Build optimal solutions for all sizes from 0 to target
     * 4. Solution reconstruction: Backtrack through choices to build result
     *
     * Dynamic Programming recurrence relation:
     *   dynProg[i + block.size/16] = min over all blocks of:
     *     dynProg[i].cost + (block.efficiency * block.size + overhead)
     *   with tie-breaking: prefer solution with fewer blocks if costs are equal
     *
     * @param targetSize Total channels to partition (must be multiple of 16)
     * @param result Output vector filled with optimal block sizes (sum = targetSize)
     * @return true if valid partition found, false if impossible or invalid input
     */
    bool divideEfficiency(int targetSize, SplitContainer& result) const;
};

}  // namespace VPUNN

#endif  // VPUNN_EFFICIENCY_BASED_SPLITTER_H
