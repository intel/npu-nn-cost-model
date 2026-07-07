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
    int size;                  ///< Block size in channels
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
public:
    using SplitContainer = std::vector<int>;

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

    std::vector<SplitBlock> blocks;              ///< Available regular block sizes (multiples of 16)
    std::vector<SplitBlock> remainder_blocks;     ///< Blocks usable only as the final block of an unaligned split
    double block_overhead;                     ///< Fixed per-block overhead (e.g., scheduling, context switch)

public:
    /**
     * @brief Construct splitter with available block options and optional per-block overhead
     * @param blocks_ Vector of available block sizes with execution costs
     * @param overhead Fixed cost added per block (for modeling scheduling overhead)
     */
    EfficiencyBasedSplitter(const std::vector<SplitBlock>& blocks_, double overhead = 0.0);

    /**
     * @brief Construct splitter with regular blocks, remainder blocks, and optional overhead
     *
     * Use this constructor when the target channel count may NOT be a multiple of 16.
     * The splitter will find the optimal regular prefix split and append one remainder block
     * as the final element.
     *
     * @param blocks_     Aligned block sizes for the main partition (multiples of 16)
     * @param rem_blocks_ Remainder block sizes that may appear only at the end of a split
     * @param overhead    Fixed cost added per block
     */
    EfficiencyBasedSplitter(const std::vector<SplitBlock>& blocks_,
                            const std::vector<SplitBlock>& rem_blocks_,
                            double overhead = 0.0);

    /**
     * @brief Find optimal partition of targetSize channels using dynamic programming
     *
     * For aligned targets (targetSize % 16 == 0) the original DP over regular blocks runs unchanged.
     *
     * For unaligned targets (targetSize % 16 != 0) the algorithm tries each compatible remainder block
     * R (where R % 16 == targetSize % 16 and R <= targetSize) as the final piece, solves the aligned
     * prefix (targetSize - R) via recursion into the aligned path, and picks the lowest total cost.
     *
     * @param targetSize Total channels to partition
     * @param result     Output vector filled with optimal block sizes (sum == targetSize)
     * @return true if valid partition found, false otherwise
     */
    bool divideEfficiency(int targetSize, SplitContainer& result) const;
};

}  // namespace VPUNN

#endif  // VPUNN_EFFICIENCY_BASED_SPLITTER_H
