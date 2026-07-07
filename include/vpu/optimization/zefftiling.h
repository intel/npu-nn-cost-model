// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_ZEFFTILING_H
#define VPUNN_ZEFFTILING_H

#include <list>
#include <set>
#include <string>
#include <vector>

#include "efficiency_based_splitter.h"
#include "tiler.h"
#include "vpu/dpu_types.h"
#include "vpu/layer.h"
#include "vpu/layer_split_info.h"

namespace VPUNN {

/**
 * @brief Z-dimension tiler using efficiency-based dynamic programming split algorithm
 *
 * This tiler uses EfficiencyBasedSplitter to find optimal channel splits based on
 * actual execution costs rather than simple balanced or power-of-2 splits.
 * Particularly useful for DW_CONVOLUTION where channel configurations
 * have varying efficiency characteristics.
 *
 * Usage example:
 * Example usage with immediate blocks:
 * \code{.cpp}
 * std::vector<SplitBlock> blocks = {{16, 1600}, {32, 2800}, {64, 5000}};
 * ZEffTiling tiler(layer, maxWorkloads, blocks, 100.0);
 * std::list<DPUWorkloadsWithCyclesSplit> splitPool;
 * tiler.tileMultipleWl(splitPool, ExecutionMode::CUBOID_16x16, 0);
 * \endcode
 *
 * Example with deferred blocks:
 * \code{.cpp}
 * ZEffTiling tiler(layer, maxWorkloads, 100.0);
 * // Measure costs...
 * tiler.setBlocks(measured_basic_blocks);
 * tiler.tileMultipleWl(splitPool, mode, 0);
 * \endcode
 */
class ZEffTiling : public ITilerAlgorithm {
public:
    /**
     * @brief Construct ZEffTiling without pre-defined blocks
     *
     * Blocks must be set later using setBlocks() before tiling can be performed.
     *
     * @param layer The layer to tile
     * @param maxWorkloads_ Maximum number of workloads to generate
     * @param overhead Per-block overhead cost (default: 0.0)
     */
    ZEffTiling(const DPULayer& layer, const unsigned int maxWorkloads_, double overhead = 0.0);

    /**
     * @brief Construct with pre-defined building blocks
     *
     * @param layer The layer to tile
     * @param maxWorkloads_ Maximum number of workloads to generate
     * @param basic_blocks Vector of available channel blocks with their costs
     * @param overhead Per-block overhead cost (default: 0.0)
     */
    ZEffTiling(const DPULayer& layer, const unsigned int maxWorkloads_,
               const std::vector<SplitBlock>& basic_blocks, double overhead = 0.0);

    /**
     * @brief Set the building blocks for efficiency-based splitting
     *
     * Must be called before tileMultipleWl() if blocks were not provided in constructor.
     * Can also be used to update blocks dynamically.
     *
     * @param basic_blocks Vector of available channel blocks with their costs
     */
    void setBlocks(const std::vector<SplitBlock>& basic_blocks);

    /**
     * @brief Set remainder blocks for unaligned output channel splits
     *
     * Remainder blocks are candidate sizes whose lower bits (size % 16) match
     * the unaligned residue.  They may only appear as the last element of a
     * split.  Each block should carry a measured execution cost so the DP can
     * pick the cheapest combination of prefix + remainder.
     *
     * @param remainder_blocks Vector of measured remainder block sizes with costs
     */
    void setRemainderBlocks(const std::vector<SplitBlock>& remainder_blocks);

    /**
     * @brief Check if building blocks are available
     *
     * @return true if blocks have been set and are ready for use
     */
    bool hasBlocks() const {
        return !basic_blocks_.empty();
    }

    /**
     * @brief Generate valid split pool for the layer
     *
     * @param numDPU Number of DPUs to optimize with
     * @param valid_execution_mode Valid DPU ExecutionMode
     * @return Set of valid split counts to try
     */
    std::set<unsigned int> generateSplitPool(const unsigned int numDPU,
                                             const ExecutionMode& valid_execution_mode) const override;

    /**
     * @brief Tile multiple workloads using efficiency-based algorithm
     *
     * @param splitPool [out] Pool of valid splits generated
     * @param mode MPE execution mode
     * @param nWorkloads Number of workloads to generate (ignored and considers that we find this dynamic inside)
     */
    void tileMultipleWl(std::list<DPUWorkloadsWithCyclesSplit>& splitPool, const ExecutionMode mode,
                        const unsigned int nWorkloads) override;

    std::string name() const override {
        return "ZEffTiling";
    }

protected:
    /**
     * @brief Split layer over Z using efficiency-based DP algorithm
     *
     * @param layer Layer to split
     * @param splitPool [out] Pool of valid splits generated
     * @param mode Execution mode
     */
    void splitOverZEfficiency(const DPULayer& layer, std::list<DPUWorkloadsWithCyclesSplit>& splitPool,
                               const ExecutionMode mode);

private:
    std::vector<SplitBlock> basic_blocks_;      ///< Available building blocks with costs (multiples of 16)
    std::vector<SplitBlock> remainder_blocks_;  ///< Remainder blocks for unaligned splits (size % 16 != 0)
    double block_overhead_;                     ///< Per-block overhead cost
};

}  // namespace VPUNN

#endif  // VPUNN_ZEFFTILING_H
