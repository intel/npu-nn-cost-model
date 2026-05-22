// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_HWTILING_H
#define VPUNN_HWTILING_H

#include <list>
#include <set>
#include <string>

#include "tiler.h"
#include "vpu/dpu_halo.h"  // For HaloWorkload
#include "vpu/dpu_types.h"
#include "vpu/layer_split_info.h"

namespace VPUNN {

/**
 * @brief A Tiler child class that implement the HWTiling algorithm
 *
 */
class HWTiling : public ITilerAlgorithm {
public:
    /**
     * @brief Using the Tiler class constructor
     *
     */
    using ITilerAlgorithm::ITilerAlgorithm;

    /**
     * @brief Generate a valid pool of splits
     *
     * @param numDPU number of DPUs to optimize with
     * @param valid_execution_mode valid DPU ExecutionMode
     * @return std::vector<unsigned int>
     */
    std::set<unsigned int> generateSplitPool(const unsigned int numDPU,
                                             const ExecutionMode& valid_execution_mode) const override;

    /**
     * @brief Tile multiple workloads using the HWTiling algorithm
     *
     * @param splitPool the pool of valid splits returned by this function
     * @param mode the MPE mode
     * @param nWorkloads the number of workloads to generate
     */
    virtual void tileMultipleWl(std::list<DPUWorkloadsWithCyclesSplit>& splitPool, const ExecutionMode mode,
                                const unsigned int nWorkloads) override;

    std::string name() const override {
        return "HWTiling";
    }

protected:
    /**
     * @brief Tile a DPULayer over H and W
     *
     * @param splitPool the pool of valid splits returned by this function
     * @param widthFactor the number of splits in the X dimension
     * @param heightFactor  the number of splits in the Y dimension
     * @param mode the selected ExecutionMode
     */
    void tileOverHW(std::list<DPUWorkloadsWithCyclesSplit>& splitPool, const unsigned int widthFactor,
                    const unsigned int heightFactor, const ExecutionMode mode);

    /**
     * @brief Split a DPULayer over the H and W dimensions, appending the result to the splitPool list
     *
     * HALO ASPECTS: @todo:
     * Pass halo to intra tile splits. Especially edge tiles are influenced by halo. Important are what NN uses and 2nd
     * the memory aspects Input halo influences NN up to v2.7. To be ported if positive to edge tiles. Output halo + cnt
     * influences NN v4.0 and onwards. To be ported to edge tiles (+ inner tiles if larger than edge tile)
     * output_inbound halo is not used by NN, only for memory, if left unchanged increases memory but the layer was
     * already checked to fit in memory.
     *
     * @param layer a DPULayer
     * @param splitPool the pool of valid split
     * @param widthFactor the number of splits in the X dimension
     * @param heightFactor  the number of splits in the Y dimension
     * @param mode the valid ExecutionMode
     */
    void splitOverHW(const DPULayer& layer, std::list<DPUWorkloadsWithCyclesSplit>& splitPool,
                     const unsigned int widthFactor, const unsigned int heightFactor, const ExecutionMode mode);

    DPUWorkload createTileHW(const DPULayer& layer, const unsigned int width, const unsigned int height,
                             const unsigned int offset_width, const unsigned int offset_height, HaloWorkload halo);

private:
    /**
     * @brief Get the prime factor of N
     *
     * @param n a natural number
     * @return std::list<std::pair<unsigned int, unsigned int>>
     */
    std::list<std::pair<unsigned int, unsigned int>> getFactors(unsigned int n);
};

}  // namespace VPUNN

#endif  // HWTILING_H
