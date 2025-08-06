// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_TILER_H
#define VPUNN_TILER_H

#include <list>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "vpu/layer.h"
#include "vpu/optimization/workload_optimization_types.h"
#include "vpu/types.h"
#include "vpu/utils.h"

#include "vpu/layer_split_info.h"

namespace VPUNN {

/**
 * @brief A generic interface for intra tile tiling algorithms (splitting a tile(Layer) into more workloads)
 *
 */
class ITilerAlgorithm {
protected:
    const DPULayer layer_on_tile;  ///< The layer to optimize, should be the Layer that is assigned to a tile and has to
                                   ///< be split in workloads
    const unsigned int maxWorkloads;  ///< The maximum number of workloads to generate

    /**
     * @brief Interface for a specific implementation of a multi WL split
     *
     * @param splitPool the pool of valid split
     * @param mode the selected ExecutionMode
     * @param nWorkloads number of splits to generate
     */
    virtual void tileMultipleWl(std::list<DPUWorkloadsWithCyclesSplit>& splitPool, const ExecutionMode mode,
                                const unsigned int nWorkloads) = 0;

public:
    /**
     * @brief Construct a new Tiler object
     *
     * @param layer_ the DPULayer
     * @param maxWorkloads_ maximum number of workloads
     */
    explicit ITilerAlgorithm(const DPULayer& layer_, const unsigned int maxWorkloads_)
            : layer_on_tile(layer_), maxWorkloads(maxWorkloads_) {
    }

    /**
     * @brief Generates a number of possible splits
     * @details This function generates a number of possible splits considering
     * - the amount of DPUs available
     * - the maximum number of splits available
     *
     * @param numDPU available DPUs
     * @param valid_execution_mode valid ExecutionMode
     * @return std::vector<unsigned int>
     */
    virtual std::set<unsigned int> generateSplitPool(const unsigned int numDPU,
                                                     const ExecutionMode& valid_execution_mode) const = 0;

    /**
     * @brief Split a layer based on the algorithm specified in the implementation
     *
     * @param mode the selected ExecutionMode
     * @param nWorkloads number of splits to generate
     * @returns the pool of valid split
     */
    std::list<DPUWorkloadsWithCyclesSplit> split_tile_in_workloads(const ExecutionMode mode,
                                                                   const unsigned int nWorkloads);

    /**
     * @brief Set the mode for list of workloads
     *
     * @param workloads the list of workloads
     * @param mode the selected ExecutionMode
     * @param originalLayer the original DPULayer
     */
    static void setWorkloadsModeAndInfereInputShape(DPUWorkloadsWithCyclesSplit& workloads, const ExecutionMode mode,
                                                    const DPULayer& originalLayer);

    /**
     * @brief Destroy the Tiler object
     *
     */
    virtual ~ITilerAlgorithm() = default;

    ///  @brief name of the actual type, for debug purposes
    virtual std::string name() const = 0;
};

/**
 * @brief A list of algorithms
 */

using TilingAlgorithmsContainer = std::list<std::unique_ptr<ITilerAlgorithm>>;
/**
 * @brief Factory functions that returns a list of algorithms from the strategies
 *
 * @param layer the DPULayer
 * @param options the split algorithm optimization options
 * @return TilingAlgorithms
 */
TilingAlgorithmsContainer getTilingAlgorithms(const DPULayer& layer, const SplitOptions& options);

}  // namespace VPUNN

#endif  // VPUNN_TILER_H
