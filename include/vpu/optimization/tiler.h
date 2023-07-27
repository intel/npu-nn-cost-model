// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_TILER_H
#define VPUNN_TILER_H

#include <string>
#include "vpu/optimization/workload_optimization.h"
#include "vpu/types.h"
#include "vpu/utils.h"

// VPU split constants
constexpr unsigned int DEFAULT_ZTILE_VALUE = 16;
constexpr unsigned int MIN_VALID_ZTILE_EXPONENT = 4;
constexpr unsigned int MAX_VALID_ZTILE_EXPONENT = 8;

namespace VPUNN {

/**
 * @brief Split a DPULayer over the Z dimension, appending the result to the splitPool list
 *
 * @param layer a DPULayer
 * @param splitPool the pool of valid split
 * @param mode the valid ExecutionMode
 * @param nWorkloads number of splits
 * @param validZTiles valid Z dimension for the splits
 */
void splitOverZ(const DPULayer& layer, std::list<DPUWorkloads>& splitPool, const ExecutionMode mode,
                const unsigned int nWorkloads, const std::vector<unsigned int>& validZTiles);

/**
 * @brief Split a DPULayer over the H and W dimensions, appending the result to the splitPool list
 *
 * @param layer a DPULayer
 * @param splitPool the pool of valid split
 * @param widthFactor the number of splits in the X dimension
 * @param heightFactor  the number of splits in the Y dimension
 * @param mode the valid ExecutionMode
 */
void splitOverHW(const DPULayer& layer, std::list<DPUWorkloads>& splitPool, const unsigned int widthFactor,
                 const unsigned int heightFactor, const ExecutionMode mode);

/**
 * @brief A generic interface for tiling algorithms (splitting a tile(Layer) into more workloads)
 *
 */
class Tiler {
protected:
    DPULayer layer_on_tile;     ///< The layer to optimize, should be the Layer that is assigned to a tile and has to be
                                ///< split in workloads
    unsigned int maxWorkloads;  ///< The maximum number of workloads to generate

    /**
     * @brief Interface for a specific implementation of a multi WL split
     *
     * @param splitPool the pool of valid split
     * @param mode the selected ExecutionMode
     * @param nWorkloads number of splits to generate
     */
    virtual void tileMultipleWl(std::list<DPUWorkloads>& splitPool, const ExecutionMode mode,
                                const unsigned int nWorkloads) = 0;

public:
    /**
     * @brief Construct a new Tiler object
     *
     * @param layer_ the DPULayer
     * @param maxWorkloads_ maximum number of workloads
     */
    explicit Tiler(const DPULayer& layer_, const unsigned int maxWorkloads_ = 50)
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
    std::list<DPUWorkloads> split_tile_in_workloads(const ExecutionMode mode, const unsigned int nWorkloads);

    /**
     * @brief Set the mode for list of workloads
     *
     * @param workloads the list of workloads
     * @param mode the selected ExecutionMode
     * @param originalLayer the original DPULayer
     */
    static void setWorkloadsMode(DPUWorkloads& workloads, const ExecutionMode mode, const DPULayer& originalLayer);

    /**
     * @brief Destroy the Tiler object
     *
     */
    virtual ~Tiler() = default;

    ///  @brief name of the actual type, for debug purposes
    virtual std::string name() const = 0;
};

/**
 * @brief A list of algorithms
 *
 */
typedef std::list<std::unique_ptr<Tiler>> TilingAlgorithms;
/**
 * @brief Factory functions that returns a list of algorithms from the strategies
 *
 * @param layer the DPULayer
 * @param options the split algorithm optimization options
 * @return TilingAlgorithms
 */
TilingAlgorithms getTilingAlgorithms(const DPULayer& layer, const SplitOptions& options);
}  // namespace VPUNN

#endif  // VPUNN_TILER_H