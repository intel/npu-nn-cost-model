// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_WORKLOAD_COST_EVALUATOR_H
#define VPUNN_WORKLOAD_COST_EVALUATOR_H

#include <vector>
#include "vpu/cycles_interface_types.h"
#include "vpu/layer.h"
#include "vpu/layer_split_info.h"
#include "vpu/optimization/efficiency_based_splitter.h"
#include "vpu/types.h"

namespace VPUNN {

// Forward declarations
class VPUCostModel;
class HWPerformanceModel;

/**
 * @brief Utility class for evaluating workload execution costs
 *
 * This class encapsulates the logic for querying the VPUCostModel to measure
 * execution costs of DPU workloads. It is primarily used by efficiency-based
 * tiling algorithms to measure candidate block sizes before performing optimization.
 */
class WorkloadCostEvaluator {
private:
    const VPUCostModel& cost_model_;  ///< Reference to the cost model for querying workload costs

    /**
     * @brief Extract and validate the device from a workloads split
     *
     * All workloads must target the same device. Throws if the list is empty or contains mixed devices.
     *
     * @param workloads The workloads split to query
     * @return The common VPUDevice shared by all workloads
     * @throws std::invalid_argument if the workloads list is empty or contains mixed devices
     */
    VPUDevice getWorkloadsDevice(const DPUWorkloadsWithCyclesSplit& workloads) const;

    /**
     * @brief Get the hardware performance model from the cost model
     *
     * @return Reference to the HWPerformanceModel for hardware characteristics
     */
    const HWPerformanceModel& getHWPerformance() const;

public:
    /**
     * @brief Construct a WorkloadCostEvaluator
     *
     * @param cost_model Reference to the VPUCostModel to use for cost queries
     */
    explicit WorkloadCostEvaluator(const VPUCostModel& cost_model);

    /**
     * @brief Measure the execution cost of a single DPU workload
     *
     * Queries the cost model to determine the cycle count for executing the given workload.
     *
     * @param workload The DPU workload to evaluate
     * @param info [out] Optional string to receive additional information from the cost model
     * @return The cycle cost of executing the workload, or an error code if measurement fails
     */
    CyclesInterfaceType measureWorkload(const DPUWorkload& workload, std::string& info) const;

    /**
     * @brief Compute execution cycles for all workloads in a split
     *
     * Iterates through all workloads in the split and queries the cost model for each,
     * storing the cycle counts in the workloads_split.cycles array.
     *
     * @param workloads_split [in/out] The workload split with workloads to evaluate;
     *                        cycles array will be populated with measured costs
     */
    void computeWorkloadCycles(DPUWorkloadsWithCyclesSplit& workloads_split) const;

    /**
     * @brief Measure costs for multiple candidate block sizes
     *
     * For each candidate size, creates a test layer with that channel count, constructs
     * a workload, and measures its execution cost. Only valid (non-error, positive cost)
     * measurements are returned as SplitBlocks.
     *
     * This method is designed for efficiency-based tiling algorithms that need to measure
     * the cost of different channel configurations before running dynamic programming.
     *
     * @param base_layer Template layer to use (channel count will be modified per candidate)
     * @param mode Execution mode to use for the test workloads
     * @param candidate_sizes Vector of channel counts to measure (must be multiples of 16)
     * @return Vector of SplitBlocks containing size and measured cost for valid candidates
     */
    std::vector<SplitBlock> measureCandidateBlocks(
            const DPULayer& base_layer,
            const ExecutionMode& mode,
            const std::vector<unsigned int>& candidate_sizes) const;

    /**
     * @brief Create a test layer with modified output channel count
     *
     * Takes a base layer and creates a copy with the specified number of output channels.
     * The input tensor spatial dimensions remain unchanged. For depthwise-convolution and
     * pooling layer types (e.g. DW_CONVOLUTION, MAXPOOL, AVEPOOL), the input channel count
     * may be updated to match the requested output channels; for other layer types, the
     * input tensor (including its channel count) is preserved.
     *
     * @param base_layer The layer to use as a template
     * @param channels The desired output channel count
     * @return A new DPULayer with modified output channels
     */
    DPULayer createTestLayer(const DPULayer& base_layer, unsigned int channels) const;

    /**
     * @brief Count error codes in a cycle results vector
     *
     * @param workloads_cycles The cycle times to inspect
     * @return Number of error codes present (zero cycles is not counted as error)
     */
    int countErrors(const std::vector<CyclesInterfaceType>& workloads_cycles) const;

    /**
     * @brief Find the index of the first error code in a cycle results vector
     *
     * @param workloads_cycles The cycle times to inspect
     * @return Zero-based index of the first error code, or -1 if no errors present
     */
    int firstErrorIndex(const std::vector<CyclesInterfaceType>& workloads_cycles) const;

    /**
     * @brief Compute the total scheduled cycle cost of a workload split
     *
     * This method performs the complete cycle evaluation:
     * 1. Computes individual workload cycles via the cost model
     * 2. Checks for error codes and returns the first error if found
     * 3. Applies DPU scheduling based on hardware characteristics
     * 4. Adds runtime overhead per workload
     *
     * @param workloads_split [in/out] The split to evaluate; cycles array is updated in-place
     * @param runtimeOverhead Per-workload scheduling overhead in cycles (default: 0)
     * @return Total scheduled cycles for the split, or the first error code encountered
     */
    CyclesInterfaceType computeSplitCycles(DPUWorkloadsWithCyclesSplit& workloads_split,
                                           const unsigned int runtimeOverhead = 0) const;
};

}  // namespace VPUNN

#endif  // VPUNN_WORKLOAD_COST_EVALUATOR_H
