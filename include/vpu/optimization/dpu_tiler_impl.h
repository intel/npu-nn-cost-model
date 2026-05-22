// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_DPU_TILER_IMPL_H
#define VPUNN_DPU_TILER_IMPL_H

#include "dpu_tiler_intf.h"

#include <list>
#include "split_options.h"
#include "tiling_algorithm_factory.h"  // for TilingAlgorithmsContainer
#include "vpu/layer.h"
#include "vpu/types.h"
#include "vpu_cost_model.h"
#include "workload_cost_evaluator.h"  // for WorkloadCostEvaluator

namespace VPUNN {

// Forward declaration
class ZEffTiling;

/**
 * @brief One of the implementation of the DPUTiler interface
 *
 */
class DPUTilerImplementation : public IDPUTiler {
private:
    const WorkloadCostEvaluator evaluator_;  ///< Utility for measuring workload costs

    /// @brief Generates all candidate workload splits and computes their cycle cost.
    /// Iterates over every combination of tiling algorithm, valid execution mode, and split
    /// count derived from @p options.nDPU. For each candidate split the cost is computed via
    /// computeSplitCycles(). Both valid costs and error codes are retained in the returned
    /// list so that the caller can pick the optimal result. Evaluation stops early when the
    /// elapsed time exceeds @p options.maxLatencyUs (if set). POWER optimisation target is
    /// not supported and will throw.
    ///
    /// @param algorithms     container of tiling algorithm instances to evaluate
    /// @param valid_execution_modes execution modes accepted by the layer (e.g. CUBOID_16x16)
    /// @param options        split options controlling nDPU, runtime overhead, timeout, etc.
    /// @returns a list of (cycle-cost, workloads-split) pairs for every evaluated candidate
    /// @throws std::runtime_error if @p options.target is VPUOptimizationTarget::POWER
    std::list<DPUWorkloadsWithCycleCost> generateSplits(const TilingAlgorithmsContainer& algorithms,
                                                        const std::vector<ExecutionMode>& valid_execution_modes,
                                                        const SplitOptions& options) const;

    /// @brief Finds ZEffTiling algorithm in the algorithms container if present.
    /// Iterates through the algorithms and returns a pointer to the first ZEffTiling
    /// instance found, or nullptr if none exists.
    ///
    /// @param algorithms container of tiling algorithm instances to search
    /// @returns pointer to ZEffTiling if found, nullptr otherwise
    ZEffTiling* findZEffTilingAlgorithm(const TilingAlgorithmsContainer& algorithms) const;

    /// @brief Checks if ZEffTiling algorithm exists in the algorithms container.
    /// Convenience wrapper around findZEffTilingAlgorithm for boolean checks.
    ///
    /// @param algorithms container of tiling algorithm instances to search
    /// @returns true if ZEffTiling is found, false otherwise
    bool hasZEffTilingAlgorithm(const TilingAlgorithmsContainer& algorithms) const;

    /// @brief Processes ZEffTiling algorithm to generate workload splits.
    /// Finds ZEffTiling in the algorithms container, measures candidate block costs based on
    /// layer characteristics, configures the algorithm with these blocks, and generates all
    /// workload splits for the valid execution modes. Handles special cases like DW convolutions
    /// with small spatial dimensions that require limited block sizes.
    ///
    /// @param algorithms         container of tiling algorithm instances to search for ZEffTiling
    /// @param layer              the DPU layer being processed
    /// @param valid_execution_modes execution modes accepted by the layer
    /// @param options            split options containing runtime overhead settings
    /// @returns a list of (cycle-cost, workloads-split) pairs for every evaluated candidate,
    ///          or an empty list if ZEffTiling is not found in algorithms
    std::list<DPUWorkloadsWithCycleCost> generateSplits_processZEffTilingSplits(
            const TilingAlgorithmsContainer& algorithms, const DPULayer& layer,
            const std::vector<ExecutionMode>& valid_execution_modes, const SplitOptions& options) const;

public:
    /**
     * @brief Construct a new DPUTilerImplementation object
     *
     * @param _model a reference to a VPUCostModel object
     */
    explicit DPUTilerImplementation(const VPUCostModel& _model): evaluator_{_model} {
    }

    /// @brief Computes the optimal intra-tile workload split for a DPU layer.
    /// Determines the valid execution modes for @p layer, instantiates all applicable
    /// tiling algorithms, and evaluates every generated split via generateSplits().
    /// The split with the lowest non-error cycle cost is returned. Optionally, all
    /// evaluated splits (before selection) are appended to @p complete_output_splits.
    ///
    /// @param layer                  the DPU layer to be split into workloads
    /// @param options                split options (nDPU, runtime overhead, timeout, etc.)
    /// @param complete_output_splits if non-null, all evaluated splits are appended here
    /// @returns a DPUWorkloadsCost containing the optimal cycle count and the corresponding
    ///          vector of DPUWorkload objects
    /// @throws std::runtime_error if no valid workload split can be generated
    DPUWorkloadsCost intraTileSplit(
            const DPULayer& layer, const SplitOptions& options,
            std::vector<DPUWorkloadsWithCyclesSplit>* complete_output_splits = nullptr) const override;

protected:
    /// @brief Evaluates the cycle cost of a single workload split candidate.
    /// Delegates all cycle computation, error checking, and scheduling logic to the
    /// WorkloadCostEvaluator. This is a thin wrapper maintained for interface compatibility.
    ///
    /// @param workloads_split  the split to evaluate; its cycles array is updated in-place
    /// @param runtimeOverhead  per-workload scheduling overhead in cycles (default: 0)
    /// @returns total scheduled cycles for the split, or the first DPU error code encountered
    CyclesInterfaceType computeSplitCycles(DPUWorkloadsWithCyclesSplit& workloads_split,
                                           const unsigned int runtimeOverhead = 0) const override;
};

}  // namespace VPUNN

#endif  // DPU_TILER_IMPL_H
