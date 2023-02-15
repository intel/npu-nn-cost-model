// Copyright © 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include <memory>

#include "core/profiling.h"
#include "vpu/optimization/tiler.h"
#include "vpu/optimization/workload_optimization.h"

namespace VPUNN {

// A type that describes a pair of cost and the associated DPUWorkloads
typedef std::pair<float, DPUWorkloads> DPUWorkloadsCost;

// When minimizing the cost we should prioritize choosing a smaller amount of workloads
inline bool operator<(const DPUWorkloads& lhs, const DPUWorkloads& rhs) {
    return lhs.size() < rhs.size();
}

/**
 * @brief Private implementation of the DPUTiler interface
 *
 */
class DPUTilerImplementation : public DPUTiler {
private:
    std::shared_ptr<VPUCostModel> model;

    VPUDevice getWorkloadsDevice(const DPUWorkloads& workloads) {
        if (workloads.size() == 0) {
            throw_error<std::invalid_argument>("getWorkloadsDevice:empty workloads list");
        }
        VPUDevice device = workloads[0].device;
        for (unsigned int idx = 1; idx < workloads.size(); idx++) {
            if (workloads[idx].device != device) {
                throw_error<std::invalid_argument>("getWorkloadsDevice: more than one device for a workloads list");
            }
        }
        return device;
    }

    float cost(const DPUWorkloads& workloads, const SplitOptions& options) {
        auto pnp = getLayerPerformance(workloads, options.runtimeOverhead);
        switch (options.target) {
        case VPUOptimizationTarget::LATENCY:
            return static_cast<float>(pnp.cycles);
        case VPUOptimizationTarget::POWER:
            return static_cast<float>(pnp.power);
        default:
            return static_cast<float>(pnp.cycles) / static_cast<float>(pnp.power);
        }
    }

    void generateSplits(std::list<DPUWorkloadsCost>& splits_costs, const TilingAlgorithms& algorithms,
                        const std::vector<ExecutionMode>& valid_execution_modes, const SplitOptions& options) {
        // Loop algorithms, splits, modes and polulate the DPUWorkloadsCost list
        auto timeout = SyncStopWatch<std::micro>();
        if (options.maxLatencyUs > 0)
            timeout.start();
        for (auto& algo : algorithms) {
            for (auto& mode : valid_execution_modes) {
                for (auto nWorkloads : algo->generateSplitPool(options.nDPU, {mode})) {
                    // A call to tile can generate multiple splits
                    std::list<DPUWorkloads> splitPool;
                    // Return if the max time has elapsed
                    if (options.maxLatencyUs > 0 && timeout.interval() > options.maxLatencyUs)
                        return;
                    algo->tile(splitPool, mode, nWorkloads);
                    for (auto& workloads : splitPool) {
                        auto wl_cost = cost(workloads, options);
                        // If the workload cost is == 0 the workload
                        // is invalid and shoudn't be inserted
                        if (wl_cost > 0)
                            splits_costs.push_back({wl_cost, workloads});
                    }
                }
            }
        }
    }

public:
    /**
     * @brief Construct a new DPUTilerImplementation object
     *
     * @param _model a shared pointer to a VPUCostModel object
     */
    DPUTilerImplementation(std::shared_ptr<VPUCostModel> const& _model) {
        model = _model;
    }

    DPUWorkloads intraTileSplit(const DPULayer& layer, const SplitOptions& options) override {
        // Get the number of DPUs for that layer
        auto valid_execution_modes = getValidExecutionMode(layer);

        // get all tiling algorithms
        TilingAlgorithms algorithms = getTilingAlgorithms(layer, options);

        // Compute the cost of each split type.
        std::list<DPUWorkloadsCost> splits_costs;
        generateSplits(splits_costs, algorithms, valid_execution_modes, options);

        if (splits_costs.size() == 0) {
            throw_error<std::runtime_error>("intraTileSplit: no valid workload generated");
        }

        // Return the split with min cost (the optimal one)
        return (*std::min_element(splits_costs.begin(), splits_costs.end())).second;
    }

    // Get the cycles and power estimate for a list of workloads.
    // This function does not optimize any workloads but simply calculate the cost of that configuration
    PnPEstimates getLayerPerformance(const DPUWorkloads& workloads, const unsigned int runtimeOverhead = 0) override {
        // For an empty list of workloads immedeately return 0
        if (workloads.size() == 0)
            return {0, 0.0f};
        // Get the execution time in cycles of the workloads
        auto workload_cycles = model->DPU(workloads);
        // Compute the total execution cycles
        auto total_cycles = dpu_schedule<unsigned int>(nDPU_per_tile(getWorkloadsDevice(workloads)), workload_cycles,
                                                       runtimeOverhead);
        // Get the average power by computing the workload on ratio by dividing its cycles by the total layer cycles
        float average_power = 0.0f;
        unsigned int idx = 0;
        for (auto wl : workloads) {
            float workload_on_ratio = total_cycles > 0 ? workload_cycles[idx++] / total_cycles : 0.0f;
            average_power += model->DPUPower(wl) * workload_on_ratio;
        }

        // Return a PnP structure with total cycles and average poower
        return {total_cycles, average_power};
    }

    // Get the cycles and power estimate for a layer.
    // This function optimize the layer and returns the optimal configuration as well as expected PnP estimates
    virtual std::pair<DPUWorkloads, PnPEstimates> getLayerPerformance(const DPULayer& layer,
                                                                      const SplitOptions& options) override {
        // Get the optimal workload list for that layer
        auto workloads = intraTileSplit(layer, options);
        // Compute the power and performance estimate for that workloads
        auto pnp = getLayerPerformance(workloads, options.runtimeOverhead);
        // Return a list fo workloads and estimates
        return {workloads, pnp};
    }
};

std::unique_ptr<DPUTiler> getDPUTiler(std::shared_ptr<VPUCostModel> const& _model) {
    return std::make_unique<DPUTilerImplementation>(_model);
}

}  // namespace VPUNN
