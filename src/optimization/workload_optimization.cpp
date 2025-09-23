// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include <memory>

#include "core/profiling.h"
#include "vpu/dpu_layer_modes.h"
#include "vpu/optimization/tiler.h"
#include "vpu/optimization/workload_optimization.h"

namespace VPUNN {

// When minimizing the cost we should prioritize choosing a smaller amount of workloads
inline bool operator<(const DPUWorkloadsWithCyclesSplit& lhs, const DPUWorkloadsWithCyclesSplit& rhs) {
    return lhs.workloads.size() < rhs.workloads.size();
}

/**
 * @brief Private implementation of the DPUTiler interface
 *
 */
class DPUTilerImplementation : public IDPUTiler {
private:
    VPUCostModel& model;  /// < The DPU cost model used for performance estimation
    const HWPerformanceModel& get_HWPerformance() const {
        return model.getPerformanceModel();
    }

    VPUDevice getWorkloadsDevice(const DPUWorkloadsWithCyclesSplit& workloads) const {
        if (workloads.workloads.size() == 0) {
            throw_error<std::invalid_argument>("getWorkloadsDevice:empty workloads list");
        }
        VPUDevice device = workloads.workloads[0].device;
        for (unsigned int idx = 1; idx < workloads.workloads.size(); idx++) {
            if (workloads.workloads[idx].device != device) {
                throw_error<std::invalid_argument>("getWorkloadsDevice: more than one device for a workloads list");
            }
        }
        return device;
    }

    std::list<DPUWorkloadsWithCycleCost> generateSplits(const TilingAlgorithmsContainer& algorithms,
                                                        const std::vector<ExecutionMode>& valid_execution_modes,
                                                        const SplitOptions& options) const {
        std::list<DPUWorkloadsWithCycleCost> splits_costs;
        // Loop algorithms, splits, modes and populate the DPUWorkloadsCost list
        auto timeout = SyncStopWatch<std::micro>();
        if (options.maxLatencyUs > 0)
            timeout.start();

        if (options.target == VPUOptimizationTarget::POWER) {
            throw_error<std::runtime_error>("generateSplits: not Handling VPUOptimizationTarget::POWER");
        }

        for (auto& algo : algorithms) {
            for (auto& mode : valid_execution_modes) {
                // in how many pieces to be tried to be split
                const auto split_count_variants = algo->generateSplitPool(options.nDPU, mode);
                for (auto nWorkloads : split_count_variants) {
                    // Return if the max time has elapsed
                    if (options.maxLatencyUs > 0 && timeout.interval() > options.maxLatencyUs) {
                        return splits_costs;
                    }

                    // populates splitVariants with 0, 1 or more workloads vectors
                    std::list<DPUWorkloadsWithCyclesSplit> splitVariants{
                            algo->split_tile_in_workloads(mode, nWorkloads)};
                    for (auto& workloads : splitVariants) {
                        // measure  this variant. try catch , and check its output for errors
                        try {
                            const auto pnp = getLayerPerformance(workloads, options.runtimeOverhead);  // may throw

                            const CyclesInterfaceType wl_cost{
                                    pnp.cycles <= 0 ? Cycles::ERROR_TILE_SPLIT_ZERO_CYC_OUTPUT  // no zero allowed
                                                    : pnp.cycles};

                            if (Cycles::isErrorCode(wl_cost)) {
                                Logger::warning() << "\n Error result (or zero cycles) while computing the performance "
                                                  << "of workloads split variants! "
                                                  << "ERROR code: " << wl_cost << " : " << Cycles::toErrorText(wl_cost)
                                                  << "\n Execution mode: " << (int)mode << " : "
                                                  << ExecutionMode_ToText.at(static_cast<int>(mode))
                                                  << "\n nWorkloads: " << nWorkloads << "\n Algo : " << algo->name()
                                                  << " \n Result: ignoring the cost of this workloads split \n";
                            }

                            // good or bad we keep the result
                            splits_costs.push_back({wl_cost, workloads});

                        } catch (const std::exception& e) {
                            Logger::warning() << "\n Exception thrown while computing the performance of workloads "
                                              << "split variants! "
                                              << "\n Execution mode: " << (int)mode << " : "
                                              << ExecutionMode_ToText.at(static_cast<int>(mode))
                                              << "\n nWorkloads: " << nWorkloads << "\n Algo : " << algo->name()
                                              << "\n Exception: " << e.what() << "\n "
                                              << "\nResult: ignoring the cost of this workloads split \n";

                            // add the error result
                            splits_costs.push_back(
                                    {(CyclesInterfaceType)Cycles::ERROR_TILE_SPLIT_EXCEPTION, workloads});
                        }

                    }  // cost of workloads
                }
            }
        }
        return splits_costs;
    }

public:
    /**
     * @brief Construct a new DPUTilerImplementation object
     *
     * @param _model a shared pointer to a VPUCostModel object
     */
    explicit DPUTilerImplementation(VPUCostModel& _model): model{_model} {
    }

    DPUWorkloadsCost intraTileSplit(
            const DPULayer& layer, const SplitOptions& options,
            std::vector<DPUWorkloadsWithCyclesSplit>* complete_output_splits = nullptr) const override {
        // Get execution modes accepted  (e.g.: ExecutionMode::CUBOID_16x16,.....)
        auto valid_execution_modes = DPULayerModes::getValidExecutionMode(layer);  // based on operation

        // get all in-tile tiling algorithms. Each algo has a copy of Layer.
        TilingAlgorithmsContainer algorithms{getTilingAlgorithms(layer, options)};

        // Compute the cost of each split type.
        // compute splits(one is a vector of DPUWorkload)  and cost for each split.
        std::list<DPUWorkloadsWithCycleCost> splits_costs{generateSplits(algorithms, valid_execution_modes, options)};

        if (splits_costs.size() == 0) {  // nothing to return
            throw_error<std::runtime_error>("intraTileSplit: no valid workload generated");
        }

        if (complete_output_splits != nullptr) {
            for (const auto& split : splits_costs) {
                complete_output_splits->push_back(split.second);  // second is the DPUWorkloads
            }
        }

        // lambda comparator for obtaining the minimum one that has no errors and is not zero!
        auto comp = [](const DPUWorkloadsWithCycleCost& a, const DPUWorkloadsWithCycleCost& b) {
            // zero is not a min candidate
            // error is not a min candidate
            // .first is the  cycle time of the workloads split
            if (Cycles::isErrorCode(a.first) || a.first <= 0) {
                return false;  // a not < b, b might be good or not. If both bad they are equal
            }
            // a is valid here
            if (Cycles::isErrorCode(b.first) || b.first <= 0) {
                return true;  // keep a<b if b is invalid value, and "a" valid
            }
            // both valid
            return (a.first) < (b.first);
        };

        // Return the split with min cost (the optimal one). or the first error code (or zero)
        // iterator to min
        const auto& minimum_split{std::min_element(splits_costs.begin(), splits_costs.end(), comp)};

        return {minimum_split->first, minimum_split->second.workloads};  // DPUWorkloadsCost pair
    }

    PnPEstimates getLayerPerformance(DPUWorkloadsWithCyclesSplit& workloads_split,
                                     const unsigned int runtimeOverhead = 0,
                                     const bool skip_power = true) const override {
        // For an empty list of workloads immediately return 0
        if (workloads_split.workloads.size() == 0)
            return {0, 0.0f};  // no runtime to execute nothing

        // std::vector<CyclesInterfaceType> workload_cycles;
        // workload_cycles.reserve(workloads.size());
        //  Iterate over the workloads and compute the execution time in cycles
        //  The result is stored in the workload_cycles vector
        //  In practice, it was observed that caching DPU calls has lower runtime than doing batched DPU calls.
        std::string info;
        for (size_t idx = 0; idx < workloads_split.workloads.size(); idx++) {
            workloads_split.cycles[idx] = model.DPU(workloads_split.workloads[idx], info);
        }

        const auto how_many_errors{countErrors(workloads_split.cycles)};

        if (how_many_errors > 0) {  // errors
            const auto errIndex =
                    (firstErrorIndex(workloads_split.cycles) >= 0) ? firstErrorIndex(workloads_split.cycles) : 0;
            Logger::warning() << "\n Error result returned by DPU for workloads"
                              << "\n Errors cnt: " << how_many_errors
                              << " , from a wl_list size: " << workloads_split.cycles.size()
                              << "\nFirst ERROR code: " << workloads_split.cycles[errIndex] << " : "
                              << Cycles::toErrorText(workloads_split.cycles[errIndex])
                              << "\n runtimeOverhead: " << runtimeOverhead
                              << "\n Workload of first error: " << workloads_split.workloads[errIndex]
                              << "\n Returning first error for entire workloads";

            return {workloads_split.cycles[errIndex], 0.0f};  // return first error code
        }

        // Compute the total execution cycles, on good values (no overflow protection)
        auto total_cycles = dpu_schedule<CyclesInterfaceType>(
                get_HWPerformance().get_hw_info(getWorkloadsDevice(workloads_split)).nDPU_per_tile(),
                // GlobalHarwdwareCharacteristics::nDPU_per_tile(getWorkloadsDevice(workloads_split)),
                workloads_split.cycles, runtimeOverhead);

        // Get the average power by computing the workload on ratio by dividing its cycles by the total layer cycles
        float average_power = 0.0f;
        if (!skip_power) {
            /*unsigned int idx = 0;
            for (DPUWorkload wl : workloads) {
                float workload_on_ratio = total_cycles > 0 ? (float)workload_cycles[idx++] / total_cycles : 0.0f;
                average_power += 0 * workload_on_ratio;
            }*/
            average_power = 0.0f;
        }

        // Return a PnP structure with total cycles and average power
        return {total_cycles, average_power};
    }

private:
    /// @brief Checks a list of cycle times for errors. counts the errors
    ///
    /// @param workloads_cycles the cycles list
    /// @returns how many errors are present  (zero cycles is not error)
    int countErrors(const std::vector<CyclesInterfaceType>& workloads_cycles) const {
        int counter{0};
        for (const auto t : workloads_cycles) {
            if (Cycles::isErrorCode(t)) {
                ++counter;
            }
        }
        return counter;
    }

    int firstErrorIndex(const std::vector<CyclesInterfaceType>& workloads_cycles) const {
        if (countErrors(workloads_cycles) > 0) {
            for (decltype(workloads_cycles.size()) i = 0; i < workloads_cycles.size(); ++i) {
                if (Cycles::isErrorCode(workloads_cycles[i])) {
                    return static_cast<int>(i);
                }
            }
        }
        return -1;
    }
};

std::unique_ptr<IDPUTiler> getDPUTiler(VPUCostModel& _model) {
    return std::make_unique<DPUTilerImplementation>(_model);
}

}  // namespace VPUNN
