// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/optimization/workload_cost_evaluator.h"
#include "core/logger.h"
#include "vpu/utils.h"
#include "vpu_cost_model.h"
#include <exception>

namespace VPUNN {

WorkloadCostEvaluator::WorkloadCostEvaluator(const VPUCostModel& cost_model)
        : cost_model_(cost_model) {
}

CyclesInterfaceType WorkloadCostEvaluator::measureWorkload(const DPUWorkload& workload, std::string& info) const {
    return cost_model_.DPU(workload, info);
}

void WorkloadCostEvaluator::computeWorkloadCycles(DPUWorkloadsWithCyclesSplit& workloads_split) const {
    std::string info;
    // Ensure cycles vector is large enough to hold all workload cycle values.
    workloads_split.cycles.resize(workloads_split.workloads.size());
    for (size_t idx = 0; idx < workloads_split.workloads.size(); idx++) {
        workloads_split.cycles[idx] = cost_model_.DPU(workloads_split.workloads[idx], info);
    }
}

DPULayer WorkloadCostEvaluator::createTestLayer(const DPULayer& base_layer, unsigned int channels) const {
    DPULayer test_layer = base_layer;
    
    // Modify output channels while preserving other dimensions
    test_layer.outputs[0] = VPUTensor(
        {base_layer.outputs[0].width(), 
         base_layer.outputs[0].height(), 
         channels, 
         base_layer.outputs[0].batches()},
        base_layer.outputs[0]);

    // For DW_CONVOLUTION (and similar operations), input and output channels must match
    // Update input channels to match output channels
    if (base_layer.op == Operation::DW_CONVOLUTION || 
        base_layer.op == Operation::AVEPOOL ||
        base_layer.op == Operation::MAXPOOL) {
        test_layer.inputs[0] = VPUTensor(
            {base_layer.inputs[0].width(), 
             base_layer.inputs[0].height(), 
             channels,  // Match output channels
             base_layer.inputs[0].batches()},
            base_layer.inputs[0]);
    }

    return test_layer;
}

std::vector<SplitBlock> WorkloadCostEvaluator::measureCandidateBlocks(
        const DPULayer& base_layer,
        const ExecutionMode& mode,
        const std::vector<unsigned int>& candidate_sizes) const {
    std::vector<SplitBlock> measured_blocks;
    measured_blocks.reserve(candidate_sizes.size());

    std::string info;
    for (auto size : candidate_sizes) {
        // Create a test layer with this channel count
        DPULayer test_layer = createTestLayer(base_layer, size);

        // Create workload from test layer
        DPUWorkload test_wl(test_layer);
        test_wl.execution_order = mode;

        // Measure the cost
        CyclesInterfaceType measured_cost = measureWorkload(test_wl, info);

        // Only add blocks with valid (non-error) costs
        if (!Cycles::isErrorCode(measured_cost) && measured_cost > 0) {
            measured_blocks.emplace_back(static_cast<int>(size), measured_cost);
        }
    }

    return measured_blocks;
}

const HWPerformanceModel& WorkloadCostEvaluator::getHWPerformance() const {
    return cost_model_.getPerformanceModel();
}

VPUDevice WorkloadCostEvaluator::getWorkloadsDevice(const DPUWorkloadsWithCyclesSplit& workloads) const {
    if (workloads.workloads.size() == 0) {
        throw_error<std::invalid_argument>("getWorkloadsDevice: empty workloads list");
    }
    VPUDevice device = workloads.workloads[0].device;
    for (unsigned int idx = 1; idx < workloads.workloads.size(); idx++) {
        if (workloads.workloads[idx].device != device) {
            throw_error<std::invalid_argument>("getWorkloadsDevice: more than one device for a workloads list");
        }
    }
    return device;
}

int WorkloadCostEvaluator::countErrors(const std::vector<CyclesInterfaceType>& workloads_cycles) const {
    int counter{0};
    for (const auto t : workloads_cycles) {
        if (Cycles::isErrorCode(t)) {
            ++counter;
        }
    }
    return counter;
}

int WorkloadCostEvaluator::firstErrorIndex(const std::vector<CyclesInterfaceType>& workloads_cycles) const {
    if (countErrors(workloads_cycles) > 0) {
        for (decltype(workloads_cycles.size()) i = 0; i < workloads_cycles.size(); ++i) {
            if (Cycles::isErrorCode(workloads_cycles[i])) {
                return static_cast<int>(i);
            }
        }
    }
    return -1;
}

CyclesInterfaceType WorkloadCostEvaluator::computeSplitCycles(DPUWorkloadsWithCyclesSplit& workloads_split,
                                                               const unsigned int runtimeOverhead) const {
    // For an empty list of workloads immediately return 0
    if (workloads_split.workloads.size() == 0)
        return 0;  // no runtime to execute nothing

    // Compute execution cycles for all workloads
    computeWorkloadCycles(workloads_split);

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

        return workloads_split.cycles[errIndex];  // return first error code
    }

    // Compute the total execution cycles with scheduling
    auto total_cycles = dpu_schedule<CyclesInterfaceType>(
            getHWPerformance().get_hw_info(getWorkloadsDevice(workloads_split)).nDPU_per_tile(),
            workloads_split.cycles, runtimeOverhead);

    return total_cycles;
}

}  // namespace VPUNN
