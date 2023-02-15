// Copyright © 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_SELECT_OPTIMAL_SPLIT_H
#define VPUNN_SELECT_OPTIMAL_SPLIT_H

#include <vpu/types.h>
#include <vpu/utils.h>
#include <vpu_cost_model.h>
#include <vpunn.h>
#include <algorithm>
#include <string>
#include <vector>

namespace VPUNN {

typedef std::tuple<VPUTensor, VPUTensor, ExecutionMode> VPUWorkloadSplit;

// Return the index of the best split from the ones evaluated
std::tuple<unsigned int, unsigned int> select_optimal_split(VPUCostModel& model, unsigned int nDPU, VPUDevice device,
                                                            Operation op,
                                                            const std::vector<std::vector<VPUWorkloadSplit>>& splits,
                                                            const std::array<unsigned int, 2>& kernels,
                                                            const std::array<unsigned int, 2>& strides,
                                                            const std::array<unsigned int, 4>& padding) {
    std::vector<unsigned int> split_cost;

    for (unsigned int split_idx = 0; split_idx < splits.size(); split_idx++) {
        std::vector<unsigned int> wl_cost;
        for (unsigned int wl_idx = 0; wl_idx < splits[split_idx].size(); wl_idx++) {
            // Compute the cost of a single wl
            auto cost = model.DPU(
                    {device,
                     op,
                     {std::get<0>(splits[split_idx][wl_idx])},  // input_0 (activations)
                     //{std::get<0>(splits[split_idx][wl_idx])},  // input_1 (weights) todo: derive appropriately
                     {std::get<1>(splits[split_idx][wl_idx])},  // output
                     kernels,
                     strides,
                     padding,
                     std::get<2>(splits[split_idx][wl_idx])});
            wl_cost.push_back(cost);
        }
        // Compute the cost of running those wl using nDPUs
        split_cost.push_back(dpu_schedule(nDPU, wl_cost));
    }

    // Get the index with minimum cost
    auto min_idx =
            static_cast<unsigned int>(std::min_element(split_cost.begin(), split_cost.end()) - split_cost.begin());

    return std::make_tuple(min_idx, split_cost[min_idx]);
}

}  // namespace VPUNN

#endif  // VPUNN_SELECT_OPTIMAL_SPLIT_H
