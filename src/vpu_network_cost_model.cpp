// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu_network_cost_model.h"

#include <stdexcept>
#include "vpu/cycles_interface_types.h"

namespace VPUNN {

unsigned long int VPUNetworkCostModel::Network(VPUComputationDAG& dag, VPUNetworkStrategy& strategy) {
    unsigned long int cost = 0;
    for (auto layer : dag) {
        if (strategy.exists(layer)) {
            cost = Cycles::cost_adder(cost, layer->cycles(*this, strategy[layer]));
        } else {
            throw_error<std::runtime_error>("Impossible to find a strategy for a layer");
        }
    }

    return cost;
}

}  // namespace VPUNN
