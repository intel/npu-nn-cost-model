// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_NETWORK_COST_MODEL_H
#define VPUNN_NETWORK_COST_MODEL_H

#include "vpu/graph.h"
#include "vpu_layer_cost_model.h"

namespace VPUNN {

/**
 * @brief VPU Network strategy type
 *
 */
class VPUNetworkStrategy {
private:
    VPUComputeNodeMap<VPULayerStrategy> _map;

public:
    explicit VPUNetworkStrategy() {};

    VPULayerStrategy& operator[](const std::shared_ptr<VPUComputeNode>& _key) {
        return _map[_key];
    }

    bool exists(const std::shared_ptr<VPUComputeNode>& _key) {
        return _map.exists(_key);
    }

    VPUNetworkStrategy& set(const std::shared_ptr<VPUComputeNode>& _key, const VPULayerStrategy& val) {
        _map[_key] = val;
        return *this;
    }
};

/**
 * @brief The VPUNN network cost model (also called VPUNN Level3 API)
 *
 */
class VPUNN_API VPUNetworkCostModel
        : public VPULayerCostModel,
          virtual protected VPU_MutexAcces  // for mutex access
{
public:
    /**
     * @brief Using the same VPULayerCostModel constructor
     *
     */
    using VPULayerCostModel::VPULayerCostModel;

    /**
     * @brief Compute the cost of executing a network with a specific per-layer strategy
     *
     * @param dag a VPUComputationDAG representing the network to estimate
     * @param strategy a per-layer strategy
     * @return unsigned long int
     */
    unsigned long int Network(VPUComputationDAG& dag, VPUNetworkStrategy& strategy) {
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
};

}  // namespace VPUNN

#endif  // VPUNN_NETWORK_COST_MODEL_H
