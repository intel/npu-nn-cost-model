// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_SHAVE_PREFETCH_COST_STRATEGY_H
#define VPUNN_SHAVE_PREFETCH_COST_STRATEGY_H

#include <map>
#include <string>
#include <unordered_map>
#include <utility>

#include "core/vpunn_api.h"
#include "vpu/cycles_interface_types.h"
#include "vpu/dpu_types.h"
#include "vpu/shave_workload.h"

namespace VPUNN {

/**
 * @brief Strategy interface for SHAVE code-prefetch cost lookup.
 *
 * This interface separates cold-start prefetch lookup from runtime cycle estimation.
 * Supports table-backed implementations at the moment.
 */
class VPUNN_API IPrefetchCostStrategy {
public:
    virtual ~IPrefetchCostStrategy() = default;

    /**
     * @brief Returns the one-time SHAVE kernel code-prefetch cost for an operator on a device.
     * @param swl SHAVE workload carrying the operator name and target VPU device
     * @return prefetch cost in cycles or an error code from Cycles
     */
    virtual CyclesInterfaceType get_code_prefetch_cost(const SHAVEWorkload& swl) const = 0;
};

using OperationPrefetchLUT = std::unordered_map<std::string, CyclesInterfaceType>;

/**
 * @brief Strategy backed by measured operation costs for one specific device.
 *
 * Operation lookup uses an unordered_map for O(1) name resolution.
 * Missing operation entries return Cycles::ERROR_SHAVE_PREFETCH_NOT_FOUND.
 */
class SingleDevicePrefetchCostStrategy final : public IPrefetchCostStrategy {
private:
    // Intentionally non-owning: this strategy is expected to bind to a static per-device LUT.
    const OperationPrefetchLUT& measured_costs_;

public:
    /// @brief Construct with a non-owning LUT reference.
    /// @note The LUT must outlive this strategy (production path uses static generated LUTs).
    explicit SingleDevicePrefetchCostStrategy(const OperationPrefetchLUT& measured_costs)
            : measured_costs_(measured_costs) {
    }

    CyclesInterfaceType get_code_prefetch_cost(const SHAVEWorkload& swl) const override {
        const auto op_it = measured_costs_.find(swl.get_name());
        if (op_it == measured_costs_.end()) {
            return Cycles::ERROR_SHAVE_PREFETCH_NOT_FOUND;
        }

        return op_it->second;
    }
};

/**
 * @brief Device-mapped prefetch strategy that delegates to a per-device strategy.
 *
 * Missing device entries return Cycles::ERROR_SHAVE_PREFETCH_DEVICE_NOT_COVERED.
 */
class DeviceMappedPrefetchCostStrategy final : public IPrefetchCostStrategy {
public:
    using DeviceToStrategyMap = std::map<VPUDevice, SingleDevicePrefetchCostStrategy>;

private:
    // Intentionally non-owning: this strategy is expected to bind to a static device->strategy map.
    const DeviceToStrategyMap& device_to_strategy_;

public:
    /// @brief Construct with a non-owning device->strategy map reference.
    /// @note The map must outlive this strategy (production path uses a static map in provider bundles).
    explicit DeviceMappedPrefetchCostStrategy(const DeviceToStrategyMap& device_to_strategy)
            : device_to_strategy_(device_to_strategy) {
    }

    CyclesInterfaceType get_code_prefetch_cost(const SHAVEWorkload& swl) const override {
        const auto strategy_it = device_to_strategy_.find(swl.get_device());
        if (strategy_it == device_to_strategy_.end()) {
            return Cycles::ERROR_SHAVE_PREFETCH_DEVICE_NOT_COVERED;
        }

        return strategy_it->second.get_code_prefetch_cost(swl);
    }
};

}  // namespace VPUNN

#endif  // VPUNN_SHAVE_PREFETCH_COST_STRATEGY_H
