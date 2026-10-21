// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_LEGACY_SHV_COMPAT_PROVIDER_H
#define VPUNN_LEGACY_SHV_COMPAT_PROVIDER_H

#include <memory>
#include <stdexcept>
#include <string>

#include "vpu/cycles_interface_types.h"
#include "vpu/shave/shave_cost_providers/shave_cost_provider_interface.h"

namespace VPUNN {

/**
 * @brief Compatibility shim for legacy compiler consumers.
 *
 * Wraps an underlying provider and applies an inverse DPU->SHV scaling on
 * successful cycle values.
 *
 * The conversion uses the DPU/SHV frequency pair provided at construction
 * time. This keeps wrapper behavior explicit and avoids per-call capability
 * checks for conversion inputs.
 */
class LegacyShvCompatProvider final : public IShaveCostProvider {
private:
    const Shared_IShaveCostProvider underlying_provider;
    const std::pair<unsigned int, unsigned int> frequencies;

    CyclesInterfaceType convert_dpu_cycles_to_legacy_shv_cycles(CyclesInterfaceType dpu_cycles) const {
        const auto& [dpu_freq, shv_freq] = frequencies;
        if (dpu_freq <= 0 || shv_freq <= 0) {
            return Cycles::ERROR_INVALID_CONVERSION_TO_CYCLES;
        }

        return Cycles::multiply_by_ratio(dpu_cycles, shv_freq, dpu_freq);
    }

public:
    /// @brief Construct compatibility wrapper with explicit conversion frequencies.
    /// @param provider Underlying provider to delegate base cost computation to.
    /// @param freqs Pair of {DPU frequency, SHV frequency} used for DPU->SHV conversion.
    explicit LegacyShvCompatProvider(const Shared_IShaveCostProvider& provider,
                                     std::pair<unsigned int, unsigned int> freqs)
            : underlying_provider(provider), frequencies(freqs) {
        if (!underlying_provider) {
            throw std::invalid_argument("Underlying provider cannot be null");
        }
    }

    CyclesInterfaceType get_cost(const SHAVEWorkload& workload, std::string* cost_source = nullptr) const override {
        auto cost = underlying_provider->get_cost(workload, cost_source);
        // Preserve failure results without changing them.
        if (!Cycles::isErrorCode(cost)) {
            cost = convert_dpu_cycles_to_legacy_shv_cycles(cost);
        }

        return cost;
    }

    int get_max_num_params() const override {
        return underlying_provider->get_max_num_params();
    }

    std::vector<std::string> get_shave_supported_ops(VPUDevice& device) const override {
        return underlying_provider->get_shave_supported_ops(device);
    }

    std::optional<std::reference_wrapper<const ShaveOpExecutor>> get_shave_instance(const std::string& name,
                                                                                    VPUDevice& device) const override {
        return underlying_provider->get_shave_instance(name, device);
    }
};

}  // namespace VPUNN

#endif  // VPUNN_LEGACY_SHV_COMPAT_PROVIDER_H
