// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_VPU_PERFORMANCE_MODEL_H
#define VPUNN_VPU_PERFORMANCE_MODEL_H

#include "vpu/cycles_interface_types.h"

#include "core/vpunn_api.h"
#include "vpu/datatype_collection_size.h"
#include "vpu/types.h"
#include "vpu/utils.h"

#include "performance.h"  // to be replaced with the hw characteristics provider.

#include "performance_mode.h"
#include "vpu/hw_characteristics/itf_device_HW_characteristics.h"

#include "vpu/hw_characteristics/HW_characteristics_set_base.h"
#include "vpu/hw_characteristics/HW_characteristics_supersets.h"
#include "vpu/hw_characteristics/device_HW_characteristics_const_repo.h"

namespace VPUNN {

// ?Q do we need different models  for different HWCharacteristicsSet details? DO we templatize?

/**
 * @brief Provides idealized performance modeling for DPU workloads.
 * IT is based on main set of HW characteristics
 *
 * This class is responsible for estimating the theoretical, best-case execution characteristics of DPU operations,
 * such as the number of cycles required and the number of MAC operations performed.
 * It serves as a reference model, ignoring non-ideal hardware effects and focusing on the maximum achievable
 * performance under optimal conditions (e.g., full MAC utilization, ideal sparsity handling).
 *
 * An instance of this class is intended to be use as a provider for performance modeling for DPU workloads.
 * DEvice dependent configuration is provided by the hw_characteristics sets.
 * IN case Performance depends on Devices this should be redesigned
 */
class VPUNN_API HWPerformanceModel {  // to be renamed DpuOpsPerformance !?
protected:
    // use the default main one
    const IHWCharacteristicsSet& hw_characteristics{HWCharacteristicsSuperSets::get_mainConfigurationRef()};

public:
    // provides access to the hardware characteristics of a particular device
    const IDeviceHWCharacteristics& get_hw_info(VPUDevice device) const {
        return hw_characteristics.device(device);
    }

public:
    /**
     * @brief Compute the DPU ideal cycles, considers HW optimizations like sparsity
     * @details Calculates cycles that a single issue scalar CPU would require to execute
     * a DPUWorkload then divides by number of MACs which can be performed in
     * parallel by DPU. All operations are base-lined in the same manner with no
     * non ideal factors considered at all.
     * Like: Number of cycles if all the MAC resources are used 100%.
     * Sparsity is considered for inputs and weights
     *
     * @param wl a DPUWorkload
     * @return  ideal execution DPU cycles
     */
    unsigned long int DPU_Power_IdealCycles(const DPUWorkload& wl) const {
        // Compute the MACs needed to generate the output tensor
        const unsigned long int operations_cnt = compute_HW_MAC_operations_cnt(wl);
        return DPU_MAC_based_cycles(wl, operations_cnt);
    }
    /**
     * @brief Compute the DPU ideal cycles, pure MAC based, no hw optimizations
     * @details Calculates cycles that a single issue scalar CPU would require to execute
     * a DPUWorkload then divides by number of MACs which can be performed in
     * parallel by DPU. All operations are base-lined in the same manner with no
     * non ideal factors considered at all.
     * Like: Number of cycles if all the MAC resources are used 100%.
     *
     * @param wl a DPUWorkload
     * @return  ideal execution DPU cycles
     */
    unsigned long int DPU_Efficency_IdealCycles(const DPUWorkload& wl) const {
        // Compute the MACs needed to generate the output tensor
        const unsigned long int operations_cnt = compute_Ideal_MAC_operations_cnt(wl);
        return DPU_MAC_based_cycles(wl, operations_cnt);
    }

    // protected:
    /**
     * @brief Compute the DPU ideal cycles
     * @details Calculates cycles that a single issue scalar CPU would require to execute
     * a DPUWorkload then divides by number of MACs which can be performed in
     * parallel by DPU. All operations are base-lined in the same manner with no
     * non ideal factors considered at all.
     * Like: Number of cycles if all the MAC resources are used 100%.
     *
     * @param wl a DPUWorkload
     * @param MACs_to_compute how many MAC operations are required to do for the wl. (computed outsided, may or may not
     * consider HW optimizations like sparsity)
     * @return  ideal execution DPU cycles
     */
    unsigned long int DPU_MAC_based_cycles(const DPUWorkload& wl, const unsigned long int MACs_to_compute) const {
        if (wl.outputs.size() == 0) {  // If it computes no output, its duration is 0 cycles
            return 0;
        }
        const IDeviceHWCharacteristics& hw{get_hw_info(wl.device)};
        const unsigned int nr_macs{hw.get_nr_macs()};
        const unsigned int fp_to_int_resource_ratio{hw.get_fp_ratio()};  // more cycles for fp vs int

        const unsigned int nr_macs_adjusted_with_type{
                native_comp_on_fp16(wl) ? ceil_division(nr_macs, fp_to_int_resource_ratio) : nr_macs};

        // Compute the MACs needed to generate the output tensor
        const unsigned long int operations_cnt = MACs_to_compute;

        // Ceil division cycles by DPU MACs for all operations
        const unsigned long int cycles = ceil_division<unsigned long int>(operations_cnt, nr_macs_adjusted_with_type);

        return cycles;
    }

    /**
     * @brief Computes how many MACs are required to generate this output
     * @details Calculates operations that a single issue scalar CPU would require to execute a DPUWorkload, no sparsity
     * or other HW details are taken in consideration
     *
     * @param wl a DPUWorkload
     * @return number of operations
     */
    unsigned long int compute_Ideal_MAC_operations_cnt(const DPUWorkload& wl) const {
        // Compute the MACs needed to generate the output tensor
        unsigned long int operations_cnt{0};
        if (wl.op == Operation::CONVOLUTION || wl.op == Operation::CM_CONVOLUTION) {
            const unsigned long int operations_cnt_base = (unsigned long int)multiply_vector(wl.kernels) *
                                                          (unsigned long int)multiply_vector(wl.outputs[0].get_shape());
            const auto channels{wl.inputs[0].channels()};
            if (wl.device < VPUDevice::VPU_2_7) {
                operations_cnt = operations_cnt_base * channels;
            } else {
                // NPU2.7 or newer. Channel less than 16 are special
                if (channels < 16) {
                    operations_cnt = operations_cnt_base * 16;
                } else {
                    operations_cnt = operations_cnt_base * channels;
                }
            }

        } else if (wl.op == Operation::ELTWISE) {
            operations_cnt = multiply_vector(wl.inputs[0].get_shape());  // kernel is 1
        } else {  // All other operations, including DW convolution and pooling
            operations_cnt = (unsigned long int)multiply_vector(wl.kernels) *
                             (unsigned long int)multiply_vector(wl.outputs[0].get_shape());
        }
        return operations_cnt;
    }
    /**
     * @brief Computes how many MACs are required to generate this output
     * @details Calculates operations that a single issue scalar CPU would require to execute a DPUWorkload considering
     * hardware details like sparsity.
     *
     * @param wl a DPUWorkload
     * @return number of operations
     */
    unsigned long int compute_HW_MAC_operations_cnt(const DPUWorkload& wl) const {
        // Compute the MACs needed to generate the output tensor
        const unsigned long int ideal_operations_cnt{compute_Ideal_MAC_operations_cnt(wl)};
        unsigned long int hw_operations_cnt{ideal_operations_cnt};

        // not active will say 1.0 = dense, 100%
        auto calc_non_zero_operations = [](bool enabled, const float sparsity) {
            if (enabled) {
                const float non_zero_operations_factorWeights_raw{1.0f - sparsity};
                const float non_zero_operations_factorWeights{
                        std::max(0.0f, std::min(1.0f, non_zero_operations_factorWeights_raw))};
                return non_zero_operations_factorWeights;
            }
            return 1.0f;
        };

        // sparsities are present
        if ((wl.weight_sparsity_enabled) || (wl.inputs[0].get_sparsity())) {
            //  model  sparse acceleration for w. OR activation

            const float act_non_zero_operations_fact{
                    calc_non_zero_operations(wl.inputs[0].get_sparsity(), wl.act_sparsity)};

            const float wts_non_zero_operations_fact{
                    calc_non_zero_operations(wl.weight_sparsity_enabled, wl.weight_sparsity)};

            // combined polity is the most influential (this is a conservative approach). Has to be correlated with the
            // implementation in VPUCostModel::runNN_dualsparsity
            const float combined_non_zero_fact{std::min(act_non_zero_operations_fact, wts_non_zero_operations_fact)};

            hw_operations_cnt = static_cast<unsigned long int>(std::ceil(hw_operations_cnt * combined_non_zero_fact));
        }

        return hw_operations_cnt;
    }

    /**
     * @brief Determine whether native computation for workload is floating point or int
     *
     * @param DPUWorkload a DPUWorkload
     * @return bool
     */
    inline bool native_comp_is_any_fp(const DPUWorkload& wl) const {
        // If either activations or weights are FP16/BF16 then native computation is FP16/BF16
        bool found_at_least_one_float = false;
        for (const auto& i : wl.inputs) {
            found_at_least_one_float = found_at_least_one_float || i.is_any_float();
        }
        return found_at_least_one_float;
    }

    inline bool native_comp_on_fp16(const DPUWorkload& wl) const {
        // If either activations or weights are FP16/BF16 then native computation is FP16/BF16
        static_assert(std::tuple_size<decltype(wl.inputs)>{} == 1, "only one input");

        return wl.inputs[0].is_fp16family();
    }

    inline bool native_comp_on_fp8(const DPUWorkload& wl) const {
        // to do : look at weights also?
        static_assert(std::tuple_size<decltype(wl.inputs)>{} == 1, "only one input");
        // to do  redesign xx family methods to be based on Datatype operations
        const VPUTensor wts({1, 1, 1, 1}, wl.get_weight_type());
        return wl.inputs[0].is_fp8family() && (!wts.is_fp16family());
    }

    inline bool native_comp_on_i8(const DPUWorkload& wl) const {
        static_assert(std::tuple_size<decltype(wl.inputs)>{} == 1, "only one input");

        // to do  redesign xx family methods to be based on Datatype operations
        const VPUTensor wts({1, 1, 1, 1}, wl.get_weight_type());

        return wl.inputs[0].is_i8family() && (!wts.is_any_float());
    }
};

}  // namespace VPUNN

#endif  // VPUNN_PERFORMANCE_H
