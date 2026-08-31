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

#include "core/vpunn_api.h"
#include "vpu/dpu_types.h"
#include "vpu/dpu_workload.h"
#include "vpu/hw_characteristics/HW_characteristics_set_base.h"
#include "vpu/hw_characteristics/HW_characteristics_supersets.h"
#include "vpu/hw_characteristics/itf_HW_characteristics_set.h"
#include "vpu/hw_characteristics/itf_device_HW_characteristics.h"
#include "vpu/vpu_tensor.h"

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
     * Note: HW MACs are normally given for 8-bit ops; if our workload uses something different (e.g. 16-bit), this
     * has to be taken into consideration
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
     * Note: HW MACs are normally given for 8-bit ops; if our workload uses something different (e.g. 16-bit), this
     * has to be taken into consideration
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
     * Note: the HW MACS are normally given for 8 bit ops, if our workload uses something different, like 16bit, this
     * has to be taken in consideration
     *
     * @param wl a DPUWorkload
     * @param MACs_to_compute how many MAC operations are required to do for the wl. (computed outside, may or may not
     * consider HW optimizations like sparsity)
     * @return  ideal execution DPU cycles
     */
    unsigned long int DPU_MAC_based_cycles(const DPUWorkload& wl, const unsigned long int MACs_to_compute) const;

    /**
     * @brief Returns the factor by which a workload's datatype increases MAC hardware resource consumption
     *        relative to the 8-bit baseline.
     *
     * @details HW MAC units are rated for 8-bit operations.  Wider datatypes consume more MAC units
     *          per logical operation, reducing the effective throughput:
     *          - FP16-family input or weights (FLOAT16, BFLOAT16, FLOAT32): uses fp_ratio from HW info.
     *          - INT16/UINT16 input with non-float weights (i16 family): uses fp_ratio from HW info.
     *            Both 16-bit paths share the same ratio because they consume the same number of MAC
     *            hardware units per operation.
     *          - All other datatypes (INT8, FP8, sub-8-bit ...): factor = 1 (no adjustment).
     *
     * @note  fp_ratio is currently used as a unified proxy for the 8-bit-to-16-bit MAC cost ratio.
     *        It captures both the FP16 and the INT16 cases until a more granular HW field is available.
     *
     * @param wl  The DPU workload whose input and weight types are inspected.
     * @param hw  HW characteristics of the target device, used to retrieve fp_ratio.
     * @return    The MAC resource factor (>= 1).  1 means no extra cost; values > 1 indicate that
     *            effective MACs are reduced by this factor.
     */
    unsigned int get_dtype_mac_factor(const DPUWorkload& wl, const IDeviceHWCharacteristics& hw) const;

    /**
     * @brief Computes the product of all tensor shape dimensions, optionally aligning the channel
     *        dimension (index 2) up to the next multiple of 16 before multiplying.
     *
     * @param tensor          The tensor whose shape is used.
     * @param align_channels  When true, channels are rounded up to the nearest multiple of 16
     *                        before the product is computed.
     * @return                Product of W * H * aligned_C * B.
     */
    unsigned long int multiply_tensor_shape(const VPUTensor& tensor, bool align_channels) const;

    /**
     * @brief Computes how many MACs are required to generate this output
     * @details Calculates operations that a single issue scalar CPU would require to execute a DPUWorkload, no sparsity
     * or other HW details are taken in consideration
     *
     * @param wl a DPUWorkload
     * @return number of operations, does not depend on datatype or how many HW MACS are available.
     */
    unsigned long int compute_Ideal_MAC_operations_cnt(const DPUWorkload& wl) const;

    /**
     * @brief Computes how many MACs are required to generate this output
     * @details Calculates operations that a single issue scalar CPU would require to execute a DPUWorkload considering
     * hardware details like sparsity.
     *
     * @param wl a DPUWorkload
     * @return number of operations
     */
    unsigned long int compute_HW_MAC_operations_cnt(const DPUWorkload& wl) const;

    bool native_comp_on_fp16(const DPUWorkload& wl) const;
    bool native_comp_on_fp8(const DPUWorkload& wl) const;
    bool native_comp_on_i16(const DPUWorkload& wl) const;
    bool native_comp_on_i8(const DPUWorkload& wl) const;
};

}  // namespace VPUNN

#endif  // VPUNN_PERFORMANCE_H
