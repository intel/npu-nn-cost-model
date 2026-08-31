// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/vpu_performance_model.h"

#include "vpu/dim_enum.h"
#include "vpu/dpu_dtypes_category_info.h"
#include "vpu/dpu_types.h"
#include "vpu/dpu_workload.h"
#include "vpu/hw_characteristics/itf_device_HW_characteristics.h"
#include "vpu/utils.h"
#include "vpu/vpu_tensor.h"

#include <algorithm>
#include <cmath>
#include <tuple>

namespace VPUNN {

unsigned long int HWPerformanceModel::DPU_MAC_based_cycles(const DPUWorkload& wl,
                                                           const unsigned long int MACs_to_compute) const {
    const IDeviceHWCharacteristics& hw{get_hw_info(wl.device)};
    const unsigned int nr_macs{hw.get_nr_macs()};

    // Effective MACs: reduced when the datatype requires more HW MAC units per operation
    const unsigned int nr_macs_adjusted_with_type{ceil_division(nr_macs, get_dtype_mac_factor(wl, hw))};

    // Ceil division cycles by effective MACs for all operations
    const unsigned long int cycles = ceil_division<unsigned long int>(MACs_to_compute, nr_macs_adjusted_with_type);

    return cycles;
}

unsigned int HWPerformanceModel::get_dtype_mac_factor(const DPUWorkload& wl, const IDeviceHWCharacteristics& hw) const {
    if (native_comp_on_fp16(wl) || native_comp_on_i16(wl)) {
        return hw.get_fp_ratio();
    }
    return 1u;
}

unsigned long int HWPerformanceModel::multiply_tensor_shape(const VPUTensor& tensor, bool align_channels) const {
    auto shape = tensor.get_shape();  // {W, H, C, B}
    if (align_channels) {
        shape[Dim::Act::Z] = round_up(shape[Dim::Act::Z], 16u);
    }
    return static_cast<unsigned long int>(multiply_vector(shape));
}

unsigned long int HWPerformanceModel::compute_Ideal_MAC_operations_cnt(const DPUWorkload& wl) const {
    // When output_autopad=true the ODU pads the output channel dimension to a multiple of 16,
    // so the hardware performs work on the aligned channel count.
    const bool align_out_channels{wl.output_autopad.value_or(false)};

    // Compute the MACs needed to generate the output tensor
    unsigned long int operations_cnt{0};
    if ((wl.op == Operation::CONVOLUTION || wl.op == Operation::CM_CONVOLUTION)  //
        || (wl.op == Operation::REDUCE_MS ||
            wl.op == Operation::REDUCE_SUMSQUARES)  // these are like CONV 1x1 with some special aspects but the
                                                    // same execution pattern, so we can use the same formula.
    ) {
        // REDUCE_ operations collapse all output channels into one; channel alignment does not apply.
        const bool is_reduce{wl.op == Operation::REDUCE_MS || wl.op == Operation::REDUCE_SUMSQUARES};
        const unsigned long int operations_cnt_base{
                is_reduce ? static_cast<unsigned long int>(multiply_vector(
                                    wl.outputs[0].get_shape()))  // reduce ops: equivalent kernel 1x1,
                                                                 // collapses all outputs channels into one
                                                                 // does NOT consider out ch alignment (conservative)
                          : ((unsigned long int)multiply_vector(wl.kernels) *
                             multiply_tensor_shape(wl.outputs[0], align_out_channels))  // considers out ch alignment
        };

        const auto in_channels{wl.inputs[0].channels()};

        if (wl.device < VPUDevice::VPU_2_7) {  // this is for backwards compatibility
            operations_cnt = operations_cnt_base * in_channels;
        } else {
            // NPU2.7 or newer. Channel less than 16 are special
            if (in_channels < 16) {
                // this is normally a Compress conv , or a conv with <16. COmpress conv with <=4 input channels get
                // padded to 4. space optimization.
                unsigned long int in_channels_aligned{16};  // default

                // to keep unaffected already release devices(their behavior), the special Compres_conv is not
                // applied to past ones
                if ((in_channels <= 4) && (wl.op == Operation::CM_CONVOLUTION) && (wl.device > VPUDevice::NPU_5_0_W)) {
                    in_channels_aligned = 4;  // compress conv special
                }
                operations_cnt = operations_cnt_base * in_channels_aligned;
            } else {                                                 // more than 16 channels
                operations_cnt = operations_cnt_base * in_channels;  // input channels are taken as they are
            }
        }

    } else if (wl.op == Operation::ELTWISE) {
        operations_cnt = multiply_vector(wl.inputs[0].get_shape());  // kernel is 1
    } else {  // All other operations, including DW convolution and pooling
        operations_cnt = (unsigned long int)multiply_vector(wl.kernels) *
                         multiply_tensor_shape(wl.outputs[0], align_out_channels);  // considers out ch alignment
    }
    return operations_cnt;
}

unsigned long int HWPerformanceModel::compute_HW_MAC_operations_cnt(const DPUWorkload& wl) const {
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

bool HWPerformanceModel::native_comp_on_fp16(const DPUWorkload& wl) const {
    // If either activations or weights are FP16/BF16 then native computation is FP16/BF16
    static_assert(std::tuple_size<decltype(wl.inputs)>{} == 1, "only one input");
    return DTypeCategoryInfo::is_fp16family_dtype(wl.inputs[0].get_dtype()) || DTypeCategoryInfo::is_fp16family_dtype(wl.get_weight_type());
}

bool HWPerformanceModel::native_comp_on_fp8(const DPUWorkload& wl) const {
    // Native computation is FP8 when the input is FP8-family and weights are not FP16-family.
    static_assert(std::tuple_size<decltype(wl.inputs)>{} == 1, "only one input");
    return DTypeCategoryInfo::is_fp8family_dtype(wl.inputs[0].get_dtype()) && (!DTypeCategoryInfo::is_fp16family_dtype(wl.get_weight_type()));
}

bool HWPerformanceModel::native_comp_on_i16(const DPUWorkload& wl) const {
    static_assert(std::tuple_size<decltype(wl.inputs)>{} == 1, "only one input");
    // Check if input is 16-bit integer family and weights are not floating point.
    return DTypeCategoryInfo::is_i16family_dtype(wl.inputs[0].get_dtype()) && (!DTypeCategoryInfo::is_any_float_dtype(wl.get_weight_type()));
}

bool HWPerformanceModel::native_comp_on_i8(const DPUWorkload& wl) const {
    static_assert(std::tuple_size<decltype(wl.inputs)>{} == 1, "only one input");
    return DTypeCategoryInfo::is_i8family_dtype(wl.inputs[0].get_dtype()) && (!DTypeCategoryInfo::is_any_float_dtype(wl.get_weight_type()));
}

}  // namespace VPUNN
