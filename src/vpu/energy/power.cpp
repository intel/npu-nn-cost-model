// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/energy/power.h"
#include "vpu/energy/power_lut_2x.h"
#include "vpu/energy/power_lut_4x.h"

#include "vpu/energy/power_lut_5x.h"

#include "vpu/energy/power_lut_6x.h"


namespace VPUNN {

VPUPowerFactorLUT::dev_lut_t VPUPowerFactorLUT::create_pf_lut() {
    // make it return with nrvo move semantics
    // if it were const, it would do a copy instead of move
    dev_lut_t result_dev_lut{
            {PowerVPU2x::make_lut()}, {PowerVPU27::make_lut()}, {PowerVPU40::make_lut()},
            {PowerVPU50::make_lut()},
            {PowerVPU60::make_lut()},
    };

    // clang and gcc does not support to use std::move here, so we need suppression
    /* coverity[copy_instead_of_move] */
    return result_dev_lut;
}

float VPUPowerFactorLUT::getValueInterpolation(const unsigned int input_ch,
                                               const std::map<unsigned int, float>& table) {
    assert(table.size() >= 1);
    if (table.size() == 1) {
        // Single entry table - no interpolation
        return table.begin()->second;
    }

    // Get the smaller and greater neighbor
    // const unsigned int max_ch_log2 = (unsigned int)ceil(log2(8192));  // Max input channels
    unsigned int smaller = table.cbegin()->first;   // what's before first value is equal to it
    unsigned int greater = table.crbegin()->first;  // what's after last value is equal to it

    const float input_ch_log2 = std::log2((float)input_ch);

    for (const auto& it : table) {
        // Find the index below or at input_ch
        if (((float)it.first <= input_ch_log2) && (it.first > smaller))
            smaller = it.first;

        // Find the index above or at input_ch
        if (((float)it.first >= input_ch_log2) && (it.first < greater))
            greater = it.first;

        if (smaller == greater)
            break;
    }

    const float interval = (float)(greater - smaller);
    float interp_value = 0;
    if (interval > 0) {
        // Logarithmic interpolation between entries
        interp_value = table.at(smaller) +
                       ((input_ch_log2 - (float)smaller) / interval) * (table.at(greater) - table.at(smaller));
    } else {
        // Direct hit - no interpolation required
        interp_value = table.at(smaller);
    }
    return interp_value;
}

float VPUPowerFactorLUT::get_Virus_logical_limit(const VPUDevice device) {
    for (const auto& i_dev : dev_lut) {
        if (i_dev.has_device(device)) {
            return i_dev.maxvirus;
        }
    }
    return 1.0f;  // nothing found , use default
}

const details::PowerFactor& VPUPowerFactorLUT::resolve_power_factor(const DPUWorkload& wl,
                                                                    const details::DevicePowerLUT& device_lut,
                                                                    const HWPerformanceModel& performanceInfo) {
    static const details::PowerFactor empty_power_factor{};  ///< returned when type/engine is unsupported

    // Find the engine map
    // First we try to find the requested engine and type for the LUT
    const auto engine_it = device_lut.factors.find(wl.mpe_engine);
    if (engine_it != device_lut.factors.end()) {  // Engine is found in the device LUT
        const auto& type_class_map = engine_it->second;

        // Determine which type class to look for based on native computation
        // Order matters: check more specific types before more general ones
        // INT16 must be checked before INT8 since INT16 types were previously falling into INT8
        details::ComputePowerTypeClass target_class =
                details::ComputePowerTypeClass::INT8;  // default to INT8 as the base reference

        if (performanceInfo.native_comp_on_i16(wl)) {
            target_class = details::ComputePowerTypeClass::INT16;
        } else if (performanceInfo.native_comp_on_i8(wl)) {
            target_class = details::ComputePowerTypeClass::INT8;
        } else if (performanceInfo.native_comp_on_fp16(wl)) {
            target_class = details::ComputePowerTypeClass::FP16;
        } else if (performanceInfo.native_comp_on_fp8(wl)) {
            target_class = details::ComputePowerTypeClass::FP8;
        }

        // Search for requested type class in the engine map
        // Direct O(log n) lookup using type class
        const auto type_it = type_class_map.find(target_class);
        if (type_it != type_class_map.end()) {
            return type_it->second;
        }

        // If the requested type class is not found, we can consider fallback strategies here if needed
        // For SCL: fallback to INT8 (SCL should always have INT8 as base reference)
        // For DCIM: fallback to INT8 (if DCIM is defined, then INT8 should be defined as well, as it's the base
        // reference for power factors)
        if (engine_it->first == VPUNN::MPEEngine::SCL ||   // fallback to INT8 for SCL if requested type is not found
            engine_it->first == VPUNN::MPEEngine::DCIM) {  // fallback to INT8 for DCIM if requested type is not found
            const auto int8_it = type_class_map.find(details::ComputePowerTypeClass::INT8);
            if (int8_it != type_class_map.end()) {
                return int8_it->second;
            }
        }
        /**
         * @brief If at the fallback mechanism the method does not return a details::PowerFactor, it means that the
         * device with requested engine does not define INT8 for it And this behavior should not happen, an assert or a
         * throw may break the existing functionality, that's why we rely on tests to avoid wrong configuration of not
         * specifying INT8 for any engine type
         */
    }

    // The requested engine is not found in the device LUT or no suitable type class is found, return empty power factor
    return empty_power_factor;
}

const details::pf_lut_t& VPUPowerFactorLUT::get_power_lut_based_on_type(const DPUWorkload& wl,
                                                                        const details::DevicePowerLUT& device_lut,
                                                                        const HWPerformanceModel& performanceInfo) {
    const auto& pf = resolve_power_factor(wl, device_lut, performanceInfo);
    return pf.values_lut;
}

float VPUPowerFactorLUT::get_adjustment_factor_based_on_type(const DPUWorkload& wl,
                                                             const details::DevicePowerLUT& device_lut,
                                                             const HWPerformanceModel& performanceInfo) {
    const auto& pf = resolve_power_factor(wl, device_lut, performanceInfo);
    return pf.adjustor;
}

float VPUPowerFactorLUT::getOperationAndPowerVirusAdjustementFactor(const DPUWorkload& wl,
                                                                    const HWPerformanceModel& performanceInfo) {
    //  Get values table for the device
    for (const auto& i_dev : dev_lut) {
        if (i_dev.has_device(wl.device)) {
            const details::pf_lut_t& operations_table{get_power_lut_based_on_type(wl, i_dev, performanceInfo)};

            // Get the power factor value
            for (const auto& i : operations_table) {
                const Operation operation = std::get<0>(i);

                if (operation == wl.op) {
                    const std::map<unsigned int, float>& op_values_map = std::get<1>(i);
                    const auto pf_interpolated{
                            getValueInterpolation(wl.inputs[0].channels(), op_values_map)};  // type knowing factor
                    const float pf_adjusted{pf_interpolated * get_adjustment_factor_based_on_type(
                                                                      wl, i_dev, performanceInfo)};  // type adjuster
                    return pf_adjusted;                                                              // early exit OK
                }
            }
            return 0.0f;  // error fast, no operation found
        }  // device was found
    }
    return 0.0f;  // error , nothing found
}

float VPUPowerFactorLUT::get_PowerVirus_exceed_factor(VPUDevice device) {
    const float factor{1.0f * get_Virus_logical_limit(device)};
    return std::max(1.0F, factor);  // no less than power virus
}

}  // namespace VPUNN