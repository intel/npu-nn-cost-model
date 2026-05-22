// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/power.h"

namespace VPUNN {

class PowerVPU2x {
public:
    constexpr static float getFP_overI8_maxPower_ratio() {
        return 0.87f;  ///< this implies INT is more power hungry (=> power virus int  is the max!)
    }
    static details::DevicePowerLUT make_lut() {
        auto int8_scl_pt =
                details::PowerFactor{1.0f,  ///< adjustor
                                     {      ///< values_lut
                                      {VPUNN::Operation::CONVOLUTION,
                                       {{4, 0.87f}, {5, 0.92f}, {6, 1.0f}, {7, 0.95f}, {8, 0.86f}, {9, 0.87f}}},
                                      {VPUNN::Operation::DW_CONVOLUTION, {{6, 5.84f}}},
                                      {VPUNN::Operation::AVEPOOL, {{6, 32.60f}}},
                                      {VPUNN::Operation::MAXPOOL, {{6, 5.29f}}},
                                      {VPUNN::Operation::ELTWISE, {{7, 232.71f}}}}};

        const float fp_ratio = getFP_overI8_maxPower_ratio();
        auto fp16_scl_pt = details::PowerFactor{1.0f,  ///< adjustor
                                                {      ///< values_lut
                                                 {VPUNN::Operation::CONVOLUTION,
                                                  {{4, 0.87f * fp_ratio},
                                                   {5, 0.92f * fp_ratio},
                                                   {6, 1.0f * fp_ratio},
                                                   {7, 0.95f * fp_ratio},
                                                   {8, 0.86f * fp_ratio},
                                                   {9, 0.87f * fp_ratio}}},
                                                 {VPUNN::Operation::DW_CONVOLUTION, {{6, 5.84f * fp_ratio}}},
                                                 {VPUNN::Operation::AVEPOOL, {{6, 32.60f * fp_ratio}}},
                                                 {VPUNN::Operation::MAXPOOL, {{6, 5.29f * fp_ratio}}},
                                                 {VPUNN::Operation::ELTWISE, {{7, 232.71f * fp_ratio}}}}};
        // VPU2.0 values (Op type: {log2(input_channels): power_factor}))
        return details::DevicePowerLUT{
                {VPUNN::VPUDevice::VPU_2_0, VPUNN::VPUDevice::VPU_2_1},  ///< devices sharing VPU 2.x power LUT
                0.87f,                                                   ///< maxvirus
                {{VPUNN::MPEEngine::SCL,
                  {///< factors
                   {details::ComputePowerTypeClass::INT8, std::move(int8_scl_pt)},
                   {details::ComputePowerTypeClass::FP16, std::move(fp16_scl_pt)}}}}};
    }
};  // PowerVPU2x

class PowerVPU27 {
public:
    constexpr static float getFP_overI8_maxPower_ratio() {
        return 1.3f;  // float more hungry
    }

    static details::DevicePowerLUT make_lut() {
        auto int8_scl_pt = details::PowerFactor{1.0f,  ///< adjustor
                                                {      ///< values_lut
                                                 {VPUNN::Operation::CONVOLUTION, {{6, 1.0f}}},
                                                 {VPUNN::Operation::CM_CONVOLUTION, {{6, 1.0f}}},
                                                 {VPUNN::Operation::DW_CONVOLUTION, {{6, 21.0f}}},
                                                 {VPUNN::Operation::AVEPOOL, {{6, 21.0f}}},
                                                 {VPUNN::Operation::MAXPOOL, {{6, 11.0f}}},
                                                 {VPUNN::Operation::ELTWISE, {{8, 5.0f}}}}};

        const float fp_ratio = getFP_overI8_maxPower_ratio();
        auto fp16_scl_pt = details::PowerFactor{1.0f,  ///< adjustor
                                                {      ///< values_lut
                                                 {VPUNN::Operation::CONVOLUTION, {{6, 1.0f * fp_ratio}}},
                                                 {VPUNN::Operation::CM_CONVOLUTION, {{6, 1.0f * fp_ratio}}},
                                                 {VPUNN::Operation::DW_CONVOLUTION, {{6, 21.0f * fp_ratio}}},
                                                 {VPUNN::Operation::AVEPOOL, {{6, 21.0f * fp_ratio}}},
                                                 {VPUNN::Operation::MAXPOOL, {{6, 11.0f * fp_ratio}}},
                                                 {VPUNN::Operation::ELTWISE, {{8, 5.0f * fp_ratio}}}}};

        return details::DevicePowerLUT{{VPUNN::VPUDevice::VPU_2_7},  ///< devices
                                       1.3f,                         ///< maxvirus
                                       {{VPUNN::MPEEngine::SCL,
                                         {///< factors
                                          {details::ComputePowerTypeClass::INT8, std::move(int8_scl_pt)},
                                          {details::ComputePowerTypeClass::FP16, std::move(fp16_scl_pt)}}}}};
    }
};  // PowerVPU27

class PowerVPU40 : protected PowerVPU27 {
public:
    static details::DevicePowerLUT make_lut() {
        details::DevicePowerLUT this_device = PowerVPU27::make_lut();
        this_device.devices = {VPUNN::VPUDevice::VPU_4_0};
        return this_device;
    }
};  // PowerVPU40

class PowerVPU50 {
public:
    static details::DevicePowerLUT make_lut() {
        auto int8_scl_pt = details::PowerFactor{0.88f,  ///< adjustor
                                                {
                                                        ///< values_lut
                                                        {VPUNN::Operation::CONVOLUTION, {{6, 0.61f}}},
                                                        {VPUNN::Operation::CM_CONVOLUTION, {{6, 0.61f}}},
                                                        {VPUNN::Operation::DW_CONVOLUTION, {{6, 9.63f}}},
                                                        {VPUNN::Operation::AVEPOOL, {{6, 3.57f}}},
                                                        {VPUNN::Operation::MAXPOOL, {{6, 3.23f}}},
                                                        {VPUNN::Operation::ELTWISE, {{8, 29.44f}}},
                                                        {VPUNN::Operation::ELTWISE_MUL, {{8, 29.87f}}},
                                                        {
                                                                VPUNN::Operation::LAYER_NORM,
                                                                {{8, 5.0f}}  // unknown
                                                        },
                                                }};

        auto fp8_scl_pt = details::PowerFactor{0.88f,  ///< adjustor
                                               {
                                                       ///< values_lut
                                                       {VPUNN::Operation::CONVOLUTION, {{6, 0.72f}}},
                                                       {VPUNN::Operation::CM_CONVOLUTION, {{6, 0.72f}}},
                                                       {VPUNN::Operation::DW_CONVOLUTION, {{6, 11.51f}}},
                                                       {VPUNN::Operation::AVEPOOL, {{6, 3.88f}}},
                                                       {VPUNN::Operation::MAXPOOL, {{6, 2.85f}}},
                                                       {VPUNN::Operation::ELTWISE, {{8, 39.87f}}},
                                                       {VPUNN::Operation::ELTWISE_MUL, {{8, 33.30f}}},
                                                       {
                                                               VPUNN::Operation::LAYER_NORM,
                                                               {{8, 5.0f}}  // unknown
                                                       },
                                               }};

        auto fp16_scl_pt = details::PowerFactor{0.88f,  ///< adjustor
                                                {
                                                        ///< values_lut
                                                        {VPUNN::Operation::CONVOLUTION, {{6, 1.0f}}},
                                                        {VPUNN::Operation::CM_CONVOLUTION, {{6, 1.0f}}},
                                                        {VPUNN::Operation::DW_CONVOLUTION, {{6, 13.82f}}},
                                                        {VPUNN::Operation::AVEPOOL, {{6, 3.88f}}},
                                                        {VPUNN::Operation::MAXPOOL, {{6, 2.91f}}},
                                                        {VPUNN::Operation::ELTWISE, {{8, 29.12f}}},
                                                        {VPUNN::Operation::ELTWISE_MUL, {{8, 26.62f}}},
                                                        {VPUNN::Operation::LAYER_NORM, {{8, 5.0f}}},
                                                }};

        return details::DevicePowerLUT{
                {VPUNN::VPUDevice::NPU_5_0, VPUNN::VPUDevice::NPU_5_0_W},  ///< devices sharing NPU 5.x power LUT
                1.0f,                                                      ///< maxvirus
                {{VPUNN::MPEEngine::SCL,
                  {///< factors
                   {details::ComputePowerTypeClass::INT8, std::move(int8_scl_pt)},
                   {details::ComputePowerTypeClass::FP8, std::move(fp8_scl_pt)},
                   {details::ComputePowerTypeClass::FP16, std::move(fp16_scl_pt)}}}}};
    }
};  // PowerVPU50



VPUPowerFactorLUT::dev_lut_t VPUPowerFactorLUT::create_pf_lut() {
    // make it return with nrvo move semantics
    // if it were const, it would do a copy instead of move
    dev_lut_t result_dev_lut{
            {PowerVPU2x::make_lut()}, {PowerVPU27::make_lut()}, {PowerVPU40::make_lut()},
            {PowerVPU50::make_lut()},
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
        details::ComputePowerTypeClass target_class =
                details::ComputePowerTypeClass::INT8;  // default to INT8 as the base reference
        if (performanceInfo.native_comp_on_i8(wl)) {
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