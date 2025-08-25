// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_POWER_H
#define VPUNN_POWER_H

#include <cmath>
#include <list>
#include <map>
#include <tuple>
#include <utility>
#include "vpu/performance.h"
#include "vpu/types.h"
#include "vpu/utils.h"
// #include "vpunn.h"

namespace VPUNN {

/**
 * @brief VPU Power factor LUTs
 * @details The power factor LUT is lookup table that will be indexed by operation
 * and will return another LUT that will be indexed by the number of input channels
 * When there is no entry in the second LUT, the value returned will be the interpolation between its smaller and
 * greater match in table
 */
class VPUPowerFactorLUT {
private:
    using lut_t = std::vector<std::tuple<Operation, std::map<unsigned int, float>>>;
    // unsigned int key represents the log2(number of input channels)
    // and the float value represents the power factor calculated based on simulation measurements
    using pf_lut_t = std::vector<std::tuple<VPUDevice, lut_t>>;
    const pf_lut_t pf_lut{create_pf_lut()};

    static pf_lut_t create_pf_lut() {
        // the values are expected to be for Float operations. For it we will adjust them using
        // getFP_overI8_maxPower_ratio()

        // VPU2.0 values (Op type: {log2(input_channels): power_factor}))
        const lut_t vpu_2_0_values{
                {Operation::CONVOLUTION,
                 {
                         {4, 0.87f},
                         {5, 0.92f},
                         {6, 1.0f},
                         {7, 0.95f},
                         {8, 0.86f},
                         {9, 0.87f},
                 }},
                {Operation::DW_CONVOLUTION,
                 {
                         {6, 5.84f},
                 }},
                {Operation::AVEPOOL,
                 {
                         {6, 32.60f},
                 }},
                {Operation::MAXPOOL,
                 {
                         {6, 5.29f},
                 }},
                {Operation::ELTWISE,
                 {
                         {7, 232.71f},
                 }},
        };

        // VPU2.7 values (Op type: {log2(input_channels): power_factor}))
        const lut_t vpu_2_7_values{
                {Operation::CONVOLUTION,
                 {
                         {6, 1.0f},
                 }},
                {Operation::CM_CONVOLUTION,  // CM_Conv is ConvCompressed for VPU2.7
                 {
                         {6, 1.0f},  // keep like for ZMCONV?
                 }},
                {Operation::DW_CONVOLUTION,
                 {
                         {6, 21.0f},
                 }},
                {Operation::AVEPOOL,
                 {
                         {6, 21.0f},
                 }},
                {Operation::MAXPOOL,
                 {
                         {6, 11.0f},
                 }},
                {Operation::ELTWISE,
                 {
                         {8, 5.0f},
                 }},
        };

        // @todo: add VPU 4.0 values
        const lut_t vpu_4_0_values{vpu_2_7_values};

        const pf_lut_t pf_lut_l{
                {VPUDevice::VPU_2_0, vpu_2_0_values},
                {VPUDevice::VPU_2_7, vpu_2_7_values},
                {VPUDevice::VPU_4_0, vpu_4_0_values},
        };

         /* coverity[copy_instead_of_move] */
        return pf_lut_l;
    }

    /**
     * @brief Logarithmic interpolation between entries of the power factor LUT
     * @details the per operation tables are indexed by log2(input channels)
     * Linear interpolation between entries based on log2(input channels) effectively
     * implements logarithmic interpolation.
     *
     * @precondition table must have at least 1 entry and to be ordered low to high
     */
    float getValueInterpolation(const unsigned int input_ch, const std::map<unsigned int, float>& table) const {
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

public:
    /**
     * @brief Get the value from the LUT+ extra info for a specific workload, represents the relative power factor
     * adjustment towards the PowerVirus (INT8). The factor will take in consideration all aspects of the WL ,
     * operation, type, etc
     *
     * @param wl the workload for which to compute the factor.
     * @return  the adjustment factor
     */
    float getOperationAndPowerVirusAdjustementFactor(const DPUWorkload& wl) const {
        //  Get values table for the device
        for (const auto& i_dev : pf_lut) {
            if (std::get<0>(i_dev) == wl.device) {
                const lut_t& device_table{std::get<1>(i_dev)};

                // Get the power factor value
                for (const auto& i : device_table) {
                    const Operation operation = std::get<0>(i);
                    const std::map<unsigned int, float>& op_values_map = std::get<1>(i);

                    if (operation == wl.op) {
                        const auto basic_pf_interpolated{
                                getValueInterpolation(wl.inputs[0].channels(), op_values_map)};  // type agnostic factor
                        const float fp16_int8_ratio{getFP_overI8_maxPower_ratio(wl.device)};
                        const float pf_value{
                                native_comp_is_fp(wl) ? (basic_pf_interpolated * fp16_int8_ratio)  // FP16 factor
                                                      : (basic_pf_interpolated)                    // INT8 factor
                        };
                        return pf_value;  // early exit
                    }
                }
                return 0.0f;  // error fast
            }                 // device was found
        }
        return 0.0f;  // error , nothing found
    }
    /**
     * @brief Scale the power factor value according to data type [FPmaxP/I8MaxP]
     * @details The entries in the power factor LUTs above are based on UINT8 operation
     * The power for FLOAT16 is different (depending on VPUDevice), here we approximate the
     * difference by scaling by a fixed amount/ratio.
     *
     * @param device that is used
     * @returns the float to int ratio for power considering the device
     */
    constexpr float getFP_overI8_maxPower_ratio(VPUDevice device) const {
        float fp_to_int_ratio{1.0f};
        if (device == VPUDevice::VPU_2_0)
            fp_to_int_ratio = 0.87f;  // this implies INT is more power hungry (=> power virus int  is the max!)
        else if (device == VPUDevice::VPU_2_7)
            fp_to_int_ratio = 1.3f;
        else if (device == VPUDevice::VPU_4_0)
            fp_to_int_ratio = 1.3f;  // mock
        else if (device == VPUDevice::NPU_RESERVED || device == VPUDevice::NPU_RESERVED_W)
            fp_to_int_ratio = 1.3f;  // mock
        else
            fp_to_int_ratio = 1.0f;

        return fp_to_int_ratio;
    }

    /*@brief how much the power virus can be exceeded (because is not the max type)*/
    constexpr float get_PowerVirus_exceed_factor(VPUDevice device) const {
        const float factor{1.0f * getFP_overI8_maxPower_ratio(device)};
        // it may be that not INT8 is the most power hungry mode, we know for sure it is FP16, and reference is now the
        // INT8

        return std::max(1.0F, factor);  // no less than power virus
    }
};

}  // namespace VPUNN

#endif  // VPUNN_POWER_H
