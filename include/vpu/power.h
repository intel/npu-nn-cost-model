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

    struct Device_LUT {
        VPUDevice device;
        float maxvirus;
        lut_t int8_lut;
        lut_t fp16_lut;
        lut_t fp8_lut{};
    };

    // using device_lut_t = std::tuple<VPUDevice, float /*maxvirus*/, lut_t /*int*/, lut_t /*fp16*/>;
    using device_lut_t = Device_LUT;

    class PowerVPU2x {
    protected:
        static inline const float virus_logical_limit{0.87f};

    public:
        constexpr static float getFP_overI8_maxPower_ratio() {
            return 0.87f;  // this implies INT is more power hungry (=> power virus int  is the max!)
        }
        static device_lut_t make_lut() {
            // VPU2.0 values (Op type: {log2(input_channels): power_factor}))
            const lut_t vpu_values_int{
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
            const lut_t vpu_values_fp16{
                    {Operation::CONVOLUTION,
                     {
                             {4, 0.87f * getFP_overI8_maxPower_ratio()},
                             {5, 0.92f * getFP_overI8_maxPower_ratio()},
                             {6, 1.0f * getFP_overI8_maxPower_ratio()},
                             {7, 0.95f * getFP_overI8_maxPower_ratio()},
                             {8, 0.86f * getFP_overI8_maxPower_ratio()},
                             {9, 0.87f * getFP_overI8_maxPower_ratio()},
                     }},
                    {Operation::DW_CONVOLUTION,
                     {
                             {6, 5.84f * getFP_overI8_maxPower_ratio()},
                     }},
                    {Operation::AVEPOOL,
                     {
                             {6, 32.60f * getFP_overI8_maxPower_ratio()},
                     }},
                    {Operation::MAXPOOL,
                     {
                             {6, 5.29f * getFP_overI8_maxPower_ratio()},
                     }},
                    {Operation::ELTWISE,
                     {
                             {7, 232.71f * getFP_overI8_maxPower_ratio()},
                     }},
            };
            const device_lut_t this_device{VPUDevice::VPU_2_0, virus_logical_limit, vpu_values_int, vpu_values_fp16};
            return this_device;
        }
    };  // PowerVPU2x

    class PowerVPU27 {
    protected:
        static inline const float virus_logical_limit{1.3f};

    public:
        constexpr static float getFP_overI8_maxPower_ratio() {
            return 1.3f;  // float more hungry
        }
        static device_lut_t make_lut() {
            const lut_t vpu_values_int{
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
            const lut_t vpu_values_fp16{
                    {Operation::CONVOLUTION,
                     {
                             {6, 1.0f * getFP_overI8_maxPower_ratio()},
                     }},
                    {Operation::CM_CONVOLUTION,  // CM_Conv is ConvCompressed for VPU2.7
                     {
                             {6, 1.0f * getFP_overI8_maxPower_ratio()},  // keep like for ZMCONV?
                     }},
                    {Operation::DW_CONVOLUTION,
                     {
                             {6, 21.0f * getFP_overI8_maxPower_ratio()},
                     }},
                    {Operation::AVEPOOL,
                     {
                             {6, 21.0f * getFP_overI8_maxPower_ratio()},
                     }},
                    {Operation::MAXPOOL,
                     {
                             {6, 11.0f * getFP_overI8_maxPower_ratio()},
                     }},
                    {Operation::ELTWISE,
                     {
                             {8, 5.0f * getFP_overI8_maxPower_ratio()},
                     }},
            };
            const device_lut_t this_device{VPUDevice::VPU_2_7, virus_logical_limit, vpu_values_int, vpu_values_fp16};
            return this_device;
        }
    };  // PowerVPU27

    class PowerVPU40 : protected PowerVPU27 {
    public:
        static device_lut_t make_lut() {
            device_lut_t this_device = PowerVPU27::make_lut();
            this_device.device = VPUDevice::VPU_4_0;
            // std::get<0>(this_device) = VPUDevice::VPU_4_0;
            return this_device;
        }
    };  // PowerVPU40


    using pf_lut_t = std::vector<device_lut_t>;  ///> device discriminates,

    static pf_lut_t create_pf_lut() {
        const pf_lut_t pf_lut_l{
                {PowerVPU2x::make_lut() /*VPUDevice::VPU_2_0, vpu_2_0_values_int, vpu_2_0_values_fp16*/},
                {PowerVPU27::make_lut() /*VPUDevice::VPU_2_7, vpu_2_7_values_int, vpu_2_7_values_fp16*/},
                {PowerVPU40::make_lut() /*VPUDevice::VPU_4_0, vpu_4_0_values_int, vpu_4_0_values_fp16*/},
        };

        /* coverity[copy_instead_of_move] */
        return pf_lut_l;
    }

    static inline const pf_lut_t pf_lut{create_pf_lut()};  // the only instance

    /**
     * @brief Logarithmic interpolation between entries of the power factor LUT
     * @details the per operation tables are indexed by log2(input channels)
     * Linear interpolation between entries based on log2(input channels) effectively
     * implements logarithmic interpolation.
     *
     * @precondition table must have at least 1 entry and to be ordered low to high
     */
    static float getValueInterpolation(const unsigned int input_ch, const std::map<unsigned int, float>& table) {
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

    static float get_Virus_logical_limit(const VPUDevice device) {
        for (const auto& i_dev : pf_lut) {
            if (i_dev.device == device) {
                return i_dev.maxvirus;  // error fast, no operation found
            }  // device was found
        }
        return 1.0f;  // nothing found , use default
    }

    inline static const lut_t& get_power_lut_based_on_type(const DPUWorkload& wl, const device_lut_t& device_lut) {
        if (native_comp_on_fp16(wl)) {
            return device_lut.fp16_lut;  // FP16;
        } else if (native_comp_on_fp8(wl)) {
            return device_lut.fp8_lut;  // FP8
        } else if (native_comp_on_i8(wl)) {
            return device_lut.int8_lut;  // INT8
        }

        return device_lut.int8_lut;  // default to INT8;
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
    static float getOperationAndPowerVirusAdjustementFactor(const DPUWorkload& wl) {
        //  Get values table for the device
        for (const auto& i_dev : pf_lut) {
            if (i_dev.device == wl.device) {
                const lut_t& operations_table{get_power_lut_based_on_type(wl, i_dev)};

                // Get the power factor value
                for (const auto& i : operations_table) {
                    const Operation operation = std::get<0>(i);

                    if (operation == wl.op) {
                        const std::map<unsigned int, float>& op_values_map = std::get<1>(i);
                        const auto pf_interpolated{
                                getValueInterpolation(wl.inputs[0].channels(), op_values_map)};  // type knowing factor
                        return pf_interpolated;                                                  // early exit OK
                    }
                }
                return 0.0f;  // error fast, no operation found
            }  // device was found
        }
        return 0.0f;  // error , nothing found
    }

    // redesign this
    /*@brief how much the power virus can be exceeded (because is not the max type)*/
    static float get_PowerVirus_exceed_factor(VPUDevice device) {
        const float factor{1.0f * get_Virus_logical_limit(device)};
        return std::max(1.0F, factor);  // no less than power virus
    }
};

}  // namespace VPUNN

#endif  // VPUNN_POWER_H
