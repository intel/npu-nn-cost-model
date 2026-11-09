// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_VPU_ENERGY_POWER_LUT_6X_H
#define VPUNN_VPU_ENERGY_POWER_LUT_6X_H

#include "power.h"

namespace VPUNN {

/// NPU 6.0 power factor LUT based on NVL Activity Factor measurements
class PowerVPU60 {
public:
    static details::DevicePowerLUT make_lut() {
        /**
         * All values were taken from NVL Activity Factor measurements.
         * The values for zm_conv were calculated by fitting weighted inverse function to the dataset.
         */
        auto int8_scl_pt = details::PowerFactor{
                1.0f,  ///< adjustor (measurement-based tuning , initially 1.0 = no impact)
                {
                        ///< values_lut
                        {VPUNN::Operation::CONVOLUTION, {{6, 0.82f}, {7, 0.76f}, {8, 0.71f}, {9, 0.67f}, {10, 0.64f}}},
                        {VPUNN::Operation::CM_CONVOLUTION,
                         {{6, 0.82f}, {7, 0.76f}, {8, 0.71f}, {9, 0.67f}, {10, 0.64f}}},
                        {VPUNN::Operation::DW_CONVOLUTION, {{6, 12.41f}}},
                        {VPUNN::Operation::AVEPOOL, {{6, 5.12f}}},
                        {VPUNN::Operation::MAXPOOL, {{6, 4.75f}}},
                        {VPUNN::Operation::ELTWISE, {{8, 38.69f}}},
                        {VPUNN::Operation::ELTWISE_MUL, {{8, 39.52f}}},
                        /// REDUCE_MS power factor is identical to 1x1 CONVOLUTION (INT8).
                        /// Although REDUCE_MS collapses all channels, measurements show it has the same power signature
                        /// as a 1x1 CONV.
                        {VPUNN::Operation::REDUCE_MS, {{6, 0.82f}, {7, 0.76f}, {8, 0.71f}, {9, 0.67f}, {10, 0.64f}}},
                        // LAYER_NORM and REDUCE_SUMSQUARES not in LUT => lookup returns 0.0f (unsupported)
                }};

        auto fp8_scl_pt = details::PowerFactor{
                1.0f,  ///< adjustor (measurement-based tuning , initially 1.0 = no impact)
                {
                        ///< values_lut
                        {VPUNN::Operation::CONVOLUTION, {{6, 0.96f}, {7, 0.88f}, {8, 0.82f}, {9, 0.78f}, {10, 0.74f}}},
                        {VPUNN::Operation::CM_CONVOLUTION,
                         {{6, 0.96f}, {7, 0.88f}, {8, 0.82f}, {9, 0.78f}, {10, 0.74f}}},
                        {VPUNN::Operation::DW_CONVOLUTION, {{6, 14.03f}}},
                        {VPUNN::Operation::AVEPOOL, {{6, 5.34f}}},
                        {VPUNN::Operation::MAXPOOL, {{6, 4.39f}}},
                        {VPUNN::Operation::ELTWISE, {{8, 46.68f}}},
                        {VPUNN::Operation::ELTWISE_MUL, {{8, 41.95f}}},
                        /// REDUCE_MS power factor is identical to 1x1 CONVOLUTION (FP8).
                        {VPUNN::Operation::REDUCE_MS, {{6, 0.96f}, {7, 0.88f}, {8, 0.82f}, {9, 0.78f}, {10, 0.74f}}},
                        // LAYER_NORM and REDUCE_SUMSQUARES not in LUT => lookup returns 0.0f (unsupported)
                }};

        auto fp16_scl_pt = details::PowerFactor{
                1.0f,  ///< adjustor (measurement-based tuning , initially 1.0 = no impact)
                {
                        ///< values_lut
                        {VPUNN::Operation::CONVOLUTION, {{6, 1.18f}, {7, 1.12f}, {8, 1.07f}, {9, 1.04f}, {10, 1.01f}}},
                        {VPUNN::Operation::CM_CONVOLUTION,
                         {{6, 1.18f}, {7, 1.12f}, {8, 1.07f}, {9, 1.04f}, {10, 1.01f}}},
                        {VPUNN::Operation::DW_CONVOLUTION, {{6, 15.83f}}},
                        {VPUNN::Operation::AVEPOOL, {{6, 4.69f}}},
                        {VPUNN::Operation::MAXPOOL, {{6, 3.80f}}},
                        {VPUNN::Operation::ELTWISE, {{8, 32.63f}}},
                        {VPUNN::Operation::ELTWISE_MUL, {{8, 31.11f}}},
                        /// REDUCE_MS power factor is identical to 1x1 CONVOLUTION (FP16).
                        {VPUNN::Operation::REDUCE_MS, {{6, 1.18f}, {7, 1.12f}, {8, 1.07f}, {9, 1.04f}, {10, 1.01f}}},
                        // LAYER_NORM and REDUCE_SUMSQUARES not in LUT => lookup returns 0.0f (unsupported)
                }};

        auto int8_dcim_pt = details::PowerFactor{
                1.0f,  // adjustor (measurement-based tuning , initially 1.0 = no impact)
                {      ///< values_lut
                 {VPUNN::Operation::CONVOLUTION, {{6, 0.42f}, {7, 0.36f}, {8, 0.33f}, {9, 0.29f}, {10, 0.26f}}},
                 {VPUNN::Operation::CM_CONVOLUTION, {{6, 0.42f}, {7, 0.36f}, {8, 0.33f}, {9, 0.29f}, {10, 0.26f}}}}};

        // Here we would use a scaling factor which is the avg of measured data of uint16 NxRN and int16 RxR for 256
        // channels divided by uint8 NxRN and RxR 256 channels of DCIM
        const float scaling_factor = 0.940f;
        auto int16_dcim_pt =
                details::PowerFactor{1.0f,  ///< adjustor (measurement-based tuning , initially 1.0 = no impact)
                                     {      ///< values_lut
                                      {VPUNN::Operation::CONVOLUTION,
                                       {{6, 0.42f * scaling_factor},
                                        {7, 0.36f * scaling_factor},
                                        {8, 0.33f * scaling_factor},
                                        {9, 0.29f * scaling_factor},
                                        {10, 0.26f * scaling_factor}}},
                                      {VPUNN::Operation::CM_CONVOLUTION,
                                       {{6, 0.42f * scaling_factor},
                                        {7, 0.36f * scaling_factor},
                                        {8, 0.33f * scaling_factor},
                                        {9, 0.29f * scaling_factor},
                                        {10, 0.26f * scaling_factor}}}

                                     }};

        return details::DevicePowerLUT{{VPUNN::VPUDevice::NPU_6_0},  ///< devices
                                       1.0f,                         ///< maxvirus
                                       {{VPUNN::MPEEngine::SCL,
                                         {///< factors
                                          {details::ComputePowerTypeClass::INT8, std::move(int8_scl_pt)},
                                          {details::ComputePowerTypeClass::FP8, std::move(fp8_scl_pt)},
                                          {details::ComputePowerTypeClass::FP16, std::move(fp16_scl_pt)}}},
                                        {VPUNN::MPEEngine::DCIM,
                                         {{details::ComputePowerTypeClass::INT8, std::move(int8_dcim_pt)},
                                          {details::ComputePowerTypeClass::INT16, std::move(int16_dcim_pt)}}}}};
    }
};

}  // namespace VPUNN

#endif  // VPUNN_VPU_ENERGY_POWER_LUT_6X_H
