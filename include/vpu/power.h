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

#include <algorithm>
#include <cmath>
#include <list>
#include <map>
#include <tuple>
#include <utility>
#include <vector>
#include "vpu/performance.h"
#include "vpu/types.h"
#include "vpu/utils.h"
// #include "vpunn.h"
#include "vpu/vpu_performance_model.h"

namespace VPUNN {

namespace details {
using pf_lut_t = std::vector<std::tuple<VPUNN::Operation, std::map<unsigned int, float>>>;

/// @brief Compute type classification based on bit representation and computational convention
enum class ComputePowerTypeClass {
    INT8,   ///< 8-bit integer types (INT8, UINT8)
    INT16,  ///< 16-bit integer types (INT16, UINT16)
    FP8,    ///< 8-bit floating point types (BF8, HF8)
    FP16,   ///< 16-bit floating point types (FLOAT16, BFLOAT16)
};

/// @brief Structure to hold power factor information for a specific operation and compute type class
struct PowerFactor {
    float adjustor{1.0f};   ///< this is an adjustor to be applied at the end of the computation, to be used for fine
                            ///< tuning the model
    pf_lut_t values_lut{};  ///< this is the main LUT, indexed by operation, then by log2(input_channels) with the power
                            ///< factor value
};

struct DevicePowerLUT {
    std::vector<VPUNN::VPUDevice> devices;  ///< one or more devices sharing the same power LUT
    float maxvirus{};                       ///< this is the maximum power virus value for this device
    std::map<VPUNN::MPEEngine, std::map<ComputePowerTypeClass, PowerFactor>>
            factors{};  ///< nested map for O(log n) lookup: engine -> type class -> power info
                        ///< normally we will have one main type (like int8) that is the reference for the power virus,
                        ///< and other types with their max power ratio compared to it.

    /// @brief checks if this LUT covers the given device
    bool has_device(VPUNN::VPUDevice d) const {
        return std::find(devices.begin(), devices.end(), d) != devices.end();
    }
};

}  // namespace details

/**
 * @brief VPU Power factor LUTs
 * @details The power factor LUT is lookup table that will be indexed by operation
 * and will return another LUT that will be indexed by the number of input channels
 * When there is no entry in the second LUT, the value returned will be the interpolation between its smaller and
 * greater match in table
 */
class VPUPowerFactorLUT {
private:
    using dev_lut_t = std::vector<details::DevicePowerLUT>;  ///> device discriminates,

    static dev_lut_t create_pf_lut();

protected:
    static inline const dev_lut_t dev_lut{create_pf_lut()};  // the only instance

private:
    /**
     * @brief Logarithmic interpolation between entries of the power factor LUT
     * @details the per operation tables are indexed by log2(input channels)
     * Linear interpolation between entries based on log2(input channels) effectively
     * implements logarithmic interpolation.
     *
     * @precondition table must have at least 1 entry and to be ordered low to high
     */
    static float getValueInterpolation(const unsigned int input_ch, const std::map<unsigned int, float>& table);

    /**
     * @brief Get the logical limit of the Power Virus for a given device, used for the adjustment factor calculation
     * @details This is used to compute the adjustment factor that will be applied to the power
     * @param device the device for which to get the limit
     * @return the logical limit of the power virus for the device
     */
    static float get_Virus_logical_limit(const VPUDevice device);

    /**
     * @brief Resolve the PowerFactor for a workload's engine and compute type.
     *
     * @return reference to the matching PowerFactor, or a static empty PowerFactor when
     *         the type/engine combination is not supported.
     * @details For SCL and DCIM engine: if the requested type is not found, falls back to INT8.
     *          For requested engines that are not found in the device LUT then returns a static empty PowerFactor.
     */
    static const details::PowerFactor& resolve_power_factor(const DPUWorkload& wl,
                                                            const details::DevicePowerLUT& device_lut,
                                                            const HWPerformanceModel& performanceInfo);

    /**
     * @brief Get the values LUT for a specific workload, represents the relative power factor adjustment towards the
     * PowerVirus (INT8). The factor will take in consideration all aspects of the WL , operation, type, etc
     * @param wl the dpu workload for which to compute the factor.
     * @param device_lut the LUT for the device, used to resolve the PowerFactor and get the values LUT from it.
     * @param performanceInfo the performance model, used to determine the native computation type for the workload
     * which is used as a key to resolve the PowerFactor.
     * @return the values LUT for the workload's operation
     */
    static const details::pf_lut_t& get_power_lut_based_on_type(const DPUWorkload& wl,
                                                                const details::DevicePowerLUT& device_lut,
                                                                const HWPerformanceModel& performanceInfo);

    /**
     * @brief Get the adjustment factor for a specific workload, represents the relative power factor adjustment towards
     * the PowerVirus (INT8). The factor will take in consideration all aspects of the WL , operation, type, etc
     * @param wl the dpu workload for which to compute the factor.
     * @param device_lut the LUT for the device, used to resolve the PowerFactor and get the adjustment factor from it.
     * @param performanceInfo the performance model, used to determine the native computation type for the
     * @return the adjustment factor for the workload's operation and type
     */
    static float get_adjustment_factor_based_on_type(const DPUWorkload& wl, const details::DevicePowerLUT& device_lut,
                                                     const HWPerformanceModel& performanceInfo);

public:
    /**
     * @brief Get the value from the LUT+ extra info for a specific workload, represents the relative power factor
     * adjustment towards the PowerVirus (INT8). The factor will take in consideration all aspects of the WL ,
     * operation, type, etc
     *
     * @param wl the workload for which to compute the factor.
     * @return  the adjustment factor
     */
    static float getOperationAndPowerVirusAdjustementFactor(const DPUWorkload& wl,
                                                            const HWPerformanceModel& performanceInfo);

    /**
     * @brief how much the power virus can be exceeded (because is not the max type)
     * @param device the device for which to get the exceed factor
     * @return the exceed factor for the device
     */
    static float get_PowerVirus_exceed_factor(VPUDevice device);
};

}  // namespace VPUNN

#endif  // VPUNN_POWER_H
