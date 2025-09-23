// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_PERFORMANCE_H
#define VPUNN_PERFORMANCE_H

#include <cassert>
#include <tuple>
#include "vpu/cycles_interface_types.h"
#include "vpu/dma_types.h"
#include "vpu/hw_characteristics/device_HW_characteristics_const_repo.h"
#include "vpu/types.h"
#include "vpu/utils.h"

namespace VPUNN {

// used in VPUX :
//  get_dpu_fclk
//  get_dram_bandwidth_MBps_Legacy
//  get_dram_bandwidth_MBps

/// USes the global defaults configurations
/* coverity[rule_of_five_violation:FALSE] */
class GlobalHarwdwareCharacteristics {
private:
    GlobalHarwdwareCharacteristics() = default;  ///< private constructor to prevent instantiation
    GlobalHarwdwareCharacteristics(const GlobalHarwdwareCharacteristics&) = delete;  ///< no copy constructor
    GlobalHarwdwareCharacteristics& operator=(const GlobalHarwdwareCharacteristics&) =
            delete;                                                             ///< no assignment operator
    GlobalHarwdwareCharacteristics(GlobalHarwdwareCharacteristics&&) = delete;  ///< no move constructor
    GlobalHarwdwareCharacteristics& operator=(GlobalHarwdwareCharacteristics&&) =
            delete;  ///< no move assignment operator

public:  // used in VPUX
    /// @brief Get the DPU default frequency in MHz
    /// Used in VPUX
    /// @param device a VPUDevice
    /// @return unsigned int
    inline static constexpr unsigned int get_dpu_fclk(VPUDevice device) {
        DeviceHWCharacteristicsVariant config = DeviceHWCHaracteristicsConstRepo::get_HWCharacteristics(device);
        return std::visit(
                [](auto& obj) {
                    return obj.get_dpu_freq_clk();
                },
                config);
    }

    /// @brief Get the DRAM bandwidth in MB/s for a specific VPU IP
    /// Actual speed is limited also by the CMX clock  bandwidth
    /// Used in VPUX
    /// @param device a VPUDevice
    /// @return float
    inline static constexpr float get_dram_bandwidth_MBps(VPUDevice device) {
        DeviceHWCharacteristicsVariant config = DeviceHWCHaracteristicsConstRepo::get_HWCharacteristics(device);
        return std::visit(
                [](auto& obj) {
                    return obj.get_dram_bandwidth_MBps();
                },
                config);
    }

    /// @brief Get the DRAM bandwidth in MB/s for a specific VPU IP
    /// Actual speed is limited also by the CMX clock  bandwidth
    /// Used in VPUX
    /// @param device a VPUDevice
    /// @return float
    inline static constexpr float get_dram_bandwidth_MBps_Legacy(VPUDevice device) {
        DeviceHWCharacteristicsVariant config =
                DeviceHWCHaracteristicsConstRepo::get_HWCharacteristics_Legacy(device);  // LEGACY
        return std::visit(
                [](auto& obj) {
                    return obj.get_dram_bandwidth_MBps();
                },
                config);
    }

public:  // others
    /**
     * @brief Get the CMX default frequency in MHz
     *
     * @param device a VPUDevice
     * @return unsigned int
     */
    inline static constexpr unsigned int get_cmx_fclk(VPUDevice device) {
        DeviceHWCharacteristicsVariant config = DeviceHWCHaracteristicsConstRepo::get_HWCharacteristics(device);
        return std::visit(
                [](auto& obj) {
                    return obj.get_cmx_freq_clk();
                },
                config);
    }

    /**
     * @brief Get the MAC/input channels/cycles for a specific VPU IP
     *
     * @param device a VPUDevice
     * @return unsigned int
     */
    inline static constexpr unsigned int nDPU_per_tile(VPUDevice device) {
        DeviceHWCharacteristicsVariant config = DeviceHWCHaracteristicsConstRepo::get_HWCharacteristics(device);
        return std::visit(
                [](auto& obj) {
                    return obj.get_nDPU_per_tile();
                },
                config);
    }

    /**
     * @brief Get the channels/Ports of DMA.
     * Can be used to run one channel per separate tile (like for DDR to CM in case of weights for SOK),
     * cannot be used to run multiple channels when transferring to same tile
     * @param device a VPUDevice
     * @return int
     */
    inline static constexpr int get_dma_ports(VPUDevice device) {
        DeviceHWCharacteristicsVariant config = DeviceHWCHaracteristicsConstRepo::get_HWCharacteristics(device);
        return std::visit(
                [](auto& obj) {
                    return obj.get_dma_ports();
                },
                config);
    }

    /**
     * @brief Get DMA engine bytes per cycle
     *
     * @param device a VPUDevice
     * @return unsigned int
     */
    inline static constexpr int get_DMA_DDR_interface_bytes(VPUDevice device) {
        DeviceHWCharacteristicsVariant config = DeviceHWCHaracteristicsConstRepo::get_HWCharacteristics(device);
        return std::visit(
                [](auto& obj) {
                    return obj.get_DMA_DDR_interface_bytes();
                },
                config);
    }

    ~GlobalHarwdwareCharacteristics() = default;
};

/// helper on top of DeviceHWCharacteristicsVariant configuration specific for a device
/// wraps the visit idiom
class HWCharacteristicsVariantWrap {
private:
    HWCharacteristicsVariantWrap() = default;  ///< private constructor to prevent instantiation
    HWCharacteristicsVariantWrap(const HWCharacteristicsVariantWrap&) = delete;             ///< no copy constructor
    HWCharacteristicsVariantWrap& operator=(const HWCharacteristicsVariantWrap&) = delete;  ///< no assignment operator
    HWCharacteristicsVariantWrap(HWCharacteristicsVariantWrap&&) = delete;                  ///< no move constructor
    HWCharacteristicsVariantWrap& operator=(HWCharacteristicsVariantWrap&&) = delete;  ///< no move assignment operator
public:
    /// dedicated for avoiding the visit in the user code, legacy independent
    static inline constexpr unsigned int get_dpu_fclk(DeviceHWCharacteristicsVariant config) {
        return std::visit(
                [](auto& obj) {
                    return obj.get_dpu_freq_clk();
                },
                config);
    }

    /// dedicated for avoiding the visit in the user code, legacy independent
    static inline constexpr auto get_DMA_latency(DeviceHWCharacteristicsVariant config, MemoryLocation location) {
        return std::visit(
                [location](auto& obj) {
                    return obj.get_DMA_latency(location);
                },
                config);
    }
};

//// global functions temporary kept here due to VPUX compatibility

inline constexpr unsigned int get_dpu_fclk(VPUDevice device) {
    return GlobalHarwdwareCharacteristics::get_dpu_fclk(device);
}

inline constexpr float get_dram_bandwidth_MBps_Legacy(VPUDevice device) {
    return GlobalHarwdwareCharacteristics::get_dram_bandwidth_MBps_Legacy(device);
}

inline constexpr float get_dram_bandwidth_MBps(VPUDevice device) {
    return GlobalHarwdwareCharacteristics::get_dram_bandwidth_MBps(device);
}

}  // namespace VPUNN

#endif  // VPUNN_PERFORMANCE_H
