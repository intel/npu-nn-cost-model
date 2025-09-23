// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_HW_CHARACTERISTICS_VPU2_7_H
#define VPUNN_HW_CHARACTERISTICS_VPU2_7_H

#include "vpu/cycles_interface_types.h"

#include "vpu/dma_types.h"
#include "vpu/hw_characteristics/device_HW_characteristics_base.h"
#include "vpu/types.h"
#include "vpu/utils.h"

#include "vpu/hw_characteristics/device_HW_characterisics_itf_impl.h"  // for ALTERNATIVE interface

namespace VPUNN {

/// @brief specific VPU 2.7 HW characteristics values
class VPU2_7_HWCharacteristics : public DeviceHWCharacteristicsBase {
public:
    constexpr VPU2_7_HWCharacteristics(): DeviceHWCharacteristicsBase(hw_characteristics_def, latency_values_def) {};

private:
    inline static constexpr HWCharacteristicsRawData hw_characteristics_def{
            1300,          // dpu_freq_clk
            975,           // cmx_freq_clk
            16,            // cmx_word_size_B
            32,            // DMA_engine_B
            8,             // dpu_cmx_num_read_ports
            27000.0f,      // dram_bandwidth_MBps
            2048,          // nr_macs
            2,             // fp_ratio
            64,            // nr_ppe
            8,             // input_channels_mac
            1,             // nDPU_per_tile
            2,             // dma_ports
            PROF_CLK_MHz,  // profiling_clk_MHz
            PROF_CLK_Hz    // profiling_clk_Hz
    };

    inline static constexpr LatencyValuesRawData latency_values_def{
            956,  // dramLatency_Nanoseconds
            16,   // cmxLatency_CMXClockCycles
            0     // postTimeLatency_CMXClockCycles
    };
};

class VPU2_7_HWCharacteristics_legacy : public DeviceHWCharacteristicsBase {
public:
    constexpr VPU2_7_HWCharacteristics_legacy()
            : DeviceHWCharacteristicsBase(hw_characteristics_def, latency_values_def) {};

private:
    inline static constexpr HWCharacteristicsRawData hw_characteristics_def{
            1300,          // dpu_freq_clk
            975,           // cmx_freq_clk
            16,            // cmx_word_size_B
            32,            // DMA_engine_B
            8,             // dpu_cmx_num_read_ports
            27000.0f,      // dram_bandwidth_MBps
            2048,          // nr_macs
            2,             // fp_ratio
            64,            // nr_ppe
            8,             // input_channels_mac
            1,             // nDPU_per_tile
            2,             // dma_ports
            PROF_CLK_MHz,  // profiling_clk_MHz
            PROF_CLK_Hz    // profiling_clk_Hz
    };

    inline static constexpr LatencyValuesRawData latency_values_def{
            956,  // dramLatency_Nanoseconds
            16,   // cmxLatency_CMXClockCycles
            0     // postTimeLatency_CMXClockCycles
    };
};

using IAlt_VPU2_7_HWCharacteristics = ALT_VPUXX_HWCharacteristics<VPU2_7_HWCharacteristics>;
using IAlt_VPU2_7_HWCharacteristics_legacy = ALT_VPUXX_HWCharacteristics<VPU2_7_HWCharacteristics_legacy>;

}  // namespace VPUNN

#endif  // VPUNN_HW_CHARACTERISTICS_VPU2_7_H
