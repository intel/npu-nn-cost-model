// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_HW_CHARACTERISTICS_DEFAULT_H
#define VPUNN_HW_CHARACTERISTICS_DEFAULT_H

#include "vpu/cycles_interface_types.h"

#include "vpu/dma_types.h"
#include "vpu/hw_characteristics/device_HW_characteristics_base.h"
#include "vpu/types.h"
#include "vpu/utils.h"

#include "vpu/hw_characteristics/device_HW_characterisics_itf_impl.h"  // for ALTERNATIVE interface

namespace VPUNN {
/// @brief default/basic HW characteristics values
/// here we have HW characteristics for unknown devices
class Default_HWCharacteristics : public DeviceHWCharacteristicsBase {
public:
    constexpr Default_HWCharacteristics(): DeviceHWCharacteristicsBase(hw_characteristics_def, latency_values_def) {};

private:
    inline static constexpr HWCharacteristicsRawData hw_characteristics_def{
            1,     // dpu_freq_clk
            1,     // cmx_freq_clk
            16,    // cmx_word_size_B
            -1,    // DMA_engine_B
            8,     // dpu_cmx_num_read_ports
            1.0F,  // dram_bandwidth_MBps
            2048,  // nr_macs
            2,     // fp_ratio
            64,    // nr_ppe
            8,     // input_channels_mac
            1,     // nDPU_per_tile
            1,     // dma_ports
            0.0F,  // profiling_clk_MHz
            0      // profiling_clk_Hz
    };

    inline static constexpr LatencyValuesRawData latency_values_def{0, 0, 0};  // no action
};

using IAlt_Default_HWCharacteristics = ALT_VPUXX_HWCharacteristics<Default_HWCharacteristics>;

}  // namespace VPUNN

#endif
