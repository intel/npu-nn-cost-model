// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_HW_CHARACTERISTICS_VPU2_1_H
#define VPUNN_HW_CHARACTERISTICS_VPU2_1_H

#include "vpu/cycles_interface_types.h"

#include "vpu/device_HW_characteristics.h"
#include "vpu/dma_types.h"
#include "vpu/types.h"
#include "vpu/utils.h"

namespace VPUNN {

/// @brief specific VPU 2.1 HW characteristics values
class VPU2_1_HWCharacteristics : public DeviceHWCharacteristics<VPU2_1_HWCharacteristics> {
public:
    constexpr VPU2_1_HWCharacteristics(): DeviceHWCharacteristics(hw_characteristics_def){};

private:
    inline static constexpr HWCharacteristics hw_characteristics_def{
            850,           // dpu_freq_clk
            850,           // cmx_freq_clk
            16,            // cmx_word_size_B
            32,            // DMA_engine_B
            4,             // dpu_cmx_num_read_ports
            20000.0f,      // dram_bandwidth_MBps
            256,           // nr_macs
            4,             // fp_ratio
            16,            // nr_ppe
            1,             // input_channels_mac
            5,             // nDPU_per_tile
            2,             // dma_ports
            PROF_CLK_MHz,  // profiling_clk_MHz
            PROF_CLK_Hz    // profiling_clk_Hz
    };

public:
    /// @return 0 to keep the old implementation behavior
    constexpr CyclesInterfaceType get_DMA_latency(MemoryLocation /*location*/) {
        return 0;
    }
};

}  // namespace VPUNN

#endif  // VPUNN_HW_CHARACTERISTICS_VPU2_1_H
