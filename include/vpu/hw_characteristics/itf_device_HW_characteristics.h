// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_ITF_DEVICE_HW_CHARACTERISTICS_H
#define VPUNN_ITF_DEVICE_HW_CHARACTERISTICS_H

#include "vpu/cycles_interface_types.h"

#include "vpu/dma_types.h"
#include "vpu/types.h"
#include "vpu/utils.h"

namespace VPUNN {

/// Device based characteristics
class IDeviceHWCharacteristics {
public:
    virtual ~IDeviceHWCharacteristics() = default;

    // DPU/CMX frequencies
    virtual unsigned int get_dpu_fclk() const = 0;
    virtual unsigned int get_cmx_fclk() const = 0;
    virtual unsigned int get_cmx_word_size_bytes() const = 0;

    // DMA/CMX/DRAM
    virtual int get_DMA_DDR_interface_bytes() const = 0;
    virtual unsigned int get_dpu_cmx_num_read_ports() const = 0;
    virtual float get_dram_bandwidth_MBps() const = 0;

    // MACs, PPE, ratios, etc
    virtual unsigned int get_nr_macs() const = 0;
    virtual unsigned int get_fp_ratio() const = 0;
    virtual unsigned int get_nr_ppe() const = 0;
    virtual unsigned int input_channels_mac() const = 0;
    virtual unsigned int nDPU_per_tile() const = 0;
    virtual int get_dma_ports() const = 0;

    // Profiling
    virtual float get_profiling_clk_MHz() const = 0;
    virtual int get_profiling_clk_Hz() const = 0;

    // dma
    virtual CyclesInterfaceType get_DMA_latency(MemoryLocation location) const = 0;
    virtual std::tuple<float, float> getDMATRansferEfficiency(const MemoryLocation s, const MemoryLocation d) const = 0;
};

}  // namespace VPUNN

#endif  //
