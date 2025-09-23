// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_DEVICE_HW_CHARACTERISTICS_ITF_IMPLEMENTATION_H
#define VPUNN_DEVICE_HW_CHARACTERISTICS_ITF_IMPLEMENTATION_H

#include "vpu/cycles_interface_types.h"

#include "vpu/dma_types.h"
#include "vpu/types.h"
#include "vpu/utils.h"

#include "vpu/hw_characteristics/itf_device_HW_characteristics.h"

namespace VPUNN {

template <typename HWCharT>
class ALT_VPUXX_HWCharacteristics : public IDeviceHWCharacteristics {
private:
    const HWCharT inner_characteristics;

public:
    constexpr ALT_VPUXX_HWCharacteristics() = default;

    // DPU/CMX frequencies
    unsigned int get_dpu_fclk() const override {
        return inner_characteristics.get_dpu_freq_clk();
    }
    unsigned int get_cmx_fclk() const override {
        return inner_characteristics.get_cmx_freq_clk();
    }
    unsigned int get_cmx_word_size_bytes() const override {
        return inner_characteristics.get_cmx_word_size_B();
    }

    // DMA/CMX/DRAM
    int get_DMA_DDR_interface_bytes() const override {
        return inner_characteristics.get_DMA_DDR_interface_bytes();
    }
    unsigned int get_dpu_cmx_num_read_ports() const override {
        return inner_characteristics.get_dpu_cmx_num_read_ports();
    }
    float get_dram_bandwidth_MBps() const override {
        return inner_characteristics.get_dram_bandwidth_MBps();
    }

    // MACs, PPE, ratios, etc
    unsigned int get_nr_macs() const override {
        return inner_characteristics.get_nr_macs();
    }
    unsigned int get_fp_ratio() const override {
        return inner_characteristics.get_fp_ratio();
    }
    unsigned int get_nr_ppe() const override {
        return inner_characteristics.get_nr_ppe();
    }
    unsigned int input_channels_mac() const override {
        return inner_characteristics.get_input_channels_mac();
    }
    unsigned int nDPU_per_tile() const override {
        return inner_characteristics.get_nDPU_per_tile();
    }
    int get_dma_ports() const override {
        return inner_characteristics.get_dma_ports();
    }

    // Profiling
    float get_profiling_clk_MHz() const override {
        return inner_characteristics.get_profiling_clk_MHz();
    }
    int get_profiling_clk_Hz() const override {
        return inner_characteristics.get_profiling_clk_Hz();
    }

    // DMA latency
    CyclesInterfaceType get_DMA_latency(MemoryLocation location) const override {
        return inner_characteristics.get_DMA_latency(location);
    }
    std::tuple<float, float> getDMATRansferEfficiency(const MemoryLocation s, const MemoryLocation d) const override {
        return inner_characteristics.getDMATRansferEfficiency(s, d);
    };

};

}  // namespace VPUNN

#endif  //
