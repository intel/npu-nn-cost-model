// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_DEVICE_HW_CHARACTERISTICS_H
#define VPUNN_DEVICE_HW_CHARACTERISTICS_H

#include "vpu/cycles_interface_types.h"

#include "vpu/dma_types.h"
#include "vpu/types.h"
#include "vpu/utils.h"

namespace VPUNN {
/// @brief this struct contains all hardware characteristics a device can have
///     we use a struct to hold together all characteristics, and we will create a HWcharateristics object with specific
///     values for each device in a device HW characteristics dedicated header file
struct HWCharacteristics {
    unsigned int dpu_freq_clk;            ///< the DPU default frequency in MHz
    unsigned int cmx_freq_clk;            ///< the CMX default frequency in MHz
    unsigned int cmx_word_size_B;         ///< CMX word size in bytes
    int DMA_engine_B;                     ///< DMA engine bytes per cycle
    unsigned int dpu_cmx_num_read_ports;  ///< DPU number of CMX read ports
    float dram_bandwidth_MBps;            ///< the DRAM bandwidth in MB/s for a specific VPU IP

    unsigned int nr_macs;             ///< the DPU number of MACs
    unsigned int fp_ratio;            ///< the ratio of int compute to fp16 compute
    unsigned int nr_ppe;              ///< the number of PPE
    unsigned int input_channels_mac;  ///< the MAC/input channels/cycles for a specific VPU IP
    unsigned int nDPU_per_tile;       ///< the MAC/input channels/cycles for a specific VPU IP
    int dma_ports;                    ///< the channels/Ports of DMA
    float profiling_clk_MHz;          ///< the Profiler clock in MHz
    int profiling_clk_Hz;             ///< the Profiler clock in Hz
};

struct LatencyValues {
    int dramLatency_Nanoseconds;    ///< nanoseconds
    int cmxLatency_CMXClockCycles;  ///< 16 clock cycles at VPU frequency
};

/// this class uses CRTP ( Curiously Recurring Template Pattern )
/// the template parameter SpecificCharacteristics should be the derived class itself
/// this enables static polymorphism, so the base class can call methods from the derived class without using virtual
/// functions (eg: get_DMA_latency() is one of them)
template <typename SpecificCharacteristics>
class DeviceHWCharacteristics {
public:
    constexpr DeviceHWCharacteristics(const HWCharacteristics& hw_characteristics_)
            : hw_characteristics(hw_characteristics_) {
    }

protected:
    const HWCharacteristics& hw_characteristics;

    static constexpr float PROF_CLK_MHz{38.4f};
    static constexpr int PROF_CLK_Hz{38400000};

public:
    constexpr unsigned int get_dpu_freq_clk() const {
        return hw_characteristics.dpu_freq_clk;
    }

    constexpr unsigned int get_cmx_freq_clk() const {
        return hw_characteristics.cmx_freq_clk;
    }

    constexpr unsigned int get_cmx_word_size_B() const {
        return hw_characteristics.cmx_word_size_B;
    }

    constexpr int get_DMA_engine_B() const {
        return hw_characteristics.DMA_engine_B;
    }

    constexpr unsigned int get_dpu_cmx_num_read_ports() const {
        return hw_characteristics.dpu_cmx_num_read_ports;
    }

    constexpr float get_dram_bandwidth_MBps() const {
        return hw_characteristics.dram_bandwidth_MBps;
    }

    /// get the DMA latency in DPU cycles
    constexpr CyclesInterfaceType get_DMA_latency(MemoryLocation location) {
        return static_cast<SpecificCharacteristics*>(this)->get_DMA_latency(location);
    }

    constexpr unsigned int get_nr_macs() const {
        return hw_characteristics.nr_macs;
    }

    constexpr unsigned int get_fp_ratio() const {
        return hw_characteristics.fp_ratio;
    }

    constexpr unsigned int get_nr_ppe() const {
        return hw_characteristics.nr_ppe;
    }

    constexpr unsigned int get_input_channels_mac() const {
        return hw_characteristics.input_channels_mac;
    }

    constexpr unsigned int get_nDPU_per_tile() const {
        return hw_characteristics.nDPU_per_tile;
    }

    constexpr int get_dma_ports() const {
        return hw_characteristics.dma_ports;
    }

    constexpr float get_profiling_clk_MHz() const {
        return hw_characteristics.profiling_clk_MHz;
    }

    constexpr int get_profiling_clk_Hz() const {
        return hw_characteristics.profiling_clk_Hz;
    }
};

/// @brief default/basic HW characteristics values
/// here we have HW characteristics for unknown devices
class Default_HWCharacteristics : public DeviceHWCharacteristics<Default_HWCharacteristics> {
public:
    constexpr Default_HWCharacteristics(): DeviceHWCharacteristics(hw_characteristics_def){};

private:
    inline static constexpr HWCharacteristics hw_characteristics_def{
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

public:
    /// @return 0 to keep the old implementation behavior
    constexpr CyclesInterfaceType get_DMA_latency(MemoryLocation /* location*/) {
        return 0;
    }
};

/// this class keep the common implementation for function get_DMA_latency() for device 2.7, 4.0, 5.0 HW characteristics
/// classes
class DMALatencyCharacteristic {
protected:
    inline static constexpr CyclesInterfaceType get_DMA_latency(MemoryLocation location,
                                                                const HWCharacteristics& hw_characteristics_def,
                                                                const LatencyValues latency_values) {
        //[ns]*[MHz] ->normalize to SI=> [ns/1 000 000 000]*[MHz*1 000 000]=[ns/1000]*[MHz]
        const int dram_DPUCycles{(latency_values.dramLatency_Nanoseconds * (int)hw_characteristics_def.dpu_freq_clk) /
                                 1000};
        // normally (cmx_clk/CMX_frq)*DPU_frq, but multiplication done first to be OK on int
        const int cmx_DPUCycles{latency_values.cmxLatency_CMXClockCycles * (int)hw_characteristics_def.dpu_freq_clk /
                                (int)hw_characteristics_def.cmx_freq_clk};

        return (location == MemoryLocation::DRAM) ? dram_DPUCycles : cmx_DPUCycles;
    }
};

}  // namespace VPUNN

#endif  // VPUNN_DEVICE_HW_CHARACTERISTICS_H
