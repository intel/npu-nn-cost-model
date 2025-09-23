// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_DEVICE_HW_CHARACTERISTICS_BASE_H
#define VPUNN_DEVICE_HW_CHARACTERISTICS_BASE_H

#include "vpu/cycles_interface_types.h"

#include "vpu/dma_types.h"
#include "vpu/types.h"
#include "vpu/utils.h"

namespace VPUNN {
/// @brief this struct contains all hardware characteristics a device can have
///     we use a struct to hold together all characteristics, and we will create a HWcharateristics object with specific
///     values for each device in a device HW characteristics dedicated header file
struct HWCharacteristicsRawData {
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

struct LatencyValuesRawData {
    int dramLatency_Nanoseconds;         ///< nanoseconds
    int cmxLatency_CMXClockCycles;       ///< 16 clock cycles at VPU frequency
    int postTimeLatency_CMXClockCycles;  ///< common for DDR and CMX,  cc at VPU frequency
    float mixedMemoryEfficiency{1.0f};   ///< the mixed memory efficiency[0..1.0], default is 1.0f
};

/// @brief Base class for device hardware characteristics, providing access to various hardware parameters and common
/// computations.
/// All have to be constexpr
class DeviceHWCharacteristicsBase {
public:
    constexpr DeviceHWCharacteristicsBase(const HWCharacteristicsRawData& hw_characteristics_,
                                          const LatencyValuesRawData& latency_values_)
            : hw_characteristics{hw_characteristics_}, latency_values{latency_values_} {
    }

protected:
    const HWCharacteristicsRawData& hw_characteristics;
    const LatencyValuesRawData& latency_values;

    static constexpr float PROF_CLK_MHz{38.4f};
    static constexpr int PROF_CLK_Hz{38400000};

public:
    // * Used in VPUX
    constexpr unsigned int get_dpu_freq_clk() const {
        return hw_characteristics.dpu_freq_clk;
    }

    constexpr unsigned int get_cmx_freq_clk() const {
        return hw_characteristics.cmx_freq_clk;
    }

    constexpr unsigned int get_cmx_word_size_B() const {
        return hw_characteristics.cmx_word_size_B;
    }

    constexpr int get_DMA_DDR_interface_bytes() const {
        return hw_characteristics.DMA_engine_B;
    }

    constexpr unsigned int get_dpu_cmx_num_read_ports() const {
        return hw_characteristics.dpu_cmx_num_read_ports;
    }

    // * Used in VPUX
    constexpr float get_dram_bandwidth_MBps() const {
        return hw_characteristics.dram_bandwidth_MBps;
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

    constexpr CyclesInterfaceType get_DMA_latency(MemoryLocation location) const {
        ////[ns]*[MHz] ->normalize to SI=> [ns/1 000 000 000]*[MHz*1 000 000]=[ns/1000]*[MHz]
        const int dram_DPUCycles{
                ((latency_values.dramLatency_Nanoseconds * (int)hw_characteristics.dpu_freq_clk) /
                 1000)  // absolute time part
                + (latency_values.postTimeLatency_CMXClockCycles * (int)hw_characteristics.dpu_freq_clk /
                   (int)hw_characteristics.cmx_freq_clk)  // cmx cycles part
        };
        // normally (cmx_clk/CMX_frq)*DPU_frq, but multiplication done first to be OK on int
        const int cmx_DPUCycles{
                (latency_values.cmxLatency_CMXClockCycles + latency_values.postTimeLatency_CMXClockCycles) *
                (int)hw_characteristics.dpu_freq_clk / (int)hw_characteristics.cmx_freq_clk};

        return (location == MemoryLocation::DRAM) ? dram_DPUCycles : cmx_DPUCycles;
    }

    constexpr std::tuple<float, float> getDMATRansferEfficiency(const MemoryLocation s, const MemoryLocation d) const {
        // if (s == MemoryLocation::CMX && d == MemoryLocation::CMX)// 100%
        const float mixedMemoryEfficiency{latency_values.mixedMemoryEfficiency};

        if (s == MemoryLocation::DRAM && d == MemoryLocation::CMX) {
            return std::tuple(1.0f, mixedMemoryEfficiency);
        }
        if (s == MemoryLocation::CMX && d == MemoryLocation::DRAM) {
            return std::tuple(mixedMemoryEfficiency, 1.0f);
        }

        return std::tuple(1.0f, 1.0f);  // input-output efficiency in percentage [0..1]
    };
};

}  // namespace VPUNN

#endif  //
