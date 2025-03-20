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

#include "vpu/cycles_interface_types.h"

#include "vpu/dma_types.h"
#include "vpu/types.h"
#include "vpu/utils.h"

namespace VPUNN {

/**
 * @brief Get the DPU default frequency in MHz
 *
 * @param device a VPUDevice
 * @return unsigned int
 */
inline constexpr unsigned int get_dpu_fclk(const VPUDevice device) {
    switch (device) {
    case VPUDevice::VPU_2_0:
        return 700;
    case VPUDevice::VPU_2_1:
        return 850;
    case VPUDevice::VPU_2_7:
        return 1300;
    case VPUDevice::VPU_4_0:
        return 1700;
    case VPUDevice::NPU_RESERVED:
    case VPUDevice::NPU_RESERVED_W:
        return 1950;  // at 0.78V KPI point
    default:
        return 700;
    }
}

/**
 * @brief Get the CMX default frequency in MHz
 *
 * @param device a VPUDevice
 * @return unsigned int
 */
inline constexpr unsigned int get_cmx_fclk(const VPUDevice device) {
    switch (device) {
    case VPUDevice::VPU_2_0:
        return 700;
    case VPUDevice::VPU_2_1:
        return 850;
    case VPUDevice::VPU_2_7:
        return 975;
    case VPUDevice::VPU_4_0:
        return 975;
    case VPUDevice::NPU_RESERVED:
    case VPUDevice::NPU_RESERVED_W:
        return 1114;  // at KPI point
    default:
        return 700;
    }
}

/**
 * @brief Get CMX word size in bytes
 *
 * @param device a VPUDevice
 * @return unsigned int
 */
inline constexpr unsigned int get_cmx_word_size_bytes(VPUDevice device) {
    switch (device) {
    case VPUDevice::VPU_2_0:
    case VPUDevice::VPU_2_1:
    case VPUDevice::VPU_2_7:
        return 16;
    case VPUDevice::VPU_4_0:
        return 32;
    case VPUDevice::NPU_RESERVED:
    case VPUDevice::NPU_RESERVED_W:
        return 32;  // no change
    default:
        return 16;
    }
}

/**
 * @brief Get DMA engine bytes per cycle
 *
 * @param device a VPUDevice
 * @return unsigned int
 */
inline constexpr unsigned int get_DMA_DDR_interface_bytes(VPUDevice device) {
    switch (device) {
    case VPUDevice::VPU_2_0:
    case VPUDevice::VPU_2_1:
    case VPUDevice::VPU_2_7:
        return 32;
    case VPUDevice::VPU_4_0:
        return 64;  // 512 bits AXI
    case VPUDevice::NPU_RESERVED:
    case VPUDevice::NPU_RESERVED_W:
        return 64;  // no change
    default:
        return 0;
    }
}

/**
 * @brief Get DPU number of CMX read ports
 *
 * @param device a VPUDevice
 * @return unsigned int
 */
inline constexpr unsigned int get_dpu_cmx_num_read_ports(VPUDevice device) {
    switch (device) {
    case VPUDevice::VPU_2_0:
    case VPUDevice::VPU_2_1:
        return 4;  // RO
    case VPUDevice::VPU_2_7:
        return 8;  // RO
    case VPUDevice::VPU_4_0:
    case VPUDevice::NPU_RESERVED:
    case VPUDevice::NPU_RESERVED_W:
        return 8;  // 4x RO, 4x RW
    default:
        return 8;
    }
}

/**
 * @brief Get the DRAM bandwidth in MB/s for a specific VPU IP
 *
 * @param device a VPUDevice
 * @return float
 */
inline constexpr float get_dram_bandwidth_MBps(VPUDevice device) {
    switch (device) {
    case VPUDevice::VPU_2_0:
    case VPUDevice::VPU_2_1:
        return 20000.0f;
    case VPUDevice::VPU_2_7:
        return 27000.0f;
    case VPUDevice::VPU_4_0:
        return 45000.0f;
    case VPUDevice::NPU_RESERVED:
    case VPUDevice::NPU_RESERVED_W:
        return 45000.0f;  //?
    default:
        return 1;
    }
}

/**
 * @brief Get the sram word size
 *
 * @param compression if compression is enabled or not
 * @return unsigned int
 */
inline constexpr unsigned int get_sram_word_size(bool compression, bool half_duplex) {
    const int full_duplex{compression ? 64 : 32};
    return half_duplex ? full_duplex / 2 : full_duplex;
}

/**
 * @brief Get the sram word size
 *
 * @param tensor a VPUTensor
 * @param compression if compression is enabled or not
 * @param permute if a permute operation is required
 * @return  unsigned int
 */
inline unsigned int get_sram_word_size(const VPUTensor& tensor, bool compression, bool permute,
                                       bool half_duplex = false) {
    if (!permute) {
        // same layout -> linear DMA so DST width is equal to tensor size
        return std::min(tensor.size(), get_sram_word_size(compression, half_duplex));
    } else {
        return dtype_to_bytes(tensor.get_dtype());
    }
}

/**
 * @brief Get the DMA bandwidth in DPU cycles/bytes for a specific VPU IP
 *
 * @param tensor a VPUTensor
 * @param device a VPUDevice
 * @param location a memory location
 * @param compression is compression enabled
 * @param permute if a permute operation is required
 * @return float
 */
inline float get_bandwidth_cycles_per_bytes(const VPUTensor& tensor, VPUDevice device, MemoryLocation location,
                                            bool compression = false, bool permute = false, bool half_duplex = false) {
    switch (location) {
    case MemoryLocation::DRAM:
        // DRAM bw is given in MBps
        return get_dpu_fclk(device) / get_dram_bandwidth_MBps(device);
    default:
        // SRAM bw is twice in compression mode
        /* coverity[divide_by_zero] */
        return (float)get_dpu_fclk(device) / (float)get_cmx_fclk(device) /
               ((float)get_sram_word_size(tensor, compression, permute, half_duplex));
    }
}

/**
 * @brief Get the DMA bandwidth in MB/s for a specific VPU IP
 *
 * @param tensor a VPUTensor
 * @param device a VPUDevice
 * @param location a memory location
 * @param compression is compression enabled
 * @param permute if a permute operation is required
 * @return float
 */
inline float get_bandwidth_MBps(const VPUTensor& tensor, VPUDevice device, MemoryLocation location,
                                bool compression = false, bool permute = false, bool half_duplex = false) {
    auto bw_cycles_per_bytes =
            get_bandwidth_cycles_per_bytes(tensor, device, location, compression, permute, half_duplex);
    auto fclk = get_dpu_fclk(device);
    return fclk / bw_cycles_per_bytes;
}

/**
 * @brief Get the DMA latency in DPU cycles
 *
 * @param device a VPUDevice
 * @param location what memory is used
 * @return CyclesInterfaceType
 */
inline constexpr CyclesInterfaceType get_DMA_latency(VPUDevice device, MemoryLocation location) {
    switch (device) {
    case VPUDevice::VPU_2_7: {
        constexpr VPUDevice const_device{VPUDevice::VPU_2_7};
        constexpr int dramLatency_Nanoseconds{956};   // nanoseconds
        constexpr int cmxLatency_CMXClockCycles{16};  // 16 clock cycles at VPU frequency

        //[ns]*[MHz] ->normalize to SI=> [ns/1 000 000 000]*[MHz*1 000 000]=[ns/1000]*[MHz]
        constexpr int dram_DPUCycles{(dramLatency_Nanoseconds * (int)get_dpu_fclk(const_device)) / 1000};
        // normally (cmx_clk/CMX_frq)*DPU_frq, but multiplication done first to be OK on int
        constexpr int cmx_DPUCycles{cmxLatency_CMXClockCycles * (int)get_dpu_fclk(const_device) /
                                    (int)get_cmx_fclk(const_device)};

        return (location == MemoryLocation::DRAM) ? dram_DPUCycles : cmx_DPUCycles;
    } break;
    case VPUDevice::VPU_4_0: {  //@todo: update for VPU4.0 actual latency (now a clone of 2.7)
        constexpr VPUDevice const_device{VPUDevice::VPU_4_0};
        constexpr int dramLatency_Nanoseconds{956};   // nanoseconds
        constexpr int cmxLatency_CMXClockCycles{16};  // 16 clock cycles at VPU frequency

        //[ns]*[MHz] ->normalize to SI=> [ns/1 000 000 000]*[MHz*1 000 000]=[ns/1000]*[MHz]
        constexpr int dram_DPUCycles{(dramLatency_Nanoseconds * (int)get_dpu_fclk(const_device)) / 1000};
        // normally (cmx_clk/CMX_frq)*DPU_frq, but multiplication done first to be OK on int
        constexpr int cmx_DPUCycles{cmxLatency_CMXClockCycles * (int)get_dpu_fclk(const_device) /
                                    (int)get_cmx_fclk(const_device)};

        return (location == MemoryLocation::DRAM) ? dram_DPUCycles : cmx_DPUCycles;
    } break;
    case VPUDevice::NPU_RESERVED:
    case VPUDevice::NPU_RESERVED_W: {  
        constexpr VPUDevice const_device{VPUDevice::NPU_RESERVED};
        constexpr int dramLatency_Nanoseconds{956};   // nanoseconds
        constexpr int cmxLatency_CMXClockCycles{16};  // 16 clock cycles at VPU frequency

        //[ns]*[MHz] ->normalize to SI=> [ns/1 000 000 000]*[MHz*1 000 000]=[ns/1000]*[MHz]
        constexpr int dram_DPUCycles{(dramLatency_Nanoseconds * (int)get_dpu_fclk(const_device)) / 1000};
        // normally (cmx_clk/CMX_frq)*DPU_frq, but multiplication done first to be OK on int
        constexpr int cmx_DPUCycles{cmxLatency_CMXClockCycles * (int)get_dpu_fclk(const_device) /
                                    (int)get_cmx_fclk(const_device)};

        return (location == MemoryLocation::DRAM) ? dram_DPUCycles : cmx_DPUCycles;
    } break;

    default:
        return 0;
    }
}

/**
 * @brief Get the DPU number of MACs
 *
 * @param device a VPUDevice
 * @return unsigned int
 */
inline constexpr unsigned int get_nr_macs(VPUDevice device) {
    switch (device) {
    case VPUDevice::VPU_2_0:
    case VPUDevice::VPU_2_1:
        return 256;
    case VPUDevice::VPU_2_7:
    case VPUDevice::VPU_4_0:
        return 2048;
    case VPUDevice::NPU_RESERVED:
    case VPUDevice::NPU_RESERVED_W:
        return 4096;
    default:
        return 2048;
    }
}

/**
 * @brief Get the ratio of int compute to fp16 compute
 *
 * @param device a VPUDevice
 * @return
 */
inline constexpr unsigned int get_fp_ratio(VPUDevice device) {
    switch (device) {
    case VPUDevice::VPU_2_0:
    case VPUDevice::VPU_2_1:
        return 4;
    default:
        return 2;
    }
}

/**
 * @brief Get the number of PPE
 *
 * @param device a VPUDevice
 * @return unsigned int
 */
inline constexpr unsigned int get_nr_ppe(VPUDevice device) {
    switch (device) {
    case VPUDevice::VPU_2_0:
    case VPUDevice::VPU_2_1:
        return 16;
    case VPUDevice::VPU_2_7:
    case VPUDevice::VPU_4_0:
        return 64;
    case VPUDevice::NPU_RESERVED:
    case VPUDevice::NPU_RESERVED_W:
        return 32;  // less ppe
    default:
        return 64;
    }
}

/**
 * @brief Determine whether native computation for workload is floating point or int
 *
 * @param DPUWorkload a DPUWorkload
 * @return bool
 */
inline bool native_comp_is_fp(const DPUWorkload& wl) {
    // If either activations or weights are FP16/BF16 then native computation is FP16/BF16
    bool found_at_least_one_float = false;
    for (const auto& i : wl.inputs) {
        found_at_least_one_float = found_at_least_one_float || i.is_float();
    }
    return found_at_least_one_float;
}

/**
 * @brief Get the MAC/input channels/cycles for a specific VPU IP
 *
 * @param device a VPUDevice
 * @return unsigned int
 */
inline constexpr unsigned int input_channels_mac(VPUDevice device) {
    switch (device) {
    case VPUDevice::VPU_2_0:
    case VPUDevice::VPU_2_1:
        /* code */
        return 1;
    default:
        return 8;
    }
}

/**
 * @brief Get the MAC/input channels/cycles for a specific VPU IP
 *
 * @param device a VPUDevice
 * @return unsigned int
 */
inline constexpr unsigned int nDPU_per_tile(VPUDevice device) {
    switch (device) {
    case VPUDevice::VPU_2_0:
    case VPUDevice::VPU_2_1:
        return 5;
    default:
        return 1;
    }
}

/**
 * @brief Get the channels/Ports of DMA.
 * Can be used to run one channel per separate tile (like for DDR to CM in case of weights for SOK),
 * cannot be used to run multiple channels when transferring to same tile
 * @param device a VPUDevice
 * @return int
 */
inline constexpr int get_dma_ports(VPUDevice device) {
    switch (device) {
    case VPUDevice::VPU_2_0:
    case VPUDevice::VPU_2_1:
        return 2;
    case VPUDevice::VPU_2_7:
        return 2;
    case VPUDevice::VPU_4_0:
        return 2;
    case VPUDevice::NPU_RESERVED:
    case VPUDevice::NPU_RESERVED_W:
        return 2;
    default:
        return 1;
    }
}

/// provides the Profiler clock in MHz , depending on device
/// \param device for which the information is requested
/// \returns the frequency or zero in case the device is not known
inline constexpr float get_profiling_clk_MHz(VPUDevice device) {
    constexpr float PROF_CLK_MHz{38.4f};

    switch (device) {
    case VPUDevice::VPU_2_0:
    case VPUDevice::VPU_2_1:
    case VPUDevice::VPU_2_7:
        return PROF_CLK_MHz;
    case VPUDevice::VPU_4_0:
    case VPUDevice::NPU_RESERVED:
    case VPUDevice::NPU_RESERVED_W:
        return PROF_CLK_MHz / 2.0f;
    default:
        return 0;
    }
}

/// provides the Profiler clock in Hz , depending on device
/// \param device for which the information is requested
/// \returns the frequency or zero in case the device is not known
inline constexpr int get_profiling_clk_Hz(VPUDevice device) {
    constexpr int PROF_CLK{38400000};  // Hz

    switch (device) {
    case VPUDevice::VPU_2_0:
    case VPUDevice::VPU_2_1:
    case VPUDevice::VPU_2_7:
        return PROF_CLK;
    case VPUDevice::VPU_4_0:
    case VPUDevice::NPU_RESERVED:
    case VPUDevice::NPU_RESERVED_W:
        return PROF_CLK / 2;
    default:
        return 0;
    }
}

}  // namespace VPUNN

#endif  // VPUNN_PERFORMANCE_H
