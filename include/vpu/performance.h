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
#include "specific_device_HW_characteristics.h"
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

inline constexpr unsigned int get_dpu_fclk(VPUDevice device) {
    Characteristics config = get_HWCharacteristics(device);
    return std::visit(
            [](auto& obj) {
                return obj.get_dpu_freq_clk();
            },
            config);
}

/**
 * @brief Get the CMX default frequency in MHz
 *
 * @param device a VPUDevice
 * @return unsigned int
 */

inline constexpr unsigned int get_cmx_fclk(VPUDevice device) {
    Characteristics config = get_HWCharacteristics(device);
    return std::visit(
            [](auto& obj) {
                return obj.get_cmx_freq_clk();
            },
            config);
}

inline constexpr unsigned int get_cmx_fclk_Legacy(VPUDevice device) {
    Characteristics config = get_HWCharacteristics_Legacy(device);
    return std::visit(
            [](auto& obj) {
                return obj.get_cmx_freq_clk();
            },
            config);
}

/**
 * @brief Get CMX word size in bytes
 *
 * NOtice: used only in DPU Theoretical, might be obsolete.
 * @param device a VPUDevice
 * @return unsigned int
 */
inline constexpr unsigned int get_cmx_word_size_bytes(VPUDevice device) {
    Characteristics config = get_HWCharacteristics(device);
    return std::visit(
            [](auto& obj) {
                return obj.get_cmx_word_size_B();
            },
            config);
}

///// normal, non decompressed upscaled , bandwidth
// inline constexpr unsigned int get_cmx_normal_BW_bytes_per_cycle(VPUDevice device) {
//     switch (device) {
//     case VPUDevice::VPU_2_0:
//     case VPUDevice::VPU_2_1:
//     case VPUDevice::VPU_2_7:
//         return 32;
//     case VPUDevice::VPU_4_0:
//         return 64;
//     case VPUDevice::NPU_RESERVED:
//     case VPUDevice::NPU_RESERVED_W:
//         return 64;  // no change
//     default:
//         return 0;  // NA
//     }
// }

/**
 * @brief Get DMA engine bytes per cycle
 *
 * @param device a VPUDevice
 * @return unsigned int
 */
inline constexpr int get_DMA_DDR_interface_bytes(VPUDevice device) {
    Characteristics config = get_HWCharacteristics(device);
    return std::visit(
            [](auto& obj) {
                return obj.get_DMA_engine_B();
            },
            config);
}

/**
 * @brief Get DPU number of CMX read ports
 *
 * @param device a VPUDevice
 * @return unsigned int
 */

inline constexpr unsigned int get_dpu_cmx_num_read_ports(VPUDevice device) {
    Characteristics config = get_HWCharacteristics(device);
    return std::visit(
            [](auto& obj) {
                return obj.get_dpu_cmx_num_read_ports();
            },
            config);
}

/**
 * @brief Get the DRAM bandwidth in MB/s for a specific VPU IP
 * Actual speed is limited also by the CMX clock  bandwidth
 * @param device a VPUDevice
 * @return float
 */
inline constexpr float get_dram_bandwidth_MBps_Legacy(VPUDevice device) {
    Characteristics config = get_HWCharacteristics_Legacy(device);
    return std::visit(
            [](auto& obj) {
                return obj.get_dram_bandwidth_MBps();
            },
            config);
}
/**
 * @brief Get the DRAM bandwidth in MB/s for a specific VPU IP
 * Actual speed is limited also by the CMX clock  bandwidth
 * @param device a VPUDevice
 * @return float
 */
inline constexpr float get_dram_bandwidth_MBps(VPUDevice device) {
    Characteristics config = get_HWCharacteristics(device);
    return std::visit(
            [](auto& obj) {
                return obj.get_dram_bandwidth_MBps();
            },
            config);
}

/**
 * @brief Get the sram word size
 *
 * @param compression if compression is enabled or not
 * @return unsigned int
 */
inline constexpr unsigned int get_sram_word_sizeLegacy(bool compression, bool half_duplex) {
    const int full_duplex{compression ? 64 : 32};
    return half_duplex ? full_duplex / 2 : full_duplex;
}

/**
 * @brief Get the sram word size
 * Bytes per one cycle
 *
 * @param tensor a VPUTensor
 * @param compression if compression is enabled or not
 * @param permute if a permute operation is required
 * @return  unsigned int
 */
inline unsigned int get_sram_word_sizeLegacy(const VPUTensor& tensor, bool compression, bool permute,
                                             bool half_duplex = false) {
    if (!permute) {
        // same layout -> linear DMA so DST width is equal to tensor size
        return std::min(tensor.size(), get_sram_word_sizeLegacy(compression, half_duplex));
    } else {
        auto number_of_bytes = dtype_to_bytes(tensor.get_dtype());
        // if valid dtype we return number of bytes else return 1, because sram should be at least 1
        return ((number_of_bytes <= 0) ? 1 : number_of_bytes);
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

inline float get_bandwidth_cycles_per_bytesLegacy(const VPUTensor& tensor, VPUDevice device, MemoryLocation location,
                                                  bool compression = false, bool permute = false,
                                                  bool half_duplex = false) {
    switch (location) {
    case MemoryLocation::DRAM:
        // DRAM bw is given in MBps
        return get_dpu_fclk(device) / get_dram_bandwidth_MBps_Legacy(device);
    default:
        // SRAM bw is twice in compression mode
        return (float)get_dpu_fclk(device) / (float)get_cmx_fclk_Legacy(device) /
               ((float)get_sram_word_sizeLegacy(tensor, compression, permute, half_duplex));
    }
}

///**
// * @brief Get the DMA bandwidth in MB/s for a specific VPU IP
// *
// * @param tensor a VPUTensor
// * @param device a VPUDevice
// * @param location a memory location
// * @param compression is compression enabled
// * @param permute if a permute operation is required
// * @return float
// */
// inline float get_bandwidth_MBps(const VPUTensor& tensor, VPUDevice device, MemoryLocation location,
//                                bool compression = false, bool permute = false, bool half_duplex = false) {
//    auto bw_cycles_per_bytes =
//            get_bandwidth_cycles_per_bytes(tensor, device, location, compression, permute, half_duplex);
//    auto fclk = get_dpu_fclk(device);
//    return fclk / bw_cycles_per_bytes;
//}

// these functions are no longer used in get_DMA_latency_Legacy, because their content was redirected to the new design
// in device HWCharacteristics specific classes
//
// inline constexpr CyclesInterfaceType get_DMA_latency_NPU27(MemoryLocation location)
//
// inline constexpr CyclesInterfaceType get_DMA_latency_NPU40_Legacy(MemoryLocation location)
//
// inline constexpr CyclesInterfaceType get_DMA_latency_NPU_RESERVED_Legacy(MemoryLocation location)

/**
 * @brief Get the DMA latency in DPU cycles
 *
 * @param device a VPUDevice
 * @param location what memory is used
 * @return CyclesInterfaceType
 */
inline constexpr CyclesInterfaceType get_DMA_latency_Legacy(VPUDevice device, MemoryLocation location) {
    Characteristics config = get_HWCharacteristics_Legacy(device);
    return std::visit(
            [location](auto& obj) {
                return obj.get_DMA_latency(location);
            },
            config);
}

// these functions are no longer used in get_DMA_latency, because their content was redirected to the new design
// in device HWCharacteristics specific classes
//
// inline constexpr CyclesInterfaceType get_DMA_latency_NPU40(MemoryLocation location)
//
// inline constexpr CyclesInterfaceType get_DMA_latency_NPU_RESERVED(MemoryLocation location)
//

inline constexpr CyclesInterfaceType get_DMA_latency(VPUDevice device, MemoryLocation location) {
    Characteristics config = get_HWCharacteristics(device);
    return std::visit(
            [location](auto& obj) {
                return obj.get_DMA_latency(location);
            },
            config);
}

/**
 * @brief Get the DPU number of MACs
 *
 * @param device a VPUDevice
 * @return unsigned int
 */
inline constexpr unsigned int get_nr_macs(VPUDevice device) {
    Characteristics config = get_HWCharacteristics(device);
    return std::visit(
            [](auto& obj) {
                return obj.get_nr_macs();
            },
            config);
}

/**
 * @brief Get the ratio of int compute to fp16 compute
 *
 * @param device a VPUDevice
 * @return
 */
inline constexpr unsigned int get_fp_ratio(VPUDevice device) {
    Characteristics config = get_HWCharacteristics(device);
    return std::visit(
            [](auto& obj) {
                return obj.get_fp_ratio();
            },
            config);
}

/**
 * @brief Get the number of PPE
 *
 * @param device a VPUDevice
 * @return unsigned int
 */
inline constexpr unsigned int get_nr_ppe(VPUDevice device) {
    Characteristics config = get_HWCharacteristics(device);
    return std::visit(
            [](auto& obj) {
                return obj.get_nr_ppe();
            },
            config);
}

/**
 * @brief Determine whether native computation for workload is floating point or int
 *
 * @param DPUWorkload a DPUWorkload
 * @return bool
 */
inline bool native_comp_is_any_fp(const DPUWorkload& wl) {
    // If either activations or weights are FP16/BF16 then native computation is FP16/BF16
    bool found_at_least_one_float = false;
    for (const auto& i : wl.inputs) {
        found_at_least_one_float = found_at_least_one_float || i.is_any_float();
    }
    return found_at_least_one_float;
}

inline bool native_comp_on_fp16(const DPUWorkload& wl) {
    // If either activations or weights are FP16/BF16 then native computation is FP16/BF16
    static_assert(std::tuple_size<decltype(wl.inputs)>{} == 1, "only one input");

    return wl.inputs[0].is_fp16family();
}

inline bool native_comp_on_fp8(const DPUWorkload& wl) {
    // to do : look at weights also?
    static_assert(std::tuple_size<decltype(wl.inputs)>{} == 1, "only one input");
    // to do  redesign xx family methods to be based on Datatype operations
    const VPUTensor wts({1, 1, 1, 1}, wl.get_weight_type());
    return wl.inputs[0].is_fp8family() && (!wts.is_fp16family());
}

inline bool native_comp_on_i8(const DPUWorkload& wl) {
    static_assert(std::tuple_size<decltype(wl.inputs)>{} == 1, "only one input");

    // to do  redesign xx family methods to be based on Datatype operations
    const VPUTensor wts({1, 1, 1, 1}, wl.get_weight_type());

    return wl.inputs[0].is_i8family() && (!wts.is_any_float());
}

/**
 * @brief Get the MAC/input channels/cycles for a specific VPU IP
 *
 * @param device a VPUDevice
 * @return unsigned int
 */
inline constexpr unsigned int input_channels_mac(VPUDevice device) {
    Characteristics config = get_HWCharacteristics(device);
    return std::visit(
            [](auto& obj) {
                return obj.get_input_channels_mac();
            },
            config);
}

/**
 * @brief Get the MAC/input channels/cycles for a specific VPU IP
 *
 * @param device a VPUDevice
 * @return unsigned int
 */
inline constexpr unsigned int nDPU_per_tile(VPUDevice device) {
    Characteristics config = get_HWCharacteristics(device);
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
inline constexpr int get_dma_ports(VPUDevice device) {
    Characteristics config = get_HWCharacteristics(device);
    return std::visit(
            [](auto& obj) {
                return obj.get_dma_ports();
            },
            config);
}

/// provides the Profiler clock in MHz , depending on device
/// \param device for which the information is requested
/// \returns the frequency or zero in case the device is not known
inline constexpr float get_profiling_clk_MHz(VPUDevice device) {
    Characteristics config = get_HWCharacteristics(device);
    return std::visit(
            [](auto& obj) {
                return obj.get_profiling_clk_MHz();
            },
            config);
}

/// provides the Profiler clock in Hz , depending on device
/// \param device for which the information is requested
/// \returns the frequency or zero in case the device is not known
inline constexpr int get_profiling_clk_Hz(VPUDevice device) {
    Characteristics config = get_HWCharacteristics(device);
    return std::visit(
            [](auto& obj) {
                return obj.get_profiling_clk_Hz();
            },
            config);
}

}  // namespace VPUNN

#endif  // VPUNN_PERFORMANCE_H
