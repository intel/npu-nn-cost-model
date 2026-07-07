// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_DMA_THEORETICAL_COST_PROVIDER_H
#define VPUNN_DMA_THEORETICAL_COST_PROVIDER_H

#include <algorithm>                      // for std::min, std::max
#include <cmath>                          // for std::floor
#include <tuple>                          // for std::tuple and structured bindings
#include "dma_cost_provider_interface.h"  // for IDMACostProvider
#include "performance_mode.h"
#include "vpu/cycles_interface_types.h"                           // for CyclesInterfaceType, Cycles
#include "vpu/dma_workload.h"                                     // for DMAWorkloadTransformer, DMANNWorkload_NPU40_50
#include "vpu/dpu_types_info.h"                                   // for dtype_to_bytes
#include "vpu/hw_characteristics/HW_characteristics_supersets.h"  // for HWCharacteristicsSet
#include "vpu/hw_characteristics/itf_HW_characteristics_set.h"    // for IHWCharacteristicsSet
#include "vpu/types.h"

// #include "vpu/dma_cost_providers/dma_strided_math_cost_provider.h"  // for experimenting
// #include "vpu/dma_descriptor_transformers.h"  // for create_workload<VPUDMADescriptor, DMAWorkload>

namespace VPUNN {
/**
 * @brief Provides theoretical cost for DMA workloads (developed the past for LNL devices or older) , in the Legacy mode
 * of computing DMA
 *
 * This class estimates the number of execution cycles required for DMA operations
 * using legacy calculation methods, considering device type, memory locations, data layout permutations,
 * compression, and hardware bandwidth limitations. It is primarily used in class DMATheoreticalCostProvider for VPU40
 * devices or prior to VPU40 or when legacy mode is on
 */
class DMATheoreticalCostProvider_LNL_Legacy {
    const IHWCharacteristicsSet& hw_info{HWCharacteristicsSuperSets::legacyConfiguration()};

public:
    // Bandwidth cycles per bytes, LEgacy approach. Public because it has some tests
    float get_bandwidth_cycles_per_bytesLegacy(const IDeviceHWCharacteristics& hw, const VPUTensor& tensor,
                                               MemoryLocation location, bool compression = false, bool permute = false,
                                               bool half_duplex = false) const {
        switch (location) {
        case MemoryLocation::DRAM:
            return hw.get_dpu_fclk() / hw.get_dram_bandwidth_MBps();
        default:
            return static_cast<float>(hw.get_dpu_fclk()) / static_cast<float>(hw.get_cmx_fclk()) /
                   static_cast<float>(get_sram_word_sizeLegacy(tensor, compression, permute, half_duplex));
        }
    }

protected:
    // not virtual, specific implementation

    // SRAM word size legacy
    unsigned int get_sram_word_sizeLegacy(bool compression, bool half_duplex) const {
        const int full_duplex{compression ? 64 : 32};
        return half_duplex ? full_duplex / 2 : full_duplex;
    }

    unsigned int get_sram_word_sizeLegacy(const VPUTensor& tensor, bool compression, bool permute,
                                          bool half_duplex = false) const {
        if (!permute) {
            return std::min(tensor.size(), get_sram_word_sizeLegacy(compression, half_duplex));
        } else {
            auto number_of_bytes = dtype_to_bytes(tensor.get_dtype());
            return ((number_of_bytes <= 0) ? 1 : number_of_bytes);
        }
    }

public:
    /**
     * @brief Compute the DMA theoretical cycles => DMATheoreticalCyclesLegacyLNL
     *
     * @param wl a DMAWorkload
     * @return unsigned long int theoretical execution DPU cycles
     * @deprecated Will be removed in future releases
     */
    unsigned long int DMATheoreticalCyclesLegacyLNL(const DMAWorkload& wl) const {
        // CMX2CMX is half-duplex on NPU 2.x
        const bool is_half_duplex_limitation{((wl.device <= VPUDevice::VPU_2_7)  // specific device
                                              && (wl.input_location == MemoryLocation::CMX) &&
                                              (wl.output_location == MemoryLocation::CMX))
                                                     ? true
                                                     : false};

        // Get if the input is permuted or compressed
        const bool is_input_permuted = ((wl.input.get_layout() != wl.output.get_layout())  // changing layout
                                        && (wl.input_location == MemoryLocation::CMX));    // and src from CMX

        const bool is_input_compressed = ((wl.input.size() != wl.output.size())            // changing size
                                          && (wl.input_location == MemoryLocation::CMX));  // and src from CMX

        // Get the bandwidth in DPU cycles/bytes
        const float input_bandwidth = get_bandwidth_cycles_per_bytesLegacy(
                hw_info.device(wl.device),  //
                wl.input, wl.input_location, is_input_compressed, is_input_permuted, is_half_duplex_limitation);
        // Compute input cycles from dimensions and bw
        const auto input_cycles = Cycles::toCycleInterfaceType((double)wl.input.size() * (double)input_bandwidth);

        // Get if the output is permuted or compressed
        const bool is_output_permuted = ((wl.input.get_layout() != wl.output.get_layout())  // changing layout
                                         && (wl.output_location == MemoryLocation::CMX));   // and dst to CMX

        const bool is_output_compressed = ((wl.input.size() != wl.output.size())             // changing size
                                           && (wl.output_location == MemoryLocation::CMX));  // and dst to CMX

        // Get the bandwidth in DPU cycles/bytes
        const float output_bandwidth = get_bandwidth_cycles_per_bytesLegacy(
                hw_info.device(wl.device),  //
                wl.output, wl.output_location, is_output_compressed, is_output_permuted, is_half_duplex_limitation);
        // Compute output cycles from dimensions and bw
        const auto output_cycles = Cycles::toCycleInterfaceType((double)wl.output.size() * (double)output_bandwidth);

        // Get latency in cycles
        const auto input_latency = (unsigned long)hw_info.device(wl.device).get_DMA_latency(wl.input_location);
        const auto output_latency = (unsigned long)hw_info.device(wl.device).get_DMA_latency(wl.output_location);

        // Get the max between input and output cycles
        return Cycles::cost_adder(std::max(input_latency, output_latency), std::max(input_cycles, output_cycles));
    }
};

/**
 * @brief Provides updated theoretical performance modeling for DMA workloads on PTL or newer devices.
 *
 * This class estimates the number of cycles required for DMA operations using the latest calculation
 * methods, supporting advanced features such as decompression, permutation, and bandwidth aggregation.
 *
 * It is intended for use with PTL and newer devices in class DMATheoreticalCostProvider.
 *
 * TODO: If PTL have multiple theoretical implementations, clearly they should be in different classes with the same
 * interface. For now we do not know yet the final design, maybe this is just an intermediate step.
 *
 */
class DMATheoreticalCostProvider_PTL {
    /// hardware characteristics are configured at constructor
    const IHWCharacteristicsSet& hw_info;

public:
    DMATheoreticalCostProvider_PTL(const IHWCharacteristicsSet& hw_info_set): hw_info(hw_info_set) {
    }

protected:
    const IDeviceHWCharacteristics& get_hw_characteristics(const VPUDevice device) const {
        return hw_info.device(device);
    }

    int compute_DRAM_bandwith_BytesPerCyc(const VPUDevice& device) const {
        const auto& hw{hw_info.device(device)};  // device characteristics
        // DRAM bw is given in MBps
        const int ddr_BytesPerCyc{static_cast<int>(std::floor(hw.get_dram_bandwidth_MBps() / hw.get_cmx_fclk()))};
        const int cmx_bounded_maxBytesPerCyc{hw.get_DMA_DDR_interface_bytes()};
        return std::min(ddr_BytesPerCyc, cmx_bounded_maxBytesPerCyc);
    }

    int cmx_raw_word_size(const VPUDevice device) const {
        const auto& hw{hw_info.device(device)};  // device characteristics
        return hw.get_DMA_DDR_interface_bytes();
    }

    // CMX clock used. Considers also limitation like compression...
    int cmx_agregated_bytes_per_cycle_bw(const VPUTensor& tensor, VPUDevice device, bool permute, bool compression,
                                         float decompression_ratio = 1.0F, int compressed_BW_BytesPerCycle = 1) const {
        // permute limits the bw to one element per cycle
        if (permute) {
            return dtype_to_bytes(tensor.get_dtype());
        }

        const auto nominal_bw = cmx_raw_word_size(device);  // nominal
        if (compression) {
            const auto max_bw = (float)nominal_bw * 2.0f;  // bpclock
            const auto potential_speed_up_bw = (float)compressed_BW_BytesPerCycle * decompression_ratio;
            // compression speeds up the bpCyc to compressed_BW_BytesPerCycle*decompression_ratio(>1) but not more than
            // 2x of SRAM speed
            const float real_speed_up_bw = std::min(max_bw, potential_speed_up_bw);  //

            return (int)real_speed_up_bw;
        }

        // normal speed is the constant CMX bytes per cycle
        return nominal_bw;
    }

    // cmx clock
    int get_bytes_per_cycle_read_bw(const VPUTensor& tensor, VPUDevice device, MemoryLocation location) const {
        switch (location) {
        case MemoryLocation::DRAM:
            return compute_DRAM_bandwith_BytesPerCyc(device);

        case MemoryLocation::CMX:
        case MemoryLocation::CSRAM:  //?
        case MemoryLocation::UPA:    //?
        default:
            return cmx_agregated_bytes_per_cycle_bw(tensor, device, false, false);
        }
    }

    int get_bytes_per_cycle_write_bw(const VPUTensor& tensor, VPUDevice device, MemoryLocation location, bool permute,
                                     bool compression, float decompression_ratio /*= 1.0F*/,
                                     int compressed_BW_BytesPerCycle /* = 0*/) const {
        switch (location) {
        case MemoryLocation::DRAM:
            // not influenced by permuted!
            // not influenced by compression!
            return compute_DRAM_bandwith_BytesPerCyc(device);

        case MemoryLocation::CMX:
        case MemoryLocation::CSRAM:  //?
        case MemoryLocation::UPA:    //?
        default:
            return (cmx_agregated_bytes_per_cycle_bw(tensor, device, permute, compression, decompression_ratio,
                                                     compressed_BW_BytesPerCycle));
        }
    }

public:
    /**
     * @brief Estimates the theoretical DMA execution cycles for PTL or newer devices.
     *
     * This method calculates the number of execution cycles required for a DMA operation
     * using the updated PTL theoretical model.  The cost is:
     *
     *   cycles = latency + max(input_cycles, output_cycles)
     *
     * where latency, bandwidth, and clock ratios are all device-specific.
     *
     * ---
     * **CMX→CMX layout permutation**
     *
     * A DMA between two CMX tensors whose layouts differ (e.g. NHWC→NCHW) is a
     * *permutation*: the DMA engine must scatter/gather individual elements rather
     * than copying a contiguous byte stream.
     *
     * This is detected when **all three** conditions hold:
     *   - `wl.input.get_layout() != wl.output.get_layout()`
     *   - `wl.input_location  == MemoryLocation::CMX`
     *   - `wl.output_location == MemoryLocation::CMX`
     *
     * When triggered, the write-side bandwidth drops to `dtype_to_bytes(dtype)` per
     * cycle (one element per cycle) instead of the nominal CMX word width (~32 B/cycle).
     * This can be up to 32× slower than a plain CMX copy.
     *
     * Layout differences on any other transfer direction (e.g. DDR→CMX) do **not**
     * trigger the permutation penalty — the flag remains false.
     *
     * ---
     * **DDR→CMX hardware decompression**
     *
     * The NPU DMA engine supports in-flight decompression of compressed activation
     * tensors stored in DRAM.  In this mode the source tensor (compressed, in DDR) is
     * smaller than the destination tensor (expanded, in CMX):
     *   `wl.input.size() < wl.output.size()`
     *
     * This is detected when **all three** conditions hold:
     *   - `wl.input.size()    < wl.output.size()`   (compressed src, expanded dst)
     *   - `wl.input_location  == MemoryLocation::DRAM`
     *   - `wl.output_location == MemoryLocation::CMX`
     *
     * When triggered a `decompression_ratio = output_bytes / input_bytes` is computed
     * (ratio > 1).  The decompressor can write to CMX faster than data arrives from
     * DRAM, so the *effective* CMX write bandwidth is scaled up by this ratio, capped
     * at 2× the nominal SRAM bandwidth:
     *   `effective_write_bw = min(raw_bw * decompression_ratio, 2 * nominal_cmx_bw)`
     *
     * For ordinary (non-compressed) transfers `input.size() == output.size()`, the
     * ratio is 1.0 and the decompression path is a no-op.
     *
     * ---
     * @param wl The DMAWorkload describing the DMA operation.
     * @return   The estimated number of DPU cycles required to complete the DMA operation.
     */
    unsigned long int DMATheoreticalCyclesPTL_ON(const DMAWorkload& wl) const {
        // device is presumed to be at least LNL
        const auto& hw{hw_info.device(wl.device)};  // device characteristics

        const float dpuPerCmx_clock_ratio{(float)hw.get_dpu_fclk() / (float)hw.get_cmx_fclk()};

        const bool is_cmx2cmx_permutation = ((wl.input.get_layout() != wl.output.get_layout())  // changing layout
                                             && (wl.input_location == MemoryLocation::CMX)      // and src/dest from CMX
                                             && (wl.output_location == MemoryLocation::CMX));   // and src/dest from CMX

        const bool is_DDR2CMX_decompresion =
                ((wl.input.size() < wl.output.size())              // dest size is bigger (decompression)
                 && (wl.input_location == MemoryLocation::DRAM)    // and src from DDR
                 && (wl.output_location == MemoryLocation::CMX));  // and dst to CMX

        const float decompression_ratio{is_DDR2CMX_decompresion ? ((float)wl.output.size() / (float)wl.input.size())
                                                                : 1.0f};

        const auto [input_bw_efficiency, output_bw_efficiency] =
                hw.getDMATRansferEfficiency(wl.input_location, wl.output_location);

        const int input_bw_bpc = get_bytes_per_cycle_read_bw(wl.input, wl.device, wl.input_location);
        const auto CMX_cycles_read = (float)wl.input.size() / ((float)input_bw_bpc * input_bw_efficiency);
        const auto input_cycles_DPU = Cycles::toCycleInterfaceType(CMX_cycles_read * dpuPerCmx_clock_ratio);

        const int output_bw_bpc =
                get_bytes_per_cycle_write_bw(wl.output, wl.device, wl.output_location, is_cmx2cmx_permutation,
                                             is_DDR2CMX_decompresion, decompression_ratio, input_bw_bpc);
        const auto CMX_cycles_write = (float)wl.output.size() / ((float)output_bw_bpc * output_bw_efficiency);
        const auto output_cycles_DPU = Cycles::toCycleInterfaceType(CMX_cycles_write * dpuPerCmx_clock_ratio);

        // Get latency in cycles
        const auto input_latency_DPU = hw.get_DMA_latency(wl.input_location);
        const auto output_latency_DPU = hw.get_DMA_latency(wl.output_location);

        // Get the max between input and output cycles
        return Cycles::cost_adder(std::max(input_latency_DPU, output_latency_DPU),
                                  std::max(input_cycles_DPU, output_cycles_DPU));
    }
};

/**
 * @class BaseDMATheoreticalCostProvider
 * @brief Provides theoretical cycles for DMA workloads.
 *
 * This class estimates the number of execution cycles required for DMA operations on VPU hardware.
 * It selects the appropriate theoretical model based on device type and configuration, supporting both
 * legacy and updated calculation methods.
 *
 * An instance of this class is intended to be use as a provider for theoretical cost for DMA workloads
 * An example of usage can be seen in class VPUCostModel where we either need just theoretical cost or
 * we use this as a fallback when NN cost not available
 */
template <typename WlT = DMAWorkload>
class BaseDMATheoreticalCostProvider : public IDMACostProvider<WlT> {
protected:
    DMATheoreticalCostProvider_LNL_Legacy dma_theoretical_LNL;  // legacy one
    DMATheoreticalCostProvider_PTL dma_theoretical_PTL{
            HWCharacteristicsSuperSets::get_mainConfigurationRef()};  // evoluated one, default config
    // DMAStridedMathCostProvider_DescIntf dma_strided_math;             // strided Math DMA

public:
    CyclesInterfaceType get_cost(const WlT& wl, std::string* cost_source = nullptr) const override {
        if (cost_source) {
            *cost_source = "theoretical";
        }

        if (wl.device < VPUDevice::VPU_4_0) {
            return dma_theoretical_LNL.DMATheoreticalCyclesLegacyLNL(wl);
        } else {  // VPU 4.0 and newer
            if (wl.device == VPUDevice::VPU_4_0 && PerformanceMode::forceLegacy_G4) {
                return dma_theoretical_LNL.DMATheoreticalCyclesLegacyLNL(wl);
            } else if (wl.device > VPUDevice::VPU_4_0 && PerformanceMode::forceLegacy_G5) {
                return dma_theoretical_LNL.DMATheoreticalCyclesLegacyLNL(wl);
            }
            return dma_theoretical_PTL.DMATheoreticalCyclesPTL_ON(wl);  // Updated theoretical model

            // experiment: use strided math model for NPU_RESERVED
            // if (wl.device < VPUDevice::NPU_RESERVED) {
            //    return dma_theoretical_PTL.DMATheoreticalCyclesPTL_ON(wl);  // classical theoretical model for
            //    // PTL / NPU_RESERVED
            //} else {
            //    // convert from WlT to DMANNWorkload_NPU40_50, the strided math model is only for this workload type,
            //    // so we need to do the conversion
            //    // convert from WlT to the workload type expected by dma_strided_math, generically deduced
            //    using StridedWlT = typename decltype(dma_strided_math)::workload_type;
            //    StridedWlT wl_nn{DMAWorkloadTransformer::template create_workload<StridedWlT>(wl)};
            //    return dma_strided_math.get_cost(wl_nn, cost_source);
            //}
        }
    }
};

// name alias used because there is only one workload type that theoretical dma cost provider supports
// and also for backward compatibility
using DMATheoreticalCostProvider = BaseDMATheoreticalCostProvider<>;

}  // namespace VPUNN

#endif
