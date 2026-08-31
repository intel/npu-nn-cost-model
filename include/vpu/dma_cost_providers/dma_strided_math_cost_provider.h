// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_DMA_STRIDED_MATH_COST_PROVIDER_H
#define VPUNN_DMA_STRIDED_MATH_COST_PROVIDER_H

#include <algorithm>      // for std::max
#include <cassert>        // for assert
#include <cmath>          // for std::ceil
#include <map>            // for std::map (direction-aware model key)
#include <string>         // for std::string
#include <unordered_map>  // for std::unordered_map (stride penalty map)

// #include "dma_theoretical_cost_provider.h"  // for DMATheoreticalCostProvider_PTL, temporary draft
#include "performance_mode.h"
#include "vpu/dma_descriptors.h"
#include "vpu/types.h"

#include "dma_cost_provider_interface.h"                          // for IDMACostProvider
#include "vpu/hw_characteristics/HW_characteristics_supersets.h"  // for HWCharacteristicsSet
#include "vpu/hw_characteristics/itf_HW_characteristics_set.h"    // for IHWCharacteristicsSet

namespace VPUNN {

/**
 * @brief
 *
 */
class DMAStridedMathModel {
    /// hardware characteristics are configured at constructor
    const IHWCharacteristicsSet& hw_info;

public:
    DMAStridedMathModel(const IHWCharacteristicsSet& hw_info_set): hw_info(hw_info_set) {
    }

protected:
    const IDeviceHWCharacteristics& get_hw_characteristics(const VPUDevice device) const {
        return hw_info.device(device);
    }
    /**
     * @brief Inner class representing linear modeling function parameters (y = k*x + b)
     * where y is compute cycles, x is data size in bytes
     */
    struct LinearModelingParams {
        double k;  ///< slope coefficient
        double b;  ///< intercept

        LinearModelingParams(double k_val, double b_val): k(k_val), b(b_val) {
        }

        /// @brief Compute cycles using linear function y = k*x + b
        /// @param dma_size data size in bytes
        /// @return estimated cycles
        double compute_cycles(unsigned long long dma_size) const {
            return dma_size * k + b;
        }
    };

    /// Composite key for direction-aware linear model lookup: (device, transfer_direction)
    using DeviceDirectionKey = std::pair<VPUDevice, MemoryDirection>;
    using ThresholdPair = std::pair<int, int>;  // {src/read threshold, dst/write threshold}

    /**
     * @brief Per-direction modeling entry combining linear fit and src/dst thresholds.
     *
     * This keeps all parameters needed by a single (device, direction) LUT row together.
     */
    struct DirectionModelingEntry {
        const LinearModelingParams linear;  ///< Linear DMA model parameters (k, b) for this direction.
        const ThresholdPair thresholds;     ///< Src/dst thresholds {src/read, dst/write} in bytes for this direction.

        DirectionModelingEntry(const LinearModelingParams& linear_params, const ThresholdPair& threshold_pair)
                : linear(linear_params), thresholds(threshold_pair) {
        }
    };

    /// Modeling function map for DMA copy compute cycles (CMX cycles).
    /// Maps (VPUDevice, MemoryDirection) to per-direction linear parameters and src/dst thresholds.
    /// Typical src/dst threshold values are 64B for CMX and 256B for DDR.
    /// For each device there are 4 entries (DDR2CMX, CMX2DDR, CMX2CMX, DDR2DDR).
    static const std::map<DeviceDirectionKey, DirectionModelingEntry>& get_modeling_function_map() {
        static const std::map<DeviceDirectionKey, DirectionModelingEntry> modeling_function = {
                // NPU 5.0 — per-direction fits (R²=1.0000 for all)
                {{VPUDevice::NPU_5_0, MemoryDirection::DDR2CMX},
                 DirectionModelingEntry(LinearModelingParams(1.85798372e-02, 324.7775),
                                        {256, 64})},
                {{VPUDevice::NPU_5_0, MemoryDirection::CMX2DDR},
                 DirectionModelingEntry(LinearModelingParams(1.85884582e-02, 276.8316),
                                        {64, 256})},
                {{VPUDevice::NPU_5_0, MemoryDirection::CMX2CMX},
                 DirectionModelingEntry(LinearModelingParams(1.56250087e-02, 19.9909),
                                        {64, 64})},
                {{VPUDevice::NPU_5_0, MemoryDirection::DDR2DDR},
                 DirectionModelingEntry(LinearModelingParams(1.85798372e-02, 324.7775),
                                        {256, 256})
                 /*DirectionModelingEntry(LinearModelingParams(1.75977680e-02, 207.2000),
                                        {256, 256})*/},
        // reuse max one from above (D2C)
                // NPU 5.0W — same characteristics as NPU 5.0
                {{VPUDevice::NPU_5_0_W, MemoryDirection::DDR2CMX},
                 DirectionModelingEntry(LinearModelingParams(1.85798372e-02, 324.7775),
                                        {256, 64})},
                {{VPUDevice::NPU_5_0_W, MemoryDirection::CMX2DDR},
                 DirectionModelingEntry(LinearModelingParams(1.85884582e-02, 276.8316),
                                        {64, 256})},
                {{VPUDevice::NPU_5_0_W, MemoryDirection::CMX2CMX},
                 DirectionModelingEntry(LinearModelingParams(1.56250087e-02, 19.9909),
                                        {64, 64})},
                {{VPUDevice::NPU_5_0_W, MemoryDirection::DDR2DDR},
                 DirectionModelingEntry(LinearModelingParams(1.85798372e-02, 324.7775),
                                        {256, 256})},

        };
        return modeling_function;
    }

    /**
     * @brief Get src/dst-specific stride threshold from the direction LUT.
     *
     * Thresholds are stored per (device, direction) entry as a {src, dst} pair.
     * This helper selects src when @p is_source is true and dst otherwise.
     *
     * @param device VPU device type
     * @param direction Transfer direction (DDR2CMX, CMX2DDR, CMX2CMX, DDR2DDR)
     * @param is_source True for src/read, false for dst/write
     * @return Src/dst stride threshold in bytes for the selected endpoint
     */
    static int get_stride_threshold_bytes(const VPUDevice& device, MemoryDirection direction, bool is_source) {
        const auto& modeling_map = get_modeling_function_map();
        const auto it = modeling_map.find({device, direction});
        assert(it != modeling_map.end() && "Missing DMA stride-threshold LUT entry for (device, direction)");

        if (it == modeling_map.end()) {
            return 64;  // Safe fallback: smallest threshold used across all devices
        }

        return is_source ? it->second.thresholds.first : it->second.thresholds.second;
    }

    /// Stride penalty enablement map for DMA transfers
    /// Maps VPU device to boolean indicating whether stride penalty should be applied
    /// Default is false (penalty disabled) if device not in map
    static const std::unordered_map<VPUDevice, bool>& get_stride_penalty_enabled_map() {
        static const std::unordered_map<VPUDevice, bool> stride_penalty_enabled = {
                {VPUDevice::NPU_5_0, true},  // NPU 5: stride penalty enabled
                {VPUDevice::NPU_5_0_W, true},  // NPU 5W (WildcatLake): same as NPU 5
        };
        return stride_penalty_enabled;
    }

    /**
     * @brief Check if stride penalty should be applied for a given device
     *
     * @param device The VPU device to check
     * @return true if stride penalty is enabled for this device, false otherwise (default)
     */
    static bool is_stride_penalty_enabled(const VPUDevice& device) {
        const auto& penalty_map = get_stride_penalty_enabled_map();
        auto it = penalty_map.find(device);
        return (it != penalty_map.end()) ? it->second : false;  // Default: disabled
    }

    /**
     * @brief Calculate penalty factor for stride DMA transfers
     *
     * continuous_n_bytes: The maximum number of bytes that can be copied contiguously in the current stride DMA task.
     *
     * strideDMACorrectionThresholdInBytes: A threshold value used to determine when stride DMA becomes
     * inefficient. If continuous_n_bytes is less than this threshold, a penalty factor is applied to the estimated DMA
     * time. The value is selected by caller based on transfer src/dst. Typical threshold values are
     * 256 for DDR and 64 for CMX.
     *
     * penalty_factor: The factor by which the estimated DMA time is multiplied to account for inaccurate modeling of
     * stride DMA. The smaller the continuous_n_bytes (i.e., the more fragmented the data transfer), the larger the
     * penalty factor, leading to a higher estimated DMA time. This approach helps to provide a more accurate estimate
     * of DMA time in scenarios where stride DMA is used, especially when the data transfer is not contiguous. The
     * penalty factor is determined by the presence of stride DMA on the last axis of input or output tensors. If
     * neither input nor output tensors use stride DMA on their last axis, the penalty factor remains 1 (no penalty).
     *
     * @param continuous_n_bytes The maximum number of bytes that can be copied contiguously
     * @param threshold_bytes Src/dst threshold value to determine when stride DMA becomes inefficient
     *                        (typical threshold values: 256 for DDR, 64 for CMX)
     * @return penalty_factor The factor to multiply the estimated DMA time
     */
    static float get_penalty_factor_for_stride_dma(int continuous_n_bytes, int threshold_bytes) {
        assert(threshold_bytes > 0 && "threshold_bytes must be >= 1 (minimum meaningful word size)");
        if (continuous_n_bytes <= 0) {
            return 1.0f;  // No penalty if invalid input
        }
        // A = continuous_n_bytes / threshold_bytes.   >1 means larger TX than the threshold, <1 means tx(chunk) is
        // small Particular case if no stride or large chunks, continuous_n_bytes = total_bytes => A >> 1 => factor = 1
        // ceil(A)/ A
        // if A<1 => 1/A  big number if A small (scales with threshold_bytes)
        // if A > 1 =>  2/1.x => max factor 2 and decreasing towards 1

        float dma_block = std::ceil(static_cast<float>(continuous_n_bytes) / static_cast<float>(threshold_bytes));
        const float penalty_factor =
                (static_cast<float>(threshold_bytes) * dma_block) / static_cast<float>(continuous_n_bytes);

        return penalty_factor;
    }

    /**
     * @brief Helper method to compute DMA cycles with stride penalty
     *
     * This is the core stride DMA cost calculation logic:
     * 1. Find out the contiguous chunk (based on first stride that breaks continuity)
     * 2. Get the penalty factor knowing the transferred bytes (logical transfer size)
     * 3. Compute the runtime using the linear formula and penalty factor
     *
     * @param contiguous_bytes Maximum contiguous bytes that can be transferred without gaps, this will influence the
     * penalty factor, as smaller contiguous chunk will lead to higher penalty
     * @param total_bytes Total logical transfer size in bytes, this is the size that we want to transfer and will be
     * used to compute the base cycles before penalty
     * @param device VPU device type
     * @param direction Transfer direction (DDR2CMX, CMX2DDR, CMX2CMX, DDR2DDR) — selects the per-direction (k, b)
     * @param is_source True when called for src/read computation, false for dst/write computation
     * @return Estimated cycles (CMX) with stride penalty applied
     */
    unsigned long long computeCyclesWithStridePenalty(int contiguous_bytes, int total_bytes, const VPUDevice& device,
                                                      MemoryDirection direction, bool is_source) const {
        // Src/dst threshold comes from the LUT entry selected by (device, direction).
        // is_source chooses src/read threshold (true) or dst/write threshold (false).
        const int threshold_bytes = get_stride_threshold_bytes(device, direction, is_source);
        // Step 1: Get the penalty factor based on contiguous chunk size
        // Only apply penalty if:
        // - Device has stride penalty enabled
        // - We have strided access (contiguous_bytes < total_bytes)
        float penalty_factor = 1.0f;
        if (is_stride_penalty_enabled(device) && (contiguous_bytes < total_bytes)) {
            // We have stride and device supports penalty - apply penalty based on chunk size
            penalty_factor = get_penalty_factor_for_stride_dma(contiguous_bytes, threshold_bytes);
        }

        // Step 2: Compute the runtime using the linear formula and penalty factor
        // Use the linear modeling function: cycles = (k * size + b) * penalty_factor
        const unsigned long long base_cycles = query_dma_cost_by_size(total_bytes, device, direction);

        // Apply penalty factor
        const unsigned long long penalized_cycles =
                static_cast<unsigned long long>(static_cast<double>(base_cycles) * static_cast<double>(penalty_factor));

        return penalized_cycles;
    }

public:
    /**
     * @brief Compute DMA cycles for source side considering stride penalties
     *
     * This method implements the stride DMA cost calculation for the source (read) side:
     * 1. Find out the contiguous chunk (based on first stride that breaks continuity)
     * 2. Get the penalty factor knowing the transferred bytes (logical transfer size)
     * 3. Compute the runtime using the linear formula and penalty factor
     *
     * @tparam WlT DMA workload descriptor type (must expose getContiguousBytesSrc(), getReadBytes(), device)
     * @param wl The DMA workload descriptor
     * @return Estimated cycles for source-side DMA transfer (CMX cycles)
     */
    template <typename WlT>
    unsigned long long computeSrcCycles(const WlT& wl) const {
        const int contiguous_bytes_src = wl.getContiguousBytesSrc();  // how big is the innermost chunk
        const int total_bytes_src =
                wl.getReadBytes();  // total bytes of the transfer, all chunks summed up, regardless of stride, this is
                                    // the logical transfer size that we want to transfer
        const auto direction = wl.getDirection();
        const bool is_source = true;
        return computeCyclesWithStridePenalty(contiguous_bytes_src, total_bytes_src, wl.device, direction, is_source);
    }

    /**
     * @brief Compute DMA cycles for destination side considering stride penalties
     *
     * This method implements the stride DMA cost calculation for the destination (write) side:
     * 1. Find out the contiguous chunk (based on first stride that breaks continuity)
     * 2. Get the penalty factor knowing the transferred bytes (logical transfer size)
     * 3. Compute the runtime using the linear formula and penalty factor
     *
     * @tparam WlT DMA workload descriptor type (must expose getContiguousBytesDst(), getWrittenBytes(), device)
     * @param wl The DMA workload descriptor
     * @return Estimated cycles for destination-side DMA transfer (CMX cycles)
     */
    template <typename WlT>
    unsigned long long computeDstCycles(const WlT& wl) const {
        const int contiguous_bytes_dst = wl.getContiguousBytesDst();
        const int total_bytes_dst =
                wl.getWrittenBytes();  // total bytes of the transfer, all chunks summed up, regardless of stride, this
                                       // is the logical transfer size that we want to trans
        const auto direction = wl.getDirection();
        const bool is_source = false;
        return computeCyclesWithStridePenalty(contiguous_bytes_dst, total_bytes_dst, wl.device, direction, is_source);
    }

protected:
    /**
     * @brief Query DMA cost by size using the linear modeling function, with a bandwidth-based fallback.
     *
     * Selects the cost formula based on whether per-device-direction linear parameters (k, b) are
     * registered for the (@p device, @p direction) pair:
     *   - **Exact match found**: returns  (k * dma_size + b)  (cycles in CMX clock domain).
     *   - **No match found**: falls back to a bandwidth-based estimate.
     *
     *
     * @note Stride-penalty enablement does NOT influence which formula is chosen here; it only
     *       affects the multiplier applied by the caller (@ref computeCyclesWithStridePenalty).
     *
     * @param dma_size   Total logical transfer size in bytes.
     * @param device     Target VPU device; used to look up the linear model parameters.
     * @param direction  Transfer direction; used to select the per-direction (k, b) entry.
     * @return Estimated DMA cycles in the CMX clock domain (before DPU-clock conversion).
     */
    unsigned long long query_dma_cost_by_size(unsigned long long dma_size, const VPUDevice& device,
                                              MemoryDirection direction) const {
        const auto& modeling_map = get_modeling_function_map();

        // 1. Try exact (device, direction) match
        auto it = modeling_map.find({device, direction});

        // 3. Use linear model if any entry was found (no fallback if missing)
        if (it != modeling_map.end()) {
            const auto& params = it->second.linear;           // LinearModelingParams
            double cycles = params.compute_cycles(dma_size);  // linear, without penalty
            return static_cast<unsigned long long>(cycles);
        } else {
            // Device not in map at all — legacy bandwidth fallback,
            const int bw_bpc = hw_info.device(device).get_DMA_DDR_interface_bytes();  // bytes per cycle
            // Guard: non-positive bandwidth (e.g. -1 for unknown/default HW) falls back to 1 byte/cycle.
            const int safe_bw_bpc = bw_bpc > 0 ? bw_bpc : 1;
            const auto CMX_cycles = ((float)dma_size) / ((float)safe_bw_bpc);
            return std::max(1ULL, static_cast<unsigned long long>(CMX_cycles));
        }
    }

public:
    /**
     * @brief Estimates the DMA execution cycles for strided transfers
     *
     * This method calculates the number of execution cycles required for a DMA operation
     * by computing cycles for both source (read) and destination (write) sides, considering
     * stride penalties, and returns the maximum (slowest) as the bottleneck.
     *
     * @tparam WlT DMA workload descriptor type (must expose getContiguousBytesSrc/Dst(),
     *             getReadBytes(), getWrittenBytes(), and a `device` member of type VPUDevice)
     * @param wl The DMA workload descriptor
     * @return The estimated number of DPU cycles required to complete the DMA operation
     */
    template <typename WlT>
    CyclesInterfaceType DMAStridedCycles(const WlT& wl) const {
        // Compute cycles for source (read) side
        const auto src_cycles = computeSrcCycles(wl);

        // Compute cycles for destination (write) side
        const auto dst_cycles = computeDstCycles(wl);

        // src size and destination size is normally expected to be the same, otherwise we have an strange request.
        // For now allow , or ignore, this constraint. treat src and dst separately knowing the memory direction. For
        // the regular case with same size this is like taking the max between src and dst time.

        // Return the maximum (slowest path determines the total time).
        // The DMA transfer is limited by whichever side is slower for now. We so not know if DMA strided on both sides
        // changes this.
        const auto max_cycles_DMA{static_cast<unsigned long long>(std::max(src_cycles, dst_cycles))};

        const auto& hw{get_hw_characteristics(wl.device)};  // device characteristics for frequencies
        const float dpuPerCmx_clock_ratio{(float)hw.get_dpu_fclk() / (float)hw.get_cmx_fclk()};

        const auto cycles_DPU = Cycles::toCycleInterfaceType(
                max_cycles_DMA *
                dpuPerCmx_clock_ratio);  // transform to DPU cycles by applying the clock ratio between DPU and CMX, as
                                         // DMA is more related to CMX access but we want to express it in DPU cycles
        return cycles_DPU;
    }
};

/**
 * @class BaseDMAStridedCostProvider
 * @brief Provides strided-math DMA cycle estimates for NPU40/50 workloads.
 *
 * Wraps @ref DMAStridedMathModel and exposes it through the @ref IDMACostProvider interface.
 * For each workload the model:
 *   1. Determines the maximum contiguous byte run on both the source and destination sides,
 *      taking multi-dimensional strides into account (degenerate dimensions with dim_size==0
 *      are transparent — their stride is physically irrelevant and does not break contiguity).
 *   2. Derives a stride-penalty factor from the contiguous chunk size relative to the
 *      side-specific hardware threshold (CMX vs DDR):
 *      penalty = ceil(chunk/threshold) / (chunk/threshold).
 *      The penalty is only applied on devices that have it enabled.
 *   3. Evaluates a per-device linear model  cycles = k·size + b  on the total logical
 *      transfer size and multiplies by the penalty factor.
 *   4. Returns max(src_cycles, dst_cycles) converted to DPU clock domain via the
 *      DPU/CMX frequency ratio.
 *
 * The cost source tag returned through the optional @p cost_source output parameter is
 * @c "theoretical_strided".
 *
 * @tparam WlT  DMA workload descriptor type; defaults to @ref DMANNWorkload_NPU40_50.
 */
template <typename WlT>
class BaseDMAStridedCostProvider : public IDMACostProvider<WlT> {
protected:
    DMAStridedMathModel dma_theoretical_{
            HWCharacteristicsSuperSets::get_mainConfigurationRef()};  // new one, default config
public:
    CyclesInterfaceType get_cost(const WlT& wl, std::string* cost_source = nullptr) const override {
        if (cost_source) {
            *cost_source = "theoretical_strided";
        }

        return dma_theoretical_.DMAStridedCycles(wl);
    }
};

// name aliases for the two specializations of BaseDMAStridedCostProvider:
//  - DMAStridedMathCostProvider_NNIntf   : legacy interface accepting DMANNWorkload_NPU40_50
//  - DMAStridedMathCostProvider_DescIntf: new compiler-facing interface accepting VPUDMADescriptor
using DMAStridedMathCostProvider_NNIntf = BaseDMAStridedCostProvider<DMANNWorkload_NPU40_50>;
using DMAStridedMathCostProvider_DescIntf = BaseDMAStridedCostProvider<VPUDMADescriptor>;

}  // namespace VPUNN

#endif
