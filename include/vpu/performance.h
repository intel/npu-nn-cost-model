// Copyright © 2023 Intel Corporation
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
#include "vpu/shave/ShaveModel1to1.h"
#include "vpu/shave/layers.h"

#include "vpu/types.h"
#include "vpu/utils.h"
#include "vpunn.h"

namespace VPUNN {

/**
 * @brief Get the DPU default frequency in MHz
 *
 * @param device a VPUDevice
 * @return unsigned int
 */
inline constexpr unsigned int get_dpu_fclk(VPUDevice device) {
    switch (device) {
    case VPUDevice::VPU_2_0:
        return 700;
    case VPUDevice::VPU_2_1:
        return 850;
    case VPUDevice::VPU_2_7:
        return 1300;
    case VPUDevice::VPU_RESERVED:
        return 1700;
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
inline constexpr unsigned int get_cmx_fclk(VPUDevice device) {
    switch (device) {
    case VPUDevice::VPU_2_0:
        return 700;
    case VPUDevice::VPU_2_1:
        return 850;
    case VPUDevice::VPU_2_7:
        return 975;
    case VPUDevice::VPU_RESERVED:
        return 975;
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
    case VPUDevice::VPU_RESERVED:
        return 32;
    default:
        return 16;
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
    case VPUDevice::VPU_RESERVED:
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
    case VPUDevice::VPU_RESERVED:
        return 45000.0f;
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
inline constexpr unsigned int get_sram_word_size(bool compression) {
    return compression ? 64 : 32;
}

/**
 * @brief Get the sram word size
 *
 * @param tensor a VPUTensor
 * @param compression if compression is enabled or not
 * @param permute if a permute operation is required
 * @return  unsigned int
 */
inline unsigned int get_sram_word_size(const VPUTensor& tensor, bool compression, bool permute) {
    if (!permute) {
        // same layout -> linear DMA so DST width is equal to tensor size
        return std::min(tensor.size(), get_sram_word_size(compression));
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
                                            bool compression = false, bool permute = false) {
    switch (location) {
    case MemoryLocation::DRAM:
        // DRAM bw is given in MBps
        return get_dpu_fclk(device) / get_dram_bandwidth_MBps(device);
    default:
        // SRAM bw is twice in compression mode
        return (float)get_dpu_fclk(device) / (float)get_cmx_fclk(device) /
               ((float)get_sram_word_size(tensor, compression, permute));
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
                                bool compression = false, bool permute = false) {
    auto bw_cycles_per_bytes = get_bandwidth_cycles_per_bytes(tensor, device, location, compression, permute);
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
    case VPUDevice::VPU_RESERVED: {
        constexpr VPUDevice const_device{VPUDevice::VPU_RESERVED};
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
    case VPUDevice::VPU_RESERVED:
        return 2048;
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
    case VPUDevice::VPU_RESERVED:
        return 64;
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
 * @brief VPUNN performance model
 *
 */
class VPUNN_API(VPUNNPerformanceModel) {
private:
    /**
     * @brief Estimate theoretical reduction in cycles from padding for ZMAJOR convolution
     * @details DPU always treats padding as sparse
     *
     * @param wl a DPUWorkload
     * @return unsigned long theoretical CMX execution cycles
     */
    unsigned long int PaddingSkipCycles(const DPUWorkload& wl) const {
        // Extract kernels and padding cycles from workload
        unsigned int Kw = wl.kernels[0], Kh = wl.kernels[1];
        unsigned int Sw = wl.strides[0], Sh = wl.strides[1];
        unsigned int Pt = wl.padding[0], Pb = wl.padding[1];
        unsigned int Pl = wl.padding[2], Pr = wl.padding[3];
        unsigned int inp_width = wl.inputs[0].width();
        unsigned int inp_height = wl.inputs[0].height();
        unsigned int out_channels = wl.outputs[0].channels();
        unsigned int in_channels = wl.inputs[0].channels();

        // Accumulate padding zeros at top, left
        unsigned int Pt_zeros = 0, Pb_zeros = 0;
        unsigned int Pl_zeros = 0, Pr_zeros = 0;
        for (int i = Pt; i > 0; i -= Sh)
            Pt_zeros += i;
        for (int i = Pl; i > 0; i -= Sw)
            Pl_zeros += i;

        // Accumulate padding zeros at right and bottom, depending on input width and stride
        unsigned int Redge = inp_width % Sw;
        unsigned int Bedge = inp_height % Sh;
        for (int i = (Pr - Redge); i > 0; i -= Sw)
            Pr_zeros += i;
        for (int i = (Pb - Bedge); i > 0; i -= Sh)
            Pb_zeros += i;

        unsigned long int Pt_cycles, Pb_cycles;
        unsigned long int Pl_cycles, Pr_cycles;

        Pt_cycles = Pt_zeros * Kw * ceil_division(inp_width, Sw);
        Pb_cycles = Pb_zeros * Kw * ceil_division(inp_width, Sw);
        Pl_cycles = Pl_zeros * Kh * ceil_division(inp_height, Sh);
        Pr_cycles = Pr_zeros * Kh * ceil_division(inp_height, Sh);

        // Subtract double counted padding cycles at top-left corner from top
        for (int i = Pt, j = Pl; i > 0 && j > 0; i -= Sh, j -= Sw)
            Pt_cycles -= i * j;

        // Subtract double counted padding cycles at top-right corner from top
        for (int i = Pt, j = (Pr - Redge); i > 0 && j > 0; i -= Sh, j -= Sw)
            Pt_cycles -= i * j;

        // Subtract double counted padding cycles at bottom-left corner from bottom
        for (int i = (Pb - Bedge), j = Pl; i > 0 && j > 0; i -= Sh, j -= Sw)
            Pb_cycles -= i * j;

        // Subtract double counted padding cycles at bottom-right corner from bottom
        for (int i = (Pb - Bedge), j = (Pr - Redge); i > 0 && j > 0; i -= Sh, j -= Sw)
            Pb_cycles -= i * j;

        return (Pt_cycles + Pb_cycles + Pl_cycles + Pr_cycles) * in_channels * out_channels;
    }

    /**
     * @brief Get the CMX reads in DPU clock cycles
     *
     * @param wl a DPUWorkload
     * @return unsigned long theoretical CMX execution cycles
     */
    unsigned long cmx_reads(const DPUWorkload& wl) const {
        if (wl.device == VPUDevice::VPU_2_0 || wl.device == VPUDevice::VPU_2_1) {
            // For VPU2.0 CMX reads are encapsulated into the NN model
            return 0;
        }
        // Get the MPE model NTHW/NTK grid on X, Y, Z, B
        const auto grid = mpe_mode_to_nthw_ntk_grid(wl.execution_order);

        // Get the number of weights and activation grid reads
        const double num_wt_grids = ceil((double)wl.outputs[0].channels() / (double)grid[Dim::Act::Z]);
        const double num_act_grids = ceil((double)wl.outputs[0].height() / (double)grid[Dim::Act::Y]) *
                                     ceil((double)wl.outputs[0].width() / (double)grid[Dim::Act::X]);

        const auto kernel_area = wl.kernels[Dim::Grid::W] * wl.kernels[Dim::Grid::H];
        const auto bytes_per_element = dtype_to_bytes(wl.inputs[0].get_dtype());  // use input zero, wt are not present

        // Compute total number of bytes of activations and weights to read. @todo: review formulas
        const auto act_reads = num_wt_grids * wl.outputs[0].height() * wl.outputs[0].width() * wl.inputs[0].channels() *
                               bytes_per_element * kernel_area;

        const auto wt_reads =
                num_act_grids * wl.outputs[0].channels() * wl.inputs[0].channels() * bytes_per_element * kernel_area;

        // Compute idealized total number of read cycles
        const auto reads =
                (act_reads + wt_reads) / (get_cmx_word_size_bytes(wl.device) * get_dpu_cmx_num_read_ports(wl.device));

        // Return the number of CMX reads in DPU clock cycles
        return static_cast<unsigned long>(
                ceil(reads * (double)get_cmx_fclk(wl.device) / (double)get_dpu_fclk(wl.device)));
    }

public:
    /**
     * @brief Compute the DPU ideal cycles, considers HW optimizations like sparsity
     * @details Calculates cycles that a single issue scalar CPU would require to execute
     * a DPUWorkload then divides by number of MACs which can be performed in
     * parallel by DPU. All operations are base-lined in the same manner with no
     * non ideal factors considered at all.
     * Like: Number of cycles if all the MAC resources are used 100%.
     * Sparsity is considered only for weights!
     *
     * @param wl a DPUWorkload
     * @return  ideal execution DPU cycles
     */
    unsigned long int DPU_Power_IdealCycles(const DPUWorkload& wl) const {
        // Compute the MACs needed to generate the output tensor
        const unsigned long int operations_cnt = compute_HW_MAC_operations_cnt(wl);
        return DPU_MAC_based_cycles(wl, operations_cnt);
    }
    /**
     * @brief Compute the DPU ideal cycles, pure MAC based, no hw optimizations
     * @details Calculates cycles that a single issue scalar CPU would require to execute
     * a DPUWorkload then divides by number of MACs which can be performed in
     * parallel by DPU. All operations are base-lined in the same manner with no
     * non ideal factors considered at all.
     * Like: Number of cycles if all the MAC resources are used 100%.
     *
     * @param wl a DPUWorkload
     * @return  ideal execution DPU cycles
     */
    unsigned long int DPU_Efficency_IdealCycles(const DPUWorkload& wl) const {
        // Compute the MACs needed to generate the output tensor
        const unsigned long int operations_cnt = compute_Ideal_MAC_operations_cnt(wl);
        return DPU_MAC_based_cycles(wl, operations_cnt);
    }

protected:
    /**
     * @brief Compute the DPU ideal cycles
     * @details Calculates cycles that a single issue scalar CPU would require to execute
     * a DPUWorkload then divides by number of MACs which can be performed in
     * parallel by DPU. All operations are base-lined in the same manner with no
     * non ideal factors considered at all.
     * Like: Number of cycles if all the MAC resources are used 100%.
     *
     * @param wl a DPUWorkload
     * @param MACs_to_compute how many MAC operations are required to do for the wl. (computed outsided, may or may not
     * consider HW optimizations like sparsity)
     * @return  ideal execution DPU cycles
     */
    unsigned long int DPU_MAC_based_cycles(const DPUWorkload& wl, const unsigned long int MACs_to_compute) const {
        if (wl.outputs.size() == 0) {  // If it computes no output, its duration is 0 cycles
            return 0;
        }
        const unsigned int nr_macs{get_nr_macs(wl.device)};
        const unsigned int fp_to_int_resource_ratio{get_fp_ratio(wl.device)};  // more cycles for fp vs int

        const unsigned int nr_macs_adjusted_with_type{
                native_comp_is_fp(wl) ? ceil_division(nr_macs, fp_to_int_resource_ratio) : nr_macs};

        // Compute the MACs needed to generate the output tensor
        const unsigned long int operations_cnt = MACs_to_compute;

        // Ceil division cycles by DPU MACs for all operations
        const unsigned long int cycles = ceil_division<unsigned long int>(operations_cnt, nr_macs_adjusted_with_type);

        return cycles;
    }

    /**
     * @brief Computes how many MACs are required to generate this output
     * @details Calculates operations that a single issue scalar CPU would require to execute a DPUWorkload, no sparsity
     * or other HW details are taken in consideration
     *
     * @param wl a DPUWorkload
     * @return number of operations
     */
    unsigned long int compute_Ideal_MAC_operations_cnt(const DPUWorkload& wl) const {
        // Compute the MACs needed to generate the output tensor
        unsigned long int operations_cnt{0};
        if (wl.op == Operation::CONVOLUTION || wl.op == Operation::CM_CONVOLUTION) {
            const unsigned long int operations_cnt_base = (unsigned long int)multiply_vector(wl.kernels) *
                                                          (unsigned long int)multiply_vector(wl.outputs[0].get_shape());
            const auto channels{wl.inputs[0].channels()};
            if (wl.device < VPUDevice::VPU_2_7) {
                operations_cnt = operations_cnt_base * channels;
            } else {
                // NPU2.7 or newer. Channel less than 16 are special
                if (channels < 16) {
                    operations_cnt = operations_cnt_base * 16;
                } else {
                    operations_cnt = operations_cnt_base * channels;
                }
            }

        } else if (wl.op == Operation::ELTWISE) {
            operations_cnt = multiply_vector(wl.inputs[0].get_shape());  // kernel is 1
        } else {  // All other operations, including DW convolution and pooling
            operations_cnt = (unsigned long int)multiply_vector(wl.kernels) *
                             (unsigned long int)multiply_vector(wl.outputs[0].get_shape());
        }
        return operations_cnt;
    }
    /**
     * @brief Computes how many MACs are required to generate this output
     * @details Calculates operations that a single issue scalar CPU would require to execute a DPUWorkload considering
     * hardware d etails like sparsity.
     *
     * @param wl a DPUWorkload
     * @return number of operations
     */
    unsigned long int compute_HW_MAC_operations_cnt(const DPUWorkload& wl) const {
        // Compute the MACs needed to generate the output tensor
        const unsigned long int ideal_operations_cnt{compute_Ideal_MAC_operations_cnt(wl)};

        unsigned long int hw_operations_cnt{ideal_operations_cnt};
        //  model  sparse acceleration for w.
        if (wl.weight_sparsity_enabled) {
            const float non_zero_operations_factorWeights_raw{1.0f - wl.weight_sparsity};
            const float non_zero_operations_factorWeights{
                    std::max(0.0f, std::min(1.0f, non_zero_operations_factorWeights_raw))};

            hw_operations_cnt =
                    static_cast<unsigned long int>(std::ceil(hw_operations_cnt * non_zero_operations_factorWeights));
        }
        // @todo: model in case of activation sparsity also present.
        return hw_operations_cnt;
    }

public:
    /**
     * @brief Compute the DPU theoretical cycles, maximum HW knowledge
     * @details Calculates cycles that a single issue scalar CPU would require to execute
     *          a DPUWorkload then divides by number of MACs which can be performed in
     *          parallel by DPU. Also considers data type, CMX memory bandwidth and some
     *          other (non-ideal) factors.
     * NO sparsity is considered.
     * @param wl a DPUWorkload
     * @return unsigned long int theoretical execution cycles
     */
    unsigned long int DPUTheoreticalCycles(const DPUWorkload& wl) const {
        if (wl.outputs.size() == 0) {
            // If it computes no output, its duration is 0 cycles
            return 0;
        }

        const unsigned int inp_channels{wl.inputs[0].channels()};

        // Get the shape of the MPE grid
        // auto mpe_grid = mpe_mode_to_grid(wl.execution_order);
        // Compute the MACs needed to generate the output tensor
        unsigned long int cycles = (unsigned long int)multiply_vector(wl.kernels) *
                                   (unsigned long int)multiply_vector(wl.outputs[0].get_shape());

        const unsigned int mt{(wl.output_write_tiles > 1) ? 2U : 1U};
        const unsigned int nr_ppe{get_nr_ppe(wl.device)};
        const unsigned int fp_ratio = {get_fp_ratio(wl.device)};

        unsigned int nr_macs{get_nr_macs(wl.device)};

        // As per Bernard David: ELTWISE_ST = (C*H*W)/64 --- ELTWISE_MT = (C*H*W)/(64/2) --- ST = single tile --- MT
        // = multi tile The 64 is 64 Bytes per clock at the slow CMX frequency – if MC is enabled this reduces to 32
        // Bytes per clock on ODU
        if (wl.op == Operation::ELTWISE) {
            cycles = ceil_division(multiply_vector(wl.inputs[0].get_shape()), (nr_ppe / mt));
        }
        // For CONV, we multiply over the input channels and remove padding, always treated as sparse
        if (wl.op == Operation::CONVOLUTION || wl.op == Operation::CM_CONVOLUTION) {
            cycles *= (unsigned long int)inp_channels;
            cycles -= PaddingSkipCycles(wl);
        } else {
            nr_macs = nr_macs / input_channels_mac(wl.device);
        }

        // Ceil division cycles by DPU MACs
        cycles = ceil_division(cycles, (unsigned long int)nr_macs);

        // Adjust cycles for ratio of FP to int compute
        if (native_comp_is_fp(wl)) {
            cycles *= fp_ratio;
        }

        // Get CMX reads for NTHW/NTK
        auto nthw_ntk_reads = cmx_reads(wl);

        // Theoretical performance is the max between CMX reads and cycles (the bottleneck)
        return std::max<unsigned long>(cycles, nthw_ntk_reads);
    }

    /**
     * @brief Compute the DMA theoretical cycles
     *
     * @param wl a DMAWorkload
     * @return unsigned long int theoretical execution cycles
     */
    unsigned long int DMATheoreticalCycles(const DMAWorkload& wl) const {
        // Get if the input is permuted or compressed
        bool is_input_permuted =
                wl.input.get_layout() != wl.output.get_layout() && wl.input_location == MemoryLocation::CMX;
        bool is_input_compressed = wl.input.size() != wl.output.size() && wl.input_location == MemoryLocation::CMX;
        // Get the bandwidth in DPU cycles/bytes
        float input_bandwidth = get_bandwidth_cycles_per_bytes(wl.input, wl.device, wl.input_location,
                                                               is_input_permuted, is_input_compressed);
        // Compute input cycles from dimensions and bw
        auto input_cycles = Cycles::toCycleInterfaceType((double)wl.input.size() * (double)input_bandwidth);
        // Get if the output is permuted or compressed
        bool is_output_compressed = wl.input.size() != wl.output.size() && wl.output_location == MemoryLocation::CMX;
        bool is_output_permuted =
                wl.input.get_layout() != wl.output.get_layout() && wl.output_location == MemoryLocation::CMX;
        // Get the bandwidth in DPU cycles/bytes
        float output_bandwidth = get_bandwidth_cycles_per_bytes(wl.output, wl.device, wl.output_location,
                                                                is_output_compressed, is_output_permuted);
        // Compute input cycles from dimensions and bw
        auto output_cycles = Cycles::toCycleInterfaceType((double)wl.output.size() * (double)output_bandwidth);

        // Get latency in cycles
        auto input_latency = (unsigned long)get_DMA_latency(wl.device, wl.input_location);
        auto output_latency = (unsigned long)get_DMA_latency(wl.device, wl.output_location);
        // Get the max between input and output cycles
        return Cycles::cost_adder(std::max(input_latency, output_latency), std::max(input_cycles, output_cycles));
    }

    /**
     * @brief Compute the Shave Kernel theoretical cycles
     *
     * @param swl a Shave Kernel
     * @return unsigned int theoretical execution cycles
     */
    unsigned int SHAVETheoreticalCycles(const SWOperation& swl) {
        if (swl.outputs.size() == 0) {  // If it computes no output, its duration is 0 cycles
            return 0;
        }
        return swl.cycles();
    }
};

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
    case VPUDevice::VPU_RESERVED:
        return 2;
    default:
        return 1;
    }
}

}  // namespace VPUNN

#endif  // VPUNN_PERFORMANCE_H
