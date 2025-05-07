// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_VPU_PERFORMANCE_MODEL_H
#define VPUNN_VPU_PERFORMANCE_MODEL_H

#include "vpu/cycles_interface_types.h"

#include "core/vpunn_api.h"
#include "vpu/datatype_collection_size.h"
#include "vpu/types.h"
#include "vpu/utils.h"

#include "performance.h"

#include "vpu/shave_old.h"

#include "performance_mode.h"

namespace VPUNN {

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
        const double num_wt_grids =
                /*static_cast<long>*/ (std::ceil((double)wl.outputs[0].channels() / (double)grid[Dim::Act::Z]));
        const double num_act_grids =
                /*static_cast<long>*/ (std::ceil((double)wl.outputs[0].height() / (double)grid[Dim::Act::Y]) *
                                       std::ceil((double)wl.outputs[0].width() / (double)grid[Dim::Act::X]));

        const auto kernel_area = wl.kernels[Dim::Grid::W] * wl.kernels[Dim::Grid::H];

        const auto datatype{wl.inputs[0].get_dtype()};  // use input zero, wt are not present

        // Compute total number of bytes of activations and weights to read. @todo: review formulas

        // @todo, review formula , what if INT4 and alignment is at innermost dimension?
        const auto elements_count_act{
                static_cast<long>(std::ceil(num_wt_grids * wl.outputs[0].height() * wl.outputs[0].width() *
                                            wl.inputs[0].channels() * kernel_area))};
        const auto act_reads{compute_size_in_bytes(elements_count_act, datatype)};

        const auto elements_count_wt{static_cast<long>(
                std::ceil(num_act_grids * wl.outputs[0].channels() * wl.inputs[0].channels() * kernel_area))};
        const auto wt_reads{compute_size_in_bytes(elements_count_wt, datatype)};

        // Compute idealized total number of read cycles
        const auto reads = (double)(act_reads + wt_reads) /
                           (get_cmx_word_size_bytes(wl.device) * get_dpu_cmx_num_read_ports(wl.device));

        // Return the number of CMX reads in DPU clock cycles
        return static_cast<unsigned long>(
                std::ceil(reads * (double)get_cmx_fclk(wl.device) / (double)get_dpu_fclk(wl.device)));
    }

public:
    /**
     * @brief Compute the DPU ideal cycles, considers HW optimizations like sparsity
     * @details Calculates cycles that a single issue scalar CPU would require to execute
     * a DPUWorkload then divides by number of MACs which can be performed in
     * parallel by DPU. All operations are base-lined in the same manner with no
     * non ideal factors considered at all.
     * Like: Number of cycles if all the MAC resources are used 100%.
     * Sparsity is considered for inputs and weights
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
     * hardware details like sparsity.
     *
     * @param wl a DPUWorkload
     * @return number of operations
     */
    unsigned long int compute_HW_MAC_operations_cnt(const DPUWorkload& wl) const {
        // Compute the MACs needed to generate the output tensor
        const unsigned long int ideal_operations_cnt{compute_Ideal_MAC_operations_cnt(wl)};
        unsigned long int hw_operations_cnt{ideal_operations_cnt};

        // not active will say 1.0 = dense, 100%
        auto calc_non_zero_operations = [](bool enabled, const float sparsity) {
            if (enabled) {
                const float non_zero_operations_factorWeights_raw{1.0f - sparsity};
                const float non_zero_operations_factorWeights{
                        std::max(0.0f, std::min(1.0f, non_zero_operations_factorWeights_raw))};
                return non_zero_operations_factorWeights;
            }
            return 1.0f;
        };

        // sparsities are present
        if ((wl.weight_sparsity_enabled) || (wl.inputs[0].get_sparsity())) {
            //  model  sparse acceleration for w. OR activation

            const float act_non_zero_operations_fact{
                    calc_non_zero_operations(wl.inputs[0].get_sparsity(), wl.act_sparsity)};

            const float wts_non_zero_operations_fact{
                    calc_non_zero_operations(wl.weight_sparsity_enabled, wl.weight_sparsity)};

            // combined polity is the most influential (this is a conservative approach). Has to be correlated with the
            // implementation in VPUCostModel::runNN_dualsparsity
            const float combined_non_zero_fact{std::min(act_non_zero_operations_fact, wts_non_zero_operations_fact)};

            hw_operations_cnt = static_cast<unsigned long int>(std::ceil(hw_operations_cnt * combined_non_zero_fact));
        }

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
     * Note: THISIS OBSOLETE/NOT UPDATED
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

    unsigned long int DMATheoreticalCycles(const DMAWorkload& wl) const {
        if (wl.device < VPUDevice::VPU_4_0) {
            return DMATheoreticalCyclesLegacyLNL(wl);
        } else {  // VPU 4.0 and newer
            if (wl.device == VPUDevice::VPU_4_0 && PerformanceMode::forceLegacy_G4) {
                return DMATheoreticalCyclesLegacyLNL(wl);
            } else if (wl.device > VPUDevice::VPU_4_0 && PerformanceMode::forceLegacy_G5) {
                return DMATheoreticalCyclesLegacyLNL(wl);
            }
            return DMATheoreticalCyclesRESERVED_ON(wl);//Updated theoretical model
        }
    }

    /**
     * @brief Compute the DMA theoretical cycles
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
        const float input_bandwidth =
                get_bandwidth_cycles_per_bytesLegacy(wl.input, wl.device, wl.input_location, is_input_compressed,
                                                     is_input_permuted, is_half_duplex_limitation);
        // Compute input cycles from dimensions and bw
        const auto input_cycles = Cycles::toCycleInterfaceType((double)wl.input.size() * (double)input_bandwidth);

        // Get if the output is permuted or compressed
        const bool is_output_permuted = ((wl.input.get_layout() != wl.output.get_layout())  // changing layout
                                         && (wl.output_location == MemoryLocation::CMX));   // and dst to CMX

        const bool is_output_compressed = ((wl.input.size() != wl.output.size())             // changing size
                                           && (wl.output_location == MemoryLocation::CMX));  // and dst to CMX

        // Get the bandwidth in DPU cycles/bytes
        const float output_bandwidth =
                get_bandwidth_cycles_per_bytesLegacy(wl.output, wl.device, wl.output_location, is_output_compressed,
                                                     is_output_permuted, is_half_duplex_limitation);
        // Compute output cycles from dimensions and bw
        const auto output_cycles = Cycles::toCycleInterfaceType((double)wl.output.size() * (double)output_bandwidth);

        // Get latency in cycles
        const auto input_latency = (unsigned long)get_DMA_latency_Legacy(wl.device, wl.input_location);
        const auto output_latency = (unsigned long)get_DMA_latency_Legacy(wl.device, wl.output_location);

        // Get the max between input and output cycles
        return Cycles::cost_adder(std::max(input_latency, output_latency), std::max(input_cycles, output_cycles));
    }

protected:
    /// CMX2CMX is half-duplex on NPU 2.x
    bool isHalfDuplexContext(const DMAWorkload& wl) const {
        return ((wl.device <= VPUDevice::VPU_2_7)  // specific device
                && (wl.input_location == MemoryLocation::CMX) && (wl.output_location == MemoryLocation::CMX))
                       ? true
                       : false;
    }

    //// CMX CLocks
    // float compute_DRAM_bandwith_cycPerB(const VPUDevice& device) const {
    //     // DRAM bw is given in MBps
    //     const float ddr_cycPerBytes{get_cmx_fclk(device) / get_dram_bandwidth_MBps(device)};
    //     const float cmx_bounded_maxCyclesPerByte{1.0f / get_DMA_DDR_interface_bytes(device)};
    //     return std::max(ddr_cycPerBytes, cmx_bounded_maxCyclesPerByte);  // take worst case
    // }
    //  cmx clock
    int compute_DRAM_bandwith_BytesPerCyc(const VPUDevice& device) const {
        // DRAM bw is given in MBps
        const int ddr_BytesPerCyc{static_cast<int>(std::floor(get_dram_bandwidth_MBps(device) / get_cmx_fclk(device)))};
        const int cmx_bounded_maxBytesPerCyc{get_DMA_DDR_interface_bytes(device)};
        return std::min(ddr_BytesPerCyc, cmx_bounded_maxBytesPerCyc);
    }

    //// cmx clock
    // float get_cycles_per_byte_read_bw(const VPUTensor& tensor, VPUDevice device, MemoryLocation location,
    //                                   bool half_duplex = false) const {
    //     switch (location) {
    //     case MemoryLocation::DRAM:
    //         return compute_DRAM_bandwith_cycPerB(device);

    //    case MemoryLocation::CMX:
    //    case MemoryLocation::CSRAM:  //?
    //    case MemoryLocation::UPA:    //?
    //    default:
    //        return (1.0F / (float)get_sram_word_size(tensor, false, false, half_duplex));
    //    }
    //}

    // cmx clock
    int get_bytes_per_cycle_read_bw(const VPUTensor& tensor, VPUDevice device, MemoryLocation location,
                                    bool half_duplex = false) const {
        switch (location) {
        case MemoryLocation::DRAM:
            return compute_DRAM_bandwith_BytesPerCyc(device);

        case MemoryLocation::CMX:
        case MemoryLocation::CSRAM:  //?
        case MemoryLocation::UPA:    //?
        default:
            // return (get_sram_word_size(tensor, false, false, half_duplex));
            return cmx_agregated_bytes_per_cycle_bw(tensor, device, half_duplex, false, false);
        }
    }
    int cmx_raw_word_size(const VPUDevice device, bool half_duplex) const {
        if (half_duplex) {
            return get_DMA_DDR_interface_bytes(device) / 2;
        }
        return get_DMA_DDR_interface_bytes(device);
    }

    // CMX clock. COnsiders also limitation like compression...
    int cmx_agregated_bytes_per_cycle_bw(const VPUTensor& tensor, VPUDevice device, bool half_duplex, bool permute,
                                         bool compression, float decompression_ratio = 1.0F,
                                         int compressed_BW_BytesPerCycle = 1) const {
        // permute limits the bw to one element per cycle
        if (permute) {
            return dtype_to_bytes(tensor.get_dtype());  // size of one element (what about half duplex?). should not be
                                                        // larger than CMX BW?
        }

        const auto nominal_bw = cmx_raw_word_size(device, half_duplex);  // nominal
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

    // float get_cycles_per_bytes_write_bw(const VPUTensor& tensor, VPUDevice device, MemoryLocation location,
    //                                     bool half_duplex = false) const {
    //     switch (location) {
    //     case MemoryLocation::DRAM:
    //         // noy influenced by permuted!
    //         // not influenced by compression!
    //         return compute_DRAM_bandwith_cycPerB(device);

    //    case MemoryLocation::CMX:
    //    case MemoryLocation::CSRAM:  //?
    //    case MemoryLocation::UPA:    //?
    //    default:

    //        // SRAM bw is twice in compression mode
    //        return (1.0F / (float)get_sram_word_size(tensor, false, false, half_duplex));
    //    }
    //}
    int get_bytes_per_cycle_write_bw(const VPUTensor& tensor, VPUDevice device, MemoryLocation location,
                                     bool half_duplex, bool permute, bool compression,
                                     float decompression_ratio /*= 1.0F*/,
                                     int compressed_BW_BytesPerCycle /* = 0*/) const {
        switch (location) {
        case MemoryLocation::DRAM:
            // noy influenced by permuted!
            // not influenced by compression!
            return compute_DRAM_bandwith_BytesPerCyc(device);

        case MemoryLocation::CMX:
        case MemoryLocation::CSRAM:  //?
        case MemoryLocation::UPA:    //?
        default:
            return (cmx_agregated_bytes_per_cycle_bw(tensor, device, half_duplex, permute, compression,
                                                     decompression_ratio, compressed_BW_BytesPerCycle));
        }
    }

public:
    unsigned long int DMATheoreticalCyclesRESERVED_ON(const DMAWorkload& wl) const {
        const float dpuPerCmx_clock_ratio{(float)get_dpu_fclk(wl.device) / (float)get_cmx_fclk(wl.device)};
        const bool is_half_duplex_limitation{isHalfDuplexContext(wl)};

        const bool is_cmx2cmx_permutation = ((wl.input.get_layout() != wl.output.get_layout())  // changing layout
                                             && (wl.input_location == MemoryLocation::CMX)      // and src/dest from CMX
                                             && (wl.output_location == MemoryLocation::CMX));   // and src/dest from CMX

        const bool is_DDR2CMX_decompresion =
                ((wl.input.size() < wl.output.size())              // dest size is bigger (decompression)
                 && (wl.input_location == MemoryLocation::DRAM)    // and src from DDR
                 && (wl.output_location == MemoryLocation::CMX));  // and dst to CMX

        const float decompression_ratio{is_DDR2CMX_decompresion ? ((float)wl.output.size() / (float)wl.input.size())
                                                                : 1.0f};

        const int input_bw_bpc =
                get_bytes_per_cycle_read_bw(wl.input, wl.device, wl.input_location, is_half_duplex_limitation);
        const auto CMX_cycles_read = (float)wl.input.size() / (float)input_bw_bpc;
        const auto input_cycles_DPU = Cycles::toCycleInterfaceType(CMX_cycles_read * dpuPerCmx_clock_ratio);

        const int output_bw_bpc = get_bytes_per_cycle_write_bw(
                wl.output, wl.device, wl.output_location, is_half_duplex_limitation, is_cmx2cmx_permutation,
                is_DDR2CMX_decompresion, decompression_ratio, input_bw_bpc);
        const auto CMX_cycles_write = (float)wl.output.size() / (float)output_bw_bpc;
        const auto output_cycles_DPU = Cycles::toCycleInterfaceType(CMX_cycles_write * dpuPerCmx_clock_ratio);

        // Get latency in cycles
        const auto input_latency_DPU = get_DMA_latency(wl.device, wl.input_location);
        const auto output_latency_DPU = get_DMA_latency(wl.device, wl.output_location);

        // Get the max between input and output cycles
        return Cycles::cost_adder(std::max(input_latency_DPU, output_latency_DPU),
                                  std::max(input_cycles_DPU, output_cycles_DPU));
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

}  // namespace VPUNN

#endif  // VPUNN_PERFORMANCE_H
