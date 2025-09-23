// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef DPU_THEORETICAL_COST_PROVIDER_H
#define DPU_THEORETICAL_COST_PROVIDER_H

#include "vpu/performance.h"
#include "vpu/types.h"

namespace VPUNN {
/**
 * @class DPUTheoreticalCostProvider
 * @brief Provides theoretical performance modeling for DPU workloads.
 *
 * This class estimates the number of execution cycles required for various DPU operations,
 * considering hardware characteristics, workload parameters, and memory bandwidth.
 *
 * An instance of this class is intended to be use as a provider for theoretical cost for DPU workloads
 * An example of usage can be seen in class VPUCostModel where we either need just theoretical cost or
 * we use this as a fallback when NN cost not available
 */
class DPUTheoreticalCostProvider {
private:
    const HWPerformanceModel& performanceInfo;///< configured hw characteristics

public:
    DPUTheoreticalCostProvider(const HWPerformanceModel& performance_): performanceInfo(performance_) {
    }

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

        Pt_cycles =
                static_cast<unsigned long>(Pt_zeros) * static_cast<unsigned long>(Kw) * ceil_division(inp_width, Sw);
        Pb_cycles =
                static_cast<unsigned long>(Pb_zeros) * static_cast<unsigned long>(Kw) * ceil_division(inp_width, Sw);
        Pl_cycles =
                static_cast<unsigned long>(Pl_zeros) * static_cast<unsigned long>(Kh) * ceil_division(inp_height, Sh);
        Pr_cycles =
                static_cast<unsigned long>(Pr_zeros) * static_cast<unsigned long>(Kh) * ceil_division(inp_height, Sh);

        // Subtract double counted padding cycles at top-left corner from top
        for (int i = Pt, j = Pl; i > 0 && j > 0; i -= Sh, j -= Sw)
            Pt_cycles -= static_cast<unsigned long>(i) * static_cast<unsigned long>(j);

        // Subtract double counted padding cycles at top-right corner from top
        for (int i = Pt, j = (Pr - Redge); i > 0 && j > 0; i -= Sh, j -= Sw)
            Pt_cycles -= static_cast<unsigned long>(i) * static_cast<unsigned long>(j);

        // Subtract double counted padding cycles at bottom-left corner from bottom
        for (int i = (Pb - Bedge), j = Pl; i > 0 && j > 0; i -= Sh, j -= Sw)
            Pb_cycles -= static_cast<unsigned long>(i) * static_cast<unsigned long>(j);

        // Subtract double counted padding cycles at bottom-right corner from bottom
        for (int i = (Pb - Bedge), j = (Pr - Redge); i > 0 && j > 0; i -= Sh, j -= Sw)
            Pb_cycles -= static_cast<unsigned long>(i) * static_cast<unsigned long>(j);

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
        const auto& hw{performanceInfo.get_hw_info(wl.device)};  // device characteristics

        // Get the MPE model NTHW/NTK grid on X, Y, Z, B
        const auto grid = mpe_mode_to_nthw_ntk_grid(wl.execution_order);  // potentially OBSOLETE

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
        const auto reads =
                (double)(act_reads + wt_reads) / (hw.get_cmx_word_size_bytes() * hw.get_dpu_cmx_num_read_ports());

        // Return the number of CMX reads in DPU clock cycles
        return static_cast<unsigned long>(std::ceil(reads * (double)hw.get_cmx_fclk() / (double)hw.get_dpu_fclk()));
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

        const auto& hw{performanceInfo.get_hw_info(wl.device)};  // device characteristics

        const unsigned int inp_channels{wl.inputs[0].channels()};

        // Get the shape of the MPE grid
        // auto mpe_grid = mpe_mode_to_grid(wl.execution_order);
        // Compute the MACs needed to generate the output tensor
        unsigned long int cycles = (unsigned long int)multiply_vector(wl.kernels) *
                                   (unsigned long int)multiply_vector(wl.outputs[0].get_shape());

        const unsigned int mt{(wl.output_write_tiles > 1) ? 2U : 1U};
        const unsigned int nr_ppe{hw.get_nr_ppe()};
        const unsigned int fp_ratio = {hw.get_fp_ratio()};

        unsigned int nr_macs{hw.get_nr_macs()};

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
            nr_macs = nr_macs / hw.input_channels_mac();
        }

        // Ceil division cycles by DPU MACs
        cycles = ceil_division(cycles, (unsigned long int)nr_macs);

        // Adjust cycles for ratio of FP16 to int(or fp) 8 bit compute
        if (performanceInfo.native_comp_on_fp16(wl)) {
            cycles *= fp_ratio;
        }

        // Get CMX reads for NTHW/NTK
        auto nthw_ntk_reads = cmx_reads(wl);

        // Theoretical performance is the max between CMX reads and cycles (the bottleneck)
        return std::max<unsigned long>(cycles, nthw_ntk_reads);
    }
};

}  // namespace VPUNN

#endif
