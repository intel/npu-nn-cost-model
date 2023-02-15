// Copyright © 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_PERFORMANCE_H
#define VPUNN_PERFORMANCE_H

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
inline unsigned int get_dpu_fclk(VPUDevice device) {
    switch (device) {
    case VPUDevice::VPU_2_0:
        return 700;
    case VPUDevice::VPU_2_1:
        return 850;
    case VPUDevice::VPU_2_7:
        return 1300;
    case VPUDevice::VPU_4_0:
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
inline unsigned int get_cmx_fclk(VPUDevice device) {
    switch (device) {
    case VPUDevice::VPU_2_0:
        return 700;
    case VPUDevice::VPU_2_1:
        return 850;
    case VPUDevice::VPU_2_7:
        return 975;
    case VPUDevice::VPU_4_0:
        return 975;
    default:
        return 700;
    }
}

/**
 * @brief Get the DRAM bandwidth in MB/s for a specific VPU IP
 *
 * @param device a VPUDevice
 * @return float
 */
inline float get_dram_bandwidth_MBps(VPUDevice device) {
    switch (device) {
    case VPUDevice::VPU_2_0:
    case VPUDevice::VPU_2_1:
        return 20000.0f;
    case VPUDevice::VPU_2_7:
        return 27000.0f;
    case VPUDevice::VPU_4_0:
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
inline unsigned int get_sram_word_size(bool compression) {
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
 * @brief Get the DMA latency in cycles
 *
 * @param device a VPUDevice
 * @return unsigned int
 */
inline unsigned int get_DMA_latency(VPUDevice device, MemoryLocation location) {
    switch (device) {
    case VPUDevice::VPU_2_7:
        return location == MemoryLocation::DRAM ? 950 : 50;
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
inline unsigned int get_nr_macs(VPUDevice device) {
    switch (device) {
    case VPUDevice::VPU_2_0:
    case VPUDevice::VPU_2_1:
        return 256;
    case VPUDevice::VPU_2_7:
        return 2048;
    default:
        return 2048;
    }
}

/**
 * @brief Get the number of PPE
 *
 * @param device a VPUDevice
 * @return unsigned int
 */
inline unsigned int get_nr_ppe(VPUDevice device) {
    switch (device) {
    case VPUDevice::VPU_2_0:
    case VPUDevice::VPU_2_1:
        return 16;
    case VPUDevice::VPU_2_7:
        return 64;
    default:
        return 64;
    }
}

/**
 * @brief Get the MAC/input channels/cycles for a specific VPU IP
 *
 * @param device a VPUDevice
 * @return unsigned int
 */
inline unsigned int input_channels_mac(VPUDevice device) {
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
inline unsigned int nDPU_per_tile(VPUDevice device) {
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
    unsigned long int PaddingSkipCycles(DPUWorkload& wl, unsigned long int nr_macs) {
        // Extract kernels and padding cycles from workload
        unsigned int Kw = wl.kernels[0], Kh = wl.kernels[1];
        unsigned int Sw = wl.strides[0], Sh = wl.strides[1];
        unsigned int Pt = wl.padding[0], Pb = wl.padding[1];
        unsigned int Pl = wl.padding[2], Pr = wl.padding[3];
        unsigned int Pt_zeros = 0, Pb_zeros = 0;
        unsigned int Pl_zeros = 0, Pr_zeros = 0;
        unsigned int inp_width = wl.inputs[0].width();
        unsigned int inp_height = wl.inputs[0].height();
        unsigned int out_channels = wl.outputs[0].channels();
        unsigned int in_channels = wl.inputs[0].channels();

        // PAdding Skip cycles
        unsigned long int Pt_cycles, Pb_cycles;
        unsigned long int Pl_cycles, Pr_cycles;

        for (unsigned int i = Pt; i > 0; i -= Sh)
            Pt_zeros += i;
        for (unsigned int i = 1; i <= Pb; i += Sh)
            Pb_zeros += i;
        for (unsigned int i = Pl; i > 0; i -= Sw)
            Pl_zeros += i;
        for (unsigned int i = 1; i <= Pr; i += Sw)
            Pr_zeros += i;

        Pt_cycles = Pt_zeros * Kw * out_channels * ceil_division(inp_width, Sw);
        Pb_cycles = Pb_zeros * Kw * out_channels * ceil_division(inp_width, Sw);
        Pl_cycles = Pl_zeros * Kh * out_channels * ceil_division(inp_height, Sh);
        Pr_cycles = Pr_zeros * Kh * out_channels * ceil_division(inp_height, Sh);

        Pt_cycles = ceil_division(Pt_cycles * in_channels, nr_macs);
        Pb_cycles = ceil_division(Pb_cycles * in_channels, nr_macs);
        Pl_cycles = ceil_division(Pl_cycles * in_channels, nr_macs);
        Pr_cycles = ceil_division(Pr_cycles * in_channels, nr_macs);

        return Pt_cycles + Pb_cycles + Pl_cycles + Pr_cycles;
    }

    /**
     * @brief Get the CMX reads in DPU clock cycles
     *
     * @param wl a DPUWorkload
     * @return unsigned long theoretical CMX execution cycles
     */
    unsigned long cmx_reads(DPUWorkload& wl) {
        if (wl.device == VPUDevice::VPU_2_0 || wl.device == VPUDevice::VPU_2_1) {
            // For VPU20 CMX reads are encapsulated into the NN model
            return 0;
        }
        // Get the MPE model NTHW/NTK grid on X, Y, Z, B
        auto grid = mpe_mode_to_nthw_ntk_grid(wl.execution_order);

        // Get the number of weights and activation grid reads
        double num_wt_grids = ceil((double)wl.outputs[0].channels() / (double)grid[2]);
        double num_act_grids = ceil((double)wl.outputs[0].height() / (double)grid[1]) *
                               ceil((double)wl.outputs[0].width() / (double)grid[0]);

        // 16 is the CMX word size
        auto act_reads = (num_wt_grids * wl.outputs[0].height() * wl.outputs[0].width() * wl.inputs[0].channels() *
                          wl.kernels[0] * wl.kernels[1]) /
                         16.0;
        auto wt_reads =
                (num_act_grids * wl.outputs[0].channels() * wl.inputs[0].channels() * wl.kernels[0] * wl.kernels[1]) /
                16.0;

        // Return the number of CMX reads in DPU clock cycles
        return static_cast<unsigned long>(
                ceil((act_reads + wt_reads) * (double)get_cmx_fclk(wl.device) / (double)get_dpu_fclk(wl.device)));
    }

public:
    /**
     * @brief Compute the DPU theoretical cycles
     *
     * @param wl a DPUWorkload
     * @return unsigned long int theoretical execution cycles
     */
    unsigned long int DPUTheoreticalCycles(DPUWorkload& wl) {
        if (wl.outputs.size() == 0) {
            // If it computes no output, its duration is 0 cycles
            return 0;
        }

        if (wl.kernels[0] > wl.inputs[0].width()) {
            wl.kernels[0] = wl.inputs[0].width();
        }
        if (wl.kernels[1] > wl.inputs[0].height()) {
            wl.kernels[1] = wl.inputs[0].height();
        }
        unsigned int mt, nr_macs, nr_ppe;
        unsigned int inp_channels = wl.inputs[0].channels();

        // Get the shape of the MPE grid
        auto mpe_grid = mpe_mode_to_grid(wl.execution_order);
        // Compute the MACs needed to generate the output tensor
        unsigned long int cycles = (unsigned long int)multiply_vector(wl.kernels) *
                                   (unsigned long int)multiply_vector(wl.outputs[0].get_shape());
        mt = 1;
        if (wl.output_write_tiles > 1) {
            mt = 2;
        }
        nr_macs = get_nr_macs(wl.device);
        nr_ppe = get_nr_ppe(wl.device);
        // As per Bernard David: ELTWISE_ST = (C*H*W)/64 --- ELTWISE_MT = (C*H*W)/(64/2) --- ST = single tile --- MT =
        // multi tile The 64 is 64 Bytes per clock at the slow CMX frequency – if MC is enabled this reduces to 32 Bytes
        // per clock on ODU
        if (wl.op == Operation::ELTWISE) {
            cycles = ceil_division(multiply_vector(wl.inputs[0].get_shape()), (nr_ppe / mt));
        }
        // For CONV, we multiply over the input channels
        if (wl.op == Operation::CONVOLUTION || wl.op == Operation::CM_CONVOLUTION) {
            // Ceil division between input channels and the DPU mac
            cycles *= (unsigned long int)inp_channels;
        } else {
            nr_macs = nr_macs / input_channels_mac(wl.device);
        }

        cycles = ceil_division(cycles, (unsigned long int)nr_macs);
#ifdef VPUNN_USE_PADDING
        cycles -= PaddingSkipCycles(wl, nr_macs);
#endif
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
    unsigned long int DMATheoreticalCycles(const DMAWorkload& wl) {
        // Get if the input is permuted or compressed
        bool is_input_permuted =
                wl.input.get_layout() != wl.output.get_layout() && wl.input_location == MemoryLocation::CMX;
        bool is_input_compressed = wl.input.size() != wl.output.size() && wl.input_location == MemoryLocation::CMX;
        // Get the bandwidth in DPU cycles/bytes
        float input_bandwidth = get_bandwidth_cycles_per_bytes(wl.input, wl.device, wl.input_location,
                                                               is_input_permuted, is_input_compressed);
        // Compute input cycles from dimensions and bw
        auto input_cycles = (unsigned long)ceil((double)wl.input.size() * (double)input_bandwidth);
        // Get if the output is permuted or compressed
        bool is_output_compressed = wl.input.size() != wl.output.size() && wl.output_location == MemoryLocation::CMX;
        bool is_output_permuted =
                wl.input.get_layout() != wl.output.get_layout() && wl.output_location == MemoryLocation::CMX;
        // Get the bandwidth in DPU cycles/bytes
        float output_bandwidth = get_bandwidth_cycles_per_bytes(wl.output, wl.device, wl.output_location,
                                                                is_output_compressed, is_output_permuted);
        // Compute input cycles from dimensions and bw
        auto output_cycles = (unsigned long)ceil((double)wl.output.size() * (double)output_bandwidth);

        // Get latency in cycles
        auto input_latency = (unsigned long)get_DMA_latency(wl.device, wl.input_location);
        auto output_latency = (unsigned long)get_DMA_latency(wl.device, wl.output_location);
        // Get the max between input and output cycles
        return std::max(input_latency, output_latency) + std::max(input_cycles, output_cycles);
    }

    /**
     * @brief Compute the Shave Kernel theoretical cycles
     *
     * @param swl a Shave Kernel
     * @return unsigned int theoretical execution cycles
     */
    unsigned int SHAVETheoreticalCycles(SWOperation& swl) {
        if (swl.outputs.size() == 0) {
            // If it computes no output, its duration is 0 cycles
            return 0;
        }

        return swl.cycles();
    }
};

}  // namespace VPUNN

#endif  // VPUNN_PERFORMANCE_H
