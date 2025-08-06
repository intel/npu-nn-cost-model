// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef DMA_POSTPROCESSING_MOCKS_H
#define DMA_POSTPROCESSING_MOCKS_H

#include "dma_post_process.h"

namespace VPUNN {

template <class DMADesc>
class IPostProcessDMA {
private:
    const float low_threshold;
    const float high_threshold;

public:
    virtual CyclesInterfaceType process(float, const DMADesc&, std::string&) const = 0;
    virtual ~IPostProcessDMA() = default;
    IPostProcessDMA(float low_threshold_, float high_threshold_)
            : low_threshold(low_threshold_), high_threshold(high_threshold_) {};

    /// @brief checks if the NN returned value is invalid, is outside of usable range
    /// @param nn_output_cycles , the value to be analyzed, this is assumed given by the NN inference
    /// @return true if invalid value
    bool is_NN_value_invalid(float nn_output_cycles) const {
        bool validity = false;
        if ((nn_output_cycles > high_threshold) || (nn_output_cycles < low_threshold)) {
            validity = true;
        }
        return validity;
    }

    /// @brief provides the value interval where the NN raw outputs are considered valid and will be used to further
    /// compute information
    ///
    /// @returns a pair containing (minimum_valid_value maximum_valid_value)
    std::pair<float, float> get_NN_Valid_interval() const {
        return std::make_pair(low_threshold, high_threshold);
    }
};

template <class DMADesc>
class ConvertFromSizeDivCycleToDPUCyc : public IPostProcessDMA<DMADesc> {
protected:
    /// NN clamping from minimum bandwith to maximum one
    static constexpr float max_NN_bandwith{1.F};
    static constexpr float min_bandwith{1.0f / 200.0F};  // 1 byte per 200 VPU cycles

    /// NN output limit, any higher should be treated like a not in range value given by the NN
    static constexpr float high_threshold{1.1F};
    static constexpr float low_threshold{-0.1F};

public:
    ConvertFromSizeDivCycleToDPUCyc(): IPostProcessDMA<DMADesc>(low_threshold, high_threshold) {};
    CyclesInterfaceType process(float nn_size_div_cycle,  // [0 to 1], 0% to 100% of  device's bytes per cycle
                                const DMADesc& wl, std::string& info) const override {
        const auto maxBytesPerCycle{get_DMA_DDR_interface_bytes(wl.device)};
        const auto raw_bandwith_BPC{nn_size_div_cycle *
                                    maxBytesPerCycle};  // range 0 to device bytes per cycle (full max speed) .

        // BW is limited , the NN might be with error that go slightly over those limits
        const auto bandwith_BPC = std::clamp(raw_bandwith_BPC, min_bandwith, max_NN_bandwith * maxBytesPerCycle);

        const auto size = wl.getAccessedBytes();

        if (bandwith_BPC > 0.0f) {
            const float vpu_cycles_f{size / bandwith_BPC};  // measured in VPU cycles

            const float dpu_cycles_f{vpu_cycles_f * get_dpu_fclk(wl.device) / get_cmx_fclk(wl.device)};
            const CyclesInterfaceType dpu_cycles{Cycles::toCycleInterfaceType(dpu_cycles_f)};

            return dpu_cycles;
        } else {
            {
                std::stringstream buffer;
                buffer << "\nThe computed DMA bandwidth is zero or negative :  " << bandwith_BPC
                       << ". NN returned: " << nn_size_div_cycle << ". "
                       << ".  Exiting with : " << Cycles::toErrorText(Cycles::ERROR_INVALID_OUTPUT_RANGE) << "\n";
                std::string details = buffer.str();
                info = info + details;
            }
            return Cycles::ERROR_INVALID_OUTPUT_RANGE;
        }
    }
};

template <class DMADesc>
class ConvertFromDirectCycleToDPUCyc : public IPostProcessDMA<DMADesc> {
private:
    static constexpr float high_threshold{1e9f};  // max reasonable cycles, 1 billion cycles max
    static constexpr float low_threshold{0.0f};

public:
    ConvertFromDirectCycleToDPUCyc(): IPostProcessDMA<DMADesc>(low_threshold, high_threshold) {};

    /// NN direct cycles conversion - no bandwidth calculation needed
    CyclesInterfaceType process(float nn_direct_cycles,  // DPU cycles directly from NN
                                const DMADesc&, std::string& info) const override {
        // Validate NN output range (should be positive cycles)
        if (nn_direct_cycles < 0.0f || std::isnan(nn_direct_cycles) || std::isinf(nn_direct_cycles)) {
            std::stringstream buffer;
            buffer << "\nThe DMA NN returned invalid direct cycles value: " << nn_direct_cycles
                   << ". Expected non-negative finite value."
                   << " Exiting with: " << Cycles::toErrorText(Cycles::ERROR_INVALID_OUTPUT_RANGE) << "\n";
            std::string details = buffer.str();
            info = info + details;
            return Cycles::ERROR_INVALID_OUTPUT_RANGE;
        }

        // Apply reasonable upper bound to prevent overflow
        if (nn_direct_cycles > high_threshold) {
            std::stringstream buffer;
            buffer << "\nThe DMA NN returned unreasonably large cycles value: " << nn_direct_cycles
                   << ". Clamping to maximum: " << high_threshold << "\n";
            std::string details = buffer.str();
            info = info + details;
            nn_direct_cycles = high_threshold;
        }

        // Convert to CyclesInterfaceType
        const CyclesInterfaceType dpu_cycles{Cycles::toCycleInterfaceType(nn_direct_cycles)};
        return dpu_cycles;
    }
};

}  // namespace VPUNN
#endif
