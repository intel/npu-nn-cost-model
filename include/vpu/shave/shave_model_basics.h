// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef SHAVE_MODEL_BASICS_H
#define SHAVE_MODEL_BASICS_H

#include <iostream>

#include "vpu/cycles_interface_types.h"
#include "vpu/types.h"

namespace VPUNN {
/// converts to Cycles knowing frequencies (at acquisition and at present intended  target)
class ShaveConverter {
public:
    const unsigned int dpu_freq_;  ///< Measured in MHz, the Frequency at which the profiling was made.
    const unsigned int shv_freq_;  ///< Measured in MHz, the Frequency of a Activation Shave. Ex. Is the VPU_Freq in

    /**
     * @brief Determines the number of cycles related to the profiling DPU freq,
     * @return the number of cycles required based on CyclesInterfaceType
     */
    CyclesInterfaceType toDPUCycles(const float us) const {
        const float raw_cycles = us * dpu_freq_;
        const auto cycles = Cycles::toCycleInterfaceType(raw_cycles);
        return cycles;
    }
    /**
     * @brief Determines the number of cycles related to the input value of a DPU freq given as a parameter
     *
     * @return the number of cycles required based on CyclesInterfaceType
     */
    CyclesInterfaceType toDPUCycles(const float us, const int present_dpu_frq) const {
        const float raw_cycles = us * present_dpu_frq;
        const auto cycles = Cycles::toCycleInterfaceType(raw_cycles);
        return cycles;
    }
    /**
     * @brief Determines the number of cycles related to the input value of a DPU freq and SHV freq given as a
     * parameter. In order to get accurate numbers we use a change factor based on the
     * freq we made profile and the given value for the profiling
     *
     * @return the number of cycles required based on CyclesInterfaceType
     */
    CyclesInterfaceType toDPUCycles(const float us, const int present_dpu_frq, const int present_shv_frq) const {
        const float frq_change_factor = static_cast<float>(shv_freq_) / present_shv_frq;
        const float raw_cycles = (us * frq_change_factor) * present_dpu_frq;
        const auto cycles = Cycles::toCycleInterfaceType(raw_cycles);
        return cycles;
    }

    CyclesInterfaceType toDPUCyclesFromShaveCycles(const int shaveCycles, const int present_dpu_frq, const int present_shv_frq) const {
        const int raw_cycles = (shaveCycles * present_dpu_frq) / present_shv_frq;
        const auto cycles = Cycles::toCycleInterfaceType(raw_cycles);
        return cycles;
    }
};

inline std::ostream& operator<<(std::ostream& stream, const VPUNN::ShaveConverter& d) {
    stream << " dpu_freq_   : \t" << d.dpu_freq_ << " ;\n"  //
           << " shv_freq_   : \t" << d.shv_freq_ << " ;\n"  //
            ;
    return stream;
}

/// provides a  template interface for obtaining DPU cycles transparent of signature for  getMicroSeconds
/// @tapara, Derived, the derived Class of this  one. Must have a float getMicroSeconds(...) signature
template <typename Derived>
class ShaveCyclesProvider {
protected:
    const ShaveConverter converter;
    ShaveCyclesProvider(unsigned int DpuFreq, unsigned int ShvFreq): converter{DpuFreq, ShvFreq} {
    }

public:
    /**
     * @brief Determines the number of cycles related to the profiling DPU freq, based on the size of output
     *
     * @return the number of cycles required based on CyclesInterfaceType
     */
    template <typename... Ts>
    CyclesInterfaceType getDPUCycles(Ts... args) const {
        const float us = static_cast<const Derived*>(this)->getMicroSeconds(args...);
        return converter.toDPUCycles(us);
    }
    ///**
    // * @brief Determines the number of cycles related to the input value of a DPU freq given as a parameter based on the
    // * size of output
    // *
    // * @return the number of cycles required based on CyclesInterfaceType
    // */
    //template <typename... Ts>
    //CyclesInterfaceType getDPUCyclesAnotherFreqDPU(const int present_dpu_frq, Ts... args) const {
    //    const float us = static_cast<const Derived*>(this)->getMicroSeconds(args...);
    //    return converter.toDPUCycles(us, present_dpu_frq);
    //}
    /**
     * @brief Determines the number of cycles related to the input value of a DPU freq and SHV freq given as a parameter
     * based on the size of output. In order to get accurate numbers we use a change factor based on the freq we made
     * profile and the given value for the profiling
     *
     * @return the number of cycles required based on CyclesInterfaceType
     */
    template <typename... Ts>
    CyclesInterfaceType getDPUCyclesAnotherFreqDPU_SHV(const int present_dpu_frq, const int present_shv_frq,
                                                       Ts... args) const {
        const float us = static_cast<const Derived*>(this)->getMicroSeconds(args...);
        return converter.toDPUCycles(us, present_dpu_frq, present_shv_frq);
    }

    int getNominalDPUFrq() const {
        return converter.dpu_freq_;
    }
    int getNominalSHVFrq() const {
        return converter.shv_freq_;
    }
};


/// provides a  template interface for obtaining DPU cycles transparent of signature for  getMicroSeconds
/// @tapara, Derived, the derived Class of this  one. Must have a float getMicroSeconds(...) signature
template <typename Derived>
class VPUEMShaveCyclesProvider {
protected:
    const ShaveConverter converter;
    VPUEMShaveCyclesProvider(unsigned int DpuFreq, unsigned int ShvFreq): converter{DpuFreq, ShvFreq} {
    }

public:
   
    ///**
    // * @brief Determines the number of cycles related to the input value of a DPU freq given as a parameter based on
    // the
    // * size of output
    // *
    // * @return the number of cycles required based on CyclesInterfaceType
    // */
    // template <typename... Ts>
    // CyclesInterfaceType getDPUCyclesAnotherFreqDPU(const int present_dpu_frq, Ts... args) const {
    //    const float us = static_cast<const Derived*>(this)->getMicroSeconds(args...);
    //    return converter.toDPUCycles(us, present_dpu_frq);
    //}
    /**
     * @brief Determines the number of cycles related to the input value of a DPU freq and SHV freq given as a parameter
     * based on the size of output. In order to get accurate numbers we use a change factor based on the freq we made
     * profile and the given value for the profiling
     *
     * @return the number of cycles required based on CyclesInterfaceType
     */
    template <typename... Ts>
    CyclesInterfaceType getDPUCyclesAnotherFreqDPU_SHV(const int present_dpu_frq, const int present_shv_frq,
                                                       /* coverity[pass_by_value] */
                                                       Ts... args) const {
        const int shaveCycles = static_cast<const Derived*>(this)->getShaveCycles(args...);
        return converter.toDPUCyclesFromShaveCycles(shaveCycles, present_dpu_frq, present_shv_frq);
    }

    int getNominalDPUFrq() const {
        return converter.dpu_freq_;
    }
    int getNominalSHVFrq() const {
        return converter.shv_freq_;
    }
};

}  // namespace VPUNN

#endif