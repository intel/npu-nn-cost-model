// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_SHAVE_MODEL_BASICS_H
#define VPUNN_SHAVE_MODEL_BASICS_H

#include <iostream>
#include <limits>

#include "vpu/cycles_interface_types.h"
#include "vpu/types.h"

namespace VPUNN {
/// Conversion helpers between time (microseconds), SHAVE-cycle counts and DPU-cycle counts.
///
/// The `ShaveConverter` holds the nominal profiling frequencies and provides
/// conversion routines used by the provider adapters below. Conversions are
/// careful about integer overflow and expose standardized `CyclesInterfaceType`
/// error codes on invalid inputs.
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

    CyclesInterfaceType toDPUCyclesFromShaveCycles(const long long shaveCycles, const int present_dpu_frq,
                                                   const int present_shv_frq) const {
        if ((shaveCycles < 0) || (present_dpu_frq <= 0) || (present_shv_frq <= 0) ||
            (shaveCycles > static_cast<long long>(std::numeric_limits<CyclesInterfaceType>::max()))) {
            return Cycles::ERROR_INVALID_CONVERSION_TO_CYCLES;
        }

        // narrowing is safe here: shaveCycles was just bounded to the CyclesInterfaceType range above
        return Cycles::multiply_by_ratio(static_cast<CyclesInterfaceType>(shaveCycles),
                                         static_cast<long long>(present_dpu_frq),
                                         static_cast<long long>(present_shv_frq));
    }
};

inline std::ostream& operator<<(std::ostream& stream, const VPUNN::ShaveConverter& d) {
    stream << " dpu_freq_   : \t" << d.dpu_freq_ << " ;\n"  //
           << " shv_freq_   : \t" << d.shv_freq_ << " ;\n"  //
            ;
    return stream;
}

/// Template provider for models that estimate runtime in microseconds.
///
/// Contract for `Derived`:
/// - Must provide a `float getMicroSeconds(...)` method that returns the model
///   execution time in microseconds for the given arguments.
/// - The provider converts that microsecond value to DPU cycles using the
///   `ShaveConverter` instance (nominal DPU/SHV frequencies configured at
///   construction).
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

/// Adapter for VPUEM models that produce SHAVE-cycle counts.
///
/// Contract for `Derived`:
/// - Must provide an `int getShaveCycles(...)` method returning SHAVE cycles for
///   the given inputs.
/// - This provider converts those SHAVE-cycle counts to DPU cycles for a
///   requested DPU/SHV operating point using
///   `ShaveConverter::toDPUCyclesFromShaveCycles`.
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
    /* coverity[pass_by_value] */
    CyclesInterfaceType getDPUCyclesAnotherFreqDPU_SHV(
            const int present_dpu_frq, const int present_shv_frq,
            Ts... args) const {  // disabled coverity pass by value finding for Ts.. args because we don't want to force
                                 // arguments to be const or/and references
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

/// Adapter for heuristic-style models that return SHAVE cycles via
/// `getCycles(...)`.
///
/// Contract for `Derived`:
/// - Must expose `CyclesInterfaceType getCycles(...)` returning the SHAVE-cycle
///   cost for the given inputs.
/// - The adapter provides `getDPUCyclesAnotherFreqDPU_SHV(...)` which converts
///   those SHAVE cycles into DPU cycles for a requested DPU/SHV frequency.
template <typename Derived>
class HeuristicShaveCyclesProvider {
protected:
    const ShaveConverter converter;
    HeuristicShaveCyclesProvider(unsigned int DpuFreq, unsigned int ShvFreq): converter{DpuFreq, ShvFreq} {
    }

public:
    /**
     * @brief Determines the number of cycles for a given DPU/SHV operating point,
     * based on SHAVE cycles returned by Derived::getCycles(...)
     *
     * @return the number of cycles required based on CyclesInterfaceType
     */
    template <typename... Ts>
    CyclesInterfaceType getDPUCyclesAnotherFreqDPU_SHV(const int present_dpu_frq, const int present_shv_frq,
                                                       Ts... args) const {
        const CyclesInterfaceType shaveCycles = static_cast<const Derived*>(this)->getCycles(args...);
        if (Cycles::isErrorCode(shaveCycles)) {
            return shaveCycles;
        }
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

#endif // VPUNN_SHAVE_MODEL_BASICS_H
