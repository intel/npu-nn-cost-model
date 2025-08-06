// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef ENERGY_INTERFACE_H
#define ENERGY_INTERFACE_H

#include <type_traits>
#include <utility>

#include "vpu/dpu_info_pack.h"
#include "vpu/power.h"
#include "vpu/shave_old.h"
#include "vpu/types.h"
#include "vpu/vpu_performance_model.h"

namespace VPUNN {

class VPUCostModel;  // fw declaration, at least for now we depend on the big cost model  as a provider for almost
                     // everything

/// provides energy computation
class IEnergy {
private:
    const VPUCostModel& all_service_provider;  ///< service as dpu, dma, performance.. providers

    const VPUPowerFactorLUT
            power_factor_lut{};  /// < this is the lookup table for power factors. to be split on generations

public:
    IEnergy(VPUCostModel& all_service_provider_): all_service_provider(all_service_provider_) {
    }

    /**
     * @brief Compute the energy of a DPUWorkload.
     * @details This is a relative energy metric with a time base in DPU clock cyles. Energy of
     * 1000 would mean energy of worst case power for 1000 DPU clock cycles at reference dynamic power (power virus
     * for INT08/FP16/FP8 operations). measured in PowerVirusJoules = PowerVirus*cycle
     * POwerVIrus reference type is device dependent.Normally choosing the type with max power
     * @param wl a DPUWorkload
     * @return float the DPUWorkload energy, measured  PowerVirus*cycle
     */
    float DPUEnergy(const DPUWorkload& wl) const {
        // const float activity_factor_powerVirus = DPU_PowerActivityFactor(wl);
        // const CyclesInterfaceType cycles{DPU(wl)};
        // return calculateEnergyFromAFandTime(activity_factor_powerVirus, cycles);

        // can be further reduced to power_ideal_cycles * power_factor_value  if no limitation desired
        const VPUNNPerformanceModel perf;
        return calculateEnergyFromIdealCycles(wl, perf.DPU_Power_IdealCycles(wl));
    }

    /**
     * @brief Compute the energy of a SHAVE SWOperation.
     * @details Energy here is a relative metric, but the activity factor of the SWOperation multiplied by
     *          its cost (number of clock cycles). We assume a constant activity factor of 0.5 for all and a max
     *          power of 5% of the DPU max power.
     *
     * @param swl a SWOperation
     * @return float the SWOperation energy , in units relative to DPU PowerVirus
     * \deprecated
     */
    float SHAVEEnergy(const SWOperation& swl) const;

    /** @brief Compute the energy of a SHAVE SHAVEWorkload.
     * @details Energy here is a relative metric, but the activity factor of the operation multiplied by
     *          its cost (number of clock cycles). We assume a constant activity factor of 0.5 for all and a max
     *          power of 5% of the DPU max power.
     *
     * @param swl a SHAVEWorkload
     * @return float the operation energy, in units relative to DPU PowerVirus. WIl return zero in case of error
     */
    float SHAVEEnergy(const SHAVEWorkload& swl) const;

    /**
     * @brief proxy for DPU_RelativeActivityFactor_hw
     */
    float DPUActivityFactor(const DPUWorkload& wl) const {
        return DPU_PowerActivityFactor(wl);
    }

    /** @brief integrates activity factor over the cycles duration=> from power to energy
     */
    float calculateEnergyFromAFandTime(const float activity_factor_powerVirus,
                                       const CyclesInterfaceType& cycles) const {
        const float checked_cycles{Cycles::isErrorCode(cycles) ? 0.0F : (float)cycles};  // zero if error
        const float energy = activity_factor_powerVirus * checked_cycles;
        return energy;
    }

    /**
     * @brief Compute DPUWorkload hw utilization based on ideal cycles considering also HW/sparsity.
     * This is in the context of the operation's datatype. (do not compare float with int values)
     * Represents the percentage [0,1+] of ideal resources(MAC based) used by this workload.
     * 1 = 100% of MACs are used
     * The value is calculated using the Estimated Runtime (cycles) by VPUNN.
     * If VPUNN is missing the TheoreticalCycles are used
     *
     * @param workload a DPUWorkload
     * @return  DPUWorkload hardware utilization (zero signals problems)
     */
    float hw_utilization(const DPUWorkload& wl) const {
        return power_mac_hw_utilization(wl);
    }

    /**
     * @brief Compute DPUWorkload hw utilization based on ideal cycles considering also HW/sparsity.
     * This is in the context of the operation's datatype. (do not compare float with int values)
     * Represents the percentage [0,1+] of ideal resources(MAC based) used by this workload.
     * 1 = 100% of MACs are used
     * The value is calculated using the Estimated Runtime (cycles) by VPUNN.
     * If VPUNN is missing the TheoreticalCycles are used
     *
     * @param workload a DPUWorkload
     * @return  DPUWorkload hardware utilization (zero signals problems)
     */
    float power_mac_hw_utilization(const DPUWorkload& wl) const {
        return mac_hw_utilization(wl, &VPUNNPerformanceModel::DPU_Power_IdealCycles);
    }

    /** @bief utilization without sparsity, can be larger than one, uses CostModel */
    float efficiency_mac_hw_utilization(const DPUWorkload& wl) const {
        return mac_hw_utilization(wl, &VPUNNPerformanceModel::DPU_Efficency_IdealCycles);
    }

public:
    /**
     * @brief Compute DPUWorkload hw utilization based on received ideal cycles.
     * This is in the context of the operation's datatype. (do not compare float with int values)
     * Represents the percentage [0,1+] of ideal resources(MAC based) used by this workload.
     * 1 = 100% of MACs are used
     * The value is calculated using the Estimated Runtime (cycles) by VPUNN.
     * If VPUNN is missing the TheoreticalCycles are used
     * Values larger than 1 can  be obtained if the ideal_cycles are larger than eNN estimated ones
     * result = ideal_cycles/estimatedNNCycles
     *
     * @param workload a DPUWorkload
     * @param ideal_cycles the reference ideal cycles against to compute the utilization
     * @return  DPUWorkload hardware utilization (zero signals problems)
     */
    float relative_mac_hw_utilization(const CyclesInterfaceType real_cycles,
                                      const unsigned long int ideal_cycles) const {
        float utilization = 0.0F;  // zero signals problems
        const auto& nn_output_cycles = real_cycles;
        if ((!Cycles::isErrorCode(nn_output_cycles)) && nn_output_cycles != 0) {
            utilization = (float)ideal_cycles / nn_output_cycles;  // NORMAL CASE,
        }

        return utilization;
    }

    /**
     * @brief Compute the activity factor of a DPUWorkload.
     * @details Activity factor is an estimation of the dynamic power of the DPUWorkload
     * relative to the worst case (reference dynamic power) DPUWorkload.
     * Interval [0, 1 or more], where 1 means the power virus activity factor
     * reference dynamic power is considered for INT8 operations
     * It can be more than 1 in case the PowerViruschosen for reference is not the fact the highest (like if reference
     * is power virus INT8,  the float operations can have the AF >1).
     * Internally uses CostModel
     *
     * @param wl a DPUWorkload
     * @return float the DPUWorkload activity factor relative to reference PowerVirus  (now is INT8)
     */
    float DPU_PowerActivityFactor(const DPUWorkload& wl) const {
        const float mac_utilization_rate = power_mac_hw_utilization(wl);  // if zero will propagate error

        // if we have sparsity , the power per cycle might be higher (more hardware firing  for the same operation)?
        // do we need a power correction here or only at energy computation.
        // what happens when sparsity (w) is o n but we have dense values, no sparsity gain, but should be more energy
        // spent also?

        const float rough_powerVirus_relative_af = DPU_AgnosticActivityFactor(wl, mac_utilization_rate);

        const float maximum_acepted_af{power_factor_lut.get_PowerVirus_exceed_factor(wl.device)};

        const float restricted_powerVirus_relative_af = std::min(rough_powerVirus_relative_af, maximum_acepted_af);

        return restricted_powerVirus_relative_af;
    }

    /// Internally uses CostModel
    float DPU_EfficiencyActivityFactor(const DPUWorkload& wl) const {
        const float mac_utilization_rate = efficiency_mac_hw_utilization(wl);  // if zero will propagate error

        const float powerVirus_relative_af = DPU_AgnosticActivityFactor(wl, mac_utilization_rate);
        // no limitation  known to be applied
        return powerVirus_relative_af;
    }

    // protected: now public to can be accessed in VPUCostModel
    float DPU_AgnosticActivityFactor(const DPUWorkload& wl, const float reference_hw_util,
                                     const float sparse_correction_factor_experimental = 1.0F) const {
        const float power_factor_value = power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl);

        return DPU_AgnosticActivityFactor_formula(power_factor_value, reference_hw_util,
                                                  sparse_correction_factor_experimental);
    }

    float DPU_AgnosticActivityFactor_formula(const float power_factor_value, const float reference_hw_util,
                                             const float sparse_correction_factor_experimental = 1.0F) const {
        const float rough_powerVirus_relative_af{(reference_hw_util * power_factor_value) *
                                                 sparse_correction_factor_experimental};

        return rough_powerVirus_relative_af;
    }

    float calculateEnergyFromIdealCycles(const DPUWorkload& wl, const unsigned long int reference_ideal_cycles) const {
        const float power_factor_value = power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl);

        // should we scale with sparse ON but dense?
        // is there a limit, probably not as long as this is time independent

        const float energy = reference_ideal_cycles * power_factor_value;
        return energy;
    }

    /// @brief fills in the fields of DPUInfoPack
    /// some fields have to be already populated
    void fillDPUInfo(DPUInfoPack& allData, const DPUWorkload& w) const {
        {
            allData.sparse_mac_operations = getPerformanceModel().compute_HW_MAC_operations_cnt(w);
            allData.power_ideal_cycles = getPerformanceModel().DPU_Power_IdealCycles(w);
            allData.power_mac_utilization = relative_mac_hw_utilization(allData.DPUCycles, allData.power_ideal_cycles);
            // to be restricted
            {
                const float rough_powerVirus_relative_af =
                        DPU_AgnosticActivityFactor(w, allData.power_mac_utilization);  // DPU_PowerActivityFactor(w);

                const float nominal_allowed_Virus_exceed_factor{
                        power_factor_lut.get_PowerVirus_exceed_factor(w.device)};
                const float restricted_powerVirus_relative_af =
                        std::min(rough_powerVirus_relative_af, nominal_allowed_Virus_exceed_factor);
                allData.power_activity_factor = restricted_powerVirus_relative_af;
            }

            // allData.energy = calculateEnergyFromAFandTime(allData.power_activity_factor, allData.DPUCycles);
            allData.energy = calculateEnergyFromIdealCycles(w, allData.power_ideal_cycles);
        }

        {
            allData.dense_mac_operations = getPerformanceModel().compute_Ideal_MAC_operations_cnt(w);
            allData.efficiency_ideal_cycles = getPerformanceModel().DPU_Efficency_IdealCycles(w);
            allData.efficiency_mac_utilization =
                    relative_mac_hw_utilization(allData.DPUCycles, allData.efficiency_ideal_cycles);
            allData.efficiency_activity_factor = DPU_AgnosticActivityFactor(
                    w, allData.efficiency_mac_utilization);  // DPU_EfficiencyActivityFactor(w);
        }
    }

protected:
    static_assert(std::is_same<decltype(&VPUNN::VPUNNPerformanceModel::DPU_Efficency_IdealCycles),
                               decltype(&VPUNN::VPUNNPerformanceModel::DPU_Power_IdealCycles)>::value,
                  "must be same signature ");

    float mac_hw_utilization(const DPUWorkload& wl,
                             decltype(&VPUNNPerformanceModel::DPU_Efficency_IdealCycles) CalculateCycles) const;

private:
    const VPUCostModel& getDPUprovider() const {
        return all_service_provider;
    }

    const VPUCostModel& getSHAVEprovider() const {
        return all_service_provider;
    }
    const VPUNNPerformanceModel& getPerformanceModel() const;
};

}  // namespace VPUNN
#endif
