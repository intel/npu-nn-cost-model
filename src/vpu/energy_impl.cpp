#include "vpu/energy_interface.h"
#include "vpu_cost_model.h"

float VPUNN::IEnergy::mac_hw_utilization(
        const DPUWorkload& wl, decltype(&VPUNNPerformanceModel::DPU_Efficency_IdealCycles) CalculateCycles) const{
    std::string dummy_info{};
    DPUWorkload w{wl};
    const auto nn_output_cycles = getDPUprovider().DPU(w, dummy_info);      // might change W, considers sparsities
    const auto ideal_cycles = (getPerformanceModel().*CalculateCycles)(w);  //< this is independent of NN cycles

    return relative_mac_hw_utilization(nn_output_cycles, ideal_cycles);
}

float VPUNN::IEnergy::SHAVEEnergy(const SWOperation& swl) const {
    constexpr float activity_factor{0.5f};      //<assume a constant activity factor of 0.5
    const float max_power_ratio_to_DPU{0.05f};  //<assume a max power of 5% of the DPU max power.
    const float energy = (activity_factor * max_power_ratio_to_DPU) * (float)getSHAVEprovider().SHAVE(swl);

    return energy;
}

float VPUNN::IEnergy::SHAVEEnergy(const SHAVEWorkload& swl) const {
    constexpr float activity_factor{0.5f};      //<assume a constant activity factor of 0.5
    const float max_power_ratio_to_DPU{0.05f};  //<assume a max power of 5% of the DPU max power.

    std::string infoOut;
    const auto shave_raw_time{getSHAVEprovider().SHAVE_2(swl, infoOut)};
    const float shave_ftime{Cycles::isErrorCode(shave_raw_time) ? 0.0f : (float)(shave_raw_time)};
    const float energy = (activity_factor * max_power_ratio_to_DPU) * shave_ftime;

    return energy;
}


const VPUNN::VPUNNPerformanceModel& VPUNN::IEnergy::getPerformanceModel() const {
    return all_service_provider;
}
