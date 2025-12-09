// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include "vpu/power.h"
#include "vpu_cost_model.h"

#include <gtest/gtest.h>
#include "common/common_helpers.h"
#include "vpu/cycles_interface_types.h"
#include "vpu/dpu_theoretical_cost_provider.h"

#include <algorithm>
#include <unordered_map>

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {
using namespace VPUNN;

class TestEnergyandPF_CostModel : public ::testing::Test /* TestCostModel*/ {
public:
protected:
    
    const HWPerformanceModel performanceProvider{};

    void SetUp() override {
        // TestCostModel::SetUp();
    }

    void basicTest(const DPUWorkload& workload, VPUCostModel& crt_model, std::string info = "") {
        DPUWorkload wl{workload};

        const VPUPowerFactorLUT power_factor_lut;
        const float operation_pf = power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider);
        const float exceedMax = power_factor_lut.get_PowerVirus_exceed_factor(wl.device);

        const IEnergy& my_energy = crt_model.getEnergyInterface();
        const HWPerformanceModel& performance{crt_model.getPerformanceModel()};
        const DPUTheoreticalCostProvider& dpu_theoretical{performanceProvider};

        const auto energy = crt_model.DPUEnergy(wl);
        const auto af = my_energy.DPUActivityFactor(wl);
        const auto util = my_energy.hw_utilization(wl);
        const auto util_idealCyc = performance.DPU_Power_IdealCycles(wl);
        const auto efficiency_idealCyc = performance.DPU_Efficency_IdealCycles(wl);
        const auto theorCyc = dpu_theoretical.DPUTheoreticalCycles(wl);
        std::string errInfo;
        const auto nnCyc = crt_model.DPU(wl, errInfo);

        std::stringstream buffer;
        buffer << info << " Op:" << Operation_ToText.at(static_cast<int>(wl.op)) << "\n  NN cyc:" << nnCyc
               << ", ThCyc: " << theorCyc << ", Power Ideal Cyc: " << util_idealCyc
               << ", Efficiency Ideal Cyc: " << efficiency_idealCyc << "\n Utilization(ideal/NNcyc): " << util
               << " Energy: " << energy << " powerAF: " << af << " pf oper correlation :" << operation_pf << "\n ";
        std::string details = buffer.str();

        EXPECT_TRUE(!Cycles::isErrorCode(nnCyc)) << info << wl << errInfo << details;

        EXPECT_GT(energy, 0) << details;

        EXPECT_GT(af, 0) << details;
        EXPECT_LE(af, 1.0f * exceedMax) << details;
        EXPECT_GT(util, 0) << details;
        EXPECT_LE(util, 1.0f) << details;

        EXPECT_NEAR(energy, (float)util_idealCyc * operation_pf, 1) << details;

        std::cout << details
                  << "-X-------------------------------------------------------------------------------------------"
                     "-\n";
    }

    void basicDPUPackEquivalenceTest(const DPUWorkload& workload, VPUCostModel& crt_model, std::string info = "") {
        DPUWorkload wl{workload};
        const VPUPowerFactorLUT power_factor_lut;
        const float operation_pf = power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider);

        const IEnergy& my_energy = crt_model.getEnergyInterface();
        const HWPerformanceModel& performance{crt_model.getPerformanceModel()};
        const DPUTheoreticalCostProvider& dpu_theoretical{performanceProvider};

        const auto energy = crt_model.DPUEnergy(wl);
        const auto af = my_energy.DPUActivityFactor(wl);
        const auto util = my_energy.hw_utilization(wl);
        const auto util_idealCyc = performance.DPU_Power_IdealCycles(wl);
        const auto efficiency_idealCyc = performance.DPU_Efficency_IdealCycles(wl);
        const auto theorCyc = dpu_theoretical.DPUTheoreticalCycles(wl);
        std::string errInfo;
        const auto nnCyc = crt_model.DPU(wl, errInfo);

        std::stringstream buffer;
        buffer << info << " Op:" << Operation_ToText.at(static_cast<int>(wl.op)) << "\n  NN cyc:" << nnCyc
               << ", ThCyc: " << theorCyc << ",Power Ideal Cyc: " << util_idealCyc
               << ", Efficiency Ideal Cyc: " << efficiency_idealCyc << "\n Utilization(ideal/NNcyc): " << util
               << " Energy: " << energy << " powerAF: " << af << " pf oper correlation :" << operation_pf << "\n ";
        std::string details = buffer.str();

        DPUWorkload wl_pack{workload};
        const DPUInfoPack allinfo = crt_model.DPUInfo(wl_pack);

        EXPECT_TRUE(!Cycles::isErrorCode(nnCyc)) << info << wl << errInfo << details;

        EXPECT_EQ(allinfo.DPUCycles, nnCyc) << info << wl << errInfo << details << allinfo;
        EXPECT_EQ(allinfo.errInfo, errInfo) << info << wl << errInfo << details << allinfo;

        EXPECT_FLOAT_EQ(allinfo.energy, energy) << info << wl << errInfo << details << allinfo;
        EXPECT_FLOAT_EQ(allinfo.power_activity_factor, af) << info << wl << errInfo << details << allinfo;
        EXPECT_FLOAT_EQ(allinfo.power_mac_utilization, util) << info << wl << errInfo << details << allinfo;
        EXPECT_EQ(allinfo.power_ideal_cycles, util_idealCyc) << info << wl << errInfo << details << allinfo;
        EXPECT_EQ(allinfo.efficiency_ideal_cycles, efficiency_idealCyc) << info << wl << errInfo << details << allinfo;
        EXPECT_EQ(allinfo.hw_theoretical_cycles, theorCyc) << info << wl << errInfo << details << allinfo;

        std::cout << details
                  << "-X-------------------------------------------------------------------------------------------"
                     "-\n";
    }

    void basicSparseTest(const DPUWorkload& workload, VPUCostModel& crt_model, std::string info = "") {
        DPUWorkload wl{workload};

        const VPUPowerFactorLUT power_factor_lut;
        const float operation_pf = power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl, performanceProvider);

        // const auto energy = crt_model.DPUEnergy(wl);
        // const auto af = crt_model.DPUActivityFactor(wl);
        // const auto util = crt_model.hw_utilization(wl);
        // const auto idealCyc = crt_model.DPUIdealCycles(wl);
        // const auto theorCyc = crt_model.DPUTheoreticalCycles(wl);
        // std::string errInfo;
        // const auto nnCyc = crt_model.DPU(wl, errInfo);

        DPUWorkload wl_pack{workload};
        const DPUInfoPack dpu = crt_model.DPUInfo(wl_pack);

        std::stringstream buffer;
        buffer << info << " Op:" << Operation_ToText.at(static_cast<int>(wl.op)) << "\n"
               << "W sparsity: " << wl.weight_sparsity << ", Act sparsity: " << wl.act_sparsity << "\t" << dpu
               << " pf oper correlation :" << operation_pf << "\n ";
        std::string details = buffer.str();

        EXPECT_TRUE(!Cycles::isErrorCode(dpu.DPUCycles)) << info << wl << dpu.errInfo << details;

        EXPECT_GT(dpu.energy, 0) << details;

        EXPECT_GT(dpu.power_activity_factor, 0) << details;
        EXPECT_LE(dpu.power_activity_factor, 1.0f) << details;
        EXPECT_GT(dpu.power_mac_utilization, 0) << details;
        EXPECT_LE(dpu.power_mac_utilization, 1.0f) << details;

        EXPECT_NEAR(dpu.energy, (float)dpu.power_ideal_cycles * operation_pf, 1) << details;
        //-----------------
        DPUWorkload wl_pack_denseW{workload};
        wl_pack_denseW.weight_sparsity_enabled = false;
        wl_pack_denseW.weight_sparsity = 0.0f;
        const DPUInfoPack dpuD_W = crt_model.DPUInfo(wl_pack_denseW);
        {
            EXPECT_TRUE(!Cycles::isErrorCode(dpuD_W.DPUCycles)) << info << dpuD_W << dpuD_W.errInfo << details;

            EXPECT_GT(dpuD_W.energy, 0) << details;

            EXPECT_GT(dpuD_W.energy, dpu.energy);  // dense should be higher  (if not much sparse HW overhead)

            // this raises also the problem : is power virus to be done with sparse enabled but no sparsity at all?,
            // or we allow for a potentially >1 Activity factor

            EXPECT_GT(dpuD_W.power_ideal_cycles, dpu.power_ideal_cycles);  // dense should be higher
            EXPECT_EQ(dpuD_W.efficiency_ideal_cycles,
                      dpu.efficiency_ideal_cycles);  // dense should be equal with sparse for efficiency
            EXPECT_EQ(dpuD_W.hw_theoretical_cycles, dpu.hw_theoretical_cycles);  // unimplemented part
        }

        std::cout << details << "\t" << dpuD_W
                  << "-X-------------------------------------------------------------------------------------------"
                     "-\n";
    }

private:
};

class TestVPUPowerFactorLUT : public ::testing::Test {
public:
protected:
    const DataType defaultTensorType{DataType::FLOAT16};

    const std::array<VPUTensor, 1> outputs{VPUTensor(56, 56, 32, 1, defaultTensorType)};
    const std::array<unsigned int, 2> kernels{3, 3};                   ///< kernel sizes WH
    const std::array<unsigned int, 2> strides{1, 1};                   ///< kernel strides WH
    const std::array<unsigned int, 4> padding{1, 1, 1, 1};             ///< kernel padding  Top, Bottom, Left,  Right
    const ExecutionMode execution_order{ExecutionMode::CUBOID_16x16};  ///< execution mod

    const HWPerformanceModel performanceProvider{};

    void SetUp() override {
        //        TestCostModel::SetUp();
    }
    TestVPUPowerFactorLUT() {
    }

private:
};


}  // namespace VPUNN_unit_tests